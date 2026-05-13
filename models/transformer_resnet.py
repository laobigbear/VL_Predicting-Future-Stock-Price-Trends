"""
主線 A 核心模型：Transformer + ResNet skip connection + Multi-head Attention。

架構：
  Input Projection (F → d_model=128)
  + Sinusoidal PE
  → 4 × TransformerEncoderLayer  (ResNet skip每2層施加一次)
  → Global Average Pooling
  → Classification Head (3-class)

Attention weights 從每層 MHA 中提取，存於 forward() 的第二回傳值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerResNetConfig:
    input_size: int = 45
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512  # 4 × d_model
    dropout_attn: float = 0.1
    dropout_ff: float = 0.2
    max_seq_len: int = 40
    num_classes: int = 3
    resnet_every_n: int = 2  # add skip connection every N encoder layers


class SinusoidalPE(nn.Module):
    """Sinusoidal Positional Encoding (fixed, not learnable)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerEncoderLayerWithAttn(nn.Module):
    """TransformerEncoderLayer that also returns attention weights.

    Mirrors nn.TransformerEncoderLayer but exposes attn_weights.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (output, attn_weights).

        attn_weights: (batch, nhead, T, T) or None if need_weights=False.
        """
        attn_out, attn_w = self.self_attn(
            x, x, x, need_weights=need_weights, average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x, attn_w


class TransformerResNet(nn.Module):
    """Transformer + ResNet skip connections every `resnet_every_n` layers.

    forward() returns:
        logits: (B, num_classes)
        attn_weights_per_layer: list of (B, nhead, T, T) — one per encoder layer
    """

    def __init__(self, cfg: TransformerResNetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TransformerResNetConfig()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)
        self.pos_enc = SinusoidalPE(cfg.d_model, cfg.max_seq_len + 10, dropout=cfg.dropout_attn)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttn(
                    cfg.d_model, cfg.nhead, cfg.dim_feedforward, dropout=cfg.dropout_attn
                )
                for _ in range(cfg.num_encoder_layers)
            ]
        )

        # LayerNorm applied after each ResNet skip merge
        self.resnet_norms = nn.ModuleList(
            [nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_encoder_layers // cfg.resnet_every_n)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_ff),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            x: (batch, seq_len, input_size)
            return_attn: whether to return attention weights

        Returns:
            logits or (logits, attn_weights_per_layer)
        """
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)

        attn_weights_per_layer: list[torch.Tensor] = []
        skip_origin = x  # residual anchor for first block
        norm_idx = 0

        for i, layer in enumerate(self.layers):
            x, attn_w = layer(x, need_weights=return_attn)
            if return_attn and attn_w is not None:
                attn_weights_per_layer.append(attn_w)

            # Apply ResNet skip every `resnet_every_n` layers
            if (i + 1) % self.cfg.resnet_every_n == 0:
                x = self.resnet_norms[norm_idx](x + skip_origin)
                skip_origin = x  # update anchor for next block
                norm_idx += 1

        pooled = x.mean(dim=1)  # Global Average Pooling over T
        logits = self.classifier(pooled)

        if return_attn:
            return logits, attn_weights_per_layer
        return logits


def build_transformer_resnet(input_size: int, **kwargs) -> TransformerResNet:
    """Factory helper."""
    cfg = TransformerResNetConfig(input_size=input_size, **kwargs)
    return TransformerResNet(cfg)


def smoke_test() -> None:
    B, T, F = 8, 40, 45
    model = build_transformer_resnet(F)
    x = torch.randn(B, T, F)

    # Without attention
    logits = model(x, return_attn=False)
    assert logits.shape == (B, 3), f"Expected ({B},3), got {logits.shape}"

    # With attention
    logits2, attn_list = model(x, return_attn=True)
    assert len(attn_list) == 4
    assert attn_list[0].shape == (B, model.cfg.nhead, T, T)
    print(f"logits: {logits2.shape}")
    print(f"attn per layer: {[a.shape for a in attn_list]}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
