"""
純 Transformer 基準模型（Baseline B4）：d_model=128, nhead=8, num_layers=4。
無 ResNet skip connection；與主線 A 差異在於此處不加殘差連接。
Sinusoidal Positional Encoding。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerBaselineConfig:
    input_size: int = 45
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 2  # B4 uses 2 layers (strong baseline, not 4)
    dim_feedforward: int = 256  # 2 × d_model for B4
    dropout_attn: float = 0.1
    dropout_ff: float = 0.1
    max_seq_len: int = 40
    num_classes: int = 3


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE; shape (1, max_len, d_model)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """Vanilla Transformer encoder classifier (no ResNet skip connections)."""

    def __init__(self, cfg: TransformerBaselineConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TransformerBaselineConfig()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout_attn,
            batch_first=True,
            norm_first=False,  # post-LN (standard)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_ff),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)  # (B, T, d_model)
        pooled = x.mean(dim=1)  # Global Average Pooling over T
        return self.classifier(pooled)


def build_transformer_baseline(input_size: int, **kwargs) -> TransformerBaseline:
    """Factory helper."""
    cfg = TransformerBaselineConfig(input_size=input_size, **kwargs)
    return TransformerBaseline(cfg)


def smoke_test() -> None:
    B, T, F = 8, 40, 45
    model = build_transformer_baseline(F)
    x = torch.randn(B, T, F)
    out = model(x)
    assert out.shape == (B, 3)
    print(
        f"TransformerBaseline output: {out.shape}  params={sum(p.numel() for p in model.parameters()):,}"
    )
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
