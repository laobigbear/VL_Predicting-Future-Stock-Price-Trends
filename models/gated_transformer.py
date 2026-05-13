"""
延伸 B：Gated Transformer（主線 A + Soft Gate by Markov Regime）。

Gate Network：
  regime_prob [B, 3]
  → Linear(3, d_gate=32) → ReLU
  → Linear(d_gate, n_features) → Sigmoid
  = gate_vector [B, n_features]

gated_input = x_features * gate_vector.unsqueeze(1)  → [B, T, F]
再餵入與主線 A 相同的 Transformer + ResNet Encoder。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.transformer_resnet import TransformerResNet, TransformerResNetConfig


@dataclass
class GatedTransformerConfig:
    input_size: int = 45
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout_attn: float = 0.1
    dropout_ff: float = 0.2
    max_seq_len: int = 40
    num_classes: int = 3
    resnet_every_n: int = 2

    # Gate network
    n_regimes: int = 3
    d_gate: int = 32

    # Ablation: set to True for A3 (fixed gate = 1, no dynamic weighting)
    fixed_gate: bool = False


class GatingNetwork(nn.Module):
    """Soft feature gate conditioned on Markov regime probabilities."""

    def __init__(self, n_regimes: int, n_features: int, d_gate: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_regimes, d_gate),
            nn.ReLU(),
            nn.Linear(d_gate, n_features),
            nn.Sigmoid(),
        )

    def forward(self, regime_prob: torch.Tensor) -> torch.Tensor:
        """Args:
            regime_prob: (batch, n_regimes) — softmax probabilities sum to 1.
        Returns:
            gate: (batch, n_features) in (0, 1).
        """
        return self.net(regime_prob)


class GatedTransformer(nn.Module):
    """Transformer + ResNet with Markov-conditioned feature gating (Extension B).

    forward() args:
        x: (batch, seq_len, input_size)
        regime_prob: (batch, n_regimes)  — filtered Markov state probabilities
        return_attn: bool

    Returns:
        logits or (logits, attn_weights_per_layer)
    """

    def __init__(self, cfg: GatedTransformerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GatedTransformerConfig()
        self.cfg = cfg

        # Gating network
        if not cfg.fixed_gate:
            self.gate_net = GatingNetwork(cfg.n_regimes, cfg.input_size, cfg.d_gate)

        # Reuse TransformerResNet backbone
        backbone_cfg = TransformerResNetConfig(
            input_size=cfg.input_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout_attn=cfg.dropout_attn,
            dropout_ff=cfg.dropout_ff,
            max_seq_len=cfg.max_seq_len,
            num_classes=cfg.num_classes,
            resnet_every_n=cfg.resnet_every_n,
        )
        self.backbone = TransformerResNet(backbone_cfg)

    def forward(
        self,
        x: torch.Tensor,
        regime_prob: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply gate then Transformer+ResNet backbone."""
        if self.cfg.fixed_gate:
            # A3 ablation: gate = 1 everywhere (no regime adjustment)
            x_gated = x
        else:
            gate = self.gate_net(regime_prob)  # (B, F)
            x_gated = x * gate.unsqueeze(1)  # (B, T, F)

        return self.backbone(x_gated, return_attn=return_attn)


def build_gated_transformer(
    input_size: int, fixed_gate: bool = False, **kwargs
) -> GatedTransformer:
    """Factory helper. fixed_gate=True produces the A3 ablation."""
    cfg = GatedTransformerConfig(input_size=input_size, fixed_gate=fixed_gate, **kwargs)
    return GatedTransformer(cfg)


def smoke_test() -> None:
    B, T, F = 8, 40, 45
    model = build_gated_transformer(F)
    x = torch.randn(B, T, F)
    regime = torch.softmax(torch.randn(B, 3), dim=-1)

    logits = model(x, regime, return_attn=False)
    assert logits.shape == (B, 3)
    print(f"logits: {logits.shape}")

    logits2, attn = model(x, regime, return_attn=True)
    assert len(attn) == 4
    print(f"attn layers: {[a.shape for a in attn]}")

    # A3 ablation (fixed gate)
    model_a3 = build_gated_transformer(F, fixed_gate=True)
    out_a3 = model_a3(x, regime)
    assert out_a3.shape == (B, 3)
    print(f"A3 ablation logits: {out_a3.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"params={total:,}")
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
