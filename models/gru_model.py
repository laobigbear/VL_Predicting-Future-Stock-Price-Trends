"""
GRU 基準模型（Baseline B3）。
hidden_size=128, dropout=0.2, 3-class output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GRUConfig:
    input_size: int = 45
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    num_classes: int = 3
    bidirectional: bool = False


class GRUClassifier(nn.Module):
    """GRU-based sequence classifier (Baseline B3)."""

    def __init__(self, cfg: GRUConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = GRUConfig()
        self.cfg = cfg

        gru_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        self.dropout = nn.Dropout(cfg.dropout)
        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        gru_out, _ = self.gru(x)
        last = self.dropout(gru_out[:, -1, :])
        return self.classifier(last)


def build_gru(input_size: int, **kwargs) -> GRUClassifier:
    """Factory helper."""
    cfg = GRUConfig(input_size=input_size, **kwargs)
    return GRUClassifier(cfg)


def smoke_test() -> None:
    B, T, F = 16, 40, 45
    model = build_gru(F)
    x = torch.randn(B, T, F)
    out = model(x)
    assert out.shape == (B, 3)
    print(f"GRU output shape: {out.shape}  params={sum(p.numel() for p in model.parameters()):,}")
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
