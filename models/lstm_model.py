"""
LSTM 基準模型（B1：單層；B2：雙層）。
hidden_size=128, dropout=0.2, 3-class Softmax output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_size: int = 45  # n_features; overridden at runtime
    hidden_size: int = 128
    num_layers: int = 1  # 1 → B1, 2 → B2
    dropout: float = 0.2
    num_classes: int = 3
    bidirectional: bool = False


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier for stock trend prediction.

    Supports both single-layer (B1) and double-layer (B2) configurations.
    """

    def __init__(self, cfg: LSTMConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = LSTMConfig()
        self.cfg = cfg

        # inter-layer dropout only meaningful when num_layers > 1
        lstm_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=lstm_dropout,
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
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, num_classes)
        """
        # lstm_out: (batch, seq_len, hidden * directions)
        lstm_out, _ = self.lstm(x)
        # Use last time-step representation
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)


def build_lstm(input_size: int, num_layers: int = 1, **kwargs) -> LSTMClassifier:
    """Factory helper to build B1 (num_layers=1) or B2 (num_layers=2)."""
    cfg = LSTMConfig(input_size=input_size, num_layers=num_layers, **kwargs)
    return LSTMClassifier(cfg)


def smoke_test() -> None:
    """Shape and forward-pass sanity check."""
    B, T, F = 16, 40, 45

    # B1
    model_b1 = build_lstm(F, num_layers=1)
    x = torch.randn(B, T, F)
    out = model_b1(x)
    assert out.shape == (B, 3), f"Expected ({B}, 3), got {out.shape}"
    print(f"B1 output shape: {out.shape}  params={sum(p.numel() for p in model_b1.parameters()):,}")

    # B2
    model_b2 = build_lstm(F, num_layers=2)
    out2 = model_b2(x)
    assert out2.shape == (B, 3)
    print(
        f"B2 output shape: {out2.shape}  params={sum(p.numel() for p in model_b2.parameters()):,}"
    )

    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
