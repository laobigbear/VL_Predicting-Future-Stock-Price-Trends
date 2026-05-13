"""
統一訓練腳本：支援所有深度學習模型（LSTM/GRU/Transformer/主線A/延伸B）。
特性：
  - class_weight 加權 CrossEntropy（處理類別不平衡）
  - 早停準則：val_macro_f1（patience=10/15/20）
  - CosineAnnealingLR scheduler
  - AdamW / Adam 選擇
  - TensorBoard 日誌
  - --smoke 模式（2 batch，1 epoch）
  - 隨機種子固定
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    model_name: str = "transformer_resnet"  # lstm_b1 / lstm_b2 / gru / transformer_b4 / transformer_resnet / gated_transformer
    input_size: int = 45
    num_classes: int = 3

    # Optimizer
    optimizer: str = "adamw"  # "adam" or "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Scheduler
    t_max: int = 50  # CosineAnnealingLR T_max

    # Training loop
    epochs: int = 100
    batch_size: int = 64
    early_stopping_patience: int = 20  # Transformer; use 10 for RNN baselines

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Reproducibility
    seed: int = 42

    # Smoke test
    smoke_batches: int = 2  # only used when smoke=True

    # Extra model kwargs passed to factory
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: TrainConfig) -> nn.Module:
    """Instantiate the model specified by cfg.model_name."""
    kw = cfg.model_kwargs

    if cfg.model_name == "lstm_b1":
        from models.lstm_model import build_lstm

        return build_lstm(cfg.input_size, num_layers=1, **kw)

    if cfg.model_name == "lstm_b2":
        from models.lstm_model import build_lstm

        return build_lstm(cfg.input_size, num_layers=2, **kw)

    if cfg.model_name == "gru":
        from models.gru_model import build_gru

        return build_gru(cfg.input_size, **kw)

    if cfg.model_name == "transformer_b4":
        from models.transformer_baseline import build_transformer_baseline

        return build_transformer_baseline(cfg.input_size, **kw)

    if cfg.model_name == "transformer_resnet":
        from models.transformer_resnet import build_transformer_resnet

        return build_transformer_resnet(cfg.input_size, **kw)

    if cfg.model_name == "gated_transformer":
        from models.gated_transformer import build_gated_transformer

        return build_gated_transformer(cfg.input_size, **kw)

    raise ValueError(f"Unknown model_name: {cfg.model_name}")


def _make_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return Adam(model.parameters(), lr=cfg.learning_rate)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_gated: bool = False,
    smoke: bool = False,
    smoke_batches: int = 2,
    train: bool = True,
) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, macro_f1)."""
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if smoke and batch_idx >= smoke_batches:
                break

            if is_gated:
                x, y, regime = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(x, regime)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)

            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    # Macro F1 computed without sklearn to avoid DLL issues on Windows
    n_classes = 3
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((all_preds_np == c) & (all_labels_np == c)))
        fp = float(np.sum((all_preds_np == c) & (all_labels_np != c)))
        fn = float(np.sum((all_preds_np != c) & (all_labels_np == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    macro_f1 = float(np.mean(f1s))
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, macro_f1


class EarlyStopping:
    """Stop training when val_macro_f1 does not improve for `patience` epochs."""

    def __init__(self, patience: int = 20, delta: float = 1e-5) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score: float = -np.inf
        self.counter: int = 0
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    smoke: bool = False,
    run_label: str = "run",
) -> nn.Module:
    """Full training loop with early stopping and TensorBoard logging.

    Returns the model with best val_macro_f1 weights restored.
    """
    set_seed(cfg.seed)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = _make_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max)
    stopper = EarlyStopping(patience=cfg.early_stopping_patience)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg.log_dir) / run_label))

    is_gated = cfg.model_name == "gated_transformer"
    epochs = 1 if smoke else cfg.epochs

    for epoch in range(epochs):
        tr_loss, tr_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_gated=is_gated,
            smoke=smoke,
            smoke_batches=cfg.smoke_batches,
            train=True,
        )
        val_loss, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            is_gated=is_gated,
            smoke=smoke,
            smoke_batches=cfg.smoke_batches,
            train=False,
        )
        scheduler.step()

        writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("macro_f1", {"train": tr_f1, "val": val_f1}, epoch)

        if (epoch + 1) % 10 == 0 or smoke:
            print(
                f"[Epoch {epoch + 1:03d}] loss={tr_loss:.4f}/{val_loss:.4f}  f1={tr_f1:.4f}/{val_f1:.4f}"
            )

        if stopper.step(val_f1, model):
            print(f"[EarlyStopping] epoch={epoch + 1}, best_val_f1={stopper.best_score:.4f}")
            break

    stopper.restore_best(model)
    writer.close()

    # Save best checkpoint
    ckpt_path = ckpt_dir / f"{run_label}_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Saved] {ckpt_path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument("--model", default="transformer_resnet", help="Model name")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (2 batches, 1 epoch)")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def smoke_test() -> None:
    """Verify training loop with dummy data (no real data needed)."""
    from torch.utils.data import TensorDataset

    B, T, F = 32, 40, 45
    rng = np.random.default_rng(0)

    N_train, N_val = B * 3, B
    X_tr = torch.from_numpy(rng.standard_normal((N_train, T, F)).astype(np.float32))
    y_tr = torch.from_numpy(rng.integers(0, 3, N_train).astype(np.int64))
    X_vl = torch.from_numpy(rng.standard_normal((N_val, T, F)).astype(np.float32))
    y_vl = torch.from_numpy(rng.integers(0, 3, N_val).astype(np.int64))

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_vl, y_vl)
    train_dl = DataLoader(train_ds, batch_size=16)
    val_dl = DataLoader(val_ds, batch_size=16)

    cfg = TrainConfig(model_name="lstm_b1", input_size=F, epochs=2, early_stopping_patience=2)
    model = build_model(cfg)
    weights = torch.ones(3)
    device = torch.device("cpu")
    trained_model = train(
        model, train_dl, val_dl, weights, cfg, device, smoke=True, run_label="smoke"
    )
    print(f"smoke_test passed. Model type: {type(trained_model).__name__}")


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke_test()
    else:
        print("Please integrate with walk-forward runner. Use --smoke for quick test.")
        sys.exit(0)
