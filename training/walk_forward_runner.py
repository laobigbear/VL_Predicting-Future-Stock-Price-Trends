"""Walk-forward validation runner.

Ties together WalkForwardSplitter + StockSequenceDataset + train().
Called by main.py --mode train; can also be run directly:
  uv run python -m training.walk_forward_runner --model transformer_resnet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data_pipeline.dataset import (
    StockSequenceDataset,
    WalkForwardConfig,
    WalkForwardSplitter,
    compute_class_weights,
)
from training.train import TrainConfig, build_model, set_seed
from training.train import train as train_one_fold


def load_processed(processed_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in sorted(processed_dir.glob("*_features.parquet")):
        ticker = path.stem.replace("_features", "")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        result[ticker] = df
    return result


def _make_loader(
    frames: dict[str, pd.DataFrame],
    date_start: str,
    date_end: str,
    wf_cfg: WalkForwardConfig,
    shuffle: bool,
) -> tuple[DataLoader | None, torch.Tensor | None]:
    datasets: list[StockSequenceDataset] = []
    for df in frames.values():
        try:
            sliced = df.loc[date_start:date_end]
            if len(sliced) <= wf_cfg.lookback:
                continue
            ds = StockSequenceDataset(sliced, horizon=wf_cfg.horizon, lookback=wf_cfg.lookback)
            if len(ds) > 0:
                datasets.append(ds)
        except Exception:  # noqa: BLE001
            continue

    if not datasets:
        return None, None

    all_y = np.concatenate([ds.y for ds in datasets])
    counts = np.bincount(all_y, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * 3.0
    class_weights = torch.tensor(w, dtype=torch.float32)

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=wf_cfg.batch_size,
        shuffle=shuffle,
        num_workers=wf_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return loader, class_weights


def run(cfg: dict[str, Any], model_name: str, device_str: str, seed: int, smoke: bool = False) -> None:
    set_seed(seed)

    processed_dir = Path(cfg["paths"]["processed_data"])
    frames = load_processed(processed_dir)
    if not frames:
        raise FileNotFoundError(
            f"No feature parquets found in {processed_dir}. Run --mode features first."
        )
    print(f"[INFO] Loaded {len(frames)} tickers")

    # Infer input_size from data
    sample_df = next(iter(frames.values()))
    feat_cols = [c for c in sample_df.columns if not c.startswith("target_")]
    input_size = len(feat_cols)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[INFO] device={device}  model={model_name}  input_size={input_size}")

    # Configs
    wf_yaml = cfg["walk_forward"]
    tr_yaml = cfg["training"]
    mc_yaml = cfg["model"]

    wf_cfg = WalkForwardConfig(
        train_start=wf_yaml["train_start"],
        val_start=wf_yaml["val_start"],
        test_end=wf_yaml["test_end"],
        lookback=int(wf_yaml["lookback"]),
        step_size=int(wf_yaml["step_size"]),
        val_size=int(wf_yaml["val_size"]),
        horizon=int(wf_yaml["horizon"]),
        batch_size=int(wf_yaml["batch_size"]),
        num_workers=int(wf_yaml["num_workers"]),
    )

    train_cfg = TrainConfig(
        model_name=model_name,
        input_size=input_size,
        num_classes=int(mc_yaml["num_classes"]),
        optimizer=tr_yaml["optimizer"],
        learning_rate=float(tr_yaml["learning_rate"]),
        weight_decay=float(tr_yaml["weight_decay"]),
        t_max=int(tr_yaml["t_max"]),
        epochs=2 if smoke else int(tr_yaml["epochs"]),
        batch_size=int(wf_yaml["batch_size"]),
        early_stopping_patience=int(tr_yaml["early_stopping_patience"]),
        checkpoint_dir=cfg["paths"]["checkpoints"],
        log_dir=cfg["paths"]["logs"],
        seed=seed,
    )

    trading_dates = sample_df.index
    splitter = WalkForwardSplitter(wf_cfg, trading_dates)
    steps = list(splitter.steps())
    if smoke:
        steps = steps[:1]

    n_steps = len(steps)
    print(f"[INFO] Walk-forward steps: {n_steps}")

    # W&B config
    wb_cfg = cfg.get("wandb", {})
    use_wandb = wb_cfg.get("enabled", False)
    if use_wandb:
        try:
            import wandb as _wb
            print(f"[W&B] project={wb_cfg.get('project', 'stock-prediction')}")
        except ImportError:
            use_wandb = False
            print("[WARN] wandb not installed, skipping W&B logging")

    step_bar = tqdm(steps, desc=f"WF [{model_name}]", unit="step", dynamic_ncols=True)
    t_start = time.time()

    for step in step_bar:
        step_bar.set_description(
            f"WF [{model_name}] step {step.step_idx + 1}/{n_steps}"
            f" | train~{step.train_end}"
        )

        train_loader, class_weights = _make_loader(
            frames, wf_cfg.train_start, step.train_end, wf_cfg, shuffle=True
        )
        val_loader, _ = _make_loader(
            frames, step.val_start_date, step.val_end_date, wf_cfg, shuffle=False
        )

        if train_loader is None or val_loader is None:
            step_bar.write(f"  [SKIP] step {step.step_idx}: insufficient data")
            continue

        n_train = sum(len(ds) for ds in train_loader.dataset.datasets) if hasattr(train_loader.dataset, "datasets") else len(train_loader.dataset)
        step_bar.write(
            f"\n[Step {step.step_idx + 1}/{n_steps}]"
            f"  train ~ {step.train_end}"
            f"  | val {step.val_start_date} ~ {step.val_end_date}"
            f"  | train_samples={n_train:,}"
        )

        model = build_model(train_cfg)
        run_label = f"{model_name}_step{step.step_idx:02d}"

        # Init W&B run for this step
        if use_wandb:
            try:
                _wb.init(
                    project=wb_cfg.get("project", "stock-prediction"),
                    entity=wb_cfg.get("entity") or None,
                    name=run_label,
                    group=model_name,
                    config={
                        **{k: v for k, v in train_cfg.__dict__.items()
                           if not isinstance(v, dict)},
                        "upload_artifact": wb_cfg.get("upload_artifact", True),
                        "step_idx": step.step_idx,
                        "train_end": step.train_end,
                        "val_start": step.val_start_date,
                        "val_end": step.val_end_date,
                        "train_samples": n_train,
                        "smoke": smoke,
                    },
                    reinit=True,
                )
            except Exception as e:
                step_bar.write(f"[WARN] W&B init failed: {e}")

        t0 = time.time()
        train_one_fold(
            model, train_loader, val_loader, class_weights, train_cfg, device,
            smoke=smoke, run_label=run_label,
        )
        elapsed = time.time() - t0
        step_bar.write(f"  [Step {step.step_idx + 1} done]  elapsed={elapsed:.0f}s")

        # Finish W&B run for this step
        if use_wandb:
            try:
                _wb.finish()
            except Exception:
                pass

    total = time.time() - t_start
    print(f"\n[DONE] Walk-forward complete: {model_name}  total={total/60:.1f} min")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward training runner")
    parser.add_argument("--model", default="transformer_resnet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml

    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args.model, args.device, args.seed, smoke=args.smoke)
