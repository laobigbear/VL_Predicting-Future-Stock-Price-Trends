"""
Walk-forward Validation 資料集切割與 PyTorch Dataset / DataLoader 介面。

Walk-forward 策略：Expanding Window
  - 起點固定在 2010-01-01
  - 每步向前滾動 step_size 個交易日
  - 驗證集 = 訓練集末尾最近 val_size 個交易日
  - 測試集 = 緊接驗證集後的 step_size 個交易日
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class WalkForwardConfig:
    # date strings in "YYYY-MM-DD"
    train_start: str = "2010-01-01"
    val_start: str = "2021-01-01"  # first test window begins here
    test_end: str = "2025-12-31"

    lookback: int = 40  # sequence length T
    step_size: int = 63  # roll forward by ~3 months
    val_size: int = 126  # ~6 months validation window per step

    horizon: int = 5  # prediction horizon (N)
    batch_size: int = 64
    num_workers: int = 0  # set >0 on Linux for multiprocessing


@dataclass
class WalkForwardStep:
    step_idx: int
    train_end: str
    val_start_date: str
    val_end_date: str
    test_start_date: str
    test_end_date: str


class StockSequenceDataset(Dataset):
    """Sliding-window sequence dataset for a single walk-forward split.

    Each sample is (X, y) where:
        X: float32 tensor of shape (lookback, n_features)
        y: int64 label {0, 1, 2} mapped from {-1, 0, 1}
    """

    LABEL_MAP: dict[int, int] = {-1: 0, 0: 1, 1: 2}  # down=0, flat=1, up=2

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        horizon: int = 5,
        lookback: int = 40,
        feature_cols: list[str] | None = None,
    ) -> None:
        target_col = f"target_h{horizon}"
        if target_col not in feature_matrix.columns:
            raise ValueError(
                f"Column '{target_col}' not found. Available: {list(feature_matrix.columns)}"
            )

        if feature_cols is None:
            feature_cols = [c for c in feature_matrix.columns if not c.startswith("target_")]

        self.feature_cols = feature_cols
        self.lookback = lookback

        # Drop rows where target is NaN (last `horizon` rows)
        df = feature_matrix[feature_cols + [target_col]].dropna(subset=[target_col])

        # Forward-fill then zero-fill remaining NaN in features
        df[feature_cols] = df[feature_cols].ffill().fillna(0.0)

        self.X = df[feature_cols].values.astype(np.float32)
        raw_labels = df[target_col].astype(int).values
        self.y = np.array([self.LABEL_MAP.get(int(v), 1) for v in raw_labels], dtype=np.int64)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.lookback)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.lookback]  # (lookback, F)
        label = self.y[idx + self.lookback - 1]  # label at last step of window
        return torch.from_numpy(x_seq), torch.tensor(label, dtype=torch.long)


def make_dataloader(
    dataset: StockSequenceDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a StockSequenceDataset in a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if shuffle else False,
    )


class WalkForwardSplitter:
    """Generate (train, val, test) date-range splits for expanding-window WFV.

    Usage:
        splitter = WalkForwardSplitter(cfg)
        for step in splitter.steps():
            train_df = full_df.loc[cfg.train_start : step.train_end]
            val_df   = full_df.loc[step.val_start_date : step.val_end_date]
            test_df  = full_df.loc[step.test_start_date : step.test_end_date]
    """

    def __init__(self, cfg: WalkForwardConfig, trading_dates: pd.DatetimeIndex) -> None:
        self.cfg = cfg
        # Filter to dates within [train_start, test_end]
        self.dates = trading_dates[
            (trading_dates >= cfg.train_start) & (trading_dates <= cfg.test_end)
        ]

    def steps(self) -> Iterator[WalkForwardStep]:
        cfg = self.cfg
        dates = self.dates

        # Index of the first date in the test region (2021-01-01)
        val_start_idx = dates.searchsorted(cfg.val_start)

        # Walk forward from val_start to test_end
        cursor = val_start_idx
        step_idx = 0

        while True:
            test_start_idx = cursor
            test_end_idx = min(cursor + cfg.step_size - 1, len(dates) - 1)

            if test_start_idx >= len(dates):
                break

            # Validation window: immediately before test window
            val_end_idx = test_start_idx - 1
            val_start_curr_idx = max(0, val_end_idx - cfg.val_size + 1)

            # Training: from 0 to just before val_start
            train_end_idx = val_start_curr_idx - 1
            if train_end_idx < 0:
                cursor += cfg.step_size
                step_idx += 1
                continue

            yield WalkForwardStep(
                step_idx=step_idx,
                train_end=dates[train_end_idx].strftime("%Y-%m-%d"),
                val_start_date=dates[val_start_curr_idx].strftime("%Y-%m-%d"),
                val_end_date=dates[val_end_idx].strftime("%Y-%m-%d"),
                test_start_date=dates[test_start_idx].strftime("%Y-%m-%d"),
                test_end_date=dates[test_end_idx].strftime("%Y-%m-%d"),
            )

            cursor += cfg.step_size
            step_idx += 1

            if dates[test_end_idx].strftime("%Y-%m-%d") >= cfg.test_end:
                break


def compute_class_weights(dataset: StockSequenceDataset) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(dataset.y, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3.0  # normalise so mean weight = 1
    return torch.tensor(weights, dtype=torch.float32)


def smoke_test() -> None:
    """Quick sanity check with synthetic data."""
    import numpy as np

    from data_pipeline.feature_engineering import FeatureConfig, build_feature_matrix

    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )
    feature_cfg = FeatureConfig()
    df = build_feature_matrix(ohlcv, cfg=feature_cfg)

    cfg = WalkForwardConfig(
        train_start="2020-01-01",
        val_start="2022-01-01",
        test_end="2022-12-31",
        lookback=40,
        step_size=63,
        val_size=126,
        horizon=5,
    )
    splitter = WalkForwardSplitter(cfg, df.index)
    steps = list(splitter.steps())
    print(f"Total walk-forward steps: {len(steps)}")
    if steps:
        s = steps[0]
        print(
            f"Step 0: train_end={s.train_end}, val={s.val_start_date}~{s.val_end_date}, test={s.test_start_date}~{s.test_end_date}"
        )
        train_df = df.loc[cfg.train_start : s.train_end]
        ds = StockSequenceDataset(train_df, horizon=cfg.horizon, lookback=cfg.lookback)
        print(f"Train dataset size: {len(ds)}")
        if len(ds) > 0:
            x, y = ds[0]
            print(f"Sample x shape: {x.shape}, y={y}")
        dl = make_dataloader(ds, batch_size=cfg.batch_size)
        xb, yb = next(iter(dl))
        print(f"Batch x: {xb.shape}, y: {yb.shape}")
        w = compute_class_weights(ds)
        print(f"Class weights: {w}")


if __name__ == "__main__":
    smoke_test()
