# TW Institutional Stock Prediction

Predicting future stock price trends (TAIEX components, 2010-2025) using
Transformer + ResNet + Multi-head Attention, with institutional investor
heterogeneity features (FINI/ITF/DEALER) and Markov Regime Switching.

## Requirements

- Python 3.10+
- uv (package manager)

## Installation

```bash
# Install uv if not already present
pip install uv

# Install all dependencies
cd phase3_code
uv sync
```

## Quick Smoke Test (no data needed)

```bash
# Verify each module independently
uv run python data_pipeline/twse_crawler.py
uv run python data_pipeline/price_downloader.py
uv run python data_pipeline/feature_engineering.py
uv run python data_pipeline/dataset.py
uv run python models/xgboost_model.py
uv run python models/lstm_model.py
uv run python models/gru_model.py
uv run python models/transformer_baseline.py
uv run python models/transformer_resnet.py
uv run python models/markov_regime.py
uv run python models/gated_transformer.py
uv run python evaluation/metrics.py
uv run python evaluation/shap_analysis.py
uv run python evaluation/attention_viz.py

# Training smoke test (2 batches, 1 epoch)
uv run python training/train.py --smoke
```

## Data Collection

```bash
# Step 1: Download TWSE institutional trading (will take time; resumes on restart)
uv run python -c "
from data_pipeline.twse_crawler import crawl_range, CrawlerConfig
crawl_range(CrawlerConfig(start_date='20100101', end_date='20251231'))
"

# Step 2: Download OHLCV prices + macro variables
uv run python -c "
from data_pipeline.price_downloader import DownloadConfig, download_macro, load_taiex_components, download_stock_universe
cfg = DownloadConfig()
cfg.tickers = load_taiex_components()
download_stock_universe(cfg)
download_macro(cfg)
"
```

## Walk-forward Training (single model, example)

```python
from data_pipeline.dataset import WalkForwardConfig, WalkForwardSplitter, StockSequenceDataset, make_dataloader, compute_class_weights
from training.train import TrainConfig, build_model, train
import torch, pandas as pd

# Load pre-built feature matrix (run feature_engineering first)
df = pd.read_parquet("data/processed/features_2330.parquet")
wf_cfg = WalkForwardConfig(horizon=5)
splitter = WalkForwardSplitter(wf_cfg, df.index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for step in splitter.steps():
    train_df = df.loc[wf_cfg.train_start : step.train_end]
    val_df   = df.loc[step.val_start_date : step.val_end_date]
    train_ds = StockSequenceDataset(train_df, horizon=wf_cfg.horizon)
    val_ds   = StockSequenceDataset(val_df,   horizon=wf_cfg.horizon)
    weights  = compute_class_weights(train_ds)

    cfg = TrainConfig(model_name="transformer_resnet", input_size=len(train_ds.feature_cols))
    model = build_model(cfg)
    model = train(model, make_dataloader(train_ds), make_dataloader(val_ds, shuffle=False),
                  weights, cfg, device, run_label=f"step{step.step_idx}")
```

## Evaluation

```python
from evaluation.metrics import compute_classification_metrics, compute_backtest_metrics
import numpy as np, pandas as pd

y_true = np.array([...])
y_pred = np.array([...])
print(compute_classification_metrics(y_true, y_pred))

returns = pd.Series([...])
signals = pd.Series([...])
print(compute_backtest_metrics(returns, signals))
```

## Project Structure

```
phase3_code/
├── pyproject.toml
├── README.md
├── data_pipeline/
│   ├── twse_crawler.py        TWSE 三大法人爬蟲
│   ├── price_downloader.py    Yahoo Finance OHLCV
│   ├── feature_engineering.py 法人/技術/總經特徵 + 目標變數
│   └── dataset.py             Walk-forward Dataset / DataLoader
├── models/
│   ├── xgboost_model.py       B5: XGBoost
│   ├── lstm_model.py          B1/B2: LSTM 單/雙層
│   ├── gru_model.py           B3: GRU
│   ├── transformer_baseline.py B4: 純 Transformer
│   ├── transformer_resnet.py  主線 A: Transformer + ResNet
│   ├── markov_regime.py       延伸 B: Markov Regime Switching
│   └── gated_transformer.py   延伸 B: Gated Transformer
├── training/
│   └── train.py               統一訓練腳本 (早停 / CosineAnnealingLR)
└── evaluation/
    ├── metrics.py             Accuracy/F1/MCC/Sharpe/MDD/DM Test
    ├── shap_analysis.py       SHAP 特徵重要性
    └── attention_viz.py       Attention Weight 熱力圖
```

## Reproducibility

All training scripts accept a `--seed` argument (default: 42).
`set_seed(42)` is called at the start of each training run.

## Models and Experiments

| ID | Model | Purpose |
|----|-------|---------|
| B1 | LSTM (1-layer) | Baseline |
| B2 | LSTM (2-layer) | Baseline |
| B3 | GRU | Baseline |
| B4 | Transformer (no ResNet) | Strong baseline |
| B5 | XGBoost | Non-DL baseline |
| A  | Transformer + ResNet | Core model (Main Line A) |
| AB | Gated Transformer + Markov | Extension B |
