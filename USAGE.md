# USAGE.md — Phase 3 程式碼執行說明

**專案：** Predicting Future Stock Price Trends Using Deep Learning and Institutional Trading Behavior Heterogeneity
**Python：** >= 3.10（建議 3.11 / 3.12）
**環境管理：** uv（不使用 pip / conda）

---

## 目錄

0. [快速開始（main.py）](#0-快速開始mainpy)
1. [環境需求與安裝](#1-環境需求與安裝)
2. [資料收集](#2-資料收集)
3. [模型訓練](#3-模型訓練)
4. [完整 Walk-forward 實驗流程](#4-完整-walk-forward-實驗流程)
5. [評估與可解釋性分析](#5-評估與可解釋性分析)
6. [輸出檔案說明](#6-輸出檔案說明)
7. [常見問題](#7-常見問題-faq)

---

## 0. 快速開始（`main.py`）

`main.py` 是本專案的統一執行入口。**使用者只需修改 `configs/config.yaml`，然後執行一行指令**，無需了解各模組細節。

### 基本用法

```bash
# 完整 pipeline（資料爬取 → 下載 → 特徵工程 → 訓練）
uv run python main.py

# 冒煙測試：快速驗證環境是否正常（縮小資料量，2 batch × 1 epoch）
uv run python main.py --smoke

# 僅執行訓練（資料已下載完成時）
uv run python main.py --mode train

# 臨時指定其他模型（不需修改 config.yaml）
uv run python main.py --mode train --model gru

# 消融實驗：依序訓練所有 6 個模型
uv run python main.py --mode ablation

# 消融實驗快速驗證
uv run python main.py --mode ablation --smoke

# 偵測 GPU / CUDA，自動更新 pyproject.toml（第一次使用請先執行）
uv run python tools/env_check.py
uv run python tools/env_check.py --dry-run  # 僅預覽，不寫入
```

### 執行模式一覽

| `--mode` | 執行內容 | 適用時機 |
|---------|---------|---------|
| `env` | 偵測 GPU/CUDA，更新 pyproject.toml 至最適 PyTorch build | **第一次使用前先執行** |
| `pipeline`（預設）| 爬蟲 → 下載 → 特徵工程 → 訓練 | 第一次執行完整實驗 |
| `crawl` | 僅爬取 TWSE 三大法人資料 | 定期更新最新法人資料 |
| `download` | 僅下載個股 OHLCV + 總經指標 | 補抓缺失股票資料 |
| `features` | 僅執行特徵工程 | 調整特徵參數後重算 |
| `train` | 僅訓練 config 指定的模型 | 資料已備妥，跑特定模型 |
| `ablation` | 依序訓練全部 6 個模型 | 消融實驗 |
| `shap` | 執行 SHAP 特徵重要性分析 | 訓練完成後解釋模型 |
| `viz` | 執行 Attention Weight 視覺化 | 訓練完成後分析注意力 |

### CLI 參數說明

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `--mode` | `pipeline` | 執行模式（見上表） |
| `--config` | `configs/config.yaml` | 設定檔路徑 |
| `--model` | （來自 config） | 臨時覆蓋模型選擇，不需修改 config.yaml |
| `--device` | （來自 config） | `cuda` / `cpu` / `auto`（自動偵測 GPU） |
| `--seed` | （來自 config） | 隨機種子 |
| `--smoke` | 關閉 | 縮小資料量快速驗證整條流程可執行 |

### 設定檔 `configs/config.yaml`

所有參數集中於 `configs/config.yaml`，**使用者唯一需要修改的檔案**。常見調整項目：

```yaml
# 1. 選擇模型
model:
  name: "transformer_resnet"  # lstm_b1 | lstm_b2 | gru | transformer_b4 | transformer_resnet | gated_transformer

# 2. 調整訓練超參數
training:
  epochs: 100
  learning_rate: 0.0001
  device: "auto"              # 自動偵測 GPU；可改為 "cuda" 或 "cpu"
  seed: 42

# 3. 調整資料收集範圍
data_collection:
  twse:
    start_date: "20100101"
    end_date: "20251231"

# 4. 調整預測目標
walk_forward:
  horizon: 5                  # 1 / 5 / 20 日趨勢預測
```

> **提示：** `--model`、`--device`、`--seed` 可在指令列臨時覆蓋 config.yaml 的設定，適合快速測試不同模型，**無需修改設定檔**。

---

## 1. 環境需求與安裝

### 1.1 系統需求

| 項目 | 最低需求 | 建議 |
|------|---------|------|
| Python | 3.10 | 3.11 / 3.12 |
| CUDA | 無（可 CPU 執行） | CUDA 11.8+ |
| RAM | 8 GB | 16 GB+ |
| 磁碟 | 5 GB（資料 + 模型） | 20 GB+ |
| 作業系統 | Windows / Linux / macOS | **Linux / Google Colab（強烈建議）** |

> **Windows 使用者注意：** sklearn 在部分 Windows 環境有 DLL 載入問題，建議優先使用 Linux 環境或 Google Colab。詳見 [常見問題](#7-常見問題-faq)。

### 1.2 安裝 uv

```bash
pip install uv
```

或使用官方安裝腳本（Linux / macOS）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.3 安裝專案依賴

**步驟 1：評估 GPU / CUDA 環境（重要，請先執行）**

```bash
uv run python tools/env_check.py
```

腳本會自動偵測 NVIDIA GPU 與驅動版本，選擇最適合的 PyTorch CUDA build，並更新 `pyproject.toml`。

**步驟 2：安裝所有依賴**

```bash
uv sync --all-groups
```

> **注意：** 若無 NVIDIA GPU，腳本會選擇 CPU 版 PyTorch，不需額外操作。  
> **首次執行** `tools/env_check.py` 前，請先安裝基礎環境：`uv sync`，讓 uv 可以執行 Python。

主要依賴套件（由 `pyproject.toml` 管理）：

```
torch>=2.0,<2.1    # CUDA 11.7 → cu117 build（由 env_check.py 自動設定）
torchvision>=0.15  # 配合 torch 版本（由 env_check.py 自動調整）
numpy>=1.24
pandas>=2.0
pyarrow>=24.0.0    # Parquet 讀寫（必要）
scikit-learn==1.3.2
xgboost>=2.0
statsmodels>=0.14
shap>=0.44
yfinance>=0.2
scipy>=1.11
matplotlib>=3.7
tensorboard>=2.13
```

> **PyTorch CUDA 版本對照：**
> | GPU 驅動版本 | CUDA 支援 | PyTorch build | torch 版本約束 |
> |-------------|---------|--------------|--------------|
> | ≥ 527.41 | 12.4+ | cu124 | >=2.1 |
> | ≥ 527.41 | 12.1+ | cu121 | >=2.1 |
> | ≥ 522.06 | 11.8+ | cu118 | >=2.1 |
> | ≥ 516.94 | 11.7 | cu117 | >=2.0,<2.1 |
> | 無 GPU | — | cpu | >=2.1 |

### 1.4 匯出 requirements.txt（供 Colab 等環境使用）

```bash
uv export --no-hashes > requirements.txt
```

---

## 2. 資料收集

所有資料收集腳本位於 `data_pipeline/` 目錄。執行前請確保已進入 `phase3_code/` 目錄。

### 2.1 `twse_crawler.py` — 爬取 TWSE 三大法人買賣超

**功能：** 從 TWSE 公開資訊觀測站爬取每日三大法人（外資 FINI、投信 ITF、自營商 Dealer）買賣超資料。支援斷點續爬（resume）。

**直接執行（smoke test，僅抓單日驗證連線）：**

```bash
uv run python data_pipeline/twse_crawler.py
```

**在程式中呼叫（完整爬取）：**

```python
from data_pipeline.twse_crawler import CrawlerConfig, crawl_range

cfg = CrawlerConfig(
    start_date="20100101",   # 起始日期（YYYYMMDD）
    end_date="20251231",     # 結束日期（YYYYMMDD）
    output_dir="data/raw/twse",  # 輸出目錄
    sleep_sec=0.5,           # 每次請求間隔（秒）；請勿低於 0.3 秒
    max_retries=3            # 失敗重試次數
)
df = crawl_range(cfg)
print(df.shape)  # 預期: (數百萬筆, 16 欄)
```

**輸出格式：**

- 儲存路徑：`data/raw/twse/institutional_trading.parquet`
- 欄位：`date, ticker, name, fini_buy, fini_sell, fini_net, itf_buy, itf_sell, itf_net, dealer_self_buy, dealer_self_sell, dealer_self_net, dealer_hedge_buy, dealer_hedge_sell, dealer_hedge_net, dealer_net`
- 型態：Parquet（支援增量更新，二次執行自動略過已有日期）

**重要說明：**
- TWSE API 在非交易日（假日、休市日）回傳空資料，屬正常現象
- 完整下載 2010–2025 約需 2–4 小時，建議分批執行

---

### 2.2 `price_downloader.py` — 下載 Yahoo Finance OHLCV

**功能：** 下載個股及總經指標（TAIEX、VIX、美元/台幣）的每日 OHLCV 資料。

**直接執行（smoke test，下載 TSMC 3 日資料）：**

```bash
uv run python data_pipeline/price_downloader.py
```

**在程式中呼叫：**

```python
from data_pipeline.price_downloader import (
    DownloadConfig,
    download_stock_universe,
    download_macro,
    load_taiex_components,
)

# 載入股票清單（從 CSV 或使用內建 20 檔藍籌股）
tickers = load_taiex_components("data/taiex_components.csv")
# CSV 格式：一欄名稱為 "ticker"，內容為股票代號（不含 .TW）

cfg = DownloadConfig(
    start_date="2010-01-01",
    end_date="2025-12-31",
    output_dir="data/raw/prices",
    sleep_sec=0.3,
    tickers=tickers,
)

# 下載個股 OHLCV
stock_data = download_stock_universe(cfg)
# stock_data: dict，key 為股票代號，value 為 DataFrame

# 下載總經指標（TAIEX 指數、VIX、美元/台幣匯率）
macro_data = download_macro(cfg)
```

**股票代號格式：**
- 輸入：台股代號不含後綴，例如 `"2330"`（台積電）
- 內部自動轉換為 Yahoo Finance 格式 `"2330.TW"`
- 總經指標使用 Yahoo 代號：`^TWII`（TAIEX）、`^VIX`、`TWD=X`（美元/台幣）

**輸出格式：**

| 檔案 | 路徑 | 欄位 |
|------|------|------|
| 個股 OHLCV | `data/raw/prices/stocks/{股票代號}.parquet` | date（index）, open, high, low, close, volume, adj_close |
| TAIEX | `data/raw/prices/macro/TAIEX.parquet` | 同上 |
| VIX | `data/raw/prices/macro/VIX.parquet` | 同上 |
| USD/TWD | `data/raw/prices/macro/USDTWD.parquet` | 同上 |

---

### 2.3 `feature_engineering.py` — 建構特徵矩陣

**功能：** 整合 OHLCV、法人買賣超、總經變數，計算以下特徵：
- **法人異質性指標：** BSR（Buy-Sell Ratio）、CNBS-5/20（累計淨買賣超）、成交量周轉率
- **技術指標：** MA(5/20/60)、MA Ratio、RSI(14)、MACD、Bollinger Bands、ATR(14)、對數報酬率
- **總經控制：** VIX、VIX MA20、USD/TWD、各自的日報酬
- **滾動 Z-Score 正規化**（252 日視窗，防止前視偏差）
- **目標變數：** 1/5/20 日三分類標籤（漲=1，盤=0，跌=-1，門檻 ±1.5%）

**直接執行（smoke test）：**

```bash
uv run python data_pipeline/feature_engineering.py
```

**在程式中呼叫：**

```python
import pandas as pd
from data_pipeline.feature_engineering import FeatureConfig, build_feature_matrix

# 讀取已下載的資料
ohlcv = pd.read_parquet("data/raw/prices/stocks/2330.parquet")

# 讀取法人資料（篩選單一股票）
inst_all = pd.read_parquet("data/raw/twse/institutional_trading.parquet")
inst_df = inst_all[inst_all["ticker"] == "2330"].set_index("date").sort_index()

# 讀取總經特徵（需先用 compute_macro_features 處理）
from data_pipeline.feature_engineering import compute_macro_features
macro_raw = {
    "VIX": pd.read_parquet("data/raw/prices/macro/VIX.parquet"),
    "USDTWD": pd.read_parquet("data/raw/prices/macro/USDTWD.parquet"),
}
macro_df = compute_macro_features(macro_raw)

# 建立完整特徵矩陣
cfg = FeatureConfig(
    ma_windows=(5, 20, 60),
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_window=20,
    atr_period=14,
    cnbs_windows=(5, 20),
    zscore_window=252,
    zscore_min_periods=63,
    horizons=(1, 5, 20),
    threshold_pct=1.5,  # ±1.5% 為上漲/下跌門檻
)

feature_matrix = build_feature_matrix(
    ohlcv=ohlcv,
    inst_df=inst_df,
    macro_df=macro_df,
    float_shares=None,   # 若有流通股數資料可傳入 pd.Series
    cfg=cfg,
    normalize=True,      # 滾動 Z-Score 正規化
)
print(feature_matrix.shape)  # (交易日數, 特徵數 + 目標數)
```

**輸入要求：**

| 參數 | 說明 | 必要欄位 |
|------|------|---------|
| `ohlcv` | 單股 OHLCV，date 為 index | open, high, low, close, adj_close, volume |
| `inst_df` | 單股法人資料，date 為 index | fini_buy, fini_sell, fini_net, itf_buy, itf_sell, itf_net, dealer_net |
| `macro_df` | 總經特徵，date 為 index | vix, vix_ret, vix_ma20, usdtwd, usdtwd_ret |

**輸出格式：**
- DataFrame，index 為 date（DatetimeIndex）
- 特徵欄（約 35–45 欄） + 目標欄 `target_h1`, `target_h5`, `target_h20`
- 目標值：Int8 型別，{1=Up, 0=Flat, -1=Down}；最後 N 筆（horizon 天）為 NaN

---

### 2.4 `dataset.py` — 建立 Walk-forward DataLoader

**功能：** 實作 Expanding-Window Walk-forward Validation，將特徵矩陣切割為多個 (train, val, test) 時間段，並包裝為 PyTorch DataLoader。

**直接執行（smoke test）：**

```bash
uv run python data_pipeline/dataset.py
```

**在程式中呼叫：**

```python
import pandas as pd
from data_pipeline.dataset import (
    WalkForwardConfig,
    WalkForwardSplitter,
    StockSequenceDataset,
    make_dataloader,
    compute_class_weights,
)

# 設定 Walk-forward 參數
cfg = WalkForwardConfig(
    train_start="2010-01-01",  # 訓練集起點（固定）
    val_start="2021-01-01",    # Walk-forward 起始日（第一個測試窗口開始於此）
    test_end="2025-12-31",     # 實驗終點
    lookback=40,               # 輸入序列長度 T（交易日數）
    step_size=63,              # 每步向前滾動的天數（約 3 個月）
    val_size=126,              # 驗證集大小（約 6 個月）
    horizon=5,                 # 預測目標：N 日後趨勢（1/5/20）
    batch_size=64,
    num_workers=0,             # Windows 建議設 0；Linux 可設 4
)

# feature_matrix 為 build_feature_matrix() 的輸出
splitter = WalkForwardSplitter(cfg, feature_matrix.index)

for step in splitter.steps():
    print(f"Step {step.step_idx}: "
          f"train_end={step.train_end}, "
          f"val={step.val_start_date}~{step.val_end_date}, "
          f"test={step.test_start_date}~{step.test_end_date}")

    train_df = feature_matrix.loc[cfg.train_start : step.train_end]
    val_df   = feature_matrix.loc[step.val_start_date : step.val_end_date]
    test_df  = feature_matrix.loc[step.test_start_date : step.test_end_date]

    train_ds = StockSequenceDataset(train_df, horizon=cfg.horizon, lookback=cfg.lookback)
    val_ds   = StockSequenceDataset(val_df,   horizon=cfg.horizon, lookback=cfg.lookback)
    test_ds  = StockSequenceDataset(test_df,  horizon=cfg.horizon, lookback=cfg.lookback)

    # 類別不平衡加權
    class_weights = compute_class_weights(train_ds)

    train_loader = make_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = make_dataloader(test_ds,  batch_size=cfg.batch_size, shuffle=False)
```

**標籤映射（`StockSequenceDataset.LABEL_MAP`）：**

| 原始標籤 | 意義 | PyTorch 標籤 |
|---------|------|-------------|
| -1 | Down（下跌） | 0 |
| 0 | Flat（盤整） | 1 |
| 1 | Up（上漲） | 2 |

**DataLoader 輸出格式：**
- `x`：`torch.float32`，shape `(batch, lookback, n_features)`，即 `(B, 40, F)`
- `y`：`torch.int64`，shape `(batch,)`，值域 `{0, 1, 2}`

---

### 2.5 工具腳本（`tools/`）

#### `tools/env_check.py` — GPU/CUDA 環境評估

偵測 NVIDIA GPU 驅動與 CUDA 版本，自動更新 `pyproject.toml` 並給出安裝指令。

```bash
# 評估並更新 pyproject.toml（執行後需 uv sync）
uv run python tools/env_check.py

# 僅預覽，不寫入檔案
uv run python tools/env_check.py --dry-run
```

輸出範例：
```
==============================================================
  PyTorch 環境評估報告
==============================================================
  Python : 3.11.4
  OS     : Windows 10

  GPU    : NVIDIA GeForce MX450  (2048 MB VRAM)
  Driver : 517.00
  CUDA   : 11.7  ← driver 支援上限

  [OK]  目前 PyTorch : 2.0.1+cu117  (tag=cu117)
  [OK]  建議 tag     : cu117  (torch >=2.0,<2.1)

  [OK]  PyTorch 版本符合，CUDA 加速已啟用
==============================================================
```

---

#### `tools/view_stock.py` — 查看下載的 Parquet 資料

快速檢視 `data/raw/` 下的個股或總經 Parquet 檔案。

```bash
# 列出所有可用股票與總經代碼
uv run python tools/view_stock.py --list

# 查看前 10 筆（預設）
uv run python tools/view_stock.py 2330

# 查看最後 20 筆
uv run python tools/view_stock.py 2330 --tail -n 20

# 統計摘要（min, max, mean, std 等）
uv run python tools/view_stock.py 2330 --stats

# 查看總經指標
uv run python tools/view_stock.py TAIEX
uv run python tools/view_stock.py VIX
```

---

## 3. 模型訓練

### 3.1 支援的模型與 `--model` 參數

| 模型名稱 | `--model` 參數 | 說明 | 實驗代碼 |
|---------|--------------|------|---------|
| XGBoost | （另見 3.2） | 展平滑動視窗特徵 | Baseline B5 |
| 單層 LSTM | `lstm_b1` | hidden=128, layers=1 | Baseline B1 |
| 雙層 LSTM | `lstm_b2` | hidden=128, layers=2 | Baseline B2 |
| GRU | `gru` | hidden=128 | Baseline B3 |
| Transformer（無 ResNet） | `transformer_b4` | 純 Transformer Encoder | Baseline B4 |
| Transformer + ResNet | `transformer_resnet` | 核心模型，每 2 層加 skip | 主線 A |
| Gated Transformer | `gated_transformer` | 主線 A + Markov 軟性 Gate | 延伸 B |

### 3.2 XGBoost Baseline（`models/xgboost_model.py`）

XGBoost 使用原生 DMatrix API，**不**透過 `train.py` 執行，直接在 Python 中呼叫：

```python
from models.xgboost_model import XGBConfig, XGBoostStockClassifier
import numpy as np

cfg = XGBConfig(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=5.0,
    random_state=42,
    lookback=40,       # 與 WalkForwardConfig.lookback 一致
    param_grid={       # 設 None 可跳過 Grid Search
        "n_estimators": [200, 500],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
)
clf = XGBoostStockClassifier(cfg)

# X_train shape: (N, lookback, n_features) 或 (N, lookback*n_features)
clf.fit(X_train, y_train, X_val, y_val)
preds = clf.predict(X_test)          # 回傳 int 陣列 {0, 1, 2}
proba = clf.predict_proba(X_test)    # 回傳 (N, 3) 機率矩陣
imp   = clf.feature_importances()   # Gain-based 特徵重要性 pd.Series
```

**smoke test：**

```bash
uv run python models/xgboost_model.py
```

---

### 3.3 深度學習模型（`training/train.py`）

訓練透過 `training/walk_forward_runner.py` 統一執行，由 `main.py` 自動呼叫。`train.py` 提供底層訓練迴圈，`walk_forward_runner.py` 負責：
1. 載入 `data/processed/` 下的特徵 Parquet 檔
2. 依 Walk-forward 切割建立 DataLoader
3. 對每個時間窗口呼叫 `train()` 並儲存 checkpoint

**建議透過 `main.py` 執行，無需直接呼叫 `train.py`：**

```bash
uv run python main.py --mode train --model transformer_resnet --device cuda
```

#### 完整參數說明

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `--model` | `transformer_resnet` | 模型名稱（見 3.1 表格） |
| `--device` | `auto` | `cuda` / `cpu` / `auto`（auto 自動偵測 GPU） |
| `--seed` | `42` | 隨機種子（固定確保可重現） |
| `--smoke` | （旗標） | 加上此旗標執行 2 batch × 1 epoch 快速驗證 |

訓練超參數（在 `TrainConfig` 中設定，無對應 CLI 參數，請在程式中傳入）：

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `epochs` | 100 | 最大訓練 epoch 數 |
| `batch_size` | 64 | Batch size |
| `learning_rate` | 1e-4 | 初始學習率 |
| `weight_decay` | 1e-4 | AdamW L2 正則化係數 |
| `early_stopping_patience` | 20 | 以 val_macro_f1 為準則，Transformer 建議 20；RNN 建議 10 |
| `optimizer` | `adamw` | `adamw` 或 `adam` |
| `t_max` | 50 | CosineAnnealingLR T_max |

#### 快速冒煙測試（驗證環境是否正常）

```bash
uv run python training/train.py --smoke
```

#### 各模型訓練範例

**Baseline B1 — 單層 LSTM：**

```python
from training.train import TrainConfig, build_model, train
import torch

cfg = TrainConfig(
    model_name="lstm_b1",
    input_size=45,       # feature_matrix 的特徵數，需依實際調整
    epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    early_stopping_patience=10,  # RNN 建議較小的 patience
    seed=42,
    checkpoint_dir="checkpoints",
    log_dir="logs",
)
model = build_model(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model = train(
    model, train_loader, val_loader, class_weights, cfg, device,
    run_label="lstm_b1_step0"
)
```

**Baseline B2 — 雙層 LSTM（只需修改 model_name）：**

```python
cfg = TrainConfig(model_name="lstm_b2", ...)
```

**Baseline B3 — GRU：**

```python
cfg = TrainConfig(model_name="gru", ...)
```

**Baseline B4 — Transformer（無 ResNet）：**

```python
cfg = TrainConfig(
    model_name="transformer_b4",
    early_stopping_patience=15,
    ...
)
```

**主線 A — Transformer + ResNet（核心模型）：**

```python
cfg = TrainConfig(
    model_name="transformer_resnet",
    early_stopping_patience=20,
    learning_rate=1e-4,
    optimizer="adamw",
    ...
)
```

**延伸 B — Gated Transformer（需搭配 Markov 狀態機率）：**

GatedTransformer 的 DataLoader 需額外提供 `regime_prob` tensor（shape `(B, 3)`）。請使用包含三份資料的自訂 Dataset：

```python
from models.gated_transformer import build_gated_transformer

cfg = TrainConfig(
    model_name="gated_transformer",
    early_stopping_patience=20,
    ...
)
# train_loader 的 batch 需回傳 (x, y, regime_prob) 三元組
# 詳見 models/gated_transformer.py 的 smoke_test 示例
```

---

## 4. 完整 Walk-forward 實驗流程

以下為從原始資料到最終結果的完整步驟，按序執行：

### Step 1 — 資料下載

```bash
# 1a. 下載 TWSE 三大法人資料（約 2–4 小時）
# 建議在 Python 互動模式或 Jupyter 中執行以便監控進度
uv run python -c "
from data_pipeline.twse_crawler import CrawlerConfig, crawl_range
crawl_range(CrawlerConfig(start_date='20100101', end_date='20251231'))
"

# 1b. 下載個股 OHLCV（準備 taiex_components.csv）
uv run python -c "
from data_pipeline.price_downloader import DownloadConfig, download_stock_universe, download_macro, load_taiex_components
tickers = load_taiex_components('data/taiex_components.csv')
cfg = DownloadConfig(start_date='2010-01-01', end_date='2025-12-31', tickers=tickers)
download_stock_universe(cfg)
download_macro(cfg)
"
```

### Step 2 — 特徵工程（以單股為例，實驗中對所有股票迴圈執行）

```python
import pandas as pd
from data_pipeline.feature_engineering import FeatureConfig, build_feature_matrix, compute_macro_features

# 讀取資料
ohlcv = pd.read_parquet("data/raw/prices/stocks/2330.parquet")
inst_all = pd.read_parquet("data/raw/twse/institutional_trading.parquet")
inst_df = inst_all[inst_all["ticker"] == "2330"].set_index("date").sort_index()
macro_raw = {
    "VIX": pd.read_parquet("data/raw/prices/macro/VIX.parquet"),
    "USDTWD": pd.read_parquet("data/raw/prices/macro/USDTWD.parquet"),
}
macro_df = compute_macro_features(macro_raw)

# 建立特徵矩陣
feature_matrix = build_feature_matrix(ohlcv, inst_df, macro_df, normalize=True)

# 儲存（可選）
feature_matrix.to_parquet("data/processed/2330_features.parquet")
```

### Step 3 — Walk-forward 資料集準備

```python
from data_pipeline.dataset import WalkForwardConfig, WalkForwardSplitter

wf_cfg = WalkForwardConfig(
    train_start="2010-01-01",
    val_start="2021-01-01",
    test_end="2025-12-31",
    lookback=40,
    step_size=63,
    val_size=126,
    horizon=5,
    batch_size=64,
    num_workers=0,  # Windows 請保持 0
)
splitter = WalkForwardSplitter(wf_cfg, feature_matrix.index)
steps = list(splitter.steps())
print(f"Walk-forward steps 總數：{len(steps)}")  # 預期約 19–20 步
```

### Step 4 — 各模型訓練（Walk-forward 迴圈）

```python
import torch
from data_pipeline.dataset import StockSequenceDataset, make_dataloader, compute_class_weights
from training.train import TrainConfig, build_model, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ["lstm_b1", "lstm_b2", "gru", "transformer_b4", "transformer_resnet"]

for model_name in model_names:
    all_test_preds = []
    all_test_labels = []

    for step in steps:
        train_df = feature_matrix.loc[wf_cfg.train_start : step.train_end]
        val_df   = feature_matrix.loc[step.val_start_date : step.val_end_date]
        test_df  = feature_matrix.loc[step.test_start_date : step.test_end_date]

        train_ds = StockSequenceDataset(train_df, horizon=wf_cfg.horizon, lookback=wf_cfg.lookback)
        val_ds   = StockSequenceDataset(val_df,   horizon=wf_cfg.horizon, lookback=wf_cfg.lookback)
        test_ds  = StockSequenceDataset(test_df,  horizon=wf_cfg.horizon, lookback=wf_cfg.lookback)

        class_weights = compute_class_weights(train_ds)
        train_loader  = make_dataloader(train_ds, wf_cfg.batch_size, shuffle=True)
        val_loader    = make_dataloader(val_ds,   wf_cfg.batch_size, shuffle=False)
        test_loader   = make_dataloader(test_ds,  wf_cfg.batch_size, shuffle=False)

        n_features = len(train_ds.feature_cols)
        cfg = TrainConfig(
            model_name=model_name,
            input_size=n_features,
            epochs=100,
            batch_size=wf_cfg.batch_size,
            early_stopping_patience=10 if "lstm" in model_name or model_name == "gru" else 20,
            seed=42,
            checkpoint_dir=f"checkpoints/{model_name}",
            log_dir=f"logs/{model_name}",
        )

        model = build_model(cfg)
        trained_model = train(
            model, train_loader, val_loader, class_weights, cfg, device,
            run_label=f"{model_name}_step{step.step_idx}"
        )

        # 收集測試集預測結果
        trained_model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                logits = trained_model(x_batch.to(device))
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_test_preds.extend(preds.tolist())
                all_test_labels.extend(y_batch.numpy().tolist())
```

### Step 5 — 評估指標計算

```python
import numpy as np
from evaluation.metrics import compute_classification_metrics, compute_backtest_metrics, diebold_mariano_test

y_true = np.array(all_test_labels)
y_pred = np.array(all_test_preds)

clf_metrics = compute_classification_metrics(y_true, y_pred)
print(f"Accuracy: {clf_metrics['accuracy']:.4f}")
print(f"Macro F1: {clf_metrics['macro_f1']:.4f}")
print(f"MCC:      {clf_metrics['mcc']:.4f}")

# 回測指標（需提供對應的日報酬率序列）
# returns = pd.Series(...)  # 對應測試期間的日對數報酬
# signals = pd.Series(y_pred, index=returns.index)
# bt_metrics = compute_backtest_metrics(returns, signals)
```

---

## 5. 評估與可解釋性分析

### 5.1 `evaluation/metrics.py` — 評估指標

**直接執行（smoke test）：**

```bash
uv run python evaluation/metrics.py
```

**分類指標：**

```python
from evaluation.metrics import compute_classification_metrics
import numpy as np

metrics = compute_classification_metrics(y_true, y_pred)
# 回傳 dict:
# {
#   "accuracy":  0.xxxx,  # 整體準確率
#   "macro_f1":  0.xxxx,  # 三類別 Macro F1（主要比較指標）
#   "mcc":       0.xxxx,  # Multiclass MCC（Gorodkin 2004）
# }
```

**回測指標（Sharpe Ratio、MDD）：**

```python
from evaluation.metrics import compute_backtest_metrics, BacktestConfig
import pandas as pd

cfg = BacktestConfig(
    risk_free_annual=0.015,  # 年化無風險利率 1.5%
    trading_days=252,
    tx_cost=0.001425,        # 買賣各 0.1425% 手續費
    tx_tax=0.003,            # 賣出 0.3% 交易稅
)

# returns: 日對數報酬率 pd.Series（index 為 date）
# signals: 預測標籤 pd.Series（{0=Down, 1=Flat, 2=Up}，同 index）
bt_metrics = compute_backtest_metrics(returns, signals, cfg)
# 回傳 dict:
# {
#   "sharpe_ratio":  float,  # 年化 Sharpe Ratio
#   "max_drawdown":  float,  # MDD（負值，如 -0.153 = -15.3%）
#   "total_return":  float,  # 累計報酬率
#   "ann_return":    float,  # 年化報酬率
# }
```

**Diebold-Mariano Test（模型間統計顯著性比較）：**

```python
from evaluation.metrics import diebold_mariano_test, run_dm_comparison
import numpy as np

# loss1, loss2：各 Walk-forward 步驟的損失值（例如 1 - macro_f1）
loss_transformer = 1 - np.array([0.45, 0.48, 0.51, ...])  # 主線 A 每步損失
loss_lstm        = 1 - np.array([0.40, 0.43, 0.47, ...])  # LSTM 每步損失

dm_result = diebold_mariano_test(
    loss1=loss_transformer,
    loss2=loss_lstm,
    horizon=5,       # 與預測 horizon 一致（1/5/20）
    two_sided=True,  # 雙尾檢定
)
# 回傳 dict:
# {
#   "dm_stat":  float,  # DM 統計量（Harvey et al. 1997 小樣本修正版）
#   "p_value":  float,  # p-value
#   "n_obs":    int,    # Walk-forward 步數 T
# }

# 一次比較所有模型 vs. 指定基準
results = {
    "lstm_b1": [0.55, 0.57, 0.58, ...],  # 每步 macro_f1 損失
    "transformer_resnet": [0.50, 0.52, 0.53, ...],
    "gated_transformer": [0.49, 0.50, 0.51, ...],
}
dm_table = run_dm_comparison(results, baseline_key="lstm_b1", metric="macro_f1", horizon=5)
print(dm_table)
```

---

### 5.2 `evaluation/shap_analysis.py` — SHAP 特徵重要性分析

> **需求：** `shap>=0.44`（已含於 `pyproject.toml`）

**直接執行（smoke test）：**

```bash
uv run python evaluation/shap_analysis.py
```

**深度學習模型（DeepExplainer）：**

```python
import torch
from evaluation.shap_analysis import SHAPConfig, compute_shap_dl, group_shap_summary, save_shap_results

cfg = SHAPConfig(
    n_background=100,          # 背景樣本數（用於估計基線期望值）
    n_explain=200,             # 待解釋樣本數
    output_dir="outputs/shap",
    seed=42,
)

# X_tensor: (N, T, F) 的 float32 tensor（從 test StockSequenceDataset 取得）
# feature_names: 長度 F 的特徵名稱列表（= train_ds.feature_cols）
shap_df = compute_shap_dl(
    model=trained_model,          # 已訓練的 TransformerResNet 或其他 nn.Module
    X_tensor=X_tensor,
    feature_names=feature_names,
    cfg=cfg,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
# shap_df 欄位：feature, mean_abs_shap, group（FINI / ITF / DEALER / Technical/Macro）
# 已按 mean_abs_shap 降序排列

# 按法人類型分組摘要
grp_summary = group_shap_summary(shap_df)
print(grp_summary)
# 欄位：avg_shap, total_shap, n_features

# 儲存結果
save_shap_results(shap_df, name="transformer_resnet_step0", cfg=cfg)
# 輸出: outputs/shap/transformer_resnet_step0_shap.csv
```

**XGBoost 模型（TreeExplainer）：**

```python
from evaluation.shap_analysis import compute_shap_xgb

# xgb_model: XGBoostStockClassifier 實例的 .booster 屬性
# X_flat: (N, lookback * n_features) numpy 陣列
shap_df = compute_shap_xgb(
    xgb_model=clf.booster,
    X_flat=X_flat,
    feature_names=feature_names,
    cfg=cfg,
)
```

**輸入模型格式要求：**
- 必須是已訓練（fitted）的模型
- PyTorch 模型需支援 `model(x)` 呼叫，輸出 logits `(B, 3)`
- TransformerResNet / GatedTransformer 在 SHAP 呼叫時使用 DeepExplainer（自動 fallback 至 GradientExplainer）
- GatedTransformer 因需 `regime_prob` 第二輸入，建議先包裝為單輸入模型再傳入 SHAP

---

### 5.3 `evaluation/attention_viz.py` — Attention Weight 視覺化

> **需求：** `matplotlib>=3.7`（已含於 `pyproject.toml`）

**直接執行（smoke test）：**

```bash
uv run python evaluation/attention_viz.py
```

**提取 Attention Weights 並繪製熱力圖：**

```python
import torch
from evaluation.attention_viz import (
    extract_attention_weights,
    average_attention,
    plot_attention_heatmap,
    plot_all_layers,
    compute_temporal_importance,
)

# x: (B, T, F) 輸入 tensor（單一 batch 或單一樣本 unsqueeze(0)）
# trained_model: TransformerResNet 或 GatedTransformer

# 1. 提取所有層的 attention weights
attn_list = extract_attention_weights(
    model=trained_model,
    x=x,
    regime_prob=regime_prob,  # GatedTransformer 需要；TransformerResNet 傳 None
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
# attn_list: 長度 4 的 list，每元素 shape (B, 8, T, T)

# 2. 跨層跨頭平均
avg_attn = average_attention(attn_list)  # (B, T, T)

# 3. 繪製單一樣本的平均熱力圖
plot_attention_heatmap(
    attn_matrix=avg_attn[0].numpy(),     # 取第 0 個樣本
    title="Averaged Attention (all layers & heads)",
    save_path="outputs/attention/avg_attn_sample0.png",
    figsize=(10, 8),
)

# 4. 一次輸出所有層 × 所有頭的熱力圖（共 4 layers × 8 heads + 1 averaged = 33 張）
plot_all_layers(
    attn_list=attn_list,
    sample_idx=0,
    output_dir="outputs/attention",
    prefix="transformer_resnet",
)

# 5. 時間步重要性分析（每個 time step 受到多少 attention）
temporal_imp = compute_temporal_importance(attn_list)  # (B, T)
print(f"最重要的 time step: {temporal_imp[0].argmax().item()}")
```

**輸出格式：**
- 熱力圖：PNG 格式，儲存至 `outputs/attention/`
- x 軸 = Key（被關注的時間步），y 軸 = Query（發出 attention 的時間步）
- 顏色越亮（viridis colormap）表示 attention 越強

---

## 6. 輸出檔案說明

### 資料目錄（`data/`）

| 路徑 | 腳本 | 格式 | 說明 |
|------|------|------|------|
| `data/raw/twse/institutional_trading.parquet` | `twse_crawler.py` | Parquet | 全市場三大法人每日買賣超 |
| `data/raw/prices/stocks/{代號}.parquet` | `price_downloader.py` | Parquet | 個股 OHLCV（adj_close 已還原） |
| `data/raw/prices/macro/TAIEX.parquet` | `price_downloader.py` | Parquet | 台灣加權指數 |
| `data/raw/prices/macro/VIX.parquet` | `price_downloader.py` | Parquet | 芝加哥 VIX 波動率指數 |
| `data/raw/prices/macro/USDTWD.parquet` | `price_downloader.py` | Parquet | 美元/台幣即期匯率 |
| `data/processed/{代號}_features.parquet` | `feature_engineering.py` | Parquet | 完整特徵矩陣（含目標變數） |

### 模型輸出目錄（`checkpoints/`、`logs/`）

| 路徑 | 說明 |
|------|------|
| `checkpoints/{model_name}/{run_label}_best.pt` | 最佳驗證集 val_macro_f1 對應的模型權重 |
| `logs/{model_name}/{run_label}/` | TensorBoard 事件檔案（loss、macro_f1 曲線） |

**查看 TensorBoard：**

```bash
uv run tensorboard --logdir logs/
# 瀏覽器開啟 http://localhost:6006
```

### 評估輸出目錄（`outputs/`）

| 路徑 | 腳本 | 說明 |
|------|------|------|
| `outputs/shap/{name}_shap.csv` | `shap_analysis.py` | 每特徵平均 |SHAP| 值及法人分組 |
| `outputs/attention/{prefix}_layer{i}_head{j}.png` | `attention_viz.py` | 各層各頭 attention 熱力圖 |
| `outputs/attention/{prefix}_averaged.png` | `attention_viz.py` | 跨層跨頭平均 attention 熱力圖 |

---

## 7. 常見問題（FAQ）

### Q1. Windows 上出現 sklearn DLL 錯誤

**症狀：**

```
ImportError: DLL load failed while importing _sklearn...
```

**解決方案（按優先序）：**

1. **優先建議：使用 Google Colab 或 Linux 環境**，可完全避免此問題
2. 安裝 Visual C++ 可轉散發套件：[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
3. 使用 Miniforge（conda-forge channel）安裝 scikit-learn，再用 uv 管理其他套件
4. 本專案所有 sklearn 依賴均有 **pure-numpy fallback**（`metrics.py`、`train.py` 的 F1/Accuracy 均為自實作）；若只需訓練和評估，不安裝 sklearn 也可執行

---

### Q2. `statsmodels` 未安裝導致 Markov Regime Switching 失敗

**症狀：**

```
ModuleNotFoundError: No module named 'statsmodels'
```

**解決方案：**

```bash
uv add statsmodels
```

`statsmodels` 是 `markov_regime.py`（延伸 B）的唯一額外依賴。若只執行主線 A（`transformer_resnet`）及 Baseline 模型，不需安裝。

---

### Q3. Google Colab 使用建議

```python
# Colab Cell 1：安裝 uv 並建立環境
!pip install uv
!uv sync --all-groups

# Colab Cell 2：掛載 Google Drive（儲存資料）
from google.colab import drive
drive.mount('/content/drive')

# Colab Cell 3：執行訓練
!uv run python training/train.py --model transformer_resnet --device cuda --smoke
```

**Colab 注意事項：**
- 免費版 Colab 記憶體限制約 12.7 GB；完整資料集建議使用 Colab Pro 或直接在本機 Linux 執行
- 將 `num_workers=0` 保持不變（Colab 不支援 multiprocessing DataLoader）
- 建議將 `data/` 目錄儲存至 Google Drive 以避免 session 重置後重新下載

---

### Q4. GPU 記憶體不足（OOM）

**症狀：**

```
RuntimeError: CUDA out of memory
```

**解決方案：**

1. 降低 `batch_size`（從 64 降至 32 或 16）
2. 降低 `lookback`（從 40 降至 20）
3. 降低 `d_model`（從 128 降至 64）
4. 使用 CPU 訓練（加上 `--device cpu`）

---

### Q5. TWSE 爬蟲被封鎖或回傳空資料

**症狀 1：** 連續收到 `[WARN] XXXXXXXX fetch failed after 3 attempts` 且速度很慢
**症狀 2：** 爬取完成但 parquet 為 0 rows
**症狀 3：** API 回傳 HTTP 307 + HTML 錯誤頁面

**說明：**
- TWSE T86 API 對特定日期（如假日、非常早期日期）回傳空內容，屬正常現象，爬蟲會直接跳過（不重試）
- 若多個爬蟲程序同時執行，TWSE 可能暫時封鎖 IP（約 10–30 分鐘後自動解除）
- 完整下載 2010–2025 約需 30–60 分鐘

**解決方案：**

1. 等待 10–30 分鐘後重新執行（支援斷點續爬，不會重複下載已有日期）
2. 增加 `sleep_sec`（建議 ≥ 1.0 秒）
3. 分批爬取（縮小 `start_date` / `end_date` 範圍）
4. 避免在開盤時段（9:00–13:30 台灣時間）執行

---

### Q6. `shap` 安裝失敗或版本衝突

```bash
# 手動指定版本安裝
uv add "shap>=0.44"

# 若 shap 未安裝，SHAP 分析會跳過並印出提示
# [SKIP] shap not installed; install with: uv add shap
```

---

### Q7. 如何確認模型訓練結果可重現

本專案所有隨機種子已在 `training/train.py` 的 `set_seed()` 函式中集中設定：

```python
# 訓練開始時自動呼叫 set_seed(cfg.seed)，確保：
# random, numpy, torch, torch.cuda 均使用相同種子
# cudnn.deterministic = True（關閉非決定性算法）
```

若要完整確保可重現：
- 固定 `--seed 42`（預設值）
- 使用相同的 Walk-forward 切割（`WalkForwardConfig` 參數不變）
- PyTorch >= 2.1 在 CPU 上完全確定性；GPU 上受 cuDNN 演算法影響，建議設 `torch.backends.cudnn.benchmark = False`（已預設關閉）

---

### Q8. 如何確認 CUDA 加速是否啟用

```bash
uv run python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

若顯示 `CUDA available: False`，請執行 `tools/env_check.py` 重新評估環境。

---

### Q9. NVIDIA 驅動版本不足

**症狀：** `env_check.py` 顯示 `[!!] 驅動版本不足`

**解決方案：**
1. 前往 [NVIDIA Driver Downloads](https://www.nvidia.com/drivers) 更新驅動
2. 更新後重新執行 `tools/env_check.py` → `uv sync`
3. 各 CUDA 版本最低驅動需求：
   - cu124 / cu121：≥ 527.41（Windows）
   - cu118：≥ 522.06（Windows）
   - cu117：≥ 516.94（Windows）
