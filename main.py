"""
股價趨勢預測研究實驗 — 一鍵執行入口。

使用方式：
    uv run python main.py                          # 完整 pipeline
    uv run python main.py --mode train             # 僅訓練
    uv run python main.py --mode ablation --smoke  # 消融實驗（快速驗證）
    uv run python main.py --config configs/config.yaml --model gru
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ─── Config ──────────────────────────────────────────────────────────────────


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """從 YAML 檔案載入設定。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到設定檔：{path}\n請確認 configs/config.yaml 存在。")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(
        description="股價趨勢預測研究實驗一鍵執行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行模式：
  pipeline   完整流程：爬蟲 → 下載 → 特徵工程 → 訓練 → 評估（預設）
  crawl      僅爬取 TWSE 三大法人資料
  download   僅下載個股 OHLCV + 總經指標
  features   僅執行特徵工程
  train      僅執行模型訓練
  ablation   對所有模型依序執行消融實驗
  shap       執行 SHAP 特徵重要性分析
  viz        執行 Attention Weight 視覺化

範例：
  uv run python main.py
  uv run python main.py --mode train --model transformer_resnet
  uv run python main.py --mode ablation --smoke
  uv run python main.py --config configs/config.yaml --device cuda
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "crawl", "download", "features", "train", "ablation", "shap", "viz"],
        default="pipeline",
        help="執行模式（預設：pipeline）",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="設定檔路徑（預設：configs/config.yaml）",
    )
    parser.add_argument(
        "--model",
        choices=[
            "lstm_b1",
            "lstm_b2",
            "gru",
            "transformer_b4",
            "transformer_resnet",
            "gated_transformer",
        ],
        default=None,
        help="指定模型（覆蓋 config.yaml 的 model.name）",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="運算裝置（覆蓋 config.yaml 的 training.device，預設：auto）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="隨機種子（覆蓋 config.yaml 的 training.seed，預設：42）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="冒煙測試模式：縮小資料量快速驗證整條流程可執行",
    )
    return parser.parse_args()


# ─── Pipeline steps ───────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_crawl(cfg: dict[str, Any], *, smoke: bool = False) -> None:
    """爬取 TWSE 三大法人每日買賣超。"""
    from data_pipeline.twse_crawler import CrawlerConfig, crawl_range

    _header("STEP 1／5  TWSE 三大法人爬蟲")
    dc = cfg["data_collection"]["twse"]
    crawler_cfg = CrawlerConfig(
        start_date="20251201" if smoke else dc["start_date"],
        end_date="20251231" if smoke else dc["end_date"],
        output_dir=dc["output_dir"],
        sleep_sec=dc["sleep_sec"],
        max_retries=dc["max_retries"],
    )
    if smoke:
        print("[SMOKE] 爬取範圍縮短至 2025-12-01 ~ 2025-12-31")
    crawl_range(crawler_cfg)
    print("[DONE] TWSE 爬取完成")


def run_download(cfg: dict[str, Any], *, smoke: bool = False) -> None:
    """下載個股 OHLCV + 總經指標（VIX、USDTWD、TAIEX）。"""
    from data_pipeline.price_downloader import (
        DownloadConfig,
        download_macro,
        download_stock_universe,
        load_taiex_components,
    )

    _header("STEP 2／5  個股 OHLCV + 總經指標下載")
    dc = cfg["data_collection"]["prices"]
    tickers: list[str] = dc.get("tickers") or load_taiex_components(
        cfg["paths"].get("taiex_components")
    )
    if smoke:
        tickers = tickers[:3]
        print(f"[SMOKE] 僅下載前 3 檔：{tickers}")

    dl_cfg = DownloadConfig(
        start_date=dc["start_date"],
        end_date=dc["end_date"],
        output_dir=dc["output_dir"],
        sleep_sec=dc["sleep_sec"],
        tickers=tickers,
    )
    download_stock_universe(dl_cfg)
    download_macro(dl_cfg)
    print("[DONE] 資料下載完成")


def run_features(cfg: dict[str, Any]) -> None:
    """整合 OHLCV、法人資料、總經指標，產出特徵矩陣。"""
    import pandas as pd

    from data_pipeline.feature_engineering import (
        FeatureConfig,
        build_feature_matrix,
        compute_macro_features,
    )

    _header("STEP 3／5  特徵工程")
    fc = cfg["features"]
    feat_cfg = FeatureConfig(
        ma_windows=tuple(fc["ma_windows"]),
        rsi_period=fc["rsi_period"],
        macd_fast=fc["macd_fast"],
        macd_slow=fc["macd_slow"],
        macd_signal=fc["macd_signal"],
        bb_window=fc["bb_window"],
        bb_std=fc["bb_std"],
        atr_period=fc["atr_period"],
        cnbs_windows=tuple(fc["cnbs_windows"]),
        zscore_window=fc["zscore_window"],
        zscore_min_periods=fc["zscore_min_periods"],
        horizons=tuple(fc["horizons"]),
        threshold_pct=fc["threshold_pct"],
    )

    raw_dir = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    macro_raw = {
        "VIX": pd.read_parquet(raw_dir / "prices/macro/VIX.parquet"),
        "USDTWD": pd.read_parquet(raw_dir / "prices/macro/USDTWD.parquet"),
    }
    macro_df = compute_macro_features(macro_raw)
    inst_all = pd.read_parquet(raw_dir / "twse/institutional_trading.parquet")

    stocks_dir = raw_dir / "prices/stocks"
    ok, skip = 0, 0
    for stock_file in sorted(stocks_dir.glob("*.parquet")):
        ticker = stock_file.stem
        try:
            ohlcv = pd.read_parquet(stock_file)
            inst_df = (
                inst_all[inst_all["ticker"] == ticker].set_index("date").sort_index()
            )
            feat = build_feature_matrix(ohlcv, inst_df, macro_df, cfg=feat_cfg, normalize=True)
            feat.to_parquet(processed_dir / f"{ticker}_features.parquet")
            ok += 1
            print(f"  [OK] {ticker}  ({len(feat)} rows, {feat.shape[1]} cols)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {ticker}: {exc}")
            skip += 1

    print(f"[DONE] 特徵工程完成：成功 {ok} 檔，跳過 {skip} 檔")


def _resolve_model_device_seed(
    cfg: dict[str, Any],
    model_name: str | None,
    device: str | None,
    seed: int | None,
) -> tuple[str, str, int]:
    """從 cfg 與 CLI 引數解析最終使用的 model / device / seed。"""
    tc = cfg["training"]
    mc = cfg["model"]
    resolved_model = model_name or mc["name"]
    resolved_device = device or tc.get("device", "auto")
    resolved_seed = seed if seed is not None else int(tc.get("seed", 42))
    return resolved_model, resolved_device, resolved_seed


def run_train(
    cfg: dict[str, Any],
    model_name: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    *,
    smoke: bool = False,
) -> None:
    """呼叫 training/train.py 執行單一模型訓練。"""
    _header("STEP 4／5  模型訓練")
    model, dev, s = _resolve_model_device_seed(cfg, model_name, device, seed)
    cmd = [
        sys.executable, "-m", "training.train",
        "--model", model,
        "--device", dev,
        "--seed", str(s),
    ]
    if smoke:
        cmd.append("--smoke")
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] 訓練失敗，returncode={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[DONE] 訓練完成：{model}")


def run_ablation(
    cfg: dict[str, Any],
    device: str | None = None,
    seed: int | None = None,
    *,
    smoke: bool = False,
) -> None:
    """對所有模型依序執行消融實驗。"""
    _header("消融實驗  — 所有模型")
    all_models = [
        "lstm_b1",
        "lstm_b2",
        "gru",
        "transformer_b4",
        "transformer_resnet",
        "gated_transformer",
    ]
    _, dev, s = _resolve_model_device_seed(cfg, None, device, seed)
    for i, model in enumerate(all_models, 1):
        print(f"\n[{i}/{len(all_models)}] 訓練模型：{model}")
        cmd = [
            sys.executable, "-m", "training.train",
            "--model", model,
            "--device", dev,
            "--seed", str(s),
        ]
        if smoke:
            cmd.append("--smoke")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARN] {model} 訓練失敗，跳過並繼續")
    print("\n[DONE] 消融實驗完成")


def run_shap(cfg: dict[str, Any]) -> None:
    """執行 SHAP 特徵重要性分析（需先完成訓練）。"""
    _header("SHAP 特徵重要性分析")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if not any(ckpt_dir.rglob("*.pt")):
        print("[WARN] 找不到已訓練的 checkpoint，請先執行 --mode train")
        return
    result = subprocess.run([sys.executable, "-m", "evaluation.shap_analysis"])
    if result.returncode != 0:
        print("[ERROR] SHAP 分析失敗", file=sys.stderr)
        sys.exit(result.returncode)
    print("[DONE] SHAP 分析完成，結果存於 outputs/shap/")


def run_viz(cfg: dict[str, Any]) -> None:
    """執行 Attention Weight 視覺化（需先完成訓練）。"""
    _header("Attention Weight 視覺化")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if not any(ckpt_dir.rglob("*.pt")):
        print("[WARN] 找不到已訓練的 checkpoint，請先執行 --mode train")
        return
    result = subprocess.run([sys.executable, "-m", "evaluation.attention_viz"])
    if result.returncode != 0:
        print("[ERROR] 視覺化失敗", file=sys.stderr)
        sys.exit(result.returncode)
    print("[DONE] 視覺化完成，結果存於 outputs/attention/")


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """整合所有實驗階段的主入口。使用者只需修改 configs/config.yaml。"""
    args = parse_args()
    cfg = load_config(args.config)

    print(f"\n{'=' * 60}")
    print("  股價趨勢預測研究實驗")
    print(f"  模式：{args.mode}{'  [SMOKE TEST]' if args.smoke else ''}")
    print(f"  設定：{args.config}")
    print(f"{'=' * 60}")

    mode = args.mode

    if mode == "crawl":
        run_crawl(cfg, smoke=args.smoke)

    elif mode == "download":
        run_download(cfg, smoke=args.smoke)

    elif mode == "features":
        run_features(cfg)

    elif mode == "train":
        run_train(cfg, args.model, args.device, args.seed, smoke=args.smoke)

    elif mode == "ablation":
        run_ablation(cfg, args.device, args.seed, smoke=args.smoke)

    elif mode == "shap":
        run_shap(cfg)

    elif mode == "viz":
        run_viz(cfg)

    else:  # pipeline (default)
        print("\n[PIPELINE] 開始完整實驗流程（共 5 步）")
        run_crawl(cfg, smoke=args.smoke)
        run_download(cfg, smoke=args.smoke)
        run_features(cfg)
        run_train(cfg, args.model, args.device, args.seed, smoke=args.smoke)
        print(f"\n{'=' * 60}")
        print("  [PIPELINE 完成]")
        print("  結果位置：")
        print("    checkpoints/  — 模型權重")
        print("    logs/         — TensorBoard 日誌（uv run tensorboard --logdir logs/）")
        print("    outputs/      — SHAP / Attention 分析")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
