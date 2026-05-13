"""
評估指標計算模組：
  - Accuracy, Macro F1, MCC（多類別廣義版）
  - Sharpe Ratio（年化）、Maximum Drawdown
  - Diebold-Mariano Test（Newey-West HAC 標準誤）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# sklearn is imported lazily to avoid DLL issues on some Windows environments.
# Pure-numpy fallback implementations are provided below.


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _mcc_multiclass(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    # Gorodkin (2004) generalised MCC for K classes
    n = len(y_true)
    c_mat = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        c_mat[int(t), int(p)] += 1.0
    # Formula: (sum_k c[k,k]*n - sum_k p[k]*t[k]) / sqrt((n^2 - sum_k p[k]^2)(n^2 - sum_k t[k]^2))
    # Use simplified element-wise form from scikit-learn docs
    t_k = c_mat.sum(axis=1)  # true per class
    p_k = c_mat.sum(axis=0)  # pred per class
    c_kk = np.trace(c_mat)
    num = c_kk * n - float(np.dot(t_k, p_k))
    denom = np.sqrt((n**2 - float(np.dot(p_k, p_k))) * (n**2 - float(np.dot(t_k, t_k))))
    return float(num / denom) if denom > 0 else 0.0


@dataclass
class BacktestConfig:
    risk_free_annual: float = 0.015  # 1.5% 年化無風險利率
    trading_days: int = 252
    tx_cost: float = 0.001425  # 買賣各 0.1425%
    tx_tax: float = 0.003  # 賣出 0.3% 交易稅


# Label mapping: {0: Down, 1: Flat, 2: Up} (matches dataset.py LABEL_MAP)
_LABEL_UP = 2
_LABEL_DOWN = 0
_LABEL_FLAT = 1


# ─── 分類指標 ──────────────────────────────────────────────────────────────────


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Accuracy, Macro F1, and MCC.

    Args:
        y_true, y_pred: integer arrays in {0, 1, 2}.

    Returns:
        dict with keys: accuracy, macro_f1, mcc.
    """
    return {
        "accuracy": _accuracy_score(y_true, y_pred),
        "macro_f1": _f1_macro(y_true, y_pred),
        "mcc": _mcc_multiclass(y_true, y_pred),
    }


# ─── 金融回測指標 ─────────────────────────────────────────────────────────────


def simulate_backtest(
    returns: pd.Series,
    signals: pd.Series,
    cfg: BacktestConfig | None = None,
) -> pd.Series:
    """Simulate strategy portfolio returns given daily signals.

    Args:
        returns: pd.Series of daily asset log returns indexed by date.
        signals: pd.Series of integer predictions (0=Down,1=Flat,2=Up) same index.
        cfg: BacktestConfig.

    Returns:
        pd.Series of daily strategy returns (after costs).
    """
    cfg = cfg or BacktestConfig()

    pos = signals.map({_LABEL_UP: 1.0, _LABEL_FLAT: 0.0, _LABEL_DOWN: -1.0}).fillna(0.0)
    # Transaction cost: incurred when position changes
    pos_change = pos.diff().abs().fillna(0.0)
    # cost = half-turn cost on each side
    cost = pos_change * (cfg.tx_cost + cfg.tx_tax * (pos_change > 0).astype(float))

    strategy_ret = pos.shift(1).fillna(0.0) * returns - cost
    return strategy_ret


def sharpe_ratio(returns: pd.Series, cfg: BacktestConfig | None = None) -> float:
    """Annualised Sharpe Ratio."""
    cfg = cfg or BacktestConfig()
    rf_daily = cfg.risk_free_annual / cfg.trading_days
    excess = returns - rf_daily
    std = excess.std()
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(cfg.trading_days))


def maximum_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative fraction, e.g. -0.153 = -15.3%)."""
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def compute_backtest_metrics(
    returns: pd.Series,
    signals: pd.Series,
    cfg: BacktestConfig | None = None,
) -> dict[str, float]:
    """Compute Sharpe Ratio and MDD for the strategy signal."""
    strat_ret = simulate_backtest(returns, signals, cfg)
    return {
        "sharpe_ratio": sharpe_ratio(strat_ret, cfg),
        "max_drawdown": maximum_drawdown(strat_ret),
        "total_return": float((1.0 + strat_ret).prod() - 1.0),
        "ann_return": float((1.0 + strat_ret).prod() ** (252 / max(len(strat_ret), 1)) - 1.0),
    }


# ─── Diebold-Mariano Test ─────────────────────────────────────────────────────


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """Newey-West HAC variance estimate for DM test."""
    T = len(d)
    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    # Bartlett kernel weights
    nw_sum = 0.0
    for j in range(1, h):
        w = 1.0 - j / h
        gamma_j = np.mean((d[j:] - d_bar) * (d[:-j] - d_bar))
        nw_sum += 2 * w * gamma_j
    return (gamma0 + nw_sum) / T


def diebold_mariano_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    horizon: int = 1,
    two_sided: bool = True,
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[L1 - L2] = 0 (no significant difference)
    H1 (two-sided): E[L1 - L2] != 0

    Args:
        loss1, loss2: per-period loss arrays (e.g., 1 - macro_f1 per WFV step).
        horizon: forecast horizon h (1, 5, or 20).
        two_sided: if True, return two-sided p-value.

    Returns:
        dict with dm_stat, p_value, n_obs.
    """
    d = loss1 - loss2
    T = len(d)
    d_bar = d.mean()
    var_d = _newey_west_variance(d, horizon)

    if var_d <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n_obs": T}

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey et al. (1997) small-sample correction
    correction = np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    dm_stat_corrected = dm_stat * correction

    if two_sided:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat_corrected)))
    else:
        p_value = 1.0 - stats.norm.cdf(dm_stat_corrected)

    return {
        "dm_stat": float(dm_stat_corrected),
        "p_value": float(p_value),
        "n_obs": T,
    }


def run_dm_comparison(
    results_dict: dict[str, dict[str, float]],
    baseline_key: str,
    metric: str = "macro_f1",
    horizon: int = 5,
) -> pd.DataFrame:
    """Run DM tests comparing all models against a baseline.

    Args:
        results_dict: {model_name: {wfv_step_0: metrics, ...}} structure.
                      Each model should have a list of per-step metric values.
        baseline_key: key of the baseline model to compare against.
        metric: metric to use as loss proxy (uses 1 - metric as loss).
        horizon: forecast horizon for NW bandwidth.

    Returns:
        DataFrame with columns: model, dm_stat, p_value, n_obs.
    """
    rows = []
    base_losses = np.array(results_dict[baseline_key])
    for model_name, losses in results_dict.items():
        if model_name == baseline_key:
            continue
        arr = np.array(losses)
        dm = diebold_mariano_test(arr, base_losses, horizon=horizon)
        rows.append({"model": model_name, **dm})
    return pd.DataFrame(rows)


def smoke_test() -> None:
    """Verify all metric functions with synthetic data."""
    rng = np.random.default_rng(42)
    n = 200

    y_true = rng.integers(0, 3, n)
    y_pred = rng.integers(0, 3, n)
    clf_metrics = compute_classification_metrics(y_true, y_pred)
    print(f"Classification: {clf_metrics}")

    dates = pd.bdate_range("2023-01-01", periods=n)
    rets = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    signals = pd.Series(rng.choice([0, 1, 2], n), index=dates)
    bt_metrics = compute_backtest_metrics(rets, signals)
    print(f"Backtest: {bt_metrics}")

    loss1 = rng.uniform(0.3, 0.6, 20)
    loss2 = rng.uniform(0.25, 0.55, 20)
    dm = diebold_mariano_test(loss1, loss2, horizon=5)
    print(f"DM Test: {dm}")


if __name__ == "__main__":
    smoke_test()
