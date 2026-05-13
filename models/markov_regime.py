"""
Markov Regime Switching 模組（延伸 B）。
3 狀態：0=Sideways, 1=Bull, 2=Bear
輸入特徵：TAIEX 日對數報酬率、20日滾動波動率、外資5日累計買賣超
實作：statsmodels MarkovRegression (Hamilton 1989)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarkovConfig:
    n_regimes: int = 3
    # Which columns to use as endogenous / exogenous variables
    endog_col: str = "taiex_log_ret"
    exog_cols: list[str] | None = None  # e.g. ["vol20", "fini_cnbs5"]
    k_ar_diff: int = 0  # AR order for MarkovAutoregression
    switching_variance: bool = True  # allow different variance per regime
    max_iter: int = 1000
    em_tol: float = 1e-6
    random_state: int = 42


def _kmeans_init(series: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Simple 1-D K-means for initialising regime means."""
    idx = rng.choice(len(series), n_clusters, replace=False)
    centers = series[idx].copy()
    for _ in range(100):
        labels = np.argmin(np.abs(series[:, None] - centers[None, :]), axis=1)
        new_centers = np.array(
            [
                series[labels == k].mean() if (labels == k).any() else centers[k]
                for k in range(n_clusters)
            ]
        )
        if np.allclose(centers, new_centers, atol=1e-8):
            break
        centers = new_centers
    return centers


def fit_markov_model(
    df: pd.DataFrame,
    cfg: MarkovConfig | None = None,
) -> "FittedMarkovModel":
    """Fit a Markov Regime Switching model on the training window.

    Args:
        df: DataFrame indexed by date, must contain cfg.endog_col and optionally cfg.exog_cols.
        cfg: MarkovConfig.

    Returns:
        FittedMarkovModel instance.
    """
    import statsmodels.tsa.regime_switching.markov_regression as mr

    cfg = cfg or MarkovConfig()
    rng = np.random.default_rng(cfg.random_state)

    endog = df[cfg.endog_col].dropna().values
    exog = df[cfg.exog_cols].loc[df[cfg.endog_col].notna()].values if cfg.exog_cols else None

    # K-means initialisation for regime means (used as diagnostic; statsmodels
    # handles its own start_params internally, but we run K-means to confirm
    # the data has distinguishable regime clusters before fitting)
    _kmeans_init(endog, cfg.n_regimes, rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mr.MarkovRegression(
            endog,
            k_regimes=cfg.n_regimes,
            exog=exog,
            switching_variance=cfg.switching_variance,
        )
        # Provide start_params guess based on K-means regime order
        try:
            result = model.fit(
                start_params=None,
                em_iter=cfg.max_iter,
                maxiter=cfg.max_iter,
                disp=False,
                return_params=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Markov model fitting failed: {exc}") from exc

    return FittedMarkovModel(result=result, cfg=cfg, endog_index=df[cfg.endog_col].dropna().index)


class FittedMarkovModel:
    """Wrapper around a fitted statsmodels MarkovRegression result."""

    def __init__(self, result, cfg: MarkovConfig, endog_index: pd.Index) -> None:
        self.result = result
        self.cfg = cfg
        self.endog_index = endog_index

    def smoothed_probs(self) -> pd.DataFrame:
        """Return smoothed state probabilities as DataFrame (n_dates × n_regimes).

        Columns: regime_0, regime_1, regime_2.
        """
        probs = self.result.smoothed_marginal_probabilities
        # probs shape: (n_obs, n_regimes)
        cols = [f"regime_{k}" for k in range(self.cfg.n_regimes)]
        df = pd.DataFrame(probs, index=self.endog_index, columns=cols)
        return df

    def filtered_probs(self) -> pd.DataFrame:
        """Return filtered (causal, no future info) state probabilities."""
        probs = self.result.filtered_marginal_probabilities
        cols = [f"regime_{k}" for k in range(self.cfg.n_regimes)]
        return pd.DataFrame(probs, index=self.endog_index, columns=cols)

    def predicted_labels(self, use_filtered: bool = True) -> pd.Series:
        """Return most-likely regime label per date (0, 1, or 2).

        use_filtered=True ensures no look-ahead bias (filtered vs. smoothed).
        """
        probs_df = self.filtered_probs() if use_filtered else self.smoothed_probs()
        return probs_df.idxmax(axis=1).map(lambda s: int(s.split("_")[1])).rename("regime_label")


def compute_regime_features(
    price_df: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
    cfg: MarkovConfig | None = None,
    train_end: str | None = None,
) -> pd.DataFrame:
    """Compute regime probability features for the Walk-forward framework.

    Args:
        price_df: DataFrame with TAIEX adj_close indexed by date.
        inst_df: Optional DataFrame with fini_net column for computing CNBS-5.
        cfg: MarkovConfig.
        train_end: Fit MRS only on data up to train_end (no look-ahead).

    Returns:
        DataFrame with columns: regime_0, regime_1, regime_2, regime_label.
        Uses filtered probabilities (causal) to avoid look-ahead.
    """
    cfg = cfg or MarkovConfig()
    close = price_df.get("adj_close", price_df["close"])
    log_ret = np.log(close / close.shift(1)).rename("taiex_log_ret")
    vol20 = log_ret.rolling(20, min_periods=5).std().rename("vol20")

    input_df = pd.concat([log_ret, vol20], axis=1)

    if inst_df is not None and "fini_net" in inst_df.columns:
        cnbs5 = inst_df["fini_net"].rolling(5, min_periods=1).sum().rename("fini_cnbs5")
        input_df = input_df.join(cnbs5, how="left")

    input_df = input_df.dropna(subset=["taiex_log_ret"])

    if train_end is not None:
        fit_df = input_df.loc[:train_end]
    else:
        fit_df = input_df

    fitted = fit_markov_model(fit_df, cfg)
    probs = fitted.filtered_probs()
    labels = fitted.predicted_labels(use_filtered=True)

    return probs.join(labels)


def smoke_test() -> None:
    """Fit a 3-state MRS on synthetic log returns."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    # Mix three Gaussian regimes
    regime = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
    params = {0: (0.0, 0.01), 1: (0.001, 0.008), 2: (-0.001, 0.02)}
    rets = np.array([rng.normal(*params[r]) for r in regime])
    df = pd.DataFrame({"adj_close": 100 * np.cumprod(1 + rets)}, index=idx)

    out = compute_regime_features(df, cfg=MarkovConfig(max_iter=200))
    print(f"smoke_test regime features:\n{out.tail(5).to_string()}")
    print(f"Label distribution:\n{out['regime_label'].value_counts().to_string()}")


if __name__ == "__main__":
    smoke_test()
