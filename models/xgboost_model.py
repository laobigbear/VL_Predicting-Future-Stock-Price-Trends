"""
XGBoost 分類器（Baseline B5）。
使用 XGBoost 原生 DMatrix API（不依賴 sklearn wrapper）。
輸入：展平的滑動視窗特徵向量 (lookback × n_features,)
輸出：3 類別預測（0=Down, 1=Flat, 2=Up）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class XGBConfig:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 5.0
    random_state: int = 42
    n_jobs: int = -1
    lookback: int = 40
    num_classes: int = 3

    # Grid search; set to None to skip
    param_grid: dict | None = field(
        default_factory=lambda: {
            "n_estimators": [200, 500],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
    )
    cv_folds: int = 3


def _flatten_windows(X_3d: np.ndarray) -> np.ndarray:
    """Reshape (N, T, F) → (N, T*F) for XGBoost."""
    return X_3d.reshape(X_3d.shape[0], -1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Compute Macro F1 without sklearn."""
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s))


class XGBoostStockClassifier:
    """XGBoost 3-class stock trend classifier using native DMatrix API."""

    def __init__(self, cfg: XGBConfig | None = None) -> None:
        self.cfg = cfg or XGBConfig()
        self.booster: xgb.Booster | None = None

    def _base_params(self, n_estimators: int | None = None, **overrides) -> dict:
        cfg = self.cfg
        p = {
            "objective": "multi:softmax",
            "num_class": cfg.num_classes,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "min_child_weight": cfg.min_child_weight,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "seed": cfg.random_state,
            "nthread": cfg.n_jobs if cfg.n_jobs > 0 else -1,
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }
        p.update(overrides)
        return p

    def _cv_fold_score(self, X: np.ndarray, y: np.ndarray, params: dict, n_rounds: int) -> float:
        """K-fold macro F1 for a given hyperparameter combination."""
        n = len(X)
        fold_size = n // self.cfg.cv_folds
        scores = []
        for k in range(self.cfg.cv_folds):
            val_slice = slice(k * fold_size, (k + 1) * fold_size)
            mask = np.ones(n, dtype=bool)
            mask[val_slice] = False
            dtr = xgb.DMatrix(X[mask], label=y[mask])
            dvl = xgb.DMatrix(X[val_slice], label=y[val_slice])
            bst = xgb.train(params, dtr, num_boost_round=n_rounds, verbose_eval=False)
            preds = bst.predict(dvl).astype(int)
            scores.append(_macro_f1(y[val_slice].astype(int), preds, self.cfg.num_classes))
        return float(np.mean(scores))

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> tuple[dict, int]:
        """Return best (params, n_rounds) from param_grid."""
        grid = self.cfg.param_grid or {}
        keys = list(grid.keys())
        values = list(grid.values())
        best_score = -1.0
        best_params = self._base_params()
        best_rounds = self.cfg.n_estimators

        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            n_rounds = int(overrides.pop("n_estimators", self.cfg.n_estimators))
            p = self._base_params(**overrides)
            score = self._cv_fold_score(X, y, p, n_rounds)
            if score > best_score:
                best_score = score
                best_params = p
                best_rounds = n_rounds
        return best_params, best_rounds

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostStockClassifier":
        """Fit the model.

        Args:
            X_train: (N, T, F) or (N, n_features).
            y_train: integer class labels {0, 1, 2}.
            X_val, y_val: optional for early stopping / grid search.
        """
        if X_train.ndim == 3:
            X_train = _flatten_windows(X_train)
        if X_val is not None and X_val.ndim == 3:
            X_val = _flatten_windows(X_val)

        if self.cfg.param_grid and X_val is not None:
            X_all = np.vstack([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])
            best_params, best_rounds = self._grid_search(X_all, y_all)
        else:
            best_params = self._base_params()
            best_rounds = self.cfg.n_estimators

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        if X_val is not None:
            evals = [(xgb.DMatrix(X_val, label=y_val), "val")]

        self.booster = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_rounds,
            evals=evals,
            verbose_eval=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class labels {0, 1, 2}."""
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        if X.ndim == 3:
            X = _flatten_windows(X)
        return self.booster.predict(xgb.DMatrix(X)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (N, 3) using softprob objective."""
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        if X.ndim == 3:
            X = _flatten_windows(X)
        # Temporarily switch to softprob
        bst_copy = self.booster.copy()
        bst_copy.set_param("objective", "multi:softprob")
        bst_copy.set_param("num_class", str(self.cfg.num_classes))
        proba = bst_copy.predict(xgb.DMatrix(X))
        return proba.reshape(-1, self.cfg.num_classes)

    def feature_importances(self, feature_names: list[str] | None = None) -> pd.Series:
        """Return XGBoost gain-based feature importances."""
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        scores = self.booster.get_score(importance_type="gain")
        if feature_names is not None:
            imp = pd.Series(
                [scores.get(f"f{i}", 0.0) for i in range(len(feature_names))],
                index=feature_names,
            )
        else:
            imp = pd.Series(scores)
        return imp.sort_values(ascending=False)


def smoke_test() -> None:
    """Quick end-to-end test with synthetic data (no GPU required)."""
    rng = np.random.default_rng(42)
    N, T, F = 200, 40, 20
    X_train = rng.standard_normal((N, T, F)).astype(np.float32)
    y_train = rng.integers(0, 3, N)
    X_val = rng.standard_normal((50, T, F)).astype(np.float32)
    y_val = rng.integers(0, 3, 50)

    cfg = XGBConfig(n_estimators=50, param_grid=None)
    clf = XGBoostStockClassifier(cfg)
    clf.fit(X_train, y_train, X_val, y_val)
    preds = clf.predict(X_val)
    print(f"smoke_test predictions sample: {preds[:10]}")
    proba = clf.predict_proba(X_val)
    print(f"proba shape: {proba.shape}, row sum: {proba[0].sum():.4f}")
    imp = clf.feature_importances()
    print(f"Top-5 feature importance:\n{imp.head()}")
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
