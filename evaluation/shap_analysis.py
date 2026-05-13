"""
SHAP 特徵重要性分析模組。
- 深度學習模型：DeepExplainer（GradientExplainer 作為後備）
- XGBoost：TreeExplainer
- 輸出：per-feature mean |SHAP| 排序，支援法人類型分組比較
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class SHAPConfig:
    n_background: int = 100  # background samples for DeepExplainer / KernelExplainer
    n_explain: int = 200  # samples to explain
    output_dir: str = "outputs/shap"
    seed: int = 42


# Groups for institutional heterogeneity analysis (column prefix → group label)
_INST_GROUPS: dict[str, str] = {
    "fini": "FINI",
    "itf": "ITF",
    "dealer": "DEALER",
}


def _get_shap_values_deep(
    model: nn.Module,
    background: torch.Tensor,
    explain_data: torch.Tensor,
) -> np.ndarray:
    """Use shap.DeepExplainer; falls back to GradientExplainer on failure.

    Args:
        model: PyTorch model (must accept input shape matching background).
        background: (n_bg, ...) tensor.
        explain_data: (n_explain, ...) tensor.

    Returns:
        shap_values: (n_explain, ...) numpy array for the predicted class.
    """
    import shap

    model.eval()
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(explain_data)
    except Exception:  # noqa: BLE001
        # Fallback to GradientExplainer (more robust for custom modules)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_data)

    # shap_values: list of (n_explain, T, F) arrays, one per class
    # Take mean absolute over classes and time dimension
    if isinstance(shap_values, list):
        arr = np.stack([np.abs(sv).mean(axis=1) for sv in shap_values], axis=0)
        return arr.mean(axis=0)  # (n_explain, F)
    return np.abs(shap_values).mean(axis=1)  # (n_explain, F)


def _get_shap_values_tree(model, X_flat: np.ndarray) -> np.ndarray:
    """TreeExplainer SHAP values for XGBoost.

    Returns mean absolute SHAP over classes: shape (n_explain, n_features).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_flat)
    if isinstance(shap_values, list):
        arr = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        return arr.mean(axis=0)
    return np.abs(shap_values)


def compute_shap_dl(
    model: nn.Module,
    X_tensor: torch.Tensor,
    feature_names: list[str],
    cfg: SHAPConfig | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Compute mean |SHAP| per feature for a PyTorch model.

    Args:
        model: Fitted PyTorch model.
        X_tensor: (N, T, F) or (N, F) tensor of samples to explain.
        feature_names: list of length F.
        cfg: SHAPConfig.
        device: computation device.

    Returns:
        DataFrame with columns: feature, mean_abs_shap, group.
        Sorted descending by mean_abs_shap.
    """
    cfg = cfg or SHAPConfig()
    torch.manual_seed(cfg.seed)
    device = device or torch.device("cpu")
    model = model.to(device)

    N = X_tensor.shape[0]
    bg_idx = np.random.default_rng(cfg.seed).choice(N, min(cfg.n_background, N), replace=False)
    ex_idx = np.random.default_rng(cfg.seed + 1).choice(N, min(cfg.n_explain, N), replace=False)

    background = X_tensor[bg_idx].to(device)
    explain_data = X_tensor[ex_idx].to(device)

    shap_mat = _get_shap_values_deep(model, background, explain_data)  # (n_explain, F)

    mean_abs = shap_mat.mean(axis=0)  # (F,)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["group"] = df["feature"].map(
        lambda f: next((v for k, v in _INST_GROUPS.items() if f.startswith(k)), "Technical/Macro")
    )
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compute_shap_xgb(
    xgb_model,
    X_flat: np.ndarray,
    feature_names: list[str],
    cfg: SHAPConfig | None = None,
) -> pd.DataFrame:
    """Compute mean |SHAP| for XGBoost model.

    Args:
        xgb_model: Fitted XGBClassifier (from XGBoostStockClassifier.model).
        X_flat: (N, n_features) flattened feature array.
        feature_names: list of length n_features.

    Returns:
        DataFrame with feature, mean_abs_shap, group.
    """
    cfg = cfg or SHAPConfig()
    rng = np.random.default_rng(cfg.seed)
    N = X_flat.shape[0]
    idx = rng.choice(N, min(cfg.n_explain, N), replace=False)
    shap_mat = _get_shap_values_tree(xgb_model, X_flat[idx])  # (n_explain, n_features)

    mean_abs = shap_mat.mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["group"] = df["feature"].map(
        lambda f: next((v for k, v in _INST_GROUPS.items() if f.startswith(k)), "Technical/Macro")
    )
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_shap_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean |SHAP| by institutional group for cross-group comparison."""
    return (
        shap_df.groupby("group")["mean_abs_shap"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "avg_shap", "sum": "total_shap", "count": "n_features"})
        .sort_values("avg_shap", ascending=False)
    )


def save_shap_results(shap_df: pd.DataFrame, name: str, cfg: SHAPConfig | None = None) -> None:
    """Save SHAP DataFrame to CSV."""
    cfg = cfg or SHAPConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_shap.csv"
    try:
        shap_df.to_csv(path, index=False)
        print(f"[INFO] SHAP results saved to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Cannot save SHAP results: {exc}")


def smoke_test() -> None:
    """Verify SHAP pipeline with a tiny dummy model (no GPU needed)."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(40 * 10, 3))

        def forward(self, x):
            return self.net(x)

    F = 10
    model = TinyModel()
    X = torch.randn(50, 40, F)
    feat_names = [f"feat_{i}" for i in range(F)]
    cfg = SHAPConfig(n_background=10, n_explain=20)

    try:
        df = compute_shap_dl(model, X, feat_names, cfg)
        print(f"SHAP DL result:\n{df.head().to_string()}")
        grp = group_shap_summary(df)
        print(f"Group summary:\n{grp.to_string()}")
        print("smoke_test passed.")
    except ImportError:
        print("[SKIP] shap not installed; install with: uv add shap")


if __name__ == "__main__":
    smoke_test()
