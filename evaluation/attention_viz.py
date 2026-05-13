"""
Attention Weight 視覺化模組。
- 從 TransformerResNet / GatedTransformer forward(return_attn=True) 提取 attention weights
- 產生熱力圖：x 軸 = query time step, y 軸 = key time step
- 支援跨層、跨頭平均，以及逐層逐頭的詳細視覺化
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def extract_attention_weights(
    model: nn.Module,
    x: torch.Tensor,
    regime_prob: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Run forward pass and return attention weights from all layers.

    Args:
        model: TransformerResNet or GatedTransformer (must support return_attn=True).
        x: (batch, seq_len, n_features) input tensor.
        regime_prob: (batch, n_regimes) required for GatedTransformer, else None.
        device: target device.

    Returns:
        List of attn tensors, each shape (batch, nhead, T, T).
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    x = x.to(device)

    with torch.no_grad():
        if regime_prob is not None:
            result = model(x, regime_prob.to(device), return_attn=True)
        else:
            result = model(x, return_attn=True)

    # result is (logits, attn_list)
    if isinstance(result, tuple):
        _, attn_list = result
    else:
        return []

    return [a.cpu() for a in attn_list]


def average_attention(attn_list: list[torch.Tensor]) -> torch.Tensor:
    """Average attention across all layers and heads.

    Returns:
        Tensor of shape (batch, T, T).
    """
    # Stack: (n_layers, batch, nhead, T, T)
    stacked = torch.stack(attn_list, dim=0)
    return stacked.mean(dim=(0, 2))  # mean over layers and heads → (batch, T, T)


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    title: str = "Attention Weight",
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot a single attention weight matrix as a heatmap.

    Args:
        attn_matrix: (T, T) numpy array.
        title: plot title.
        x_labels, y_labels: optional tick labels (time steps or feature names).
        save_path: if provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn_matrix, aspect="auto", cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key (time step)")
    ax.set_ylabel("Query (time step)")

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=6)

    plt.tight_layout()
    if save_path is not None:
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Attention heatmap saved to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Cannot save figure: {exc}")
    plt.close()


def plot_all_layers(
    attn_list: list[torch.Tensor],
    sample_idx: int = 0,
    output_dir: str = "outputs/attention",
    prefix: str = "attn",
) -> None:
    """Save per-layer, per-head attention heatmaps for one sample.

    Args:
        attn_list: list of (batch, nhead, T, T) tensors (one per layer).
        sample_idx: which sample in the batch to visualise.
        output_dir: directory to save PNG files.
        prefix: filename prefix.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, attn in enumerate(attn_list):
        nhead = attn.shape[1]
        for head_idx in range(nhead):
            mat = attn[sample_idx, head_idx].numpy()  # (T, T)
            save_path = str(out_dir / f"{prefix}_layer{layer_idx}_head{head_idx}.png")
            plot_attention_heatmap(
                mat,
                title=f"Layer {layer_idx} Head {head_idx}",
                save_path=save_path,
            )

    # Also save averaged attention
    avg = average_attention(attn_list)[sample_idx].numpy()  # (T, T)
    plot_attention_heatmap(
        avg,
        title="Averaged Attention (all layers & heads)",
        save_path=str(out_dir / f"{prefix}_averaged.png"),
    )
    print(f"[INFO] Saved {len(attn_list) * attn_list[0].shape[1] + 1} attention plots to {out_dir}")


def compute_temporal_importance(attn_list: list[torch.Tensor]) -> torch.Tensor:
    """Derive per-time-step importance as mean attention received.

    Each time step's importance = how much attention it receives across all queries,
    averaged over layers and heads.

    Returns:
        Tensor of shape (batch, T) — attention importance per time step.
    """
    avg = average_attention(attn_list)  # (batch, T, T)
    return avg.mean(dim=1)  # mean over query dim → (batch, T)


def smoke_test() -> None:
    """Test with dummy Transformer model."""
    from models.transformer_resnet import build_transformer_resnet

    B, T, F = 4, 40, 45
    model = build_transformer_resnet(F)
    x = torch.randn(B, T, F)

    attn_list = extract_attention_weights(model, x)
    assert len(attn_list) == 4, f"Expected 4 layers, got {len(attn_list)}"
    assert attn_list[0].shape == (B, 8, T, T)

    avg = average_attention(attn_list)
    assert avg.shape == (B, T, T)
    print(f"avg attn shape: {avg.shape}")

    ti = compute_temporal_importance(attn_list)
    assert ti.shape == (B, T)
    print(f"temporal importance shape: {ti.shape}")

    # Save one heatmap
    plot_attention_heatmap(
        avg[0].numpy(),
        title="Smoke Test Averaged Attention",
        save_path="outputs/attention/smoke_test_avg.png",
    )
    print("smoke_test passed.")


if __name__ == "__main__":
    smoke_test()
