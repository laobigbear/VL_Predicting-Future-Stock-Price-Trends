"""
環境評估工具

偵測 GPU / CUDA 版本，決定最適合的 PyTorch CUDA build，
並自動更新 pyproject.toml 的 [tool.uv.sources] / [[tool.uv.indexes]]。
執行後只需 `uv sync` 即可完成安裝。

用法：
  uv run python tools/env_check.py           # 評估 + 更新 pyproject.toml
  uv run python tools/env_check.py --dry-run # 僅顯示，不寫檔
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ── 常數 ─────────────────────────────────────────────────────────────────────

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# (最低 CUDA 版本, tag, torch 版本約束, torchvision 版本約束, Windows 最低驅動版本)
_CUDA_TABLE: list[tuple[float, str, str, str, float]] = [
    (12.4, "cu124", ">=2.1",      ">=0.16", 527.41),
    (12.1, "cu121", ">=2.1",      ">=0.16", 527.41),
    (11.8, "cu118", ">=2.1",      ">=0.16", 522.06),
    (11.7, "cu117", ">=2.0,<2.1", ">=0.15,<0.16", 516.94),
]

_INDEX_URL = "https://download.pytorch.org/whl/{tag}"

# ── 資料結構 ─────────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    name: str
    vram_mb: int
    driver: float
    driver_str: str   # 原始字串，保留完整版本號
    cuda_ver: float   # driver 所支援的最高 CUDA 版本


@dataclass
class Recommendation:
    tag: str               # e.g. "cu118" or "cpu"
    torch_spec: str        # version constraint, e.g. ">=2.1"
    torchvision_spec: str  # e.g. ">=0.16"
    driver_ok: bool        # 目前驅動是否符合
    min_driver: float      # 需要的最低驅動版本（0 = 無要求）
    upgrade_tag: str       # 如果更新驅動可獲得的更好 tag（可能同 tag）


# ── 偵測函式 ─────────────────────────────────────────────────────────────────

def _run(*cmd: str) -> str:
    try:
        return subprocess.check_output(list(cmd), stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ""


def detect_gpu() -> GpuInfo | None:
    out = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ).strip()
    if not out:
        return None

    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 3:
        return None

    driver_str = parts[2].strip()
    try:
        vram_mb = int(parts[1])
        driver  = float(driver_str)
    except ValueError:
        return None

    smi_full = _run("nvidia-smi")
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi_full)
    if not m:
        return None
    try:
        cuda_ver = float(m.group(1))
    except ValueError:
        return None

    return GpuInfo(name=parts[0], vram_mb=vram_mb, driver=driver,
                   driver_str=driver_str, cuda_ver=cuda_ver)


def detect_torch() -> tuple[str, str]:
    """返回 (torch_version, cuda_tag)，e.g. ('2.11.0+cpu', 'cpu')。"""
    try:
        import torch
        ver = torch.__version__
        m = re.search(r"\+(cu\d+|cpu)", ver)
        tag = m.group(1) if m else "cpu"
        return ver, tag
    except ImportError:
        return "未安裝", "cpu"


def recommend(gpu: GpuInfo | None) -> Recommendation:
    if gpu is None:
        return Recommendation(tag="cpu", torch_spec=">=2.1", torchvision_spec=">=0.16",
                              driver_ok=True, min_driver=0, upgrade_tag="cpu")

    # 找最高可用 tag（依現有驅動）
    best_current = None
    for cuda_min, tag, torch_spec, tv_spec, min_drv in _CUDA_TABLE:
        if gpu.cuda_ver >= cuda_min and gpu.driver >= min_drv:
            best_current = (tag, torch_spec, tv_spec, min_drv)
            break

    # 找最高可用 tag（若驅動全部更新）
    best_ideal = None
    for cuda_min, tag, _, _, _ in _CUDA_TABLE:
        if gpu.cuda_ver >= cuda_min:
            best_ideal = tag
            break
    if best_ideal is None:
        best_ideal = "cpu"

    if best_current:
        tag, torch_spec, tv_spec, min_drv = best_current
        return Recommendation(tag=tag, torch_spec=torch_spec, torchvision_spec=tv_spec,
                              driver_ok=True, min_driver=min_drv, upgrade_tag=best_ideal)
    else:
        # 驅動太舊，選驅動需求最低的 tag
        _, tag, torch_spec, tv_spec, min_drv = _CUDA_TABLE[-1]
        return Recommendation(tag=tag, torch_spec=torch_spec, torchvision_spec=tv_spec,
                              driver_ok=False, min_driver=min_drv, upgrade_tag=best_ideal)


# ── 報告 ─────────────────────────────────────────────────────────────────────

def _ok(cond: bool) -> str:
    return "[OK]" if cond else "[!!]"


def print_report(gpu: GpuInfo | None, rec: Recommendation,
                 torch_ver: str, torch_tag: str) -> None:
    W = 62
    print("=" * W)
    print("  PyTorch 環境評估報告")
    print("=" * W)
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  OS     : {platform.system()} {platform.release()}")
    print()

    if gpu:
        print(f"  GPU    : {gpu.name}  ({gpu.vram_mb} MB VRAM)")
        print(f"  Driver : {gpu.driver_str}")
        print(f"  CUDA   : {gpu.cuda_ver}  ← driver 支援上限")
    else:
        print("  GPU    : 未偵測到 NVIDIA GPU")
    print()

    tag_match = (torch_tag == rec.tag)
    print(f"  {_ok(True)}  目前 PyTorch : {torch_ver}  (tag={torch_tag})")
    print(f"  {_ok(rec.driver_ok)}  建議 tag     : {rec.tag}  (torch {rec.torch_spec})")

    if not rec.driver_ok:
        print()
        print(f"  [!!] 驅動版本不足（需 >= {rec.min_driver}，目前 {gpu.driver_str}）")
        print(f"    請至 https://www.nvidia.com/drivers 更新驅動。")
        if rec.upgrade_tag != rec.tag:
            print(f"    更新後可使用更高版本：{rec.upgrade_tag}")

    print()
    if tag_match:
        print(f"  {_ok(True)}  pyproject.toml 無需變更")
    else:
        print(f"  {_ok(False)}  需更新 pyproject.toml → 執行後請跑 uv sync")

    print("=" * W)


# ── pyproject.toml 更新 ───────────────────────────────────────────────────────

def _strip_uv_sections(text: str) -> str:
    """移除既有的 [tool.uv.sources] 及 [[tool.uv.indexes]] 區塊。"""
    # 移除 [tool.uv.sources] 整塊（到下一個 [ 開頭或 EOF）
    text = re.sub(
        r"\[tool\.uv\.sources\][^\[]*",
        "",
        text,
        flags=re.DOTALL,
    )
    # 移除所有 [[tool.uv.index]] 整塊
    text = re.sub(
        r"\[\[tool\.uv\.index\]\][^\[]*",
        "",
        text,
        flags=re.DOTALL,
    )
    return text.rstrip() + "\n"


def _update_torch_spec(text: str, torch_spec: str, tv_spec: str) -> str:
    """更新 dependencies 裡 torch 與 torchvision 的版本約束。"""
    # torch only（不影響 torchvision/torchaudio）
    text = re.sub(r'"torch([>=<~][^"]*)"', f'"torch{torch_spec}"', text)
    # torchvision
    text = re.sub(r'"torchvision([>=<~][^"]*)"', f'"torchvision{tv_spec}"', text)
    return text


def build_uv_block(rec: Recommendation) -> str:
    if rec.tag == "cpu":
        return ""   # CPU 版不需要特殊 index，PyPI 預設即可

    index_name = f"pytorch-{rec.tag}"
    url = _INDEX_URL.format(tag=rec.tag)
    lines = [
        "",
        "[tool.uv.sources]",
        f'torch      = {{ index = "{index_name}" }}',
        f'torchvision = {{ index = "{index_name}" }}',
        f'torchaudio  = {{ index = "{index_name}" }}',
        "",
        "[[tool.uv.index]]",
        f'name     = "{index_name}"',
        f'url      = "{url}"',
        "explicit = true",
        "",
    ]
    return "\n".join(lines)


def update_pyproject(rec: Recommendation, dry_run: bool = False) -> None:
    original = PYPROJECT.read_text(encoding="utf-8")

    updated = _strip_uv_sections(original)
    updated = _update_torch_spec(updated, rec.torch_spec, rec.torchvision_spec)
    block = build_uv_block(rec)
    if block:
        updated = updated.rstrip() + "\n" + block

    if dry_run:
        print("\n── pyproject.toml 預覽（dry-run，不寫入）──")
        print(updated)
        return

    if updated == original:
        print("\n[INFO] pyproject.toml 無變更。")
        return

    PYPROJECT.write_text(updated, encoding="utf-8")
    print(f"\n[OK] 已更新 {PYPROJECT}")
    print("     請執行：uv sync")
    if rec.tag != "cpu":
        print(f"     安裝後可透過 --device cuda 啟用 GPU 加速")


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="偵測 GPU/CUDA，更新 pyproject.toml")
    parser.add_argument("--dry-run", action="store_true",
                        help="僅顯示預覽，不寫入 pyproject.toml")
    args = parser.parse_args()

    gpu = detect_gpu()
    rec = recommend(gpu)
    torch_ver, torch_tag = detect_torch()

    print_report(gpu, rec, torch_ver, torch_tag)

    tag_match = (torch_tag == rec.tag)
    if not tag_match or args.dry_run:
        update_pyproject(rec, dry_run=args.dry_run)
    else:
        print("\n[INFO] 環境已是最佳狀態，pyproject.toml 無需變更。")


if __name__ == "__main__":
    main()
