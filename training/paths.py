"""Environment-aware paths: the same code runs on the laptop (smoke) and on Kaggle (full)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or KAGGLE_WORKING.is_dir()


def output_root() -> Path:
    """Where runs are written. Override with CODMAV_OUTPUT_ROOT."""
    if env := os.environ.get("CODMAV_OUTPUT_ROOT"):
        return Path(env)
    return KAGGLE_WORKING / "runs" if is_kaggle() else REPO_ROOT / "runs"


def input_roots() -> list[Path]:
    """Where datasets / previous run outputs are looked up, in priority order."""
    roots: list[Path] = []
    if env := os.environ.get("CODMAV_INPUT_ROOTS"):
        roots += [Path(p) for p in env.split(os.pathsep) if p]
    if is_kaggle():
        roots.append(KAGGLE_INPUT)
    roots.append(REPO_ROOT / "data" / "raw")
    return roots
