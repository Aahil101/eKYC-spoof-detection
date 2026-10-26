"""Import every head_Mx.pkl from a train_heads run into one bundle (one module per artifact).

    python -m training.import_heads --run-dir downloads/th/runs/train_heads --bundle kyc-v0.3 --semver 0.3.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.import_weights import import_run
from training.paths import REPO_ROOT


def import_all(run_dir: Path, bundle: str, semver: str | None, bundles_root: Path) -> str:
    v = ""
    for f in sorted((run_dir / "artifacts").glob("head_M*.pkl")):
        module = f.stem.split("_")[1]
        v = import_run(run_dir, module, bundle, semver, bundles_root, artifacts=[f.name])
        print(f"{module}: {f.name} -> {bundle} {v}")
    if not v:
        raise SystemExit("no head_M*.pkl artifacts found")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--semver", default=None)
    ap.add_argument("--bundles-root", type=Path, default=REPO_ROOT / "models" / "bundles")
    a = ap.parse_args()
    import_all(a.run_dir, a.bundle, a.semver, a.bundles_root)


if __name__ == "__main__":
    main()
