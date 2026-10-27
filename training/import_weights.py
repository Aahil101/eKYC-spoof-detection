"""Import trained weights from a Kaggle run into a versioned model bundle.

Download the run output first (only the small artifacts, not checkpoints), e.g.:
    kaggle kernels output <owner>/codmav-<run-slug> -p downloads/<run-slug>
or download /kaggle/working/runs/<run_name>/ from the notebook's Output tab.

Then:
    python -m training.import_weights --run-dir downloads/<run-slug>/runs/<run_name> \
        --module M2 --bundle kyc-v0.3 --semver 0.3.0

This verifies artifact hashes against run_manifest.json, copies artifacts to
models/bundles/<bundle>/<module>/, records the training run (git sha, config sha,
kaggle kernel, metrics) in manifest.yaml and recomputes bundle_version.
Serve it with MODEL_BUNDLE=<bundle> in .env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.registry.bundle import BundleError, import_module_artifacts, sha256_file
from training.paths import REPO_ROOT


def import_run(run_dir: Path, module: str, bundle: str, semver: str | None = None,
               bundles_root: Path = REPO_ROOT / "models" / "bundles", allow_incomplete: bool = False,
               artifacts: list[str] | None = None) -> str:
    man_path = run_dir / "run_manifest.json"
    if not man_path.exists():
        raise BundleError(f"{man_path} not found - point --run-dir at runs/<run_name>/")
    man = json.loads(man_path.read_text())
    if man.get("status") != "completed" and not allow_incomplete:
        raise BundleError(f"run status is {man.get('status')!r}; refusing to import (use --allow-incomplete)")
    expected_module = man.get("config", {}).get("module")
    if expected_module and expected_module not in (module, "ALL"):
        raise BundleError(f"run trained module {expected_module}, not {module}")
    art_dir = run_dir / "artifacts"
    files = []
    for name, digest in (man.get("artifacts") or {}).items():
        if artifacts is not None and name not in artifacts:
            continue
        p = art_dir / name
        if not p.exists():
            raise BundleError(f"artifact missing: {p}")
        if sha256_file(p) != digest:
            raise BundleError(f"artifact hash mismatch: {p} (corrupted download?)")
        files.append(p)
    if not files:
        raise BundleError("run has no artifacts")
    training_run = {k: man.get(k) for k in ("run_name", "job", "mode", "git_sha", "config_sha256",
                                             "started_at", "finished_at", "metrics", "data", "status")}
    training_run["kaggle_kernel"] = man.get("env", {}).get("kaggle_kernel")
    return import_module_artifacts(bundles_root / bundle, module, files, training_run, semver)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--module", required=True, choices=["M1", "M2", "M3", "M4", "M5", "FUSION"])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--semver", default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--artifacts", nargs="*", default=None,
                    help="only these artifact files (e.g. head_M2.pkl); default: all")
    ap.add_argument("--bundles-root", type=Path, default=REPO_ROOT / "models" / "bundles")
    a = ap.parse_args()
    v = import_run(a.run_dir, a.module, a.bundle, a.semver, bundles_root=a.bundles_root,
                   allow_incomplete=a.allow_incomplete, artifacts=a.artifacts)
    print(f"bundle {a.bundle} -> version {v}")


if __name__ == "__main__":
    main()
