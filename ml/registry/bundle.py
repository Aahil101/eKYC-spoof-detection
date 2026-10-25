"""Versioned model bundles (docs/ARCHITECTURE.md section 13).

Layout:
    models/bundles/<bundle_name>/
        manifest.yaml
        <module>/<files...>      e.g. M2/head.pt, M2/calibrator.json

manifest.yaml:
    name: kyc-bundle
    semver: 0.3.0
    created_at: ...
    modules:
      M2:
        artifacts: {head.pt: <sha256>, calibrator.json: <sha256>}
        training_run: {run_id, kaggle_kernel, git_sha, config_sha256, data_manifest_sha256, metrics}
    fusion: {...}  policy: {...}  reference_window: <id or null>

bundle_version = "<semver>+<first 10 hex of sha256 over all artifact hashes + fusion + policy>"
so any change to weights, calibrators, fusion or thresholds yields a new version.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER_BUNDLE_VERSION = "0.0.0+placeholder"


class BundleError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def compute_bundle_version(manifest: dict[str, Any]) -> str:
    material = {
        "modules": {m: sorted((manifest.get("modules") or {}).get(m, {}).get("artifacts", {}).items())
                    for m in sorted(manifest.get("modules") or {})},
        "fusion": manifest.get("fusion"),
        "policy": manifest.get("policy"),
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{manifest.get('semver', '0.0.0')}+{digest}"


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    p = bundle_dir / "manifest.yaml"
    if not p.exists():
        raise BundleError(f"no manifest.yaml in {bundle_dir}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def verify_bundle(bundle_dir: Path) -> str:
    """Check every artifact hash; return the bundle version. Raises on mismatch."""
    man = load_manifest(bundle_dir)
    for module, spec in (man.get("modules") or {}).items():
        for rel, expected in (spec.get("artifacts") or {}).items():
            f = bundle_dir / module / rel
            if not f.exists():
                raise BundleError(f"missing artifact {f}")
            actual = sha256_file(f)
            if actual != expected:
                raise BundleError(f"hash mismatch for {f}: {actual} != {expected}")
    version = compute_bundle_version(man)
    if man.get("bundle_version") and man["bundle_version"] != version:
        raise BundleError(f"manifest bundle_version {man['bundle_version']} != computed {version}")
    return version


def import_module_artifacts(bundle_dir: Path, module: str, artifact_paths: list[Path],
                            training_run: dict[str, Any] | None = None,
                            semver: str | None = None, name: str = "kyc-bundle") -> str:
    """Copy trained artifacts for one module into a bundle, rehash, and bump version.

    Used by `python -m training.import_weights` after downloading Kaggle outputs.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    man_path = bundle_dir / "manifest.yaml"
    man: dict[str, Any] = load_manifest(bundle_dir) if man_path.exists() else {
        "name": name, "semver": semver or "0.1.0", "modules": {}, "fusion": None, "policy": None,
        "reference_window": None,
    }
    if semver:
        man["semver"] = semver
    dest = bundle_dir / module
    dest.mkdir(exist_ok=True)
    artifacts: dict[str, str] = {}
    for src in artifact_paths:
        if not src.is_file():
            raise BundleError(f"artifact not found: {src}")
        target = dest / src.name
        shutil.copy2(src, target)
        artifacts[src.name] = sha256_file(target)
    man.setdefault("modules", {})[module] = {"artifacts": artifacts, "training_run": training_run or {}}
    man["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    man["bundle_version"] = compute_bundle_version(man)
    man_path.write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    return man["bundle_version"]


def resolve_active_bundle(bundles_root: Path, name: str | None) -> tuple[str, Path | None]:
    """Return (bundle_version, dir). Empty name -> placeholder bundle."""
    if not name:
        return PLACEHOLDER_BUNDLE_VERSION, None
    d = bundles_root / name
    return verify_bundle(d), d
