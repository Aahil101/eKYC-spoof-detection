"""Pretrained third-party weights: one place for URLs, licenses, local paths and hashes.

    python -m ml.common.weights fetch [--only mediapipe_landmarker ...]
    python -m ml.common.weights list

Files go to models/store/ (git-ignored) or to CODMAV_WEIGHTS_DIR. On Kaggle, attached
`codmav-weights-*` dirs (CODMAV_WEIGHTS_DIRS) are searched first, then the files are
downloaded (internet on). The first download writes the sha256 into models/store/weights.lock.json
and later loads are checked against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WeightSpec:
    key: str
    url: str
    filename: str            # file used by the code (inside the zip if archive)
    license: str
    archive: bool = False
    note: str = ""


SPECS: dict[str, WeightSpec] = {s.key: s for s in [
    WeightSpec("mediapipe_landmarker",
               "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
               "face_landmarker.task", "Apache-2.0", note="MediaPipe Face Landmarker (478 pts incl. iris, blendshapes)"),
    WeightSpec("mediapipe_detector",
               "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
               "blaze_face_short_range.tflite", "Apache-2.0", note="BlazeFace short-range detector"),
    WeightSpec("arcface_buffalo_l",
               "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
               "w600k_r50.onnx", "InsightFace models: NON-COMMERCIAL RESEARCH ONLY", archive=True,
               note="ArcFace R50 (WebFace600K) from the buffalo_l pack"),
    WeightSpec("vosk_small_en",
               "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
               "vosk-model-small-en-us-0.15", "Apache-2.0 (per alphacephei model list; verify)", archive=True,
               note="offline ASR for the read-digits challenge"),
]}


def store_dir() -> Path:
    return Path(os.environ.get("CODMAV_WEIGHTS_DIR", REPO_ROOT / "models" / "store"))


def _lock_path() -> Path:
    return store_dir() / "weights.lock.json"


def _read_lock() -> dict:
    p = _lock_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(p).as_posix().encode())
                h.update(f.read_bytes())
    else:
        with open(p, "rb") as f:
            while b := f.read(1 << 20):
                h.update(b)
    return h.hexdigest()


def _search_attached(spec: WeightSpec) -> Path | None:
    for d in filter(None, os.environ.get("CODMAV_WEIGHTS_DIRS", "").split(os.pathsep)):
        for cand in Path(d).rglob(spec.filename):
            return cand
    return None


def path_for(key: str, download: bool = True) -> Path:
    """Return a local path to the weight file/dir, downloading if allowed."""
    spec = SPECS[key]
    attached = _search_attached(spec)
    if attached:
        return attached
    target_dir = store_dir() / key
    hits = list(target_dir.rglob(spec.filename)) if target_dir.exists() else []
    if hits:
        p = hits[0]
    elif download:
        p = _download(spec, target_dir)
    else:
        raise FileNotFoundError(f"weights {key} not found; run: python -m ml.common.weights fetch --only {key}")
    lock = _read_lock()
    digest = lock.get(key, {}).get("sha256")
    if digest is None:
        lock[key] = {"sha256": _sha256(p), "url": spec.url, "license": spec.license, "path": str(p.relative_to(store_dir()))}
        _lock_path().parent.mkdir(parents=True, exist_ok=True)
        _lock_path().write_text(json.dumps(lock, indent=2))
    elif os.environ.get("CODMAV_VERIFY_WEIGHTS", "1") == "1" and p.is_file() and _sha256(p) != digest:
        raise RuntimeError(f"weights {key} hash mismatch vs weights.lock.json")
    return p


def _download(spec: WeightSpec, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / (Path(spec.url).name + ".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "codmav/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if spec.archive:
        with zipfile.ZipFile(tmp) as z:
            for m in z.infolist():
                dest = (target_dir / m.filename).resolve()
                if not str(dest).startswith(str(target_dir.resolve())):
                    raise RuntimeError("unsafe archive path")
            z.extractall(target_dir)
        tmp.unlink()
    else:
        tmp.replace(target_dir / spec.filename)
    hits = list(target_dir.rglob(spec.filename))
    if not hits:
        raise RuntimeError(f"{spec.filename} not found after downloading {spec.url}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "list"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for key, spec in SPECS.items():
        if a.only and key not in a.only:
            continue
        if a.cmd == "list":
            print(f"{key:22s} {spec.license:45s} {spec.url}")
        else:
            print(key, "->", path_for(key))


if __name__ == "__main__":
    main()
