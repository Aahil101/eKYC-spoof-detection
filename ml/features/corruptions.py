"""Controlled capture corruptions for drift experiments (EXPERIMENT_PLAN E5a/E5b).

Applied to decoded BGR frames BEFORE all feature extraction, so the quality vector and every module see the
same degraded input, as they would for a real change of camera / network / lighting.
Severity 1..3 per type. 'jpeg' is per-frame JPEG re-encoding (a proxy for heavy H.264 compression).
"""
from __future__ import annotations

import cv2
import numpy as np

LEVELS = {
    "blur": {1: 1.0, 2: 2.0, 3: 3.5},                 # Gaussian sigma (px at 720p)
    "downscale": {1: 0.6, 2: 0.4, 3: 0.25},           # scale factor, then upsampled back
    "jpeg": {1: 40, 2: 20, 3: 10},                    # JPEG quality
    "lowlight": {1: 0.6, 2: 0.4, 3: 0.25},            # gain (with gamma 1.5 and shot noise)
    "noise": {1: 6.0, 2: 12.0, 3: 20.0},              # Gaussian sigma (8-bit levels)
}


def corrupt(img: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    v = LEVELS[kind][severity]
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), v * img.shape[0] / 720)
    if kind == "downscale":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * v)), max(8, int(h * v))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "lowlight":
        x = (img.astype(np.float32) / 255.0) ** 1.5 * v
        x = rng.poisson(x * 255.0 * 0.5) / (255.0 * 0.5)          # photon (shot) noise at low light
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if kind == "noise":
        return np.clip(img.astype(np.float32) + rng.normal(0, v, img.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def corrupt_all(frames: list[np.ndarray], kind: str | None, severity: int, seed: int = 0) -> list[np.ndarray]:
    if not kind or kind == "none":
        return frames
    rng = np.random.default_rng(seed)
    return [corrupt(f, kind, severity, rng) for f in frames]
