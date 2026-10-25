"""Per-session capture-quality vector q (docs/ARCHITECTURE.md section 5).

Used for (a) fusion gating and (b) drift detection. Only classical image statistics
here; face size / head pose are added once the face pipeline exists (milestone 3)
and are reported as NaN until then (never guessed).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

QUALITY_FEATURES: tuple[str, ...] = (
    "video_width", "video_height", "fps_measured", "frame_interval_jitter", "bitrate_bps",
    "bits_per_pixel", "duration_ms",
    "blur_laplacian_var", "tenengrad", "brightness_mean", "contrast_rms", "clipped_frac",
    "noise_sigma",
    "doc_width", "doc_height", "doc_blur_laplacian_var", "doc_jpeg_blockiness",
    "face_height_frac", "head_yaw_mean", "head_yaw_std", "head_pitch_mean",
)

NAN = float("nan")


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian (Pech-Pacheco et al., ICPR 2000)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def noise_sigma(gray: np.ndarray) -> float:
    """Robust noise estimate: MAD of a Laplacian-like high-pass residual (Immerkaer-style kernel)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    r = cv2.filter2D(gray.astype(np.float64), -1, k)
    return float(np.median(np.abs(r - np.median(r))) * 1.4826 / 6.0)


def jpeg_blockiness(gray: np.ndarray) -> float:
    """Ratio of mean absolute gradient across 8-px block boundaries to that inside blocks.

    ~1.0 for unblocked images, > 1 when 8x8 JPEG blocking is visible.
    """
    g = gray.astype(np.float64)
    dx = np.abs(np.diff(g, axis=1))
    if dx.shape[1] < 16:
        return NAN
    cols = np.arange(dx.shape[1])
    boundary = dx[:, (cols % 8) == 7].mean()
    inner = dx[:, (cols % 8) != 7].mean()
    return float(boundary / inner) if inner > 0 else NAN


def frame_stats(img: np.ndarray) -> dict[str, float]:
    g = _gray(img)
    gf = g.astype(np.float64)
    return {
        "blur_laplacian_var": laplacian_var(g),
        "tenengrad": tenengrad(g),
        "brightness_mean": float(gf.mean()),
        "contrast_rms": float(gf.std()),
        "clipped_frac": float(np.mean((g <= 2) | (g >= 253))),
        "noise_sigma": noise_sigma(g),
    }


def compute_quality(frames: list[np.ndarray], video_meta: dict[str, Any],
                    document: np.ndarray | None,
                    face_info: dict[str, float] | None = None) -> dict[str, float]:
    q: dict[str, float] = {k: NAN for k in QUALITY_FEATURES}
    if frames:
        per = [frame_stats(f) for f in frames]
        for k in per[0]:
            q[k] = float(np.median([p[k] for p in per]))
    w = float(video_meta.get("width", NAN))
    h = float(video_meta.get("height", NAN))
    fps = float(video_meta.get("fps_measured", NAN))
    br = float(video_meta.get("bitrate_bps", NAN))
    q.update(video_width=w, video_height=h, fps_measured=fps, bitrate_bps=br,
             frame_interval_jitter=float(video_meta.get("frame_interval_jitter", NAN)),
             duration_ms=float(video_meta.get("duration_ms", NAN)))
    if all(math.isfinite(x) and x > 0 for x in (w, h, fps, br)):
        q["bits_per_pixel"] = br / (w * h * fps)
    if document is not None:
        dg = _gray(document)
        q["doc_height"], q["doc_width"] = float(dg.shape[0]), float(dg.shape[1])
        q["doc_blur_laplacian_var"] = laplacian_var(dg)
        q["doc_jpeg_blockiness"] = jpeg_blockiness(dg)
    if face_info:
        for k, v in face_info.items():
            if k in q:
                q[k] = float(v)
    return q
