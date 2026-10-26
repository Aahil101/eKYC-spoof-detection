"""Image-level features for CelebA-Spoof (M3) and SIDTD (M1). Shared by Kaggle jobs and the API."""
from __future__ import annotations

import cv2
import numpy as np

from ml.features import forensics as X
from ml.features.extract import m1_features


def celeba_spoof_bbox(img: np.ndarray, bb_txt: str) -> tuple[int, int, int, int] | None:
    """CelebA-Spoof *_BB.txt: 'x y w h score' in a 224x224 reference frame (dataset README)."""
    try:
        x, y, w, h = [float(v) for v in bb_txt.split()[:4]]
    except ValueError:
        return None
    H, W = img.shape[:2]
    sx, sy = W / 224.0, H / 224.0
    x0, y0 = int(max(0, x * sx)), int(max(0, y * sy))
    x1, y1 = int(min(W - 1, (x + w) * sx)), int(min(H - 1, (y + h) * sy))
    return (x0, y0, x1, y1) if x1 > x0 + 10 and y1 > y0 + 10 else None


def m3_image_features(img: np.ndarray, bbox) -> tuple[dict, dict, np.ndarray]:
    """Same feature names as the video M3 path (motion features are absent -> NaN in heads)."""
    from ml.common.face import square_crop
    crop = square_crop(img, bbox, 0.25, 224)
    s = X.ida_features(crop)
    mf, mfull = X.moire_features(crop), X.moire_features(img)
    for k in mf:
        s[f"m3_face_{k}"] = mf[k]
        s[f"m3_full_{k}"] = mfull[k]
    s["m3_n_faces"] = 1.0
    return s, {"lbp": X.colour_lbp(crop).astype(np.float32)}, crop


def m1_image_features(img: np.ndarray) -> tuple[dict, dict]:
    s, extra = m1_features(img)
    return s, extra["vec"]


def read_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img
