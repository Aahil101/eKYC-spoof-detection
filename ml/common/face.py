"""Face pipeline: MediaPipe Face Landmarker (478 pts incl. iris) on every analysed frame.

Outputs per frame: landmarks (pixel coords, z in pixel units), bbox, head pose
(yaw/pitch/roll from the landmarker's facial transformation matrix), blendshape
scores (eyeBlink*, jawOpen) and aligned crops:
  * clip crop 224x224 (square bbox + margin) for M2/M3;
  * arcface crop 112x112 aligned to the standard 5-point template for M5.

Landmark index conventions follow the MediaPipe canonical face mesh (RESEARCH_NOTES [43,44,45]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

# --- MediaPipe face-mesh landmark indices --------------------------------------------
# Eye contour points for EAR (Soukupova & Cech, RESEARCH_NOTES [42]): p1..p6
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)    # subject's left eye
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_IRIS = (473, 474, 475, 476, 477)            # centre + 4 boundary points (refined iris)
RIGHT_IRIS = (468, 469, 470, 471, 472)
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
UPPER_LIP, LOWER_LIP = 13, 14
JAW_LEFT, JAW_RIGHT = 234, 454                   # face width at cheek/ear level
JAW_LOW_LEFT, JAW_LOW_RIGHT = 172, 397           # lower jaw (gonion-ish)
NOSE_LEFT, NOSE_RIGHT = 129, 358
LEFT_EYE_OUTER, LEFT_EYE_INNER = 263, 362
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
BROW_MID = 9
# rPPG skin ROIs (polygons on the mesh): forehead, left cheek, right cheek
ROI_FOREHEAD = (109, 10, 338, 337, 151, 108)
ROI_LEFT_CHEEK = (425, 411, 376, 352, 346, 347, 330)
ROI_RIGHT_CHEEK = (205, 187, 147, 123, 117, 118, 101)

# ArcFace 112x112 reference 5 points (InsightFace arcface_dst)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class FaceFrame:
    t_ms: float
    landmarks: np.ndarray | None          # (478, 3) pixel coords
    bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1
    yaw: float = float("nan")
    pitch: float = float("nan")
    roll: float = float("nan")
    blend: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.landmarks is not None


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from a 4x4 facial transformation matrix (rotation part)."""
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch, yaw, roll = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class FaceAnalyzer:
    """Wraps MediaPipe FaceLandmarker in VIDEO mode (temporal tracking across frames)."""

    def __init__(self, model_path: str | None = None, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        from ml.common.weights import path_for
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path or path_for("mediapipe_landmarker"))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def close(self) -> None:
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def process(self, frame_bgr: np.ndarray, t_ms: float) -> tuple[FaceFrame, int]:
        """Returns (largest face, number of faces detected)."""
        h, w = frame_bgr.shape[:2]
        ts = max(int(round(t_ms)), self._last_ts + 1)   # VIDEO mode needs strictly increasing ints
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        res = self._lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return FaceFrame(t_ms, None, None), 0
        sizes = []
        for lms in res.face_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            sizes.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
        i = int(np.argmax(sizes))
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[i]], dtype=np.float32)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        ff = FaceFrame(t_ms, pts, (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w - 1)), int(min(y1, h - 1))))
        if res.facial_transformation_matrixes:
            ff.yaw, ff.pitch, ff.roll = _euler_from_matrix(np.asarray(res.facial_transformation_matrixes[i]))
        if res.face_blendshapes:
            want = {"eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight"}
            ff.blend = {c.category_name: float(c.score) for c in res.face_blendshapes[i] if c.category_name in want}
        return ff, len(res.face_landmarks)

    def process_all(self, frames: Iterable[np.ndarray], times_ms: Iterable[float]) -> tuple[list[FaceFrame], int]:
        out, max_faces = [], 0
        for f, t in zip(frames, times_ms):
            ff, n = self.process(f, t)
            out.append(ff)
            max_faces = max(max_faces, n)
        return out, max_faces


# --- crops ----------------------------------------------------------------------------
def square_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.25,
                size: int = 224) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    M = np.array([[size / s, 0, size / 2 - cx * size / s], [0, size / s, size / 2 - cy * size / s]], np.float32)
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def five_points(lm: np.ndarray) -> np.ndarray:
    """Left eye, right eye (image order), nose tip, mouth corners - matching ARCFACE_DST order."""
    re = lm[list(RIGHT_IRIS[:1])][0, :2]   # subject right eye appears on image left
    le = lm[list(LEFT_IRIS[:1])][0, :2]
    return np.array([re, le, lm[NOSE_TIP, :2], lm[MOUTH_LEFT, :2], lm[MOUTH_RIGHT, :2]], np.float32)


def arcface_crop(frame: np.ndarray, lm: np.ndarray) -> np.ndarray:
    src = five_points(lm)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        raise ValueError("alignment failed")
    return cv2.warpAffine(frame, M, (112, 112), borderValue=0)


def roi_mask(shape: tuple[int, int], lm: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [lm[list(idx), :2].astype(np.int32)], 1)
    return m.astype(bool)


def face_info(faces: list[FaceFrame], frame_shape: tuple[int, int]) -> dict[str, float]:
    """Face-related quality features (fill the NaNs left in the quality vector)."""
    ok = [f for f in faces if f.ok]
    nan = float("nan")
    if not ok:
        return {"face_height_frac": nan, "head_yaw_mean": nan, "head_yaw_std": nan, "head_pitch_mean": nan}
    hs = [(f.bbox[3] - f.bbox[1]) / frame_shape[0] for f in ok]
    yaw = np.array([f.yaw for f in ok])
    pitch = np.array([f.pitch for f in ok])
    return {"face_height_frac": float(np.median(hs)), "head_yaw_mean": float(np.nanmean(yaw)),
            "head_yaw_std": float(np.nanstd(yaw)), "head_pitch_mean": float(np.nanmean(pitch))}
