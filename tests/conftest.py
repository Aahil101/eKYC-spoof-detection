"""Shared fixtures. Synthetic media only: no real faces or documents are used."""
from __future__ import annotations

import io

import av
import cv2
import numpy as np
import pytest


def make_video_bytes(n_frames: int = 30, fps: int = 15, size: tuple[int, int] = (160, 120),
                     container: str = "webm", blur: int = 0) -> bytes:
    """Encode a small synthetic moving-pattern video (VP8/WebM or H.264/MP4)."""
    buf = io.BytesIO()
    fmt, codec = ("webm", "libvpx") if container == "webm" else ("mp4", "libx264")
    with av.open(buf, mode="w", format=fmt) as out:
        st = out.add_stream(codec, rate=fps)
        st.width, st.height = size
        st.pix_fmt = "yuv420p"
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        for i in range(n_frames):
            img = np.roll(base, i * 2, axis=1)
            cv2.circle(img, (20 + i * 3 % size[0], size[1] // 2), 15, (255, 255, 255), -1)
            if blur:
                img = cv2.GaussianBlur(img, (0, 0), blur)
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
    return buf.getvalue()


def make_document_png(w: int = 400, h: int = 250) -> bytes:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (40, 60, 120), 3)
    cv2.putText(img, "SPECIMEN - SYNTHETIC", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "P<UTOSPECIMEN<<TEST<<<<<<<<<<<", (20, h - 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


@pytest.fixture(scope="session")
def video_webm() -> bytes:
    return make_video_bytes()


@pytest.fixture(scope="session")
def document_png() -> bytes:
    return make_document_png()
