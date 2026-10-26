"""Video and document decoding.

PyAV is used for video because it exposes codec, bitrate and per-frame timestamps,
which the quality vector needs. Browser MediaRecorder WebM often lacks a duration
header, so frames are decoded once and sampled afterwards.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import av
import cv2
import numpy as np


class MediaError(ValueError):
    """Raised for unreadable or unsupported uploads."""


@dataclass
class DecodedVideo:
    frames: list[np.ndarray]        # sampled BGR frames
    times_ms: list[float]
    meta: dict[str, Any]
    dense_frames: list[np.ndarray] | None = None   # every decoded frame, downscaled (rPPG / blink / motion)
    dense_times_ms: list[float] | None = None


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def decode_video(data: bytes, sample_frames: int = 16, max_seconds: float = 12.0,
                 max_side: int = 720, dense_max_side: int = 0) -> DecodedVideo:
    try:
        container = av.open(io.BytesIO(data))
    except Exception as e:  # PyAV raises several error types
        raise MediaError(f"cannot open video: {e}") from e
    try:
        if not container.streams.video:
            raise MediaError("no video stream")
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        all_frames: list[np.ndarray] = []
        times: list[float] = []
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base) * 1000.0 if frame.pts is not None else len(times) * 33.3
            if t > max_seconds * 1000.0:
                break
            all_frames.append(_resize_max_side(frame.to_ndarray(format="bgr24"), max_side))
            times.append(t)
        bitrate = container.bit_rate or stream.bit_rate
        width, height = stream.codec_context.width, stream.codec_context.height
    except MediaError:
        raise
    except Exception as e:
        raise MediaError(f"cannot decode video: {e}") from e
    finally:
        container.close()

    if len(all_frames) < 2:
        raise MediaError("video has fewer than 2 decodable frames")

    t0 = times[0]
    times = [t - t0 for t in times]
    duration_ms = times[-1]
    dts = np.diff(times)
    fps = float(1000.0 / np.median(dts)) if np.median(dts) > 0 else 0.0
    jitter = float(np.std(dts) / np.mean(dts)) if np.mean(dts) > 0 else 0.0
    if not bitrate and duration_ms > 0:
        bitrate = int(len(data) * 8 / (duration_ms / 1000.0))

    n = min(sample_frames, len(all_frames))
    idx = np.linspace(0, len(all_frames) - 1, n).round().astype(int)
    meta = {
        "codec": codec,
        "width": int(width or all_frames[0].shape[1]),
        "height": int(height or all_frames[0].shape[0]),
        "n_frames_decoded": len(all_frames),
        "duration_ms": float(duration_ms),
        "fps_measured": fps,
        "frame_interval_jitter": jitter,
        "bitrate_bps": float(bitrate or 0),
        "bytes": len(data),
    }
    dense = [_resize_max_side(f, dense_max_side) for f in all_frames] if dense_max_side else None
    return DecodedVideo(frames=[all_frames[i] for i in idx], times_ms=[times[i] for i in idx], meta=meta,
                        dense_frames=dense, dense_times_ms=list(times) if dense else None)


def decode_document(data: bytes, content_type: str | None, max_side: int = 2000, pdf_dpi: int = 200) -> np.ndarray:
    """Return a BGR image. PDFs: first page only, rasterized with pdfium."""
    is_pdf = (content_type == "application/pdf") or data[:5] == b"%PDF-"
    if is_pdf:
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(data)
            if len(pdf) == 0:
                raise MediaError("PDF has no pages")
            pil = pdf[0].render(scale=pdf_dpi / 72.0).to_pil().convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        except MediaError:
            raise
        except Exception as e:
            raise MediaError(f"cannot read PDF: {e}") from e
    else:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise MediaError("cannot decode document image")
    return _resize_max_side(img, max_side)
