import math

import cv2
import numpy as np
import pytest

from backend.app.challenges import challenge_from_json, challenge_to_json, generate_challenge
from ml.common.interface import ModuleOutput
from ml.common.media import MediaError, decode_document, decode_video
from ml.common.quality import QUALITY_FEATURES, compute_quality, jpeg_blockiness, laplacian_var
from tests.conftest import make_video_bytes


def test_module_output_contract():
    with pytest.raises(ValueError):
        ModuleOutput(module="M2", status="ok", model_version="v")             # ok needs score
    with pytest.raises(ValueError):
        ModuleOutput(module="M2", status="placeholder", score=0.5, model_version="v")  # no invented scores
    with pytest.raises(ValueError):
        ModuleOutput(module="M2", status="ok", score=1.5, confidence=0.5, model_version="v")
    o = ModuleOutput(module="M2", status="ok", score=0.9, confidence=0.7, model_version="v")
    assert o.score == 0.9


@pytest.mark.parametrize("container", ["webm", "mp4"])
def test_decode_video(container):
    v = decode_video(make_video_bytes(n_frames=30, fps=15, container=container), sample_frames=8)
    assert len(v.frames) == 8
    assert v.meta["n_frames_decoded"] == 30
    assert abs(v.meta["fps_measured"] - 15) < 1.0
    assert v.meta["bitrate_bps"] > 0
    assert v.times_ms == sorted(v.times_ms)


def test_decode_video_rejects_garbage():
    with pytest.raises(MediaError):
        decode_video(b"not a video at all")


def test_decode_document_png_and_pdf(document_png):
    img = decode_document(document_png, "image/png")
    assert img.shape == (250, 400, 3)
    import pypdfium2 as pdfium
    import io
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(300, 200)
    buf = io.BytesIO()
    pdf.save(buf)
    page = decode_document(buf.getvalue(), "application/pdf", pdf_dpi=72)
    assert page.shape[:2] == (200, 300)
    with pytest.raises(MediaError):
        decode_document(b"\x00\x01garbage", "image/png")


def test_quality_vector_responds_to_blur():
    sharp = decode_video(make_video_bytes(blur=0), sample_frames=6)
    blurry = decode_video(make_video_bytes(blur=3), sample_frames=6)
    qs = compute_quality(sharp.frames, sharp.meta, None)
    qb = compute_quality(blurry.frames, blurry.meta, None)
    assert set(qs) == set(QUALITY_FEATURES)
    assert qb["blur_laplacian_var"] < qs["blur_laplacian_var"] * 0.5
    assert math.isnan(qs["face_height_frac"])        # not computed yet -> NaN, never guessed
    assert qs["bits_per_pixel"] > 0


def test_blockiness_detects_jpeg():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (256, 256), dtype=np.uint8), (0, 0), 2)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    jpg = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    assert jpeg_blockiness(jpg) > jpeg_blockiness(img) + 0.1
    assert laplacian_var(img) > 0


def test_challenge_random_and_well_formed():
    a = generate_challenge(7000, 3, 3)
    b = generate_challenge(7000, 3, 3)
    assert len(a) == 3 and sum(s.kind == "read_digits" for s in a) == 1
    digits = next(s for s in a if s.kind == "read_digits").payload["digits"]
    assert len(digits) == 3 and all(0 <= d <= 9 for d in digits)
    assert all(s.start_ms < s.end_ms <= 7000 for s in a)
    assert all(x.end_ms <= y.start_ms for x, y in zip(a, a[1:]))
    assert challenge_from_json(challenge_to_json(a)) == a
    # randomness: 20 draws should not all be identical
    scripts = {tuple((s.kind, s.prompt) for s in generate_challenge()) for _ in range(20)}
    assert len(scripts) > 1 or a != b
