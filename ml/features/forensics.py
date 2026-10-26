"""Hand-crafted image forensics shared by M1/M2/M3.

* azimuthal power spectrum (Frank et al. [23]) - generator upsampling artefacts (M2);
* moire / periodic-peak energy in the 2-D FFT (Patel et al. [33]) - screen recapture (M1, M3);
* image distortion analysis (Wen et al. [34]): specular ratio, blur, chromatic moments, colour diversity;
* colour LBP histograms in HSV and YCrCb (Boulkenafet et al. [35]);
* ELA multi-quality (Krawetz [1]), JPEG DCT blockiness/double-quantisation periodicity ([2]),
  block-wise noise-residual inconsistency ([3]).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, stats


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def azimuthal_spectrum(img: np.ndarray, bins: int = 24, size: int = 224) -> np.ndarray:
    g = cv2.resize(_gray(img), (size, size)).astype(np.float64)
    g -= g.mean()
    f = np.fft.fftshift(np.abs(np.fft.fft2(g * np.outer(np.hanning(size), np.hanning(size)))) ** 2)
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - size / 2, xx - size / 2)
    rmax = size / 2
    edges = np.linspace(0, rmax, bins + 1)
    lp = np.log1p(f)
    prof = np.array([lp[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return (prof - prof.mean()) / (prof.std() + 1e-9)


def moire_features(img: np.ndarray, size: int = 256) -> dict[str, float]:
    """Strength of isolated periodic peaks in the mid/high-frequency band (screen pixel grid / moire)."""
    g = cv2.resize(_gray(img), (size, size)).astype(np.float64)
    g -= g.mean()
    F = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(g * np.outer(np.hanning(size), np.hanning(size))))))
    yy, xx = np.indices(F.shape)
    r = np.hypot(yy - size / 2, xx - size / 2) / (size / 2)
    band = (r > 0.25) & (r < 0.95)
    bg = ndimage.median_filter(F, size=9)
    resid = (F - bg)[band]
    hf = F[r > 0.5].mean() - F[(r > 0.05) & (r < 0.25)].mean()
    return {"moire_peak_max": float(resid.max()), "moire_peak_p999": float(np.percentile(resid, 99.9)),
            "moire_peak_count": float((resid > 1.5).sum()), "hf_energy_ratio": float(hf)}


def ida_features(img: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    v, s = hsv[..., 2] / 255.0, hsv[..., 1] / 255.0
    spec = (v > 0.9) & (s < 0.2)
    g = _gray(img)
    blur_lap = cv2.Laplacian(g, cv2.CV_64F).var()
    # blur via re-blur difference (Crete et al.-style): sharp images lose more under extra blur
    gb = cv2.blur(g, (9, 9)).astype(np.float64)
    reblur = float(np.abs(np.diff(g.astype(np.float64), axis=1)).mean() / (np.abs(np.diff(gb, axis=1)).mean() + 1e-9))
    feats = {"ida_specular_frac": float(spec.mean()),
             "ida_specular_intensity": float(v[spec].mean()) if spec.any() else 0.0,
             "ida_blur_laplacian": float(blur_lap), "ida_reblur_ratio": reblur}
    for name, ch in (("h", hsv[..., 0] / 180.0), ("s", s), ("v", v)):
        x = ch.ravel()
        feats[f"ida_{name}_mean"] = float(x.mean())
        feats[f"ida_{name}_std"] = float(x.std())
        feats[f"ida_{name}_skew"] = float(stats.skew(x))
    q = (img // 32).reshape(-1, 3).astype(np.int32)
    codes = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    feats["ida_colour_diversity"] = float(len(np.unique(codes)) / 512.0)
    return feats


def _lbp(ch: np.ndarray) -> np.ndarray:
    c = ch.astype(np.int16)
    p = np.pad(c, 1, mode="edge")
    code = np.zeros_like(c, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offs):
        nb = p[1 + dy:1 + dy + c.shape[0], 1 + dx:1 + dx + c.shape[1]]
        code |= ((nb >= c).astype(np.uint8) << bit)
    return code


# uniform-pattern mapping (59 bins) for 8-neighbour LBP
_U = np.zeros(256, np.int32)
_k = 0
for _v in range(256):
    _b = [(_v >> i) & 1 for i in range(8)]
    if sum(_b[i] != _b[(i + 1) % 8] for i in range(8)) <= 2:
        _U[_v] = _k
        _k += 1
_U[_U == 0] = 0
_NONUNI = _k
for _v in range(256):
    _b = [(_v >> i) & 1 for i in range(8)]
    if sum(_b[i] != _b[(i + 1) % 8] for i in range(8)) > 2:
        _U[_v] = _NONUNI


def colour_lbp(img: np.ndarray, size: int = 128) -> np.ndarray:
    im = cv2.resize(img, (size, size))
    out = []
    for conv in (cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YCrCb):
        cs = cv2.cvtColor(im, conv)
        for c in range(3):
            h = np.bincount(_U[_lbp(cs[..., c])].ravel(), minlength=_NONUNI + 1).astype(np.float64)
            out.append(h / h.sum())
    return np.concatenate(out)  # 6 * 59 = 354


# ---------------------------- document-specific ---------------------------------------
def ela_features(img: np.ndarray, qualities=(95, 90, 75)) -> tuple[dict[str, float], np.ndarray]:
    feats = {}
    ela_map = None
    for q in qualities:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        re = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        d = np.abs(img.astype(np.int16) - re.astype(np.int16)).max(axis=2).astype(np.float64)
        blocks = d[: d.shape[0] // 16 * 16, : d.shape[1] // 16 * 16]
        bm = blocks.reshape(blocks.shape[0] // 16, 16, -1, 16).mean(axis=(1, 3))
        feats[f"ela_q{q}_mean"] = float(d.mean())
        feats[f"ela_q{q}_p99"] = float(np.percentile(d, 99))
        feats[f"ela_q{q}_block_cv"] = float(bm.std() / (bm.mean() + 1e-9))
        feats[f"ela_q{q}_block_maxz"] = float((bm.max() - bm.mean()) / (bm.std() + 1e-9))
        if q == 90:
            ela_map = d
    return feats, ela_map


def dct_features(img: np.ndarray) -> dict[str, float]:
    """Periodicity of 8x8 DCT coefficient histograms (double quantisation leaves comb patterns)."""
    g = _gray(img).astype(np.float32) - 128
    h, w = g.shape[0] // 8 * 8, g.shape[1] // 8 * 8
    blocks = g[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    if len(blocks) > 4000:
        blocks = blocks[np.random.default_rng(0).choice(len(blocks), 4000, replace=False)]
    coefs = np.array([cv2.dct(b) for b in blocks])
    feats = {}
    peaks = []
    for (u, v) in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0)):
        c = np.round(coefs[:, u, v]).astype(int)
        c = c[np.abs(c) <= 50]
        if len(c) < 50:
            continue
        hist = np.bincount(c + 50, minlength=101).astype(np.float64)
        spec = np.abs(np.fft.rfft(hist - hist.mean()))
        peaks.append(spec[2:].max() / (spec[2:].mean() + 1e-9))
    feats["dct_hist_periodicity"] = float(np.mean(peaks)) if peaks else float("nan")
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    feats["dct_blockiness"] = float(dx[:, cols % 8 == 7].mean() / (dx[:, cols % 8 != 7].mean() + 1e-9))
    return feats


def noise_residual_features(img: np.ndarray, block: int = 32) -> tuple[dict[str, float], np.ndarray]:
    g = _gray(img).astype(np.float64)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float64)
    h, w = res.shape[0] // block * block, res.shape[1] // block * block
    if h == 0 or w == 0:
        return {"noise_block_cv": float("nan"), "noise_block_maxz": float("nan"), "noise_block_kurt": float("nan")}, res
    b = res[:h, :w].reshape(h // block, block, w // block, block)
    sd = b.std(axis=(1, 3))
    return {"noise_block_cv": float(sd.std() / (sd.mean() + 1e-9)),
            "noise_block_maxz": float((sd.max() - sd.mean()) / (sd.std() + 1e-9)),
            "noise_block_kurt": float(stats.kurtosis(sd.ravel()))}, sd


def glare_features(img: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] > 240) & (hsv[..., 1] < 40)
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    largest = float(st[1:, cv2.CC_STAT_AREA].max() / m.size) if n > 1 else 0.0
    return {"glare_frac": float(m.mean()), "glare_largest_blob": largest}


def mrz_check(lines: list[str]) -> dict[str, float]:
    """ICAO 9303 [12] 7-3-1 check digits for TD3 (2x44) or TD1 (3x30). NaN if no MRZ given."""
    def cd(s: str) -> int:
        val = {**{str(i): i for i in range(10)}, **{chr(65 + i): 10 + i for i in range(26)}, "<": 0}
        return sum(val.get(c, 0) * w for c, w in zip(s, [7, 3, 1] * 20)) % 10

    nan = float("nan")
    if not lines:
        return {"mrz_present": 0.0, "mrz_checks_pass_frac": nan}
    L = [x.strip().upper().replace(" ", "") for x in lines]
    checks = []
    if len(L) >= 2 and len(L[-1]) == 44:                       # TD3 passport
        l2 = L[-1]
        checks = [(l2[0:9], l2[9]), (l2[13:19], l2[19]), (l2[21:27], l2[27])]
    elif len(L) >= 3 and len(L[0]) == 30:                      # TD1 ID card
        l1, l2 = L[0], L[1]
        checks = [(l1[5:14], l1[14]), (l2[0:6], l2[6]), (l2[8:14], l2[14])]
    ok = [c.isdigit() and cd(f) == int(c) for f, c in checks]
    return {"mrz_present": 1.0, "mrz_checks_pass_frac": float(np.mean(ok)) if ok else nan}
