"""M4 liveness features: rPPG (POS + CHROM), blink (EAR), head-motion, challenge compliance.

rPPG: Wang et al. POS (RESEARCH_NOTES [39]) and de Haan & Jeanne CHROM [38] on mean skin RGB of
landmark ROIs (forehead, cheeks), band 0.7-3.0 Hz (42-180 bpm). Liveness evidence is the
spectral peak SNR and inter-ROI agreement (Li et al. [41]): replayed/printed faces give flat
or incoherent spectra. Blink: EAR [42]. Challenges are checked against the server script.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from ml.common import face as F

HR_BAND = (0.7, 3.0)


def ear(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    p = lm[list(idx), :2]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float(v / (2 * h)) if h > 0 else float("nan")


def roi_rgb_traces(frames: list[np.ndarray], faces: list[F.FaceFrame]) -> dict[str, np.ndarray]:
    """Mean RGB per ROI per frame (NaN where no face)."""
    rois = {"forehead": F.ROI_FOREHEAD, "lcheek": F.ROI_LEFT_CHEEK, "rcheek": F.ROI_RIGHT_CHEEK}
    out = {k: np.full((len(frames), 3), np.nan) for k in rois}
    for i, (img, ff) in enumerate(zip(frames, faces)):
        if not ff.ok:
            continue
        sx, sy = img.shape[1], img.shape[0]
        lm = ff.landmarks
        for k, idx in rois.items():
            m = F.roi_mask((sy, sx), lm, idx)
            if m.sum() >= 20:
                out[k][i] = img[m][:, ::-1].mean(axis=0)   # BGR -> RGB
    return out


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for c in range(x.shape[1]):
        col = x[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            return np.full_like(x, np.nan)
        col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    return x


def pos(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (Wang et al. 2017), sliding window 1.6 s, overlap-add."""
    n = len(rgb)
    w = max(int(1.6 * fs), 4)
    h = np.zeros(n)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    for t in range(0, n - w + 1):
        c = rgb[t:t + w].T
        mu = c.mean(axis=1, keepdims=True)
        if np.any(mu <= 0):
            continue
        cn = c / mu
        s = P @ cn
        std1 = s[1].std()
        hh = s[0] + (s[0].std() / std1 if std1 > 0 else 0.0) * s[1]
        h[t:t + w] += hh - hh.mean()
    return h


def chrom(rgb: np.ndarray) -> np.ndarray:
    mu = rgb.mean(axis=0)
    if np.any(mu <= 0):
        return np.zeros(len(rgb))
    rn = rgb / mu
    x = 3 * rn[:, 0] - 2 * rn[:, 1]
    y = 1.5 * rn[:, 0] + rn[:, 1] - 1.5 * rn[:, 2]
    a = x.std() / y.std() if y.std() > 0 else 0.0
    return x - a * y


def bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    if len(x) < 16 or fs <= 2 * HR_BAND[1]:
        return x - x.mean()
    b, a = signal.butter(2, [HR_BAND[0] / (fs / 2), HR_BAND[1] / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def spectrum_stats(x: np.ndarray, fs: float) -> dict[str, float]:
    """Peak frequency, SNR (peak +/-0.1 Hz and harmonic vs. rest of band, dB), peak prominence."""
    nan = float("nan")
    if len(x) < 32 or not np.isfinite(x).all() or x.std() == 0:
        return {"hr_bpm": nan, "snr_db": nan, "peak_ratio": nan}
    f, p = signal.periodogram(x * signal.windows.hann(len(x)), fs=fs, nfft=max(1024, len(x)))
    band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    if band.sum() < 3:
        return {"hr_bpm": nan, "snr_db": nan, "peak_ratio": nan}
    fb, pb = f[band], p[band]
    k = int(np.argmax(pb))
    f0 = fb[k]
    sig = (np.abs(fb - f0) <= 0.1) | (np.abs(fb - 2 * f0) <= 0.2)
    noise = pb[~sig].sum()
    snr = 10 * math.log10(pb[sig].sum() / noise) if noise > 0 else nan
    away = np.abs(fb - f0) > 0.15                      # second peak outside the main lobe
    second = pb[away].max() if away.any() else 0.0
    return {"hr_bpm": float(f0 * 60), "snr_db": float(snr), "peak_ratio": float(pb[k] / (second + 1e-12))}


def rppg_features(frames: list[np.ndarray], faces: list[F.FaceFrame], times_ms: list[float]) -> tuple[dict, dict]:
    """Returns (features, explanation payload with waveform for the UI)."""
    nan = float("nan")
    feats: dict[str, float] = {}
    t = np.asarray(times_ms) / 1000.0
    if len(t) < 32 or t[-1] - t[0] < 3.0:
        return {k: nan for k in ("rppg_pos_snr_db", "rppg_chrom_snr_db", "rppg_hr_bpm", "rppg_roi_hr_spread",
                                 "rppg_roi_coherence", "rppg_peak_ratio", "rppg_valid_frac")}, {}
    fs = (len(t) - 1) / (t[-1] - t[0])
    traces = roi_rgb_traces(frames, faces)
    valid = np.mean([np.isfinite(v[:, 0]).mean() for v in traces.values()])
    sigs, hrs = {}, []
    for k, v in traces.items():
        v = _interp_nan(v)
        if not np.isfinite(v).all():
            continue
        s = bandpass(pos(v, fs), fs)
        sigs[k] = s
        hrs.append(spectrum_stats(s, fs)["hr_bpm"])
    if not sigs:
        return {k: nan for k in ("rppg_pos_snr_db", "rppg_chrom_snr_db", "rppg_hr_bpm", "rppg_roi_hr_spread",
                                 "rppg_roi_coherence", "rppg_peak_ratio", "rppg_valid_frac")}, {}
    allrgb = _interp_nan(np.nanmean(np.stack([traces[k] for k in sigs]), axis=0))
    s_pos = bandpass(pos(allrgb, fs), fs)
    s_chr = bandpass(chrom(allrgb), fs)
    st_pos, st_chr = spectrum_stats(s_pos, fs), spectrum_stats(s_chr, fs)
    keys = list(sigs)
    cors = [np.corrcoef(sigs[a], sigs[b])[0, 1] for i, a in enumerate(keys) for b in keys[i + 1:]]
    feats.update(rppg_pos_snr_db=st_pos["snr_db"], rppg_chrom_snr_db=st_chr["snr_db"], rppg_hr_bpm=st_pos["hr_bpm"],
                 rppg_roi_hr_spread=float(np.nanstd(hrs)) if len(hrs) > 1 else nan,
                 rppg_roi_coherence=float(np.nanmean(cors)) if cors else nan,
                 rppg_peak_ratio=st_pos["peak_ratio"], rppg_valid_frac=float(valid))
    step = max(1, len(s_pos) // 300)
    expl = {"rppg_waveform": {"t": [round(float(x - t[0]), 3) for x in t[::step]],
                              "pos": [round(float(x), 4) for x in s_pos[::step]], "fs": fs}}
    return feats, expl


def count_blinks(e: np.ndarray, times_ms, base: float, min_gap_ms: float = 250.0) -> int:
    """Hysteresis blink counter: close below 0.70*base, reopen above 0.85*base, >= min_gap between blinks."""
    blinks, closed, last = 0, False, -1e9
    for v, t in zip(e, times_ms):
        if not np.isfinite(v):
            continue
        if not closed and v < 0.70 * base:
            closed = True
            if t - last >= min_gap_ms:
                blinks += 1
                last = t
        elif closed and v > 0.85 * base:
            closed = False
    return blinks


def blink_features(faces: list[F.FaceFrame], times_ms: list[float]) -> tuple[dict, np.ndarray]:
    e = np.array([0.5 * (ear(f.landmarks, F.LEFT_EYE_EAR) + ear(f.landmarks, F.RIGHT_EYE_EAR)) if f.ok else np.nan
                  for f in faces])
    nan = float("nan")
    if np.isfinite(e).sum() < 10:
        return {"ear_mean": nan, "ear_std": nan, "blink_count": nan, "blink_rate_per_min": nan}, e
    base = np.nanpercentile(e, 80)     # relative threshold: robust to per-person eye shape
    blinks = count_blinks(e, times_ms, base)
    dur_min = (times_ms[-1] - times_ms[0]) / 60000.0
    return {"ear_mean": float(np.nanmean(e)), "ear_std": float(np.nanstd(e)), "blink_count": float(blinks),
            "blink_rate_per_min": float(blinks / dur_min) if dur_min > 0 else nan}, e


def motion_features(faces: list[F.FaceFrame]) -> dict[str, float]:
    ok = [f for f in faces if f.ok]
    nan = float("nan")
    if len(ok) < 5:
        return {"yaw_range": nan, "pitch_range": nan, "landmark_jitter": nan, "nonrigid_motion": nan}
    yaw = np.array([f.yaw for f in ok])
    pitch = np.array([f.pitch for f in ok])
    L = np.stack([f.landmarks[:, :2] for f in ok])
    scale = np.linalg.norm(L[:, F.JAW_LEFT] - L[:, F.JAW_RIGHT], axis=1, keepdims=True)[..., None]
    Ln = (L - L.mean(axis=1, keepdims=True)) / np.maximum(scale, 1)
    jitter = float(np.median(np.abs(np.diff(Ln, axis=0)).mean(axis=(1, 2))))
    # non-rigid motion: residual after best rigid (Procrustes) alignment to the first frame -
    # a flat photo moved in front of the camera has ~zero non-rigid deformation
    ref = Ln[0]
    res = []
    for X in Ln[1:]:
        u, _, vt = np.linalg.svd(X.T @ ref)
        R = u @ vt
        res.append(np.linalg.norm(X @ R - ref) / math.sqrt(len(ref)))
    return {"yaw_range": float(np.nanpercentile(yaw, 95) - np.nanpercentile(yaw, 5)),
            "pitch_range": float(np.nanpercentile(pitch, 95) - np.nanpercentile(pitch, 5)),
            "landmark_jitter": jitter, "nonrigid_motion": float(np.median(res))}


def challenge_compliance(challenge: list[Any], faces: list[F.FaceFrame], times_ms: list[float],
                         ear_series: np.ndarray, transcript: str | None,
                         asr_method: str) -> tuple[dict[str, float], list[dict]]:
    """Per-step pass/fail from landmarks/blendshapes/ASR within each step's time window (+/-500 ms)."""
    t = np.asarray(times_ms)
    results, passed = [], []
    words = {"zero": 0, "oh": 0, "one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "ate": 8, "nine": 9}
    for st in challenge:
        win = (t >= st.start_ms - 500) & (t <= st.end_ms + 500)
        fw = [f for f, w in zip(faces, win) if w and f.ok]
        ok, evidence = False, {}
        if st.kind in ("turn_left", "turn_right", "look_up") and fw:
            base = np.nanmedian([f.yaw for f in faces if f.ok][:5])
            yaw = np.array([f.yaw for f in fw]) - base
            pitch = np.array([f.pitch for f in fw])
            if st.kind == "look_up":
                # MediaPipe pitch sign: looking up is negative in our convention; accept either sign w/ magnitude
                val = float(np.nanmax(np.abs(pitch - np.nanmedian(pitch[:3]))))
                ok = val >= 10
            else:
                # Raw video is not mirrored: subject turning to THEIR left -> positive yaw here.
                val = float(np.nanmax(yaw) if st.kind == "turn_left" else -np.nanmin(yaw))
                ok = val >= 15
            evidence = {"delta_deg": round(val, 1)}
        elif st.kind == "blink_twice":
            tw = t[win]
            n = count_blinks(ear_series[win], tw, float(np.nanpercentile(ear_series, 80))) \
                if np.isfinite(ear_series[win]).sum() > 3 else 0
            ok, evidence = n >= 2, {"blinks": n}
        elif st.kind == "smile" and fw:
            s = max(0.5 * (f.blend.get("mouthSmileLeft", 0) + f.blend.get("mouthSmileRight", 0)) for f in fw)
            ok, evidence = s >= 0.4, {"smile": round(s, 2)}
        elif st.kind == "read_digits":
            target = st.payload.get("digits", [])
            if transcript is not None:
                heard = [words[w] for w in transcript.lower().split() if w in words]
                heard += [int(c) for c in transcript if c.isdigit()]
                hit = sum(1 for d in target if d in heard)
                ok, evidence = hit == len(target), {"method": asr_method, "heard": heard, "target": target}
            elif fw:
                jaw = np.array([f.blend.get("jawOpen", 0) for f in fw])
                act = float(jaw.std() + (jaw > 0.15).mean())
                ok, evidence = act >= 0.12, {"method": "mouth_motion_fallback", "activity": round(act, 3)}
        results.append({"kind": st.kind, "prompt": st.prompt, "passed": bool(ok), **evidence})
        passed.append(ok)
    feats = {"challenge_pass_frac": float(np.mean(passed)) if passed else float("nan"),
             "challenge_all_passed": float(all(passed)) if passed else float("nan")}
    return feats, results


def transcribe_vosk(wav_pcm16: bytes, sample_rate: int) -> str | None:
    """Offline ASR restricted to digit words (grammar) -> robust and fast. None if Vosk unavailable."""
    try:
        import json

        import vosk

        from ml.common.weights import path_for
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(path_for("vosk_small_en", download=False)))
        grammar = json.dumps(["zero oh one two three four five six seven eight nine", "[unk]"])
        rec = vosk.KaldiRecognizer(model, sample_rate, grammar)
        rec.AcceptWaveform(wav_pcm16)
        return json.loads(rec.FinalResult()).get("text", "")
    except Exception:
        return None


def decode_audio(data: bytes, rate: int = 16000) -> bytes | None:
    """Mono s16 PCM from the uploaded webm/mp4 (None if no audio track)."""
    import io

    import av
    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.audio:
                return None
            res = av.AudioResampler(format="s16", layout="mono", rate=rate)
            out = bytearray()
            for fr in c.decode(c.streams.audio[0]):
                for r in res.resample(fr):
                    out += r.to_ndarray().tobytes()
            return bytes(out)
    except Exception:
        return None
