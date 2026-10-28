"""Per-module feature extraction from a PreparedSession / raw media.

These functions are used IDENTICALLY by (a) the Kaggle extraction jobs and (b) the API at
inference time, so training and serving features cannot drift apart. Feature names are
stable keys; vector-valued features (CLIP embedding, LBP histogram, spectrum) are kept as
named arrays and flattened by the heads.
"""
from __future__ import annotations

import numpy as np

from ml.common import face as F
from ml.features import biometrics as B
from ml.features import forensics as X
from ml.features import liveness as L

FEATURE_VERSION = "fv2"   # fv2: M1 document normalisation (resize+re-encode); bump when any extractor changes


class ClipEmbedder:
    """Frozen CLIP ViT-B/16 (OpenAI weights) image encoder - UnivFD-style backbone [21]."""

    name = "clip_vitb16_openai"

    def __init__(self, device: str | None = None):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # OpenAI CLIP weights were trained with QuickGELU; the plain "ViT-B-16" config would silently mismatch.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai",
                                                                                device=self.device)
        self.model.eval()

    def embed(self, crops_bgr: list[np.ndarray], batch: int = 32) -> np.ndarray:
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops_bgr), batch):
                x = self.torch.stack([self.preprocess(Image.fromarray(c[:, :, ::-1])) for c in crops_bgr[i:i + batch]])
                e = self.model.encode_image(x.to(self.device)).float()
                out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


# ------------------------------------------------------------------------------------------
def face_crops(frames, faces):
    clip, arc, idx = [], [], []
    for i, (fr, ff) in enumerate(zip(frames, faces)):
        if ff.ok:
            clip.append(F.square_crop(fr, ff.bbox, 0.25, 224))
            try:
                arc.append(F.arcface_crop(fr, ff.landmarks))
            except ValueError:
                pass
            idx.append(i)
    return clip, arc, idx


def m2_handcrafted(crops: list[np.ndarray], faces) -> tuple[dict, dict]:
    if not crops:
        return {"m2_n_faces": 0.0}, {}
    spec = np.mean([X.azimuthal_spectrum(c) for c in crops], axis=0)
    mot = L.motion_features(faces)
    return ({"m2_n_faces": float(len(crops)), "m2_landmark_jitter": mot["landmark_jitter"],
             "m2_hf_spectrum_tail": float(spec[-6:].mean())}, {"spectrum": spec.astype(np.float32)})


def m2_clip(clip_embs: np.ndarray) -> tuple[dict, dict]:
    nan = float("nan")
    if len(clip_embs) == 0:
        return {}, {}
    mean = clip_embs.mean(axis=0)
    sims = clip_embs @ (mean / np.linalg.norm(mean))
    d = np.linalg.norm(np.diff(clip_embs, axis=0), axis=1) if len(clip_embs) > 1 else np.array([nan])
    return ({"m2_clip_temporal_mean_step": float(np.nanmean(d)), "m2_clip_temporal_max_step": float(np.nanmax(d)),
             "m2_clip_consistency": float(sims.mean())}, {"clip_mean": mean.astype(np.float32)})


def m2_features(clip_embs: np.ndarray, crops: list[np.ndarray], faces) -> tuple[dict, dict]:
    s1, v1 = m2_handcrafted(crops, faces)
    s2, v2 = m2_clip(clip_embs)
    return {**s1, **s2}, {**v1, **v2}


# ------------------------------------------------------------------------------------------
# Shared video analysis (API + Kaggle jobs). Embeddings are computed afterwards (GPU-batched in jobs).
VIDEO_DEFAULTS = {"sample_frames": 16, "max_seconds": 12.0, "max_side": 720, "dense_side": 320,
                  "dense_seconds": 8.0, "dense_fps": 15.0, "arc_frames": 4,
                  # serving-only options (defaults reproduce the training-time extraction exactly):
                  "full_every_ms": 0.0,     # >0: keep a full-res frame only every N ms (long free recordings)
                  "frontal_pool": 0,        # >0: sample this many frames, keep the sample_frames most frontal ones
                  "actions": False,         # detect self-paced blink / head-turn actions over the whole video
                  "actions_fps": 8.0}


def _frontal_subset(frames, times, faces, k):
    """The k most frontal frames (training data - FF++/Celeb-DF/MSU - is mostly frontal), in temporal order.
    Frames without a face are used only if fewer than k frontal faces exist."""
    from ml.features.liveness import head_pose_series
    h, v = head_pose_series(faces)
    ok = np.isfinite(h) & np.isfinite(v)
    if ok.sum() == 0:
        sel = np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)
    else:
        h0, v0 = np.median(h[ok]), np.median(v[ok])
        dev = np.where(ok, np.abs(h - h0) / 0.18 + np.abs(v - v0) / 0.08, np.inf)
        sel = np.sort(np.argsort(dev, kind="stable")[:k])
    sel = sorted(set(int(i) for i in sel))
    return [frames[i] for i in sel], [times[i] for i in sel], [faces[i] for i in sel]


def _subsample_dense(frames, times, seconds, fps):
    keep, last = [], -1e9
    for i, t in enumerate(times):
        if t > seconds * 1000:
            break
        if t - last >= 1000.0 / fps - 1e-6:
            keep.append(i)
            last = t
    return [frames[i] for i in keep], [times[i] for i in keep]


def analyse_video(data: bytes, cfg: dict | None = None, challenge=None, audio_transcript=None,
                  asr_method: str = "none") -> dict:
    """Decode + faces + all embedding-free features. Returns a picklable dict (crops included)."""
    from ml.common.face import FaceAnalyzer, face_info
    from ml.common.media import decode_video
    from ml.common.quality import compute_quality
    c = {**VIDEO_DEFAULTS, **(cfg or {})}
    n_sample = int(c["frontal_pool"]) or c["sample_frames"]
    v = decode_video(data, n_sample, c["max_seconds"], c["max_side"], dense_max_side=c["dense_side"],
                     full_every_ms=float(c["full_every_ms"]))
    dframes, dtimes = _subsample_dense(v.dense_frames, v.dense_times_ms, c["dense_seconds"], c["dense_fps"])
    actions = None
    if c["actions"]:
        aframes, atimes = _subsample_dense(v.dense_frames, v.dense_times_ms, c["max_seconds"], c["actions_fps"])
        from ml.features.liveness import detect_actions
        with FaceAnalyzer() as fa:
            afaces, _ = fa.process_all(aframes, atimes)
        actions = detect_actions(afaces, atimes)
    v.dense_frames = None                         # free memory early (long recordings)
    if c.get("corruption"):                       # E5: controlled capture degradation before ALL features
        from ml.features.corruptions import corrupt_all
        kind, sev = c["corruption"]["kind"], int(c["corruption"]["severity"])
        v.frames = corrupt_all(v.frames, kind, sev, seed=1)
        dframes = corrupt_all(dframes, kind, sev, seed=2)
    with FaceAnalyzer() as fa:
        faces, n_faces = fa.process_all(v.frames, v.times_ms)
    if c["frontal_pool"]:
        v.frames, v.times_ms, faces = _frontal_subset(v.frames, v.times_ms, faces, c["sample_frames"])
    with FaceAnalyzer() as fa:
        dfaces, _ = fa.process_all(dframes, dtimes)
    crops, arc_crops, crop_idx = face_crops(v.frames, faces)
    # ArcFace: the most frontal frames only
    order = sorted([i for i in crop_idx if faces[i].ok and np.isfinite(faces[i].yaw)], key=lambda i: abs(faces[i].yaw))
    arc_sel = []
    for i in order[: c["arc_frames"]]:
        try:
            arc_sel.append(F.arcface_crop(v.frames[i], faces[i].landmarks))
        except ValueError:
            pass
    q = compute_quality(v.frames, v.meta, None, face_info(faces, v.frames[0].shape))
    s2, v2 = m2_handcrafted(crops, faces)
    s3, v3 = m3_features(v.frames, faces, crops)
    s4, x4 = m4_features(dframes, dfaces, dtimes, challenge, audio_transcript, asr_method)
    g, med_ratios = B.geometry_features(faces)
    lm_frame = next((i for i in range(len(faces)) if faces[i].ok), None)
    return {"meta": v.meta, "quality": q, "n_faces_max": n_faces, "face_frames": int(sum(f.ok for f in faces)),
            "m2": s2, "m2_vec": v2, "m3": s3, "m3_vec": v3, "m4": s4, "m4_expl": x4, "m5_geo": g,
            "m5_ratios": med_ratios, "clip_crops": crops, "arc_crops": arc_sel,
            "crop_times_ms": [float(v.times_ms[i]) for i in crop_idx], "actions": actions,
            "overlay": {"frame_index": lm_frame,
                        "landmarks": faces[lm_frame].landmarks[:, :2].round(1).tolist() if lm_frame is not None else None,
                        "frame_shape": list(v.frames[0].shape[:2])},
            "first_frame": v.frames[lm_frame] if lm_frame is not None else v.frames[0]}


def finish_embeddings(res: dict, clip: "ClipEmbedder", arc, keep_frame_embs: bool = False) -> dict:
    """Adds CLIP-based M2 features and ArcFace embeddings; drops crops. keep_frame_embs (serving only) keeps
    the per-frame CLIP embeddings for the frame-level deepfake timeline."""
    ce = clip.embed(res.pop("clip_crops"))
    if keep_frame_embs:
        res["clip_frame_embs"] = ce.astype(np.float32)
    s, vec = m2_clip(ce)
    res["m2"].update(s)
    res["m2_vec"].update(vec)
    ae = arc.embed(res.pop("arc_crops"))
    res["arc_embs"] = ae.astype(np.float32)
    return res


def m3_features(frames, faces, crops) -> tuple[dict, dict]:
    if not crops:
        return {"m3_n_faces": 0.0}, {}
    ida = [X.ida_features(c) for c in crops[:: max(1, len(crops) // 6)]]
    scal = {k: float(np.median([d[k] for d in ida])) for k in ida[0]}
    moi_face = [X.moire_features(c) for c in crops[:: max(1, len(crops) // 6)]]
    moi_full = [X.moire_features(f) for f in frames[:: max(1, len(frames) // 4)]]
    for k in moi_face[0]:
        scal[f"m3_face_{k}"] = float(np.median([m[k] for m in moi_face]))
        scal[f"m3_full_{k}"] = float(np.median([m[k] for m in moi_full]))
    # face-vs-background motion correlation (Anjos & Marcel [36]): printed photo / screen moves rigidly with scene
    scal.update(_face_bg_motion(frames, faces))
    scal["m3_n_faces"] = float(len(crops))
    lbp = np.mean([X.colour_lbp(c) for c in crops[:: max(1, len(crops) // 4)]], axis=0)
    return scal, {"lbp": lbp.astype(np.float32)}


def _face_bg_motion(frames, faces) -> dict[str, float]:
    import cv2
    nan = float("nan")
    fm, bm = [], []
    prev = None
    for fr, ff in zip(frames, faces):
        g = cv2.cvtColor(cv2.resize(fr, (160, int(160 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        if prev is not None and ff.ok:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            s = 160 / fr.shape[1]
            x0, y0, x1, y1 = (np.array(ff.bbox) * s).astype(int)
            mask = np.zeros(g.shape, bool)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True
            mag = np.linalg.norm(flow, axis=2)
            if mask.any() and (~mask).any():
                fm.append(mag[mask].mean())
                bm.append(mag[~mask].mean())
        prev = g
    if len(fm) < 4:
        return {"m3_face_bg_motion_corr": nan, "m3_bg_over_face_motion": nan}
    fm, bm = np.array(fm), np.array(bm)
    c = float(np.corrcoef(fm, bm)[0, 1]) if fm.std() > 0 and bm.std() > 0 else nan
    return {"m3_face_bg_motion_corr": c, "m3_bg_over_face_motion": float(bm.mean() / (fm.mean() + 1e-9))}


def m4_features(dense_frames, dense_faces, dense_times, challenge=None, transcript=None,
                asr_method="none") -> tuple[dict, dict]:
    rp, expl = L.rppg_features(dense_frames, dense_faces, dense_times)
    bl, ear_series = L.blink_features(dense_faces, dense_times)
    mo = L.motion_features(dense_faces)
    feats = {**rp, **bl, **mo}
    if challenge:
        cf, steps = L.challenge_compliance(challenge, dense_faces, dense_times, ear_series, transcript, asr_method)
        feats.update(cf)
        expl["challenge_steps"] = steps
    return feats, expl


def m1_features(doc_bgr: np.ndarray, mrz_lines: list[str] | None = None) -> tuple[dict, dict]:
    ela, ela_map = X.ela_features(doc_bgr)
    nr, noise_map = X.noise_residual_features(doc_bgr)
    feats = {**ela, **X.dct_features(doc_bgr), **nr, **X.glare_features(doc_bgr),
             **{f"m1_{k}": v for k, v in X.moire_features(doc_bgr).items()},
             **X.mrz_check(mrz_lines or [])}
    h, w = doc_bgr.shape[:2]
    feats["m1_aspect"] = float(w / h)
    vec = {"spectrum": X.azimuthal_spectrum(doc_bgr).astype(np.float32)}
    return feats, {"vec": vec, "ela_map": ela_map, "noise_map": noise_map}
