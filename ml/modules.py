"""Real KYC modules M1-M5 (milestone 3).

Each module loads `head_Mx.pkl` (a HeadArtifact) from the active model bundle. If the bundle has
no head for a module, it falls back to PlaceholderModule (status="placeholder", no score).
Features are computed by the SAME functions used in the Kaggle extraction jobs (ml.features.*),
once per session in `analyse_session`, and every module reads from that shared analysis.
"""
from __future__ import annotations

import base64
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.common.interface import ModuleOutput, PreparedSession
from ml.common.placeholders import PlaceholderModule

_CACHE: dict[str, Any] = {}


def _clip():
    if "clip" not in _CACHE:
        from ml.features.extract import ClipEmbedder
        _CACHE["clip"] = ClipEmbedder()
    return _CACHE["clip"]


def _arc():
    if "arc" not in _CACHE:
        from ml.features.biometrics import FaceEmbedder
        _CACHE["arc"] = FaceEmbedder()
    return _CACHE["arc"]


def _png_b64(img: np.ndarray, max_side: int = 320) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, enc = cv2.imencode(".png", img)
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


def _heatmap_b64(m: np.ndarray, base: np.ndarray) -> str:
    m = np.asarray(m, np.float64)
    m = (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m) + 1e-9)
    hm = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    return _png_b64(cv2.addWeighted(base, 0.55, hm, 0.45, 0))


# ------------------------------------------------------------------------------------- analysis
def analyse_session(video_bytes: bytes, doc_img: np.ndarray | None, challenge, vcfg: dict | None = None) -> dict:
    """One pass over the media shared by all modules (expensive steps run once)."""
    from ml.features import extract as E
    from ml.features import liveness as L
    from ml.features.biometrics import ratios
    from ml.features.images import m1_image_features

    t0 = time.perf_counter()
    audio = L.decode_audio(video_bytes)
    transcript, asr = None, "none"
    if audio:
        transcript = L.transcribe_vosk(audio, 16000)
        asr = "vosk" if transcript is not None else "none"
    res = E.analyse_video(video_bytes, vcfg, challenge, transcript, asr)
    crops = res.get("clip_crops") or []
    res["gradcam_crop"] = crops[len(crops) // 2] if crops else None
    res = E.finish_embeddings(res, _clip(), _arc())
    res["asr"] = {"method": asr, "audio": bool(audio)}
    # ID document: forensics + portrait (for M5)
    res["doc"] = None
    if doc_img is not None:
        s1, v1 = m1_image_features(doc_img)
        from ml.features.images import normalize_document
        _, extra = E.m1_features(normalize_document(doc_img))
        res["doc"] = {"m1": s1, "m1_vec": v1, "ela_map": extra["ela_map"], "noise_map": extra["noise_map"]}
        res["doc_portrait"] = _doc_portrait(doc_img)
        if res["doc_portrait"]:
            res["doc_portrait"]["ratios"] = ratios(res["doc_portrait"]["landmarks"])
    res["analysis_ms"] = (time.perf_counter() - t0) * 1000
    return res


def _doc_portrait(doc: np.ndarray) -> dict | None:
    """Largest face on the ID document -> ArcFace embedding (upscaled: ID portraits are small)."""
    from ml.common.face import FaceAnalyzer, arcface_crop
    img = doc
    if max(doc.shape[:2]) < 1200:
        s = 1200 / max(doc.shape[:2])
        img = cv2.resize(doc, (int(doc.shape[1] * s), int(doc.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    with FaceAnalyzer(min_conf=0.3) as fa:
        ff, n = fa.process(img, 0)
    if not ff.ok:
        return None
    crop = arcface_crop(img, ff.landmarks)
    emb = _arc().embed([crop])[0]
    return {"embedding": emb, "landmarks": ff.landmarks, "crop_b64": _png_b64(crop, 112)}


# ------------------------------------------------------------------------------------- modules
class TrainedModule:
    name = "M?"
    record_key = ""

    def __init__(self, art):
        self.art = art
        self.version = f"{self.name}-{art.kind}-{art.feature_version}"

    def record(self, a: dict) -> dict | None:
        raise NotImplementedError

    def explain(self, a: dict, s: PreparedSession, feats: dict) -> dict:
        return {}

    def predict(self, session: PreparedSession) -> ModuleOutput:
        t0 = time.perf_counter()
        a = session.analysis
        rec = self.record(a) if a else None
        if rec is None:
            return ModuleOutput(module=self.name, status="not_applicable", model_version=self.version,
                                latency_ms=0.0, message=self.na_message(a))
        X = self.art.featurize(rec)
        p, z, conf = self.art.predict(X)
        feats = {k: (None if not np.isfinite(v) else float(v)) for k, v in (rec.get(self.art.scalar_key) or {}).items()}
        expl = self.explain(a, session, feats)
        expl["top_features"] = self.top_features(X[0])
        return ModuleOutput(module=self.name, status="ok", score=float(p[0]), raw_score=float(z[0]),
                            confidence=float(conf[0]), features={k: v for k, v in feats.items() if v is not None},
                            explanations=expl, model_version=self.version,
                            latency_ms=(time.perf_counter() - t0) * 1000 + a.get("analysis_ms", 0) / 5)

    def na_message(self, a) -> str:
        return "No face detected in the selfie video." if a and a.get("face_frames", 0) == 0 else "Input not available."

    def top_features(self, x: np.ndarray, k: int = 5) -> list[dict]:
        """Signed contributions over scalar features (vector blocks summarised as one entry each)."""
        cols = self.art.columns()
        c = self.art.contributions(x)
        agg: dict[str, float] = {}
        for name, v in zip(cols, c):
            key = name.split("[")[0] + ("[*]" if "[" in name else "")
            agg[key] = agg.get(key, 0.0) + float(v)
        top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:k]
        kind = "coef*z (+ pushes genuine)" if self.art.kind == "logreg" else "z-score vs training mean"
        return [{"feature": n, "contribution": round(v, 3), "kind": kind} for n, v in top]


class M1Document(TrainedModule):
    name = "M1"

    def record(self, a):
        return {"m1": a["doc"]["m1"], "m1_vec": a["doc"]["m1_vec"]} if a.get("doc") else None

    def na_message(self, a):
        return "No ID document image."

    def explain(self, a, s, feats):
        d = s.document_image
        return {"ela_heatmap_png": _heatmap_b64(a["doc"]["ela_map"], d),
                "noise_heatmap_png": _heatmap_b64(a["doc"]["noise_map"], d),
                "explain_method": "ELA (q=90) and block noise-residual maps (not Grad-CAM: M1 is a feature model)"}


class M2Deepfake(TrainedModule):
    name = "M2"

    def record(self, a):
        return {"m2": a["m2"], "m2_vec": a["m2_vec"]} if a.get("face_frames") else None

    def explain(self, a, s, feats):
        out = {}
        try:
            out["gradcam_png"] = _clip_gradcam(a, self.art)
            out["explain_method"] = "Grad-CAM on CLIP ViT-B/16 last block w.r.t. the linear-probe logit"
        except Exception as e:  # explanation must never break scoring
            out["explain_error"] = f"{type(e).__name__}"
        return out


class M3Replay(TrainedModule):
    name = "M3"

    def record(self, a):
        return {"m3": a["m3"], "m3_vec": a["m3_vec"]} if a.get("face_frames") else None


class M4Liveness(TrainedModule):
    name = "M4"

    def record(self, a):
        return {"m4": a["m4"]} if a.get("face_frames") else None

    def explain(self, a, s, feats):
        e = dict(a.get("m4_expl") or {})
        e["asr"] = a.get("asr")
        steps = e.get("challenge_steps") or []
        e["challenge_failed"] = bool(steps) and not all(x["passed"] for x in steps)
        return e


class M5Biometrics(TrainedModule):
    name = "M5"

    def record(self, a):
        from ml.features.biometrics import match_features
        port = a.get("doc_portrait")
        if not a.get("face_frames") or port is None or len(a.get("arc_embs", [])) == 0:
            return None
        f = match_features(port["embedding"], a["arc_embs"], port.get("ratios"), a["m5_ratios"] or {})
        return {"m5": {**f, **a["m5_geo"]}}

    def na_message(self, a):
        if a and a.get("doc") and not a.get("doc_portrait"):
            return "No face found on the ID document."
        return super().na_message(a)

    def explain(self, a, s, feats):
        port = a.get("doc_portrait") or {}
        return {"id_portrait_png": port.get("crop_b64"), "overlay": a.get("overlay")}


MODULE_CLASSES = {"M1": M1Document, "M2": M2Deepfake, "M3": M3Replay, "M4": M4Liveness, "M5": M5Biometrics}


def load_modules(bundle_dir: Path | None) -> dict:
    mods: dict = {}
    for name, cls in MODULE_CLASSES.items():
        f = bundle_dir / name / f"head_{name}.pkl" if bundle_dir else None
        if f is not None and f.exists():
            with open(f, "rb") as fh:          # bundle hashes were verified at startup (verify_bundle)
                mods[name] = cls(pickle.load(fh))
        else:
            mods[name] = PlaceholderModule(name)
    return mods


def _clip_gradcam(a: dict, art) -> str:
    """Grad-CAM (Selvaraju et al. [85]) over CLIP patch tokens for the M2 linear probe."""
    import torch
    from PIL import Image
    clip = _clip()
    crop = a.get("gradcam_crop")
    if crop is None:
        raise ValueError("no crop")
    m = clip.model.visual
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["x"] = out

    blk = m.transformer.resblocks[-1]
    h = blk.register_forward_hook(hook)
    try:
        x = clip.preprocess(Image.fromarray(crop[:, :, ::-1])).unsqueeze(0).to(clip.device)
        e = m(x)
        e = e / e.norm(dim=-1, keepdim=True)
        # linear-probe weights restricted to the CLIP block (in the learner's standardised space)
        cols = art.columns()
        coef = art.learner[-1].coef_[0]
        scaler = art.learner[0]
        sel = [i for i, n in enumerate(cols) if n.startswith("clip_mean[")]
        if not sel:
            raise ValueError("no clip features in head")
        dims = [int(cols[i][len("clip_mean["):-1]) for i in sel]
        wfull = np.zeros(e.shape[-1])
        wfull[dims] = coef[sel] / scaler.scale_[sel]
        w = torch.tensor(wfull, dtype=e.dtype, device=e.device)
        score = (e[0] * w).sum()           # logit of P(genuine) up to a constant
        (-score).backward()                # explain evidence for "fake"
        t = acts["x"]                      # (tokens, batch, width) or (batch, tokens, width)
        if t.shape[0] != 1:
            t, g = t.permute(1, 0, 2), t.grad.permute(1, 0, 2)
        else:
            g = t.grad
        t, g = t[0, 1:], g[0, 1:]
        cam = torch.relu((g.mean(0, keepdim=True) * t).sum(-1)).detach().cpu().numpy()
        side = int(round(len(cam) ** 0.5))
        cam = cam.reshape(side, side)
    finally:
        h.remove()
    return _heatmap_b64(cam, cv2.resize(crop, (224, 224)))
