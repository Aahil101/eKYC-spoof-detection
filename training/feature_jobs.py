"""Kaggle feature-extraction and head-training jobs (milestone 3).

extract_features (one notebook per dataset, resume-safe):
    items are listed deterministically and chunked into shards; each finished shard is written
    atomically to <out>/feats/<dataset>/shard_XXXXX.pkl. On restart (or when a previous version's
    output is attached) finished shards are found and skipped, so a dead 12 h session loses at most
    one shard. Videos: worker processes do decode + MediaPipe + handcrafted features; the main
    process batches CLIP (GPU) and ArcFace. Images (CelebA-Spoof, SIDTD) need no GPU.

train_heads: loads all shards from attached kernel outputs, trains + calibrates M1-M5 heads with
    group-disjoint / official splits, writes head_Mx.pkl artifacts and a metrics report.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from training.audit import _find_mount
from training.jobs import register
from training.paths import KAGGLE_WORKING, REPO_ROOT, input_roots, is_kaggle

# ================================================================== listers
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def list_ffpp(root: Path, splits: dict[str, str]) -> list[dict]:
    base = next(root.glob("*/original")).parent if not (root / "original").exists() else root
    items = []
    for p in sorted((base / "original").glob("*.mp4")):
        vid = p.stem
        items.append({"id": f"ffpp/original/{vid}", "path": str(p), "label": 1, "attack_type": "bona_fide",
                      "group": vid, "split": splits.get(vid, "unknown"), "identity": vid, "source": "ffpp"})
    for m in FFPP_METHODS:
        for p in sorted((base / m).glob("*.mp4")):
            a, b = p.stem.split("_")[:2]
            items.append({"id": f"ffpp/{m}/{p.stem}", "path": str(p), "label": 0, "attack_type": m,
                          "group": a, "split": splits.get(a, "unknown"), "identity": a, "source_identity": b,
                          "source": "ffpp"})
    return items


def list_celebdf(root: Path, n_fake_extra: int = 600, seed: int = 0) -> list[dict]:
    test = {}
    tl = next(root.rglob("List_of_testing_videos.txt"), None)
    if tl:
        for line in tl.read_text().splitlines():
            if line.strip():
                lab, rel = line.split()
                test[rel] = int(lab)          # Celeb-DF list: 1 = real, 0 = fake
    base = tl.parent if tl else root
    items = []
    rng = np.random.default_rng(seed)
    synth = sorted((base / "Celeb-synthesis").glob("*.mp4"))
    extra = set(rng.choice([s.name for s in synth if f"Celeb-synthesis/{s.name}" not in test],
                           size=min(n_fake_extra, len(synth)), replace=False).tolist())
    for sub, lab in (("Celeb-real", 1), ("YouTube-real", 1), ("Celeb-synthesis", 0)):
        for p in sorted((base / sub).glob("*.mp4")):
            rel = f"{sub}/{p.name}"
            if sub == "Celeb-synthesis" and rel not in test and p.name not in extra:
                continue
            m = re.match(r"id(\d+)_(?:id(\d+)_)?(\d+)", p.stem)
            ident = f"cdf{m.group(1)}" if m and sub != "YouTube-real" else f"yt{p.stem}"
            items.append({"id": f"celebdf/{rel}", "path": str(p), "label": lab,
                          "attack_type": "bona_fide" if lab else "celebdf_faceswap", "group": ident,
                          "split": "test" if rel in test else "pool", "identity": ident,
                          "source_identity": f"cdf{m.group(2)}" if m and m.group(2) else None, "source": "celebdf"})
    return items


def list_msu(root: Path) -> list[dict]:
    tr = next(root.rglob("train_sub_list.txt"))
    te = next(root.rglob("test_sub_list.txt"))
    split = {int(x): "train" for x in tr.read_text().split()} | {int(x): "test" for x in te.read_text().split()}
    base = tr.parent
    items = []
    for sub, lab in (("real", 1), ("attack", 0)):
        for p in sorted((base / "scene01" / sub).glob("*")):
            if p.suffix.lower() not in (".mp4", ".mov"):
                continue
            m = re.search(r"client(\d+)", p.name)
            cid = int(m.group(1)) if m else -1
            at = "bona_fide"
            for k in ("printed_photo", "ipad_video", "iphone_video"):
                if k in p.name:
                    at = k
            items.append({"id": f"msu/{sub}/{p.stem}", "path": str(p), "label": lab, "attack_type": at,
                          "group": f"client{cid:03d}", "split": split.get(cid, "unknown"),
                          "identity": f"msu{cid:03d}", "device": "android" if "android" in p.name else "laptop",
                          "source": "msu"})
    return items


def list_celeba_spoof(root: Path, per_type_train: int, n_test: int, seed: int = 0) -> list[dict]:
    # Bounded search (never rglob: the mount holds 1.1M files on a slow network FS).
    meta = next((p for d in range(0, 4) for p in root.glob("*/" * d + "metas/intra_test") if p.is_dir()), None)
    if meta is None:
        raise FileNotFoundError("CelebA-Spoof metas/intra_test not found within depth 3")
    base = meta.parents[1]
    rng = np.random.default_rng(seed)
    items = []

    def sample_existing(ks: list[str], n: int) -> list[str]:
        """Sample first, then stat only the sampled files (mirror holds ~90% of images)."""
        order = rng.permutation(len(ks))
        out = []
        for i in order:
            if len(out) >= n:
                break
            if (base / ks[i]).exists():
                out.append(ks[i])
        return out

    for part, fn in (("train", "train_label.json"), ("test", "test_label.json")):
        lab = json.loads((meta / fn).read_text())
        keys = sorted(lab)
        if part == "train":
            by_type: dict[int, list[str]] = {}
            for k in keys:
                by_type.setdefault(int(lab[k][40]), []).append(k)
            chosen = []
            for t, ks in sorted(by_type.items()):
                n = per_type_train * (3 if t == 0 else 1)          # more live samples (balance ~ attacks)
                chosen += sample_existing(ks, n)
        else:
            chosen = sample_existing(keys, n_test)
        for k in sorted(chosen):
            v = lab[k]
            subj = k.split("/")[2]
            items.append({"id": f"celebaspoof/{k}", "path": str(base / k), "label": int(v[43] == 0),
                          "attack_type": f"spoof_type_{int(v[40])}", "group": subj,
                          "split": part, "illumination": int(v[41]), "environment": int(v[42]),
                          "source": "celeba_spoof"})
    return items


def list_sidtd(root: Path) -> list[dict]:
    img_root = next(root.rglob("templates/Images"))
    items = []
    for sub, lab in (("reals", 1), ("fakes", 0)):
        for p in sorted((img_root / sub).glob("*.jpg")):
            m = re.match(r"([a-z]+_(?:id|passport))_(\d+)", p.stem)
            tmpl = f"{m.group(1)}_{m.group(2)}" if m else p.stem
            ft = re.search(r"fake_(\d+)", p.stem)
            items.append({"id": f"sidtd/{sub}/{p.stem}", "path": str(p), "label": lab,
                          "attack_type": f"forgery_{ft.group(1)}" if ft else ("bona_fide" if lab else "forgery"),
                          "group": tmpl, "split": "pool", "doc_type": m.group(1) if m else "unknown",
                          "source": "sidtd"})
    return items


def ffpp_official_splits(cache: Path) -> dict[str, str]:
    """FaceForensics++ official splits (github ondyari/FaceForensics dataset/splits/*.json)."""
    import urllib.request
    out = {}
    cache.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        f = cache / f"ffpp_{s}.json"
        if not f.exists():
            url = f"https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{s}.json"
            f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        for a, b in json.loads(f.read_text()):
            out[a] = out[b] = s
    return out


# ================================================================== workers
def _video_worker(args):
    item, vcfg = args
    try:
        from ml.features.extract import analyse_video
        res = analyse_video(Path(item["path"]).read_bytes(), vcfg)
        res.pop("first_frame", None)
        res.pop("overlay", None)
        res["m4_expl"] = {}
        return item, res, None
    except Exception as e:
        return item, None, f"{type(e).__name__}: {e}"


def _image_worker(args):
    item, kind = args
    try:
        import cv2

        from ml.common.quality import compute_quality
        from ml.features import images as I
        img = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("unreadable image")
        if kind == "celeba_spoof":
            bbt = Path(item["path"]).with_name(Path(item["path"]).stem + "_BB.txt")
            bb = I.celeba_spoof_bbox(img, bbt.read_text()) if bbt.exists() else None
            if bb is None:
                raise ValueError("no bbox")
            s, vec, _ = I.m3_image_features(img, bb)
            res = {"m3": s, "m3_vec": vec}
        else:
            s, vec = I.m1_image_features(img)
            res = {"m1": s, "m1_vec": vec}
        res["quality"] = compute_quality([img], {"width": img.shape[1], "height": img.shape[0]}, None)
        return item, res, None
    except Exception as e:
        return item, None, f"{type(e).__name__}: {e}"


# ================================================================== extraction job
def _out_root() -> Path:
    return Path(os.environ.get("CODMAV_FEATS_ROOT", (KAGGLE_WORKING if is_kaggle() else REPO_ROOT / "runs")))


def _find_existing_shard(dataset: str, name: str, own: Path) -> Path | None:
    if (own / name).exists():
        return own / name
    for r in input_roots():
        if not r.is_dir():
            continue
        for d in range(0, 5):
            for p in r.glob("*/" * d + f"feats/{dataset}/{name}"):
                return p
    return None


def _resolve_root(reg_name: str) -> Path:
    import yaml
    reg = yaml.safe_load((REPO_ROOT / "training" / "datasets.yaml").read_text(encoding="utf-8"))
    e = reg["datasets"][reg_name]
    if env := os.environ.get(f"CODMAV_DATA_{reg_name.upper()}"):
        return Path(env)
    slug = e["kaggle_ref"].split("/")[-1]
    m = _find_mount(slug, input_roots())
    if m is None:
        raise FileNotFoundError(f"dataset {reg_name} ({e['kaggle_ref']}) is not attached")
    return m


@register("extract_features")
def extract_features(cfg: dict, ctx) -> dict:
    p = cfg["params"]
    ds = p["dataset"]
    root = _resolve_root(p.get("registry_name", ds))
    if ds == "ffpp_c23":
        items = list_ffpp(root, ffpp_official_splits(_out_root() / "cache"))
    elif ds == "celebdf_v2":
        items = list_celebdf(root, int(p.get("n_fake_extra", 600)), cfg["seed"])
    elif ds == "msu_mfsd":
        items = list_msu(root)
    elif ds == "celeba_spoof":
        items = list_celeba_spoof(root, int(p["per_type_train"]), int(p["n_test"]), cfg["seed"])
    elif ds == "sidtd":
        items = list_sidtd(root)
    else:
        raise ValueError(ds)
    if p.get("max_items"):
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(items), size=min(int(p["max_items"]), len(items)), replace=False)
        items = [items[i] for i in sorted(idx)]
    is_video = ds in ("ffpp_c23", "celebdf_v2", "msu_mfsd")
    shard_size = int(p.get("shard_size", 100))
    shards = [items[i:i + shard_size] for i in range(0, len(items), shard_size)]
    out = _out_root() / "feats" / ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps({"n": len(items), "items_sha256": hashlib.sha256(
        json.dumps([it["id"] for it in items]).encode()).hexdigest(), "shard_size": shard_size}, indent=1))
    ctx.log("listed", dataset=ds, n_items=len(items), n_shards=len(shards), root=str(root))

    clip = arc = None
    from ml.common.weights import path_for
    path_for("mediapipe_landmarker")              # fetch once before workers start (no download race)
    if is_video:
        path_for("arcface_buffalo_l")
        from ml.features.biometrics import FaceEmbedder
        from ml.features.extract import ClipEmbedder
        clip, arc = ClipEmbedder(p.get("device")), FaceEmbedder()
    workers = int(p.get("workers", max(1, (os.cpu_count() or 2))))
    pool = get_context("spawn").Pool(workers) if workers > 1 else None
    done = skipped = errors = 0
    t0 = time.monotonic()
    try:
        for si, shard in enumerate(shards):
            name = f"shard_{si:05d}.pkl"
            ex = _find_existing_shard(ds, name, out)
            if ex is not None:
                if ex != out / name:
                    shutil.copy2(ex, out / name)
                skipped += 1
                continue
            if ctx.out_of_time():
                ctx.log("budget_exhausted", shard=si)
                break
            args = [(it, p.get("video", {})) if is_video else (it, ds) for it in shard]
            fn = _video_worker if is_video else _image_worker
            results = pool.imap(fn, args, chunksize=1) if pool else map(fn, args)
            recs = []
            for item, res, err in results:
                if err:
                    errors += 1
                    recs.append({**item, "error": err})
                    continue
                if is_video:
                    from ml.features.extract import finish_embeddings
                    res = finish_embeddings(res, clip, arc)
                recs.append({**item, **res})
            tmp = out / (name + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, out / name)
            done += 1
            el = time.monotonic() - t0
            ctx.log("shard_done", shard=si, of=len(shards), errors=errors,
                    s_per_item=round(el / max(1, done * shard_size), 2))
    finally:
        if pool:
            pool.close()
            pool.join()
    complete = all((out / f"shard_{i:05d}.pkl").exists() for i in range(len(shards)))
    summary = {"dataset": ds, "n_items": len(items), "n_shards": len(shards), "shards_done_now": done,
               "shards_reused": skipped, "errors": errors, "finished": complete}
    ctx.save_artifact("extract_summary.json", json.dumps(summary, indent=2).encode())
    return summary


# ================================================================== training
def load_records(dataset: str) -> list[dict]:
    roots = [_out_root()] + [r for r in input_roots() if r.is_dir()]
    seen, recs = set(), []
    for r in roots:
        for d in range(0, 5):
            for f in sorted(r.glob("*/" * d + f"feats/{dataset}/shard_*.pkl")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                with open(f, "rb") as fh:
                    recs += pickle.load(fh)
    return [x for x in recs if "error" not in x]


def _m5_pairs(recs: list[dict], seed: int, n_imp: int = 3):
    """Identity-verification pairs from real videos: ID 'photo' = most frontal ArcFace embedding of
    video i; selfie = embeddings of video j (same identity -> genuine, other identity -> impostor)."""
    from ml.features.biometrics import match_features
    rng = np.random.default_rng(seed)
    by_id: dict[str, list[dict]] = {}
    for r in recs:
        if r["label"] == 1 and len(r.get("arc_embs", [])) > 0 and r.get("m5_ratios"):
            by_id.setdefault(r["identity"], []).append(r)
    ids = sorted(by_id)
    out = []
    for ident in ids:
        vids = by_id[ident]
        for i, a in enumerate(vids):
            id_emb, id_rat = a["arc_embs"][0], a["m5_ratios"]
            partners = [v for j, v in enumerate(vids) if j != i][:2]
            for b in partners:
                out.append(({"m5": {**match_features(id_emb, b["arc_embs"], id_rat, b["m5_ratios"]), **b["m5_geo"]}},
                            1, ident))
            for o in rng.choice([x for x in ids if x != ident], size=min(n_imp, len(ids) - 1), replace=False):
                b = by_id[o][rng.integers(len(by_id[o]))]
                out.append(({"m5": {**match_features(id_emb, b["arc_embs"], id_rat, b["m5_ratios"]), **b["m5_geo"]}},
                            0, ident))
    return out


@register("train_heads")
def train_heads(cfg: dict, ctx) -> dict:
    from ml.features.extract import FEATURE_VERSION
    from ml.heads import build_matrix, fit_head, group_split, pad_metrics
    p = cfg["params"]
    seed = cfg["seed"]
    report: dict = {"feature_version": FEATURE_VERSION, "modules": {}}

    def save(art, extra_eval=None):
        if extra_eval:
            art.metrics.update(extra_eval)
        ctx.save_artifact(f"head_{art.module}.pkl", pickle.dumps(art, protocol=pickle.HIGHEST_PROTOCOL))
        report["modules"][art.module] = {"metrics": art.metrics, "data": art.data_info, "kind": art.kind,
                                         "n_features": len(art.feature_names) + sum(art.vec_dims.values())}
        ctx.log("head_trained", module=art.module, test=art.metrics.get("test", {}))

    def counts(recs, split):
        c: dict = {}
        for r, s in zip(recs, split):
            c.setdefault(s, {}).setdefault(r["attack_type"], 0)
            c[s][r["attack_type"]] += 1
        return c

    # ---------------- M2 deepfake: FF++ official splits; cross-dataset Celeb-DF test list
    if "M2" in p["modules"]:
        ff = load_records("ffpp_c23")
        ff = [r for r in ff if r["split"] in ("train", "val", "test")]
        X, names, vd, _ = build_matrix(ff, "m2", ("clip_mean", "spectrum"))
        y = np.array([r["label"] for r in ff])
        split = np.array([{"train": "train", "val": "cal", "test": "test"}[r["split"]] for r in ff])
        art = fit_head("M2", X, y, split, names, vd, ("clip_mean", "spectrum"), "m2", kind="logreg",
                       calib="temperature", seed=seed, feature_version=FEATURE_VERSION)
        art.data_info = {"train_data": "FaceForensics++ c23 (official splits), real vs DF/F2F/FS/NT/FaceShifter",
                         "counts": counts(ff, split)}
        extra = {}
        test_mask = split == "test"
        pt, _, _ = art.predict(X[test_mask])
        yt = y[test_mask]
        at = np.array([r["attack_type"] for r in ff])[test_mask]
        extra["test_per_method_auc"] = {m: pad_metrics(np.r_[pt[yt == 1], pt[at == m]],
                                                       np.r_[np.ones((yt == 1).sum()), np.zeros((at == m).sum())])
                                        .get("auc") for m in FFPP_METHODS}
        cdf = [r for r in load_records("celebdf_v2") if r["split"] == "test"]
        if cdf:
            Xc, *_ = build_matrix(cdf, "m2", ("clip_mean", "spectrum"), names, vd)
            pc, _, _ = art.predict(Xc)
            extra["cross_celebdf_test"] = pad_metrics(pc, np.array([r["label"] for r in cdf]))
        save(art, extra)

    # ---------------- M3 replay/PAD: CelebA-Spoof images; cross-dataset MSU-MFSD videos
    if "M3" in p["modules"]:
        cs = load_records("celeba_spoof")
        X, names, vd, _ = build_matrix(cs, "m3", ("lbp",))
        y = np.array([r["label"] for r in cs])
        gs = group_split([r["group"] for r in cs], seed, (0.85, 0.15, 0.0, 0.0))
        split = np.array(["test" if r["split"] == "test" else s for r, s in zip(cs, gs)])
        art = fit_head("M3", X, y, split, names, vd, ("lbp",), "m3", kind="hgb", calib="isotonic", seed=seed,
                       feature_version=FEATURE_VERSION)
        art.data_info = {"train_data": "CelebA-Spoof Kaggle mirror (subset, official intra-test labels)",
                         "counts": counts(cs, split)}
        msu = load_records("msu_mfsd")
        extra = {}
        if msu:
            Xm, *_ = build_matrix(msu, "m3", ("lbp",), names, vd)
            pm, _, _ = art.predict(Xm)
            extra["cross_msu_all"] = pad_metrics(pm, np.array([r["label"] for r in msu]))
        save(art, extra)

    # ---------------- M4 liveness: MSU-MFSD official subject protocol (train 15 / test 20 subjects)
    if "M4" in p["modules"]:
        msu = load_records("msu_mfsd")
        X, names, vd, _ = build_matrix(msu, "m4")
        y = np.array([r["label"] for r in msu])
        inner = group_split([r["group"] for r in msu], seed, (0.75, 0.25, 0.0, 0.0))
        split = np.array(["test" if r["split"] == "test" else s for r, s in zip(msu, inner)])
        art = fit_head("M4", X, y, split, names, vd, (), "m4", kind="logreg", calib="temperature", seed=seed,
                       feature_version=FEATURE_VERSION)
        art.data_info = {"train_data": "MSU-MFSD (passive rPPG/blink/motion cues; official subject split)",
                         "counts": counts(msu, split),
                         "note": "challenge-response compliance is rule-based (not learned) - no public data"}
        save(art)

    # ---------------- M1 document: SIDTD templates, template-disjoint split
    if "M1" in p["modules"]:
        sd = load_records("sidtd")
        X, names, vd, _ = build_matrix(sd, "m1", ("spectrum",))
        y = np.array([r["label"] for r in sd])
        split = group_split([r["group"] for r in sd], seed)
        art = fit_head("M1", X, y, split, names, vd, ("spectrum",), "m1", kind="hgb", calib="isotonic",
                       seed=seed, feature_version=FEATURE_VERSION)
        art.data_info = {"train_data": "SIDTD templates (reals vs crop&replace / inpainting fakes)",
                         "counts": counts(sd, split)}
        save(art)

    # ---------------- M5 face match: Celeb-DF real identities (+ FF++ originals as extra impostors)
    if "M5" in p["modules"]:
        recs = [r for r in load_records("celebdf_v2") if r["label"] == 1]
        pairs = _m5_pairs(recs, seed)
        X, names, vd, _ = build_matrix([q for q, _, _ in pairs], "m5")
        y = np.array([lab for _, lab, _ in pairs])
        split = group_split([g for _, _, g in pairs], seed)
        art = fit_head("M5", X, y, split, names, vd, (), "m5", kind="logreg", calib="temperature", seed=seed,
                       feature_version=FEATURE_VERSION)
        art.data_info = {"train_data": "Celeb-DF v2 real videos: same-identity (genuine) vs other-identity pairs",
                         "n_pairs": int(len(pairs)), "n_genuine": int(y.sum()),
                         "note": "FAR = APCER, FRR = BPCER in the metrics block"}
        save(art)

    ctx.save_artifact("heads_report.json", json.dumps(report, indent=2, default=float).encode())
    return {"modules": list(report["modules"]),
            "test_auc": {m: v["metrics"].get("test", {}).get("auc") for m, v in report["modules"].items()},
            "finished": True}
