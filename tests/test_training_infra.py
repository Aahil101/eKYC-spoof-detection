import json
from pathlib import Path

import pytest
import yaml

from ml.registry.bundle import BundleError, resolve_active_bundle, verify_bundle
from training.checkpoint import CheckpointManager, find_resume_checkpoint
from training.config import load_job_config
from training.fetch_subset import plan
from training.import_weights import import_run
from training.make_notebook import generate
from training.package_src import build
from training.paths import REPO_ROOT
from training.run import run

SELFTEST = REPO_ROOT / "training" / "configs" / "selftest.yaml"


def write_cfg(tmp_path: Path, **smoke_over) -> Path:
    cfg = yaml.safe_load(SELFTEST.read_text())
    cfg["modes"]["smoke"].update(smoke_over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_mode_merge():
    s, f = load_job_config(SELFTEST, "smoke"), load_job_config(SELFTEST, "full")
    assert s["params"]["steps"] == 300 and f["params"]["steps"] == 20000
    assert s["params"]["lr"] == f["params"]["lr"] == 0.1
    assert s["config_sha256"] != f["config_sha256"]
    with pytest.raises(ValueError):
        load_job_config(SELFTEST, "nope")


def test_unimplemented_job_exits_clearly(tmp_path):
    with pytest.raises(SystemExit, match="not implemented"):
        run(REPO_ROOT / "training" / "configs" / "m2_clip_probe.yaml", "smoke", out_root=tmp_path)


def test_crash_then_resume_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    ref = run(write_cfg(tmp_path), "smoke", resume="none", run_name="ref", out_root=tmp_path / "a")

    crash_cfg = write_cfg(tmp_path, simulate_crash_at_step=175)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(crash_cfg, "smoke", resume="auto", run_name="r", out_root=tmp_path / "b")
    man = json.loads((tmp_path / "b" / "r" / "run_manifest.json").read_text())
    assert man["status"] == "failed"
    ck = CheckpointManager.latest_in(tmp_path / "b" / "r" / "checkpoints")
    assert ck is not None and int(ck.stem.split("_")[1]) == 150     # every_steps=50

    resumed = run(write_cfg(tmp_path), "smoke", resume="auto", run_name="r", out_root=tmp_path / "b")
    assert resumed["status"] == "completed"
    assert resumed["resumed_from"].endswith("ckpt_000000150.pkl")
    # resumed training reproduces the uninterrupted run exactly (RNG state is checkpointed)
    assert resumed["metrics"]["train_accuracy_synthetic"] == ref["metrics"]["train_accuracy_synthetic"]
    assert resumed["artifacts"]["head.npz"] == ref["artifacts"]["head.npz"]


def test_resume_from_attached_previous_version(tmp_path, monkeypatch):
    """Simulates Kaggle: a previous version's output is attached under /kaggle/input/<x>/runs/<run>/."""
    cfg = write_cfg(tmp_path, simulate_crash_at_step=120)
    with pytest.raises(RuntimeError):
        run(cfg, "smoke", resume="none", run_name="kag", out_root=tmp_path / "prev_version_output" / "runs")
    fake_input = tmp_path / "kaggle_input"
    (fake_input).mkdir()
    (tmp_path / "prev_version_output").rename(fake_input / "codmav-kag-output")
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", str(fake_input))
    found = find_resume_checkpoint("kag", tmp_path / "fresh" / "kag" / "checkpoints", [fake_input])
    assert found is not None and found.name == "ckpt_000000100.pkl"
    m = run(write_cfg(tmp_path), "smoke", resume="auto", run_name="kag", out_root=tmp_path / "fresh")
    assert m["status"] == "completed" and "codmav-kag-output" in m["resumed_from"]


def test_corrupt_latest_checkpoint_falls_back(tmp_path):
    mgr = CheckpointManager(tmp_path, every_steps=1, keep_last=5)
    mgr.save({"step": 1}, 1)
    mgr.save({"step": 2}, 2)
    (tmp_path / "ckpt_000000003.pkl").write_bytes(b"truncated")
    assert CheckpointManager.latest_in(tmp_path).name == "ckpt_000000002.pkl"


def test_checkpoint_pruning(tmp_path):
    mgr = CheckpointManager(tmp_path, every_steps=1, keep_last=2)
    for s in range(5):
        mgr.save({"s": s}, s)
    assert sorted(p.name for p in tmp_path.glob("ckpt_*")) == ["ckpt_000000003.pkl", "ckpt_000000004.pkl"]


def test_budget_stop_marks_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    cfg = yaml.safe_load(SELFTEST.read_text())
    cfg["budget"]["max_wall_minutes"] = 1e-9
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    m = run(p, "smoke", resume="none", out_root=tmp_path)
    assert m["status"] == "incomplete_budget"


def test_import_weights_into_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    m = run(SELFTEST, "smoke", resume="none", out_root=tmp_path / "runs")
    run_dir = tmp_path / "runs" / m["run_name"]
    bundles = tmp_path / "bundles"
    v1 = import_run(run_dir, "M2", "b1", semver="0.3.0", bundles_root=bundles)
    assert v1.startswith("0.3.0+")
    assert verify_bundle(bundles / "b1") == v1
    assert resolve_active_bundle(bundles, "b1")[0] == v1
    man = yaml.safe_load((bundles / "b1" / "manifest.yaml").read_text())
    assert man["modules"]["M2"]["training_run"]["config_sha256"] == m["config_sha256"]
    # adding another module changes the version
    v2 = import_run(run_dir, "M3", "b1", bundles_root=bundles)
    assert v2 != v1
    # tampering is detected
    (bundles / "b1" / "M2" / "head.npz").write_bytes(b"tampered")
    with pytest.raises(BundleError):
        verify_bundle(bundles / "b1")


def test_import_rejects_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    m = run(SELFTEST, "smoke", resume="none", out_root=tmp_path / "runs")
    run_dir = tmp_path / "runs" / m["run_name"]
    (run_dir / "artifacts" / "head.npz").write_bytes(b"broken")
    with pytest.raises(BundleError, match="hash mismatch"):
        import_run(run_dir, "M2", "b", bundles_root=tmp_path / "bundles")


def test_notebook_generation(tmp_path):
    out = generate(REPO_ROOT / "training" / "configs" / "m2_clip_probe.yaml", "me", tmp_path,
                   repo_url="https://github.com/me/codmav.git")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False
    assert "xdxd003/ff-c23" in meta["dataset_sources"]
    assert {"me/codmav-src", "me/codmav-wheels", "me/codmav-weights-openclip-vitb16"} <= set(meta["dataset_sources"])
    nb = json.loads((out / meta["code_file"]).read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "--mode full --resume auto" in code and "training/configs/m2_clip_probe.yaml" in code
    assert "--no-index --find-links" in code
    assert "recursive=True" not in code and "**" not in code     # never walk all of /kaggle/input
    compile(code, "nb", "exec")       # code cells are valid Python

    out2 = generate(SELFTEST, "me", tmp_path)
    meta2 = json.loads((out2 / "kernel-metadata.json").read_text())
    assert meta2["enable_gpu"] is False and meta2["enable_internet"] is True
    assert meta2["dataset_sources"] == ["me/codmav-src"]          # no repo url -> source dataset
    out3 = generate(SELFTEST, "me", tmp_path / "x", repo_url="https://github.com/me/codmav.git")
    assert json.loads((out3 / "kernel-metadata.json").read_text())["dataset_sources"] == []


def test_competition_source(tmp_path):
    cfg = yaml.safe_load(SELFTEST.read_text())
    cfg["data"]["datasets"] = ["dfdc", "sidtd"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    meta = json.loads((generate(p, "me", tmp_path) / "kernel-metadata.json").read_text())
    assert meta["competition_sources"] == ["deepfake-detection-challenge"]
    assert meta["kernel_sources"] == ["aahil1011/codmav-fetch-sidtd-templates"]


def test_fetch_archive_job(tmp_path, monkeypatch):
    """Serves a zip over local HTTP; checks hash, extraction, zip-slip guard and manifest."""
    import hashlib
    import http.server
    import io
    import threading
    import zipfile as zf

    buf = io.BytesIO()
    with zf.ZipFile(buf, "w") as z:
        z.writestr("templates/Images/reals/a.jpg", b"x" * 10)
        z.writestr("templates/Images/fakes/b.jpg", b"y" * 10)
    payload = buf.getvalue()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = yaml.safe_load((REPO_ROOT / "training" / "configs" / "fetch_sidtd.yaml").read_text())
    cfg["params"]["url"] = f"http://127.0.0.1:{srv.server_address[1]}/templates.zip"
    cfg["params"]["sha256"] = hashlib.sha256(payload).hexdigest()
    cfg["modes"]["smoke"] = {"max_bytes": 0, "extract": True}
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CODMAV_FETCH_DEST", str(tmp_path / "data"))
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    m = run(p, "smoke", resume="none", out_root=tmp_path / "runs")
    srv.shutdown()
    assert m["status"] == "completed" and m["metrics"]["extracted_files"] == 2
    assert (tmp_path / "data" / "sidtd" / "templates" / "Images" / "fakes" / "b.jpg").exists()
    assert "CC-BY-4.0" in (tmp_path / "data" / "sidtd" / "LICENSE_NOTICE.txt").read_text()


def test_package_src_excludes_secrets_and_data(tmp_path):
    import zipfile
    z = zipfile.ZipFile(build("me", tmp_path))
    names = z.namelist()
    assert "training/run.py" in names and "GIT_SHA" in names
    assert "+src." in z.read("GIT_SHA").decode()
    assert not any(n.startswith(("data/", "runs/", "frontend/")) or n.endswith((".db", ".keys", ".env"))
                   for n in names)


def test_audit_tree_counts_and_layout(tmp_path):
    from training.audit import _find_mount, _has_entry, audit_tree
    root = tmp_path / "datasets" / "owner" / "celeb-df-v2"
    for sub, n in (("Celeb-real", 3), ("Celeb-synthesis", 5)):
        (root / sub).mkdir(parents=True)
        for i in range(n):
            (root / sub / f"id{i}.mp4").write_bytes(b"x")
    (root / "List_of_testing_videos.txt").write_text("x")
    assert _find_mount("celeb-df-v2", [tmp_path]) == root
    r = audit_tree(root, depth=1)
    assert r["n_files"] == 9 and r["files_per_dir_depth1"]["Celeb-synthesis"] == 5 and not r["partial"]
    assert r["extensions"][".mp4"] == 8
    assert _has_entry(root, "Celeb-real") == "Celeb-real"


def test_fetch_subset_plan_respects_caps():
    files = [("a.mp4", 50_000_000), ("b.mp4", 60_000_000), ("c.json", 10), ("d.mp4", 300_000_000)]
    assert plan(files, "*.mp4", 10, 120) == ["a.mp4", "b.mp4"]
    assert plan(files, "*.mp4", 1, 1000) == ["a.mp4"]
