"""Execute a generated Kaggle notebook's code cells inside a Docker container that mimics
the Kaggle layout (/kaggle/input/<dataset>/..., /kaggle/working). Used to validate
notebooks before spending Kaggle quota.

    python scripts/simulate_kaggle.py kaggle/build/selftest --src-zip dist/codmav-src/codmav_src.zip

The container is python:3.12-slim with numpy preinstalled (Kaggle images ship numpy/torch).
Internet is left ON in the container so the notebook's pip cell works; use --offline to
run with --network none (then the notebook must find wheels in an attached dataset).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook_dir", type=Path)
    ap.add_argument("--src-zip", type=Path, required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--keep", type=Path, default=None, help="copy /kaggle/working here afterwards")
    a = ap.parse_args()

    meta = json.loads((a.notebook_dir / "kernel-metadata.json").read_text())
    nb = json.loads((a.notebook_dir / meta["code_file"]).read_text())
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src_ds = root / "input" / "codmav-src"
        src_ds.mkdir(parents=True)
        shutil.copy2(a.src_zip, src_ds / "codmav_src.zip")
        (root / "working").mkdir()
        (root / "nb.py").write_text(code, encoding="utf-8")
        cmd = ["docker", "run", "--rm",
               "-v", f"{root / 'input'}:/kaggle/input:ro",
               "-v", f"{root / 'working'}:/kaggle/working",
               "-v", f"{root / 'nb.py'}:/nb.py:ro",
               "-e", "KAGGLE_KERNEL_RUN_TYPE=Batch"]
        if a.offline:
            cmd += ["--network", "none"]
        cmd += ["python:3.12-slim", "sh", "-c", "pip install -q numpy && python /nb.py"]
        rc = subprocess.call(cmd)
        if a.keep:
            if a.keep.exists():
                shutil.rmtree(a.keep)
            shutil.copytree(root / "working", a.keep)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
