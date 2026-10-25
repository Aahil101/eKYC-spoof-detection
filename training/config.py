"""Training job configs: one YAML per job with `smoke` and `full` mode overrides.

    job: selftest_logreg          # key in training.jobs.JOBS
    run_name: selftest
    implemented: true
    seed: 0
    params: {...}                 # shared
    modes:
      smoke: {...}                # merged over params for laptop CPU runs
      full: {...}                 # merged over params for Kaggle runs
    checkpoint: {every_steps, every_minutes, keep_last}
    budget: {max_wall_minutes}    # stop cleanly before Kaggle's 12 h limit
    data: {datasets: [logical names from training/datasets.yaml]}
    kaggle: {accelerator: gpu|none, internet: true|false, pip_packages: [...], weights_datasets: [...]}
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_job_config(path: Path, mode: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if mode not in (raw.get("modes") or {}):
        raise ValueError(f"mode {mode!r} not defined in {path}")
    for key in ("job", "run_name"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")
    cfg = {k: v for k, v in raw.items() if k != "modes"}
    cfg["params"] = deep_merge(raw.get("params") or {}, raw["modes"][mode] or {})
    cfg["mode"] = mode
    cfg["config_path"] = str(path)
    cfg.setdefault("seed", 0)
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("budget", {})
    cfg["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    return cfg
