"""Crash-safe checkpointing with resume across Kaggle sessions.

Kaggle sessions can die (12 h limit, idle timeout, preemption). Strategy:
  * save atomically (write tmp -> os.replace) every N steps and every M minutes;
  * keep the last K checkpoints plus `latest.json` pointing at the newest valid one;
  * on start with --resume auto, search (1) this run's own output dir and (2) any
    attached input dataset that contains runs/<run_name>/checkpoints (attach the
    previous notebook version's output via "Add Input -> Your Work").

Serialization: pickle for numpy/python state, torch.save when the state holds
tensors. Checkpoints are only loaded from directories this project wrote; never
load checkpoints from untrusted sources (pickle executes code).
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any


def _has_torch_tensors(obj: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(_has_torch_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_torch_tensors(v) for v in obj)
    return False


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, every_steps: int = 500, every_minutes: float = 10.0, keep_last: int = 3):
        self.dir = ckpt_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every_steps = max(1, int(every_steps))
        self.every_seconds = float(every_minutes) * 60.0
        self.keep_last = max(1, int(keep_last))
        self._last_save_time = time.monotonic()
        self._last_save_step = -1

    def should_save(self, step: int) -> bool:
        # Step multiples (stay aligned after resume) or wall-clock interval, never twice per step.
        if step == self._last_save_step:
            return False
        return step % self.every_steps == 0 or time.monotonic() - self._last_save_time >= self.every_seconds

    def save(self, state: dict[str, Any], step: int) -> Path:
        use_torch = _has_torch_tensors(state)
        name = f"ckpt_{step:09d}.{'pt' if use_torch else 'pkl'}"
        final = self.dir / name
        tmp = self.dir / (name + ".tmp")
        if use_torch:
            import torch
            torch.save(state, tmp)
        else:
            with open(tmp, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)
        latest_tmp = self.dir / "latest.json.tmp"
        latest_tmp.write_text(json.dumps({"file": name, "step": step, "saved_at": time.time()}))
        os.replace(latest_tmp, self.dir / "latest.json")
        self._last_save_time, self._last_save_step = time.monotonic(), step
        self._prune()
        return final

    def _prune(self) -> None:
        ckpts = sorted(p for p in self.dir.iterdir() if p.name.startswith("ckpt_") and not p.name.endswith(".tmp"))
        for p in ckpts[: -self.keep_last]:
            p.unlink(missing_ok=True)

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        if path.suffix == ".pt":
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def latest_in(ckpt_dir: Path) -> Path | None:
        """Newest loadable checkpoint in a dir (falls back past corrupt files)."""
        if not ckpt_dir.is_dir():
            return None
        candidates = sorted((p for p in ckpt_dir.iterdir()
                             if p.name.startswith("ckpt_") and p.suffix in (".pkl", ".pt")), reverse=True)
        for p in candidates:
            try:
                CheckpointManager.load(p)
                return p
            except Exception:
                continue
        return None


def find_resume_checkpoint(run_name: str, own_ckpt_dir: Path, input_roots: list[Path]) -> Path | None:
    """Pick the checkpoint with the highest step among own dir and attached inputs (any mount depth)."""
    found: list[tuple[int, Path]] = []
    dirs = [own_ckpt_dir]
    for root in input_roots:
        if root.is_dir():
            dirs += [p for p in root.glob(f"**/{run_name}/checkpoints") if p.is_dir()]
    for d in dirs:
        p = CheckpointManager.latest_in(d)
        if p is not None:
            step = int(p.stem.split("_")[1])
            found.append((step, p))
    return max(found)[1] if found else None
