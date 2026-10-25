"""Placeholder modules.

These exist so the end-to-end pipeline, logging and UI can be exercised before the
real modules land in milestone 3. They NEVER return a score: status="placeholder",
score=None. Fusion excludes them and the UI shows a PLACEHOLDER badge.
"""
from __future__ import annotations

import time

from ml.common.interface import MODULE_TITLES, ModuleOutput, PreparedSession


class PlaceholderModule:
    def __init__(self, name: str):
        self.name = name
        self.version = f"{name}-placeholder-0"

    def predict(self, session: PreparedSession) -> ModuleOutput:
        t0 = time.perf_counter()
        return ModuleOutput(
            module=self.name,  # type: ignore[arg-type]
            status="placeholder",
            model_version=self.version,
            latency_ms=(time.perf_counter() - t0) * 1000,
            message=f"{MODULE_TITLES[self.name]}: not implemented yet (milestone 3). No score produced.",
        )
