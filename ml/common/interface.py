"""Common interface shared by all KYC modules (M1-M5).

Every module returns a ModuleOutput. Fusion, logging and monitoring only ever see
this object, so module implementations can be swapped (e.g. InsightFace -> facenet).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, field_validator

ModuleName = Literal["M1", "M2", "M3", "M4", "M5"]
ModuleStatus = Literal["ok", "placeholder", "failed", "not_applicable"]

MODULE_NAMES: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5")
MODULE_TITLES = {
    "M1": "Document authenticity",
    "M2": "Video deepfake detection",
    "M3": "Replay / presentation attack",
    "M4": "Liveness",
    "M5": "Facial biometrics & face match",
}


class ModuleOutput(BaseModel):
    """Output contract. score: calibrated P(genuine), 1 = genuine.

    For status != "ok", score/confidence are None: no number is ever invented.
    """

    module: ModuleName
    status: ModuleStatus
    score: float | None = None
    raw_score: float | None = None
    confidence: float | None = None
    features: dict[str, float] = Field(default_factory=dict)
    embedding: list[float] | None = None
    explanations: dict[str, Any] = Field(default_factory=dict)
    model_version: str
    latency_ms: float = 0.0
    message: str | None = None

    @field_validator("score", "confidence")
    @classmethod
    def _unit_interval(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("must be in [0, 1]")
        return v

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        if self.status == "ok" and (self.score is None or self.confidence is None):
            raise ValueError("status 'ok' requires score and confidence")
        if self.status != "ok" and self.score is not None:
            raise ValueError(f"status '{self.status}' must not carry a score")


@dataclass
class ChallengeStep:
    kind: Literal["turn_left", "turn_right", "blink_twice", "read_digits", "look_up", "smile"]
    prompt: str
    start_ms: int
    end_ms: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSession:
    """Everything a module may need, prepared once per session."""

    session_id: str
    document_image: np.ndarray | None          # BGR uint8
    frames: list[np.ndarray]                   # sampled BGR frames
    frame_times_ms: list[float]
    video_meta: dict[str, Any]
    challenge: list[ChallengeStep]
    quality: dict[str, float]
    # Populated in milestone 3 by the face pipeline (MediaPipe):
    landmarks: list[np.ndarray | None] = field(default_factory=list)
    face_boxes: list[tuple[int, int, int, int] | None] = field(default_factory=list)
    audio_path: str | None = None


@runtime_checkable
class KYCModule(Protocol):
    name: str
    version: str

    def predict(self, session: PreparedSession) -> ModuleOutput: ...
