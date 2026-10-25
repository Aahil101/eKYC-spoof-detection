"""API request/response schemas."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

CONSENT_VERSION = "2026-10-v1"


class CreateSessionRequest(BaseModel):
    consent: bool
    consent_version: str
    client_meta: dict[str, Any] = Field(default_factory=dict)


class ChallengeStepOut(BaseModel):
    kind: str
    prompt: str
    start_ms: int
    end_ms: int


class CreateSessionResponse(BaseModel):
    session_id: str
    token: str
    challenge: list[ChallengeStepOut]
    duration_ms: int


class ModuleOut(BaseModel):
    module: str
    title: str
    status: str
    score: float | None
    confidence: float | None
    model_version: str
    latency_ms: float
    message: str | None
    features: dict[str, Any]
    explanations: dict[str, Any]


class SessionOut(BaseModel):
    session_id: str
    status: str
    created_at: dt.datetime
    bundle_version: str | None
    decision: str | None
    decision_reason: str | None
    fused_score: float | None
    fused_uncertainty: float | None
    fusion_strategy: str | None
    fusion_weights: dict[str, Any] | None
    veto_module: str | None
    error: str | None
    challenge: list[ChallengeStepOut]
    video_meta: dict[str, Any] | None
    quality: dict[str, float | None] | None
    modules: list[ModuleOut]
    placeholder: bool


class LabelIn(BaseModel):
    label: str = Field(pattern="^(genuine|attack)$")
    attack_type: str | None = None


class PublicConfig(BaseModel):
    demo_mode: bool
    consent_version: str
    media_ttl_days: int
    bundle_version: str
    placeholder_bundle: bool
    max_document_bytes: int
    max_video_bytes: int
