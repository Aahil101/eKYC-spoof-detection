"""SQLAlchemy models: session log store (docs/ARCHITECTURE.md section 7).

JSON columns hold feature dicts so the schema is stable while module features evolve.
Identity templates (face-match embeddings) are stored encrypted in a separate table
and purged with media; artifact embeddings (for novelty) live in `embeddings`.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (JSON, Float, ForeignKey, Integer, LargeBinary, String, Text, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class KYCSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    status: Mapped[str] = mapped_column(String(24), default="created", index=True)
    # created -> uploading -> queued -> processing -> completed | failed ; deleted
    token_hash: Mapped[str] = mapped_column(String(64))
    consent_version: Mapped[str] = mapped_column(String(32))
    consent_at: Mapped[dt.datetime] = mapped_column(default=utcnow)
    challenge_script: Mapped[dict] = mapped_column(JSON)
    bundle_version: Mapped[str | None] = mapped_column(String(64), index=True)
    decision: Mapped[str | None] = mapped_column(String(24))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    fused_score: Mapped[float | None] = mapped_column(Float)
    fused_uncertainty: Mapped[float | None] = mapped_column(Float)
    fusion_strategy: Mapped[str | None] = mapped_column(String(16))
    fusion_weights: Mapped[dict | None] = mapped_column(JSON)
    veto_module: Mapped[str | None] = mapped_column(String(4))
    video_meta: Mapped[dict | None] = mapped_column(JSON)
    client_meta: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    media_expires_at: Mapped[dt.datetime | None] = mapped_column()
    log_expires_at: Mapped[dt.datetime | None] = mapped_column()
    source: Mapped[str] = mapped_column(String(16), default="live")  # live | simulated | dataset

    modules: Mapped[list["ModuleOutputRow"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    quality: Mapped["QualityRow | None"] = relationship(back_populates="session", cascade="all, delete-orphan",
                                                        uselist=False)


class ModuleOutputRow(Base):
    __tablename__ = "module_outputs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    module: Mapped[str] = mapped_column(String(4))
    status: Mapped[str] = mapped_column(String(16))
    score: Mapped[float | None] = mapped_column(Float)
    raw_score: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str] = mapped_column(String(64))
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    explanations: Mapped[dict] = mapped_column(JSON, default=dict)
    message: Mapped[str | None] = mapped_column(Text)
    session: Mapped[KYCSession] = relationship(back_populates="modules")


class QualityRow(Base):
    __tablename__ = "quality"
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)
    values: Mapped[dict] = mapped_column(JSON)
    session: Mapped[KYCSession] = relationship(back_populates="quality")


class EmbeddingRow(Base):
    """Artifact-space embeddings only (m2_clip, m1_forensic, m3_texture). Never identity."""
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    space: Mapped[str] = mapped_column(String(32))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32 little-endian


class IdentityTemplate(Base):
    __tablename__ = "identity_templates"
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True)
    vector_encrypted: Mapped[bytes] = mapped_column(LargeBinary)
    expires_at: Mapped[dt.datetime] = mapped_column()


class Label(Base):
    __tablename__ = "labels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(16))          # genuine | attack
    attack_type: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16))         # human_review | dataset | simulated
    labelled_at: Mapped[dt.datetime] = mapped_column(default=utcnow)


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)
    window_start: Mapped[dt.datetime] = mapped_column()
    window_end: Mapped[dt.datetime] = mapped_column()
    reference_id: Mapped[str | None] = mapped_column(String(64))
    bundle_version: Mapped[str] = mapped_column(String(64))
    per_feature: Mapped[dict] = mapped_column(JSON, default=dict)
    multivariate: Mapped[dict] = mapped_column(JSON, default=dict)
    streaming: Mapped[dict] = mapped_column(JSON, default=dict)
    label: Mapped[str | None] = mapped_column(String(24))


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow, index=True)
    type: Mapped[str] = mapped_column(String(24))
    severity: Mapped[str] = mapped_column(String(12))
    confidence: Mapped[float | None] = mapped_column(Float)
    bundle_version: Mapped[str] = mapped_column(String(64))
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    recommendation: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="open")
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))


class ReferenceWindow(Base):
    __tablename__ = "reference_windows"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    bundle_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(default=utcnow)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


def make_engine(url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk(dbapi_conn, _):  # enforce ON DELETE CASCADE in SQLite
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return engine


def make_sessionmaker(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
