"""Structured JSON logging. Never log raw media, names, ID numbers or embeddings."""
from __future__ import annotations

import logging

import structlog

_FORBIDDEN_KEYS = {"embedding", "image", "frame", "video", "document", "name", "id_number", "ocr_text"}


def _scrub(_, __, event_dict):
    for k in list(event_dict):
        if k in _FORBIDDEN_KEYS:
            event_dict[k] = "[redacted]"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _scrub,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "codmav"):
    return structlog.get_logger(name)
