"""Retention: purge expired media/identity templates and expired logs."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import delete, select

from backend.app.db import IdentityTemplate, KYCSession
from backend.app.media_store import MediaStore


def purge_expired(sm, store: MediaStore, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    media_purged = logs_purged = 0
    with sm() as db:
        for s in db.scalars(select(KYCSession).where(KYCSession.media_expires_at.is_not(None))):
            exp = s.media_expires_at if s.media_expires_at.tzinfo else s.media_expires_at.replace(tzinfo=dt.timezone.utc)
            if exp <= now:
                store.delete_session(s.id)
                db.execute(delete(IdentityTemplate).where(IdentityTemplate.session_id == s.id))
                s.media_expires_at = None
                media_purged += 1
        for s in db.scalars(select(KYCSession).where(KYCSession.log_expires_at.is_not(None))).all():
            exp = s.log_expires_at if s.log_expires_at.tzinfo else s.log_expires_at.replace(tzinfo=dt.timezone.utc)
            if exp <= now:
                store.delete_session(s.id)
                db.delete(s)
                logs_purged += 1
        db.commit()
    return {"media_purged": media_purged, "logs_purged": logs_purged}
