"""Encrypted media store on local disk with TTL-based purge."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from backend.app.crypto import KeyRing

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class MediaStore:
    def __init__(self, root: Path, keyring: KeyRing):
        self.root = root
        self.keyring = keyring
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str, name: str) -> Path:
        if not _SAFE.match(session_id) or not _SAFE.match(name):
            raise ValueError("invalid media identifier")
        return self.root / session_id / f"{name}.bin"

    @staticmethod
    def _aad(session_id: str, name: str) -> bytes:
        return f"{session_id}/{name}".encode()

    def put(self, session_id: str, name: str, data: bytes) -> str:
        p = self._path(session_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(self.keyring.encrypt(data, self._aad(session_id, name)))
        tmp.replace(p)
        return f"{session_id}/{name}"

    def get(self, session_id: str, name: str) -> bytes:
        return self.keyring.decrypt(self._path(session_id, name).read_bytes(), self._aad(session_id, name))

    def exists(self, session_id: str, name: str) -> bool:
        return self._path(session_id, name).exists()

    def delete_session(self, session_id: str) -> None:
        if not _SAFE.match(session_id):
            raise ValueError("invalid session id")
        d = self.root / session_id
        if d.exists():
            shutil.rmtree(d)
