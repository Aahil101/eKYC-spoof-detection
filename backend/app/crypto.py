"""AES-256-GCM envelope for media at rest.

File format: b"CMV1" | key_id_len(1) | key_id | nonce(12) | ciphertext+tag
The key id in the header allows rotation: old files stay readable while new files
use MEDIA_ACTIVE_KEY_ID. Associated data binds the ciphertext to its logical name,
so a file cannot be swapped for another session's file undetected.
"""
from __future__ import annotations

import base64
import os
import secrets
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"CMV1"


class CryptoError(RuntimeError):
    pass


class KeyRing:
    def __init__(self, keys: dict[str, bytes], active_id: str):
        if not keys:
            raise CryptoError("empty key ring")
        if active_id not in keys:
            raise CryptoError(f"active key id {active_id!r} not in key ring")
        for kid, k in keys.items():
            if len(k) != 32:
                raise CryptoError(f"key {kid!r} must be 32 bytes")
            if not kid or len(kid.encode()) > 255:
                raise CryptoError("invalid key id")
        self._keys = keys
        self.active_id = active_id

    @classmethod
    def parse(cls, spec: str, active_id: str = "") -> "KeyRing":
        keys: dict[str, bytes] = {}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            kid, _, b64 = part.partition(":")
            keys[kid] = base64.b64decode(b64)
        return cls(keys, active_id or next(iter(keys)))

    @classmethod
    def load_or_create(cls, spec: str, active_id: str, secrets_dir: Path, allow_create: bool) -> "KeyRing":
        if spec:
            return cls.parse(spec, active_id)
        f = secrets_dir / "media.keys"
        if f.exists():
            return cls.parse(f.read_text().strip(), active_id)
        if not allow_create:
            raise CryptoError("MEDIA_KEYS not set (auto-generation only allowed in DEMO_MODE)")
        secrets_dir.mkdir(parents=True, exist_ok=True)
        line = new_key_spec()
        f.write_text(line)
        try:
            os.chmod(f, 0o600)
        except OSError:
            pass
        return cls.parse(line, active_id)

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        kid = self.active_id.encode()
        nonce = secrets.token_bytes(12)
        ct = AESGCM(self._keys[self.active_id]).encrypt(nonce, plaintext, aad)
        return MAGIC + bytes([len(kid)]) + kid + nonce + ct

    def decrypt(self, blob: bytes, aad: bytes) -> bytes:
        if blob[:4] != MAGIC:
            raise CryptoError("not an encrypted media blob")
        n = blob[4]
        kid = blob[5:5 + n].decode()
        nonce = blob[5 + n:17 + n]
        if kid not in self._keys:
            raise CryptoError(f"unknown key id {kid!r}")
        try:
            return AESGCM(self._keys[kid]).decrypt(nonce, blob[17 + n:], aad)
        except Exception as e:
            raise CryptoError("decryption failed (tampered or wrong key)") from e


def new_key_spec(key_id: str | None = None) -> str:
    kid = key_id or f"k{secrets.token_hex(3)}"
    return f"{kid}:{base64.b64encode(AESGCM.generate_key(bit_length=256)).decode()}"


if __name__ == "__main__":
    if "--new-key" in sys.argv:
        print(new_key_spec())
