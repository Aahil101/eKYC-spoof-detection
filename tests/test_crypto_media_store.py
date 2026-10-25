import pytest

from backend.app.crypto import CryptoError, KeyRing, new_key_spec
from backend.app.media_store import MediaStore


def test_roundtrip_and_aad_binding():
    kr = KeyRing.parse(new_key_spec("a"))
    blob = kr.encrypt(b"secret media", b"s1/video")
    assert b"secret media" not in blob
    assert kr.decrypt(blob, b"s1/video") == b"secret media"
    with pytest.raises(CryptoError):
        kr.decrypt(blob, b"s2/video")          # swapped into another session


def test_tamper_detected():
    kr = KeyRing.parse(new_key_spec("a"))
    blob = bytearray(kr.encrypt(b"x" * 100, b"aad"))
    blob[-1] ^= 1
    with pytest.raises(CryptoError):
        kr.decrypt(bytes(blob), b"aad")


def test_key_rotation_keeps_old_files_readable():
    old, new = new_key_spec("old"), new_key_spec("new")
    blob = KeyRing.parse(old, "old").encrypt(b"legacy", b"aad")
    ring = KeyRing.parse(f"{old},{new}", "new")
    assert ring.decrypt(blob, b"aad") == b"legacy"
    assert ring.encrypt(b"fresh", b"aad")[5:8] == b"new"


def test_bad_key_rejected():
    with pytest.raises(CryptoError):
        KeyRing({"k": b"short"}, "k")


def test_auto_create_only_in_demo(tmp_path):
    with pytest.raises(CryptoError):
        KeyRing.load_or_create("", "", tmp_path / "s", allow_create=False)
    kr = KeyRing.load_or_create("", "", tmp_path / "s", allow_create=True)
    kr2 = KeyRing.load_or_create("", "", tmp_path / "s", allow_create=False)   # reuses file
    assert kr2.decrypt(kr.encrypt(b"z", b"a"), b"a") == b"z"


def test_media_store(tmp_path):
    store = MediaStore(tmp_path, KeyRing.parse(new_key_spec("k")))
    store.put("abc123", "video", b"\x1a\x45\xdf\xa3data")
    raw = (tmp_path / "abc123" / "video.bin").read_bytes()
    assert raw.startswith(b"CMV1") and b"data" not in raw
    assert store.get("abc123", "video") == b"\x1a\x45\xdf\xa3data"
    store.delete_session("abc123")
    assert not store.exists("abc123", "video")


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 65])
def test_media_store_rejects_path_traversal(tmp_path, bad):
    store = MediaStore(tmp_path, KeyRing.parse(new_key_spec("k")))
    with pytest.raises(ValueError):
        store.put(bad, "video", b"x")
