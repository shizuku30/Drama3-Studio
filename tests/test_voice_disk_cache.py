from pathlib import Path
import os

from drama_low_vram.voice_disk_cache import DiskVoiceCache, studio_voice_cache, voice_cache_identity


def role(number):
    return voice_cache_identity("test-card", number)


def test_more_than_twelve_distinct_cards_persist_without_opening_prefixes(tmp_path):
    root = tmp_path / "cache"
    cache = DiskVoiceCache(root, byte_budget=8192, retain_free=0)
    for i in range(20):
        assert cache.write_codes(role(i), "reference.wav", "fingerprint", [[i]])
    reopened = DiskVoiceCache(root, byte_budget=8192, retain_free=0)
    assert [reopened.read_codes(role(i), "reference.wav", "fingerprint") for i in range(20)] == [
        [[i]] for i in range(20)]
    assert len(list(root.glob("*.codes"))) == 20
    assert reopened.read_codes(role(0), "reference.wav", "wrong") is None


def test_disk_limit_evicts_the_oldest_prefix_but_keeps_codec_and_card(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    original = library / "reference.wav"
    original.write_bytes(b"the original card")

    root = tmp_path / "cache"
    cache = DiskVoiceCache(root, byte_budget=3000, retain_free=0)
    for i in range(3):
        key = role(i)
        assert cache.write_codes(key, "reference.wav", "same-model", [[i]])
        pending, limit = cache.plan_prefix(key, maximum=1200)
        assert pending is not None and limit == 1200
        pending.write_bytes(b"p" * limit)
        if i == 2:
            # Mark the first role as used after the second.
            cache.touch_prefix(role(0))
            os.utime(root / (role(1) + ".prefix"), ns=(1, 1))
            os.utime(root / (role(1) + ".codes"), ns=(1, 1))
        assert cache.publish_prefix(key, pending)
        assert not pending.exists()
    assert cache.read_prefix(role(0))
    assert cache.read_prefix(role(1)) is None
    assert cache.read_prefix(role(2))
    assert cache.read_codes(role(1), "reference.wav", "same-model") == [[1]]
    assert original.read_bytes() == b"the original card"
    assert sum(path.stat().st_size for path in root.iterdir()) <= 3000


def test_prefix_too_large_and_disk_floor_leave_no_pending_file(tmp_path):
    root = tmp_path / "cache"
    cache = DiskVoiceCache(root, byte_budget=200, retain_free=0)
    pending, limit = cache.plan_prefix(role(0), maximum=500)
    assert limit == 200
    pending.write_bytes(b"x" * 201)
    assert not cache.publish_prefix(role(0), pending)
    assert list(root.iterdir()) == []
    huge_floor = DiskVoiceCache(root, byte_budget=200, retain_free=10**30)
    assert huge_floor.plan_prefix(role(0), maximum=150)[0] is None


def test_cache_location_separates_installations(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_GGUF_VOICE_CACHE_DIR", str(tmp_path / "network-safe-cache"))
    left = studio_voice_cache(tmp_path / "first")
    right = studio_voice_cache(tmp_path / "second")
    assert left.parent == right.parent == tmp_path / "network-safe-cache"
    assert left != right
