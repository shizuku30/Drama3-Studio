"""Bounded, rebuildable voice acceleration files, separate from voice-library assets."""

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid


MIB = 1024**2
GIB = 1024 * MIB
DEFAULT_VOICE_CACHE_BYTES = 2 * GIB
MAX_CODEC_FILE = 8 * MIB
MAX_PREFIX_FILE = 512 * MIB
RETAIN_FREE_BYTES = GIB
ENTRY = re.compile(r"([a-f0-9]{64})\.(codes|prefix)$")
PENDING = re.compile(r"[a-f0-9]{64}\.[a-f0-9]{32}\.pending$")


def voice_cache_identity(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def studio_voice_cache(root):
    """Namespace separate Studio installations so one instance cannot evict another's request."""
    root = Path(root).resolve()
    override = os.environ.get("DRAMA_GGUF_VOICE_CACHE_DIR")
    local = os.environ.get("LOCALAPPDATA")
    if override or local:
        base = Path(override) if override else Path(local) / "Drama3Studio" / "voice-cache"
        return base / voice_cache_identity(os.path.normcase(str(root)))[:20]
    return root / ".cache" / "voice-cache"


class DiskVoiceCache:
    """Store one codec map and one native prefix per card/model key.

    There is no card count limit. Recency and sizes live in each file's
    directory entry; reads never load other cards or whole prefix blobs.
    """

    def __init__(self, root, *, byte_budget=DEFAULT_VOICE_CACHE_BYTES, retain_free=RETAIN_FREE_BYTES):
        self.root = Path(root).resolve()
        self.byte_budget = byte_budget
        self.retain_free = retain_free
        self._lock = threading.RLock()
        self._prepared = False

    def _path(self, key, kind):
        if re.fullmatch(r"[a-f0-9]{64}", key) is None or kind not in ("codes", "prefix"):
            raise ValueError("Invalid voice cache entry")
        return self.root / f"{key}.{kind}"

    def _files(self):
        files = []
        for path in self.root.iterdir():
            match = ENTRY.fullmatch(path.name)
            if match and not path.is_symlink() and path.is_file():
                stat = path.stat()
                files.append((match[1], match[2], stat.st_size, stat.st_mtime_ns, path))
        return files

    @staticmethod
    def _touch(path):
        with suppress(OSError):
            os.utime(path, None)

    def _read_map(self, path):
        try:
            if path.is_symlink():
                return {}
            with path.open("rb") as handle:
                data = handle.read(MAX_CODEC_FILE + 1)
            if len(data) <= MAX_CODEC_FILE:
                result = json.loads(data)
                if isinstance(result, dict):
                    return result
        except (OSError, ValueError, RecursionError):
            pass
        return {}

    def read_codes(self, key, clip, fingerprint):
        with self._lock:
            path = self._path(key, "codes")
            entry = self._read_map(path).get(clip)
            if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
                return None
            self._touch(path)
            return entry.get("codes")

    def write_codes(self, key, clip, fingerprint, codes):
        pending = None
        with self._lock:
            try:
                data = self._read_map(self._path(key, "codes"))
                data[clip] = {"fingerprint": fingerprint, "codes": codes}
                blob = json.dumps(data, separators=(",", ":")).encode("utf-8")
                if len(blob) > MAX_CODEC_FILE:
                    return False
                pending = self._pending(key, len(blob))
                if pending is None:
                    return False
                pending.write_bytes(blob)
                return self._publish(key, "codes", pending)
            except OSError:
                return False
            finally:
                self.discard(pending)

    def read_prefix(self, key):
        with self._lock:
            path = self._path(key, "prefix")
            try:
                if not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= MAX_PREFIX_FILE:
                    return path
            except OSError:
                pass
            return None

    def touch_prefix(self, key):
        self._touch(self._path(key, "prefix"))

    def _pending(self, key, size):
        self._path(key, "prefix")
        if not 0 < size <= self.byte_budget:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._prepared:
            # Normal cancellation removes the pending file in finally. Reclaim
            # leftovers from abruptly terminated processes on the next write.
            stale = time.time_ns() - 24 * 60 * 60 * 10**9
            for path in self.root.iterdir():
                if PENDING.fullmatch(path.name):
                    with suppress(OSError):
                        if path.lstat().st_mtime_ns < stale:
                            path.unlink()
            self._prepared = True
        if shutil.disk_usage(self.root).free - self.retain_free < size:
            self._trim(self._files(), key, shortage=size)
        if shutil.disk_usage(self.root).free - self.retain_free < size:
            return None
        return self.root / f"{key}.{uuid.uuid4().hex}.pending"

    def plan_prefix(self, key, *, maximum):
        with self._lock:
            try:
                limit = min(maximum, MAX_PREFIX_FILE, self.byte_budget)
                if limit <= 0:
                    return None, 0
                return self._pending(key, limit), limit
            except OSError:
                return None, 0

    def _trim(self, files, keep, *, shortage=0, projected=None):
        ages = defaultdict(int)
        for key, _, _, age, _ in files:
            ages[key] = max(ages[key], age)
        # Drop older prefix files before reclaiming the far smaller codec maps.
        for key, kind, size, _, path in sorted(files, key=lambda row: (row[1] == "codes", ages[row[0]])):
            if key == keep:
                continue
            if projected is not None and projected <= self.byte_budget:
                break
            if projected is None and shutil.disk_usage(self.root).free >= self.retain_free + shortage:
                break
            path.unlink(missing_ok=True)
            if projected is not None:
                projected -= size
        return projected

    def _publish(self, key, kind, pending):
        destination = self._path(key, kind)
        if pending.parent != self.root or not PENDING.fullmatch(pending.name) or not pending.name.startswith(key + "."):
            raise ValueError("Invalid pending cache file")
        if pending.is_symlink():
            return False
        size = pending.stat().st_size
        limit = MAX_PREFIX_FILE if kind == "prefix" else MAX_CODEC_FILE
        if not 0 < size <= min(limit, self.byte_budget):
            return False
        files = [item for item in self._files() if item[4] != destination]
        projected = size + sum(item[2] for item in files)
        if self._trim(files, key, projected=projected) > self.byte_budget:
            return False
        os.replace(pending, destination)
        self._touch(destination)
        return True

    def publish_prefix(self, key, pending):
        with self._lock:
            try:
                return self._publish(key, "prefix", pending)
            except OSError:
                return False
            finally:
                self.discard(pending)

    def discard(self, pending):
        if pending is not None and pending.parent == self.root and PENDING.fullmatch(pending.name):
            with suppress(OSError):
                pending.unlink(missing_ok=True)
