"""TTL result cache whose keys are namespaced by config-snapshot fingerprint.

Changing any effective config value (e.g. sma_short 20 -> 50) changes the
fingerprint, so stale results computed under the old generation can never be
served under the new one. ``clear()`` drops everything; expired/fingerprint
mismatches simply miss.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class _Entry:
    fingerprint: str
    expires_at: float
    value: Any


class SnapshotCache:
    def __init__(self) -> None:
        self._store: Dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(fingerprint: str, namespace: str, key: str) -> str:
        return f"{fingerprint}:{namespace}:{key}"

    def get(self, fingerprint: str, namespace: str, key: str) -> Optional[Any]:
        full_key = self._make_key(fingerprint, namespace, key)
        with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                return None
            if entry.fingerprint != fingerprint or entry.expires_at < time.monotonic():
                del self._store[full_key]
                return None
            return entry.value

    def set(
        self, fingerprint: str, namespace: str, key: str,
        value: Any, ttl_seconds: int,
    ) -> None:
        full_key = self._make_key(fingerprint, namespace, key)
        with self._lock:
            self._store[full_key] = _Entry(
                fingerprint=fingerprint,
                expires_at=time.monotonic() + max(0, int(ttl_seconds)),
                value=value,
            )

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def entries(self) -> Tuple[str, ...]:
        """Inspect live (non-expired, current fingerprint) cache keys."""
        with self._lock:
            now = time.monotonic()
            return tuple(
                key for key, entry in self._store.items()
                if entry.expires_at >= now
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
