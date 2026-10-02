"""Configuration layer for the stock dashboard.

Precedence: environment variables > config.yaml > built-in defaults.

Merge rules (implemented in :func:`load_config`):

* Scalar leaves are replaced wholesale by the higher-precedence layer.
* List leaves are merged index by index; items at out-of-range indices are
  dropped and a diagnostic is recorded.
* Invalid overlay values never raise: the leaf keeps the lower-layer value
  (ultimately the built-in default) and a diagnostic is recorded.
* An environment variable that is unset differs from one set to an empty
  string: unset means "no override"; an empty string on a list leaf clears the
  whole list, on a scalar leaf it is recorded as an invalid explicit-empty.

Other modules consume only the returned :class:`ConfigSnapshot`, a deep-frozen
read-only view.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Mapping

import yaml

ENV_PREFIX = "SD_"
CONFIG_PATH_ENV = "STOCKDASH_CONFIG"
DEFAULT_YAML_PATH = "config.yaml"

_MISSING = object()


# ---------------------------------------------------------------------------
# Built-in defaults (lowest precedence)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "indicators": {
        "sma_windows": [20, 50, 200],
        "ema_spans": [12, 26],
        "rsi_window": 14,
        "macd_signal": 9,
        "volume_window": 20,
        "bollinger": {"window": 20, "std": 2.0},
        "atr_window": 14,
        "stoch": {"k_window": 14, "d_window": 3},
    },
    "cache": {"ttl_seconds": 300, "max_entries": 128},
    "sources": {
        "yfinance": {"enabled": True},
        "local_fixture": {"enabled": True, "path": "tests/fixtures/sample_prices.csv"},
    },
    "data": {"period": "1y"},
    "server": {"host": "127.0.0.1", "port": 8000},
}

# Leaf validation table.  kind: str | int | float | bool | list[str] | list[int]
# constraint: None | allowed-value set | per-value predicate
_LEAF_SPECS: dict[str, tuple[str, Any]] = {
    "tickers": ("list[str]", None),
    "indicators.sma_windows": ("list[int]", lambda v: 1 <= v <= 1000),
    "indicators.ema_spans": ("list[int]", lambda v: 1 <= v <= 1000),
    "indicators.rsi_window": ("int", lambda v: 1 <= v <= 500),
    "indicators.macd_signal": ("int", lambda v: 1 <= v <= 500),
    "indicators.volume_window": ("int", lambda v: 1 <= v <= 500),
    "indicators.bollinger.window": ("int", lambda v: 2 <= v <= 500),
    "indicators.bollinger.std": ("float", lambda v: 0 < v <= 10),
    "indicators.atr_window": ("int", lambda v: 1 <= v <= 500),
    "indicators.stoch.k_window": ("int", lambda v: 1 <= v <= 500),
    "indicators.stoch.d_window": ("int", lambda v: 1 <= v <= 500),
    "cache.ttl_seconds": ("int", lambda v: 0 <= v <= 24 * 3600),
    "cache.max_entries": ("int", lambda v: 1 <= v <= 100000),
    "sources.yfinance.enabled": ("bool", None),
    "sources.local_fixture.enabled": ("bool", None),
    "sources.local_fixture.path": ("str", bool),
    "data.period": ("str", {"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}),
    "server.host": ("str", bool),
    "server.port": ("int", lambda v: 1 <= v <= 65535),
}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Frozen read-only containers
# ---------------------------------------------------------------------------


class FrozenDict(dict):
    """Immutable dict; raises TypeError on any mutation attempt."""

    __slots__ = ("_frozen_hash",)

    def __readonly(self, *_args, **_kwargs):
        raise TypeError("config snapshot is read-only")

    __setitem__ = __readonly
    __delitem__ = __readonly
    pop = __readonly
    popitem = __readonly
    clear = __readonly
    update = __readonly
    setdefault = __readonly

    def __hash__(self):
        cached = getattr(self, "_frozen_hash", None)
        if cached is None:
            cached = hash(json.dumps(to_plain(self), sort_keys=True, default=str))
            object.__setattr__(self, "_frozen_hash", cached)
        return cached


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def to_plain(value: Any) -> Any:
    """Convert a frozen snapshot into plain JSON-friendly containers."""
    if isinstance(value, Mapping):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Validation / coercion
# ---------------------------------------------------------------------------


def _valid(kind: str, constraint: Any, value: Any) -> bool:
    if kind == "str":
        if not isinstance(value, str):
            return False
    elif kind == "bool":
        if not isinstance(value, bool):
            return False
    elif kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    elif kind in ("list[str]", "list[int]"):
        if not isinstance(value, (list, tuple)):
            return False
        if kind == "list[str]":
            if not all(isinstance(v, str) and v for v in value):
                return False
        elif not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return False
    else:  # pragma: no cover - defensive
        return False
    if callable(constraint):
        if kind in ("list[str]", "list[int]"):
            return all(constraint(v) for v in value)
        return bool(constraint(value))
    if isinstance(constraint, set):
        return value in constraint
    return True


def _coerce_scalar(kind: str, raw: str) -> tuple[bool, Any]:
    if kind == "str":
        return True, raw
    if kind == "bool":
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True, True
        if low in _BOOL_FALSE:
            return True, False
        return False, None
    if kind == "int":
        try:
            return True, int(raw.strip())
        except (TypeError, ValueError):
            return False, None
    if kind == "float":
        try:
            return True, float(raw.strip())
        except (TypeError, ValueError):
            return False, None
    return False, None


def _diag(diagnostics: list[dict[str, str]], code: str, path: str, message: str) -> None:
    diagnostics.append({"code": code, "path": path, "message": message})


# ---------------------------------------------------------------------------
# Merge core
# ---------------------------------------------------------------------------


def _env_name(path: str) -> str:
    return ENV_PREFIX + path.upper().replace(".", "_")


def _merge_yaml_list(
    path: str,
    base: list[Any],
    overlay: Any,
    kind: str,
    constraint: Any,
    diagnostics: list[dict[str, str]],
) -> list[Any]:
    """Index-wise list merge for the yaml layer; out-of-range items dropped."""
    result = list(base)
    if not isinstance(overlay, (list, tuple)):
        _diag(diagnostics, "invalid_yaml_value", path,
              f"expected list, got {type(overlay).__name__}; using defaults")
        return result
    for i, item in enumerate(overlay):
        if i >= len(result):
            _diag(diagnostics, "yaml_index_out_of_range", path,
                  f"yaml list index {i} out of range; dropped")
            continue
        if not _valid(kind, constraint, [item]):
            _diag(diagnostics, "invalid_yaml_value", path,
                  f"invalid yaml list item at index {i}: {item!r}; keeping default")
            continue
        result[i] = item
    return result


def _merge_leaf(
    path: str,
    default: Any,
    yaml_value: Any,
    env_raw: Any,
    diagnostics: list[dict[str, str]],
) -> tuple[Any, str]:
    """Merge one leaf across the three layers; returns (value, source)."""
    kind, constraint = _LEAF_SPECS[path]
    value = default
    source = "default"

    # ---- yaml layer (lists already index-merged) ----
    if yaml_value is not _MISSING:
        if _valid(kind, constraint, yaml_value):
            value = list(yaml_value) if isinstance(yaml_value, (list, tuple)) else yaml_value
            source = "yaml"
        else:
            _diag(diagnostics, "invalid_yaml_value", path,
                  f"invalid yaml value {yaml_value!r}; falling back")

    # ---- env layer ----
    if env_raw is not _MISSING:
        if kind in ("list[str]", "list[int]"):
            if env_raw == "":
                # Explicitly empty differs from unset: clear the whole list.
                return [], "env"
            merged = list(value)
            for i, part in enumerate(p.strip() for p in env_raw.split(",")):
                if kind == "list[str]":
                    ok, candidate = True, part
                    if not part:
                        ok = False
                else:
                    ok, candidate = _coerce_scalar("int", part)
                if not ok:
                    _diag(diagnostics, "invalid_env_value", path,
                          f"invalid env list item at index {i}: {part!r}; keeping lower layer")
                    continue
                if i >= len(merged):
                    _diag(diagnostics, "env_index_out_of_range", path,
                          f"env list index {i} out of range; dropped")
                    continue
                if not _valid(kind, constraint, [candidate]):
                    _diag(diagnostics, "invalid_env_value", path,
                          f"env list item at index {i} rejected: {candidate!r}")
                    continue
                merged[i] = candidate
            return merged, "env" if merged != value else source

        if env_raw == "":
            _diag(diagnostics, "env_explicit_empty", path,
                  "env var set to empty string; ignored (unset != empty)")
            return value, source
        ok, candidate = _coerce_scalar(kind, env_raw)
        if ok and _valid(kind, constraint, candidate):
            return candidate, "env"
        _diag(diagnostics, "invalid_env_value", path,
              f"invalid env value {env_raw!r}; keeping lower layer")

    return value, source


def _merge(
    path: str,
    default_node: Any,
    yaml_node: Any,
    env: Mapping[str, str],
    diagnostics: list[dict[str, str]],
) -> tuple[Any, Any]:
    if isinstance(default_node, dict):
        if yaml_node is not _MISSING and not isinstance(yaml_node, dict):
            _diag(diagnostics, "invalid_yaml_value", path or "<root>",
                  "expected mapping; ignoring section")
            yaml_node = _MISSING
        if yaml_node is not _MISSING:
            for unknown in yaml_node.keys() - default_node.keys():
                unknown_path = f"{path}.{unknown}" if path else unknown
                _diag(diagnostics, "unknown_yaml_key", unknown_path, "unknown key ignored")
        config_out: dict[str, Any] = {}
        source_out: dict[str, Any] = {}
        for key, default_child in default_node.items():
            child_path = f"{path}.{key}" if path else key
            yaml_child = yaml_node.get(key, _MISSING) if yaml_node is not _MISSING else _MISSING
            config_out[key], source_out[key] = _merge(
                child_path, default_child, yaml_child, env, diagnostics
            )
        return config_out, source_out

    kind, constraint = _LEAF_SPECS[path]
    env_raw = env.get(_env_name(path), _MISSING)

    if kind in ("list[str]", "list[int]") and yaml_node is not _MISSING:
        yaml_value: Any = _merge_yaml_list(
            path, default_node, yaml_node, kind, constraint, diagnostics
        )
    else:
        yaml_value = yaml_node

    return _merge_leaf(path, default_node, yaml_value, env_raw, diagnostics)


def _fingerprint(config: Mapping[str, Any]) -> str:
    blob = json.dumps(to_plain(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class ConfigSnapshot:
    """Immutable read-only configuration snapshot."""

    __slots__ = ("config", "sources", "fingerprint", "diagnostics",
                 "yaml_path", "loaded_at", "mtime_ns")

    def __init__(self, config, sources, fingerprint, diagnostics, yaml_path, loaded_at, mtime_ns):
        for slot, val in (
            ("config", config),
            ("sources", sources),
            ("fingerprint", fingerprint),
            ("diagnostics", diagnostics),
            ("yaml_path", yaml_path),
            ("loaded_at", loaded_at),
            ("mtime_ns", mtime_ns),
        ):
            object.__setattr__(self, slot, val)

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __contains__(self, key):
        return key in self.config

    def to_plain(self):
        return {
            "fingerprint": self.fingerprint,
            "yaml_path": self.yaml_path,
            "loaded_at": self.loaded_at,
            "mtime_ns": self.mtime_ns,
            "diagnostics": to_plain(self.diagnostics),
            "sources": to_plain(self.sources),
            "config": to_plain(self.config),
        }


def load_config(
    yaml_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ConfigSnapshot:
    """Merge defaults < yaml < env into one frozen snapshot.

    Never raises on bad overlay input: problems are recorded in
    ``snapshot.diagnostics`` and the affected leaves keep lower-layer values.
    """
    if yaml_path is None:
        yaml_path = os.environ.get(CONFIG_PATH_ENV, DEFAULT_YAML_PATH)
    if env is None:
        env = os.environ

    diagnostics: list[dict[str, str]] = []
    raw_yaml: Any = None
    mtime_ns: int | None = None

    try:
        mtime_ns = os.stat(yaml_path).st_mtime_ns
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw_yaml = yaml.safe_load(fh)
    except FileNotFoundError:
        _diag(diagnostics, "yaml_missing", yaml_path, "config file not found; using defaults")
    except (OSError, yaml.YAMLError) as exc:
        _diag(diagnostics, "yaml_unreadable", yaml_path, f"{type(exc).__name__}: {exc}; using defaults")

    if raw_yaml is not None and not isinstance(raw_yaml, dict):
        _diag(diagnostics, "invalid_yaml_value", yaml_path,
              f"top-level yaml is {type(raw_yaml).__name__}, expected mapping; using defaults")
        raw_yaml = None

    merged, sources = _merge("", DEFAULT_CONFIG, raw_yaml if raw_yaml is not None else _MISSING,
                             env, diagnostics)
    frozen_config = freeze(merged)
    frozen_sources = freeze(sources)
    frozen_diag = tuple(freeze(d) for d in diagnostics)

    return ConfigSnapshot(
        config=frozen_config,
        sources=frozen_sources,
        fingerprint=_fingerprint(frozen_config),
        diagnostics=frozen_diag,
        yaml_path=yaml_path,
        loaded_at=time.time(),
        mtime_ns=mtime_ns,
    )


# ---------------------------------------------------------------------------
# mtime-based hot reload store
# ---------------------------------------------------------------------------


class ConfigStore:
    """Holds the one live :class:`ConfigSnapshot`.

    ``snapshot()`` reloads from yaml when its mtime changes.  The new snapshot
    is built fully before the reference is swapped, so readers either see the
    old or the new snapshot - never a half-merged one.
    """

    def __init__(self, yaml_path: str | None = None, env: Mapping[str, str] | None = None):
        self._yaml_path = yaml_path
        self._env = env
        self._lock = threading.RLock()
        self._snapshot = load_config(yaml_path=yaml_path, env=env)

    def snapshot(self, force: bool = False) -> ConfigSnapshot:
        with self._lock:
            current = self._snapshot
            if not force:
                try:
                    mtime_ns = os.stat(current.yaml_path).st_mtime_ns
                except OSError:
                    mtime_ns = None
                if mtime_ns == current.mtime_ns:
                    return current
            new_snapshot = load_config(yaml_path=current.yaml_path, env=self._env)
            # Atomic reference swap.
            self._snapshot = new_snapshot
            return new_snapshot


# ---------------------------------------------------------------------------
# Snapshot-scoped cache
# ---------------------------------------------------------------------------


class SnapshotCache:
    """TTL cache whose keys are namespaced by the config snapshot fingerprint.

    Once the snapshot changes fingerprint (yaml hot reload / env change), old
    entries are invisible to the new snapshot and expire lazily.
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 128):
        self._lock = threading.RLock()
        self._store: dict[tuple[str, str], tuple[float, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

    def _full_key(self, snapshot: ConfigSnapshot, key: str) -> tuple[str, str]:
        return (snapshot.fingerprint, key)

    def get(self, snapshot: ConfigSnapshot, key: str) -> Any:
        with self._lock:
            entry = self._store.get(self._full_key(snapshot, key))
            if entry is None:
                return None
            expires_at, value = entry
            if self.ttl_seconds > 0 and expires_at < time.time():
                self._store.pop(self._full_key(snapshot, key), None)
                return None
            return value

    def set(self, snapshot: ConfigSnapshot, key: str, value: Any) -> None:
        with self._lock:
            if self.ttl_seconds <= 0:
                return
            full = self._full_key(snapshot, key)
            if full not in self._store and len(self._store) >= self.max_entries:
                # Evict oldest entry overall.
                oldest = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest, None)
            self._store[full] = (time.time() + self.ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
