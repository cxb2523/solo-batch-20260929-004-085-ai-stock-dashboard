"""
Configuration layer for the stock dashboard.

Precedence (highest wins):
    environment variables  >  config.yaml  >  built-in DEFAULTS

Merge rules (see :func:`load_config`):
  * scalars are replaced wholesale -- the overriding value replaces the
    lower-priority value, or, when it fails validation, falls back to the
    built-in default. Bad input never raises;
  * lists are merged item-by-item by index. Overrides can only replace
    existing indices; an override item whose index is >= the base-list length
    is dropped and one diagnostic is recorded;
  * an unset environment variable means "no override" (yaml/default shine
    through); a variable set to an empty string means "reset this key to its
    built-in default" -- the two cases are handled differently;
  * every leaf carries provenance ("env" / "yaml" / "default") and every
    rejected input produces a diagnostic.

The returned :class:`ConfigSnapshot` is a deep-frozen, read-only view. Modules
never mutate it; :class:`ConfigStore` does mtime based hot reload and swaps
snapshots atomically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger("stock_dashboard.config")

# ---------------------------------------------------------------------------
# Built-in defaults (lowest precedence)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "tickers": ["AAPL", "MSFT", "GOOG"],
    "indicator_windows": {
        "sma_short": 20,
        "sma_long": 50,
        "rsi_window": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    },
    "cache": {
        "ttl_seconds": 300,
    },
    "datasources": {
        "yfinance_enabled": True,
        "local_fixture_enabled": True,
    },
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Minimum sane value for numeric windows. Bigger-than-data windows are fine
# (they just yield NaN until enough history accumulates).
_MIN_POSITIVE_INT = 2


# ---------------------------------------------------------------------------
# Freeze / thaw
# ---------------------------------------------------------------------------
def _freeze(value: Any) -> Any:
    """Recursively make *value* immutable (dict -> proxy, list -> tuple)."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze` -- plain dict/list primitives."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _iter_leaves(
    schema: Any, path: Tuple[str, ...] = ()
) -> Iterable[Tuple[Tuple[str, ...], str, Any]]:
    """Walk the default schema, yielding ``(path, kind, default_value)``."""
    if isinstance(schema, Mapping):
        for key, sub in schema.items():
            yield from _iter_leaves(sub, path + (str(key),))
    elif isinstance(schema, (list, tuple)):
        yield path, "list", list(schema)
    elif isinstance(schema, bool):
        yield path, "bool", schema
    elif isinstance(schema, int):
        yield path, "int", schema
    else:
        yield path, "str", schema


_LEAF_KINDS = {path: kind for path, kind, _ in _iter_leaves(DEFAULTS)}


def env_var_name(path: Sequence[str]) -> str:
    """Config path -> same-named env var.

    ``indicator_windows.sma_short`` -> ``INDICATOR_WINDOWS__SMA_SHORT``.
    Nested keys join on a double underscore; list values are comma separated.
    """
    return "__".join(part.upper() for part in path)


# ---------------------------------------------------------------------------
# Scalar / list coercion -- invalid input falls back, never raises
# ---------------------------------------------------------------------------
def _coerce_scalar(raw: Any, kind: str, default: Any) -> Tuple[bool, Any]:
    """Return ``(ok, value)``; on failure caller falls back to *default*."""
    try:
        if kind == "bool":
            if isinstance(raw, bool):
                return True, raw
            text = str(raw).strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True, True
            if text in ("0", "false", "no", "off"):
                return True, False
            return False, default
        if kind == "int":
            if isinstance(raw, bool):  # bool is an int subclass -- reject
                return False, default
            if isinstance(raw, int):
                value = raw
            else:
                value = int(str(raw).strip())  # int("12.0") raises, intended
            if value < _MIN_POSITIVE_INT:
                return False, default
            return True, value
        text = str(raw).strip()  # kind == "str"
        if not text:
            return False, default
        return True, text.upper()
    except (TypeError, ValueError):
        return False, default


def _coerce_env_list(raw: str, default: Sequence[Any]) -> Tuple[bool, List[Any]]:
    """Parse a comma-separated env override into ticker-like strings."""
    cleaned: List[Any] = []
    for piece in str(raw).split(","):
        ok, value = _coerce_scalar(piece, "str", None)
        if not ok:
            return False, list(default)
        cleaned.append(value)
    return True, cleaned


def _valid_ticker(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 20 and " " not in value


# ---------------------------------------------------------------------------
# Merge primitives
# ---------------------------------------------------------------------------
def _merge_lists(
    path: Tuple[str, ...],
    base: Sequence[Any],
    override: Sequence[Any],
    source: str,
    sources: Dict[str, Any],
    diagnostics: List[str],
) -> List[Any]:
    """Merge lists by index; out-of-range override indices are dropped."""
    merged = list(base)
    index_sources = sources.setdefault("__list__", {})
    dropped: List[Tuple[int, Any]] = []
    for index, new_value in enumerate(override):
        if index >= len(merged):
            dropped.append((index, new_value))
            continue
        if not _valid_ticker(new_value):
            diagnostics.append(
                f"{'.'.join(path)}[{index}]: invalid {source} value "
                f"{new_value!r}; kept {merged[index]!r}"
            )
            continue
        merged[index] = new_value
        index_sources[index] = source
    if dropped:
        shown = ", ".join(f"[{idx}]={val!r}" for idx, val in dropped)
        diagnostics.append(
            f"{'.'.join(path)}: {len(dropped)} override item(s) out of range "
            f"for base list of length {len(merged)} dropped ({shown})"
        )
    return merged


def _merge_mapping(
    path: Tuple[str, ...],
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    source: str,
    sources: Dict[str, Any],
    diagnostics: List[str],
) -> Dict[str, Any]:
    """Recursive, schema-aware merge of *override* on top of *base*."""
    result: Dict[str, Any] = {}
    for key, base_value in base.items():
        key_path = path + (key,)
        if key not in override:
            result[key] = deepcopy(base_value)
            continue
        override_value = override[key]
        if isinstance(base_value, Mapping):
            if not isinstance(override_value, Mapping):
                diagnostics.append(
                    f"{'.'.join(key_path)}: expected mapping from {source}, "
                    f"got {type(override_value).__name__}; using built-in default"
                )
                result[key] = deepcopy(base_value)
                sources[key] = {}
                continue
            sub_sources: Dict[str, Any] = {}
            result[key] = _merge_mapping(
                key_path, base_value, override_value, source,
                sub_sources, diagnostics,
            )
            sources[key] = sub_sources
        elif isinstance(base_value, (list, tuple)):
            if not isinstance(override_value, (list, tuple)):
                diagnostics.append(
                    f"{'.'.join(key_path)}: expected list from {source}, got "
                    f"{type(override_value).__name__}; using built-in default"
                )
                result[key] = list(base_value)
                sources[key] = {}
                continue
            sub_sources = {}
            result[key] = _merge_lists(
                key_path, base_value, override_value, source,
                sub_sources, diagnostics,
            )
            sources[key] = sub_sources
        else:
            kind = _LEAF_KINDS[key_path]
            ok, coerced = _coerce_scalar(override_value, kind, base_value)
            if ok:
                result[key] = coerced
                sources[key] = source
            else:
                diagnostics.append(
                    f"{'.'.join(key_path)}: invalid {source} value "
                    f"{override_value!r}; fell back to built-in default "
                    f"{base_value!r}"
                )
                result[key] = deepcopy(base_value)
    for key in override:
        if key not in base:
            diagnostics.append(
                f"{'.'.join(path + (key,))}: unknown key from {source} ignored"
            )
    return result


def _apply_env(
    config: Dict[str, Any],
    sources: Dict[str, Any],
    diagnostics: List[str],
    environ: Mapping[str, str],
) -> None:
    """Apply env overrides in place.

    Absent variable: no override. Present-but-empty: reset to default.
    """
    for leaf_path, kind, default_value in _iter_leaves(DEFAULTS):
        name = env_var_name(leaf_path)
        if name not in environ:
            continue
        raw = environ[name]

        node = config
        src_node = sources
        for part in leaf_path[:-1]:
            node = node[part]
            src_node = src_node[part]
        key = leaf_path[-1]
        dotted = ".".join(leaf_path)

        if raw == "":
            node[key] = deepcopy(default_value)
            if kind == "list":
                src_node[key] = {
                    "__list__": {i: "default" for i in range(len(default_value))}
                }
            else:
                src_node[key] = "default"
            diagnostics.append(
                f"{dotted}: {name} set to empty string; reset to built-in "
                f"default {default_value!r}"
            )
            continue

        if kind == "list":
            ok, coerced = _coerce_env_list(raw, default_value)
            if not ok:
                diagnostics.append(
                    f"{dotted}: invalid env value {raw!r}; kept current "
                    f"value {node[key]!r}"
                )
                continue
            sub_sources: Dict[str, Any] = {}
            node[key] = _merge_lists(
                leaf_path, node[key], coerced, "env", sub_sources, diagnostics
            )
            src_node[key] = sub_sources
        else:
            ok, coerced = _coerce_scalar(raw, kind, default_value)
            if ok:
                node[key] = coerced
                src_node[key] = "env"
            else:
                diagnostics.append(
                    f"{dotted}: invalid env value {raw!r}; fell back to "
                    f"built-in default {default_value!r}"
                )
                node[key] = deepcopy(default_value)
                src_node[key] = "default"


def _default_sources() -> Dict[str, Any]:
    """Provenance tree where every leaf is "default"."""
    sources: Dict[str, Any] = {}
    for path, kind, default_value in _iter_leaves(DEFAULTS):
        node = sources
        for part in path[:-1]:
            node = node.setdefault(part, {})
        if kind == "list":
            node[path[-1]] = {
                "__list__": {i: "default" for i in range(len(default_value))}
            }
        else:
            node[path[-1]] = "default"
    return sources


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
class ConfigSnapshot:
    """Immutable, read-only view of one generation of configuration."""

    def __init__(
        self,
        values: Mapping[str, Any],
        sources: Mapping[str, Any],
        diagnostics: Sequence[str],
        config_path: Optional[Path],
        yaml_mtime: Optional[float],
    ) -> None:
        self._values = _freeze(values)
        self._sources = _freeze(sources)
        self._diagnostics = tuple(diagnostics)
        self._config_path = str(config_path) if config_path else None
        self._yaml_mtime = yaml_mtime
        self._fingerprint = hashlib.sha256(
            json.dumps(_thaw(self._values), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def values(self) -> Mapping[str, Any]:
        return self._values

    @property
    def sources(self) -> Mapping[str, Any]:
        return self._sources

    @property
    def diagnostics(self) -> Tuple[str, ...]:
        return self._diagnostics

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @property
    def yaml_mtime(self) -> Optional[float]:
        return self._yaml_mtime

    # Typed convenience accessors used by other (read-only) modules --------
    @property
    def tickers(self) -> Tuple[str, ...]:
        return tuple(self._values["tickers"])

    @property
    def indicator_windows(self) -> Mapping[str, int]:
        return self._values["indicator_windows"]

    @property
    def cache_ttl_seconds(self) -> int:
        return int(self._values["cache"]["ttl_seconds"])

    @property
    def datasources(self) -> Mapping[str, bool]:
        return self._values["datasources"]

    def to_plain(self) -> Dict[str, Any]:
        """JSON-serialisable copy (plain dicts/lists) for API responses."""
        return {
            "fingerprint": self._fingerprint,
            "config_path": self._config_path,
            "yaml_mtime": self._yaml_mtime,
            "values": _thaw(self._values),
            "sources": _thaw(self._sources),
            "diagnostics": list(self._diagnostics),
        }


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
def _read_yaml(path: Path, diagnostics: List[str]) -> Tuple[Dict[str, Any], Optional[float]]:
    """Load YAML defensively. Missing/unreadable/bad YAML -> ({}, mtime)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        diagnostics.append(
            f"{path}: could not parse config ({exc.__class__.__name__}: "
            f"{exc}); using built-in defaults"
        )
        return {}, mtime
    if loaded is None:
        return {}, mtime
    if not isinstance(loaded, Mapping):
        diagnostics.append(
            f"{path}: top-level YAML must be a mapping, got "
            f"{type(loaded).__name__}; using built-in defaults"
        )
        return {}, mtime
    return dict(loaded), mtime


def load_config(
    config_path: Optional[os.PathLike | str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ConfigSnapshot:
    """Build one frozen :class:`ConfigSnapshot`.

    Precedence: *environ* > YAML file at *config_path* > :data:`DEFAULTS`.

    Any problem (missing file, bad YAML, bad scalar, out-of-range list index,
    ...) is recorded as a diagnostic and degraded to a safe value -- this
    function never raises.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    diagnostics: List[str] = []
    raw_yaml, yaml_mtime = _read_yaml(path, diagnostics)

    values = deepcopy(DEFAULTS)
    sources = _default_sources()

    # Layer 1: YAML on top of defaults (scalars replace, lists merge by index).
    if raw_yaml:
        values = _merge_mapping(
            (), values, raw_yaml, "yaml", sources, diagnostics
        )

    # Layer 2: environment variables win over everything.
    _apply_env(values, sources, diagnostics, env)

    snapshot = ConfigSnapshot(
        values=values,
        sources=sources,
        diagnostics=diagnostics,
        config_path=path,
        yaml_mtime=yaml_mtime,
    )
    for message in diagnostics:
        logger.warning("config diagnostic: %s", message)
    return snapshot


# ---------------------------------------------------------------------------
# ConfigStore -- mtime hot reload with atomic snapshot swaps
# ---------------------------------------------------------------------------
class ConfigStore:
    """Holds the current :class:`ConfigSnapshot` and hot-reloads on mtime change.

    Reload is detected lazily (stat on :meth:`snapshot`, throttled to once per
    ``poll_interval`` seconds, also on changes to relevant env vars). The new
    snapshot is built fully before being published: the reference swap under a
    lock is atomic, so readers always see exactly one consistent generation,
    never a half-merged config.
    """

    def __init__(
        self,
        config_path: Optional[os.PathLike | str] = None,
        poll_interval: float = 0.1,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._path = (
            Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        )
        self._poll_interval = poll_interval
        self._environ = environ
        self._lock = threading.Lock()
        self._snapshot: ConfigSnapshot = self._build()
        self._last_stat_mtime = self._snapshot.yaml_mtime
        self._last_env_sig = self._env_signature(
            os.environ if environ is None else environ
        )
        self._last_check = 0.0

    def _build(self) -> ConfigSnapshot:
        return load_config(self._path, self._environ)

    @staticmethod
    def _env_signature(environ: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            sorted(
                (name, environ.get(name, ""))
                for path, _, _ in _iter_leaves(DEFAULTS)
                for name in (env_var_name(path),)
                if name in environ
            )
        )

    def _current_signature(self) -> Tuple[Optional[float], Tuple[Tuple[str, str], ...]]:
        try:
            current_mtime: Optional[float] = self._path.stat().st_mtime
        except OSError:
            current_mtime = None
        env = os.environ if self._environ is None else self._environ
        return current_mtime, self._env_signature(env)

    def snapshot(self) -> ConfigSnapshot:
        """Return the current snapshot, reloading first if sources changed.

        Stat is throttled to once per ``poll_interval`` seconds. The rebuild
        happens outside the lock; publishing is a single locked attribute
        assignment, so concurrent readers either get the old generation or the
        new one -- never a partially merged snapshot.
        """
        import time

        now = time.monotonic()
        if now - self._last_check >= self._poll_interval:
            mtime, env_sig = self._current_signature()
            if mtime != self._last_stat_mtime or env_sig != self._last_env_sig:
                new_snapshot = self._build()
                with self._lock:
                    self._snapshot = new_snapshot  # atomic reference swap
                    self._last_stat_mtime = mtime
                    self._last_env_sig = env_sig
                    self._last_check = time.monotonic()
            else:
                self._last_check = now
        return self._snapshot

    def force_reload(self) -> ConfigSnapshot:
        """Force a rebuild regardless of mtime/poll throttle."""
        with self._lock:
            self._snapshot = self._build()
            self._last_stat_mtime = self._snapshot.yaml_mtime
            env = os.environ if self._environ is None else self._environ
            self._last_env_sig = self._env_signature(env)
        return self._snapshot


# Process-wide singleton used by the web layer and other read-only modules.
store = ConfigStore()


def get_snapshot() -> ConfigSnapshot:
    """Convenience accessor: current frozen config snapshot."""
    return store.snapshot()
