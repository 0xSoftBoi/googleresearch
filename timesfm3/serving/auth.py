"""API keys, plans, quotas and usage metering.

The billable unit is a **forecast point**: one series x one horizon step.
A forecast of 3 series over 24 steps costs 72 points; a backtest costs
series x windows x horizon x models; anomaly detection costs one point per
scored observation.  That is the unit competitors meter on (predictions,
API calls) expressed in a way that scales with actual compute.

Configuration (any combination):

- ``TIMESFM3_API_KEY=secret`` -- one unlimited key named ``default``.
- ``TIMESFM3_API_KEYS=name:key[:monthly_points],...`` -- several keys inline.
- ``TIMESFM3_API_KEYS_FILE=keys.json`` -- ``{"keys": [{"key": "...",
  "name": "...", "plan": "team", "monthly_points": 5000000}]}``.
- ``TIMESFM3_USAGE_FILE=usage.json`` -- persist counters across restarts.

With no keys configured the service is open and metered under one
anonymous key, so ``/v1/usage`` and the usage headers still work.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import tempfile
import threading

ANONYMOUS = "anonymous"


@dataclasses.dataclass(frozen=True)
class ApiKey:
    key: str
    name: str
    plan: str = "custom"
    monthly_points: int | None = None  # None = unlimited

    @property
    def unlimited(self) -> bool:
        return self.monthly_points is None


class KeyStore:
    def __init__(self, keys: list[ApiKey] | None = None):
        self._by_key: dict[str, ApiKey] = {}
        self._by_name: dict[str, ApiKey] = {}
        for k in keys or []:
            self.add(k)

    def add(self, key: ApiKey) -> None:
        if key.name in self._by_name:
            raise ValueError(f"duplicate API key name {key.name!r}")
        self._by_key[key.key] = key
        self._by_name[key.name] = key

    @property
    def open(self) -> bool:
        """True when no keys are configured (unauthenticated access)."""
        return not self._by_key

    def lookup(self, key: str | None) -> ApiKey | None:
        if key is None:
            return None
        return self._by_key.get(key)

    def names(self) -> list[str]:
        return list(self._by_name)

    @classmethod
    def from_env(cls, api_key: str | None = None) -> "KeyStore":
        store = cls()
        single = api_key or os.environ.get("TIMESFM3_API_KEY")
        if single:
            store.add(ApiKey(key=single, name="default", plan="unlimited"))
        inline = os.environ.get("TIMESFM3_API_KEYS", "")
        for spec in filter(None, (s.strip() for s in inline.split(","))):
            parts = spec.split(":")
            if len(parts) < 2:
                raise ValueError(
                    f"TIMESFM3_API_KEYS entry {spec!r} must be name:key[:monthly_points]"
                )
            quota = int(parts[2]) if len(parts) > 2 and parts[2] else None
            store.add(ApiKey(key=parts[1], name=parts[0],
                             plan="metered" if quota else "unlimited", monthly_points=quota))
        path = os.environ.get("TIMESFM3_API_KEYS_FILE")
        if path:
            with open(path) as f:
                doc = json.load(f)
            for entry in doc.get("keys", []):
                store.add(ApiKey(
                    key=entry["key"], name=entry["name"], plan=entry.get("plan", "custom"),
                    monthly_points=entry.get("monthly_points"),
                ))
        return store


class QuotaExceeded(Exception):
    def __init__(self, key: ApiKey, used: int, requested: int):
        self.key, self.used, self.requested = key, used, requested
        super().__init__(
            f"Monthly quota of {key.monthly_points} forecast points exhausted for "
            f"{key.name!r} ({used} used, {requested} requested)."
        )


def _month(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m")


class UsageMeter:
    """Per-key monthly counters; optional write-through JSON persistence."""

    def __init__(self, path: str | None = None):
        self._lock = threading.Lock()
        self._path = path
        self._usage: dict[str, dict[str, dict[str, int]]] = {}  # month -> name -> counters
        if path and os.path.exists(path):
            with open(path) as f:
                self._usage = json.load(f)

    def _bucket(self, name: str, month: str) -> dict[str, int]:
        return self._usage.setdefault(month, {}).setdefault(
            name, {"points": 0, "requests": 0}
        )

    def _persist(self) -> None:
        if not self._path:
            return
        d = os.path.dirname(os.path.abspath(self._path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".usage-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(self._usage, f)
        os.replace(tmp, self._path)

    def charge(self, key: ApiKey, points: int, month: str | None = None) -> dict[str, int]:
        """Adds ``points`` for the key; raises :class:`QuotaExceeded` first if over."""
        month = month or _month()
        with self._lock:
            bucket = self._bucket(key.name, month)
            if not key.unlimited and bucket["points"] + points > key.monthly_points:
                raise QuotaExceeded(key, bucket["points"], points)
            bucket["points"] += points
            bucket["requests"] += 1
            self._persist()
            return dict(bucket)

    def usage(self, key: ApiKey, month: str | None = None) -> dict:
        month = month or _month()
        with self._lock:
            bucket = dict(self._bucket(key.name, month))
        remaining = None if key.unlimited else max(0, key.monthly_points - bucket["points"])
        return {
            "name": key.name, "plan": key.plan, "month": month,
            "points_used": bucket["points"], "requests": bucket["requests"],
            "monthly_quota": key.monthly_points, "points_remaining": remaining,
        }

    def all_usage(self, month: str | None = None) -> dict[str, dict[str, int]]:
        month = month or _month()
        with self._lock:
            return {k: dict(v) for k, v in self._usage.get(month, {}).items()}
