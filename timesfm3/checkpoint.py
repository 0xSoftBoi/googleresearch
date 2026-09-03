"""Checkpoint packaging: small, self-describing model files.

``timesfm3.train`` writes ``{"config", "model"}`` in full precision.  A
*packaged* checkpoint adds a ``meta`` dict (name, provenance, training
budget, validation loss) and stores weights in half precision, which
halves the file so a starter model can ship inside the repository and the
Docker image.  :meth:`TimesFM3Forecaster.from_checkpoint` reads both
layouts and always restores float32 weights.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import torch

from . import __version__

CHECKPOINT_FORMAT = 1


def package_checkpoint(
    src: str, dst: str, meta: dict[str, Any] | None = None, half: bool = True
) -> dict[str, Any]:
    """Re-saves ``src`` (a training checkpoint) as a packaged checkpoint."""
    state = torch.load(src, map_location="cpu", weights_only=False)
    weights = state["model"]
    if half:
        weights = {
            k: (v.half() if torch.is_floating_point(v) else v)
            for k, v in weights.items()
        }
    packaged = {
        "format": CHECKPOINT_FORMAT,
        "config": state["config"],
        "model": weights,
        "meta": {
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "timesfm3_version": __version__,
            "dtype": "float16" if half else "float32",
            **(meta or {}),
        },
    }
    torch.save(packaged, dst)
    return packaged["meta"]


def load_checkpoint(path: str) -> dict[str, Any]:
    """Loads either layout; weights come back as float32, ``meta`` always present."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in state["model"].items()
    }
    return {
        "config": state["config"],
        "model": weights,
        "meta": dict(state.get("meta") or {}),
    }
