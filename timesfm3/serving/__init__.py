"""Forecasting-as-a-service: model registry, REST API, dashboard.

``create_app`` needs the ``serve`` extra (FastAPI, uvicorn, pydantic);
:class:`ModelRegistry` only needs the core package.
"""

from .registry import ModelEntry, ModelRegistry

__all__ = ["ModelEntry", "ModelRegistry", "create_app"]


def create_app(*args, **kwargs):  # lazy: keeps fastapi optional
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
