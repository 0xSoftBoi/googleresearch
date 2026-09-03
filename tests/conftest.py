import numpy as np
import pytest

from timesfm3 import TimesFM3Config, TimesFM3Forecaster
from timesfm3.serving.registry import ModelRegistry


@pytest.fixture(scope="session")
def tiny_forecaster():
    import torch

    torch.manual_seed(0)
    return TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu")


@pytest.fixture(scope="session")
def registry(tiny_forecaster):
    reg = ModelRegistry.from_env(include_bundled=False)
    reg.add_forecaster("tiny", tiny_forecaster, "tiny random-init model for tests")
    return reg


@pytest.fixture(scope="session")
def seasonal():
    rng = np.random.default_rng(0)
    t = np.arange(400)
    return 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, len(t))
