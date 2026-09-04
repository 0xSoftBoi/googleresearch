"""Configuration for TimesFM-3.

The base configuration targets roughly 330M parameters, matching the model
size described in the TimesFM-3 announcement. A small configuration is
provided for local experimentation.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TimesFM3Config:
    """Hyper-parameters of the TimesFM-3 architecture.

    Attributes:
        patch_len: Number of contiguous time steps grouped into one patch.
            TimesFM-3 uses patches of 32 time steps.
        model_dim: Transformer hidden dimension.
        num_layers: Number of alternating transformer layers. Layers with an
            even index apply causal temporal attention (across time, within a
            series); layers with an odd index apply cross-variate attention
            (across series, within a time step).
        num_heads: Attention heads for both attention types.
        ff_dim: Hidden dimension of the feed-forward blocks.
        dropout: Dropout rate (0 for inference / zero-shot use).
        lookahead_patches: For past-future ("known ahead") covariates, each
            token concatenates the current patch with this many future
            patches so the model can peek at upcoming known signals.
        quantiles: Quantile levels predicted per target and horizon step.
            TimesFM-3 predicts 9 quantiles from the 10th to the 90th
            percentile, alongside a point (mean) forecast.
        max_context_patches: Maximum number of context patches the
            forecaster feeds the model (older history is cropped). The
            TimesFM 2.5 documents a 16k-step context; Google's TimesFM-3
            blog post and model card do not state a limit, so 16k is this
            implementation's choice, not a verified released-model figure.
        max_horizon_patches: Maximum number of contiguous masked patches the
            model decodes in a single forward pass. The high-level
            forecaster extends beyond this by rolling the decode forward.
    """

    patch_len: int = 32
    model_dim: int = 1280
    num_layers: int = 20
    num_heads: int = 16
    ff_dim: int = 3840
    dropout: float = 0.0
    lookahead_patches: int = 1
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    max_context_patches: int = 512
    max_horizon_patches: int = 8

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def output_dim_per_step(self) -> int:
        # One point forecast plus one value per quantile level.
        return 1 + self.num_quantiles

    @property
    def max_context_len(self) -> int:
        return self.max_context_patches * self.patch_len

    @property
    def max_horizon_len(self) -> int:
        return self.max_horizon_patches * self.patch_len

    @classmethod
    def base(cls) -> "TimesFM3Config":
        """~330M parameter configuration.

        Matches the released TimesFM-3 dimensions: 20 transformer layers,
        model dimension 1280, 16 attention heads, 32-step input patches.
        """
        return cls()

    @classmethod
    def small(cls) -> "TimesFM3Config":
        """A small configuration for demos and CPU experimentation."""
        return cls(
            model_dim=256,
            num_layers=6,
            num_heads=8,
            ff_dim=1024,
            max_context_patches=64,
        )

    @classmethod
    def tiny(cls) -> "TimesFM3Config":
        """A tiny configuration that trains in minutes on CPU."""
        return cls(
            model_dim=128,
            num_layers=4,
            num_heads=4,
            ff_dim=512,
            max_context_patches=32,
            max_horizon_patches=4,
        )
