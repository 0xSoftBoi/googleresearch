"""Fine-tune a checkpoint on a customer's own series, then prove it helped.

The recipe follows what the finance literature says about foundation
models (generic pre-training transfers weakly; domain adaptation is where
the gains are): start from a pre-trained checkpoint, train briefly at a low
learning rate on windows from the first ``train_fraction`` of the customer
panel, validate on windows from the held-out tail, package the result --
and then run the *same walk-forward backtest the service exposes* on that
tail so the report says whether fine-tuning beat the base model and the
classical baselines on data neither has seen.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np

from .checkpoint import load_checkpoint, package_checkpoint
from .configuration import TimesFM3Config
from .data.real import MixedCorpus, RealSource, RealWindowDataset
from .data.synthetic import SyntheticMultivariateCorpus
from .train import train


@dataclasses.dataclass
class FinetuneReport:
    output: str
    base_checkpoint: str
    steps: int
    best_val_loss: float
    minutes: float
    evaluation: dict | None  # run_backtest report on the held-out tail, or None


def finetune(
    values: np.ndarray,
    base_checkpoint: str,
    output: str,
    name: str = "finetuned",
    steps: int = 300,
    batch_size: int = 16,
    lr: float = 1e-4,
    context_patches: int = 8,
    horizon_patches: int = 2,
    train_fraction: float = 0.8,
    periods: tuple[int, ...] = (),
    synthetic_fraction: float = 0.2,
    device: str | None = None,
    evaluate: bool = True,
    eval_windows: int = 20,
    seed: int = 0,
    verbose: bool = True,
    scratch: str | None = None,
) -> FinetuneReport:
    """``values`` is (num_series, num_steps); NaN marks missing observations."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("values must be (num_series, num_steps).")
    state = load_checkpoint(base_checkpoint)
    cfg: TimesFM3Config = state["config"]
    window = (context_patches + horizon_patches) * cfg.patch_len
    n = values.shape[1]
    if int(n * train_fraction) < window or n - int(n * train_fraction) < window:
        raise ValueError(
            f"Need at least {int(np.ceil(window / min(train_fraction, 1 - train_fraction)))} "
            f"steps so both the training and held-out regions hold one "
            f"{window}-step window; got {n}."
        )
    source = RealSource(name=name, values=values, periods=tuple(periods))
    real = RealWindowDataset(
        cfg, [source], context_patches=context_patches, horizon_patches=horizon_patches,
        train_fraction=train_fraction, calendar=bool(periods), seed=seed,
    )
    dataset = real
    if synthetic_fraction > 0:
        synth = SyntheticMultivariateCorpus(
            cfg, context_patches=context_patches, horizon_patches=horizon_patches, seed=seed + 1
        )
        dataset = MixedCorpus(real, synth, primary_prob=1.0 - synthetic_fraction, seed=seed + 2)
    val = RealWindowDataset(
        cfg, [source], context_patches=context_patches, horizon_patches=horizon_patches,
        train_fraction=train_fraction, calendar=bool(periods), seed=seed + 3, tail=True,
    )
    scratch = scratch or output + ".train.pt"
    history: list = []
    t0 = time.time()
    train(
        cfg, steps=steps, batch_size=batch_size, peak_lr=lr,
        warmup_steps=max(1, min(50, steps // 10)), context_patches=context_patches,
        horizon_patches=horizon_patches, device=device, log_every=max(1, steps // 10),
        val_every=max(1, steps // 5), val_batches=2, checkpoint_path=scratch,
        dataset=dataset, val_dataset=val, history=history, init_state=state["model"],
        verbose=verbose,
    )
    minutes = (time.time() - t0) / 60
    best_val = min(h["val_loss"] for h in history if "val_loss" in h)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    package_checkpoint(scratch, output, meta={
        **{k: v for k, v in state["meta"].items() if k not in ("created", "dtype")},
        "name": name,
        "description": f"Fine-tuned from {state['meta'].get('name', base_checkpoint)} "
                       f"on {values.shape[0]} customer series ({steps} steps).",
        "base_checkpoint": base_checkpoint,
        "finetune_steps": steps,
        "finetune_lr": lr,
        "finetune_series": int(values.shape[0]),
        "finetune_steps_per_series": int(n),
        "train_fraction": train_fraction,
        "best_val_loss": float(best_val),
    })
    os.remove(scratch)

    evaluation = None
    if evaluate:
        from .serving.app import run_backtest
        from .serving.registry import ModelRegistry

        reg = ModelRegistry.from_env(include_bundled=False, device=device)
        reg.add_checkpoint(base_checkpoint, name="base", device=device)
        reg.add_checkpoint(output, name=name, device=device)
        context = context_patches * cfg.patch_len
        horizon = horizon_patches * cfg.patch_len
        tail = [row[int(n * train_fraction):].astype(np.float64) for row in values]
        try:
            evaluation = run_backtest(
                reg, tail, context, horizon, ["base", name, "ewma", "ar4", "last-value"],
                "last-value", eval_windows,
            )
        except ValueError as e:  # tail too short for a full evaluation window
            evaluation = {"error": str(e)}
    return FinetuneReport(
        output=output, base_checkpoint=base_checkpoint, steps=steps,
        best_val_loss=float(best_val), minutes=minutes, evaluation=evaluation,
    )
