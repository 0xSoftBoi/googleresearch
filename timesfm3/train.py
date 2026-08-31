"""Pre-training loop for TimesFM-3.

Trains under the same contiguous-patch-masking regime used at inference:
each batch masks the trailing horizon patches of targets and past-only
covariates, keeps known-future covariates visible, and optimizes a combined
point (MSE) + quantile (pinball) objective on the masked target steps.

Usage:
    python -m timesfm3.train --config small --steps 1000
"""

from __future__ import annotations

import argparse
import math
import random

import torch

from .configuration import TimesFM3Config
from .data.synthetic import SyntheticMultivariateCorpus, collate
from .loss import forecast_loss
from .model import TimesFM3Model


def cosine_lr(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(
    config: TimesFM3Config,
    steps: int = 10_000,
    batch_size: int = 32,
    peak_lr: float = 3e-4,
    warmup_steps: int = 500,
    horizon_patches: int = 4,
    context_patches: int = 16,
    device: str | None = None,
    log_every: int = 50,
    checkpoint_path: str = "timesfm3_checkpoint.pt",
) -> TimesFM3Model:
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = TimesFM3Model(config).to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"TimesFM-3 model with {num_params / 1e6:.1f}M parameters on {device}.")

    corpus = SyntheticMultivariateCorpus(
        config,
        context_patches=context_patches,
        horizon_patches=horizon_patches,
    )
    loader = torch.utils.data.DataLoader(
        corpus, batch_size=batch_size, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    step = 0
    running = 0.0
    for batch in loader:
        step += 1
        lr = cosine_lr(step, warmup_steps, steps, peak_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        values = batch["values"].to(device)
        observed = batch["observed"].to(device)
        roles = batch["roles"].to(device)
        variate_mask = batch["variate_mask"].to(device)

        # Vary the masked-horizon length so the model learns every offset it
        # will see at inference time (excess patches simply become context).
        step_horizon = random.randint(1, horizon_patches)
        output = model(
            values=values,
            observed=observed,
            roles=roles,
            num_horizon_patches=step_horizon,
            variate_mask=variate_mask,
        )
        loss = forecast_loss(config, output, values, roles, variate_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += loss.item()
        if step % log_every == 0:
            print(f"step {step:>7d}  lr {lr:.2e}  loss {running / log_every:.4f}")
            running = 0.0
        if step >= steps:
            break

    torch.save({"config": config, "model": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}.")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train TimesFM-3.")
    parser.add_argument("--config", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context-patches", type=int, default=16)
    parser.add_argument("--horizon-patches", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="timesfm3_checkpoint.pt")
    args = parser.parse_args()

    config = getattr(TimesFM3Config, args.config)()
    train(
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        peak_lr=args.lr,
        context_patches=args.context_patches,
        horizon_patches=args.horizon_patches,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
