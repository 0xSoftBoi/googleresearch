"""Export a TimesFM-3 checkpoint to ONNX for in-browser inference.

The graph takes ``values`` (1, N, T) float32, ``observed`` (1, N, T) bool,
``roles`` (1, N) int64 and ``horizon_patches`` (1,) int64, and returns the
denormalized ``point`` (1, N, T) and ``quantiles`` (1, N, T, Q) for every
position -- the same contract as :meth:`TimesFM3Model.forward`, with the
horizon length as a tensor so one graph serves any horizon.  N and T are
dynamic.  ``timesfm3-onnx.js`` reproduces the forecaster's padding, rolling
decode and quantile repair around it.

    python scripts/export_onnx.py timesfm3/assets/starter-small.pt cloudflare/public/models/starter-small.onnx
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from timesfm3.attention import RotaryEmbedding
from timesfm3.checkpoint import load_checkpoint
from timesfm3.embedding import ROLE_FUTURE_COVARIATE, patchify
from timesfm3.model import TimesFM3Model
from timesfm3.normalization import PerSeriesNormalizer


class ExportRMSNorm(nn.Module):
    """``nn.RMSNorm`` spelled out in ops the ONNX exporter understands."""

    def __init__(self, ref: nn.RMSNorm):
        super().__init__()
        self.weight = nn.Parameter(ref.weight.detach().clone())
        self.eps = ref.eps if ref.eps is not None else torch.finfo(torch.float32).eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ExportRotary(nn.Module):
    """RoPE tables via broadcasting (the traced ``torch.outer`` mis-shapes)."""

    def __init__(self, ref):
        super().__init__()
        self.register_buffer("inv_freq", ref.inv_freq.detach().clone(), persistent=False)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = positions.reshape(-1, 1) * self.inv_freq.reshape(1, -1)
        emb = torch.cat([freqs, freqs], dim=1)
        return emb.cos(), emb.sin()


def replace_rmsnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ExportRMSNorm(child))
        elif isinstance(child, RotaryEmbedding):
            setattr(module, name, ExportRotary(child))
        else:
            replace_rmsnorm(child)


class ExportWrapper(nn.Module):
    """``TimesFM3Model.forward`` with the horizon length as a tensor input."""

    def __init__(self, model: TimesFM3Model):
        super().__init__()
        self.model = model
        self.cfg = model.config

    def forward(self, values, observed, roles, horizon_patches):
        cfg, m = self.cfg, self.model
        b, n, t = values.shape
        num_patches = t // cfg.patch_len
        values = torch.where(observed, values, torch.zeros_like(values))
        patch_idx = torch.arange(num_patches, device=values.device)
        in_horizon = patch_idx >= (num_patches - horizon_patches.reshape(()))
        hidden_role = roles != ROLE_FUTURE_COVARIATE
        masked = (hidden_role[:, :, None] & in_horizon[None, None, :]).float()
        visible_steps = observed & ~masked.bool().repeat_interleave(cfg.patch_len, dim=-1)
        normalizer = PerSeriesNormalizer(values, visible_steps)
        norm_values = normalizer.normalize(values)
        visible = visible_steps.to(norm_values.dtype)
        norm_values = norm_values * visible
        patches = patchify(norm_values, cfg.patch_len)
        patch_observed = patchify(visible, cfg.patch_len)
        tokens = m.embedding(patches, patch_observed, masked, roles)
        for layer in m.layers:
            tokens = layer(tokens)
        tokens = m.final_norm(tokens)
        out = m.head(tokens)
        out = out.reshape(b, n, num_patches, cfg.patch_len, cfg.output_dim_per_step)
        out = out.reshape(b, n, t, cfg.output_dim_per_step)
        out = normalizer.denormalize(out)
        return out[..., 0], out[..., 1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint)
    cfg = state["config"]
    model = TimesFM3Model(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    replace_rmsnorm(model)
    wrapper = ExportWrapper(model).eval()

    n, t = 3, 8 * cfg.patch_len
    values = torch.randn(1, n, t)
    observed = torch.ones(1, n, t, dtype=torch.bool)
    observed[0, 0, :5] = False
    roles = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    hp = torch.tensor([2], dtype=torch.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    series = torch.export.Dim("series", min=1, max=256)
    time = torch.export.Dim("time", min=64, max=4096)
    onnx_program = torch.onnx.export(
        wrapper, (values, observed, roles, hp),
        input_names=["values", "observed", "roles", "horizon_patches"],
        output_names=["point", "quantiles"],
        dynamic_shapes={
            "values": {1: series, 2: time}, "observed": {1: series, 2: time},
            "roles": {1: series}, "horizon_patches": None,
        },
        opset_version=args.opset, dynamo=True, optimize=True,
    )
    onnx_program.save(args.output)

    # Parity check against PyTorch on a different shape than the trace.
    import onnxruntime as ort

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    n2, t2 = 5, 12 * cfg.patch_len
    v2 = (torch.randn(1, n2, t2) * 3 + 10)
    o2 = torch.ones(1, n2, t2, dtype=torch.bool)
    o2[0, 1, 40:50] = False
    o2[0, 4, :] = True
    r2 = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.int64)
    for h in (1, 3):
        with torch.no_grad():
            p_ref, q_ref = wrapper(v2, o2, r2, torch.tensor([h]))
        p_onnx, q_onnx = sess.run(None, {
            "values": v2.numpy(), "observed": o2.numpy(), "roles": r2.numpy(),
            "horizon_patches": np.array([h], dtype=np.int64),
        })
        err_p = float(np.abs(p_onnx - p_ref.numpy()).max())
        err_q = float(np.abs(q_onnx - q_ref.numpy()).max())
        print(f"horizon_patches={h}: max |onnx - torch| point {err_p:.2e} quantiles {err_q:.2e}")
        assert err_p < 1e-3 and err_q < 1e-3, "ONNX parity check failed"

    meta = {
        **{k: v for k, v in state["meta"].items()},
        "patch_len": cfg.patch_len,
        "max_context_len": cfg.max_context_len,
        "max_horizon_len": cfg.max_horizon_len,
        "quantiles": list(cfg.quantiles),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "onnx_opset": args.opset,
        "onnx_bytes": os.path.getsize(args.output),
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.output} ({meta['onnx_bytes'] / 1e6:.1f} MB) and its .json card")


if __name__ == "__main__":
    main()
