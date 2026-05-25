"""Optimizer factory. AdamW with proper param groups; Muon slot."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from torch.optim import Optimizer

from nano_gpt.config import TrainConfig


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split model params into two AdamW groups by dimensionality.

    Returns a list of two dicts:
      - decay group:    params with ndim >= 2 (weight matrices, embeddings)
                        -> weight_decay applied
      - no-decay group: params with ndim <  2 (biases, LayerNorm gamma/beta)
                        -> weight_decay = 0.0
    """
    # Weight decay regularizes weight matrices (ndim >= 2). For 1D params
    # (biases, LayerNorm γ/β) decay is meaningless/harmful, so they get 0.
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    """AdamW with decay/no-decay param groups. Uses fused=True on CUDA. Muon -> Step 6."""
    if cfg.optimizer == "muon":
        raise NotImplementedError("Muon optimizer implemented in Step 6")

    optim_groups = make_param_groups(model, cfg.weight_decay)

    # Fused AdamW is a faster CUDA kernel; only available on CUDA + recent torch.
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()

    return torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=use_fused,
    )
