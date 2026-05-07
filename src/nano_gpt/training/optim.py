"""Optimizer factory. AdamW with proper param groups; Muon slot."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from nano_gpt.config import TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    """
    AdamW with two groups:
      - decay:    params with ndim >= 2 (matrices, embeddings)
      - no-decay: params with ndim < 2  (biases, norm gains)

    Use fused=True on CUDA. If cfg.optimizer == 'muon', route matrix params
    through Muon and keep AdamW for the rest.
    """
    raise NotImplementedError
