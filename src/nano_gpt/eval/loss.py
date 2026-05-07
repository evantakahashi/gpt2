"""Validation loss / perplexity over a held-out shard."""

from __future__ import annotations

import torch.nn as nn

from nano_gpt.data.loader import DistributedDataLoader


def estimate_val_loss(model: nn.Module, val_loader: DistributedDataLoader, n_batches: int = 20) -> float:
    """model.eval(); average CE loss over n_batches; reduce across ranks."""
    raise NotImplementedError
