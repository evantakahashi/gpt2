"""HellaSwag zero-shot eval: pick continuation with lowest mean NLL over candidate tokens."""

from __future__ import annotations

import torch.nn as nn


def evaluate_hellaswag(model: nn.Module, split: str = "val", limit: int | None = None) -> dict[str, float]:
    """
    For each example: 4 candidate continuations; score = sum log p(token | context) / n_completion_tokens
    Returns {'acc': ..., 'acc_norm': ...}. Distribute examples across ranks.
    """
    raise NotImplementedError
