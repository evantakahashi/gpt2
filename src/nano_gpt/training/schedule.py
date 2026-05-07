"""LR schedules. Cosine with linear warmup; WSD slot."""

from __future__ import annotations

import math


def cosine_with_warmup(step: int, *, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup [0, lr] for `warmup` steps, then cosine to `min_lr` at `max_steps`."""
    raise NotImplementedError


def wsd(step: int, *, warmup: int, max_steps: int, decay_frac: float, lr: float, min_lr: float) -> float:
    """Warmup -> Stable -> Decay schedule (Hagele et al)."""
    raise NotImplementedError
