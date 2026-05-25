"""LR schedules. Cosine with linear warmup; WSD slot."""

from __future__ import annotations

import math


def cosine_with_warmup(step: int, *, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup [0, lr] over `warmup` steps, then cosine decay to `min_lr` at `max_steps`.

    Returns the learning rate to use AT this step.
    """
    # TODO(you): implement the three regimes of the schedule.
    #
    #   1. WARMUP (step < warmup):
    #        linear ramp from ~0 up to `lr`.
    #        return lr * (step + 1) / warmup
    #      (step+1 so step 0 isn't exactly 0; the very first step gets a tiny nonzero lr)
    if step < warmup:
        return lr * (step + 1) / warmup
    #
    #   2. POST-TRAINING (step > max_steps):
    #        clamp at the floor.
    #        return min_lr
    elif step > max_steps:
        return min_lr
    #
    #   3. COSINE DECAY (warmup <= step <= max_steps):
    #        decay_ratio goes 0 -> 1 across the decay phase:
    #          decay_ratio = (step - warmup) / (max_steps - warmup)
    #        cosine coefficient goes 1 -> 0:
    #          coeff = 0.5 * (1 + cos(pi * decay_ratio))
    #        interpolate between lr (coeff=1) and min_lr (coeff=0):
    #          return min_lr + coeff * (lr - min_lr)
    elif warmup <= step <= max_steps:
        decay_ratio = (step - warmup) / (max_steps - warmup)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)
    #
    # math.cos and math.pi are imported.
    # raise NotImplementedError("implement cosine_with_warmup")


def wsd(step: int, *, warmup: int, max_steps: int, decay_frac: float, lr: float, min_lr: float) -> float:
    """Warmup -> Stable -> Decay schedule (Hagele et al). Deferred."""
    raise NotImplementedError("WSD schedule deferred (alternative to cosine; not needed for baseline)")
