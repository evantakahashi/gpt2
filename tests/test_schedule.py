import math

import pytest

from nano_gpt.training.schedule import cosine_with_warmup


# Common schedule params for tests.
KW = dict(warmup=100, max_steps=1000, lr=6e-4, min_lr=6e-5)


def test_warmup_starts_near_zero():
    """At step 0, lr should be small (the first warmup step), not the full lr."""
    lr0 = cosine_with_warmup(0, **KW)
    assert 0 < lr0 < KW["lr"]
    # step 0 should be lr * 1/warmup
    assert math.isclose(lr0, KW["lr"] * 1 / KW["warmup"], rel_tol=1e-6)


def test_warmup_is_linear():
    """During warmup, lr should ramp linearly."""
    lr_25 = cosine_with_warmup(24, **KW)   # step 24 -> 25/100 of lr
    lr_50 = cosine_with_warmup(49, **KW)   # step 49 -> 50/100 of lr
    assert math.isclose(lr_25, KW["lr"] * 25 / 100, rel_tol=1e-6)
    assert math.isclose(lr_50, KW["lr"] * 50 / 100, rel_tol=1e-6)


def test_peak_lr_at_end_of_warmup():
    """At the end of warmup (step == warmup), lr should be ~ the peak lr."""
    lr_peak = cosine_with_warmup(KW["warmup"], **KW)
    assert math.isclose(lr_peak, KW["lr"], rel_tol=1e-3)


def test_cosine_decays_to_min_at_max_steps():
    """At max_steps, cosine should have decayed to min_lr."""
    lr_end = cosine_with_warmup(KW["max_steps"], **KW)
    assert math.isclose(lr_end, KW["min_lr"], rel_tol=1e-6)


def test_clamps_at_min_after_max_steps():
    """Past max_steps, lr stays at min_lr (no negative or runaway values)."""
    lr_past = cosine_with_warmup(KW["max_steps"] + 500, **KW)
    assert math.isclose(lr_past, KW["min_lr"], rel_tol=1e-6)


def test_midpoint_of_cosine_is_halfway():
    """At the midpoint of the decay phase, lr should be halfway between lr and min_lr."""
    mid_step = KW["warmup"] + (KW["max_steps"] - KW["warmup"]) // 2
    lr_mid = cosine_with_warmup(mid_step, **KW)
    expected = KW["min_lr"] + 0.5 * (KW["lr"] - KW["min_lr"])
    assert math.isclose(lr_mid, expected, rel_tol=1e-2)


def test_monotonic_decrease_during_cosine():
    """LR should monotonically decrease through the cosine phase."""
    prev = cosine_with_warmup(KW["warmup"], **KW)
    for step in range(KW["warmup"] + 1, KW["max_steps"] + 1, 50):
        cur = cosine_with_warmup(step, **KW)
        assert cur <= prev + 1e-9
        prev = cur


def test_lr_never_exceeds_peak_and_stays_positive():
    """LR is always positive and never exceeds the peak lr, at any step."""
    for step in range(0, KW["max_steps"] + 200, 17):
        lr = cosine_with_warmup(step, **KW)
        assert 0 < lr <= KW["lr"] + 1e-9


def test_min_lr_floor_applies_only_after_warmup():
    """During the cosine decay phase (post-warmup), lr stays >= min_lr.

    Note: during WARMUP, lr legitimately starts below min_lr (it ramps from ~0),
    so the min_lr floor only applies once warmup is done.
    """
    for step in range(KW["warmup"], KW["max_steps"] + 200, 17):
        lr = cosine_with_warmup(step, **KW)
        assert lr >= KW["min_lr"] - 1e-9
