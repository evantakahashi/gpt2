import pytest
import torch

from nano_gpt.config import ModelConfig, TrainConfig
from nano_gpt.model.gpt import GPT
from nano_gpt.training.optim import build_optimizer, make_param_groups


def tiny_model():
    return GPT(ModelConfig(vocab_size=100, n_embd=32, n_head=4, n_layer=2, block_size=8))


# --- make_param_groups (your TODO) ---


def test_returns_two_groups():
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2


def test_decay_group_only_2d_params():
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.1)
    decay = [g for g in groups if g["weight_decay"] > 0][0]
    assert all(p.dim() >= 2 for p in decay["params"])


def test_nodecay_group_only_1d_params():
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.1)
    nodecay = [g for g in groups if g["weight_decay"] == 0][0]
    assert all(p.dim() < 2 for p in nodecay["params"])


def test_decay_group_uses_configured_weight_decay():
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.137)
    decay = [g for g in groups if g["weight_decay"] > 0][0]
    assert decay["weight_decay"] == 0.137


def test_nodecay_group_weight_decay_zero():
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.1)
    nodecay = [g for g in groups if g["weight_decay"] == 0][0]
    assert nodecay["weight_decay"] == 0.0


def test_all_params_accounted_for_exactly_once():
    """Every trainable param must appear in exactly one group (no drops, no dupes).

    Note: token_emb.weight is tied to lm_head.weight (same Parameter), so it's
    counted once by id.
    """
    model = tiny_model()
    groups = make_param_groups(model, weight_decay=0.1)

    grouped_ids = []
    for g in groups:
        for p in g["params"]:
            grouped_ids.append(id(p))

    model_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # No param in both groups.
    assert len(grouped_ids) == len(set(grouped_ids)), "a param appears in multiple groups"
    # Exactly the set of trainable params is covered.
    assert set(grouped_ids) == model_ids


# --- build_optimizer (scaffolded) ---


def test_build_optimizer_returns_adamw():
    model = tiny_model()
    opt = build_optimizer(model, TrainConfig())
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_sets_lr_and_betas():
    model = tiny_model()
    cfg = TrainConfig(lr=3e-4, beta1=0.9, beta2=0.95)
    opt = build_optimizer(model, cfg)
    # Both groups should carry the configured lr and betas.
    for g in opt.param_groups:
        assert g["lr"] == 3e-4
        assert g["betas"] == (0.9, 0.95)


def test_muon_raises():
    model = tiny_model()
    with pytest.raises(NotImplementedError):
        build_optimizer(model, TrainConfig(optimizer="muon"))


def test_optimizer_can_step():
    """End-to-end: optimizer actually updates params without error."""
    model = tiny_model()
    opt = build_optimizer(model, TrainConfig(lr=1e-3))
    idx = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    _, loss = model(idx, targets=targets)
    opt.zero_grad()
    loss.backward()
    opt.step()  # should not raise
