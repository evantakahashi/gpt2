import math

import pytest
import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig
from nano_gpt.model.block import Block
from nano_gpt.model.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from nano_gpt.model.gpt import GPT
from nano_gpt.model.norm import LayerNorm


def make_tiny_cfg(vocab_size=100, n_embd=32, n_head=4, n_layer=2, block_size=8):
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )


# --- __init__ wiring (scaffolded; should pass) ---


def test_submodules_exist():
    model = GPT(make_tiny_cfg())
    assert isinstance(model.token_emb, TokenEmbedding)
    assert isinstance(model.pos_emb, LearnedPositionalEmbedding)
    assert isinstance(model.blocks, nn.ModuleList)
    assert isinstance(model.ln_f, LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


def test_block_count_matches_n_layer():
    cfg = make_tiny_cfg(n_layer=4)
    model = GPT(cfg)
    assert len(model.blocks) == 4
    for blk in model.blocks:
        assert isinstance(blk, Block)


def test_lm_head_no_bias():
    model = GPT(make_tiny_cfg())
    assert model.lm_head.bias is None


def test_weight_tying():
    """lm_head.weight and token_emb.weight must be the SAME Parameter object."""
    model = GPT(make_tiny_cfg())
    assert model.lm_head.weight is model.token_emb.weight, (
        "Weight tying broken: lm_head.weight should be the same Parameter as token_emb.weight."
    )


def test_init_residual_projections_have_smaller_std():
    """Residual projections (c_proj in attn and mlp) should have init std = 0.02 / sqrt(2*n_layer).

    Other Linears should have init std ~ 0.02.
    """
    cfg = make_tiny_cfg(n_layer=12, n_embd=256)  # n_layer=12 gives the GPT-2 124M scaling
    model = GPT(cfg)
    target_std_normal = 0.02
    target_std_resproj = 0.02 / math.sqrt(2 * cfg.n_layer)

    # Pick a residual projection from the first block and check its std is roughly correct.
    res_proj = model.blocks[0].attn.c_proj
    assert getattr(res_proj, "RESIDUAL_PROJ", False) is True
    measured_std = res_proj.weight.std().item()
    # Allow generous tolerance — std of a Gaussian sample is noisy.
    assert abs(measured_std - target_std_resproj) < target_std_resproj * 0.5

    # Pick a non-residual Linear (c_attn or c_fc) and check std is closer to 0.02.
    non_res = model.blocks[0].attn.c_attn
    assert getattr(non_res, "RESIDUAL_PROJ", False) is False
    measured_std_normal = non_res.weight.std().item()
    assert abs(measured_std_normal - target_std_normal) < target_std_normal * 0.3


def test_num_params_for_default_124m_config():
    """At default ModelConfig (vocab=50257, D=768, n_layer=12), total params ≈ 124M.

    Note: this allocates the full 124M model; slow but only runs once.
    """
    model = GPT(ModelConfig())
    total = model.num_params(non_embedding=False)
    # Should be in the ballpark of 124M ± a few percent.
    assert 120_000_000 < total < 130_000_000, f"got {total:,} params"


def test_num_params_tiny():
    """Sanity check on the param-counting function with a tiny model."""
    cfg = make_tiny_cfg(vocab_size=100, n_embd=32, n_layer=2)
    model = GPT(cfg)
    # All counted params should be positive.
    assert model.num_params(non_embedding=False) > 0
    assert model.num_params(non_embedding=True) > 0
    # non_embedding subtracts pos_emb only (token_emb is tied, included).
    diff = model.num_params(non_embedding=False) - model.num_params(non_embedding=True)
    assert diff == model.pos_emb.weight.numel()


# --- forward (your TODO; fails until you implement it) ---


def test_forward_shape_no_targets():
    cfg = make_tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, loss = model(idx)
    assert logits.shape == (2, 4, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets():
    cfg = make_tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    targets = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 4, cfg.vocab_size)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0  # nonzero loss


def test_initial_loss_near_log_vocab():
    """At init, the model has no information. Predictions are ~uniform.
    Expected initial loss ≈ ln(vocab_size).
    """
    torch.manual_seed(0)
    cfg = make_tiny_cfg(vocab_size=100)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 8))
    targets = torch.randint(0, cfg.vocab_size, (4, 8))
    _, loss = model(idx, targets=targets)
    expected = math.log(cfg.vocab_size)
    # Allow generous tolerance — init noise can shift loss a bit.
    assert abs(loss.item() - expected) < 0.5, f"got {loss.item():.3f}, expected ~{expected:.3f}"


def test_forward_raises_on_too_long_sequence():
    """T > block_size should be rejected (positional embedding has no row for it)."""
    cfg = make_tiny_cfg(block_size=4)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 8))  # T=8 > block_size=4
    with pytest.raises(Exception):  # assertion or runtime error
        model(idx)


# --- generate (your TODO) ---


def test_generate_appends_correct_number_of_tokens():
    cfg = make_tiny_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (1, 8)  # 3 prompt + 5 generated


def test_generate_dtype_is_int():
    cfg = make_tiny_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    out = model.generate(prompt, max_new_tokens=3)
    assert out.dtype == prompt.dtype


def test_generate_token_ids_in_vocab():
    cfg = make_tiny_cfg(vocab_size=50)
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    out = model.generate(prompt, max_new_tokens=10)
    assert (out >= 0).all()
    assert (out < cfg.vocab_size).all()


def test_generate_with_top_k():
    """Exercises the top_k path (the other generate tests use top_k=None)."""
    cfg = make_tiny_cfg(vocab_size=50)
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(prompt, max_new_tokens=5, top_k=10)
    assert out.shape == (1, 8)
    assert (out >= 0).all()
    assert (out < cfg.vocab_size).all()


def test_generate_top_k_larger_than_vocab():
    """top_k > vocab_size must not crash (min(top_k, V) guards this)."""
    cfg = make_tiny_cfg(vocab_size=20)
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    out = model.generate(prompt, max_new_tokens=3, top_k=1000)  # 1000 >> vocab 20
    assert out.shape == (1, 5)


def test_generate_with_temperature():
    """Temperature scaling path should run without error."""
    cfg = make_tiny_cfg(vocab_size=50)
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    out = model.generate(prompt, max_new_tokens=4, temperature=0.7, top_k=5)
    assert out.shape == (1, 6)


# --- THE BIG ONE: overfit a single batch (Step 2 smoke test) ---


def test_smoke_overfit_single_batch():
    """Train on ONE batch for ~100 steps. Loss should drop dramatically.

    This is the gold-standard sanity check that the whole architecture is wired
    correctly: residuals, embeddings, attention, MLP, init, loss — all of it.
    A correctly-built model can memorize a single batch quickly. If loss plateaus
    above the initial value, something fundamental is broken.
    """
    torch.manual_seed(0)
    cfg = make_tiny_cfg(vocab_size=32, n_embd=64, n_head=4, n_layer=2, block_size=8)
    model = GPT(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _, loss0 = model(idx, targets=targets)
    initial_loss = loss0.item()

    for _ in range(100):
        _, loss = model(idx, targets=targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Should drop dramatically — typical end-state is < 0.5 for this tiny config.
    assert final_loss < initial_loss * 0.2, (
        f"Smoke test failed: loss only went from {initial_loss:.3f} -> {final_loss:.3f}. "
        "Expected at least 80% reduction. Something in the architecture is broken."
    )
