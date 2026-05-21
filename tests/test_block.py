import pytest
import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig
from nano_gpt.model.attention import CausalSelfAttention
from nano_gpt.model.block import Block
from nano_gpt.model.mlp import GELU_MLP
from nano_gpt.model.norm import LayerNorm


def make_cfg(n_embd=64, n_head=4, block_size=16, bias=True):
    return ModelConfig(n_embd=n_embd, n_head=n_head, block_size=block_size, bias=bias)


# --- __init__ wiring (scaffolded; should pass) ---


def test_submodules_exist():
    """Block must have ln_1, attn, ln_2, mlp as direct attributes."""
    blk = Block(make_cfg())
    assert isinstance(blk.ln_1, LayerNorm)
    assert isinstance(blk.attn, CausalSelfAttention)
    assert isinstance(blk.ln_2, LayerNorm)
    assert isinstance(blk.mlp, GELU_MLP)


def test_submodule_order_pre_ln():
    """Pre-LN convention: LN comes before attn and before mlp (two LNs per block)."""
    blk = Block(make_cfg())
    # We expect exactly TWO LayerNorms — no LN after the residual add.
    n_lns = sum(1 for m in blk.modules() if isinstance(m, LayerNorm))
    assert n_lns == 2


# --- forward (your TODO; fails until you implement it) ---


def test_forward_shape_preserved():
    blk = Block(make_cfg(n_embd=64, n_head=4, block_size=16))
    x = torch.randn(2, 8, 64)
    out = blk(x)
    assert out.shape == (2, 8, 64)


def test_forward_dtype_preserved():
    blk = Block(make_cfg(n_embd=32, n_head=4, block_size=8))
    x = torch.randn(1, 4, 32, dtype=torch.float32)
    assert blk(x).dtype == torch.float32


def test_residual_stream_is_additive():
    """With both sub-layers zeroed out, block should return its input unchanged.

    This verifies the residual `x + ...` is wired correctly. If the block accidentally
    REPLACES x with sublayer output, this test will fail.
    """
    blk = Block(make_cfg(n_embd=32, n_head=4, block_size=8)).eval()
    # Zero out attention's output projection -> attn(...) = 0.
    with torch.no_grad():
        blk.attn.c_proj.weight.zero_()
        if blk.attn.c_proj.bias is not None:
            blk.attn.c_proj.bias.zero_()
        # Zero out MLP's output projection -> mlp(...) = 0.
        blk.mlp.c_proj.weight.zero_()
        if blk.mlp.c_proj.bias is not None:
            blk.mlp.c_proj.bias.zero_()

    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        out = blk(x)
    # With both sub-layers contributing 0, output = x + 0 + 0 = x.
    assert torch.allclose(out, x, atol=1e-6), (
        "When both sublayers output 0, block should be the identity (residual passthrough). "
        "Got a non-trivial difference, meaning the residual `x +` is missing or wrong."
    )


def test_causal_property_propagates():
    """Block must preserve the causal property of attention.

    Perturbing the last token's input shouldn't change earlier output positions.
    """
    torch.manual_seed(0)
    blk = Block(make_cfg(n_embd=16, n_head=2, block_size=8)).eval()
    x1 = torch.randn(1, 4, 16)
    x2 = x1.clone()
    x2[0, 3] = torch.randn(16) * 5

    with torch.no_grad():
        out1 = blk(x1)
        out2 = blk(x2)

    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5), (
        "Earlier output positions changed when a future input position was perturbed — "
        "the block is leaking future information."
    )
    # Sanity: position 3 should differ.
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-3)


def test_gradient_flows_to_all_submodules():
    """Backward through Block must produce gradients in every submodule with params."""
    blk = Block(make_cfg(n_embd=16, n_head=2, block_size=8))
    x = torch.randn(1, 4, 16, requires_grad=True)
    out = blk(x)
    out.sum().backward()

    # All four submodules should have nonzero gradients in their main weights.
    assert (blk.ln_1.weight.grad != 0).any()
    assert (blk.attn.c_attn.weight.grad != 0).any()
    assert (blk.attn.c_proj.weight.grad != 0).any()
    assert (blk.ln_2.weight.grad != 0).any()
    assert (blk.mlp.c_fc.weight.grad != 0).any()
    assert (blk.mlp.c_proj.weight.grad != 0).any()


def test_rope_passes_through_to_attention():
    """If rope is provided, the block should pass it to attn — which will raise (RoPE not impl)."""
    blk = Block(make_cfg(n_embd=16, n_head=2, block_size=8))
    x = torch.randn(1, 4, 16)
    cos = torch.ones(4, 8)
    sin = torch.zeros(4, 8)
    with pytest.raises(NotImplementedError):
        blk(x, rope=(cos, sin))
