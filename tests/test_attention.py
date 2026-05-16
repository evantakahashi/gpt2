import pytest
import torch
import torch.nn.functional as F

from nano_gpt.config import ModelConfig
from nano_gpt.model.attention import CausalSelfAttention


def make_cfg(n_embd=64, n_head=4, block_size=32, bias=True):
    return ModelConfig(
        vocab_size=50257,
        block_size=block_size,
        n_layer=12,
        n_head=n_head,
        n_embd=n_embd,
        bias=bias,
    )


# --- __init__ boilerplate (Claude's scaffold should already pass these) ---


def test_n_embd_divisible_by_n_head_required():
    with pytest.raises(ValueError):
        CausalSelfAttention(make_cfg(n_embd=64, n_head=7))


def test_causal_mask_is_buffer_not_parameter():
    attn = CausalSelfAttention(make_cfg())
    assert "causal_mask" in dict(attn.named_buffers())
    assert "causal_mask" not in dict(attn.named_parameters())


def test_causal_mask_shape():
    attn = CausalSelfAttention(make_cfg(block_size=32))
    assert attn.causal_mask.shape == (1, 1, 32, 32)


def test_causal_mask_is_lower_triangular():
    attn = CausalSelfAttention(make_cfg(block_size=8))
    m = attn.causal_mask.squeeze()  # (8, 8)
    expected = torch.tril(torch.ones(8, 8))
    assert torch.equal(m, expected)


def test_c_proj_marked_as_residual_projection():
    """Marker attribute consumed by the model-level GPT-2 init pass (1/sqrt(2*n_layer) scaling)."""
    attn = CausalSelfAttention(make_cfg())
    assert getattr(attn.c_proj, "RESIDUAL_PROJ", False) is True


def test_use_flash_raises():
    with pytest.raises(NotImplementedError):
        CausalSelfAttention(ModelConfig(use_flash=True))


def test_gqa_raises():
    with pytest.raises(NotImplementedError):
        CausalSelfAttention(ModelConfig(attn_impl="gqa"))


def test_rope_pos_raises():
    with pytest.raises(NotImplementedError):
        CausalSelfAttention(ModelConfig(pos="rope"))


# --- forward (your TODO; these will fail until you implement it) ---


def test_forward_shape_preserved():
    """Input (B, T, D) must produce output (B, T, D)."""
    attn = CausalSelfAttention(make_cfg(n_embd=64, n_head=4, block_size=16))
    x = torch.randn(2, 8, 64)
    out = attn(x)
    assert out.shape == (2, 8, 64)


def test_forward_dtype_preserved():
    attn = CausalSelfAttention(make_cfg())
    x = torch.randn(2, 8, 64, dtype=torch.float32)
    out = attn(x)
    assert out.dtype == torch.float32


def test_causal_constraint_via_input_perturbation():
    """Token i's output must not depend on tokens at positions > i.

    Concretely: perturb the LAST input position; outputs at all earlier
    positions must be unchanged. If they change, the mask isn't blocking
    future tokens — your causal constraint is broken.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(make_cfg(n_embd=16, n_head=2, block_size=8)).eval()

    x1 = torch.randn(1, 4, 16)
    x2 = x1.clone()
    x2[0, 3] = torch.randn(16) * 10   # large perturbation at last position

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Positions 0, 1, 2 must be identical (they cannot have seen position 3).
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6), (
        "Earlier positions changed when a future position was perturbed — causal mask broken."
    )
    # Position 3 must differ (sanity: confirms the perturbation actually flows through).
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-3)


def test_matches_torch_sdpa_reference():
    """Apples-to-apples: our hand-rolled forward must match F.scaled_dot_product_attention.

    Given the same Q, K, V (via the same c_attn projection), our output (after c_proj)
    must equal the reference's output (after c_proj) to within fp32 precision.
    """
    torch.manual_seed(0)
    cfg = make_cfg(n_embd=64, n_head=4, block_size=16, bias=False)
    attn = CausalSelfAttention(cfg).eval()
    B, T, D = 2, 8, cfg.n_embd

    x = torch.randn(B, T, D)

    with torch.no_grad():
        ours = attn(x)

        # Reproduce Q, K, V from the same c_attn, then run SDPA as the reference.
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(cfg.n_embd, dim=-1)
        head_dim = cfg.n_embd // cfg.n_head
        q = q.view(B, T, cfg.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, cfg.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, cfg.n_head, head_dim).transpose(1, 2)

        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = ref.transpose(1, 2).contiguous().view(B, T, D)
        ref = attn.c_proj(ref)

    assert torch.allclose(ours, ref, atol=1e-5), (
        "Output disagrees with F.scaled_dot_product_attention. Check the math:"
        " QK^T scaling, mask application, softmax dim, weighted sum, head reshape."
    )


def test_gradient_flows_to_all_params():
    """Backward through attention must produce gradients for c_attn and c_proj."""
    attn = CausalSelfAttention(make_cfg(n_embd=16, n_head=2, block_size=8))
    x = torch.randn(1, 4, 16, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.c_attn.weight.grad is not None
    assert (attn.c_attn.weight.grad != 0).any()
    assert attn.c_proj.weight.grad is not None
    assert (attn.c_proj.weight.grad != 0).any()
    # Mask buffer has no gradient (it's not a Parameter).
    assert attn.causal_mask.grad is None
