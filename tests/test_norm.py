import pytest
import torch
import torch.nn as nn

from nano_gpt.model.norm import LayerNorm, make_norm


def test_layernorm_matches_torch_reference():
    """Our LayerNorm must produce the same output as torch.nn.LayerNorm."""
    torch.manual_seed(0)
    n_embd = 64
    ours = LayerNorm(n_embd, bias=True)
    ref = nn.LayerNorm(n_embd, bias=True, eps=ours.eps)

    # share the same (weight, bias) so output differences are due to math only.
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)

    x = torch.randn(2, 8, n_embd)
    y_ours = ours(x)
    y_ref = ref(x)
    assert torch.allclose(y_ours, y_ref, atol=1e-6)


def test_shape_and_dtype_preserved():
    ln = LayerNorm(32)
    x = torch.randn(4, 16, 32, dtype=torch.float32)
    y = ln(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_normalizes_last_dim_only():
    """After LayerNorm, each (b, t) vector has mean~0 and var~1 (before scale/bias)."""
    n_embd = 128
    ln = LayerNorm(n_embd, bias=True)
    # Zero out learnable params so we read the raw normalized values.
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()
    x = torch.randn(3, 5, n_embd) * 10 + 7  # weird mean and var
    y = ln(x)
    means = y.mean(dim=-1)
    vars_ = y.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4)


def test_bias_false_matches_torch_reference():
    n_embd = 48
    ours = LayerNorm(n_embd, bias=False)
    assert ours.bias is None
    ref = nn.LayerNorm(n_embd, bias=False, eps=ours.eps)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
    x = torch.randn(2, 7, n_embd)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_eps_prevents_div_by_zero():
    """All-equal input → variance is 0; eps must keep the math finite."""
    ln = LayerNorm(16)
    x = torch.full((1, 1, 16), 3.14)
    y = ln(x)
    assert torch.isfinite(y).all()


def test_make_norm_factory():
    m = make_norm("ln", 64, bias=True)
    assert isinstance(m, LayerNorm)
    with pytest.raises(ValueError):
        make_norm("bogus", 64)
