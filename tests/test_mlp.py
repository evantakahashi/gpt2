import pytest
import torch

from nano_gpt.config import ModelConfig
from nano_gpt.model.mlp import GELU_MLP, make_mlp


def make_cfg(n_embd=64, bias=True, mlp="gelu"):
    return ModelConfig(n_embd=n_embd, bias=bias, mlp=mlp)


# --- __init__ boilerplate (scaffolded, should pass) ---


def test_c_fc_shape_is_4x_expansion():
    mlp = GELU_MLP(make_cfg(n_embd=64))
    assert mlp.c_fc.weight.shape == (256, 64)  # nn.Linear stores (out, in)


def test_c_proj_shape_compresses_back():
    mlp = GELU_MLP(make_cfg(n_embd=64))
    assert mlp.c_proj.weight.shape == (64, 256)


def test_c_proj_marked_as_residual_projection():
    mlp = GELU_MLP(make_cfg())
    assert getattr(mlp.c_proj, "RESIDUAL_PROJ", False) is True


def test_bias_false_drops_biases():
    mlp = GELU_MLP(make_cfg(bias=False))
    assert mlp.c_fc.bias is None
    assert mlp.c_proj.bias is None


def test_make_mlp_factory_gelu():
    m = make_mlp(make_cfg(mlp="gelu"))
    assert isinstance(m, GELU_MLP)


def test_make_mlp_factory_unknown_raises():
    with pytest.raises(ValueError):
        make_mlp(ModelConfig(mlp="totally_made_up"))  # type: ignore[arg-type]


def test_swiglu_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        make_mlp(make_cfg(mlp="swiglu"))


# --- forward (your TODO; these fail until you implement it) ---


def test_forward_shape_preserved():
    mlp = GELU_MLP(make_cfg(n_embd=64))
    x = torch.randn(2, 8, 64)
    out = mlp(x)
    assert out.shape == (2, 8, 64)


def test_forward_dtype_preserved():
    mlp = GELU_MLP(make_cfg(n_embd=32))
    x = torch.randn(1, 4, 32, dtype=torch.float32)
    out = mlp(x)
    assert out.dtype == torch.float32


def test_no_cross_token_mixing():
    """MLP processes each token independently. Perturbing token 3 must NOT change tokens 0, 1, 2."""
    torch.manual_seed(0)
    mlp = GELU_MLP(make_cfg(n_embd=32)).eval()

    x1 = torch.randn(1, 4, 32)
    x2 = x1.clone()
    x2[0, 3] = torch.randn(32) * 5   # perturb only position 3

    with torch.no_grad():
        out1 = mlp(x1)
        out2 = mlp(x2)

    # Positions 0, 1, 2 must be IDENTICAL (MLP doesn't see across positions).
    assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6), (
        "MLP output at earlier positions changed when a later position was perturbed — "
        "this means MLP is mixing across tokens, which it shouldn't."
    )
    # Position 3 should differ (sanity: the perturbation actually propagates).
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-3)


def test_no_cross_batch_mixing():
    """MLP processes each (b, t) independently. Perturbing batch row 1 must NOT change batch row 0."""
    torch.manual_seed(0)
    mlp = GELU_MLP(make_cfg(n_embd=16)).eval()

    x1 = torch.randn(2, 4, 16)
    x2 = x1.clone()
    x2[1] = torch.randn(4, 16) * 5

    with torch.no_grad():
        out1 = mlp(x1)
        out2 = mlp(x2)

    # Batch row 0 should be identical.
    assert torch.allclose(out1[0], out2[0], atol=1e-6)


def test_gradient_flows_to_both_linears():
    """Backward through MLP must produce gradients for both c_fc and c_proj."""
    mlp = GELU_MLP(make_cfg(n_embd=16))
    x = torch.randn(1, 4, 16, requires_grad=True)
    out = mlp(x)
    out.sum().backward()

    assert mlp.c_fc.weight.grad is not None
    assert (mlp.c_fc.weight.grad != 0).any()
    assert mlp.c_proj.weight.grad is not None
    assert (mlp.c_proj.weight.grad != 0).any()
