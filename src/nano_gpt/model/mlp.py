"""Feed-forward block. Paper-faithful 4x GELU MLP; SwiGLU slot."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class GELU_MLP(nn.Module):
    """Paper-faithful: c_fc (4x) -> GELU -> c_proj.

    Per-token feedforward: each (b, t) D-dim vector is processed independently.
    NO mixing across tokens (that's attention's job). Shape preserved (B, T, D) -> (B, T, D).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # 4× expansion is the original Transformer convention; carried into GPT-2.
        hidden = 4 * cfg.n_embd

        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        # GPT-2 specifically used the tanh-approximation of GELU. Numerical difference
        # vs the exact erf-based version is tiny (~1e-5) but matters for paper fidelity.
        self.gelu = nn.GELU(approximate="tanh")
        # Output projection back to n_embd. Marked as a residual projection so the
        # model-level GPT-2 init scheme scales its weight by 1/sqrt(2*n_layer).
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, T, n_embd)
        returns: shape (B, T, n_embd)
        """
        # TODO(you): implement the 3-line MLP forward.
        #
        # Steps:
        #   1. Apply self.c_fc to x          -> shape (B, T, 4 * n_embd)
        #   2. Apply self.gelu (elementwise) -> shape (B, T, 4 * n_embd)
        #   3. Apply self.c_proj             -> shape (B, T, n_embd)
        #   4. Return the result.
        #
        # No reshapes, no masks, no head splits. Each token's D-dim vector is
        # processed independently — the Linears only operate on the last dim.
        # raise NotImplementedError("implement GELU_MLP.forward")
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class SwiGLU_MLP(nn.Module):
    """Llama-style: SiLU(w1(x)) * w3(x) -> w2. Hidden ~= 8/3 * n_embd to match params. (Step 6.)"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("SwiGLU implemented in Step 6")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SwiGLU implemented in Step 6")


def make_mlp(cfg: ModelConfig) -> nn.Module:
    """Factory used by Block. Selects GELU vs SwiGLU based on ModelConfig.mlp."""
    if cfg.mlp == "gelu":
        return GELU_MLP(cfg)
    if cfg.mlp == "swiglu":
        return SwiGLU_MLP(cfg)
    raise ValueError(f"unknown mlp kind: {cfg.mlp!r}")
