"""Feed-forward block. Paper-faithful 4x GELU MLP; SwiGLU slot."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class GELU_MLP(nn.Module):
    """Paper-faithful: c_fc (4x) -> GELU -> c_proj."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SwiGLU_MLP(nn.Module):
    """Llama-style: w1, w3 -> SiLU(w1(x)) * w3(x) -> w2. Hidden ~= 8/3 * n_embd to match params."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def make_mlp(cfg: ModelConfig) -> nn.Module:
    """Factory used by Block."""
    raise NotImplementedError
