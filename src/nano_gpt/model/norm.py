"""Normalization layers. Default LayerNorm; RMSNorm slot for SOTA configs."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Paper-faithful LayerNorm with optional bias.

    Normalizes over the last dim:
        y = (x - mean(x, dim=-1)) / sqrt(var(x, dim=-1) + eps) * weight + bias
    """

    def __init__(self, n_embd: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) * torch.rsqrt(var + self.eps)
        y = self.weight * normalized_x
        if self.bias is not None:
            y = y + self.bias
        return y



class RMSNorm(nn.Module):
    """Llama-style RMSNorm. No mean centering, no bias. (Used in Step 6.)"""

    def __init__(self, n_embd: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("RMSNorm implemented in Step 6")


def make_norm(kind: str, n_embd: int, bias: bool = True) -> nn.Module:
    """Factory used by Block. Selects LayerNorm or RMSNorm based on ModelConfig.norm."""
    if kind == "ln":
        return LayerNorm(n_embd, bias=bias)
    if kind == "rmsnorm":
        return RMSNorm(n_embd)
    raise ValueError(f"unknown norm kind: {kind!r}")
