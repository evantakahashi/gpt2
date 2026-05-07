"""Normalization layers. Default LayerNorm; RMSNorm slot for SOTA configs."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Paper-faithful LayerNorm with optional bias."""

    def __init__(self, n_embd: int, bias: bool = True) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RMSNorm(nn.Module):
    """Llama-style RMSNorm. No mean centering, no bias."""

    def __init__(self, n_embd: int, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def make_norm(kind: str, n_embd: int, bias: bool = True) -> nn.Module:
    """Factory used by Block."""
    raise NotImplementedError
