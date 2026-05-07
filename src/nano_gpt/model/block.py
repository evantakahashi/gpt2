"""Pre-LN transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # ln_1, attn, ln_2, mlp -- built via make_norm / CausalSelfAttention / make_mlp
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
