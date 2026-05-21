"""Pre-LN transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig
from nano_gpt.model.attention import CausalSelfAttention
from nano_gpt.model.mlp import make_mlp
from nano_gpt.model.norm import make_norm


class Block(nn.Module):
    """One transformer block in Pre-LN style.

    Each sub-layer reads a normalized view of the residual stream and writes back
    by addition. The stream itself (x) is never normalized in flight — each block's
    contributions accumulate on the raw stream.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = make_norm(cfg.norm, cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg.norm, cfg.n_embd, bias=cfg.bias)
        self.mlp = make_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        x:    shape (B, T, n_embd)
        rope: optional (cos, sin) tables, passed through to attention.
        returns: shape (B, T, n_embd)
        """
        # TODO(you): implement the two-line Pre-LN block forward.
        #
        # Pattern (this is the core "residual stream + Pre-LN" idea):
        #
        #   1. x = x + self.attn(self.ln_1(x), rope=rope)
        #      - LN_1 normalizes the input view for attention.
        #      - attention reads from that normalized view.
        #      - the residual `x +` writes the attention output back to the
        #        UNNORMALIZED stream.
        x = x + self.attn(self.ln_1(x), rope=rope)
        #
        #   2. x = x + self.mlp(self.ln_2(x))
        #      - Same pattern: LN_2 normalizes for the MLP; MLP writes back via residual.
        x = x + self.mlp(self.ln_2(x))
        return x
        #
        #   3. return x
        #
        # Note the residual `x +` is OUTSIDE the LN call. LN is applied only to
        # the input view that the sub-layer sees. The stream itself is never
        # normalized — each block adds a small contribution to the raw stream.
        raise NotImplementedError("implement Block.forward")
