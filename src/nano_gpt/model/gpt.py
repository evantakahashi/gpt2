"""GPT model: token+pos embed -> N x Block -> final norm -> tied lm_head."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class GPT(nn.Module):
    """
    Owns:
      - embeddings (token + learned-pos OR token + RoPE cache)
      - n_layer Blocks
      - final norm
      - lm_head with weight tying to token embedding
      - paper init: N(0, 0.02); residual projections scaled by 1/sqrt(2*n_layer)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError

    def _init_weights(self, module: nn.Module) -> None:
        """Apply GPT-2 paper init scheme."""
        raise NotImplementedError

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx: (B, T) int64 token ids.
        targets: (B, T) int64 next-token ids; if given, returns CE loss.
        Returns (logits, loss). Logits shape (B, T, vocab_size).
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy / sampled autoregressive generation."""
        raise NotImplementedError

    def num_params(self, non_embedding: bool = True) -> int:
        raise NotImplementedError
