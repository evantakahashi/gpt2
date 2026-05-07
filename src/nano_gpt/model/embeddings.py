"""Token + position embeddings. Learned (paper) or RoPE (SOTA)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Standard token embedding; weight tied with lm_head in GPT."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LearnedPositionalEmbedding(nn.Module):
    """Paper-faithful: learned absolute position embedding of shape (block_size, n_embd)."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, t: int) -> torch.Tensor:
        """Return positional embedding for the first `t` positions."""
        raise NotImplementedError


class RoPECache(nn.Module):
    """Precomputed cos/sin tables for RoPE; applied inside attention to q,k."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape (seq_len, head_dim)."""
        raise NotImplementedError


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate q and k in pairs by (cos, sin). Returns (q_rot, k_rot)."""
    raise NotImplementedError
