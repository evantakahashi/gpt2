"""Token + position embeddings. Learned (paper) or RoPE (SOTA)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Standard token embedding; weight tied with lm_head in GPT.

    Internally a learnable (vocab_size, n_embd) lookup table. Calling
    `forward(idx)` for idx of shape (B, T) returns (B, T, n_embd) by
    indexing row `idx[b, t]` of the weight matrix at each position.
    """

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the underlying lookup matrix for weight-tying with lm_head."""
        return self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class LearnedPositionalEmbedding(nn.Module):
    """Paper-faithful: learned absolute position embedding.

    A (block_size, n_embd) parameter matrix. Row `i` is the vector added to
    the token at position `i` in the sequence. Capped at block_size — the
    model cannot extrapolate to sequences longer than training context.
    """

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        # Raw Parameter so the indexing in `forward` is explicit.
        # Default init is small (zeros); the model-level _init_weights pass
        # will overwrite with N(0, 0.02) later. For now zeros is fine for tests.
        self.weight = nn.Parameter(torch.empty(block_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, t: int) -> torch.Tensor:
        """Return positional embedding for the first `t` positions.

        Output shape: (t, n_embd). Will broadcast against token embeddings
        of shape (B, t, n_embd) when summed.
        """
        # TODO(you): return the first `t` rows of self.weight.
        #
        # Spec:
        #   - Input  t: int, 1 <= t <= self.block_size.
        #   - Output:  torch.Tensor of shape (t, n_embd).
        #   - Should be a slice of self.weight, NOT a copy.
        #     (We want gradients to flow back to self.weight.)
        #
        # Hint: standard Python slicing works on nn.Parameter — they ARE tensors.
        # Hint: think (T_indices,) → returns (T, n_embd).
        return self.weight[:t]


# ── RoPE bits below are stubs; implemented in Step 6 (SOTA extensions). ──


class RoPECache(nn.Module):
    """Precomputed cos/sin tables for RoPE; applied inside attention to q,k."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        raise NotImplementedError("RoPECache implemented in Step 6")

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("RoPECache implemented in Step 6")


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate q and k in pairs by (cos, sin). Returns (q_rot, k_rot)."""
    raise NotImplementedError("apply_rope implemented in Step 6")
