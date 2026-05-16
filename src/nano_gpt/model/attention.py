"""Causal self-attention. Paper-faithful MHA by default; GQA + RoPE + SDPA slots."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Paper-faithful causal MHA: c_attn (qkv proj) -> split heads -> softmax(QK^T/sqrt(d)) @ V -> c_proj.

    Extension slots (deferred):
      - cfg.use_flash         -> route through F.scaled_dot_product_attention (Step 5)
      - cfg.attn_impl=="gqa"  -> q has n_head heads, k/v have n_kv_heads (Step 6)
      - cfg.pos=="rope"       -> apply RoPE to q, k before the matmul (Step 6)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(
                f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
            )
        if cfg.use_flash:
            raise NotImplementedError("use_flash path implemented in Step 5")
        if cfg.attn_impl != "mha":
            raise NotImplementedError(
                f"attn_impl={cfg.attn_impl!r} implemented in Step 6 (paper-faithful is 'mha')"
            )
        if cfg.pos == "rope":
            raise NotImplementedError("RoPE path implemented in Step 6")

        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # Combined QKV projection: one big linear instead of three separate ones.
        # Output is concatenated along the last dim; we split it into Q, K, V in forward.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)

        # Output projection back to (B, T, n_embd). The marker attribute below
        # tells the model-level GPT-2 init scheme to scale this weight by
        # 1 / sqrt(2 * n_layer) — keeps the residual stream's variance from
        # blowing up as more block contributions accumulate.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True

        # Causal mask, stored as a non-trainable buffer. Shape (1, 1, block_size, block_size)
        # so it broadcasts over the leading (B, n_head) dims of the attention scores.
        # Slice to :T at forward time.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        x:    shape (B, T, n_embd)
        rope: optional (cos, sin) tables when cfg.pos == "rope". Unused in paper-faithful path.
        returns: shape (B, T, n_embd)
        """
        if rope is not None:
            raise NotImplementedError("RoPE path implemented in Step 6")

        # TODO(you): implement multi-head causal self-attention forward pass.
        #
        # Input  x: shape (B, T, n_embd)
        # Output:    shape (B, T, n_embd)   — same shape, refined representation
        #
        # Recommended steps (matches the shape walkthrough in attention_shapes.html):
        #
        B, T, _ = x.shape
        #
        #   2.  Combined QKV projection:
        #          qkv = self.c_attn(x)            # (B, T, 3 * n_embd)
        qkv = self.c_attn(x)
        #
        #   3.  Split into Q, K, V along the last dim:
        #          q, k, v = qkv.split(self.n_embd, dim=-1)   # each (B, T, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        #
        #   4.  Reshape and transpose each into per-head layout:
        #          q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        #          # Result: (B, n_head, T, head_dim). Same for k, v.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        #
        #   5.  Attention scores (Q @ K^T) and scale by 1/sqrt(head_dim):
        #          scores = q @ k.transpose(-2, -1)            # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) = (B, n_head, T, T)
        #          scores = scores / math.sqrt(self.head_dim)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim) #scale because scores grows linearly with size of head dim
        #
        #   6.  Apply causal mask (set "future" positions to -inf):
        #          mask = self.causal_mask[:, :, :T, :T]       # (1, 1, T, T)
        #          scores = scores.masked_fill(mask == 0, float("-inf"))
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        #
        #   7.  Softmax over the last dim:
        #          weights = scores.softmax(dim=-1)            # (B, n_head, T, T)
        weights = scores.softmax(dim=-1)
        #
        #   8.  Weighted sum of V:
        #          out = weights @ v                           # (B, n_head, T, head_dim)
        out = weights @ v
        #
        #   9.  Transpose back and merge heads:
        #          out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        #
        #   10. Output projection:
        #          return self.c_proj(out)
        return self.c_proj(out)
        #
        # References (open in browser):
        #   docs/notes/visuals/attention_shapes.html — every step's tensor shape, with sliders
        #   docs/notes/visuals/attention.html        — small-numbers compute walkthrough
        #   docs/notes/visuals/attention_study.html  — sections 3 (shapes), 5 (mask), 6 (skeleton)
        # raise NotImplementedError("implement CausalSelfAttention.forward")
