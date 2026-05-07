"""Causal self-attention. Paper-faithful MHA by default; GQA + RoPE + SDPA slots."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_gpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Paper-faithful causal MHA: c_attn (qkv proj) -> split heads -> softmax(QK^T/sqrt(d)) @ V -> c_proj.

    Extension slots:
      - cfg.use_flash       -> route through F.scaled_dot_product_attention (causal=True)
      - cfg.attn_impl=="gqa" -> q has n_head heads, k/v have n_kv_heads; broadcast to q
      - cfg.pos=="rope"      -> apply RoPE to q,k before the matmul
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # cfg.n_embd, cfg.n_head, cfg.n_kv_heads, cfg.bias, cfg.use_flash, cfg.attn_impl, cfg.pos
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, C). Returns (B, T, C).
        rope: optional (cos, sin) tables when cfg.pos == "rope".
        """
        raise NotImplementedError
