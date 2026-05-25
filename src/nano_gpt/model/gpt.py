"""GPT model: token+pos embed -> N x Block -> final norm -> tied lm_head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_gpt.config import ModelConfig
from nano_gpt.model.block import Block
from nano_gpt.model.embeddings import LearnedPositionalEmbedding, TokenEmbedding
from nano_gpt.model.norm import make_norm


class GPT(nn.Module):
    """
    Owns:
      - embeddings (token + learned position; RoPE deferred to Step 6)
      - n_layer Blocks
      - final LayerNorm (ln_f)
      - lm_head with weight tying to token embedding
      - paper init: N(0, 0.02); residual projections scaled by 1/sqrt(2*n_layer)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embeddings.
        self.token_emb = TokenEmbedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = LearnedPositionalEmbedding(cfg.block_size, cfg.n_embd)

        # Stack of n_layer transformer blocks.
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # Final LayerNorm before the lm_head projection.
        self.ln_f = make_norm(cfg.norm, cfg.n_embd, bias=cfg.bias)

        # Vocabulary projection.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # ────────────────────────────────────────────────────────────────
        # Weight tying — the input embedding matrix IS the output projection.
        # This single assignment makes lm_head.weight and token_emb.weight the
        # SAME Parameter object (same memory, same gradient). Saves ~38.6M params
        # at GPT-2 124M scale (~30% of the model).
        self.lm_head.weight = self.token_emb.weight
        # ────────────────────────────────────────────────────────────────

        # Apply GPT-2 paper init to all submodules.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 paper init: N(0, 0.02) everywhere; residual projections scaled by 1/sqrt(2*n_layer).

        The "residual projection" downscale prevents the residual stream's variance from
        compounding as more block contributions accumulate. Tagged on c_proj layers (in
        attention and MLP) via the RESIDUAL_PROJ attribute.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std = std / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     shape (B, T), int64 — token ids.
        targets: shape (B, T), int64 — next-token ids. Optional. If given, returns CE loss.
        Returns: (logits, loss). logits shape (B, T, vocab_size). loss is scalar or None.
        """
        # TODO(you): implement the full forward pass.
        #
        # Spec (roughly 7 lines):
        #

        #   1. Unpack shapes:
        #        B, T = idx.shape
        #      Also assert T <= self.cfg.block_size (positional embedding ceiling).
        B, T = idx.shape
        assert(T <= self.cfg.block_size)
        #
        #   2. Embed tokens and positions, sum them:
        #        tok = self.token_emb(idx)   # (B, T, D)
        #        pos = self.pos_emb(T)       # (T, D), broadcasts over batch
        #        x = tok + pos               # (B, T, D)
        tok = self.token_emb(idx)
        pos = self.pos_emb(T)
        x = tok + pos
        #   3. Pass through every block in sequence:
        #        for block in self.blocks:
        #            x = block(x)            # (B, T, D), shape preserved
        for block in self.blocks:
            x = block(x)
        #
        #   4. Final LayerNorm:
        #        x = self.ln_f(x)            # (B, T, D)
        x = self.ln_f(x)
        #
        #   5. Project to vocab logits:
        #        logits = self.lm_head(x)    # (B, T, vocab_size)
        logits = self.lm_head(x)
        #
        #   6. Compute cross-entropy loss if targets given. Use the flatten trick:
        #        if targets is not None:
        #            loss = F.cross_entropy(
        #                logits.view(-1, logits.size(-1)),   # (B*T, V)
        #                targets.view(-1),                    # (B*T,)
        #            )
        #        else:
        #            loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        else:
            loss = None
        return (logits, loss)
        #
        #   7. Return (logits, loss).
        # raise NotImplementedError("implement GPT.forward")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Returns idx with max_new_tokens appended.

        idx:           shape (B, T_prompt) - the prompt token ids.
        max_new_tokens: how many new tokens to generate.
        temperature:   logits are divided by this. <1 sharpens, >1 flattens, 1 = trained dist.
        top_k:         optionally restrict to the top-k highest-probability tokens.
        """
        # TODO(you): implement the generation loop.
        #
        # Spec (~10 lines):
        for _ in range(max_new_tokens):
            idx_con = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_con)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx
        #
        #   for _ in range(max_new_tokens):
        #
        #     1. Truncate context to block_size if it's grown beyond:
        #          idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
        #
        #     2. Forward pass (no targets, just want logits):
        #          logits, _ = self(idx_cond)
        #
        #     3. Take the LAST position's logits, divide by temperature:
        #          logits = logits[:, -1, :] / temperature      # (B, V)
        #
        #     4. Top-k filtering (optional):
        #          if top_k is not None:
        #              v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #              logits[logits < v[:, [-1]]] = float("-inf")
        #
        #     5. Softmax to probabilities, then sample:
        #          probs = F.softmax(logits, dim=-1)            # (B, V)
        #          next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
        #
        #     6. Append to idx:
        #          idx = torch.cat((idx, next_id), dim=1)
        #
        #   return idx
        # raise NotImplementedError("implement GPT.generate")

    def num_params(self, non_embedding: bool = True) -> int:
        """Count parameters. By default subtracts position embeddings (convention).

        Note: token_emb.weight is tied to lm_head.weight, so PyTorch's parameters()
        iterator counts it ONCE — that's the conventional behavior we want.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n
