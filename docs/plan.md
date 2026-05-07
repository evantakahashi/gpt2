# Build plan

Roadmap from empty repo to SOTA-extended 350M. Each step ends with a working repo.

## 0. Setup
- `pip install -e .`
- Skim GPT-2 paper §2 (architecture) and §3 (training).

## 1. Data path (tiny slice first)
- Implement `data/tokenizer.py`.
- Implement `data/prepare_fineweb.py` against a 100MB slice.
- Implement `data/loader.py`; verify `next_batch()` returns `(B,T)` with `y == x[:, 1:]` shape.

## 2. Paper-faithful 124M model
- `model/norm.py` (LayerNorm only)
- `model/embeddings.py` (TokenEmbedding + LearnedPositionalEmbedding)
- `model/attention.py` (MHA, hand-written `softmax(QK^T/sqrt(d))`)
- `model/mlp.py` (GELU 4x)
- `model/block.py` (pre-LN)
- `model/gpt.py` (assembly + tied lm_head + GPT-2 init scheme)
- Smoke test: forward shapes; overfit a single batch to ~0 loss.

## 3. Single-GPU training
- `training/optim.py` (AdamW, param groups)
- `training/schedule.py` (cosine + warmup)
- `training/trainer.py` (bf16 autocast, grad accum, clip, ckpt)
- Run `gpt2_124m` on full FineWeb-Edu sample-10BT. Target val loss curve from build-nanogpt.

## 4. DDP
- `training/distributed.py`
- Verify single-GPU vs 8x same total batch -> ~same loss curve.

## 5. Free perf wins
- Flip `compile=True` (torch.compile).
- Set `use_flash=True` (torch SDPA).
- Confirm step time drops; loss unchanged.

## 6. SOTA extensions (A/B vs §3-4 baseline ckpt)
Pick any order; gate behind `ModelConfig` flags.
- RoPE -- swap learned-pos for `pos="rope"`.
- RMSNorm -- `norm="rmsnorm"`, drop biases (`bias=False`).
- SwiGLU -- `mlp="swiglu"`, retune hidden ratio.
- GQA -- `attn_impl="gqa"`, `n_kv_heads=4`.
- Muon optimizer -- `optimizer="muon"` for matrix params.
- Stability: QK-norm, z-loss.
- HellaSwag eval (`eval/hellaswag.py`).

## 7. Scale to 350M
- Same code, `configs/gpt2_350m.py` and `configs/sota_350m.py`.
- Compare baseline 350M vs SOTA 350M val loss + HellaSwag.

## Notes
- Reading notes per topic go in `docs/notes/`.
- Keep a numbers table in `docs/results.md` (val loss, tok/s, HellaSwag) per config.
