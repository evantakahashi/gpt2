# nano_gpt_scratch -- design

Date: 2026-05-07

## Goal
Reimplement GPT-2 from the paper, single-node multi-GPU, scaling 124M -> 350M. Keep
extension points clean so SOTA components (RoPE, RMSNorm, SwiGLU, GQA, FlashAttn, Muon,
QK-norm, z-loss, FineWeb data, HellaSwag eval) can be A/B'd against a paper-faithful
baseline without rewriting the model.

## Non-goals
- Multi-node training.
- Inference-time KV cache (generate is just a sampling stub).
- Hydra / YAML configs (Python dataclasses are enough).
- Tests (skipped per user, can be added later).

## Architecture

Hybrid layout: flat top-level scripts; modular `src/nano_gpt/` package with one
responsibility per file.

```
configs/                       # per-run dataclass instantiations (124m, 350m, sota)
src/nano_gpt/
  config.py                    # ModelConfig, TrainConfig, DataConfig
  model/{gpt, block, attention, mlp, embeddings, norm}.py
  data/{tokenizer, prepare_fineweb, loader}.py
  training/{trainer, optim, schedule, distributed}.py
  eval/{loss, hellaswag}.py
scripts/{train, sample, prepare_data, eval}.py
docs/{plan.md, notes/}
```

## Data flow
1. `prepare_fineweb.py` writes uint16 shards under `data/fineweb_edu_10B/`.
2. `DistributedDataLoader(rank, world_size)` mmaps shards, emits `(x, y)` of `(B, T)`.
3. `Trainer.fit()`: bf16 autocast forward in `GPT.forward(idx, targets)` -> `(logits, loss)`.
4. grad accum -> clip -> AdamW step -> cosine LR update -> log -> periodic eval/ckpt.

## Extension strategy
Every SOTA swap is a `ModelConfig` flag:
- `norm`: ln | rmsnorm
- `pos`: learned | rope
- `mlp`: gelu | swiglu
- `attn_impl`: mha | gqa  (+ `n_kv_heads`)
- `use_flash`: routes attention through torch SDPA / FA-2
- `bias`: turn off biases for modern arches

Optimizer swap is a `TrainConfig` flag (`optimizer: adamw | muon`).

This means the scaffold itself never changes shape -- only flags + new module classes.

## Build order
See `docs/plan.md`. Each step ends with a runnable repo so progress is bisectable.

## Conventions
- bf16 autocast assumed (user has bf16 GPU); no fp16 fallback.
- tiktoken `gpt2` only; no tokenizer swap slot.
- Weight init: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer) (paper).
- Param groups: weight decay only on params with ndim >= 2.

## Open questions
- Final eval suite scope: HellaSwag only, or add MMLU-lite / ARC-easy?
- Checkpoint format: raw `torch.save` or safetensors?
- Logging: stdout only, or wandb/tensorboard?
