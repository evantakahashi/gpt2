# nano_gpt_scratch

Reimplementation of GPT-2 from the paper, with hooks for SOTA extensions.

## Goal
Build GPT-2 124M faithfully from "Language Models are Unsupervised Multitask Learners",
then extend with modern methods (RoPE, RMSNorm, SwiGLU, GQA, FlashAttn, Muon, etc.) and
A/B against the baseline. Single-node multi-GPU, scaling to 350M.

## Layout
- `configs/` — dataclass configs per run
- `src/nano_gpt/model/` — model components, each pluggable
- `src/nano_gpt/data/` — tokenization, sharding, loader
- `src/nano_gpt/training/` — trainer, optim, schedule, distributed
- `src/nano_gpt/eval/` — val loss, downstream evals
- `scripts/` — CLI entries (train, sample, prepare_data, eval)
- `docs/plan.md` — build order / roadmap

## Build order
See `docs/plan.md`.
