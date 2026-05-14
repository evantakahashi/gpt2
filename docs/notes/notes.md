# Learning notes — nano_gpt_scratch

Reference doc capturing everything covered in the build sessions. Topics ordered by when they came up. Each section: **concept → why → how it shows up in code → things to think about**. Blank "Your notes" blocks at the end of each section for your own annotations.

---

## 1. Tokenizers (BPE) and GPT-2's `gpt2` encoding

### Why a tokenizer exists
The model operates on integers, not text. The tokenizer is the boundary translator:
- `encode(string) → list[int]`
- `decode(list[int]) → string`
- Round-trip property: `decode(encode(s)) == s` must hold.

### Byte-Pair Encoding (BPE), short version
1. Start with every byte as its own token (256 base tokens).
2. Find the most common adjacent pair in your corpus, merge it into a new token id. Repeat ~50000 times.
3. Add 1 special token (`<|endoftext|>`, the "EOT").
4. **GPT-2 vocab = 256 base bytes + 50000 merges + 1 EOT = 50257.**

### Why **byte-level** BPE
- Base alphabet is bytes, not Unicode codepoints.
- Consequence: **any byte sequence is encodable.** Emojis, malformed UTF-8, binary blobs — all round-trip losslessly.
- No `<UNK>` token needed.

### Why **`gpt2`** encoding specifically (vs. cl100k, o200k, SentencePiece)
1. We're reimplementing GPT-2 — model's embedding matrix is sized `(50257, n_embd)`. Tokenizer choice fixes that dimension.
2. Lets us load OpenAI's pretrained GPT-2 weights as a sanity check if we want.
3. The reference run (build-nanogpt) uses it → comparable loss curves.
4. `tiktoken` is fast (Rust) and battle-tested.

### `tiktoken` details that surface in code
- `tiktoken.get_encoding("gpt2")` loads merges from `~/.cache/tiktoken` (one-time download).
- `enc.eot_token` = 50256, `enc.n_vocab` = 50257.
- `enc.encode(s, allowed_special=set())` raises if the literal text `"<|endoftext|>"` appears in the input — safety against accidental special-token injection. To allow it, pass `allowed_special={"<|endoftext|>"}`.
- `enc.decode(ids)` accepts a list of ints; bad ids → U+FFFD replacement char (graceful).

### Observed tokenization behavior worth knowing
| Input | Tokens | Note |
|---|---|---|
| `"hello world"` | `[31373, 995]` | Leading space gets bundled with `' world'`. |
| `"hello\nworld"` | `[31373, 198, 6894]` | Newline is its own token. |
| `"hello 🦀 world"` | `[31373, 12520, 99, 222, 995]` | 4-byte emoji → 3 byte-fallback tokens. |
| `"   leading spaces"` | `[220, 220, 3756, 9029]` | Single space is token 220. |
| `""` | `[]` | Empty round-trips fine. |

### Open questions / things to think about
- How would the loss curve change if we used `cl100k_base` instead? (15% fewer tokens per word → similar text → fewer training steps for same text.)
- What does the BPE "merge order" look like? Why does " world" beat "world"? Hint: frequency in WebText.

### Code reference
- `src/nano_gpt/data/tokenizer.py`

**Your notes:**

---

## 2. uint16 shards on disk

### Why uint16
- GPT-2 vocab = 50257. uint16 range = 0..65535. Fits.
- **4× smaller than int64.** For 10B tokens: 20GB vs 80GB on disk.
- We cast to int64 once at load time because `nn.Embedding` indexing requires int64.

### Why **memory-map** the file (not `torch.load`)
- `np.memmap(path, dtype=uint16, mode="r")` exposes the file as a numpy array via the OS page cache. Only the bytes you read are paged into RAM.
- Multiple processes can memmap the same read-only file with zero locking overhead — kernel handles it.
- Without memmap: a 20GB shard would force a 20GB allocation per rank. Untenable.

### Shard file format
A `.bin` file is a flat array of uint16 ints with **no header, no metadata**. `numpy.ndarray.tofile()` writes raw bytes; `np.memmap` reads them back. The shape is implicit: `(filesize / 2,)`.

### Document boundaries via EOT
Inside a shard, documents are concatenated with EOT (50256) as the **separator that the model can recognize**. Convention is to **prepend** EOT to each document:
```
<EOT> doc1_tokens <EOT> doc2_tokens <EOT> doc3_tokens ...
```
Why prepend not append? Convention from Karpathy / build-nanogpt. Either would work; consistency matters.

### Shard naming
- `train_000000.bin`, `train_000001.bin`, ... and `val_000000.bin`.
- **Zero-padded** to 6 digits so `sorted(glob(...))` is lexicographic == numeric.
- The first `val_shards` shards (default 1) go to val; the rest go to train.

### Subtle artifact: cross-shard document continuation
The writer flushes mid-document when the buffer fills. So:
- `val_000000.bin` starts with EOT.
- `train_000000.bin` does **not** start with EOT — it's the middle of the doc that straddled the seam.
- The loader resets to per-rank offset on shard advance, so the tail of the previous shard's straddler doc is effectively discarded.
- Acceptable artifact at our scale (one seam per 100M tokens vs. millions of clean EOT starts).

### Open questions
- Could we flush only at document boundaries? Yes, but you'd get variable shard sizes which complicates "one big int array per shard" handling. Not worth it.

### Code reference
- `src/nano_gpt/data/prepare_fineweb.py` (writer)
- `src/nano_gpt/data/loader.py:46-49` (reader: memmap + uint16→int64 cast)

**Your notes:**

---

## 3. DataLoader: the shift trick and per-rank striding

### What a training step actually consumes
Each step we want `(x, y)` where `x` is `(B, T)` input tokens and `y` is `(B, T)` target tokens **shifted by one**:
```
x = [t0, t1, t2, ..., t_{T-1}]
y = [t1, t2, t3, ..., t_T]
```
For each position in x, the loss compares the model's prediction against the corresponding y. This shift IS the entire training signal for a causal LM.

### How we implement it cheaply
Pull a 1-D slice of `B*T + 1` tokens. Reshape twice:
```python
buf = tokens[pos : pos + B*T + 1]
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
```
- `B*T + 1` because x and y share `B*T - 1` tokens; only one new token at the right edge.
- `.view()` is a zero-copy stride trick.
- Subtle: across row boundaries in `(B, T)`, x and y are still "concatenated docs" — the attention mask is causal *within* a row, so rows don't see each other anyway.

### Distributed loading without coordination
**N GPUs = N independent processes** (launched by `torchrun --nproc-per-node=N`). Each gets a unique `RANK` env var (0..N-1).

All ranks operate on the **same shard file** at the same time. **Read-only memmap is concurrency-safe** — kernel page cache, no locks needed.

To give each rank disjoint tokens, we use **arithmetic, not synchronization**:
- Rank `r` starts at position `r * B * T`.
- Every step, every rank advances by `world_size * B * T`.
- These per-rank windows tile the shard end-to-end with no gaps and no overlaps.

```
shard: [r0 r1 r2 r3 | r0 r1 r2 r3 | r0 r1 r2 r3 | ...]
       └──step 0──┘ └──step 1──┘ └──step 2──┘
```

### Effective batch sizes
- **Micro-batch**: one rank's `(B, T)` from `next_batch()`.
- **Per-step batch**: `world_size * B` sequences.
- **Effective / global batch**: `world_size * B * grad_accum_steps`.
- For GPT-2 124M defaults (B=64, T=1024, accum=8, world=8): 4.2M tokens per optimizer step.

### Shard cycling
When a shard runs out (`position + B*T+1 > len(tokens)`), advance to the next shard, wrap with `% len(shards)`. Loader is **infinite** — trainer decides when to stop via `max_steps`.

### state_dict
For resume: save `(current_shard, position)`. On load, re-memmap that shard and seek. Tiny but essential for long training runs that crash.

### Open questions
- Why not give each rank its own shard? Different doc distributions per rank → less uniform gradient. Striding mixes data more.
- Could the per-rank-window scheme miss tokens? Yes — partial windows at end-of-shard are discarded. At scale this is negligible.

### Code reference
- `src/nano_gpt/data/loader.py`

**Your notes:**

---

## 4. HuggingFace `datasets` streaming + FineWeb-Edu

### FineWeb-Edu (`sample-10BT`)
- 10B-token slice of the web, filtered for educational quality.
- Pre-shuffled, pre-deduplicated upstream — we don't redo that.
- Lives at `HuggingFaceFW/fineweb-edu` on the HF Hub.

### Streaming vs. full download
- `load_dataset(..., streaming=True)` returns an `IterableDataset` instead of downloading everything to local cache.
- Iterator yields one document at a time; HF lazily fetches Parquet shards in the background.
- For tiny experiments this is essential — full sample-10BT is ~27GB cached.

### The HF prefetcher hang (a real wart)
When you `break` out of the stream early, HF's background prefetch threads keep running and prevent process exit. Workaround: `os._exit(0)` at the end of the prep script.

### Code reference
- `src/nano_gpt/data/prepare_fineweb.py:85-92` (`_stream_fineweb_docs`)
- `scripts/prepare_data.py:25-29` (`os._exit` workaround)

**Your notes:**

---

## 5. Distributed training process model (DDP, high level)

### What `torchrun` does
`torchrun --nproc-per-node=4 train.py`:
1. Spawns 4 separate Python processes.
2. Sets per-process env vars: `RANK={0,1,2,3}`, `LOCAL_RANK={0,1,2,3}`, `WORLD_SIZE=4`, `MASTER_ADDR/PORT` (for inter-process comm).
3. Each process runs `train.py` independently.

### What our code does with it (Step 4 territory)
```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
```

This is essentially `int(os.environ["RANK"])` plus telling NCCL "here are your N peers; here's how to reach them on the network." After init, all-reduce works.

### Two **separate** guarantees that need to hold for DDP correctness
- **torchrun**: each process has a unique rank.
- **Our offset arithmetic**: unique ranks → disjoint data slices.

These are independent. torchrun can't make your loader correct; our loader can't make rank assignment safe. Each layer does its own job.

### Wall-clock timeline of one optimizer step
```
  rank 0:  load batch ─▶ forward ─▶ backward ─▶┐
  rank 1:  load batch ─▶ forward ─▶ backward ─▶├─▶ all-reduce ─▶ optimizer.step()
  rank 2:  load batch ─▶ forward ─▶ backward ─▶│
  rank 3:  load batch ─▶ forward ─▶ backward ─▶┘
                                  (NCCL ring averages gradients across ranks)
```

**Key facts:**
- Forward and backward run **independently per rank**, in parallel, no cross-rank comm.
- Each rank's forward sees only its own micro-batch (its own `(B, T)` slice from the shard).
- Gradients **diverge** between ranks during backward (different data).
- All-reduce **averages** gradients across all ranks — every rank ends up with the same averaged gradient and applies the same optimizer update. Model weights stay in sync because they started in sync and got the same update.
- This makes DDP mathematically equivalent (modulo numerics) to a single-GPU run with batch = `world_size × B`.

### Lockstep vs. interleaving — clearing up a common confusion
- **Spatially** (across the shard): per-rank windows interleave — `[r0 r1 r2 r3 | r0 r1 r2 r3 | ...]`.
- **Temporally** (across time): ranks run **in parallel**, not in turns. Every rank is on step N at the same wall-clock moment. DDP blocks at the all-reduce to enforce this.

Picture: N parallel timelines that synchronize at each step boundary, NOT one timeline where ranks take turns.

### Grad accumulation: deferred all-reduce within a step
Within a single optimizer step, each rank can do `grad_accum_steps` forward/backward passes, **accumulating gradients without all-reducing**. The all-reduce only fires on the last micro-batch.

```
rank r:  fwd→bwd (μb1) → fwd→bwd (μb2) → ... → fwd→bwd (μbK) → all-reduce → opt.step
```

Lets effective batch be much larger than GPU memory allows. Implemented in DDP via `model.require_backward_grad_sync = False` for the first K-1 micro-batches. (We'll write this in Step 3/4.)

### Open questions
- What does NCCL all-reduce actually do under the hood? (Ring-reduce; topology-aware on multi-node.)
- Why nccl over gloo? GPU-aware, faster on H100/A100.
- Bandwidth cost of all-reduce per step? `2 × (N-1)/N × |gradient|` bytes per rank in ring-reduce.

**Your notes:**

---

## 6. LayerNorm (Step 2, in progress)

### Why normalization exists at all
- Without it: deep networks have activations that explode or vanish across layers. Gradients too.
- With it: each layer's activations are forced to (roughly) zero mean, unit variance. Stable gradients, higher LR works, decouples scale from direction.

### LayerNorm vs. BatchNorm — what dimension do we normalize over?
Input shape `(B, T, D)`:
- **BatchNorm**: stats over `(B, T)` for each feature dim. Per-feature mean/var. Used in CNNs.
- **LayerNorm**: stats over `(D,)` for each `(b, t)` token. **Per-token** mean/var.

### Why transformers use LN, not BN
1. **Batch-size independent.** BN needs batch stats; tiny per-rank batches break BN.
2. **No train/eval mismatch.** BN keeps running mean/var for inference; LN behaves identically in both modes.
3. **No DDP all-reduce of stats.** LN's stats are purely local to a token.

### The math
For a vector `x ∈ R^D` (one token's features):

```
μ  = mean(x)            -- scalar
σ² = var(x)             -- scalar (population variance, divide by D not D-1)
x̂ = (x - μ) / sqrt(σ² + ε)
y  = γ ⊙ x̂ + β         -- elementwise scale and offset
```

Where:
- `γ` = `self.weight` ∈ R^D, learned, initialized to 1.
- `β` = `self.bias`   ∈ R^D, learned, initialized to 0. (GPT-2 has it; modern arches drop it.)
- `ε` = `self.eps` ≈ 1e-5, prevents divide-by-zero.

### Why γ and β exist (they seem to undo the normalization)
The normalization forces a fixed distribution. γ and β let the network **un-normalize on purpose** if a particular feature should be a different scale or offset. The model gets the benefit of stable gradients during training PLUS the flexibility to learn its own per-feature scale.

### Pre-LN vs Post-LN (will come up in Block)
- **Post-LN** (original Transformer paper): `LN(x + sublayer(x))`. Hard to train deep networks with this.
- **Pre-LN** (GPT-2 and almost everyone since): `x + sublayer(LN(x))`. LN applied before each sublayer, residual added after. Stabler, allows much deeper stacks.

### Open questions
- Why `unbiased=False` for the variance? Convention to match `nn.LayerNorm`. Population variance, not sample. Difference is negligible at large D.
- What happens with mixed precision (bf16)? Variance can underflow if computed in bf16. Pytorch's LN does it in fp32 internally. Worth checking ours later.

### Code reference
- `src/nano_gpt/model/norm.py` (skeleton in place; LayerNorm.forward is yours to implement)

### Sources to watch/read before implementing
**Primary (watch first, ~10 min):**
- Karpathy, "Let's build GPT: from scratch, in code, spelled out" (Jan 2023, ~1h56m total) — https://youtu.be/kCc8FmEb1nY
  - LayerNorm section is around **~1:28–1:35** (verify against the chapter list — I couldn't fetch chapter timestamps reliably).
  - He builds a tiny BatchNorm1d first, then contrasts with LayerNorm. Best pedagogical walk-through of the math + why-not-BN.

**Secondary (architectural placement, ~30 min):**
- Karpathy, "Let's reproduce GPT-2 (124M)" / build-nanogpt (June 2024, ~4h) — https://youtu.be/l8pRSuU81PU
  - First ~30 min is the model build-out. LayerNorm appears inside the Block class in Pre-LN placement. Math is not re-explained.

**Papers (optional, if curious):**
- Ba, Kiros, Hinton (2016), "Layer Normalization" — https://arxiv.org/abs/1607.06450 — §3 (formulation), §5 (experiments). ~6 pages.
- Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture" — https://arxiv.org/abs/2002.04745 — §3 explains why Pre-LN converges where Post-LN doesn't.
- Radford et al. (2019), GPT-2 paper — §2.1 has one sentence on LN placement.

**Your notes:**

---

## Cheat sheet: numbers worth memorizing

| Name | Value | Where it appears |
|---|---|---|
| `vocab_size` | 50257 | Tokenizer, embedding `(50257, n_embd)`, lm_head |
| `EOT` token id | 50256 | Doc separator in shards |
| `n_embd` (124M) | 768 | Model hidden dim |
| `n_head` (124M) | 12 | Attention heads |
| `n_layer` (124M) | 12 | Block stack depth |
| `block_size` (context) | 1024 | Max sequence length |
| Per-token bytes on disk | 2 | uint16 |
| Per-token bytes in model | 8 | int64 (after cast) |

## Cheat sheet: derived quantities

| Quantity | Formula | At our defaults |
|---|---|---|
| Effective batch (tokens) | `world_size × B × T × grad_accum` | 8 × 64 × 1024 × 8 = 4,194,304 |
| Tokens per shard | `shard_size` parameter | 100M default; 5M for tiny slice |
| Step advance (per rank) | `world_size × B × T` | 8 × 64 × 1024 = 524,288 |

---

## Suggested study order (for going deeper)
1. Karpathy's "Let's build GPT" video (YouTube) — fastest way to internalize the shapes.
2. GPT-2 paper, §2 (architecture) and §3 (training).
3. "Layer Normalization" (Ba et al., 2016) — the original LN paper.
4. "Attention Is All You Need" (Vaswani et al., 2017) — Transformer foundation, though Post-LN.
5. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) — why Pre-LN matters.
