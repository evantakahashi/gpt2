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

## 7. Embeddings (token + positional)

### Why we need embeddings at all
Model input: token ids of shape `(B, T)` — integers in `[0, vocab_size)`. Transformer wants vectors of shape `(B, T, D)`. The embedding layer is the integer → vector translation. Two pieces of information need to be encoded:

1. **What the token *is*** — handled by `TokenEmbedding`.
2. **Where the token is in the sequence** — handled by `LearnedPositionalEmbedding`.

Why both? **Self-attention is permutation-invariant.** If you shuffle input tokens, attention by itself produces the same result. The model literally cannot distinguish "the cat sat" from "sat cat the" without position info. So we inject "where am I" as an additive vector.

### TokenEmbedding — the lookup table
- A learnable `(vocab_size, n_embd)` matrix.
- Forward: for each id in `idx[b, t]`, return the corresponding row.
- At 124M: `(50257, 768)` ≈ 39M params — about **30% of the whole model**.
- Implemented via `nn.Embedding` (which is itself just `Parameter + F.embedding` indexing under the hood).
- Used twice: as input embedding **and** as the final `lm_head` projection (weight tying — Step 2 final assembly). The same matrix that converts id → vector is transposed to convert hidden_state → logits over vocab. Saves 39M params.

### LearnedPositionalEmbedding — the position table
- A learnable `(block_size, n_embd)` matrix.
- Forward: return rows `0..t-1` for an input of length `t`.
- Capped at `block_size = 1024` for GPT-2. **Cannot extrapolate** to longer sequences than seen at training.
- Added (elementwise sum) to token embeddings:
  ```python
  tok_emb = self.token_emb(idx)        # (B, T, D)
  pos_emb = self.pos_emb(T)            # (T, D)
  h = tok_emb + pos_emb                # broadcast over batch → (B, T, D)
  ```
- The `+` works because PyTorch broadcasts `(T, D)` against `(B, T, D)` — same positional vector added to every batch row.

### Why the model learns to use the sum
Naively, adding position to content seems to corrupt both. But because gradients flow back through both embeddings independently, the model learns to allocate **different subspaces of D** for content vs. position, and to disentangle them at downstream attention layers. With D=768 there's plenty of room.

### Alternatives (will come up later)
- **Sinusoidal positions** (original Transformer, "Attention Is All You Need"): fixed sin/cos functions of position, no learned params. Extrapolates to any length in principle, though attention quality degrades.
- **RoPE (Rotary Position Embedding)** — Llama, modern models: a rotation matrix applied to **Q and K inside attention**, not added to the embedding. Encodes *relative* positions, extrapolates better. Will swap in during Step 6.
- **ALiBi**: a position-dependent bias added to attention logits. Different approach again.

### Sources to watch/read before implementing
**Karpathy "Let's build GPT: from scratch"** (https://youtu.be/kCc8FmEb1nY):
- Token embedding as bigram model: **~20:00–35:00** (verify against chapters).
- Positional embedding addition: **~1:00:00–1:10:00**.

**Karpathy "Let's reproduce GPT-2 (124M)"** (https://youtu.be/l8pRSuU81PU):
- First ~15–25 minutes: `wte` (token) and `wpe` (position) defined inside the `GPT` class; `tok_emb + pos_emb` add.

**Papers:**
- GPT-2 §2.1 — one paragraph on learned positions.
- "Attention Is All You Need" §3.5 — original sinusoidal positions.
- RoFormer (Su et al. 2021) — original RoPE paper.

### Code reference
- `src/nano_gpt/model/embeddings.py` — `TokenEmbedding` (full), `LearnedPositionalEmbedding` (your `forward` TODO), `RoPECache`/`apply_rope` stubs for Step 6.

### Are embeddings trained? YES — they're learned end-to-end
Both `TokenEmbedding.weight` and `LearnedPositionalEmbedding.weight` are `nn.Parameter` — exactly like attention weights, MLP weights, LN γ/β. They start as random Gaussian noise (`N(0, 0.02)` per GPT-2 init) and are updated by AdamW via standard backprop, jointly with the rest of the model.

**No pre-training, no hand-engineering.** We don't load word2vec or GloVe or anything. Both matrices start as noise; the next-token-prediction loss is the only training signal.

**Param share at 124M (vocab=50257, n_embd=768, block_size=1024):**
| Component | Params | Share |
|---|--:|--:|
| TokenEmbedding | 38.6M | **31%** |
| LearnedPositionalEmbedding | 0.79M | 0.6% |
| 12 transformer blocks | ~85M | ~68% |

About a third of the entire model is just the token lookup table.

### What learning produces
- **Token rows** end up encoding **semantic similarity**: tokens with similar context (king/queen, cat/dog) get similar vectors. The classic "king − man + woman ≈ queen" property emerges from learned embeddings.
- **Position rows** end up encoding **smooth position structure**: nearby positions are nearby in D-space; far positions are distinguishable. Exact structure is opaque but visualizable.

### Weight tying (preview for `gpt.py`)
GPT-2 uses the **same `(vocab_size, n_embd)` matrix in two places**:
- Input: token id → vector (token_emb).
- Output: hidden vector → logits over vocab (lm_head, the matrix transposed).

```python
self.lm_head = nn.Linear(D, vocab, bias=False)
self.lm_head.weight = self.token_emb.weight   # tied — same Parameter object
```

Forces the model to use the same representation for "what this token looks like (input)" and "what we'd predict for this token (output)." Saves 38.6M params. Works better empirically. The `weight` property on `TokenEmbedding` exists specifically to make this line work.

### Open questions
- Why is sum the right combination, not concat? Concat would burn capacity (D split between content and position). Sum is cheaper and works.
- Could we initialize positional embeddings to zero? Probably fine — the model just learns them from scratch.
- What about pretrained embeddings (like GloVe)? Possible but no longer common — modern LMs train from scratch because joint training with the LM objective outperforms frozen pretrained embeddings.

**Your notes:**

---

## 8. The full GPT-2 forward pass (where does the prediction come from?)

### What attention actually outputs
Attention takes `(B, T, D)` and returns `(B, T, D)` — **same shape**. Each output position is the same token's representation, but now contextually informed by the other tokens it attended to.

**Attention does NOT produce vocabulary predictions.** It produces *better token vectors*.

### The full pipeline (forward pass)
```
ids (B, T) int
    │
    ▼   token_emb(idx) + pos_emb(T)
    │
(B, T, D)
    │
    ▼   ┌─── Block 1: attn + residual + MLP + residual ───┐
    │   ...                                                  ×12 blocks
    ▼   └─── Block 12 ────────────────────────────────────┘
    │
(B, T, D)
    │
    ▼   final LayerNorm
(B, T, D)
    │
    ▼   lm_head: nn.Linear(D, V, bias=False)
(B, T, V)   ← logits over the 50257-token vocabulary, at EVERY position
    │
    ▼   training:   cross_entropy(logits, targets) → scalar loss
        inference:  softmax(logits[:, -1, :]) → sample → next token id
```

### Where the "next token prediction" comes from
- The **last linear layer**, `lm_head`, projects each `D`-dim hidden vector to `V = 50257` logits (one per possible token).
- `softmax(logits[i])` gives the probability distribution over what token comes after position i.
- To **generate**, we softmax the logits at the **last position** of the sequence, then either argmax (greedy) or sample.

### Training vs inference — same forward, different post-processing
**Training:** every position contributes a loss signal. Given input `[t0, t1, ..., t_{T-1}]`, the model produces logits at all T positions. Position i should predict t_{i+1}. We compute cross-entropy at every position. **T loss terms per sequence — extremely sample-efficient.**

**Inference:** we only care about the last position's logits. Sample → append new token → re-run forward on the longer sequence → sample → repeat.

### Why 12 stacked blocks
Each block produces a refined `(B, T, D)` representation. Layer 1 might capture simple local relationships; layer 6 longer-range patterns; layer 12 abstract concepts useful for prediction. This specialization is emergent — gradient signal teaches it, we don't program it. Only the final block's output is read by `lm_head`.

### The shape progression — internalize this
| Stage | Shape | Meaning |
|---|---|---|
| Input ids | `(B, T)` int | token id integers |
| After embedding | `(B, T, D)` float | content + position vectors |
| After block 1 | `(B, T, D)` float | enriched, level 1 |
| After block 12 | `(B, T, D)` float | enriched, level 12 |
| After lm_head | `(B, T, V)` float | logits over vocab, at every position |
| After softmax | `(B, T, V)` float | probability distribution per position |
| After sample (last pos only) | `(B,)` int | next token ids |

**Key:** D stays constant through the whole transformer stack. Only the **last linear** changes the dimensionality from D to V.

### Open questions
- Do we have to compute logits at every position during inference? No — but we do during training because it's free (the matmul is the same cost) and provides T loss signals. During inference we could project only the last position.
- How big is `lm_head`? `D × V = 768 × 50257` = 38.6M params. Same matrix as `token_emb` (weight tying!).

**Your notes:**

---

## 9. What do B, T, D, V actually mean? (terminology check)

These four letters appear in nearly every shape annotation. Worth understanding what each represents.

### B — batch size
- Number of **independent** sequences processed in parallel.
- "Independent" matters: rows in a batch never share information through the model. They're parallel universes that just happen to run on the same GPU at the same time.
- Why we batch: GPUs are massively parallel; processing 64 sequences at once is roughly the same wall-clock cost as processing 1.
- Affects: memory (linearly), gradient noise (inversely with sqrt(B)). Does NOT affect model architecture or parameter count.
- Our config: per-rank `batch_size=64`, with `grad_accum=8`, world_size=8 → effective batch = 4096 sequences/step.

### T — sequence length (a.k.a. block_size when at max)
- How many tokens are in each sequence in the current batch.
- Each sequence is a contiguous chunk of tokens from the dataset (e.g. a 1024-token excerpt from a FineWeb article).
- Attention's compute and memory cost grow as **O(T²)** — long sequences are expensive.
- Capped at `block_size` (model's max context length, 1024 for GPT-2). Cannot exceed because positional embedding table only has block_size rows.

### D — embedding dimension (a.k.a. n_embd)
- The **width** of every internal representation. Every (b, t) cell of the hidden state is a D-dim vector.
- Where it shows up:
  - Token embedding rows are D-dim each.
  - Hidden state through every block: `(B, T, D)`.
  - Q/K/V projections: D → D internally (split into n_head pieces).
  - MLP hidden: 4×D wide internally.
- D is the **capacity knob**. GPT-2 scales it across sizes:
  - 124M: D=768
  - 350M: D=1024
  - 774M: D=1280
  - 1.5B: D=1600
- Most Linear params scale as **D²**, so doubling D ~4×s those params.

### V — vocabulary size
- Total number of distinct tokens the model knows. Fixed by the tokenizer.
- For GPT-2's BPE: V = 50257.
- Appears in two places: token embedding `(V, D)` and lm_head `(D, V)` (which are the same matrix, tied).

### The key difference — B vs D
| | B (batch) | D (embed dim) |
|---|---|---|
| What it indexes | independent examples | features of one token |
| Coupling between slices | none (parallel universes) | dense (Linears mix all D features) |
| Set at... | training time (data axis) | architecture time (fixed for the run) |
| Affects parameters? | no | yes — every Parameter |
| Cost when doubled | ~2× memory, ~2× time | ~4× params for D×D layers |

### A `(B, T, D)` tensor in plain English
`(B=4, T=8, D=768)` =
- 4 independent sequences,
- each with 8 tokens,
- each token represented as a 768-dim vector.

Attention mixes across T (within a sequence). MLP processes each (b, t) cell independently (along the D axis). **Neither operation mixes across B.**

### Quick sanity calculation
For `B=8, T=1024, D=768`:
- Independent training examples per forward pass: 8
- Total tokens processed: 8 × 1024 = 8,192
- D-dim vectors held in one hidden state: 8,192
- Memory for one block's hidden state (fp32): 8 × 1024 × 768 × 4 bytes ≈ **25 MB**

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
