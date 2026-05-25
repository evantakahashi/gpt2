# Learning notes — nano_gpt_scratch

Reference doc capturing everything covered in the build sessions. Topics ordered by when they came up. Each section: **concept → why → how it shows up in code → things to think about**. Blank "Your notes" blocks at the end of each section for your own annotations.

---

## ⚡ Quiz review — misconceptions to nail down (Step 2 oral quiz)

Things I got wrong or fuzzy on during the Step 2 quiz, with the correct explanation. Re-read these — they're the exact spots where my mental model needed fixing.

### Tokenization (Q1)
- **Tokenization is deterministic, not dataset-dependent at use time.** The 50000 BPE merges were learned once (on WebText, by OpenAI). After `get_encoding("gpt2")`, "hello world" → `[31373, 995]` every time. The dataset only mattered when BPE was originally trained.
- **Why the model needs integers:** it's a neural net — it does linear algebra on numbers. The first op `token_emb(idx)` is a row-lookup that requires integer indices. Strings can't go in.
- **Round-trip `decode(encode(s)) == s` proves losslessness** — no bytes lost. Critical for byte-level BPE (handles any input including emoji/binary). Not just "decode undoes encode."

### Embeddings (Q2)
- **Why position matters — the precise reason: self-attention is permutation-invariant.** Shuffle the input tokens → attention gives the same output. The model literally can't tell "the cat sat" from "sat cat the" without positional info. (Not just "for grammar.")
- **Token embedding is ~30% of params, not 15%.** 50257 × 768 ≈ 38.6M out of 124M.
- **Positional embedding output is (T, D)** — no batch dim. It broadcasts over B when summed with the (B, T, D) token embeddings.

### LayerNorm (Q3)
- **γ is initialized to 1, β to 0** — NOT `N(0, 0.02²)`. (That `N(0, 0.02²)` init is for Linear/Embedding weights.) γ=1, β=0 means LN starts as pure standardization, then learns deviations.
- **Without-LN failure mechanism:** activations drift across depth → softmax/GELU saturate (gradient ≈ 0) → gradients vanish/explode → can't train past a few layers.
- **Two LNs per block (ln_1, ln_2)** because γ and β are separate learnable params; attention and MLP want different input statistics. Sharing one LN couples them.

### Attention (Q4)
- **Causal mask is about TRAINING-time anti-cheating, not stability.** Without it, position i could attend to token i+1 (the answer it's predicting) and trivially copy it → zero loss, no learning. The mask forces "predict next token from past only."
- **Attention weights ARE a probability distribution.** There are TWO softmaxes in the model: (1) attention softmax over (T,T) scores → attention weights summing to 1 per row, inside every block; (2) final softmax over (B,T,V) logits → vocab distribution, at the very end. Don't conflate.
- **Scaling order:** divide by √d_head BETWEEN `Q@K.T` and softmax, not after the V multiply.
- **√d_head not d_head:** variance scales with d_head, so std scales with √d_head. Without scaling, softmax saturates.
- **Multi-head buys specialization:** different heads attend to different relationships in parallel (syntax, coreference, content similarity). Same param count, n_head independent softmaxes.

### MLP (Q5)
- **Attention mixes across TOKENS; MLP processes each token independently (across FEATURES).** I had this backwards — MLP does NOT mix across tokens or across attention heads. (Head-mixing is done by attention's c_proj.) Framing: attention = communication, MLP = computation.
- **Without a nonlinearity, two stacked Linears collapse to one Linear:** `C(Ax+b₁)+b₂ = (CA)x + b'`. The 4× expansion would be wasted. GELU is what unlocks universal approximation.

### Block / Pre-LN / residual stream (Q6)
- **Why the residual stream stays unnormalized — TWO reasons:**
  - *Forward:* the stream accumulates contributions across 12 blocks. Normalizing it each block (Post-LN) would rescale/wash out earlier blocks' contributions, diluting early-block info. Keeping it unnormalized lets block 1's signal survive to block 12. *(This is the reason I missed.)*
  - *Backward:* the unnormalized residual gives gradients a clean `+1` identity path. Normalizing the stream routes gradient through 24 LN Jacobians in series (bad for flow).
- **Why block input/output is the same shape (B,T,D) — it's stackability, NOT "losslessness."** Uniform `(B,T,D) → (B,T,D)` interface means you can stack 12 blocks in a simple loop (`for block: x = block(x)`) with no reshaping glue. Like standardized LEGO connectors. ("Lossless" isn't a meaningful property — the block transforms the data.)

### GPT assembly / init scheme (Q7)
- **Why only residual projections (`c_proj`) get the 1/√(2·n_layer) downscale:** because `c_proj` layers are the ones that WRITE into the residual stream (their output is what gets `x +`'d). The other Linears (c_attn, c_fc) are intermediate and don't write directly to the stream. Only stream-writers need scaling to control how much variance each block dumps in.
- **Where the `2` and `√` come from (NOT arbitrary):** variance-matching. Each block adds 2 contributions to the stream (attn c_proj + mlp c_proj) → `2·n_layer` total. To keep `Var(stream) ≈ 1` after summing `2·n_layer` contributions, each needs variance `1/(2·n_layer)`. Variance = std², so scale std by `1/√(2·n_layer)`.
- **Initial loss = ln(V) ≈ ln(50257) ≈ 10.83.** Random init → uniform logits → P(target) ≈ 1/V → loss = -log(1/V) = log(V). Concrete day-1 sanity check: a fresh GPT-2 shows ~10.8 on the first batch. Wildly different (0, NaN, 50) = bug before training.

### Forward shape progression (Q8)
- **Loss is ALWAYS a scalar `()`.** I confused the flatten INPUT with the OUTPUT. `logits.view(-1, V)` is `(B*T, V)` — that's what goes INTO cross_entropy. `F.cross_entropy(...)` returns a single scalar (mean over all B*T positions). You can't backprop from a tensor; backward needs one number.
- **Shape progression:** `(B,T)` ids → `(B,T,D)` after embed → `(B,T,D)` through all blocks → `(B,T,D)` after ln_f → `(B,T,V)` after lm_head → `()` scalar after CE. Only lm_head changes the last dim (D→V).
- **Inference uses only the LAST position's logits** (`logits[:, -1, :]`). Reason: position t predicts token t+1. The last position predicts what comes after the WHOLE prompt — the only genuinely new token. Earlier positions predict tokens we already have. (Training uses ALL positions — each is a known training signal; that's what makes training sample-efficient.)

### Capstone: full pipeline (Q12)
- **Pre-LN ordering — LN comes BEFORE each sublayer, not between attention and MLP.** Per block: `ln_1 → attention → (add to stream) → ln_2 → MLP → (add to stream)`. I described it as attention→LN→MLP, but LN normalizes the INPUT view each sublayer reads, not attention's output. The unnormalized residual stream flows between sublayers; each sublayer gets a normalized view via its own LN.
- **lm_head is a single Linear (D→V), NOT an MLP.** No hidden layer, no GELU. One matrix multiply from hidden dim to vocab.
- **Training endpoint = cross-entropy loss; inference endpoint = sampled token.** Training: logits at all positions → CE loss (internally log-softmax + NLL). Inference: last position's logits → softmax → top-k → multinomial → token. Don't describe training as "softmax then sample" — that's the inference path.

### Generation: multinomial vs argmax (Q10)
- **`torch.multinomial(probs, n)` samples RANDOMLY from the distribution, weighted by probability.** A token with prob 0.6 gets drawn ~60% of the time. **`argmax` always picks the single max** (deterministic).
- **Why multinomial:** variety. Greedy/argmax gives the same (often repetitive) output every time; multinomial sampling produces diverse, natural generations.
- **The pipeline ties together:** `logits ÷ temperature` (shape: sharper/flatter) → `top-k filter` (restrict to k most likely) → `softmax` → `multinomial sample`. Temperature and top-k SHAPE the distribution; multinomial SAMPLES from it. With argmax, temperature/top-k (beyond k=1) would be pointless — argmax ignores distribution shape.
- **temp < 1** = sharper/deterministic (temp→0 = argmax); **temp > 1** = flatter/random. **top_k=1** = greedy = argmax.

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

### LayerNorm vs BatchNorm in Karpathy's "rows vs columns" framing
Picture activations as a 2D matrix where:
- **Rows** = tokens (one row per (b, t) position; for `(B, T, D)`, flatten B and T → `N = B*T` rows)
- **Columns** = feature dims (D of them — the embedding axis)

Then:
- **LayerNorm normalizes rows.** For each row, compute mean/std across its D feature values. Each row gets its own (mean, std). N independent normalizations.
- **BatchNorm normalizes columns.** For each column, compute mean/std across all rows. Each column gets its own (mean, std). D independent normalizations.

This is **exactly what `dim=-1` does** in our code:
```python
mean = x.mean(dim=-1, keepdim=True)    # reduce over the last (feature) dim → per-row mean
```
"Reduce over dim=-1" = "average across columns per row" = LayerNorm. If you swapped this to `dim=0`, you'd compute per-column means (across rows) — that's the BatchNorm direction.

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

### γ and β are SHARED across all tokens (per-feature, not per-token)
A common confusion: do γ and β change per token? **No.** They're each shape `(D,)` — one scalar per feature dim, shared across:
- Every batch row (B)
- Every position (T)
- Within a single forward pass and the whole training run

Stage 1 (standardization) IS per-token — each (b, t) row uses its own mean/var. But Stage 2 applies the **same** γ and β to every token. So if `γ[47] = 2.0`, then feature 47 gets doubled for **every** token in the batch (and every token in every batch). γ encodes "for the model's representation as a whole, this is how loud each feature should be."

**Refined mental model:**
```
Stage 1 (per-token): every (b, t) row → mean=0, var=1     ← rigid, local
Stage 2 (shared):    same γ ⊙ x̂ + β applied to every row  ← flexible, global
```

### Features = "neurons" (with caveats)
Each of the D dims at a (b, t) position is a "feature" or "neuron activation". Terminology is loose:
- **Activation**: scalar at one (b, t, d) location.
- **Neuron** (= "feature dim"): the index `d`. "Neuron 47" = feature 47 across all tokens.

**Crucial caveat:** features are NOT single-concept. Modern networks learn **distributed representations** — concepts are encoded in **directions** (combinations of dims), not in individual dims. Most neurons are polysemantic (encode multiple concepts mixed together). Sometimes you find monosemantic neurons (single clean concept), but they're the exception. This is the central object of interpretability research.

So when we say "γ[47] amplifies feature 47," we don't mean "amplifies a specific human-interpretable concept." We mean "amplifies whatever distributed bits of information happen to be encoded in dim 47 by training."

### How γ and β learn their values
Same as every other Parameter: gradient descent on the loss.
```
loss → backward → ∂loss/∂γ[i] computed → optimizer step → γ[i] updates
```
If increasing γ[i] would reduce loss, γ[i] increases. If not, it decreases (or stays put). Over many steps, γ and β converge to whatever values minimize loss on the training data. No human intervention; pure mechanical optimization.

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

### Embedding quality scales with token frequency (the long-tail problem)
Each gradient update to `token_emb.weight` only touches the rows for tokens that **appeared in the batch**. So embedding quality scales with how often a token appears in training data:
- Common tokens (" the", " of") get millions of updates → highly refined.
- Rare tokens get tens or hundreds → close to random init.

Natural text follows **Zipf's law** (power-law frequency distribution): the top ~1000 tokens account for ~80% of all updates. The long tail of rare tokens learns much less.

**Weight tying mitigates this.** With `lm_head.weight = token_emb.weight`, the lm_head's softmax-over-vocab gradient touches **every row** at every step (as "wrong-class" gradient signal). Rare token rows still move during training, just not from input-side. Empirically, weight tying improves long-tail performance noticeably.

**BPE also mitigates this.** Byte-level BPE breaks rare words into subwords that appear frequently. "myocardiopathy" → " my", "ocard", "iopathy" — each subword sees enough training to be useful, even if the full word is rare.

Practical implication: model + vocab size must match training corpus size. Tiny corpus + huge vocab = useless tail.

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

### Terminology clarification: "batch" vs "sequence" vs "token"
A common confusion: "is the batch multiple tokens or multiple batches?"

- **One batch = one tensor** of shape `(B, T)` for ids (or `(B, T, D)` after embedding).
- That ONE tensor contains **B independent sequences** (rows).
- Each sequence contains **T sequential tokens** (columns within a row).

Sequences in different rows have nothing to do with each other (different chunks of FineWeb text). Tokens within a row ARE related — they're successive tokens of one document. Attention mixes along T (within a sequence); B is pure parallelism.

So "the batch has many tokens" is true in TWO senses: B sequences and T tokens per sequence. Don't conflate them.

### What is `idx`?
By Karpathy convention: `idx` is the input token-ids tensor, shape `(B, T)`, dtype int64. Each entry is an integer in `[0, vocab_size)`. The first thing `GPT.forward` does is `self.token_emb(idx)`, which uses `idx` as a row-index into the embedding table — hence the name "idx" (= indices). After embedding lookup, the variable gets renamed `x` (now a float `(B, T, D)` tensor) for the rest of the forward.

### Quick sanity calculation
For `B=8, T=1024, D=768`:
- Independent training examples per forward pass: 8
- Total tokens processed: 8 × 1024 = 8,192
- D-dim vectors held in one hidden state: 8,192
- Memory for one block's hidden state (fp32): 8 × 1024 × 768 × 4 bytes ≈ **25 MB**

**Your notes:**

---

## 10. MLP (feedforward) — the "computation" half of every block

### What it is
A tiny per-token feedforward inside every transformer block:
```python
x → c_fc(x)                 # Linear(D, 4*D), expand
  → GELU(x)                 # elementwise nonlinearity
  → c_proj(x)               # Linear(4*D, D), compress
```
Three operations. Input shape `(B, T, D)`, output shape `(B, T, D)`.

### What it does (and DOESN'T do)
- **Does:** apply a learned nonlinear transform to each token's D-dim vector, independently per (b, t).
- **Doesn't:** mix information across tokens. The Linears only operate on the last dim; positions don't see each other. (Compare attention, which is exactly the "mix across tokens" operation.)
- **Doesn't:** predict next tokens. MLP output is still `(B, T, D)` hidden states. Next-token prediction happens ONCE at the end of the model via `lm_head` — a single Linear that projects D → V.

### Attention vs MLP — the "communication vs computation" framing
| | Attention | MLP |
|---|---|---|
| Mixes across tokens? | YES | NO |
| Compute per-token? | Minor (Q, K, V projections) | YES — the whole point |
| Shape | (B, T, D) → (B, T, D) | (B, T, D) → (B, T, D) |
| Conceptual role | "communication" — gather info from other tokens | "computation" — process each token's info |
| Inside a block | runs first | runs second |

Both are per-block; each block has exactly one of each. The model has 12 blocks → 12 attentions + 12 MLPs.

### Why 4× expansion?
The hidden dim is `4 * D` (= 3072 for GPT-2 124M). This came from the original Transformer paper; modern arches sometimes use ~2.67× with SwiGLU to match param count.

Intuition: a nonlinearity sandwiched between two linears is a universal function approximator. The wider the hidden, the more expressivity. Too wide wastes params. 4× is the empirical sweet spot.

### Why a nonlinearity at all? (the math that justifies GELU's existence)

Without a nonlinearity in the middle, the MLP would mathematically collapse to a single Linear:
```
Linear(D, 4D)   then   Linear(4D, D)   with no nonlinearity between
  y = Ax + b₁              z = Cy + b₂

  z = C(Ax + b₁) + b₂  =  (CA)x + (Cb₁ + b₂)  =  one Linear(D, D)
```
The 4× expansion buys nothing. Two stacked Linears = one Linear.

Linear functions can only rotate, scale, reflect, and shift vectors. They cannot:
- Make decisions ("activate this feature only if input > 0")
- Compute non-linear compositions of features (XOR is the canonical example — no single Linear can compute it)
- Approximate arbitrary continuous functions

A nonlinearity in the middle unlocks the **universal approximation theorem**: with enough hidden dim, `Linear → nonlinearity → Linear` can approximate ANY continuous function. That's the whole game.

For language modeling specifically, nonlinearity is needed for:
- Selective activation ("fire on this pattern, suppress otherwise")
- Feature conjunctions ("if A AND B, then C")
- Hierarchical composition across stacked layers

Without GELU, GPT-2 would be a 124M-parameter linear regression. Loss would never converge.

### Why GELU, not ReLU?
- ReLU's gradient is exactly 0 for x < 0 ("dying ReLU"). Many neurons can get stuck.
- GELU is smooth: small but nonzero gradient for slightly-negative x. All neurons keep getting signal.
- GPT-2 specifically uses the **tanh approximation** to GELU (`nn.GELU(approximate="tanh")`). Numerical difference vs exact is ~1e-5; we use the tanh version for paper fidelity.

### Param accounting
- Per MLP: `D × 4D + 4D × D = 8D²` (ignoring biases). For D=768 → 4.7M params.
- Across 12 blocks: 57M params just for MLPs.
- About **half of GPT-2 124M's total params live in MLPs.** Compare to attention's 4D² per block (Q, K, V, output) = half of MLP per-block.

### What "MLP encapsulates attention's learning" actually means (or doesn't)
A common intuition pitfall: thinking MLP somehow stores or summarizes what attention computed. **It doesn't.** Both layers have their own learned parameters. After attention enriches a token's vector with context from other tokens, MLP applies a per-token nonlinear transform to that enriched vector. They're complementary, not nested. Both train independently via backprop.

A better mental model: attention is "gather info from neighbors"; MLP is "now think privately about what you gathered."

### Sources to watch/read
- Karpathy "Let's build GPT: from scratch" (https://youtu.be/kCc8FmEb1nY) — feedforward section ~1:15–1:20 (verify in chapters).
- Karpathy "Let's reproduce GPT-2 (124M)" (https://youtu.be/l8pRSuU81PU) — MLP class in the GPT build, first ~30 min.

### Code reference
- `src/nano_gpt/model/mlp.py` — `GELU_MLP` (full implementation); `SwiGLU_MLP` is a Step 6 stub.

### Open questions
- Could we use ReLU instead and skip the GELU approximation hassle? Yes, marginal performance loss (~1% perplexity). GELU is the GPT-2 paper choice; we follow it.
- Why is the hidden dim 4×D, not 2×D or 8×D? Empirical sweet spot. Modern Llama-style models with SwiGLU use ~2.67× because the gated nonlinearity is more expressive per-dim.

**Your notes:**

---

## 11. Residual connections — why they exist

### The historical problem: vanishing/exploding gradients
Without residuals, gradients in a deep network are the **product of N layer-derivatives**:
```
∂L/∂x_0 = ∂L/∂x_N · ∂f_N/∂x · ∂f_{N-1}/∂x · ... · ∂f_1/∂x
```
If each `∂f/∂x ≈ 0.5`, then after 24 layers the product is `0.5^24 ≈ 6×10⁻⁸` → gradient vanishes. If each ≈ 1.5, product is `~16,000` → gradient explodes. Pre-ResNet (2015), this locked deep networks at ~20 layers.

### The fix: `x_next = x + f(x)`
Derivative becomes `1 + ∂f/∂x`. Even if `∂f/∂x = 0`, the identity term gives gradients a **direct skip path** backward. After N layers, the product is `∏(1 + small) ≈ 1` regardless of N. **Gradients survive depth.**

This single change unlocked depth:
- Pre-ResNet: best CNN = 19 layers (VGG-19).
- Post-ResNet (2015): 152 layers, then 1000+ layers.
- Transformers: 12+ blocks routine; ~100 blocks now feasible.

### Five concrete benefits

1. **Gradient flow.** Direct skip path → no vanishing/exploding regardless of depth.

2. **"Easy default" at init.** With small init, `f(x)` is small, so `x + f(x) ≈ x`. Each block learns small **corrections** to the identity, not "transform the input from scratch." Vastly easier optimization.

3. **Feature reuse via the residual stream.** Every block adds to the stream; nothing replaces it. Information from block 1 can persist to block 12 unchanged. Different blocks can specialize on shallow vs deep features simultaneously.

4. **Stable deep training** (with LayerNorm). LN keeps each sub-layer's input at unit scale; residuals keep gradients from vanishing. Together they make 12+ block training reliable.

5. **Smoother loss surface.** Empirical: residual networks have dramatically smoother loss landscapes than plain stacks. Easier to optimize.

### What we use: plain residual, NOT highway networks
- **Highway Networks** (Srivastava 2015): `out = T(x) · f(x) + (1−T(x)) · x` — learned gates.
- **Residual** (He 2015, ResNet, what we use): `out = x + f(x)` — no gates, simpler.

GPT-2 and all modern transformers use plain residuals.

### In our Block code
```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # residual add #1
    x = x + self.mlp(self.ln_2(x))    # residual add #2
    return x
```
Two `+`s. That's the entire residual mechanism. Removing either of them would break gradient flow through that sub-layer and likely break training at depth.

### What WOULD go wrong without residuals
Concretely: if you remove the `x +` and write `x = sublayer(LN(x))`:
- Initial loss = ln(50257) ≈ 10.8.
- Loss decreases very slowly or not at all — gradient at input embedding is near zero.
- Or activations explode → NaN within 100 steps.
- Either way: no useful training. Empirically tested many times.

### Open questions
- Why don't we scale the residual contribution (e.g. `x + α·f(x)`)? Some papers do this. Empirically, plain `x + f(x)` works well for transformers. The init scheme (1/sqrt(2*n_layer) on c_proj weights) achieves a similar effect.
- ResNet style stays `x + f(x)`; DenseNet uses `concat(x, f(x))`. Why not DenseNet for transformers? Concat would force a wider model with each block. Sum is cheaper and equally effective.

**Your notes:**

---

## 12. Cross-entropy loss for language modeling

### Intuition
Cross-entropy measures **how surprised the model was by the correct answer**. High prob assigned to the target → low loss; tiny prob to the target → high loss.

### Math (per prediction)
Given logits `z = [z_0, ..., z_{V-1}]` and target class `y`:
```
P(k) = exp(z_k) / Σ_j exp(z_j)             # softmax
loss = -log P(y) = -z_y + log_sum_exp(z)
```
The second form is what `F.cross_entropy` actually computes (numerically stable, no exp overflow).

### Quantitative anchor: initial loss ≈ ln(V)
At random init, the model has no information. Logits are roughly uniform → `P(any) ≈ 1/V` → `loss ≈ ln(V)`.
For V = 50257 (GPT-2 vocab): **initial loss ≈ 10.83**.

This is a critical sanity check: if your day-1 initial loss is wildly different (NaN, 0, 50), something is broken **before** training even starts.

### Properties
- **Non-negative**, min 0 (perfect prediction).
- **Unbounded above**: as P(target) → 0, loss → +∞.
- **Heavily penalizes confident wrong predictions** — `-log(0.01) = 4.6` vs `-log(0.5) = 0.7`.
- **Smooth + differentiable** → gradient descent friendly.

### Why CE specifically (not MSE)
Cross-entropy IS the maximum-likelihood objective for categorical distributions. Equivalent to "maximize log P(data | model)." MSE would treat all wrong classes equally; CE punishes confident wrong predictions much harder. CE wins empirically and theoretically for classification.

### The flatten trick (used in GPT.forward)
Logits are `(B, T, V)`, targets are `(B, T)`. `F.cross_entropy` wants 2D inputs `(N, V)` and 1D targets `(N,)`. Flatten:
```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (B*T, V)
    targets.view(-1),                    # (B*T,)
)
```
Returns one scalar — the **mean** of the B*T per-position losses. Every (b, t) position contributes a training signal. This is why transformer training is so sample-efficient.

### Code reference
- `src/nano_gpt/model/gpt.py` — `GPT.forward` computes the loss this way.

**Your notes:**

---

## 13. GPT-2 init scheme — the 1/√(2·n_layer) residual projection scaling

### What we initialize
Every learnable Parameter starts as a Gaussian sample. GPT-2 paper specifies:
- **All Linear weights**: `N(0, 0.02²)` — small.
- **All biases**: 0.
- **Embeddings**: `N(0, 0.02²)`.
- **Residual projections** (`c_proj` in attention and MLP, **only these**): `N(0, (0.02/√(2·n_layer))²)`.

For n_layer=12: residual proj std = `0.02 / √24 ≈ 0.00408`. About **5× smaller** than other Linears.

### Why scale down ONLY the residual projections
At every block, two sub-layer contributions are added to the residual stream. With 12 blocks → 24 sublayer additions. If each contribution had unit variance, the stream's variance would compound additively: `Var(stream) ≈ Var(initial) + 24 × Var(contribution)`. After block 12, the stream would have variance ~25 — way larger than what LN inside the next block expects to see.

Scaling residual projection weights by `1/√(2·n_layer)` makes each contribution have variance `~1/(2·n_layer)` — small enough that 24 additions sum to variance ~1. The stream stays well-conditioned all the way through.

### How we tag residual projections in code
A marker attribute on the Linear:
```python
self.c_proj = nn.Linear(D, D)
self.c_proj.RESIDUAL_PROJ = True
```
Then `GPT._init_weights` checks for this attribute and applies the extra downscaling. Tagged in `attention.py` (c_proj) and `mlp.py` (c_proj).

### What happens without this scaling
Loss converges much slower, may diverge to NaN early, sensitive to learning rate. Empirically tested in the GPT-2 paper appendix.

### Code reference
- `src/nano_gpt/model/gpt.py:_init_weights` — the init function applied via `self.apply(self._init_weights)`.

**Your notes:**

---

## 14. Weight tying — the gradient alignment story

§7 covered weight tying mechanically (`self.lm_head.weight = self.token_emb.weight`). Here's the deeper "why does this work" answer.

### The two uses are different operations on the same matrix
Let W be the shared `(V, D)` matrix:
- **Input embedding lookup**: for token id `i`, return `W[i]` — pick out one row.
- **Output projection (lm_head)**: for hidden state `h`, return `h @ W.T` — V dot products of h with each row.

These are mathematically different operations. The model doesn't need a flag to "distinguish" them — the operations themselves are what they are.

### The shared geometric interpretation
W is a **codebook of token directions** in D-dim space. Each row k IS "the geometric direction representing token k."
- Used as input: "I'm token k → my vector representation is row k."
- Used as output: "Given hidden h, how aligned is h with each token's direction? Higher alignment → higher logit."

Both uses are consistent with the same geometric meaning. The matrix encodes "what each token looks like in D-dim space"; that single notion serves both jobs.

### How gradients work with the shared parameter
PyTorch's autograd handles this automatically via the **sum rule**. When `loss.backward()` runs, contributions from BOTH forward sites accumulate into the same `.grad` attribute. The optimizer reads the combined gradient and takes one step.

### Are the two gradient contributions contradictory?
**No, they're aligned.**
- Input-site gradient: "row k should change so that when used as input, it leads to a hidden state that predicts better."
- Output-site gradient: "row k should change so that as a prediction direction, it's more aligned with hidden states that should predict it."

Both push row k toward the same target: a geometrically coherent representation of token k. They REINFORCE each other rather than fighting. That's why tying empirically improves sample efficiency — every gradient step gets 2× the learning signal per parameter.

### Saves ~30% of model params
At GPT-2 124M: `V × D = 50257 × 768 = 38.6M params` of lm_head are tied to token_emb. ~30% of the model.

### Why not just remove lm_head and reshape the embeddings?
We still need an `nn.Linear(D, V, bias=False)` module so the forward pass can write `self.lm_head(x)`. The trick is that its weight is shared with token_emb. The "Linear module" exists for forward-pass ergonomics; the weight is shared underneath.

### Code reference
- `src/nano_gpt/model/gpt.py:__init__` — the single line `self.lm_head.weight = self.token_emb.weight` that does the tying.

**Your notes:**

---

## 15. Generation — autoregressive sampling

### The loop
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = self(idx_cond)              # (B, T, V)
        logits = logits[:, -1, :] / temperature  # (B, V) - take LAST position only
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx
```

### Why we only look at the last position's logits
At inference, we want `P(next token | seen so far)`. The model produces a (B, T, V) tensor where each (b, t) cell is "what token comes after position t in batch b". We only care about position T-1 — the actual next position. Earlier positions' predictions are wasted (they'd predict tokens we already know).

### Temperature
- `T = 1`: trained distribution as-is.
- `T < 1`: sharpens (more deterministic). At `T → 0`, becomes argmax (greedy).
- `T > 1`: flattens (more random).
Concretely: `logits / temperature` makes the softmax peakier or flatter. Higher T → more diverse output; lower T → more deterministic.

### Top-k sampling
Restrict sampling to the k highest-probability tokens. Setting non-top-k logits to `-inf` makes their softmax prob = 0. Prevents sampling truly weird tokens (like the model giving 0.1% prob to "the weirdest token in vocab" and getting unlucky).
`top_k=1` is argmax (greedy). `top_k=None` means sample from the full distribution.

### Multinomial sampling
`torch.multinomial(probs, num_samples=1)` draws one sample from the probability distribution. Each token's probability of being drawn matches its softmax value.

### Why context-truncate to block_size?
The positional embedding table has only `block_size` rows. If the sequence grows beyond that, we can't embed positions > block_size-1. So we slice to the LAST block_size tokens: `idx[:, -block_size:]`. The model "forgets" anything earlier than the past block_size tokens.

### Code reference
- `src/nano_gpt/model/gpt.py:generate` — the autoregressive loop.
- `scripts/sample.py` will be the CLI that calls this.

**Your notes:**

---

## 16. The Step 2 smoke test — overfit a single batch

### What it is
The gold-standard sanity check that the entire architecture is wired correctly. Train on ONE fixed mini-batch for ~100 steps with AdamW. A correctly-built model with sufficient capacity should be able to **memorize that single batch** quickly, driving loss from ~ln(V) toward 0.

### Why this works as a sanity check
A correctly-built model CAN memorize a tiny dataset. So if you train on a single batch and loss DOESN'T drop, something is broken — typically:
- Missing residual connection (gradient can't flow through depth)
- Wrong target alignment (off-by-one in `y` vs `x`)
- Wrong softmax dim (e.g. softmax over batch instead of vocab)
- Init too aggressive or too small (NaN, or no learning)
- LayerNorm bug
- Embedding not learnable (frozen by accident)
- ...

A correct model usually drops loss from ~ln(V) (e.g., ~3.5 for V=32) to < 0.5 within 100 steps on a tiny batch.

### What gets verified end-to-end
- Forward pass produces logits.
- Cross-entropy is computed at every position.
- Backward populates `.grad` for every Parameter in the model.
- Gradient signal reaches embeddings, attention weights, MLP weights, LayerNorm params.
- Optimizer applies updates that actually reduce loss.
- Init wasn't pathological.

### Code reference
- `tests/test_gpt.py:test_smoke_overfit_single_batch` — exactly this pattern.

### What it doesn't test
- Generalization (overfitting a single batch is the OPPOSITE of generalization).
- Real-corpus learning (no FineWeb here).
- Distributed training, mixed precision, gradient accumulation — all that comes in Step 3+.

The smoke test is a **necessary, not sufficient** condition. Passing it means the architecture is sound. Passing real training is a separate step.

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
