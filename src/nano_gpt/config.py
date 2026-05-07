"""Frozen dataclasses for model, training, and data configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Architecture knobs. Defaults reproduce GPT-2 124M."""

    # core
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # extension slots (defaults = paper-faithful)
    norm: Literal["ln", "rmsnorm"] = "ln"
    pos: Literal["learned", "rope"] = "learned"
    mlp: Literal["gelu", "swiglu"] = "gelu"
    attn_impl: Literal["mha", "gqa"] = "mha"
    n_kv_heads: int | None = None  # for GQA; None => n_head

    # perf
    use_flash: bool = False  # torch SDPA / FlashAttention-2
    bias: bool = True        # GPT-2 used bias on linear+ln; modern arches drop it


@dataclass(frozen=True)
class TrainConfig:
    """Optimization + schedule + run loop knobs."""

    # batch
    batch_size: int = 64           # per-rank micro-batch
    grad_accum_steps: int = 8      # global batch = B * accum * world_size
    seq_len: int = 1024

    # optim
    optimizer: Literal["adamw", "muon"] = "adamw"
    lr: float = 6e-4
    min_lr: float = 6e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # schedule
    warmup_steps: int = 715
    max_steps: int = 19_073        # ~10B tokens at 524288 tok/step
    schedule: Literal["cosine", "wsd"] = "cosine"

    # precision / compile
    dtype: Literal["bf16", "fp32"] = "bf16"
    compile: bool = True

    # logging / ckpt
    eval_interval: int = 500
    log_interval: int = 10
    ckpt_interval: int = 2000
    ckpt_dir: str = "checkpoints"
    seed: int = 1337


@dataclass(frozen=True)
class DataConfig:
    """Where shards live and how to read them."""

    data_dir: str = "data/fineweb_edu_10B"
    train_shard_glob: str = "train_*.bin"
    val_shard_glob: str = "val_*.bin"
    tokenizer: Literal["gpt2"] = "gpt2"
    dtype: Literal["uint16", "uint32"] = "uint16"
