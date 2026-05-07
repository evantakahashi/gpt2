"""SOTA-extended 350M. Fill in extension flags as you implement them."""

from nano_gpt.config import DataConfig, ModelConfig, TrainConfig

model = ModelConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    # toggle as components land:
    # norm="rmsnorm",
    # pos="rope",
    # mlp="swiglu",
    # attn_impl="gqa", n_kv_heads=4,
    # use_flash=True,
    # bias=False,
)

train = TrainConfig(
    batch_size=32,
    grad_accum_steps=16,
    lr=3e-4,
    max_steps=19_073,
    # optimizer="muon",
)

data = DataConfig()
