"""Paper-faithful GPT-2 medium (350M)."""

from nano_gpt.config import DataConfig, ModelConfig, TrainConfig

model = ModelConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
)

train = TrainConfig(
    batch_size=32,
    grad_accum_steps=16,
    lr=3e-4,
    max_steps=19_073,
)

data = DataConfig()
