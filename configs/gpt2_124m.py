"""Paper-faithful GPT-2 124M baseline."""

from nano_gpt.config import DataConfig, ModelConfig, TrainConfig

model = ModelConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
)

train = TrainConfig(
    batch_size=64,
    grad_accum_steps=8,
    lr=6e-4,
    max_steps=19_073,
)

data = DataConfig()
