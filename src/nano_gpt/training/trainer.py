"""Training loop: bf16 autocast, grad accum, clip, LR schedule, eval, ckpt."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer

from nano_gpt.config import DataConfig, TrainConfig
from nano_gpt.data.loader import DistributedDataLoader
from nano_gpt.training.distributed import DistEnv


class Trainer:
    """
    Owns the inner loop; stateless w.r.t. model construction.

    Per step:
      1. zero_grad
      2. for k in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            with autocast(bf16): logits, loss = model(x, y)
            (loss / grad_accum_steps).backward()
            (DDP no_sync on all but last micro-step)
      3. clip grads
      4. step optimizer; update LR via schedule
      5. log; periodic eval; periodic ckpt
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DistributedDataLoader,
        val_loader: DistributedDataLoader,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
        env: DistEnv,
    ) -> None:
        raise NotImplementedError

    def fit(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Run val_loader for N batches, return {'val_loss': ...}."""
        raise NotImplementedError

    def save_checkpoint(self, step: int) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> int:
        """Returns step to resume from."""
        raise NotImplementedError
