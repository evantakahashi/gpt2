"""Train entry point.

Usage:
    torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/gpt2_124m.py
"""

from __future__ import annotations

import argparse


def load_config(path: str):
    """exec config file; return its (model, train, data) module-level objects."""
    raise NotImplementedError


def main() -> None:
    """
    1. parse --config
    2. init_distributed()
    3. seed; build model; wrap (DDP/FSDP); torch.compile if cfg.compile
    4. build optimizer + loaders
    5. Trainer(model, optim, train_loader, val_loader, ...).fit()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
