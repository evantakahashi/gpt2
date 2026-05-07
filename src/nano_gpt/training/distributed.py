"""DDP setup + rank-aware utilities. FSDP slot for later."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    is_master: bool
    device: str  # "cuda:N" or "cpu"


def init_distributed() -> DistEnv:
    """Read RANK/LOCAL_RANK/WORLD_SIZE; init_process_group('nccl'); set CUDA device."""
    raise NotImplementedError


def shutdown_distributed() -> None:
    raise NotImplementedError


def wrap_model(model: nn.Module, env: DistEnv, strategy: str = "ddp") -> nn.Module:
    """strategy: 'ddp' | 'fsdp' (later) | 'none'."""
    raise NotImplementedError
