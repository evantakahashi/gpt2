"""Sharded distributed data loader. Each rank reads a strided window of one shard at a time."""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch


class DistributedDataLoader:
    """
    Memory-maps shard files and emits (x, y) of shape (B, T) where y = x shifted by 1.

    Per-rank stride: each rank starts at offset rank*B*T and advances by world_size*B*T.
    When the current shard is exhausted, advance to next shard (cycle at end).
    """

    def __init__(
        self,
        data_dir: str,
        shard_glob: str,
        batch_size: int,
        seq_len: int,
        rank: int,
        world_size: int,
        dtype: str = "uint16",
    ) -> None:
        self.data_dir = data_dir
        self.shard_glob = shard_glob
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype

        self.shards = sorted(glob(str(Path(data_dir) / shard_glob)))
        if not self.shards:
            raise FileNotFoundError(f"no shards matched {data_dir}/{shard_glob}")

        self.current_shard = 0
        self.tokens = self._load_shard(self.shards[self.current_shard])
        self.position = rank * batch_size * seq_len

    def _load_shard(self, path: str) -> torch.Tensor:
        np_dtype = np.uint16 if self.dtype == "uint16" else np.uint32
        arr = np.memmap(path, dtype=np_dtype, mode="r")
        return torch.from_numpy(arr.astype(np.int64))

    def _advance_shard(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = self._load_shard(self.shards[self.current_shard])
        self.position = self.rank * self.batch_size * self.seq_len

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.seq_len
        need = B * T + 1
        if self.position + need > len(self.tokens):
            self._advance_shard()

        buf = self.tokens[self.position : self.position + need]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.position += self.world_size * B * T
        return x, y

    def state_dict(self) -> dict:
        return {"current_shard": self.current_shard, "position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.current_shard = state["current_shard"]
        self.tokens = self._load_shard(self.shards[self.current_shard])
        self.position = state["position"]
