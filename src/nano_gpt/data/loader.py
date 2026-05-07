"""Sharded distributed data loader. Each rank reads a strided window of one shard at a time."""

from __future__ import annotations

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
        # self.shards = sorted(glob(...))
        # self.current_shard = 0
        # self.tokens = self._load_shard(self.shards[0])
        # self.position = rank * batch_size * seq_len
        raise NotImplementedError

    def _load_shard(self, path: str):
        """np.memmap or np.fromfile -> torch.from_numpy (long)."""
        raise NotImplementedError

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance position; on shard end, load next shard. Returns (x, y) on CPU pinned."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        """For deterministic resume."""
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
