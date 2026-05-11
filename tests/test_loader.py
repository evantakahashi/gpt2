import numpy as np
import pytest
import torch

from nano_gpt.data.loader import DistributedDataLoader


def _write_shard(path, n_tokens, vocab_size=50257, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    arr.tofile(path)
    return arr


@pytest.fixture
def shard_dir(tmp_path):
    _write_shard(tmp_path / "train_000000.bin", 4096)
    _write_shard(tmp_path / "train_000001.bin", 4096, seed=1)
    return tmp_path


def test_batch_shapes(shard_dir):
    B, T = 4, 128
    loader = DistributedDataLoader(
        data_dir=str(shard_dir),
        shard_glob="train_*.bin",
        batch_size=B,
        seq_len=T,
        rank=0,
        world_size=1,
    )
    x, y = loader.next_batch()
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_targets_are_shifted_inputs(shard_dir):
    B, T = 4, 128
    loader = DistributedDataLoader(
        data_dir=str(shard_dir),
        shard_glob="train_*.bin",
        batch_size=B,
        seq_len=T,
        rank=0,
        world_size=1,
    )
    x, y = loader.next_batch()
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_multiple_batches_advance(shard_dir):
    B, T = 4, 64
    loader = DistributedDataLoader(
        data_dir=str(shard_dir),
        shard_glob="train_*.bin",
        batch_size=B,
        seq_len=T,
        rank=0,
        world_size=1,
    )
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert not torch.equal(x1, x2)


def test_per_rank_stride_disjoint(shard_dir):
    B, T = 4, 64
    l0 = DistributedDataLoader(
        data_dir=str(shard_dir), shard_glob="train_*.bin",
        batch_size=B, seq_len=T, rank=0, world_size=2,
    )
    l1 = DistributedDataLoader(
        data_dir=str(shard_dir), shard_glob="train_*.bin",
        batch_size=B, seq_len=T, rank=1, world_size=2,
    )
    x0, _ = l0.next_batch()
    x1, _ = l1.next_batch()
    assert not torch.equal(x0, x1)


def test_shard_cycles(shard_dir):
    B, T = 4, 64
    loader = DistributedDataLoader(
        data_dir=str(shard_dir),
        shard_glob="train_*.bin",
        batch_size=B,
        seq_len=T,
        rank=0,
        world_size=1,
    )
    # Each shard has 4096 tokens; one batch consumes B*T+1 = 257 tokens.
    # Pull enough batches to exhaust both shards and confirm we don't crash.
    for _ in range(40):
        x, y = loader.next_batch()
        assert x.shape == (B, T)
