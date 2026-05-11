import numpy as np
import torch

from nano_gpt.data.loader import DistributedDataLoader
from nano_gpt.data.prepare_fineweb import write_shards


def test_write_shards_produces_uint16_bins(tmp_path):
    docs = ["the quick brown fox", "jumps over the lazy dog", "lorem ipsum dolor sit amet"] * 50
    paths = write_shards(iter(docs), out_dir=str(tmp_path), shard_size=200, val_shards=1)
    assert len(paths) >= 2
    val_paths = [p for p in paths if "val_" in p]
    train_paths = [p for p in paths if "train_" in p]
    assert len(val_paths) == 1
    assert len(train_paths) >= 1
    arr = np.memmap(val_paths[0], dtype=np.uint16, mode="r")
    assert arr.size == 200


def test_prepared_shards_load_via_loader(tmp_path):
    docs = ["a short document here."] * 1000
    write_shards(iter(docs), out_dir=str(tmp_path), shard_size=512, val_shards=1)
    loader = DistributedDataLoader(
        data_dir=str(tmp_path),
        shard_glob="train_*.bin",
        batch_size=2,
        seq_len=16,
        rank=0,
        world_size=1,
    )
    x, y = loader.next_batch()
    assert x.shape == (2, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_eot_separates_documents(tmp_path):
    docs = ["hello", "world"]
    paths = write_shards(iter(docs), out_dir=str(tmp_path), shard_size=10, val_shards=1)
    arr = np.memmap(paths[0], dtype=np.uint16, mode="r")
    # EOT (50256) should appear at the start of each doc.
    assert arr[0] == 50256
