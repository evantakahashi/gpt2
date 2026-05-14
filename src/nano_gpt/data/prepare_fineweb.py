"""Tokenize FineWeb-Edu (sample-10BT) into uint16 .bin shards.

Output layout (under DataConfig.data_dir):
    train_000000.bin  ... train_NNNNNN.bin
    val_000000.bin

Each shard is a flat array of token ids; documents separated by EOT (50256).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nano_gpt.data.tokenizer import Tokenizer


def write_shards(
    docs: Iterable[str],
    out_dir: str,
    shard_size: int,
    val_shards: int = 1,
    tokenizer: Tokenizer | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Tokenize `docs` and write fixed-size uint16 shards.

    First `val_shards` shards go to val_*.bin, the rest to train_*.bin.
    Stops once `max_tokens` have been written (across all shards) if set.
    Returns the list of shard paths written.
    """
    tokenizer = tokenizer or Tokenizer()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf = np.empty(shard_size, dtype=np.uint16)
    pos = 0
    shard_idx = 0
    val_written = 0
    train_written = 0
    total = 0
    paths: list[str] = []

    def _flush() -> None:
        nonlocal pos, shard_idx, val_written, train_written
        if pos == 0:
            return
        split = "val" if shard_idx < val_shards else "train"
        idx = val_written if split == "val" else train_written
        path = out / f"{split}_{idx:06d}.bin"
        buf[:pos].tofile(path)
        paths.append(str(path))
        if split == "val":
            val_written += 1
        else:
            train_written += 1
        shard_idx += 1
        pos = 0

    pbar = tqdm(unit="tok", total=max_tokens) if max_tokens else None
    eot = tokenizer.eot
    for doc in docs:
        ids = [eot, *tokenizer.encode(doc)]
        for tok in ids:
            buf[pos] = tok
            pos += 1
            total += 1
            if pos == shard_size:
                _flush()
            if max_tokens is not None and total >= max_tokens:
                break
        if pbar:
            pbar.update(len(ids))
        if max_tokens is not None and total >= max_tokens:
            break
    if pbar:
        pbar.close()
    _flush()
    return paths


def _stream_fineweb_docs() -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for row in ds:
        yield row["text"]


def main(
    out_dir: str = "data/fineweb_edu_10B",
    shard_size: int = 100_000_000,
    val_shards: int = 1,
    max_tokens: int | None = None,
) -> None:
    write_shards(
        _stream_fineweb_docs(),
        out_dir=out_dir,
        shard_size=shard_size,
        val_shards=val_shards,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    main()
