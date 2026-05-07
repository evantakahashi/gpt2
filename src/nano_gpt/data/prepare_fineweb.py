"""Tokenize FineWeb-Edu (sample-10BT) into uint16 .bin shards.

Output layout (under DataConfig.data_dir):
    train_000000.bin  ... train_NNNNNN.bin
    val_000000.bin

Each shard is a flat array of token ids; documents separated by EOT (50256).
Default shard size ~= 100M tokens.
"""

from __future__ import annotations


def main(
    out_dir: str = "data/fineweb_edu_10B",
    shard_size: int = 100_000_000,
    val_shards: int = 1,
    num_proc: int | None = None,
) -> None:
    """
    1. Stream from huggingface: load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT')
    2. Tokenize each doc with tiktoken gpt2; prepend EOT.
    3. Pack tokens into uint16 numpy arrays of `shard_size`; flush as <split>_NNNNNN.bin.
    4. First `val_shards` go to val, rest to train.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
