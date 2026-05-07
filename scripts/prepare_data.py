"""One-shot data prep wrapper. Calls into nano_gpt.data.prepare_fineweb."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fineweb_edu_10B")
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--val-shards", type=int, default=1)
    args = parser.parse_args()
    # from nano_gpt.data.prepare_fineweb import main as prep
    # prep(args.out_dir, args.shard_size, args.val_shards)
    raise NotImplementedError


if __name__ == "__main__":
    main()
