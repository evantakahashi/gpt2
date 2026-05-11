"""One-shot data prep wrapper. Calls into nano_gpt.data.prepare_fineweb."""

from __future__ import annotations

import argparse

from nano_gpt.data.prepare_fineweb import main as prep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fineweb_edu_10B")
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--val-shards", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Stop after writing this many tokens total (useful for tiny slices).",
    )
    args = parser.parse_args()
    prep(args.out_dir, args.shard_size, args.val_shards, args.max_tokens)


if __name__ == "__main__":
    import os

    main()
    # HF streaming spawns prefetcher threads that don't tear down on iterator exit;
    # hard-exit so the script returns instead of hanging after the work is done.
    os._exit(0)
