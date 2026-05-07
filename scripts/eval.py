"""Run eval harness against a checkpoint (val loss, HellaSwag)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", default=["val_loss", "hellaswag"])
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
