"""tiktoken `gpt2` wrapper. Single source of truth for vocab_size and EOT id."""

from __future__ import annotations


class Tokenizer:
    """Thin wrapper around tiktoken.get_encoding('gpt2')."""

    def __init__(self) -> None:
        # self.enc = tiktoken.get_encoding("gpt2")
        # self.eot = self.enc.eot_token  # 50256
        # self.vocab_size = self.enc.n_vocab  # 50257
        raise NotImplementedError

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
