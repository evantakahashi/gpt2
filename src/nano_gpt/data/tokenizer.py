"""tiktoken `gpt2` wrapper. Single source of truth for vocab_size and EOT id."""

from __future__ import annotations

import tiktoken


class Tokenizer:
    """Thin wrapper around tiktoken.get_encoding('gpt2')."""

    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        return self.enc.encode(text, allowed_special=allowed_special or set())

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)
