import torch

from nano_gpt.model.embeddings import LearnedPositionalEmbedding, TokenEmbedding


# --- TokenEmbedding ---

def test_token_emb_shape_and_dtype():
    vocab, D, B, T = 100, 32, 4, 16
    te = TokenEmbedding(vocab, D)
    idx = torch.randint(0, vocab, (B, T))
    out = te(idx)
    assert out.shape == (B, T, D)
    assert out.dtype == torch.float32


def test_token_emb_weight_shape_for_tying():
    vocab, D = 50257, 768
    te = TokenEmbedding(vocab, D)
    assert te.weight.shape == (vocab, D)
    # Same object as the underlying lookup, so tying lm_head.weight = te.weight works.
    assert te.weight is te.emb.weight


def test_token_emb_same_id_same_vector():
    """Different positions with the same token id must produce identical vectors."""
    vocab, D = 100, 32
    te = TokenEmbedding(vocab, D)
    idx = torch.tensor([[7, 7, 7], [7, 0, 7]])
    out = te(idx)
    # All the 7s map to the same row of the table.
    assert torch.equal(out[0, 0], out[0, 1])
    assert torch.equal(out[0, 0], out[0, 2])
    assert torch.equal(out[0, 0], out[1, 0])
    assert torch.equal(out[0, 0], out[1, 2])


# --- LearnedPositionalEmbedding ---

def test_pos_emb_shape():
    block_size, D, T = 1024, 768, 64
    pe = LearnedPositionalEmbedding(block_size, D)
    out = pe(T)
    assert out.shape == (T, D)


def test_pos_emb_full_block_size():
    block_size, D = 128, 16
    pe = LearnedPositionalEmbedding(block_size, D)
    out = pe(block_size)
    assert out.shape == (block_size, D)


def test_pos_emb_returns_view_not_copy():
    """Sliced rows should share storage with self.weight so grads flow back."""
    pe = LearnedPositionalEmbedding(16, 8)
    out = pe(4)
    # Mutating the slice mutates the weight (same storage).
    with torch.no_grad():
        out.zero_()
    assert torch.equal(pe.weight[:4], torch.zeros(4, 8))


def test_pos_emb_distinct_positions_distinct_rows():
    """Positions 0..t-1 should yield t distinct vectors (with prob ≈ 1 at init)."""
    pe = LearnedPositionalEmbedding(64, 32)
    out = pe(8)
    # Pairwise check: no two rows are identical.
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.equal(out[i], out[j])


def test_pos_emb_broadcasts_with_token_emb():
    """The (T, D) pos emb should broadcast cleanly with (B, T, D) tok emb.

    Structural check: the same `pos` vector gets added to every batch row.
    We verify this by adding pos manually to each batch row of tok and comparing
    against `tok + pos` (which relies on broadcasting). No round-trip arithmetic
    on floats, so no precision issues.
    """
    vocab, block_size, D, B, T = 100, 64, 16, 3, 5
    te = TokenEmbedding(vocab, D)
    pe = LearnedPositionalEmbedding(block_size, D)
    idx = torch.randint(0, vocab, (B, T))
    tok = te(idx)
    pos = pe(T)
    via_broadcast = tok + pos
    via_explicit = torch.stack([tok[b] + pos for b in range(B)])
    assert via_broadcast.shape == (B, T, D)
    assert torch.equal(via_broadcast, via_explicit)
