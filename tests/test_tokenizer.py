from nano_gpt.data.tokenizer import Tokenizer


def test_round_trip_ascii():
    tok = Tokenizer()
    s = "hello world, this is a test."
    assert tok.decode(tok.encode(s)) == s


def test_round_trip_unicode():
    tok = Tokenizer()
    s = "café — naïve résumé 🌍"
    assert tok.decode(tok.encode(s)) == s


def test_vocab_size_and_eot():
    tok = Tokenizer()
    assert tok.vocab_size == 50257
    assert tok.eot == 50256


def test_encode_returns_ints():
    tok = Tokenizer()
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)
