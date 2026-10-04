import pytest
from synor.bpe_tokenizer import BPETokenizer


def test_bpe_encode_decode():
    tokenizer = BPETokenizer()
    text = "Synor AI is a native personal assistant foundation model."
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0

    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_empty_string():
    tokenizer = BPETokenizer()
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
