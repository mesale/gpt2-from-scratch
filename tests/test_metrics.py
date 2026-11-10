import math
import pytest
from src.metrics import ngrams, repetition_rate, distinct_n


def test_ngrams():
    assert ngrams(list("abcd"), 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ngrams(["a"], 3) == []


def test_repetition_rate():
    assert repetition_rate("a b c a b c a b c".split(), n=3) == pytest.approx(4 / 7)
    assert repetition_rate("a b c d e f".split(), n=3) == 0.0
    assert repetition_rate("a b".split(), n=3) == 0.0


def test_distinct_n():
    same = ["a b c".split(), "a b c".split()]
    assert distinct_n(same, 2) == pytest.approx(0.5)
    assert distinct_n([["a", "b", "c"], ["d", "e", "f"]], 2) == 1.0
    assert math.isnan(distinct_n([["a"]], 2))