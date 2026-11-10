import numpy as np
import pytest
from src.experiments import get_stories, summarize


def test_get_stories():
    data = np.array([5, 6, 7, 0, 8, 0, 9, 9, 9, 9, 0], dtype=np.uint16)
    assert get_stories(data, 0, n_stories=5, min_len=3) == [[5, 6, 7], [9, 9, 9, 9]]
    assert get_stories(data, 0, n_stories=1, min_len=1) == [[5, 6, 7]]


def test_summarize():
    s = summarize(["a b c a b c a b c", "x y z"], [True, False], n_samples=1)
    assert s["pct_ended"] == 0.5 and s["mean_words"] == 6.0
    assert s["rep3"] == pytest.approx((4 / 7 + 0.0) / 2)
    assert np.isnan(s["distinct2"])
    assert summarize(["a b c", "a b c"], [True, True], n_samples=2)["distinct2"] == pytest.approx(0.5)