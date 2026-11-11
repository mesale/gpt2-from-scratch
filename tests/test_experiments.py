import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import GPTConfig
from src.model import GPT
from src.experiments import get_stories, summarize, loss_vs_context


class FakeTok:
    def token_to_id(self, s):
        return 0


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


def _tiny():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=32, vocab_size=20)
    stories = [torch.randint(1, 20, (30,)).tolist() for _ in range(6)]
    return GPT(cfg).eval(), stories


def test_loss_vs_context_full_equals_k_when_k_covers_prefix():
    model, stories = _tiny()
    res = loss_vs_context(model, FakeTok(), stories, "cpu", ks=(1, 8), targets=[8], batch=4)
    assert res[8] == pytest.approx(res["full"])          # k = 8 = the whole prefix of target 8
    assert all(math.isfinite(v) for v in res.values())


def test_loss_vs_context_matches_manual():
    model, stories = _tiny()
    seqs = torch.tensor([[0] + s for s in stories])
    manual = F.cross_entropy(model(seqs[:, 6:10])[0][:, -1, :], seqs[:, 10]).item()
    res = loss_vs_context(model, FakeTok(), stories, "cpu", ks=(4,), targets=[10], batch=4)
    assert res[4] == pytest.approx(manual, abs=1e-6)