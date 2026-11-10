import torch
from src.config import GPTConfig
from src.model import GPT
from src.generate import truncate_at_eos


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=50)).eval()


def test_shape_and_prompt_preserved():
    m, idx = tiny_model(), torch.randint(0, 50, (3, 5))
    out = m.generate(idx, max_new_tokens=7, temperature=1.0)
    assert out.shape == (3, 12) and torch.equal(out[:, :5], idx)


def test_greedy_deterministic_and_is_argmax():
    m, idx = tiny_model(), torch.randint(0, 50, (1, 5))
    a, b = m.generate(idx, 6, temperature=0), m.generate(idx, 6, temperature=0)
    assert torch.equal(a, b)
    assert a[0, 5].item() == m(idx)[0][0, -1].argmax().item()


def test_sampling_reproducible_with_seed():
    m, idx = tiny_model(), torch.randint(0, 50, (2, 4))
    torch.manual_seed(1); a = m.generate(idx, 8)
    torch.manual_seed(1); b = m.generate(idx, 8)
    assert torch.equal(a, b)


def test_longer_than_context_window():
    out = tiny_model().generate(torch.randint(0, 50, (1, 10)), max_new_tokens=30)   # 40 > block_size 16
    assert out.shape == (1, 40)


def test_stops_at_eos():
    m, idx = tiny_model(), torch.randint(0, 50, (1, 4))
    first = m.generate(idx, 1, temperature=0)[0, -1].item()   # whatever greedy picks first
    assert m.generate(idx, 20, temperature=0, eos_id=first).shape == (1, 5)


def test_restores_training_mode():
    m = tiny_model(); m.train()
    m.generate(torch.randint(0, 50, (1, 4)), 2)
    assert m.training


def test_truncate_at_eos():
    assert truncate_at_eos([1, 2, 3, 2], 3) == [1, 2]
    assert truncate_at_eos([1, 2], 3) == [1, 2]