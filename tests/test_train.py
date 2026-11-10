import numpy as np
import pytest
import torch
from src.config import GPTConfig, TrainConfig, load_config
from src.model import GPT
from src.train import get_lr, configure_optimizer, train


def test_lr_schedule():
    tc = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert get_lr(0, tc) == pytest.approx(1e-4)       # (0+1)/10 of peak
    assert get_lr(9, tc) == pytest.approx(1e-3)       # end of warmup
    assert get_lr(10, tc) == pytest.approx(1e-3)      # cosine starts at peak
    assert get_lr(60, tc) == pytest.approx(5.5e-4)    # halfway: average of peak and floor
    assert get_lr(110, tc) == pytest.approx(1e-4)
    lrs = [get_lr(s, tc) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_optimizer_groups():
    tc = TrainConfig()
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=100))
    decay, no_decay = configure_optimizer(model, tc).param_groups
    assert decay["weight_decay"] == tc.weight_decay and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    n = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert n == model.num_params()                    # every parameter in exactly one group


def test_config_overrides(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("model:\n  n_layer: 2\ntrain:\n  lr: 1e-3\n")
    m, t = load_config(str(f), ["model.n_head=3", "train.max_steps=5"])
    assert (m.n_layer, m.n_head, t.max_steps) == (2, 3, 5)
    assert t.lr == 1e-3                               # "1e-3" parsed as a float, not a string
    with pytest.raises(AssertionError):
        load_config(str(f), ["train.nonsense=1"])


def test_train_and_resume(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("tr.bin", "va.bin"):
        rng.integers(0, 100, 5000).astype(np.uint16).tofile(tmp_path / name)
    mc = GPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=100)
    tc = TrainConfig(run_name="t", out_dir=str(tmp_path / "runs"), data_dir=str(tmp_path),
                     train_file="tr.bin", val_file="va.bin", batch_size=4, max_steps=20,
                     warmup_steps=2, eval_interval=10, eval_iters=2, log_interval=5, amp=False)
    train(mc, tc)
    ckpt_path = tmp_path / "runs" / "t" / "ckpt.pt"
    assert torch.load(ckpt_path)["step"] == 20
    tc.max_steps = 30
    train(mc, tc)                                     # should resume from step 20
    assert torch.load(ckpt_path)["step"] == 30