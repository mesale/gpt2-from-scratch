import matplotlib
matplotlib.use("Agg")
import torch

from src.config import GPTConfig
from src.model import GPT
from src.viz import attention_maps, plot_attention


def _model():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=50)
    return GPT(cfg).eval(), cfg


def test_attention_maps_masked_and_unmasked():
    model, cfg = _model()
    idx = torch.randint(0, 50, (1, 10))
    masked = attention_maps(model, idx, layer=1, causal=True)
    full = attention_maps(model, idx, layer=1, causal=False)
    ones = torch.ones(cfg.n_head, 10)
    assert masked.shape == full.shape == (cfg.n_head, 10, 10)
    assert torch.all(masked.triu(1) == 0)                       # no attention to the future
    assert torch.allclose(masked[:, 0, 0], torch.ones(cfg.n_head))   # first token only sees itself
    assert torch.allclose(masked.sum(-1), ones, atol=1e-5)
    assert torch.allclose(full.sum(-1), ones, atol=1e-5)
    assert full.triu(1).sum() > 0                                # unmasked attends to the future
    assert model.blocks[1].attn.causal is True                   # the model itself is untouched


def test_plot_attention_runs(tmp_path):
    model, _ = _model()
    idx = torch.randint(0, 50, (1, 6))
    m, f = attention_maps(model, idx, 0, True), attention_maps(model, idx, 0, False)
    out = tmp_path / "a.png"
    plot_attention(list("abcdef"), m, f, heads=(0, 1), save_path=str(out))
    assert out.exists() and out.stat().st_size > 0