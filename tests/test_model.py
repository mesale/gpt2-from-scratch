import torch
from src.config import GPTConfig
from src.model import Embeddings

def test_embeddings_shape():
    cfg = GPTConfig()
    out = Embeddings(cfg)(torch.randint(0, cfg.vocab_size, (2, 10)))
    assert out.shape == (2, 10, cfg.n_embd)

def test_position_matters():
    emb = Embeddings(GPTConfig()).eval()
    idx = torch.full((1, 4), 7)                        # token 7 at all four positions
    tok = emb.wte(idx)
    assert torch.equal(tok[0, 0], tok[0, 1])           # token embedding alone: identical vectors
    out = emb(idx)
    assert not torch.allclose(out[0, 0], out[0, 1])    # with positions: different vectors