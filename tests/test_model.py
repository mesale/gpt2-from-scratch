import torch
import torch.nn.functional as F
from src.config import GPTConfig
from src.model import Embeddings, CausalSelfAttention, LayerNorm, FeedForward, Block

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

def test_attention_shape_and_params():
    cfg = GPTConfig()
    attn = CausalSelfAttention(cfg)
    assert attn(torch.randn(2, 10, cfg.n_embd)).shape == (2, 10, cfg.n_embd)
    assert sum(p.numel() for p in attn.parameters()) == 4 * (cfg.n_embd**2 + cfg.n_embd)

def test_causal_no_future_leak():
    torch.manual_seed(0)
    cfg = GPTConfig()
    x = torch.randn(1, 10, cfg.n_embd)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, cfg.n_embd)             # change only positions 6..9
    causal = CausalSelfAttention(cfg, causal=True).eval()
    assert torch.allclose(causal(x)[:, :6], causal(x2)[:, :6], atol=1e-6)   # past unaffected
    full = CausalSelfAttention(cfg, causal=False).eval()
    assert not torch.allclose(full(x)[:, :6], full(x2)[:, :6])              # unmasked DOES leak

def test_attention_weights():
    cfg = GPTConfig()
    attn = CausalSelfAttention(cfg).eval()
    _, att = attn(torch.randn(1, 10, cfg.n_embd), return_attn=True)
    assert att.shape == (1, cfg.n_head, 10, 10)
    assert torch.allclose(att.sum(-1), torch.ones(1, cfg.n_head, 10), atol=1e-5)
    assert torch.all(att.triu(1) == 0)                    # nothing above the diagonal

def test_matches_torch_sdpa():
    cfg = GPTConfig()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 12, cfg.n_embd)
    q, k, v = (attn._split_heads(p(x)) for p in (attn.q_proj, attn.k_proj, attn.v_proj))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.out_proj(ref.transpose(1, 2).reshape(2, 12, cfg.n_embd))
    assert torch.allclose(attn(x), ref, atol=1e-5)

def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16) * 3 + 1
    mine, ref = LayerNorm(16), torch.nn.LayerNorm(16, eps=1e-5)
    with torch.no_grad():                               # random γ, β so the test isn't trivial
        g, b = torch.randn(16), torch.randn(16)
        mine.weight.copy_(g); mine.bias.copy_(b)
        ref.weight.copy_(g); ref.bias.copy_(b)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)

def test_layernorm_statistics():
    out = LayerNorm(32)(torch.randn(4, 7, 32) * 10 + 5)
    assert torch.allclose(out.mean(-1), torch.zeros(4, 7), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(4, 7), atol=1e-3)

def test_ffn_shape_and_params():
    cfg = GPTConfig(); C = cfg.n_embd
    ffn = FeedForward(cfg)
    assert ffn(torch.randn(2, 10, C)).shape == (2, 10, C)
    assert sum(p.numel() for p in ffn.parameters()) == 8 * C * C + 5 * C

def test_block_params_shape_causality():
    torch.manual_seed(0)
    cfg = GPTConfig(); C = cfg.n_embd
    blk = Block(cfg).eval()
    assert sum(p.numel() for p in blk.parameters()) == 12 * C * C + 13 * C   # = 1,774,464
    x = torch.randn(1, 10, C)
    x2 = x.clone(); x2[:, 6:] = torch.randn(1, 4, C)
    assert blk(x).shape == x.shape
    assert torch.allclose(blk(x)[:, :6], blk(x2)[:, :6], atol=1e-5)          # no leak through a block

def test_post_ln_block():
    cfg = GPTConfig(pre_ln=False)
    out = Block(cfg).eval()(torch.randn(2, 10, cfg.n_embd) * 4 + 2)
    assert out.shape == (2, 10, cfg.n_embd)
    assert torch.allclose(out.mean(-1), torch.zeros(2, 10), atol=1e-5)       # last op is a LayerNorm