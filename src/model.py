import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from src.config import GPTConfig

class Embeddings(nn.Module):
    """Token IDs (B, T) -> input vectors (B, T, C) = token embedding + positional embedding."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # table of shape (vocab, C)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # table of shape (block_size, C)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx):                        # idx: (B, T), int64
        B, T = idx.shape
        assert T <= self.wpe.num_embeddings, f"sequence length {T} exceeds block_size"
        pos_ids = torch.arange(T, device=idx.device)   # (T,) = [0, 1, ..., T-1]
        tok_emb = self.wte(idx)                        # (B, T, C)
        pos_emb = self.wpe(pos_ids)                    # (T, C)
        return self.drop(tok_emb + pos_emb)            # (B, T, C)

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention. (B, T, C) -> (B, T, C). Equations: paper §3.2.1-3.2.2."""

    def __init__(self, cfg: GPTConfig, causal: bool = True):
        super().__init__()
        C = cfg.n_embd
        self.n_head, self.head_dim, self.causal = cfg.n_head, cfg.head_dim, causal
        self.q_proj = nn.Linear(C, C, bias=cfg.bias)     # W_Q for all heads stacked
        self.k_proj = nn.Linear(C, C, bias=cfg.bias)     # W_K
        self.v_proj = nn.Linear(C, C, bias=cfg.bias)     # W_V
        self.out_proj = nn.Linear(C, C, bias=cfg.bias)   # W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def _split_heads(self, t):                           # (B, T, C) -> (B, h, T, d)
        B, T, _ = t.shape
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x))            # (B, h, T, d)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, h, T, T)
        if self.causal:
            scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(scores, dim=-1)                  # (B, h, T, T), rows sum to 1
        y = self.attn_drop(att) @ v                      # (B, h, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads: (B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return (y, att) if return_attn else y
class LayerNorm(nn.Module):
    """Normalize each token's C features to mean 0 / variance 1, then rescale: y = γ·x̂ + β."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))    # γ, learnable scale
        self.bias = nn.Parameter(torch.zeros(dim))     # β, learnable shift

    def forward(self, x):                              # (..., C)
        xf = x.float()                                 # statistics in fp32: safer under fp16 training
        mean = xf.mean(dim=-1, keepdim=True)           # (..., 1)
        var = xf.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = ((xf - mean) / torch.sqrt(var + self.eps)).to(x.dtype)
        return self.weight * x_hat + self.bias


class FeedForward(nn.Module):
    """Position-wise FFN, paper §3.3 with GELU as in GPT-2: (B,T,C) -> (B,T,4C) -> (B,T,C)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.n_embd
        self.fc1 = nn.Linear(C, 4 * C, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * C, C, bias=cfg.bias)
        self.act = nn.GELU(approximate="tanh")         # the GELU variant GPT-2 used
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """One Transformer block: attention + FFN, each wrapped in a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.pre_ln = cfg.pre_ln
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x):                              # (B, T, C) -> (B, T, C)
        if self.pre_ln:                                # GPT-2: normalize each sublayer's input
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:                                          # original paper: normalize after the add
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x
class GPT(nn.Module):
    """Decoder-only Transformer: token IDs (B, T) -> logits (B, T, vocab)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd) if cfg.pre_ln else nn.Identity()   # post-LN blocks already end in a LayerNorm                         # final LN (needed in pre-LN)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.wte.weight            # one shared tensor

        self.apply(self._init_weights)                             # GPT-2 init everywhere...
        for name, p in self.named_parameters():                    # ...then shrink residual writers
            if name.endswith("attn.out_proj.weight") or name.endswith("ffn.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)                                        # (B, T, C)
        for block in self.blocks:
            x = block(x)                                           # (B, T, C)
        logits = self.lm_head(self.ln_f(x))                        # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())           # shared tensors counted once
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None):
        """idx: (B, T) prompt IDs -> (B, T + new) IDs. temperature=0 means greedy.
        Stops early once every row has produced eos_id; rows that finished sooner keep
        generating, so truncate each row at its first eos_id afterwards."""
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                  # crop to the context window
            logits, _ = self(idx_cond)                                # (B, T', V)
            logits = logits[:, -1, :]                                 # (B, V): last position only
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)         # (B, 1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)       # (B, V)
                next_id = torch.multinomial(probs, num_samples=1)     # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                    # (B, T + 1)
            if eos_id is not None:
                finished |= next_id.squeeze(1) == eos_id
                if finished.all():
                    break
        self.train(was_training)
        return idx