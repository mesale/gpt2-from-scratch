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