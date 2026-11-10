import torch
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