from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256   # max context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384       # model width, called d_model in the paper
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must divide evenly across heads"
        self.head_dim = self.n_embd // self.n_head   # d_k = d_v = 64 here