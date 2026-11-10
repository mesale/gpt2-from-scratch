import yaml
from dataclasses import dataclass, fields, asdict

@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256   # max context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384       # model width, called d_model in the paper
    dropout: float = 0.0
    bias: bool = True
    pre_ln: bool = True     # True: GPT-2 style. False: original-paper post-LN
    tie_weights: bool = True

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must divide evenly across heads"
        self.head_dim = self.n_embd // self.n_head   # d_k = d_v = 64 here

@dataclass
class TrainConfig:
    run_name: str = "debug"
    data_dir: str = "data"
    train_file: str = ""
    val_file: str = ""
    out_dir: str = "runs"
    batch_size: int = 64          # micro-batch (sequences per forward pass)
    grad_accum: int = 1           # micro-batches per optimizer step
    max_steps: int = 1000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    amp: bool = True
    seed: int = 1337
    resume: bool = True


def _build(cls, d):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    assert not unknown, f"unknown config keys for {cls.__name__}: {unknown}"
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kwargs[f.name] = f.type(v) if f.type in (int, float, str) else v
    return cls(**kwargs)


def load_config(path, overrides=()):
    """YAML with 'model:' and 'train:' sections, plus overrides like 'train.lr=0.001'."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    sections = {"model": dict(raw.get("model", {})), "train": dict(raw.get("train", {}))}
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        sections[section][name] = yaml.safe_load(value)
    return _build(GPTConfig, sections["model"]), _build(TrainConfig, sections["train"])