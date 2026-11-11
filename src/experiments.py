"""Analysis experiments on a trained model (temperature, prompt length, loss vs context).

python -m src.experiments --ckpt runs/tinystories_main/best.pt --data_dir DATA_DIR --out results/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.checkpoint import load_model
from src.data import load_bin
from src.generate import sample_texts
from src.metrics import repetition_rate, distinct_n
from src.tokenizer import load_tokenizer, EOT

HAND_PROMPTS = [
    "Once upon a time, there was a little girl named",
    "Tom and his dog went to the park. They",
    "The little bird was sad because",
    "One day, a big storm came and",
    "Lily wanted to bake a cake for her mom, but",
]


def get_stories(data, eot_id, n_stories, min_len, max_scan=3_000_000):
    """First n_stories stories (token-id lists, without the EOT) that have >= min_len tokens."""
    ids = np.asarray(data[:max_scan]).astype(np.int64)
    stories, start = [], 0
    for end in np.flatnonzero(ids == eot_id):
        if end - start >= min_len:
            stories.append(ids[start:end].tolist())
            if len(stories) == n_stories:
                break
        start = end + 1
    return stories


@torch.no_grad()
def generate_continuations(model, tok, prompts, n_samples, temperature, max_new, device, seed=0):
    """prompts: equal-length token-id lists. Returns (texts, ended), one entry per (prompt, sample)."""
    eot = tok.token_to_id(EOT)
    torch.manual_seed(seed)
    rows = [[eot] + p for p in prompts for _ in range(n_samples)]    # start marker + prompt
    idx = torch.tensor(rows, dtype=torch.long, device=device)
    out = model.generate(idx, max_new, temperature=temperature, eos_id=eot).tolist()
    texts, ended = [], []
    for row in out:
        new = row[idx.size(1):]
        hit = eot in new
        texts.append(tok.decode(new[: new.index(eot)] if hit else new))
        ended.append(hit)
    return texts, ended


def summarize(texts, ended, n_samples):
    words = [t.split() for t in texts]
    res = {"rep3": float(np.mean([repetition_rate(w, 3) for w in words])),
           "mean_words": float(np.mean([len(w) for w in words])),
           "pct_ended": float(np.mean(ended))}
    if n_samples > 1:   # diversity across the samples drawn from the same prompt
        groups = [words[i:i + n_samples] for i in range(0, len(words), n_samples)]
        res["distinct2"] = float(np.nanmean([distinct_n(g, 2) for g in groups]))
    else:
        res["distinct2"] = float("nan")
    return res


def temperature_experiment(model, tok, stories, device, out_dir, n_prompts, n_samples, max_new,
                           temps=(0, 0.3, 0.7, 1.0, 1.5), prompt_len=16):
    prompts = [s[:prompt_len] for s in stories[:n_prompts]]
    rows = []
    for t in temps:
        k = 1 if t == 0 else n_samples            # greedy is deterministic: one sample is enough
        texts, ended = generate_continuations(model, tok, prompts, k, t, max_new, device)
        rows.append({"temperature": t, **summarize(texts, ended, k)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "temperature_metrics.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(df.temperature, df.rep3, "o-")
    axes[0].set_ylabel("repeated word 3-grams (fraction)")
    axes[1].plot(df.temperature, df.distinct2, "o-")
    axes[1].set_ylabel("distinct-2 across samples")
    for ax in axes:
        ax.set_xlabel("temperature")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature_metrics.png"), dpi=150)
    plt.close(fig)


def prompt_length_experiment(model, tok, stories, device, out_dir, n_prompts, n_samples, max_new,
                             lengths=(1, 4, 16, 64), temperature=0.7):
    rows, lines = [], []
    for L in lengths:
        prompts = [s[:L] for s in stories[:n_prompts]]
        texts, ended = generate_continuations(model, tok, prompts, n_samples, temperature, max_new, device)
        rows.append({"prompt_tokens": L, **summarize(texts, ended, n_samples)})
        print(rows[-1], flush=True)
        for i in range(3):                         # qualitative: same 3 stories at every length
            lines.append(f"### prompt_tokens={L} | story {i}\nPROMPT: {tok.decode(prompts[i])}\n"
                         f"CONTINUATION: {texts[i * n_samples]}\n")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "prompt_length_metrics.csv"), index=False)
    with open(os.path.join(out_dir, "prompt_length_samples.txt"), "w") as f:
        f.write("\n".join(lines))


@torch.no_grad()
def loss_by_position(model, tok, stories, device, length=200, n=500, batch=50):
    """Mean next-token loss at each position, for story-aligned windows [EOT] + story[:length]."""
    eot = tok.token_to_id(EOT)
    seqs = [[eot] + s[:length] for s in stories[:n]]           # length + 1 tokens each
    chunks = []
    for i in range(0, len(seqs), batch):
        b = torch.tensor(seqs[i:i + batch], dtype=torch.long, device=device)
        x, y = b[:, :-1], b[:, 1:]                             # position t predicts token t+1
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        chunks.append(loss.view(x.size(0), -1).float().cpu())
    return torch.cat(chunks).mean(0).numpy()                   # (length,)


def position_experiment(model, tok, stories, device, out_dir):
    curve = loss_by_position(model, tok, stories, device)
    ctx = np.arange(1, len(curve) + 1)                         # context tokens, including the start marker
    pd.DataFrame({"context_tokens": ctx, "loss": curve}).to_csv(
        os.path.join(out_dir, "loss_by_position.csv"), index=False)
    for c in (1, 2, 4, 8, 16, 32, 64, 128, len(curve)):
        print(f"context {c:>3} tokens: loss {curve[c - 1]:.3f}")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ctx, curve)
    ax.set_xscale("log")
    ax.set_xlabel("context length (tokens, incl. start marker)")
    ax.set_ylabel("next-token loss (nats)")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "loss_by_position.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def loss_vs_context(model, tok, stories, device, ks=(1, 2, 4, 8, 16, 32, 64, 127),
                    targets=range(128, 200, 8), n=500, batch=100):
    """Loss on FIXED target tokens when the model only sees the previous k tokens.

    The target at position t is token t of [EOT] + story[:200]; the input is the k tokens just
    before it (k=None: the whole prefix). Returns {k: mean loss, ..., "full": mean loss}."""
    eot = tok.token_to_id(EOT)
    seqs = torch.tensor([[eot] + s[:200] for s in stories[:n]], dtype=torch.long, device=device)
    results = {}
    for k in list(ks) + [None]:
        total, count = 0.0, 0
        for t in targets:
            lo = 0 if k is None else t - k
            x, y = seqs[:, lo:t], seqs[:, t]                  # window before t, and the target tokens
            for i in range(0, x.size(0), batch):
                logits, _ = model(x[i:i + batch])
                total += F.cross_entropy(logits[:, -1, :], y[i:i + batch], reduction="sum").item()
                count += x[i:i + batch].size(0)
        results["full" if k is None else k] = total / count
    return results


def context_experiment(model, tok, stories, device, out_dir):
    res = loss_vs_context(model, tok, stories, device)
    full = res.pop("full")
    for k, v in res.items():
        print(f"previous {k:>3} tokens: loss {v:.3f}")
    print(f"full prefix (128-192 tokens): loss {full:.3f}")
    pd.DataFrame({"context": [str(k) for k in res] + ["full"], "loss": list(res.values()) + [full]}).to_csv(
        os.path.join(out_dir, "loss_vs_context.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(list(res), list(res.values()), "o-", label="only the previous k tokens")
    ax.axhline(full, ls="--", color="gray", label="full prefix")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length k (tokens)")
    ax.set_ylabel("loss on fixed target tokens (nats)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "loss_vs_context.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def report_samples(model, tok, device, out_dir, temps=(0, 0.7, 1.0), max_new=150):
    with open(os.path.join(out_dir, "report_samples.txt"), "w") as f:
        for t in temps:
            for prompt in HAND_PROMPTS:
                for text in sample_texts(model, tok, prompt, n=1 if t == 0 else 2, max_new_tokens=max_new,
                                         temperature=t, seed=0, device=device):
                    f.write(f"### temperature={t} | prompt={prompt!r}\n{text}\n\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--tokenizer", default="configs/tokenizer.json")
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--max_new", type=int, default=100)
    p.add_argument("--n_stories", type=int, default=500)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(a.ckpt, device), load_tokenizer(a.tokenizer)
    val = load_bin(os.path.join(a.data_dir, "tinystories_val.bin"))
    stories = get_stories(val, tok.token_to_id(EOT), n_stories=a.n_stories, min_len=200)
    print(f"{len(stories)} validation stories with >= 200 tokens | device={device}", flush=True)

    print("\n== loss by position ==");  position_experiment(model, tok, stories, device, a.out)
    print("\n== loss vs context ==");  context_experiment(model, tok, stories, device, a.out)
    print("\n== temperature ==");       temperature_experiment(model, tok, stories, device, a.out,
                                                              a.n_prompts, a.n_samples, a.max_new)
    print("\n== prompt length ==");     prompt_length_experiment(model, tok, stories, device, a.out,
                                                                a.n_prompts, a.n_samples, a.max_new)
    print("\n== report samples ==");    report_samples(model, tok, device, a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()