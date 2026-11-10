"""python -m src.generate --ckpt runs/NAME/best.pt --prompt "Once upon a time" --temperature 0.8 --n 3"""
import argparse
import torch
from src.checkpoint import load_model
from src.tokenizer import load_tokenizer, EOT


def truncate_at_eos(ids, eos_id):
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def sample_texts(model, tok, prompt, n=1, max_new_tokens=200, temperature=0.8,
                 seed=None, device="cpu", use_bos=True):
    """Return n continuations of `prompt` (the prompt text is included in each result)."""
    if seed is not None:
        torch.manual_seed(seed)
    bos = tok.token_to_id(EOT)                       # <|endoftext|> doubles as start-of-story marker
    prompt_ids = tok.encode(prompt).ids
    prefix = ([bos] if use_bos else []) + prompt_ids
    assert prefix, "empty prompt needs use_bos=True"
    idx = torch.tensor([prefix] * n, dtype=torch.long, device=device)       # (n, T)
    out = model.generate(idx, max_new_tokens, temperature=temperature, eos_id=bos)
    texts = []
    for row in out.tolist():
        new = truncate_at_eos(row[len(prefix):], bos)
        texts.append(tok.decode(prompt_ids + new))
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="configs/tokenizer.json")
    p.add_argument("--prompt", default="")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_bos", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(a.ckpt, device), load_tokenizer(a.tokenizer)
    texts = sample_texts(model, tok, a.prompt, a.n, a.max_new_tokens, a.temperature,
                         a.seed, device, use_bos=not a.no_bos)
    for i, t in enumerate(texts, 1):
        print(f"--- sample {i} (temperature={a.temperature}) ---\n{t}\n")


if __name__ == "__main__":
    main()