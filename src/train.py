"""python -m src.train --config configs/shakespeare.yaml [--set train.max_steps=500 model.n_layer=4]"""
import argparse, csv, math, os, time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import yaml

from src.config import load_config
from src.data import load_bin, get_batch
from src.model import GPT
from src.checkpoint import save_checkpoint, load_checkpoint


def get_lr(step, tc):
    """Linear warmup to tc.lr, then cosine decay down to tc.min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= tc.max_steps:
        return tc.min_lr
    progress = (step - tc.warmup_steps) / (tc.max_steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (tc.lr - tc.min_lr)


def configure_optimizer(model, tc):
    """AdamW; weight decay only on matrices/embeddings (dim >= 2), not biases or LayerNorm."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2),
                             fused=torch.cuda.is_available())


@torch.no_grad()
def estimate_loss(model, splits, tc, block_size, device, ctx):
    model.eval()
    out = {}
    for name, data in splits.items():
        g = torch.Generator().manual_seed(0)           # identical batches at every evaluation
        total = 0.0
        for _ in range(tc.eval_iters):
            x, y = get_batch(data, tc.batch_size, block_size, device, generator=g)
            with ctx:
                _, loss = model(x, y)
            total += loss.item()
        out[name] = total / tc.eval_iters
    model.train()
    return out


def open_log(run_dir, name, header):
    path = os.path.join(run_dir, name)
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if is_new:
        w.writerow(header)
    return f, w


def train(model_cfg, tc):
    torch.manual_seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = tc.amp and device == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    run_dir = os.path.join(tc.out_dir, tc.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path, best_path = os.path.join(run_dir, "ckpt.pt"), os.path.join(run_dir, "best.pt")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({"model": asdict(model_cfg), "train": asdict(tc)}, f)

    splits = {"train": load_bin(os.path.join(tc.data_dir, tc.train_file)),
              "val": load_bin(os.path.join(tc.data_dir, tc.val_file))}

    model = GPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, tc)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_val = 0, float("inf")
    if tc.resume and os.path.exists(ckpt_path):
        step, best_val = load_checkpoint(ckpt_path, model, optimizer, scaler, device)
        print(f"Resumed from {ckpt_path} at step {step}")

    train_f, train_w = open_log(run_dir, "train_log.csv", ["step", "lr", "loss", "grad_norm", "tokens_per_sec"])
    eval_f, eval_w = open_log(run_dir, "eval_log.csv", ["step", "train_loss", "val_loss"])

    tokens_per_step = tc.batch_size * tc.grad_accum * model_cfg.block_size
    print(f"{model.num_params():,} params | {tokens_per_step:,} tokens/step | device={device} amp={use_amp}")

    model.train()
    loss_sum, n_logged, t_last = 0.0, 0, time.time()
    while True:
        if step % tc.eval_interval == 0 or step == tc.max_steps:
            losses = estimate_loss(model, splits, tc, model_cfg.block_size, device, ctx)
            print(f"[eval] step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            eval_w.writerow([step, losses["train"], losses["val"]]); eval_f.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "model_cfg": asdict(model_cfg)}, best_path)
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, best_val, model_cfg, tc)
            t_last = time.time()                       # exclude eval time from throughput
        if step >= tc.max_steps:
            break

        lr = get_lr(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(tc.grad_accum):
            x, y = get_batch(splits["train"], tc.batch_size, model_cfg.block_size, device)
            with ctx:
                _, loss = model(x, y)
            loss = loss / tc.grad_accum                # so the accumulated gradient is an average
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)                     # gradients back to true scale before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        scaler.step(optimizer)                         # skipped automatically if grads had inf/NaN
        scaler.update()
        step += 1

        loss_sum += step_loss; n_logged += 1
        if step % tc.log_interval == 0:
            now = time.time()
            avg, tps = loss_sum / n_logged, n_logged * tokens_per_step / (now - t_last)
            print(f"step {step:>6} | loss {avg:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | {tps:,.0f} tok/s")
            train_w.writerow([step, lr, avg, grad_norm, tps]); train_f.flush()
            loss_sum, n_logged, t_last = 0.0, 0, now

    train_f.close(); eval_f.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    args = parser.parse_args()
    train(*load_config(args.config, args.set))