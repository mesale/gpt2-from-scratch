import os
from dataclasses import asdict
import torch
from src.config import GPTConfig
from src.model import GPT

def save_checkpoint(path, model, optimizer, scaler, step, best_val, model_cfg, tc):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "best_val": best_val,
                "model_cfg": asdict(model_cfg), "train_cfg": asdict(tc)}, tmp)
    os.replace(tmp, path)     # atomic: a crash mid-save never corrupts the previous checkpoint

def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["best_val"]

def load_model(path, device="cpu"):
    """Rebuild a model for inference from a best.pt or ckpt.pt file (used by generation)."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()