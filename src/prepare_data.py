"""Download TinyStories + Tiny Shakespeare and encode them to uint16 .bin files."""
import os
import urllib.request
from huggingface_hub import hf_hub_download
from src.tokenizer import load_tokenizer
from src.data import encode_file

RAW = os.environ.get("RAW_DIR", "/kaggle/temp/raw")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/data")
SHK_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def encode(name, txt_path, tok):
    n = encode_file(txt_path, os.path.join(OUT, name + ".bin"), tok)
    print(f"{name}: {n:,} tokens", flush=True)

def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    tok = load_tokenizer("configs/tokenizer.json")   # the committed tokenizer, never retrain

    # Tiny Shakespeare: 90/10 train/val split by characters
    shk = os.path.join(RAW, "shakespeare.txt")
    urllib.request.urlretrieve(SHK_URL, shk)
    text = open(shk, encoding="utf-8").read()
    cut = int(0.9 * len(text))
    for name, chunk in [("shk_train", text[:cut]), ("shk_val", text[cut:])]:
        path = os.path.join(RAW, name + ".txt")
        open(path, "w", encoding="utf-8").write(chunk)
        encode(name, path, tok)

    # TinyStories: validation first (small), so problems show up early
    for split, name in [("valid", "tinystories_val"), ("train", "tinystories_train")]:
        txt = hf_hub_download("roneneldan/TinyStories", f"TinyStoriesV2-GPT4-{split}.txt",
                              repo_type="dataset", local_dir=RAW)
        encode(name, txt, tok)

if __name__ == "__main__":
    main()