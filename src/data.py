import numpy as np
from itertools import islice
from src.tokenizer import EOT

def iter_stories(path, block_chars=1 << 24):
    """Yield one story at a time without loading the whole file into memory."""
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        for block in iter(lambda: f.read(block_chars), ""):
            buf += block
            *stories, buf = buf.split(EOT)   # last piece may be an unfinished story: keep it
            for s in stories:
                s = s.strip()
                if s:
                    yield s
    if buf.strip():
        yield buf.strip()

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def encode_file(txt_path, bin_path, tok, batch_size=20_000, max_stories=None):
    """Tokenize a text file; write story tokens + EOT as raw uint16 to bin_path."""
    eot = tok.token_to_id(EOT)
    total = 0
    stories = islice(iter_stories(txt_path), max_stories)   # None means no limit
    with open(bin_path, "wb") as out:
        for batch in batched(stories, batch_size):
            for enc in tok.encode_batch(batch):              # tokenizes in parallel (Rust)
                ids = np.array(enc.ids + [eot], dtype=np.uint16)
                out.write(ids.tobytes())
                total += len(ids)
    return total