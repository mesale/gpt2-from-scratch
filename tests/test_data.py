import numpy as np
import torch
from src.tokenizer import EOT
from src.data import encode_file, load_bin, get_batch

def test_vocab_and_roundtrip(tok):
    assert tok.get_vocab_size() == 4096
    for s in ["Once upon a time, there was a little girl.", "Hello\n\nWorld!  ", "café 日本語 🙂"]:
        assert tok.decode(tok.encode(s).ids, skip_special_tokens=False) == s

def test_eot_is_single_token(tok):
    ids = tok.encode("hello" + EOT + "world").ids
    assert ids.count(tok.token_to_id(EOT)) == 1

def test_encode_file(tmp_path, tok):
    txt = tmp_path / "a.txt"
    txt.write_text("Story one.\n<|endoftext|>\nStory two.\n<|endoftext|>\n", encoding="utf-8")
    n = encode_file(str(txt), str(tmp_path / "a.bin"), tok)
    data = load_bin(str(tmp_path / "a.bin"))
    assert len(data) == n
    assert tok.decode(data.tolist(), skip_special_tokens=False) == "Story one.<|endoftext|>Story two.<|endoftext|>"

def test_get_batch():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert torch.equal(y, x + 1)   # on arange data, the target is always the next number