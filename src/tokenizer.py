from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

EOT = "<|endoftext|>"

def make_sample(src, dst, n_chars=200 * 1024**2):
    """Copy the first ~200M characters of the corpus, cut at a story boundary."""
    with open(src, "r", encoding="utf-8") as f:
        text = f.read(n_chars)
    text = text[: text.rfind(EOT)]
    # Replace the marker with a newline so BPE doesn't waste vocab slots on its characters.
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text.replace(EOT, "\n"))

def train_bpe(files, vocab_size=4096, save_path="configs/tokenizer.json"):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],                                  # added as one atomic token
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
    )
    tok.train(files, trainer)
    tok.save(save_path)
    return tok

def load_tokenizer(path="configs/tokenizer.json"):
    return Tokenizer.from_file(path)