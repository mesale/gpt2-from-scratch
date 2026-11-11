# GPT-2 from Scratch: a small decoder-only Transformer for story generation

A ~12M-parameter GPT-2-style language model implemented from scratch in PyTorch (attention, multi-head attention,
feed-forward network, layer norm, residual connections, Transformer blocks, training loop, sampling), trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (V2, GPT-4 generated) with a byte-level BPE
tokenizer (vocab 4096). No pretrained models, Hugging Face model code, or existing Transformer blocks are used; the
only library pieces are tensors, autograd, optimizers, and basic layers (`nn.Linear`, `nn.Embedding`, `nn.Dropout`, `nn.GELU`).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mesale/gpt2-from-scratch/blob/main/notebooks/demo.ipynb)

**Technical report:** [`report/report.md`](report/report.md)  
**Demo notebook** (runs the tests, loads the model, generates text, shows results and attention maps, and trains a small model from scratch; no local setup needed): [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

## Example output
Prompt: `The little dog was sad because` (greedy decoding, `models/tinystories_main_best.pt`):

> The little dog was sad because he had no friends. He walked around the park, looking for someone to play with. He saw a big tree and thought, "Maybe I can find a friend there."
> The little dog walked to the tree and saw a little girl. She was playing with a ball. The little dog walked up to her and said, "Hi, I am a dog. Do you want to play with me?" The little girl smiled and said, "Yes, I am Lily. I am a dog, but I have no friends."
> The little dog was happy to have a new friend. They played together all day. The little dog was not sad anymore. He had a new friend, and they had lots of fun.

Grammar, dialogue, and story structure are good for a 12M-parameter model, but there are logic slips (the girl says
"I am a dog"). More samples: [`results/tinystories_main/samples.txt`](results/tinystories_main/samples.txt),
[`results/analysis/report_samples.txt`](results/analysis/report_samples.txt).

## Model
| | |
|---|---|
| Layers / heads / width | 6 / 6 / 384 (head size 64) |
| Context length | 256 tokens |
| Vocabulary | 4096 (byte-level BPE trained on a ~200 MB TinyStories sample) |
| Parameters | 12,318,720 (output head tied to the token embedding) |
| Architecture | GPT-2 style: learned positional embeddings, pre-LayerNorm, GELU, final LayerNorm |
| Initialization | N(0, 0.02); residual projections scaled by 1/sqrt(2 * n_layer) |

## Results at a glance
All numbers are cross-entropy in nats per token. Each run's logs, config, and curves are in `results/<run>/`.

**Main run** (8,000 steps x 32,768 tokens = 262M tokens, about one hour on one Kaggle GPU):
validation loss **1.372** (perplexity about 3.9), training loss 1.363 (no overfitting gap).
Curve: [`results/tinystories_main/loss_curves.png`](results/tinystories_main/loss_curves.png).

**Debug run on Tiny Shakespeare** (405k training tokens): validation loss bottoms out at 3.63 around step 1200,
then rises to 3.67 while training loss keeps falling to 2.10: clear overfitting, as expected for 12M parameters on
~0.4M tokens. Curve: [`results/shakespeare_debug/loss_curves.png`](results/shakespeare_debug/loss_curves.png).

**Number of attention heads** (1,500 steps, identical parameter count; `results/ablation_heads/`):

| heads | head size | validation loss | tokens/s |
|---|---|---|---|
| 1 | 384 | 1.720 | ~89.7k |
| 2 | 192 | 1.712 | ~81.9k |
| 6 | 64 | 1.703 (second seed: 1.698) | ~71k |
| 12 | 32 | 1.705 | ~59k |

The seed-to-seed gap at 6 heads is about 0.004, so 1 head is clearly worse, 2 heads is borderline, and 6 vs 12 is a tie.

**Pre-LN vs post-LN** (1,500 steps; `results/ablation_ln/`):

| | lr 1e-3, warmup 100 | lr 3e-4, warmup 300 |
|---|---|---|
| pre-LN | 1.703 | 1.973 |
| post-LN | 5.888 (stalled) | 2.458 |

**Context length** (loss on fixed target tokens given only the previous k tokens; `results/analysis/`):
k=1: 3.95, k=8: 2.36, k=32: 1.74, k=127: 1.43, full prefix: 1.41. Most of the benefit comes from the first 16-32 tokens.

**Temperature** (50 prompts x 4 samples x 200 new tokens; `results/analysis/`):

| temperature | repeated word 3-grams | distinct-2 (across samples) |
|---|---|---|
| 0 (greedy) | 7.4% | n/a |
| 0.3 | 7.0% | 0.67 |
| 0.7 | 3.9% | 0.78 |
| 1.0 | 1.8% | 0.86 |
| 1.5 | 0.1% | 0.99 (text becomes incoherent) |

## Repository layout
```
configs/      YAML run configs (shakespeare, tinystories_main, ablation_heads) and tokenizer.json
src/          config.py, tokenizer.py, data.py, prepare_data.py, model.py, train.py, checkpoint.py,
              generate.py, metrics.py, experiments.py, viz.py
tests/        pytest suite (shapes, causality, LayerNorm vs torch, attention vs torch SDPA, training, generation, ...)
models/       trained checkpoint (tinystories_main_best.pt)
results/      logs, configs, figures, and samples for every run
report/       technical report
notebooks/    demo.ipynb (demo + tests, runs on Colab/Kaggle) and the Kaggle notebooks used to launch the runs
```

## Setup
```bash
git clone https://github.com/mesale/gpt2-from-scratch.git
cd gpt2-from-scratch
pip install -r requirements.txt
python -m pytest -q
```

## Generate text with the included model (CPU is fine)
```bash
python -m src.generate --ckpt models/tinystories_main_best.pt --prompt "Once upon a time" --temperature 0.7 --n 3
python -m src.generate --ckpt models/tinystories_main_best.pt --prompt "The little dog was sad because" --temperature 0   # greedy
```
`--temperature 0` is greedy decoding; larger values sample more diversely. Generation stops at the end-of-story token or after `--max_new_tokens`.

## Reproduce the training
```bash
# 1) download TinyStories + Tiny Shakespeare and encode them to uint16 .bin files (about 2.2 GB download)
RAW_DIR=data/raw OUT_DIR=data python -m src.prepare_data

# 2) quick debug run on Shakespeare (minutes on a GPU)
python -m src.train --config configs/shakespeare.yaml --set train.data_dir=data train.out_dir=runs

# 3) main run (about an hour on a Kaggle-class GPU)
python -m src.train --config configs/tinystories_main.yaml --set train.data_dir=data train.out_dir=runs

# 4) ablations and analysis
python -m src.train --config configs/ablation_heads.yaml --set model.n_head=1 train.run_name=heads_1 train.data_dir=data train.out_dir=runs
python -m src.experiments --ckpt runs/tinystories_main/best.pt --data_dir data --out results/analysis --n_prompts 50 --n_samples 4 --max_new 200
```
Any config value can be overridden with `--set section.key=value`. Training is resumable: rerunning the same command
continues from `runs/<run_name>/ckpt.pt`. Mixed precision (fp16) is on by default on GPU; disable it with `--set train.amp=false`.

The runs in this repository were produced on Kaggle (one GPU, `Nvidia T4 x2`), launched from the notebooks in `notebooks/`.

## Limitations
- The model writes locally coherent stories but loses track of who is who and drifts in plot logic.
- Ablations are short (1,500 steps) with one seed per setting (two for 6 heads), so small differences are within noise.
- The context experiment re-numbers positions for truncated windows (learned absolute positions), which slightly confounds it.
- Generation recomputes the whole context at every step (no KV cache).

## References
- Vaswani et al. (2017). *Attention Is All You Need.* https://arxiv.org/abs/1706.03762
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Eldan and Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Karpathy (2022). *Let's build GPT: from scratch, in code, spelled out.*
- Alammar (2018). *The Illustrated Transformer.* https://jalammar.github.io/illustrated-transformer/