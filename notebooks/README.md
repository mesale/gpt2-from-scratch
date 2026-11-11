# Notebooks

These are the Kaggle notebooks used to produce the results, kept as a record of how each run was launched.
They use Kaggle paths (`/kaggle/...`), Kaggle Secrets, and an attached dataset, so they are not meant to be
rerun elsewhere. To reproduce the results, use the commands in the top-level README, or run `demo.ipynb`.

| Notebook | What it does |
|---|---|
| `data-prep.ipynb` | Downloads TinyStories and Tiny Shakespeare, encodes them to `.bin` files |
| `tinystories-main.ipynb` | Main TinyStories run (8,000 steps) |
| `tinystories-ablations.ipynb` | Attention-head ablation (1, 2, 6, 12 heads) and a seed-noise run |
| `tinystories-ln.ipynb` | Pre-LN vs post-LN comparison |
| `demo.ipynb` | Runs the tests, loads the model, generates text, shows results and attention maps, trains a small model |