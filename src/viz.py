"""Attention inspection helpers (used by notebooks/demo.ipynb)."""
import copy

import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def attention_maps(model, idx, layer, causal=True):
    """Attention weights of one layer for token ids idx of shape (1, T). Returns (n_head, T, T).

    causal=False runs the SAME trained weights with the mask removed (an intervention: the model
    was never trained that way), which shows what the causal mask forbids."""
    model.eval()
    x = model.embed(idx)
    for block in model.blocks[:layer]:
        x = block(x)
    block = model.blocks[layer]
    attn = block.attn
    if attn.causal != causal:
        attn = copy.deepcopy(attn)
        attn.causal = causal
    h = block.ln1(x) if block.pre_ln else x
    _, att = attn(h, return_attn=True)
    return att[0]


def plot_attention(labels, att_masked, att_unmasked, heads=(0, 1, 2), save_path=None):
    """Grid of heatmaps: one row per head, left = masked (causal), right = unmasked."""
    fig, axes = plt.subplots(len(heads), 2, figsize=(11, 4.8 * len(heads)), squeeze=False)
    for r, h in enumerate(heads):
        for c, (att, name) in enumerate([(att_masked, "masked (causal)"), (att_unmasked, "unmasked")]):
            ax = axes[r, c]
            ax.imshow(att[h].cpu().numpy(), cmap="viridis", vmin=0)
            ax.set_title(f"head {h}, {name}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("key (token being attended to)")
            ax.set_ylabel("query (current token)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig