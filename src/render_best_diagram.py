"""Render outputs/diagrams/best_model.png from the winning architecture in
outputs/results/all_runs.json, using only matplotlib (no external draw.io
dependency).

The drawio XML in outputs/diagrams/best_architecture.drawio remains the
authoritative editable source; this script is a one-shot rasteriser for the
embedded README image so the figure tracks the actual experiment winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
OUT_PNG = ROOT / "outputs" / "diagrams" / "best_model.png"


# Per-architecture layer descriptors. Each layer is (text, fillcolor).
# Read top-to-bottom; arrows are drawn between adjacent entries.
LAYERS = {
    "lstm_small": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 64", "#FFF3E0"),
        ("LSTM\ninput=64, hidden=128", "#E8F5E9"),
        ("h_T = H[:, -1, :]   (B, 128)", "#F3E5F5"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 128 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "lstm_large": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("LSTM\ninput=128, hidden=256", "#E8F5E9"),
        ("h_T = H[:, -1, :]   (B, 256)", "#F3E5F5"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 256 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "lstm_stacked": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("LSTM layer 1\ninput=128, hidden=256", "#E8F5E9"),
        ("Inter-layer Dropout(p=0.4)", "#ECEFF1"),
        ("LSTM layer 2\ninput=256, hidden=256", "#E8F5E9"),
        ("h_T = H[:, -1, :]   (B, 256)", "#F3E5F5"),
        ("Dropout(p=0.4)", "#ECEFF1"),
        ("Linear: 256 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "bilstm": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("Bidirectional LSTM\ninput=128, hidden=256/direction\noutput H = (B, T, 512)", "#E8F5E9"),
        ("last timestep H[:, -1, :]   (B, 512)", "#F3E5F5"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 512 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "lstm_bahdanau": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("LSTM\ninput=128, hidden=256\noutput H = (B, T, 256)", "#E8F5E9"),
        ("Bahdanau additive attention\nq=h_T, k=v=H\ncontext = Σ α_t h_t   (B, 256)", "#FFEBEE"),
        ("concat([context ; h_T])   (B, 512)", "#ECEFF1"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 512 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "lstm_mhsa": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("LSTM\ninput=128, hidden=256", "#E8F5E9"),
        ("MultiheadAttention\nnum_heads=4, Q=K=V=H", "#FFEBEE"),
        ("LayerNorm(H + attn_out)", "#ECEFF1"),
        ("h_T = H[:, -1, :]   (B, 256)", "#F3E5F5"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 256 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "gru_bahdanau": [
        ("Input token ids\n(B, T=20)", "#E1F5FE"),
        ("Embedding\nvocab → 128", "#FFF3E0"),
        ("GRU\ninput=128, hidden=256\noutput H = (B, T, 256)", "#E8F5E9"),
        ("Bahdanau additive attention\nq=h_T, k=v=H\ncontext = Σ α_t h_t   (B, 256)", "#FFEBEE"),
        ("concat([context ; h_T])   (B, 512)", "#ECEFF1"),
        ("Dropout(p=0.3)", "#ECEFF1"),
        ("Linear: 512 → vocab", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "lstm_enhanced": [
        ("Input token ids\n(B, T=40)", "#E1F5FE"),
        ("Embedding\nvocab → 256\n(weights TIED with output FC)", "#FFF3E0"),
        ("Variational Dropout(p=0.3)\n(one mask shared across time)", "#ECEFF1"),
        ("LSTM layer 1\ninput=256, hidden=256", "#E8F5E9"),
        ("Inter-layer Dropout(p=0.4)", "#ECEFF1"),
        ("LSTM layer 2\ninput=256, hidden=256", "#E8F5E9"),
        ("Variational Dropout(p=0.4)", "#ECEFF1"),
        ("h_T = H[:, -1, :]   (B, 256)", "#F3E5F5"),
        ("Linear: 256 → vocab\n(weights SHARED with embedding)", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
    "awd_lstm": [
        ("Input token ids\n(B, T=40)", "#E1F5FE"),
        ("Embedding Dropout(p=0.1)\n(zero whole vocabulary rows)", "#ECEFF1"),
        ("Embedding\nvocab → 256\n(weights TIED with output FC)", "#FFF3E0"),
        ("Variational Dropout(p=0.4)\n(one mask shared across time)", "#ECEFF1"),
        ("Weight-Dropped LSTM layer 1\nDropConnect(p=0.5) on weight_hh_l0", "#E8F5E9"),
        ("Inter-layer Dropout(p=0.3)", "#ECEFF1"),
        ("Weight-Dropped LSTM layer 2\nDropConnect(p=0.5) on weight_hh_l1", "#E8F5E9"),
        ("Variational Dropout(p=0.4)", "#ECEFF1"),
        ("h_T = H[:, -1, :]   (B, 256)", "#F3E5F5"),
        ("Linear: 256 → vocab\n(weights SHARED with embedding)", "#FFF8E1"),
        ("softmax → top-k for next word", "#E1F5FE"),
    ],
}


def render(name: str, label: str, out_path: Path) -> None:
    if name not in LAYERS:
        raise SystemExit(f"No diagram template for model '{name}'")
    layers = LAYERS[name]

    box_w = 5.8
    box_h = 0.85
    gap = 0.45
    fig_h = (box_h + gap) * len(layers) + 1.0
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    title_y = fig_h - 0.35
    ax.text(3.5, title_y, label, ha="center", va="center", fontsize=13, fontweight="bold")

    y = fig_h - 1.2
    centers = []
    for text, fill in layers:
        x = (7 - box_w) / 2
        box = mpatches.FancyBboxPatch(
            (x, y - box_h), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2, edgecolor="#37474F", facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y - box_h / 2, text,
                ha="center", va="center", fontsize=9)
        centers.append((x + box_w / 2, y - box_h / 2))
        y -= box_h + gap

    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(x2, y2 + box_h / 2 + 0.02),
                    xytext=(x1, y1 - box_h / 2 - 0.02),
                    arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.1))

    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    runs = json.loads((RESULTS / "all_runs.json").read_text())
    runs.sort(key=lambda r: r["test"]["perplexity"])
    best = runs[0]
    render(best["name"], best["label"], OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
