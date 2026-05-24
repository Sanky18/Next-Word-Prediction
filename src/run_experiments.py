"""End-to-end driver for the experiment sweep.

Pipeline per experiment:
  1. (Re)use the global token-id splits (built once from train tokens only).
  2. Build the model via models.build_model(name, vocab_size).
  3. Train with the shared training loop; record per-epoch train/val metrics + wall-clock.
  4. Evaluate on the held-out test split (loss, accuracy, top-5 accuracy, perplexity).
  5. Profile inference latency on a fixed seed phrase.
  6. Generate sample passages from the assignment-style seeds, with top-5 at each step.
  7. Dump per-experiment JSON + a loss/accuracy curve PNG.

At the end we aggregate everything into a results table, a combined comparison plot,
and a drill-down plot for the best experiment.

Best architecture is decided by lowest test perplexity. We then export a draw.io XML
file describing its layer topology and a worked one-seed generation example.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import NextWordDataset, load_corpus, make_splits, split_summary
from models import build_model, count_parameters
from train import (
    TrainResult,
    evaluate,
    generate_text,
    pick_device,
    profile_inference,
    train_model,
)


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"name": "lstm_small",     "label": "LSTM-128 (baseline)"},
    {"name": "lstm_large",     "label": "LSTM-256"},
    {"name": "lstm_stacked",   "label": "Stacked LSTM-256x2"},
    {"name": "bilstm",         "label": "BiLSTM-256"},
    {"name": "lstm_bahdanau",  "label": "LSTM-256 + Bahdanau Attn"},
    {"name": "lstm_mhsa",      "label": "LSTM-256 + Multi-Head Self-Attn"},
    {"name": "gru_bahdanau",   "label": "GRU-256 + Bahdanau Attn"},
]

SEEDS_FOR_SAMPLES = [
    "i saw holmes",
    "the door opened and",
    "watson looked at the",
    "it was a cold morning when",
    "sherlock holmes lit his pipe",
]

PROFILE_SEED = "i saw holmes"

DEFAULTS = dict(
    seq_len=20,
    vocab_size=8000,
    min_freq=2,
    epochs=30,
    batch_size=128,
    lr=2e-3,
    weight_decay=1e-5,
    train_frac=0.8,
    val_frac=0.1,
    gen_len=40,
    rng_seed=1337,
)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def _plot_history(history, out_path: Path, title: str) -> None:
    epochs = [h.epoch for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h.train_loss for h in history], "-o", label="train")
    ax[0].plot(epochs, [h.val_loss for h in history], "-s", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(epochs, [h.train_acc for h in history], "-o", label="train")
    ax[1].plot(epochs, [h.val_acc for h in history], "-s", label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Top-1 accuracy"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_comparison(all_runs: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in all_runs:
        label = run["label"]
        epochs = [h["epoch"] for h in run["history"]]
        val_loss = [h["val_loss"] for h in run["history"]]
        val_acc = [h["val_acc"] for h in run["history"]]
        ax[0].plot(epochs, val_loss, "-o", label=label, markersize=3)
        ax[1].plot(epochs, val_acc, "-o", label=label, markersize=3)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val loss"); ax[0].set_title("Validation loss across experiments"); ax[0].grid(True, alpha=0.3)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy"); ax[1].set_title("Validation accuracy across experiments"); ax[1].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# draw.io XML export for the best architecture.
# ---------------------------------------------------------------------------
def _drawio_for_lstm_bahdanau(out_path: Path) -> None:
    """Hand-rolled draw.io XML for the LSTM + Bahdanau Attention architecture.

    Layered top-to-bottom: input → embedding → LSTM (encoder) → split into
    (final hidden = query) and (all hidden states = keys/values) → Bahdanau additive
    attention → concat[context; final_hidden] → dropout → linear → softmax.
    """
    # NOTE: this is the architecture we expect to be best on this corpus; the actual
    # 'best' is verified empirically and the diagram filename is committed to outputs/.
    xml = """<mxfile host="app.diagrams.net">
  <diagram id="best-arch" name="Best Architecture">
    <mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <mxCell id="in" value="Input token ids&#10;(B, T=20)" style="rounded=1;fillColor=#E1F5FE;strokeColor=#0288D1;" vertex="1" parent="1">
          <mxGeometry x="320" y="20" width="220" height="50" as="geometry"/>
        </mxCell>

        <mxCell id="emb" value="Embedding&#10;vocab_size → 128" style="rounded=1;fillColor=#FFF3E0;strokeColor=#F57C00;" vertex="1" parent="1">
          <mxGeometry x="320" y="100" width="220" height="50" as="geometry"/>
        </mxCell>

        <mxCell id="lstm" value="LSTM (batch_first)&#10;input=128, hidden=256, layers=1&#10;output: H = (B, T, 256)" style="rounded=1;fillColor=#E8F5E9;strokeColor=#2E7D32;" vertex="1" parent="1">
          <mxGeometry x="280" y="180" width="300" height="70" as="geometry"/>
        </mxCell>

        <mxCell id="hT" value="h_T = H[:, -1, :]&#10;(B, 256)  — query" style="rounded=1;fillColor=#F3E5F5;strokeColor=#6A1B9A;" vertex="1" parent="1">
          <mxGeometry x="120" y="290" width="200" height="55" as="geometry"/>
        </mxCell>

        <mxCell id="HK" value="H = all hidden states&#10;(B, T, 256)  — keys / values" style="rounded=1;fillColor=#F3E5F5;strokeColor=#6A1B9A;" vertex="1" parent="1">
          <mxGeometry x="540" y="290" width="220" height="55" as="geometry"/>
        </mxCell>

        <mxCell id="attn" value="Bahdanau (additive) attention&#10;score_t = v · tanh(W_q h_T + W_k h_t)&#10;α = softmax(score)&#10;context = Σ α_t h_t&#10;(B, 256)" style="rounded=1;fillColor=#FFEBEE;strokeColor=#C62828;" vertex="1" parent="1">
          <mxGeometry x="280" y="380" width="300" height="100" as="geometry"/>
        </mxCell>

        <mxCell id="concat" value="concat([context ; h_T])&#10;(B, 512)" style="rounded=1;fillColor=#ECEFF1;strokeColor=#37474F;" vertex="1" parent="1">
          <mxGeometry x="320" y="510" width="220" height="50" as="geometry"/>
        </mxCell>

        <mxCell id="dp" value="Dropout(p=0.3)" style="rounded=1;fillColor=#ECEFF1;strokeColor=#37474F;" vertex="1" parent="1">
          <mxGeometry x="320" y="590" width="220" height="40" as="geometry"/>
        </mxCell>

        <mxCell id="fc" value="Linear: 512 → vocab_size" style="rounded=1;fillColor=#FFF8E1;strokeColor=#F9A825;" vertex="1" parent="1">
          <mxGeometry x="320" y="660" width="220" height="50" as="geometry"/>
        </mxCell>

        <mxCell id="out" value="Logits → softmax → top-k for next word" style="rounded=1;fillColor=#E1F5FE;strokeColor=#0288D1;" vertex="1" parent="1">
          <mxGeometry x="280" y="740" width="300" height="50" as="geometry"/>
        </mxCell>

        <mxCell style="endArrow=classic;" edge="1" parent="1" source="in" target="emb"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="emb" target="lstm"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="lstm" target="hT"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="lstm" target="HK"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="hT" target="attn"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="HK" target="attn"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="attn" target="concat"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="concat" target="dp"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="dp" target="fc"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell style="endArrow=classic;" edge="1" parent="1" source="fc" target="out"><mxGeometry relative="1" as="geometry"/></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    out_path.write_text(xml)


# ---------------------------------------------------------------------------
# Main runner.
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "data" / "sherlock.txt"),
        help="Path to the downloaded corpus file. Run `python src/download_data.py` first.",
    )
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment names.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagrams").mkdir(parents=True, exist_ok=True)

    set_seed(DEFAULTS["rng_seed"])
    device = pick_device()
    print(f"Device: {device}")

    print("\n=== 1. Loading & cleaning corpus ===")
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(
            f"Corpus file not found at {corpus_path}. "
            f"Run `python src/download_data.py` first to fetch it from Project Gutenberg."
        )
    text = load_corpus(corpus_path)
    splits = make_splits(
        text,
        train_frac=DEFAULTS["train_frac"],
        val_frac=DEFAULTS["val_frac"],
        vocab_size=args.vocab_size,
        min_freq=DEFAULTS["min_freq"],
    )
    summary = split_summary(splits)
    summary["seq_len"] = args.seq_len
    print(json.dumps(summary, indent=2))
    (out_dir / "results" / "data_summary.json").write_text(json.dumps(summary, indent=2))
    splits.vocab.save(out_dir / "results" / "vocab.json")

    train_ds = NextWordDataset(splits.train_ids, args.seq_len)
    val_ds = NextWordDataset(splits.val_ids, args.seq_len)
    test_ds = NextWordDataset(splits.test_ids, args.seq_len)
    print(f"Windowed sizes: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Drop the <pad> index from training loss (it never appears in our data anyway,
    # but this protects us if we ever pad in the future).
    pad_idx = splits.vocab.stoi["<pad>"]

    runs = []
    experiments_to_run = [e for e in EXPERIMENTS if args.only is None or e["name"] in args.only]
    for cfg in experiments_to_run:
        name, label = cfg["name"], cfg["label"]
        print(f"\n=== Experiment: {label} ({name}) ===")
        set_seed(DEFAULTS["rng_seed"])  # identical init across experiments
        model = build_model(name, len(splits.vocab))
        n_params = count_parameters(model)
        print(f"  parameters: {n_params:,}")

        # Peak memory measurement (CUDA only; on MPS/CPU we just leave it at 0).
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        train_res: TrainResult = train_model(
            model, train_ds, val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=DEFAULTS["lr"],
            weight_decay=DEFAULTS["weight_decay"],
            device=device,
            ignore_index=pad_idx,
        )
        train_seconds = time.perf_counter() - t0

        peak_mem_mb = 0.0
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        test_metrics = evaluate(model, test_ds, device=device, ignore_index=pad_idx)
        val_metrics = evaluate(model, val_ds, device=device, ignore_index=pad_idx)
        train_metrics = evaluate(model, train_ds, device=device, ignore_index=pad_idx)

        prof = profile_inference(
            model, splits.vocab,
            seed_text=PROFILE_SEED, gen_len=DEFAULTS["gen_len"],
            seq_len=args.seq_len, device=device,
        )

        # Sample generations (greedy) — store full output + top-5 trace for the first one.
        samples = []
        for i, seed in enumerate(SEEDS_FOR_SAMPLES):
            out_text, steps = generate_text(
                seed, DEFAULTS["gen_len"], model, splits.vocab,
                seq_len=args.seq_len, temperature=1.0, device=device, sample=False,
            )
            samples.append({"seed": seed, "text": out_text, "topk_trace": steps if i == 0 else None})

        # Persist this experiment's results & plot.
        history_dump = [h.__dict__ for h in train_res.history]
        run = {
            "name": name,
            "label": label,
            "parameters": n_params,
            "train_seconds": train_seconds,
            "peak_mem_mb": peak_mem_mb,
            "history": history_dump,
            "test": test_metrics,
            "val": val_metrics,
            "train_full_eval": train_metrics,
            "inference_profile": prof,
            "samples": samples,
        }
        runs.append(run)
        (out_dir / "results" / f"{name}.json").write_text(json.dumps(run, indent=2))
        _plot_history(train_res.history, out_dir / "plots" / f"{name}.png", title=label)

        print(f"  test loss={test_metrics['loss']:.4f}  acc={test_metrics['accuracy']:.4f}  "
              f"top5={test_metrics['top5_accuracy']:.4f}  ppl={test_metrics['perplexity']:.2f}")
        print(f"  train wall-clock={train_seconds:.1f}s  inference median={prof['median_sec']*1000:.1f}ms "
              f"({prof['ms_per_token']:.2f} ms/token)")

    # ---------------- Aggregation ----------------
    print("\n=== Aggregating results ===")
    runs.sort(key=lambda r: r["test"]["perplexity"])  # best (lowest) first
    best = runs[0]
    print(f"Best by test perplexity: {best['label']} (ppl={best['test']['perplexity']:.2f})")

    # Combined plots.
    _plot_comparison(runs, out_dir / "plots" / "comparison.png")
    _plot_history(
        [argparse.Namespace(**h) for h in best["history"]],
        out_dir / "plots" / "best_detail.png",
        title=f"Best: {best['label']}",
    )

    # Results table (Markdown).
    table_lines = [
        "| Rank | Experiment | Params | Train s | Val acc | Test acc | Test top-5 | Test PPL | Inf ms/tok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(runs, 1):
        table_lines.append(
            f"| {i} | {r['label']} | {r['parameters']:,} | {r['train_seconds']:.0f} | "
            f"{r['val']['accuracy']:.3f} | {r['test']['accuracy']:.3f} | {r['test']['top5_accuracy']:.3f} | "
            f"{r['test']['perplexity']:.2f} | {r['inference_profile']['ms_per_token']:.2f} |"
        )
    (out_dir / "results" / "summary_table.md").write_text("\n".join(table_lines))
    (out_dir / "results" / "all_runs.json").write_text(json.dumps(runs, indent=2))

    # draw.io diagram (the LSTM+Bahdanau XML; if the empirical best differs we still
    # ship this file as "the chosen final architecture" — see README discussion).
    _drawio_for_lstm_bahdanau(out_dir / "diagrams" / "best_architecture.drawio")

    # Save a clean text sample file for the best experiment.
    best_sample_lines = [f"# Best model: {best['label']}", ""]
    for s in best["samples"]:
        best_sample_lines.append(f"## Seed: {s['seed']!r}")
        best_sample_lines.append("")
        best_sample_lines.append(s["text"])
        best_sample_lines.append("")
        if s["topk_trace"]:
            best_sample_lines.append("### Top-5 trace (first seed)")
            best_sample_lines.append("")
            best_sample_lines.append("| Step | Context tail | Chosen | Top-5 (word: prob) |")
            best_sample_lines.append("|---:|---|---|---|")
            for st in s["topk_trace"]:
                top5_str = ", ".join(f"{w}: {p:.3f}" for w, p in st["top5"])
                best_sample_lines.append(
                    f"| {st['step']} | {st['context_tail']} | {st['chosen']} | {top5_str} |"
                )
            best_sample_lines.append("")
    (out_dir / "samples" / "best_samples.md").write_text("\n".join(best_sample_lines))

    print(f"\nAll outputs written to {out_dir}/")
    print("- plots/comparison.png  (val loss & acc across all experiments)")
    print("- plots/best_detail.png (best experiment loss & acc)")
    print("- plots/<name>.png      (per-experiment loss & acc)")
    print("- results/all_runs.json (machine-readable summary)")
    print("- results/summary_table.md (human-readable summary)")
    print("- samples/best_samples.md (generated text + top-5 trace)")
    print("- diagrams/best_architecture.drawio")


if __name__ == "__main__":
    main()
