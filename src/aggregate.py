"""Rebuild outputs/results/all_runs.json and summary_table.md from the
per-experiment JSON files currently on disk.

Useful when you've trained additional experiments separately (`--only`) and
want the aggregate views to include them without re-training everything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
sys.path.insert(0, str(ROOT / "src"))

import argparse

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from run_experiments import EXPERIMENTS, _plot_comparison, _plot_history  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-plots", action="store_true",
                   help="Skip regenerating comparison.png / best_detail.png.")
    args = p.parse_args(argv)

    runs = []
    # Iterate in the canonical EXPERIMENTS order so the comparison plot legend
    # stays stable across reruns.
    for cfg in EXPERIMENTS:
        path = RESULTS / f"{cfg['name']}.json"
        if not path.exists():
            print(f"  (skipping {cfg['name']} — no JSON on disk)")
            continue
        runs.append(json.loads(path.read_text()))

    runs.sort(key=lambda r: r["test"]["perplexity"])

    (RESULTS / "all_runs.json").write_text(json.dumps(runs, indent=2))

    table = [
        "| Rank | Experiment | Params | Train s | Val acc | Test acc | Test top-5 | Test PPL | Inf ms/tok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(runs, 1):
        table.append(
            f"| {i} | {r['label']} | {r['parameters']:,} | {r['train_seconds']:.0f} | "
            f"{r['val']['accuracy']:.3f} | {r['test']['accuracy']:.3f} | {r['test']['top5_accuracy']:.3f} | "
            f"{r['test']['perplexity']:.2f} | {r['inference_profile']['ms_per_token']:.2f} |"
        )
    (RESULTS / "summary_table.md").write_text("\n".join(table))

    if not args.no_plots and runs:
        plots = ROOT / "outputs" / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        _plot_comparison(runs, plots / "comparison.png")
        # Best-detail plot needs the EpochStats-shaped history; argparse.Namespace
        # works as a duck-typed stand-in (`.train_loss`, `.val_loss`, etc).
        best = runs[0]
        ns_history = [argparse.Namespace(**h) for h in best["history"]]
        _plot_history(ns_history, plots / "best_detail.png", title=f"Best: {best['label']}")

    print(f"Aggregated {len(runs)} experiments:")
    for r in runs:
        print(f"  {r['label']:<70} ppl={r['test']['perplexity']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
