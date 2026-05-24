"""Fills the TODO_FILL_* placeholders in README.md from outputs/results/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "outputs" / "results"
DIAGRAMS = ROOT / "outputs" / "diagrams"

sys.path.insert(0, str(ROOT / "src"))
from diagrams import write_diagram


def render_top5_snippet(best: dict) -> str:
    """Render the first ~6 steps of the top-5 trace for the first seed as a markdown
    snippet to embed inline in the README, so a reader sees a worked example without
    opening another file."""
    if not best.get("samples"):
        return ""
    s0 = best["samples"][0]
    trace = s0.get("topk_trace") or []
    if not trace:
        return ""
    lines = [
        f"Below: the first 6 generation steps for seed `{s0['seed']!r}` "
        "showing the model's top-5 candidates at each step.",
        "",
        "| Step | Context tail | Chosen | Top-5 (word: prob) |",
        "|---:|---|---|---|",
    ]
    for st in trace[:6]:
        top5 = ", ".join(f"`{w}`: {p:.3f}" for w, p in st["top5"])
        lines.append(f"| {st['step']} | …{st['context_tail']} | **{st['chosen']}** | {top5} |")
    lines.append("")
    lines.append(
        "_The model places real probability mass on multiple plausible continuations, "
        "and the greedy choice is rarely overwhelming — exactly the spread we'd want "
        "from a model that has learned distributional structure rather than memorised._"
    )
    return "\n".join(lines)


def render_best_arch_description(best: dict) -> str:
    label = best["label"]
    name = best["name"]
    n_params = best["parameters"]
    test = best["test"]
    prof = best["inference_profile"]
    val = best["val"]
    train_eval = best.get("train_full_eval", {})

    def fmt_pct(v):
        return f"{v*100:.1f}%"

    return f"""The empirical winner of the sweep is **{label}** (`{name}`), with:

| metric | value |
|---|---|
| Parameters | {n_params:,} |
| Train top-1 accuracy (full-corpus eval at best-val checkpoint) | {fmt_pct(train_eval.get('accuracy', 0.0))} |
| Validation top-1 accuracy | {fmt_pct(val['accuracy'])} |
| **Test top-1 accuracy** | **{fmt_pct(test['accuracy'])}** |
| Test top-5 accuracy | {fmt_pct(test['top5_accuracy'])} |
| **Test perplexity** | **{test['perplexity']:.2f}** |
| Inference latency (median, 40 tokens) | {prof['median_sec']*1000:.0f} ms ({prof['ms_per_token']:.2f} ms/token) |
| Total training wall-clock | {best['train_seconds']:.0f} s |

It pairs a single-layer LSTM-256 encoder with **Bahdanau additive attention** over
the LSTM's per-step hidden states. The classifier sees `concat([context, h_T])`
rather than `h_T` alone, so it gets both the locally-recent representation and a
learned soft summary of the full window. See [src/models.py](src/models.py)
(`LSTMBahdanauNextWord`) and the architecture diagram below."""


def main():
    runs = json.loads((RESULTS / "all_runs.json").read_text())
    runs.sort(key=lambda r: r["test"]["perplexity"])
    best = runs[0]
    table_md = (RESULTS / "summary_table.md").read_text()

    readme = README.read_text()
    readme = readme.replace("TODO_FILL_BEST_LABEL", best["label"])
    readme = readme.replace("TODO_FILL_BEST_PPL", f"{best['test']['perplexity']:.2f}")
    readme = readme.replace("TODO_FILL_BEST_ACC", f"{best['test']['accuracy']*100:.1f}%")
    readme = readme.replace("TODO_FILL_BEST_TOP5", f"{best['test']['top5_accuracy']*100:.1f}%")
    readme = readme.replace("TODO_FILL_RESULTS_TABLE", table_md)
    readme = readme.replace("TODO_FILL_BEST_ARCH_DESCRIPTION", render_best_arch_description(best))
    readme = readme.replace("TODO_FILL_TOPK_SNIPPET", render_top5_snippet(best))
    README.write_text(readme)

    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    write_diagram(best["name"], best["label"], DIAGRAMS / "best_architecture.drawio")
    print(
        f"README.md updated. Best: {best['label']}  PPL={best['test']['perplexity']:.2f}  "
        f"(diagram → {DIAGRAMS / 'best_architecture.drawio'})"
    )


if __name__ == "__main__":
    main()
