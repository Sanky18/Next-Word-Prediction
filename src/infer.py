"""Load the saved best model and generate text from a seed phrase.

This is the script you want for *using* a trained checkpoint without retraining.
It reads:
  - outputs/best_model.pt     (weights + architecture name; saved by rerun_best.py)
  - outputs/results/vocab.json (the token ↔ id mapping; saved by run_experiments.py)

…rebuilds the same model, restores the weights, and runs `generate_text` on the
seed phrase you provide. Both greedy and top-k sampled decoding are exposed.

Usage:
    # Greedy (default)
    python src/infer.py "i saw holmes"

    # Top-k sampled with a temperature knob
    python src/infer.py "watson opened the door" --sample --temperature 0.9

    # Customise length and show the per-step top-5 trace
    python src/infer.py "it was a cold morning" --length 60 --show-topk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import Vocab  # noqa: E402
from models import build_model  # noqa: E402
from train import generate_text, pick_device  # noqa: E402


DEFAULT_CKPT = ROOT / "outputs" / "best_model.pt"
DEFAULT_VOCAB = ROOT / "outputs" / "results" / "vocab.json"
DEFAULT_SEQ_LEN = 20


def load_model_and_vocab(
    ckpt_path: Path = DEFAULT_CKPT,
    vocab_path: Path = DEFAULT_VOCAB,
    device: torch.device | None = None,
):
    if not ckpt_path.exists():
        raise SystemExit(
            f"No checkpoint at {ckpt_path}. Run `python src/rerun_best.py` first "
            f"(it trains the winning architecture and saves best_model.pt)."
        )
    if not vocab_path.exists():
        raise SystemExit(
            f"No vocab at {vocab_path}. Run `python src/run_experiments.py` first "
            f"(it persists the train-only vocabulary)."
        )
    device = device or pick_device()
    vocab = Vocab.load(vocab_path)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(blob["name"], vocab_size=len(vocab)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, vocab, blob.get("label", blob["name"]), device


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("seed", help="Seed phrase to continue, e.g. 'i saw holmes'")
    p.add_argument("--length", type=int, default=40, help="Number of tokens to generate (default: 40).")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                   help="Context window length the model was trained on (default: 20).")
    p.add_argument("--sample", action="store_true",
                   help="Use top-k multinomial sampling instead of greedy argmax.")
    p.add_argument("--top-k", type=int, default=5, help="Top-k restriction for both display and sampling.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--show-topk", action="store_true",
                   help="Print the top-k candidates and probabilities at every step.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    args = p.parse_args(argv)

    model, vocab, label, device = load_model_and_vocab(
        ckpt_path=Path(args.ckpt), vocab_path=Path(args.vocab),
    )
    print(f"Loaded {label}  (vocab={len(vocab):,}, device={device})")

    text, steps = generate_text(
        args.seed, args.length, model, vocab,
        seq_len=args.seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        sample=args.sample,
    )

    print(f"\nSeed:    {args.seed!r}")
    print(f"Mode:    {'top-k sample' if args.sample else 'greedy'}  "
          f"(k={args.top_k}, T={args.temperature})")
    print(f"\n{text}\n")

    if args.show_topk:
        print("Per-step top-k:")
        print(f"{'step':>4}  {'chosen':<14}  top-k (word: prob)")
        for st in steps:
            top_str = ", ".join(f"{w}:{prob:.3f}" for w, prob in st["top5"])
            print(f"{st['step']:>4}  {st['chosen']:<14}  {top_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
