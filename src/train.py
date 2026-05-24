"""Training, evaluation, perplexity, and text generation utilities.

All functions are model-agnostic — they only depend on the (B, T) → (B, V) contract.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import NextWordDataset, UNK, Vocab


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_ppl: float
    seconds: float


@dataclass
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    total_train_seconds: float = 0.0
    best_val_loss: float = math.inf
    best_state_dict: dict | None = None


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one pass over `loader`. `optimizer=None` ⇒ evaluation mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Modest gradient clipping — RNNs occasionally produce large spikes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(-1) == y).sum().item()
            total += bs
    return total_loss / total, total_correct / total


def train_model(
    model: nn.Module,
    train_ds: NextWordDataset,
    val_ds: NextWordDataset,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
    log_fn: Callable[[str], None] = print,
    ignore_index: int | None = None,
) -> TrainResult:
    device = device or pick_device()
    model.to(device)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    if ignore_index is not None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    result = TrainResult()
    t0_all = time.perf_counter()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = _epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _epoch(model, val_loader, device, criterion, None)
        scheduler.step(val_loss)
        secs = time.perf_counter() - t0
        ppl = math.exp(min(val_loss, 20.0))
        stats = EpochStats(
            epoch=ep,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_ppl=ppl,
            seconds=secs,
        )
        result.history.append(stats)
        log_fn(
            f"  ep {ep:>2}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={ppl:.2f}  ({secs:.1f}s)"
        )
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            # Keep a CPU copy of the best weights so we can later restore without
            # blowing up MPS / CUDA memory.
            result.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result.total_train_seconds = time.perf_counter() - t0_all
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    ds: NextWordDataset,
    *,
    batch_size: int = 256,
    device: torch.device | None = None,
    ignore_index: int | None = None,
) -> dict:
    device = device or pick_device()
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if ignore_index is not None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        pred = logits.argmax(-1)
        total_correct += (pred == y).sum().item()
        top5 = logits.topk(5, dim=-1).indices
        total_top5 += (top5 == y.unsqueeze(-1)).any(-1).sum().item()
        total += y.size(0)
    avg_loss = total_loss / total
    return {
        "loss": avg_loss,
        "accuracy": total_correct / total,
        "top5_accuracy": total_top5 / total,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "n": total,
    }


# ---------------------------------------------------------------------------
# Text generation with top-5 breakdown.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(
    seed_text: str,
    max_phrase_len: int,
    model: nn.Module,
    vocab: Vocab,
    seq_len: int = 20,
    temperature: float = 0.8,
    top_k: int = 5,
    device: torch.device | None = None,
    sample: bool = False,
) -> tuple[str, list[dict]]:
    """Greedy (or top-k sampled) generation.

    For each predicted word we also return the top-`top_k` candidate words and their
    softmax probabilities, so the caller can introspect the model's decision at each
    step (assignment requirement 5f).

    `sample=False` → greedy argmax (deterministic, easier to interpret).
    `sample=True`  → multinomial sample restricted to top_k logits.
    """
    device = device or pick_device()
    model.to(device).eval()

    from data import normalize_text, tokenize as _tok  # local import: avoid cycles
    seed_tokens = _tok(normalize_text(seed_text))
    if not seed_tokens:
        raise ValueError("seed_text became empty after normalization")
    generated_tokens = list(seed_tokens)
    steps: list[dict] = []

    for _ in range(max_phrase_len):
        context = generated_tokens[-seq_len:]
        # Left-pad if the context is shorter than seq_len so the model sees a fixed window.
        if len(context) < seq_len:
            context = [vocab.itos[0]] * (seq_len - len(context)) + context  # <pad>
        ids = torch.tensor(vocab.encode(context), dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ids).squeeze(0)
        # Block <pad> and <unk> from being generated — they're never useful output.
        logits[0] = float("-inf")
        logits[1] = float("-inf")
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        top_words = [vocab.itos[i] for i in top_ids.tolist()]
        top_pairs = list(zip(top_words, [float(p) for p in top_probs.tolist()]))

        if sample:
            choice = torch.multinomial(top_probs, num_samples=1).item()
            next_id = top_ids[choice].item()
        else:
            next_id = top_ids[0].item()
        next_word = vocab.itos[next_id]
        generated_tokens.append(next_word)
        steps.append({
            "step": len(steps) + 1,
            "context_tail": " ".join(context[-6:]),
            "chosen": next_word,
            "top5": top_pairs,
        })

    output_text = " ".join(generated_tokens)
    return output_text, steps


# ---------------------------------------------------------------------------
# Inference profiling.
# ---------------------------------------------------------------------------
@torch.no_grad()
def profile_inference(
    model: nn.Module,
    vocab: Vocab,
    *,
    seed_text: str = "i saw holmes",
    gen_len: int = 30,
    seq_len: int = 20,
    n_warmup: int = 2,
    n_repeats: int = 5,
    device: torch.device | None = None,
) -> dict:
    """Wall-clock time for one full generate_text call. Reports median + IQR."""
    device = device or pick_device()
    # Warm-up — first MPS/CUDA call includes kernel compilation; throw it out.
    for _ in range(n_warmup):
        generate_text(seed_text, gen_len, model, vocab, seq_len=seq_len, device=device)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        generate_text(seed_text, gen_len, model, vocab, seq_len=seq_len, device=device)
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "seed_text": seed_text,
        "gen_len": gen_len,
        "n_repeats": n_repeats,
        "median_sec": float(np.median(arr)),
        "mean_sec": float(arr.mean()),
        "p25_sec": float(np.percentile(arr, 25)),
        "p75_sec": float(np.percentile(arr, 75)),
        "ms_per_token": float(np.median(arr) / gen_len * 1000.0),
    }
