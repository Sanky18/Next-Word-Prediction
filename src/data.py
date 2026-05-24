"""Data loading, cleaning, vocabulary construction, and sequence generation.

Critical invariant: train/val/test splits are made at the *raw token stream* level
BEFORE turning the streams into overlapping (context, target) windows. The vocabulary
is built from the TRAIN tokens only; val/test tokens not in the train vocab map to
<unk>. This prevents both (a) overlapping windows from leaking across the split and
(b) the vocabulary itself from being a function of held-out text.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------- RTF stripping (Project Gutenberg file shipped as .rtf) ----------

_RTF_GROUP_RE = re.compile(r"\\\*?\{[^{}]*\}")  # {\* ...} groups (fontTbl, colorTbl, etc.)
_RTF_CTRL_RE = re.compile(r"\\[a-zA-Z]+-?\d*\s?")  # \controlword or \controlword123
_RTF_HEX_RE = re.compile(r"\\'[0-9a-fA-F]{2}")
_RTF_BRACES_RE = re.compile(r"[{}]")


def strip_rtf(rtf_text: str) -> str:
    """Minimal RTF → plain text. Good enough for the Gutenberg eBook export."""
    txt = rtf_text
    # Drop font/color tables and similar \* groups (may nest one level — repeat).
    for _ in range(4):
        new = _RTF_GROUP_RE.sub("", txt)
        if new == txt:
            break
        txt = new
    txt = _RTF_HEX_RE.sub("", txt)
    txt = _RTF_CTRL_RE.sub(" ", txt)
    txt = _RTF_BRACES_RE.sub("", txt)
    txt = txt.replace("\\\n", "\n").replace("\\", "")
    return txt


# ---------- Gutenberg header/footer stripping & normalisation ----------

_GUT_START_RE = re.compile(
    r"(?:\*\*\*\s*)?START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*",
    re.IGNORECASE,
)
_GUT_END_RE = re.compile(
    r"(?:\*\*\*\s*)?END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*",
    re.IGNORECASE,
)


def clean_gutenberg(text: str) -> str:
    """Remove Project Gutenberg legal header/footer; keep only the novel body.

    The RTF-stripping step in this pipeline removes the literal asterisks, so the
    regexes accept the markers with or without the surrounding `***`.
    """
    start_m = _GUT_START_RE.search(text)
    if start_m:
        text = text[start_m.end():]
    end_m = _GUT_END_RE.search(text)
    if end_m:
        text = text[: end_m.start()]
    # Drop any remaining "Produced by ..." line that precedes the title page.
    text = re.sub(r"^\s*Produced by[^\n]*\n", "", text, count=1, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, keep alphanumerics + a few punctuation marks."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Keep sentence-boundary punctuation as separate tokens — useful signal for an LM.
    text = re.sub(r"([.,!?;:\"'()])", r" \1 ", text)
    text = re.sub(r"[^a-z0-9.,!?;:\"'()\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


# ---------- Vocabulary ----------

PAD = "<pad>"
UNK = "<unk>"


@dataclass
class Vocab:
    itos: List[str]
    stoi: dict

    @classmethod
    def build(cls, tokens: List[str], max_size: int = 8000, min_freq: int = 2) -> "Vocab":
        counter = Counter(tokens)
        # Reserve PAD=0, UNK=1.
        most_common = [w for w, c in counter.most_common() if c >= min_freq][: max_size - 2]
        itos = [PAD, UNK] + most_common
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> List[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.loads(path.read_text())
        itos = data["itos"]
        return cls(itos=itos, stoi={w: i for i, w in enumerate(itos)})


# ---------- Sequence dataset ----------

class NextWordDataset(Dataset):
    """Fixed-length sliding window over a token-id stream → (context, next-token)."""

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        assert token_ids.ndim == 1
        self.token_ids = token_ids.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + self.seq_len]
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)


# ---------- Top-level pipeline ----------

@dataclass
class Splits:
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    vocab: Vocab
    raw_train_tokens: int
    raw_val_tokens: int
    raw_test_tokens: int


def load_corpus(path: Path) -> str:
    """Load the corpus from disk. Auto-detects RTF vs plain text by suffix.

    The downloader (`src/download_data.py`) writes plain UTF-8 from Project
    Gutenberg directly; the original file shipped with this repo happened to be
    an RTF export, so we keep the RTF strip path for backward compatibility.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    plain = strip_rtf(raw) if path.suffix.lower() == ".rtf" else raw
    body = clean_gutenberg(plain)
    return normalize_text(body)


def make_splits(
    text: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    vocab_size: int = 8000,
    min_freq: int = 2,
) -> Splits:
    """Token-level contiguous split.

    Contiguous (not random) split keeps long-range narrative structure intact for each
    fold, mirrors how language models are typically evaluated on continuous text, and —
    crucially — guarantees that a (context, target) window in one split cannot share
    even a single token with another split.
    """
    tokens = tokenize(text)
    n = len(tokens)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_toks = tokens[:n_train]
    val_toks = tokens[n_train : n_train + n_val]
    test_toks = tokens[n_train + n_val :]

    # Vocab built from TRAIN ONLY.
    vocab = Vocab.build(train_toks, max_size=vocab_size, min_freq=min_freq)

    train_ids = np.asarray(vocab.encode(train_toks), dtype=np.int64)
    val_ids = np.asarray(vocab.encode(val_toks), dtype=np.int64)
    test_ids = np.asarray(vocab.encode(test_toks), dtype=np.int64)

    return Splits(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        vocab=vocab,
        raw_train_tokens=len(train_toks),
        raw_val_tokens=len(val_toks),
        raw_test_tokens=len(test_toks),
    )


def split_summary(splits: Splits) -> dict:
    v = splits.vocab
    train_unk = float((splits.train_ids == v.stoi[UNK]).mean())
    val_unk = float((splits.val_ids == v.stoi[UNK]).mean())
    test_unk = float((splits.test_ids == v.stoi[UNK]).mean())
    return {
        "vocab_size": len(v),
        "train_tokens": int(splits.train_ids.size),
        "val_tokens": int(splits.val_ids.size),
        "test_tokens": int(splits.test_ids.size),
        "train_unk_rate": train_unk,
        "val_unk_rate": val_unk,
        "test_unk_rate": test_unk,
    }
