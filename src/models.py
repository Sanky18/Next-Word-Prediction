"""Model architectures used in the experiment sweep.

Every model has the same I/O contract:
    input:  LongTensor of shape (B, T)   — token ids
    output: FloatTensor of shape (B, V)  — logits over the vocabulary for the next token

That uniform contract is what lets the training loop, evaluator, and generator be
written once and reused across all six experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Experiment 1 / 2: baseline single-layer LSTM (vary hidden_dim across runs).
# ---------------------------------------------------------------------------
class LSTMNextWord(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Bidirectional LSTM.
#
# Subtle but important: for *next-word prediction* the forward pass classically
# must not see future tokens. Here we are predicting token y = x_{T+1}, where
# x_{T+1} is **outside** the input window x_{1..T}. The bidirectional pass runs
# *inside* the observed context window only — the forward direction encodes the
# left-to-right view of x_{1..T}, the backward direction encodes its
# right-to-left view, and neither direction ever sees the target y. So this is
# legal: the model is reading the *given context* in both directions and then
# extrapolating one step forward. The Dataset and the contiguous split
# guarantee that x_{T+1} is in a different position in the sequence and is held
# out at training time.
# ---------------------------------------------------------------------------
class BiLSTMNextWord(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # `hidden_dim` here is the per-direction hidden size, matching the
        # convention used in the unidirectional LSTM models above so parameter
        # counts are comparable.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # PyTorch's bidirectional LSTM concatenates the two directions along the
        # last dim, so the per-timestep output is (B, T, 2*hidden_dim). At the
        # last timestep the forward direction has seen the whole window
        # left-to-right and the backward direction has seen it right-to-left
        # (collapsed into a single representation at t=T).
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Experiment 3: stacked LSTM with inter-layer dropout — more capacity, regularised.
# ---------------------------------------------------------------------------
class StackedLSTMNextWord(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Experiment 4: LSTM + Bahdanau (additive) attention.
#
# Why Bahdanau here? For next-word prediction on a fixed window the natural decoder
# state is the LSTM's *last* hidden vector h_T. Bahdanau computes a content-based,
# learned-MLP score between h_T (the "query") and every encoder state h_t (the
# "keys/values"), producing a soft summary of the whole window rather than just the
# final step. That gives the classifier:
#   - an explicit, differentiable focus on relevant earlier positions
#     (e.g. the subject of the sentence even when it's 15 tokens back),
#   - and a residual signal that survives the vanishing-gradient bottleneck of
#     squeezing all history into h_T.
# Additive (Bahdanau) was preferred over multiplicative (Luong) here because the
# encoder and decoder dimensions are identical, so the learned MLP gives an extra
# nonlinearity at little parameter cost — and additive attention is empirically
# slightly more stable on small datasets where the dot-product scores can saturate.
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.W_q = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # query: (B, H);   keys: (B, T, H)
        q = self.W_q(query).unsqueeze(1)        # (B, 1, A)
        k = self.W_k(keys)                       # (B, T, A)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=-1)              # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)   # (B, H)
        return context, weights


class LSTMBahdanauNextWord(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = BahdanauAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Concatenate [context ; final_hidden] before classification.
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)                 # (B, T, H)
        last = out[:, -1, :]
        context, weights = self.attn(last, out)
        feat = torch.cat([context, last], dim=-1)
        logits = self.fc(self.dropout(feat))
        if return_attn:
            return logits, weights
        return logits


# ---------------------------------------------------------------------------
# Experiment 5: LSTM + multi-head self-attention on top of the LSTM outputs.
# The MHSA layer lets each position re-mix information from every other position
# through several parallel attention subspaces — a richer family than the single
# Bahdanau head and a closer cousin to the Transformer block.
# ---------------------------------------------------------------------------
class LSTMMultiHeadAttnNextWord(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.mhsa = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        attn_out, _ = self.mhsa(out, out, out, need_weights=False)
        # Residual + LayerNorm — standard Transformer-style stabilisation.
        out = self.ln(out + attn_out)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Experiment 6: GRU + Bahdanau attention. Same recipe as Experiment 4 with a
# different RNN cell — included to isolate "LSTM vs GRU" with everything else held
# constant.
# ---------------------------------------------------------------------------
class GRUBahdanauNextWord(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = BahdanauAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        last = out[:, -1, :]
        context, _ = self.attn(last, out)
        feat = torch.cat([context, last], dim=-1)
        return self.fc(self.dropout(feat))


# ---------------------------------------------------------------------------
# Factory: name → constructed model. Used by run_experiments.py.
# ---------------------------------------------------------------------------
def build_model(name: str, vocab_size: int) -> nn.Module:
    name = name.lower()
    if name == "lstm_small":
        return LSTMNextWord(vocab_size, embed_dim=64, hidden_dim=128, dropout=0.3)
    if name == "lstm_large":
        return LSTMNextWord(vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3)
    if name == "lstm_stacked":
        return StackedLSTMNextWord(vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.4)
    if name == "bilstm":
        # Per-direction hidden = 256 → effective representation dim = 512.
        return BiLSTMNextWord(vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3)
    if name == "lstm_bahdanau":
        return LSTMBahdanauNextWord(vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3)
    if name == "lstm_mhsa":
        return LSTMMultiHeadAttnNextWord(vocab_size, embed_dim=128, hidden_dim=256, num_heads=4, dropout=0.3)
    if name == "gru_bahdanau":
        return GRUBahdanauNextWord(vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
