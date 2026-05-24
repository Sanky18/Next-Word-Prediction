"""draw.io XML generators for each architecture in the sweep.

Used after the sweep completes to export a diagram for the empirical winner. Each
generator produces a self-contained `.drawio` XML file you can open at
https://app.diagrams.net or with the VS Code "Draw.io Integration" extension.
"""

from __future__ import annotations

from pathlib import Path


def _box(cid: str, x: int, y: int, w: int, h: int, value: str, fill: str, stroke: str) -> str:
    return (
        f'<mxCell id="{cid}" value="{value}" '
        f'style="rounded=1;fillColor={fill};strokeColor={stroke};whiteSpace=wrap;html=1;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f'</mxCell>'
    )


def _arrow(src: str, dst: str) -> str:
    return (
        f'<mxCell style="endArrow=classic;html=1;" edge="1" parent="1" source="{src}" target="{dst}">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f'</mxCell>'
    )


def _wrap(body: str, name: str = "Best Architecture") -> str:
    return f"""<mxfile host="app.diagrams.net">
  <diagram id="best-arch" name="{name}">
    <mxGraphModel dx="1000" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# Colors used per layer-type for consistency across diagrams.
C_IN = ("#E1F5FE", "#0288D1")
C_EMB = ("#FFF3E0", "#F57C00")
C_RNN = ("#E8F5E9", "#2E7D32")
C_SPLIT = ("#F3E5F5", "#6A1B9A")
C_ATTN = ("#FFEBEE", "#C62828")
C_MIX = ("#ECEFF1", "#37474F")
C_FC = ("#FFF8E1", "#F9A825")


def diagram_lstm_simple(label: str, hidden: int, layers: int = 1) -> str:
    cells, x = [], 320
    cells.append(_box("in", x, 30, 220, 50, f"Input token ids&#10;(B, T=20)", *C_IN))
    cells.append(_box("emb", x, 110, 220, 50, "Embedding&#10;vocab → 128", *C_EMB))
    rnn_value = f"LSTM&#10;input=128, hidden={hidden}, layers={layers}&#10;output: H = (B, T, {hidden})"
    cells.append(_box("lstm", x - 40, 190, 300, 70, rnn_value, *C_RNN))
    cells.append(_box("hT", x, 290, 220, 50, f"h_T = H[:, -1, :]&#10;(B, {hidden})", *C_SPLIT))
    cells.append(_box("dp", x, 370, 220, 40, "Dropout(p=0.3)", *C_MIX))
    cells.append(_box("fc", x, 440, 220, 50, f"Linear: {hidden} → vocab_size", *C_FC))
    cells.append(_box("out", x - 20, 520, 260, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "lstm"), ("lstm", "hT"), ("hT", "dp"), ("dp", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_lstm_bahdanau(label: str = "LSTM + Bahdanau Attention") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=20)", *C_IN))
    cells.append(_box("emb", 320, 100, 220, 50, "Embedding&#10;vocab → 128", *C_EMB))
    cells.append(_box("lstm", 280, 180, 300, 70,
                      "LSTM (batch_first)&#10;input=128, hidden=256, layers=1&#10;output: H = (B, T, 256)", *C_RNN))
    cells.append(_box("hT", 120, 290, 200, 55,
                      "h_T = H[:, -1, :]&#10;(B, 256)  — query", *C_SPLIT))
    cells.append(_box("HK", 540, 290, 220, 55,
                      "H = all hidden states&#10;(B, T, 256)  — keys / values", *C_SPLIT))
    cells.append(_box("attn", 280, 380, 300, 100,
                      "Bahdanau (additive) attention&#10;"
                      "score_t = v · tanh(W_q h_T + W_k h_t)&#10;"
                      "α = softmax(score)&#10;"
                      "context = Σ α_t h_t   (B, 256)", *C_ATTN))
    cells.append(_box("concat", 320, 510, 220, 50,
                      "concat([context ; h_T])&#10;(B, 512)", *C_MIX))
    cells.append(_box("dp", 320, 590, 220, 40, "Dropout(p=0.3)", *C_MIX))
    cells.append(_box("fc", 320, 660, 220, 50, "Linear: 512 → vocab_size", *C_FC))
    cells.append(_box("out", 280, 740, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "lstm"), ("lstm", "hT"), ("lstm", "HK"),
             ("hT", "attn"), ("HK", "attn"), ("attn", "concat"),
             ("concat", "dp"), ("dp", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_lstm_mhsa(label: str = "LSTM + Multi-Head Self-Attention") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=20)", *C_IN))
    cells.append(_box("emb", 320, 100, 220, 50, "Embedding&#10;vocab → 128", *C_EMB))
    cells.append(_box("lstm", 280, 180, 300, 70,
                      "LSTM&#10;input=128, hidden=256&#10;output: H = (B, T, 256)", *C_RNN))
    cells.append(_box("mhsa", 280, 280, 300, 80,
                      "MultiheadAttention&#10;num_heads=4, dropout=0.3&#10;Q=K=V=H&#10;attn_out (B, T, 256)", *C_ATTN))
    cells.append(_box("ln", 280, 380, 300, 50, "LayerNorm(H + attn_out)&#10;(B, T, 256)", *C_MIX))
    cells.append(_box("last", 320, 460, 220, 50, "take last position&#10;(B, 256)", *C_SPLIT))
    cells.append(_box("dp", 320, 540, 220, 40, "Dropout(p=0.3)", *C_MIX))
    cells.append(_box("fc", 320, 610, 220, 50, "Linear: 256 → vocab_size", *C_FC))
    cells.append(_box("out", 280, 690, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "lstm"), ("lstm", "mhsa"), ("mhsa", "ln"),
             ("ln", "last"), ("last", "dp"), ("dp", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_stacked_lstm(label: str = "Stacked LSTM-256 × 2") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=20)", *C_IN))
    cells.append(_box("emb", 320, 100, 220, 50, "Embedding&#10;vocab → 128", *C_EMB))
    cells.append(_box("lstm1", 280, 180, 300, 60,
                      "LSTM layer 1&#10;input=128, hidden=256", *C_RNN))
    cells.append(_box("inter_dp", 320, 250, 220, 40, "Inter-layer Dropout(p=0.4)", *C_MIX))
    cells.append(_box("lstm2", 280, 300, 300, 60,
                      "LSTM layer 2&#10;input=256, hidden=256&#10;output: H = (B, T, 256)", *C_RNN))
    cells.append(_box("hT", 320, 380, 220, 50,
                      "h_T = H[:, -1, :]&#10;(B, 256)", *C_SPLIT))
    cells.append(_box("dp", 320, 450, 220, 40, "Dropout(p=0.4)", *C_MIX))
    cells.append(_box("fc", 320, 510, 220, 50, "Linear: 256 → vocab_size", *C_FC))
    cells.append(_box("out", 280, 590, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "lstm1"), ("lstm1", "inter_dp"),
             ("inter_dp", "lstm2"), ("lstm2", "hT"), ("hT", "dp"),
             ("dp", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_gru_bahdanau(label: str = "GRU + Bahdanau Attention") -> str:
    xml = diagram_lstm_bahdanau(label=label)
    return xml.replace("LSTM (batch_first)", "GRU (batch_first)").replace("LSTM", "GRU", 1)


def diagram_bilstm(label: str = "BiLSTM-256") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=20)", *C_IN))
    cells.append(_box("emb", 320, 100, 220, 50, "Embedding&#10;vocab → 128", *C_EMB))
    cells.append(_box("bilstm", 260, 180, 340, 90,
                      "Bidirectional LSTM&#10;input=128, hidden=256/direction&#10;"
                      "→ forward H_f (B, T, 256)&#10;→ backward H_b (B, T, 256)&#10;"
                      "concat: H = (B, T, 512)", *C_RNN))
    cells.append(_box("last", 320, 290, 220, 60,
                      "last timestep: H[:, -1, :]&#10;(B, 512) — "
                      "f-direction has seen the full window;&#10;"
                      "b-direction has the right-to-left summary collapsed at t=T", *C_SPLIT))
    cells.append(_box("dp", 320, 370, 220, 40, "Dropout(p=0.3)", *C_MIX))
    cells.append(_box("fc", 320, 430, 220, 50, "Linear: 512 → vocab_size", *C_FC))
    cells.append(_box("out", 280, 510, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "bilstm"), ("bilstm", "last"),
             ("last", "dp"), ("dp", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_lstm_enhanced(label: str = "Stacked LSTM + Tied Embed + Variational Dropout") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=40)", *C_IN))
    cells.append(_box("emb", 280, 100, 300, 60,
                      "Embedding&#10;vocab → 256&#10;weights TIED with output FC", *C_EMB))
    cells.append(_box("vd_in", 320, 180, 220, 40, "Variational Dropout(p=0.3)&#10;(one mask shared across time)", *C_MIX))
    cells.append(_box("lstm1", 280, 240, 300, 55, "LSTM layer 1&#10;input=256, hidden=256", *C_RNN))
    cells.append(_box("inter", 320, 310, 220, 40, "Inter-layer Dropout(p=0.4)", *C_MIX))
    cells.append(_box("lstm2", 280, 365, 300, 55, "LSTM layer 2&#10;input=256, hidden=256", *C_RNN))
    cells.append(_box("vd_out", 320, 435, 220, 40, "Variational Dropout(p=0.4)", *C_MIX))
    cells.append(_box("last", 320, 495, 220, 40, "h_T = H[:, -1, :]  (B, 256)", *C_SPLIT))
    cells.append(_box("fc", 280, 555, 300, 60,
                      "Linear: 256 → vocab&#10;weights SHARED with embedding", *C_FC))
    cells.append(_box("out", 280, 640, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb"), ("emb", "vd_in"), ("vd_in", "lstm1"), ("lstm1", "inter"),
             ("inter", "lstm2"), ("lstm2", "vd_out"), ("vd_out", "last"),
             ("last", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_awd_lstm(label: str = "AWD-LSTM") -> str:
    cells = []
    cells.append(_box("in", 320, 20, 220, 50, "Input token ids&#10;(B, T=40)", *C_IN))
    cells.append(_box("emb_drop", 280, 100, 300, 50,
                      "Embedding Dropout(p=0.1)&#10;(zero whole vocabulary rows)", *C_MIX))
    cells.append(_box("emb", 280, 165, 300, 55,
                      "Embedding&#10;vocab → 256 (tied with output FC)", *C_EMB))
    cells.append(_box("vd_in", 320, 235, 220, 40, "Variational Dropout(p=0.4)", *C_MIX))
    cells.append(_box("wd_lstm1", 240, 295, 380, 60,
                      "Weight-Dropped LSTM layer 1&#10;DropConnect(p=0.5) on weight_hh_l0", *C_RNN))
    cells.append(_box("inter", 320, 370, 220, 40, "Inter-layer Dropout(p=0.3)", *C_MIX))
    cells.append(_box("wd_lstm2", 240, 425, 380, 60,
                      "Weight-Dropped LSTM layer 2&#10;DropConnect(p=0.5) on weight_hh_l1", *C_RNN))
    cells.append(_box("vd_out", 320, 500, 220, 40, "Variational Dropout(p=0.4)", *C_MIX))
    cells.append(_box("last", 320, 560, 220, 40, "h_T = H[:, -1, :]  (B, 256)", *C_SPLIT))
    cells.append(_box("fc", 280, 620, 300, 60,
                      "Linear: 256 → vocab&#10;weights SHARED with embedding", *C_FC))
    cells.append(_box("out", 280, 700, 300, 50, "softmax → top-k for next word", *C_IN))
    edges = [("in", "emb_drop"), ("emb_drop", "emb"), ("emb", "vd_in"),
             ("vd_in", "wd_lstm1"), ("wd_lstm1", "inter"), ("inter", "wd_lstm2"),
             ("wd_lstm2", "vd_out"), ("vd_out", "last"), ("last", "fc"), ("fc", "out")]
    cells.extend(_arrow(a, b) for a, b in edges)
    return _wrap("\n".join(cells), name=label)


def diagram_for(name: str, label: str) -> str:
    name = name.lower()
    if name == "lstm_small":
        return diagram_lstm_simple(label, hidden=128, layers=1)
    if name == "lstm_large":
        return diagram_lstm_simple(label, hidden=256, layers=1)
    if name == "lstm_stacked":
        return diagram_stacked_lstm(label)
    if name == "bilstm":
        return diagram_bilstm(label)
    if name == "lstm_enhanced":
        return diagram_lstm_enhanced(label)
    if name == "awd_lstm":
        return diagram_awd_lstm(label)
    if name == "lstm_bahdanau":
        return diagram_lstm_bahdanau(label)
    if name == "lstm_mhsa":
        return diagram_lstm_mhsa(label)
    if name == "gru_bahdanau":
        return diagram_gru_bahdanau(label)
    raise ValueError(f"No diagram for model name: {name}")


def write_diagram(name: str, label: str, out_path: Path) -> None:
    out_path.write_text(diagram_for(name, label))
