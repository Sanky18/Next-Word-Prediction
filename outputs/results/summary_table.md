| Rank | Experiment | Params | Train s | Val acc | Test acc | Test top-5 | Test PPL | Inf ms/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | AWD-LSTM (weight-drop + tied embed + var dropout, seq=40) | 2,057,285 | 575 | 0.220 | 0.228 | 0.470 | 65.41 | 1.81 |
| 2 | Stacked LSTM + tied embeddings + variational dropout (seq=40) | 2,057,285 | 582 | 0.219 | 0.222 | 0.465 | 70.40 | 1.77 |
| 3 | Stacked LSTM-256x2 | 2,426,565 | 312 | 0.210 | 0.218 | 0.458 | 70.95 | 1.50 |
| 4 | LSTM-256 | 1,900,229 | 188 | 0.198 | 0.216 | 0.447 | 72.78 | 1.51 |
| 5 | LSTM-256 + Bahdanau Attn | 3,032,261 | 250 | 0.205 | 0.210 | 0.450 | 73.24 | 1.73 |
| 6 | LSTM-128 (baseline) | 853,765 | 113 | 0.204 | 0.210 | 0.444 | 74.67 | 1.43 |
| 7 | BiLSTM-256 | 3,296,197 | 349 | 0.202 | 0.212 | 0.449 | 75.29 | 1.95 |
| 8 | GRU-256 + Bahdanau Attn | 2,933,445 | 317 | 0.191 | 0.202 | 0.441 | 78.83 | 3.00 |
| 9 | LSTM-256 + Multi-Head Self-Attn | 2,163,909 | 300 | 0.196 | 0.199 | 0.441 | 79.46 | 1.75 |