# Best model: AWD-LSTM (weight-drop + tied embed + var dropout, seq=40)

- Re-trained test perplexity: **64.19**
- Re-trained test top-1 accuracy: **22.91%**
- Re-trained test top-5 accuracy: **47.38%**
- Inference latency (median, 40 tokens): **90 ms** (2.26 ms/token)

## Greedy generation (argmax at each step)

_Greedy decoding intentionally surfaces the model's single most-confident continuation; as the assignment's own 'Bad Output' example warns, this can spiral into repetition loops on small RNN LMs. The samples below match that pattern, which is informative — see the top-5 trace for the first seed: when several candidates share probability, greedy locks onto one path and can't escape._

### Seed: `'i saw holmes'`

> i saw holmes , and the whole of the lady was a small , and the other of the country , the other of the city of the country , which was a small , and the other of the country , the

#### Step-by-step top-5 trace

| Step | Context tail | Chosen | Top-5 (word: prob) |
|---:|---|---|---|
| 1 | …<pad> <pad> <pad> i saw holmes | **,** | `,`: 0.110, `.`: 0.074, `in`: 0.047, `that`: 0.034, `the`: 0.028 |
| 2 | …<pad> <pad> i saw holmes , | **and** | `and`: 0.073, `the`: 0.057, `as`: 0.031, `with`: 0.030, `i`: 0.021 |
| 3 | …<pad> i saw holmes , and | **the** | `the`: 0.087, `i`: 0.044, `that`: 0.031, `a`: 0.029, `his`: 0.028 |
| 4 | …i saw holmes , and the | **whole** | `whole`: 0.019, `only`: 0.013, `other`: 0.012, `man`: 0.011, `lady`: 0.009 |
| 5 | …saw holmes , and the whole | **of** | `of`: 0.041, `,`: 0.030, `was`: 0.019, `point`: 0.010, `and`: 0.009 |
| 6 | …holmes , and the whole of | **the** | `the`: 0.121, `a`: 0.030, `his`: 0.013, `my`: 0.011, `which`: 0.010 |
| 7 | …, and the whole of the | **lady** | `lady`: 0.010, `house`: 0.010, `man`: 0.009, `case`: 0.008, `matter`: 0.008 |
| 8 | …and the whole of the lady | **was** | `was`: 0.116, `of`: 0.110, `,`: 0.106, `had`: 0.101, `who`: 0.040 |
| 9 | …the whole of the lady was | **a** | `a`: 0.046, `in`: 0.023, `the`: 0.019, `very`: 0.017, `not`: 0.015 |
| 10 | …whole of the lady was a | **small** | `small`: 0.038, `very`: 0.024, `little`: 0.021, `man`: 0.016, `long`: 0.015 |
| 11 | …of the lady was a small | **,** | `,`: 0.089, `one`: 0.041, `of`: 0.030, `man`: 0.022, `and`: 0.020 |
| 12 | …the lady was a small , | **and** | `and`: 0.034, `the`: 0.023, `of`: 0.018, `with`: 0.013, `a`: 0.011 |
| 13 | …lady was a small , and | **the** | `the`: 0.052, `a`: 0.051, `his`: 0.017, `,`: 0.015, `of`: 0.012 |
| 14 | …was a small , and the | **other** | `other`: 0.011, `man`: 0.011, `rain`: 0.010, `windows`: 0.009, `light`: 0.008 |
| 15 | …a small , and the other | **of** | `of`: 0.138, `was`: 0.094, `,`: 0.050, `had`: 0.021, `which`: 0.021 |
| 16 | …small , and the other of | **the** | `the`: 0.194, `a`: 0.081, `his`: 0.028, `which`: 0.025, `my`: 0.016 |
| 17 | …, and the other of the | **country** | `country`: 0.009, `man`: 0.008, `windows`: 0.008, `society`: 0.008, `red`: 0.007 |
| 18 | …and the other of the country | **,** | `,`: 0.085, `of`: 0.068, `.`: 0.045, `was`: 0.037, `had`: 0.026 |
| 19 | …the other of the country , | **the** | `the`: 0.066, `and`: 0.058, `which`: 0.053, `a`: 0.030, `who`: 0.022 |
| 20 | …other of the country , the | **other** | `other`: 0.018, `man`: 0.010, `body`: 0.008, `only`: 0.007, `door`: 0.005 |
| 21 | …of the country , the other | **of** | `of`: 0.178, `was`: 0.099, `,`: 0.060, `which`: 0.027, `is`: 0.020 |
| 22 | …the country , the other of | **the** | `the`: 0.216, `a`: 0.063, `which`: 0.056, `his`: 0.023, `my`: 0.014 |
| 23 | …country , the other of the | **city** | `city`: 0.012, `most`: 0.010, `red`: 0.009, `old`: 0.009, `country`: 0.009 |
| 24 | …, the other of the city | **of** | `of`: 0.125, `,`: 0.111, `.`: 0.091, `was`: 0.045, `and`: 0.037 |
| 25 | …the other of the city of | **the** | `the`: 0.132, `which`: 0.032, `a`: 0.027, `stoke`: 0.026, `whom`: 0.015 |
| 26 | …other of the city of the | **country** | `country`: 0.012, `red`: 0.012, `city`: 0.011, `bride`: 0.011, `most`: 0.010 |
| 27 | …of the city of the country | **,** | `,`: 0.127, `.`: 0.091, `of`: 0.083, `was`: 0.024, `had`: 0.017 |
| 28 | …the city of the country , | **which** | `which`: 0.072, `the`: 0.070, `and`: 0.067, `a`: 0.024, `who`: 0.022 |
| 29 | …city of the country , which | **was** | `was`: 0.126, `had`: 0.058, `is`: 0.046, `were`: 0.040, `i`: 0.027 |
| 30 | …of the country , which was | **a** | `a`: 0.056, `the`: 0.043, `in`: 0.022, `found`: 0.017, `to`: 0.016 |
| 31 | …the country , which was a | **small** | `small`: 0.044, `very`: 0.022, `little`: 0.018, `singular`: 0.017, `long`: 0.014 |
| 32 | …country , which was a small | **,** | `,`: 0.053, `one`: 0.030, `of`: 0.025, `in`: 0.014, `and`: 0.014 |
| 33 | …, which was a small , | **and** | `and`: 0.024, `of`: 0.022, `the`: 0.021, `with`: 0.011, `which`: 0.010 |
| 34 | …which was a small , and | **the** | `the`: 0.058, `a`: 0.050, `,`: 0.013, `of`: 0.013, `in`: 0.012 |
| 35 | …was a small , and the | **other** | `other`: 0.012, `windows`: 0.010, `rain`: 0.009, `man`: 0.009, `light`: 0.008 |
| 36 | …a small , and the other | **of** | `of`: 0.145, `was`: 0.076, `,`: 0.043, `which`: 0.026, `the`: 0.018 |
| 37 | …small , and the other of | **the** | `the`: 0.207, `a`: 0.078, `which`: 0.030, `his`: 0.025, `my`: 0.016 |
| 38 | …, and the other of the | **country** | `country`: 0.010, `windows`: 0.008, `red`: 0.007, `society`: 0.007, `man`: 0.007 |
| 39 | …and the other of the country | **,** | `,`: 0.078, `of`: 0.066, `.`: 0.052, `was`: 0.034, `were`: 0.024 |
| 40 | …the other of the country , | **the** | `the`: 0.066, `which`: 0.060, `and`: 0.059, `a`: 0.029, `who`: 0.022 |

### Seed: `'the door opened and'`

> the door opened and the door . the door was a small , and a pair of a man who was a small , and a pair of a man who is a very deep , and a pair of a man who is

### Seed: `'watson looked at the'`

> watson looked at the last , and the other was a small , the old man , and the other was a small , the old man , and the other of the country , the other of the city of the city .

### Seed: `'it was a cold morning when'`

> it was a cold morning when the night was a little , and the whole of the country , and the other was a small , and the rain of the drug , and the windows of the wood . the other was a small ,

### Seed: `'sherlock holmes lit his pipe'`

> sherlock holmes lit his pipe , and the lamp was a small , and a pair of a man who is a very deep , and a pair of a man who is a very deep , and a little , the man who is

## Top-k sampled generation (k=5, temperature=0.9)

_Sampling from the model's top-5 candidates at each step (rather than always taking the argmax) preserves the model's distributional knowledge while breaking out of greedy loops. This is the same model — only the decoding strategy differs._

### Seed: `'i saw holmes'`

> i saw holmes , and his whole eyes were at one side , and a long man , a man who had been a small one of those , which was a little of the old country . the lady had not been

### Seed: `'the door opened and'`

> the door opened and , and i could hardly see that the matter was the only of the same time . i was in the house , and it was the very thought of the same . i was not to be a little

### Seed: `'watson looked at the'`

> watson looked at the hotel , i saw that the door was still the other of the floor . i found a few moments and was a small , the old man of the wood , and the windows were to be the very

### Seed: `'it was a cold morning when'`

> it was a cold morning when the matter was a little . i have a good time , and the other had not the time , and i was very much in the way . it was a little thing , but it is a very

### Seed: `'sherlock holmes lit his pipe'`

> sherlock holmes lit his pipe , and his eyes was a black , black face , and a pair of a red red hair , and a broad black hat , a broad brimmed hat of a large , black , white , thin ,
