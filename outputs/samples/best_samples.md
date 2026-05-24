# Best model: Stacked LSTM-256x2

- Re-trained test perplexity: **69.86**
- Re-trained test top-1 accuracy: **22.33%**
- Re-trained test top-5 accuracy: **46.07%**
- Inference latency (median, 40 tokens): **60 ms** (1.50 ms/token)

## Greedy generation (argmax at each step)

_Greedy decoding intentionally surfaces the model's single most-confident continuation; as the assignment's own 'Bad Output' example warns, this can spiral into repetition loops on small RNN LMs. The samples below match that pattern, which is informative — see the top-5 trace for the first seed: when several candidates share probability, greedy locks onto one path and can't escape._

### Seed: `'i saw holmes'`

> i saw holmes , and i was a little of the matter , and i had been in the time of the matter . i have been able to see that i have been able to see the matter . i have been

#### Step-by-step top-5 trace

| Step | Context tail | Chosen | Top-5 (word: prob) |
|---:|---|---|---|
| 1 | …<pad> <pad> <pad> i saw holmes | **,** | `,`: 0.115, `.`: 0.101, `and`: 0.059, `in`: 0.052, `to`: 0.041 |
| 2 | …<pad> <pad> i saw holmes , | **and** | `and`: 0.360, `but`: 0.082, `with`: 0.058, `for`: 0.033, `which`: 0.030 |
| 3 | …<pad> i saw holmes , and | **i** | `i`: 0.125, `the`: 0.101, `he`: 0.052, `that`: 0.048, `a`: 0.043 |
| 4 | …i saw holmes , and i | **was** | `was`: 0.104, `had`: 0.079, `could`: 0.061, `have`: 0.061, `saw`: 0.036 |
| 5 | …saw holmes , and i was | **a** | `a`: 0.037, `not`: 0.033, `glad`: 0.021, `in`: 0.020, `very`: 0.018 |
| 6 | …holmes , and i was a | **little** | `little`: 0.080, `very`: 0.062, `man`: 0.035, `good`: 0.019, `considerable`: 0.019 |
| 7 | …, and i was a little | **of** | `of`: 0.031, `,`: 0.022, `man`: 0.019, `more`: 0.019, `little`: 0.016 |
| 8 | …and i was a little of | **the** | `the`: 0.166, `a`: 0.095, `my`: 0.047, `his`: 0.019, `them`: 0.016 |
| 9 | …i was a little of the | **matter** | `matter`: 0.018, `house`: 0.010, `same`: 0.010, `city`: 0.010, `room`: 0.009 |
| 10 | …was a little of the matter | **,** | `,`: 0.164, `.`: 0.160, `to`: 0.052, `in`: 0.051, `and`: 0.046 |
| 11 | …a little of the matter , | **and** | `and`: 0.310, `but`: 0.129, `which`: 0.051, `for`: 0.035, `so`: 0.035 |
| 12 | …little of the matter , and | **i** | `i`: 0.127, `the`: 0.101, `that`: 0.090, `he`: 0.046, `yet`: 0.035 |
| 13 | …of the matter , and i | **had** | `had`: 0.086, `have`: 0.085, `was`: 0.083, `could`: 0.058, `should`: 0.027 |
| 14 | …the matter , and i had | **been** | `been`: 0.063, `not`: 0.049, `no`: 0.042, `a`: 0.041, `seen`: 0.037 |
| 15 | …matter , and i had been | **in** | `in`: 0.026, `a`: 0.024, `able`: 0.023, `seen`: 0.023, `so`: 0.020 |
| 16 | …, and i had been in | **the** | `the`: 0.224, `a`: 0.118, `my`: 0.067, `his`: 0.024, `an`: 0.019 |
| 17 | …and i had been in the | **time** | `time`: 0.024, `room`: 0.020, `way`: 0.015, `matter`: 0.014, `morning`: 0.014 |
| 18 | …i had been in the time | **of** | `of`: 0.327, `that`: 0.206, `,`: 0.107, `to`: 0.047, `.`: 0.045 |
| 19 | …had been in the time of | **the** | `the`: 0.355, `my`: 0.089, `a`: 0.059, `this`: 0.030, `his`: 0.022 |
| 20 | …been in the time of the | **matter** | `matter`: 0.024, `same`: 0.014, `house`: 0.013, `time`: 0.011, `sort`: 0.010 |
| 21 | …in the time of the matter | **.** | `.`: 0.306, `,`: 0.179, `to`: 0.055, `and`: 0.039, `which`: 0.033 |
| 22 | …the time of the matter . | **i** | `i`: 0.126, `it`: 0.079, `the`: 0.066, `he`: 0.049, `but`: 0.035 |
| 23 | …time of the matter . i | **have** | `have`: 0.092, `had`: 0.068, `was`: 0.059, `am`: 0.057, `think`: 0.031 |
| 24 | …of the matter . i have | **been** | `been`: 0.103, `not`: 0.071, `no`: 0.064, `heard`: 0.058, `a`: 0.044 |
| 25 | …the matter . i have been | **able** | `able`: 0.031, `a`: 0.028, `seen`: 0.022, `in`: 0.022, `so`: 0.020 |
| 26 | …matter . i have been able | **to** | `to`: 0.884, `in`: 0.019, `for`: 0.018, `,`: 0.018, `with`: 0.009 |
| 27 | …. i have been able to | **see** | `see`: 0.069, `be`: 0.059, `think`: 0.057, `know`: 0.042, `hear`: 0.035 |
| 28 | …i have been able to see | **that** | `that`: 0.092, `the`: 0.091, `you`: 0.088, `it`: 0.066, `him`: 0.056 |
| 29 | …have been able to see that | **i** | `i`: 0.101, `the`: 0.083, `you`: 0.081, `it`: 0.056, `this`: 0.043 |
| 30 | …been able to see that i | **have** | `have`: 0.149, `am`: 0.097, `should`: 0.075, `was`: 0.074, `had`: 0.072 |
| 31 | …able to see that i have | **been** | `been`: 0.093, `not`: 0.050, `a`: 0.045, `ever`: 0.039, `no`: 0.037 |
| 32 | …to see that i have been | **able** | `able`: 0.038, `a`: 0.032, `seen`: 0.030, `in`: 0.024, `so`: 0.024 |
| 33 | …see that i have been able | **to** | `to`: 0.908, `,`: 0.018, `for`: 0.016, `.`: 0.013, `in`: 0.010 |
| 34 | …that i have been able to | **see** | `see`: 0.057, `be`: 0.056, `think`: 0.042, `know`: 0.041, `have`: 0.032 |
| 35 | …i have been able to see | **the** | `the`: 0.098, `you`: 0.096, `it`: 0.062, `that`: 0.056, `him`: 0.051 |
| 36 | …have been able to see the | **matter** | `matter`: 0.045, `little`: 0.022, `police`: 0.017, `man`: 0.015, `facts`: 0.012 |
| 37 | …been able to see the matter | **.** | `.`: 0.274, `,`: 0.148, `to`: 0.062, `of`: 0.047, `?`: 0.039 |
| 38 | …able to see the matter . | **i** | `i`: 0.153, `it`: 0.073, `but`: 0.050, `and`: 0.045, `you`: 0.043 |
| 39 | …to see the matter . i | **have** | `have`: 0.120, `am`: 0.102, `shall`: 0.060, `was`: 0.043, `think`: 0.042 |
| 40 | …see the matter . i have | **been** | `been`: 0.096, `no`: 0.074, `not`: 0.074, `heard`: 0.061, `a`: 0.054 |

### Seed: `'the door opened and'`

> the door opened and a little man , and the whole man was a little of the door , and the whole of the other of the window , and the whole man was a little of the window , and the whole of

### Seed: `'watson looked at the'`

> watson looked at the room . i have been able to see that i have been able to be a little more than a little problem . i have been able to see you . i have been able to see you . i

### Seed: `'it was a cold morning when'`

> it was a cold morning when he had been in the room . i have been able to see that i have been able to see the matter . i have been able to see you . i have been able to see you . i

### Seed: `'sherlock holmes lit his pipe'`

> sherlock holmes lit his pipe , and the whole man was a little of the door , and the whole of the other , and the whole of the other of the other , and the lady , the lady , and the lady of

## Top-k sampled generation (k=5, temperature=0.9)

_Sampling from the model's top-5 candidates at each step (rather than always taking the argmax) preserves the model's distributional knowledge while breaking out of greedy loops. This is the same model — only the decoding strategy differs._

### Seed: `'i saw holmes'`

> i saw holmes , and the lady and the man had been in the table of a man who is a little of the most man . there was no sign of it , and i could not have the little , i

### Seed: `'the door opened and'`

> the door opened and his eyes and found , and the lady were a small man who would have been a very strong man . it is a common , i have a little man for a little business , and i have been

### Seed: `'watson looked at the'`

> watson looked at the time of this case , and that i could not have a cab to be the machine to have the first of the matter . it was a very little , and the other is the most , the matter

### Seed: `'it was a cold morning when'`

> it was a cold morning when a man who was a man who had been in my companion . it would be , however , but i had been able to see the police . it was a little man , i have not seen a

### Seed: `'sherlock holmes lit his pipe'`

> sherlock holmes lit his pipe , and his eyes were in his chair , and the man were a very very very man . i have seen it , and i am sorry to be in the house . i was glad to hear a
