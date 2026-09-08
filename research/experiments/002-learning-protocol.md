# E002: fixed tiny-model screening protocol

Status: proposed, not implemented or executed. Read T08 and the architecture definitions first. This protocol supplies defaults so an implementation agent does not improvise a different training experiment for each candidate.

## Stage0: plumbing

Use synthetic next-token/copy sequences and16–64 short examples. Require falling training loss, finite gradients, exact parameter/state accounting and a successful checkpoint/resume that preserves the next minibatch and RNG. CPU reference is sufficient. Compare full-sequence and chunked recurrent forward/backward on tiny lengths. For QAT, compare fake-quant forward outputs with the intended integer/LUT reference separately from STE gradients.

## Stage1: language screen

- Data: a deterministically selected public TinyStories subset with its original held-out split. Pin revision and content checksums; do not download the entire corpus if streaming a fixed small subset suffices. Save exact selected document IDs and split. Training tokenizer:8192 vocabulary entries trained once on training-only text and reused by every model.
- Token budgets:100k tokens plumbing, then5,000,000 training tokens for the first screen. Validation: fixed100k held-out tokens and an independently held-out natural-text sample for transfer diagnostics. No evaluation data in tokenizer training or calibration.
- Sequence length256; effective batch2048 tokens via microbatch/gradient accumulation appropriate to measured memory. Reset recurrent state at document boundaries. Do not give one candidate extra cross-document context. Longer512/2048/8192 recall tests are separate, clearly labeled extrapolation tests.
- AdamW, initial LR3e-4, betas(0.9,0.95), epsilon1e-8, weight decay0.1 on matrix weights, gradient norm clip1.0. Linear warmup5% of updates then cosine to10% of peak LR. FP32 masters, activations/state/optimizer initially. Record exactly which biases/norms are exempt from decay.
- Seed17 for screening; survivors run17/29/43. If an architecture's faithful implementation has incompatible training requirements, document the reason and run a bounded equal-budget LR comparison(1e-4,3e-4,1e-3) for both it and the Transformer control. Never tune only the favored model extensively.
- QAT: initially symmetric per-row/group weight scales and per-token/group activation clipping; preserve FP32 recurrent state and normalization. Start with FP32 learning-control and separate QAT arm. Do not change quantization method mid-run without a new experiment ID.
- Output: parameter/active/state counts, loss curves, held-out NLL and PPL, actual training tokens/updates, achieved throughput, wall time, peak GPU/RAM, estimated arithmetic, sampled GPU joules and thermal history. Record loss in nats/token; do not compare PPL across different tokenizers.

## Stage2: useful representation checks

Fixed generated datasets for copy, delayed associative recall and finite-state/modular counting; separate seeds for train and evaluation. Use delays64/256/1024 and test held-out combinations, not memorized sequence tables. Report exact-match accuracy and confidence intervals. Include equal-state-byte and equal-parameter controls. A simple vector recurrence may need more state; this is a tested information bottleneck, not a kernel defect.

## Stage3: trained inference performance

Same trained checkpoint for arithmetic A/Bs; batch1/2/4/8, context128/512/2048 and fixed128 or512 generation budget after canaries. Separate prefill, decode and complete-answer timing. Use teacher-forced common token traces to compare math at identical workload, then free-running generation for behavioral checks. Compare correctness against FP32, include activation quantization/LUT construction, and report device residency and transfers.

Default screening failure is >3% relative PPL regression or>2 percentage-point recall decline without a compensating, explicitly reported quality/speed tradeoff. A model must beat the unigram baseline and improve held-out loss before hardware speed is scientifically meaningful. One5M-token run only screens candidates; it does not establish scalability or broad language quality. Review three-seed results before50M-scale work.
