# Five P40-native architecture candidates

These are proposed, separately trained ablations grounded in [prior work](../papers.md), not novel architectures by assertion or conversions of the 35B Qwen checkpoint. Parameter counts below are pencil estimates, excluding small biases/norms; the implementation must report exact trainable, active and stored counts.

## Shared experimental scaffold

First serious screen: width D=384, L=8 blocks, tied vocabulary embedding/output V=8192, float recurrent state. The tied embedding contributes VD=3,145,728 parameters; the full output projection is executed per token and must be included in performance. Increase/decrease width or FFN rank to match a ~12M Transformer control within 5%. Also report equal-active-compute and equal-state-byte controls when different architectures make total-parameter matching misleading.

Each model has normalization, residual paths and a standard next-token cross-entropy objective. Tiny synthetic overfit precedes 5M-token language screening. FP32 master weights/gradients/Adam state remain during QAT; INT8/ternary inference is not evidence of integer-only training. Approximate weight/activation paths use straight-through gradients with calibrated clipping as an experimental choice.

The diagrams are block patterns repeated L times; they specify data flow, not an executable implementation.

### A — diagonal recurrence with integer projections

`token → W8A8 input/decay projections → FP32 diagonal state → output projection → small channel mixer → logits`

For channel d and state coordinate n:

```
a_t = sigmoid(W_a x_t), b_t = W_b x_t
h_t[d,n] = a_t[d] h_(t-1)[d,n] + B[d,n] b_t[d]
u_t[d] = sum_n C[d,n] h_t[d,n]
y_t = W_o u_t
channel(x) = V_m silu(U_m x)
```

Use N=16, mixer rank r=2D. Core parameters approximately `L(3D² + 2Dr + 2DN)+VD` =11.50M. State `4LDN` =196,608 bytes (0.188 MiB). Linear projections cost about twice their active parameter count in multiply/add-equivalent operations per token (~23M including output head), plus O(LDN) recurrence and nonlinearities. W8 storage is roughly 11.5 MB plus scales; BF16/FP32 retained tensors change that total.

Why useful: decaying content traces can summarize history; input-dependent gates choose retention. Why Pascal: small streaming state, quantizable projections, no KV growth. Bottleneck: diagonal/vector compression may lose associative recall and state tracking. Training: this is an intentionally simple selective/diagonal SSM ablation inspired by S4/Mamba, not a faithful implementation of Mamba-3. Compare an actual small Mamba reference too. Kill or enrich state if retrieval fails despite adequate training.

### B — ternary gated vector recurrence

`token → ternary gate/candidate projections → FP32 gated recurrence → ternary channel mixer → logits`

```
f = sigmoid(T_f x), z = sigmoid(T_z x), c = silu(T_c x)
h_t = f * h_(t-1) + (1-f) * c
y_t = T_o(z * h_t)
channel(x) = T_down(silu(T_gate x) * (T_up x))
T = scale * round_and_clip(W_master) in {-1,0,+1}
```

With mixer rank 2D, core `L(4D²+6D²)+VD` ≈14.94M; reduce mixer rank for the matched-parameter control. State `4LD` =12,288 bytes. Approximately 29.9M dense-equivalent operations/token; executed arithmetic can be additions/LUT sums but still reads weights/scales and reduces rows. If only block matrices (11.80M) are packed at 2 bits and the tied embedding remains FP32, weights total about 15.53 MB plus scales, not a misleading uniform 1.58-bit total. Evaluate quantized output embeddings separately.

Why useful: learnable retention/candidate generation supports sequence prediction with tiny state. Why Pascal: additions/DP4A or LUT linear operations plus a very small FP32 recurrent vector. Weakness: recurrence compresses history aggressively; gates and output head may dominate after weight math is cheap. Train from scratch with ternary-aware forward passes, grounded in MatMul-free LM/BitNet. Compare unpacked ternary-DP4A versus packed ternary-LUT on identical trained weights. Kill if it cannot learn recall or gives no useful speed/quality gain.

### C — quantized projections with an FP32 delta-rule memory

`token → quantized q/k/v projections → FP32 associative state update/read → output + channel mixer → logits`

```
S' = alpha_t S_(t-1)
delta = beta_t (v_t - k_t^T S')
S_t = S' + k_t delta^T
u_t = q_t^T S_t
```

Use 12 heads of 32×32 state and mixer rank 2D. Four D×D q/k/v/o projections plus two mixer matrices: `8LD²+VD` ≈12.58M. State `4LHn²` =393,216 bytes (0.375 MiB). Roughly 25.2M projection operations plus O(LHn²) reads/writes and outer products per token. W8 projection storage about12.6 MB plus scales; output/state precision is recorded separately.

Why useful: a content-addressed read/write correction is more expressive than independent scalar decay. Why Pascal: fuse update/read, keep state local and use DP4A only for suitable projections. Weakness: state bandwidth and sequential dependencies; increasing state can erase gains. Closest work is gated DeltaNet/RWKV-7, already related to Qwen's mixer. Preserve FP32 state first; integer-state ablations come only after long-rollout error tests. Training uses differentiable recurrence/chunking; compare to a faithful RWKV-7 small reference rather than calling this equation RWKV-7.

### D — conditionally updated recurrent memory banks

`token → shared projections/router → one selected bank + shared bank → merge → mixer → logits`

Maintain five C-like banks (four routed, one shared). Select top1 or top2 via a trained router; update only selected banks and shared bank, leave others stationary. Read/merge with normalized routing weights. Decay semantics for skipped banks must be explicit: first use no hidden per-token decay; a time-aware decay-on-next-touch is a separate ablation.

Share q/k/v/o across banks initially to isolate memory conditionality. Parameters roughly A/C-style shared projection count plus `L*4D` router (~12.60M using C); state five times C =1.875 MiB per sequence. Top1+shared touches two banks versus five for an all-bank control. Touched-state lower bound is at least read+write for two selected states; actual read/update passes can be greater. Weight traffic is mostly shared and does not shrink fivefold.

Why useful: specialized memories may reduce interference and preserve different topics. Why Pascal: stationary state and block-local updates, small activation routing across GPUs later. Weakness: router collapse, inactive-bank staleness, more total state, imbalance and scatter overhead. Train router and memory with a measured load-balancing term; do not force routing locality without evaluating quality. Grounding: Mixture-of-Memories. Equal-total-state and equal-active-state controls are mandatory.

### E — small hybrid with compressed routed FFN experts

`token → delta recurrence (3/4 blocks) or tiled attention (1/4) → top1 of four low-bit experts → residual → logits`

With per-expert SwiGLU width r=D, four experts and top1, mixing projections4D²: total `L(4D²+3E D r)+VD` ≈22.02M; active matrix parameters/token `L(4D²+3Dr)+VD` ≈11.40M. This is deliberately larger total than the ~12M active control; report both. Compare a smaller equal-total model too. Projection operations roughly22.8M/token plus attention/state/nonlinearities.

At context T, two full multihead-attention layers use approximately `2 layers *2(K,V)*T*D*4 =6144T` bytes KV (12 MiB at2048 tokens), plus six recurrent matrix states (~0.281 MiB). FFN weights may use group-scaled INT4 or additive vector codebooks; distinguish storage/calibration from inference kernels. Packing all non-embedding weights at four bits would be a roughly9.44 MB body, but actual dense/codebook/scales precision must be counted rather than assumed.

Why useful: occasional explicit attention relieves the recurrence information bottleneck, while experts add capacity without activating every weight. Why Pascal: small active weight set, reuse across tokens/agents, expert placement and LUT kernels. Weakness: routing overhead and attention/KV traffic; codebook reconstructions may lose on narrow experts. Train tiny hybrid first in FP32, then calibrate low-bit/codebook alternatives, or use QAT as a separately labeled arm. Prior art: hybrid SSM/attention, MoE, LUT-GEMM/AQLM; no novelty claim.

## Relative priority, before measurements

Higher qualitative scores mean a stronger initial hypothesis, not benchmark evidence. Quality is plausibility at equal adequate training, not a promised ranking.

| Candidate | Quality plausibility | P40 efficiency | Training simplicity | Scaling potential | DP4A fit | PCIe locality | Priority |
|---|---|---|---|---|---|---|---|
| A diagonal recurrence | Medium | High | High | Medium | High projections | High | 1: cheap falsification |
| B ternary vector recurrence | Medium-low on retrieval | High if LUT wins | Medium (QAT) | Medium | Medium, LUT alternative | High | 2 |
| C delta memory | High relative to small vector state | Medium-high | Medium | High | High projections | High state locality | 1: best link to current Qwen |
| D routed memory | Unknown/medium-high | Unknown | Low-medium | High capacity | High projections | High potential, routing risk | 3 |
| E compressed hybrid MoE | High | Medium | Medium-low | High | Medium/format-dependent | High with expert locality | 3 |

Run A and C first; B after primitive evidence; D/E only when a simpler control has learned. Diffusion is a useful later comparator for arithmetic intensity, not the first candidate to train on this server.

## Smallest decisive learning tests

1. Overfit16–64 examples with forward/backward parity: catches incorrect recurrence/STE/chunking cheaply.
2. Copy, delayed key-value associative recall, finite-state transitions and modular counting with held-out sequence lengths. Sweep vector/state size. Kill the claim that tiny recurrence preserves needed information if the matched Transformer succeeds and candidate systematically fails.
3. Fixed5M training tokens, identical tokenizer/data order; report held-out loss, speed/energy and state/weight traffic. A single undertrained run cannot establish architectural inferiority; three seeds and a modest learning-rate check for survivors.
4. Compare trained FP32→same architecture low-bit on fixed teacher-forced traces before free decoding. Approximately>3% PPL degradation or>2-point retrieval decline triggers investigation/rejection of that precision setting.
5. Optimize only the winning operator; use end-to-end latency at equal quality to promote50M. Novelty would require a comprehensive close-prior-work comparison, multiple tasks/seeds/scales/hardware, ablations and reproducible code, beyond this pilot.
