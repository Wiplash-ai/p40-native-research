# Colibrì on P40: ten hypotheses for eliminating expensive work

Date: 2026-09-09. Research baseline: repository `8e447340c7d88a6f5cc35b3986e6e9a634bc9801`, retained T24 experimental engine, pinned Qwen3.6 checkpoint. This report proposes experiments; it does not implement or execute them. All proposed P40 speed and quality outcomes are **`not_measured`**. Equations and cost scenarios are analytical, not benchmark results. No novelty is claimed.

> Repository reconciliation, later on 2026-09-09: the report was authored
> against its stated baseline. Later T28–T30 groupwise/top-K real-Qwen shadows
> completed and rejected the generic residual-capacity direction. T31 then
> attributed the remaining tail to gate/up input quantization (45/2,397 groups
> above 5%), while isolated post-SiLU down quantization had 0/2,397 failures.
> Accordingly, R1 should begin with gate/up input capture/replay and an
> influence-aware selector, rather than treating T28 as pending. See
> `research/results/T31-real-qwen-w4a8-stage-attribution-shadow.md`.

**Recommendation:** investigate a cheap bulk computation plus a small, explicitly measured correction. The most useful question is no longer “Can Pascal multiply these weights faster?” It is “Which parts of this particular token's computation actually require the full representation?” Start with sparse activation-error correction and low-rank error-response correction. In parallel as research, test whether co-selected experts share reusable computations and whether successive tokens have cheap innovations. Speculative verification is the strongest distinct route to doubling whole-model throughput, but its verifier should be falsified before building a draft.

## The assumptions worth challenging first

**An expert-kernel win is not a model-speed win.** T24 reports 64.04 ms/token in its decode timer, including 30.79 ms MoE and 24.38 ms DeltaNet. The 17.12 ms shared-MLP interval is inside MoE, not another additive cost. The engine reports 12.92 output tok/s on the fixed 64-output fixture; that rate is not the reciprocal of the decode-only timer. These are short, fixed-prompt observations, not a repeated multi-domain benchmark. [T24 result](research/results/T24-exact-qwen-deltanet-pair-64.md)

Holding all other work fixed gives the following optimistic Amdahl calculation:

`speedup = 64.04 / (64.04 - 30.79 + 30.79 / s)`

| Speedup of the entire inclusive MoE interval, s | Overall decode speedup |
| --- | ---: |
| 2× | 1.316× |
| 4× | 1.564× |
| Infinite; MoE takes zero time | 1.926× |

Routed experts alone have less headroom. CPU/GPU intervals overlap, so subtracting shared time does not establish an exclusive routed-kernel budget. A 2× overall proposal must also reach dense/shared projections, remove scheduling boundaries, or amortize an entire model pass over multiple accepted tokens. For example, accelerating a genuinely exclusive 75% of total time by 3× would yield 2× overall; we have not established that coverage for any proposed kernel.

**“94% kernel time” does not mean compute-bound.** T22's device events are mostly kernels, but kernels can be limited by weight reads, reductions, occupancy or instructions. Their sums also span both devices and prefill. DP4A does not shrink the existing W4 weight payload. T20's synthetic 1.694× result is evidence for that control, not the production-shaped implementation's attainable speed. [T22](research/results/T22-exact-qwen-async-expert-profile.md), [T20](research/results/T20-w4a8-dp4a-full-routed-mlp-gpu0-125w.md)

**T21 diagnosed a failed formulation, not the cause of failure.** Its 2,397 records aggregate `rows × width` within device groups. They are not 2,397 independently evaluated experts. The 36.37% maximum relative L2 error and 288 groups above 5% reject the tested formulation, but do not reveal input-quantization error versus post-SwiGLU error, sensitive experts, cancellation, or subsequent logit impact. The exact stdout passed because the original W4A32 path supplied the returned values. At this report's baseline, T28's groupwise scaling was unmeasured; later T28–T31 evidence is reconciled above. [T21](research/results/T21-real-qwen-expert-w4a8-shadow.md), [shadow implementation](patches/0004-qwen36-w4a8-dp4a-shadow.patch), [T28 design](research/experiments/021-real-qwen-groupwise-w4a8-shadow.md)

**CPU bit identity is a valuable control, not a complete definition of useful numerical accuracy.** Keep that control and every existing gate. In separately declared approximation experiments, assess the error's effect on logits, routing, recurrent state and language loss. Neither a changed reduction tree nor low local L2 alone proves a candidate acceptable. T06 already demonstrated that correct text can coexist with severe CUDA failures and CPU fallback. [T06 history](research/experiments/004-qwen-deltanet-exact-integration.md)

## Mathematical reference

For each of the unchanged eight selected experts, use column-vector notation:

`g_e = G_e x;  u_e = U_e x;  h_e = SiLU(g_e) ⊙ u_e`

`y_e = D_e h_e;  z = Σ_e p_e y_e + γ y_shared`

Here `x ∈ R^2048`, `G_e,U_e ∈ R^(512×2048)`, and `D_e ∈ R^(2048×512)`. The router coefficients `p_e`, shared gate `γ`, expert identities, tokenizer and checkpoint stay fixed in the proposed runtime controls. `W` below means the mathematical matrix represented by the existing quantized weights and scales, not an unavailable original FP32 checkpoint. Arithmetic uses FP32 unless explicitly stated.

The routed linear work is `3 × 8 × 512 × 2048 = 25,165,824 MACs/layer`. Its bare active packed weights are about 12 MiB/layer, or 480 MiB across 40 layers before scales/reuse. These are operation/storage counts, not measured DRAM traffic.

Quantization enters twice. If gate/up errors are `δg,δu`, and `η` is the quantization error of the resulting approximate hidden vector before down, then:

`δy = D [u ⊙ (φ(g+δg)-φ(g)) + φ(g) ⊙ δu + (φ(g+δg)-φ(g)) ⊙ δu + η]`, with `φ=SiLU`.

Consequently, repairing the down input cannot generally undo earlier nonlinear error. The first-order sensitivity is `J(x) = D [diag(u⊙φ'(g))G + diag(φ(g))U]`; a Jacobian score is a useful diagnostic, but is not a certified bound on the omitted nonlinear remainder.

## Ten hypotheses

### H1. Correct the activation residual that matters, not merely the largest activation

**Challenged assumption:** every value must receive the same precision, and magnitude identifies the values needing protection.

For a dequantized Q8 approximation `x̂`, define `r=x-x̂`. The identity is:

`Wx = W x̂ + W r`.

Execute the bulk with DP4A and correct only a small coordinate set `S` in FP32:

`ỹ = W x̂ + Σ_(j∈S) W[:,j] r_j`.

For down, rank by `|r_j| ||D[:,j]||₂`, rather than `|h_j|`. The omitted linear error obeys `||D r_notS||₂ ≤ Σ_(j∉S) |r_j| ||D[:,j]||₂`. For gate/up, begin with a cheap stored joint-column sensitivity; use the expensive full-Jacobian ranking only as an offline oracle. Correct gate and up before SiLU, then independently correct hidden quantization before down.

Test a second, distinct arm that excludes a few large values before selecting the bulk quantization scale, then restores them. Protecting outliers before absmax can improve every remaining value; correcting residuals afterward targets the errors actually produced. These mechanisms should not be conflated.

Eight corrected coordinates at each boundary add `2×512×8 + 2048×8 = 24,576` FP32 MACs/expert, less than 1% of its dense MAC count. Scattered column reads, selection and extra launches can nevertheless dominate. Fuse correction while weights are available or cache only demonstrated hot columns; do not expand all W4 weights into FP32.

**Falsifier:** if even an oracle choosing 16 coordinates at each boundary cannot remove the held-out error tail, the “tiny sparse correction” claim fails. **2× route:** an enabling technique for faster W4A8 and separately Q8A8 dense/shared projections; not a standalone 2× model claim.

Outlier decomposition has precedent in [LLM.int8()](https://arxiv.org/abs/2208.07339) and [QUIK](https://arxiv.org/abs/2310.09259). The proposal here is fixed-W4 residual correction selected by the complete expert's sensitivity, using FP32 corrections on Pascal.

### H2. Spend a second INT8 digit only on difficult groups, experts and tokens

**Challenged assumption:** the only alternatives are one Q8 activation or an entire FP32 fallback.

Represent an activation as `x = a q₀ + b q₁ + ε`, where `q₀=round(x/a)` and `q₁=round((x-aq₀)/b)`. With unclipped symmetric absmax quantizers, exact scale arithmetic and round-to-nearest, `||ε||∞ ≤ a/508`; the single-digit bound is `a/2`. Actual FP32 scale/rounding behavior must be tested.

Compute `W x ≈ a·dot(W4,q₀) + b·dot(W4,q₁)`, using two DP4A accumulators but one unpacked weight load. Select second digits only where residual norms, stored sensitivities and the fixed router weight predict excessive output error. Unassessed experts use the exact fallback. Apply the same decision separately before gate/up and before down.

If a fraction `f` of equally sized groups needs refinement, dot work is approximately `1+f` times the first pass, excluding selection/reduction overhead. An expert with small `p_e` may need less refinement, but correlated errors from the common input can add coherently. Keep every selected expert; do not interpret a small router weight as permission to drop it.

**Falsifier:** if two digits everywhere plus substantial FP32 correction are needed, or packed-weight reuse cannot preserve a complete-operator advantage, abandon this path. **2× route:** selective refinement could keep integer computation viable across more than MoE; it does not reduce W4 traffic. This residual-expansion control is a proposed extension, not a measured result or a claim that published W4A8 systems support Pascal. [DP4A semantics](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a)

### H3. Approximate the quantization-error operator, not the model matrix

**Challenged assumption:** useful low-rank structure must exist in `W` itself.

Keep `W` unchanged. Fit `L=AB`, of small rank, to predict only its response to actual quantization residuals:

`ỹ = W Q(x) + A[B(x-Q(x))]`.

The remaining error is `(W-L)r`. Fit using `min_(rank(L)≤ρ) E||Wr-Lr||²`, equivalently `||(W-L) C_r^(1/2)||F²`, where `r` is the activation residual, `ρ` is the rank budget, and `C_r` is the residual second moment. Regularize poorly sampled directions; ordinary activation PCA or weight SVD optimizes a different objective.

Use a joint `[G;U]` corrector so the small input projection repairs both branches before SiLU. Give down its own corrector. Joint gate/up plus down at rank eight costs 45,056 FP32 MACs and 176 KiB of factors per expert. Across 10,240 experts that is approximately 1.72 GiB of additional factor storage. Small arithmetic does not mean negligible bandwidth.

**Falsifier:** rank 4/8/16 improves calibration but not held-out residuals, or low-rank correction fails on the hard tail. Quantization noise may be too diffuse for this to work. **2× route:** the same technique applies to dominant dense Q8 projections as a separately evaluated Q8A8 candidate, potentially crossing MoE's Amdahl limit.

[ARHQ](https://arxiv.org/abs/2605.00140) motivates residual-aware low-rank fitting, but its report covers a different Qwen model/projection scope and substantially larger correction rank. Our proposed adaptation evaluates `L(x-Q(x))` directly, avoiding a new quantization of `W-L`. Rank-eight success on real Qwen experts is an untested hypothesis.

### H4. Change the order of hidden neurons before adding more quantization scales

**Challenged assumption:** adjacent stored channels are a good activation-quantization group.

For a hidden-neuron permutation `P`, set `G'=PG`, `U'=PU`, `D'=DPᵀ`. Then:

`D'[SiLU(G'x) ⊙ U'x] = D[SiLU(Gx) ⊙ Ux]` in real arithmetic.

Jointly reorder gate/up rows, their scales and down columns. Cluster neurons by calibrated range and outlier frequency so a few large channels do not set the step size for unrelated small channels. This can retain the same number of groups as T28. The packed W4 values can be rearranged losslessly; the represented weights need not be requantized. Reduction order can still change, so algebraic preservation is not a bitwise-output guarantee.

**Falsifier:** at the same scale count, a calibration-derived permutation fails to improve held-out error against both original order and a random permutation. This is a cheap accuracy enabler, not an independent 2× claim. [RPTQ](https://arxiv.org/abs/2304.01089) supplies the range-reordering precedent.

Do not substitute arbitrary smoothing/rotation and call it equally lossless: multiplying existing W4 columns by different factors generally cannot be absorbed into their per-output scales. Nor is `SiLU(cg)=c·SiLU(g)`. [SmoothQuant](https://arxiv.org/abs/2211.10438) and [QuaRot](https://arxiv.org/abs/2404.00456) are useful separate representation experiments, not free switches on the existing container.

### H5. Make co-selected experts corrections to shared computations

**Challenged assumption:** eight expert identities require eight independent traversals of every matrix.

Start with local anchors rather than one average across all 256 experts. For an expert in cluster `c`, approximate each matrix by an anchor plus rank-`r` residual: `G_e≈G_c+B_ge A_ge`, and likewise for `U_e,D_e`. Compute the shared gate/up projections once per touched cluster, add each expert's residual before SiLU, then exploit the exact linear identity for the down anchor:

`z_c = D_c [Σ_(e∈S_c) p_e h_e] + Σ_(e∈S_c) p_e B_de(A_de h_e)`.

The `h_e` remain different nonlinear vectors. Only their common linear down map moves outside the sum. Averaging expert weights through SiLU is invalid. Align gate/up rows and down columns with the same permutation before judging residual compressibility.

With `C` touched clusters, the ideal linear cost is `3 C D M + 3 K r(D+M)`, versus `3 K D M`. For `D=2048,M=512,K=8,r=64`, one/two/three touched clusters give 3.56×/2.46×/1.88× fewer MACs. Thus low residual rank and frequent co-selection within at most two clusters must both hold for a credible 2× expert-compute target. Higher-precision factors and memory traffic can erase it.

A more aggressive architecture is shared input/output bases: `G_e≈A_ge B`, `U_e≈A_ue B`, `D_e≈C A_de`. Compute `Bx` once and `C[Σ p_e A_de h_e]` once. Rank 128 gives 12× fewer ideal expert MACs at these dimensions, but demands much stronger shared structure. If existing weights resist this, train a tiny model with that structure and evaluate it under the existing learning protocol; do not claim a converted checkpoint preserves quality.

**Falsifier:** actual routing touches too many anchors, or low-rank residuals fail held-out mixture reconstruction. This remains an MoE-stage gain until it changes wider execution. [D²-MoE](https://arxiv.org/abs/2502.17298) and [TD-MoE](https://github.com/ust-xu/TD-MoE) are precedents. The recent [LorExperts/BTExperts preprint](https://arxiv.org/abs/2608.07814) specifically argues against one global shared base; its compression results are not proof of preserved quality under our gates.

### H6. Predict negligible neuron contributions before computing gate and up

**Challenged assumption:** fixed top-eight routing implies all 512 neurons in every selected expert must be fully evaluated.

For down column `d_ej`, omitted neurons obey:

`||δz||₂ ≤ Σ_(e,j omitted) |p_e|·|h_ej|·||d_ej||₂`.

Use a cheap predictor or coarse bounded estimate to identify aligned groups of 16/32 neurons whose entire contribution is small. Compute gate/up/down only for survivors and refine ambiguous groups. Bounds on gate/up must account for SiLU's negative, non-monotonic region; a negative gate is not a zero activation.

**Falsifier:** first give the idea an oracle that knows every exact hidden activation. If retaining only half the neuron groups already violates the error gate, no cheaper predictor will rescue the 2× arithmetic proposition. Then price predictor misses and repairs on held-out prompts. Computing all gate/up before skipping down removes at most one of three matrix traversals: its ideal arithmetic ceiling is 1.5×, not 2×.

[Deja Vu](https://proceedings.mlr.press/v202/liu23am.html) and [PowerInfer](https://arxiv.org/abs/2312.12456) motivate contextual/neuron sparsity, but do not establish it for stock Qwen SwiGLU. Replacing SiLU with ReLU or retraining for block sparsity belongs to the tiny-model architecture lane.

### H7. Compute token-to-token innovations instead of full projections

**Challenged assumption:** a previously computed projection becomes useless at the next token.

For fixed `W`, `W x_t = W x_ref + W(x_t-x_ref)`. Cache projection outputs for each layer/expert and use sparse or low-rank innovations. Preserve each expert's reference activation and age; use the last invocation of that expert, not blindly the previous token. Reset on sequence boundaries.

For an expert, update cached `g,u` from the input innovation, recompute SiLU, then update the cached down result through `D(h_t-h_ref)`. A first-order alternative is `y_t≈y_ref+J_ref δx`, but forming an ordinary full Jacobian-vector product still requires gate/up/down work. It saves nothing unless the innovation or Jacobian is cheaper to represent.

Sparse column corrections potentially avoid both matrix reads and arithmetic. Quantizing a dense innovation into another dense GEMV does not. Cache only active/recent experts initially; account for eviction, fallback, drift and hot-cache capacity. High cosine similarity of inputs is insufficient evidence: it does not bound `Wδx` or nonlinear output error.

**Falsifier:** oracle sparse-innovation replay cannot recover the expert output cheaply, route churn makes most references unusable, or exact refreshes consume the saving. This can reach 2× overall only if useful across enough dense/shared as well as routed work. Temporal-activity observations in [NeuroPrefetcher](https://arxiv.org/abs/2608.22643) concern weight availability, not equality of expert outputs; this proposal must test value reuse separately.

### H8. Eliminate DP4A and activation quantization with bitplane lookup algebra

**Challenged assumption:** the P40's best low-bit primitive must be an integer dot product.

For signed W4, `q=b₀+2b₁+4b₂-8b₃`. For each block of eight FP32 input values, build 256 subset sums `T[m]=Σ_j bit_j(m)x_j`. A weight-row block becomes:

`dot(q,x) = T[m₀]+2T[m₁]+4T[m₂]-8T[m₃]`.

Reuse these tables across gate/up outputs and selected experts sharing the same input. Down requires separate tables for each expert's hidden vector. Apply existing weight scales. This retains the represented W4 weights and avoids A8 error, but reassociates floating-point sums.

The complete D=2048 table bank would occupy 256 KiB, exceeding one P40 block's shared-memory capacity. Tile it: eight tables occupy 8 KiB, leaving room for other state. Price table construction, repeated construction across blocks, index extraction, shared-memory bank conflicts and reductions. Four lookups for eight inputs are not automatically faster than two DP4As. `__popc` cannot compute arbitrary multi-bit signed products by itself.

**Falsifier:** the transfer-inclusive full MLP, including all table builds, does not materially beat the same W4A32 control. The real engine uses two's-complement nibbles; the synthetic benchmark's `code−8` packing is different. Validate all 16 codes before reusing any packing fixture. [Pinned decoder](https://github.com/JustVugg/colibri/blob/12a5c464b5c1f8292d578c62458706bc32d6ac95/c/backend_cuda.cu), [LUT-GEMM](https://arxiv.org/abs/2206.09557), [T-MAC](https://arxiv.org/abs/2407.00088)

**2× route:** possible operator improvement without activation quantization; model-wide gains require reaching other phases. Q8 needs eight bitplanes, so extending the same algorithm to dense weights is not automatically profitable.

### H9. Treat approximate computation as a draft, then verify several tokens together

**Challenged assumption:** an approximation must be accurate enough to become the final model output to be useful.

A rejected approximate expert formulation could still inform a cheap draft, provided an exact target verifies it and rejection is handled correctly. Alternatively begin with prompt/ngram proposals so no draft training or download is required. For greedy generation, commit only the target-verified prefix and its correct continuation. Sampling requires the proper acceptance/resampling algorithm, not merely an argmax comparison. [Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html)

The governing quantity is:

`speedup = E[correctly emitted tokens] × T_step / (T_draft + T_verify + T_commit/rollback)`.

A four-token draft does not give 4× if verification performs four ordinary serial forwards. With one free verified bonus/correction token accounted for, the optimistic yield under independent acceptance probability `a` is `1+a+a²+a³+a⁴`. At `a=0.8` this is 3.3616 tokens; 2× needs all costs below 1.6808 baseline steps. These are hypothetical conditions, not acceptance measurements.

Qwen's recurrent state cannot roll back by deleting KV entries. Preserve recurrent matrices, convolution rings, positions, KV and sampling state. [TreeWY](https://arxiv.org/abs/2608.20961) provides recent Gated DeltaNet-specific reconstruction research, but its memory improvements and modern implementation do not establish a P40 latency win. Its author explicitly reports wider trees are not yet a throughput win.

**Falsifier:** give the verifier perfect future tokens first. If even this oracle cannot verify and commit a block cheaply enough for 2×, do not build a draft. Existing Qwen `DRAFT`/MTP support remains `unsupported` in the audited engine; this is an implementation project, not a flag.

### H10. Partition the sum of nonlinear neuron paths, not expert identities

**Challenged assumption:** `expert_id modulo GPU_count` is the natural mathematical decomposition.

For disjoint hidden-neuron sets `J₀,J₁`, compute on each GPU:

`z_a = Σ_e p_e D_e[:,J_a] [SiLU(G_e[J_a,:]x) ⊙ U_e[J_a,:]x]`, then `z=z₀+z₁`.

Each GPU owns complete gate/up/down paths for its neurons. Add a partition of the shared expert to the same local sum. This retains all experts and arithmetic in real numbers while balancing their work and returning one partial hidden vector per GPU instead of every expert output. Reassociation still needs numerical validation. H5's shared output basis could further reduce partial-vector dimension.

The gain would come from balance, fewer host reductions and integration with device-resident layer state, not from aggregating already-small PCIe bytes. Splitting a dependent bulk path onto one GPU and its mandatory pre-SiLU correction onto the other may add a synchronization barrier and lose. Likewise, a layer pipeline does not execute two dependent halves of a single token concurrently; aggregate request throughput and single-sequence latency are different targets.

**Falsifier:** replay actual per-device expert counts and measured service times. If avoided imbalance/coordination is too small to pay for the extra boundaries, stop. Under an illustrative independent 50/50 eight-expert assignment, balancing max occupancy from 5.094 to 4 provides only 1.273× ideal stage improvement. No 2× claim follows. [Current execution map](research/colibri_execution.md), [T26 rejected shared overlap](research/results/T26-exact-qwen-shared-pair-64.md)

## Top five experiments, ranked by payoff versus difficulty

This ranking values the cost of reaching a decisive answer, not a speculative best-case speed headline. Confidence describes the rationale for testing, not probability of achieving 2×. The `R1`–`R5` labels are local to this report and do not replace execution task IDs.

| Rank | Experiment | Expected payoff if it survives | Difficulty: falsifier / eventual integration | Why this order |
| --- | --- | --- | --- | --- |
| 1 | R1: influence-ranked sparse correction, H1 | Establish whether a tiny FP32 fraction can make fast W4A8 useful; possible extension to dense Q8A8 | Low–medium / medium | Closest to T28, no weight refit, high diagnostic value even on failure |
| 2 | R2: low-rank quantization-error response, H3 | Cheap correction even when errors are not coordinate-sparse; reaches dense/shared projections | Medium / medium–high | Tests a different structural assumption from outlier clipping |
| 3 | R3: co-activation anchors, H5 | More than 2× expert MAC reduction under explicit rank/cluster conditions; byte savings require validated factor precision | Medium / high | Eliminates matrix work rather than changing instruction type |
| 4 | R4: sparse temporal innovations, H7 | Potentially avoid repeated weight reads in both dense and expert paths | Low–medium / high | A cheap oracle can kill an appealing but often false reuse assumption |
| 5 | R5: oracle speculative verifier, H9 | Direct route to 2×+ useful whole-model throughput without accepting approximate target output | Medium–high / very high | Biggest overall upside, but current recurrent/serving implementation is a substantial obstacle |

H4 is a useful inexpensive control within the quantization investigation. H6's oracle sparsity curve is worth collecting from the same trace, but a deployable predictor is uncertain. H8 is the strongest alternative if A8 accuracy remains irreducible; it ranks below the five because unchanged weight traffic and lookup costs may erase its advantage. H10 should be driven by measured imbalance, not intuition about two GPUs.

### R1 — sparse correction with a complete SwiGLU error budget

**Smallest experiment:** add a bounded, default-off capture/replay interface after the existing T28 decision, as its own task. Current raw logs contain aggregate errors, not the tensors needed for this test. Capture exact inputs and intermediate/output vectors for a small, predeclared selection of early/middle/late layers on public prompts. Capture complete device groups and all eight routed experts for each token/layer bundle used to score mixture errors, even when only a subset receives the candidate correction. Return the original exact path throughout capture.

Implement a CPU replay, initially one expert at a time, that uses actual packed W4 values and original scales. Compare input-only quantization, hidden-only quantization and both. With all other settings frozen, sweep 0/4/8/16/32 corrected coordinates at each boundary. Compare magnitude selection, residual selection, stored-column sensitivity and an oracle influence ranking. Add clipping-before-scale only as a separate comparison.

**Proposed screening gate, fixed before looking at new results:** on the held-out replay, require all finite values, p99 per-expert relative L2 ≤1%, no replay group above the existing 5% gate, and reduced absolute/routed-sum error. This is a new experiment's stringent screening rule, not a retrospective change to T21 or a language-quality guarantee. Report low-reference-norm cases separately with the same fixed denominator floor.

**Kill:** even the oracle needs more than 16 corrected coordinates at both boundaries to meet the screen; or the deployable selector needs extensive exact fallback. A few hard experts may instead justify explicit exact fallback, whose measured cost must enter the decision.

**Next only if numerical success:** one full routed-MLP CUDA replay with fused bulk/correction and INT64 checks of all integer accumulators. Include selection, quantization, correction reads, SiLU, requantization, synchronization and transfers. Seek ≥2× operator speed as a stretch gate; classify a smaller gain honestly. Then test an independent dense/shared Q8A8 operator, since routed-only gains cannot double T24.

**Suggested implementation surface:** extend the shadow path's optional capture in `patches/0009-qwen36-w4a8-groupwise-shadow.patch`; add a proposed `scripts/replay_expert_error.py` and a later standalone benchmark. No such implementation is included in this report.

### R2 — a small low-rank model of the error itself

**Smallest experiment:** reuse R1's immutable capture with a separate calibration/holdout split. For a few adequately sampled experts, form residual inputs and exact projection errors. Fit rank 4/8/16 joint gate/up correctors and separate down correctors. Compare residual-aware fitting against ordinary weight-SVD and an equal-storage sparse correction. Fit in FP32/FP64 on bounded CPU data; do not first convert the whole checkpoint.

The deployed coefficients must be computable as `B r`; a fit requiring the exact live `W r` as input has cheated. Test down correction on hidden vectors produced by the candidate's corrected gate/up, not only on oracle hidden vectors. For rare experts without adequate calibration, record `inconclusive` and retain exact execution.

**Kill:** rank 16 does not meet R1's held-out screen, performance depends on calibration prompt memorization, or factor traffic/launches consume the bulk-path saving. A flat held-out error spectrum is useful negative evidence: there is no tiny error subspace to exploit.

**Next only if success:** a fused full-expert replay with factors resident. Measure total extra bytes and latency; rank-eight factors already add roughly 1.72 GiB if replicated across every expert. A shared residual-input basis across experts could reduce storage, but requires another independently tested approximation. Separately test a representative DeltaNet or shared-Q8 projection to determine whether this idea can address enough time for a 2× overall target.

**Core decision:** prefer this over sparse correction only if it repairs diffuse errors at lower measured total cost. Do not assume the sparse and low-rank gains multiply; test their combination separately if both survive.

### R3 — one common computation plus small expert-specific differences

**Smallest experiment:** first collect route IDs and coefficients, with no matrix approximation. Cluster from calibration co-activation and count touched clusters on held-out tokens. Then inspect 8–16 frequently co-selected experts from one layer, retaining original top-eight selection. Jointly align each expert's gate/up hidden rows and corresponding down columns; fit rank 32/64/128 deltas against local anchors using actual W4-dequantized weights.

Replay the entire nonlinear expert mixture using the H5 equations. Count anchor computations actually reused per token. Weight similarity alone is insufficient; co-activation alone is also insufficient because experts may be complementary rather than redundant.

**Kill:** rank 64 requires three or more touched anchors often enough that average modeled work exceeds half baseline, or the candidate fails the held-out numerical screen. Require an ideal MAC ratio comfortably above 2×—for example ≥2.5× as a proposed engineering gate—before accepting extra launches and factor-format costs. If a compression reduces disk storage but not active computation, reject it for this objective.

**Next only if success:** one-layer guarded CUDA replay, including anchor work, residuals, nonlinearities and routed reduction. Require measured full-layer improvement at declared quality. At rank 64, eight experts' residual factors contain 3,932,160 coefficients: 15 MiB in FP32 before anchors. One/two W4 anchors raise that to 16.5/18 MiB, versus the original 12 MiB active W4 payload. Even Q8 factors plus two W4 anchors total 6.75 MiB, only 1.78× fewer bytes. These counts exclude scales and cache reuse. Factor quantization is another numerical experiment, not an assumed implementation detail.

**Architecture alternative:** if the pretrained weights cannot support the factorization, use H5's shared encode/nonlinear experts/shared decode as a tiny trained-model experiment. Use the fixed learning protocol and equal-quality comparisons. That is an alternate architecture, not a runtime optimization of the existing checkpoint.

### R4 — oracle temporal reuse before writing a cache

**Smallest experiment:** capture contiguous teacher-forced token positions, reset markers and expert visit histories. Random independent activation rows cannot test this hypothesis. For each eligible projection, compare the current input with its cached reference and form exact innovations. Use an oracle to select sparse columns; evaluate 8/16/32/64/128 selected input dimensions and neuron-block variants. Repeat after the actual nonlinear hidden update for the down projection.

Start with within-expert projection reuse, not reuse of the whole residual layer. Always run the original recurrence in the control; expert output similarity does not justify skipping recurrent-state evolution. Report eligibility, cache-hit age, changed-dimension fraction, error tails and required refresh frequency. Include cold starts and route churn in every speed model.

**Kill:** the oracle cannot meet the numerical screen with less than half the original weighted work after refreshes and selection, or most tokens require full recomputation. A useful explicit rule is `a + (1-a)s + o < 0.5`, where `a` is the full-refresh fraction, `s` is measured sparse-update cost relative to a full call, and `o` is amortized lookup/selection overhead. Dimension sparsity is not a substitute for measuring `s`.

**Next only if success:** implement a bounded cache and sparse-column replay for one projection family. Stress reference age, sequence reset, long teacher-forced drift and fallback. Move to full-model evaluation only if success covers sufficient dense/shared time as well as MoE; a beautiful expert-only cache still faces the Amdahl ceiling.

### R5 — can an exact verifier amortize the target at all?

**Smallest experiment:** first implement state snapshot/restore and a two-token reject/commit CPU fixture in the isolated source. Preserve every recurrent state, convolution ring, attention KV position and sampling state. At each possible rejection boundary, continuing from restored state must match an isolated exact control. Do not start by finding or training a draft model.

After that correctness fixture, use known future control tokens as an oracle draft of length 2 and then 4. Measure teacher-forced block execution, expert grouping, all output-head work, commit and rollback against serial T24-derived execution. Qwen's single-outstanding expert buffers and row limit require explicit bounds; simply sending a larger batch is not a verifier implementation.

**Kill for the 2× target:** even perfect proposals cannot exceed 2× after measured verification/commit cost. For a four-token proposal with a genuinely available fifth bonus/correction token, verification plus commit must cost less than 2.5 baseline steps even with free drafting. If verification costs four steps, its absolute best ratio is only 5/4. Without a bonus token, use four in the numerator. Real draft cost and rejection lower both ceilings.

**Next only if success:** test free prompt/ngram proposals, then a separately justified compatible draft. Measure actual accepted tokens, not predicted acceptance. An approximate copy using our own corrected low-bit path is worthwhile only if it is much cheaper than the verifier and preserves enough acceptance. Do not add a second giant checkpoint by default.

Gated DeltaNet's transition contains a rank-one update, not merely diagonal decay; generic diagonal-SSM tree algorithms are not drop-in solutions. TreeWY is a relevant mathematical reference if snapshots prove costly, but a small exact chain with snapshots is the cheaper first falsifier. Its kernels/framework must be reimplemented or independently shown compatible with sm_61. [TreeWY author RFC](https://github.com/vllm-project/vllm/issues/54080)

## A common protocol that makes these answers credible

**Capture what can distinguish the hypotheses.** Keep layer, expert, device-group and token-position IDs; original route coefficients; input, gate, up, hidden and output vectors; original packed weights/scales; and exact model/tokenizer/prompt identities. Do not infer these tensors from T21's aggregate logs. For capacity sizing, 32 experts × 256 rows storing `2048+512+512+512+2048` FP32 values per row is 176 MiB, plus about 48 MiB W4 weights and 0.375 MiB scales. This is an example allocation, not a promise that 32 selected experts cover complete mixtures. Admit complete token/layer bundles and reduce their count when necessary to enforce a 512 MiB file budget including metadata and extra traces; process one expert at a time. Mark group/routed-sum metrics `inconclusive` for any incomplete bundle rather than summing only its captured experts. Filling the budget may need more than one short canary; stop at the workload envelope instead of extending a run to obtain samples.

Split by prompt/document before selecting corrections or fitting factors, with public text spanning prose, code, reasoning and repetitive continuations. Include an unbiased held-out sample plus a separately labeled hard-case diagnostic sample. Do not cherry-pick only the 288 known failures or fit and evaluate on adjacent rows from the same prompt. Sparse calibration for an expert means `inconclusive`, not a guessed pass.

**Separate algebra, local numerical accuracy and language behavior.** First verify CPU reference identities and signed packing. Then compare quantized integer accumulators to INT64 references, and complete outputs to the existing W4A32/Q8A32 controls. Report per-expert and per-device-group errors, fixed-coefficient routed-sum error, and absolute error relative to the residual-stream norm. Local triangle bounds can be conservative; cancellation may help or hurt. Neither proves final-logit stability.

For any later separately scoped approximate model replay, teacher-force identical tokens and measure held-out NLL/perplexity, logit KL, top-k/argmax agreement, downstream routing differences and long recurrent-state drift. Keep routing frozen only for local ablations; subsequently evaluate the original router on candidate activations so changed later-layer selections are visible. A proposed broader screen is ≤1% relative held-out perplexity increase with no material domain/task regression; the eventual acceptance criterion must be fixed before that experiment. An exact-output shadow run cannot satisfy this approximate-path gate by itself.

**Measure the computation someone would actually run.** Include bulk quantization, scale creation, selector, corrections, table construction, factor reads, transfers, synchronization, cache refreshes and fallbacks. Use actual selected-kernel records. Separate primitive time, decode, prefill, loading, request queueing and useful output rate. After brief canaries, use matched alternating control/candidate repetitions, median and range; do not treat one short sample as a general winner. Collect energy only through available telemetry and label unavailable measurements explicitly.

**Stay within the existing execution contract.** This report authorizes no hardware dispatch. Preserve production and use isolated pinned sources. Before future workload tasks, reconcile the checked-in remote runners with the documented independent watchdog, host-wide lock, stale/slope checks and cleanup requirements, then satisfy current thermal gates. Begin with bounded canaries and matched power caps. No giant downloads, global runtime changes, or automatic task advancement follow from this research.

At the report baseline, the current execution task was **T28**. Its comparison
and T31 attribution have since completed; the next explicitly identified task
is a bounded input-side capture/replay that can falsify R1 and supply data for
R2–R4. R5 still starts with CPU state fixtures and a cost ceiling, not a draft
deployment.

## Research and verification record

Primary papers, author repositories, the pinned Colibrì decoder and local experimental evidence were reviewed for this report. Newer preprints are used as hypotheses and mathematical precedents, not validated Qwen3.6/P40 recipes. Modern Tensor-Core and CPU-LUT benchmark rates were not transferred to Pascal. Pascal's DP4A and shared-memory constraints were checked against [NVIDIA's CUDA 12.9.1 Pascal guide](https://docs.nvidia.com/cuda/archive/12.9.1/pascal-tuning-guide/index.html) and the repository's device records.

CPU-only checks in this session verified eight algebraic relationships, including sparse residual repair, low-rank residual repair, the two-digit error bound, hidden permutations, neuron partitioning, common-down factoring and the full SwiGLU perturbation expansion; 100 random signed-W4 bitplane cases also passed. These are toy mathematical checks, not model validation. No server access, model inference, GPU kernels, training or performance benchmarks were run.
