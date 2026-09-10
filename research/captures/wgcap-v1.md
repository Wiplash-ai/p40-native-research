# WGCAP v1 — bounded routed-expert gate/up capture

`WGCAP` is the self-contained binary sidecar for T32.  It exists to make an
offline replay of the **actual** Qwen routed-expert gate/up inputs possible;
it is not a model format, a new runtime cache, or a production telemetry
format.

## Fixed T32 collection

The guarded T32 run captures exactly:

| Selector | Fixed value |
| --- | --- |
| Model geometry | `D=2048`, `I=512` |
| Activation quantizer | signed Q8, absmax scale per 256 coordinates |
| Decode positions | `0, 1, 2, 3` after the fixed prompt prefill |
| MoE layers | `0, 20, 39` |
| Routed experts | all eight top-k experts at every selected `(position, layer)` |
| Maximum records | `4 × 3 × 8 = 96` |
| Maximum artifact size | 512 MiB |

Decode positions 0–1 are the calibration partition and positions 2–3 are the
holdout partition.  The selector is fixed before the run: capture code must
not inspect approximation error, choose an expert, or change the returned
model values.

Each record is self-contained so replay does not need to reopen a checkpoint:

- routing provenance: layer, decode position, device, expert ID, top-k rank,
  route weight, and the size of the original top-k bundle;
- the exact FP32 input `x`, its captured groupwise-Q8 values and scales;
- exact FP32 gate `g`, up `u`, SwiGLU hidden `h`, and exact down output `y`;
- original packed signed-nibble W4 matrices for gate/up/down, plus their
  per-row FP32 scales.

The packed nibbles use Colibri's actual two's-complement layout: low nibble is
the even input coordinate and high nibble is the odd input coordinate.  Decode
as `v < 8 ? v : v - 16`; do **not** use the synthetic control's `code - 8`
layout.

At the T32 shape, one record is 1,609,800 bytes and the complete 96-record
sidecar is 154,540,864 bytes including its 64-byte header.  The cap therefore
has more than 3× headroom while still preventing an unrestricted tensor dump.

## Binary layout

All integers and floats are little-endian.  The reader rejects a non-little
endian marker, truncated data, trailing data, inconsistent dimensions, an
oversize count, or any record whose payload geometry differs from its header.

Header: `8s + 14 × u32` (`64` bytes)

| Field | Value |
| --- | --- |
| magic | `WGCAP01\\0` |
| version | `1` |
| byte order marker | `0x01020304` |
| hidden, intermediate | `2048`, `512` |
| activation group width | `256` |
| record bytes | fixed function of the geometry below |
| expected records | `96` |

Record header: `8 × u32 + f32 + u32` (`40` bytes), followed by fields in this
order:

1. `x[D]` FP32
2. `q[D]` signed int8
3. `q_scale[ceil(D/256)]` FP32
4. `g[I]`, `u[I]`, `h[I]` FP32
5. `y[D]` FP32
6. packed `G[I,D]`, `U[I,D]`, `Down[D,I]` W4 bytes
7. `G_scale[I]`, `U_scale[I]`, `Down_scale[D]` FP32

The header's `record_bytes` includes its 40-byte record header.  The fixed
record header carries `layer`, `decode_step`, `device`, `expert_id`,
`route_rank`, `route_bundle_size`, the shared gate/up packed-weight byte count,
the down packed-weight byte count, router weight, and a reserved zero field.

## Correctness and provenance gates

The T32 remote guard must require all of these before it calls the capture a
pass:

1. The existing exact stdout SHA-256 remains unchanged.
2. Capture summary reports 96 records and zero source-side capture errors.
3. The artifact SHA-256 and byte count match the guard result.
4. The strict reader accepts the file and finds exactly one record for every
   `(decode_step, layer, route_rank)` tuple.
5. Every selected tuple has eight distinct top-k ranks and a bundle size of
   eight.

T32 is a data-collection run, not a latency result.  Its extra copies and raw
gate/up kernel are deliberately excluded from performance comparisons.
