// T20: isolated full routed-MoE W4A8/DP4A MLP control.  No model state.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int kExperts = 4;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kTile = 8;
constexpr const char* kProfile = "w4a8-dp4a-expert-mlp-4x-2048-512-2048";
constexpr const char* kSchema = "t20-w4a8-dp4a-expert-mlp-v1";

struct Options { std::string profile; int gpu = -1, repetitions = 0, calls = 0, cap = 0, seed = 0; bool dry = false; };
struct Matrix { std::vector<uint8_t> w; std::vector<float> s; int rows, cols; };
struct Device {
  uint8_t *gw = nullptr, *uw = nullptr, *dw = nullptr;
  float *gs = nullptr, *us = nullptr, *ds = nullptr, *input = nullptr, *ins = nullptr,
        *gate = nullptr, *up = nullptr, *hidden = nullptr, *hs = nullptr, *output = nullptr;
  int8_t *inq = nullptr, *hiddenq = nullptr;
  int32_t *gatei = nullptr, *upi = nullptr, *downi = nullptr;
};
struct Result { std::vector<float> out; std::vector<int8_t> inq, hiddenq; std::vector<int32_t> gate, up, down; };
struct Error { double e = 0, ref = 0, max_abs = 0, max_rel = 0; };

[[noreturn]] void usage() {
  std::fprintf(stderr, "usage: qwen_w4a8_dp4a_mlp_control --profile w4a8-dp4a-expert-mlp-4x-2048-512-2048 --gpu 0 --repetitions 1..3 --calls-per-sample 32 --memory-cap-mib 24..48 --seed N [--dry-run]\n");
  std::exit(2);
}
bool parse_int(const char* text, int* out) { char* end = nullptr; long v = std::strtol(text, &end, 10); if (!text[0] || *end || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false; *out = static_cast<int>(v); return true; }
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) { std::string a(argv[i]); if (a == "--dry-run") { o.dry = true; continue; } if (++i == argc) usage(); const char* v = argv[i]; if (a == "--profile") o.profile = v; else if (a == "--gpu") { if (!parse_int(v, &o.gpu)) usage(); } else if (a == "--repetitions") { if (!parse_int(v, &o.repetitions)) usage(); } else if (a == "--calls-per-sample") { if (!parse_int(v, &o.calls)) usage(); } else if (a == "--memory-cap-mib") { if (!parse_int(v, &o.cap)) usage(); } else if (a == "--seed") { if (!parse_int(v, &o.seed)) usage(); } else usage(); }
  if (o.profile != kProfile || o.gpu != 0 || o.repetitions < 1 || o.repetitions > 3 || o.calls != 32 || o.cap < 24 || o.cap > 48) usage();
  return o;
}
uint32_t mix32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }
bool ok(cudaError_t e, const char* what) { if (e == cudaSuccess) return true; std::fprintf(stderr, "[T20 CUDA] %s: %s\n", what, cudaGetErrorString(e)); return false; }
size_t count(int rows) { return static_cast<size_t>(kExperts) * rows; }
size_t weights(const Matrix& m) { return static_cast<size_t>(kExperts) * m.rows * (m.cols / 2); }
__host__ __device__ int nibble(uint8_t p, int col) { return static_cast<int>((col & 1) ? p >> 4 : p & 15U) - 8; }

void make_matrix(Matrix* m, int rows, int cols, uint32_t seed) {
  m->rows = rows; m->cols = cols; m->w.resize(static_cast<size_t>(kExperts) * rows * (cols / 2)); m->s.resize(count(rows));
  for (int e = 0; e < kExperts; ++e) for (int r = 0; r < rows; ++r) {
    const size_t ix = static_cast<size_t>(e) * rows + r; m->s[ix] = .001F + static_cast<float>(mix32(seed + ix) % 2048U) / 262144.0F;
    uint8_t* dst = m->w.data() + ix * (cols / 2);
    for (int c = 0; c < cols; c += 2) { const uint8_t lo = static_cast<uint8_t>(mix32(seed ^ static_cast<uint32_t>(ix * cols + c)) & 15U); const uint8_t hi = static_cast<uint8_t>(mix32(seed ^ static_cast<uint32_t>(ix * cols + c + 1)) & 15U); dst[c / 2] = static_cast<uint8_t>(lo | (hi << 4)); }
  }
}
void make_input(std::vector<float>* x, uint32_t seed) { x->resize(count(kHidden)); for (size_t i = 0; i < x->size(); ++i) (*x)[i] = static_cast<float>(static_cast<int>(mix32(seed + i) % 65536U) - 32768) / 8192.0F; }
void quant_cpu(const std::vector<float>& x, int width, std::vector<int8_t>* q, std::vector<float>* s) {
  q->resize(count(width)); s->resize(kExperts);
  for (int e = 0; e < kExperts; ++e) { const float* src = x.data() + static_cast<size_t>(e) * width; float hi = 0; for (int c = 0; c < width; ++c) hi = std::max(hi, std::fabs(src[c])); (*s)[e] = hi > 0 ? hi / 127.0F : 1.0F; for (int c = 0; c < width; ++c) { int v = static_cast<int>(std::nearbyint(src[c] / (*s)[e])); (*q)[static_cast<size_t>(e) * width + c] = static_cast<int8_t>(std::max(-127, std::min(127, v))); } }
}
void ref_dot(const Matrix& m, const std::vector<int8_t>& x, std::vector<int32_t>* y) {
  y->assign(count(m.rows), 0); for (int e = 0; e < kExperts; ++e) for (int r = 0; r < m.rows; ++r) { const size_t oi = static_cast<size_t>(e) * m.rows + r; const uint8_t* w = m.w.data() + oi * (m.cols / 2); const int8_t* q = x.data() + static_cast<size_t>(e) * m.cols; int64_t a = 0; for (int c = 0; c < m.cols; ++c) a += static_cast<int64_t>(q[c]) * nibble(w[c / 2], c); if (a < std::numeric_limits<int32_t>::min() || a > std::numeric_limits<int32_t>::max()) std::abort(); (*y)[oi] = static_cast<int32_t>(a); }
}
void dequant_cpu(const std::vector<int32_t>& x, const std::vector<float>& xs, const Matrix& m, std::vector<float>* y) { y->resize(count(m.rows)); for (int e = 0; e < kExperts; ++e) for (int r = 0; r < m.rows; ++r) { const size_t ix = static_cast<size_t>(e) * m.rows + r; (*y)[ix] = static_cast<float>(x[ix]) * xs[e] * m.s[ix]; } }
void cpu_hidden(const std::vector<float>& gate, const std::vector<float>& up, std::vector<float>* out) { out->resize(count(kIntermediate)); for (size_t i = 0; i < out->size(); ++i) (*out)[i] = (gate[i] / (1.0F + std::exp(-gate[i]))) * up[i]; }

__device__ __forceinline__ int w4x4(uint16_t p) { const int w0 = (static_cast<int>(p & 15U) - 8) & 255; const int w1 = (static_cast<int>((p >> 4) & 15U) - 8) & 255; const int w2 = (static_cast<int>((p >> 8) & 15U) - 8) & 255; const int w3 = (static_cast<int>((p >> 12) & 15U) - 8) & 255; return w0 | w1 << 8 | w2 << 16 | w3 << 24; }
template <int Width> __global__ void quant_rows(const float* x, int8_t* q, float* s) { const int e = blockIdx.x; __shared__ float maxima[8]; float hi = 0; const float* src = x + static_cast<size_t>(e) * Width; for (int c = threadIdx.x; c < Width; c += blockDim.x) hi = fmaxf(hi, fabsf(src[c])); for (int off = 16; off > 0; off >>= 1) hi = fmaxf(hi, __shfl_down_sync(0xffffffffU, hi, off)); if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = hi; __syncthreads(); if (threadIdx.x < 32) { hi = threadIdx.x < 8 ? maxima[threadIdx.x] : 0; for (int off = 16; off > 0; off >>= 1) hi = fmaxf(hi, __shfl_down_sync(0xffffffffU, hi, off)); if (threadIdx.x == 0) s[e] = hi > 0 ? hi / 127.0F : 1.0F; } __syncthreads(); const float inv = 1.0F / s[e]; int8_t* dst = q + static_cast<size_t>(e) * Width; for (int c = threadIdx.x; c < Width; c += blockDim.x) { int v = __float2int_rn(src[c] * inv); dst[c] = static_cast<int8_t>(max(-127, min(127, v))); } }
template <int Inputs, int Outputs> __global__ void dp4a_rows(const uint8_t* w, const int8_t* x, int32_t* y) { const int e = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.x * kTile + warp; if (row >= Outputs) return; const size_t oi = static_cast<size_t>(e) * Outputs + row; const uint8_t* roww = w + oi * (Inputs / 2); const int8_t* q = x + static_cast<size_t>(e) * Inputs; int a = 0;
#pragma unroll
  for (int c = lane * 4; c < Inputs; c += 128) { const uint16_t p = static_cast<uint16_t>(roww[c / 2]) | static_cast<uint16_t>(roww[c / 2 + 1]) << 8; a = __dp4a(*reinterpret_cast<const int*>(q + c), w4x4(p), a); }
  for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffU, a, off); if (lane == 0) y[oi] = a; }
template <int Outputs> __global__ void dequant_rows(const int32_t* x, const float* xs, const float* ws, float* y) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < kExperts * Outputs) { const int e = i / Outputs; y[i] = __int2float_rn(x[i]) * xs[e] * ws[i]; } }
__global__ void silu_mul(const float* gate, const float* up, float* out) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < kExperts * kIntermediate) out[i] = (gate[i] / (1.0F + expf(-gate[i]))) * up[i]; }
template <int Inputs, int Outputs> __global__ void w4a32_rows(const uint8_t* w, const float* x, const float* s, float* y) { const int e = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.x * kTile + warp; if (row >= Outputs) return; const size_t oi = static_cast<size_t>(e) * Outputs + row; const uint8_t* roww = w + oi * (Inputs / 2); const float* src = x + static_cast<size_t>(e) * Inputs; float a = 0; for (int c = lane; c < Inputs; c += 32) a = __fmaf_rn(src[c], static_cast<float>(nibble(roww[c / 2], c)), a); for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffU, a, off); if (lane == 0) y[oi] = a * s[oi]; }

bool alloc(Device* d) { return ok(cudaMalloc(&d->gw, static_cast<size_t>(kExperts) * kIntermediate * kHidden / 2), "gate weights") && ok(cudaMalloc(&d->uw, static_cast<size_t>(kExperts) * kIntermediate * kHidden / 2), "up weights") && ok(cudaMalloc(&d->dw, static_cast<size_t>(kExperts) * kHidden * kIntermediate / 2), "down weights") && ok(cudaMalloc(&d->gs, count(kIntermediate) * sizeof(float)), "gate scales") && ok(cudaMalloc(&d->us, count(kIntermediate) * sizeof(float)), "up scales") && ok(cudaMalloc(&d->ds, count(kHidden) * sizeof(float)), "down scales") && ok(cudaMalloc(&d->input, count(kHidden) * sizeof(float)), "input") && ok(cudaMalloc(&d->inq, count(kHidden)), "input q8") && ok(cudaMalloc(&d->ins, kExperts * sizeof(float)), "input scale") && ok(cudaMalloc(&d->gatei, count(kIntermediate) * sizeof(int32_t)), "gate int") && ok(cudaMalloc(&d->upi, count(kIntermediate) * sizeof(int32_t)), "up int") && ok(cudaMalloc(&d->downi, count(kHidden) * sizeof(int32_t)), "down int") && ok(cudaMalloc(&d->gate, count(kIntermediate) * sizeof(float)), "gate") && ok(cudaMalloc(&d->up, count(kIntermediate) * sizeof(float)), "up") && ok(cudaMalloc(&d->hidden, count(kIntermediate) * sizeof(float)), "hidden") && ok(cudaMalloc(&d->hiddenq, count(kIntermediate)), "hidden q8") && ok(cudaMalloc(&d->hs, kExperts * sizeof(float)), "hidden scale") && ok(cudaMalloc(&d->output, count(kHidden) * sizeof(float)), "output"); }
void release(Device* d) { cudaFree(d->gw); cudaFree(d->uw); cudaFree(d->dw); cudaFree(d->gs); cudaFree(d->us); cudaFree(d->ds); cudaFree(d->input); cudaFree(d->inq); cudaFree(d->ins); cudaFree(d->gatei); cudaFree(d->upi); cudaFree(d->downi); cudaFree(d->gate); cudaFree(d->up); cudaFree(d->hidden); cudaFree(d->hiddenq); cudaFree(d->hs); cudaFree(d->output); *d = Device{}; }
bool upload(Device* d, const Matrix& g, const Matrix& u, const Matrix& down) { return ok(cudaMemcpy(d->gw, g.w.data(), weights(g), cudaMemcpyHostToDevice), "gate W4 upload") && ok(cudaMemcpy(d->uw, u.w.data(), weights(u), cudaMemcpyHostToDevice), "up W4 upload") && ok(cudaMemcpy(d->dw, down.w.data(), weights(down), cudaMemcpyHostToDevice), "down W4 upload") && ok(cudaMemcpy(d->gs, g.s.data(), count(kIntermediate) * sizeof(float), cudaMemcpyHostToDevice), "gate scale upload") && ok(cudaMemcpy(d->us, u.s.data(), count(kIntermediate) * sizeof(float), cudaMemcpyHostToDevice), "up scale upload") && ok(cudaMemcpy(d->ds, down.s.data(), count(kHidden) * sizeof(float), cudaMemcpyHostToDevice), "down scale upload"); }
bool run_a8(Device* d, const std::vector<float>& x, Result* r, bool detail) { if (!ok(cudaMemcpy(d->input, x.data(), count(kHidden) * sizeof(float), cudaMemcpyHostToDevice), "A8 input upload")) return false; quant_rows<kHidden><<<kExperts, 256>>>(d->input, d->inq, d->ins); const dim3 mid((kIntermediate + kTile - 1) / kTile, kExperts), out((kHidden + kTile - 1) / kTile, kExperts); dp4a_rows<kHidden, kIntermediate><<<mid, kTile * 32>>>(d->gw, d->inq, d->gatei); dp4a_rows<kHidden, kIntermediate><<<mid, kTile * 32>>>(d->uw, d->inq, d->upi); dequant_rows<kIntermediate><<<(count(kIntermediate) + 255) / 256, 256>>>(d->gatei, d->ins, d->gs, d->gate); dequant_rows<kIntermediate><<<(count(kIntermediate) + 255) / 256, 256>>>(d->upi, d->ins, d->us, d->up); silu_mul<<<(count(kIntermediate) + 255) / 256, 256>>>(d->gate, d->up, d->hidden); quant_rows<kIntermediate><<<kExperts, 256>>>(d->hidden, d->hiddenq, d->hs); dp4a_rows<kIntermediate, kHidden><<<out, kTile * 32>>>(d->dw, d->hiddenq, d->downi); dequant_rows<kHidden><<<(count(kHidden) + 255) / 256, 256>>>(d->downi, d->hs, d->ds, d->output); if (!ok(cudaGetLastError(), "A8 launches")) return false; r->out.resize(count(kHidden)); if (!ok(cudaMemcpy(r->out.data(), d->output, count(kHidden) * sizeof(float), cudaMemcpyDeviceToHost), "A8 output")) return false; if (!detail) return true; r->inq.resize(count(kHidden)); r->hiddenq.resize(count(kIntermediate)); r->gate.resize(count(kIntermediate)); r->up.resize(count(kIntermediate)); r->down.resize(count(kHidden)); return ok(cudaMemcpy(r->inq.data(), d->inq, count(kHidden), cudaMemcpyDeviceToHost), "A8 input Q8") && ok(cudaMemcpy(r->hiddenq.data(), d->hiddenq, count(kIntermediate), cudaMemcpyDeviceToHost), "A8 hidden Q8") && ok(cudaMemcpy(r->gate.data(), d->gatei, count(kIntermediate) * sizeof(int32_t), cudaMemcpyDeviceToHost), "A8 gate int") && ok(cudaMemcpy(r->up.data(), d->upi, count(kIntermediate) * sizeof(int32_t), cudaMemcpyDeviceToHost), "A8 up int") && ok(cudaMemcpy(r->down.data(), d->downi, count(kHidden) * sizeof(int32_t), cudaMemcpyDeviceToHost), "A8 down int"); }
bool run_a32(Device* d, const std::vector<float>& x, Result* r) { if (!ok(cudaMemcpy(d->input, x.data(), count(kHidden) * sizeof(float), cudaMemcpyHostToDevice), "A32 input upload")) return false; const dim3 mid((kIntermediate + kTile - 1) / kTile, kExperts), out((kHidden + kTile - 1) / kTile, kExperts); w4a32_rows<kHidden, kIntermediate><<<mid, kTile * 32>>>(d->gw, d->input, d->gs, d->gate); w4a32_rows<kHidden, kIntermediate><<<mid, kTile * 32>>>(d->uw, d->input, d->us, d->up); silu_mul<<<(count(kIntermediate) + 255) / 256, 256>>>(d->gate, d->up, d->hidden); w4a32_rows<kIntermediate, kHidden><<<out, kTile * 32>>>(d->dw, d->hidden, d->ds, d->output); if (!ok(cudaGetLastError(), "A32 launches")) return false; r->out.resize(count(kHidden)); return ok(cudaMemcpy(r->out.data(), d->output, count(kHidden) * sizeof(float), cudaMemcpyDeviceToHost), "A32 output"); }
Error compare(const std::vector<float>& actual, const std::vector<float>& reference) { Error z; for (size_t i = 0; i < actual.size(); ++i) { const double d = static_cast<double>(actual[i]) - reference[i]; z.e += d * d; z.ref += static_cast<double>(reference[i]) * reference[i]; z.max_abs = std::max(z.max_abs, std::fabs(d)); z.max_rel = std::max(z.max_rel, std::fabs(d) / std::max(1.0, std::fabs(static_cast<double>(reference[i])))); } return z; }
double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double med(std::vector<double> x) { std::sort(x.begin(), x.end()); return x[x.size() / 2]; }
void samples(const std::vector<double>& x) { std::cout << '['; for (size_t i = 0; i < x.size(); ++i) { if (i) std::cout << ','; std::cout << x[i]; } std::cout << ']'; }
int run(const Options& o) {
  Matrix g, u, down; make_matrix(&g, kIntermediate, kHidden, static_cast<uint32_t>(o.seed)); make_matrix(&u, kIntermediate, kHidden, static_cast<uint32_t>(o.seed) + 0x10000U); make_matrix(&down, kHidden, kIntermediate, static_cast<uint32_t>(o.seed) + 0x20000U); std::vector<float> x; make_input(&x, static_cast<uint32_t>(o.seed) + 17U);
  std::vector<int8_t> cq, chq; std::vector<float> cis, cg, cu, ch; std::vector<int32_t> cgi, cui, cdi; quant_cpu(x, kHidden, &cq, &cis); ref_dot(g, cq, &cgi); ref_dot(u, cq, &cui); dequant_cpu(cgi, cis, g, &cg); dequant_cpu(cui, cis, u, &cu); cpu_hidden(cg, cu, &ch); quant_cpu(ch, kIntermediate, &chq, &cis); ref_dot(down, chq, &cdi);
  if (!ok(cudaSetDevice(o.gpu), "device selection")) return 1; Device d; if (!alloc(&d) || !upload(&d, g, u, down)) { release(&d); return 1; } Result a8, a32; if (!run_a8(&d, x, &a8, true) || !run_a32(&d, x, &a32)) { release(&d); return 1; }
  if (a8.inq != cq || a8.gate != cgi || a8.up != cui || a8.hiddenq != chq || a8.down != cdi) { std::fprintf(stderr, "[T20] staged W4/Q8 integer reference mismatch\n"); release(&d); return 1; }
  const Error err = compare(a8.out, a32.out); const double rel = std::sqrt(err.e / std::max(err.ref, 1e-30)); if (rel > .05) { std::fprintf(stderr, "[T20] full-MLP relative L2 gate failed: %.8f\n", rel); release(&d); return 1; }
  Result scratch; for (int i = 0; i < 3; ++i) if (!run_a8(&d, x, &scratch, false) || !run_a32(&d, x, &scratch)) { release(&d); return 1; } std::vector<double> a8ms, a32ms;
  for (int r = 0; r < o.repetitions; ++r) { double t = now(); for (int c = 0; c < o.calls; ++c) if (!run_a8(&d, x, &scratch, false)) { release(&d); return 1; } a8ms.push_back((now() - t) / o.calls); t = now(); for (int c = 0; c < o.calls; ++c) if (!run_a32(&d, x, &scratch)) { release(&d); return 1; } a32ms.push_back((now() - t) / o.calls); }
  release(&d); const double a8m = med(a8ms), a32m = med(a32ms); std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile << "\",\"cuda_initialized\":true,\"tile\":8,\"experts\":4,\"staged_integer_exact\":true,\"relative_l2_error\":" << rel << ",\"max_absolute_error\":" << err.max_abs << ",\"max_relative_error\":" << err.max_rel << ",\"w4a8_ms\":" << a8m << ",\"w4a32_ms\":" << a32m << ",\"speedup\":" << a32m / a8m << ",\"w4a8_samples\":"; samples(a8ms); std::cout << ",\"w4a32_samples\":"; samples(a32ms); std::cout << ",\"advancement_eligible\":" << (a32m / a8m >= 1.15 ? "true" : "false") << "}\n"; return 0;
}
}  // namespace
int main(int argc, char** argv) { const Options o = parse(argc, argv); if (o.dry) { std::cout << "{\"schema_version\":\"t20-w4a8-dp4a-expert-mlp-v1\",\"profile\":\"w4a8-dp4a-expert-mlp-4x-2048-512-2048\",\"cuda_initialized\":false}\n"; return 0; } return run(o); }
