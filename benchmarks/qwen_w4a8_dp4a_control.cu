// T17: isolated Pascal W4A8/DP4A MoE-projection control.
// It is synthetic, owns no model state, and does not modify Colibri.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int kExperts = 4;       // representative per-P40 top-k share
constexpr int kInput = 2048;      // Qwen routed expert hidden width
constexpr int kOutput = 512;      // Qwen routed expert intermediate width
#ifndef P40_W4A8_TILE
#define P40_W4A8_TILE 8
#endif
constexpr int kTile = P40_W4A8_TILE;  // warp-per-row outputs per CTA
constexpr const char* kProfile = "w4a8-dp4a-expert-proj-4x-2048-512";
constexpr const char* kSchema = "t17-w4a8-dp4a-expert-proj-v1";
#ifndef P40_W4A8_UNROLL
#define P40_W4A8_UNROLL 1
#endif

struct Options {
  std::string profile;
  int gpu = -1;
  int repetitions = 0;
  int calls_per_sample = 0;
  int memory_cap_mib = 0;
  int seed = 0;
  bool dry_run = false;
};

struct Device {
  uint8_t* w4 = nullptr;
  float* wscale = nullptr;
  float* input = nullptr;
  int8_t* input_q8 = nullptr;
  float* input_scale = nullptr;
  int32_t* accum = nullptr;
  float* output = nullptr;
};

struct Error {
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double max_absolute = 0.0;
  double max_relative = 0.0;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
      "usage: qwen_w4a8_dp4a_control --profile w4a8-dp4a-expert-proj-4x-2048-512 --gpu 0 "
      "--repetitions 1..3 --calls-per-sample 64 --memory-cap-mib 24..48 --seed N [--dry-run]\n");
  std::exit(2);
}

bool parse_int(const char* text, int* output) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) return false;
  *output = static_cast<int>(value);
  return true;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--dry-run") { options.dry_run = true; continue; }
    if (index + 1 >= argc) usage();
    const char* value = argv[++index];
    if (argument == "--profile") options.profile = value;
    else if (argument == "--gpu") { if (!parse_int(value, &options.gpu)) usage(); }
    else if (argument == "--repetitions") { if (!parse_int(value, &options.repetitions)) usage(); }
    else if (argument == "--calls-per-sample") { if (!parse_int(value, &options.calls_per_sample)) usage(); }
    else if (argument == "--memory-cap-mib") { if (!parse_int(value, &options.memory_cap_mib)) usage(); }
    else if (argument == "--seed") { if (!parse_int(value, &options.seed)) usage(); }
    else usage();
  }
  if (options.profile != kProfile || options.gpu != 0 || options.repetitions < 1 ||
      options.repetitions > 3 || options.calls_per_sample != 64 ||
      options.memory_cap_mib < 24 || options.memory_cap_mib > 48 ||
      (kTile != 8 && kTile != 16)) usage();
  return options;
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[T17 CUDA] %s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

size_t row_bytes() { return static_cast<size_t>(kInput / 2); }
size_t weight_bytes() { return static_cast<size_t>(kExperts) * kOutput * row_bytes(); }
size_t activation_count() { return static_cast<size_t>(kExperts) * kInput; }
size_t output_count() { return static_cast<size_t>(kExperts) * kOutput; }

void make_weights(std::vector<uint8_t>* packed, std::vector<float>* scales, uint32_t seed) {
  packed->resize(weight_bytes());
  scales->resize(output_count());
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int row = 0; row < kOutput; ++row) {
      const size_t scale_index = static_cast<size_t>(expert) * kOutput + row;
      (*scales)[scale_index] = 0.001F +
          static_cast<float>(mix32(seed + static_cast<uint32_t>(scale_index)) % 2048U) / 262144.0F;
      uint8_t* destination = packed->data() + scale_index * row_bytes();
      for (int column = 0; column < kInput; column += 2) {
        const uint8_t low = static_cast<uint8_t>(mix32(seed ^ static_cast<uint32_t>(scale_index * kInput + column)) & 0x0fU);
        const uint8_t high = static_cast<uint8_t>(mix32(seed ^ static_cast<uint32_t>(scale_index * kInput + column + 1)) & 0x0fU);
        destination[column / 2] = static_cast<uint8_t>(low | (high << 4));
      }
    }
  }
}

void make_inputs(std::vector<float>* input, uint32_t seed) {
  input->resize(activation_count());
  for (size_t index = 0; index < input->size(); ++index) {
    const int value = static_cast<int>(mix32(seed + static_cast<uint32_t>(index)) % 65536U) - 32768;
    (*input)[index] = static_cast<float>(value) / 8192.0F;
  }
}

int signed_nibble(uint8_t value, int column) {
  return static_cast<int>((column & 1) ? (value >> 4) : (value & 0x0fU)) - 8;
}

void quantize_cpu(const std::vector<float>& input, std::vector<int8_t>* quantized,
                  std::vector<float>* scales) {
  quantized->resize(activation_count());
  scales->resize(kExperts);
  for (int expert = 0; expert < kExperts; ++expert) {
    const float* source = input.data() + static_cast<size_t>(expert) * kInput;
    float maximum = 0.0F;
    for (int column = 0; column < kInput; ++column) maximum = std::max(maximum, std::fabs(source[column]));
    const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
    (*scales)[expert] = scale;
    for (int column = 0; column < kInput; ++column) {
      int value = static_cast<int>(std::nearbyint(source[column] / scale));
      value = std::max(-127, std::min(127, value));
      (*quantized)[static_cast<size_t>(expert) * kInput + column] = static_cast<int8_t>(value);
    }
  }
}

void cpu_integer_reference(const std::vector<uint8_t>& weights, const std::vector<int8_t>& input,
                           std::vector<int32_t>* output) {
  output->assign(output_count(), 0);
  for (int expert = 0; expert < kExperts; ++expert) {
    const int8_t* activation = input.data() + static_cast<size_t>(expert) * kInput;
    for (int row = 0; row < kOutput; ++row) {
      const size_t output_index = static_cast<size_t>(expert) * kOutput + row;
      const uint8_t* weight = weights.data() + output_index * row_bytes();
      int64_t accumulator = 0;
      for (int column = 0; column < kInput; ++column) {
        accumulator += static_cast<int64_t>(activation[column]) * signed_nibble(weight[column / 2], column);
      }
      if (accumulator < std::numeric_limits<int32_t>::min() || accumulator > std::numeric_limits<int32_t>::max()) std::abort();
      (*output)[output_index] = static_cast<int32_t>(accumulator);
    }
  }
}

void cpu_float_reference(const std::vector<uint8_t>& weights, const std::vector<float>& scales,
                         const std::vector<float>& input, std::vector<float>* output) {
  output->assign(output_count(), 0.0F);
  for (int expert = 0; expert < kExperts; ++expert) {
    const float* activation = input.data() + static_cast<size_t>(expert) * kInput;
    for (int row = 0; row < kOutput; ++row) {
      const size_t output_index = static_cast<size_t>(expert) * kOutput + row;
      const uint8_t* weight = weights.data() + output_index * row_bytes();
      float accumulator = 0.0F;
      for (int column = 0; column < kInput; ++column) accumulator += activation[column] * static_cast<float>(signed_nibble(weight[column / 2], column));
      (*output)[output_index] = accumulator * scales[output_index];
    }
  }
}

Error compare_approximation(const std::vector<float>& reference, const std::vector<int32_t>& integer_output,
                            const std::vector<float>& input_scales, const std::vector<float>& weight_scales) {
  Error error;
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int row = 0; row < kOutput; ++row) {
      const size_t index = static_cast<size_t>(expert) * kOutput + row;
      const float actual = static_cast<float>(integer_output[index]) * input_scales[expert] * weight_scales[index];
      const double delta = static_cast<double>(actual) - reference[index];
      error.squared_error += delta * delta;
      error.squared_reference += static_cast<double>(reference[index]) * reference[index];
      error.max_absolute = std::max(error.max_absolute, std::fabs(delta));
      error.max_relative = std::max(error.max_relative,
          std::fabs(delta) / std::max(1.0, std::fabs(static_cast<double>(reference[index]))));
    }
  }
  return error;
}

__device__ __forceinline__ int packed_signed_w4x4(uint16_t packed) {
  const int w0 = (static_cast<int>(packed & 0x0fU) - 8) & 0xff;
  const int w1 = (static_cast<int>((packed >> 4) & 0x0fU) - 8) & 0xff;
  const int w2 = (static_cast<int>((packed >> 8) & 0x0fU) - 8) & 0xff;
  const int w3 = (static_cast<int>((packed >> 12) & 0x0fU) - 8) & 0xff;
  return w0 | (w1 << 8) | (w2 << 16) | (w3 << 24);
}

__global__ void quantize_rows_i8(const float* input, int8_t* output, float* scales) {
  const int expert = static_cast<int>(blockIdx.x);
  __shared__ float maxima[8];
  float maximum = 0.0F;
  const float* source = input + static_cast<size_t>(expert) * kInput;
  for (int column = threadIdx.x; column < kInput; column += blockDim.x) maximum = fmaxf(maximum, fabsf(source[column]));
  for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x < 32) {
    maximum = threadIdx.x < 8 ? maxima[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    if (threadIdx.x == 0) scales[expert] = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  }
  __syncthreads();
  const float inverse = 1.0F / scales[expert];
  int8_t* destination = output + static_cast<size_t>(expert) * kInput;
  for (int column = threadIdx.x; column < kInput; column += blockDim.x) {
    int value = __float2int_rn(source[column] * inverse);
    value = max(-127, min(127, value));
    destination[column] = static_cast<int8_t>(value);
  }
}

__global__ void w4a8_dp4a_rows(const uint8_t* weights, const int8_t* input, int32_t* output) {
  const int expert = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kTile + warp;
  if (row >= kOutput) return;
  const size_t output_index = static_cast<size_t>(expert) * kOutput + row;
  const uint8_t* weight = weights + output_index * (kInput / 2);
  const int8_t* activation = input + static_cast<size_t>(expert) * kInput;
  int accumulator = 0;
#if P40_W4A8_UNROLL
#pragma unroll
#else
#pragma unroll 1
#endif
  for (int column = lane * 4; column < kInput; column += 32 * 4) {
    const uint16_t packed = static_cast<uint16_t>(weight[column / 2]) |
                            (static_cast<uint16_t>(weight[column / 2 + 1]) << 8);
    const int activation4 = *reinterpret_cast<const int*>(activation + column);
    accumulator = __dp4a(activation4, packed_signed_w4x4(packed), accumulator);
  }
  for (int offset = 16; offset > 0; offset >>= 1) accumulator += __shfl_down_sync(0xffffffffU, accumulator, offset);
  if (lane == 0) output[output_index] = accumulator;
}

__global__ void dequantize_rows(const int32_t* input, const float* activation_scales,
                                const float* weight_scales, float* output) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kExperts * kOutput) return;
  const int expert = index / kOutput;
  output[index] = __fmul_rn(__int2float_rn(input[index]),
                            __fmul_rn(activation_scales[expert], weight_scales[index]));
}

__global__ void w4a32_rows(const uint8_t* weights, const float* input, const float* scales, float* output) {
  const int expert = static_cast<int>(blockIdx.y);
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kTile + warp;
  if (row >= kOutput) return;
  const size_t output_index = static_cast<size_t>(expert) * kOutput + row;
  const uint8_t* weight = weights + output_index * (kInput / 2);
  const float* activation = input + static_cast<size_t>(expert) * kInput;
  float accumulator = 0.0F;
  for (int column = lane; column < kInput; column += 32) {
    const uint8_t packed = weight[column / 2];
    const int value = static_cast<int>((column & 1) ? (packed >> 4) : (packed & 0x0fU)) - 8;
    accumulator = __fmaf_rn(activation[column], static_cast<float>(value), accumulator);
  }
  for (int offset = 16; offset > 0; offset >>= 1) accumulator += __shfl_down_sync(0xffffffffU, accumulator, offset);
  if (lane == 0) output[output_index] = accumulator * scales[output_index];
}

bool allocate(Device* device) {
  return cuda_ok(cudaMalloc(&device->w4, weight_bytes()), "W4 allocation") &&
         cuda_ok(cudaMalloc(&device->wscale, output_count() * sizeof(float)), "scale allocation") &&
         cuda_ok(cudaMalloc(&device->input, activation_count() * sizeof(float)), "input allocation") &&
         cuda_ok(cudaMalloc(&device->input_q8, activation_count() * sizeof(int8_t)), "Q8 allocation") &&
         cuda_ok(cudaMalloc(&device->input_scale, kExperts * sizeof(float)), "input scale allocation") &&
         cuda_ok(cudaMalloc(&device->accum, output_count() * sizeof(int32_t)), "accumulator allocation") &&
         cuda_ok(cudaMalloc(&device->output, output_count() * sizeof(float)), "output allocation");
}

void release(Device* device) {
  cudaFree(device->w4); cudaFree(device->wscale); cudaFree(device->input); cudaFree(device->input_q8);
  cudaFree(device->input_scale); cudaFree(device->accum); cudaFree(device->output); *device = Device{};
}

bool run_w4a8(Device* device, const std::vector<float>& input, std::vector<int32_t>* integer_output,
              std::vector<float>* output) {
  if (!cuda_ok(cudaMemcpy(device->input, input.data(), activation_count() * sizeof(float), cudaMemcpyHostToDevice), "W4A8 input upload")) return false;
  quantize_rows_i8<<<kExperts, 256>>>(device->input, device->input_q8, device->input_scale);
  w4a8_dp4a_rows<<<dim3((kOutput + kTile - 1) / kTile, kExperts), kTile * 32>>>(device->w4, device->input_q8, device->accum);
  dequantize_rows<<<(kExperts * kOutput + 255) / 256, 256>>>(device->accum, device->input_scale, device->wscale, device->output);
  if (!cuda_ok(cudaGetLastError(), "W4A8 DP4A launch")) return false;
  integer_output->resize(output_count()); output->resize(output_count());
  return cuda_ok(cudaMemcpy(integer_output->data(), device->accum, output_count() * sizeof(int32_t), cudaMemcpyDeviceToHost), "W4A8 integer download") &&
         cuda_ok(cudaMemcpy(output->data(), device->output, output_count() * sizeof(float), cudaMemcpyDeviceToHost), "W4A8 output download");
}

bool run_w4a32(Device* device, const std::vector<float>& input, std::vector<float>* output) {
  if (!cuda_ok(cudaMemcpy(device->input, input.data(), activation_count() * sizeof(float), cudaMemcpyHostToDevice), "W4A32 input upload")) return false;
  w4a32_rows<<<dim3((kOutput + kTile - 1) / kTile, kExperts), kTile * 32>>>(device->w4, device->input, device->wscale, device->output);
  if (!cuda_ok(cudaGetLastError(), "W4A32 launch")) return false;
  output->resize(output_count());
  return cuda_ok(cudaMemcpy(output->data(), device->output, output_count() * sizeof(float), cudaMemcpyDeviceToHost), "W4A32 output download");
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void print_samples(const std::vector<double>& samples) {
  std::cout << '[';
  for (size_t index = 0; index < samples.size(); ++index) { if (index) std::cout << ','; std::cout << samples[index]; }
  std::cout << ']';
}

int run(const Options& options) {
  std::vector<uint8_t> weights; std::vector<float> weight_scales; std::vector<float> input;
  std::vector<int8_t> cpu_q8; std::vector<float> cpu_input_scales; std::vector<int32_t> cpu_integer;
  make_weights(&weights, &weight_scales, static_cast<uint32_t>(options.seed));
  make_inputs(&input, static_cast<uint32_t>(options.seed) + 17U);
  quantize_cpu(input, &cpu_q8, &cpu_input_scales);
  cpu_integer_reference(weights, cpu_q8, &cpu_integer);
  std::vector<float> cpu_float; cpu_float_reference(weights, weight_scales, input, &cpu_float);
  if (!cuda_ok(cudaSetDevice(options.gpu), "device selection")) return 1;
  Device device;
  if (!allocate(&device) ||
      !cuda_ok(cudaMemcpy(device.w4, weights.data(), weight_bytes(), cudaMemcpyHostToDevice), "W4 upload") ||
      !cuda_ok(cudaMemcpy(device.wscale, weight_scales.data(), output_count() * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
    release(&device); return 1;
  }
  std::vector<int32_t> gpu_integer; std::vector<float> gpu_output; std::vector<float> scratch;
  if (!run_w4a8(&device, input, &gpu_integer, &gpu_output)) { release(&device); return 1; }
  if (gpu_integer != cpu_integer) { std::fprintf(stderr, "[T17] DP4A integer reference mismatch\n"); release(&device); return 1; }
  Error error = compare_approximation(cpu_float, gpu_integer, cpu_input_scales, weight_scales);
  const double relative_l2 = std::sqrt(error.squared_error / std::max(error.squared_reference, 1e-30));
  if (relative_l2 > 0.02) { std::fprintf(stderr, "[T17] quantization error gate failed: %.8f\n", relative_l2); release(&device); return 1; }
  for (int warmup = 0; warmup < 3; ++warmup) {
    if (!run_w4a8(&device, input, &gpu_integer, &gpu_output) || !run_w4a32(&device, input, &scratch)) { release(&device); return 1; }
  }
  std::vector<double> w4a8_samples, w4a32_samples;
  for (int repetition = 0; repetition < options.repetitions; ++repetition) {
    const double w4a8_started = now_ms();
    for (int call = 0; call < options.calls_per_sample; ++call)
      if (!run_w4a8(&device, input, &gpu_integer, &gpu_output)) { release(&device); return 1; }
    w4a8_samples.push_back((now_ms() - w4a8_started) / options.calls_per_sample);
    const double w4a32_started = now_ms();
    for (int call = 0; call < options.calls_per_sample; ++call)
      if (!run_w4a32(&device, input, &scratch)) { release(&device); return 1; }
    w4a32_samples.push_back((now_ms() - w4a32_started) / options.calls_per_sample);
  }
  release(&device);
  const double w4a8_ms = median(w4a8_samples), w4a32_ms = median(w4a32_samples);
  std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"tile\":" << kTile
            << ",\"experts\":" << kExperts << ",\"integer_exact\":true"
            << ",\"unrolled\":" << (P40_W4A8_UNROLL ? "true" : "false")
            << ",\"relative_l2_error\":" << relative_l2
            << ",\"max_absolute_error\":" << error.max_absolute
            << ",\"max_relative_error\":" << error.max_relative
            << ",\"w4a8_ms\":" << w4a8_ms << ",\"w4a32_ms\":" << w4a32_ms
            << ",\"speedup\":" << (w4a32_ms / w4a8_ms) << ",\"w4a8_samples\":";
  print_samples(w4a8_samples); std::cout << ",\"w4a32_samples\":"; print_samples(w4a32_samples);
  std::cout << ",\"advancement_eligible\":" << ((w4a32_ms / w4a8_ms) >= 1.15 ? "true" : "false") << "}\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile
              << "\",\"cuda_initialized\":false}\n";
    return 0;
  }
  return run(options);
}
