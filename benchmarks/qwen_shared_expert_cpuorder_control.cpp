// T14: exact-order Q8 shared-expert sweep for Qwen3.6's 40 MoE layers.
// Synthetic only: this does not load a model or modify the Colibri engine.
#include "qwen_cpuorder_cuda.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
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
constexpr int kLayers = 40;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr const char* kProfile = "shared-expert-cpuorder-40x-2048-512-2048";
constexpr const char* kSchema = "t14-shared-expert-cpuorder-q8-v1";

struct Options {
  std::string profile;
  int gpu = -1;
  int repetitions = 0;
  int calls_per_sample = 0;
  int memory_cap_mib = 0;
  int seed = 0;
  bool dry_run = false;
};

struct Matrix {
  int input_width = 0;
  int output_width = 0;
  std::vector<int8_t> weights;
  std::vector<float> scales;
  P40CpuOrderTensor* tensor = nullptr;
};

struct Layer {
  Matrix gate;
  Matrix up;
  Matrix down;
  std::vector<float> input;
  std::vector<float> gate_reference;
  std::vector<float> up_reference;
  std::vector<float> activation_reference;
  std::vector<float> down_reference;
  std::vector<float> gate_scratch;
  std::vector<float> up_scratch;
  std::vector<float> activation_scratch;
  std::vector<float> down_scratch;
  std::vector<float> gate_gpu;
  std::vector<float> up_gpu;
  std::vector<float> activation_gpu;
  std::vector<float> down_gpu;
};

struct Comparison {
  bool bit_exact = true;
  size_t mismatch_count = 0;
  double max_absolute = 0.0;
  double max_relative = 0.0;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
      "usage: qwen_shared_expert_cpuorder_control --profile shared-expert-cpuorder-40x-2048-512-2048 --gpu 0 "
      "--repetitions 1..3 --calls-per-sample 1 --memory-cap-mib 144..176 --seed N [--dry-run]\\n");
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
      options.repetitions > 3 || options.calls_per_sample != 1 ||
      options.memory_cap_mib < 144 || options.memory_cap_mib > 176) usage();
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

Matrix make_matrix(int input_width, int output_width, uint32_t seed) {
  Matrix matrix;
  matrix.input_width = input_width;
  matrix.output_width = output_width;
  matrix.weights.resize(static_cast<size_t>(input_width) * output_width);
  matrix.scales.resize(output_width);
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(matrix.weights.size()); ++index) {
    matrix.weights[static_cast<size_t>(index)] = static_cast<int8_t>(
        static_cast<int>(mix32(seed ^ static_cast<uint32_t>(index)) % 255U) - 127);
  }
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(matrix.scales.size()); ++index) {
    matrix.scales[static_cast<size_t>(index)] = 0.000125F +
        static_cast<float>(mix32(seed + static_cast<uint32_t>(index)) % 2048U) / 131072.0F;
  }
  return matrix;
}

Layer make_layer(uint32_t seed) {
  Layer layer;
  layer.gate = make_matrix(kHidden, kIntermediate, seed + 1U);
  layer.up = make_matrix(kHidden, kIntermediate, seed + 2U);
  layer.down = make_matrix(kIntermediate, kHidden, seed + 3U);
  layer.input.resize(kHidden);
  for (size_t index = 0; index < layer.input.size(); ++index) {
    layer.input[index] = static_cast<float>(static_cast<int>(
        mix32(seed + 0x9e3779b9U + static_cast<uint32_t>(index)) % 65536U) - 32768) / 32768.0F;
  }
  layer.gate_reference.resize(kIntermediate);
  layer.up_reference.resize(kIntermediate);
  layer.activation_reference.resize(kIntermediate);
  layer.down_reference.resize(kHidden);
  layer.gate_scratch.resize(kIntermediate);
  layer.up_scratch.resize(kIntermediate);
  layer.activation_scratch.resize(kIntermediate);
  layer.down_scratch.resize(kHidden);
  layer.gate_gpu.resize(kIntermediate);
  layer.up_gpu.resize(kIntermediate);
  layer.activation_gpu.resize(kIntermediate);
  layer.down_gpu.resize(kHidden);
  return layer;
}

// Byte-for-byte the S=1 dense-Q8 reduction order used by qwen36.c:matmul_q.
void qwen_cpu_q8_matvec(float* output, const float* input, const Matrix& matrix) {
#if !defined(__AVX2__) || !defined(__FMA__)
#error "T14 requires the Qwen x86 AVX2/FMA CPU control"
#endif
  if (matrix.input_width % 32) std::abort();
#pragma omp parallel for schedule(static)
  for (int row = 0; row < matrix.output_width; ++row) {
    const int8_t* weight = matrix.weights.data() + static_cast<size_t>(row) * matrix.input_width;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int column = 0; column < matrix.input_width; column += 32) {
      const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + column));
      const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + column + 16));
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8))), a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8))), a3);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    reduced = _mm_add_ps(reduced, _mm_movehl_ps(reduced, reduced));
    reduced = _mm_add_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
    output[row] = _mm_cvtss_f32(reduced) * matrix.scales[row];
  }
}

void activate(float* output, const float* gate, const float* up) {
  for (int index = 0; index < kIntermediate; ++index) {
    const float value = gate[index];
    output[index] = (value / (1.0F + std::exp(-value))) * up[index];
  }
}

void cpu_sweep(std::vector<Layer>* layers, bool scratch) {
  for (Layer& layer : *layers) {
    float* gate = scratch ? layer.gate_scratch.data() : layer.gate_reference.data();
    float* up = scratch ? layer.up_scratch.data() : layer.up_reference.data();
    float* activation = scratch ? layer.activation_scratch.data() : layer.activation_reference.data();
    float* down = scratch ? layer.down_scratch.data() : layer.down_reference.data();
    qwen_cpu_q8_matvec(gate, layer.input.data(), layer.gate);
    qwen_cpu_q8_matvec(up, layer.input.data(), layer.up);
    activate(activation, gate, up);
    qwen_cpu_q8_matvec(down, activation, layer.down);
  }
}

bool gpu_sweep(std::vector<Layer>* layers) {
  for (Layer& layer : *layers) {
    if (!p40_cpuorder_matvec(layer.gate.tensor, layer.gate_gpu.data(), layer.input.data()) ||
        !p40_cpuorder_matvec(layer.up.tensor, layer.up_gpu.data(), layer.input.data())) return false;
    activate(layer.activation_gpu.data(), layer.gate_gpu.data(), layer.up_gpu.data());
    if (!p40_cpuorder_matvec(layer.down.tensor, layer.down_gpu.data(), layer.activation_gpu.data())) return false;
  }
  return true;
}

void compare_vector(const std::vector<float>& reference, const std::vector<float>& actual,
                    Comparison* comparison) {
  if (std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0) return;
  comparison->bit_exact = false;
  for (size_t index = 0; index < reference.size(); ++index) {
    if (std::memcmp(&reference[index], &actual[index], sizeof(float)) == 0) continue;
    ++comparison->mismatch_count;
    const double absolute = std::fabs(static_cast<double>(actual[index]) - reference[index]);
    comparison->max_absolute = std::max(comparison->max_absolute, absolute);
    comparison->max_relative = std::max(comparison->max_relative,
        absolute / std::max(1.0, std::fabs(static_cast<double>(reference[index]))));
  }
}

Comparison compare_bits(const std::vector<Layer>& layers) {
  Comparison result;
  for (const Layer& layer : layers) {
    compare_vector(layer.gate_reference, layer.gate_gpu, &result);
    compare_vector(layer.up_reference, layer.up_gpu, &result);
    compare_vector(layer.activation_reference, layer.activation_gpu, &result);
    compare_vector(layer.down_reference, layer.down_gpu, &result);
  }
  return result;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void print_samples(const std::vector<double>& samples) {
  std::cout << '[';
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << samples[index];
  }
  std::cout << ']';
}

size_t matrix_bytes(const Matrix& matrix) {
  return matrix.weights.size() * sizeof(int8_t) + matrix.scales.size() * sizeof(float);
}

size_t host_accounted_bytes(const std::vector<Layer>& layers) {
  size_t bytes = 0;
  for (const Layer& layer : layers) {
    bytes += matrix_bytes(layer.gate) + matrix_bytes(layer.up) + matrix_bytes(layer.down);
    bytes += layer.input.size() * sizeof(float);
    bytes += (layer.gate_reference.size() + layer.up_reference.size() + layer.activation_reference.size() + layer.down_reference.size()) * sizeof(float);
    bytes += (layer.gate_scratch.size() + layer.up_scratch.size() + layer.activation_scratch.size() + layer.down_scratch.size()) * sizeof(float);
    bytes += (layer.gate_gpu.size() + layer.up_gpu.size() + layer.activation_gpu.size() + layer.down_gpu.size()) * sizeof(float);
  }
  return bytes;
}

bool upload(Matrix* matrix) {
  return p40_cpuorder_upload(&matrix->tensor, matrix->weights.data(), matrix->scales.data(),
                              matrix->input_width, matrix->output_width);
}

void release(std::vector<Layer>* layers) {
  for (Layer& layer : *layers) {
    p40_cpuorder_tensor_free(layer.gate.tensor);
    p40_cpuorder_tensor_free(layer.up.tensor);
    p40_cpuorder_tensor_free(layer.down.tensor);
    layer.gate.tensor = nullptr;
    layer.up.tensor = nullptr;
    layer.down.tensor = nullptr;
  }
}

int run(const Options& options) {
  std::vector<Layer> layers;
  layers.reserve(kLayers);
  for (int layer = 0; layer < kLayers; ++layer) {
    layers.push_back(make_layer(static_cast<uint32_t>(options.seed) + static_cast<uint32_t>(layer * 17)));
  }
  const size_t bytes = host_accounted_bytes(layers);
  if (bytes > static_cast<size_t>(options.memory_cap_mib) * 1024U * 1024U) {
    std::fprintf(stderr, "T14 host allocation exceeds memory cap\\n");
    return 2;
  }
  cpu_sweep(&layers, false);
  std::vector<double> cpu_samples;
  for (int sample = 0; sample < options.repetitions; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    cpu_sweep(&layers, true);
    cpu_samples.push_back(elapsed_ms(started));
  }
  if (!p40_cpuorder_init(options.gpu)) return 1;
  const auto first_started = std::chrono::steady_clock::now();
  bool gpu_ok = true;
  for (Layer& layer : layers) {
    gpu_ok = upload(&layer.gate) && upload(&layer.up) && upload(&layer.down);
    if (!gpu_ok) break;
  }
  if (gpu_ok) gpu_ok = gpu_sweep(&layers);
  const double first_upload_and_sweep_ms = elapsed_ms(first_started);
  const Comparison comparison = compare_bits(layers);
  std::vector<double> gpu_samples;
  for (int sample = 0; sample < options.repetitions && gpu_ok; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    gpu_ok = gpu_sweep(&layers);
    if (gpu_ok) gpu_samples.push_back(elapsed_ms(started));
  }
  release(&layers);
  p40_cpuorder_shutdown();
  const double cpu_median = median(cpu_samples);
  const double gpu_median = gpu_samples.empty() ? 0.0 : median(gpu_samples);
  const bool correct = gpu_ok && comparison.bit_exact;
  const size_t q8_weight_bytes = static_cast<size_t>(kLayers) * 3U * kHidden * kIntermediate;
  std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"gpu\":" << options.gpu
            << ",\"layers\":" << kLayers << ",\"matrix_count\":" << kLayers * 3
            << ",\"q8_weight_bytes\":" << q8_weight_bytes
            << ",\"host_accounted_bytes\":" << bytes
            << ",\"first_upload_and_sweep_ms\":" << first_upload_and_sweep_ms
            << ",\"cpu_ms_per_sweep\":";
  print_samples(cpu_samples);
  std::cout << ",\"gpu_complete_ms_per_sweep\":";
  print_samples(gpu_samples);
  std::cout << ",\"cpu_median_ms\":" << cpu_median
            << ",\"gpu_complete_median_ms\":" << gpu_median
            << ",\"speedup_cpu_over_gpu\":" << (gpu_median > 0.0 ? cpu_median / gpu_median : 0.0)
            << ",\"bit_exact\":" << (comparison.bit_exact ? "true" : "false")
            << ",\"mismatch_count\":" << comparison.mismatch_count
            << ",\"max_abs_error\":" << comparison.max_absolute
            << ",\"max_relative_error\":" << comparison.max_relative
            << ",\"correct\":" << (correct ? "true" : "false") << "}" << std::endl;
  return correct ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile
              << "\",\"cuda_initialized\":false}" << std::endl;
    return 0;
  }
  return run(options);
}
