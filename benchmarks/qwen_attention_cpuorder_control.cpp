// T11: exact-order Q8 projection sweep for Qwen3.6's ten full-attention layers.
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
constexpr int kLayers = 10;
constexpr int kHidden = 2048;
constexpr int kQueryOutput = 8192;
constexpr int kKvOutput = 512;
constexpr int kOutputInput = 4096;
constexpr int kOutputOutput = 2048;
constexpr int kMatricesPerLayer = 4;
constexpr const char* kProfile = "attention-projections-cpuorder-10x-2048-8192-512-4096-2048";
constexpr const char* kSchema = "t11-attention-projections-cpuorder-q8-v1";

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
  const char* stage = nullptr;
  int layer = 0;
  int input_width = 0;
  int output_width = 0;
  std::vector<int8_t> weights;
  std::vector<float> scales;
  std::vector<float> input;
  std::vector<float> reference;
  std::vector<float> scratch;
  std::vector<float> gpu;
  P40CpuOrderTensor* tensor = nullptr;
};

struct Comparison {
  bool bit_exact = true;
  size_t mismatch_count = 0;
  double max_absolute = 0.0;
  double max_relative = 0.0;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
      "usage: qwen_attention_cpuorder_control --profile attention-projections-cpuorder-10x-2048-8192-512-4096-2048 --gpu 0 "
      "--repetitions 1..3 --calls-per-sample 1 --memory-cap-mib 288..320 --seed N [--dry-run]\\n");
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
      options.memory_cap_mib < 288 || options.memory_cap_mib > 320) usage();
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

void fill_matrix(Matrix* matrix, uint32_t seed) {
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(matrix->weights.size()); ++index) {
    matrix->weights[static_cast<size_t>(index)] = static_cast<int8_t>(
        static_cast<int>(mix32(seed ^ static_cast<uint32_t>(index)) % 255U) - 127);
  }
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(matrix->scales.size()); ++index) {
    matrix->scales[static_cast<size_t>(index)] = 0.000125F +
        static_cast<float>(mix32(seed + static_cast<uint32_t>(index)) % 2048U) / 131072.0F;
  }
  for (size_t index = 0; index < matrix->input.size(); ++index) {
    matrix->input[index] = static_cast<float>(static_cast<int>(
        mix32(seed + 0x9e3779b9U + static_cast<uint32_t>(index)) % 65536U) - 32768) / 32768.0F;
  }
}

Matrix make_matrix(const char* stage, int layer, int input_width, int output_width, uint32_t seed) {
  Matrix matrix;
  matrix.stage = stage;
  matrix.layer = layer;
  matrix.input_width = input_width;
  matrix.output_width = output_width;
  matrix.weights.resize(static_cast<size_t>(input_width) * output_width);
  matrix.scales.resize(output_width);
  matrix.input.resize(input_width);
  matrix.reference.resize(output_width);
  matrix.scratch.resize(output_width);
  matrix.gpu.resize(output_width);
  fill_matrix(&matrix, seed);
  return matrix;
}

// Byte-for-byte the S=1 dense-Q8 reduction order used by qwen36.c:matmul_q.
void qwen_cpu_q8_matvec(float* output, const float* input, const int8_t* weights,
                        const float* scales, int input_width, int output_width) {
#if !defined(__AVX2__) || !defined(__FMA__)
#error "T11 requires the Qwen x86 AVX2/FMA CPU control"
#endif
  if (input_width % 32) std::abort();
#pragma omp parallel for schedule(static)
  for (int row = 0; row < output_width; ++row) {
    const int8_t* weight = weights + static_cast<size_t>(row) * input_width;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (int column = 0; column < input_width; column += 32) {
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
    output[row] = _mm_cvtss_f32(reduced) * scales[row];
  }
}

void cpu_sweep(std::vector<Matrix>* matrices, bool scratch) {
  for (Matrix& matrix : *matrices) {
    qwen_cpu_q8_matvec(scratch ? matrix.scratch.data() : matrix.reference.data(),
                        matrix.input.data(), matrix.weights.data(), matrix.scales.data(),
                        matrix.input_width, matrix.output_width);
  }
}

bool gpu_sweep(std::vector<Matrix>* matrices) {
  for (Matrix& matrix : *matrices) {
    if (!p40_cpuorder_matvec(matrix.tensor, matrix.gpu.data(), matrix.input.data())) return false;
  }
  return true;
}

Comparison compare_bits(const std::vector<Matrix>& matrices) {
  Comparison result;
  for (const Matrix& matrix : matrices) {
    if (std::memcmp(matrix.reference.data(), matrix.gpu.data(),
                    matrix.reference.size() * sizeof(float)) == 0) continue;
    result.bit_exact = false;
    for (size_t index = 0; index < matrix.reference.size(); ++index) {
      if (std::memcmp(&matrix.reference[index], &matrix.gpu[index], sizeof(float)) == 0) continue;
      ++result.mismatch_count;
      const double absolute = std::fabs(static_cast<double>(matrix.gpu[index]) - matrix.reference[index]);
      result.max_absolute = std::max(result.max_absolute, absolute);
      result.max_relative = std::max(result.max_relative,
          absolute / std::max(1.0, std::fabs(static_cast<double>(matrix.reference[index]))));
    }
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

size_t host_accounted_bytes(const std::vector<Matrix>& matrices) {
  size_t bytes = 0;
  for (const Matrix& matrix : matrices) {
    bytes += matrix.weights.size() * sizeof(int8_t);
    bytes += matrix.scales.size() * sizeof(float);
    bytes += matrix.input.size() * sizeof(float);
    bytes += (matrix.reference.size() + matrix.scratch.size() + matrix.gpu.size()) * sizeof(float);
  }
  return bytes;
}

void release_tensors(std::vector<Matrix>* matrices) {
  for (Matrix& matrix : *matrices) {
    p40_cpuorder_tensor_free(matrix.tensor);
    matrix.tensor = nullptr;
  }
}

int run(const Options& options) {
  std::vector<Matrix> matrices;
  matrices.reserve(kLayers * kMatricesPerLayer);
  for (int layer = 0; layer < kLayers; ++layer) {
    const uint32_t base = static_cast<uint32_t>(options.seed) + static_cast<uint32_t>(layer * 17);
    matrices.push_back(make_matrix("q", layer, kHidden, kQueryOutput, base + 1));
    matrices.push_back(make_matrix("k", layer, kHidden, kKvOutput, base + 2));
    matrices.push_back(make_matrix("v", layer, kHidden, kKvOutput, base + 3));
    matrices.push_back(make_matrix("o", layer, kOutputInput, kOutputOutput, base + 4));
  }
  const size_t bytes = host_accounted_bytes(matrices);
  if (bytes > static_cast<size_t>(options.memory_cap_mib) * 1024U * 1024U) {
    std::fprintf(stderr, "T11 host allocation exceeds memory cap\\n");
    return 2;
  }
  cpu_sweep(&matrices, false);

  std::vector<double> cpu_samples;
  for (int sample = 0; sample < options.repetitions; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    cpu_sweep(&matrices, true);
    cpu_samples.push_back(elapsed_ms(started));
  }

  if (!p40_cpuorder_init(options.gpu)) return 1;
  const auto first_started = std::chrono::steady_clock::now();
  bool gpu_ok = true;
  for (Matrix& matrix : matrices) {
    gpu_ok = p40_cpuorder_upload(&matrix.tensor, matrix.weights.data(), matrix.scales.data(),
                                  matrix.input_width, matrix.output_width);
    if (!gpu_ok) break;
  }
  if (gpu_ok) gpu_ok = gpu_sweep(&matrices);
  const double first_upload_and_sweep_ms = elapsed_ms(first_started);
  const Comparison comparison = compare_bits(matrices);
  std::vector<double> gpu_samples;
  for (int sample = 0; sample < options.repetitions && gpu_ok; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    gpu_ok = gpu_sweep(&matrices);
    if (gpu_ok) gpu_samples.push_back(elapsed_ms(started));
  }
  release_tensors(&matrices);
  p40_cpuorder_shutdown();

  const double cpu_median = median(cpu_samples);
  const double gpu_median = gpu_samples.empty() ? 0.0 : median(gpu_samples);
  const bool correct = gpu_ok && comparison.bit_exact;
  const size_t q8_weight_bytes = static_cast<size_t>(kLayers) *
      (static_cast<size_t>(kHidden) * kQueryOutput + 2U * kHidden * kKvOutput +
       static_cast<size_t>(kOutputInput) * kOutputOutput);
  std::cout << "{\"schema_version\":\"" << kSchema << "\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"gpu\":" << options.gpu
            << ",\"layers\":" << kLayers << ",\"matrix_count\":" << matrices.size()
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
