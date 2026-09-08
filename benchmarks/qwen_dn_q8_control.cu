// T05A: bounded cached Q8-weight/FP32-activation control for one Qwen
// DeltaNet projection. This fixture has no model reader and no model weights.
#include "backend_cuda.h"

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

constexpr int kInput = 2048;
constexpr int kOutput = 8192;
constexpr const char* kProfile = "dn_qkv-2048x8192";
constexpr double kAbsoluteTolerance = 1e-4;
constexpr double kRelativeTolerance = 1e-4;

struct Options {
  std::string profile;
  int gpu = -1;
  int repetitions = 0;
  int calls_per_sample = 0;
  int memory_cap_mib = 0;
  int seed = 0;
  bool dry_run = false;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: qwen_dn_q8_control --profile dn_qkv-2048x8192 --gpu 0 "
               "--repetitions 1..5 --calls-per-sample 1..8 "
               "--memory-cap-mib 32..64 --seed N [--dry-run]\n");
  std::exit(2);
}

bool parse_int(const char* text, int* output) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return false;
  }
  *output = static_cast<int>(value);
  return true;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--dry-run") {
      options.dry_run = true;
      continue;
    }
    if (index + 1 >= argc) usage();
    const char* value = argv[++index];
    if (argument == "--profile") options.profile = value;
    else if (argument == "--gpu") {
      if (!parse_int(value, &options.gpu)) usage();
    } else if (argument == "--repetitions") {
      if (!parse_int(value, &options.repetitions)) usage();
    } else if (argument == "--calls-per-sample") {
      if (!parse_int(value, &options.calls_per_sample)) usage();
    } else if (argument == "--memory-cap-mib") {
      if (!parse_int(value, &options.memory_cap_mib)) usage();
    } else if (argument == "--seed") {
      if (!parse_int(value, &options.seed)) usage();
    } else {
      usage();
    }
  }
  if (options.profile != kProfile || options.gpu != 0 ||
      options.repetitions < 1 || options.repetitions > 5 ||
      options.calls_per_sample < 1 || options.calls_per_sample > 8 ||
      options.memory_cap_mib < 32 || options.memory_cap_mib > 64) {
    usage();
  }
  return options;
}

uint32_t next_random(uint32_t* state) {
  *state = *state * 1664525U + 1013904223U;
  return *state;
}

void fill_inputs(std::vector<int8_t>* weights, std::vector<float>* scales,
                 std::vector<float>* input, int seed) {
  uint32_t state = static_cast<uint32_t>(seed);
  for (int8_t& weight : *weights) {
    weight = static_cast<int8_t>(static_cast<int>(next_random(&state) % 255U) - 127);
  }
  for (float& scale : *scales) {
    scale = 0.000125F + static_cast<float>(next_random(&state) % 2048U) / 131072.0F;
  }
  for (float& value : *input) {
    value = static_cast<float>(static_cast<int>(next_random(&state) % 65536U) - 32768) /
            32768.0F;
  }
}

// This is the same AVX2/FMA accumulation structure as qwen36.c:matmul_q for
// S=1 decode, including its four independent accumulators and output scale.
void qwen_cpu_q8_matvec(float* output, const float* input, const int8_t* weights,
                        const float* scales) {
#if !defined(__AVX2__) || !defined(__FMA__)
#error "T05A requires the Qwen x86 AVX2/FMA CPU control"
#endif
#pragma omp parallel for schedule(static) if(kOutput >= 256)
  for (int row = 0; row < kOutput; ++row) {
    const int8_t* weight = weights + static_cast<size_t>(row) * kInput;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int column = 0;
    for (; column + 32 <= kInput; column += 32) {
      const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + column));
      const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + column + 16));
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column),
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 8),
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8))), a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 16),
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(input + column + 24),
                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8))), a3);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    reduced = _mm_add_ps(reduced, _mm_movehl_ps(reduced, reduced));
    reduced = _mm_add_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
    float total = _mm_cvtss_f32(reduced);
    for (; column < kInput; ++column) total += input[column] * static_cast<float>(weight[column]);
    output[row] = total * scales[row];
  }
}

double elapsed_ms(const std::chrono::steady_clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2.0;
}

void print_samples(const std::vector<double>& samples) {
  std::cout << '[';
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << samples[index];
  }
  std::cout << ']';
}

struct ErrorSummary {
  bool correct = true;
  double max_abs = 0.0;
  double max_relative = 0.0;
};

ErrorSummary compare_outputs(const std::vector<float>& reference, const std::vector<float>& observed) {
  ErrorSummary result;
  for (size_t index = 0; index < reference.size(); ++index) {
    const double expected = reference[index];
    const double actual = observed[index];
    const double absolute = std::fabs(actual - expected);
    const double relative = absolute / std::max(1.0, std::fabs(expected));
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_relative = std::max(result.max_relative, relative);
    if (!std::isfinite(actual) || absolute > kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected)) {
      result.correct = false;
    }
  }
  return result;
}

int run(const Options& options) {
  const size_t weight_bytes = static_cast<size_t>(kInput) * kOutput;
  const size_t host_bytes = weight_bytes + static_cast<size_t>(kOutput) * sizeof(float) * 3U +
                            static_cast<size_t>(kInput) * sizeof(float);
  if (host_bytes > static_cast<size_t>(options.memory_cap_mib) * 1024U * 1024U) {
    std::fprintf(stderr, "T05A host allocation exceeds memory cap\n");
    return 2;
  }
  std::vector<int8_t> weights(weight_bytes);
  std::vector<float> scales(kOutput);
  std::vector<float> input(kInput);
  std::vector<float> reference(kOutput);
  std::vector<float> scratch(kOutput);
  std::vector<float> gpu_output(kOutput);
  fill_inputs(&weights, &scales, &input, options.seed);

  qwen_cpu_q8_matvec(reference.data(), input.data(), weights.data(), scales.data());
  std::vector<double> cpu_samples;
  for (int sample = 0; sample < options.repetitions; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (int call = 0; call < options.calls_per_sample; ++call) {
      qwen_cpu_q8_matvec(scratch.data(), input.data(), weights.data(), scales.data());
    }
    cpu_samples.push_back(elapsed_ms(started) / options.calls_per_sample);
  }

  const int device = options.gpu;
  if (!coli_cuda_init(&device, 1)) {
    std::fprintf(stderr, "T05A CUDA initialization failed\n");
    return 1;
  }
  ColiCudaTensor* tensor = nullptr;
  const auto first_started = std::chrono::steady_clock::now();
  const int first_ok = coli_cuda_matmul(&tensor, gpu_output.data(), input.data(), weights.data(),
                                        scales.data(), 1, 1, kInput, kOutput, device, 0);
  const double first_call_ms = elapsed_ms(first_started);
  if (!first_ok || !tensor) {
    std::fprintf(stderr, "T05A first CUDA Q8 matvec failed\n");
    coli_cuda_shutdown();
    return 1;
  }
  const ErrorSummary error = compare_outputs(reference, gpu_output);

  std::vector<double> gpu_samples;
  bool cached_ok = true;
  for (int sample = 0; sample < options.repetitions && cached_ok; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (int call = 0; call < options.calls_per_sample; ++call) {
      if (!coli_cuda_matmul(&tensor, gpu_output.data(), input.data(), nullptr, nullptr,
                            1, 1, kInput, kOutput, device, 0)) {
        cached_ok = false;
        break;
      }
    }
    if (cached_ok) gpu_samples.push_back(elapsed_ms(started) / options.calls_per_sample);
  }
  size_t tensor_count = 0;
  size_t tensor_bytes = 0;
  coli_cuda_stats(device, &tensor_count, &tensor_bytes);
  coli_cuda_tensor_free(tensor);
  coli_cuda_shutdown();

  const double cpu_median = median(cpu_samples);
  const double gpu_median = gpu_samples.empty() ? 0.0 : median(gpu_samples);
  const double speedup = gpu_median > 0.0 ? cpu_median / gpu_median : 0.0;
  std::cout << "{\"schema_version\":\"t05a-q8-gemv-v1\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"gpu\":" << device
            << ",\"input\":" << kInput << ",\"output\":" << kOutput
            << ",\"weight_bytes\":" << weight_bytes
            << ",\"first_call_ms\":" << first_call_ms
            << ",\"cpu_ms_per_call\":";
  print_samples(cpu_samples);
  std::cout << ",\"gpu_complete_ms_per_call\":";
  print_samples(gpu_samples);
  std::cout << ",\"cpu_median_ms\":" << cpu_median
            << ",\"gpu_complete_median_ms\":" << gpu_median
            << ",\"speedup_cpu_over_gpu\":" << speedup
            << ",\"cached_tensor_count\":" << tensor_count
            << ",\"cached_tensor_bytes\":" << tensor_bytes
            << ",\"per_stage_metrics\":null"
            << ",\"correct\":" << (error.correct && cached_ok ? "true" : "false")
            << ",\"max_abs_error\":" << error.max_abs
            << ",\"max_relative_error\":" << error.max_relative << "}" << std::endl;
  return error.correct && cached_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"t05a-q8-gemv-v1\",\"profile\":\"" << kProfile
              << "\",\"cuda_initialized\":false}" << std::endl;
    return 0;
  }
  return run(options);
}
