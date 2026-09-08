// T05B: a bounded synthetic DeltaNet Q8 projection triplet. It shares the
// actual host boundary that Qwen has between dn_qkv/dn_z and dn_out, but does
// not load a model or emulate the recurrent state math between those phases.
#include "backend_cuda.h"

#include <immintrin.h>
#include <omp.h>

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

constexpr int kHidden = 2048;
constexpr int kConv = 8192;
constexpr int kValue = 4096;
constexpr const char* kProfile = "dn-triplet-2048-8192-4096";
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

struct Projection {
  int input = 0;
  int output = 0;
  std::vector<int8_t> weights;
  std::vector<float> scales;
  std::vector<float> cpu;
  std::vector<float> gpu;
  ColiCudaTensor* tensor = nullptr;

  Projection(int input_width, int output_width)
      : input(input_width), output(output_width),
        weights(static_cast<size_t>(input_width) * output_width),
        scales(output_width), cpu(output_width), gpu(output_width) {}
};

struct ErrorSummary {
  bool correct = true;
  double max_absolute = 0.0;
  double max_relative = 0.0;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: qwen_dn_triplet_control --profile dn-triplet-2048-8192-4096 --gpu 0 "
               "--repetitions 1..5 --calls-per-sample 1..4 --memory-cap-mib 48..64 --seed N [--dry-run]\n");
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
    if (argument == "--dry-run") {
      options.dry_run = true;
      continue;
    }
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
      options.repetitions > 5 || options.calls_per_sample < 1 || options.calls_per_sample > 4 ||
      options.memory_cap_mib < 48 || options.memory_cap_mib > 64) usage();
  return options;
}

uint32_t next_random(uint32_t* state) {
  *state = *state * 1664525U + 1013904223U;
  return *state;
}

void fill_projection(Projection* projection, uint32_t* state) {
  for (int8_t& weight : projection->weights) {
    weight = static_cast<int8_t>(static_cast<int>(next_random(state) % 255U) - 127);
  }
  for (float& scale : projection->scales) {
    scale = 0.000125F + static_cast<float>(next_random(state) % 2048U) / 131072.0F;
  }
}

void qwen_cpu_q8_matvec(float* output, const float* input, const Projection& projection) {
#if !defined(__AVX2__) || !defined(__FMA__)
#error "T05B requires the Qwen x86 AVX2/FMA CPU control"
#endif
#pragma omp parallel for schedule(static) if(projection.output >= 256)
  for (int row = 0; row < projection.output; ++row) {
    const int8_t* weight = projection.weights.data() + static_cast<size_t>(row) * projection.input;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int column = 0;
    for (; column + 32 <= projection.input; column += 32) {
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
    float total = _mm_cvtss_f32(reduced);
    for (; column < projection.input; ++column) total += input[column] * static_cast<float>(weight[column]);
    output[row] = total * projection.scales[row];
  }
}

// This intentionally small CPU boundary establishes the direction and host
// residency of the actual DeltaNet phase boundary without pretending to model
// its recurrent operation. Both controls execute it on the CPU.
void make_out_input(float* output, const float* qkv, const float* z) {
  for (int index = 0; index < kValue; ++index) output[index] = 0.5F * qkv[index] + 0.5F * z[index];
}

void cpu_triplet(Projection* qkv, Projection* z, Projection* out,
                 const std::vector<float>& hidden, std::vector<float>* out_input) {
  qwen_cpu_q8_matvec(qkv->cpu.data(), hidden.data(), *qkv);
  qwen_cpu_q8_matvec(z->cpu.data(), hidden.data(), *z);
  make_out_input(out_input->data(), qkv->cpu.data(), z->cpu.data());
  qwen_cpu_q8_matvec(out->cpu.data(), out_input->data(), *out);
}

bool gpu_projection(Projection* projection, const float* input, int device, bool cached) {
  return coli_cuda_matmul(&projection->tensor, projection->gpu.data(), input,
                           cached ? nullptr : projection->weights.data(),
                           cached ? nullptr : projection->scales.data(),
                           1, 1, projection->input, projection->output, device, 0) != 0;
}

bool gpu_triplet(Projection* qkv, Projection* z, Projection* out,
                 const std::vector<float>& hidden, std::vector<float>* out_input,
                 int device, bool cached) {
  if (!gpu_projection(qkv, hidden.data(), device, cached) ||
      !gpu_projection(z, hidden.data(), device, cached)) return false;
  make_out_input(out_input->data(), qkv->gpu.data(), z->gpu.data());
  return gpu_projection(out, out_input->data(), device, cached);
}

ErrorSummary compare(const std::vector<float>& expected, const std::vector<float>& actual) {
  ErrorSummary result;
  for (size_t index = 0; index < expected.size(); ++index) {
    const double absolute = std::fabs(static_cast<double>(actual[index]) - expected[index]);
    const double relative = absolute / std::max(1.0, std::fabs(static_cast<double>(expected[index])));
    result.max_absolute = std::max(result.max_absolute, absolute);
    result.max_relative = std::max(result.max_relative, relative);
    if (!std::isfinite(actual[index]) || absolute > kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected[index])) {
      result.correct = false;
    }
  }
  return result;
}

void merge(ErrorSummary* total, const ErrorSummary& next) {
  total->correct = total->correct && next.correct;
  total->max_absolute = std::max(total->max_absolute, next.max_absolute);
  total->max_relative = std::max(total->max_relative, next.max_relative);
}

double milliseconds(const std::chrono::steady_clock::time_point& started) {
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

int run(const Options& options) {
  Projection qkv(kHidden, kConv), z(kHidden, kValue), out(kValue, kHidden);
  const size_t q8_bytes = qkv.weights.size() + z.weights.size() + out.weights.size();
  const size_t host_bytes = q8_bytes + (qkv.scales.size() + z.scales.size() + out.scales.size()) * sizeof(float) +
      static_cast<size_t>(kHidden + kValue) * sizeof(float) +
      (qkv.cpu.size() + qkv.gpu.size() + z.cpu.size() + z.gpu.size() + out.cpu.size() + out.gpu.size()) * sizeof(float);
  if (host_bytes > static_cast<size_t>(options.memory_cap_mib) * 1024U * 1024U) {
    std::fprintf(stderr, "T05B host allocation exceeds memory cap\n");
    return 2;
  }
  uint32_t state = static_cast<uint32_t>(options.seed);
  fill_projection(&qkv, &state); fill_projection(&z, &state); fill_projection(&out, &state);
  std::vector<float> hidden(kHidden), cpu_out_input(kValue), gpu_out_input(kValue);
  for (float& value : hidden) value = static_cast<float>(static_cast<int>(next_random(&state) % 65536U) - 32768) / 32768.0F;

  cpu_triplet(&qkv, &z, &out, hidden, &cpu_out_input);
  std::vector<double> cpu_samples;
  for (int sample = 0; sample < options.repetitions; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (int call = 0; call < options.calls_per_sample; ++call) cpu_triplet(&qkv, &z, &out, hidden, &cpu_out_input);
    cpu_samples.push_back(milliseconds(started) / options.calls_per_sample);
  }

  const int device = options.gpu;
  if (!coli_cuda_init(&device, 1)) {
    std::fprintf(stderr, "T05B CUDA initialization failed\n");
    return 1;
  }
  const auto first_started = std::chrono::steady_clock::now();
  const bool first_ok = gpu_triplet(&qkv, &z, &out, hidden, &gpu_out_input, device, false);
  const double first_triplet_ms = milliseconds(first_started);
  ErrorSummary error;
  merge(&error, compare(qkv.cpu, qkv.gpu));
  merge(&error, compare(z.cpu, z.gpu));
  merge(&error, compare(out.cpu, out.gpu));

  std::vector<double> gpu_samples;
  bool cached_ok = first_ok;
  for (int sample = 0; sample < options.repetitions && cached_ok; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    for (int call = 0; call < options.calls_per_sample; ++call) {
      if (!gpu_triplet(&qkv, &z, &out, hidden, &gpu_out_input, device, true)) {
        cached_ok = false;
        break;
      }
    }
    if (cached_ok) gpu_samples.push_back(milliseconds(started) / options.calls_per_sample);
  }
  size_t tensor_count = 0, tensor_bytes = 0;
  coli_cuda_stats(device, &tensor_count, &tensor_bytes);
  coli_cuda_tensor_free(qkv.tensor); coli_cuda_tensor_free(z.tensor); coli_cuda_tensor_free(out.tensor);
  coli_cuda_shutdown();

  const double cpu_median = median(cpu_samples);
  const double gpu_median = gpu_samples.empty() ? 0.0 : median(gpu_samples);
  const double speedup = gpu_median > 0.0 ? cpu_median / gpu_median : 0.0;
  std::cout << "{\"schema_version\":\"t05b-q8-triplet-v1\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"gpu\":" << device
            << ",\"q8_weight_bytes\":" << q8_bytes
            << ",\"first_triplet_ms\":" << first_triplet_ms
            << ",\"cpu_ms_per_triplet\":";
  print_samples(cpu_samples);
  std::cout << ",\"gpu_complete_ms_per_triplet\":";
  print_samples(gpu_samples);
  std::cout << ",\"cpu_median_ms\":" << cpu_median
            << ",\"gpu_complete_median_ms\":" << gpu_median
            << ",\"speedup_cpu_over_gpu\":" << speedup
            << ",\"cached_tensor_count\":" << tensor_count
            << ",\"cached_tensor_bytes\":" << tensor_bytes
            << ",\"per_stage_metrics\":null"
            << ",\"correct\":" << (first_ok && cached_ok && error.correct ? "true" : "false")
            << ",\"max_abs_error\":" << error.max_absolute
            << ",\"max_relative_error\":" << error.max_relative << "}" << std::endl;
  return first_ok && cached_ok && error.correct ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"t05b-q8-triplet-v1\",\"profile\":\"" << kProfile
              << "\",\"cuda_initialized\":false}" << std::endl;
    return 0;
  }
  return run(options);
}
