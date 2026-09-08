// T05C: cache-aware Qwen-shaped DeltaNet projection sweep.
//
// This fixture deliberately owns 30 distinct (qkv, z, out) Q8 projection
// triplets: the same count and 32 MiB/layer Q8 working set as Qwen3.6's 30
// DeltaNet layers.  It is still synthetic and does not parse model weights.
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
constexpr int kLayers = 30;
constexpr const char* kProfile = "dn-sweep-30x-triplet-2048-8192-4096";
constexpr double kAbsoluteTolerance = 1e-4;
constexpr double kRelativeTolerance = 1e-4;

struct Options {
  std::string profile;
  int gpu = -1;
  int repetitions = 0;
  int sweeps_per_sample = 0;
  int memory_cap_mib = 0;
  int seed = 0;
  bool dry_run = false;
};

struct Projection {
  int input;
  int output;
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

struct Layer {
  Projection qkv{kHidden, kConv};
  Projection z{kHidden, kValue};
  Projection out{kValue, kHidden};
};

struct ErrorSummary {
  bool correct = true;
  double max_absolute = 0.0;
  double max_relative = 0.0;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: qwen_dn_sweep_control --profile dn-sweep-30x-triplet-2048-8192-4096 --gpu 0 "
               "--repetitions 1..3 --sweeps-per-sample 1 --memory-cap-mib 1024..1088 --seed N [--dry-run]\\n");
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
    else if (argument == "--sweeps-per-sample") { if (!parse_int(value, &options.sweeps_per_sample)) usage(); }
    else if (argument == "--memory-cap-mib") { if (!parse_int(value, &options.memory_cap_mib)) usage(); }
    else if (argument == "--seed") { if (!parse_int(value, &options.seed)) usage(); }
    else usage();
  }
  if (options.profile != kProfile || options.gpu != 0 || options.repetitions < 1 ||
      options.repetitions > 3 || options.sweeps_per_sample != 1 ||
      options.memory_cap_mib < 1024 || options.memory_cap_mib > 1088) usage();
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

void fill_projection(Projection* projection, uint32_t seed) {
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(projection->weights.size()); ++index) {
    const uint32_t value = mix32(seed ^ static_cast<uint32_t>(index));
    projection->weights[static_cast<size_t>(index)] = static_cast<int8_t>(static_cast<int>(value % 255U) - 127);
  }
#pragma omp parallel for schedule(static)
  for (long long index = 0; index < static_cast<long long>(projection->scales.size()); ++index) {
    const uint32_t value = mix32(seed + static_cast<uint32_t>(index));
    projection->scales[static_cast<size_t>(index)] = 0.000125F + static_cast<float>(value % 2048U) / 131072.0F;
  }
}

void fill_hidden(std::vector<float>* hidden, uint32_t seed) {
  for (size_t index = 0; index < hidden->size(); ++index) {
    const uint32_t value = mix32(seed ^ static_cast<uint32_t>(index));
    (*hidden)[index] = static_cast<float>(static_cast<int>(value % 65536U) - 32768) / 32768.0F;
  }
}

// Qwen's x86 S=1 Q8 dense operator: FP32 activation, int8 row weights,
// per-output FP32 scale, and four AVX2/FMA accumulators.
void qwen_cpu_q8_matvec(float* output, const float* input, const Projection& projection) {
#if !defined(__AVX2__) || !defined(__FMA__)
#error "T05C requires the Qwen x86 AVX2/FMA CPU control"
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

// The real recurrence remains CPU-owned. This intentionally small, explicit
// CPU boundary tests the actual host round trip between qkv/z and dn_out.
void make_out_input(float* output, const float* qkv, const float* z) {
  for (int index = 0; index < kValue; ++index) output[index] = 0.5F * qkv[index] + 0.5F * z[index];
}

void cpu_sweep(std::vector<Layer>* layers, const std::vector<float>& hidden,
               std::vector<float>* boundary) {
  for (Layer& layer : *layers) {
    qwen_cpu_q8_matvec(layer.qkv.cpu.data(), hidden.data(), layer.qkv);
    qwen_cpu_q8_matvec(layer.z.cpu.data(), hidden.data(), layer.z);
    make_out_input(boundary->data(), layer.qkv.cpu.data(), layer.z.cpu.data());
    qwen_cpu_q8_matvec(layer.out.cpu.data(), boundary->data(), layer.out);
  }
}

bool gpu_projection(Projection* projection, const float* input, int device, bool cached) {
  return coli_cuda_matmul(&projection->tensor, projection->gpu.data(), input,
                           cached ? nullptr : projection->weights.data(),
                           cached ? nullptr : projection->scales.data(),
                           1, 1, projection->input, projection->output, device, 0) != 0;
}

bool gpu_sweep(std::vector<Layer>* layers, const std::vector<float>& hidden,
               std::vector<float>* boundary, int device, bool cached) {
  for (Layer& layer : *layers) {
    if (!gpu_projection(&layer.qkv, hidden.data(), device, cached) ||
        !gpu_projection(&layer.z, hidden.data(), device, cached)) return false;
    make_out_input(boundary->data(), layer.qkv.gpu.data(), layer.z.gpu.data());
    if (!gpu_projection(&layer.out, boundary->data(), device, cached)) return false;
  }
  return true;
}

void merge(ErrorSummary* total, const std::vector<float>& expected, const std::vector<float>& actual) {
  for (size_t index = 0; index < expected.size(); ++index) {
    const double absolute = std::fabs(static_cast<double>(actual[index]) - expected[index]);
    const double relative = absolute / std::max(1.0, std::fabs(static_cast<double>(expected[index])));
    total->max_absolute = std::max(total->max_absolute, absolute);
    total->max_relative = std::max(total->max_relative, relative);
    if (!std::isfinite(actual[index]) || absolute > kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected[index])) {
      total->correct = false;
    }
  }
}

void print_error(const ErrorSummary& error) {
  std::cout << "{\"correct\":" << (error.correct ? "true" : "false")
            << ",\"max_abs_error\":" << error.max_absolute
            << ",\"max_relative_error\":" << error.max_relative << '}';
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

int run(const Options& options) {
  std::vector<Layer> layers;
  layers.reserve(kLayers);
  for (int layer = 0; layer < kLayers; ++layer) layers.emplace_back();
  size_t q8_bytes = 0;
  size_t host_bytes = 0;
  for (const Layer& layer : layers) {
    for (const Projection* projection : {&layer.qkv, &layer.z, &layer.out}) {
      q8_bytes += projection->weights.size();
      host_bytes += projection->weights.size() * sizeof(int8_t);
      host_bytes += projection->scales.size() * sizeof(float);
      host_bytes += (projection->cpu.size() + projection->gpu.size()) * sizeof(float);
    }
  }
  host_bytes += static_cast<size_t>(kHidden + kValue * 2) * sizeof(float);
  if (host_bytes > static_cast<size_t>(options.memory_cap_mib) * 1024U * 1024U) {
    std::fprintf(stderr, "T05C host allocation exceeds memory cap\\n");
    return 2;
  }
  for (int layer = 0; layer < kLayers; ++layer) {
    Layer& current = layers[static_cast<size_t>(layer)];
    const uint32_t base = static_cast<uint32_t>(options.seed) + static_cast<uint32_t>(layer) * 0x9e3779b9U;
    fill_projection(&current.qkv, mix32(base ^ 0x11111111U));
    fill_projection(&current.z, mix32(base ^ 0x22222222U));
    fill_projection(&current.out, mix32(base ^ 0x33333333U));
  }
  std::vector<float> hidden(kHidden), cpu_boundary(kValue), gpu_boundary(kValue);
  fill_hidden(&hidden, static_cast<uint32_t>(options.seed));

  // This establishes the exact CPU reference. Each timed CPU sample walks the
  // 30 distinct layers once; it cannot reduce to repeated work on one layer.
  cpu_sweep(&layers, hidden, &cpu_boundary);
  std::vector<double> cpu_samples;
  for (int sample = 0; sample < options.repetitions; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    cpu_sweep(&layers, hidden, &cpu_boundary);
    cpu_samples.push_back(elapsed_ms(started));
  }

  const int device = options.gpu;
  if (!coli_cuda_init(&device, 1)) {
    std::fprintf(stderr, "T05C CUDA initialization failed\\n");
    return 1;
  }
  const auto first_started = std::chrono::steady_clock::now();
  const bool first_ok = gpu_sweep(&layers, hidden, &gpu_boundary, device, false);
  const double first_sweep_ms = elapsed_ms(first_started);
  ErrorSummary error;
  ErrorSummary qkv_error;
  ErrorSummary z_error;
  ErrorSummary out_error;
  ErrorSummary out_kernel_error;
  ErrorSummary boundary_error;
  std::vector<float> diagnostic_boundary(kValue);
  std::vector<float> cpu_from_gpu_boundary(kHidden);
  for (const Layer& layer : layers) {
    merge(&qkv_error, layer.qkv.cpu, layer.qkv.gpu);
    merge(&z_error, layer.z.cpu, layer.z.gpu);
    merge(&out_error, layer.out.cpu, layer.out.gpu);
    make_out_input(diagnostic_boundary.data(), layer.qkv.gpu.data(), layer.z.gpu.data());
    qwen_cpu_q8_matvec(cpu_from_gpu_boundary.data(), diagnostic_boundary.data(), layer.out);
    merge(&out_kernel_error, cpu_from_gpu_boundary, layer.out.gpu);
    merge(&boundary_error, layer.out.cpu, cpu_from_gpu_boundary);
  }
  error.correct = qkv_error.correct && z_error.correct && out_error.correct;
  error.max_absolute = std::max({qkv_error.max_absolute, z_error.max_absolute, out_error.max_absolute});
  error.max_relative = std::max({qkv_error.max_relative, z_error.max_relative, out_error.max_relative});

  std::vector<double> gpu_samples;
  bool cached_ok = first_ok;
  for (int sample = 0; sample < options.repetitions && cached_ok; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    cached_ok = gpu_sweep(&layers, hidden, &gpu_boundary, device, true);
    if (cached_ok) gpu_samples.push_back(elapsed_ms(started));
  }
  size_t tensor_count = 0;
  size_t tensor_bytes = 0;
  coli_cuda_stats(device, &tensor_count, &tensor_bytes);
  for (Layer& layer : layers) {
    coli_cuda_tensor_free(layer.qkv.tensor);
    coli_cuda_tensor_free(layer.z.tensor);
    coli_cuda_tensor_free(layer.out.tensor);
  }
  coli_cuda_shutdown();

  const double cpu_median = median(cpu_samples);
  const double gpu_median = gpu_samples.empty() ? 0.0 : median(gpu_samples);
  const bool correct = first_ok && cached_ok && error.correct;
  std::cout << "{\"schema_version\":\"t05c-q8-sweep-v1\",\"profile\":\"" << kProfile
            << "\",\"cuda_initialized\":true,\"gpu\":" << device
            << ",\"layers\":" << kLayers << ",\"q8_weight_bytes\":" << q8_bytes
            << ",\"host_accounted_bytes\":" << host_bytes
            << ",\"first_sweep_ms\":" << first_sweep_ms
            << ",\"cpu_ms_per_sweep\":";
  print_samples(cpu_samples);
  std::cout << ",\"gpu_complete_ms_per_sweep\":";
  print_samples(gpu_samples);
  std::cout << ",\"cpu_median_ms\":" << cpu_median
            << ",\"gpu_complete_median_ms\":" << gpu_median
            << ",\"speedup_cpu_over_gpu\":" << (gpu_median > 0.0 ? cpu_median / gpu_median : 0.0)
            << ",\"cached_tensor_count\":" << tensor_count
            << ",\"cached_tensor_bytes\":" << tensor_bytes
            << ",\"per_stage_metrics\":null"
            << ",\"error_by_stage\":{\"qkv\":";
  print_error(qkv_error);
  std::cout << ",\"z\":";
  print_error(z_error);
  std::cout << ",\"out_composite\":";
  print_error(out_error);
  std::cout << ",\"out_kernel_with_gpu_boundary\":";
  print_error(out_kernel_error);
  std::cout << ",\"boundary_propagation\":";
  print_error(boundary_error);
  std::cout << '}'
            << ",\"correct\":" << (correct ? "true" : "false")
            << ",\"max_abs_error\":" << error.max_absolute
            << ",\"max_relative_error\":" << error.max_relative << "}" << std::endl;
  return correct ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"t05c-q8-sweep-v1\",\"profile\":\"" << kProfile
              << "\",\"cuda_initialized\":false}" << std::endl;
    return 0;
  }
  return run(options);
}
