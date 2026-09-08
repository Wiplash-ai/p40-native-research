// Small, bounded P40 primitive suite.  It intentionally has no model loader.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string primitive;
  int gpu = -1;
  size_t bytes = 0;
  int duration_seconds = 0;
  int memory_cap_mib = 0;
  int seed = 0;
  bool dry_run = false;
};

void check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(status));
    std::exit(1);
  }
}

__global__ void fill_kernel(float* values, size_t count, float seed) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) values[index] = seed + static_cast<float>(index % 251) * 0.00390625f;
}

__global__ void copy_kernel(float* output, const float* input, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index];
}

__global__ void dp4a_gemv_kernel(const int8_t* weights, const int8_t* input,
                                  int32_t* output, int rows, int columns) {
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  int accumulator = 0;
  for (int column = thread * 4; column < columns; column += blockDim.x * 4) {
    const int packed_weight = *reinterpret_cast<const int*>(weights + row * columns + column);
    const int packed_input = *reinterpret_cast<const int*>(input + column);
    accumulator = __dp4a(packed_weight, packed_input, accumulator);
  }
  __shared__ int32_t partial[256];
  partial[thread] = accumulator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (thread < stride) partial[thread] += partial[thread + stride];
    __syncthreads();
  }
  if (thread == 0) output[row] = partial[0];
}

bool parse_int(const char* value, int* result) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!value[0] || *end || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) return false;
  *result = static_cast<int>(parsed);
  return true;
}

bool parse_size(const char* value, size_t* result) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (!value[0] || *end || parsed > std::numeric_limits<size_t>::max()) return false;
  *result = static_cast<size_t>(parsed);
  return true;
}

void usage() {
  std::cerr << "usage: p40_bench --primitive P01-copy|P03-dp4a-gemv --gpu 0|1 --bytes N --duration 1|5|10 --memory-cap-mib N --seed N [--dry-run]\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--dry-run") { options.dry_run = true; continue; }
    if (index + 1 >= argc) { usage(); std::exit(2); }
    const char* value = argv[++index];
    if (argument == "--primitive") options.primitive = value;
    else if (argument == "--gpu") { if (!parse_int(value, &options.gpu)) { usage(); std::exit(2); } }
    else if (argument == "--bytes") { if (!parse_size(value, &options.bytes)) { usage(); std::exit(2); } }
    else if (argument == "--duration") { if (!parse_int(value, &options.duration_seconds)) { usage(); std::exit(2); } }
    else if (argument == "--memory-cap-mib") { if (!parse_int(value, &options.memory_cap_mib)) { usage(); std::exit(2); } }
    else if (argument == "--seed") { if (!parse_int(value, &options.seed)) { usage(); std::exit(2); } }
    else { usage(); std::exit(2); }
  }
  if ((options.primitive != "P01-copy" && options.primitive != "P03-dp4a-gemv") ||
      (options.gpu != 0 && options.gpu != 1) || options.bytes == 0 ||
      (options.duration_seconds != 1 && options.duration_seconds != 5 && options.duration_seconds != 10) ||
      options.memory_cap_mib < 1) {
    usage(); std::exit(2);
  }
  return options;
}

double timed_copy(float* output, const float* input, size_t count, int seconds, uint64_t* iterations) {
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  double milliseconds = 0.0;
  *iterations = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    check(cudaEventRecord(start), "record start event");
    for (int repeat = 0; repeat < 32; ++repeat) {
      copy_kernel<<<blocks, threads>>>(output, input, count);
      ++*iterations;
    }
    check(cudaGetLastError(), "launch copy kernel");
    check(cudaEventRecord(stop), "record stop event");
    check(cudaEventSynchronize(stop), "synchronize copy event");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "measure copy event");
    milliseconds += elapsed;
  }
  check(cudaEventDestroy(start), "destroy start event");
  check(cudaEventDestroy(stop), "destroy stop event");
  return milliseconds;
}

double timed_dp4a(const int8_t* weights, const int8_t* input, int32_t* output,
                  int rows, int columns, int seconds, uint64_t* iterations) {
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  double milliseconds = 0.0;
  *iterations = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    check(cudaEventRecord(start), "record start event");
    for (int repeat = 0; repeat < 32; ++repeat) {
      dp4a_gemv_kernel<<<rows, 256>>>(weights, input, output, rows, columns);
      ++*iterations;
    }
    check(cudaGetLastError(), "launch DP4A GEMV kernel");
    check(cudaEventRecord(stop), "record stop event");
    check(cudaEventSynchronize(stop), "synchronize DP4A event");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "measure DP4A event");
    milliseconds += elapsed;
  }
  check(cudaEventDestroy(start), "destroy start event");
  check(cudaEventDestroy(stop), "destroy stop event");
  return milliseconds;
}

void print_copy(const Options& options) {
  const size_t count = options.bytes / sizeof(float);
  const size_t allocation = options.bytes * 2;
  if (allocation > static_cast<size_t>(options.memory_cap_mib) * 1024 * 1024) {
    std::fprintf(stderr, "P01 allocation exceeds memory cap\n"); std::exit(2);
  }
  float *input = nullptr, *output = nullptr;
  check(cudaMalloc(&input, options.bytes), "allocate P01 input");
  check(cudaMalloc(&output, options.bytes), "allocate P01 output");
  const int blocks = static_cast<int>((count + 255) / 256);
  fill_kernel<<<blocks, 256>>>(input, count, static_cast<float>(options.seed));
  check(cudaGetLastError(), "initialize P01 input");
  check(cudaDeviceSynchronize(), "synchronize P01 initialization");
  uint64_t iterations = 0;
  const double milliseconds = timed_copy(output, input, count, options.duration_seconds, &iterations);
  float expected = 0, observed = 0;
  check(cudaMemcpy(&expected, input + count - 1, sizeof(float), cudaMemcpyDeviceToHost), "copy P01 expected");
  check(cudaMemcpy(&observed, output + count - 1, sizeof(float), cudaMemcpyDeviceToHost), "copy P01 observed");
  const bool correct = expected == observed;
  const double bytes_moved = static_cast<double>(options.bytes) * 2.0 * static_cast<double>(iterations);
  const double gb_per_second = bytes_moved / (milliseconds * 1.0e6);
  std::cout << "{\"schema_version\":\"p40-bench-v1\",\"primitive\":\"P01-copy\",\"gpu\":" << options.gpu
            << ",\"bytes_per_array\":" << options.bytes << ",\"iterations\":" << iterations
            << ",\"elapsed_ms\":" << milliseconds << ",\"algorithmic_gb_per_s\":" << gb_per_second
            << ",\"correct\":" << (correct ? "true" : "false") << "}" << std::endl;
  check(cudaFree(input), "free P01 input");
  check(cudaFree(output), "free P01 output");
  if (!correct) std::exit(1);
}

uint32_t next_random(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

void print_dp4a(const Options& options) {
  constexpr int rows = 512;
  constexpr int columns = 2048;  // multiple of 4; one Qwen-like decode shape.
  std::vector<int8_t> weights(static_cast<size_t>(rows) * columns);
  std::vector<int8_t> input(columns);
  uint32_t random = static_cast<uint32_t>(options.seed);
  for (int8_t& value : weights) value = static_cast<int8_t>(static_cast<int>(next_random(&random) % 15) - 7);
  for (int8_t& value : input) value = static_cast<int8_t>(static_cast<int>(next_random(&random) % 15) - 7);
  std::vector<int64_t> reference(rows, 0);
  for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column)
    reference[row] += static_cast<int64_t>(weights[static_cast<size_t>(row) * columns + column]) * input[column];
  const size_t allocation = weights.size() + input.size() + static_cast<size_t>(rows) * sizeof(int32_t);
  if (allocation > static_cast<size_t>(options.memory_cap_mib) * 1024 * 1024) {
    std::fprintf(stderr, "P03 allocation exceeds memory cap\n"); std::exit(2);
  }
  int8_t *device_weights = nullptr, *device_input = nullptr;
  int32_t* device_output = nullptr;
  check(cudaMalloc(&device_weights, weights.size()), "allocate P03 weights");
  check(cudaMalloc(&device_input, input.size()), "allocate P03 input");
  check(cudaMalloc(&device_output, static_cast<size_t>(rows) * sizeof(int32_t)), "allocate P03 output");
  check(cudaMemcpy(device_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice), "upload P03 weights");
  check(cudaMemcpy(device_input, input.data(), input.size(), cudaMemcpyHostToDevice), "upload P03 input");
  uint64_t iterations = 0;
  const double milliseconds = timed_dp4a(device_weights, device_input, device_output, rows, columns, options.duration_seconds, &iterations);
  std::vector<int32_t> observed(rows);
  check(cudaMemcpy(observed.data(), device_output, observed.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download P03 output");
  bool correct = true;
  for (int row = 0; row < rows; ++row) if (reference[row] != observed[row]) correct = false;
  const double macs = static_cast<double>(rows) * columns * iterations;
  const double gigaops = macs * 2.0 / (milliseconds * 1.0e6);
  std::cout << "{\"schema_version\":\"p40-bench-v1\",\"primitive\":\"P03-dp4a-gemv\",\"gpu\":" << options.gpu
            << ",\"rows\":" << rows << ",\"columns\":" << columns << ",\"iterations\":" << iterations
            << ",\"elapsed_ms\":" << milliseconds << ",\"int8_gops\":" << gigaops
            << ",\"correct\":" << (correct ? "true" : "false") << "}" << std::endl;
  check(cudaFree(device_weights), "free P03 weights");
  check(cudaFree(device_input), "free P03 input");
  check(cudaFree(device_output), "free P03 output");
  if (!correct) std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.dry_run) {
    std::cout << "{\"schema_version\":\"p40-bench-v1\",\"primitive\":\"" << options.primitive
              << "\",\"gpu\":" << options.gpu << ",\"cuda_initialized\":false}" << std::endl;
    return 0;
  }
  check(cudaSetDevice(options.gpu), "select GPU");
  if (options.primitive == "P01-copy") print_copy(options); else print_dp4a(options);
  return 0;
}
