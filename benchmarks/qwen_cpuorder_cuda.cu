// Experimental Q8 GEMV path. This is deliberately not a Colibri backend
// change. It tests whether Pascal can reproduce Qwen's CPU reduction order.
#include "qwen_cpuorder_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

struct P40CpuOrderTensor {
  int input;
  int output;
  int8_t* weights;
  float* scales;
};

namespace {

struct Context {
  int device = -1;
  float* input = nullptr;
  float* output = nullptr;
  size_t input_capacity = 0;
  size_t output_capacity = 0;
} context;

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "[T05E CUDA] %s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool reserve(float** pointer, size_t* capacity, size_t bytes) {
  if (*capacity >= bytes) return true;
  if (*pointer) cudaFree(*pointer);
  *pointer = nullptr;
  *capacity = 0;
  if (!cuda_ok(cudaMalloc(pointer, bytes), "buffer allocation")) return false;
  *capacity = bytes;
  return true;
}

// One warp maps to one output row. Each lane performs the same serial FMA
// stream as one AVX lane in qwen36.c:matmul_q: columns lane + 32*k. Lane zero
// then executes the CPU's fixed 4x8-vector reduction tree in the same order.
__global__ void qwen_cpuorder_q8_matvec(float* output, const float* input,
                                         const int8_t* weights,
                                         const float* scales, int width) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  __shared__ float partial[32];
  const int8_t* weight = weights + static_cast<size_t>(row) * width;
  float accumulator = 0.0f;
  for (int column = lane; column < width; column += 32) {
    accumulator = __fmaf_rn(input[column], static_cast<float>(weight[column]), accumulator);
  }
  partial[lane] = accumulator;
  __syncthreads();
  if (lane == 0) {
    float lanes[8];
    #pragma unroll
    for (int index = 0; index < 8; ++index) {
      const float first = partial[index] + partial[index + 8];
      const float second = partial[index + 16] + partial[index + 24];
      lanes[index] = first + second;
    }
    const float r0 = lanes[0] + lanes[4];
    const float r1 = lanes[1] + lanes[5];
    const float r2 = lanes[2] + lanes[6];
    const float r3 = lanes[3] + lanes[7];
    output[row] = ((r0 + r2) + (r1 + r3)) * scales[row];
  }
}

}  // namespace

extern "C" int p40_cpuorder_init(int device) {
  if (context.device >= 0) return context.device == device;
  if (!cuda_ok(cudaSetDevice(device), "device selection")) return 0;
  context.device = device;
  return 1;
}

extern "C" int p40_cpuorder_upload(P40CpuOrderTensor** tensor,
                                     const int8_t* weights, const float* scales,
                                     int input, int output) {
  if (!tensor || !weights || !scales || input < 32 || output < 1 || (input % 32) != 0 ||
      context.device < 0) return 0;
  if (*tensor) {
    if ((*tensor)->input != input || (*tensor)->output != output) return 0;
    return 1;
  }
  P40CpuOrderTensor* created = new (std::nothrow) P40CpuOrderTensor{input, output, nullptr, nullptr};
  if (!created) return 0;
  const size_t weight_bytes = static_cast<size_t>(input) * output;
  const size_t scale_bytes = static_cast<size_t>(output) * sizeof(float);
  if (!cuda_ok(cudaMalloc(&created->weights, weight_bytes), "weight allocation") ||
      !cuda_ok(cudaMalloc(&created->scales, scale_bytes), "scale allocation") ||
      !cuda_ok(cudaMemcpy(created->weights, weights, weight_bytes, cudaMemcpyHostToDevice), "weight upload") ||
      !cuda_ok(cudaMemcpy(created->scales, scales, scale_bytes, cudaMemcpyHostToDevice), "scale upload")) {
    cudaFree(created->weights);
    cudaFree(created->scales);
    delete created;
    return 0;
  }
  *tensor = created;
  return 1;
}

extern "C" int p40_cpuorder_matvec(P40CpuOrderTensor* tensor, float* output,
                                     const float* input) {
  if (!tensor || !output || !input || context.device < 0) return 0;
  const size_t input_bytes = static_cast<size_t>(tensor->input) * sizeof(float);
  const size_t output_bytes = static_cast<size_t>(tensor->output) * sizeof(float);
  if (!reserve(&context.input, &context.input_capacity, input_bytes) ||
      !reserve(&context.output, &context.output_capacity, output_bytes) ||
      !cuda_ok(cudaMemcpy(context.input, input, input_bytes, cudaMemcpyHostToDevice), "input upload")) return 0;
  qwen_cpuorder_q8_matvec<<<tensor->output, 32>>>(context.output, context.input,
                                                    tensor->weights, tensor->scales,
                                                    tensor->input);
  if (!cuda_ok(cudaGetLastError(), "kernel launch") ||
      !cuda_ok(cudaMemcpy(output, context.output, output_bytes, cudaMemcpyDeviceToHost), "output download")) return 0;
  return 1;
}

extern "C" void p40_cpuorder_tensor_free(P40CpuOrderTensor* tensor) {
  if (!tensor) return;
  cudaFree(tensor->weights);
  cudaFree(tensor->scales);
  delete tensor;
}

extern "C" void p40_cpuorder_shutdown() {
  if (context.device < 0) return;
  cudaSetDevice(context.device);
  cudaFree(context.input);
  cudaFree(context.output);
  context = Context{};
}
