#pragma once

#include <cstdint>

struct P40CpuOrderTensor;

extern "C" {
int p40_cpuorder_init(int device);
int p40_cpuorder_upload(P40CpuOrderTensor** tensor, const int8_t* weights,
                         const float* scales, int input, int output);
int p40_cpuorder_matvec(P40CpuOrderTensor* tensor, float* output,
                         const float* input);
void p40_cpuorder_tensor_free(P40CpuOrderTensor* tensor);
void p40_cpuorder_shutdown();
}
