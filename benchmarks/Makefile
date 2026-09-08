NVCC ?= /usr/bin/nvcc
CUDA_ARCH ?= sm_61
CXXFLAGS := -std=c++17 -O3 -lineinfo -Xcompiler=-Wall,-Wextra,-Werror

.PHONY: all clean

all: p40_bench

p40_bench: p40_bench.cu
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) $< -o $@

clean:
	rm -f p40_bench
