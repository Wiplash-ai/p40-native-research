NVCC ?= /usr/bin/nvcc
CUDA_ARCH ?= sm_61
CXXFLAGS := -std=c++17 -O3 -lineinfo -Xcompiler=-Wall,-Wextra,-Werror
COLIBRI_C_DIR ?= ../colibri-t05a/c
CXX ?= g++
CUDA_LIBDIR ?= /usr/lib64

.PHONY: all clean

all: p40_bench qwen_dn_q8_control qwen_dn_triplet_control qwen_dn_sweep_control qwen_dn_out_cpuorder_control qwen_lmhead_cpuorder_control qwen_attention_cpuorder_control

p40_bench: p40_bench.cu
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) $< -o $@

# T05A is intentionally linked against a clean Colibri worktree.  The generic
# CUDA backend owns the resident Q8 weights; this fixture never parses a model.
qwen_dn_q8_control: qwen_dn_q8_control.cu
	test -f $(COLIBRI_C_DIR)/backend_cuda.cu
	$(MAKE) -C $(COLIBRI_C_DIR) backend_cuda.o CUDA=1 CUDA_ARCH=$(CUDA_ARCH) NVCC=$(NVCC)
	$(CXX) -x c++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		-I$(COLIBRI_C_DIR) $< -x none $(COLIBRI_C_DIR)/backend_cuda.o -o $@ \
		-L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -lcublasLt -lcublas -ldl -lpthread -lm

qwen_dn_triplet_control: qwen_dn_triplet_control.cu
	test -f $(COLIBRI_C_DIR)/backend_cuda.cu
	$(MAKE) -C $(COLIBRI_C_DIR) backend_cuda.o CUDA=1 CUDA_ARCH=$(CUDA_ARCH) NVCC=$(NVCC)
	$(CXX) -x c++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		-I$(COLIBRI_C_DIR) $< -x none $(COLIBRI_C_DIR)/backend_cuda.o -o $@ \
		-L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -lcublasLt -lcublas -ldl -lpthread -lm

qwen_dn_sweep_control: qwen_dn_sweep_control.cu
	test -f $(COLIBRI_C_DIR)/backend_cuda.cu
	$(MAKE) -C $(COLIBRI_C_DIR) backend_cuda.o CUDA=1 CUDA_ARCH=$(CUDA_ARCH) NVCC=$(NVCC)
	$(CXX) -x c++ -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		-I$(COLIBRI_C_DIR) $< -x none $(COLIBRI_C_DIR)/backend_cuda.o -o $@ \
		-L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -lcublasLt -lcublas -ldl -lpthread -lm

qwen_cpuorder_cuda.o: qwen_cpuorder_cuda.cu qwen_cpuorder_cuda.h
	$(NVCC) -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra,-Werror -c $< -o $@

qwen_dn_out_cpuorder_control: qwen_dn_out_cpuorder_control.cpp qwen_cpuorder_cuda.h qwen_cpuorder_cuda.o
	$(CXX) -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		$< qwen_cpuorder_cuda.o -o $@ -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -ldl -lpthread -lm

qwen_lmhead_cpuorder_control: qwen_lmhead_cpuorder_control.cpp qwen_cpuorder_cuda.h qwen_cpuorder_cuda.o
	$(CXX) -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		$< qwen_cpuorder_cuda.o -o $@ -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -ldl -lpthread -lm

qwen_attention_cpuorder_control: qwen_attention_cpuorder_control.cpp qwen_cpuorder_cuda.h qwen_cpuorder_cuda.o
	$(CXX) -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		$< qwen_cpuorder_cuda.o -o $@ -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -ldl -lpthread -lm

qwen_dn_sweep_cpuorder_control: qwen_dn_sweep_control.cu qwen_cpuorder_cuda.h qwen_cpuorder_cuda.o
	$(CXX) -x c++ -DCPUORDER -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Werror \
		$< -x none qwen_cpuorder_cuda.o -o $@ -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -ldl -lpthread -lm

clean:
	rm -f p40_bench qwen_dn_q8_control qwen_dn_triplet_control qwen_dn_sweep_control qwen_cpuorder_cuda.o qwen_dn_out_cpuorder_control qwen_lmhead_cpuorder_control qwen_attention_cpuorder_control qwen_dn_sweep_cpuorder_control
