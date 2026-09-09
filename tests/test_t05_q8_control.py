from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks" / "qwen_dn_q8_control.cu").read_text()
MAKEFILE = (ROOT / "benchmarks" / "Makefile").read_text()


class T05Q8ControlSourceTests(unittest.TestCase):
    def test_dry_run_returns_before_cuda_initialization(self):
        main = SOURCE.index("int main(int argc, char** argv)")
        dry_run = SOURCE.index("if (options.dry_run)", main)
        cuda_init = SOURCE.index("return run(options);", main)
        self.assertLess(dry_run, cuda_init)
        self.assertIn('"cuda_initialized\\":false', SOURCE)

    def test_shape_and_resource_envelope_are_fixed(self):
        self.assertIn('constexpr int kInput = 2048;', SOURCE)
        self.assertIn('constexpr int kOutput = 8192;', SOURCE)
        self.assertIn('constexpr const char* kProfile = "dn_qkv-2048x8192";', SOURCE)
        self.assertIn('options.gpu != 0', SOURCE)
        self.assertIn('options.repetitions > 5', SOURCE)
        self.assertIn('options.calls_per_sample > 8', SOURCE)
        self.assertIn('options.memory_cap_mib > 64', SOURCE)

    def test_cached_gpu_call_and_cpu_numerical_gate_exist(self):
        self.assertIn('coli_cuda_matmul(&tensor, gpu_output.data(), input.data(), nullptr, nullptr,', SOURCE)
        self.assertIn('kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected)', SOURCE)
        self.assertIn('qwen_cpu_q8_matvec', SOURCE)
        self.assertIn('coli_cuda_tensor_free(tensor)', SOURCE)
        self.assertIn('coli_cuda_shutdown()', SOURCE)

    def test_build_links_clean_colibri_backend_with_pascal_target(self):
        self.assertIn('COLIBRI_C_DIR ?= ../colibri-t05a/c', MAKEFILE)
        self.assertIn('qwen_dn_q8_control: qwen_dn_q8_control.cu', MAKEFILE)
        self.assertIn('$(COLIBRI_C_DIR)/backend_cuda.cu', MAKEFILE)
        self.assertIn('CUDA_ARCH=$(CUDA_ARCH)', MAKEFILE)
        self.assertIn('$(CXX) -x c++', MAKEFILE)
        self.assertIn('$< -x none $(COLIBRI_C_DIR)/backend_cuda.o', MAKEFILE)

    def test_triplet_target_keeps_the_host_boundary_and_fixed_shapes(self):
        triplet = (ROOT / "benchmarks" / "qwen_dn_triplet_control.cu").read_text()
        self.assertIn('constexpr int kConv = 8192;', triplet)
        self.assertIn('constexpr int kValue = 4096;', triplet)
        self.assertIn('make_out_input(out_input->data(), qkv->gpu.data(), z->gpu.data())', triplet)
        self.assertIn('qwen_dn_triplet_control: qwen_dn_triplet_control.cu', MAKEFILE)
        self.assertIn('if (options.dry_run)', triplet)

    def test_sweep_target_exceeds_cache_with_the_full_qwen_deltanet_working_set(self):
        sweep = (ROOT / "benchmarks" / "qwen_dn_sweep_control.cu").read_text()
        self.assertIn('constexpr int kLayers = 30;', sweep)
        self.assertIn('dn-sweep-30x-triplet-2048-8192-4096', sweep)
        self.assertIn('for (Layer& layer : *layers)', sweep)
        self.assertIn('make_out_input(boundary->data(), layer.qkv.gpu.data(), layer.z.gpu.data())', sweep)
        self.assertIn('out_kernel_with_gpu_boundary', sweep)
        self.assertIn('boundary_propagation', sweep)
        self.assertIn('options.memory_cap_mib < 1024 || options.memory_cap_mib > 1088', sweep)
        self.assertIn('qwen_dn_sweep_control: qwen_dn_sweep_control.cu', MAKEFILE)
        self.assertIn('#ifdef CPUORDER', sweep)
        self.assertIn('p40_cpuorder_matvec', sweep)
        self.assertIn('qwen_dn_sweep_cpuorder_control:', MAKEFILE)

    def test_cpuorder_out_control_is_standalone_and_uses_the_fixed_reduction_tree(self):
        kernel = (ROOT / "benchmarks" / "qwen_cpuorder_cuda.cu").read_text()
        control = (ROOT / "benchmarks" / "qwen_dn_out_cpuorder_control.cpp").read_text()
        self.assertIn('__fmaf_rn', kernel)
        self.assertIn('__shfl_sync', kernel)
        self.assertNotIn('__syncthreads()', kernel)
        self.assertIn('qwen_cpuorder_q8_matvec', kernel)
        self.assertIn('constexpr int kInput = 4096;', control)
        self.assertIn('constexpr int kOutput = 2048;', control)
        self.assertIn('p40_cpuorder_matvec', control)
        self.assertIn('qwen_dn_out_cpuorder_control:', MAKEFILE)

    def test_lmhead_control_matches_the_real_q8_shape_and_requires_bit_identity(self):
        control = (ROOT / "benchmarks" / "qwen_lmhead_cpuorder_control.cpp").read_text()
        self.assertIn('constexpr int kInput = 2048;', control)
        self.assertIn('constexpr int kOutput = 248044;', control)
        self.assertIn('lmhead-cpuorder-2048x248044', control)
        self.assertIn('std::memcmp(expected.data(), actual.data()', control)
        self.assertIn('p40_cpuorder_matvec', control)
        self.assertIn('options.memory_cap_mib < 512 || options.memory_cap_mib > 544', control)
        self.assertIn('qwen_lmhead_cpuorder_control:', MAKEFILE)

    def test_attention_projection_control_covers_all_ten_full_attention_layers(self):
        control = (ROOT / "benchmarks" / "qwen_attention_cpuorder_control.cpp").read_text()
        self.assertIn('constexpr int kLayers = 10;', control)
        self.assertIn('constexpr int kQueryOutput = 8192;', control)
        self.assertIn('constexpr int kKvOutput = 512;', control)
        self.assertIn('constexpr int kOutputInput = 4096;', control)
        self.assertIn('constexpr int kOutputOutput = 2048;', control)
        self.assertIn('attention-projections-cpuorder-10x-2048-8192-512-4096-2048', control)
        self.assertIn('matrices.reserve(kLayers * kMatricesPerLayer)', control)
        self.assertIn('p40_cpuorder_upload', control)
        self.assertIn('std::memcmp(matrix.reference.data(), matrix.gpu.data()', control)
        self.assertIn('std::cout << "{\\"schema_version\\":\\""', control)
        self.assertNotIn('std::cout << "{\\\\\\"schema_version', control)
        self.assertIn('options.memory_cap_mib < 288 || options.memory_cap_mib > 320', control)
        self.assertIn('qwen_attention_cpuorder_control:', MAKEFILE)

    def test_attention_integration_patch_is_opt_in_and_count_gated(self):
        patch = (ROOT / "patches" / "0002-qwen36-attention-cpuorder.patch").read_text()
        self.assertIn('#define QDW_ATTN_CPUORDER 4u', patch)
        self.assertIn('COLI_CUDA_ATTN_CPUORDER', patch)
        self.assertIn('attn_count != (attn_cpuorder_on() ? 40 : 0)', patch)
        self.assertIn('qdw_register_with_flags(l->q', patch)
        self.assertIn('qdw_register_with_flags(l->k', patch)
        self.assertIn('qdw_register_with_flags(l->v', patch)
        self.assertIn('qdw_register_with_flags(l->o', patch)
        self.assertNotIn('coli_cuda_attention_', patch)

    def test_shared_expert_control_has_the_real_40_layer_q8_mlp_sequence(self):
        control = (ROOT / "benchmarks" / "qwen_shared_expert_cpuorder_control.cpp").read_text()
        self.assertIn('constexpr int kLayers = 40;', control)
        self.assertIn('constexpr int kHidden = 2048;', control)
        self.assertIn('constexpr int kIntermediate = 512;', control)
        self.assertIn('shared-expert-cpuorder-40x-2048-512-2048', control)
        self.assertIn('layers.reserve(kLayers)', control)
        self.assertIn('qwen_cpu_q8_matvec(gate, layer.input.data(), layer.gate)', control)
        self.assertIn('qwen_cpu_q8_matvec(up, layer.input.data(), layer.up)', control)
        self.assertIn('activate(activation, gate, up)', control)
        self.assertIn('qwen_cpu_q8_matvec(down, activation, layer.down)', control)
        self.assertIn('p40_cpuorder_matvec(layer.down.tensor', control)
        self.assertIn('options.memory_cap_mib < 144 || options.memory_cap_mib > 176', control)
        self.assertIn('qwen_shared_expert_cpuorder_control:', MAKEFILE)

    def test_shared_expert_integration_patch_is_opt_in_and_count_gated(self):
        patch = (ROOT / "patches" / "0003-qwen36-shared-expert-cpuorder.patch").read_text()
        self.assertIn('#define QDW_SHARED_CPUORDER 8u', patch)
        self.assertIn('COLI_CUDA_SHARED_CPUORDER', patch)
        self.assertIn('shared_count != (shared_cpuorder_on() ? 120 : 0)', patch)
        self.assertIn('qdw_register_with_flags(l->sh_g', patch)
        self.assertIn('qdw_register_with_flags(l->sh_u', patch)
        self.assertIn('qdw_register_with_flags(l->sh_d', patch)
        self.assertIn('120 shared MLP matrices', patch)
        self.assertNotIn('coli_cuda_expert_', patch)

    def test_w4a8_dp4a_control_keeps_packed_w4_and_the_real_expert_projection_shape(self):
        control = (ROOT / "benchmarks" / "qwen_w4a8_dp4a_control.cu").read_text()
        self.assertIn('constexpr int kExperts = 4;', control)
        self.assertIn('constexpr int kInput = 2048;', control)
        self.assertIn('constexpr int kOutput = 512;', control)
        self.assertIn('constexpr int kTile = 8;', control)
        self.assertIn('__dp4a', control)
        self.assertIn('quantize_rows_i8', control)
        self.assertIn('w4a32_rows', control)
        self.assertIn('gpu_integer != cpu_integer', control)
        self.assertIn('relative_l2 > 0.02', control)
        self.assertIn('options.calls_per_sample != 64', control)
        self.assertIn('"cuda_initialized\\":false', control)
        self.assertIn('qwen_w4a8_dp4a_control:', MAKEFILE)


if __name__ == "__main__":
    unittest.main()
