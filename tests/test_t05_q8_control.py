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


if __name__ == "__main__":
    unittest.main()
