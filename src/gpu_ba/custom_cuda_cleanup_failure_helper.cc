#include "gpu_ba/custom_cuda.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace {

const std::map<std::string, colmap::gpu_ba::CudaTeardownType> kTypes = {
    {"get_device", colmap::gpu_ba::CudaTeardownType::kGetDevice},
    {"set_device", colmap::gpu_ba::CudaTeardownType::kSetDevice},
    {"solver", colmap::gpu_ba::CudaTeardownType::kSolver},
    {"blas", colmap::gpu_ba::CudaTeardownType::kBlas},
    {"event", colmap::gpu_ba::CudaTeardownType::kEvent},
    {"stream", colmap::gpu_ba::CudaTeardownType::kStream},
    {"allocation", colmap::gpu_ba::CudaTeardownType::kAllocation},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0]
              << " get_device|set_device|solver|blas|event|stream|allocation\n";
    return EXIT_FAILURE;
  }
  const auto it = kTypes.find(argv[1]);
  if (it == kTypes.end()) {
    std::cerr << "unknown teardown type: " << argv[1] << '\n';
    return EXIT_FAILURE;
  }
  colmap::gpu_ba::CudaFullLmResult result;
  std::string error;
  const bool passed = colmap::gpu_ba::RunCudaCleanupFailureSelfTest(
      it->second, &result, &error);
  std::cout << "{\"teardown_type\":\"" << argv[1]
            << "\",\"self_test_pass\":" << (passed ? "true" : "false")
            << ",\"error_classification\":"
            << static_cast<int>(result.error_classification)
            << ",\"final_resource_health\":"
            << static_cast<int>(result.runtime.final_resource_health)
            << ",\"cleanup_complete\":"
            << (result.runtime.resource_cleanup_complete ? "true" : "false")
            << ",\"quarantine_count\":"
            << result.runtime.resource_quarantine_count
            << ",\"next_solve_advance_events\":"
            << result.runtime.resource_generation_advance_events
            << ",\"error\":\"" << error << "\"}\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
