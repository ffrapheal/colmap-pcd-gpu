#include "gpu_ba/custom_cuda.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string JsonString(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (const char character : value) {
    if (character == '"') stream << "\\\"";
    else if (character == '\\') stream << "\\\\";
    else if (character == '\n') stream << "\\n";
    else stream << character;
  }
  stream << '"';
  return stream.str();
}

struct TransferCounts {
  uint64_t variable_images = 0;
  uint64_t propagated_images = 0;
  uint64_t variable_translation_components = 0;
  uint64_t propagated_translation_components = 0;
  uint64_t variable_points = 0;
  uint64_t propagated_points = 0;
};

TransferCounts PropagateState(const Snapshot& previous, Snapshot* next) {
  TransferCounts counts;
  std::map<uint32_t, const ImageSnapshot*> images;
  std::map<uint64_t, const PointSnapshot*> points;
  for (const auto& image : previous.images)
    images.emplace(image.image_id, &image);
  for (const auto& point : previous.points)
    points.emplace(point.point3D_id, &point);
  for (auto& image : next->images) {
    if (image.pose_constant) continue;
    ++counts.variable_images;
    const auto found = images.find(image.image_id);
    if (found == images.end()) continue;
    image.qvec = found->second->qvec;
    ++counts.propagated_images;
    for (size_t component = 0; component < 3; ++component) {
      if ((image.constant_tvec_mask & (1u << component)) != 0) continue;
      ++counts.variable_translation_components;
      image.tvec[component] = found->second->tvec[component];
      ++counts.propagated_translation_components;
    }
  }
  for (auto& point : next->points) {
    if (point.constant) continue;
    ++counts.variable_points;
    const auto found = points.find(point.point3D_id);
    if (found == points.end()) continue;
    point.xyz = found->second->xyz;
    ++counts.propagated_points;
  }
  return counts;
}

CudaFullLmOptions Options(const CudaArithmeticPrecision precision) {
  CudaFullLmOptions options;
  options.arithmetic_precision = precision;
  options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  options.execution_profile = CudaExecutionProfile::kCompactControl;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  options.layer_c.layer_b.hessian_assembly_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  options.layer_c.layer_b.hessian_segment_size_for_testing = 64;
  options.layer_c.schur_contribution_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  options.layer_c.schur_segment_size_for_testing = 64;
  options.layer_c.layer_b.layer_a.residual_order =
      CudaResidualOrder::kSourceInsertion;
  options.layer_c.layer_b.cost_reduction_threads = 128;
  options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
  options.capture_state_trace = false;
  options.max_solver_time_in_seconds = 1e9;
  return options;
}

int Main(const int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " OUTPUT_JSON STATE_ROOT fp64|fp32_core|fp32_state_quantized|fp32_mixed "
                 "SNAPSHOT1 SNAPSHOT2 [SNAPSHOT...]\n";
    return 2;
  }
  const std::string precision_name = argv[3];
  CudaArithmeticPrecision precision = CudaArithmeticPrecision::kFp64;
  if (precision_name == "fp32_core") {
    precision = CudaArithmeticPrecision::kFp32Core;
  } else if (precision_name == "fp32_state_quantized") {
    precision = CudaArithmeticPrecision::kFp32StateQuantizedMixed;
  } else if (precision_name == "fp32_mixed") {
    precision = CudaArithmeticPrecision::kFp32MixedStable;
  } else if (precision_name != "fp64") {
    std::cerr << "invalid precision\n";
    return 2;
  }
  std::ofstream output(argv[1], std::ios::trunc);
  if (!output) return 1;
  output << std::setprecision(17)
         << "{\n  \"schema\":\"phase10p1a_precision_sequence_v1\","
         << "\n  \"precision\":" << JsonString(precision_name)
         << ",\n  \"state_root\":" << JsonString(argv[2])
         << ",\n  \"steps\":[";
  Snapshot previous;
  bool have_previous = false;
  bool all_success = true;
  size_t completed = 0;
  for (int argument = 4; argument < argc; ++argument) {
    Snapshot input;
    SnapshotReadResult read;
    std::string error;
    if (!ReadSnapshot(argv[argument], &input, &read, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    TransferCounts transfer;
    if (have_previous) transfer = PropagateState(previous, &input);
    CudaFullLmResult result;
    const auto start = std::chrono::steady_clock::now();
    const bool success = RunCustomCudaSolve(
        input, Options(precision), &result, &error);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    SnapshotWriteResult write;
    std::string write_error;
    if (success) {
      const std::string state_dir = std::string(argv[2]) + "/step-" +
          std::to_string(argument - 4);
      Snapshot state = result.final_state;
      state.metadata.snapshot_id += "-" + precision_name + "-sequence-step-" +
          std::to_string(argument - 4);
      if (!WriteSnapshot(state, state_dir, &write, &write_error)) {
        std::cerr << write_error << '\n';
        return 1;
      }
      previous = result.final_state;
      have_previous = true;
      ++completed;
    } else {
      all_success = false;
    }
    const auto& runtime = result.runtime.persistent_device;
    output << (argument == 4 ? "\n    " : ",\n    ")
           << "{\"index\":" << (argument - 4)
           << ",\"input\":" << JsonString(argv[argument])
           << ",\"snapshot_id\":" << JsonString(input.metadata.snapshot_id)
           << ",\"ba_kind\":" << JsonString(BaKindName(input.metadata.ba_kind))
           << ",\"success\":" << (success ? "true" : "false")
           << ",\"error\":" << JsonString(error)
           << ",\"wall_milliseconds\":" << wall_ms
           << ",\"initial_cost\":" << result.initial_cost
           << ",\"final_cost\":" << result.final_cost
           << ",\"trial_iterations\":" << result.trial_iterations
           << ",\"accepted_commits\":" << result.accepted_commits
           << ",\"rejected_steps\":" << result.rejected_steps
           << ",\"invalid_steps\":" << result.invalid_steps
           << ",\"factorization_failures\":"
           << result.factorization_failures
           << ",\"termination\":" << JsonString(result.termination_reason)
           << ",\"precision_requested\":"
           << JsonString(runtime.arithmetic_precision_requested)
           << ",\"precision_effective\":"
           << JsonString(runtime.arithmetic_precision_effective)
           << ",\"hessian_effective\":"
           << JsonString(runtime.hessian_assembly_backend_effective)
           << ",\"schur_effective\":"
           << JsonString(runtime.schur_contribution_backend_effective)
           << ",\"spotrf_calls\":" << runtime.spotrf_calls
           << ",\"spotrs_calls\":" << runtime.spotrs_calls
           << ",\"dpotrf_calls\":" << runtime.dpotrf_calls
           << ",\"dpotrs_calls\":" << runtime.dpotrs_calls
           << ",\"fp64_cost_calls\":" << runtime.fp64_cost_calls
           << ",\"state_quantization_calls\":"
           << runtime.state_quantization_calls
           << ",\"precision_mirror_cross_hits\":"
           << runtime.precision_mirror_cross_hits
           << ",\"fixed_external_write_attempts\":"
           << runtime.fixed_external_write_attempts
           << ",\"state_transfer\":{\"variable_images\":"
           << transfer.variable_images << ",\"propagated_images\":"
           << transfer.propagated_images
           << ",\"variable_translation_components\":"
           << transfer.variable_translation_components
           << ",\"propagated_translation_components\":"
           << transfer.propagated_translation_components
           << ",\"variable_points\":" << transfer.variable_points
           << ",\"propagated_points\":" << transfer.propagated_points
           << "},\"final_state_manifest\":"
           << (write.manifest_path.empty() ? "null"
                                           : JsonString(write.manifest_path))
           << '}';
    if (!success) break;
  }
  output << "\n  ],\n  \"completed_steps\":" << completed
         << ",\n  \"all_success\":" << (all_success ? "true" : "false")
         << "\n}\n";
  std::string shutdown_error;
  if (precision == CudaArithmeticPrecision::kFp64 &&
      !ShutdownCudaRuntimePool(&shutdown_error)) {
    std::cerr << shutdown_error << '\n';
    return 1;
  }
  return all_success ? 0 : 1;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  return colmap::gpu_ba::Main(argc, argv);
}
