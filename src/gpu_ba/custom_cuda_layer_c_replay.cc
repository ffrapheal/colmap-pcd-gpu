#include "gpu_ba/custom_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_ba/fixed_linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

struct Metric {
  std::string name;
  double atol = 1e-10;
  double rtol = 1e-8;
  uint64_t count = 0;
  uint64_t failures = 0;
  uint64_t nonfinite = 0;
  double maximum = 0.0;
  double squared_sum = 0.0;
  double rms = 0.0;
  double p95 = 0.0;
  std::vector<double> errors;
  std::string first_id;
  double first_reference = 0.0;
  double first_candidate = 0.0;
  bool scale_floor_one = false;
  bool bitwise = false;

  void Add(double reference, double candidate, const std::string& id) {
    ++count;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      ++failures;
      ++nonfinite;
      if (first_id.empty()) {
        first_id = id;
        first_reference = reference;
        first_candidate = candidate;
      }
      return;
    }
    if (bitwise &&
        std::memcmp(&reference, &candidate, sizeof(double)) != 0) {
      ++failures;
      if (first_id.empty()) {
        first_id = id;
        first_reference = reference;
        first_candidate = candidate;
      }
    }
    const double error = std::abs(reference - candidate);
    const double scale = scale_floor_one
        ? std::max(1.0, std::abs(reference))
        : std::max(std::abs(reference), std::abs(candidate));
    const double limit = atol + rtol * scale;
    maximum = std::max(maximum, error);
    squared_sum += error * error;
    errors.push_back(error);
    if (!bitwise && error > limit) {
      ++failures;
      if (first_id.empty()) {
        first_id = id;
        first_reference = reference;
        first_candidate = candidate;
      }
    }
  }

  void Finish() {
    if (errors.empty()) return;
    rms = std::sqrt(squared_sum / errors.size());
    std::sort(errors.begin(), errors.end());
    const size_t index = std::min(
        errors.size() - 1,
        static_cast<size_t>(std::ceil(0.95 * errors.size()) - 1));
    p95 = errors[index];
  }
};

std::string JsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (char character : value) {
    if (character == '"') output << "\\\"";
    else if (character == '\\') output << "\\\\";
    else if (character == '\n') output << "\\n";
    else output << character;
  }
  output << '"';
  return output.str();
}

std::string NumberOrNull(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

void WriteMetric(std::ostream& output, const Metric& metric) {
  output << "{\"name\":" << JsonString(metric.name)
         << ",\"atol\":" << metric.atol
         << ",\"rtol\":" << metric.rtol
         << ",\"scale_floor_one\":"
         << (metric.scale_floor_one ? "true" : "false")
         << ",\"bitwise\":" << (metric.bitwise ? "true" : "false")
         << ",\"count\":" << metric.count
         << ",\"failures\":" << metric.failures
         << ",\"nonfinite\":" << metric.nonfinite
         << ",\"max_absolute\":" << NumberOrNull(metric.maximum)
         << ",\"rms_absolute\":" << NumberOrNull(metric.rms)
         << ",\"p95_absolute\":" << NumberOrNull(metric.p95)
         << ",\"first_id\":"
         << (metric.first_id.empty() ? "null" : JsonString(metric.first_id))
         << ",\"first_reference\":"
         << (metric.first_id.empty() ? "null"
                                     : NumberOrNull(metric.first_reference))
         << ",\"first_candidate\":"
         << (metric.first_id.empty() ? "null"
                                     : NumberOrNull(metric.first_candidate))
         << '}';
}

bool CompareStates(const Snapshot& reference,
                   const Snapshot& candidate,
                   std::vector<Metric>* metrics,
                   std::string* topology_error) {
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  for (const ImageSnapshot& value : candidate.images)
    images.emplace(value.image_id, &value);
  for (const PointSnapshot& value : candidate.points)
    points.emplace(value.point3D_id, &value);
  for (const ImageSnapshot& value : reference.images) {
    const auto it = images.find(value.image_id);
    if (it == images.end()) {
      *topology_error = "trial state is missing image=" +
                        std::to_string(value.image_id);
      return false;
    }
    for (size_t i = 0; i < 4; ++i)
      (*metrics)[6].Add(value.qvec[i], it->second->qvec[i],
                        "image=" + std::to_string(value.image_id) +
                            ":q[" + std::to_string(i) + "]");
    for (size_t i = 0; i < 3; ++i)
      (*metrics)[7].Add(value.tvec[i], it->second->tvec[i],
                        "image=" + std::to_string(value.image_id) +
                            ":t[" + std::to_string(i) + "]");
  }
  for (const PointSnapshot& value : reference.points) {
    const auto it = points.find(value.point3D_id);
    if (it == points.end()) {
      *topology_error = "trial state is missing point3D=" +
                        std::to_string(value.point3D_id);
      return false;
    }
    for (size_t i = 0; i < 3; ++i)
      (*metrics)[8].Add(value.xyz[i], it->second->xyz[i],
                        "point3D=" + std::to_string(value.point3D_id) +
                            "[" + std::to_string(i) + "]");
  }
  return true;
}

bool Compare(const CustomCpuCanonicalStepExport& reference,
             const CudaLayerCResult& candidate,
             std::vector<Metric>* metrics,
             std::string* topology_error) {
  if (reference.pose_dimension != candidate.pose_dimension ||
      reference.point_dimension != candidate.point_dimension ||
      reference.schur.size() != candidate.schur.size() ||
      reference.rhs.size() != candidate.rhs.size() ||
      reference.camera_delta.size() != candidate.camera_delta.size() ||
      reference.point_delta.size() != candidate.point_delta.size()) {
    *topology_error = "Layer C vector/matrix dimensions differ";
    return false;
  }
  const size_t dimension = reference.pose_dimension;
  for (size_t row = 0; row < dimension; ++row) {
    for (size_t col = 0; col < dimension; ++col) {
      const size_t index = row * dimension + col;
      (*metrics)[0].Add(reference.schur[index], candidate.schur[index],
                        "S[" + std::to_string(row) + "," +
                            std::to_string(col) + "]");
    }
    (*metrics)[1].Add(reference.rhs[row], candidate.rhs[row],
                      "rhs[" + std::to_string(row) + "]");
    (*metrics)[2].Add(reference.camera_delta[row],
                      candidate.camera_delta[row],
                      "camera_delta[" + std::to_string(row) + "]");
  }
  for (size_t i = 0; i < reference.point_delta.size(); ++i) {
    const size_t point = i / 3;
    const size_t component = i % 3;
    const uint64_t point_id = reference.linearization.points[point].point3D_id;
    (*metrics)[3].Add(reference.point_delta[i], candidate.point_delta[i],
                      "point3D=" + std::to_string(point_id) + "[" +
                          std::to_string(component) + "]");
  }
  (*metrics)[4].Add(reference.predicted_reduction,
                    candidate.predicted_reduction, "predicted_reduction");
  (*metrics)[5].Add(reference.trial_cost, candidate.trial_cost, "trial_cost");
  return CompareStates(reference.trial_state, candidate.trial_state, metrics,
                       topology_error);
}

bool CompareCuda(const CudaLayerCResult& reference,
                 const CudaLayerCResult& candidate,
                 std::vector<Metric>* metrics,
                 std::string* topology_error) {
  if (reference.pose_dimension != candidate.pose_dimension ||
      reference.point_dimension != candidate.point_dimension ||
      reference.schur.size() != candidate.schur.size() ||
      reference.rhs.size() != candidate.rhs.size() ||
      reference.camera_delta.size() != candidate.camera_delta.size() ||
      reference.point_delta.size() != candidate.point_delta.size()) {
    *topology_error = "CUDA reference/candidate Layer C dimensions differ";
    return false;
  }
  const size_t dimension = reference.pose_dimension;
  for (size_t row = 0; row < dimension; ++row) {
    for (size_t col = 0; col < dimension; ++col) {
      const size_t index = row * dimension + col;
      (*metrics)[0].Add(reference.schur[index], candidate.schur[index],
                        "S[" + std::to_string(row) + "," +
                            std::to_string(col) + "]");
    }
    (*metrics)[1].Add(reference.rhs[row], candidate.rhs[row],
                      "rhs[" + std::to_string(row) + "]");
    (*metrics)[2].Add(reference.camera_delta[row],
                      candidate.camera_delta[row],
                      "camera_delta[" + std::to_string(row) + "]");
  }
  for (size_t i = 0; i < reference.point_delta.size(); ++i) {
    (*metrics)[3].Add(reference.point_delta[i], candidate.point_delta[i],
                      "point_delta[" + std::to_string(i) + "]");
  }
  (*metrics)[4].Add(reference.predicted_reduction,
                    candidate.predicted_reduction, "predicted_reduction");
  (*metrics)[5].Add(reference.trial_cost, candidate.trial_cost, "trial_cost");
  return CompareStates(reference.trial_state, candidate.trial_state, metrics,
                       topology_error);
}

void WriteFailure(const std::string& path,
                  const Snapshot& snapshot,
                  const CudaLayerCResult& result,
                  const std::string& error) {
  std::ofstream output(path);
  output << "{\"phase\":\"phase6p3-custom-cuda-audit-v1\","
         << "\"layer\":\"C_schur_factor_backsub\","
         << "\"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
         << ",\"cuda_ran\":false,\"pass\":false,\"error\":" << JsonString(error)
         << ",\"cusolver_potrf_status\":"
         << result.runtime.cusolver_potrf_status
         << ",\"cusolver_potrs_status\":"
         << result.runtime.cusolver_potrs_status
         << ",\"cusolver_dev_info\":"
         << result.runtime.cusolver_dev_info
         << ",\"point_factorization_failures\":"
         << result.runtime.point_factorization_failures << "}\n";
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap::gpu_ba;
  if (argc < 3 || argc > 7) {
    std::cerr << "usage: gpu_ba_custom_cuda_layer_c_replay "
                 "SNAPSHOT REPORT [explicit|unified] "
                 "[canonical|source] [REDUCTION_THREADS] [parallel]\n"
                 "   or: gpu_ba_custom_cuda_layer_c_replay "
                 "SNAPSHOT REPORT component SEGMENT_SIZE PAIRED_SAMPLES\n";
    return 2;
  }
  const bool source_order = argc >= 5 && std::string(argv[4]) == "source";
  const int reduction_threads = argc >= 6 ? std::max(1, std::stoi(argv[5])) : 1;
  bool parallel = false;
  for (int i = 3; i < argc; ++i)
    parallel = parallel || std::string(argv[i]) == "parallel";
  Snapshot snapshot;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (argc >= 4 && std::string(argv[3]) == "component") {
    const uint32_t segment_size = argc >= 5
        ? static_cast<uint32_t>(std::stoul(argv[4])) : 32;
    const size_t samples = argc >= 6
        ? static_cast<size_t>(std::stoull(argv[5])) : 5;
    CudaSchurContributionComponentResult result;
    if (!RunCudaSchurContributionComponentForTesting(
            snapshot, segment_size, samples, &result, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    const bool numeric_pass =
        result.element_contract_violations == 0 &&
        result.normalized_max_error <= 5e-11 &&
        result.normalized_frobenius_error <= 5e-11 &&
        result.repeated_candidate_max_abs_error == 0.0 &&
        result.coverage_violations == 0;
    std::ofstream output(argv[2]);
    if (!output) return 1;
    const auto write_samples = [&](const std::vector<double>& values) {
      output << '[';
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ',';
        output << NumberOrNull(values[i]);
      }
      output << ']';
    };
    output << std::setprecision(17)
           << "{\n  \"phase\":\"phase10p0a-fixed-workload-component\",\n"
           << "  \"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
           << ",\n  \"pass\":" << (numeric_pass ? "true" : "false")
           << ",\n  \"segment_size\":" << result.segment_size
           << ",\n  \"pose_dimension\":" << result.pose_dimension
           << ",\n  \"pair_count\":" << result.pair_count
           << ",\n  \"contribution_count\":" << result.contribution_count
           << ",\n  \"direct_pair_count\":" << result.direct_pair_count
           << ",\n  \"segmented_pair_count\":" << result.segmented_pair_count
           << ",\n  \"segment_count\":" << result.segment_count
           << ",\n  \"max_segments_per_pair\":" << result.max_segments_per_pair
           << ",\n  \"partial_workspace_bytes\":"
           << result.partial_workspace_bytes
           << ",\n  \"coverage_violations\":" << result.coverage_violations
           << ",\n  \"element_contract_violations\":"
           << result.element_contract_violations
           << ",\n  \"max_abs_error\":" << NumberOrNull(result.max_abs_error)
           << ",\n  \"normalized_max_error\":"
           << NumberOrNull(result.normalized_max_error)
           << ",\n  \"normalized_frobenius_error\":"
           << NumberOrNull(result.normalized_frobenius_error)
           << ",\n  \"repeated_candidate_max_abs_error\":"
           << NumberOrNull(result.repeated_candidate_max_abs_error)
           << ",\n  \"direct_complete_milliseconds\":";
    write_samples(result.direct_complete_milliseconds);
    output << ",\n  \"segmented_complete_milliseconds\":";
    write_samples(result.segmented_complete_milliseconds);
    output << ",\n  \"segmented_short_direct_milliseconds\":";
    write_samples(result.segmented_short_direct_milliseconds);
    output << ",\n  \"segmented_partial_milliseconds\":";
    write_samples(result.segmented_partial_milliseconds);
    output << ",\n  \"segmented_merge_milliseconds\":";
    write_samples(result.segmented_merge_milliseconds);
    output << "\n}\n";
    return numeric_pass ? 0 : 1;
  }
  CustomCpuCanonicalStepExport reference;
  const bool cpu_exported = source_order
      ? ExportCustomCpuSourceStep(snapshot, reduction_threads, 1e-4, 1e-6,
                                  1e32, &reference, &error)
      : ExportCustomCpuCanonicalStep(snapshot, 1e-4, 1e-6, 1e32,
                                     &reference, &error);
  if (!cpu_exported) {
    std::cerr << "CPU step export failed: " << error << '\n';
    return 1;
  }
  CudaLayerCOptions options;
  if (argc >= 4 && std::string(argv[3]) == "unified")
    options.layer_b.layer_a.memory_mode = CudaMemoryMode::kUnifiedManaged;
  if (source_order) {
    options.layer_b.layer_a.residual_order =
        CudaResidualOrder::kSourceInsertion;
  }
  options.layer_b.cost_reduction_threads = reduction_threads;
  if (parallel) {
    options.layer_b.reduction_mode =
        CudaReductionMode::kParallelDeterministic;
    options.layer_b.cost_reduction_threads = std::max(128, reduction_threads);
  }
  const char* selector_value =
      std::getenv("COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION");
  const std::string selector = selector_value == nullptr
      ? "reference"
      : std::string(selector_value);
  const bool transformed = selector == "transformed";
  CudaLayerCResult cuda_reference;
  if (transformed) {
    setenv("COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "reference", 1);
    if (!RunCudaSnapshotLayerC(snapshot, options, &cuda_reference, &error)) {
      setenv("COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed", 1);
      WriteFailure(argv[2], snapshot, cuda_reference, error);
      std::cerr << "CUDA reference Layer C failed: " << error << '\n';
      return 1;
    }
    setenv("COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed", 1);
  }
  CudaLayerCResult candidate;
  if (!RunCudaSnapshotLayerC(snapshot, options, &candidate, &error)) {
    WriteFailure(argv[2], snapshot, candidate, error);
    std::cerr << "CUDA Layer C failed: " << error << '\n';
    return 1;
  }
  const char* names[] = {"schur_S", "rhs", "camera_delta", "point_delta",
                         "predicted_reduction", "trial_cost",
                         "trial_quaternion", "trial_translation",
                         "trial_point"};
  std::vector<Metric> metrics;
  for (const char* name : names) {
    Metric metric;
    metric.name = name;
    metrics.push_back(metric);
  }
  metrics[4].atol = 1e-9;
  metrics[5].atol = 1e-9;
  if (transformed) {
    for (Metric& metric : metrics) metric.scale_floor_one = true;
    metrics[0].atol = 1e-10;
    metrics[0].rtol = 5e-11;
    metrics[1].atol = 0.0;
    metrics[1].rtol = 0.0;
    metrics[1].bitwise = true;
    metrics[2].atol = 1e-10;
    metrics[2].rtol = 1e-8;
    metrics[3].atol = 1e-10;
    metrics[3].rtol = 1e-8;
    metrics[4].atol = 1e-10;
    metrics[4].rtol = 1e-8;
    metrics[5].atol = 1e-8;
    metrics[5].rtol = 1e-10;
  }
  long double schur_difference_squared = 0.0L;
  long double schur_reference_squared = 0.0L;
  const std::vector<double>& schur_reference =
      transformed ? cuda_reference.schur : reference.schur;
  if (schur_reference.size() == candidate.schur.size()) {
    for (size_t i = 0; i < schur_reference.size(); ++i) {
      const long double difference = static_cast<long double>(
          candidate.schur[i] - schur_reference[i]);
      schur_difference_squared += difference * difference;
      const long double value = schur_reference[i];
      schur_reference_squared += value * value;
    }
  }
  const double schur_relative_frobenius = static_cast<double>(
      std::sqrt(schur_difference_squared) /
      std::max<long double>(1.0L, std::sqrt(schur_reference_squared)));
  std::string topology_error;
  const bool topology_pass = transformed
      ? CompareCuda(cuda_reference, candidate, &metrics, &topology_error)
      : Compare(reference, candidate, &metrics, &topology_error);
  const double reference_backward_error = transformed
      ? cuda_reference.backward_error
      : reference.backward_error;
  bool pass = topology_pass && candidate.backward_error <= 1e-8 &&
              candidate.symmetry_error <= 1e-10 &&
              candidate.runtime.point_factorization_failures == 0 &&
              candidate.runtime.cusolver_create_status == 0 &&
              candidate.runtime.cusolver_buffer_size_status == 0 &&
              candidate.runtime.cusolver_potrf_status == 0 &&
              candidate.runtime.cusolver_potrs_status == 0 &&
              candidate.runtime.cusolver_dev_info == 0 &&
              candidate.runtime.cublas_create_status == 0 &&
              candidate.runtime.cublas_copy_status == 0;
  if (transformed) {
    pass = pass && schur_relative_frobenius <= 5e-11 &&
           std::isfinite(reference_backward_error) &&
           std::isfinite(candidate.backward_error) &&
           reference_backward_error >= 0.0 &&
           candidate.backward_error >= 0.0 &&
           candidate.backward_error <=
               std::max(1e-12, 10.0 * reference_backward_error);
  }
  for (Metric& metric : metrics) {
    metric.Finish();
    pass = pass && metric.failures == 0;
  }

  std::ofstream output(argv[2]);
  if (!output) return 1;
  output << std::setprecision(17)
         << "{\n  \"phase\":\"phase6p3-custom-cuda-audit-v1\",\n"
         << "  \"layer\":\"C_schur_factor_backsub\",\n"
         << "  \"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
         << ",\n  \"cuda_ran\":true"
         << ",\n  \"residual_parameter_order\":"
         << JsonString(source_order ? "source" : "canonical")
         << ",\n  \"hot_kernel_implementation\":"
         << JsonString(selector)
         << ",\n  \"component_reference\":"
         << JsonString(transformed ? "same_binary_cuda_reference"
                                   : "custom_cpu_canonical")
         << ",\n  \"schur_relative_frobenius\":"
         << NumberOrNull(schur_relative_frobenius)
         << ",\n  \"cost_reduction_threads\":" << reduction_threads
         << ",\n  \"pass\":" << (pass ? "true" : "false")
         << ",\n  \"topology_pass\":"
         << (topology_pass ? "true" : "false")
         << ",\n  \"topology_error\":"
         << (topology_error.empty() ? "null" : JsonString(topology_error))
         << ",\n  \"pose_dimension\":" << candidate.pose_dimension
         << ",\n  \"point_dimension\":" << candidate.point_dimension
         << ",\n  \"symmetry_error\":"
         << NumberOrNull(candidate.symmetry_error)
         << ",\n  \"reference_backward_error\":"
         << NumberOrNull(reference_backward_error)
         << ",\n  \"cuda_backward_error\":"
         << NumberOrNull(candidate.backward_error) << ",\n"
         << "  \"runtime\":{\"predicted_peak_bytes\":"
         << candidate.runtime.predicted_peak_bytes
         << ",\"active_bytes\":" << candidate.runtime.active_bytes
         << ",\"cached_bytes\":" << candidate.runtime.cached_bytes
         << ",\"resident_bytes\":" << candidate.runtime.resident_bytes
         << ",\"predicted_additional_bytes\":"
         << candidate.runtime.predicted_additional_bytes
         << ",\"reduction_mode\":"
         << JsonString(candidate.layer_b.runtime.reduction_mode ==
                               CudaReductionMode::kParallelDeterministic
                           ? "parallel_deterministic"
                           : "serial_deterministic")
         << ",\"configured_worker_count\":"
         << candidate.layer_b.runtime.configured_worker_count
         << ",\"effective_worker_count\":"
         << candidate.layer_b.runtime.effective_worker_count
         << ",\"cost_reduction_parallel\":"
         << (candidate.layer_b.runtime.cost_reduction_parallel ? "true"
                                                               : "false")
         << ",\"gradient_reduction_parallel\":"
         << (candidate.layer_b.runtime.gradient_reduction_parallel ? "true"
                                                                   : "false")
         << ",\"free_bytes_before\":"
         << candidate.runtime.free_bytes_before
         << ",\"total_bytes\":" << candidate.runtime.total_bytes
         << ",\"pair_adjacency_bytes\":"
         << candidate.runtime.pair_adjacency_bytes
         << ",\"layer_c_pair_chunk_launch_calls\":"
         << candidate.runtime.layer_c_pair_chunk_launch_calls
         << ",\"layer_c_max_chunks_per_step\":"
         << candidate.runtime.layer_c_max_chunks_per_step
         << ",\"layer_c_max_chunk_bytes\":"
         << candidate.runtime.layer_c_max_chunk_bytes
         << ",\"workspace_bytes\":"
         << candidate.runtime.cusolver_workspace_bytes
         << ",\"cusolver_create\":"
         << candidate.runtime.cusolver_create_status
         << ",\"cusolver_set_stream\":"
         << candidate.runtime.cusolver_set_stream_status
         << ",\"cusolver_buffer_size\":"
         << candidate.runtime.cusolver_buffer_size_status
         << ",\"cusolver_potrf\":"
         << candidate.runtime.cusolver_potrf_status
         << ",\"cusolver_potrs\":"
         << candidate.runtime.cusolver_potrs_status
         << ",\"cusolver_dev_info\":"
         << candidate.runtime.cusolver_dev_info
         << ",\"cublas_create\":"
         << candidate.runtime.cublas_create_status
         << ",\"cublas_set_stream\":"
         << candidate.runtime.cublas_set_stream_status
         << ",\"cublas_copy\":"
         << candidate.runtime.cublas_copy_status
         << ",\"point_factor_ms\":"
         << NumberOrNull(candidate.runtime.point_kernel_milliseconds)
         << ",\"schur_ms\":"
         << NumberOrNull(candidate.runtime.schur_kernel_milliseconds)
         << ",\"factorization_ms\":"
         << NumberOrNull(candidate.runtime.factorization_milliseconds)
         << ",\"back_substitution_ms\":"
         << NumberOrNull(candidate.runtime.back_substitution_milliseconds)
         << ",\"trial_cost_ms\":"
         << NumberOrNull(candidate.runtime.trial_cost_kernel_milliseconds)
         << ",\"allocation_ms\":"
         << NumberOrNull(candidate.runtime.allocation_milliseconds)
         << ",\"copy_ms\":"
         << NumberOrNull(candidate.runtime.copy_milliseconds)
         << ",\"synchronization_ms\":"
         << NumberOrNull(candidate.runtime.synchronization_milliseconds)
         << ",\"host_wall_ms\":"
         << NumberOrNull(candidate.runtime.host_wall_milliseconds) << "},\n"
         << "  \"metrics\":[\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    output << "    ";
    WriteMetric(output, metrics[i]);
    output << (i + 1 == metrics.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return pass ? 0 : 1;
}
