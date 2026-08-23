#include "gpu_ba/custom_cuda.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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
  double rms = 0.0;
  double p95 = 0.0;
  double squared_sum = 0.0;
  std::vector<double> errors;
  std::string first_id;
  double first_reference = 0.0;
  double first_candidate = 0.0;

  void Add(double reference, double candidate, const std::string& id) {
    ++count;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      ++nonfinite;
      ++failures;
      if (first_id.empty()) {
        first_id = id;
        first_reference = reference;
        first_candidate = candidate;
      }
      return;
    }
    const double absolute = std::abs(reference - candidate);
    const double limit = atol + rtol *
        std::max(std::abs(reference), std::abs(candidate));
    maximum = std::max(maximum, absolute);
    squared_sum += absolute * absolute;
    errors.push_back(absolute);
    if (absolute > limit) {
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
  for (const char character : value) {
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

bool Compare(const CustomCpuLinearizationExport& reference,
             const CudaLayerBResult& candidate,
             std::vector<Metric>* metrics,
             std::string* topology_error) {
  if (reference.poses.size() != candidate.poses.size() ||
      reference.points.size() != candidate.points.size() ||
      reference.edges.size() != candidate.edges.size()) {
    *topology_error = "pose/point/edge counts differ";
    return false;
  }
  for (size_t block = 0; block < reference.poses.size(); ++block) {
    const auto& lhs = reference.poses[block];
    const auto& rhs = candidate.poses[block];
    if (lhs.image_id != rhs.image_id || lhs.dimension != rhs.dimension ||
        lhs.free_translation_indices.size() != lhs.dimension - 3) {
      *topology_error = "pose identity/dimension differs at block=" +
                        std::to_string(block);
      return false;
    }
    for (size_t i = 0; i < lhs.free_translation_indices.size(); ++i) {
      if (lhs.free_translation_indices[i] !=
          rhs.free_translation_indices[i]) {
        *topology_error = "pose subset differs at image=" +
                          std::to_string(lhs.image_id);
        return false;
      }
    }
    const std::string prefix = "image=" + std::to_string(lhs.image_id);
    for (size_t row = 0; row < lhs.dimension; ++row) {
      (*metrics)[3].Add(lhs.gradient[row], rhs.gradient[row],
                        prefix + ":g[" + std::to_string(row) + "]");
      (*metrics)[5].Add(lhs.jacobi_scaling[row], rhs.jacobi_scaling[row],
                        prefix + ":scale[" + std::to_string(row) + "]");
      (*metrics)[6].Add(lhs.damping[row], rhs.damping[row],
                        prefix + ":D[" + std::to_string(row) + "]");
      for (size_t col = 0; col < lhs.dimension; ++col) {
        const size_t index = row * lhs.dimension + col;
        (*metrics)[0].Add(lhs.hessian[index], rhs.hessian[index],
                          prefix + ":B[" + std::to_string(row) + "," +
                              std::to_string(col) + "]");
      }
    }
  }
  for (size_t block = 0; block < reference.points.size(); ++block) {
    const auto& lhs = reference.points[block];
    const auto& rhs = candidate.points[block];
    if (lhs.point3D_id != rhs.point3D_id) {
      *topology_error = "point identity differs at block=" +
                        std::to_string(block);
      return false;
    }
    const std::string prefix = "point3D=" + std::to_string(lhs.point3D_id);
    for (size_t i = 0; i < 9; ++i)
      (*metrics)[1].Add(lhs.hessian[i], rhs.hessian[i],
                        prefix + ":C[" + std::to_string(i) + "]");
    for (size_t i = 0; i < 3; ++i) {
      (*metrics)[4].Add(lhs.gradient[i], rhs.gradient[i],
                        prefix + ":g[" + std::to_string(i) + "]");
      (*metrics)[5].Add(lhs.jacobi_scaling[i], rhs.jacobi_scaling[i],
                        prefix + ":scale[" + std::to_string(i) + "]");
      (*metrics)[6].Add(lhs.damping[i], rhs.damping[i],
                        prefix + ":D[" + std::to_string(i) + "]");
    }
  }
  for (size_t block = 0; block < reference.edges.size(); ++block) {
    const auto& lhs = reference.edges[block];
    const auto& rhs = candidate.edges[block];
    if (lhs.pose_index != rhs.pose_index ||
        lhs.point_index != rhs.point_index ||
        lhs.pose_dimension != rhs.pose_dimension) {
      *topology_error = "edge identity differs at block=" +
                        std::to_string(block);
      return false;
    }
    const std::string prefix =
        "image=" + std::to_string(reference.poses[lhs.pose_index].image_id) +
        ":point3D=" +
        std::to_string(reference.points[lhs.point_index].point3D_id);
    for (size_t i = 0; i < lhs.pose_dimension * 3; ++i)
      (*metrics)[2].Add(lhs.value[i], rhs.value[i],
                        prefix + ":E[" + std::to_string(i) + "]");
  }
  (*metrics)[7].Add(reference.gradient_norms.projected_max_norm,
                    candidate.projected_gradient_max_norm, "global");
  (*metrics)[8].Add(reference.gradient_norms.raw_tangent_max_norm,
                    candidate.raw_tangent_gradient_max_norm, "global");
  (*metrics)[9].Add(reference.gradient_norms.scaled_max_norm,
                    candidate.scaled_gradient_max_norm, "global");
  return true;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap::gpu_ba;
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: gpu_ba_custom_cuda_layer_b_replay "
                 "SNAPSHOT REPORT [explicit|unified] [parallel]\n"
                 "   or: SNAPSHOT REPORT component SEGMENT_SIZE PAIRED_SAMPLES\n";
    return 2;
  }
  Snapshot snapshot;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (argc == 6 && std::string(argv[3]) == "component") {
    const uint32_t segment_size =
        static_cast<uint32_t>(std::stoul(argv[4]));
    const size_t paired_samples = static_cast<size_t>(std::stoul(argv[5]));
    CudaHessianAssemblyComponentResult component;
    if (!RunCudaHessianAssemblyComponentForTesting(
            snapshot, segment_size, paired_samples, &component, &error)) {
      std::cerr << "Hessian component failed: " << error << '\n';
      return 1;
    }
    std::ofstream output(argv[2]);
    if (!output) return 1;
    const auto vector = [&output](const std::vector<double>& values) {
      output << '[';
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ',';
        output << std::setprecision(17) << values[i];
      }
      output << ']';
    };
    output << std::setprecision(17)
           << "{\n  \"phase\":\"phase10p0b-fixed-workload-component\",\n"
           << "  \"snapshot_id\":"
           << JsonString(snapshot.metadata.snapshot_id)
           << ",\n  \"visual_count\":" << component.visual_count
           << ",\"lidar_count\":" << component.lidar_count
           << ",\"pose_count\":" << component.pose_count
           << ",\"point_count\":" << component.point_count
           << ",\"edge_count\":" << component.edge_count
           << ",\n  \"segment_size\":" << component.segment_size
           << ",\"pose_segment_count\":" << component.pose_segment_count
           << ",\"point_segment_count\":" << component.point_segment_count
           << ",\"point_direct_count\":" << component.point_direct_count
           << ",\"partial_workspace_bytes\":"
           << component.partial_workspace_bytes
           << ",\n  \"atomic_add_estimate\":{\"pose\":"
           << component.atomic_pose_add_estimate << ",\"point\":"
           << component.atomic_point_add_estimate << ",\"edge\":"
           << component.atomic_edge_add_estimate << "},\n"
           << "  \"correctness\":{\"atomic_violations\":"
           << component.element_contract_violations_atomic
           << ",\"segmented_violations\":"
           << component.element_contract_violations_segmented
           << ",\"atomic_max_abs_error\":"
           << component.atomic_max_abs_error
           << ",\"atomic_normalized_max_error\":"
           << component.atomic_normalized_max_error
           << ",\"atomic_normalized_frobenius_error\":"
           << component.atomic_normalized_frobenius_error
           << ",\"segmented_max_abs_error\":"
           << component.segmented_max_abs_error
           << ",\"segmented_normalized_max_error\":"
           << component.segmented_normalized_max_error
           << ",\"segmented_normalized_frobenius_error\":"
           << component.segmented_normalized_frobenius_error
           << ",\"atomic_repeat_max_abs_error\":"
           << component.atomic_repeated_max_abs_error
           << ",\"segmented_repeat_max_abs_error\":"
           << component.segmented_repeated_max_abs_error
           << ",\"atomic_first_violation\":{\"index\":"
           << component.atomic_first_violation_index << ",\"reference\":"
           << component.atomic_first_violation_reference << ",\"candidate\":"
           << component.atomic_first_violation_candidate
           << "},\"segmented_first_violation\":{\"index\":"
           << component.segmented_first_violation_index
           << ",\"reference\":"
           << component.segmented_first_violation_reference
           << ",\"candidate\":"
           << component.segmented_first_violation_candidate << "}},\n"
           << "  \"pose_owned_complete_milliseconds\":";
    vector(component.pose_owned_complete_milliseconds);
    output << ",\n  \"atomic_complete_milliseconds\":";
    vector(component.atomic_complete_milliseconds);
    output << ",\n  \"segmented_complete_milliseconds\":";
    vector(component.segmented_complete_milliseconds);
    output << "\n}\n";
    return 0;
  }
  CustomCpuLinearizationExport reference;
  if (!ExportCustomCpuCanonicalLinearization(snapshot, 1e-6, 1e32,
                                               &reference, &error)) {
    std::cerr << "CPU export failed: " << error << '\n';
    return 1;
  }
  CudaLayerBOptions options;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "unified")
      options.layer_a.memory_mode = CudaMemoryMode::kUnifiedManaged;
    if (std::string(argv[i]) == "parallel") {
      options.reduction_mode = CudaReductionMode::kParallelDeterministic;
      options.cost_reduction_threads = 128;
    }
  }
  CudaLayerBResult candidate;
  if (!RunCudaSnapshotLayerB(snapshot, options, &candidate, &error)) {
    std::cerr << "CUDA Layer B failed: " << error << '\n';
    return 1;
  }
  const char* names[] = {"B_pose_blocks", "C_point_blocks", "E_edges",
                         "pose_gradient", "point_gradient",
                         "jacobi_scaling", "lm_damping",
                         "projected_gradient_max_norm",
                         "raw_tangent_gradient_max_norm",
                         "scaled_gradient_max_norm"};
  std::vector<Metric> metrics;
  for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
    Metric metric;
    metric.name = names[index];
    // These are the already-frozen fixed-linearization gates from
    // fixed_linearization.cc, not a Phase-6 runtime relaxation.
    if (index <= 4) {
      metric.atol = 1e-9;
      metric.rtol = 1e-9;
    } else if (index <= 6) {
      metric.atol = 1e-12;
      metric.rtol = 1e-10;
    }
    metrics.push_back(metric);
  }
  std::string topology_error;
  const bool topology_pass =
      Compare(reference, candidate, &metrics, &topology_error);
  bool pass = topology_pass;
  for (Metric& metric : metrics) {
    metric.Finish();
    pass = pass && metric.failures == 0;
  }

  std::ofstream output(argv[2]);
  if (!output) {
    std::cerr << "Cannot open report path\n";
    return 1;
  }
  output << std::setprecision(17)
         << "{\n  \"phase\":\"phase6p3-custom-cuda-audit-v1\",\n"
         << "  \"layer\":\"B_assembly_gradient_damping\",\n"
         << "  \"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
         << ",\n  \"cuda_ran\":true"
         << ",\n  \"pass\":" << (pass ? "true" : "false")
         << ",\n  \"topology_pass\":"
         << (topology_pass ? "true" : "false")
         << ",\n  \"topology_error\":"
         << (topology_error.empty() ? "null" : JsonString(topology_error))
         << ",\n  \"pose_blocks\":" << candidate.poses.size()
         << ",\n  \"point_blocks\":" << candidate.points.size()
         << ",\n  \"edge_blocks\":" << candidate.edges.size()
         << ",\n  \"runtime\":{\"layer_a_kernel_ms\":"
         << NumberOrNull(candidate.layer_a.runtime.kernel_milliseconds)
         << ",\"assembly_kernel_ms\":"
         << NumberOrNull(candidate.runtime.assembly_kernel_milliseconds)
         << ",\"cost_kernel_ms\":"
         << NumberOrNull(candidate.runtime.cost_kernel_milliseconds)
         << ",\"gradient_kernel_ms\":"
         << NumberOrNull(candidate.runtime.gradient_kernel_milliseconds)
         << ",\"cost_reduction_parallel\":"
         << (candidate.runtime.cost_reduction_parallel ? "true" : "false")
         << ",\"gradient_reduction_parallel\":"
         << (candidate.runtime.gradient_reduction_parallel ? "true" : "false")
         << ",\"configured_worker_count\":"
         << candidate.runtime.configured_worker_count
         << ",\"effective_worker_count\":"
         << candidate.runtime.effective_worker_count
         << ",\"reduction_mode\":"
         << JsonString(candidate.runtime.reduction_mode ==
                               CudaReductionMode::kParallelDeterministic
                           ? "parallel_deterministic"
                           : "serial_deterministic")
         << ",\"active_bytes\":" << candidate.runtime.active_bytes
         << ",\"cached_bytes\":" << candidate.runtime.cached_bytes
         << ",\"resident_bytes\":" << candidate.runtime.resident_bytes
         << ",\"predicted_additional_bytes\":"
         << candidate.runtime.predicted_additional_bytes
         << ",\"predicted_peak_bytes\":"
         << candidate.runtime.predicted_peak_bytes
         << ",\"free_bytes_before\":"
         << candidate.runtime.free_bytes_before
         << ",\"total_bytes\":" << candidate.runtime.total_bytes
         << ",\"copy_ms\":"
         << NumberOrNull(candidate.runtime.copy_milliseconds)
         << ",\"host_wall_ms\":"
         << NumberOrNull(candidate.runtime.host_wall_milliseconds)
         << ",\"metadata_bytes\":" << candidate.runtime.metadata_bytes
         << ",\"adjacency_bytes\":" << candidate.runtime.adjacency_bytes
         << ",\"output_bytes\":" << candidate.runtime.output_bytes
         << ",\"launch_status\":" << candidate.runtime.launch_status
         << ",\"synchronize_status\":"
         << candidate.runtime.synchronize_status << "},\n"
         << "  \"metrics\":[\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    output << "    ";
    WriteMetric(output, metrics[i]);
    output << (i + 1 == metrics.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return pass ? 0 : 1;
}
