#include "gpu_ba/custom_cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
  double max_absolute = 0.0;
  double rms_absolute = 0.0;
  double p95_absolute = 0.0;
  double sum_squared = 0.0;
  std::vector<double> absolute_errors;
  std::string first_id;
  double first_reference = 0.0;
  double first_candidate = 0.0;

  void Add(const double reference, const double candidate, const std::string& id) {
    ++count;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      if (std::isfinite(reference) != std::isfinite(candidate)) {
        ++failures;
        if (first_id.empty()) {
          first_id = id;
          first_reference = reference;
          first_candidate = candidate;
        }
      }
      ++nonfinite;
      return;
    }
    const double absolute = std::abs(reference - candidate);
    const double scale = std::max(std::abs(reference), std::abs(candidate));
    const double limit = atol + rtol * scale;
    absolute_errors.push_back(absolute);
    max_absolute = std::max(max_absolute, absolute);
    sum_squared += absolute * absolute;
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
    if (!absolute_errors.empty()) {
      std::sort(absolute_errors.begin(), absolute_errors.end());
      const size_t index = std::min(
          absolute_errors.size() - 1,
          static_cast<size_t>(std::ceil(0.95 * absolute_errors.size()) - 1));
      p95_absolute = absolute_errors[index];
      rms_absolute = std::sqrt(sum_squared / absolute_errors.size());
    }
  }
};

std::string JsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      default: output << character; break;
    }
  }
  output << '"';
  return output.str();
}

std::string NumberOrNull(const double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

void WriteMetric(std::ostream& output, const Metric& metric) {
  output << "{\"name\":" << JsonString(metric.name)
         << ",\"atol\":" << NumberOrNull(metric.atol)
         << ",\"rtol\":" << NumberOrNull(metric.rtol)
         << ",\"count\":" << metric.count
         << ",\"failures\":" << metric.failures
         << ",\"nonfinite\":" << metric.nonfinite
         << ",\"max_absolute\":" << NumberOrNull(metric.max_absolute)
         << ",\"rms_absolute\":" << NumberOrNull(metric.rms_absolute)
         << ",\"p95_absolute\":" << NumberOrNull(metric.p95_absolute)
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

void AddVisual(const CudaVisualInput& input,
               const CudaVisualOutput& candidate,
               std::vector<Metric>* metrics,
               bool* finite_match) {
  const std::array<double, 4> q{{input.quaternion[0], input.quaternion[1],
                                  input.quaternion[2], input.quaternion[3]}};
  const std::array<double, 3> t{{input.translation[0], input.translation[1],
                                  input.translation[2]}};
  const std::array<double, 3> p{{input.point[0], input.point[1],
                                  input.point[2]}};
  const std::array<double, 8> camera{{input.camera[0], input.camera[1],
                                      input.camera[2], input.camera[3],
                                      input.camera[4], input.camera[5],
                                      input.camera[6], input.camera[7]}};
  const std::array<double, 2> observation{{input.observation[0],
                                            input.observation[1]}};
  VisualEvaluation reference;
  const bool reference_ok =
      EvaluateOpenCVVisual(q, t, p, camera, observation, &reference);
  if (reference_ok != (candidate.finite != 0)) *finite_match = false;
  const std::string prefix = "source=" + std::to_string(input.source_index);
  auto add = [&](const size_t metric_index, const double reference_value,
                 const double candidate_value, const size_t element) {
    (*metrics)[metric_index].Add(reference_value, candidate_value,
                                  prefix + ":element=" +
                                      std::to_string(element));
  };
  for (size_t i = 0; i < 2; ++i) add(0, reference.residual[i], candidate.residual[i], i);
  for (size_t i = 0; i < 8; ++i) {
    add(1, reference.ambient_quaternion_jacobian[i],
        candidate.ambient_quaternion_jacobian[i], i);
  }
  for (size_t i = 0; i < 12; ++i)
    add(2, reference.plus_jacobian[i], candidate.plus_jacobian[i], i);
  for (size_t i = 0; i < 6; ++i) {
    add(3, reference.local_rotation_jacobian[i],
        candidate.local_rotation_jacobian[i], i);
    add(4, reference.translation_jacobian[i],
        candidate.translation_jacobian[i], i);
    add(5, reference.point_jacobian[i], candidate.point_jacobian[i], i);
  }
  for (size_t i = 0; i < 16; ++i)
    add(6, reference.camera_jacobian[i], candidate.camera_jacobian[i], i);
}

void AddLidar(const CudaLidarInput& input,
              const CudaLidarOutput& candidate,
              std::vector<Metric>* metrics,
              bool* finite_match) {
  const std::array<double, 3> point{{input.point[0], input.point[1],
                                      input.point[2]}};
  const std::array<double, 4> plane{{input.plane[0], input.plane[1],
                                     input.plane[2], input.plane[3]}};
  const LidarEvaluation reference = EvaluateLidar(
      point, plane, input.weight, static_cast<LidarResidualMode>(input.mode),
      input.near_zero_threshold);
  if (reference.finite != (candidate.finite != 0)) *finite_match = false;
  const std::string prefix = "source=" + std::to_string(input.source_index);
  (*metrics)[7].Add(reference.signed_distance, candidate.signed_distance,
                    prefix + ":signed_distance");
  (*metrics)[8].Add(reference.residual, candidate.residual,
                    prefix + ":residual");
  for (size_t i = 0; i < 3; ++i) {
    (*metrics)[9].Add(reference.point_jacobian[i], candidate.point_jacobian[i],
                      prefix + ":element=" + std::to_string(i));
  }
}

bool CompareSnapshot(const Snapshot& snapshot,
                     const CudaLayerAResult& result,
                     const std::vector<CudaVisualInput>& visual,
                     const std::vector<CudaLidarInput>& lidar,
                     std::vector<Metric>* metrics,
                     bool* finite_match,
                     std::string* error) {
  if (result.visual.size() != visual.size() || result.lidar.size() != lidar.size()) {
    *error = "CUDA output count does not match packed input count";
    return false;
  }
  *finite_match = true;
  for (size_t i = 0; i < visual.size(); ++i)
    AddVisual(visual[i], result.visual[i], metrics, finite_match);
  for (size_t i = 0; i < lidar.size(); ++i)
    AddLidar(lidar[i], result.lidar[i], metrics, finite_match);
  return true;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap::gpu_ba;
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: gpu_ba_custom_cuda_replay SNAPSHOT_MANIFEST "
                 "REPORT_JSON [explicit|unified]\n";
    return 2;
  }
  Snapshot snapshot;
  SnapshotReadResult read_result;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read_result, &error)) {
    std::cerr << "ReadSnapshot failed: " << error << '\n';
    return 1;
  }
  std::vector<CudaVisualInput> visual;
  std::vector<CudaLidarInput> lidar;
  if (!BuildCudaLayerAInputs(snapshot, &visual, &lidar, &error)) {
    std::cerr << "BuildCudaLayerAInputs failed: " << error << '\n';
    return 1;
  }
  CudaLayerAOptions options;
  if (argc == 4 && std::string(argv[3]) == "unified") {
    options.memory_mode = CudaMemoryMode::kUnifiedManaged;
  }
  CudaLayerAResult result;
  if (!RunCudaLayerA(visual, lidar, options, &result)) {
    std::cerr << "RunCudaLayerA failed: " << result.error << '\n';
    return 1;
  }
  std::vector<Metric> metrics;
  const char* names[] = {"visual_residual", "ambient_quaternion_jacobian",
                         "plus_jacobian", "local_rotation_jacobian",
                         "translation_jacobian", "point_jacobian",
                         "camera_jacobian", "lidar_signed_distance",
                         "lidar_residual", "lidar_point_jacobian"};
  for (const char* name : names) {
    Metric metric;
    metric.name = name;
    metrics.push_back(metric);
  }
  bool finite_match = true;
  if (!CompareSnapshot(snapshot, result, visual, lidar, &metrics,
                       &finite_match, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  bool pass = finite_match;
  for (Metric& metric : metrics) {
    metric.Finish();
    pass = pass && metric.failures == 0;
  }

  std::ofstream output(argv[2]);
  if (!output) {
    std::cerr << "Cannot open report path: " << argv[2] << '\n';
    return 1;
  }
  output << std::setprecision(17);
  output << "{\n  \"phase\":\"phase6p3-custom-cuda-audit-v1\",\n"
         << "  \"layer\":\"A_residual_jacobian\",\n"
         << "  \"snapshot_id\":" << JsonString(snapshot.metadata.snapshot_id)
         << ",\n  \"memory_mode\":"
         << JsonString(result.runtime.used_unified_memory ? "unified"
                                                           : "explicit")
         << ",\n  \"visual_count\":" << visual.size()
         << ",\n  \"lidar_count\":" << lidar.size()
         << ",\n  \"finite_match\":" << (finite_match ? "true" : "false")
         << ",\n  \"cuda_ran\":true"
         << ",\n  \"pass\":" << (pass ? "true" : "false") << ",\n"
         << "  \"runtime\":{\"device\":" << result.runtime.device
         << ",\"compute_major\":" << result.runtime.device_major
         << ",\"compute_minor\":" << result.runtime.device_minor
         << ",\"launch_status\":" << result.runtime.launch_status
         << ",\"synchronize_status\":"
         << result.runtime.synchronize_status
         << ",\"free_bytes_before\":"
         << result.runtime.free_bytes_before
         << ",\"total_bytes\":" << result.runtime.total_bytes
         << ",\"active_bytes\":" << result.runtime.active_bytes
         << ",\"cached_bytes\":" << result.runtime.cached_bytes
         << ",\"kernel_milliseconds\":"
         << NumberOrNull(result.runtime.kernel_milliseconds)
         << ",\"copy_milliseconds\":"
         << NumberOrNull(result.runtime.copy_milliseconds)
         << ",\"host_wall_milliseconds\":"
         << NumberOrNull(result.runtime.host_wall_milliseconds)
         << ",\"visual_bytes\":" << result.runtime.visual_bytes
         << ",\"lidar_bytes\":" << result.runtime.lidar_bytes
         << ",\"output_bytes\":" << result.runtime.output_bytes << "},\n"
         << "  \"metrics\":[\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    output << "    ";
    WriteMetric(output, metrics[i]);
    output << (i + 1 == metrics.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return pass ? 0 : 1;
}
