#include "gpu_ba/custom_cuda.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu_ba/fixed_linearization.h"
#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {
namespace {

constexpr char kPhaseName[] = "phase6p3-custom-cuda-audit-v1";

std::string JsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const char c : value) {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else out << c;
  }
  out << '"';
  return out.str();
}

std::string NumberOrNull(const double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

const char* ReductionModeName(const CudaReductionMode mode) {
  return mode == CudaReductionMode::kParallelDeterministic
             ? "parallel_deterministic"
             : "serial_deterministic";
}

struct Difference {
  bool size_match = true;
  bool finite_match = true;
  bool topology_match = true;
  bool numeric_match = true;
  double atol = 0.0;
  double rtol = 0.0;
  std::string first_error;
  double max_absolute = 0.0;
  double max_relative = 0.0;
  size_t count = 0;
  size_t first_index = std::numeric_limits<size_t>::max();
  double first_reference = 0.0;
  double first_candidate = 0.0;

  bool Pass() const {
    return size_match && finite_match && topology_match && numeric_match;
  }
};

void SetError(Difference* result, const bool topology, const std::string& error) {
  if (topology) result->topology_match = false;
  else if (error.find("size") != std::string::npos)
    result->size_match = false;
  else result->finite_match = false;
  if (result->first_error.empty()) result->first_error = error;
}

void AddDifference(const double reference, const double candidate,
                   const size_t index, Difference* result) {
  ++result->count;
  if (!std::isfinite(reference) || !std::isfinite(candidate)) {
    SetError(result, false, "non-finite element at index=" +
                                std::to_string(index));
    return;
  }
  const double absolute = std::abs(reference - candidate);
  const double relative = absolute /
      std::max({std::abs(reference), std::abs(candidate), 1e-300});
  result->max_absolute = std::max(result->max_absolute, absolute);
  result->max_relative = std::max(result->max_relative, relative);
  const double threshold = result->atol +
      result->rtol * std::max(std::abs(reference), std::abs(candidate));
  if (absolute > threshold) {
    result->numeric_match = false;
  }
  if (absolute > threshold && result->first_index ==
      std::numeric_limits<size_t>::max()) {
    result->first_index = index;
    result->first_reference = reference;
    result->first_candidate = candidate;
  }
}

Difference CompareVector(const std::vector<double>& reference,
                         const std::vector<double>& candidate,
                         const double atol = 1e-12,
                         const double rtol = 1e-10) {
  Difference result;
  result.atol = atol;
  result.rtol = rtol;
  if (reference.size() != candidate.size())
    SetError(&result, false, "size mismatch");
  const size_t count = std::min(reference.size(), candidate.size());
  for (size_t i = 0; i < count; ++i) AddDifference(reference[i], candidate[i], i,
                                                     &result);
  for (size_t i = count; i < reference.size(); ++i) {
    if (!std::isfinite(reference[i])) SetError(&result, false, "non-finite reference tail");
  }
  for (size_t i = count; i < candidate.size(); ++i) {
    if (!std::isfinite(candidate[i])) SetError(&result, false, "non-finite candidate tail");
  }
  return result;
}

template <size_t N>
bool FiniteArray(const std::array<double, N>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool FiniteSnapshot(const Snapshot& snapshot, std::string* error) {
  const auto fail = [&](const std::string& where) {
    if (error != nullptr) *error = "non-finite snapshot field: " + where;
    return false;
  };
  const double metadata_values[] = {
      snapshot.metadata.proj_lidar_weight,
      snapshot.metadata.icp_lidar_weight,
      snapshot.metadata.icp_ground_lidar_weight,
      snapshot.metadata.function_tolerance,
      snapshot.metadata.gradient_tolerance,
      snapshot.metadata.parameter_tolerance};
  for (const double value : metadata_values) {
    if (!std::isfinite(value)) return fail("metadata");
  }
  for (const CameraSnapshot& camera : snapshot.cameras) {
    for (const double value : camera.params)
      if (!std::isfinite(value)) return fail("camera");
  }
  for (const ImageSnapshot& image : snapshot.images) {
    if (!FiniteArray(image.qvec) || !FiniteArray(image.tvec))
      return fail("image=" + std::to_string(image.image_id));
  }
  for (const PointSnapshot& point : snapshot.points) {
    if (!FiniteArray(point.xyz) ||
        (point.has_search_range && !std::isfinite(point.search_range)))
      return fail("point=" + std::to_string(point.point3D_id));
  }
  for (const ObservationSnapshot& observation : snapshot.observations) {
    if (!FiniteArray(observation.xy)) return fail("observation");
  }
  for (const LidarSnapshot& lidar : snapshot.lidar) {
    if (!FiniteArray(lidar.lidar_xyz) || !FiniteArray(lidar.plane) ||
        !std::isfinite(lidar.weight) ||
        (lidar.has_search_range && !std::isfinite(lidar.search_range)))
      return fail("lidar");
  }
  return true;
}

bool ValidateSnapshotTopology(const Snapshot& lhs, const Snapshot& rhs,
                              std::string* error) {
  if (lhs.images.size() != rhs.images.size() ||
      lhs.points.size() != rhs.points.size() ||
      lhs.cameras.size() != rhs.cameras.size() ||
      lhs.observations.size() != rhs.observations.size() ||
      lhs.tracks.size() != rhs.tracks.size() ||
      lhs.lidar.size() != rhs.lidar.size() ||
      lhs.parameter_blocks_source_order.size() !=
          rhs.parameter_blocks_source_order.size() ||
      lhs.source_insertion_order.size() != rhs.source_insertion_order.size() ||
      lhs.canonical_order.size() != rhs.canonical_order.size()) {
    *error = "snapshot topology size mismatch";
    return false;
  }
  for (size_t i = 0; i < lhs.source_insertion_order.size(); ++i) {
    const auto& a = lhs.source_insertion_order[i];
    const auto& b = rhs.source_insertion_order[i];
    if (a.source_index != b.source_index || a.residual_kind != b.residual_kind ||
        a.image_id != b.image_id || a.point2D_idx != b.point2D_idx ||
        a.point3D_id != b.point3D_id) {
      *error = "source insertion topology mismatch at index=" +
               std::to_string(i);
      return false;
    }
  }
  for (size_t i = 0; i < lhs.canonical_order.size(); ++i) {
    const auto& a = lhs.canonical_order[i];
    const auto& b = rhs.canonical_order[i];
    if (a.source_index != b.source_index || a.residual_kind != b.residual_kind ||
        a.image_id != b.image_id || a.point2D_idx != b.point2D_idx ||
        a.point3D_id != b.point3D_id) {
      *error = "canonical topology mismatch at index=" + std::to_string(i);
      return false;
    }
  }
  return true;
}

double LossCost(const double squared_norm, const Snapshot& snapshot) {
  std::string loss = snapshot.metadata.loss_function;
  std::transform(loss.begin(), loss.end(), loss.begin(),
                 [](const unsigned char c) { return std::toupper(c); });
  if (loss != "SOFT_L1") return 0.5 * squared_norm;
  const double scale = 1.0;
  return scale * scale *
         (std::sqrt(1.0 + squared_norm / (scale * scale)) - 1.0);
}

// This deliberately uses the same scalar double accumulation as custom_cpu.
bool HostSourceCost(const Snapshot& snapshot, double* cost,
                    uint64_t* near_zero, std::string* error) {
  std::vector<CudaVisualInput> visual;
  std::vector<CudaLidarInput> lidar;
  if (!BuildCudaLayerAInputs(snapshot, CudaResidualOrder::kSourceInsertion,
                             &visual, &lidar, error)) return false;
  std::unordered_map<uint64_t, const CudaVisualInput*> visual_by_source;
  std::unordered_map<uint64_t, const CudaLidarInput*> lidar_by_source;
  for (const CudaVisualInput& value : visual)
    visual_by_source.emplace(value.source_index, &value);
  for (const CudaLidarInput& value : lidar)
    lidar_by_source.emplace(value.source_index, &value);
  double total = 0.0;
  *near_zero = 0;
  for (const OrderEntrySnapshot& entry : snapshot.source_insertion_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      const auto it = visual_by_source.find(entry.source_index);
      if (it == visual_by_source.end()) {
        *error = "source-order visual is missing from packed inputs";
        return false;
      }
      const CudaVisualInput& value = *it->second;
      VisualEvaluation evaluation;
      if (!EvaluateOpenCVVisual(
              {{value.quaternion[0], value.quaternion[1], value.quaternion[2],
                value.quaternion[3]}},
              {{value.translation[0], value.translation[1], value.translation[2]}},
              {{value.point[0], value.point[1], value.point[2]}},
              {{value.camera[0], value.camera[1], value.camera[2], value.camera[3],
                value.camera[4], value.camera[5], value.camera[6], value.camera[7]}},
              {{value.observation[0], value.observation[1]}}, &evaluation)) {
        *error = "host visual evaluation returned non-finite";
        return false;
      }
      const double squared = evaluation.residual[0] * evaluation.residual[0] +
                             evaluation.residual[1] * evaluation.residual[1];
      total += LossCost(squared, snapshot);
    } else {
      const auto it = lidar_by_source.find(entry.source_index);
      if (it == lidar_by_source.end()) {
        *error = "source-order lidar is missing from packed inputs";
        return false;
      }
      const CudaLidarInput& value = *it->second;
      LidarEvaluation evaluation = EvaluateLidar(
          {{value.point[0], value.point[1], value.point[2]}},
          {{value.plane[0], value.plane[1], value.plane[2], value.plane[3]}},
          value.weight, static_cast<LidarResidualMode>(value.mode),
          value.near_zero_threshold);
      if (evaluation.near_zero) ++*near_zero;
      if (!evaluation.finite) {
        *error = "host lidar evaluation returned non-finite";
        return false;
      }
      total += LossCost(evaluation.residual * evaluation.residual, snapshot);
    }
  }
  *cost = total;
  return std::isfinite(*cost);
}

bool HostSourceResiduals(const Snapshot& snapshot,
                         std::vector<double>* residuals,
                         std::string* error) {
  std::vector<CudaVisualInput> visual;
  std::vector<CudaLidarInput> lidar;
  if (!BuildCudaLayerAInputs(snapshot, CudaResidualOrder::kSourceInsertion,
                             &visual, &lidar, error)) return false;
  std::unordered_map<uint64_t, const CudaVisualInput*> visual_by_source;
  std::unordered_map<uint64_t, const CudaLidarInput*> lidar_by_source;
  for (const auto& value : visual) visual_by_source[value.source_index] = &value;
  for (const auto& value : lidar) lidar_by_source[value.source_index] = &value;
  residuals->clear();
  for (const auto& entry : snapshot.source_insertion_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      const auto it = visual_by_source.find(entry.source_index);
      if (it == visual_by_source.end()) {
        *error = "host residual source visual missing";
        return false;
      }
      const auto& value = *it->second;
      VisualEvaluation evaluation;
      if (!EvaluateOpenCVVisual(
              {{value.quaternion[0], value.quaternion[1], value.quaternion[2],
                value.quaternion[3]}},
              {{value.translation[0], value.translation[1], value.translation[2]}},
              {{value.point[0], value.point[1], value.point[2]}},
              {{value.camera[0], value.camera[1], value.camera[2], value.camera[3],
                value.camera[4], value.camera[5], value.camera[6], value.camera[7]}},
              {{value.observation[0], value.observation[1]}}, &evaluation)) {
        *error = "host residual visual evaluation failed";
        return false;
      }
      residuals->push_back(evaluation.residual[0]);
      residuals->push_back(evaluation.residual[1]);
    } else {
      const auto it = lidar_by_source.find(entry.source_index);
      if (it == lidar_by_source.end()) {
        *error = "host residual source lidar missing";
        return false;
      }
      const auto& value = *it->second;
      const LidarEvaluation evaluation = EvaluateLidar(
          {{value.point[0], value.point[1], value.point[2]}},
          {{value.plane[0], value.plane[1], value.plane[2], value.plane[3]}},
          value.weight, static_cast<LidarResidualMode>(value.mode),
          value.near_zero_threshold);
      residuals->push_back(evaluation.residual);
    }
  }
  return true;
}

std::vector<double> CudaSourceResiduals(const Snapshot& snapshot,
                                        const CudaLayerBResult& result) {
  std::vector<CudaVisualInput> visual;
  std::vector<CudaLidarInput> lidar;
  std::string ignored;
  if (!BuildCudaLayerAInputs(snapshot, CudaResidualOrder::kSourceInsertion,
                             &visual, &lidar, &ignored)) {
    return {};
  }
  std::unordered_map<uint64_t, size_t> visual_index;
  std::unordered_map<uint64_t, size_t> lidar_index;
  for (size_t i = 0; i < visual.size(); ++i)
    visual_index[visual[i].source_index] = i;
  for (size_t i = 0; i < lidar.size(); ++i)
    lidar_index[lidar[i].source_index] = i;
  std::vector<double> residuals;
  for (const auto& entry : snapshot.source_insertion_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      const auto it = visual_index.find(entry.source_index);
      if (it == visual_index.end() || it->second >= result.layer_a.visual.size())
        return {};
      residuals.push_back(result.layer_a.visual[it->second].residual[0]);
      residuals.push_back(result.layer_a.visual[it->second].residual[1]);
    } else {
      const auto it = lidar_index.find(entry.source_index);
      if (it == lidar_index.end() || it->second >= result.layer_a.lidar.size())
        return {};
      residuals.push_back(result.layer_a.lidar[it->second].residual);
    }
  }
  return residuals;
}

std::vector<double> SnapshotStateValues(const Snapshot& snapshot) {
  std::vector<double> values;
  for (const ImageSnapshot& image : snapshot.images) {
    values.insert(values.end(), image.qvec.begin(), image.qvec.end());
    values.insert(values.end(), image.tvec.begin(), image.tvec.end());
  }
  for (const CameraSnapshot& camera : snapshot.cameras)
    values.insert(values.end(), camera.params.begin(), camera.params.end());
  for (const PointSnapshot& point : snapshot.points)
    values.insert(values.end(), point.xyz.begin(), point.xyz.end());
  return values;
}

Difference CompareLinearization(const CustomCpuLinearizationExport& reference,
                                const CudaLayerBResult& candidate,
                                const double atol = 1e-10,
                                const double rtol = 1e-7) {
  Difference result;
  result.atol = atol;
  result.rtol = rtol;
  if (reference.poses.size() != candidate.poses.size() ||
      reference.points.size() != candidate.points.size() ||
      reference.edges.size() != candidate.edges.size()) {
    SetError(&result, true, "linearization topology size mismatch");
  }
  size_t index = 0;
  const size_t pose_count = std::min(reference.poses.size(), candidate.poses.size());
  for (size_t i = 0; i < pose_count; ++i) {
    const auto& a = reference.poses[i];
    const auto& b = candidate.poses[i];
    if (a.image_id != b.image_id || a.dimension != b.dimension ||
        a.free_translation_indices.size() != 3) {
      SetError(&result, true, "pose topology mismatch at index=" + std::to_string(i));
    }
    for (size_t j = 0; j < a.free_translation_indices.size(); ++j) {
      if (j >= 3 || a.free_translation_indices[j] != b.free_translation_indices[j])
        SetError(&result, true, "pose mask mismatch at index=" + std::to_string(i));
    }
    if (a.hessian.size() != static_cast<size_t>(a.dimension) * a.dimension ||
        b.dimension != a.dimension) {
      SetError(&result, true, "pose hessian size mismatch");
    }
    for (size_t j = 0; j < a.hessian.size(); ++j)
      AddDifference(a.hessian[j], j < 36 ? b.hessian[j] : std::numeric_limits<double>::quiet_NaN(), index++, &result);
    for (size_t j = 0; j < a.gradient.size(); ++j)
      AddDifference(a.gradient[j], j < 6 ? b.gradient[j] : std::numeric_limits<double>::quiet_NaN(), index++, &result);
    for (size_t j = 0; j < a.jacobi_scaling.size(); ++j)
      AddDifference(a.jacobi_scaling[j],
                    j < 6 ? b.jacobi_scaling[j]
                           : std::numeric_limits<double>::quiet_NaN(),
                    index++, &result);
    for (size_t j = 0; j < a.damping.size(); ++j)
      AddDifference(a.damping[j],
                    j < 6 ? b.damping[j]
                           : std::numeric_limits<double>::quiet_NaN(),
                    index++, &result);
  }
  const size_t point_count = std::min(reference.points.size(), candidate.points.size());
  for (size_t i = 0; i < point_count; ++i) {
    const auto& a = reference.points[i];
    const auto& b = candidate.points[i];
    if (a.point3D_id != b.point3D_id)
      SetError(&result, true, "point topology mismatch at index=" + std::to_string(i));
    for (size_t j = 0; j < 9; ++j) AddDifference(a.hessian[j], b.hessian[j], index++, &result);
    for (size_t j = 0; j < 3; ++j) AddDifference(a.gradient[j], b.gradient[j], index++, &result);
    for (size_t j = 0; j < 3; ++j) {
      AddDifference(a.jacobi_scaling[j], b.jacobi_scaling[j], index++, &result);
      AddDifference(a.damping[j], b.damping[j], index++, &result);
    }
  }
  const size_t edge_count = std::min(reference.edges.size(), candidate.edges.size());
  for (size_t i = 0; i < edge_count; ++i) {
    const auto& a = reference.edges[i];
    const auto& b = candidate.edges[i];
    if (a.pose_index != b.pose_index || a.point_index != b.point_index ||
        a.pose_dimension != b.pose_dimension)
      SetError(&result, true, "edge topology mismatch at index=" + std::to_string(i));
    for (size_t j = 0; j < a.pose_dimension * 3; ++j)
      AddDifference(a.value[j], j < 18 ? b.value[j] : std::numeric_limits<double>::quiet_NaN(), index++, &result);
  }
  return result;
}

void WriteDifference(std::ostream& out, const Difference& value) {
  out << "{\"pass\":" << (value.Pass() ? "true" : "false")
      << ",\"size_match\":" << (value.size_match ? "true" : "false")
      << ",\"finite_match\":" << (value.finite_match ? "true" : "false")
      << ",\"topology_match\":" << (value.topology_match ? "true" : "false")
      << ",\"numeric_match\":" << (value.numeric_match ? "true" : "false")
      << ",\"atol\":" << NumberOrNull(value.atol)
      << ",\"rtol\":" << NumberOrNull(value.rtol)
      << ",\"error\":"
      << (value.first_error.empty() ? "null" : JsonString(value.first_error))
      << ",\"count\":" << value.count
      << ",\"max_absolute\":" << NumberOrNull(value.max_absolute)
      << ",\"max_relative\":" << NumberOrNull(value.max_relative)
      << ",\"first_index\":"
      << (value.first_index == std::numeric_limits<size_t>::max()
              ? "null" : std::to_string(value.first_index))
      << ",\"first_reference\":" << NumberOrNull(value.first_reference)
      << ",\"first_candidate\":" << NumberOrNull(value.first_candidate)
      << '}';
}

struct CudaReductionRun {
  CudaReductionMode mode = CudaReductionMode::kSerialDeterministic;
  int configured_workers = 1;
  int effective_workers = 1;
  CudaLayerBResult initial;
  CudaLayerCResult step;
  CudaLayerBResult trial;
};

bool RunReductionPath(const Snapshot& state, const double lambda,
                      const int workers,
                      const CudaReductionMode mode, CudaReductionRun* run,
                      std::string* error) {
  run->mode = mode;
  run->configured_workers = workers;
  run->effective_workers = mode == CudaReductionMode::kParallelDeterministic
                               ? workers : 1;
  CudaLayerCOptions options;
  options.lambda = lambda;
  options.layer_b.layer_a.residual_order = CudaResidualOrder::kSourceInsertion;
  options.layer_b.cost_reduction_threads = workers;
  options.layer_b.reduction_mode = mode;
  if (!RunCudaSnapshotLayerB(state, options.layer_b, &run->initial, error) ||
      !RunCudaSnapshotLayerC(state, options, &run->step, error) ||
      !RunCudaSnapshotLayerB(run->step.trial_state, options.layer_b,
                             &run->trial, error)) {
    return false;
  }
  return true;
}

void WriteReductionRun(std::ostream& out, const CudaReductionRun& run) {
  out << "{\"reduction_mode\":" << JsonString(ReductionModeName(run.mode))
      << ",\"effective_worker_count\":" << run.effective_workers
      << ",\"configured_worker_count\":" << run.configured_workers
      << ",\"initial_cost\":" << NumberOrNull(run.initial.cost)
      << ",\"trial_cost_from_layer_c\":" << NumberOrNull(run.step.trial_cost)
      << ",\"trial_cost_from_layer_b\":" << NumberOrNull(run.trial.cost)
      << ",\"cost_reduction_parallel\":"
      << (run.initial.runtime.cost_reduction_parallel ? "true" : "false")
      << ",\"gradient_reduction_parallel\":"
      << (run.initial.runtime.gradient_reduction_parallel ? "true" : "false")
      << ",\"initial_runtime_host_ms\":"
      << NumberOrNull(run.initial.runtime.host_wall_milliseconds)
      << ",\"trial_runtime_host_ms\":"
      << NumberOrNull(run.trial.runtime.host_wall_milliseconds)
      << '}';
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  using namespace colmap::gpu_ba;
  if (argc < 5 || argc > 7) {
    std::cerr << "usage: gpu_ba_custom_cuda_global120_diagnostic "
                 "CPU_STATE CUDA_STATE LAMBDA REPORT [CUDA_THREADS] "
                 "[STATE_MODE]\n";
    return 2;
  }
  const double lambda = std::stod(argv[3]);
  const int workers = argc >= 6 ? std::max(1, std::stoi(argv[5])) : 8;
  const std::string state_mode = argc == 7 ? argv[6] : "common_fixed_state";
  if (state_mode != "common_fixed_state" &&
      state_mode != "trajectory_separate_accepted_states") {
    std::cerr << "invalid STATE_MODE: " << state_mode << '\n';
    return 2;
  }
  Snapshot cpu_state;
  Snapshot cuda_state;
  SnapshotReadResult ignored;
  std::string error;
  if (!ReadSnapshot(argv[1], &cpu_state, &ignored, &error) ||
      !ReadSnapshot(argv[2], &cuda_state, &ignored, &error)) {
    std::cerr << "snapshot read failed: " << error << '\n';
    return 1;
  }
  if (!FiniteSnapshot(cpu_state, &error) || !FiniteSnapshot(cuda_state, &error) ||
      !ValidateSnapshotTopology(cpu_state, cuda_state, &error)) {
    std::cerr << "snapshot validation failed: " << error << '\n';
    return 1;
  }

  CustomCpuCanonicalStepExport cpu_step;
  if (!ExportCustomCpuSourceStep(cpu_state, workers, lambda, 1e-6, 1e32,
                                 &cpu_step, &error)) {
    std::cerr << "CPU source step failed: " << error << '\n';
    return 1;
  }
  if (!FiniteSnapshot(cpu_step.trial_state, &error)) {
    std::cerr << "CPU trial state validation failed: " << error << '\n';
    return 1;
  }
  CudaReductionRun serial;
  CudaReductionRun parallel;
  if (!RunReductionPath(cuda_state, lambda, workers,
                        CudaReductionMode::kSerialDeterministic, &serial,
                        &error) ||
      !RunReductionPath(cuda_state, lambda, workers,
                        CudaReductionMode::kParallelDeterministic, &parallel,
                        &error)) {
    std::cerr << "CUDA reduction path failed: " << error << '\n';
    return 1;
  }

  double host_cost = 0.0;
  double host_trial_cost = 0.0;
  std::vector<double> host_residuals;
  std::vector<double> host_trial_residuals;
  std::vector<double> candidate_host_residuals;
  std::vector<double> candidate_host_trial_residuals;
  uint64_t near_zero = 0;
  uint64_t trial_near_zero = 0;
  uint64_t candidate_near_zero = 0;
  uint64_t candidate_trial_near_zero = 0;
  double candidate_host_cost = 0.0;
  double candidate_host_trial_cost = 0.0;
  if (!HostSourceCost(cpu_state, &host_cost, &near_zero, &error) ||
      !HostSourceCost(cpu_step.trial_state, &host_trial_cost,
                      &trial_near_zero, &error) ||
      !HostSourceCost(cuda_state, &candidate_host_cost, &candidate_near_zero,
                      &error) ||
      !HostSourceCost(parallel.step.trial_state, &candidate_host_trial_cost,
                      &candidate_trial_near_zero, &error)) {
    std::cerr << "host source cost failed: " << error << '\n';
    return 1;
  }
  if (!HostSourceResiduals(cpu_state, &host_residuals, &error) ||
      !HostSourceResiduals(cpu_step.trial_state, &host_trial_residuals,
                           &error) ||
      !HostSourceResiduals(cuda_state, &candidate_host_residuals, &error) ||
      !HostSourceResiduals(parallel.step.trial_state,
                           &candidate_host_trial_residuals, &error)) {
    std::cerr << "host source residuals failed: " << error << '\n';
    return 1;
  }

  const Difference serial_linearization =
      CompareLinearization(cpu_step.linearization, serial.step.layer_b,
                           1e-10, 1e-7);
  const Difference parallel_linearization =
      CompareLinearization(cpu_step.linearization, parallel.step.layer_b,
                           1e-10, 1e-7);
  const Difference serial_schur =
      CompareVector(cpu_step.schur, serial.step.schur, 1e-10, 1e-8);
  const Difference parallel_schur =
      CompareVector(cpu_step.schur, parallel.step.schur, 1e-10, 1e-8);
  const Difference serial_rhs =
      CompareVector(cpu_step.rhs, serial.step.rhs, 1e-10, 1e-8);
  const Difference parallel_rhs_diff =
      CompareVector(cpu_step.rhs, parallel.step.rhs, 1e-10, 1e-8);
  const Difference serial_camera_delta =
      CompareVector(cpu_step.camera_delta, serial.step.camera_delta,
                    1e-10, 1e-8);
  const Difference parallel_camera_delta =
      CompareVector(cpu_step.camera_delta, parallel.step.camera_delta,
                    1e-10, 1e-8);
  const Difference serial_point_delta =
      CompareVector(cpu_step.point_delta, serial.step.point_delta,
                    1e-10, 1e-8);
  const Difference parallel_point_delta =
      CompareVector(cpu_step.point_delta, parallel.step.point_delta,
                    1e-10, 1e-8);
  const Difference serial_parallel_schur =
      CompareVector(serial.step.schur, parallel.step.schur, 1e-10, 1e-8);
  const Difference serial_parallel_rhs =
      CompareVector(serial.step.rhs, parallel.step.rhs, 1e-10, 1e-8);
  const Difference serial_parallel_camera_delta =
      CompareVector(serial.step.camera_delta, parallel.step.camera_delta,
                    1e-10, 1e-8);
  const Difference serial_residuals = CompareVector(
      candidate_host_residuals, CudaSourceResiduals(cuda_state, serial.initial),
      1e-12, 1e-10);
  const Difference parallel_residuals = CompareVector(
      candidate_host_residuals,
      CudaSourceResiduals(cuda_state, parallel.initial),
      1e-12, 1e-10);
  const Difference serial_trial_residuals = CompareVector(
      candidate_host_trial_residuals,
      CudaSourceResiduals(parallel.step.trial_state, serial.trial),
      1e-12, 1e-10);
  const Difference parallel_trial_residuals = CompareVector(
      candidate_host_trial_residuals,
      CudaSourceResiduals(parallel.step.trial_state, parallel.trial),
      1e-12, 1e-10);
  const Difference trajectory_initial_residuals =
      CompareVector(host_residuals, candidate_host_residuals, 1e-12, 1e-10);
  const Difference trajectory_trial_residuals =
      CompareVector(host_trial_residuals, candidate_host_trial_residuals,
                    1e-12, 1e-10);
  const Difference trajectory_trial_state = CompareVector(
      SnapshotStateValues(cpu_step.trial_state),
      SnapshotStateValues(parallel.step.trial_state), 1e-10, 1e-8);
  const Difference serial_parallel_trial_state = CompareVector(
      SnapshotStateValues(serial.step.trial_state),
      SnapshotStateValues(parallel.step.trial_state), 1e-10, 1e-8);

  if (parallel.step.pose_dimension == 0 ||
      parallel.step.schur.size() !=
          parallel.step.pose_dimension * parallel.step.pose_dimension ||
      parallel.step.rhs.size() != parallel.step.pose_dimension ||
      parallel.step.camera_delta.size() != parallel.step.pose_dimension ||
      parallel.step.factor.size() !=
          parallel.step.pose_dimension * parallel.step.pose_dimension) {
    std::cerr << "diagnostic matrix/vector bounds validation failed\n";
    return 1;
  }
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
      parallel_s(parallel.step.schur.data(), parallel.step.pose_dimension,
                 parallel.step.pose_dimension);
  Eigen::VectorXd parallel_rhs_vector = Eigen::Map<const Eigen::VectorXd>(
      parallel.step.rhs.data(), parallel.step.rhs.size());
  Eigen::LLT<Eigen::MatrixXd> eigen_llt(parallel_s);
  Eigen::VectorXd eigen_delta = eigen_llt.solve(parallel_rhs_vector);
  std::vector<double> eigen_delta_vector(eigen_delta.data(),
                                         eigen_delta.data() + eigen_delta.size());
  const Difference eigen_vs_cusolver =
      CompareVector(eigen_delta_vector, parallel.step.camera_delta,
                    1e-10, 1e-8);
  std::vector<double> eigen_factor;
  std::vector<double> cuda_factor;
  const Eigen::MatrixXd eigen_lower = eigen_llt.matrixL();
  for (size_t row = 0; row < static_cast<size_t>(eigen_lower.rows()); ++row) {
    for (size_t col = 0;
         col <= row && col < static_cast<size_t>(eigen_lower.cols()); ++col) {
      const size_t index =
          col * static_cast<size_t>(eigen_lower.rows()) + row;
      eigen_factor.push_back(eigen_lower(row, col));
      cuda_factor.push_back(index < parallel.step.factor.size()
                                ? parallel.step.factor[index]
                                : std::numeric_limits<double>::quiet_NaN());
    }
  }
  const Difference eigen_factor_vs_cusolver =
      CompareVector(eigen_factor, cuda_factor, 1e-10, 1e-8);

  const Difference* all_differences[] = {
      &serial_linearization, &parallel_linearization, &serial_schur,
      &parallel_schur, &serial_rhs, &parallel_rhs_diff,
      &serial_camera_delta, &parallel_camera_delta, &serial_point_delta,
      &parallel_point_delta, &serial_parallel_schur, &serial_parallel_rhs,
      &serial_parallel_camera_delta, &serial_residuals, &parallel_residuals,
      &serial_trial_residuals, &parallel_trial_residuals,
      &trajectory_initial_residuals, &trajectory_trial_residuals,
      &trajectory_trial_state, &serial_parallel_trial_state,
      &eigen_vs_cusolver, &eigen_factor_vs_cusolver};
  bool structural_pass = true;
  bool finite_pass = true;
  bool numeric_pass = true;
  for (const Difference* difference : all_differences) {
    structural_pass = structural_pass && difference->size_match &&
                      difference->topology_match;
    finite_pass = finite_pass && difference->finite_match;
    numeric_pass = numeric_pass && difference->numeric_match;
  }

  std::ofstream output(argv[4]);
  if (!output) return 1;
  output << std::setprecision(17)
         << "{\n  \"phase\":\"" << kPhaseName << "\",\n"
         << "  \"diagnostic\":\"global120-layered-reduction-audit\",\n"
         << "  \"state_mode\":" << JsonString(state_mode) << ",\n"
         << "  \"structural_pass\":"
         << (structural_pass ? "true" : "false") << ",\n"
         << "  \"finite_pass\":" << (finite_pass ? "true" : "false")
         << ",\n  \"numeric_pass\":"
         << (numeric_pass ? "true" : "false") << ",\n"
         << "  \"pass\":"
         << ((structural_pass && finite_pass && numeric_pass) ? "true"
                                                               : "false")
         << ",\n"
         << "  \"attribution\":\"UNRESOLVED\",\n"
         << "  \"cpu_state\":" << JsonString(argv[1]) << ",\n"
         << "  \"cuda_state\":" << JsonString(argv[2]) << ",\n"
         << "  \"lambda\":" << NumberOrNull(lambda) << ",\n"
         << "  \"requested_parallel_workers\":" << workers << ",\n"
         << "  \"numeric_tolerances\":{\"residual\":{\"atol\":1e-12,\"rtol\":1e-10},"
            "\"linearization\":{\"atol\":1e-10,\"rtol\":1e-7},"
            "\"linear_solve\":{\"atol\":1e-10,\"rtol\":1e-8}},\n"
         << "  \"near_zero_lidar_samples\":" << near_zero << ",\n"
         << "  \"trial_near_zero_lidar_samples\":" << trial_near_zero << ",\n"
         << "  \"candidate_near_zero_lidar_samples\":"
         << candidate_near_zero << ",\n"
         << "  \"candidate_trial_near_zero_lidar_samples\":"
         << candidate_trial_near_zero << ",\n"
         << "  \"host_cost_arithmetic\":\"double_source_order\",\n"
         << "  \"host_source_order_cost\":" << NumberOrNull(host_cost) << ",\n"
         << "  \"host_source_order_trial_cost\":"
         << NumberOrNull(host_trial_cost) << ",\n"
         << "  \"candidate_host_source_order_cost\":"
         << NumberOrNull(candidate_host_cost) << ",\n"
         << "  \"candidate_host_source_order_trial_cost\":"
         << NumberOrNull(candidate_host_trial_cost) << ",\n"
         << "  \"reductions\":{\"serial\":";
  WriteReductionRun(output, serial);
  output << ",\"parallel\":";
  WriteReductionRun(output, parallel);
  output << "},\n  \"cost_comparison\":{\n"
         << "    \"serial_initial_abs\":"
         << NumberOrNull(std::abs(host_cost - serial.initial.cost)) << ",\n"
         << "    \"parallel_initial_abs\":"
         << NumberOrNull(std::abs(host_cost - parallel.initial.cost)) << ",\n"
         << "    \"serial_trial_layer_c_abs\":"
         << NumberOrNull(std::abs(host_trial_cost - serial.step.trial_cost)) << ",\n"
         << "    \"parallel_trial_layer_c_abs\":"
         << NumberOrNull(std::abs(host_trial_cost - parallel.step.trial_cost)) << ",\n"
         << "    \"serial_trial_layer_b_abs\":"
         << NumberOrNull(std::abs(candidate_host_trial_cost - serial.trial.cost)) << ",\n"
         << "    \"parallel_trial_layer_b_abs\":"
         << NumberOrNull(std::abs(candidate_host_trial_cost - parallel.trial.cost)) << ",\n"
         << "    \"candidate_host_vs_reference_initial_abs\":"
         << NumberOrNull(std::abs(host_cost - candidate_host_cost)) << ",\n"
         << "    \"candidate_host_vs_reference_trial_abs\":"
         << NumberOrNull(std::abs(host_trial_cost - candidate_host_trial_cost)) << "\n  },\n"
         << "  \"same_state_layers\":{\n"
         << "    \"serial_linearization\":"; WriteDifference(output, serial_linearization);
  output << ",\n    \"parallel_linearization\":"; WriteDifference(output, parallel_linearization);
  output << ",\n    \"serial_schur\":"; WriteDifference(output, serial_schur);
  output << ",\n    \"parallel_schur\":"; WriteDifference(output, parallel_schur);
  output << ",\n    \"serial_rhs\":"; WriteDifference(output, serial_rhs);
  output << ",\n    \"parallel_rhs\":"; WriteDifference(output, parallel_rhs_diff);
  output << ",\n    \"serial_camera_delta\":"; WriteDifference(output, serial_camera_delta);
  output << ",\n    \"parallel_camera_delta\":"; WriteDifference(output, parallel_camera_delta);
  output << ",\n    \"serial_point_delta\":"; WriteDifference(output, serial_point_delta);
  output << ",\n    \"parallel_point_delta\":"; WriteDifference(output, parallel_point_delta);
  output << ",\n    \"serial_source_residuals\":"; WriteDifference(output, serial_residuals);
  output << ",\n    \"parallel_source_residuals\":"; WriteDifference(output, parallel_residuals);
  output << ",\n    \"serial_trial_source_residuals\":"; WriteDifference(output, serial_trial_residuals);
  output << ",\n    \"parallel_trial_source_residuals\":"; WriteDifference(output, parallel_trial_residuals);
  output << ",\n    \"serial_parallel_schur\":"; WriteDifference(output, serial_parallel_schur);
  output << ",\n    \"serial_parallel_rhs\":"; WriteDifference(output, serial_parallel_rhs);
  output << ",\n    \"serial_parallel_camera_delta\":"; WriteDifference(output, serial_parallel_camera_delta);
  output << ",\n    \"trajectory_initial_residuals\":"; WriteDifference(output, trajectory_initial_residuals);
  output << ",\n    \"trajectory_trial_residuals\":"; WriteDifference(output, trajectory_trial_residuals);
  output << ",\n    \"trajectory_trial_state\":"; WriteDifference(output, trajectory_trial_state);
  output << ",\n    \"serial_parallel_trial_state\":"; WriteDifference(output, serial_parallel_trial_state);
  output << "\n  },\n  \"same_cuda_schur_factorization\":{\n"
         << "    \"eigen_llt_info\":" << static_cast<int>(eigen_llt.info())
         << ",\n    \"eigen_delta_vs_cusolver\":";
  WriteDifference(output, eigen_vs_cusolver);
  output << ",\n    \"eigen_factor_vs_cusolver\":";
  WriteDifference(output, eigen_factor_vs_cusolver);
  output << ",\n    \"cuda_potrf_status\":" << parallel.step.runtime.cusolver_potrf_status
         << ",\n    \"cuda_potrs_status\":" << parallel.step.runtime.cusolver_potrs_status
         << ",\n    \"cuda_dev_info\":" << parallel.step.runtime.cusolver_dev_info
         << "\n  },\n  \"trial_state\":{\n"
         << "    \"cpu_trial_cost\":" << NumberOrNull(cpu_step.trial_cost)
         << ",\n    \"serial_trial_cost\":" << NumberOrNull(serial.step.trial_cost)
         << ",\n    \"parallel_trial_cost\":" << NumberOrNull(parallel.step.trial_cost)
         << ",\n    \"candidate_host_trial_cost\":"
         << NumberOrNull(candidate_host_trial_cost)
         << ",\n    \"cpu_backward_error\":" << NumberOrNull(cpu_step.backward_error)
         << ",\n    \"serial_backward_error\":" << NumberOrNull(serial.step.backward_error)
         << ",\n    \"parallel_backward_error\":" << NumberOrNull(parallel.step.backward_error)
         << "\n  }\n}\n";
  return 0;
}
