#include "gpu_ba/fixed_linearization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <ceres/local_parameterization.h>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

constexpr double kLinearAtol = 1e-12;
constexpr double kLinearRtol = 1e-10;
// B/C/E and gradient are sums/products of already-gated analytic Jacobians.
// Their gate is deliberately still 100x tighter in relative terms than the
// Jacobian-vs-AutoDiff gate, while allowing cancellation around zero.
constexpr double kAssemblyAtol = 1e-9;
constexpr double kAssemblyRtol = 1e-9;
constexpr double kCostAtol = 1e-12;
constexpr double kCostRtol = 1e-10;

enum class EvaluationSource { kAutoDiff, kAnalytic };

struct PoseVariable {
  uint32_t image_id = 0;
  size_t offset = 0;
  std::vector<int> free_translation_indices;

  size_t Dimension() const {
    return 3 + free_translation_indices.size();
  }
};

struct PointVariable {
  uint64_t point3D_id = 0;
  size_t offset = 0;
};

struct CanonicalLayout {
  std::vector<PoseVariable> poses;
  std::vector<PointVariable> points;
  std::unordered_map<uint32_t, size_t> pose_index;
  std::unordered_map<uint64_t, size_t> point_index;
  size_t pose_dimension = 0;
  size_t point_dimension = 0;
  std::string sha256;
};

struct SnapshotLookup {
  std::unordered_map<uint32_t, const CameraSnapshot*> cameras;
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  std::vector<const ObservationSnapshot*> observations_by_source;
  std::vector<const LidarSnapshot*> lidar_by_source;
};

struct EdgeBlock {
  size_t pose_index = 0;
  size_t point_index = 0;
  Eigen::MatrixXd value;
};

struct BlockSystem {
  double cost = 0.0;
  std::vector<Eigen::MatrixXd> pose_hessian;
  std::vector<Eigen::VectorXd> pose_gradient;
  std::vector<Eigen::MatrixXd> point_hessian;
  std::vector<Eigen::VectorXd> point_gradient;
  std::vector<EdgeBlock> edges;
  std::unordered_map<uint64_t, size_t> edge_lookup;
};

struct FrozenResiduals {
  std::vector<std::array<double, 2>> visual;
  std::vector<double> lidar;
  std::vector<uint8_t> has_visual;
  std::vector<uint8_t> has_lidar;
};

struct DampingDiagonal {
  Eigen::VectorXd camera;
  Eigen::VectorXd point;
};

struct SolveData {
  bool point_factorization_success = false;
  bool schur_factorization_success = false;
  double min_point_cholesky_diagonal = 0.0;
  double min_schur_cholesky_diagonal = 0.0;
  double condition_estimate = 0.0;
  double symmetry_error = 0.0;
  double backward_error = 0.0;
  double predicted_reduction = 0.0;
  std::vector<Eigen::MatrixXd> damped_point_hessian;
  std::vector<Eigen::MatrixXd> point_inverse;
  Eigen::MatrixXd schur;
  Eigen::VectorXd rhs;
  Eigen::VectorXd camera_delta;
  Eigen::VectorXd point_delta;
};

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

class FixedAccumulator {
 public:
  FixedAccumulator(std::string name, double atol, double rtol)
      : name_(std::move(name)), atol_(atol), rtol_(rtol) {}

  void Add(double reference, double candidate, const std::string& id) {
    ++count_;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      ++failures_;
      ++nonfinite_;
      if (worst_id_.empty()) {
        worst_id_ = id;
        worst_reference_ = reference;
        worst_candidate_ = candidate;
        worst_gate_ratio_ = std::numeric_limits<double>::max();
      }
      return;
    }
    const double absolute = std::abs(reference - candidate);
    const double scale = std::max(std::abs(reference), std::abs(candidate));
    const double relative = scale == 0.0 ? 0.0 : absolute / scale;
    const double threshold = atol_ + rtol_ * scale;
    const double gate_ratio = threshold == 0.0
                                  ? (absolute == 0.0
                                         ? 0.0
                                         : std::numeric_limits<double>::max())
                                  : absolute / threshold;
    absolute_errors_.push_back(absolute);
    relative_errors_.push_back(relative);
    absolute_sum_ += absolute;
    absolute_squared_sum_ += absolute * absolute;
    relative_sum_ += relative;
    relative_squared_sum_ += relative * relative;
    if (absolute > threshold) ++failures_;
    if (worst_id_.empty() || gate_ratio > worst_gate_ratio_) {
      worst_gate_ratio_ = gate_ratio;
      worst_id_ = id;
      worst_reference_ = reference;
      worst_candidate_ = candidate;
    }
  }

  ErrorSummary Finish() const {
    ErrorSummary summary;
    summary.name = name_;
    summary.atol = atol_;
    summary.rtol = rtol_;
    summary.count = count_;
    summary.failures = failures_;
    summary.nonfinite = nonfinite_;
    if (!absolute_errors_.empty()) {
      summary.max_absolute =
          *std::max_element(absolute_errors_.begin(), absolute_errors_.end());
      summary.mean_absolute =
          static_cast<double>(absolute_sum_ / absolute_errors_.size());
      summary.rms_absolute = std::sqrt(static_cast<double>(
          absolute_squared_sum_ / absolute_errors_.size()));
      summary.median_absolute = Percentile(absolute_errors_, 0.5);
      summary.p95_absolute = Percentile(absolute_errors_, 0.95);
      summary.max_relative =
          *std::max_element(relative_errors_.begin(), relative_errors_.end());
      summary.mean_relative =
          static_cast<double>(relative_sum_ / relative_errors_.size());
      summary.rms_relative = std::sqrt(static_cast<double>(
          relative_squared_sum_ / relative_errors_.size()));
      summary.median_relative = Percentile(relative_errors_, 0.5);
      summary.p95_relative = Percentile(relative_errors_, 0.95);
    }
    summary.worst_gate_ratio = worst_gate_ratio_;
    summary.worst_element_id = worst_id_;
    summary.worst_reference = worst_reference_;
    summary.worst_candidate = worst_candidate_;
    summary.pass = failures_ == 0 && nonfinite_ == 0;
    return summary;
  }

 private:
  std::string name_;
  double atol_ = 0.0;
  double rtol_ = 0.0;
  uint64_t count_ = 0;
  uint64_t failures_ = 0;
  uint64_t nonfinite_ = 0;
  long double absolute_sum_ = 0.0;
  long double absolute_squared_sum_ = 0.0;
  long double relative_sum_ = 0.0;
  long double relative_squared_sum_ = 0.0;
  std::vector<double> absolute_errors_;
  std::vector<double> relative_errors_;
  double worst_gate_ratio_ = 0.0;
  std::string worst_id_;
  double worst_reference_ = 0.0;
  double worst_candidate_ = 0.0;
};

std::string EscapeJson(const std::string& value) {
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
      case '\\': escaped << "\\\\"; break;
      case '"': escaped << "\\\""; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default: escaped << character; break;
    }
  }
  return escaped.str();
}

void AppendJsonNumber(std::ostringstream* stream, double value) {
  if (std::isfinite(value)) {
    *stream << value;
  } else {
    *stream << "null";
  }
}

bool AllFinite(const Eigen::MatrixXd& matrix) {
  return matrix.array().isFinite().all();
}

bool BuildLookup(const Snapshot& snapshot,
                 SnapshotLookup* lookup,
                 std::string* error) {
  const size_t residual_count = snapshot.source_insertion_order.size();
  lookup->observations_by_source.assign(residual_count, nullptr);
  lookup->lidar_by_source.assign(residual_count, nullptr);
  for (const CameraSnapshot& camera : snapshot.cameras) {
    if (!lookup->cameras.emplace(camera.camera_id, &camera).second) {
      *error = "Duplicate camera ID in snapshot";
      return false;
    }
  }
  for (const ImageSnapshot& image : snapshot.images) {
    if (!lookup->images.emplace(image.image_id, &image).second) {
      *error = "Duplicate image ID in snapshot";
      return false;
    }
  }
  for (const PointSnapshot& point : snapshot.points) {
    if (!lookup->points.emplace(point.point3D_id, &point).second) {
      *error = "Duplicate point3D ID in snapshot";
      return false;
    }
  }
  for (const ObservationSnapshot& observation : snapshot.observations) {
    if (observation.source_index >= residual_count ||
        lookup->observations_by_source[observation.source_index] != nullptr) {
      *error = "Invalid visual residual source index";
      return false;
    }
    lookup->observations_by_source[observation.source_index] = &observation;
  }
  for (const LidarSnapshot& lidar : snapshot.lidar) {
    if (lidar.source_index >= residual_count ||
        lookup->lidar_by_source[lidar.source_index] != nullptr) {
      *error = "Invalid LiDAR residual source index";
      return false;
    }
    lookup->lidar_by_source[lidar.source_index] = &lidar;
  }
  return true;
}

bool BuildLayout(const Snapshot& snapshot,
                 CanonicalLayout* layout,
                 std::string* error) {
  if (snapshot.metadata.refine_focal_length ||
      snapshot.metadata.refine_principal_point ||
      snapshot.metadata.refine_extra_params) {
    *error = "Custom fixed linearization requires fixed intrinsics";
    return false;
  }
  for (const CameraSnapshot& camera : snapshot.cameras) {
    if (camera.model_id != OpenCVCameraModel::kModelId ||
        camera.params.size() != OpenCVCameraModel::kNumParams) {
      *error = "Custom fixed linearization only supports OPENCV cameras";
      return false;
    }
    if (!camera.constant) {
      *error = "Custom fixed linearization does not support variable intrinsics";
      return false;
    }
  }
  if (snapshot.metadata.lidar_residual_mode != "legacy_exact") {
    *error = "Phase-4 strict parity requires legacy_exact LiDAR residual";
    return false;
  }
  if (snapshot.metadata.loss_function != "trivial" &&
      snapshot.metadata.loss_function != "TRIVIAL") {
    *error = "Phase-4 strict parity requires TRIVIAL loss";
    return false;
  }

  std::map<std::pair<ParameterKind, uint64_t>, const ParameterBlockSnapshot*>
      parameters;
  for (const ParameterBlockSnapshot& parameter :
       snapshot.parameter_blocks_source_order) {
    parameters.emplace(std::make_pair(parameter.kind, parameter.entity_id),
                       &parameter);
  }

  std::vector<const ImageSnapshot*> variable_images;
  for (const ImageSnapshot& image : snapshot.images) {
    if (!image.pose_constant) variable_images.push_back(&image);
  }
  std::sort(variable_images.begin(), variable_images.end(),
            [](const ImageSnapshot* lhs, const ImageSnapshot* rhs) {
              return lhs->image_id < rhs->image_id;
            });
  std::ostringstream layout_text;
  layout_text << "canonical-layout-v1\n";
  for (const ImageSnapshot* image : variable_images) {
    if (!image->has_pose_parameter_blocks) {
      *error = "Variable pose is missing recorded parameter blocks: image=" +
               std::to_string(image->image_id);
      return false;
    }
    const auto q_it = parameters.find(
        std::make_pair(ParameterKind::kQuaternion, image->image_id));
    const auto t_it = parameters.find(
        std::make_pair(ParameterKind::kTranslation, image->image_id));
    if (q_it == parameters.end() || t_it == parameters.end()) {
      *error = "Variable pose parameter identity is absent: image=" +
               std::to_string(image->image_id);
      return false;
    }
    PoseVariable pose;
    pose.image_id = image->image_id;
    pose.offset = layout->pose_dimension;
    for (int index = 0; index < 3; ++index) {
      if ((image->constant_tvec_mask & (1u << index)) == 0) {
        pose.free_translation_indices.push_back(index);
      }
    }
    if (q_it->second->constant || q_it->second->ambient_size != 4 ||
        q_it->second->tangent_size != 3 || t_it->second->constant ||
        t_it->second->ambient_size != 3 ||
        t_it->second->tangent_size !=
            pose.free_translation_indices.size()) {
      *error = "Recorded pose tangent state disagrees with snapshot flags: image=" +
               std::to_string(image->image_id);
      return false;
    }
    layout->pose_index.emplace(pose.image_id, layout->poses.size());
    layout->pose_dimension += pose.Dimension();
    layout_text << "pose " << pose.image_id << " offset " << pose.offset
                << " dim " << pose.Dimension() << " free_t";
    for (const int index : pose.free_translation_indices) {
      layout_text << ' ' << index;
    }
    layout_text << '\n';
    layout->poses.push_back(std::move(pose));
  }

  std::vector<const PointSnapshot*> variable_points;
  for (const PointSnapshot& point : snapshot.points) {
    if (!point.constant) variable_points.push_back(&point);
  }
  std::sort(variable_points.begin(), variable_points.end(),
            [](const PointSnapshot* lhs, const PointSnapshot* rhs) {
              return lhs->point3D_id < rhs->point3D_id;
            });
  for (const PointSnapshot* point : variable_points) {
    const auto parameter_it = parameters.find(
        std::make_pair(ParameterKind::kPoint3D, point->point3D_id));
    if (parameter_it == parameters.end() || parameter_it->second->constant ||
        parameter_it->second->ambient_size != 3 ||
        parameter_it->second->tangent_size != 3) {
      *error = "Recorded point tangent state disagrees with snapshot flags: point=" +
               std::to_string(point->point3D_id);
      return false;
    }
    PointVariable variable;
    variable.point3D_id = point->point3D_id;
    variable.offset = layout->point_dimension;
    layout->point_index.emplace(variable.point3D_id, layout->points.size());
    layout->point_dimension += 3;
    layout_text << "point " << variable.point3D_id << " offset "
                << variable.offset << " dim 3\n";
    layout->points.push_back(variable);
  }
  layout->sha256 = Sha256Hex(layout_text.str());
  return true;
}

bool EvaluateVisual(const SnapshotLookup& lookup,
                    const ObservationSnapshot& observation,
                    EvaluationSource source,
                    VisualEvaluation* evaluation,
                    std::string* error) {
  const auto image_it = lookup.images.find(observation.image_id);
  const auto point_it = lookup.points.find(observation.point3D_id);
  if (image_it == lookup.images.end() || point_it == lookup.points.end()) {
    *error = "Visual residual references missing image or point";
    return false;
  }
  const ImageSnapshot& image = *image_it->second;
  const auto camera_it = lookup.cameras.find(image.camera_id);
  if (camera_it == lookup.cameras.end()) {
    *error = "Visual residual references missing camera";
    return false;
  }
  const CameraSnapshot& camera_snapshot = *camera_it->second;
  std::array<double, 8> camera;
  std::copy(camera_snapshot.params.begin(), camera_snapshot.params.end(),
            camera.begin());
  const PointSnapshot& point = *point_it->second;
  if (source == EvaluationSource::kAnalytic) {
    if (!EvaluateOpenCVVisual(image.qvec, image.tvec, point.xyz, camera,
                              observation.xy, evaluation)) {
      *error = "Analytic visual evaluation is non-finite: source=" +
               std::to_string(observation.source_index);
      return false;
    }
    return true;
  }

  std::unique_ptr<ceres::CostFunction> cost(
      BundleAdjustmentCostFunction<OpenCVCameraModel>::Create(
          Eigen::Vector2d(observation.xy[0], observation.xy[1])));
  const double* parameters[] = {image.qvec.data(), image.tvec.data(),
                                point.xyz.data(), camera.data()};
  double* jacobians[] = {evaluation->ambient_quaternion_jacobian.data(),
                         evaluation->translation_jacobian.data(),
                         evaluation->point_jacobian.data(),
                         evaluation->camera_jacobian.data()};
  if (!cost->Evaluate(parameters, evaluation->residual.data(), jacobians)) {
    *error = "Ceres visual evaluation failed: source=" +
             std::to_string(observation.source_index);
    return false;
  }
  ceres::QuaternionParameterization parameterization;
  if (!parameterization.ComputeJacobian(image.qvec.data(),
                                        evaluation->plus_jacobian.data())) {
    *error = "Ceres PlusJacobian failed";
    return false;
  }
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      double value = 0.0;
      for (size_t ambient = 0; ambient < 4; ++ambient) {
        value += evaluation->ambient_quaternion_jacobian[row * 4 + ambient] *
                 evaluation->plus_jacobian[ambient * 3 + col];
      }
      evaluation->local_rotation_jacobian[row * 3 + col] = value;
    }
  }
  evaluation->finite = true;
  for (const double value : evaluation->residual) {
    evaluation->finite = evaluation->finite && std::isfinite(value);
  }
  for (const double value : evaluation->local_rotation_jacobian) {
    evaluation->finite = evaluation->finite && std::isfinite(value);
  }
  for (const double value : evaluation->translation_jacobian) {
    evaluation->finite = evaluation->finite && std::isfinite(value);
  }
  for (const double value : evaluation->point_jacobian) {
    evaluation->finite = evaluation->finite && std::isfinite(value);
  }
  if (!evaluation->finite) {
    *error = "Ceres visual evaluation is non-finite: source=" +
             std::to_string(observation.source_index);
    return false;
  }
  return true;
}

bool EvaluateLidarResidual(const SnapshotLookup& lookup,
                           const LidarSnapshot& lidar,
                           EvaluationSource source,
                           double* residual,
                           std::array<double, 3>* jacobian,
                           std::string* error) {
  const auto point_it = lookup.points.find(lidar.point3D_id);
  if (point_it == lookup.points.end()) {
    *error = "LiDAR residual references missing point";
    return false;
  }
  const PointSnapshot& point = *point_it->second;
  if (source == EvaluationSource::kAnalytic) {
    const LidarEvaluation evaluation = EvaluateLidar(
        point.xyz, lidar.plane, lidar.weight,
        LidarResidualMode::kLegacyExact);
    *residual = evaluation.residual;
    *jacobian = evaluation.point_jacobian;
    if (!evaluation.finite) {
      *error = "legacy_exact LiDAR derivative is non-finite at point=" +
               std::to_string(lidar.point3D_id);
      return false;
    }
    return true;
  }
  Eigen::Matrix<double, 4, 1> plane;
  for (size_t i = 0; i < 4; ++i) plane[i] = lidar.plane[i];
  std::unique_ptr<ceres::CostFunction> cost(
      BundleAdjustmentLidarCostFunction::Create(plane, lidar.weight));
  const double* parameters[] = {point.xyz.data()};
  double* jacobians[] = {jacobian->data()};
  if (!cost->Evaluate(parameters, residual, jacobians)) {
    *error = "Ceres LiDAR evaluation failed";
    return false;
  }
  if (!std::isfinite(*residual) || !std::isfinite((*jacobian)[0]) ||
      !std::isfinite((*jacobian)[1]) || !std::isfinite((*jacobian)[2])) {
    *error = "Ceres legacy_exact LiDAR derivative is non-finite at point=" +
             std::to_string(lidar.point3D_id);
    return false;
  }
  return true;
}

void InitializeSystem(const CanonicalLayout& layout, BlockSystem* system) {
  for (const PoseVariable& pose : layout.poses) {
    system->pose_hessian.push_back(
        Eigen::MatrixXd::Zero(pose.Dimension(), pose.Dimension()));
    system->pose_gradient.push_back(
        Eigen::VectorXd::Zero(pose.Dimension()));
  }
  for (size_t i = 0; i < layout.points.size(); ++i) {
    system->point_hessian.push_back(Eigen::MatrixXd::Zero(3, 3));
    system->point_gradient.push_back(Eigen::VectorXd::Zero(3));
  }
  system->edge_lookup.reserve(layout.points.size() * 2);
}

EdgeBlock* GetEdge(size_t pose_index,
                   size_t point_index,
                   size_t pose_dimension,
                   BlockSystem* system) {
  const uint64_t key = (static_cast<uint64_t>(pose_index) << 32) |
                       static_cast<uint64_t>(point_index);
  const auto existing = system->edge_lookup.find(key);
  if (existing != system->edge_lookup.end()) {
    return &system->edges[existing->second];
  }
  EdgeBlock edge;
  edge.pose_index = pose_index;
  edge.point_index = point_index;
  edge.value = Eigen::MatrixXd::Zero(pose_dimension, 3);
  const size_t index = system->edges.size();
  system->edges.push_back(std::move(edge));
  system->edge_lookup.emplace(key, index);
  return &system->edges.back();
}

void AccumulateVisual(const CanonicalLayout& layout,
                      const ObservationSnapshot& observation,
                      const VisualEvaluation& evaluation,
                      BlockSystem* system) {
  const auto pose_it = layout.pose_index.find(observation.image_id);
  const auto point_it = layout.point_index.find(observation.point3D_id);
  const bool variable_pose = pose_it != layout.pose_index.end();
  const bool variable_point = point_it != layout.point_index.end();
  Eigen::MatrixXd pose_jacobian;
  if (variable_pose) {
    const PoseVariable& pose = layout.poses[pose_it->second];
    pose_jacobian = Eigen::MatrixXd::Zero(2, pose.Dimension());
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        pose_jacobian(row, col) =
            evaluation.local_rotation_jacobian[row * 3 + col];
      }
      for (size_t free_index = 0;
           free_index < pose.free_translation_indices.size(); ++free_index) {
        pose_jacobian(row, 3 + free_index) =
            evaluation.translation_jacobian[
                row * 3 + pose.free_translation_indices[free_index]];
      }
    }
  }
  Eigen::Matrix<double, 2, 3> point_jacobian;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      point_jacobian(row, col) =
          evaluation.point_jacobian[row * 3 + col];
    }
  }

  for (size_t row = 0; row < 2; ++row) {
    const double residual = evaluation.residual[row];
    system->cost += 0.5 * residual * residual;
    if (variable_pose) {
      Eigen::MatrixXd& hessian = system->pose_hessian[pose_it->second];
      Eigen::VectorXd& gradient = system->pose_gradient[pose_it->second];
      for (Eigen::Index i = 0; i < pose_jacobian.cols(); ++i) {
        gradient[i] += pose_jacobian(row, i) * residual;
        for (Eigen::Index j = 0; j < pose_jacobian.cols(); ++j) {
          hessian(i, j) += pose_jacobian(row, i) * pose_jacobian(row, j);
        }
      }
    }
    if (variable_point) {
      Eigen::MatrixXd& hessian = system->point_hessian[point_it->second];
      Eigen::VectorXd& gradient = system->point_gradient[point_it->second];
      for (size_t i = 0; i < 3; ++i) {
        gradient[i] += point_jacobian(row, i) * residual;
        for (size_t j = 0; j < 3; ++j) {
          hessian(i, j) += point_jacobian(row, i) * point_jacobian(row, j);
        }
      }
    }
    if (variable_pose && variable_point) {
      const PoseVariable& pose = layout.poses[pose_it->second];
      EdgeBlock* edge = GetEdge(pose_it->second, point_it->second,
                                pose.Dimension(), system);
      for (Eigen::Index i = 0; i < pose_jacobian.cols(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
          edge->value(i, j) +=
              pose_jacobian(row, i) * point_jacobian(row, j);
        }
      }
    }
  }
}

void AccumulateLidar(const CanonicalLayout& layout,
                     const LidarSnapshot& lidar,
                     double residual,
                     const std::array<double, 3>& jacobian,
                     BlockSystem* system) {
  system->cost += 0.5 * residual * residual;
  const auto point_it = layout.point_index.find(lidar.point3D_id);
  if (point_it == layout.point_index.end()) return;
  Eigen::MatrixXd& hessian = system->point_hessian[point_it->second];
  Eigen::VectorXd& gradient = system->point_gradient[point_it->second];
  for (size_t i = 0; i < 3; ++i) {
    gradient[i] += jacobian[i] * residual;
    for (size_t j = 0; j < 3; ++j) {
      hessian(i, j) += jacobian[i] * jacobian[j];
    }
  }
}

bool AssembleSystem(const Snapshot& snapshot,
                    const SnapshotLookup& lookup,
                    const CanonicalLayout& layout,
                    EvaluationSource source,
                    const FrozenResiduals* frozen_input,
                    FrozenResiduals* frozen_output,
                    BlockSystem* system,
                    std::string* error) {
  InitializeSystem(layout, system);
  if (frozen_output != nullptr) {
    const size_t count = snapshot.source_insertion_order.size();
    frozen_output->visual.resize(count);
    frozen_output->lidar.assign(count, 0.0);
    frozen_output->has_visual.assign(count, 0);
    frozen_output->has_lidar.assign(count, 0);
  }
  for (const OrderEntrySnapshot& entry : snapshot.canonical_order) {
    if (entry.source_index >= lookup.observations_by_source.size()) {
      *error = "Canonical residual source index is out of range";
      return false;
    }
    if (entry.residual_kind == ResidualKind::kVisual) {
      const ObservationSnapshot* observation =
          lookup.observations_by_source[entry.source_index];
      if (observation == nullptr) {
        *error = "Canonical visual entry has no observation";
        return false;
      }
      VisualEvaluation evaluation;
      if (!EvaluateVisual(lookup, *observation, source, &evaluation, error)) {
        return false;
      }
      if (frozen_output != nullptr) {
        frozen_output->visual[entry.source_index] = evaluation.residual;
        frozen_output->has_visual[entry.source_index] = 1;
      }
      if (frozen_input != nullptr) {
        if (entry.source_index >= frozen_input->has_visual.size() ||
            frozen_input->has_visual[entry.source_index] == 0) {
          *error = "Frozen visual residual is missing";
          return false;
        }
        evaluation.residual = frozen_input->visual[entry.source_index];
      }
      AccumulateVisual(layout, *observation, evaluation, system);
    } else {
      const LidarSnapshot* lidar = lookup.lidar_by_source[entry.source_index];
      if (lidar == nullptr) {
        *error = "Canonical LiDAR entry has no correspondence";
        return false;
      }
      double residual = 0.0;
      std::array<double, 3> jacobian;
      if (!EvaluateLidarResidual(lookup, *lidar, source, &residual, &jacobian,
                                 error)) {
        return false;
      }
      if (frozen_output != nullptr) {
        frozen_output->lidar[entry.source_index] = residual;
        frozen_output->has_lidar[entry.source_index] = 1;
      }
      if (frozen_input != nullptr) {
        if (entry.source_index >= frozen_input->has_lidar.size() ||
            frozen_input->has_lidar[entry.source_index] == 0) {
          *error = "Frozen LiDAR residual is missing";
          return false;
        }
        residual = frozen_input->lidar[entry.source_index];
      }
      AccumulateLidar(layout, *lidar, residual, jacobian, system);
    }
  }
  std::sort(system->edges.begin(), system->edges.end(),
            [](const EdgeBlock& lhs, const EdgeBlock& rhs) {
              return std::tie(lhs.pose_index, lhs.point_index) <
                     std::tie(rhs.pose_index, rhs.point_index);
            });
  system->edge_lookup.clear();
  return std::isfinite(system->cost);
}

DampingDiagonal ComputeDamping(const CanonicalLayout& layout,
                               const BlockSystem& system,
                               double diagonal_floor) {
  DampingDiagonal damping;
  damping.camera = Eigen::VectorXd::Zero(layout.pose_dimension);
  damping.point = Eigen::VectorXd::Zero(layout.point_dimension);
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      damping.camera[pose.offset + i] =
          std::max(system.pose_hessian[pose_index](i, i), diagonal_floor);
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    for (size_t i = 0; i < 3; ++i) {
      damping.point[point.offset + i] =
          std::max(system.point_hessian[point_index](i, i), diagonal_floor);
    }
  }
  return damping;
}

std::string DampingSha256(const DampingDiagonal& damping) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "damping-v1\n";
  for (Eigen::Index i = 0; i < damping.camera.size(); ++i) {
    stream << "camera " << i << ' ' << damping.camera[i] << '\n';
  }
  for (Eigen::Index i = 0; i < damping.point.size(); ++i) {
    stream << "point " << i << ' ' << damping.point[i] << '\n';
  }
  return Sha256Hex(stream.str());
}

void AddMatrix(FixedAccumulator* accumulator,
               const Eigen::MatrixXd& reference,
               const Eigen::MatrixXd& candidate,
               const std::string& prefix) {
  for (Eigen::Index row = 0; row < reference.rows(); ++row) {
    for (Eigen::Index col = 0; col < reference.cols(); ++col) {
      accumulator->Add(reference(row, col), candidate(row, col),
                       prefix + ":row=" + std::to_string(row) +
                           ":col=" + std::to_string(col));
    }
  }
}

void AddVector(FixedAccumulator* accumulator,
               const Eigen::VectorXd& reference,
               const Eigen::VectorXd& candidate,
               const std::string& prefix) {
  for (Eigen::Index i = 0; i < reference.size(); ++i) {
    accumulator->Add(reference[i], candidate[i],
                     prefix + ":element=" + std::to_string(i));
  }
}

bool CompareUndampedSystems(const CanonicalLayout& layout,
                            const BlockSystem& reference,
                            const BlockSystem& candidate,
                            const DampingDiagonal& reference_damping,
                            const DampingDiagonal& candidate_damping,
                            std::vector<ErrorSummary>* metrics,
                            std::string* error) {
  if (reference.pose_hessian.size() != candidate.pose_hessian.size() ||
      reference.point_hessian.size() != candidate.point_hessian.size() ||
      reference.edges.size() != candidate.edges.size()) {
    *error = "Reference/candidate block topology differs";
    return false;
  }
  FixedAccumulator b("canonical_B_pose_blocks", kAssemblyAtol,
                     kAssemblyRtol);
  FixedAccumulator c("canonical_C_point_blocks", kAssemblyAtol,
                     kAssemblyRtol);
  FixedAccumulator e("canonical_E_pose_point_blocks", kAssemblyAtol,
                     kAssemblyRtol);
  FixedAccumulator camera_gradient("canonical_camera_gradient",
                                   kAssemblyAtol, kAssemblyRtol);
  FixedAccumulator point_gradient("canonical_point_gradient",
                                  kAssemblyAtol, kAssemblyRtol);
  FixedAccumulator damping("diagonal_D_before_freeze", kLinearAtol,
                           kLinearRtol);
  for (size_t i = 0; i < layout.poses.size(); ++i) {
    const std::string id = "image=" +
                           std::to_string(layout.poses[i].image_id);
    AddMatrix(&b, reference.pose_hessian[i], candidate.pose_hessian[i], id);
    AddVector(&camera_gradient, reference.pose_gradient[i],
              candidate.pose_gradient[i], id);
  }
  for (size_t i = 0; i < layout.points.size(); ++i) {
    const std::string id = "point3D=" +
                           std::to_string(layout.points[i].point3D_id);
    AddMatrix(&c, reference.point_hessian[i], candidate.point_hessian[i], id);
    AddVector(&point_gradient, reference.point_gradient[i],
              candidate.point_gradient[i], id);
  }
  for (size_t i = 0; i < reference.edges.size(); ++i) {
    const EdgeBlock& reference_edge = reference.edges[i];
    const EdgeBlock& candidate_edge = candidate.edges[i];
    if (reference_edge.pose_index != candidate_edge.pose_index ||
        reference_edge.point_index != candidate_edge.point_index ||
        reference_edge.value.rows() != candidate_edge.value.rows()) {
      *error = "Reference/candidate canonical E edge identity differs";
      return false;
    }
    const std::string id =
        "image=" +
        std::to_string(layout.poses[reference_edge.pose_index].image_id) +
        ":point3D=" +
        std::to_string(layout.points[reference_edge.point_index].point3D_id);
    AddMatrix(&e, reference_edge.value, candidate_edge.value, id);
  }
  AddVector(&damping, reference_damping.camera, candidate_damping.camera,
            "camera");
  AddVector(&damping, reference_damping.point, candidate_damping.point,
            "point");
  metrics->push_back(b.Finish());
  metrics->push_back(c.Finish());
  metrics->push_back(e.Finish());
  metrics->push_back(camera_gradient.Finish());
  metrics->push_back(point_gradient.Finish());
  metrics->push_back(damping.Finish());
  return true;
}

Eigen::VectorXd ConcatenatePoseGradient(const CanonicalLayout& layout,
                                        const BlockSystem& system) {
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(layout.pose_dimension);
  for (size_t i = 0; i < layout.poses.size(); ++i) {
    const PoseVariable& pose = layout.poses[i];
    gradient.segment(pose.offset, pose.Dimension()) = system.pose_gradient[i];
  }
  return gradient;
}

Eigen::VectorXd ConcatenatePointGradient(const CanonicalLayout& layout,
                                         const BlockSystem& system) {
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(layout.point_dimension);
  for (size_t i = 0; i < layout.points.size(); ++i) {
    gradient.segment(layout.points[i].offset, 3) = system.point_gradient[i];
  }
  return gradient;
}

bool BuildAndSolveSchur(const CanonicalLayout& layout,
                        const BlockSystem& system,
                        const DampingDiagonal& damping,
                        double lambda,
                        SolveData* solve,
                        std::string* error) {
  solve->schur =
      Eigen::MatrixXd::Zero(layout.pose_dimension, layout.pose_dimension);
  solve->rhs = -ConcatenatePoseGradient(layout, system);
  solve->damped_point_hessian.resize(layout.points.size());
  solve->point_inverse.resize(layout.points.size());

  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    solve->schur.block(pose.offset, pose.offset, pose.Dimension(),
                       pose.Dimension()) = system.pose_hessian[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      solve->schur(pose.offset + i, pose.offset + i) +=
          lambda * damping.camera[pose.offset + i];
    }
  }

  solve->point_factorization_success = true;
  solve->min_point_cholesky_diagonal =
      layout.points.empty() ? 0.0 : std::numeric_limits<double>::max();
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    Eigen::MatrixXd damped = system.point_hessian[point_index];
    for (size_t i = 0; i < 3; ++i) {
      damped(i, i) +=
          lambda * damping.point[layout.points[point_index].offset + i];
    }
    solve->damped_point_hessian[point_index] = damped;
    Eigen::LLT<Eigen::MatrixXd> llt(damped);
    if (llt.info() != Eigen::Success) {
      solve->point_factorization_success = false;
      *error = "Damped point block Cholesky failed: point=" +
               std::to_string(layout.points[point_index].point3D_id);
      return false;
    }
    const double min_diagonal = llt.matrixL().toDenseMatrix()
                                    .diagonal()
                                    .minCoeff();
    solve->min_point_cholesky_diagonal =
        std::min(solve->min_point_cholesky_diagonal, min_diagonal);
    solve->point_inverse[point_index] =
        llt.solve(Eigen::MatrixXd::Identity(3, 3));
    if (!AllFinite(solve->point_inverse[point_index])) {
      *error = "Damped point inverse is non-finite";
      return false;
    }
  }

  std::vector<std::vector<size_t>> point_edges(layout.points.size());
  for (size_t edge_index = 0; edge_index < system.edges.size(); ++edge_index) {
    point_edges[system.edges[edge_index].point_index].push_back(edge_index);
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const Eigen::MatrixXd& inverse = solve->point_inverse[point_index];
    const Eigen::VectorXd inverse_gradient =
        inverse * system.point_gradient[point_index];
    const std::vector<size_t>& edges = point_edges[point_index];
    for (const size_t edge_index : edges) {
      const EdgeBlock& edge = system.edges[edge_index];
      const PoseVariable& pose = layout.poses[edge.pose_index];
      solve->rhs.segment(pose.offset, pose.Dimension()) +=
          edge.value * inverse_gradient;
    }
    for (size_t lhs_index = 0; lhs_index < edges.size(); ++lhs_index) {
      const EdgeBlock& lhs = system.edges[edges[lhs_index]];
      const PoseVariable& lhs_pose = layout.poses[lhs.pose_index];
      for (size_t rhs_index = lhs_index; rhs_index < edges.size();
           ++rhs_index) {
        const EdgeBlock& rhs = system.edges[edges[rhs_index]];
        const PoseVariable& rhs_pose = layout.poses[rhs.pose_index];
        Eigen::MatrixXd contribution =
            lhs.value * inverse * rhs.value.transpose();
        if (lhs.pose_index == rhs.pose_index) {
          contribution =
              (0.5 * (contribution + contribution.transpose())).eval();
        }
        solve->schur.block(lhs_pose.offset, rhs_pose.offset,
                           lhs_pose.Dimension(), rhs_pose.Dimension()) -=
            contribution;
        if (lhs.pose_index != rhs.pose_index) {
          solve->schur.block(rhs_pose.offset, lhs_pose.offset,
                             rhs_pose.Dimension(), lhs_pose.Dimension()) -=
              contribution.transpose();
        }
      }
    }
  }

  solve->symmetry_error = layout.pose_dimension == 0
                              ? 0.0
                              : (solve->schur - solve->schur.transpose())
                                    .cwiseAbs()
                                    .maxCoeff();
  if (!AllFinite(solve->schur) || !solve->rhs.array().isFinite().all()) {
    *error = "Schur system is non-finite";
    return false;
  }
  if (layout.pose_dimension == 0) {
    solve->schur_factorization_success = true;
    solve->min_schur_cholesky_diagonal = 0.0;
    solve->condition_estimate = 1.0;
    solve->camera_delta = Eigen::VectorXd();
  } else {
    Eigen::LLT<Eigen::MatrixXd> llt(solve->schur);
    if (llt.info() != Eigen::Success) {
      *error = "Dense Schur Cholesky failed";
      return false;
    }
    solve->schur_factorization_success = true;
    solve->min_schur_cholesky_diagonal =
        llt.matrixL().toDenseMatrix().diagonal().minCoeff();
    solve->camera_delta = llt.solve(solve->rhs);
    if (!solve->camera_delta.array().isFinite().all()) {
      *error = "Camera delta is non-finite";
      return false;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(
        solve->schur, Eigen::EigenvaluesOnly);
    if (eigen_solver.info() != Eigen::Success) {
      *error = "Schur condition estimation failed";
      return false;
    }
    const double minimum = eigen_solver.eigenvalues().minCoeff();
    const double maximum = eigen_solver.eigenvalues().maxCoeff();
    solve->condition_estimate =
        minimum > 0.0 ? maximum / minimum
                      : std::numeric_limits<double>::max();
  }

  solve->point_delta = Eigen::VectorXd::Zero(layout.point_dimension);
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    Eigen::VectorXd right = system.point_gradient[point_index];
    for (const size_t edge_index : point_edges[point_index]) {
      const EdgeBlock& edge = system.edges[edge_index];
      const PoseVariable& pose = layout.poses[edge.pose_index];
      right += edge.value.transpose() *
               solve->camera_delta.segment(pose.offset, pose.Dimension());
    }
    solve->point_delta.segment(layout.points[point_index].offset, 3) =
        -solve->point_inverse[point_index] * right;
  }

  Eigen::VectorXd camera_residual =
      ConcatenatePoseGradient(layout, system);
  Eigen::VectorXd point_residual =
      ConcatenatePointGradient(layout, system);
  std::vector<double> camera_row_sum(layout.pose_dimension, 0.0);
  std::vector<double> point_row_sum(layout.point_dimension, 0.0);
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    Eigen::MatrixXd damped = system.pose_hessian[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      damped(i, i) += lambda * damping.camera[pose.offset + i];
    }
    camera_residual.segment(pose.offset, pose.Dimension()) +=
        damped * solve->camera_delta.segment(pose.offset, pose.Dimension());
    for (size_t row = 0; row < pose.Dimension(); ++row) {
      camera_row_sum[pose.offset + row] += damped.row(row).cwiseAbs().sum();
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    point_residual.segment(point.offset, 3) +=
        solve->damped_point_hessian[point_index] *
        solve->point_delta.segment(point.offset, 3);
    for (size_t row = 0; row < 3; ++row) {
      point_row_sum[point.offset + row] +=
          solve->damped_point_hessian[point_index]
              .row(row)
              .cwiseAbs()
              .sum();
    }
  }
  for (const EdgeBlock& edge : system.edges) {
    const PoseVariable& pose = layout.poses[edge.pose_index];
    const PointVariable& point = layout.points[edge.point_index];
    camera_residual.segment(pose.offset, pose.Dimension()) +=
        edge.value * solve->point_delta.segment(point.offset, 3);
    point_residual.segment(point.offset, 3) +=
        edge.value.transpose() *
        solve->camera_delta.segment(pose.offset, pose.Dimension());
    for (size_t row = 0; row < pose.Dimension(); ++row) {
      camera_row_sum[pose.offset + row] += edge.value.row(row).cwiseAbs().sum();
    }
    for (size_t row = 0; row < 3; ++row) {
      point_row_sum[point.offset + row] +=
          edge.value.col(row).cwiseAbs().sum();
    }
  }
  const double residual_norm = std::max(
      camera_residual.size() == 0 ? 0.0
                                  : camera_residual.cwiseAbs().maxCoeff(),
      point_residual.size() == 0 ? 0.0
                                 : point_residual.cwiseAbs().maxCoeff());
  const double hessian_norm = std::max(
      camera_row_sum.empty()
          ? 0.0
          : *std::max_element(camera_row_sum.begin(), camera_row_sum.end()),
      point_row_sum.empty()
          ? 0.0
          : *std::max_element(point_row_sum.begin(), point_row_sum.end()));
  const Eigen::VectorXd camera_gradient =
      ConcatenatePoseGradient(layout, system);
  const Eigen::VectorXd point_gradient =
      ConcatenatePointGradient(layout, system);
  const double gradient_norm = std::max(
      camera_gradient.size() == 0
          ? 0.0
          : camera_gradient.cwiseAbs().maxCoeff(),
      point_gradient.size() == 0 ? 0.0
                                 : point_gradient.cwiseAbs().maxCoeff());
  const double delta_norm = std::max(
      solve->camera_delta.size() == 0
          ? 0.0
          : solve->camera_delta.cwiseAbs().maxCoeff(),
      solve->point_delta.size() == 0
          ? 0.0
          : solve->point_delta.cwiseAbs().maxCoeff());
  const double denominator = hessian_norm * delta_norm + gradient_norm;
  solve->backward_error = denominator == 0.0 ? 0.0
                                             : residual_norm / denominator;

  double damping_quadratic = 0.0;
  double gradient_dot_delta = 0.0;
  if (solve->camera_delta.size() > 0) {
    damping_quadratic +=
        (damping.camera.array() * solve->camera_delta.array().square()).sum();
    gradient_dot_delta += camera_gradient.dot(solve->camera_delta);
  }
  if (solve->point_delta.size() > 0) {
    damping_quadratic +=
        (damping.point.array() * solve->point_delta.array().square()).sum();
    gradient_dot_delta += point_gradient.dot(solve->point_delta);
  }
  solve->predicted_reduction =
      0.5 * (lambda * damping_quadratic - gradient_dot_delta);
  return std::isfinite(solve->backward_error) &&
         std::isfinite(solve->predicted_reduction);
}

bool ApplyDelta(const CanonicalLayout& layout,
                const SolveData& solve,
                EvaluationSource source,
                Snapshot* state,
                std::string* error) {
  std::unordered_map<uint32_t, ImageSnapshot*> images;
  std::unordered_map<uint64_t, PointSnapshot*> points;
  for (ImageSnapshot& image : state->images) {
    images.emplace(image.image_id, &image);
  }
  for (PointSnapshot& point : state->points) {
    points.emplace(point.point3D_id, &point);
  }
  ceres::QuaternionParameterization parameterization;
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    const auto image_it = images.find(pose.image_id);
    if (image_it == images.end()) {
      *error = "Trial update references missing image";
      return false;
    }
    ImageSnapshot& image = *image_it->second;
    std::array<double, 3> rotation_delta;
    for (size_t i = 0; i < 3; ++i) {
      rotation_delta[i] = solve.camera_delta[pose.offset + i];
    }
    std::array<double, 4> updated;
    if (source == EvaluationSource::kAutoDiff) {
      if (!parameterization.Plus(image.qvec.data(), rotation_delta.data(),
                                 updated.data())) {
        *error = "Ceres trial quaternion Plus failed";
        return false;
      }
      std::array<double, 4> normalized;
      if (!NormalizeQuaternion(updated, &normalized)) {
        *error = "Ceres trial quaternion normalization failed";
        return false;
      }
      image.qvec = normalized;
    } else {
      if (!QuaternionPlus(image.qvec, rotation_delta, &updated)) {
        *error = "Analytic trial quaternion Plus failed";
        return false;
      }
      image.qvec = updated;
    }
    for (size_t free_index = 0;
         free_index < pose.free_translation_indices.size(); ++free_index) {
      image.tvec[pose.free_translation_indices[free_index]] +=
          solve.camera_delta[pose.offset + 3 + free_index];
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& variable = layout.points[point_index];
    const auto point_it = points.find(variable.point3D_id);
    if (point_it == points.end()) {
      *error = "Trial update references missing point";
      return false;
    }
    for (size_t i = 0; i < 3; ++i) {
      point_it->second->xyz[i] += solve.point_delta[variable.offset + i];
    }
  }
  return true;
}

bool EvaluateTotalCost(const Snapshot& state,
                       EvaluationSource source,
                       double* cost,
                       std::string* error) {
  SnapshotLookup lookup;
  if (!BuildLookup(state, &lookup, error)) return false;
  *cost = 0.0;
  for (const OrderEntrySnapshot& entry : state.canonical_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      const ObservationSnapshot* observation =
          lookup.observations_by_source[entry.source_index];
      if (observation == nullptr) {
        *error = "Cost evaluation is missing a visual residual";
        return false;
      }
      if (source == EvaluationSource::kAnalytic) {
        VisualEvaluation evaluation;
        if (!EvaluateVisual(lookup, *observation, source, &evaluation, error)) {
          return false;
        }
        for (const double residual : evaluation.residual) {
          *cost += 0.5 * residual * residual;
        }
      } else {
        const ImageSnapshot& image = *lookup.images.at(observation->image_id);
        const PointSnapshot& point = *lookup.points.at(observation->point3D_id);
        const CameraSnapshot& camera_snapshot =
            *lookup.cameras.at(image.camera_id);
        std::array<double, 8> camera;
        std::copy(camera_snapshot.params.begin(), camera_snapshot.params.end(),
                  camera.begin());
        std::unique_ptr<ceres::CostFunction> ceres_cost(
            BundleAdjustmentCostFunction<OpenCVCameraModel>::Create(
                Eigen::Vector2d(observation->xy[0], observation->xy[1])));
        const double* parameters[] = {image.qvec.data(), image.tvec.data(),
                                      point.xyz.data(), camera.data()};
        std::array<double, 2> residual;
        if (!ceres_cost->Evaluate(parameters, residual.data(), nullptr)) {
          *error = "Ceres trial residual evaluation failed";
          return false;
        }
        *cost += 0.5 * (residual[0] * residual[0] +
                       residual[1] * residual[1]);
      }
    } else {
      const LidarSnapshot* lidar = lookup.lidar_by_source[entry.source_index];
      if (lidar == nullptr) {
        *error = "Cost evaluation is missing a LiDAR residual";
        return false;
      }
      const PointSnapshot& point = *lookup.points.at(lidar->point3D_id);
      double residual = 0.0;
      if (source == EvaluationSource::kAnalytic) {
        residual = EvaluateLidar(point.xyz, lidar->plane, lidar->weight,
                                 LidarResidualMode::kLegacyExact)
                       .residual;
      } else {
        Eigen::Matrix<double, 4, 1> plane;
        for (size_t i = 0; i < 4; ++i) plane[i] = lidar->plane[i];
        std::unique_ptr<ceres::CostFunction> ceres_cost(
            BundleAdjustmentLidarCostFunction::Create(plane, lidar->weight));
        const double* parameters[] = {point.xyz.data()};
        if (!ceres_cost->Evaluate(parameters, &residual, nullptr)) {
          *error = "Ceres trial LiDAR residual evaluation failed";
          return false;
        }
      }
      *cost += 0.5 * residual * residual;
    }
  }
  return std::isfinite(*cost);
}

void CompareSolveData(const CanonicalLayout& layout,
                      const SolveData& reference,
                      const SolveData& candidate,
                      std::vector<ErrorSummary>* metrics) {
  FixedAccumulator damped_points("damped_point_blocks", kLinearAtol,
                                 kLinearRtol);
  FixedAccumulator point_inverse("point_inverse_blocks", kLinearAtol,
                                 kLinearRtol);
  for (size_t i = 0; i < layout.points.size(); ++i) {
    const std::string id = "point3D=" +
                           std::to_string(layout.points[i].point3D_id);
    AddMatrix(&damped_points, reference.damped_point_hessian[i],
              candidate.damped_point_hessian[i], id);
    AddMatrix(&point_inverse, reference.point_inverse[i],
              candidate.point_inverse[i], id);
  }
  FixedAccumulator schur("canonical_schur_S", kLinearAtol, kLinearRtol);
  FixedAccumulator rhs("canonical_schur_rhs", kLinearAtol, kLinearRtol);
  FixedAccumulator camera_delta("camera_delta", kLinearAtol, kLinearRtol);
  FixedAccumulator point_delta("point_delta", kLinearAtol, kLinearRtol);
  AddMatrix(&schur, reference.schur, candidate.schur, "schur");
  AddVector(&rhs, reference.rhs, candidate.rhs, "rhs");
  AddVector(&camera_delta, reference.camera_delta, candidate.camera_delta,
            "camera");
  AddVector(&point_delta, reference.point_delta, candidate.point_delta,
            "point");
  metrics->push_back(damped_points.Finish());
  metrics->push_back(point_inverse.Finish());
  metrics->push_back(schur.Finish());
  metrics->push_back(rhs.Finish());
  metrics->push_back(camera_delta.Finish());
  metrics->push_back(point_delta.Finish());
}

void CompareUnfrozenSolveDiagnostics(
    const SolveData& reference,
    const SolveData& candidate,
    std::vector<ErrorSummary>* diagnostics) {
  FixedAccumulator rhs("unfrozen_canonical_schur_rhs", kLinearAtol,
                       kLinearRtol);
  FixedAccumulator camera_delta("unfrozen_camera_delta", kLinearAtol,
                                kLinearRtol);
  FixedAccumulator point_delta("unfrozen_point_delta", kLinearAtol,
                               kLinearRtol);
  FixedAccumulator predicted("unfrozen_predicted_reduction", kLinearAtol,
                             kLinearRtol);
  AddVector(&rhs, reference.rhs, candidate.rhs, "rhs");
  AddVector(&camera_delta, reference.camera_delta, candidate.camera_delta,
            "camera");
  AddVector(&point_delta, reference.point_delta, candidate.point_delta,
            "point");
  predicted.Add(reference.predicted_reduction, candidate.predicted_reduction,
                "predicted");
  diagnostics->push_back(rhs.Finish());
  diagnostics->push_back(camera_delta.Finish());
  diagnostics->push_back(point_delta.Finish());
  diagnostics->push_back(predicted.Finish());
}

void CountConstantResiduals(const Snapshot& snapshot,
                            const CanonicalLayout& layout,
                            FixedLinearizationResult* result) {
  for (const ObservationSnapshot& observation : snapshot.observations) {
    const bool pose_constant =
        layout.pose_index.count(observation.image_id) == 0;
    const bool point_constant =
        layout.point_index.count(observation.point3D_id) == 0;
    if (pose_constant) ++result->constant_pose_residuals;
    if (point_constant) ++result->constant_point_residuals;
    if (pose_constant && point_constant) ++result->fully_constant_residuals;
  }
  for (const LidarSnapshot& lidar : snapshot.lidar) {
    if (layout.point_index.count(lidar.point3D_id) == 0) {
      ++result->constant_point_residuals;
      ++result->fully_constant_residuals;
    }
  }
}

}  // namespace

bool RunFixedLinearizationComparison(
    const Snapshot& snapshot,
    const FixedLinearizationOptions& options,
    FixedLinearizationResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = FixedLinearizationResult();
  if (!std::isfinite(options.lambda) || options.lambda < 0.0 ||
      !std::isfinite(options.diagonal_floor) || options.diagonal_floor <= 0.0) {
    *error = "Invalid fixed-linearization lambda or diagonal floor";
    return false;
  }
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, &layout, error)) return false;
  SnapshotLookup lookup;
  if (!BuildLookup(snapshot, &lookup, error)) return false;
  result->variable_pose_blocks = layout.poses.size();
  result->variable_point_blocks = layout.points.size();
  result->pose_tangent_dimension = layout.pose_dimension;
  result->point_tangent_dimension = layout.point_dimension;
  result->layout_sha256 = layout.sha256;
  CountConstantResiduals(snapshot, layout, result);

  BlockSystem reference;
  BlockSystem candidate;
  FrozenResiduals frozen_residuals;
  if (!AssembleSystem(snapshot, lookup, layout, EvaluationSource::kAutoDiff,
                      nullptr, &frozen_residuals, &reference, error) ||
      !AssembleSystem(snapshot, lookup, layout, EvaluationSource::kAnalytic,
                      &frozen_residuals, nullptr, &candidate, error)) {
    return false;
  }
  result->pose_point_edges = reference.edges.size();
  const DampingDiagonal reference_damping =
      ComputeDamping(layout, reference, options.diagonal_floor);
  const DampingDiagonal candidate_damping =
      ComputeDamping(layout, candidate, options.diagonal_floor);
  result->damping_sha256 = DampingSha256(reference_damping);
  if (!CompareUndampedSystems(layout, reference, candidate,
                              reference_damping, candidate_damping,
                              &result->metrics, error)) {
    return false;
  }

  SolveData reference_solve;
  SolveData candidate_solve;
  SolveData unfrozen_candidate_solve;
  // The candidate deliberately receives the frozen reference D. This mode
  // isolates fixed-linearization parity from independent LM damping choices.
  if (!BuildAndSolveSchur(layout, reference, reference_damping,
                          options.lambda, &reference_solve, error)) {
    return false;
  }
  if (!BuildAndSolveSchur(layout, candidate, reference_damping,
                          options.lambda, &unfrozen_candidate_solve, error)) {
    return false;
  }
  result->unfrozen_candidate_backward_error =
      unfrozen_candidate_solve.backward_error;
  CompareUnfrozenSolveDiagnostics(
      reference_solve, unfrozen_candidate_solve,
      &result->non_gating_end_to_end_diagnostics);
  // The gradient was compared as an assembly output above. Freeze it for the
  // Schur layer so rhs/delta parity diagnoses elimination and factorization,
  // rather than re-counting cancellation in J^T r.
  candidate.pose_gradient = reference.pose_gradient;
  candidate.point_gradient = reference.point_gradient;
  if (!BuildAndSolveSchur(layout, candidate, reference_damping,
                          options.lambda, &candidate_solve, error)) {
    return false;
  }
  CompareSolveData(layout, reference_solve, candidate_solve, &result->metrics);
  result->reference_point_factorization_success =
      reference_solve.point_factorization_success;
  result->candidate_point_factorization_success =
      candidate_solve.point_factorization_success;
  result->reference_schur_factorization_success =
      reference_solve.schur_factorization_success;
  result->candidate_schur_factorization_success =
      candidate_solve.schur_factorization_success;
  result->reference_min_point_cholesky_diagonal =
      reference_solve.min_point_cholesky_diagonal;
  result->candidate_min_point_cholesky_diagonal =
      candidate_solve.min_point_cholesky_diagonal;
  result->reference_min_schur_cholesky_diagonal =
      reference_solve.min_schur_cholesky_diagonal;
  result->candidate_min_schur_cholesky_diagonal =
      candidate_solve.min_schur_cholesky_diagonal;
  result->reference_schur_condition_estimate =
      reference_solve.condition_estimate;
  result->candidate_schur_condition_estimate =
      candidate_solve.condition_estimate;
  result->reference_schur_symmetry_error = reference_solve.symmetry_error;
  result->candidate_schur_symmetry_error = candidate_solve.symmetry_error;
  result->reference_backward_error = reference_solve.backward_error;
  result->candidate_backward_error = candidate_solve.backward_error;
  result->reference_current_cost = reference.cost;
  result->candidate_current_cost = candidate.cost;
  result->reference_predicted_reduction =
      reference_solve.predicted_reduction;
  result->candidate_predicted_reduction =
      candidate_solve.predicted_reduction;

  Snapshot reference_trial = snapshot;
  Snapshot candidate_trial = snapshot;
  if (!ApplyDelta(layout, reference_solve, EvaluationSource::kAutoDiff,
                  &reference_trial, error) ||
      !ApplyDelta(layout, candidate_solve, EvaluationSource::kAnalytic,
                  &candidate_trial, error) ||
      !EvaluateTotalCost(reference_trial, EvaluationSource::kAutoDiff,
                         &result->reference_trial_cost, error) ||
      !EvaluateTotalCost(candidate_trial, EvaluationSource::kAnalytic,
                         &result->candidate_trial_cost, error)) {
    return false;
  }
  result->reference_actual_reduction =
      result->reference_current_cost - result->reference_trial_cost;
  result->candidate_actual_reduction =
      result->candidate_current_cost - result->candidate_trial_cost;
  result->reference_rho = result->reference_predicted_reduction > 0.0
                              ? result->reference_actual_reduction /
                                    result->reference_predicted_reduction
                              : 0.0;
  result->candidate_rho = result->candidate_predicted_reduction > 0.0
                              ? result->candidate_actual_reduction /
                                    result->candidate_predicted_reduction
                              : 0.0;
  result->reference_accept =
      std::isfinite(result->reference_trial_cost) &&
      std::isfinite(result->reference_predicted_reduction) &&
      std::isfinite(result->reference_rho) &&
      result->reference_predicted_reduction > 0.0 &&
      result->reference_rho > 1e-3;
  result->candidate_accept =
      std::isfinite(result->candidate_trial_cost) &&
      std::isfinite(result->candidate_predicted_reduction) &&
      std::isfinite(result->candidate_rho) &&
      result->candidate_predicted_reduction > 0.0 &&
      result->candidate_rho > 1e-3;

  FixedAccumulator current_cost("fixed_current_cost", kCostAtol, kCostRtol);
  FixedAccumulator predicted("predicted_reduction", kLinearAtol, kLinearRtol);
  FixedAccumulator trial_cost("trial_cost", kCostAtol, kCostRtol);
  FixedAccumulator backward("linear_backward_error", kLinearAtol,
                            kLinearRtol);
  current_cost.Add(result->reference_current_cost,
                   result->candidate_current_cost, "cost");
  predicted.Add(result->reference_predicted_reduction,
                result->candidate_predicted_reduction, "predicted");
  trial_cost.Add(result->reference_trial_cost, result->candidate_trial_cost,
                 "trial");
  backward.Add(result->reference_backward_error,
               result->candidate_backward_error, "backward");
  result->metrics.push_back(current_cost.Finish());
  result->metrics.push_back(predicted.Finish());
  result->metrics.push_back(trial_cost.Finish());
  result->metrics.push_back(backward.Finish());

  result->pass = result->reference_point_factorization_success &&
                 result->candidate_point_factorization_success &&
                 result->reference_schur_factorization_success &&
                 result->candidate_schur_factorization_success &&
                 result->reference_backward_error <=
                     options.linear_backward_error_limit &&
                 result->candidate_backward_error <=
                     options.linear_backward_error_limit &&
                 result->reference_schur_symmetry_error <= kLinearAtol &&
                 result->candidate_schur_symmetry_error <= kLinearAtol &&
                 result->reference_predicted_reduction > 0.0 &&
                 result->candidate_predicted_reduction > 0.0 &&
                 result->reference_accept == result->candidate_accept;
  for (const ErrorSummary& metric : result->metrics) {
    result->pass = result->pass && metric.pass;
  }
  return true;
}

std::string FixedLinearizationJson(
    const FixedLinearizationResult& result,
    const FixedLinearizationOptions& options,
    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string indent2(indent_spaces * 2, ' ');
  const std::string indent3(indent_spaces * 3, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "\"fixed_linearization\": {\n";
  stream << indent2 << "\"scope\": "
         << "\"phase_4_canonical_blocks_and_fixed_step\",\n";
  stream << indent2 << "\"lambda\": " << options.lambda
         << ", \"diagonal_floor\": " << options.diagonal_floor
         << ", \"shared_frozen_reference_residual\": true"
         << ", \"shared_frozen_reference_gradient_for_schur\": true"
         << ", \"shared_frozen_reference_D\": true,\n";
  stream << indent2 << "\"comparison_rule\": "
         << "\"abs_error <= atol + rtol * max(abs(reference), "
            "abs(candidate))\",\n";
  stream << indent2 << "\"layout\": {\"variable_pose_blocks\": "
         << result.variable_pose_blocks
         << ", \"variable_point_blocks\": "
         << result.variable_point_blocks
         << ", \"pose_tangent_dimension\": "
         << result.pose_tangent_dimension
         << ", \"point_tangent_dimension\": "
         << result.point_tangent_dimension
         << ", \"pose_point_edges\": " << result.pose_point_edges
         << ", \"constant_pose_residuals\": "
         << result.constant_pose_residuals
         << ", \"constant_point_residuals\": "
         << result.constant_point_residuals
         << ", \"fully_constant_residuals\": "
         << result.fully_constant_residuals
         << ", \"layout_sha256\": \"" << result.layout_sha256
         << "\", \"damping_sha256\": \"" << result.damping_sha256
         << "\"},\n";
  stream << indent2 << "\"factorization\": {\n";
  stream << indent3 << "\"reference_point_success\": "
         << (result.reference_point_factorization_success ? "true" : "false")
         << ", \"candidate_point_success\": "
         << (result.candidate_point_factorization_success ? "true" : "false")
         << ",\n";
  stream << indent3 << "\"reference_schur_success\": "
         << (result.reference_schur_factorization_success ? "true" : "false")
         << ", \"candidate_schur_success\": "
         << (result.candidate_schur_factorization_success ? "true" : "false")
         << ",\n";
  stream << indent3 << "\"reference_min_point_cholesky_diagonal\": "
         << result.reference_min_point_cholesky_diagonal
         << ", \"candidate_min_point_cholesky_diagonal\": "
         << result.candidate_min_point_cholesky_diagonal << ",\n";
  stream << indent3 << "\"reference_min_schur_cholesky_diagonal\": "
         << result.reference_min_schur_cholesky_diagonal
         << ", \"candidate_min_schur_cholesky_diagonal\": "
         << result.candidate_min_schur_cholesky_diagonal << "\n";
  stream << indent2 << "},\n";
  stream << indent2 << "\"diagnostics\": {\n";
  stream << indent3 << "\"backward_error_formula\": "
         << "\"norm_inf(H_delta_plus_g) / "
            "(norm_inf(H)*norm_inf(delta)+norm_inf(g))\",\n";
  stream << indent3 << "\"backward_error_limit\": "
         << options.linear_backward_error_limit
         << ", \"reference_backward_error\": "
         << result.reference_backward_error
         << ", \"candidate_backward_error\": "
         << result.candidate_backward_error
         << ", \"unfrozen_candidate_backward_error\": "
         << result.unfrozen_candidate_backward_error << ",\n";
  stream << indent3 << "\"reference_schur_condition_estimate\": "
         << result.reference_schur_condition_estimate
         << ", \"candidate_schur_condition_estimate\": "
         << result.candidate_schur_condition_estimate << ",\n";
  stream << indent3 << "\"reference_schur_symmetry_error\": "
         << result.reference_schur_symmetry_error
         << ", \"candidate_schur_symmetry_error\": "
         << result.candidate_schur_symmetry_error << "\n";
  stream << indent2 << "},\n";
  stream << indent2 << "\"non_gating_end_to_end_diagnostics\": [\n";
  for (size_t i = 0;
       i < result.non_gating_end_to_end_diagnostics.size(); ++i) {
    const ErrorSummary& metric =
        result.non_gating_end_to_end_diagnostics[i];
    stream << indent3 << "{\"name\": \"" << EscapeJson(metric.name)
           << "\", \"atol\": " << metric.atol
           << ", \"rtol\": " << metric.rtol
           << ", \"count\": " << metric.count
           << ", \"failures\": " << metric.failures
           << ", \"nonfinite\": " << metric.nonfinite
           << ", \"absolute\": {\"max\": " << metric.max_absolute
           << ", \"mean\": " << metric.mean_absolute
           << ", \"rms\": " << metric.rms_absolute
           << ", \"median\": " << metric.median_absolute
           << ", \"p95\": " << metric.p95_absolute
           << "}, \"relative\": {\"max\": " << metric.max_relative
           << ", \"mean\": " << metric.mean_relative
           << ", \"rms\": " << metric.rms_relative
           << ", \"median\": " << metric.median_relative
           << ", \"p95\": " << metric.p95_relative
           << "}, \"worst_gate_ratio\": " << metric.worst_gate_ratio
           << ", \"worst_element_id\": \""
           << EscapeJson(metric.worst_element_id)
           << "\", \"worst_reference\": ";
    AppendJsonNumber(&stream, metric.worst_reference);
    stream << ", \"worst_candidate\": ";
    AppendJsonNumber(&stream, metric.worst_candidate);
    stream << ", \"would_pass_strict_gate\": "
           << (metric.pass ? "true" : "false") << "}";
    if (i + 1 != result.non_gating_end_to_end_diagnostics.size()) {
      stream << ',';
    }
    stream << '\n';
  }
  stream << indent2 << "],\n";
  stream << indent2 << "\"step\": {\n";
  stream << indent3 << "\"reference_current_cost\": "
         << result.reference_current_cost
         << ", \"candidate_current_cost\": "
         << result.candidate_current_cost << ",\n";
  stream << indent3 << "\"reference_trial_cost\": "
         << result.reference_trial_cost
         << ", \"candidate_trial_cost\": "
         << result.candidate_trial_cost << ",\n";
  stream << indent3 << "\"reference_predicted_reduction\": "
         << result.reference_predicted_reduction
         << ", \"candidate_predicted_reduction\": "
         << result.candidate_predicted_reduction << ",\n";
  stream << indent3 << "\"reference_actual_reduction\": "
         << result.reference_actual_reduction
         << ", \"candidate_actual_reduction\": "
         << result.candidate_actual_reduction << ",\n";
  stream << indent3 << "\"reference_rho\": " << result.reference_rho
         << ", \"candidate_rho\": " << result.candidate_rho
         << ", \"reference_accept\": "
         << (result.reference_accept ? "true" : "false")
         << ", \"candidate_accept\": "
         << (result.candidate_accept ? "true" : "false") << "\n";
  stream << indent2 << "},\n";
  stream << indent2 << "\"metrics\": [\n";
  for (size_t i = 0; i < result.metrics.size(); ++i) {
    const ErrorSummary& metric = result.metrics[i];
    stream << indent3 << "{\"name\": \"" << EscapeJson(metric.name)
           << "\", \"atol\": " << metric.atol
           << ", \"rtol\": " << metric.rtol
           << ", \"count\": " << metric.count
           << ", \"failures\": " << metric.failures
           << ", \"nonfinite\": " << metric.nonfinite
           << ", \"absolute\": {\"max\": " << metric.max_absolute
           << ", \"mean\": " << metric.mean_absolute
           << ", \"rms\": " << metric.rms_absolute
           << ", \"median\": " << metric.median_absolute
           << ", \"p95\": " << metric.p95_absolute
           << "}, \"relative\": {\"max\": " << metric.max_relative
           << ", \"mean\": " << metric.mean_relative
           << ", \"rms\": " << metric.rms_relative
           << ", \"median\": " << metric.median_relative
           << ", \"p95\": " << metric.p95_relative
           << "}, \"worst_gate_ratio\": " << metric.worst_gate_ratio
           << ", \"worst_element_id\": \""
           << EscapeJson(metric.worst_element_id)
           << "\", \"worst_reference\": ";
    AppendJsonNumber(&stream, metric.worst_reference);
    stream << ", \"worst_candidate\": ";
    AppendJsonNumber(&stream, metric.worst_candidate);
    stream << ", \"pass\": " << (metric.pass ? "true" : "false")
           << "}";
    if (i + 1 != result.metrics.size()) stream << ',';
    stream << '\n';
  }
  stream << indent2 << "],\n";
  stream << indent2 << "\"pass\": " << (result.pass ? "true" : "false")
         << '\n';
  stream << indent << '}';
  return stream.str();
}

}  // namespace gpu_ba
}  // namespace colmap
