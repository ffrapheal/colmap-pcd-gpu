#include "gpu_ba/fixed_linearization.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
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

enum class AssemblyOrder { kCanonical, kSource };

enum class LossKind { kTrivial, kSoftL1 };

struct LossConfig {
  LossKind kind = LossKind::kTrivial;
  double scale = 1.0;
  std::string name = "TRIVIAL";
  int reduction_threads = 1;
};

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

DampingDiagonal ComputeInitialJacobiScaling(
    const CanonicalLayout& layout, const BlockSystem& system);

DampingDiagonal ComputeCeres14Damping(
    const CanonicalLayout& layout,
    const BlockSystem& system,
    const DampingDiagonal& jacobi_scaling,
    double min_diagonal,
    double max_diagonal);

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

double ReduceBlockCostsCeres14(const std::vector<double>& block_costs,
                               int requested_threads) {
  const size_t thread_count = static_cast<size_t>(
      std::max(1, requested_threads));
  std::vector<double> partial(thread_count, 0.0);
  const size_t quotient = block_costs.size() / thread_count;
  const size_t remainder = block_costs.size() % thread_count;
  size_t begin = 0;
  for (size_t thread = 0; thread < thread_count; ++thread) {
    const size_t count = quotient + (thread < remainder ? 1 : 0);
    const size_t end = begin + count;
    for (size_t i = begin; i < end; ++i) partial[thread] += block_costs[i];
    begin = end;
  }
  double total = 0.0;
  for (const double value : partial) total += value;
  return total;
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

bool ParseLossConfig(const std::string& name,
                     double scale,
                     LossConfig* loss,
                     std::string* error) {
  std::string normalized = name;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  if (normalized == "TRIVIAL") {
    loss->kind = LossKind::kTrivial;
    loss->scale = 1.0;
    loss->name = "TRIVIAL";
    return true;
  }
  if (normalized == "SOFT_L1") {
    if (!std::isfinite(scale) || scale <= 0.0) {
      *error = "SOFT_L1 loss scale must be finite and positive";
      return false;
    }
    loss->kind = LossKind::kSoftL1;
    loss->scale = scale;
    loss->name = "SOFT_L1";
    return true;
  }
  *error = "custom_cpu supports only TRIVIAL and SOFT_L1 loss, got " + name;
  return false;
}

bool BuildLayout(const Snapshot& snapshot,
                 AssemblyOrder order,
                 CanonicalLayout* layout,
                 std::string* error) {
  *layout = CanonicalLayout();
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
  std::map<std::pair<ParameterKind, uint64_t>, const ParameterBlockSnapshot*>
      parameters;
  for (const ParameterBlockSnapshot& parameter :
       snapshot.parameter_blocks_source_order) {
    if (!parameters
             .emplace(std::make_pair(parameter.kind, parameter.entity_id),
                      &parameter)
             .second) {
      *error = "Duplicate parameter identity in source insertion order";
      return false;
    }
  }

  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  for (const ImageSnapshot& image : snapshot.images) {
    images.emplace(image.image_id, &image);
  }
  for (const PointSnapshot& point : snapshot.points) {
    points.emplace(point.point3D_id, &point);
  }
  std::vector<const ImageSnapshot*> variable_images;
  if (order == AssemblyOrder::kCanonical) {
    for (const ImageSnapshot& image : snapshot.images) {
      if (!image.pose_constant) variable_images.push_back(&image);
    }
    std::sort(variable_images.begin(), variable_images.end(),
              [](const ImageSnapshot* lhs, const ImageSnapshot* rhs) {
                return lhs->image_id < rhs->image_id;
              });
  } else {
    std::set<uint32_t> seen;
    for (const ParameterBlockSnapshot& parameter :
         snapshot.parameter_blocks_source_order) {
      if (parameter.kind != ParameterKind::kQuaternion) continue;
      const auto image_it = images.find(static_cast<uint32_t>(parameter.entity_id));
      if (image_it != images.end() && !image_it->second->pose_constant &&
          seen.insert(image_it->first).second) {
        variable_images.push_back(image_it->second);
      }
    }
  }
  std::ostringstream layout_text;
  layout_text << (order == AssemblyOrder::kCanonical
                      ? "canonical-layout-v1\n"
                      : "source-first-insertion-layout-v1\n");
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
  const size_t expected_variable_images = static_cast<size_t>(std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const ImageSnapshot& image) { return !image.pose_constant; }));
  if (variable_images.size() != expected_variable_images) {
    *error = "Source parameter order does not cover every variable pose";
    return false;
  }

  std::vector<const PointSnapshot*> variable_points;
  if (order == AssemblyOrder::kCanonical) {
    for (const PointSnapshot& point : snapshot.points) {
      if (!point.constant) variable_points.push_back(&point);
    }
    std::sort(variable_points.begin(), variable_points.end(),
              [](const PointSnapshot* lhs, const PointSnapshot* rhs) {
                return lhs->point3D_id < rhs->point3D_id;
              });
  } else {
    std::set<uint64_t> seen;
    for (const ParameterBlockSnapshot& parameter :
         snapshot.parameter_blocks_source_order) {
      if (parameter.kind != ParameterKind::kPoint3D) continue;
      const auto point_it = points.find(parameter.entity_id);
      if (point_it != points.end() && !point_it->second->constant &&
          seen.insert(point_it->first).second) {
        variable_points.push_back(point_it->second);
      }
    }
  }
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
  const size_t expected_variable_points = static_cast<size_t>(std::count_if(
      snapshot.points.begin(), snapshot.points.end(),
      [](const PointSnapshot& point) { return !point.constant; }));
  if (variable_points.size() != expected_variable_points) {
    *error = "Source parameter order does not cover every variable point";
    return false;
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
                      bool accumulate_quadratic_cost,
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
    if (accumulate_quadratic_cost) {
      system->cost += 0.5 * residual * residual;
    }
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
                     bool accumulate_quadratic_cost,
                     BlockSystem* system) {
  if (accumulate_quadratic_cost) {
    system->cost += 0.5 * residual * residual;
  }
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

bool ApplyLoss(const LossConfig& loss,
               double squared_norm,
               double* cost,
               double* residual_jacobian_scale,
               std::string* error) {
  if (!std::isfinite(squared_norm) || squared_norm < 0.0) {
    *error = "Loss received an invalid squared residual norm";
    return false;
  }
  if (loss.kind == LossKind::kTrivial) {
    *cost = 0.5 * squared_norm;
    *residual_jacobian_scale = 1.0;
    return true;
  }
  // Independent reproduction of Ceres 1.14 SoftLOneLoss followed by the
  // rho'' <= 0 branch of Corrector.  For SOFT_L1 this branch is always used.
  const double inverse_scale_squared = 1.0 / (loss.scale * loss.scale);
  const double sum = 1.0 + squared_norm * inverse_scale_squared;
  const double root = std::sqrt(sum);
  const double rho0 = 2.0 * loss.scale * loss.scale * (root - 1.0);
  const double rho1 =
      std::max(std::numeric_limits<double>::min(), 1.0 / root);
  *cost = 0.5 * rho0;
  *residual_jacobian_scale = std::sqrt(rho1);
  if (!std::isfinite(*cost) || !std::isfinite(*residual_jacobian_scale)) {
    *error = "SOFT_L1 correction produced a non-finite value";
    return false;
  }
  return true;
}

bool RobustifyVisual(const LossConfig& loss,
                     VisualEvaluation* evaluation,
                     double* cost,
                     bool* quadratic_cost,
                     std::string* error) {
  const double squared_norm =
      evaluation->residual[0] * evaluation->residual[0] +
      evaluation->residual[1] * evaluation->residual[1];
  double scale = 1.0;
  if (!ApplyLoss(loss, squared_norm, cost, &scale, error)) return false;
  *quadratic_cost = loss.kind == LossKind::kTrivial;
  if (*quadratic_cost) return true;
  for (double& value : evaluation->residual) value *= scale;
  for (double& value : evaluation->ambient_quaternion_jacobian) value *= scale;
  // PlusJacobian belongs to the parameterization, not the residual block.
  for (double& value : evaluation->local_rotation_jacobian) value *= scale;
  for (double& value : evaluation->translation_jacobian) value *= scale;
  for (double& value : evaluation->point_jacobian) value *= scale;
  for (double& value : evaluation->camera_jacobian) value *= scale;
  return true;
}

bool RobustifyLidar(const LossConfig& loss,
                    double* residual,
                    std::array<double, 3>* jacobian,
                    double* cost,
                    bool* quadratic_cost,
                    std::string* error) {
  double scale = 1.0;
  if (!ApplyLoss(loss, (*residual) * (*residual), cost, &scale, error)) {
    return false;
  }
  *quadratic_cost = loss.kind == LossKind::kTrivial;
  if (!*quadratic_cost) {
    *residual *= scale;
    for (double& value : *jacobian) value *= scale;
  }
  return true;
}

bool AssembleSystem(const Snapshot& snapshot,
                    const SnapshotLookup& lookup,
                    const CanonicalLayout& layout,
                    EvaluationSource source,
                    AssemblyOrder order,
                    const LossConfig& loss,
                    const FrozenResiduals* frozen_input,
                    FrozenResiduals* frozen_output,
                    BlockSystem* system,
                    std::string* error) {
  InitializeSystem(layout, system);
  std::vector<double> block_costs;
  block_costs.reserve(snapshot.source_insertion_order.size());
  if (frozen_output != nullptr) {
    const size_t count = snapshot.source_insertion_order.size();
    frozen_output->visual.resize(count);
    frozen_output->lidar.assign(count, 0.0);
    frozen_output->has_visual.assign(count, 0);
    frozen_output->has_lidar.assign(count, 0);
  }
  const std::vector<OrderEntrySnapshot>& residual_order =
      order == AssemblyOrder::kCanonical ? snapshot.canonical_order
                                         : snapshot.source_insertion_order;
  for (const OrderEntrySnapshot& entry : residual_order) {
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
      double block_cost = 0.0;
      bool quadratic_cost = true;
      if (!RobustifyVisual(loss, &evaluation, &block_cost, &quadratic_cost,
                           error)) {
        return false;
      }
      block_costs.push_back(block_cost);
      AccumulateVisual(layout, *observation, evaluation, false,
                       system);
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
      double block_cost = 0.0;
      bool quadratic_cost = true;
      if (!RobustifyLidar(loss, &residual, &jacobian, &block_cost,
                          &quadratic_cost, error)) {
        return false;
      }
      block_costs.push_back(block_cost);
      AccumulateLidar(layout, *lidar, residual, jacobian, false,
                      system);
    }
  }
  std::sort(system->edges.begin(), system->edges.end(),
            [](const EdgeBlock& lhs, const EdgeBlock& rhs) {
              return std::tie(lhs.pose_index, lhs.point_index) <
                     std::tie(rhs.pose_index, rhs.point_index);
            });
  system->edge_lookup.clear();
  system->cost =
      ReduceBlockCostsCeres14(block_costs, loss.reduction_threads);
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

std::string FixedStepSemanticsName(FixedStepSemantics semantics) {
  return semantics == FixedStepSemantics::kCeres14 ? "ceres14" : "legacy";
}

std::string CustomCpuSolveModeName(CustomCpuSolveMode mode) {
  switch (mode) {
    case CustomCpuSolveMode::kLegacySingleStep:
      return "legacy_single_step";
    case CustomCpuSolveMode::kCeres14SingleStep:
      return "ceres14_single_step";
    case CustomCpuSolveMode::kFull:
      return "full_lm";
  }
  return "unknown";
}

std::string LmDiagonalSha256(const std::vector<double>& values) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "lm-diagonal-v1\n";
  for (size_t i = 0; i < values.size(); ++i) {
    stream << i << ' ' << values[i] << '\n';
  }
  return Sha256Hex(stream.str());
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
                bool ceres14_quaternion_plus,
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
      ceres::QuaternionParameterization parameterization;
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
    } else if (ceres14_quaternion_plus) {
      if (!QuaternionPlusCeres14(image.qvec, rotation_delta, &updated)) {
        *error = "Ceres-compatible analytic trial quaternion Plus failed";
        return false;
      }
      image.qvec = updated;
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
                       AssemblyOrder order,
                       const LossConfig& loss,
                       double* cost,
                       std::string* error) {
  SnapshotLookup lookup;
  if (!BuildLookup(state, &lookup, error)) return false;
  std::vector<double> block_costs;
  block_costs.reserve(state.source_insertion_order.size());
  const std::vector<OrderEntrySnapshot>& residual_order =
      order == AssemblyOrder::kCanonical ? state.canonical_order
                                         : state.source_insertion_order;
  for (const OrderEntrySnapshot& entry : residual_order) {
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
        const double squared_norm =
            evaluation.residual[0] * evaluation.residual[0] +
            evaluation.residual[1] * evaluation.residual[1];
        double block_cost = 0.0;
        double unused_scale = 1.0;
        if (!ApplyLoss(loss, squared_norm, &block_cost, &unused_scale,
                       error)) {
          return false;
        }
        block_costs.push_back(block_cost);
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
        const double squared_norm = residual[0] * residual[0] +
                                    residual[1] * residual[1];
        double block_cost = 0.0;
        double unused_scale = 1.0;
        if (!ApplyLoss(loss, squared_norm, &block_cost, &unused_scale,
                       error)) {
          return false;
        }
        block_costs.push_back(block_cost);
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
      double block_cost = 0.0;
      double unused_scale = 1.0;
      if (!ApplyLoss(loss, residual * residual, &block_cost, &unused_scale,
                     error)) {
        return false;
      }
      block_costs.push_back(block_cost);
    }
  }
  *cost = ReduceBlockCostsCeres14(block_costs, loss.reduction_threads);
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
  result->step_semantics = FixedStepSemanticsName(options.step_semantics);
  if (!std::isfinite(options.lambda) || options.lambda < 0.0 ||
      !std::isfinite(options.diagonal_floor) || options.diagonal_floor <= 0.0) {
    *error = "Invalid fixed-linearization lambda or diagonal floor";
    return false;
  }
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, AssemblyOrder::kCanonical, &layout, error)) {
    return false;
  }
  const LossConfig trivial_loss;
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
                      AssemblyOrder::kCanonical, trivial_loss, nullptr,
                      &frozen_residuals, &reference, error) ||
      !AssembleSystem(snapshot, lookup, layout, EvaluationSource::kAnalytic,
                      AssemblyOrder::kCanonical, trivial_loss,
                      &frozen_residuals, nullptr, &candidate, error)) {
    return false;
  }
  result->pose_point_edges = reference.edges.size();
  DampingDiagonal reference_damping;
  DampingDiagonal candidate_damping;
  if (options.step_semantics == FixedStepSemantics::kCeres14) {
    const DampingDiagonal reference_scaling =
        ComputeInitialJacobiScaling(layout, reference);
    const DampingDiagonal candidate_scaling =
        ComputeInitialJacobiScaling(layout, candidate);
    reference_damping = ComputeCeres14Damping(
        layout, reference, reference_scaling, 1e-6, 1e32);
    candidate_damping = ComputeCeres14Damping(
        layout, candidate, candidate_scaling, 1e-6, 1e32);
  } else {
    reference_damping =
        ComputeDamping(layout, reference, options.diagonal_floor);
    candidate_damping =
        ComputeDamping(layout, candidate, options.diagonal_floor);
  }
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
  const bool ceres14_plus =
      options.step_semantics == FixedStepSemantics::kCeres14;
  if (!ApplyDelta(layout, reference_solve, EvaluationSource::kAutoDiff,
                  ceres14_plus, &reference_trial, error) ||
      !ApplyDelta(layout, candidate_solve, EvaluationSource::kAnalytic,
                  ceres14_plus, &candidate_trial, error) ||
      !EvaluateTotalCost(reference_trial, EvaluationSource::kAutoDiff,
                         AssemblyOrder::kCanonical, trivial_loss,
                         &result->reference_trial_cost, error) ||
      !EvaluateTotalCost(candidate_trial, EvaluationSource::kAnalytic,
                         AssemblyOrder::kCanonical, trivial_loss,
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
         << "\"phase_4_canonical_blocks_and_fixed_step_"
         << result.step_semantics << "\",\n";
  stream << indent2 << "\"step_semantics\": \""
         << result.step_semantics << "\",\n";
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

namespace {

double ScaledGradientNorm(const CanonicalLayout& layout,
                          const BlockSystem& system,
                          double diagonal_floor) {
  double norm = 0.0;
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      const double scale = std::sqrt(
          std::max(system.pose_hessian[pose_index](i, i), diagonal_floor));
      norm = std::max(norm, std::abs(system.pose_gradient[pose_index][i]) /
                                std::max(scale, 1e-12));
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    for (size_t i = 0; i < 3; ++i) {
      const double scale = std::sqrt(
          std::max(system.point_hessian[point_index](i, i), diagonal_floor));
      norm = std::max(norm, std::abs(system.point_gradient[point_index][i]) /
                                std::max(scale, 1e-12));
    }
  }
  return norm;
}

bool LinearizeState(const Snapshot& state,
                    const CanonicalLayout& layout,
                    EvaluationSource source,
                    AssemblyOrder order,
                    const LossConfig& loss,
                    BlockSystem* system,
                    std::string* error) {
  SnapshotLookup lookup;
  if (!BuildLookup(state, &lookup, error)) return false;
  return AssembleSystem(state, lookup, layout, source, order, loss, nullptr,
                        nullptr, system, error);
}

bool ComputeGradientNorms(const Snapshot& state,
                          const CanonicalLayout& layout,
                          const BlockSystem& system,
                          CpuGradientNorms* norms,
                          std::string* error) {
  *norms = CpuGradientNorms();
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  for (const ImageSnapshot& image : state.images) {
    images.emplace(image.image_id, &image);
  }
  long double ambient_squared_norm = 0.0;
  auto add_ambient_difference = [&](double difference,
                                    const std::string& id) {
    const double absolute = std::abs(difference);
    ambient_squared_norm += difference * difference;
    if (absolute > norms->projected_max_norm ||
        norms->worst_ambient_id.empty()) {
      norms->projected_max_norm = absolute;
      norms->worst_ambient_id = id;
    }
  };
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    const auto image_it = images.find(pose.image_id);
    if (image_it == images.end()) {
      *error = "Projected gradient references a missing image";
      return false;
    }
    const ImageSnapshot& image = *image_it->second;
    std::array<double, 3> negative_rotation_gradient;
    for (size_t i = 0; i < 3; ++i) {
      const double gradient = system.pose_gradient[pose_index][i];
      negative_rotation_gradient[i] = -gradient;
      norms->raw_tangent_max_norm =
          std::max(norms->raw_tangent_max_norm, std::abs(gradient));
    }
    std::array<double, 4> projected_quaternion;
    if (!QuaternionPlusCeres14(image.qvec, negative_rotation_gradient,
                               &projected_quaternion)) {
      *error = "Projected-gradient quaternion Plus failed: image=" +
               std::to_string(pose.image_id);
      return false;
    }
    for (size_t i = 0; i < 4; ++i) {
      add_ambient_difference(
          image.qvec[i] - projected_quaternion[i],
          "image=" + std::to_string(pose.image_id) + ":q[" +
              std::to_string(i) + "]");
    }
    ++norms->variable_rotation_blocks;
    for (size_t free_index = 0;
         free_index < pose.free_translation_indices.size(); ++free_index) {
      const int ambient_index = pose.free_translation_indices[free_index];
      const double gradient =
          system.pose_gradient[pose_index][3 + free_index];
      norms->raw_tangent_max_norm =
          std::max(norms->raw_tangent_max_norm, std::abs(gradient));
      add_ambient_difference(
          gradient, "image=" + std::to_string(pose.image_id) + ":t[" +
                        std::to_string(ambient_index) + "]");
      ++norms->variable_translation_components;
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    for (size_t i = 0; i < 3; ++i) {
      const double gradient = system.point_gradient[point_index][i];
      norms->raw_tangent_max_norm =
          std::max(norms->raw_tangent_max_norm, std::abs(gradient));
      add_ambient_difference(
          gradient, "point3D=" + std::to_string(point.point3D_id) + "[" +
                        std::to_string(i) + "]");
    }
    ++norms->variable_point_blocks;
  }
  norms->projected_l2_norm =
      std::sqrt(static_cast<double>(ambient_squared_norm));
  norms->scaled_max_norm = ScaledGradientNorm(layout, system, 1e-12);
  if (!std::isfinite(norms->projected_max_norm) ||
      !std::isfinite(norms->projected_l2_norm) ||
      !std::isfinite(norms->raw_tangent_max_norm) ||
      !std::isfinite(norms->scaled_max_norm)) {
    *error = "Projected-gradient norms are non-finite";
    return false;
  }
  return true;
}

DampingDiagonal ComputeInitialJacobiScaling(
    const CanonicalLayout& layout, const BlockSystem& system) {
  DampingDiagonal scaling;
  scaling.camera = Eigen::VectorXd::Ones(layout.pose_dimension);
  scaling.point = Eigen::VectorXd::Ones(layout.point_dimension);
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      const double diagonal =
          std::max(0.0, system.pose_hessian[pose_index](i, i));
      scaling.camera[pose.offset + i] =
          1.0 / (1.0 + std::sqrt(diagonal));
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    for (size_t i = 0; i < 3; ++i) {
      const double diagonal =
          std::max(0.0, system.point_hessian[point_index](i, i));
      scaling.point[point.offset + i] =
          1.0 / (1.0 + std::sqrt(diagonal));
    }
  }
  return scaling;
}

DampingDiagonal ComputeCeres14Damping(
    const CanonicalLayout& layout,
    const BlockSystem& system,
    const DampingDiagonal& jacobi_scaling,
    double min_diagonal,
    double max_diagonal) {
  DampingDiagonal damping;
  damping.camera = Eigen::VectorXd::Zero(layout.pose_dimension);
  damping.point = Eigen::VectorXd::Zero(layout.point_dimension);
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    for (size_t i = 0; i < pose.Dimension(); ++i) {
      const size_t offset = pose.offset + i;
      const double scale = jacobi_scaling.camera[offset];
      const double scaled_diagonal =
          system.pose_hessian[pose_index](i, i) * scale * scale;
      const double clamped =
          std::min(std::max(scaled_diagonal, min_diagonal), max_diagonal);
      damping.camera[offset] = clamped / (scale * scale);
    }
  }
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    for (size_t i = 0; i < 3; ++i) {
      const size_t offset = point.offset + i;
      const double scale = jacobi_scaling.point[offset];
      const double scaled_diagonal =
          system.point_hessian[point_index](i, i) * scale * scale;
      const double clamped =
          std::min(std::max(scaled_diagonal, min_diagonal), max_diagonal);
      damping.point[offset] = clamped / (scale * scale);
    }
  }
  return damping;
}

void ComputeCeres14LmDiagonalRange(
    const DampingDiagonal& effective_damping,
    const DampingDiagonal& jacobi_scaling,
    double radius,
    double* minimum,
    double* maximum) {
  *minimum = std::numeric_limits<double>::max();
  *maximum = 0.0;
  auto accumulate = [&](const Eigen::VectorXd& damping,
                        const Eigen::VectorXd& scaling) {
    for (Eigen::Index i = 0; i < damping.size(); ++i) {
      const double scaled_clamped =
          damping[i] * scaling[i] * scaling[i];
      const double value = std::sqrt(scaled_clamped / radius);
      *minimum = std::min(*minimum, value);
      *maximum = std::max(*maximum, value);
    }
  };
  accumulate(effective_damping.camera, jacobi_scaling.camera);
  accumulate(effective_damping.point, jacobi_scaling.point);
  if (*minimum == std::numeric_limits<double>::max()) *minimum = 0.0;
}

std::vector<double> ComputeCeres14LmDiagonalValues(
    const DampingDiagonal& effective_damping,
    const DampingDiagonal& jacobi_scaling,
    double radius) {
  std::vector<double> values;
  values.reserve(static_cast<size_t>(effective_damping.camera.size() +
                                     effective_damping.point.size()));
  auto append = [&](const Eigen::VectorXd& damping,
                    const Eigen::VectorXd& scaling) {
    for (Eigen::Index i = 0; i < damping.size(); ++i) {
      values.push_back(
          std::sqrt(damping[i] * scaling[i] * scaling[i] / radius));
    }
  };
  append(effective_damping.camera, jacobi_scaling.camera);
  append(effective_damping.point, jacobi_scaling.point);
  return values;
}

void ComputeDirectLmDiagonalRange(const DampingDiagonal& damping,
                                  double lambda,
                                  double* minimum,
                                  double* maximum) {
  *minimum = std::numeric_limits<double>::max();
  *maximum = 0.0;
  auto accumulate = [&](const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
      const double value = std::sqrt(lambda * values[i]);
      *minimum = std::min(*minimum, value);
      *maximum = std::max(*maximum, value);
    }
  };
  accumulate(damping.camera);
  accumulate(damping.point);
  if (*minimum == std::numeric_limits<double>::max()) *minimum = 0.0;
}

std::vector<double> ComputeDirectLmDiagonalValues(
    const DampingDiagonal& damping, double lambda) {
  std::vector<double> values;
  values.reserve(static_cast<size_t>(damping.camera.size() +
                                     damping.point.size()));
  auto append = [&](const Eigen::VectorXd& source) {
    for (Eigen::Index i = 0; i < source.size(); ++i) {
      values.push_back(std::sqrt(lambda * source[i]));
    }
  };
  append(damping.camera);
  append(damping.point);
  return values;
}

double AmbientVariableStateNorm(const Snapshot& state,
                                const CanonicalLayout& layout) {
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  for (const ImageSnapshot& image : state.images) {
    images.emplace(image.image_id, &image);
  }
  for (const PointSnapshot& point : state.points) {
    points.emplace(point.point3D_id, &point);
  }
  long double squared_norm = 0.0;
  for (const PoseVariable& pose : layout.poses) {
    const ImageSnapshot& image = *images.at(pose.image_id);
    for (const double value : image.qvec) squared_norm += value * value;
    for (const double value : image.tvec) squared_norm += value * value;
  }
  for (const PointVariable& point : layout.points) {
    for (const double value : points.at(point.point3D_id)->xyz) {
      squared_norm += value * value;
    }
  }
  return std::sqrt(static_cast<double>(squared_norm));
}

double AmbientVariableDifferenceNorm(const Snapshot& lhs,
                                     const Snapshot& rhs,
                                     const CanonicalLayout& layout) {
  std::unordered_map<uint32_t, const ImageSnapshot*> lhs_images;
  std::unordered_map<uint32_t, const ImageSnapshot*> rhs_images;
  std::unordered_map<uint64_t, const PointSnapshot*> lhs_points;
  std::unordered_map<uint64_t, const PointSnapshot*> rhs_points;
  for (const ImageSnapshot& image : lhs.images) {
    lhs_images.emplace(image.image_id, &image);
  }
  for (const ImageSnapshot& image : rhs.images) {
    rhs_images.emplace(image.image_id, &image);
  }
  for (const PointSnapshot& point : lhs.points) {
    lhs_points.emplace(point.point3D_id, &point);
  }
  for (const PointSnapshot& point : rhs.points) {
    rhs_points.emplace(point.point3D_id, &point);
  }
  long double squared_norm = 0.0;
  auto add = [&](double difference) { squared_norm += difference * difference; };
  for (const PoseVariable& pose : layout.poses) {
    const ImageSnapshot& lhs_image = *lhs_images.at(pose.image_id);
    const ImageSnapshot& rhs_image = *rhs_images.at(pose.image_id);
    for (size_t i = 0; i < 4; ++i) {
      add(lhs_image.qvec[i] - rhs_image.qvec[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
      add(lhs_image.tvec[i] - rhs_image.tvec[i]);
    }
  }
  for (const PointVariable& point : layout.points) {
    const PointSnapshot& lhs_point = *lhs_points.at(point.point3D_id);
    const PointSnapshot& rhs_point = *rhs_points.at(point.point3D_id);
    for (size_t i = 0; i < 3; ++i) {
      add(lhs_point.xyz[i] - rhs_point.xyz[i]);
    }
  }
  return std::sqrt(static_cast<double>(squared_norm));
}

CpuTerminationType CeresTermination(ceres::TerminationType type) {
  if (type == ceres::CONVERGENCE || type == ceres::USER_SUCCESS) {
    return CpuTerminationType::kConvergence;
  }
  if (type == ceres::NO_CONVERGENCE) {
    return CpuTerminationType::kNoConvergence;
  }
  return CpuTerminationType::kFailure;
}

double RootMeanSquare(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  long double squared_sum = 0.0;
  for (const double value : values) squared_sum += value * value;
  return std::sqrt(static_cast<double>(squared_sum / values.size()));
}

double SymmetricConditionEstimate(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() == 0) return 1.0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      matrix, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    return std::numeric_limits<double>::max();
  }
  const double minimum = solver.eigenvalues().minCoeff();
  const double maximum = solver.eigenvalues().maxCoeff();
  return minimum > 0.0 ? maximum / minimum
                       : std::numeric_limits<double>::max();
}

LidarNearZeroSummary ClassifyLidarNearZero(const Snapshot& snapshot) {
  LidarNearZeroSummary summary;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  for (const PointSnapshot& point : snapshot.points) {
    points.emplace(point.point3D_id, &point);
  }
  for (const LidarSnapshot& lidar : snapshot.lidar) {
    const auto point_it = points.find(lidar.point3D_id);
    if (point_it == points.end()) continue;
    const PointSnapshot& point = *point_it->second;
    const double distance = lidar.plane[0] * point.xyz[0] +
                            lidar.plane[1] * point.xyz[1] +
                            lidar.plane[2] * point.xyz[2] + lidar.plane[3];
    ++summary.total_samples;
    if (distance == 0.0) {
      ++summary.exact_zero_samples;
      summary.exact_zero_point_ids.push_back(point.point3D_id);
    }
    if (std::abs(distance) <= summary.threshold) {
      ++summary.near_zero_samples;
      summary.near_zero_point_ids.push_back(point.point3D_id);
    }
  }
  auto sort_unique = [](std::vector<uint64_t>* ids) {
    std::sort(ids->begin(), ids->end());
    ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
  };
  sort_unique(&summary.exact_zero_point_ids);
  sort_unique(&summary.near_zero_point_ids);
  return summary;
}

void AppendLidarNearZeroJson(std::ostringstream* stream,
                             const LidarNearZeroSummary& summary) {
  *stream << "{\"threshold\": " << summary.threshold
          << ", \"total_samples\": " << summary.total_samples
          << ", \"exact_zero_samples\": " << summary.exact_zero_samples
          << ", \"near_zero_samples\": " << summary.near_zero_samples
          << ", \"exact_zero_point_ids\": [";
  for (size_t i = 0; i < summary.exact_zero_point_ids.size(); ++i) {
    if (i != 0) *stream << ", ";
    *stream << summary.exact_zero_point_ids[i];
  }
  *stream << "], \"near_zero_point_ids\": [";
  for (size_t i = 0; i < summary.near_zero_point_ids.size(); ++i) {
    if (i != 0) *stream << ", ";
    *stream << summary.near_zero_point_ids[i];
  }
  *stream << "]}";
}

struct CompactSnapshotState {
  std::vector<std::array<double, 4>> image_qvecs;
  std::vector<std::array<double, 3>> image_tvecs;
  std::vector<std::array<double, 3>> point_xyzs;
};

CompactSnapshotState CaptureCompactState(const Snapshot& snapshot) {
  CompactSnapshotState state;
  state.image_qvecs.reserve(snapshot.images.size());
  state.image_tvecs.reserve(snapshot.images.size());
  state.point_xyzs.reserve(snapshot.points.size());
  for (const ImageSnapshot& image : snapshot.images) {
    state.image_qvecs.push_back(image.qvec);
    state.image_tvecs.push_back(image.tvec);
  }
  for (const PointSnapshot& point : snapshot.points) {
    state.point_xyzs.push_back(point.xyz);
  }
  return state;
}

bool RestoreCompactState(const CompactSnapshotState& values,
                         Snapshot* snapshot) {
  if (values.image_qvecs.size() != snapshot->images.size() ||
      values.image_tvecs.size() != snapshot->images.size() ||
      values.point_xyzs.size() != snapshot->points.size()) {
    return false;
  }
  for (size_t i = 0; i < snapshot->images.size(); ++i) {
    snapshot->images[i].qvec = values.image_qvecs[i];
    snapshot->images[i].tvec = values.image_tvecs[i];
  }
  for (size_t i = 0; i < snapshot->points.size(); ++i) {
    snapshot->points[i].xyz = values.point_xyzs[i];
  }
  return true;
}

class DeterministicOracleCallback : public ceres::IterationCallback {
 public:
  explicit DeterministicOracleCallback(const Snapshot* state)
      : state_(state) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary&) override {
    states.push_back(CaptureCompactState(*state_));
    return ceres::SOLVER_CONTINUE;
  }

  std::vector<CompactSnapshotState> states;

 private:
  const Snapshot* state_;
};

uint64_t CountBaImages(const Snapshot& snapshot) {
  return static_cast<uint64_t>(std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const ImageSnapshot& image) { return image.selected; }));
}

ceres::LinearSolverType SelectOriginalLinearSolver(uint64_t ba_image_count) {
  if (ba_image_count <= 50) return ceres::DENSE_SCHUR;
  if (ba_image_count <= 1000) return ceres::SPARSE_SCHUR;
  return ceres::ITERATIVE_SCHUR;
}

bool BuildCeresOracleTrace(const Snapshot& initial_state,
                           const CanonicalLayout& layout,
                           const LossConfig& loss,
                           const ceres::Solver::Summary& summary,
                           const std::vector<CompactSnapshotState>& states,
                           const std::vector<double>& observed_lm_diagonal,
                           std::vector<CustomCpuIteration>* trace,
                           std::string* error) {
  if (states.size() != summary.iterations.size()) {
    *error = "Ceres callback/summary iteration counts differ";
    return false;
  }
  trace->clear();
  trace->reserve(summary.iterations.size());
  Snapshot state = initial_state;
  double current_cost = summary.initial_cost;
  for (size_t i = 0; i < summary.iterations.size(); ++i) {
    if (!RestoreCompactState(states[i], &state)) {
      *error = "Ceres callback state dimensions differ from snapshot";
      return false;
    }
    BlockSystem system;
    if (!LinearizeState(state, layout, EvaluationSource::kAutoDiff,
                        AssemblyOrder::kSource, loss, &system, error)) {
      return false;
    }
    const ceres::IterationSummary& source = summary.iterations[i];
    CustomCpuIteration item;
    item.iteration = source.iteration;
    item.cost_before = i == 0 ? source.cost : current_cost;
    item.trial_cost = source.cost;
    item.actual_reduction = source.cost_change;
    item.rho = source.relative_decrease;
    item.predicted_reduction =
        source.relative_decrease == 0.0
            ? 0.0
            : source.cost_change / source.relative_decrease;
    item.accepted = source.step_is_successful;
    item.invalid = !source.step_is_valid;
    item.step_valid = source.step_is_valid;
    item.factorization_success = source.step_is_valid;
    item.trial_finite = std::isfinite(source.cost);
    item.step_norm = source.step_norm;
    item.eta = source.eta;
    item.linear_solver_iterations = source.linear_solver_iterations;
    item.radius_before = i == 0
                             ? source.trust_region_radius
                             : summary.iterations[i - 1].trust_region_radius;
    item.radius_after = source.trust_region_radius;
    item.lambda_before = 1.0 / item.radius_before;
    item.lambda_after = 1.0 / item.radius_after;
    if (source.iteration == 1 && !observed_lm_diagonal.empty()) {
      const auto range = std::minmax_element(observed_lm_diagonal.begin(),
                                             observed_lm_diagonal.end());
      item.lm_diagonal_min = *range.first;
      item.lm_diagonal_max = *range.second;
      item.lm_diagonal_observed = true;
    }
    if (source.step_is_successful) {
      current_cost = source.cost;
    }
    item.cost_after = current_cost;
    item.projected_gradient_max_norm = source.gradient_max_norm;
    item.scaled_gradient_norm = ScaledGradientNorm(layout, system, 1e-12);
    item.gradient_norm = source.gradient_norm;
    if (i + 1 == summary.iterations.size()) {
      item.termination_reason = summary.message;
    }
    trace->push_back(std::move(item));
  }
  return true;
}

}  // namespace

bool EvaluateCustomCpuGradientNorms(const Snapshot& snapshot,
                                    CpuGradientNorms* norms,
                                    std::string* error) {
  if (norms == nullptr || error == nullptr) return false;
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, AssemblyOrder::kSource, &layout, error)) {
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(snapshot.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : snapshot.metadata.loss_function,
                       1.0, &loss, error)) {
    return false;
  }
  loss.reduction_threads = 1;
  BlockSystem system;
  if (!LinearizeState(snapshot, layout, EvaluationSource::kAnalytic,
                      AssemblyOrder::kSource, loss, &system, error)) {
    return false;
  }
  return ComputeGradientNorms(snapshot, layout, system, norms, error);
}

bool ExportCustomCpuLinearizationImpl(
    const Snapshot& snapshot,
    AssemblyOrder assembly_order,
    int reduction_threads,
    double min_lm_diagonal,
    double max_lm_diagonal,
    CustomCpuLinearizationExport* output,
    std::string* error) {
  if (output == nullptr || error == nullptr) return false;
  *output = CustomCpuLinearizationExport();
  if (reduction_threads <= 0 || !std::isfinite(min_lm_diagonal) ||
      !std::isfinite(max_lm_diagonal) || min_lm_diagonal <= 0.0 ||
      max_lm_diagonal < min_lm_diagonal) {
    *error = "Invalid LM diagonal limits for custom_cpu export";
    return false;
  }
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, assembly_order, &layout, error)) {
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(snapshot.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : snapshot.metadata.loss_function,
                       1.0, &loss, error)) {
    return false;
  }
  loss.reduction_threads = reduction_threads;
  BlockSystem system;
  if (!LinearizeState(snapshot, layout, EvaluationSource::kAnalytic,
                      assembly_order, loss, &system, error)) {
    return false;
  }
  const DampingDiagonal jacobi_scaling =
      ComputeInitialJacobiScaling(layout, system);
  const DampingDiagonal damping = ComputeCeres14Damping(
      layout, system, jacobi_scaling, min_lm_diagonal, max_lm_diagonal);
  output->cost = system.cost;
  output->layout_sha256 = layout.sha256;
  output->poses.reserve(layout.poses.size());
  for (size_t pose_index = 0; pose_index < layout.poses.size(); ++pose_index) {
    const PoseVariable& pose = layout.poses[pose_index];
    CustomCpuPoseBlockExport block;
    block.image_id = pose.image_id;
    block.dimension = static_cast<uint32_t>(pose.Dimension());
    block.free_translation_indices.assign(
        pose.free_translation_indices.begin(),
        pose.free_translation_indices.end());
    block.hessian.resize(pose.Dimension() * pose.Dimension());
    block.gradient.resize(pose.Dimension());
    block.jacobi_scaling.resize(pose.Dimension());
    block.damping.resize(pose.Dimension());
    for (size_t row = 0; row < pose.Dimension(); ++row) {
      block.gradient[row] = system.pose_gradient[pose_index][row];
      block.jacobi_scaling[row] =
          jacobi_scaling.camera[pose.offset + row];
      block.damping[row] = damping.camera[pose.offset + row];
      for (size_t col = 0; col < pose.Dimension(); ++col) {
        block.hessian[row * pose.Dimension() + col] =
            system.pose_hessian[pose_index](row, col);
      }
    }
    output->poses.push_back(std::move(block));
  }
  output->points.reserve(layout.points.size());
  for (size_t point_index = 0; point_index < layout.points.size();
       ++point_index) {
    const PointVariable& point = layout.points[point_index];
    CustomCpuPointBlockExport block;
    block.point3D_id = point.point3D_id;
    for (size_t row = 0; row < 3; ++row) {
      block.gradient[row] = system.point_gradient[point_index][row];
      block.jacobi_scaling[row] =
          jacobi_scaling.point[point.offset + row];
      block.damping[row] = damping.point[point.offset + row];
      for (size_t col = 0; col < 3; ++col) {
        block.hessian[row * 3 + col] =
            system.point_hessian[point_index](row, col);
      }
    }
    output->points.push_back(std::move(block));
  }
  output->edges.reserve(system.edges.size());
  for (const EdgeBlock& edge : system.edges) {
    CustomCpuEdgeBlockExport block;
    block.pose_index = static_cast<uint32_t>(edge.pose_index);
    block.point_index = static_cast<uint32_t>(edge.point_index);
    block.pose_dimension =
        static_cast<uint32_t>(layout.poses[edge.pose_index].Dimension());
    for (size_t row = 0; row < block.pose_dimension; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        block.value[row * 3 + col] = edge.value(row, col);
      }
    }
    output->edges.push_back(std::move(block));
  }
  return ComputeGradientNorms(snapshot, layout, system,
                              &output->gradient_norms, error);
}

bool ExportCustomCpuCanonicalLinearization(
    const Snapshot& snapshot,
    double min_lm_diagonal,
    double max_lm_diagonal,
    CustomCpuLinearizationExport* output,
    std::string* error) {
  return ExportCustomCpuLinearizationImpl(
      snapshot, AssemblyOrder::kCanonical, 1, min_lm_diagonal,
      max_lm_diagonal, output, error);
}

bool ExportCustomCpuSourceLinearization(
    const Snapshot& snapshot,
    int reduction_threads,
    double min_lm_diagonal,
    double max_lm_diagonal,
    CustomCpuLinearizationExport* output,
    std::string* error) {
  return ExportCustomCpuLinearizationImpl(
      snapshot, AssemblyOrder::kSource, reduction_threads, min_lm_diagonal,
      max_lm_diagonal, output, error);
}

bool ExportCustomCpuStepImpl(const Snapshot& snapshot,
                             AssemblyOrder assembly_order,
                             int reduction_threads,
                             double lambda,
                             double min_lm_diagonal,
                             double max_lm_diagonal,
                             CustomCpuCanonicalStepExport* output,
                             std::string* error) {
  if (output == nullptr || error == nullptr) return false;
  *output = CustomCpuCanonicalStepExport();
  if (!std::isfinite(lambda) || lambda < 0.0) {
    *error = "Invalid lambda for custom_cpu canonical step export";
    return false;
  }
  if (!ExportCustomCpuLinearizationImpl(
          snapshot, assembly_order, reduction_threads, min_lm_diagonal,
          max_lm_diagonal, &output->linearization, error)) {
    return false;
  }
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, assembly_order, &layout, error)) {
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(snapshot.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : snapshot.metadata.loss_function,
                       1.0, &loss, error)) {
    return false;
  }
  loss.reduction_threads = reduction_threads;
  BlockSystem system;
  if (!LinearizeState(snapshot, layout, EvaluationSource::kAnalytic,
                      assembly_order, loss, &system, error)) {
    return false;
  }
  const DampingDiagonal jacobi_scaling =
      ComputeInitialJacobiScaling(layout, system);
  const DampingDiagonal damping = ComputeCeres14Damping(
      layout, system, jacobi_scaling, min_lm_diagonal, max_lm_diagonal);
  SolveData solve;
  if (!BuildAndSolveSchur(layout, system, damping, lambda, &solve, error)) {
    return false;
  }
  output->pose_dimension = layout.pose_dimension;
  output->point_dimension = layout.point_dimension;
  output->schur.resize(layout.pose_dimension * layout.pose_dimension);
  for (size_t row = 0; row < layout.pose_dimension; ++row) {
    for (size_t col = 0; col < layout.pose_dimension; ++col) {
      output->schur[row * layout.pose_dimension + col] =
          solve.schur(row, col);
    }
  }
  output->rhs.assign(solve.rhs.data(),
                     solve.rhs.data() + solve.rhs.size());
  output->camera_delta.assign(solve.camera_delta.data(),
                              solve.camera_delta.data() +
                                  solve.camera_delta.size());
  output->point_delta.assign(solve.point_delta.data(),
                             solve.point_delta.data() +
                                 solve.point_delta.size());
  output->backward_error = solve.backward_error;
  output->predicted_reduction = solve.predicted_reduction;
  output->trial_state = snapshot;
  if (!ApplyDelta(layout, solve, EvaluationSource::kAnalytic, true,
                  &output->trial_state, error) ||
      !EvaluateTotalCost(output->trial_state, EvaluationSource::kAnalytic,
                         assembly_order, loss,
                         &output->trial_cost, error)) {
    return false;
  }
  return true;
}

bool ExportCustomCpuCanonicalStep(const Snapshot& snapshot,
                                  double lambda,
                                  double min_lm_diagonal,
                                  double max_lm_diagonal,
                                  CustomCpuCanonicalStepExport* output,
                                  std::string* error) {
  return ExportCustomCpuStepImpl(
      snapshot, AssemblyOrder::kCanonical, 1, lambda, min_lm_diagonal,
      max_lm_diagonal, output, error);
}

bool ExportCustomCpuSourceStep(const Snapshot& snapshot,
                               int reduction_threads,
                               double lambda,
                               double min_lm_diagonal,
                               double max_lm_diagonal,
                               CustomCpuCanonicalStepExport* output,
                               std::string* error) {
  return ExportCustomCpuStepImpl(
      snapshot, AssemblyOrder::kSource, reduction_threads, lambda,
      min_lm_diagonal, max_lm_diagonal, output, error);
}

std::string CpuTerminationTypeName(CpuTerminationType type) {
  switch (type) {
    case CpuTerminationType::kConvergence: return "CONVERGENCE";
    case CpuTerminationType::kNoConvergence: return "NO_CONVERGENCE";
    case CpuTerminationType::kFailure: return "FAILURE";
  }
  return "UNKNOWN";
}

bool RunCustomCpuSolve(const Snapshot& snapshot,
                       double initial_lambda,
                       CustomCpuSolveMode mode,
                       CustomCpuSolveResult* result,
                       Snapshot* final_state,
                       std::string* error) {
  CustomCpuSolverOptions options;
  options.linear_solver_type =
      CountBaImages(snapshot) <= 50 ? "DENSE_SCHUR" : "UNSUPPORTED";
  options.loss_type = snapshot.metadata.loss_function.empty()
                          ? "TRIVIAL"
                          : snapshot.metadata.loss_function;
  options.loss_scale = 1.0;
  options.max_num_iterations = snapshot.metadata.max_num_iterations;
  options.max_linear_solver_iterations =
      snapshot.metadata.max_linear_solver_iterations;
  options.max_num_consecutive_invalid_steps =
      snapshot.metadata.max_consecutive_invalid_steps > 0
          ? snapshot.metadata.max_consecutive_invalid_steps
          : 10;
  options.function_tolerance = snapshot.metadata.function_tolerance;
  options.gradient_tolerance = snapshot.metadata.gradient_tolerance;
  options.parameter_tolerance = snapshot.metadata.parameter_tolerance;
  if (!std::isfinite(initial_lambda) || initial_lambda <= 0.0) {
    if (error != nullptr) {
      *error = "Custom LM initial lambda must be finite and positive";
    }
    return false;
  }
  options.initial_trust_region_radius = 1.0 / initial_lambda;
  return RunCustomCpuSolve(snapshot, options, mode, result, final_state,
                           error);
}

bool RunCustomCpuSolve(const Snapshot& snapshot,
                       const CustomCpuSolverOptions& options,
                       CustomCpuSolveMode mode,
                       CustomCpuSolveResult* result,
                       Snapshot* final_state,
                       std::string* error) {
  if (result == nullptr || final_state == nullptr || error == nullptr) {
    return false;
  }
  *result = CustomCpuSolveResult();
  result->solve_mode = mode;
  *final_state = snapshot;
  const bool single_step = mode != CustomCpuSolveMode::kFull;
  const bool use_ceres14_math =
      mode != CustomCpuSolveMode::kLegacySingleStep;
  if (options.linear_solver_type != "DENSE_SCHUR") {
    *error = "custom_cpu phase5p6 supports only DENSE_SCHUR";
    return false;
  }
  if (options.residual_order != "source_insertion_order" ||
      options.parameter_order != "source_first_insertion_order") {
    *error = "custom_cpu requires captured source residual/parameter order";
    return false;
  }
  if (!options.jacobi_scaling || options.use_nonmonotonic_steps ||
      options.use_inner_iterations) {
    *error = "custom_cpu phase5p6 supports jacobi-scaled monotonic LM only";
    return false;
  }
  if (!std::isfinite(options.initial_trust_region_radius) ||
      options.initial_trust_region_radius <= 0.0 ||
      !std::isfinite(options.min_trust_region_radius) ||
      options.min_trust_region_radius < 0.0 ||
      !std::isfinite(options.max_trust_region_radius) ||
      options.max_trust_region_radius <= 0.0 ||
      !std::isfinite(options.max_solver_time_in_seconds) ||
      options.max_solver_time_in_seconds <= 0.0 ||
      options.initial_trust_region_radius > options.max_trust_region_radius ||
      options.min_trust_region_radius > options.initial_trust_region_radius ||
      options.max_num_consecutive_invalid_steps <= 0) {
    *error = "Custom LM effective options are invalid";
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(options.loss_type, options.loss_scale, &loss, error)) {
    return false;
  }
  loss.reduction_threads = std::max(1, options.effective_num_threads);
  result->effective_options = options;
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, AssemblyOrder::kSource, &layout, error)) {
    return false;
  }
  BlockSystem system;
  if (!LinearizeState(*final_state, layout, EvaluationSource::kAnalytic,
                      AssemblyOrder::kSource, loss, &system, error)) {
    return false;
  }
  CpuGradientNorms gradient_norms;
  if (!ComputeGradientNorms(*final_state, layout, system, &gradient_norms,
                            error)) {
    return false;
  }
  double projected_gradient_norm = gradient_norms.projected_max_norm;
  double projected_gradient_l2_norm = gradient_norms.projected_l2_norm;
  double scaled_gradient_norm = gradient_norms.scaled_max_norm;
  result->initial_cost = system.cost;
  result->initial_projected_gradient_max_norm = projected_gradient_norm;
  result->initial_scaled_gradient_norm = scaled_gradient_norm;
  result->function_tolerance = options.function_tolerance;
  result->gradient_tolerance = options.gradient_tolerance;
  result->parameter_tolerance = options.parameter_tolerance;
  result->initial_lidar_near_zero = ClassifyLidarNearZero(snapshot);
  const auto solve_start = std::chrono::steady_clock::now();
  const double max_radius = options.max_trust_region_radius;
  const double min_radius = options.min_trust_region_radius;
  double radius = options.initial_trust_region_radius;
  if (!std::isfinite(radius) || radius <= 0.0 || radius > max_radius) {
    *error = "Custom LM initial trust-region radius is invalid";
    return false;
  }
  double decrease_factor = 2.0;
  int32_t consecutive_invalid_steps = 0;
  const int32_t max_iterations =
      single_step ? 1 : std::max(0, options.max_num_iterations);
  result->max_trial_iterations = max_iterations;
  const int32_t invalid_limit = options.max_num_consecutive_invalid_steps;
  const DampingDiagonal jacobi_scaling =
      ComputeInitialJacobiScaling(layout, system);
  DampingDiagonal damping;
  bool refresh_damping = true;

  CustomCpuIteration iteration_zero;
  iteration_zero.iteration = 0;
  iteration_zero.cost_before = system.cost;
  iteration_zero.trial_cost = system.cost;
  iteration_zero.cost_after = system.cost;
  iteration_zero.projected_gradient_max_norm = projected_gradient_norm;
  iteration_zero.scaled_gradient_norm = scaled_gradient_norm;
  iteration_zero.gradient_norm = projected_gradient_l2_norm;
  iteration_zero.radius_before = radius;
  iteration_zero.radius_after = radius;
  iteration_zero.lambda_before = 1.0 / radius;
  iteration_zero.lambda_after = 1.0 / radius;
  iteration_zero.factorization_success = true;
  iteration_zero.step_valid = true;
  iteration_zero.trial_finite = true;
  iteration_zero.accepted = true;
  iteration_zero.eta = options.eta;
  iteration_zero.successful_steps = 1;
  result->trace.push_back(iteration_zero);
  result->accepted_steps = 1;
  if (options.capture_state_trace) {
    result->accepted_state_trace.push_back(*final_state);
  }

  const auto elapsed_seconds = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         solve_start)
        .count();
  };
  if (!single_step &&
      elapsed_seconds() >= options.max_solver_time_in_seconds) {
    result->success = true;
    result->termination_type = CpuTerminationType::kNoConvergence;
    result->termination_reason = "maximum_solver_time_at_iteration_0";
  } else if (!single_step && max_iterations == 0) {
    result->success = true;
    result->termination_type = CpuTerminationType::kNoConvergence;
    result->termination_reason = "maximum_trial_iterations_at_iteration_0";
  } else if (!single_step &&
             projected_gradient_norm <=
                 options.gradient_tolerance) {
    result->success = true;
    result->termination_type = CpuTerminationType::kConvergence;
    result->termination_reason = "gradient_tolerance_at_iteration_0";
  } else if (!single_step && radius <= min_radius) {
    result->success = true;
    result->termination_type = CpuTerminationType::kConvergence;
    result->termination_reason = "minimum_trust_region_radius_at_iteration_0";
  }

  while (result->termination_reason.empty()) {
    if (refresh_damping) {
      damping = use_ceres14_math
                    ? ComputeCeres14Damping(layout, system, jacobi_scaling,
                                            options.min_lm_diagonal,
                                            options.max_lm_diagonal)
                    : ComputeDamping(layout, system, 1e-12);
      refresh_damping = false;
    }
    CustomCpuIteration iteration;
    iteration.iteration = result->trial_iterations + 1;
    iteration.cost_before = system.cost;
    iteration.cost_after = system.cost;
    iteration.trial_cost = system.cost;
    iteration.projected_gradient_max_norm = projected_gradient_norm;
    iteration.scaled_gradient_norm = scaled_gradient_norm;
    iteration.gradient_norm = projected_gradient_l2_norm;
    iteration.radius_before = radius;
    iteration.radius_after = iteration.radius_before;
    iteration.lambda_before = 1.0 / radius;
    if (!use_ceres14_math) {
      ComputeDirectLmDiagonalRange(
          damping, iteration.lambda_before, &iteration.lm_diagonal_min,
          &iteration.lm_diagonal_max);
    } else {
      ComputeCeres14LmDiagonalRange(damping, jacobi_scaling, radius,
                                    &iteration.lm_diagonal_min,
                                    &iteration.lm_diagonal_max);
    }
    iteration.lm_diagonal_observed = true;
    if (result->trial_iterations == 0) {
      result->first_lm_diagonal =
          use_ceres14_math
              ? ComputeCeres14LmDiagonalValues(damping, jacobi_scaling, radius)
              : ComputeDirectLmDiagonalValues(damping,
                                              iteration.lambda_before);
      result->first_lm_diagonal_sha256 =
          LmDiagonalSha256(result->first_lm_diagonal);
    }

    SolveData solve;
    std::string solve_error;
    const bool solved = BuildAndSolveSchur(
        layout, system, damping, iteration.lambda_before, &solve,
        &solve_error);
    ++result->trial_iterations;
    ++result->num_linear_solves;
    iteration.linear_solver_iterations = solved ? 1 : 0;
    iteration.factorization_success = solved;
    iteration.predicted_reduction =
        solved ? solve.predicted_reduction : 0.0;
    const bool model_step_valid =
        solved && std::isfinite(solve.predicted_reduction) &&
        solve.predicted_reduction > 0.0;
    iteration.step_valid = model_step_valid;
    if (!model_step_valid) {
      if (!solved) ++result->factorization_failures;
      ++result->invalid_steps;
      ++consecutive_invalid_steps;
      iteration.invalid = true;
      iteration.eta = options.eta;
      if (consecutive_invalid_steps >= invalid_limit) {
        result->termination_type = CpuTerminationType::kFailure;
        result->termination_reason =
            "maximum_consecutive_invalid_steps: " + solve_error;
        break;
      }
      radius /= decrease_factor;
      decrease_factor *= 2.0;
      iteration.radius_after = radius;
      iteration.lambda_after = 1.0 / radius;
      iteration.cost_after = system.cost;
      ++result->rejected_steps;
      iteration.successful_steps = result->accepted_steps;
      iteration.unsuccessful_steps = result->rejected_steps;
      iteration.cumulative_invalid_steps = result->invalid_steps;
      result->trace.push_back(iteration);
      if (!single_step &&
          elapsed_seconds() >= options.max_solver_time_in_seconds) {
        result->success = true;
        result->termination_type = CpuTerminationType::kNoConvergence;
        result->termination_reason = "maximum_solver_time";
      } else if (result->trial_iterations >= max_iterations) {
        result->success = true;
        result->termination_type = CpuTerminationType::kNoConvergence;
        result->termination_reason = "maximum_trial_iterations";
      } else if (radius <= min_radius) {
        result->success = true;
        result->termination_type = CpuTerminationType::kConvergence;
        result->termination_reason = "minimum_trust_region_radius";
      }
      continue;
    }
    consecutive_invalid_steps = 0;

    iteration.backward_error = solve.backward_error;
    Snapshot trial = *final_state;
    std::string trial_error;
    bool trial_valid = ApplyDelta(layout, solve, EvaluationSource::kAnalytic,
                                  use_ceres14_math, &trial, &trial_error);
    double trial_cost = std::numeric_limits<double>::max();
    if (trial_valid) {
      double evaluated_cost = 0.0;
      if (EvaluateTotalCost(trial, EvaluationSource::kAnalytic,
                            AssemblyOrder::kSource, loss, &evaluated_cost,
                            &trial_error) &&
          std::isfinite(evaluated_cost)) {
        trial_cost = evaluated_cost;
      } else {
        trial_valid = false;
      }
    }
    iteration.trial_cost = trial_cost;
    iteration.trial_finite = trial_valid && std::isfinite(trial_cost);
    iteration.actual_reduction = system.cost - trial_cost;
    iteration.rho =
        iteration.actual_reduction / solve.predicted_reduction;
    iteration.step_norm = trial_valid
                              ? AmbientVariableDifferenceNorm(
                                    *final_state, trial, layout)
                              : std::numeric_limits<double>::max();
    const double state_norm = AmbientVariableStateNorm(*final_state, layout);
    iteration.parameter_metric =
        iteration.step_norm /
        (state_norm + options.parameter_tolerance);
    iteration.function_metric = std::abs(iteration.actual_reduction) /
                                std::max(system.cost, 1e-300);

    const double parameter_threshold =
        options.parameter_tolerance *
        (state_norm + options.parameter_tolerance);
    if (!single_step && iteration.step_norm <= parameter_threshold) {
      result->success = true;
      result->termination_type = CpuTerminationType::kConvergence;
      result->termination_reason = "parameter_tolerance";
      break;
    }
    const double function_threshold =
        options.function_tolerance * system.cost;
    if (!single_step &&
        std::abs(iteration.actual_reduction) <= function_threshold) {
      result->success = true;
      result->termination_type = CpuTerminationType::kConvergence;
      result->termination_reason = "function_tolerance";
      break;
    }

    iteration.accepted = std::isfinite(iteration.rho) &&
                         iteration.rho > options.min_relative_decrease;

    if (iteration.accepted) {
      ++result->accepted_steps;
      *final_state = std::move(trial);
      if (options.capture_state_trace) {
        result->accepted_state_trace.push_back(*final_state);
      }
      const double update =
          std::max(1.0 / 3.0,
                   1.0 - std::pow(2.0 * iteration.rho - 1.0, 3.0));
      radius = std::min(max_radius, radius / update);
      decrease_factor = 2.0;
      iteration.cost_after = trial_cost;
      system = BlockSystem();
      if (!LinearizeState(*final_state, layout, EvaluationSource::kAnalytic,
                          AssemblyOrder::kSource, loss, &system, error)) {
        result->termination_type = CpuTerminationType::kFailure;
        result->termination_reason = "accepted_state_relinearization_failed";
        break;
      }
      if (!ComputeGradientNorms(*final_state, layout, system,
                                &gradient_norms, error)) {
        result->termination_type = CpuTerminationType::kFailure;
        result->termination_reason =
            "accepted_state_projected_gradient_failed";
        break;
      }
      projected_gradient_norm = gradient_norms.projected_max_norm;
      projected_gradient_l2_norm = gradient_norms.projected_l2_norm;
      scaled_gradient_norm = gradient_norms.scaled_max_norm;
      refresh_damping = true;
    } else {
      ++result->rejected_steps;
      radius /= decrease_factor;
      decrease_factor *= 2.0;
      // Ceres starts an ordinary trial IterationSummary with zero gradient
      // fields and only populates them after an accepted relinearization.
      iteration.projected_gradient_max_norm = 0.0;
      iteration.gradient_norm = 0.0;
    }
    if (iteration.accepted) {
      iteration.projected_gradient_max_norm = projected_gradient_norm;
      iteration.gradient_norm = projected_gradient_l2_norm;
    }
    iteration.scaled_gradient_norm = scaled_gradient_norm;
    iteration.radius_after = radius;
    iteration.lambda_after = 1.0 / radius;
    iteration.successful_steps = result->accepted_steps;
    iteration.unsuccessful_steps = result->rejected_steps;
    iteration.cumulative_invalid_steps = result->invalid_steps;
    result->trace.push_back(iteration);

    if (single_step) {
      result->success = true;
      result->termination_type = CpuTerminationType::kNoConvergence;
      result->termination_reason =
          mode == CustomCpuSolveMode::kCeres14SingleStep
              ? "ceres14_single_step_complete"
              : "legacy_single_step_complete";
    } else if (elapsed_seconds() >= options.max_solver_time_in_seconds) {
      result->success = true;
      result->termination_type = CpuTerminationType::kNoConvergence;
      result->termination_reason = "maximum_solver_time";
    } else if (result->trial_iterations >= max_iterations) {
      result->success = true;
      result->termination_type = CpuTerminationType::kNoConvergence;
      result->termination_reason = "maximum_trial_iterations";
    } else if (iteration.accepted &&
               projected_gradient_norm <=
                   options.gradient_tolerance) {
      result->success = true;
      result->termination_type = CpuTerminationType::kConvergence;
      result->termination_reason = "gradient_tolerance";
    } else if (radius <= min_radius) {
      result->success = true;
      result->termination_type = CpuTerminationType::kConvergence;
      result->termination_reason = "minimum_trust_region_radius";
    }
  }
  if (!result->trace.empty()) {
    result->trace.back().termination_reason = result->termination_reason;
  }
  BlockSystem final_system;
  if (!LinearizeState(*final_state, layout, EvaluationSource::kAnalytic,
                      AssemblyOrder::kSource, loss, &final_system, error)) {
    result->success = false;
    result->termination_type = CpuTerminationType::kFailure;
    result->termination_reason = "final_state_linearization_failed";
    return false;
  }
  result->final_cost = final_system.cost;
  CpuGradientNorms final_gradient_norms;
  if (!ComputeGradientNorms(*final_state, layout, final_system,
                            &final_gradient_norms, error)) {
    result->success = false;
    result->termination_type = CpuTerminationType::kFailure;
    result->termination_reason = "final_projected_gradient_failed";
    return false;
  }
  result->final_projected_gradient_max_norm =
      final_gradient_norms.projected_max_norm;
  result->final_scaled_gradient_norm = final_gradient_norms.scaled_max_norm;
  result->final_lambda = 1.0 / radius;
  result->final_lidar_near_zero = ClassifyLidarNearZero(*final_state);
  return true;
}

namespace {

std::string JoinOraclePath(const std::string& directory,
                           const std::string& filename) {
  if (directory.empty() || directory.back() == '/') return directory + filename;
  return directory + "/" + filename;
}

bool ReadCeresDampingDump(const std::string& directory,
                          CeresCpuSolveResult* result,
                          std::string* error) {
  const std::string base = "ceres_solver_iteration_001";
  const std::string damping_path = JoinOraclePath(directory, base + "_D.txt");
  std::ifstream input(damping_path, std::ios::binary);
  if (!input.is_open()) {
    *error = "Ceres damping oracle did not produce " + damping_path;
    return false;
  }
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    *error = "Failed to read Ceres damping oracle " + damping_path;
    return false;
  }
  std::istringstream values(contents);
  double value = 0.0;
  while (values >> value) {
    if (!std::isfinite(value) || value < 0.0) {
      *error = "Ceres damping oracle contains an invalid LM diagonal";
      return false;
    }
    result->damping_oracle_lm_diagonal.push_back(value);
  }
  if (!values.eof() || result->damping_oracle_lm_diagonal.empty()) {
    *error = "Ceres damping oracle is empty or malformed: " + damping_path;
    return false;
  }
  result->damping_oracle_observed = true;
  result->damping_oracle_source =
      "ceres_1.14_trust_region_problem_dump_solve_options_D";
  result->damping_oracle_path = damping_path;
  result->damping_oracle_sha256 = Sha256Hex(contents);
  result->damping_oracle_text_precision =
      "Ceres 1.14 WriteArrayToFileOrDie %17f (six fractional digits)";

  const std::string matrix_path = JoinOraclePath(directory, base + "_A.txt");
  std::ifstream matrix_input(matrix_path);
  if (!matrix_input.is_open()) {
    *error = "Ceres damping oracle did not produce " + matrix_path;
    return false;
  }
  std::vector<long double> scaled_column_norms(
      result->damping_oracle_lm_diagonal.size(), 0.0L);
  int64_t row = 0;
  int64_t column = 0;
  double matrix_value = 0.0;
  while (matrix_input >> row >> column >> matrix_value) {
    if (row < 0 || column < 0 ||
        static_cast<size_t>(column) >= scaled_column_norms.size() ||
        !std::isfinite(matrix_value)) {
      *error = "Ceres scaled-Jacobian dump contains an invalid triplet";
      return false;
    }
    scaled_column_norms[static_cast<size_t>(column)] +=
        static_cast<long double>(matrix_value) * matrix_value;
    ++result->damping_oracle_scaled_jacobian_nonzeros;
  }
  if (!matrix_input.eof()) {
    *error = "Ceres scaled-Jacobian dump is malformed: " + matrix_path;
    return false;
  }
  result->damping_oracle_scaled_column_norm_min =
      std::numeric_limits<double>::max();
  for (const long double squared_norm : scaled_column_norms) {
    const double value = static_cast<double>(squared_norm);
    result->damping_oracle_scaled_column_norm_min =
        std::min(result->damping_oracle_scaled_column_norm_min, value);
    result->damping_oracle_scaled_column_norm_max =
        std::max(result->damping_oracle_scaled_column_norm_max, value);
  }
  if (scaled_column_norms.empty()) {
    result->damping_oracle_scaled_column_norm_min = 0.0;
  }
  result->damping_oracle_clamped_diagonal_min =
      std::numeric_limits<double>::max();
  for (const double lm_diagonal : result->damping_oracle_lm_diagonal) {
    const double clamped = lm_diagonal * lm_diagonal *
                           result->initial_trust_region_radius;
    result->damping_oracle_clamped_diagonal_min =
        std::min(result->damping_oracle_clamped_diagonal_min, clamped);
    result->damping_oracle_clamped_diagonal_max =
        std::max(result->damping_oracle_clamped_diagonal_max, clamped);
  }

  bool removed = true;
  for (const std::string& suffix : {"_A.txt", "_b.txt", "_x.txt", ".m"}) {
    const std::string path = JoinOraclePath(directory, base + suffix);
    if (std::remove(path.c_str()) != 0) removed = false;
  }
  result->damping_oracle_auxiliary_files_removed = removed;
  return true;
}

}  // namespace

bool RunCeresCpuSolve(const Snapshot& snapshot,
                      bool single_step,
                      const std::string& damping_oracle_directory,
                      CeresCpuSolveResult* result,
                      Snapshot* final_state,
                      std::string* error) {
  if (result == nullptr || final_state == nullptr || error == nullptr) {
    return false;
  }
  *result = CeresCpuSolveResult();
  *final_state = snapshot;
  result->residual_order = "source_insertion_order";
  result->initial_lidar_near_zero = ClassifyLidarNearZero(snapshot);
  CanonicalLayout layout;
  if (!BuildLayout(snapshot, AssemblyOrder::kSource, &layout, error)) {
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(snapshot.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : snapshot.metadata.loss_function,
                       1.0, &loss, error)) {
    return false;
  }
  result->ba_image_count = CountBaImages(snapshot);
  if (result->ba_image_count == 0) {
    *error = "Ceres oracle snapshot has no selected BA images";
    return false;
  }
  loss.reduction_threads = 1;
  SnapshotLookup lookup;
  if (!BuildLookup(*final_state, &lookup, error)) return false;
  ceres::Problem problem;
  ceres::LossFunction* loss_function =
      loss.kind == LossKind::kSoftL1
          ? static_cast<ceres::LossFunction*>(new ceres::SoftLOneLoss(loss.scale))
          : nullptr;
  for (const OrderEntrySnapshot& entry : final_state->source_insertion_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      const ObservationSnapshot* observation =
          lookup.observations_by_source[entry.source_index];
      if (observation == nullptr) {
        *error = "Ceres replay is missing visual residual";
        return false;
      }
      ImageSnapshot* image = const_cast<ImageSnapshot*>(
          lookup.images.at(observation->image_id));
      PointSnapshot* point = const_cast<PointSnapshot*>(
          lookup.points.at(observation->point3D_id));
      CameraSnapshot* camera =
          const_cast<CameraSnapshot*>(lookup.cameras.at(image->camera_id));
      ceres::CostFunction* cost = nullptr;
      if (image->pose_constant || observation->pose_constant) {
        cost = BundleAdjustmentConstantPoseCostFunction<
            OpenCVCameraModel>::Create(
            Eigen::Vector4d(image->qvec[0], image->qvec[1], image->qvec[2],
                            image->qvec[3]),
            Eigen::Vector3d(image->tvec[0], image->tvec[1], image->tvec[2]),
            Eigen::Vector2d(observation->xy[0], observation->xy[1]));
        problem.AddResidualBlock(cost, loss_function, point->xyz.data(),
                                 camera->params.data());
      } else {
        cost = BundleAdjustmentCostFunction<OpenCVCameraModel>::Create(
            Eigen::Vector2d(observation->xy[0], observation->xy[1]));
        problem.AddResidualBlock(cost, loss_function, image->qvec.data(),
                                 image->tvec.data(), point->xyz.data(),
                                 camera->params.data());
      }
    } else {
      const LidarSnapshot* lidar = lookup.lidar_by_source[entry.source_index];
      if (lidar == nullptr) {
        *error = "Ceres replay is missing LiDAR residual";
        return false;
      }
      PointSnapshot* point =
          const_cast<PointSnapshot*>(lookup.points.at(lidar->point3D_id));
      Eigen::Matrix<double, 4, 1> plane;
      for (size_t i = 0; i < 4; ++i) plane[i] = lidar->plane[i];
      problem.AddResidualBlock(
          BundleAdjustmentLidarCostFunction::Create(plane, lidar->weight),
          loss_function, point->xyz.data());
    }
  }
  for (const PoseVariable& pose : layout.poses) {
    ImageSnapshot* image =
        const_cast<ImageSnapshot*>(lookup.images.at(pose.image_id));
    SetQuaternionManifold(&problem, image->qvec.data());
    std::vector<int> constant_translation;
    for (int index = 0; index < 3; ++index) {
      if ((image->constant_tvec_mask & (1u << index)) != 0) {
        constant_translation.push_back(index);
      }
    }
    if (!constant_translation.empty()) {
      SetSubsetManifold(3, constant_translation, &problem,
                        image->tvec.data());
    }
  }
  for (const CameraSnapshot& const_camera : final_state->cameras) {
    CameraSnapshot* camera = const_cast<CameraSnapshot*>(&const_camera);
    if (problem.HasParameterBlock(camera->params.data())) {
      problem.SetParameterBlockConstant(camera->params.data());
    }
  }
  for (const PointSnapshot& const_point : final_state->points) {
    PointSnapshot* point = const_cast<PointSnapshot*>(&const_point);
    if (point->constant && problem.HasParameterBlock(point->xyz.data())) {
      problem.SetParameterBlockConstant(point->xyz.data());
    }
  }

  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type =
      SelectOriginalLinearSolver(result->ba_image_count);
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.max_num_iterations =
      single_step ? 1 : std::max(0, snapshot.metadata.max_num_iterations);
  if (snapshot.metadata.max_linear_solver_iterations > 0) {
    options.max_linear_solver_iterations =
        snapshot.metadata.max_linear_solver_iterations;
  }
  options.function_tolerance = snapshot.metadata.function_tolerance;
  options.gradient_tolerance = snapshot.metadata.gradient_tolerance;
  options.parameter_tolerance = snapshot.metadata.parameter_tolerance;
  if (snapshot.metadata.max_consecutive_invalid_steps > 0) {
    options.max_num_consecutive_invalid_steps =
        snapshot.metadata.max_consecutive_invalid_steps;
  }
  options.num_threads = 1;
  options.num_linear_solver_threads = 1;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;
  options.use_inner_iterations = false;
  options.use_nonmonotonic_steps = false;
  options.jacobi_scaling = true;
  options.initial_trust_region_radius = 1e4;
  options.min_trust_region_radius = 1e-32;
  options.max_trust_region_radius = 1e16;
  options.min_lm_diagonal = 1e-6;
  options.max_lm_diagonal = 1e32;
  options.min_relative_decrease = 1e-3;
  if (options.linear_solver_type == ceres::ITERATIVE_SCHUR) {
    options.preconditioner_type = ceres::SCHUR_JACOBI;
  }
  DeterministicOracleCallback callback(final_state);
  options.update_state_every_iteration = true;
  options.callbacks.push_back(&callback);
  if (!damping_oracle_directory.empty()) {
    options.trust_region_minimizer_iterations_to_dump = {1};
    options.trust_region_problem_dump_directory = damping_oracle_directory;
    options.trust_region_problem_dump_format_type = ceres::TEXTFILE;
  }

  result->requested_linear_solver_type =
      ceres::LinearSolverTypeToString(options.linear_solver_type);
  result->requested_num_threads = options.num_threads;
  result->jacobi_scaling = options.jacobi_scaling;
  result->initial_trust_region_radius = options.initial_trust_region_radius;
  result->min_trust_region_radius = options.min_trust_region_radius;
  result->max_trust_region_radius = options.max_trust_region_radius;
  result->min_lm_diagonal = options.min_lm_diagonal;
  result->max_lm_diagonal = options.max_lm_diagonal;
  result->min_relative_decrease = options.min_relative_decrease;
  result->max_consecutive_invalid_steps =
      options.max_num_consecutive_invalid_steps;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  result->success = summary.IsSolutionUsable();
  result->termination_type = CeresTermination(summary.termination_type);
  result->termination_reason = summary.message;
  result->initial_cost = summary.initial_cost;
  result->final_cost = summary.final_cost;
  result->iterations = static_cast<int32_t>(summary.iterations.size());
  result->accepted_steps = summary.num_successful_steps;
  result->rejected_steps = summary.num_unsuccessful_steps;
  result->invalid_steps = static_cast<int32_t>(std::count_if(
      summary.iterations.begin(), summary.iterations.end(),
      [](const ceres::IterationSummary& item) {
        return item.iteration > 0 && !item.step_is_valid;
      }));
  result->total_time_seconds = summary.total_time_in_seconds;
  result->actual_linear_solver_type =
      ceres::LinearSolverTypeToString(summary.linear_solver_type_used);
  result->effective_num_threads = summary.num_threads_used;
  if (!damping_oracle_directory.empty() &&
      !ReadCeresDampingDump(damping_oracle_directory, result, error)) {
    return false;
  }
  if (!BuildCeresOracleTrace(snapshot, layout, loss, summary, callback.states,
                             result->damping_oracle_lm_diagonal,
                             &result->trace, error)) {
    return false;
  }
  if (!result->trace.empty()) {
    result->final_gradient_max_norm =
        result->trace.back().projected_gradient_max_norm;
  }
  result->final_lidar_near_zero = ClassifyLidarNearZero(*final_state);
  return true;
}

namespace {

std::string TerminationCategory(CpuTerminationType type,
                                const std::string& reason,
                                bool single_step) {
  if (single_step && type == CpuTerminationType::kNoConvergence) {
    return "single_step_limit";
  }
  if (type == CpuTerminationType::kFailure) return "failure";
  std::string lower = reason;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char value) { return std::tolower(value); });
  if (lower.find("gradient") != std::string::npos) return "gradient";
  if (lower.find("function") != std::string::npos) return "function";
  if (lower.find("parameter") != std::string::npos) return "parameter";
  if (lower.find("maximum") != std::string::npos ||
      lower.find("max_iteration") != std::string::npos) {
    return "max_iterations";
  }
  if (lower.find("minimum trust") != std::string::npos ||
      lower.find("minimum_trust") != std::string::npos) {
    return "min_trust_region_radius";
  }
  return type == CpuTerminationType::kConvergence ? "other_convergence"
                                                  : "other_no_convergence";
}

void CompareCpuLmTraces(const CeresCpuSolveResult& reference_result,
                        const CustomCpuSolveResult& candidate_result,
                        CpuSolveComparisonResult* comparison) {
  const std::vector<CustomCpuIteration>& reference = reference_result.trace;
  const std::vector<CustomCpuIteration>& candidate = candidate_result.trace;
  const size_t common_size = std::min(reference.size(), candidate.size());
  const bool single_step =
      candidate_result.solve_mode != CustomCpuSolveMode::kFull;
  comparison->ceres_termination_category = TerminationCategory(
      reference_result.termination_type, reference_result.termination_reason,
      single_step);
  comparison->custom_termination_category = TerminationCategory(
      candidate_result.termination_type, candidate_result.termination_reason,
      single_step);
  comparison->ceres_summary_accepted_steps = reference_result.accepted_steps;
  comparison->custom_summary_accepted_steps = candidate_result.accepted_steps;
  comparison->ceres_summary_rejected_steps = reference_result.rejected_steps;
  comparison->custom_summary_rejected_steps = candidate_result.rejected_steps;
  comparison->ceres_summary_invalid_steps = reference_result.invalid_steps;
  comparison->custom_summary_invalid_steps = candidate_result.invalid_steps;

  auto record_structure = [&](int32_t iteration, const std::string& field,
                              const std::string& reference_value,
                              const std::string& candidate_value) {
    if (comparison->trace_structure_mismatch_count == 0) {
      comparison->first_trace_structure_mismatch_iteration = iteration;
      comparison->first_trace_structure_mismatch_field = field;
      comparison->first_trace_structure_reference = reference_value;
      comparison->first_trace_structure_candidate = candidate_value;
    }
    ++comparison->trace_structure_mismatch_count;
  };
  if (reference.size() != candidate.size()) {
    record_structure(-1, "trace_length", std::to_string(reference.size()),
                     std::to_string(candidate.size()));
  }
  for (size_t i = 0; i < common_size; ++i) {
    const CustomCpuIteration& ref = reference[i];
    const CustomCpuIteration& cand = candidate[i];
    ++comparison->trace_structure_iterations_checked;
    const int32_t iteration = ref.iteration;
    if (ref.iteration != cand.iteration) {
      record_structure(iteration, "iteration", std::to_string(ref.iteration),
                       std::to_string(cand.iteration));
    }
    if (ref.step_valid != cand.step_valid) {
      record_structure(iteration, "step_valid",
                       ref.step_valid ? "true" : "false",
                       cand.step_valid ? "true" : "false");
    }
    if (ref.accepted != cand.accepted) {
      record_structure(iteration, "accepted",
                       ref.accepted ? "true" : "false",
                       cand.accepted ? "true" : "false");
    }
    if (ref.invalid != cand.invalid) {
      record_structure(iteration, "invalid", ref.invalid ? "true" : "false",
                       cand.invalid ? "true" : "false");
    }
    const bool ref_rejected = !ref.accepted && !ref.invalid;
    const bool cand_rejected = !cand.accepted && !cand.invalid;
    if (ref_rejected != cand_rejected) {
      record_structure(iteration, "rejected",
                       ref_rejected ? "true" : "false",
                       cand_rejected ? "true" : "false");
    }
    const bool ref_has_reason = !ref.termination_reason.empty();
    const bool cand_has_reason = !cand.termination_reason.empty();
    if (ref_has_reason != cand_has_reason) {
      record_structure(iteration, "termination_reason_presence",
                       ref_has_reason ? "true" : "false",
                       cand_has_reason ? "true" : "false");
    }
  }
  if (reference_result.termination_type != candidate_result.termination_type) {
    record_structure(-1, "termination_type",
                     CpuTerminationTypeName(reference_result.termination_type),
                     CpuTerminationTypeName(candidate_result.termination_type));
  }
  if (comparison->ceres_termination_category !=
      comparison->custom_termination_category) {
    record_structure(-1, "termination_category",
                     comparison->ceres_termination_category,
                     comparison->custom_termination_category);
  }
  if (reference_result.accepted_steps != candidate_result.accepted_steps) {
    record_structure(-1, "summary_accepted_steps",
                     std::to_string(reference_result.accepted_steps),
                     std::to_string(candidate_result.accepted_steps));
  }
  if (reference_result.rejected_steps != candidate_result.rejected_steps) {
    record_structure(-1, "summary_rejected_steps",
                     std::to_string(reference_result.rejected_steps),
                     std::to_string(candidate_result.rejected_steps));
  }
  if (reference_result.invalid_steps != candidate_result.invalid_steps) {
    record_structure(-1, "summary_invalid_steps",
                     std::to_string(reference_result.invalid_steps),
                     std::to_string(candidate_result.invalid_steps));
  }
  comparison->trace_structure_pass =
      comparison->trace_structure_mismatch_count == 0;

  auto record_numeric = [&](int32_t iteration, const std::string& field,
                            const std::string& reason, double reference_value,
                            double candidate_value, double atol, double rtol) {
    comparison->trace_numeric_within_tolerance = false;
    comparison->first_trace_divergence_iteration = iteration;
    comparison->first_trace_divergence_field = field;
    comparison->first_trace_divergence_reason = reason;
    comparison->first_trace_reference = reference_value;
    comparison->first_trace_candidate = candidate_value;
    comparison->first_trace_atol = atol;
    comparison->first_trace_rtol = rtol;
    comparison->first_trace_absolute_error =
        std::abs(reference_value - candidate_value);
    const double scale =
        std::max(std::abs(reference_value), std::abs(candidate_value));
    comparison->first_trace_relative_error =
        scale == 0.0 ? 0.0 : comparison->first_trace_absolute_error / scale;
  };
  bool found_numeric_divergence = false;
  for (size_t i = 0; i < common_size && !found_numeric_divergence; ++i) {
    const CustomCpuIteration& ref = reference[i];
    const CustomCpuIteration& cand = candidate[i];
    comparison->trace_numeric_iterations_scanned = i + 1;
    auto compare_value = [&](const std::string& field, double ref_value,
                             double cand_value, double atol, double rtol) {
      if (!std::isfinite(ref_value) || !std::isfinite(cand_value)) {
        if (ref_value == cand_value) return true;
        record_numeric(ref.iteration, field, "nonfinite_mismatch", ref_value,
                       cand_value, atol, rtol);
        return false;
      }
      const double threshold =
          atol + rtol * std::max(std::abs(ref_value), std::abs(cand_value));
      if (std::abs(ref_value - cand_value) <= threshold) return true;
      record_numeric(ref.iteration, field, "numeric_tolerance_exceeded",
                     ref_value, cand_value, atol, rtol);
      return false;
    };
    found_numeric_divergence =
        !compare_value("cost_before", ref.cost_before, cand.cost_before,
                       1e-10, 1e-10) ||
        !compare_value("projected_gradient_max_norm",
                       ref.projected_gradient_max_norm,
                       cand.projected_gradient_max_norm, 1e-8, 1e-7) ||
        !compare_value("projected_gradient_l2_norm", ref.gradient_norm,
                       cand.gradient_norm, 1e-8, 1e-7) ||
        !compare_value("scaled_gradient_norm", ref.scaled_gradient_norm,
                       cand.scaled_gradient_norm, 1e-10, 1e-7) ||
        !compare_value("radius_before", ref.radius_before,
                       cand.radius_before, 1e-15, 1e-12) ||
        !compare_value("lambda_before", ref.lambda_before,
                       cand.lambda_before, 1e-15, 1e-12) ||
        (ref.lm_diagonal_observed && cand.lm_diagonal_observed &&
         (!compare_value("lm_diagonal_min", ref.lm_diagonal_min,
                         cand.lm_diagonal_min, 5.1e-7, 1e-6) ||
          !compare_value("lm_diagonal_max", ref.lm_diagonal_max,
                         cand.lm_diagonal_max, 5.1e-7, 1e-6))) ||
        !compare_value("predicted_reduction", ref.predicted_reduction,
                       cand.predicted_reduction, 1e-10, 1e-8) ||
        !compare_value("trial_cost", ref.trial_cost, cand.trial_cost,
                       1e-10, 1e-10) ||
        !compare_value("rho", ref.rho, cand.rho, 1e-10, 1e-8) ||
        !compare_value("step_norm", ref.step_norm, cand.step_norm, 1e-12,
                       1e-9) ||
        !compare_value("cost_after", ref.cost_after, cand.cost_after,
                       1e-10, 1e-10) ||
        !compare_value("radius_after", ref.radius_after, cand.radius_after,
                       1e-15, 1e-12);
  }
}

void CompareIndependentDampingOracle(
    const CeresCpuSolveResult& ceres_result,
    const CustomCpuSolveResult& custom_result,
    CpuSolveComparisonResult* comparison) {
  comparison->damping_oracle_required =
      custom_result.solve_mode == CustomCpuSolveMode::kCeres14SingleStep;
  comparison->damping_oracle_available = ceres_result.damping_oracle_observed;
  comparison->damping_oracle_ceres_count =
      ceres_result.damping_oracle_lm_diagonal.size();
  comparison->damping_oracle_custom_count =
      custom_result.first_lm_diagonal.size();
  if (!comparison->damping_oracle_required) {
    comparison->damping_oracle_pass = true;
    return;
  }
  if (!comparison->damping_oracle_available ||
      comparison->damping_oracle_ceres_count !=
          comparison->damping_oracle_custom_count) {
    comparison->damping_oracle_pass = false;
    return;
  }
  std::vector<double> reference = ceres_result.damping_oracle_lm_diagonal;
  std::vector<double> candidate = custom_result.first_lm_diagonal;
  std::sort(reference.begin(), reference.end());
  std::sort(candidate.begin(), candidate.end());
  double worst_ratio = -1.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double absolute = std::abs(reference[i] - candidate[i]);
    const double scale = std::max(std::abs(reference[i]), std::abs(candidate[i]));
    const double relative = scale == 0.0 ? 0.0 : absolute / scale;
    const double threshold = comparison->damping_oracle_atol +
                             comparison->damping_oracle_rtol * scale;
    comparison->damping_oracle_max_absolute_error =
        std::max(comparison->damping_oracle_max_absolute_error, absolute);
    comparison->damping_oracle_max_relative_error =
        std::max(comparison->damping_oracle_max_relative_error, relative);
    if (absolute > threshold) ++comparison->damping_oracle_failures;
    const double ratio = threshold == 0.0 ? 0.0 : absolute / threshold;
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      comparison->damping_oracle_worst_id =
          "sorted_lm_diagonal=" + std::to_string(i);
      comparison->damping_oracle_worst_ceres = reference[i];
      comparison->damping_oracle_worst_custom = candidate[i];
    }
  }
  comparison->damping_oracle_pass =
      comparison->damping_oracle_failures == 0;
}

}  // namespace

bool CompareCpuSolveResults(const Snapshot& ceres_state,
                            const Snapshot& custom_state,
                            const CeresCpuSolveResult& ceres_result,
                            const CustomCpuSolveResult& custom_result,
                            CpuSolveComparisonResult* comparison,
                            std::string* error) {
  if (comparison == nullptr || error == nullptr) return false;
  *comparison = CpuSolveComparisonResult();
  CanonicalLayout ceres_layout;
  CanonicalLayout custom_layout;
  if (!BuildLayout(ceres_state, AssemblyOrder::kSource, &ceres_layout,
                   error) ||
      !BuildLayout(custom_state, AssemblyOrder::kSource, &custom_layout,
                   error)) {
    return false;
  }
  if (ceres_layout.sha256 != custom_layout.sha256) {
    *error = "Solved-state canonical layouts differ";
    return false;
  }
  LossConfig loss;
  if (!ParseLossConfig(ceres_state.metadata.loss_function.empty()
                           ? "TRIVIAL"
                           : ceres_state.metadata.loss_function,
                       custom_result.effective_options.loss_scale, &loss,
                       error)) {
    return false;
  }
  loss.reduction_threads = std::max(
      1, custom_result.effective_options.effective_num_threads);
  BlockSystem ceres_system;
  BlockSystem custom_system;
  if (!LinearizeState(ceres_state, ceres_layout,
                      EvaluationSource::kAutoDiff, AssemblyOrder::kSource,
                      loss, &ceres_system, error) ||
      !LinearizeState(custom_state, custom_layout,
                      EvaluationSource::kAnalytic, AssemblyOrder::kSource,
                      loss, &custom_system, error)) {
    return false;
  }
  comparison->ceres_scaled_gradient_norm =
      ScaledGradientNorm(ceres_layout, ceres_system, 1e-12);
  comparison->custom_scaled_gradient_norm =
      ScaledGradientNorm(custom_layout, custom_system, 1e-12);
  comparison->ceres_projected_gradient_max_norm =
      ceres_result.final_gradient_max_norm;
  comparison->custom_projected_gradient_max_norm =
      custom_result.final_projected_gradient_max_norm;
  CompareCpuLmTraces(ceres_result, custom_result, comparison);
  CompareIndependentDampingOracle(ceres_result, custom_result, comparison);
  comparison->cost_absolute_error =
      std::abs(ceres_result.final_cost - custom_result.final_cost);
  const double cost_scale =
      std::max(std::abs(ceres_result.final_cost),
               std::abs(custom_result.final_cost));
  comparison->cost_relative_error =
      cost_scale == 0.0 ? 0.0 : comparison->cost_absolute_error / cost_scale;

  std::unordered_map<uint32_t, const ImageSnapshot*> custom_images;
  std::unordered_map<uint64_t, const PointSnapshot*> custom_points;
  for (const ImageSnapshot& image : custom_state.images) {
    custom_images.emplace(image.image_id, &image);
  }
  for (const PointSnapshot& point : custom_state.points) {
    custom_points.emplace(point.point3D_id, &point);
  }
  std::vector<double> rotations;
  std::vector<double> translations;
  std::vector<double> points;
  std::vector<double> conditioned_points;
  const double radians_to_degrees = 180.0 / std::acos(-1.0);
  for (const ImageSnapshot& reference : ceres_state.images) {
    const auto candidate_it = custom_images.find(reference.image_id);
    if (candidate_it == custom_images.end()) {
      *error = "Custom solved state is missing image=" +
               std::to_string(reference.image_id);
      return false;
    }
    const ImageSnapshot& candidate = *candidate_it->second;
    double dot = 0.0;
    for (size_t i = 0; i < 4; ++i) dot += reference.qvec[i] * candidate.qvec[i];
    dot = std::max(-1.0, std::min(1.0, std::abs(dot)));
    const double rotation = 2.0 * std::acos(dot) * radians_to_degrees;
    const double dx = reference.tvec[0] - candidate.tvec[0];
    const double dy = reference.tvec[1] - candidate.tvec[1];
    const double dz = reference.tvec[2] - candidate.tvec[2];
    const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);
    rotations.push_back(rotation);
    translations.push_back(translation);
    if (rotation > comparison->rotation_max_degrees) {
      comparison->rotation_max_degrees = rotation;
      comparison->worst_rotation_id =
          "image=" + std::to_string(reference.image_id);
    }
    if (translation > comparison->translation_max) {
      comparison->translation_max = translation;
      comparison->worst_translation_id =
          "image=" + std::to_string(reference.image_id);
    }
  }
  for (const PointSnapshot& reference : ceres_state.points) {
    const auto candidate_it = custom_points.find(reference.point3D_id);
    if (candidate_it == custom_points.end()) {
      *error = "Custom solved state is missing point=" +
               std::to_string(reference.point3D_id);
      return false;
    }
    const PointSnapshot& candidate = *candidate_it->second;
    const double dx = reference.xyz[0] - candidate.xyz[0];
    const double dy = reference.xyz[1] - candidate.xyz[1];
    const double dz = reference.xyz[2] - candidate.xyz[2];
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    points.push_back(distance);
    double ceres_point_condition = 0.0;
    double custom_point_condition = 0.0;
    const auto ceres_point_index =
        ceres_layout.point_index.find(reference.point3D_id);
    const auto custom_point_index =
        custom_layout.point_index.find(reference.point3D_id);
    if (ceres_point_index != ceres_layout.point_index.end() &&
        custom_point_index != custom_layout.point_index.end()) {
      ceres_point_condition = SymmetricConditionEstimate(
          ceres_system.point_hessian[ceres_point_index->second]);
      custom_point_condition = SymmetricConditionEstimate(
          custom_system.point_hessian[custom_point_index->second]);
    }
    // The exclusion set is reference-only. Candidate conditioning is reported
    // diagnostically but can never decide which candidate coordinates are
    // admitted to the gate.
    const bool condition_excluded =
        ceres_point_condition >=
        comparison->point_condition_exclusion_threshold;
    if (condition_excluded) {
      ++comparison->condition_excluded_points;
      comparison->condition_excluded_point_ids.push_back(
          reference.point3D_id);
    } else {
      conditioned_points.push_back(distance);
      if (distance > comparison->conditioned_point_max) {
        comparison->conditioned_point_max = distance;
        comparison->worst_conditioned_point_id =
            "point3D=" + std::to_string(reference.point3D_id);
      }
    }
    if (distance > comparison->point_max) {
      comparison->point_max = distance;
      comparison->worst_point_id =
          "point3D=" + std::to_string(reference.point3D_id);
      comparison->worst_point_ceres_xyz = reference.xyz;
      comparison->worst_point_custom_xyz = candidate.xyz;
      comparison->worst_point_observation_count = 0;
      for (const ObservationSnapshot& observation :
           ceres_state.observations) {
        if (observation.point3D_id == reference.point3D_id) {
          ++comparison->worst_point_observation_count;
        }
      }
      comparison->worst_point_track_length = 0;
      for (const TrackElementSnapshot& track : ceres_state.tracks) {
        if (track.point3D_id == reference.point3D_id) {
          ++comparison->worst_point_track_length;
        }
      }
      comparison->worst_point_has_lidar = false;
      for (const LidarSnapshot& lidar : ceres_state.lidar) {
        if (lidar.point3D_id == reference.point3D_id) {
          comparison->worst_point_has_lidar = true;
          break;
        }
      }
      comparison->worst_point_ceres_block_condition =
          ceres_point_condition;
      comparison->worst_point_custom_block_condition =
          custom_point_condition;
    }
  }
  comparison->compared_poses = rotations.size();
  comparison->compared_points = points.size();
  comparison->rotation_p95_degrees = Percentile(rotations, 0.95);
  comparison->rotation_rms_degrees = RootMeanSquare(rotations);
  comparison->translation_p95 = Percentile(translations, 0.95);
  comparison->translation_rms = RootMeanSquare(translations);
  comparison->point_p95 = Percentile(points, 0.95);
  comparison->point_rms = RootMeanSquare(points);
  comparison->conditioned_point_p95 = Percentile(conditioned_points, 0.95);
  comparison->conditioned_point_rms = RootMeanSquare(conditioned_points);
  comparison->condition_excluded_fraction =
      points.empty()
          ? 0.0
          : static_cast<double>(comparison->condition_excluded_points) /
                static_cast<double>(points.size());

  comparison->strict_cost_pass =
      comparison->cost_absolute_error <= 1e-8 + 1e-6 * cost_scale;
  const double gradient_tolerance = custom_result.gradient_tolerance;
  const double projected_gradient_scale = std::max(
      std::abs(comparison->ceres_projected_gradient_max_norm),
      std::abs(comparison->custom_projected_gradient_max_norm));
  const bool projected_gradients_close =
      std::abs(comparison->ceres_projected_gradient_max_norm -
               comparison->custom_projected_gradient_max_norm) <=
      std::max(1e-3, 1e-8 + 1e-7 * projected_gradient_scale);
  const bool both_gradients_satisfy_tolerance =
      gradient_tolerance >= 0.0 &&
      comparison->ceres_projected_gradient_max_norm <= gradient_tolerance &&
      comparison->custom_projected_gradient_max_norm <= gradient_tolerance;
  comparison->gradient_pass =
      std::isfinite(comparison->ceres_projected_gradient_max_norm) &&
      std::isfinite(comparison->custom_projected_gradient_max_norm) &&
      (projected_gradients_close || both_gradients_satisfy_tolerance);
  comparison->termination_match =
      ceres_result.termination_type == custom_result.termination_type &&
      comparison->ceres_termination_category ==
          comparison->custom_termination_category;
  comparison->conditioned_point_strict_pass =
      comparison->conditioned_point_max <= 1e-5;
  comparison->provisional_state_pass =
      comparison->rotation_max_degrees <= 1e-4 &&
      comparison->translation_max <= 1e-6 &&
      comparison->conditioned_point_strict_pass;
  const bool quality_cost_pass =
      comparison->cost_absolute_error <= 1e-8 + 1e-3 * cost_scale;
  comparison->quality_diagnostic_pass =
      ceres_result.success && custom_result.success && quality_cost_pass &&
      comparison->gradient_pass && comparison->termination_match &&
      comparison->rotation_p95_degrees <= 0.05 &&
      comparison->rotation_max_degrees <= 0.2 &&
      comparison->translation_p95 <= 0.005 &&
      comparison->translation_max <= 0.02 &&
      comparison->point_rms <= 0.005 && comparison->point_p95 <= 0.01;
  comparison->pass =
      ceres_result.success && custom_result.success &&
      comparison->strict_cost_pass && comparison->gradient_pass &&
      comparison->termination_match && comparison->provisional_state_pass &&
      comparison->trace_structure_pass && comparison->damping_oracle_pass;
  return true;
}

std::string CustomCpuSolveJson(const CustomCpuSolveResult& result,
                               size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string indent2(indent_spaces * 2, ' ');
  const std::string indent3(indent_spaces * 3, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "\"custom_cpu_solve\": {\n";
  stream << indent2 << "\"solve_mode\": \""
         << CustomCpuSolveModeName(result.solve_mode) << "\", \"success\": "
         << (result.success ? "true" : "false")
         << ", \"termination_type\": \""
         << CpuTerminationTypeName(result.termination_type)
         << "\", \"termination_reason\": \""
         << EscapeJson(result.termination_reason) << "\",\n";
  stream << indent2 << "\"initial_cost\": " << result.initial_cost
         << ", \"final_cost\": " << result.final_cost
         << ", \"initial_projected_gradient_max_norm\": "
         << result.initial_projected_gradient_max_norm
         << ", \"final_projected_gradient_max_norm\": "
         << result.final_projected_gradient_max_norm
         << ", \"initial_scaled_gradient_norm_diagnostic\": "
         << result.initial_scaled_gradient_norm
         << ", \"final_scaled_gradient_norm_diagnostic\": "
         << result.final_scaled_gradient_norm << ", \"final_lambda\": "
         << result.final_lambda << ",\n";
  stream << indent2 << "\"stopping_config\": {\"function_tolerance\": "
         << result.function_tolerance << ", \"gradient_tolerance\": "
         << result.gradient_tolerance << ", \"parameter_tolerance\": "
         << result.parameter_tolerance << ", \"max_trial_iterations\": "
         << result.max_trial_iterations << "},\n";
  const CustomCpuSolverOptions& option = result.effective_options;
  stream << indent2 << "\"effective_options\": {\"linear_solver_type\": \""
         << EscapeJson(option.linear_solver_type)
         << "\", \"residual_order\": \""
         << EscapeJson(option.residual_order)
         << "\", \"parameter_order\": \""
         << EscapeJson(option.parameter_order) << "\", \"loss_type\": \""
         << EscapeJson(option.loss_type) << "\", \"loss_scale\": "
         << option.loss_scale << ", \"requested_num_threads\": "
         << option.requested_num_threads << ", \"effective_num_threads\": "
         << option.effective_num_threads
         << ", \"actual_parallel_regions\": 0"
         << ", \"max_num_iterations\": " << option.max_num_iterations
         << ", \"min_linear_solver_iterations\": "
         << option.min_linear_solver_iterations
         << ", \"max_linear_solver_iterations\": "
         << option.max_linear_solver_iterations
         << ", \"max_num_consecutive_invalid_steps\": "
         << option.max_num_consecutive_invalid_steps
         << ", \"max_solver_time_in_seconds\": "
         << option.max_solver_time_in_seconds
         << ", \"initial_trust_region_radius\": "
         << option.initial_trust_region_radius
         << ", \"min_trust_region_radius\": "
         << option.min_trust_region_radius
         << ", \"max_trust_region_radius\": "
         << option.max_trust_region_radius
         << ", \"min_lm_diagonal\": " << option.min_lm_diagonal
         << ", \"max_lm_diagonal\": " << option.max_lm_diagonal
         << ", \"min_relative_decrease\": "
         << option.min_relative_decrease << ", \"eta\": " << option.eta
         << ", \"jacobi_scaling\": "
         << (option.jacobi_scaling ? "true" : "false")
         << ", \"use_nonmonotonic_steps\": "
         << (option.use_nonmonotonic_steps ? "true" : "false")
         << ", \"use_inner_iterations\": "
         << (option.use_inner_iterations ? "true" : "false") << "},\n";
  stream << indent2 << "\"trial_iterations\": "
         << result.trial_iterations << ", \"accepted_steps\": "
         << result.accepted_steps << ", \"rejected_steps\": "
         << result.rejected_steps << ", \"factorization_failures\": "
         << result.factorization_failures << ", \"invalid_steps\": "
         << result.invalid_steps << ", \"num_linear_solves\": "
         << result.num_linear_solves << ",\n";
  stream << indent2 << "\"first_lm_diagonal\": {\"count\": "
         << result.first_lm_diagonal.size() << ", \"sha256\": \""
         << result.first_lm_diagonal_sha256 << "\"},\n";
  stream << indent2 << "\"lidar_near_zero\": {\"initial\": ";
  AppendLidarNearZeroJson(&stream, result.initial_lidar_near_zero);
  stream << ", \"final\": ";
  AppendLidarNearZeroJson(&stream, result.final_lidar_near_zero);
  stream << "},\n";
  stream << indent2 << "\"trace\": [\n";
  for (size_t i = 0; i < result.trace.size(); ++i) {
    const CustomCpuIteration& item = result.trace[i];
    stream << indent3 << "{\"iteration\": " << item.iteration
           << ", \"cost_before\": " << item.cost_before
           << ", \"trial_cost\": ";
    AppendJsonNumber(&stream, item.trial_cost);
    stream << ", \"cost_after\": " << item.cost_after
           << ", \"projected_gradient_max_norm\": "
           << item.projected_gradient_max_norm
           << ", \"scaled_gradient_norm\": " << item.scaled_gradient_norm
           << ", \"projected_gradient_l2_norm\": " << item.gradient_norm
           << ", \"radius_before\": " << item.radius_before
           << ", \"radius_after\": " << item.radius_after
           << ", \"lambda_before\": " << item.lambda_before
           << ", \"lambda_after\": " << item.lambda_after
           << ", \"lm_diagonal_observed\": "
           << (item.lm_diagonal_observed ? "true" : "false")
           << ", \"lm_diagonal_min\": " << item.lm_diagonal_min
           << ", \"lm_diagonal_max\": " << item.lm_diagonal_max
           << ", \"predicted_reduction\": "
           << item.predicted_reduction << ", \"actual_reduction\": ";
    AppendJsonNumber(&stream, item.actual_reduction);
    stream << ", \"rho\": " << item.rho
           << ", \"function_metric\": " << item.function_metric
           << ", \"parameter_metric\": " << item.parameter_metric
           << ", \"step_norm\": " << item.step_norm
           << ", \"backward_error\": " << item.backward_error
           << ", \"eta\": " << item.eta
           << ", \"linear_solver_iterations\": "
           << item.linear_solver_iterations
           << ", \"successful_steps\": " << item.successful_steps
           << ", \"unsuccessful_steps\": " << item.unsuccessful_steps
           << ", \"cumulative_invalid_steps\": "
           << item.cumulative_invalid_steps
           << ", \"factorization_success\": "
           << (item.factorization_success ? "true" : "false")
           << ", \"step_valid\": "
           << (item.step_valid ? "true" : "false")
           << ", \"trial_finite\": "
           << (item.trial_finite ? "true" : "false")
           << ", \"invalid\": " << (item.invalid ? "true" : "false")
           << ", \"rejected\": "
           << (!item.accepted && !item.invalid ? "true" : "false")
           << ", \"accepted\": "
           << (item.accepted ? "true" : "false")
           << ", \"termination_reason\": \""
           << EscapeJson(item.termination_reason) << "\"}";
    if (i + 1 != result.trace.size()) stream << ',';
    stream << '\n';
  }
  stream << indent2 << "]\n";
  stream << indent << '}';
  return stream.str();
}

std::string CeresCpuSolveJson(const CeresCpuSolveResult& result,
                              size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string indent2(indent_spaces * 2, ' ');
  const std::string indent3(indent_spaces * 3, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "\"ceres_cpu_solve\": {\n";
  stream << indent2 << "\"success\": "
         << (result.success ? "true" : "false")
         << ", \"termination_type\": \""
         << CpuTerminationTypeName(result.termination_type)
         << "\", \"termination_reason\": \""
         << EscapeJson(result.termination_reason) << "\",\n";
  stream << indent2 << "\"initial_cost\": " << result.initial_cost
         << ", \"final_cost\": " << result.final_cost
         << ", \"final_gradient_max_norm\": "
         << result.final_gradient_max_norm << ", \"iterations\": "
         << result.iterations << ", \"accepted_steps\": "
         << result.accepted_steps << ", \"rejected_steps\": "
         << result.rejected_steps << ", \"invalid_steps\": "
         << result.invalid_steps << ", \"total_time_seconds\": "
         << result.total_time_seconds << ",\n";
  stream << indent2 << "\"oracle_options\": {\"deterministic_oracle\": "
         << (result.deterministic_oracle ? "true" : "false")
         << ", \"original_threading_emulated\": false"
         << ", \"residual_order\": \"" << result.residual_order
         << "\", \"ba_image_count\": " << result.ba_image_count
         << ", \"requested_linear_solver_type\": \""
         << result.requested_linear_solver_type
         << "\", \"actual_linear_solver_type\": \""
         << result.actual_linear_solver_type
         << "\", \"requested_num_threads\": "
         << result.requested_num_threads
         << ", \"effective_num_threads\": "
         << result.effective_num_threads << ", \"jacobi_scaling\": "
         << (result.jacobi_scaling ? "true" : "false")
         << ", \"initial_trust_region_radius\": "
         << result.initial_trust_region_radius
         << ", \"min_trust_region_radius\": "
         << result.min_trust_region_radius
         << ", \"max_trust_region_radius\": "
         << result.max_trust_region_radius << ", \"min_lm_diagonal\": "
         << result.min_lm_diagonal << ", \"max_lm_diagonal\": "
         << result.max_lm_diagonal << ", \"min_relative_decrease\": "
         << result.min_relative_decrease
         << ", \"max_consecutive_invalid_steps\": "
         << result.max_consecutive_invalid_steps
         << ", \"trace_gradient_semantics\": "
            "\"current_accepted_state\", "
         << "\"trace_lm_diagonal_source\": "
            "\"actual_ceres_dump_iteration_1_or_unobserved\", "
         << "\"trace_predicted_reduction_source\": "
            "\"cost_change_divided_by_relative_decrease\"},\n";
  stream << indent2 << "\"damping_oracle\": {\"observed\": "
         << (result.damping_oracle_observed ? "true" : "false")
         << ", \"source\": \"" << EscapeJson(result.damping_oracle_source)
         << "\", \"path\": \"" << EscapeJson(result.damping_oracle_path)
         << "\", \"sha256\": \"" << result.damping_oracle_sha256
         << "\", \"text_precision\": \""
         << EscapeJson(result.damping_oracle_text_precision)
         << "\", \"lm_diagonal_count\": "
         << result.damping_oracle_lm_diagonal.size()
         << ", \"jacobi_scaling_observation\": "
            "\"post_scaling_A_column_squared_norms\""
         << ", \"scaled_jacobian_nonzeros\": "
         << result.damping_oracle_scaled_jacobian_nonzeros
         << ", \"scaled_column_squared_norm_min\": "
         << result.damping_oracle_scaled_column_norm_min
         << ", \"scaled_column_squared_norm_max\": "
         << result.damping_oracle_scaled_column_norm_max
         << ", \"clamped_diagonal_min_from_D\": "
         << result.damping_oracle_clamped_diagonal_min
         << ", \"clamped_diagonal_max_from_D\": "
         << result.damping_oracle_clamped_diagonal_max
         << ", \"auxiliary_files_removed\": "
         << (result.damping_oracle_auxiliary_files_removed ? "true" : "false")
         << "},\n";
  stream << indent2 << "\"lidar_near_zero\": {\"initial\": ";
  AppendLidarNearZeroJson(&stream, result.initial_lidar_near_zero);
  stream << ", \"final\": ";
  AppendLidarNearZeroJson(&stream, result.final_lidar_near_zero);
  stream << "},\n";
  stream << indent2 << "\"trace\": [\n";
  for (size_t i = 0; i < result.trace.size(); ++i) {
    const CustomCpuIteration& item = result.trace[i];
    stream << indent3 << "{\"iteration\": " << item.iteration
           << ", \"cost_before\": " << item.cost_before
           << ", \"trial_cost\": ";
    AppendJsonNumber(&stream, item.trial_cost);
    stream << ", \"cost_after\": " << item.cost_after
           << ", \"projected_gradient_max_norm\": "
           << item.projected_gradient_max_norm
           << ", \"projected_gradient_l2_norm\": " << item.gradient_norm
           << ", \"scaled_gradient_norm\": " << item.scaled_gradient_norm
           << ", \"radius_before\": " << item.radius_before
           << ", \"radius_after\": " << item.radius_after
           << ", \"lambda_before\": " << item.lambda_before
           << ", \"lambda_after\": " << item.lambda_after
           << ", \"lm_diagonal_observed\": "
           << (item.lm_diagonal_observed ? "true" : "false")
           << ", \"lm_diagonal_min\": " << item.lm_diagonal_min
           << ", \"lm_diagonal_max\": " << item.lm_diagonal_max
           << ", \"predicted_reduction\": "
           << item.predicted_reduction << ", \"actual_reduction\": ";
    AppendJsonNumber(&stream, item.actual_reduction);
    stream << ", \"rho\": " << item.rho
           << ", \"step_norm\": " << item.step_norm
           << ", \"factorization_success\": "
           << (item.factorization_success ? "true" : "false")
           << ", \"step_valid\": "
           << (item.step_valid ? "true" : "false")
           << ", \"trial_finite\": "
           << (item.trial_finite ? "true" : "false")
           << ", \"invalid\": " << (item.invalid ? "true" : "false")
           << ", \"rejected\": "
           << (!item.accepted && !item.invalid ? "true" : "false")
           << ", \"accepted\": "
           << (item.accepted ? "true" : "false")
           << ", \"termination_reason\": \""
           << EscapeJson(item.termination_reason) << "\"}";
    if (i + 1 != result.trace.size()) stream << ',';
    stream << '\n';
  }
  stream << indent2 << "]\n";
  stream << indent << '}';
  return stream.str();
}

std::string CpuSolveComparisonJson(const CpuSolveComparisonResult& result,
                                   size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string indent2(indent_spaces * 2, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "\"cpu_solve_comparison\": {\n";
  stream << indent2 << "\"cost_absolute_error\": "
         << result.cost_absolute_error << ", \"cost_relative_error\": "
         << result.cost_relative_error
         << ", \"ceres_projected_gradient_max_norm\": "
         << result.ceres_projected_gradient_max_norm
         << ", \"custom_projected_gradient_max_norm\": "
         << result.custom_projected_gradient_max_norm
         << ", \"ceres_scaled_gradient_norm\": "
         << result.ceres_scaled_gradient_norm
         << ", \"custom_scaled_gradient_norm\": "
         << result.custom_scaled_gradient_norm << ",\n";
  stream << indent2 << "\"trace_structure\": {\"pass\": "
         << (result.trace_structure_pass ? "true" : "false")
         << ", \"iterations_checked\": "
         << result.trace_structure_iterations_checked
         << ", \"mismatch_count\": "
         << result.trace_structure_mismatch_count
         << ", \"first_mismatch_iteration\": "
         << result.first_trace_structure_mismatch_iteration
         << ", \"first_mismatch_field\": \""
         << EscapeJson(result.first_trace_structure_mismatch_field)
         << "\", \"reference\": \""
         << EscapeJson(result.first_trace_structure_reference)
         << "\", \"candidate\": \""
         << EscapeJson(result.first_trace_structure_candidate)
         << "\", \"ceres_termination_category\": \""
         << result.ceres_termination_category
         << "\", \"custom_termination_category\": \""
         << result.custom_termination_category
         << "\", \"summary_counts\": {\"ceres_accepted\": "
         << result.ceres_summary_accepted_steps
         << ", \"custom_accepted\": "
         << result.custom_summary_accepted_steps
         << ", \"ceres_rejected\": "
         << result.ceres_summary_rejected_steps
         << ", \"custom_rejected\": "
         << result.custom_summary_rejected_steps
         << ", \"ceres_invalid\": "
         << result.ceres_summary_invalid_steps
         << ", \"custom_invalid\": "
         << result.custom_summary_invalid_steps << "}},\n";
  stream << indent2 << "\"trace_numeric_diagnostic\": {\"within_tolerance\": "
         << (result.trace_numeric_within_tolerance ? "true" : "false")
         << ", \"iterations_scanned_to_first_divergence\": "
         << result.trace_numeric_iterations_scanned
         << ", \"first_divergence_iteration\": "
         << result.first_trace_divergence_iteration
         << ", \"first_divergence_field\": \""
         << EscapeJson(result.first_trace_divergence_field)
         << "\", \"reason\": \""
         << EscapeJson(result.first_trace_divergence_reason)
         << "\", \"reference\": " << result.first_trace_reference
         << ", \"candidate\": " << result.first_trace_candidate
         << ", \"atol\": " << result.first_trace_atol
         << ", \"rtol\": " << result.first_trace_rtol
         << ", \"absolute_error\": " << result.first_trace_absolute_error
         << ", \"relative_error\": " << result.first_trace_relative_error
         << "},\n";
  stream << indent2 << "\"independent_damping_oracle\": {\"required\": "
         << (result.damping_oracle_required ? "true" : "false")
         << ", \"available\": "
         << (result.damping_oracle_available ? "true" : "false")
         << ", \"pass\": "
         << (result.damping_oracle_pass ? "true" : "false")
         << ", \"ceres_count\": " << result.damping_oracle_ceres_count
         << ", \"custom_count\": " << result.damping_oracle_custom_count
         << ", \"comparison_order\": \"sorted_multiset\""
         << ", \"atol\": " << result.damping_oracle_atol
         << ", \"rtol\": " << result.damping_oracle_rtol
         << ", \"failures\": " << result.damping_oracle_failures
         << ", \"max_absolute_error\": "
         << result.damping_oracle_max_absolute_error
         << ", \"max_relative_error\": "
         << result.damping_oracle_max_relative_error
         << ", \"worst_id\": \""
         << EscapeJson(result.damping_oracle_worst_id)
         << "\", \"worst_ceres\": " << result.damping_oracle_worst_ceres
         << ", \"worst_custom\": " << result.damping_oracle_worst_custom
         << "},\n";
  stream << indent2 << "\"poses\": {\"count\": "
         << result.compared_poses << ", \"rotation_max_degrees\": "
         << result.rotation_max_degrees
         << ", \"rotation_p95_degrees\": "
         << result.rotation_p95_degrees
         << ", \"rotation_rms_degrees\": "
         << result.rotation_rms_degrees << ", \"translation_max\": "
         << result.translation_max << ", \"translation_p95\": "
         << result.translation_p95 << ", \"translation_rms\": "
         << result.translation_rms << ", \"worst_rotation_id\": \""
         << EscapeJson(result.worst_rotation_id)
         << "\", \"worst_translation_id\": \""
         << EscapeJson(result.worst_translation_id) << "\"},\n";
  stream << indent2 << "\"points\": {\"total_point_count\": "
         << result.compared_points << ", \"raw_point_diagnostic_only\": true"
         << ", \"raw_point_max\": " << result.point_max
         << ", \"raw_point_p95\": " << result.point_p95
         << ", \"raw_point_rms\": " << result.point_rms
         << ", \"conditioned_point_max\": "
         << result.conditioned_point_max
         << ", \"conditioned_point_p95\": "
         << result.conditioned_point_p95
         << ", \"conditioned_point_rms\": "
         << result.conditioned_point_rms
         << ", \"excluded_point_count\": "
         << result.condition_excluded_points
         << ", \"excluded_fraction\": "
         << result.condition_excluded_fraction
         << ", \"worst_raw_point_id\": \""
         << EscapeJson(result.worst_point_id)
         << "\", \"ceres_xyz\": [" << result.worst_point_ceres_xyz[0]
         << ", " << result.worst_point_ceres_xyz[1] << ", "
         << result.worst_point_ceres_xyz[2] << "], \"custom_xyz\": ["
         << result.worst_point_custom_xyz[0] << ", "
         << result.worst_point_custom_xyz[1] << ", "
         << result.worst_point_custom_xyz[2]
         << "], \"ba_observation_count\": "
         << result.worst_point_observation_count
         << ", \"track_length\": " << result.worst_point_track_length
         << ", \"has_lidar\": "
         << (result.worst_point_has_lidar ? "true" : "false")
         << ", \"ceres_block_condition\": "
         << result.worst_point_ceres_block_condition
         << ", \"custom_block_condition\": "
         << result.worst_point_custom_block_condition
         << ", \"worst_conditioned_id\": \""
         << EscapeJson(result.worst_conditioned_point_id)
         << "\", \"condition_exclusion_threshold\": "
         << result.point_condition_exclusion_threshold
         << ", \"excluded_point_ids\": [";
  for (size_t i = 0; i < result.condition_excluded_point_ids.size(); ++i) {
    if (i != 0) stream << ", ";
    stream << result.condition_excluded_point_ids[i];
  }
  stream << "]},\n";
  stream << indent2 << "\"phase5_strict_gate\": {"
         << "\"cost_atol\": 1e-8, \"cost_rtol\": 1e-6, "
         << "\"rotation_max_degrees\": 1e-4, "
         << "\"translation_max\": 1e-6, "
         << "\"conditioned_point_max\": 1e-5, "
         << "\"strict_cost_pass\": "
         << (result.strict_cost_pass ? "true" : "false")
         << ", \"gradient_pass\": "
         << (result.gradient_pass ? "true" : "false")
         << ", \"termination_match\": "
         << (result.termination_match ? "true" : "false")
         << ", \"trace_structure_pass\": "
         << (result.trace_structure_pass ? "true" : "false")
         << ", \"damping_oracle_pass\": "
         << (result.damping_oracle_pass ? "true" : "false")
         << ", \"conditioned_point_strict_pass\": "
         << (result.conditioned_point_strict_pass ? "true" : "false")
         << ", \"provisional_state_pass\": "
         << (result.provisional_state_pass ? "true" : "false") << "},\n";
  stream << indent2 << "\"quality_diagnostic_thresholds\": {"
         << "\"cost_atol\": 1e-8, \"cost_rtol\": 1e-3, "
         << "\"rotation_p95_degrees\": 0.05, "
         << "\"rotation_max_degrees\": 0.2, "
         << "\"translation_p95\": 0.005, "
         << "\"translation_max\": 0.02, \"raw_point_rms\": 0.005, "
         << "\"raw_point_p95\": 0.01, \"pass\": "
         << (result.quality_diagnostic_pass ? "true" : "false")
         << "},\n";
  stream << indent2 << "\"pass\": " << (result.pass ? "true" : "false")
         << '\n';
  stream << indent << '}';
  return stream.str();
}

}  // namespace gpu_ba
}  // namespace colmap
