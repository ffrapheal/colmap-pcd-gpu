#include "gpu_ba/validation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ceres/local_parameterization.h>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

constexpr double kResidualAtol = 1e-12;
constexpr double kResidualRtol = 1e-10;
constexpr double kJacobianAtol = 1e-10;
constexpr double kJacobianRtol = 1e-7;
constexpr double kFiniteDifferenceAtol = 1e-8;
constexpr double kFiniteDifferenceRtol = 1e-5;

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

class ErrorAccumulator {
 public:
  ErrorAccumulator(std::string name, double atol, double rtol)
      : name_(std::move(name)), atol_(atol), rtol_(rtol) {}

  void Add(double reference, double candidate, const std::string& element_id) {
    ++count_;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      ++nonfinite_;
      ++failures_;
      if (worst_element_id_.empty()) {
        worst_element_id_ = element_id;
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
                                  ? (absolute == 0.0 ? 0.0
                                                     : std::numeric_limits<double>::max())
                                  : absolute / threshold;
    absolute_errors_.push_back(absolute);
    relative_errors_.push_back(relative);
    absolute_sum_ += absolute;
    absolute_squared_sum_ += absolute * absolute;
    relative_sum_ += relative;
    relative_squared_sum_ += relative * relative;
    if (absolute > threshold) ++failures_;
    if (worst_element_id_.empty() || gate_ratio > worst_gate_ratio_) {
      worst_gate_ratio_ = gate_ratio;
      worst_element_id_ = element_id;
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
    summary.worst_element_id = worst_element_id_;
    summary.worst_reference = worst_reference_;
    summary.worst_candidate = worst_candidate_;
    summary.pass = failures_ == 0 && nonfinite_ == 0;
    return summary;
  }

 private:
  std::string name_;
  double atol_;
  double rtol_;
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
  std::string worst_element_id_;
  double worst_reference_ = 0.0;
  double worst_candidate_ = 0.0;
};

template <size_t N>
void AddArray(ErrorAccumulator* accumulator,
              const std::array<double, N>& reference,
              const std::array<double, N>& candidate,
              const std::string& prefix) {
  for (size_t i = 0; i < N; ++i) {
    accumulator->Add(reference[i], candidate[i],
                     prefix + ":element=" + std::to_string(i));
  }
}

std::string ObservationId(const ObservationSnapshot& observation) {
  std::ostringstream stream;
  stream << "source=" << observation.source_index
         << ":image=" << observation.image_id
         << ":point2D=" << observation.point2D_idx
         << ":point3D=" << observation.point3D_id;
  return stream.str();
}

bool ResidualOnly(ceres::CostFunction* cost,
                  const std::array<double, 4>& quaternion,
                  const std::array<double, 3>& translation,
                  const std::array<double, 3>& point,
                  const std::array<double, 8>& camera,
                  std::array<double, 2>* residual) {
  const double* parameters[] = {quaternion.data(), translation.data(),
                                point.data(), camera.data()};
  if (!cost->Evaluate(parameters, residual->data(), nullptr)) return false;
  return std::isfinite((*residual)[0]) && std::isfinite((*residual)[1]);
}

double DifferenceStep(double relative_step, double value) {
  return relative_step * std::max(1.0, std::abs(value));
}

template <typename Evaluator>
bool FivePointDerivative(Evaluator evaluator,
                         double step,
                         std::array<double, 2>* derivative) {
  std::array<double, 2> plus_two;
  std::array<double, 2> plus_one;
  std::array<double, 2> minus_one;
  std::array<double, 2> minus_two;
  if (!evaluator(2.0 * step, &plus_two) ||
      !evaluator(step, &plus_one) ||
      !evaluator(-step, &minus_one) ||
      !evaluator(-2.0 * step, &minus_two)) {
    return false;
  }
  for (size_t row = 0; row < 2; ++row) {
    (*derivative)[row] =
        (-plus_two[row] + 8.0 * plus_one[row] - 8.0 * minus_one[row] +
         minus_two[row]) /
        (12.0 * step);
  }
  return std::isfinite((*derivative)[0]) &&
         std::isfinite((*derivative)[1]);
}

void AppendUint64Array(std::ostringstream* stream,
                       const std::vector<uint64_t>& values) {
  *stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) *stream << ", ";
    *stream << values[i];
  }
  *stream << "]";
}

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

}  // namespace

bool ValidateResidualsAndJacobians(
    const Snapshot& snapshot,
    const LinearizationValidationOptions& options,
    LinearizationValidationResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = LinearizationValidationResult();
  if (snapshot.metadata.loss_function != "TRIVIAL" &&
      snapshot.metadata.loss_function != "trivial") {
    *error = "Phase-3 validation only supports TRIVIAL loss, got " +
             snapshot.metadata.loss_function;
    return false;
  }
  LidarResidualMode lidar_mode;
  if (!ParseLidarResidualMode(snapshot.metadata.lidar_residual_mode,
                              &lidar_mode)) {
    *error = "Invalid LiDAR residual mode in snapshot: " +
             snapshot.metadata.lidar_residual_mode;
    return false;
  }
  result->lidar_mode = LidarResidualModeName(lidar_mode);

  std::unordered_map<uint32_t, const CameraSnapshot*> cameras;
  std::unordered_map<uint32_t, const ImageSnapshot*> images;
  std::unordered_map<uint64_t, const PointSnapshot*> points;
  for (const CameraSnapshot& camera : snapshot.cameras) {
    cameras.emplace(camera.camera_id, &camera);
  }
  for (const ImageSnapshot& image : snapshot.images) {
    images.emplace(image.image_id, &image);
  }
  for (const PointSnapshot& point : snapshot.points) {
    points.emplace(point.point3D_id, &point);
  }

  std::vector<uint64_t> canonical_visual_indices;
  canonical_visual_indices.reserve(snapshot.observations.size());
  for (const OrderEntrySnapshot& entry : snapshot.canonical_order) {
    if (entry.residual_kind == ResidualKind::kVisual) {
      canonical_visual_indices.push_back(entry.source_index);
    }
  }
  const size_t sample_count = std::min(
      options.max_finite_difference_observations,
      canonical_visual_indices.size());
  std::unordered_set<uint64_t> finite_difference_indices;
  std::ostringstream sample_hash_input;
  for (size_t i = 0; i < sample_count; ++i) {
    const size_t position = sample_count == canonical_visual_indices.size()
                                ? i
                                : (i * canonical_visual_indices.size()) /
                                      sample_count;
    const uint64_t source_index = canonical_visual_indices[position];
    finite_difference_indices.insert(source_index);
    sample_hash_input << source_index << '\n';
  }
  result->finite_difference_observations = finite_difference_indices.size();
  result->finite_difference_source_indices_sha256 =
      Sha256Hex(sample_hash_input.str());

  ErrorAccumulator residual("visual_residual_vs_autodiff", kResidualAtol,
                            kResidualRtol);
  ErrorAccumulator ambient(
      "ambient_quaternion_jacobian_vs_autodiff", kJacobianAtol,
      kJacobianRtol);
  ErrorAccumulator plus("plus_jacobian_vs_ceres", kJacobianAtol,
                        kJacobianRtol);
  ErrorAccumulator local(
      "local_rotation_jacobian_vs_autodiff_times_plus", kJacobianAtol,
      kJacobianRtol);
  ErrorAccumulator translation("translation_jacobian_vs_autodiff",
                               kJacobianAtol, kJacobianRtol);
  ErrorAccumulator point_jacobian("point_jacobian_vs_autodiff",
                                  kJacobianAtol, kJacobianRtol);
  ErrorAccumulator camera_jacobian("camera_jacobian_vs_autodiff",
                                   kJacobianAtol, kJacobianRtol);
  ErrorAccumulator ambient_fd(
      "ambient_quaternion_jacobian_vs_finite_difference",
      kFiniteDifferenceAtol, kFiniteDifferenceRtol);
  ErrorAccumulator local_fd(
      "local_rotation_jacobian_vs_finite_difference", kFiniteDifferenceAtol,
      kFiniteDifferenceRtol);
  ErrorAccumulator translation_fd(
      "translation_jacobian_vs_finite_difference", kFiniteDifferenceAtol,
      kFiniteDifferenceRtol);
  ErrorAccumulator point_fd("point_jacobian_vs_finite_difference",
                            kFiniteDifferenceAtol, kFiniteDifferenceRtol);
  ErrorAccumulator camera_fd("camera_jacobian_vs_finite_difference",
                             kFiniteDifferenceAtol, kFiniteDifferenceRtol);

  ceres::QuaternionParameterization quaternion_parameterization;
  for (const ObservationSnapshot& observation : snapshot.observations) {
    ++result->visual_observations;
    const auto image_it = images.find(observation.image_id);
    const auto point_it = points.find(observation.point3D_id);
    if (image_it == images.end() || point_it == points.end()) {
      *error = "Snapshot observation references missing state: " +
               ObservationId(observation);
      return false;
    }
    const ImageSnapshot& image = *image_it->second;
    const auto camera_it = cameras.find(image.camera_id);
    if (camera_it == cameras.end()) {
      *error = "Snapshot image references missing camera: image=" +
               std::to_string(image.image_id);
      return false;
    }
    const CameraSnapshot& camera_snapshot = *camera_it->second;
    if (camera_snapshot.model_id != OpenCVCameraModel::kModelId ||
        camera_snapshot.params.size() != OpenCVCameraModel::kNumParams) {
      *error = "Unsupported camera in phase-3 validator: camera=" +
               std::to_string(camera_snapshot.camera_id) + " model=" +
               std::to_string(camera_snapshot.model_id) + " params=" +
               std::to_string(camera_snapshot.params.size());
      return false;
    }
    std::array<double, 8> camera;
    std::copy(camera_snapshot.params.begin(), camera_snapshot.params.end(),
              camera.begin());
    const PointSnapshot& point = *point_it->second;
    const std::string id = ObservationId(observation);

    VisualEvaluation analytic;
    if (!EvaluateOpenCVVisual(image.qvec, image.tvec, point.xyz, camera,
                              observation.xy, &analytic)) {
      ++result->visual_candidate_failures;
      continue;
    }

    std::unique_ptr<ceres::CostFunction> cost(
        BundleAdjustmentCostFunction<OpenCVCameraModel>::Create(
            Eigen::Vector2d(observation.xy[0], observation.xy[1])));
    std::array<double, 2> reference_residual;
    std::array<double, 8> reference_ambient;
    std::array<double, 6> reference_translation;
    std::array<double, 6> reference_point;
    std::array<double, 16> reference_camera;
    const double* parameters[] = {image.qvec.data(), image.tvec.data(),
                                  point.xyz.data(), camera.data()};
    double* jacobians[] = {reference_ambient.data(),
                           reference_translation.data(),
                           reference_point.data(), reference_camera.data()};
    if (!cost->Evaluate(parameters, reference_residual.data(), jacobians)) {
      ++result->visual_reference_failures;
      continue;
    }
    bool reference_finite = true;
    for (const double value : reference_residual) {
      reference_finite = reference_finite && std::isfinite(value);
    }
    for (const double value : reference_ambient) {
      reference_finite = reference_finite && std::isfinite(value);
    }
    for (const double value : reference_translation) {
      reference_finite = reference_finite && std::isfinite(value);
    }
    for (const double value : reference_point) {
      reference_finite = reference_finite && std::isfinite(value);
    }
    for (const double value : reference_camera) {
      reference_finite = reference_finite && std::isfinite(value);
    }
    if (!reference_finite) {
      ++result->visual_reference_failures;
      continue;
    }

    std::array<double, 12> reference_plus;
    if (!quaternion_parameterization.ComputeJacobian(image.qvec.data(),
                                                     reference_plus.data())) {
      ++result->visual_reference_failures;
      continue;
    }
    std::array<double, 6> reference_local;
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        double value = 0.0;
        for (size_t ambient_col = 0; ambient_col < 4; ++ambient_col) {
          value += reference_ambient[row * 4 + ambient_col] *
                   reference_plus[ambient_col * 3 + col];
        }
        reference_local[row * 3 + col] = value;
      }
    }

    AddArray(&residual, reference_residual, analytic.residual, id);
    AddArray(&ambient, reference_ambient,
             analytic.ambient_quaternion_jacobian, id);
    AddArray(&plus, reference_plus, analytic.plus_jacobian, id);
    AddArray(&local, reference_local, analytic.local_rotation_jacobian, id);
    AddArray(&translation, reference_translation,
             analytic.translation_jacobian, id);
    AddArray(&point_jacobian, reference_point, analytic.point_jacobian, id);
    AddArray(&camera_jacobian, reference_camera, analytic.camera_jacobian, id);

    if (finite_difference_indices.count(observation.source_index) == 0) {
      continue;
    }
    const double relative_step = options.finite_difference_relative_step;
    for (size_t col = 0; col < 4; ++col) {
      const double step = DifferenceStep(relative_step, image.qvec[col]);
      std::array<double, 2> finite_difference;
      if (!FivePointDerivative(
              [&](double offset, std::array<double, 2>* value) {
                auto perturbed = image.qvec;
                perturbed[col] += offset;
                return ResidualOnly(cost.get(), perturbed, image.tvec,
                                    point.xyz, camera, value);
              },
              step, &finite_difference)) {
        ++result->visual_reference_failures;
        continue;
      }
      for (size_t row = 0; row < 2; ++row) {
        ambient_fd.Add(
            finite_difference[row],
            analytic.ambient_quaternion_jacobian[row * 4 + col],
            id + ":row=" + std::to_string(row) +
                ":ambient_col=" + std::to_string(col));
      }
    }
    for (size_t col = 0; col < 3; ++col) {
      std::array<double, 2> finite_difference;
      if (!FivePointDerivative(
              [&](double offset, std::array<double, 2>* value) {
                std::array<double, 3> delta{{0.0, 0.0, 0.0}};
                delta[col] = offset;
                std::array<double, 4> perturbed;
                if (!quaternion_parameterization.Plus(
                        image.qvec.data(), delta.data(), perturbed.data())) {
                  return false;
                }
                return ResidualOnly(cost.get(), perturbed, image.tvec,
                                    point.xyz, camera, value);
              },
              relative_step, &finite_difference)) {
        ++result->visual_reference_failures;
        continue;
      }
      for (size_t row = 0; row < 2; ++row) {
        local_fd.Add(finite_difference[row],
                     analytic.local_rotation_jacobian[row * 3 + col],
                     id + ":row=" + std::to_string(row) +
                         ":local_rotation_col=" + std::to_string(col));
      }
    }
    for (size_t col = 0; col < 3; ++col) {
      const double step = DifferenceStep(relative_step, image.tvec[col]);
      std::array<double, 2> finite_difference;
      if (!FivePointDerivative(
              [&](double offset, std::array<double, 2>* value) {
                auto perturbed = image.tvec;
                perturbed[col] += offset;
                return ResidualOnly(cost.get(), image.qvec, perturbed,
                                    point.xyz, camera, value);
              },
              step, &finite_difference)) {
        ++result->visual_reference_failures;
        continue;
      }
      for (size_t row = 0; row < 2; ++row) {
        translation_fd.Add(
            finite_difference[row],
            analytic.translation_jacobian[row * 3 + col],
            id + ":row=" + std::to_string(row) +
                ":translation_col=" + std::to_string(col));
      }
    }
    for (size_t col = 0; col < 3; ++col) {
      const double step = DifferenceStep(relative_step, point.xyz[col]);
      std::array<double, 2> finite_difference;
      if (!FivePointDerivative(
              [&](double offset, std::array<double, 2>* value) {
                auto perturbed = point.xyz;
                perturbed[col] += offset;
                return ResidualOnly(cost.get(), image.qvec, image.tvec,
                                    perturbed, camera, value);
              },
              step, &finite_difference)) {
        ++result->visual_reference_failures;
        continue;
      }
      for (size_t row = 0; row < 2; ++row) {
        point_fd.Add(finite_difference[row],
                     analytic.point_jacobian[row * 3 + col],
                     id + ":row=" + std::to_string(row) +
                         ":point_col=" + std::to_string(col));
      }
    }
    for (size_t col = 0; col < 8; ++col) {
      const double step = DifferenceStep(relative_step, camera[col]);
      std::array<double, 2> finite_difference;
      if (!FivePointDerivative(
              [&](double offset, std::array<double, 2>* value) {
                auto perturbed = camera;
                perturbed[col] += offset;
                return ResidualOnly(cost.get(), image.qvec, image.tvec,
                                    point.xyz, perturbed, value);
              },
              step, &finite_difference)) {
        ++result->visual_reference_failures;
        continue;
      }
      for (size_t row = 0; row < 2; ++row) {
        camera_fd.Add(finite_difference[row],
                      analytic.camera_jacobian[row * 8 + col],
                      id + ":row=" + std::to_string(row) +
                          ":camera_col=" + std::to_string(col));
      }
    }
  }

  ErrorAccumulator lidar_residual("lidar_residual_vs_autodiff",
                                  kResidualAtol, kResidualRtol);
  ErrorAccumulator lidar_point("lidar_point_jacobian_vs_autodiff",
                               kJacobianAtol, kJacobianRtol);
  ErrorAccumulator lidar_cost("lidar_near_zero_cost_vs_autodiff",
                              kResidualAtol, kResidualRtol);
  ErrorAccumulator lidar_jtr("lidar_near_zero_jtr_vs_autodiff",
                             kJacobianAtol, kJacobianRtol);
  ErrorAccumulator lidar_jtj("lidar_near_zero_jtj_vs_autodiff",
                             kJacobianAtol, kJacobianRtol);
  for (const LidarSnapshot& lidar : snapshot.lidar) {
    ++result->lidar_residuals;
    const auto point_it = points.find(lidar.point3D_id);
    if (point_it == points.end()) {
      *error = "LiDAR residual references missing point: " +
               std::to_string(lidar.point3D_id);
      return false;
    }
    const PointSnapshot& point = *point_it->second;
    const LidarEvaluation analytic = EvaluateLidar(
        point.xyz, lidar.plane, lidar.weight, lidar_mode,
        options.lidar_near_zero_threshold);
    if (analytic.exact_zero) {
      result->lidar_exact_zero_point_ids.push_back(lidar.point3D_id);
    }
    if (analytic.near_zero) {
      result->lidar_near_zero_point_ids.push_back(lidar.point3D_id);
    }
    if (analytic.guarded) ++result->lidar_guarded_count;

    Eigen::Matrix<double, 4, 1> plane;
    for (size_t i = 0; i < 4; ++i) plane[i] = lidar.plane[i];
    std::unique_ptr<ceres::CostFunction> cost(
        BundleAdjustmentLidarCostFunction::Create(plane, lidar.weight));
    const double* parameters[] = {point.xyz.data()};
    double reference_residual = 0.0;
    std::array<double, 3> reference_jacobian;
    double* jacobians[] = {reference_jacobian.data()};
    if (!cost->Evaluate(parameters, &reference_residual, jacobians)) {
      ++result->lidar_reference_failures;
      continue;
    }
    const bool reference_residual_finite = std::isfinite(reference_residual);
    const bool reference_jacobian_finite =
        std::isfinite(reference_jacobian[0]) &&
        std::isfinite(reference_jacobian[1]) &&
        std::isfinite(reference_jacobian[2]);
    const std::string id = "source=" + std::to_string(lidar.source_index) +
                           ":point3D=" +
                           std::to_string(lidar.point3D_id);
    if (!reference_residual_finite || !std::isfinite(analytic.residual)) {
      ++result->lidar_reference_failures;
      continue;
    }

    const bool residual_is_source_comparable =
        lidar_mode != LidarResidualMode::kSigned;
    const bool jacobian_is_source_comparable =
        lidar_mode == LidarResidualMode::kLegacyExact ||
        (lidar_mode == LidarResidualMode::kLegacyGuarded &&
         !analytic.near_zero);
    if (residual_is_source_comparable) {
      lidar_residual.Add(reference_residual, analytic.residual, id);
    }

    if (jacobian_is_source_comparable) {
      const bool candidate_jacobian_finite =
          std::isfinite(analytic.point_jacobian[0]) &&
          std::isfinite(analytic.point_jacobian[1]) &&
          std::isfinite(analytic.point_jacobian[2]);
      if (reference_jacobian_finite != candidate_jacobian_finite) {
        ++result->lidar_classification_mismatches;
      } else if (reference_jacobian_finite) {
        AddArray(&lidar_point, reference_jacobian,
                 analytic.point_jacobian, id);
      }
    }

    if (analytic.near_zero) {
      const double reference_cost =
          0.5 * reference_residual * reference_residual;
      const double candidate_cost =
          0.5 * analytic.residual * analytic.residual;
      lidar_cost.Add(reference_cost, candidate_cost, id);
      const bool normal_equations_are_source_comparable =
          jacobian_is_source_comparable ||
          lidar_mode == LidarResidualMode::kSigned;
      if (normal_equations_are_source_comparable &&
          reference_jacobian_finite && analytic.finite) {
        for (size_t i = 0; i < 3; ++i) {
          lidar_jtr.Add(reference_jacobian[i] * reference_residual,
                        analytic.point_jacobian[i] * analytic.residual,
                        id + ":jtr_col=" + std::to_string(i));
          for (size_t j = 0; j < 3; ++j) {
            lidar_jtj.Add(reference_jacobian[i] * reference_jacobian[j],
                          analytic.point_jacobian[i] *
                              analytic.point_jacobian[j],
                          id + ":jtj=" + std::to_string(i) + "," +
                              std::to_string(j));
          }
        }
      }
    }
  }

  result->metrics = {
      residual.Finish(),       ambient.Finish(),
      plus.Finish(),           local.Finish(),
      translation.Finish(),    point_jacobian.Finish(),
      camera_jacobian.Finish(), ambient_fd.Finish(),
      local_fd.Finish(),       translation_fd.Finish(),
      point_fd.Finish(),       camera_fd.Finish(),
      lidar_residual.Finish(), lidar_point.Finish(),
      lidar_cost.Finish(),     lidar_jtr.Finish(),
      lidar_jtj.Finish()};
  result->pass = result->visual_reference_failures == 0 &&
                 result->visual_candidate_failures == 0 &&
                 result->lidar_reference_failures == 0 &&
                 result->lidar_classification_mismatches == 0;
  for (const ErrorSummary& summary : result->metrics) {
    result->pass = result->pass && summary.pass;
  }
  return true;
}

std::string LinearizationValidationJson(
    const LinearizationValidationResult& result,
    const LinearizationValidationOptions& options,
    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string indent2(indent_spaces * 2, ' ');
  const std::string indent3(indent_spaces * 3, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "\"linearization_validation\": {\n";
  stream << indent2 << "\"scope\": \"phase_3_residual_and_jacobian\",\n";
  stream << indent2 << "\"lidar_mode\": \""
         << EscapeJson(result.lidar_mode) << "\",\n";
  stream << indent2 << "\"comparison_rule\": "
         << "\"abs_error <= atol + rtol * max(abs(reference), "
            "abs(candidate))\",\n";
  stream << indent2 << "\"finite_difference\": {\"scheme\": "
         << "\"five_point_central_ceres_residual\", \"relative_step\": "
         << options.finite_difference_relative_step
         << ", \"scale\": \"max(1,abs(parameter))\", "
         << "\"max_observations\": "
         << options.max_finite_difference_observations
         << ", \"sampled_observations\": "
         << result.finite_difference_observations
         << ", \"source_indices_sha256\": \""
         << result.finite_difference_source_indices_sha256 << "\"},\n";
  stream << indent2 << "\"counts\": {\"visual_observations\": "
         << result.visual_observations << ", \"lidar_residuals\": "
         << result.lidar_residuals << ", \"visual_reference_failures\": "
         << result.visual_reference_failures
         << ", \"visual_candidate_failures\": "
         << result.visual_candidate_failures
         << ", \"lidar_reference_failures\": "
         << result.lidar_reference_failures
         << ", \"lidar_classification_mismatches\": "
         << result.lidar_classification_mismatches
         << ", \"lidar_guarded_count\": " << result.lidar_guarded_count
         << "},\n";
  stream << indent2 << "\"lidar_near_zero_threshold\": "
         << options.lidar_near_zero_threshold << ",\n";
  stream << indent2 << "\"lidar_exact_zero_point_ids\": ";
  AppendUint64Array(&stream, result.lidar_exact_zero_point_ids);
  stream << ",\n";
  stream << indent2 << "\"lidar_near_zero_point_ids\": ";
  AppendUint64Array(&stream, result.lidar_near_zero_point_ids);
  stream << ",\n";
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
    if (i + 1 != result.metrics.size()) stream << ",";
    stream << "\n";
  }
  stream << indent2 << "],\n";
  stream << indent2 << "\"pass\": " << (result.pass ? "true" : "false")
         << "\n";
  stream << indent << "}";
  return stream.str();
}

}  // namespace gpu_ba
}  // namespace colmap
