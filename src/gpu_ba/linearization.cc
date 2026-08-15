#include "gpu_ba/linearization.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace colmap {
namespace gpu_ba {
namespace {

template <size_t N>
bool AllFinite(const std::array<double, N>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void Multiply2x3By3xN(const double* lhs,
                      const double* rhs,
                      size_t columns,
                      double* result) {
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < columns; ++col) {
      result[row * columns + col] =
          lhs[row * 3 + 0] * rhs[0 * columns + col] +
          lhs[row * 3 + 1] * rhs[1 * columns + col] +
          lhs[row * 3 + 2] * rhs[2 * columns + col];
    }
  }
}

}  // namespace

bool ParseLidarResidualMode(const std::string& value,
                            LidarResidualMode* mode) {
  if (value == "legacy_exact") {
    *mode = LidarResidualMode::kLegacyExact;
  } else if (value == "legacy_guarded") {
    *mode = LidarResidualMode::kLegacyGuarded;
  } else if (value == "signed") {
    *mode = LidarResidualMode::kSigned;
  } else {
    return false;
  }
  return true;
}

std::string LidarResidualModeName(LidarResidualMode mode) {
  switch (mode) {
    case LidarResidualMode::kLegacyExact: return "legacy_exact";
    case LidarResidualMode::kLegacyGuarded: return "legacy_guarded";
    case LidarResidualMode::kSigned: return "signed";
  }
  return "unknown";
}

bool NormalizeQuaternion(const std::array<double, 4>& input,
                         std::array<double, 4>* output) {
  const double squared_norm = input[0] * input[0] + input[1] * input[1] +
                              input[2] * input[2] + input[3] * input[3];
  if (!std::isfinite(squared_norm) ||
      squared_norm <= std::numeric_limits<double>::min()) {
    return false;
  }
  const double inverse_norm = 1.0 / std::sqrt(squared_norm);
  for (size_t i = 0; i < 4; ++i) (*output)[i] = input[i] * inverse_norm;
  return AllFinite(*output);
}

void QuaternionPlusJacobian(const std::array<double, 4>& quaternion,
                            std::array<double, 12>* jacobian) {
  const double w = quaternion[0];
  const double x = quaternion[1];
  const double y = quaternion[2];
  const double z = quaternion[3];
  *jacobian = {{-x, -y, -z,
                 w,  z, -y,
                -z,  w,  x,
                 y, -x,  w}};
}

bool QuaternionPlus(const std::array<double, 4>& quaternion,
                    const std::array<double, 3>& delta,
                    std::array<double, 4>* result) {
  const double squared_theta = delta[0] * delta[0] +
                               delta[1] * delta[1] +
                               delta[2] * delta[2];
  if (!std::isfinite(squared_theta)) return false;
  const double theta = std::sqrt(squared_theta);
  double sin_theta_over_theta = 1.0;
  double cos_theta = 1.0;
  if (theta > 1e-8) {
    sin_theta_over_theta = std::sin(theta) / theta;
    cos_theta = std::cos(theta);
  } else {
    const double theta4 = squared_theta * squared_theta;
    sin_theta_over_theta =
        1.0 - squared_theta / 6.0 + theta4 / 120.0;
    cos_theta = 1.0 - squared_theta / 2.0 + theta4 / 24.0;
  }

  const double dw = cos_theta;
  const double dx = sin_theta_over_theta * delta[0];
  const double dy = sin_theta_over_theta * delta[1];
  const double dz = sin_theta_over_theta * delta[2];
  const double w = quaternion[0];
  const double x = quaternion[1];
  const double y = quaternion[2];
  const double z = quaternion[3];
  const std::array<double, 4> product{{
      dw * w - dx * x - dy * y - dz * z,
      dw * x + dx * w + dy * z - dz * y,
      dw * y - dx * z + dy * w + dz * x,
      dw * z + dx * y - dy * x + dz * w}};
  return NormalizeQuaternion(product, result);
}

bool EvaluateOpenCVVisual(const std::array<double, 4>& quaternion,
                          const std::array<double, 3>& translation,
                          const std::array<double, 3>& point,
                          const std::array<double, 8>& camera,
                          const std::array<double, 2>& observation,
                          VisualEvaluation* evaluation) {
  if (evaluation == nullptr || !AllFinite(quaternion) ||
      !AllFinite(translation) || !AllFinite(point) || !AllFinite(camera) ||
      !AllFinite(observation)) {
    return false;
  }

  const double w = quaternion[0];
  const double x = quaternion[1];
  const double y = quaternion[2];
  const double z = quaternion[3];
  const double px = point[0];
  const double py = point[1];
  const double pz = point[2];

  // Exact polynomial used by ceres::UnitQuaternionRotatePoint. This is not
  // the homogeneous non-unit quaternion formula; its ambient derivative is
  // therefore intentionally taken from this expansion.
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (w * y + x * z);
  const double r10 = 2.0 * (w * z + x * y);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (w * x + y * z);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);

  const double X = r00 * px + r01 * py + r02 * pz + translation[0];
  const double Y = r10 * px + r11 * py + r12 * pz + translation[1];
  const double Z = r20 * px + r21 * py + r22 * pz + translation[2];
  evaluation->camera_point = {{X, Y, Z}};
  if (!std::isfinite(Z) || std::abs(Z) <= std::numeric_limits<double>::min()) {
    return false;
  }

  const double u = X / Z;
  const double v = Y / Z;
  const double u2 = u * u;
  const double uv = u * v;
  const double v2 = v * v;
  const double r2 = u2 + v2;
  const double r4 = r2 * r2;
  const double fx = camera[0];
  const double fy = camera[1];
  const double cx = camera[2];
  const double cy = camera[3];
  const double k1 = camera[4];
  const double k2 = camera[5];
  const double p1 = camera[6];
  const double p2 = camera[7];
  const double radial = k1 * r2 + k2 * r4;
  const double scale = 1.0 + radial;
  const double distorted_u =
      u * scale + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
  const double distorted_v =
      v * scale + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);
  evaluation->residual = {{fx * distorted_u + cx - observation[0],
                           fy * distorted_v + cy - observation[1]}};

  const double radial_u = 2.0 * u * (k1 + 2.0 * k2 * r2);
  const double radial_v = 2.0 * v * (k1 + 2.0 * k2 * r2);
  const double ddu_du =
      scale + u * radial_u + 2.0 * p1 * v + 6.0 * p2 * u;
  const double ddu_dv =
      u * radial_v + 2.0 * p1 * u + 2.0 * p2 * v;
  const double ddv_du =
      v * radial_u + 2.0 * p2 * v + 2.0 * p1 * u;
  const double ddv_dv =
      scale + v * radial_v + 2.0 * p2 * u + 6.0 * p1 * v;
  const double inverse_z = 1.0 / Z;
  const double normalized_jacobian[6] = {
      inverse_z, 0.0, -u * inverse_z,
      0.0, inverse_z, -v * inverse_z};
  const double distortion_jacobian[4] = {
      fx * ddu_du, fx * ddu_dv,
      fy * ddv_du, fy * ddv_dv};
  double camera_point_jacobian[6];
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      camera_point_jacobian[row * 3 + col] =
          distortion_jacobian[row * 2 + 0] *
              normalized_jacobian[0 * 3 + col] +
          distortion_jacobian[row * 2 + 1] *
              normalized_jacobian[1 * 3 + col];
    }
  }
  std::copy(camera_point_jacobian, camera_point_jacobian + 6,
            evaluation->translation_jacobian.begin());

  const double rotation[9] = {
      r00, r01, r02, r10, r11, r12, r20, r21, r22};
  Multiply2x3By3xN(camera_point_jacobian, rotation, 3,
                   evaluation->point_jacobian.data());

  const double rotated_point_quaternion_jacobian[12] = {
      2.0 * (-z * py + y * pz),
      2.0 * (y * py + z * pz),
      2.0 * (-2.0 * y * px + x * py + w * pz),
      2.0 * (-2.0 * z * px - w * py + x * pz),

      2.0 * (z * px - x * pz),
      2.0 * (y * px - 2.0 * x * py - w * pz),
      2.0 * (x * px + z * pz),
      2.0 * (w * px - 2.0 * z * py + y * pz),

      2.0 * (-y * px + x * py),
      2.0 * (z * px + w * py - 2.0 * x * pz),
      2.0 * (-w * px + z * py - 2.0 * y * pz),
      2.0 * (x * px + y * py)};
  Multiply2x3By3xN(camera_point_jacobian,
                   rotated_point_quaternion_jacobian, 4,
                   evaluation->ambient_quaternion_jacobian.data());

  QuaternionPlusJacobian(quaternion, &evaluation->plus_jacobian);
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

  evaluation->camera_jacobian = {{
      distorted_u, 0.0, 1.0, 0.0,
      fx * u * r2, fx * u * r4, fx * 2.0 * uv,
      fx * (r2 + 2.0 * u2),
      0.0, distorted_v, 0.0, 1.0,
      fy * v * r2, fy * v * r4, fy * (r2 + 2.0 * v2),
      fy * 2.0 * uv}};

  evaluation->finite =
      AllFinite(evaluation->residual) &&
      AllFinite(evaluation->ambient_quaternion_jacobian) &&
      AllFinite(evaluation->plus_jacobian) &&
      AllFinite(evaluation->local_rotation_jacobian) &&
      AllFinite(evaluation->translation_jacobian) &&
      AllFinite(evaluation->point_jacobian) &&
      AllFinite(evaluation->camera_jacobian);
  return evaluation->finite;
}

LidarEvaluation EvaluateLidar(const std::array<double, 3>& point,
                              const std::array<double, 4>& plane,
                              double weight,
                              LidarResidualMode mode,
                              double near_zero_threshold) {
  LidarEvaluation evaluation;
  evaluation.signed_distance = plane[0] * point[0] + plane[1] * point[1] +
                               plane[2] * point[2] + plane[3];
  evaluation.exact_zero = evaluation.signed_distance == 0.0;
  evaluation.near_zero =
      std::abs(evaluation.signed_distance) <= near_zero_threshold;

  double derivative = 0.0;
  if (mode == LidarResidualMode::kSigned) {
    evaluation.residual = weight * evaluation.signed_distance;
    derivative = weight;
  } else {
    const double magnitude =
        std::sqrt(evaluation.signed_distance * evaluation.signed_distance);
    evaluation.residual = weight * magnitude;
    if (mode == LidarResidualMode::kLegacyGuarded && evaluation.near_zero) {
      derivative = 0.0;
      evaluation.guarded = true;
    } else {
      derivative = weight * evaluation.signed_distance / magnitude;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    evaluation.point_jacobian[i] = derivative * plane[i];
  }
  evaluation.finite = std::isfinite(evaluation.signed_distance) &&
                      std::isfinite(evaluation.residual) &&
                      AllFinite(evaluation.point_jacobian);
  return evaluation;
}

}  // namespace gpu_ba
}  // namespace colmap
