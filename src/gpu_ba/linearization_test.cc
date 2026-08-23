#define TEST_NAME "gpu_ba/linearization"
#include "util/testing.h"

#include <array>
#include <cmath>
#include <memory>

#include <ceres/local_parameterization.h>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

template <size_t N>
void CheckArrayNear(const std::array<double, N>& reference,
                    const std::array<double, N>& candidate,
                    double tolerance) {
  for (size_t i = 0; i < N; ++i) {
    BOOST_CHECK_SMALL(reference[i] - candidate[i], tolerance);
  }
}

struct VisualFixture {
  VisualFixture() {
    const std::array<double, 4> raw{{0.91, 0.13, -0.21, 0.31}};
    BOOST_REQUIRE(NormalizeQuaternion(raw, &quaternion));
  }

  std::array<double, 4> quaternion;
  std::array<double, 3> translation{{0.2, -0.1, 0.4}};
  std::array<double, 3> point{{1.2, -0.4, 4.5}};
  std::array<double, 8> camera{{610.0, 605.0, 320.0, 240.0,
                                -0.03, 0.004, 0.001, -0.0007}};
  std::array<double, 2> observation{{421.3, 187.4}};
};

VisualEvaluation CeresReference(const VisualFixture& fixture,
                                std::array<double, 8>* camera_jacobian_row0,
                                std::array<double, 8>* camera_jacobian_row1) {
  std::unique_ptr<ceres::CostFunction> cost(
      BundleAdjustmentCostFunction<OpenCVCameraModel>::Create(
          Eigen::Vector2d(fixture.observation[0], fixture.observation[1])));
  const double* parameters[] = {fixture.quaternion.data(),
                                fixture.translation.data(), fixture.point.data(),
                                fixture.camera.data()};
  VisualEvaluation reference;
  double camera_jacobian[16];
  double* jacobians[] = {reference.ambient_quaternion_jacobian.data(),
                         reference.translation_jacobian.data(),
                         reference.point_jacobian.data(), camera_jacobian};
  BOOST_REQUIRE(cost->Evaluate(parameters, reference.residual.data(), jacobians));
  for (size_t i = 0; i < 8; ++i) {
    (*camera_jacobian_row0)[i] = camera_jacobian[i];
    (*camera_jacobian_row1)[i] = camera_jacobian[8 + i];
  }
  return reference;
}

}  // namespace

BOOST_FIXTURE_TEST_CASE(AnalyticMatchesCeresAmbient, VisualFixture) {
  VisualEvaluation analytic;
  BOOST_REQUIRE(EvaluateOpenCVVisual(quaternion, translation, point, camera,
                                     observation, &analytic));
  std::array<double, 8> camera_row0;
  std::array<double, 8> camera_row1;
  const VisualEvaluation reference =
      CeresReference(*this, &camera_row0, &camera_row1);
  CheckArrayNear(reference.residual, analytic.residual, 1e-12);
  CheckArrayNear(reference.ambient_quaternion_jacobian,
                 analytic.ambient_quaternion_jacobian, 1e-10);
  CheckArrayNear(reference.translation_jacobian,
                 analytic.translation_jacobian, 1e-10);
  CheckArrayNear(reference.point_jacobian, analytic.point_jacobian, 1e-10);
  for (size_t i = 0; i < 8; ++i) {
    BOOST_CHECK_SMALL(camera_row0[i] - analytic.camera_jacobian[i], 1e-10);
    BOOST_CHECK_SMALL(camera_row1[i] - analytic.camera_jacobian[8 + i], 1e-10);
  }
}

BOOST_FIXTURE_TEST_CASE(QuaternionPlusAndLocalJacobian, VisualFixture) {
  VisualEvaluation analytic;
  BOOST_REQUIRE(EvaluateOpenCVVisual(quaternion, translation, point, camera,
                                     observation, &analytic));
  ceres::QuaternionParameterization parameterization;
  std::array<double, 12> ceres_plus_jacobian;
  BOOST_REQUIRE(parameterization.ComputeJacobian(
      quaternion.data(), ceres_plus_jacobian.data()));
  CheckArrayNear(ceres_plus_jacobian, analytic.plus_jacobian, 1e-15);

  const std::array<double, 3> delta{{2e-4, -3e-4, 4e-4}};
  std::array<double, 4> analytic_plus;
  std::array<double, 4> ceres_plus;
  BOOST_REQUIRE(QuaternionPlus(quaternion, delta, &analytic_plus));
  BOOST_REQUIRE(parameterization.Plus(quaternion.data(), delta.data(),
                                      ceres_plus.data()));
  CheckArrayNear(ceres_plus, analytic_plus, 1e-14);

  const double epsilon = 1e-7;
  for (size_t column = 0; column < 3; ++column) {
    std::array<double, 3> plus_delta{{0.0, 0.0, 0.0}};
    std::array<double, 3> minus_delta{{0.0, 0.0, 0.0}};
    plus_delta[column] = epsilon;
    minus_delta[column] = -epsilon;
    std::array<double, 4> q_plus;
    std::array<double, 4> q_minus;
    BOOST_REQUIRE(QuaternionPlus(quaternion, plus_delta, &q_plus));
    BOOST_REQUIRE(QuaternionPlus(quaternion, minus_delta, &q_minus));
    VisualEvaluation value_plus;
    VisualEvaluation value_minus;
    BOOST_REQUIRE(EvaluateOpenCVVisual(q_plus, translation, point, camera,
                                       observation, &value_plus));
    BOOST_REQUIRE(EvaluateOpenCVVisual(q_minus, translation, point, camera,
                                       observation, &value_minus));
    for (size_t row = 0; row < 2; ++row) {
      const double finite_difference =
          (value_plus.residual[row] - value_minus.residual[row]) /
          (2.0 * epsilon);
      BOOST_CHECK_SMALL(
          finite_difference - analytic.local_rotation_jacobian[row * 3 + column],
          2e-5);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(Ceres14QuaternionPlusIsIndependentAndExact,
                        VisualFixture) {
  ceres::QuaternionParameterization parameterization;
  const std::array<double, 3> delta{{0.17, -0.08, 0.03}};
  std::array<double, 4> reference;
  std::array<double, 4> candidate;
  BOOST_REQUIRE(parameterization.Plus(quaternion.data(), delta.data(),
                                      reference.data()));
  BOOST_REQUIRE(QuaternionPlusCeres14(quaternion, delta, &candidate));
  CheckArrayNear(reference, candidate, 1e-15);

  const std::array<double, 3> zero_delta{{0.0, 0.0, 0.0}};
  BOOST_REQUIRE(QuaternionPlusCeres14(quaternion, zero_delta, &candidate));
  CheckArrayNear(quaternion, candidate, 0.0);

  std::array<double, 4> non_unit = quaternion;
  for (double& value : non_unit) value *= 3.0;
  BOOST_REQUIRE(QuaternionPlusCeres14(non_unit, delta, &candidate));
  const double norm = std::sqrt(candidate[0] * candidate[0] +
                                candidate[1] * candidate[1] +
                                candidate[2] * candidate[2] +
                                candidate[3] * candidate[3]);
  BOOST_CHECK_CLOSE_FRACTION(norm, 3.0, 1e-15);
}

BOOST_AUTO_TEST_CASE(SmallAngleLeftMultiplyAndFactorTwo) {
  const std::array<double, 4> identity{{1.0, 0.0, 0.0, 0.0}};
  const std::array<double, 3> delta{{1e-12, -2e-12, 3e-12}};
  std::array<double, 4> result;
  BOOST_REQUIRE(QuaternionPlus(identity, delta, &result));
  BOOST_CHECK_SMALL(result[1] - delta[0], 1e-27);
  BOOST_CHECK_SMALL(result[2] - delta[1], 1e-27);
  BOOST_CHECK_SMALL(result[3] - delta[2], 1e-27);

  const std::array<double, 3> visible_delta{{1e-5, 0.0, 0.0}};
  BOOST_REQUIRE(QuaternionPlus(identity, visible_delta, &result));
  const double physical_angle = 2.0 * std::acos(result[0]);
  BOOST_CHECK_SMALL(physical_angle - 2.0e-5, 2e-11);

  const std::array<double, 4> q{{0.9, 0.1, 0.2, -0.3}};
  std::array<double, 4> normalized;
  std::array<double, 4> scaled_normalized;
  BOOST_REQUIRE(NormalizeQuaternion(q, &normalized));
  const std::array<double, 4> scaled{{2.7, 0.3, 0.6, -0.9}};
  BOOST_REQUIRE(NormalizeQuaternion(scaled, &scaled_normalized));
  CheckArrayNear(normalized, scaled_normalized, 1e-15);

  // A non-commuting update distinguishes q_delta * q from q * q_delta.
  const double z_half_angle = 0.3;
  const std::array<double, 4> z_rotation{{std::cos(z_half_angle), 0.0, 0.0,
                                          std::sin(z_half_angle)}};
  const std::array<double, 3> x_delta{{1e-3, 0.0, 0.0}};
  BOOST_REQUIRE(QuaternionPlus(z_rotation, x_delta, &result));
  BOOST_CHECK(result[2] < 0.0);
  BOOST_CHECK_SMALL(result[2] + std::sin(1e-3) * std::sin(z_half_angle),
                    1e-15);
}

BOOST_FIXTURE_TEST_CASE(NonUnitInputIsExplicitlyNormalized, VisualFixture) {
  VisualEvaluation reference;
  BOOST_REQUIRE(EvaluateOpenCVVisual(quaternion, translation, point, camera,
                                     observation, &reference));
  std::array<double, 4> scaled = quaternion;
  for (double& value : scaled) value *= 7.0;
  std::array<double, 4> normalized;
  BOOST_REQUIRE(NormalizeQuaternion(scaled, &normalized));
  VisualEvaluation candidate;
  BOOST_REQUIRE(EvaluateOpenCVVisual(normalized, translation, point, camera,
                                     observation, &candidate));
  CheckArrayNear(reference.residual, candidate.residual, 1e-12);
  CheckArrayNear(reference.local_rotation_jacobian,
                 candidate.local_rotation_jacobian, 1e-10);
}

BOOST_FIXTURE_TEST_CASE(AmbientFiniteDifference, VisualFixture) {
  VisualEvaluation analytic;
  BOOST_REQUIRE(EvaluateOpenCVVisual(quaternion, translation, point, camera,
                                     observation, &analytic));
  const double epsilon = 1e-7;
  for (size_t column = 0; column < 4; ++column) {
    auto plus = quaternion;
    auto minus = quaternion;
    plus[column] += epsilon;
    minus[column] -= epsilon;
    VisualEvaluation value_plus;
    VisualEvaluation value_minus;
    BOOST_REQUIRE(EvaluateOpenCVVisual(plus, translation, point, camera,
                                       observation, &value_plus));
    BOOST_REQUIRE(EvaluateOpenCVVisual(minus, translation, point, camera,
                                       observation, &value_minus));
    for (size_t row = 0; row < 2; ++row) {
      const double finite_difference =
          (value_plus.residual[row] - value_minus.residual[row]) /
          (2.0 * epsilon);
      BOOST_CHECK_SMALL(
          finite_difference -
              analytic.ambient_quaternion_jacobian[row * 4 + column],
          2e-5);
    }
  }
}

BOOST_AUTO_TEST_CASE(LidarModesAndZeroBoundary) {
  const std::array<double, 4> plane{{1.0, 0.0, 0.0, -1.0}};
  LidarEvaluation evaluation = EvaluateLidar(
      {{1.2, 2.0, 3.0}}, plane, 10.0, LidarResidualMode::kLegacyExact);
  BOOST_REQUIRE(evaluation.finite);
  BOOST_CHECK_CLOSE(evaluation.residual, 2.0, 1e-12);
  BOOST_CHECK_CLOSE(evaluation.point_jacobian[0], 10.0, 1e-12);

  evaluation = EvaluateLidar({{0.8, 2.0, 3.0}}, plane, 10.0,
                             LidarResidualMode::kLegacyExact);
  BOOST_REQUIRE(evaluation.finite);
  BOOST_CHECK_CLOSE(evaluation.residual, 2.0, 1e-12);
  BOOST_CHECK_CLOSE(evaluation.point_jacobian[0], -10.0, 1e-12);

  evaluation = EvaluateLidar({{1.0, 2.0, 3.0}}, plane, 10.0,
                             LidarResidualMode::kLegacyExact);
  BOOST_CHECK(evaluation.exact_zero);
  BOOST_CHECK(!evaluation.finite);
  BOOST_CHECK_EQUAL(evaluation.residual, 0.0);

  evaluation = EvaluateLidar({{1.0, 2.0, 3.0}}, plane, 10.0,
                             LidarResidualMode::kLegacyGuarded);
  BOOST_CHECK(evaluation.finite);
  BOOST_CHECK(evaluation.guarded);
  BOOST_CHECK_EQUAL(evaluation.point_jacobian[0], 0.0);

  evaluation = EvaluateLidar({{1.0, 2.0, 3.0}}, plane, 10.0,
                             LidarResidualMode::kSigned);
  BOOST_CHECK(evaluation.finite);
  BOOST_CHECK_EQUAL(evaluation.residual, 0.0);
  BOOST_CHECK_EQUAL(evaluation.point_jacobian[0], 10.0);

  evaluation = EvaluateLidar({{1.0 + 5e-13, 2.0, 3.0}}, plane, 10.0,
                             LidarResidualMode::kLegacyGuarded);
  BOOST_CHECK(evaluation.near_zero);
  BOOST_CHECK(evaluation.guarded);
}

}  // namespace gpu_ba
}  // namespace colmap
