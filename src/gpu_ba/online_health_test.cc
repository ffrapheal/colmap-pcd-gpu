#define TEST_NAME "gpu_ba/online_health"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "gpu_ba/online_health.h"

namespace colmap {
namespace gpu_ba {
namespace {

CudaOnlineHealthOptions EnabledOptions() {
  CudaOnlineHealthOptions options;
  options.enabled = true;
  return options;
}

void SetPointMatrix(CudaOnlineHealthCapture* capture,
                    const size_t point_index,
                    const std::array<double, 9>& matrix) {
  const size_t offset = point_index * matrix.size();
  for (size_t index = 0; index < matrix.size(); ++index) {
    capture->point_hessians[offset + index] = matrix[index];
  }
}

void SetPointDiagonal(CudaOnlineHealthCapture* capture,
                      const size_t point_index,
                      const double first,
                      const double second,
                      const double third) {
  SetPointMatrix(capture, point_index,
                 {{first, 0.0, 0.0,
                   0.0, second, 0.0,
                   0.0, 0.0, third}});
}

void SetSchurDiagonal(CudaOnlineHealthCapture* capture,
                      const std::vector<double>& diagonal) {
  BOOST_REQUIRE_EQUAL(diagonal.size(), capture->pose_dimension);
  std::fill(capture->reduced_pose_schur.begin(),
            capture->reduced_pose_schur.end(), 0.0);
  for (size_t index = 0; index < diagonal.size(); ++index) {
    capture->reduced_pose_schur[index * capture->pose_dimension + index] =
        diagonal[index];
  }
}

CudaOnlineHealthCapture IdentityCapture(const size_t point_count = 1,
                                        const size_t pose_dimension = 12) {
  CudaOnlineHealthCapture capture;
  capture.point_hessians.assign(point_count * 9, 0.0);
  for (size_t point_index = 0; point_index < point_count; ++point_index) {
    SetPointDiagonal(&capture, point_index, 1.0, 1.0, 1.0);
  }
  capture.pose_dimension = pose_dimension;
  capture.reduced_pose_schur.assign(
      capture.pose_dimension * capture.pose_dimension, 0.0);
  capture.pose_jacobi_scaling.assign(capture.pose_dimension, 1.0);
  SetSchurDiagonal(&capture,
                   std::vector<double>(capture.pose_dimension, 1.0));
  return capture;
}

struct Evaluation {
  bool api_success = false;
  CudaOnlineHealthResult result;
  std::string error;
};

Evaluation RunEvaluation(const CudaOnlineHealthOptions& options,
                         const CudaOnlineHealthCapture& capture) {
  Evaluation evaluation;
  evaluation.error = "stale error";
  evaluation.api_success = EvaluateCudaOnlineHealth(
      options, capture, &evaluation.result, &evaluation.error);
  return evaluation;
}

void CheckDefaultMatrixResult(
    const CudaOnlineHealthMatrixResult& matrix_result) {
  BOOST_CHECK(!matrix_result.requested);
  BOOST_CHECK(!matrix_result.completed);
  BOOST_CHECK(!matrix_result.passed);
  BOOST_CHECK(!matrix_result.finite);
  BOOST_CHECK(!matrix_result.full_rank);
  BOOST_CHECK(!matrix_result.condition_within_limit);
  BOOST_CHECK(!matrix_result.has_significant_negative_eigenvalue);
  BOOST_CHECK_EQUAL(matrix_result.dimension, 0);
  BOOST_CHECK_EQUAL(matrix_result.rank, 0);
  BOOST_CHECK_EQUAL(matrix_result.spectral_radius, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.rank_threshold, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.condition_number, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.min_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.max_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.relative_rank_tolerance, 0.0);
  BOOST_CHECK_EQUAL(matrix_result.max_condition_number, 0.0);
  BOOST_CHECK(matrix_result.failure_reason.empty());
}

void CheckApiRejected(const CudaOnlineHealthOptions& options,
                      const CudaOnlineHealthCapture& capture) {
  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_CHECK(!evaluation.api_success);
  BOOST_CHECK(evaluation.result.requested);
  BOOST_CHECK(!evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_CHECK(!evaluation.result.finite);
  BOOST_CHECK(!evaluation.result.point_matrices_finite);
  BOOST_CHECK(!evaluation.result.schur_finite);
  BOOST_CHECK_EQUAL(evaluation.result.point_count, 0);
  BOOST_CHECK_EQUAL(evaluation.result.full_rank_count, 0);
  BOOST_CHECK_EQUAL(evaluation.result.min_rank, 0);
  BOOST_CHECK_EQUAL(evaluation.result.worst_condition, 0.0);
  BOOST_CHECK_EQUAL(evaluation.result.pose_dimension, capture.pose_dimension);
  BOOST_CHECK_EQUAL(evaluation.result.schur_rank, 0);
  BOOST_CHECK_EQUAL(evaluation.result.schur_condition, 0.0);
  BOOST_CHECK(evaluation.result.point_results.empty());
  CheckDefaultMatrixResult(evaluation.result.schur_result);
  BOOST_CHECK(!evaluation.error.empty());
  BOOST_CHECK_EQUAL(evaluation.result.failure_reason, evaluation.error);
}

CudaOnlineHealthMatrixResult PrefilledMatrixResult() {
  CudaOnlineHealthMatrixResult result;
  result.requested = true;
  result.completed = true;
  result.passed = true;
  result.finite = true;
  result.full_rank = true;
  result.condition_within_limit = true;
  result.has_significant_negative_eigenvalue = true;
  result.dimension = 17;
  result.rank = 13;
  result.spectral_radius = 19.0;
  result.rank_threshold = 23.0;
  result.condition_number = 29.0;
  result.min_eigenvalue = 31.0;
  result.max_eigenvalue = 37.0;
  result.relative_rank_tolerance = 41.0;
  result.max_condition_number = 43.0;
  result.failure_reason = "stale matrix failure";
  return result;
}

CudaOnlineHealthResult PrefilledResult() {
  CudaOnlineHealthResult result;
  result.requested = true;
  result.completed = true;
  result.passed = true;
  result.finite = true;
  result.point_matrices_finite = true;
  result.schur_finite = true;
  result.has_significant_negative_eigenvalue = true;
  result.point_has_significant_negative_eigenvalue = true;
  result.schur_has_significant_negative_eigenvalue = true;
  result.point_count = 47;
  result.full_rank_count = 53;
  result.min_rank = 59;
  result.worst_condition = 61.0;
  result.pose_dimension = 67;
  result.schur_rank = 71;
  result.schur_condition = 73.0;
  result.schur_min_eigenvalue = 79.0;
  result.schur_max_eigenvalue = 83.0;
  result.relative_rank_tolerance = 89.0;
  result.max_condition_number = 97.0;
  result.max_pose_dimension = 101;
  result.point_results.assign(2, PrefilledMatrixResult());
  result.schur_result = PrefilledMatrixResult();
  result.failure_reason = "stale aggregate failure";
  return result;
}

void CheckCleanRequestedFailure(const CudaOnlineHealthOptions& options,
                                const CudaOnlineHealthCapture& capture,
                                const CudaOnlineHealthResult& result,
                                const std::string& error,
                                const std::string& expected_reason) {
  BOOST_CHECK(result.requested);
  BOOST_CHECK(!result.completed);
  BOOST_CHECK(!result.passed);
  BOOST_CHECK(!result.finite);
  BOOST_CHECK(!result.point_matrices_finite);
  BOOST_CHECK(!result.schur_finite);
  BOOST_CHECK(!result.has_significant_negative_eigenvalue);
  BOOST_CHECK(!result.point_has_significant_negative_eigenvalue);
  BOOST_CHECK(!result.schur_has_significant_negative_eigenvalue);
  BOOST_CHECK_EQUAL(result.point_count, 0);
  BOOST_CHECK_EQUAL(result.full_rank_count, 0);
  BOOST_CHECK_EQUAL(result.min_rank, 0);
  BOOST_CHECK_EQUAL(result.worst_condition, 0.0);
  BOOST_CHECK_EQUAL(result.pose_dimension, capture.pose_dimension);
  BOOST_CHECK_EQUAL(result.schur_rank, 0);
  BOOST_CHECK_EQUAL(result.schur_condition, 0.0);
  BOOST_CHECK_EQUAL(result.schur_min_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(result.schur_max_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(result.relative_rank_tolerance,
                    options.relative_rank_tolerance);
  BOOST_CHECK_EQUAL(result.max_condition_number,
                    options.max_condition_number);
  BOOST_CHECK_EQUAL(result.max_pose_dimension, options.max_pose_dimension);
  BOOST_CHECK(result.point_results.empty());
  CheckDefaultMatrixResult(result.schur_result);
  BOOST_CHECK_EQUAL(result.failure_reason, expected_reason);
  BOOST_CHECK_EQUAL(error, expected_reason);
}

}  // namespace

BOOST_AUTO_TEST_CASE(DefaultsDoNotClaimEvaluationRan) {
  const CudaOnlineHealthOptions options;
  const CudaOnlineHealthMatrixResult matrix_result;
  const CudaOnlineHealthResult result;

  BOOST_CHECK(!options.enabled);
  BOOST_CHECK_EQUAL(options.relative_rank_tolerance, 1e-12);
  BOOST_CHECK_EQUAL(options.max_condition_number, 1e12);
  BOOST_CHECK_EQUAL(options.max_pose_dimension, 120);
  BOOST_CHECK(!matrix_result.requested);
  BOOST_CHECK(!matrix_result.completed);
  BOOST_CHECK(!matrix_result.passed);
  BOOST_CHECK(!result.requested);
  BOOST_CHECK(!result.completed);
  BOOST_CHECK(!result.passed);
}

BOOST_AUTO_TEST_CASE(DisabledSkipsInvalidOptionsAndCapture) {
  CudaOnlineHealthOptions options;
  options.relative_rank_tolerance =
      std::numeric_limits<double>::quiet_NaN();
  options.max_condition_number = std::numeric_limits<double>::infinity();
  options.max_pose_dimension = 0;
  CudaOnlineHealthCapture capture;
  capture.point_hessians = {std::numeric_limits<double>::quiet_NaN()};

  CudaOnlineHealthResult result;
  result.requested = true;
  result.completed = true;
  result.passed = true;
  result.point_results.push_back(CudaOnlineHealthMatrixResult());
  std::string error = "stale error";
  BOOST_REQUIRE(
      EvaluateCudaOnlineHealth(options, capture, &result, &error));
  BOOST_CHECK(!result.requested);
  BOOST_CHECK(!result.completed);
  BOOST_CHECK(!result.passed);
  BOOST_CHECK(!result.finite);
  BOOST_CHECK_EQUAL(result.point_count, 0);
  BOOST_CHECK(result.point_results.empty());
  CheckDefaultMatrixResult(result.schur_result);
  BOOST_CHECK(result.failure_reason.empty());
  BOOST_CHECK(error.empty());
}

BOOST_AUTO_TEST_CASE(HealthyMultiplePointsAndTwelveDimensionalSchur) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture(2);
  SetPointDiagonal(&capture, 0, 1.0, 2.0, 4.0);
  SetPointDiagonal(&capture, 1, 2.0, 5.0, 10.0);
  std::vector<double> schur_diagonal(capture.pose_dimension);
  for (size_t index = 0; index < schur_diagonal.size(); ++index) {
    schur_diagonal[index] = static_cast<double>(index + 1);
  }
  SetSchurDiagonal(&capture, schur_diagonal);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE_MESSAGE(evaluation.api_success, evaluation.error);
  BOOST_CHECK(evaluation.result.requested);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(evaluation.result.passed);
  BOOST_CHECK(evaluation.result.finite);
  BOOST_CHECK(evaluation.result.point_matrices_finite);
  BOOST_CHECK(evaluation.result.schur_finite);
  BOOST_CHECK(!evaluation.result.has_significant_negative_eigenvalue);
  BOOST_CHECK_EQUAL(evaluation.result.point_count, 2);
  BOOST_CHECK_EQUAL(evaluation.result.full_rank_count, 2);
  BOOST_CHECK_EQUAL(evaluation.result.min_rank, 3);
  BOOST_CHECK_EQUAL(evaluation.result.worst_condition, 5.0);
  BOOST_CHECK_EQUAL(evaluation.result.pose_dimension, 12);
  BOOST_CHECK_EQUAL(evaluation.result.schur_rank, 12);
  BOOST_CHECK_EQUAL(evaluation.result.schur_condition, 12.0);
  BOOST_CHECK_EQUAL(evaluation.result.schur_min_eigenvalue, 1.0);
  BOOST_CHECK_EQUAL(evaluation.result.schur_max_eigenvalue, 12.0);
  BOOST_CHECK_EQUAL(evaluation.result.relative_rank_tolerance,
                    options.relative_rank_tolerance);
  BOOST_CHECK_EQUAL(evaluation.result.max_condition_number,
                    options.max_condition_number);
  BOOST_CHECK_EQUAL(evaluation.result.max_pose_dimension,
                    options.max_pose_dimension);
  BOOST_REQUIRE_EQUAL(evaluation.result.point_results.size(), 2);
  BOOST_CHECK(evaluation.result.point_results[0].requested);
  BOOST_CHECK(evaluation.result.point_results[0].completed);
  BOOST_CHECK(evaluation.result.point_results[0].passed);
  BOOST_CHECK(evaluation.result.point_results[0].finite);
  BOOST_CHECK(evaluation.result.point_results[0].full_rank);
  BOOST_CHECK_EQUAL(evaluation.result.point_results[0].rank, 3);
  BOOST_CHECK_EQUAL(evaluation.result.point_results[0].condition_number, 4.0);
  BOOST_CHECK_CLOSE_FRACTION(
      evaluation.result.point_results[0].rank_threshold, 4e-12, 1e-15);
  BOOST_CHECK(evaluation.result.schur_result.requested);
  BOOST_CHECK(evaluation.result.schur_result.completed);
  BOOST_CHECK(evaluation.result.schur_result.passed);
  BOOST_CHECK(evaluation.result.failure_reason.empty());
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(PointRankThresholdUsesStrictGreaterThan) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance = 0.125;
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, 0.125, 1.0, 1.0);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_REQUIRE_EQUAL(evaluation.result.point_results.size(), 1);
  const CudaOnlineHealthMatrixResult& point =
      evaluation.result.point_results.front();
  BOOST_CHECK(point.completed);
  BOOST_CHECK(!point.passed);
  BOOST_CHECK_EQUAL(point.rank_threshold, 0.125);
  BOOST_CHECK_EQUAL(point.rank, 2);
  BOOST_CHECK(!point.full_rank);
  BOOST_CHECK(!point.has_significant_negative_eigenvalue);
  BOOST_CHECK(std::isinf(point.condition_number));
  BOOST_CHECK_EQUAL(evaluation.result.full_rank_count, 0);
  BOOST_CHECK_EQUAL(evaluation.result.min_rank, 2);
  BOOST_CHECK(std::isinf(evaluation.result.worst_condition));
  BOOST_CHECK(!evaluation.result.failure_reason.empty());
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(ZeroPointHessianIsCompletedDegeneracy) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, 0.0, 0.0, 0.0);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_CHECK(evaluation.result.finite);
  BOOST_CHECK(evaluation.result.point_matrices_finite);
  BOOST_CHECK(evaluation.result.schur_finite);
  const CudaOnlineHealthMatrixResult& point =
      evaluation.result.point_results.front();
  BOOST_CHECK(point.completed);
  BOOST_CHECK(point.finite);
  BOOST_CHECK_EQUAL(point.rank, 0);
  BOOST_CHECK_EQUAL(point.spectral_radius, 0.0);
  BOOST_CHECK_EQUAL(point.rank_threshold, 0.0);
  BOOST_CHECK_EQUAL(point.min_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(point.max_eigenvalue, 0.0);
  BOOST_CHECK(!point.has_significant_negative_eigenvalue);
  BOOST_CHECK(std::isinf(point.condition_number));
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(NegativePointHessianIsCompletedDegeneracy) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance = 0.1;
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, -1.0, 2.0, 3.0);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_CHECK(evaluation.result.has_significant_negative_eigenvalue);
  BOOST_CHECK(evaluation.result.point_has_significant_negative_eigenvalue);
  BOOST_CHECK(!evaluation.result.schur_has_significant_negative_eigenvalue);
  const CudaOnlineHealthMatrixResult& point =
      evaluation.result.point_results.front();
  BOOST_CHECK(point.has_significant_negative_eigenvalue);
  BOOST_CHECK_EQUAL(point.rank, 2);
  BOOST_CHECK_EQUAL(point.min_eigenvalue, -1.0);
  BOOST_CHECK_EQUAL(point.max_eigenvalue, 3.0);
  BOOST_CHECK_CLOSE_FRACTION(point.rank_threshold, 0.3, 1e-15);
  BOOST_CHECK(!point.passed);
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(NegativeMagnitudeDefinesSpectralRadius) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance = 0.1;
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, -10.0, 1.0, 2.0);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_REQUIRE_EQUAL(evaluation.result.point_results.size(), 1);
  const CudaOnlineHealthMatrixResult& point =
      evaluation.result.point_results.front();
  BOOST_CHECK_EQUAL(point.min_eigenvalue, -10.0);
  BOOST_CHECK_EQUAL(point.max_eigenvalue, 2.0);
  BOOST_CHECK_EQUAL(point.spectral_radius, 10.0);
  BOOST_CHECK_EQUAL(point.rank_threshold, 1.0);
  BOOST_CHECK_EQUAL(point.rank, 1);
  BOOST_CHECK(point.has_significant_negative_eigenvalue);
  BOOST_CHECK(!point.passed);
}

BOOST_AUTO_TEST_CASE(NegativeSignificanceBoundaryUsesStrictLessThan) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance = 0.1;
  CudaOnlineHealthCapture capture = IdentityCapture();
  const double tau = 1.0;
  const auto evaluate_negative_eigenvalue = [&](const double eigenvalue) {
    SetPointDiagonal(&capture, 0, eigenvalue, 2.0, 10.0);
    return RunEvaluation(options, capture);
  };

  const Evaluation boundary = evaluate_negative_eigenvalue(-tau);
  BOOST_REQUIRE(boundary.api_success);
  BOOST_REQUIRE_EQUAL(boundary.result.point_results.size(), 1);
  BOOST_CHECK_EQUAL(boundary.result.point_results.front().spectral_radius,
                    10.0);
  BOOST_CHECK_EQUAL(boundary.result.point_results.front().rank_threshold, tau);
  BOOST_CHECK(!boundary.result.point_results.front()
                   .has_significant_negative_eigenvalue);

  const Evaluation below = evaluate_negative_eigenvalue(std::nextafter(
      -tau, -std::numeric_limits<double>::infinity()));
  BOOST_REQUIRE(below.api_success);
  BOOST_CHECK(below.result.point_results.front()
                  .has_significant_negative_eigenvalue);

  const Evaluation above = evaluate_negative_eigenvalue(
      std::nextafter(-tau, std::numeric_limits<double>::infinity()));
  BOOST_REQUIRE(above.api_success);
  BOOST_CHECK(!above.result.point_results.front()
                   .has_significant_negative_eigenvalue);
}

BOOST_AUTO_TEST_CASE(PointConditionBoundaryIsInclusive) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.max_condition_number = 10.0;
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, 1.0, 2.0, 10.0);

  const Evaluation boundary = RunEvaluation(options, capture);
  BOOST_REQUIRE(boundary.api_success);
  BOOST_CHECK(boundary.result.passed);
  BOOST_CHECK_EQUAL(boundary.result.point_results.front().condition_number,
                    options.max_condition_number);
  BOOST_CHECK(
      boundary.result.point_results.front().condition_within_limit);

  options.max_condition_number = std::nextafter(10.0, 0.0);
  const Evaluation over_limit = RunEvaluation(options, capture);
  BOOST_REQUIRE(over_limit.api_success);
  BOOST_CHECK(over_limit.result.completed);
  BOOST_CHECK(!over_limit.result.passed);
  BOOST_CHECK(over_limit.result.point_results.front().full_rank);
  BOOST_CHECK(
      !over_limit.result.point_results.front().condition_within_limit);
  BOOST_CHECK(!over_limit.result.failure_reason.empty());
  BOOST_CHECK(over_limit.error.empty());
}

BOOST_AUTO_TEST_CASE(SchurGaugeNullspaceIsNotRemoved) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  std::vector<double> diagonal(capture.pose_dimension, 1.0);
  diagonal.front() = 0.0;
  SetSchurDiagonal(&capture, diagonal);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_CHECK(evaluation.result.finite);
  BOOST_CHECK(evaluation.result.point_matrices_finite);
  BOOST_CHECK(evaluation.result.schur_finite);
  BOOST_CHECK_EQUAL(evaluation.result.schur_rank, 11);
  BOOST_CHECK(std::isinf(evaluation.result.schur_condition));
  BOOST_CHECK_EQUAL(evaluation.result.schur_min_eigenvalue, 0.0);
  BOOST_CHECK_EQUAL(evaluation.result.schur_max_eigenvalue, 1.0);
  BOOST_CHECK(!evaluation.result.schur_has_significant_negative_eigenvalue);
  BOOST_CHECK(!evaluation.result.schur_result.full_rank);
  BOOST_CHECK(!evaluation.result.schur_result.passed);
  BOOST_CHECK(!evaluation.result.failure_reason.empty());
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(NegativeSchurIsCompletedDegeneracy) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance = 0.1;
  CudaOnlineHealthCapture capture = IdentityCapture();
  std::vector<double> diagonal(capture.pose_dimension, 2.0);
  diagonal.front() = -1.0;
  SetSchurDiagonal(&capture, diagonal);

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.completed);
  BOOST_CHECK(!evaluation.result.passed);
  BOOST_CHECK(evaluation.result.has_significant_negative_eigenvalue);
  BOOST_CHECK(!evaluation.result.point_has_significant_negative_eigenvalue);
  BOOST_CHECK(evaluation.result.schur_has_significant_negative_eigenvalue);
  BOOST_CHECK_EQUAL(evaluation.result.schur_rank, 11);
  BOOST_CHECK_EQUAL(evaluation.result.schur_min_eigenvalue, -1.0);
  BOOST_CHECK_EQUAL(evaluation.result.schur_max_eigenvalue, 2.0);
  BOOST_CHECK(evaluation.result.schur_result
                  .has_significant_negative_eigenvalue);
  BOOST_CHECK(!evaluation.result.schur_result.passed);
  BOOST_CHECK(evaluation.error.empty());
}

BOOST_AUTO_TEST_CASE(SchurConditionBoundaryIsInclusive) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.max_condition_number = 10.0;
  CudaOnlineHealthCapture capture = IdentityCapture();
  std::vector<double> diagonal(capture.pose_dimension, 10.0);
  diagonal.front() = 1.0;
  SetSchurDiagonal(&capture, diagonal);

  const Evaluation boundary = RunEvaluation(options, capture);
  BOOST_REQUIRE(boundary.api_success);
  BOOST_CHECK(boundary.result.passed);
  BOOST_CHECK_EQUAL(boundary.result.schur_condition,
                    options.max_condition_number);
  BOOST_CHECK(boundary.result.schur_result.condition_within_limit);

  options.max_condition_number = std::nextafter(10.0, 0.0);
  const Evaluation over_limit = RunEvaluation(options, capture);
  BOOST_REQUIRE(over_limit.api_success);
  BOOST_CHECK(over_limit.result.completed);
  BOOST_CHECK(!over_limit.result.passed);
  BOOST_CHECK(over_limit.result.schur_result.full_rank);
  BOOST_CHECK(!over_limit.result.schur_result.condition_within_limit);
  BOOST_CHECK(!over_limit.result.failure_reason.empty());
  BOOST_CHECK(over_limit.error.empty());
}

BOOST_AUTO_TEST_CASE(PositiveOverallScalingPreservesRankAndCondition) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture(2);
  SetPointDiagonal(&capture, 0, 1.0, 3.0, 7.0);
  SetPointDiagonal(&capture, 1, 2.0, 8.0, 18.0);
  std::vector<double> diagonal(capture.pose_dimension);
  for (size_t index = 0; index < diagonal.size(); ++index) {
    diagonal[index] = static_cast<double>(index + 2);
  }
  SetSchurDiagonal(&capture, diagonal);

  CudaOnlineHealthCapture scaled = capture;
  const double scale = 1e40;
  for (double& value : scaled.point_hessians) value *= scale;
  for (double& value : scaled.reduced_pose_schur) value *= scale;

  const Evaluation baseline = RunEvaluation(options, capture);
  const Evaluation rescaled = RunEvaluation(options, scaled);
  BOOST_REQUIRE(baseline.api_success);
  BOOST_REQUIRE(rescaled.api_success);
  BOOST_CHECK(baseline.result.passed);
  BOOST_CHECK(rescaled.result.passed);
  BOOST_CHECK_EQUAL(rescaled.result.full_rank_count,
                    baseline.result.full_rank_count);
  BOOST_CHECK_EQUAL(rescaled.result.min_rank, baseline.result.min_rank);
  BOOST_CHECK_EQUAL(rescaled.result.schur_rank, baseline.result.schur_rank);
  BOOST_CHECK_CLOSE_FRACTION(rescaled.result.worst_condition,
                             baseline.result.worst_condition, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(rescaled.result.schur_condition,
                             baseline.result.schur_condition, 1e-14);
  BOOST_REQUIRE_EQUAL(rescaled.result.point_results.size(),
                      baseline.result.point_results.size());
  for (size_t index = 0; index < baseline.result.point_results.size();
       ++index) {
    BOOST_CHECK_EQUAL(rescaled.result.point_results[index].rank,
                      baseline.result.point_results[index].rank);
    BOOST_CHECK_CLOSE_FRACTION(
        rescaled.result.point_results[index].condition_number,
        baseline.result.point_results[index].condition_number, 1e-14);
  }
}

BOOST_AUTO_TEST_CASE(NonsymmetricInputsAreSymmetrized) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointMatrix(&capture, 0,
                 {{4.0, 2.0, 0.0,
                   0.0, 4.0, 0.0,
                   0.0, 0.0, 6.0}});
  SetSchurDiagonal(&capture,
                   std::vector<double>(capture.pose_dimension, 5.0));
  capture.reduced_pose_schur[1] = 4.0;
  capture.reduced_pose_schur[capture.pose_dimension] = 0.0;

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.passed);
  const CudaOnlineHealthMatrixResult& point =
      evaluation.result.point_results.front();
  BOOST_CHECK_CLOSE_FRACTION(point.min_eigenvalue, 3.0, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(point.max_eigenvalue, 6.0, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(point.condition_number, 2.0, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_min_eigenvalue, 3.0,
                             1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_max_eigenvalue, 7.0,
                             1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_condition, 7.0 / 3.0,
                             1e-14);
}

BOOST_AUTO_TEST_CASE(JacobiScalingEvaluatesExplicitDSD) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetSchurDiagonal(&capture,
                   std::vector<double>(capture.pose_dimension, 10.0));
  capture.reduced_pose_schur[0] = 2.0;
  capture.reduced_pose_schur[1] = 1.0;
  capture.reduced_pose_schur[capture.pose_dimension] = 1.0;
  capture.reduced_pose_schur[capture.pose_dimension + 1] = 2.0;
  capture.pose_jacobi_scaling[0] = 2.0;
  capture.pose_jacobi_scaling[1] = 3.0;

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK(evaluation.result.passed);
  BOOST_CHECK_EQUAL(evaluation.result.schur_rank, 12);
  const double expected_min = 13.0 - std::sqrt(61.0);
  const double expected_max = 13.0 + std::sqrt(61.0);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_min_eigenvalue,
                             expected_min, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_max_eigenvalue,
                             expected_max, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_condition,
                             expected_max / expected_min, 1e-14);
  BOOST_CHECK_CLOSE_FRACTION(evaluation.result.schur_result.rank_threshold,
                             expected_max * options.relative_rank_tolerance,
                             1e-14);
}

BOOST_AUTO_TEST_CASE(NonfinitePointAndSchurInputsAreRejected) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture point_nan = IdentityCapture();
  point_nan.point_hessians[0] =
      std::numeric_limits<double>::quiet_NaN();
  CheckApiRejected(options, point_nan);

  CudaOnlineHealthCapture schur_inf = IdentityCapture();
  schur_inf.reduced_pose_schur[0] =
      std::numeric_limits<double>::infinity();
  CheckApiRejected(options, schur_inf);
}

BOOST_AUTO_TEST_CASE(NonfiniteAndNonpositiveJacobiScalesAreRejected) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture scale_nan = IdentityCapture();
  scale_nan.pose_jacobi_scaling[0] =
      std::numeric_limits<double>::quiet_NaN();
  CheckApiRejected(options, scale_nan);

  CudaOnlineHealthCapture scale_inf = IdentityCapture();
  scale_inf.pose_jacobi_scaling[0] =
      std::numeric_limits<double>::infinity();
  CheckApiRejected(options, scale_inf);

  CudaOnlineHealthCapture scale_zero = IdentityCapture();
  scale_zero.pose_jacobi_scaling[0] = 0.0;
  CheckApiRejected(options, scale_zero);

  CudaOnlineHealthCapture scale_negative = IdentityCapture();
  scale_negative.pose_jacobi_scaling[0] = -1.0;
  CheckApiRejected(options, scale_negative);
}

BOOST_AUTO_TEST_CASE(ScalingOverflowIsRejectedAsNonfiniteEvaluation) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  capture.pose_jacobi_scaling.assign(
      capture.pose_dimension, std::numeric_limits<double>::max());

  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_CHECK(!evaluation.api_success);
  CheckCleanRequestedFailure(
      options, capture, evaluation.result, evaluation.error,
      "scaled reduced pose Schur is non-finite after transformation");
}

BOOST_AUTO_TEST_CASE(FullRankConditionOverflowRejectsEvaluation) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance =
      std::numeric_limits<double>::denorm_min();
  CudaOnlineHealthCapture capture = IdentityCapture();
  SetPointDiagonal(&capture, 0, 0.25, 1.0,
                   std::numeric_limits<double>::max() / 2.0);

  CudaOnlineHealthResult result = PrefilledResult();
  std::string error = "stale error";
  BOOST_CHECK(!EvaluateCudaOnlineHealth(options, capture, &result, &error));
  CheckCleanRequestedFailure(
      options, capture, result, error,
      "variable point Hessian condition number is non-finite");
}

BOOST_AUTO_TEST_CASE(CaptureShapeMismatchesAreRejected) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture point_shape = IdentityCapture();
  point_shape.point_hessians.push_back(0.0);
  CheckApiRejected(options, point_shape);

  CudaOnlineHealthCapture schur_shape = IdentityCapture();
  schur_shape.reduced_pose_schur.pop_back();
  CheckApiRejected(options, schur_shape);

  CudaOnlineHealthCapture scale_shape = IdentityCapture();
  scale_shape.pose_jacobi_scaling.pop_back();
  CheckApiRejected(options, scale_shape);
}

BOOST_AUTO_TEST_CASE(InvalidPoseDimensionsAreRejected) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture zero = IdentityCapture();
  zero.pose_dimension = 0;
  CheckApiRejected(options, zero);

  CudaOnlineHealthCapture above_limit = IdentityCapture();
  above_limit.pose_dimension = 126;
  CheckApiRejected(options, above_limit);

  CudaOnlineHealthCapture not_pose_block_aligned = IdentityCapture();
  not_pose_block_aligned.pose_dimension = 10;
  CheckApiRejected(options, not_pose_block_aligned);
}

BOOST_AUTO_TEST_CASE(PoseDimensionEndpointsAreAccepted) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  const size_t pose_dimensions[] = {6, 120};
  for (const size_t pose_dimension : pose_dimensions) {
    const CudaOnlineHealthCapture capture = IdentityCapture(1, pose_dimension);
    const Evaluation evaluation = RunEvaluation(options, capture);
    BOOST_REQUIRE_MESSAGE(evaluation.api_success, evaluation.error);
    BOOST_CHECK(evaluation.result.completed);
    BOOST_CHECK(evaluation.result.passed);
    BOOST_CHECK_EQUAL(evaluation.result.pose_dimension, pose_dimension);
    BOOST_CHECK_EQUAL(evaluation.result.schur_rank, pose_dimension);
  }
}

BOOST_AUTO_TEST_CASE(NoVariablePointsIsRejected) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture();
  capture.point_hessians.clear();
  CheckApiRejected(options, capture);
}

BOOST_AUTO_TEST_CASE(InvalidEnabledOptionsAreRejected) {
  const CudaOnlineHealthCapture capture = IdentityCapture();
  CudaOnlineHealthOptions options = EnabledOptions();
  options.relative_rank_tolerance =
      std::numeric_limits<double>::quiet_NaN();
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.relative_rank_tolerance = 0.0;
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.relative_rank_tolerance = 1.0;
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.max_condition_number = std::numeric_limits<double>::infinity();
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.max_condition_number = std::nextafter(1.0, 0.0);
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.max_pose_dimension = 0;
  CheckApiRejected(options, capture);

  options = EnabledOptions();
  options.max_pose_dimension = 121;
  CheckApiRejected(options, capture);
}

BOOST_AUTO_TEST_CASE(ConfiguredPoseDimensionLimitIsEnforced) {
  CudaOnlineHealthOptions options = EnabledOptions();
  options.max_pose_dimension = 6;
  const CudaOnlineHealthCapture capture = IdentityCapture();
  CheckApiRejected(options, capture);
}

BOOST_AUTO_TEST_CASE(EvaluationDoesNotModifyCapture) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  CudaOnlineHealthCapture capture = IdentityCapture(2);
  SetPointMatrix(&capture, 0,
                 {{4.0, 2.0, 0.0,
                   0.0, 4.0, 0.0,
                   0.0, 0.0, 6.0}});
  SetPointDiagonal(&capture, 1, 2.0, 3.0, 8.0);
  capture.reduced_pose_schur[1] = 0.25;
  capture.reduced_pose_schur[capture.pose_dimension] = 0.75;
  capture.pose_jacobi_scaling[0] = 2.0;
  capture.pose_jacobi_scaling[1] = 3.0;

  const CudaOnlineHealthCapture before = capture;
  const Evaluation evaluation = RunEvaluation(options, capture);
  BOOST_REQUIRE(evaluation.api_success);
  BOOST_CHECK_EQUAL(capture.pose_dimension, before.pose_dimension);
  BOOST_CHECK(capture.point_hessians == before.point_hessians);
  BOOST_CHECK(capture.reduced_pose_schur == before.reduced_pose_schur);
  BOOST_CHECK(capture.pose_jacobi_scaling == before.pose_jacobi_scaling);
}

BOOST_AUTO_TEST_CASE(ExceptionAfterFirstPointPublishesCleanFailureAtomically) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  const CudaOnlineHealthCapture capture = IdentityCapture(2);
  CudaOnlineHealthResult result = PrefilledResult();
  std::string error = "stale error";
  SetCudaOnlineHealthFailureInjectionForTesting(
      CudaOnlineHealthFailureInjectionForTesting::kThrowAfterFirstPoint);

  BOOST_CHECK(!EvaluateCudaOnlineHealth(options, capture, &result, &error));
  CheckCleanRequestedFailure(
      options, capture, result, error,
      "online health evaluation raised an exception");

  const Evaluation recovery = RunEvaluation(options, capture);
  BOOST_REQUIRE_MESSAGE(recovery.api_success, recovery.error);
  BOOST_CHECK(recovery.result.completed);
  BOOST_CHECK(recovery.result.passed);
  BOOST_CHECK_EQUAL(recovery.result.point_results.size(), 2);
}

BOOST_AUTO_TEST_CASE(NullOutputsAreRejectedWithoutThrowing) {
  const CudaOnlineHealthOptions options = EnabledOptions();
  const CudaOnlineHealthCapture capture = IdentityCapture();
  std::string error;
  BOOST_CHECK(!EvaluateCudaOnlineHealth(options, capture, nullptr, &error));
  BOOST_CHECK(!error.empty());

  CudaOnlineHealthResult result;
  result.requested = true;
  BOOST_CHECK(!EvaluateCudaOnlineHealth(options, capture, &result, nullptr));
  BOOST_CHECK(!result.requested);
  BOOST_CHECK(!result.completed);
  BOOST_CHECK(!result.passed);
}

}  // namespace gpu_ba
}  // namespace colmap
