#include "gpu_ba/online_health.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace colmap {
namespace gpu_ba {
namespace {

constexpr size_t kPointDimension = 3;
constexpr size_t kPointMatrixSize = kPointDimension * kPointDimension;
constexpr size_t kHardMaxPoseDimension = 120;
constexpr char kOutputRequirementFailure[] =
    "online health result and error outputs are required";
constexpr char kUnhandledFailure[] =
    "online health evaluation raised an exception";

std::atomic<CudaOnlineHealthFailureInjectionForTesting>
    g_failure_injection_for_testing(
        CudaOnlineHealthFailureInjectionForTesting::kNone);

static_assert(
    std::is_nothrow_move_assignable<CudaOnlineHealthResult>::value,
    "online health result publication must be noexcept movable");
static_assert(std::is_nothrow_move_assignable<std::string>::value,
              "online health error publication must be noexcept movable");

void SetTextNoexcept(const char* text, std::string* output) noexcept {
  if (output == nullptr) return;
  try {
    *output = text;
  } catch (...) {
    output->clear();
  }
}

void InitializeRequestedResult(const CudaOnlineHealthOptions& options,
                               const CudaOnlineHealthCapture& capture,
                               CudaOnlineHealthResult* result) noexcept {
  result->requested = true;
  result->relative_rank_tolerance = options.relative_rank_tolerance;
  result->max_condition_number = options.max_condition_number;
  result->max_pose_dimension = options.max_pose_dimension;
  result->pose_dimension = capture.pose_dimension;
}

void SetFailureNoexcept(const CudaOnlineHealthOptions& options,
                        const CudaOnlineHealthCapture& capture,
                        const bool requested,
                        const char* reason,
                        CudaOnlineHealthResult* result,
                        std::string* error) noexcept {
  CudaOnlineHealthResult failure_result;
  if (requested) {
    InitializeRequestedResult(options, capture, &failure_result);
  }
  SetTextNoexcept(reason, &failure_result.failure_reason);
  std::string failure_error;
  SetTextNoexcept(reason, &failure_error);
  *result = std::move(failure_result);
  *error = std::move(failure_error);
}

bool RejectEvaluation(const CudaOnlineHealthOptions& options,
                      const CudaOnlineHealthCapture& capture,
                      const std::string& reason,
                      CudaOnlineHealthResult* result,
                      std::string* error) {
  CudaOnlineHealthResult failure_result;
  InitializeRequestedResult(options, capture, &failure_result);
  failure_result.failure_reason = reason;
  std::string failure_error = reason;
  *result = std::move(failure_result);
  *error = std::move(failure_error);
  return false;
}

void CommitOutputsNoexcept(CudaOnlineHealthResult* local_result,
                           std::string* local_error,
                           CudaOnlineHealthResult* result,
                           std::string* error) noexcept {
  if (result != nullptr) *result = std::move(*local_result);
  if (error != nullptr) *error = std::move(*local_error);
}

bool ConsumeThrowAfterFirstPointForTesting() noexcept {
  CudaOnlineHealthFailureInjectionForTesting expected =
      CudaOnlineHealthFailureInjectionForTesting::kThrowAfterFirstPoint;
  return g_failure_injection_for_testing.compare_exchange_strong(
      expected, CudaOnlineHealthFailureInjectionForTesting::kNone,
      std::memory_order_acq_rel, std::memory_order_relaxed);
}

struct InjectedOnlineHealthFailure {};

bool AllFinite(const std::vector<double>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

template <typename MatrixType>
bool EvaluateMatrix(const MatrixType& matrix,
                    const CudaOnlineHealthOptions& options,
                    const std::string& label,
                    CudaOnlineHealthMatrixResult* result,
                    std::string* error) {
  *result = CudaOnlineHealthMatrixResult();
  result->requested = true;
  result->dimension = static_cast<size_t>(matrix.rows());
  result->relative_rank_tolerance = options.relative_rank_tolerance;
  result->max_condition_number = options.max_condition_number;

  if (!matrix.array().isFinite().all()) {
    *error = label + " is non-finite after transformation";
    return false;
  }

  Eigen::SelfAdjointEigenSolver<typename MatrixType::PlainObject> eigen_solver(
      matrix, Eigen::EigenvaluesOnly);
  if (eigen_solver.info() != Eigen::Success) {
    // Eigen exposes no deterministic finite input that forces this status. The
    // test hook therefore exercises exception rollback without fabricating it.
    *error = label + " eigensolver failed";
    return false;
  }
  const auto& eigenvalues = eigen_solver.eigenvalues();
  if (!eigenvalues.array().isFinite().all()) {
    *error = label + " eigensolver produced non-finite eigenvalues";
    return false;
  }

  result->min_eigenvalue = eigenvalues.minCoeff();
  result->max_eigenvalue = eigenvalues.maxCoeff();
  result->spectral_radius =
      std::max(std::abs(result->min_eigenvalue),
               std::abs(result->max_eigenvalue));
  result->rank_threshold =
      options.relative_rank_tolerance * result->spectral_radius;
  if (!std::isfinite(result->rank_threshold)) {
    *error = label + " rank threshold is non-finite";
    return false;
  }

  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    const double eigenvalue = eigenvalues[index];
    if (eigenvalue > result->rank_threshold) ++result->rank;
    if (eigenvalue < -result->rank_threshold) {
      result->has_significant_negative_eigenvalue = true;
    }
  }
  result->full_rank = result->rank == result->dimension;
  if (result->full_rank) {
    result->condition_number =
        result->max_eigenvalue / result->min_eigenvalue;
    if (!std::isfinite(result->condition_number)) {
      *error = label + " condition number is non-finite";
      return false;
    }
  } else {
    result->condition_number = std::numeric_limits<double>::infinity();
  }
  result->condition_within_limit =
      result->condition_number <= options.max_condition_number;
  result->passed = result->full_rank &&
                   !result->has_significant_negative_eigenvalue &&
                   result->condition_within_limit;
  result->finite = true;
  result->completed = true;

  if (result->has_significant_negative_eigenvalue) {
    result->failure_reason = "significant negative eigenvalue";
  } else if (!result->full_rank) {
    result->failure_reason = "rank deficient";
  } else if (!result->condition_within_limit) {
    result->failure_reason = "condition number exceeds maximum";
  }
  return true;
}

Eigen::Matrix3d SymmetrizePointMatrix(const std::vector<double>& values,
                                      const size_t offset) {
  Eigen::Matrix3d symmetric;
  for (size_t row = 0; row < kPointDimension; ++row) {
    for (size_t column = 0; column < kPointDimension; ++column) {
      const double forward =
          values[offset + row * kPointDimension + column];
      const double transpose =
          values[offset + column * kPointDimension + row];
      symmetric(static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(column)) =
          0.5 * forward + 0.5 * transpose;
    }
  }
  return symmetric;
}

Eigen::MatrixXd ScaledSymmetricSchur(
    const CudaOnlineHealthCapture& capture) {
  const size_t dimension = capture.pose_dimension;
  Eigen::MatrixXd scaled(static_cast<Eigen::Index>(dimension),
                         static_cast<Eigen::Index>(dimension));
  for (size_t row = 0; row < dimension; ++row) {
    for (size_t column = 0; column < dimension; ++column) {
      const double forward =
          capture.reduced_pose_schur[row * dimension + column];
      const double transpose =
          capture.reduced_pose_schur[column * dimension + row];
      const double symmetric = 0.5 * forward + 0.5 * transpose;
      scaled(static_cast<Eigen::Index>(row),
             static_cast<Eigen::Index>(column)) =
          capture.pose_jacobi_scaling[row] * symmetric *
          capture.pose_jacobi_scaling[column];
    }
  }
  return scaled;
}

bool EvaluateCudaOnlineHealthImpl(const CudaOnlineHealthOptions& options,
                                  const CudaOnlineHealthCapture& capture,
                                  CudaOnlineHealthResult* result,
                                  std::string* error) {
  *result = CudaOnlineHealthResult();
  error->clear();
  if (!options.enabled) return true;

  InitializeRequestedResult(options, capture, result);

  if (!std::isfinite(options.relative_rank_tolerance) ||
      options.relative_rank_tolerance <= 0.0 ||
      options.relative_rank_tolerance >= 1.0) {
    return RejectEvaluation(
        options, capture,
        "relative_rank_tolerance must be finite and strictly between 0 and 1",
        result, error);
  }
  if (!std::isfinite(options.max_condition_number) ||
      options.max_condition_number < 1.0) {
    return RejectEvaluation(options, capture,
                            "max_condition_number must be finite and at least 1",
                            result, error);
  }
  if (options.max_pose_dimension == 0 ||
      options.max_pose_dimension > kHardMaxPoseDimension) {
    return RejectEvaluation(options, capture,
                            "max_pose_dimension must be between 1 and 120",
                            result, error);
  }

  if (capture.pose_dimension == 0) {
    return RejectEvaluation(options, capture,
                            "pose_dimension must be positive", result, error);
  }
  if (capture.pose_dimension > options.max_pose_dimension) {
    return RejectEvaluation(options, capture,
                            "pose_dimension exceeds max_pose_dimension", result,
                            error);
  }
  if (capture.pose_dimension % 6 != 0) {
    return RejectEvaluation(options, capture,
                            "pose_dimension must be a multiple of 6", result,
                            error);
  }
  if (capture.point_hessians.empty()) {
    return RejectEvaluation(
        options, capture,
        "online health requires at least one variable point Hessian", result,
        error);
  }
  if (capture.point_hessians.size() % kPointMatrixSize != 0) {
    return RejectEvaluation(options, capture,
                            "point_hessians size must be a multiple of 9",
                            result, error);
  }

  const size_t expected_schur_size =
      capture.pose_dimension * capture.pose_dimension;
  if (capture.reduced_pose_schur.size() != expected_schur_size) {
    return RejectEvaluation(
        options, capture,
        "reduced_pose_schur size does not match pose_dimension squared", result,
        error);
  }
  if (capture.pose_jacobi_scaling.size() != capture.pose_dimension) {
    return RejectEvaluation(
        options, capture,
        "pose_jacobi_scaling size does not match pose_dimension", result,
        error);
  }
  if (!AllFinite(capture.point_hessians)) {
    return RejectEvaluation(options, capture,
                            "point_hessians contains a non-finite value", result,
                            error);
  }
  if (!AllFinite(capture.reduced_pose_schur)) {
    return RejectEvaluation(options, capture,
                            "reduced_pose_schur contains a non-finite value",
                            result, error);
  }
  for (const double scale : capture.pose_jacobi_scaling) {
    if (!std::isfinite(scale)) {
      return RejectEvaluation(
          options, capture,
          "pose_jacobi_scaling contains a non-finite value", result, error);
    }
    if (scale <= 0.0) {
      return RejectEvaluation(
          options, capture,
          "pose_jacobi_scaling values must be strictly positive", result,
          error);
    }
  }

  result->point_count = capture.point_hessians.size() / kPointMatrixSize;
  result->min_rank = kPointDimension;
  result->point_results.resize(result->point_count);
  bool point_matrices_passed = true;
  for (size_t point_index = 0; point_index < result->point_count;
       ++point_index) {
    const Eigen::Matrix3d symmetric = SymmetrizePointMatrix(
        capture.point_hessians, point_index * kPointMatrixSize);
    std::string matrix_error;
    CudaOnlineHealthMatrixResult& point_result =
        result->point_results[point_index];
    if (!EvaluateMatrix(symmetric, options, "variable point Hessian",
                        &point_result, &matrix_error)) {
      return RejectEvaluation(options, capture, matrix_error, result, error);
    }
    if (point_result.full_rank) ++result->full_rank_count;
    result->min_rank = std::min(result->min_rank, point_result.rank);
    result->worst_condition =
        std::max(result->worst_condition, point_result.condition_number);
    result->point_has_significant_negative_eigenvalue =
        result->point_has_significant_negative_eigenvalue ||
        point_result.has_significant_negative_eigenvalue;
    if (!point_result.passed) {
      point_matrices_passed = false;
      if (result->failure_reason.empty()) {
        result->failure_reason =
            "variable point Hessian " + std::to_string(point_index) +
            " failed: " + point_result.failure_reason;
      }
    }
    if (point_index == 0 && ConsumeThrowAfterFirstPointForTesting()) {
      throw InjectedOnlineHealthFailure();
    }
  }
  result->point_matrices_finite = true;

  const Eigen::MatrixXd scaled_schur = ScaledSymmetricSchur(capture);
  std::string matrix_error;
  if (!EvaluateMatrix(scaled_schur, options, "scaled reduced pose Schur",
                      &result->schur_result, &matrix_error)) {
    return RejectEvaluation(options, capture, matrix_error, result, error);
  }
  result->schur_finite = result->schur_result.finite;
  result->schur_rank = result->schur_result.rank;
  result->schur_condition = result->schur_result.condition_number;
  result->schur_min_eigenvalue = result->schur_result.min_eigenvalue;
  result->schur_max_eigenvalue = result->schur_result.max_eigenvalue;
  result->schur_has_significant_negative_eigenvalue =
      result->schur_result.has_significant_negative_eigenvalue;
  if (!result->schur_result.passed && result->failure_reason.empty()) {
    result->failure_reason =
        "scaled reduced pose Schur failed: " +
        result->schur_result.failure_reason;
  }

  result->has_significant_negative_eigenvalue =
      result->point_has_significant_negative_eigenvalue ||
      result->schur_has_significant_negative_eigenvalue;
  result->finite = true;
  result->completed = true;
  result->passed = point_matrices_passed && result->schur_result.passed;
  return true;
}

}  // namespace

void SetCudaOnlineHealthFailureInjectionForTesting(
    const CudaOnlineHealthFailureInjectionForTesting failure) noexcept {
  g_failure_injection_for_testing.store(failure, std::memory_order_release);
}

bool EvaluateCudaOnlineHealth(const CudaOnlineHealthOptions& options,
                              const CudaOnlineHealthCapture& capture,
                              CudaOnlineHealthResult* result,
                              std::string* error) noexcept {
  CudaOnlineHealthResult local_result;
  std::string local_error;
  bool success = false;

  if (result == nullptr || error == nullptr) {
    SetFailureNoexcept(options, capture, false, kOutputRequirementFailure,
                       &local_result, &local_error);
  } else {
    try {
      success = EvaluateCudaOnlineHealthImpl(options, capture, &local_result,
                                             &local_error);
    } catch (...) {
      SetFailureNoexcept(options, capture, options.enabled, kUnhandledFailure,
                         &local_result, &local_error);
    }
  }

  CommitOutputsNoexcept(&local_result, &local_error, result, error);
  return success;
}

}  // namespace gpu_ba
}  // namespace colmap
