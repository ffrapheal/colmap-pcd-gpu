#ifndef COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_
#define COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_ba/snapshot.h"
#include "gpu_ba/validation.h"

namespace colmap {
namespace gpu_ba {

struct FixedLinearizationOptions {
  double lambda = 1e-4;
  double diagonal_floor = 1e-12;
  double linear_backward_error_limit = 1e-10;
};

struct FixedLinearizationResult {
  bool pass = false;
  uint64_t variable_pose_blocks = 0;
  uint64_t variable_point_blocks = 0;
  uint64_t pose_tangent_dimension = 0;
  uint64_t point_tangent_dimension = 0;
  uint64_t pose_point_edges = 0;
  uint64_t constant_pose_residuals = 0;
  uint64_t constant_point_residuals = 0;
  uint64_t fully_constant_residuals = 0;
  std::string layout_sha256;
  std::string damping_sha256;
  bool reference_point_factorization_success = false;
  bool candidate_point_factorization_success = false;
  bool reference_schur_factorization_success = false;
  bool candidate_schur_factorization_success = false;
  double reference_min_point_cholesky_diagonal = 0.0;
  double candidate_min_point_cholesky_diagonal = 0.0;
  double reference_min_schur_cholesky_diagonal = 0.0;
  double candidate_min_schur_cholesky_diagonal = 0.0;
  double reference_schur_condition_estimate = 0.0;
  double candidate_schur_condition_estimate = 0.0;
  double reference_schur_symmetry_error = 0.0;
  double candidate_schur_symmetry_error = 0.0;
  double reference_backward_error = 0.0;
  double candidate_backward_error = 0.0;
  double unfrozen_candidate_backward_error = 0.0;
  double reference_current_cost = 0.0;
  double candidate_current_cost = 0.0;
  double reference_trial_cost = 0.0;
  double candidate_trial_cost = 0.0;
  double reference_predicted_reduction = 0.0;
  double candidate_predicted_reduction = 0.0;
  double reference_actual_reduction = 0.0;
  double candidate_actual_reduction = 0.0;
  double reference_rho = 0.0;
  double candidate_rho = 0.0;
  bool reference_accept = false;
  bool candidate_accept = false;
  std::vector<ErrorSummary> metrics;
  std::vector<ErrorSummary> non_gating_end_to_end_diagnostics;
};

bool RunFixedLinearizationComparison(
    const Snapshot& snapshot,
    const FixedLinearizationOptions& options,
    FixedLinearizationResult* result,
    std::string* error);

std::string FixedLinearizationJson(
    const FixedLinearizationResult& result,
    const FixedLinearizationOptions& options,
    size_t indent_spaces = 2);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_
