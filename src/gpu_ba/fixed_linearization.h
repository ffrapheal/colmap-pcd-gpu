#ifndef COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_
#define COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_ba/snapshot.h"
#include "gpu_ba/validation.h"

namespace colmap {
namespace gpu_ba {

enum class FixedStepSemantics : uint8_t {
  kLegacy = 0,
  kCeres14 = 1,
};

struct FixedLinearizationOptions {
  double lambda = 1e-4;
  double diagonal_floor = 1e-12;
  double linear_backward_error_limit = 1e-10;
  FixedStepSemantics step_semantics = FixedStepSemantics::kLegacy;
};

struct FixedLinearizationResult {
  bool pass = false;
  std::string step_semantics;
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

enum class CpuTerminationType : uint8_t {
  kConvergence = 0,
  kNoConvergence = 1,
  kFailure = 2,
};

enum class CustomCpuSolveMode : uint8_t {
  kLegacySingleStep = 0,
  kCeres14SingleStep = 1,
  kFull = 2,
};

struct CustomCpuIteration {
  int32_t iteration = 0;
  double cost_before = 0.0;
  double trial_cost = 0.0;
  double cost_after = 0.0;
  double projected_gradient_max_norm = 0.0;
  double scaled_gradient_norm = 0.0;
  double gradient_norm = 0.0;
  double radius_before = 0.0;
  double radius_after = 0.0;
  double lambda_before = 0.0;
  double lambda_after = 0.0;
  double lm_diagonal_min = 0.0;
  double lm_diagonal_max = 0.0;
  bool lm_diagonal_observed = false;
  double predicted_reduction = 0.0;
  double actual_reduction = 0.0;
  double rho = 0.0;
  double function_metric = 0.0;
  double parameter_metric = 0.0;
  double step_norm = 0.0;
  double backward_error = 0.0;
  double eta = 0.0;
  int32_t linear_solver_iterations = 0;
  int32_t successful_steps = 0;
  int32_t unsuccessful_steps = 0;
  int32_t cumulative_invalid_steps = 0;
  bool factorization_success = false;
  bool step_valid = false;
  bool trial_finite = false;
  bool accepted = false;
  bool invalid = false;
  std::string termination_reason;
};

// Fully materialized solve configuration. Production replay constructs this
// from the captured effective Ceres options; the convenience overload below
// exists only for older canonical diagnostic snapshots.
struct CustomCpuSolverOptions {
  std::string linear_solver_type = "DENSE_SCHUR";
  std::string residual_order = "source_insertion_order";
  std::string parameter_order = "source_first_insertion_order";
  std::string loss_type = "TRIVIAL";
  double loss_scale = 1.0;
  int32_t requested_num_threads = 1;
  int32_t effective_num_threads = 1;
  int32_t max_num_iterations = 0;
  int32_t min_linear_solver_iterations = 0;
  int32_t max_linear_solver_iterations = 0;
  int32_t max_num_consecutive_invalid_steps = 10;
  double max_solver_time_in_seconds = 1e9;
  double function_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  double parameter_tolerance = 0.0;
  double initial_trust_region_radius = 1e4;
  double min_trust_region_radius = 1e-32;
  double max_trust_region_radius = 1e16;
  double min_lm_diagonal = 1e-6;
  double max_lm_diagonal = 1e32;
  double min_relative_decrease = 1e-3;
  double eta = 0.1;
  bool jacobi_scaling = true;
  bool use_nonmonotonic_steps = false;
  bool use_inner_iterations = false;
  bool capture_state_trace = false;
};

struct LidarNearZeroSummary {
  double threshold = 1e-12;
  uint64_t total_samples = 0;
  uint64_t exact_zero_samples = 0;
  uint64_t near_zero_samples = 0;
  std::vector<uint64_t> exact_zero_point_ids;
  std::vector<uint64_t> near_zero_point_ids;
};

struct CpuGradientNorms {
  double projected_max_norm = 0.0;
  double projected_l2_norm = 0.0;
  double raw_tangent_max_norm = 0.0;
  double scaled_max_norm = 0.0;
  uint64_t variable_rotation_blocks = 0;
  uint64_t variable_translation_components = 0;
  uint64_t variable_point_blocks = 0;
  std::string worst_ambient_id;
};

// Read-only, flat export of the already-validated custom_cpu canonical
// linearization. Phase 6 uses this as the independent CPU reference for CUDA
// B/C/E, gradient, and damping comparisons; no CUDA execution depends on it.
struct CustomCpuPoseBlockExport {
  uint32_t image_id = 0;
  uint32_t dimension = 0;
  std::vector<int32_t> free_translation_indices;
  std::vector<double> hessian;  // row-major dimension x dimension
  std::vector<double> gradient;
  std::vector<double> jacobi_scaling;
  std::vector<double> damping;
};

struct CustomCpuPointBlockExport {
  uint64_t point3D_id = 0;
  std::array<double, 9> hessian{{}};
  std::array<double, 3> gradient{{}};
  std::array<double, 3> jacobi_scaling{{}};
  std::array<double, 3> damping{{}};
};

struct CustomCpuEdgeBlockExport {
  uint32_t pose_index = 0;
  uint32_t point_index = 0;
  uint32_t pose_dimension = 0;
  std::array<double, 18> value{{}};  // row-major pose_dimension x 3
};

struct CustomCpuLinearizationExport {
  double cost = 0.0;
  std::string layout_sha256;
  std::vector<CustomCpuPoseBlockExport> poses;
  std::vector<CustomCpuPointBlockExport> points;
  std::vector<CustomCpuEdgeBlockExport> edges;
  CpuGradientNorms gradient_norms;
};

struct CustomCpuCanonicalStepExport {
  CustomCpuLinearizationExport linearization;
  uint64_t pose_dimension = 0;
  uint64_t point_dimension = 0;
  std::vector<double> schur;  // row-major pose_dimension x pose_dimension
  std::vector<double> rhs;
  std::vector<double> camera_delta;
  std::vector<double> point_delta;
  double backward_error = 0.0;
  double predicted_reduction = 0.0;
  double trial_cost = 0.0;
  Snapshot trial_state;
};

struct CustomCpuSolveResult {
  bool success = false;
  CustomCpuSolveMode solve_mode = CustomCpuSolveMode::kFull;
  CpuTerminationType termination_type = CpuTerminationType::kFailure;
  std::string termination_reason;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double initial_projected_gradient_max_norm = 0.0;
  double final_projected_gradient_max_norm = 0.0;
  double initial_scaled_gradient_norm = 0.0;
  double final_scaled_gradient_norm = 0.0;
  double final_lambda = 0.0;
  double function_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  double parameter_tolerance = 0.0;
  int32_t max_trial_iterations = 0;
  int32_t trial_iterations = 0;
  int32_t accepted_steps = 0;
  int32_t rejected_steps = 0;
  int32_t invalid_steps = 0;
  int32_t factorization_failures = 0;
  int32_t num_linear_solves = 0;
  CustomCpuSolverOptions effective_options;
  LidarNearZeroSummary initial_lidar_near_zero;
  LidarNearZeroSummary final_lidar_near_zero;
  std::vector<double> first_lm_diagonal;
  std::string first_lm_diagonal_sha256;
  std::vector<CustomCpuIteration> trace;
  // Debug-only accepted-state capture. Disabled in production to avoid
  // retaining a full Snapshot for every iteration.
  std::vector<Snapshot> accepted_state_trace;
};

struct CeresCpuSolveResult {
  bool success = false;
  CpuTerminationType termination_type = CpuTerminationType::kFailure;
  std::string termination_reason;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double final_gradient_max_norm = 0.0;
  int32_t iterations = 0;
  int32_t accepted_steps = 0;
  int32_t rejected_steps = 0;
  int32_t invalid_steps = 0;
  double total_time_seconds = 0.0;
  uint64_t ba_image_count = 0;
  std::string requested_linear_solver_type;
  std::string actual_linear_solver_type;
  int32_t requested_num_threads = 1;
  int32_t effective_num_threads = 1;
  bool jacobi_scaling = true;
  bool deterministic_oracle = true;
  std::string residual_order = "canonical";
  double initial_trust_region_radius = 1e4;
  double min_trust_region_radius = 1e-32;
  double max_trust_region_radius = 1e16;
  double min_lm_diagonal = 1e-6;
  double max_lm_diagonal = 1e32;
  double min_relative_decrease = 1e-3;
  int32_t max_consecutive_invalid_steps = 5;
  bool damping_oracle_observed = false;
  std::string damping_oracle_source;
  std::string damping_oracle_path;
  std::string damping_oracle_sha256;
  std::string damping_oracle_text_precision;
  bool damping_oracle_auxiliary_files_removed = false;
  uint64_t damping_oracle_scaled_jacobian_nonzeros = 0;
  double damping_oracle_scaled_column_norm_min = 0.0;
  double damping_oracle_scaled_column_norm_max = 0.0;
  double damping_oracle_clamped_diagonal_min = 0.0;
  double damping_oracle_clamped_diagonal_max = 0.0;
  std::vector<double> damping_oracle_lm_diagonal;
  LidarNearZeroSummary initial_lidar_near_zero;
  LidarNearZeroSummary final_lidar_near_zero;
  std::vector<CustomCpuIteration> trace;
};

struct CpuSolveComparisonResult {
  bool pass = false;
  bool strict_cost_pass = false;
  bool gradient_pass = false;
  bool termination_match = false;
  bool provisional_state_pass = false;
  bool quality_diagnostic_pass = false;
  bool trace_structure_pass = false;
  uint64_t trace_structure_iterations_checked = 0;
  uint64_t trace_structure_mismatch_count = 0;
  int32_t first_trace_structure_mismatch_iteration = -1;
  std::string first_trace_structure_mismatch_field;
  std::string first_trace_structure_reference;
  std::string first_trace_structure_candidate;
  std::string ceres_termination_category;
  std::string custom_termination_category;
  int32_t ceres_summary_accepted_steps = 0;
  int32_t custom_summary_accepted_steps = 0;
  int32_t ceres_summary_rejected_steps = 0;
  int32_t custom_summary_rejected_steps = 0;
  int32_t ceres_summary_invalid_steps = 0;
  int32_t custom_summary_invalid_steps = 0;
  bool trace_numeric_within_tolerance = true;
  uint64_t trace_numeric_iterations_scanned = 0;
  int32_t first_trace_divergence_iteration = -1;
  std::string first_trace_divergence_field;
  std::string first_trace_divergence_reason;
  double first_trace_reference = 0.0;
  double first_trace_candidate = 0.0;
  double first_trace_atol = 0.0;
  double first_trace_rtol = 0.0;
  double first_trace_absolute_error = 0.0;
  double first_trace_relative_error = 0.0;
  bool damping_oracle_required = false;
  bool damping_oracle_available = false;
  bool damping_oracle_pass = false;
  uint64_t damping_oracle_ceres_count = 0;
  uint64_t damping_oracle_custom_count = 0;
  double damping_oracle_atol = 5.1e-7;
  double damping_oracle_rtol = 1e-6;
  uint64_t damping_oracle_failures = 0;
  double damping_oracle_max_absolute_error = 0.0;
  double damping_oracle_max_relative_error = 0.0;
  std::string damping_oracle_worst_id;
  double damping_oracle_worst_ceres = 0.0;
  double damping_oracle_worst_custom = 0.0;
  double ceres_projected_gradient_max_norm = 0.0;
  double custom_projected_gradient_max_norm = 0.0;
  double ceres_scaled_gradient_norm = 0.0;
  double custom_scaled_gradient_norm = 0.0;
  double cost_absolute_error = 0.0;
  double cost_relative_error = 0.0;
  double rotation_max_degrees = 0.0;
  double rotation_p95_degrees = 0.0;
  double rotation_rms_degrees = 0.0;
  double translation_max = 0.0;
  double translation_p95 = 0.0;
  double translation_rms = 0.0;
  double point_max = 0.0;
  double conditioned_point_max = 0.0;
  double point_p95 = 0.0;
  double point_rms = 0.0;
  double conditioned_point_p95 = 0.0;
  double conditioned_point_rms = 0.0;
  double condition_excluded_fraction = 0.0;
  bool conditioned_point_strict_pass = false;
  double point_condition_exclusion_threshold = 1e12;
  uint64_t condition_excluded_points = 0;
  std::vector<uint64_t> condition_excluded_point_ids;
  std::string worst_conditioned_point_id;
  uint64_t compared_poses = 0;
  uint64_t compared_points = 0;
  std::string worst_rotation_id;
  std::string worst_translation_id;
  std::string worst_point_id;
  std::array<double, 3> worst_point_ceres_xyz{{0.0, 0.0, 0.0}};
  std::array<double, 3> worst_point_custom_xyz{{0.0, 0.0, 0.0}};
  uint64_t worst_point_observation_count = 0;
  uint64_t worst_point_track_length = 0;
  bool worst_point_has_lidar = false;
  double worst_point_ceres_block_condition = 0.0;
  double worst_point_custom_block_condition = 0.0;
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

std::string CpuTerminationTypeName(CpuTerminationType type);

bool EvaluateCustomCpuGradientNorms(const Snapshot& snapshot,
                                    CpuGradientNorms* norms,
                                    std::string* error);

bool ExportCustomCpuCanonicalLinearization(
    const Snapshot& snapshot,
    double min_lm_diagonal,
    double max_lm_diagonal,
    CustomCpuLinearizationExport* output,
    std::string* error);

bool ExportCustomCpuSourceLinearization(
    const Snapshot& snapshot,
    int reduction_threads,
    double min_lm_diagonal,
    double max_lm_diagonal,
    CustomCpuLinearizationExport* output,
    std::string* error);

bool ExportCustomCpuCanonicalStep(const Snapshot& snapshot,
                                  double lambda,
                                  double min_lm_diagonal,
                                  double max_lm_diagonal,
                                  CustomCpuCanonicalStepExport* output,
                                  std::string* error);

bool ExportCustomCpuSourceStep(const Snapshot& snapshot,
                               int reduction_threads,
                               double lambda,
                               double min_lm_diagonal,
                               double max_lm_diagonal,
                               CustomCpuCanonicalStepExport* output,
                               std::string* error);

bool RunCustomCpuSolve(const Snapshot& snapshot,
                       double initial_lambda,
                       CustomCpuSolveMode mode,
                       CustomCpuSolveResult* result,
                       Snapshot* final_state,
                       std::string* error);

bool RunCustomCpuSolve(const Snapshot& snapshot,
                       const CustomCpuSolverOptions& options,
                       CustomCpuSolveMode mode,
                       CustomCpuSolveResult* result,
                       Snapshot* final_state,
                       std::string* error);

bool RunCeresCpuSolve(const Snapshot& snapshot,
                      bool single_step,
                      const std::string& damping_oracle_directory,
                      CeresCpuSolveResult* result,
                      Snapshot* final_state,
                      std::string* error);

bool CompareCpuSolveResults(const Snapshot& ceres_state,
                            const Snapshot& custom_state,
                            const CeresCpuSolveResult& ceres_result,
                            const CustomCpuSolveResult& custom_result,
                            CpuSolveComparisonResult* comparison,
                            std::string* error);

std::string CustomCpuSolveJson(const CustomCpuSolveResult& result,
                               size_t indent_spaces = 2);

std::string CeresCpuSolveJson(const CeresCpuSolveResult& result,
                              size_t indent_spaces = 2);

std::string CpuSolveComparisonJson(
    const CpuSolveComparisonResult& result,
    size_t indent_spaces = 2);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_FIXED_LINEARIZATION_H_
