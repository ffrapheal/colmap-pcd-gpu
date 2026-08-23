#ifndef COLMAP_SRC_GPU_BA_CERES_FIDELITY_H_
#define COLMAP_SRC_GPU_BA_CERES_FIDELITY_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <ceres/ceres.h>

#include "base/reconstruction.h"
#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {

constexpr uint32_t kCeresFidelitySchemaVersion = 1;

struct FidelityProvenanceSnapshot {
  std::string executable_path;
  std::string executable_sha256;
  std::string libceres_path;
  std::string libceres_sha256;
  std::string git_head;
  std::string dirty_diff_sha256;
};

struct LossSpecificationSnapshot {
  std::string residual_class;
  std::string type;
  double scale = 1.0;
  uint64_t residual_block_count = 0;
};

struct ParameterConstraintSnapshot {
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;
  bool constant = false;
  std::string local_parameterization = "euclidean";
  std::vector<int32_t> constant_indices;
};

// This is the final Solver::Options value passed by BundleAdjuster to
// ceres::Solve. The original requested thread counts are retained separately
// because BundleAdjuster may replace them after applying its residual-count
// threshold.
struct EffectiveCeresOptionsSnapshot {
  int32_t minimizer_type = 0;
  int32_t trust_region_strategy_type = 0;
  int32_t dogleg_type = 0;
  int32_t linear_solver_type = 0;
  int32_t preconditioner_type = 0;
  int32_t visibility_clustering_type = 0;
  int32_t dense_linear_algebra_library_type = 0;
  int32_t sparse_linear_algebra_library_type = 0;
  int32_t logging_type = 0;

  int32_t original_requested_num_threads = 0;
  int32_t original_requested_num_linear_solver_threads = 0;
  int32_t num_threads = 0;
  int32_t num_linear_solver_threads = 0;
  int32_t min_num_residuals_for_multi_threading = 0;

  int32_t max_num_iterations = 0;
  int32_t min_linear_solver_iterations = 0;
  int32_t max_linear_solver_iterations = 0;
  int32_t max_num_consecutive_invalid_steps = 0;
  int32_t max_consecutive_nonmonotonic_steps = 0;

  double max_solver_time_in_seconds = 0.0;
  double function_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  double parameter_tolerance = 0.0;
  double initial_trust_region_radius = 0.0;
  double min_trust_region_radius = 0.0;
  double max_trust_region_radius = 0.0;
  double min_lm_diagonal = 0.0;
  double max_lm_diagonal = 0.0;
  double min_relative_decrease = 0.0;
  double eta = 0.0;
  double inner_iteration_tolerance = 0.0;
  double gradient_check_relative_precision = 0.0;
  double gradient_check_numeric_derivative_relative_step_size = 0.0;

  bool jacobi_scaling = false;
  bool use_nonmonotonic_steps = false;
  bool use_inner_iterations = false;
  bool use_explicit_schur_complement = false;
  bool use_postordering = false;
  bool dynamic_sparsity = false;
  bool minimizer_progress_to_stdout = false;
  bool check_gradients = false;
  bool update_state_every_iteration = false;
  bool linear_solver_ordering_present = false;
  bool inner_iteration_ordering_present = false;
  bool evaluation_callback_present = false;
  uint64_t callback_count = 0;
  uint64_t trust_region_dump_iteration_count = 0;
  std::string trust_region_problem_dump_directory;
  int32_t trust_region_problem_dump_format_type = 0;
};

struct FidelityProblemSnapshot {
  uint64_t config_image_count = 0;
  uint64_t selected_image_count = 0;
  uint64_t reconstruction_image_count = 0;
  uint64_t snapshot_image_record_count = 0;
  uint64_t visual_residual_block_count = 0;
  uint64_t lidar_residual_block_count = 0;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t parameter_block_count = 0;
  uint64_t ambient_parameter_dimension = 0;
  uint64_t tangent_parameter_dimension = 0;
  uint64_t constant_pose_count = 0;
  uint64_t constant_point_count = 0;
  uint64_t constant_camera_count = 0;
  uint64_t quaternion_manifold_count = 0;
  uint64_t subset_translation_count = 0;
  bool has_bounds_or_constraints = false;
  std::vector<int32_t> camera_model_ids;
  std::vector<ParameterConstraintSnapshot> parameter_constraints;
  std::vector<LossSpecificationSnapshot> loss_specifications;
  std::string source_residual_order_sha256;
  std::string canonical_residual_order_sha256;
  std::string source_parameter_order_sha256;
  std::string canonical_parameter_order_sha256;
  std::string parameter_constraints_sha256;
  std::string loss_specification_sha256;
  std::string initial_state_sha256;
  std::string lidar_correspondence_sha256;
  std::string effective_options_sha256;
  std::string complete_fingerprint_sha256;
};

struct CeresIterationSnapshot {
  int32_t iteration = 0;
  bool step_is_valid = false;
  bool step_is_nonmonotonic = false;
  bool step_is_successful = false;
  double cost = 0.0;
  double cost_change = 0.0;
  double gradient_max_norm = 0.0;
  double gradient_norm = 0.0;
  double step_norm = 0.0;
  double relative_decrease = 0.0;
  double trust_region_radius = 0.0;
  double eta = 0.0;
  double step_size = 0.0;
  int32_t line_search_function_evaluations = 0;
  int32_t line_search_gradient_evaluations = 0;
  int32_t line_search_iterations = 0;
  int32_t linear_solver_iterations = 0;
  double iteration_time_in_seconds = 0.0;
  double step_solver_time_in_seconds = 0.0;
  double cumulative_time_in_seconds = 0.0;
};

struct CeresSummarySnapshot {
  int32_t minimizer_type = 0;
  int32_t termination_type = 0;
  std::string termination_message;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double fixed_cost = 0.0;
  int32_t successful_steps = 0;
  int32_t unsuccessful_steps = 0;
  int32_t invalid_steps = 0;
  int32_t num_inner_iteration_steps = 0;
  int32_t num_line_search_steps = 0;
  int32_t num_linear_solves = 0;
  double preprocessor_time_in_seconds = 0.0;
  double minimizer_time_in_seconds = 0.0;
  double postprocessor_time_in_seconds = 0.0;
  double total_time_in_seconds = 0.0;
  double linear_solver_time_in_seconds = 0.0;
  int32_t num_parameter_blocks = 0;
  int32_t num_parameters = 0;
  int32_t num_effective_parameters = 0;
  int32_t num_residual_blocks = 0;
  int32_t num_residuals = 0;
  int32_t num_parameter_blocks_reduced = 0;
  int32_t num_parameters_reduced = 0;
  int32_t num_effective_parameters_reduced = 0;
  int32_t num_residual_blocks_reduced = 0;
  int32_t num_residuals_reduced = 0;
  bool is_constrained = false;
  int32_t num_threads_given = 0;
  int32_t num_threads_used = 0;
  int32_t num_linear_solver_threads_given = 0;
  int32_t num_linear_solver_threads_used = 0;
  int32_t linear_solver_type_given = 0;
  int32_t linear_solver_type_used = 0;
  int32_t preconditioner_type_given = 0;
  int32_t preconditioner_type_used = 0;
  int32_t trust_region_strategy_type = 0;
  int32_t dogleg_type = 0;
  int32_t dense_linear_algebra_library_type = 0;
  int32_t sparse_linear_algebra_library_type = 0;
  bool inner_iterations_given = false;
  bool inner_iterations_used = false;
  std::vector<int32_t> linear_solver_ordering_given;
  std::vector<int32_t> linear_solver_ordering_used;
  std::vector<int32_t> inner_iteration_ordering_given;
  std::vector<int32_t> inner_iteration_ordering_used;
  std::string schur_structure_given;
  std::string schur_structure_used;
  std::vector<CeresIterationSnapshot> iterations;
};

struct CeresFidelityRecord {
  std::string record_kind = "original_bundle_adjuster";
  std::string run_id;
  FidelityProvenanceSnapshot provenance;
  std::string snapshot_id;
  std::string snapshot_manifest_path;
  std::string snapshot_payload_path;
  std::string snapshot_payload_sha256;
  std::string snapshot_manifest_sha256;
  std::string post_state_manifest_path;
  std::string post_state_payload_path;
  std::string post_state_payload_sha256;
  std::string final_state_sha256;
  EffectiveCeresOptionsSnapshot effective_options;
  FidelityProblemSnapshot problem;
  CeresSummarySnapshot summary;
};

struct CeresFidelityWriteResult {
  std::string prefix_path;
  std::string binary_path;
  std::string manifest_path;
  std::string binary_sha256;
  std::string manifest_sha256;
};

struct FidelityErrorSummary {
  uint64_t count = 0;
  uint64_t bitwise_differences = 0;
  uint64_t tolerance_failures = 0;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  double rms_absolute_error = 0.0;
  double p95_absolute_error = 0.0;
  std::string worst_id;
  double worst_reference = 0.0;
  double worst_candidate = 0.0;
  std::vector<double> worst_reference_values;
  std::vector<double> worst_candidate_values;
};

struct CeresFidelityComparison {
  bool pass = false;
  bool provenance_pass = false;
  bool problem_fingerprint_pass = false;
  bool effective_options_pass = false;
  bool actual_options_pass = false;
  bool trace_structure_pass = false;
  bool trace_numeric_pass = false;
  bool final_state_bitwise_identical = false;
  bool final_state_strict_pass = false;
  uint64_t trace_structure_mismatch_count = 0;
  int32_t first_trace_structure_mismatch_iteration = -1;
  std::string first_trace_structure_mismatch_field;
  int32_t first_trace_numeric_divergence_iteration = -1;
  std::string first_trace_numeric_divergence_field;
  double first_trace_reference = 0.0;
  double first_trace_candidate = 0.0;
  double first_trace_absolute_error = 0.0;
  double first_trace_relative_error = 0.0;
  std::string first_option_mismatch;
  std::string first_problem_mismatch;
  FidelityErrorSummary quaternion_ambient;
  FidelityErrorSummary rotation_degrees;
  FidelityErrorSummary translation;
  FidelityErrorSummary points;
  FidelityErrorSummary cameras;
};

struct CeresFidelityReplayResult {
  bool pass = false;
  std::string semantics = "original_fidelity";
  std::string perturbation = "none";
  std::string used_residual_order = "source_insertion_order";
  std::string used_parameter_order = "source_first_insertion_order";
  CeresFidelityRecord oracle_record;
  CeresFidelityRecord replay_record;
  CeresFidelityWriteResult replay_record_write;
  SnapshotWriteResult post_state_write;
  CeresFidelityComparison comparison;
};

EffectiveCeresOptionsSnapshot CaptureEffectiveCeresOptions(
    const ceres::Solver::Options& original_requested,
    const ceres::Solver::Options& effective,
    int min_num_residuals_for_multi_threading);

bool ApplyEffectiveCeresOptions(const EffectiveCeresOptionsSnapshot& snapshot,
                                ceres::Solver::Options* options,
                                std::string* error);

std::string EffectiveCeresOptionsSha256(
    const EffectiveCeresOptionsSnapshot& options);

CeresSummarySnapshot CaptureCeresSummary(
    const ceres::Solver::Summary& summary);

std::string CanonicalStateSha256(const Snapshot& snapshot);

bool UpdateSnapshotStateFromReconstruction(const Reconstruction& reconstruction,
                                           Snapshot* snapshot,
                                           std::string* error);

bool RestoreReconstructionStateFromSnapshot(const Snapshot& snapshot,
                                            Reconstruction* reconstruction,
                                            std::string* error);

bool CaptureFidelityProvenance(FidelityProvenanceSnapshot* provenance,
                               std::string* error);

FidelityProblemSnapshot BuildFidelityProblemSnapshot(
    const Snapshot& snapshot,
    const SnapshotIntegrity& integrity,
    const EffectiveCeresOptionsSnapshot& effective_options,
    uint64_t config_image_count,
    uint64_t scalar_residual_count,
    uint64_t parameter_block_count,
    uint64_t ambient_parameter_dimension,
    const std::vector<ParameterConstraintSnapshot>& parameter_constraints,
    const std::vector<LossSpecificationSnapshot>& loss_specifications,
    bool has_bounds_or_constraints);

bool WriteCeresFidelityRecord(const CeresFidelityRecord& record,
                              const std::string& output_dir,
                              CeresFidelityWriteResult* result,
                              std::string* error);

bool ReadCeresFidelityRecord(const std::string& path,
                             CeresFidelityRecord* record,
                             CeresFidelityWriteResult* result,
                             std::string* error);

bool CompareCeresFidelityRecords(const CeresFidelityRecord& reference,
                                 const Snapshot& reference_state,
                                 const CeresFidelityRecord& candidate,
                                 const Snapshot& candidate_state,
                                 CeresFidelityComparison* comparison,
                                 std::string* error);

bool RunOriginalFidelityReplay(const std::string& snapshot_path,
                               const std::string& oracle_path,
                               const std::string& output_dir,
                               const std::string& run_id,
                               const std::string& perturbation,
                               CeresFidelityReplayResult* result,
                               std::string* error);

std::string CeresFidelityRecordJson(const CeresFidelityRecord& record,
                                    size_t indent_spaces = 0);

std::string CeresFidelityComparisonJson(
    const CeresFidelityComparison& comparison,
    size_t indent_spaces = 0);

std::string CeresFidelityReplayJson(const CeresFidelityReplayResult& result,
                                    size_t indent_spaces = 0);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_CERES_FIDELITY_H_
