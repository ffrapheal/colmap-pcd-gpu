// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_
#define COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <fstream>
#include <Eigen/Core>

#include <ceres/ceres.h>
#include "lidar/lidar_point.h"
#include "lidar/ply.h"
#include "PBA/pba.h"
#include "base/camera_rig.h"
#include "base/reconstruction.h"
#include "gpu_ba/host_problem_store.h"
#include "gpu_ba/active_ba_solve_spec.h"
#include "util/alignment.h"


namespace colmap {

namespace gpu_ba {
class SnapshotRecorder;
struct Snapshot;
struct SnapshotWriteResult;
struct NativeCudaResolvedConfig;
}

//参数的结构，包括损失函数类型，
struct BundleAdjustmentOptions {
  enum class CudaExecutionProfile {
    INVALID = -1,
    BASELINE = 0,
    RUNTIME_POOL = 1,
    ARENA = 2,
    DEVICE_SCALING = 3,
    FAST_IDENTITY = 4,
    UNIFIED_BUILDER = 5,
    COMPACT_LAYER_A = 6,
    COMPACT_CONTROL = 7,
  };
  // GPU BA backend and snapshot controls. The default preserves legacy Ceres.
  std::string ba_backend = "ceres_cpu";
  bool ba_fallback_to_ceres = true;
  std::string ba_snapshot_dir;
  std::string ba_snapshot_capture = "none";
  std::string ba_snapshot_registered_images = "2,6,20,50,270";
  std::string ba_ceres_oracle_dir;
  std::string ba_ceres_oracle_run_id = "original";
  int ba_ceres_oracle_repeat_count = 1;
  std::string ba_compare_dir;
  int ba_cuda_device = 0;
  CudaExecutionProfile ba_cuda_execution_profile =
      CudaExecutionProfile::COMPACT_CONTROL;
  std::string ba_cuda_audit_profile = "compatibility_default";
  std::string ba_cuda_arithmetic_precision = "compatibility_default";
  std::string ba_cuda_hessian_assembly_backend = "compatibility_default";
  std::string ba_cuda_hot_kernel_mode = "transformed";
  std::string ba_cuda_schur_contribution_backend = "direct";
  std::string ba_cuda_schur_mode = "deterministic";
  std::string ba_cuda_host_problem_store = "disabled";
  gpu_ba::CudaProblemSource ba_cuda_problem_source =
      gpu_ba::CudaProblemSource::kLegacySnapshot;
  std::string ba_lidar_residual = "legacy_exact";
  std::string ba_telemetry_path;
  uint64_t ba_refinement_index = 0;
  image_t ba_trigger_image_id = 0;

  // Lidar poiny cloud file path
  std::string lidar_pointcloud_path;
  // If use existed lidar point cloud map to assist mapping
  bool if_add_lidar_constraint = true;
  // Weight used to optimize
  // Lidar point in projection process
  double proj_lidar_constraint_weight = 1.0;
  // Lidar point in Icp process
  double icp_lidar_constraint_weight = 100.0;
  // Lidar point (belong to ground) in Icp process
  double icp_ground_lidar_constraint_weight = 1000.0;
  bool if_add_lidar_corresponding = true;
  int ba_match_features_threshold;
  // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
  enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = false;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = false;

  // Whether to refine the extrinsic parameter group.
  bool refine_extrinsics = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Minimum number of residuals to enable multi-threading. Note that
  // single-threaded is typically better for small bundle adjustment problems
  // due to the overhead of threading.
  int min_num_residuals_for_multi_threading = 50000;

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  BundleAdjustmentOptions() {
    solver_options.function_tolerance = 0.0;//Δcost/cost
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  }

  // Create a new loss function based on the specified options. The caller
  // takes ownership of the loss function.
  ceres::LossFunction* CreateLossFunction() const;

  bool Check() const;
};

BundleAdjustmentOptions::CudaExecutionProfile
ParseBundleAdjustmentCudaExecutionProfile(const std::string& value);

struct BundleAdjustmentExecutionResult {
  uint64_t call_index = 0;
  std::string ba_kind;
  uint64_t registered_images = 0;
  std::string requested_backend;
  std::string executed_backend;
  std::string problem_source_requested = "legacy_snapshot";
  std::string problem_source_effective = "legacy_snapshot";
  std::string problem_source_fallback_reason;
  std::string selected_schur;
  std::string execution_profile;
  std::string audit_profile_requested;
  std::string audit_profile_effective;
  bool capture_state_trace_effective = false;
  bool instrumentation_effective = false;
  bool production_audit_invariants_checked = false;
  bool production_audit_invariants_pass = false;
  uint64_t production_audit_violation_count = 0;
  std::string first_production_audit_violation;
  uint64_t state_hash_computations = 0;
  uint64_t topology_fingerprint_computations = 0;
  uint64_t audit_mirror_bytes = 0;
  uint64_t optional_full_array_d2h_bytes = 0;
  uint64_t mixed_double_edge_materialization_bytes = 0;
  std::string arithmetic_precision_requested;
  std::string arithmetic_precision_effective;
  std::string hessian_backend_requested;
  std::string hessian_backend_effective;
  std::string hot_kernel_requested;
  std::string hot_kernel_effective;
  std::string schur_contribution_backend_requested;
  std::string schur_contribution_backend_effective;
  std::string state_storage_precision;
  std::string residual_jacobian_precision;
  std::string hessian_schur_precision;
  std::string factorization_routine;
  std::string delta_precision;
  std::string quaternion_plus_precision;
  std::string cost_precision;
  std::string controller_precision;
  std::string summary_source;
  bool success = false;
  std::string termination;
  std::string stable_error;
  std::string diagnostic_message;
  bool fallback_used = false;
  std::string fallback_reason;
  double wall_seconds = 0.0;
  int trial_steps = 0;
  int accepted_steps = 0;
  int accepted_commits = 0;
  int rejected_steps = 0;
  int invalid_steps = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double max_backward_error = 0.0;
  uint64_t residuals = 0;
  uint64_t parameter_blocks = 0;
  uint64_t parameters = 0;
  uint64_t residual_blocks = 0;
  uint64_t effective_parameters = 0;
  bool ceres_problem_created = false;
  uint64_t ceres_cost_function_creations = 0;
  uint64_t ceres_add_residual_calls = 0;
  bool snapshot_recorder_created = false;
  uint64_t snapshot_materialization_calls = 0;
  uint64_t active_spec_build_calls = 0;
  uint64_t residual_enumerator_passes = 0;
  uint64_t residual_enumerator_items = 0;
  uint64_t peak_cuda_bytes = 0;
  uint64_t ceres_solve_calls = 0;
  std::string ceres_solve_calls_scope = "bundle_adjuster_only";
  uint64_t transaction_qvec_count = 0;
  uint64_t transaction_parameter_block_count = 0;
  uint64_t transaction_backup_entity_count = 0;
  uint64_t transaction_backup_bytes = 0;
  uint64_t transaction_identity_index_entries = 0;
  uint64_t transaction_parameter_lookup_count = 0;
  double transaction_checkpoint_milliseconds = 0.0;
  double transaction_restore_milliseconds = 0.0;
  std::string host_store_mode_requested = "disabled";
  std::string host_store_mode_effective = "disabled";
  std::string host_store_view_action = "disabled";
  std::string host_store_rebuild_reason;
  uint64_t host_store_owner_epoch = 0;
  uint64_t host_store_catalog_generation = 0;
  uint64_t host_store_view_generation = 0;
  uint64_t host_store_lookup_calls = 0;
  uint64_t host_store_hits = 0;
  uint64_t host_store_misses = 0;
  uint64_t host_store_journal_cursor_before = 0;
  uint64_t host_store_journal_cursor_after = 0;
  uint64_t host_store_catalog_cold_builds = 0;
  uint64_t host_store_catalog_delta_updates = 0;
  uint64_t host_store_catalog_full_rebuilds = 0;
  uint64_t host_store_journal_events = 0;
  uint64_t host_store_journal_gaps = 0;
  uint64_t host_store_journal_overflows = 0;
  uint64_t host_store_journal_unknown_events = 0;
  uint64_t host_store_exact_view_reuses = 0;
  uint64_t host_store_view_patches = 0;
  uint64_t host_store_view_rebuilds = 0;
  uint64_t host_store_fallback_rebuilds = 0;
  uint64_t host_store_builder_calls_executed = 0;
  uint64_t host_store_builder_calls_saved = 0;
  uint64_t host_store_builder_traversals_executed = 0;
  uint64_t host_store_builder_traversals_saved = 0;
  uint64_t host_store_estimated_builder_bytes_executed = 0;
  uint64_t host_store_estimated_builder_bytes_saved = 0;
  uint64_t host_store_static_binding_builder_calls = 0;
  uint64_t host_store_cost_layout_builder_calls = 0;
  uint64_t host_store_hessian_topology_builder_calls = 0;
  uint64_t host_store_hessian_segment_plan_builder_calls = 0;
  uint64_t host_store_schur_topology_builder_calls = 0;
  uint64_t host_store_schur_segment_plan_builder_calls = 0;
  uint64_t host_store_dynamic_state_refresh_calls = 0;
  uint64_t host_store_indexed_catalog_full_graph_build_calls = 0;
  uint64_t host_store_indexed_catalog_journal_apply_calls = 0;
  uint64_t host_store_indexed_catalog_flatten_calls = 0;
  uint64_t host_store_indexed_catalog_reconcile_calls = 0;
  uint64_t host_store_indexed_catalog_export_calls = 0;
  uint64_t host_store_indexed_active_materializer_calls = 0;
  uint64_t host_store_indexed_full_graph_records_scanned = 0;
  uint64_t host_store_indexed_estimated_impl_copy_bytes = 0;
  uint64_t host_store_indexed_estimated_export_bytes = 0;
  uint64_t host_store_descriptor_identity = 0;
  uint64_t host_store_descriptor_items = 0;
  uint64_t host_store_descriptor_hash_updates = 0;
  uint64_t host_store_device_static_h2d_saved_calls = 0;
  uint64_t host_store_device_static_h2d_saved_bytes = 0;
  uint64_t host_store_dynamic_h2d_calls = 0;
  uint64_t host_store_dynamic_h2d_bytes = 0;
  uint64_t cuda_host_build_cuda_layer_a_inputs_calls = 0;
  uint64_t cuda_host_build_static_layout_calls = 0;
  uint64_t cuda_host_build_cost_layout_calls = 0;
  uint64_t cuda_host_build_layer_b_topology_calls = 0;
  uint64_t cuda_host_build_layer_c_topology_calls = 0;
  uint64_t indexed_device_catalog_lookup_calls = 0;
  uint64_t indexed_device_catalog_reuse_calls = 0;
  uint64_t indexed_device_catalog_revision_rebinds = 0;
  uint64_t indexed_device_catalog_full_upload_calls = 0;
  uint64_t indexed_device_catalog_full_upload_bytes = 0;
  uint64_t indexed_device_catalog_patch_upload_calls = 0;
  uint64_t indexed_device_catalog_patch_upload_bytes = 0;
  uint64_t indexed_device_catalog_invalidations = 0;
  uint64_t indexed_device_catalog_prefix_bytes = 0;
  uint64_t indexed_device_catalog_arena_generation = 0;
  uint64_t native_device_store_lookup_calls = 0;
  uint64_t native_device_store_reuse_calls = 0;
  uint64_t native_device_store_full_upload_calls = 0;
  uint64_t native_device_store_full_upload_bytes = 0;
  uint64_t native_device_store_patch_upload_calls = 0;
  uint64_t native_device_store_patch_upload_bytes = 0;
  uint64_t native_device_store_growth_d2d_calls = 0;
  uint64_t native_device_store_growth_d2d_bytes = 0;
  uint64_t native_device_store_invalidations = 0;
  uint64_t native_variable_state_d2h_calls = 0;
  uint64_t native_variable_state_d2h_bytes = 0;
  uint64_t native_legacy_kernel_input_bundle_calls = 0;
  uint64_t native_repeated_residual_state_packing_bytes = 0;
  double native_indexed_packing_milliseconds = 0.0;
  double native_variable_state_download_milliseconds = 0.0;
  double native_intent_build_milliseconds = 0.0;
  uint64_t host_store_host_resident_bytes = 0;
  uint64_t host_store_host_peak_bytes = 0;
  uint64_t host_store_owner_identity_violations = 0;
  uint64_t host_store_coverage_violations = 0;
  uint64_t host_store_busy_failures = 0;
  bool host_store_catalog_valid = false;
  bool host_store_device_prepared_context_reuse = false;
  bool host_store_full_cross_solve_initialization_removed = false;
  double host_store_descriptor_milliseconds = 0.0;
  double host_store_lookup_milliseconds = 0.0;
  double host_store_catalog_delta_milliseconds = 0.0;
  double host_store_view_build_or_patch_milliseconds = 0.0;
  double host_store_dynamic_refresh_milliseconds = 0.0;
  double host_store_preparation_total_milliseconds = 0.0;
  double host_store_prepare_and_cuda_wall_seconds = 0.0;
  double problem_source_prepare_and_cuda_wall_seconds = 0.0;
  double custom_cuda_caller_wall_seconds = 0.0;
  double bundle_adjuster_setup_milliseconds = 0.0;
  double snapshot_materialization_milliseconds = 0.0;
  double active_spec_build_milliseconds = 0.0;
  double active_spec_checkpoint_milliseconds = 0.0;
  double legacy_cpu_preparation_milliseconds = 0.0;
  double fast_cpu_preparation_milliseconds = 0.0;
  uint64_t trigger_image_id = 0;
  uint64_t refinement_index = 0;
};

#ifdef GPU_BA_ENABLED
struct ActiveBaProblemSourceComparisonForTesting {
  gpu_ba::ActiveBaSolveSpec active;
  gpu_ba::Snapshot legacy;
};
#endif

// Compute the exact options passed by the original BundleAdjuster to
// ceres::Solve. Fidelity replay serializes the returned value and restores it;
// it must not call this helper to infer options.
ceres::Solver::Options CreateEffectiveBundleAdjustmentSolverOptions(
    const BundleAdjustmentOptions& options,
    size_t config_num_images,
    int problem_num_residuals);

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
 public:
  BundleAdjustmentConfig();

  size_t NumImages() const;
  size_t NumPoints() const;
  size_t NumConstantCameras() const;
  size_t NumConstantPoses() const;
  size_t NumConstantTvecs() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

  // Determine the number of residuals for the given reconstruction. The number
  // of residuals equals the number of observations times two.
  size_t NumResiduals(const Reconstruction& reconstruction) const;

  // Add / remove images from the configuration.
  void AddImage(const image_t image_id);
  void AddLidarPoint(const point3D_t& point3D_id, const class LidarPoint& lidar_point);
  // Add lidar point cloud 
  void AddPointcloud(std::shared_ptr<lidar::PointCloudProcess> ptr);

  bool HasImage(const image_t image_id) const;
  void RemoveImage(const image_t image_id);

  // Set cameras of added images as constant or variable. By default all
  // cameras of added images are variable. Note that the corresponding images
  // have to be added prior to calling these methods.
  void SetConstantCamera(const camera_t camera_id);
  void SetVariableCamera(const camera_t camera_id);
  bool IsConstantCamera(const camera_t camera_id) const;

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantPose(const image_t image_id);
  void SetVariablePose(const image_t image_id);
  bool HasConstantPose(const image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique. Note that the
  // corresponding images have to be added prior to calling these methods.
  void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
  void RemoveConstantTvec(const image_t image_id);
  bool HasConstantTvec(const image_t image_id) const;

  // Find 3d point and relative lidar point
  void Project2Image(Reconstruction* reconstruction,const point3D_t& point3D_id, const image_t& image_id, const int& match_features_threshold);
  void MatchVariablePoint2LidarPoint(Reconstruction* reconstruction,const point3D_t point3D_id);
  void MatchClosestLidarPoint(Reconstruction* reconstruction,const point3D_t& point3D_id, double& max_search_range);
  void SetLidarPoint(const point3D_t point3D_id, const std::vector<double>& lidar_pt);
  void SetLidarSearchRange(const point3D_t point3D_id, double search_range);


  // Add / remove points from the configuration. Note that points can either
  // be variable or constant but not both at the same time.
  void AddVariablePoint(const point3D_t point3D_id);
  void AddConstantPoint(const point3D_t point3D_id);
  bool HasPoint(const point3D_t point3D_id) const;
  bool HasVariablePoint(const point3D_t point3D_id) const;
  bool HasConstantPoint(const point3D_t point3D_id) const;
  void RemoveVariablePoint(const point3D_t point3D_id);
  void RemoveConstantPoint(const point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::vector<image_t>& OrderedImages() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::vector<int>& ConstantTvec(const image_t image_id) const;
  const std::unordered_map<point3D_t, double>& LidarSearchRanges() const;

  // images that have done lidar point cloud projection
  // key: image_id, value : map (key: point3d_id, value: lidar point coordinate and normal vector)
  std::map<image_t,std::map<point3D_t,Eigen::Matrix<double,6,1>>> lidar_searched_image_ids_;

  EIGEN_STL_UMAP(point3D_t, class LidarPoint) lidar_maps_;
  std::shared_ptr<lidar::PointCloudProcess> point_cloud_process_;

 private:
  std::unordered_set<camera_t> constant_camera_ids_;
  std::unordered_set<image_t> image_ids_;
  std::vector<image_t> ordered_image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<image_t> constant_poses_;
  std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
  std::unordered_map<point3D_t, double> lidar_search_ranges_;

};

#ifdef GPU_BA_CUDA_ENABLED
bool ResolveNativeBundleAdjustmentCudaConfiguration(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_solver_options,
    uint64_t config_generation,
    gpu_ba::CudaFullLmOptions* resolved_options,
    gpu_ba::NativeCudaResolvedConfig* config,
    std::string* error);

bool BuildNativeBaSolveIntent(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction,
    uint64_t owner_epoch,
    uint64_t topology_revision,
    uint64_t selection_revision,
    gpu_ba::BaKind kind,
    gpu_ba::NativeBaSolveIntent* intent,
    std::string* error);
#endif

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
 public:
  enum class OptimazePhrase{Local, Global, WholeMap};
  enum class FailureModeForTesting {
    kNone = 0,
    kCustomCudaFailure = 1,
    kCeresFailure = 2,
    kCudaTopologyMismatch = 3,
    kRestoreValidationFailure = 4,
    kActiveSpecBuildFailure = 5,
    kCustomCudaThenCeresFailure = 6,
  };
  BundleAdjuster(const BundleAdjustmentOptions& options,//
                 const BundleAdjustmentConfig& config);//
  ~BundleAdjuster();

  void SetOptimazePhrase(const OptimazePhrase& phrase);
  void SetCudaHostStoreBinding(
      const gpu_ba::CudaHostStoreBinding& binding) noexcept;

  // Final native handoff used by Mapper integration: selection and resolved
  // CUDA identity are supplied directly, so the successful path creates no
  // Ceres Problem, ActiveBaSolveSpec, CudaSolveProblem, or Snapshot.
  bool SolveNative(Reconstruction* reconstruction,
                   const gpu_ba::NativeBaSolveIntent& intent,
                   double intent_build_milliseconds = 0.0);

  bool Solve(Reconstruction* reconstruction);
#ifdef GPU_BA_ENABLED
  bool CompareProblemSourcesForTesting(
      Reconstruction* reconstruction,
      ActiveBaProblemSourceComparisonForTesting* comparison,
      std::string* error);
#endif

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;
  const BundleAdjustmentExecutionResult& ExecutionResult() const;

  static uint64_t CeresSolveCallCountForTesting();
  static void ResetCeresSolveCallCountForTesting();
  static void SetFailureModeForTesting(FailureModeForTesting mode);

 private:
  void SetUp(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);
  void SetUpLocalByLidar(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);
  void SetUpGlobalByLidar(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);
  void SetUpAdjustWholeMapByLidar(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function);
  void TearDown(Reconstruction* reconstruction);

  void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);
  // Take the image position as the center of the circle to make a ball shape, 
  // and add images inside the sphere to the optimization
  void AddImageInSphereToProblem(const image_t image_id, Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);
  
  void AddLidarToProblem(const point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);
#ifdef GPU_BA_ENABLED
  bool CaptureSnapshotIfRequested(Reconstruction* reconstruction,
                                  const ceres::Solver::Options& solver_options,
                                  bool write_snapshot,
                                  uint64_t ba_call_index,
                                  gpu_ba::Snapshot* snapshot,
                                  gpu_ba::SnapshotWriteResult* result);
#endif
 protected:
  void ParameterizeCameras(Reconstruction* reconstruction);
  void ParameterizePoints(Reconstruction* reconstruction);

  OptimazePhrase optimize_phrase_ = OptimazePhrase::Global;
  const BundleAdjustmentOptions options_;
  BundleAdjustmentConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  bool solve_called_ = false;
  ceres::Solver::Summary summary_;
  BundleAdjustmentExecutionResult execution_result_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
#ifdef GPU_BA_ENABLED
  std::unique_ptr<gpu_ba::SnapshotRecorder> snapshot_recorder_;
  std::unique_ptr<gpu_ba::ActiveBaSolveSpecBuilder> active_spec_builder_;
#endif
  uint64_t ceres_cost_function_creations_ = 0;
  uint64_t ceres_add_residual_calls_ = 0;
  gpu_ba::CudaHostStoreBinding cuda_host_store_binding_;
};

// Bundle adjustment using PBA (GPU or CPU). Less flexible and accurate than
// Ceres-Solver bundle adjustment but much faster. Only supports SimpleRadial
// camera model.
class ParallelBundleAdjuster {
 public:
  struct Options {
    // Whether to print a final summary.
    bool print_summary = true;

    // Maximum number of iterations.
    int max_num_iterations = 50;

    // Index of the GPU used for bundle adjustment.
    int gpu_index = -1;

    // Number of threads for CPU based bundle adjustment.
    int num_threads = -1;

    // Minimum number of residuals to enable multi-threading. Note that
    // single-threaded is typically better for small bundle adjustment problems
    // due to the overhead of threading.
    int min_num_residuals_for_multi_threading = 50000;

    bool Check() const;
  };

  ParallelBundleAdjuster(const Options& options,
                         const BundleAdjustmentOptions& ba_options,
                         const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

  // Check whether PBA is supported for the given reconstruction. If the
  // reconstruction is not supported, the PBA solver will exit ungracefully.
  static bool IsSupported(const BundleAdjustmentOptions& options,
                          const Reconstruction& reconstruction);

 private:
  void SetUp(Reconstruction* reconstruction);
  void TearDown(Reconstruction* reconstruction);

  void AddImagesToProblem(Reconstruction* reconstruction);
  void AddPointsToProblem(Reconstruction* reconstruction);

  const Options options_;
  const BundleAdjustmentOptions ba_options_;
  BundleAdjustmentConfig config_;
  ceres::Solver::Summary summary_;

  size_t num_measurements_;
  std::vector<pba::CameraT> cameras_;
  std::vector<pba::Point3D> points3D_;
  std::vector<pba::Point2D> measurements_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_set<point3D_t> point3D_ids_;
  std::vector<int> camera_idxs_;
  std::vector<int> point3D_idxs_;
  std::vector<image_t> ordered_image_ids_;
  std::vector<point3D_t> ordered_point3D_ids_;
  std::unordered_map<image_t, int> image_id_to_camera_idx_;
};

class RigBundleAdjuster : public BundleAdjuster {
 public:
  struct Options {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment. Some observations might have large
    // reprojection errors due to the concatenation of the absolute and relative
    // rig poses, which might be different from the absolute pose of the image
    // in the reconstruction.
    double max_reproj_error = 1000.0;
  };

  RigBundleAdjuster(const BundleAdjustmentOptions& options,
                    const Options& rig_options,
                    const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction,
             std::vector<CameraRig>* camera_rigs);

 private:
  void SetUp(Reconstruction* reconstruction,
             std::vector<CameraRig>* camera_rigs,
             ceres::LossFunction* loss_function);
  void TearDown(Reconstruction* reconstruction,
                const std::vector<CameraRig>& camera_rigs);

  void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                         std::vector<CameraRig>* camera_rigs,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void ComputeCameraRigPoses(const Reconstruction& reconstruction,
                             const std::vector<CameraRig>& camera_rigs);

  void ParameterizeCameraRigs(Reconstruction* reconstruction);

  const Options rig_options_;

  // Mapping from images to camera rigs.
  std::unordered_map<image_t, CameraRig*> image_id_to_camera_rig_;

  // Mapping from images to the absolute camera rig poses.
  std::unordered_map<image_t, Eigen::Vector4d*> image_id_to_rig_qvec_;
  std::unordered_map<image_t, Eigen::Vector3d*> image_id_to_rig_tvec_;

  // For each camera rig, the absolute camera rig poses.
  std::vector<std::vector<Eigen::Vector4d>> camera_rig_qvecs_;
  std::vector<std::vector<Eigen::Vector3d>> camera_rig_tvecs_;

  // The Quaternions added to the problem, used to set the local
  // parameterization once after setting up the problem.
  std::unordered_set<double*> parameterized_qvec_data_;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_
