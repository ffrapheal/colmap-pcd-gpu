#ifndef COLMAP_SRC_SFM_ONLINE_LOCAL_BA_EXECUTOR_H_
#define COLMAP_SRC_SFM_ONLINE_LOCAL_BA_EXECUTOR_H_

#ifdef GPU_BA_CUDA_ENABLED

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "sfm/online_lidar_ba_intent.h"

namespace colmap {

constexpr size_t kOnlineLocalBaMaximumImageCount = 20;
constexpr uint64_t kOnlineLocalBaMinimumTriggerLidarResidualCount = 50;

enum class OnlineLocalBaMode : uint8_t {
  BOOTSTRAP = 0,
  SINGLE = 1,
  CATCHUP = 2,
  DUAL_PASS1 = 3,
  DUAL_PASS2 = 4,
};

enum class OnlineLocalBaFailureReason : uint8_t {
  NONE = 0,
  INVALID_REQUEST = 1,
  CANDIDATE_UNAVAILABLE = 2,
  FIXED_INTRINSICS_MISMATCH = 3,
  INTENT_BUILD_FAILED = 4,
  INTENT_AUDIT_FAILED = 5,
  TRIGGER_BELOW_MINIMUM = 6,
  NATIVE_RUNNER_FAILED = 7,
  SOLVE_FAILED = 8,
  BACKEND_CONTRACT_FAILED = 9,
  TERMINATION_UNUSABLE = 10,
  NONFINITE_SOLVER_RESULT = 11,
  RESIDUAL_AUDIT_MISMATCH = 12,
  CANDIDATE_STATE_INVALID = 13,
  POSTPROCESS_ADAPTER_MISSING = 14,
  POSTPROCESS_MERGE_FAILED = 15,
  POSTPROCESS_COMPLETE_FAILED = 16,
  POSTPROCESS_FILTER_FAILED = 17,
  DUAL_REQUEST_MISMATCH = 18,
  DUAL_PASS1_FAILED = 19,
  DUAL_PASS2_FAILED = 20,
  INTERNAL_ERROR = 21,
  RETRY_EVIDENCE_NOT_INCREASED = 22,
};

const char* ToString(OnlineLocalBaMode mode) noexcept;
const char* ToString(OnlineLocalBaFailureReason reason) noexcept;

bool CountOnlineLocalBaPotentialTriggerObservations(
    const Reconstruction& reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    image_t trigger_image_id,
    uint64_t* upper_bound,
    std::string* error) noexcept;

struct OnlineLocalBaFixedIntrinsicsSnapshot {
  camera_t camera_id = kInvalidCameraId;
  int model_id = -1;
  size_t width = 0;
  size_t height = 0;
  bool has_prior_focal_length = false;
  std::vector<double> parameters;

  bool operator==(const OnlineLocalBaFixedIntrinsicsSnapshot& other) const;
  bool operator!=(const OnlineLocalBaFixedIntrinsicsSnapshot& other) const;
};

struct OnlineLocalBaRequest {
  OnlineLocalBaMode mode = OnlineLocalBaMode::SINGLE;
  uint64_t attempt_id = 0;
  uint32_t pass_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  std::vector<image_t> ordered_frozen_image_ids;
  std::vector<point3D_t> point3D_ids;

  ReconstructionCanonicalVersion expected_candidate_version;
  uint64_t owner_epoch = 0;
  uint64_t topology_revision = 0;
  uint64_t selection_revision = 0;

  uint64_t expected_map_version = 0;
  uint64_t expected_max_scan_index = 0;
  std::string expected_snapshot_sha256;
  std::string expected_geometry_sha256;
  std::shared_ptr<const lidar::LidarMapSnapshot> map_snapshot;
  OnlineLidarAssociationOptions association_options;
  lidar::PcdProjectionOptions projection_options;

  std::vector<OnlineLocalBaFixedIntrinsicsSnapshot> fixed_intrinsics;
  bool all_camera_poses_variable = true;
  std::vector<image_t> constant_camera_pose_ids;
  std::string requested_backend = "custom_cuda";
  gpu_ba::CudaProblemSource requested_problem_source =
      gpu_ba::CudaProblemSource::kNativeGraph;
  bool fallback_allowed = false;
  bool allow_postprocess = true;
  bool require_trigger_residual_increase = false;
  uint64_t previous_trigger_lidar_constraint_count = 0;
};

struct OnlineLocalBaIntentBuildResult {
  OnlineLidarAssociationAudit association_audit;
  gpu_ba::NativeBaSolveIntent intent;
  OnlineLidarAssociationTriggerGate trigger_gate;
};

class OnlineLocalBaIntentBuilder {
 public:
  virtual ~OnlineLocalBaIntentBuilder() = default;

  virtual bool Build(const Reconstruction& candidate,
                     const BundleAdjustmentOptions& options,
                     const OnlineLocalBaRequest& request,
                     OnlineLocalBaIntentBuildResult* result,
                     std::string* error) = 0;
};

struct OnlineLocalBaIntentSummary {
  uint64_t owner_epoch = 0;
  uintptr_t reconstruction_identity = 0;
  uint64_t topology_revision = 0;
  uint64_t selection_revision = 0;
  uint64_t config_generation = 0;
  uint64_t lidar_map_generation = 0;
  uint64_t lidar_match_config_generation = 0;
  std::vector<image_t> active_image_ids;
  size_t fixed_pose_count = 0;
  size_t translation_policy_count = 0;
  size_t fixed_camera_count = 0;
  size_t variable_point_count = 0;
  size_t constant_point_count = 0;
  size_t lidar_constraint_count = 0;
  bool local_kind = false;
  bool active_images_only = false;
  bool all_poses_variable = false;
  bool all_intrinsics_fixed = false;
  bool config_resolved = false;
};

struct OnlineLocalBaResidualEvidence {
  // The default SolveNative runner has no public per-image materialization or
  // evaluation evidence and therefore leaves both availability flags false.
  bool materialized_counts_available = false;
  bool evaluated_counts_available = false;
  std::map<image_t, uint64_t> per_image_materialized_lidar_count;
  std::map<image_t, uint64_t> per_image_evaluated_lidar_count;
  uint64_t materialized_visual_residual_count = 0;
  uint64_t materialized_lidar_residual_count = 0;
  uint64_t evaluated_visual_residual_count = 0;
  uint64_t evaluated_lidar_residual_count = 0;
  uint64_t trigger_materialized_lidar_count = 0;
  uint64_t trigger_evaluated_lidar_count = 0;
};

struct OnlineLocalBaNativeRunOutput {
  bool solver_invoked = false;
  bool solve_returned = false;
  BundleAdjustmentExecutionResult execution;
  ceres::Solver::Summary summary;
  OnlineLocalBaResidualEvidence residual_evidence;
};

class OnlineLocalBaNativeRunner {
 public:
  virtual ~OnlineLocalBaNativeRunner() = default;

  virtual bool Run(const BundleAdjustmentOptions& options,
                   Reconstruction* candidate,
                   const OnlineLocalBaIntentBuildResult& build_result,
                   double intent_build_milliseconds,
                   OnlineLocalBaNativeRunOutput* output,
                   std::string* error) = 0;
};

class OnlineLocalBaPostprocessAdapter {
 public:
  virtual ~OnlineLocalBaPostprocessAdapter() = default;

  virtual bool Merge(Reconstruction* candidate,
                     const OnlineLocalBaRequest& request,
                     uint64_t* merged_observation_count,
                     std::string* error) = 0;
  virtual bool Complete(Reconstruction* candidate,
                        const OnlineLocalBaRequest& request,
                        uint64_t* completed_observation_count,
                        std::string* error) = 0;
  virtual bool Filter(Reconstruction* candidate,
                      const OnlineLocalBaRequest& request,
                      uint64_t* filtered_observation_count,
                      std::string* error) = 0;
};

struct OnlineLocalBaPostprocessResult {
  uint32_t merge_call_count = 0;
  uint32_t complete_call_count = 0;
  uint32_t filter_call_count = 0;
  uint64_t merged_observation_count = 0;
  uint64_t completed_observation_count = 0;
  uint64_t filtered_observation_count = 0;
};

struct OnlineLocalBaTiming {
  double request_validation_milliseconds = 0.0;
  double intent_build_milliseconds = 0.0;
  double native_run_milliseconds = 0.0;
  double merge_milliseconds = 0.0;
  double complete_milliseconds = 0.0;
  double filter_milliseconds = 0.0;
  double visual_state_hash_milliseconds = 0.0;
  double transaction_discard_milliseconds = 0.0;
  double total_milliseconds = 0.0;
};

struct OnlineLocalBaPassResult {
  OnlineLocalBaMode mode = OnlineLocalBaMode::SINGLE;
  uint64_t attempt_id = 0;
  uint32_t pass_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  bool success = false;
  bool fatal = false;
  bool intent_builder_called = false;
  bool solver_called = false;
  bool candidate_discarded = false;
  bool candidate_ready_for_commit = false;
  bool backend_contract_passed = false;
  bool termination_converged = false;
  bool finite_costs = false;
  bool submitted_constraint_audit_passed = false;
  bool solve_acceptance_passed = false;

  OnlineLidarAssociationAudit association_audit;
  OnlineLidarAssociationTriggerGate trigger_gate;
  OnlineLocalBaIntentSummary intent_summary;
  // Submitted counts describe constraints written into NativeBaSolveIntent.
  // They are the pre-solve gate and must not be reported as evaluated counts.
  std::map<image_t, uint64_t> per_image_submitted_lidar_constraint_count;
  uint64_t trigger_submitted_lidar_constraint_count = 0;
  uint64_t submitted_lidar_constraint_count = 0;
  OnlineLocalBaResidualEvidence residual_evidence;
  BundleAdjustmentExecutionResult execution;
  ceres::Solver::Summary solver_summary;
  OnlineLocalBaPostprocessResult postprocess;

  ReconstructionCanonicalVersion candidate_version_before;
  ReconstructionCanonicalVersion candidate_version_after;
  std::string candidate_visual_state_sha256_before;
  std::string candidate_visual_state_sha256_after;
  OnlineLocalBaTiming timing;
  OnlineLocalBaFailureReason failure_reason =
      OnlineLocalBaFailureReason::NONE;
  std::string failure_detail;
};

struct OnlineDualLocalBaResult {
  bool success = false;
  bool fatal = false;
  bool candidate_discarded = false;
  bool candidate_ready_for_commit = false;
  OnlineLocalBaFailureReason failure_reason =
      OnlineLocalBaFailureReason::NONE;
  std::string failure_detail;
  OnlineLocalBaPassResult pass1;
  OnlineLocalBaPassResult pass2;
  ReconstructionCanonicalVersion candidate_version_before;
  ReconstructionCanonicalVersion candidate_version_after;
  std::string candidate_visual_state_sha256_before;
  std::string candidate_visual_state_sha256_after;
};

struct OnlineLocalBaExecutorDependencies {
  OnlineLocalBaIntentBuilder* intent_builder = nullptr;
  OnlineLocalBaNativeRunner* native_runner = nullptr;
  OnlineLocalBaPostprocessAdapter* postprocess = nullptr;
  bool audit_visual_state_sha256 = false;
};

bool CaptureOnlineLocalBaFixedIntrinsics(
    const Reconstruction& reconstruction,
    const std::vector<image_t>& ordered_frozen_image_ids,
    std::vector<OnlineLocalBaFixedIntrinsicsSnapshot>* fixed_intrinsics,
    std::string* error) noexcept;

// Success leaves the caller-owned transaction candidate ready for the
// controller's atomic commit. Any failure resets the whole transaction.
OnlineLocalBaPassResult ExecuteOnlineLocalBa(
    const BundleAdjustmentOptions& options,
    const OnlineLocalBaRequest& request,
    ReconstructionTransaction* transaction,
    const OnlineLocalBaExecutorDependencies& dependencies);

OnlineDualLocalBaResult ExecuteOnlineDualLocalBa(
    const BundleAdjustmentOptions& options,
    const OnlineLocalBaRequest& pass1_request,
    const OnlineLocalBaRequest& pass2_request,
    ReconstructionTransaction* transaction,
    const OnlineLocalBaExecutorDependencies& dependencies);

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED

#endif  // COLMAP_SRC_SFM_ONLINE_LOCAL_BA_EXECUTOR_H_
