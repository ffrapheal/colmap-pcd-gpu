#ifndef COLMAP_SRC_CONTROLLERS_ONLINE_I3DGS_MAPPER_H_
#define COLMAP_SRC_CONTROLLERS_ONLINE_I3DGS_MAPPER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "controllers/online_mapper_frontend.h"
#include "controllers/online_mapper_state.h"
#include "sfm/active_covisibility_graph.h"
#include "sfm/online_dual_selection.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

enum class OnlineI3dgsMapperMode {
  NONE,
  BOOTSTRAP,
  SINGLE,
  CATCHUP,
  DUAL,
  END_OF_SEQUENCE_FLUSH,
};

enum class OnlineI3dgsMapperStatus {
  RUNNING,
  READY_TO_FLUSH,
  COMPLETE,
  INCOMPLETE,
};

enum class OnlineI3dgsBaDisposition {
  SUCCESS,
  RECOVERABLE_FAILURE,
  FATAL_FAILURE,
};

enum class OnlineI3dgsBaFailureReason {
  NONE,
  TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM,
  RETRY_EVIDENCE_NOT_INCREASED,
  PREPARE_FAILED,
  SOLVE_NATIVE_FAILED,
  POSTPROCESS_FAILED,
  PROPAGATION_FAILED,
  TRANSACTION_FAILED,
  INVALID_RESULT,
};

const char* ToString(OnlineI3dgsMapperMode value);
const char* ToString(OnlineI3dgsMapperStatus value);
const char* ToString(OnlineI3dgsBaDisposition value);
const char* ToString(OnlineI3dgsBaFailureReason value);

struct OnlineI3dgsMapperOptions {
  size_t expected_frame_count = 246;
  size_t ba_window_size = 20;

  bool Check() const;
};

struct OnlineI3dgsFrameInput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  size_t frame_index = std::numeric_limits<size_t>::max();
  image_t image_id = kInvalidImageId;
  std::string image_path;
  std::string camera_path;
  std::string odometry_path;
  std::string scan_path;
  std::string image_sha256;
  std::string camera_sha256;
  std::string odometry_sha256;
  std::string scan_sha256;
  uint64_t scan_size_bytes = 0;
  KnownPoseSE3 fastlio_T_cw;
};

struct OnlineI3dgsDependencyResult {
  bool success = false;
  std::string detail;
};

struct OnlineI3dgsLidarIngestResult : OnlineI3dgsDependencyResult {
  uint64_t opened_scan_index = 0;
  uint64_t map_version_before = 0;
  uint64_t map_version_after = 0;
  uint64_t max_scan_index = 0;
  uint32_t point_transform_count_min = 0;
  uint32_t point_transform_count_max = 0;
  uint32_t normal_transform_count_min = 0;
  uint32_t normal_transform_count_max = 0;
  std::string snapshot_sha256;
  std::string geometry_sha256;
};

struct OnlineI3dgsPairMatch {
  image_t reference_image_id = kInvalidImageId;
  size_t raw_match_count = 0;
  size_t verified_inlier_count = 0;
};

struct OnlineI3dgsMatchBatchResult : OnlineI3dgsDependencyResult {
  std::vector<OnlineI3dgsPairMatch> pairs;
};

struct OnlineI3dgsKnownPoseRegistrationResult
    : OnlineI3dgsDependencyResult {
  size_t registration_sequence = std::numeric_limits<size_t>::max();
};

struct OnlineI3dgsCatchupFailureEvidence {
  uint64_t active_edge_evidence_version = 0;
  uint64_t trigger_submitted_lidar_constraint_count = 0;
  uint64_t lidar_map_version = 0;
};

struct OnlineI3dgsCatchupFailureQueryResult
    : OnlineI3dgsDependencyResult {
  bool has_evidence = false;
  OnlineI3dgsCatchupFailureEvidence evidence;
};

class OnlineI3dgsFrontendDependency {
 public:
  virtual ~OnlineI3dgsFrontendDependency() = default;

  virtual OnlineI3dgsDependencyResult IngestFrame(
      const OnlineI3dgsFrameInput& frame,
      size_t max_visible_frame_index) = 0;
  virtual OnlineI3dgsMatchBatchResult MatchExplicitReferences(
      image_t current_image_id,
      const std::vector<image_t>& ordered_reference_image_ids,
      size_t max_visible_frame_index) = 0;
};

class OnlineI3dgsLidarDependency {
 public:
  virtual ~OnlineI3dgsLidarDependency() = default;

  virtual OnlineI3dgsLidarIngestResult IngestScan(
      const OnlineI3dgsFrameInput& frame,
      size_t max_visible_frame_index) = 0;
};

struct OnlineI3dgsPnPRequest {
  image_t current_image_id = kInvalidImageId;
  size_t frame_index = 0;
  size_t max_visible_frame_index = 0;
  uint64_t canonical_graph_version = 0;
  OnlineDualClassificationResult classification;
  std::vector<OnlineI3dgsPairMatch> matched_pairs;
};

class OnlineI3dgsPnPDependency {
 public:
  virtual ~OnlineI3dgsPnPDependency() = default;

  virtual OnlineDualPnPDecisionResult ProbeTwoSides(
      const OnlineI3dgsPnPRequest& request) = 0;
};

struct OnlineI3dgsBaPassAudit {
  uint32_t pass_index = 0;
  bool solve_native_invoked = false;
  bool solve_native_success = false;
  bool finite_costs = false;
  bool finite_output = false;
  uint64_t submitted_visual_residual_count = 0;
  uint64_t submitted_lidar_constraint_count = 0;
  uint64_t trigger_submitted_lidar_constraint_count = 0;
  bool trigger_preflight_checked = false;
  uint64_t trigger_preflight_lidar_constraint_count = 0;
  size_t constant_camera_pose_count = 0;
  size_t postprocess_count = 0;
  std::string requested_backend = "custom_cuda";
  std::string executed_backend;
  std::string requested_problem_source = "native_graph";
  std::string executed_problem_source;
  bool fallback_used = false;
  std::string termination;
  std::string detail;
};

struct OnlineI3dgsImagePoseUpdate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  image_t image_id = kInvalidImageId;
  KnownPoseSE3 latest_T_cw;
};

struct OnlineI3dgsPerImageSubmittedLidarConstraintCount {
  image_t image_id = kInvalidImageId;
  uint64_t submitted_lidar_constraint_count = 0;
};

class OnlineI3dgsBaCandidate {
 public:
  virtual ~OnlineI3dgsBaCandidate() = default;
};

struct OnlineI3dgsBaExecution {
  OnlineI3dgsBaDisposition disposition =
      OnlineI3dgsBaDisposition::FATAL_FAILURE;
  OnlineI3dgsBaFailureReason failure_reason =
      OnlineI3dgsBaFailureReason::INVALID_RESULT;
  std::string detail;
  std::vector<OnlineI3dgsBaPassAudit> passes;
  std::vector<OnlineI3dgsImagePoseUpdate,
              Eigen::aligned_allocator<OnlineI3dgsImagePoseUpdate>>
      pose_updates;
  std::vector<OnlineI3dgsPerImageSubmittedLidarConstraintCount>
      per_image_submitted_lidar_constraint_counts;
  size_t total_postprocess_count = 0;
  bool propagation_applied = false;
  std::unique_ptr<OnlineI3dgsBaCandidate> candidate;
};

struct OnlineI3dgsBaRequest {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineI3dgsMapperMode mode = OnlineI3dgsMapperMode::NONE;
  uint64_t attempt_id = 0;
  size_t frame_index = 0;
  size_t max_visible_frame_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  OnlineMapperCanonicalVersion expected_canonical_version;
  uint64_t expected_registry_version = 0;
  uint64_t expected_graph_version = 0;
  uint64_t lidar_map_version = 0;
  uint64_t max_scan_index = 0;
  std::string lidar_snapshot_sha256;
  std::string lidar_geometry_sha256;
  std::vector<image_t> frozen_image_ids;
  std::vector<ActiveCovisibilityEdgeInput> ordinary_edges;
  std::vector<ActiveCovisibilityEdgeInput> loop_edges;
  bool all_camera_poses_variable = true;
  bool intrinsics_fixed = true;
  bool pose_prior_enabled = false;
  bool fallback_allowed = false;
  std::string requested_backend = "custom_cuda";
  std::string requested_problem_source = "native_graph";
  std::string solve_api = "BundleAdjuster::SolveNative";
  size_t required_pass_count = 1;
  uint64_t minimum_trigger_submitted_lidar_constraints = 50;
  bool is_flush = false;
  bool has_previous_catchup_failure = false;
  OnlineI3dgsCatchupFailureEvidence previous_catchup_failure;
  uint64_t active_edge_evidence_version = 0;
  bool has_dual_context = false;
  OnlineDualClassificationResult dual_classification;
  OnlineDualPnPDecisionResult dual_pnp;
  OnlineDualWindowResult dual_window;
};

enum class OnlineI3dgsCommitWriteKind {
  STATE_AND_ACTIVE_NODE,
  ORDINARY_EDGES,
  RECONSTRUCTION_AND_OWNERS,
  PROPAGATION,
  LOOP_EDGES,
};

struct OnlineI3dgsCommitRequest {
  OnlineI3dgsMapperMode mode = OnlineI3dgsMapperMode::NONE;
  uint64_t attempt_id = 0;
  size_t frame_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  OnlineMapperCanonicalVersion expected_canonical_version;
  uint64_t expected_registry_version = 0;
  uint64_t expected_graph_version = 0;
  std::vector<image_t> promoted_image_ids;
  std::vector<OnlineI3dgsImagePoseUpdate,
              Eigen::aligned_allocator<OnlineI3dgsImagePoseUpdate>>
      pose_updates;
  std::vector<ActiveCovisibilityEdgeInput> ordinary_edges;
  std::vector<ActiveCovisibilityEdgeInput> loop_edges;
  std::vector<OnlineI3dgsCommitWriteKind> logical_write_order;
};

struct OnlineI3dgsCommitResult : OnlineI3dgsDependencyResult {
  bool atomic_publication = false;
  OnlineMapperCanonicalVersion canonical_version_before;
  OnlineMapperCanonicalVersion canonical_version_after;
  uint64_t registry_version_before = 0;
  uint64_t registry_version_after = 0;
  uint64_t graph_version_before = 0;
  uint64_t graph_version_after = 0;
};

class OnlineI3dgsLocalBaDependency {
 public:
  virtual ~OnlineI3dgsLocalBaDependency() = default;

  // The implementation owns one detached Reconstruction transaction candidate,
  // materializes the frozen-window intent, and invokes
  // BundleAdjuster::SolveNative exactly required_pass_count times. DUAL reuses
  // that same candidate for both calls. Execute never publishes canonical state.
  virtual OnlineI3dgsBaExecution Execute(
      const OnlineI3dgsBaRequest& request) = 0;
  virtual OnlineI3dgsCommitResult Commit(
      const OnlineI3dgsCommitRequest& request,
      OnlineI3dgsBaExecution* execution) = 0;
};

class OnlineI3dgsStateDependency {
 public:
  virtual ~OnlineI3dgsStateDependency() = default;

  virtual const KnownPoseRegistry& KnownPoses() const = 0;
  virtual const ActiveCovisibilityGraph& ActiveGraph() const = 0;
  virtual OnlineMapperCanonicalVersion CanonicalVersion() const = 0;
  virtual OnlineI3dgsKnownPoseRegistrationResult RegisterKnownPose(
      const OnlineI3dgsFrameInput& frame) = 0;
  virtual OnlineI3dgsCatchupFailureQueryResult GetCatchupFailure(
      image_t image_id) const = 0;
  virtual OnlineI3dgsDependencyResult RecordCatchupFailure(
      image_t image_id,
      const OnlineI3dgsCatchupFailureEvidence& evidence) = 0;
  virtual uint64_t GlobalBaCallCount() const = 0;
};

struct OnlineI3dgsMapperDependencies {
  OnlineI3dgsFrontendDependency* frontend = nullptr;
  OnlineI3dgsLidarDependency* lidar = nullptr;
  OnlineI3dgsPnPDependency* pnp = nullptr;
  OnlineI3dgsLocalBaDependency* local_ba = nullptr;
  OnlineI3dgsStateDependency* state = nullptr;

  bool Check() const;
};

struct OnlineI3dgsCatchupAudit {
  image_t image_id = kInvalidImageId;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  OnlineMapperCanonicalVersion canonical_version_before;
  OnlineMapperCanonicalVersion canonical_version_after;
  uint64_t registry_version_before = 0;
  uint64_t registry_version_after = 0;
  uint64_t graph_version_before = 0;
  uint64_t graph_version_after = 0;
  uint64_t active_edge_evidence_version = 0;
  bool had_previous_failure = false;
  bool attempted = false;
  bool promoted = false;
  bool retry_evidence_sufficient = false;
  OnlineI3dgsBaFailureReason failure_reason =
      OnlineI3dgsBaFailureReason::NONE;
  uint64_t trigger_submitted_lidar_constraint_count = 0;
  std::vector<image_t> frozen_image_ids;
  std::string detail;
};

struct OnlineI3dgsFrameAudit {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  size_t frame_index = 0;
  image_t image_id = kInvalidImageId;
  size_t max_visible_frame_index = 0;
  uint64_t lidar_map_version = 0;
  uint64_t max_scan_index = 0;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  OnlineMapperCanonicalVersion event_start_canonical_version;
  OnlineMapperCanonicalVersion current_recomputed_canonical_version;
  OnlineMapperCanonicalVersion canonical_version_after;
  uint64_t event_start_registry_version = 0;
  uint64_t event_start_graph_version = 0;
  uint64_t current_recomputed_registry_version = 0;
  uint64_t current_recomputed_graph_version = 0;
  std::vector<ConfirmedNormalizedSE3Reference> candidates;
  std::vector<image_t> first_batch_reference_image_ids;
  std::vector<image_t> second_batch_reference_image_ids;
  std::vector<OnlineI3dgsPairMatch> matched_pairs;
  std::vector<image_t> initial_effective_active_reference_image_ids;
  std::vector<image_t> recomputed_effective_active_reference_image_ids;
  bool matching_extended_to_ten = false;
  bool maximum_finite_hop_defined = false;
  size_t maximum_finite_hop = 0;
  bool current_recomputed_after_catchup = false;
  std::vector<OnlineI3dgsCatchupAudit> catchup;
  OnlineI3dgsMapperMode mode = OnlineI3dgsMapperMode::NONE;
  bool dual_probe_attempted = false;
  bool dual_fell_back_to_single = false;
  OnlineDualClassificationResult dual_classification;
  OnlineDualPnPDecisionResult dual_pnp;
  std::vector<image_t> frozen_image_ids;
  std::vector<OnlineI3dgsBaPassAudit> ba_passes;
  bool ba_succeeded = false;
  bool commit_attempted = false;
  bool commit_succeeded = false;
  bool current_visual_active = false;
  uint64_t registry_version_after = 0;
  uint64_t graph_version_after = 0;
  uint64_t global_ba_call_count = 0;
  std::map<std::string, double> stage_milliseconds;
  std::string termination;
  std::string detail;
};

struct OnlineI3dgsFrameResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineI3dgsMapperStatus status = OnlineI3dgsMapperStatus::INCOMPLETE;
  OnlineI3dgsFrameAudit audit;

  bool IsSuccess() const {
    return status == OnlineI3dgsMapperStatus::RUNNING ||
           status == OnlineI3dgsMapperStatus::READY_TO_FLUSH;
  }
};

struct OnlineI3dgsFlushAttemptAudit {
  size_t round = 0;
  OnlineI3dgsCatchupAudit catchup;
};

struct OnlineI3dgsRunSummary {
  OnlineI3dgsMapperStatus status = OnlineI3dgsMapperStatus::RUNNING;
  size_t arrived_event_count = 0;
  size_t known_pose_registered_count = 0;
  size_t visual_active_online_count = 0;
  size_t visual_active_after_flush_count = 0;
  size_t pose_only_final_count = 0;
  size_t bootstrap_call_count = 0;
  size_t single_call_count = 0;
  size_t catchup_call_count = 0;
  size_t dual_call_count = 0;
  size_t ba_pass_count = 0;
  size_t catchup_commit_count = 0;
  size_t dual_atomic_commit_count = 0;
  size_t partial_commit_detection_count = 0;
  size_t pnp_probe_call_count = 0;
  size_t pnp_registration_call_count = 0;
  uint64_t global_ba_call_count = 0;
  std::string last_consistent_stage;
  std::string incomplete_reason;
};

struct OnlineI3dgsFinishResult {
  OnlineI3dgsMapperStatus status = OnlineI3dgsMapperStatus::INCOMPLETE;
  std::vector<OnlineI3dgsFlushAttemptAudit> flush_attempts;
  OnlineI3dgsRunSummary summary;

  bool IsComplete() const {
    return status == OnlineI3dgsMapperStatus::COMPLETE;
  }
};

class OnlineI3dgsMapper {
 public:
  OnlineI3dgsMapper(const OnlineI3dgsMapperOptions& options,
                    OnlineI3dgsMapperDependencies dependencies);

  OnlineI3dgsFrameResult ProcessArrivedFrame(
      const OnlineI3dgsFrameInput& frame);
  OnlineI3dgsFinishResult Finish();

  OnlineI3dgsMapperStatus Status() const noexcept;
  const OnlineI3dgsRunSummary& Summary() const noexcept;

 private:
  struct PendingEdge {
    ActiveCovisibilityEdgeInput edge;
    uint64_t evidence_version = 0;
    uint64_t direct_active_evidence_version = 0;
  };

  using EdgeKey = std::pair<image_t, image_t>;

  OnlineI3dgsFrameResult FailFrame(OnlineI3dgsFrameAudit audit,
                                   const std::string& stage,
                                   const std::string& detail);
  void FailRun(const std::string& stage, const std::string& detail);
  bool CheckGlobalBaCount(std::string* error);
  bool ValidateFrameInput(const OnlineI3dgsFrameInput& frame,
                          std::string* error) const;
  bool MatchBatch(const OnlineI3dgsFrameInput& frame,
                  const std::vector<image_t>& references,
                  std::vector<OnlineI3dgsPairMatch>* matches,
                  std::string* error);
  std::vector<image_t> EffectiveActiveReferences(
      const std::vector<OnlineI3dgsPairMatch>& matches) const;
  bool ComputeMaximumFiniteHop(const std::vector<image_t>& image_ids,
                               bool* defined,
                               size_t* maximum_hop,
                               std::string* error) const;
  bool UpdatePendingEdges(image_t current_image_id,
                          size_t frame_index,
                          const std::vector<OnlineI3dgsPairMatch>& matches,
                          std::vector<ActiveCovisibilityEdgeInput>* accepted,
                          std::string* error);
  std::vector<PendingEdge> DirectPendingEdgesToActive(
      image_t image_id);
  std::vector<image_t> BuildBootstrapWindow(image_t trigger_image_id) const;
  bool BuildActiveWindow(image_t trigger_image_id,
                         const std::vector<PendingEdge>& connecting_edges,
                         std::vector<image_t>* window,
                         std::string* error) const;
  OnlineI3dgsCatchupAudit RunCatchup(image_t image_id,
                                     size_t frame_index,
                                     bool is_flush,
                                     bool* fatal);
  bool RunCatchups(size_t frame_index,
                   image_t excluded_current_image_id,
                   bool is_flush,
                   std::vector<OnlineI3dgsCatchupAudit>* audits,
                   size_t* promotion_count,
                   std::string* error);
  bool RunCurrent(const OnlineI3dgsFrameInput& frame,
                  const std::vector<OnlineI3dgsPairMatch>& matches,
                  const std::vector<ActiveCovisibilityEdgeInput>&
                      accepted_current_edges,
                  OnlineI3dgsFrameAudit* audit,
                  std::string* error);
  bool RunBootstrap(const OnlineI3dgsFrameInput& frame,
                    OnlineI3dgsFrameAudit* audit,
                    std::string* error);
  bool RunSingle(const OnlineI3dgsFrameInput& frame,
                 const std::vector<ActiveCovisibilityEdgeInput>&
                     accepted_current_edges,
                 OnlineI3dgsFrameAudit* audit,
                 std::string* error);
  bool RunDual(const OnlineI3dgsFrameInput& frame,
               const std::vector<OnlineI3dgsPairMatch>& matches,
               const std::vector<ActiveCovisibilityEdgeInput>&
                   accepted_current_edges,
               OnlineI3dgsFrameAudit* audit,
               bool* fallback_to_single,
               std::string* error);
  bool ExecuteAndCommit(const OnlineI3dgsBaRequest& request,
                        const std::vector<image_t>& promoted_image_ids,
                        const std::vector<ActiveCovisibilityEdgeInput>&
                            ordinary_edges,
                        const std::vector<ActiveCovisibilityEdgeInput>&
                            loop_edges,
                        bool bootstrap,
                        OnlineI3dgsBaExecution* execution,
                        OnlineI3dgsCommitResult* commit,
                        std::string* error);
  bool ValidateSuccessfulBa(const OnlineI3dgsBaRequest& request,
                            const OnlineI3dgsBaExecution& execution,
                            std::string* error) const;
  std::vector<image_t> BootstrapPromotedSet(
      image_t trigger_image_id,
      const std::vector<image_t>& window,
      const OnlineI3dgsBaExecution& execution) const;
  std::vector<ActiveCovisibilityEdgeInput> EdgesWithin(
      const std::vector<image_t>& image_ids) const;
  OnlineI3dgsCommitRequest BuildCommitRequest(
      const OnlineI3dgsBaRequest& request,
      const std::vector<image_t>& promoted_image_ids,
      const std::vector<ActiveCovisibilityEdgeInput>& ordinary_edges,
      const std::vector<ActiveCovisibilityEdgeInput>& loop_edges,
      const OnlineI3dgsBaExecution& execution,
      bool bootstrap) const;
  static EdgeKey MakeEdgeKey(image_t image_id1, image_t image_id2);

  const OnlineI3dgsMapperOptions options_;
  const OnlineI3dgsMapperDependencies dependencies_;
  const std::thread::id control_thread_id_;
  OnlineI3dgsMapperStatus status_ = OnlineI3dgsMapperStatus::RUNNING;
  OnlineI3dgsRunSummary summary_;
  size_t next_frame_index_ = 1;
  uint64_t last_lidar_map_version_ = 0;
  std::string last_lidar_snapshot_sha256_;
  std::string last_lidar_geometry_sha256_;
  uint64_t next_ba_attempt_id_ = 1;
  uint64_t next_edge_evidence_version_ = 1;
  std::map<EdgeKey, PendingEdge> pending_edges_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_ONLINE_I3DGS_MAPPER_H_
