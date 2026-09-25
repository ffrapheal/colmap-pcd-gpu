#ifndef COLMAP_SRC_SFM_ONLINE_DUAL_SELECTION_H_
#define COLMAP_SRC_SFM_ONLINE_DUAL_SELECTION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "sfm/active_covisibility_graph.h"
#include "util/types.h"

namespace colmap {

enum class OnlineDualResultClass {
  SUCCESS,
  SINGLE_FALLBACK,
  INPUT_ERROR,
  DUAL_FATAL,
};

enum class OnlineDualDecision {
  SINGLE,
  DUAL_CANDIDATE,
  DUAL_READY,
};

enum class OnlineDualReason {
  NONE,
  FEWER_THAN_TWO_REFERENCES,
  MAX_HOP_NOT_GREATER_THAN_TEN,
  INVALID_REFERENCE,
  REFERENCE_NOT_MATCHED_THIS_EVENT,
  REFERENCE_NOT_POSITIVE_VERIFIED,
  REFERENCE_NOT_VISUAL_ACTIVE,
  DUPLICATE_REFERENCE,
  DUPLICATE_REGISTRATION_SEQUENCE,
  REFERENCE_NOT_CANONICAL,
  REGISTRATION_SEQUENCE_MISMATCH,
  GRAPH_QUERY_FAILED,
  DISCONNECTED_ACTIVE_GRAPH,
  SUPPORT_SUM_OVERFLOW,
  INSUFFICIENT_ROBUST_PAIRS,
  WEAK_SUPPORT_NOT_STRICTLY_GREATER_THAN_HALF,
  CLASSIFICATION_NOT_DUAL_CANDIDATE,
  INVALID_PNP_INPUT,
  PNP_ADAPTER_NOT_CONFIGURED,
  PNP_SOLVER_FAILED,
  PNP_MALFORMED_INLIER_MASK,
  PNP_INLIERS_BELOW_MINIMUM,
  PNP_INLIER_RATIO_BELOW_MINIMUM,
  PNP_INVALID_SE3,
  PNP_DELTA_INVALID_SE3,
  INVALID_WINDOW_INPUT,
  STALE_FROZEN_GRAPH,
  WINDOW_EXPANSION_FAILED,
  BACKBONE_UNAVAILABLE,
  BACKBONE_QUERY_FAILED,
  EMPTY_NEWER_WINDOW,
  MISSING_WINDOW_DELTA,
  DUPLICATE_WINDOW_DELTA,
  EXTRA_WINDOW_DELTA,
  INVALID_WINDOW_DELTA,
  INVALID_BACKBONE_COST,
  OWNERSHIP_FAILED,
  OWNERSHIP_MISSING,
  DUPLICATE_POSE_ONLY_IMAGE,
  POSE_ONLY_IMAGE_IN_ACTIVE_GRAPH,
  INVALID_LINEAGE,
  DUPLICATE_POINT,
  INVALID_POINT_INPUT,
  MISSING_POINT_OWNER,
  MISSING_OWNER_CORRECTION,
};

enum class OnlineDualClusterId {
  NONE,
  FIRST,
  SECOND,
};

enum class OnlineDualVisualSide {
  NONE,
  MAIN,
  LOOP,
};

enum class OnlineDualTemporalSide {
  NONE,
  OLDER,
  NEWER,
};

const char* ToString(OnlineDualResultClass value);
const char* ToString(OnlineDualDecision value);
const char* ToString(OnlineDualReason value);
const char* ToString(OnlineDualClusterId value);
const char* ToString(OnlineDualVisualSide value);
const char* ToString(OnlineDualTemporalSide value);

struct OnlineDualResolvedPolicy {
  size_t dual_hop_threshold = 10;
  size_t robust_pair_minimum_inliers = 15;
  size_t minimum_robust_pairs_per_side = 3;
  double weak_to_strong_strict_ratio = 0.5;
  size_t minimum_pnp_inliers_per_side = 30;
  double minimum_pnp_inlier_ratio = 0.25;
  size_t maximum_window_size = 20;
  size_t maximum_window_size_per_side = 10;
};

OnlineDualResolvedPolicy GetOnlineDualResolvedPolicy();

struct OnlineDualSE3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineDualSE3();
  OnlineDualSE3(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec);

  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
};

bool IsValidOnlineDualSE3(const OnlineDualSE3& transform);
bool IsIdentityOnlineDualSE3(const OnlineDualSE3& transform);
OnlineDualSE3 InverseOnlineDualSE3(const OnlineDualSE3& transform);
OnlineDualSE3 ComposeOnlineDualSE3(const OnlineDualSE3& lhs,
                                   const OnlineDualSE3& rhs);
OnlineDualSE3 ComputeOnlineDualDelta(const OnlineDualSE3& before_T_wc,
                                     const OnlineDualSE3& after_T_wc);
Eigen::Vector3d ApplyOnlineDualSE3(const OnlineDualSE3& transform,
                                  const Eigen::Vector3d& point);

struct OnlineDualReference {
  OnlineDualReference();
  OnlineDualReference(image_t image_id,
                      size_t registration_sequence,
                      size_t verified_inliers,
                      ActiveCovisibilityNodeState visual_state,
                      bool matched_this_event,
                      bool positive_verified);

  image_t image_id = kInvalidImageId;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  size_t verified_inliers = 0;
  ActiveCovisibilityNodeState visual_state =
      ActiveCovisibilityNodeState::POSE_ONLY;
  bool matched_this_event = false;
  bool positive_verified = false;
};

struct OnlineDualReferenceAssignment {
  OnlineDualReference reference;
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  size_t hops_to_first_seed = std::numeric_limits<size_t>::max();
  size_t hops_to_second_seed = std::numeric_limits<size_t>::max();
  bool assigned_by_seed_tie_break = false;
};

struct OnlineDualClusterSummary {
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  image_t seed_image_id = kInvalidImageId;
  std::vector<image_t> reference_image_ids;
  size_t minimum_registration_sequence = std::numeric_limits<size_t>::max();
  image_t minimum_image_id = kInvalidImageId;
  size_t robust_pair_count = 0;
  size_t best_verified_inliers = 0;
  uint64_t total_verified_inliers = 0;
  OnlineDualVisualSide visual_side = OnlineDualVisualSide::NONE;
  OnlineDualTemporalSide temporal_side = OnlineDualTemporalSide::NONE;
};

struct OnlineDualClassificationResult {
  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualDecision decision = OnlineDualDecision::SINGLE;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  uint64_t canonical_graph_version = 0;
  bool max_hop_defined = false;
  size_t maximum_hops = 0;
  size_t farthest_pair_tie_count = 0;
  image_t first_seed_image_id = kInvalidImageId;
  image_t second_seed_image_id = kInvalidImageId;
  OnlineDualClusterId main_cluster_id = OnlineDualClusterId::NONE;
  OnlineDualClusterId loop_cluster_id = OnlineDualClusterId::NONE;
  OnlineDualClusterId older_cluster_id = OnlineDualClusterId::NONE;
  OnlineDualClusterId newer_cluster_id = OnlineDualClusterId::NONE;
  bool main_selected_by_tie_break = false;
  std::vector<OnlineDualReference> ordered_references;
  std::vector<OnlineDualReferenceAssignment> assignments;
  std::vector<OnlineDualClusterSummary> clusters;

  bool IsDualCandidate() const {
    return result_class == OnlineDualResultClass::SUCCESS &&
           decision == OnlineDualDecision::DUAL_CANDIDATE;
  }
};

OnlineDualClassificationResult ClassifyOnlineDualReferences(
    const ActiveCovisibilityGraph& frozen_graph,
    const std::vector<OnlineDualReference>& references);

struct OnlineDualPnPCorrespondenceSet {
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
};

struct OnlineDualPnPProbeRequest {
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  std::vector<image_t> reference_image_ids;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
};

struct OnlineDualPnPSolverOutput {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool success = false;
  OnlineDualSE3 probe_T_cw;
  std::vector<uint8_t> inlier_mask;
  std::string detail;
};

class OnlineDualPnPProbeAdapter {
 public:
  using Solver =
      std::function<OnlineDualPnPSolverOutput(const OnlineDualPnPProbeRequest&)>;

  OnlineDualPnPProbeAdapter();
  explicit OnlineDualPnPProbeAdapter(Solver solver);

  bool IsConfigured() const;
  // The callback contract is estimate-only: it receives copied probe inputs
  // and must not register an image or mutate Reconstruction/registry state.
  OnlineDualPnPSolverOutput Probe(
      const OnlineDualPnPProbeRequest& request) const;

 private:
  Solver solver_;
};

enum class OnlineDualPnPProbeStatus {
  NOT_RUN,
  SUCCESS,
  INVALID_INPUT,
  ADAPTER_NOT_CONFIGURED,
  SOLVER_FAILED,
  MALFORMED_INLIER_MASK,
  INLIERS_BELOW_MINIMUM,
  INLIER_RATIO_BELOW_MINIMUM,
  INVALID_SE3,
};

const char* ToString(OnlineDualPnPProbeStatus value);

struct OnlineDualPnPProbeAudit {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  OnlineDualPnPProbeStatus status = OnlineDualPnPProbeStatus::NOT_RUN;
  std::vector<image_t> reference_image_ids;
  size_t correspondence_count = 0;
  size_t inlier_count = 0;
  double inlier_ratio = 0.0;
  bool pose_is_valid_se3 = false;
  OnlineDualSE3 probe_T_cw;
  std::string detail;
};

using OnlineDualPnPProbeAuditVector =
    std::vector<OnlineDualPnPProbeAudit,
                EIGEN_ALIGNED_ALLOCATOR<OnlineDualPnPProbeAudit>>;

struct OnlineDualPnPDecisionResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualDecision decision = OnlineDualDecision::SINGLE;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  OnlineDualClusterId failed_cluster_id = OnlineDualClusterId::NONE;
  OnlineDualSE3 delta_init;
  double delta_translation_norm = 0.0;
  double delta_rotation_radians = 0.0;
  OnlineDualPnPProbeAuditVector probes;

  bool IsDualCandidate() const {
    return result_class == OnlineDualResultClass::SUCCESS &&
           decision == OnlineDualDecision::DUAL_CANDIDATE;
  }
};

// DUAL-only gate. A non-candidate classification is rejected before the
// adapter is invoked, so ordinary known-pose registration has no probe path.
OnlineDualPnPDecisionResult RunOnlineDualTwoSidePnPProbe(
    const OnlineDualClassificationResult& classification,
    const std::vector<OnlineDualPnPCorrespondenceSet>& correspondence_sets,
    const OnlineDualPnPProbeAdapter& adapter);

struct OnlineDualWindowImage {
  image_t image_id = kInvalidImageId;
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  OnlineDualVisualSide visual_side = OnlineDualVisualSide::NONE;
  OnlineDualTemporalSide temporal_side = OnlineDualTemporalSide::NONE;
  size_t side_selection_rank = std::numeric_limits<size_t>::max();
  uint64_t expansion_support_strength = 0;
  bool is_current = false;
  bool is_cluster_seed = false;
};

struct OnlineDualWindowSideAudit {
  OnlineDualClusterId cluster_id = OnlineDualClusterId::NONE;
  OnlineDualTemporalSide temporal_side = OnlineDualTemporalSide::NONE;
  size_t quota = 10;
  std::vector<image_t> expansion_order;
  std::vector<image_t> selected_image_ids;
  std::vector<image_t> skipped_overlap_image_ids;
  std::vector<image_t> skipped_current_wrong_side_image_ids;
};

struct OnlineDualWindowResult {
  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  uint64_t frozen_graph_version = 0;
  image_t current_image_id = kInvalidImageId;
  OnlineDualClusterId current_cluster_id = OnlineDualClusterId::NONE;
  std::vector<OnlineDualWindowImage> images;
  OnlineDualWindowSideAudit older_side;
  OnlineDualWindowSideAudit newer_side;

  bool IsSuccess() const {
    return result_class == OnlineDualResultClass::SUCCESS;
  }
};

OnlineDualWindowResult BuildOnlineDualFrozenWindow(
    const ActiveCovisibilityGraph& frozen_graph,
    const ActiveCovisibilityOverlay& current_overlay,
    image_t current_image_id,
    size_t current_registration_sequence,
    const OnlineDualClassificationResult& classification,
    size_t max_window_size = ActiveCovisibilityGraph::kMaxWindowSize);

struct OnlineDualBackboneCandidateAudit {
  image_t older_image_id = kInvalidImageId;
  image_t newer_image_id = kInvalidImageId;
  bool reachable = false;
  double total_cost = std::numeric_limits<double>::infinity();
  std::vector<image_t> path;
};

struct OnlineDualBackboneResult {
  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualDecision decision = OnlineDualDecision::SINGLE;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  uint64_t frozen_graph_version = 0;
  image_t older_endpoint_image_id = kInvalidImageId;
  image_t newer_endpoint_image_id = kInvalidImageId;
  double total_cost = std::numeric_limits<double>::infinity();
  size_t minimum_cost_endpoint_pair_count = 0;
  std::vector<image_t> path;
  std::vector<double> edge_costs;
  std::vector<OnlineDualBackboneCandidateAudit> candidates;

  bool IsDualReady() const {
    return result_class == OnlineDualResultClass::SUCCESS &&
           decision == OnlineDualDecision::DUAL_READY;
  }
};

OnlineDualBackboneResult SelectOnlineDualBackbone(
    const ActiveCovisibilityGraph& frozen_graph,
    const OnlineDualWindowResult& window);

struct OnlineDualImageDelta {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  image_t image_id = kInvalidImageId;
  OnlineDualSE3 delta;
};

using OnlineDualImageDeltaVector =
    std::vector<OnlineDualImageDelta,
                EIGEN_ALIGNED_ALLOCATOR<OnlineDualImageDelta>>;

enum class OnlineDualImageCorrectionKind {
  DIRECT_WINDOW_RESULT,
  PROPAGATED,
  IDENTITY_OLDER_DOMAIN,
  IDENTITY_TRANSFORM,
  POSE_ONLY,
};

const char* ToString(OnlineDualImageCorrectionKind value);

struct OnlineDualRepresentativeCandidateAudit {
  image_t image_id = kInvalidImageId;
  double translation_distance_to_median =
      std::numeric_limits<double>::infinity();
};

struct OnlineDualSourceCorrection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  image_t source_image_id = kInvalidImageId;
  double backbone_alpha = 0.0;
  bool forced_identity_by_older_window = false;
  OnlineDualSE3 delta;
};

using OnlineDualSourceCorrectionVector =
    std::vector<OnlineDualSourceCorrection,
                EIGEN_ALIGNED_ALLOCATOR<OnlineDualSourceCorrection>>;

struct OnlineDualImageCorrection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  image_t image_id = kInvalidImageId;
  OnlineDualImageCorrectionKind kind =
      OnlineDualImageCorrectionKind::IDENTITY_TRANSFORM;
  bool apply_correction = false;
  image_t owner_source_image_id = kInvalidImageId;
  double ownership_distance = std::numeric_limits<double>::infinity();
  double backbone_alpha = 0.0;
  OnlineDualSE3 delta;
};

using OnlineDualImageCorrectionVector =
    std::vector<OnlineDualImageCorrection,
                EIGEN_ALIGNED_ALLOCATOR<OnlineDualImageCorrection>>;

struct OnlineDualPropagationPlan {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  uint64_t frozen_graph_version = 0;
  Eigen::Vector3d componentwise_translation_median = Eigen::Vector3d::Zero();
  image_t representative_image_id = kInvalidImageId;
  size_t minimum_distance_representative_count = 0;
  OnlineDualSE3 representative_delta;
  std::vector<OnlineDualRepresentativeCandidateAudit>
      representative_candidates;
  OnlineDualSourceCorrectionVector source_corrections;
  OnlineDualImageCorrectionVector image_corrections;

  bool IsSuccess() const {
    return result_class == OnlineDualResultClass::SUCCESS;
  }
};

OnlineDualPropagationPlan BuildOnlineDualPropagationPlan(
    const ActiveCovisibilityGraph& frozen_graph,
    const OnlineDualWindowResult& window,
    const OnlineDualBackboneResult& backbone,
    const OnlineDualImageDeltaVector& window_deltas,
    const std::vector<image_t>& pose_only_image_ids =
        std::vector<image_t>());

struct OnlineDualPointLineageAncestor {
  image_t owner_image_id = kInvalidImageId;
  size_t owner_registration_sequence = std::numeric_limits<size_t>::max();
  bool directly_optimized = false;
};

struct OnlineDualPointLineage {
  point3D_t final_point3D_id = kInvalidPoint3DId;
  std::vector<OnlineDualPointLineageAncestor> ancestors;
};

struct OnlineDualResolvedPointOwner {
  point3D_t point3D_id = kInvalidPoint3DId;
  image_t owner_image_id = kInvalidImageId;
  size_t owner_registration_sequence = std::numeric_limits<size_t>::max();
  bool directly_optimized = false;
};

struct OnlineDualPointLineageResult {
  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  std::vector<OnlineDualResolvedPointOwner> owners;

  bool IsSuccess() const {
    return result_class == OnlineDualResultClass::SUCCESS;
  }
};

OnlineDualPointLineageResult ResolveOnlineDualPointLineage(
    const std::vector<OnlineDualPointLineage>& lineages);

enum class OnlineDualPointKind {
  VISUAL,
  LIDAR,
};

enum class OnlineDualPointCorrectionKind {
  DIRECT_OPTIMIZED,
  OWNER_NEWER_WINDOW_DELTA,
  OWNER_PROPAGATED_DELTA,
  OWNER_OLDER_WINDOW,
  OWNER_IDENTITY_DOMAIN,
  OWNER_POSE_ONLY,
  LIDAR_UNCHANGED,
};

const char* ToString(OnlineDualPointKind value);
const char* ToString(OnlineDualPointCorrectionKind value);

struct OnlineDualPointState {
  point3D_t point3D_id = kInvalidPoint3DId;
  OnlineDualPointKind point_kind = OnlineDualPointKind::VISUAL;
  image_t owner_image_id = kInvalidImageId;
  size_t owner_registration_sequence = std::numeric_limits<size_t>::max();
  ActiveCovisibilityNodeState owner_visual_state =
      ActiveCovisibilityNodeState::POSE_ONLY;
  bool directly_optimized = false;
};

struct OnlineDualPointCorrection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  point3D_t point3D_id = kInvalidPoint3DId;
  image_t owner_image_id = kInvalidImageId;
  OnlineDualPointCorrectionKind kind =
      OnlineDualPointCorrectionKind::OWNER_POSE_ONLY;
  bool apply_correction = false;
  OnlineDualSE3 delta;
};

using OnlineDualPointCorrectionVector =
    std::vector<OnlineDualPointCorrection,
                EIGEN_ALIGNED_ALLOCATOR<OnlineDualPointCorrection>>;

struct OnlineDualPointCorrectionPlan {
  OnlineDualResultClass result_class = OnlineDualResultClass::INPUT_ERROR;
  OnlineDualReason reason = OnlineDualReason::NONE;
  std::string detail;
  OnlineDualPointCorrectionVector corrections;

  bool IsSuccess() const {
    return result_class == OnlineDualResultClass::SUCCESS;
  }
};

OnlineDualPointCorrectionPlan BuildOnlineDualPointCorrectionPlan(
    const OnlineDualWindowResult& window,
    const OnlineDualImageDeltaVector& window_deltas,
    const OnlineDualPropagationPlan& propagation,
    const std::vector<OnlineDualPointState>& points);

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_ONLINE_DUAL_SELECTION_H_
