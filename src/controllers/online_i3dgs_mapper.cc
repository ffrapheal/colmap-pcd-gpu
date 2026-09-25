#include "controllers/online_i3dgs_mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <tuple>
#include <utility>

namespace colmap {
namespace {

constexpr size_t kFirstMatchBatchSize = 5;
constexpr size_t kMaximumMatchedReferences = 10;
constexpr size_t kDualHopThreshold = 10;
constexpr uint64_t kMinimumTriggerSubmittedLidarConstraints = 50;

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

bool IsValidPose(const KnownPoseSE3& pose) {
  return pose.qvec.allFinite() && pose.tvec.allFinite() &&
         pose.qvec.norm() > std::numeric_limits<double>::epsilon();
}

image_t OtherEndpoint(const ActiveCovisibilityEdgeInput& edge,
                      const image_t image_id) {
  if (edge.image_id1 == image_id) return edge.image_id2;
  if (edge.image_id2 == image_id) return edge.image_id1;
  return kInvalidImageId;
}

uint64_t TriggerSubmittedConstraintCount(
    const OnlineI3dgsBaExecution& execution) {
  uint64_t result = 0;
  for (const OnlineI3dgsBaPassAudit& pass : execution.passes) {
    result = std::max(result,
                      pass.trigger_preflight_checked
                          ? pass.trigger_preflight_lidar_constraint_count
                          : pass.trigger_submitted_lidar_constraint_count);
  }
  return result;
}

bool ContainsImage(const std::vector<image_t>& image_ids,
                   const image_t image_id) {
  return std::find(image_ids.begin(), image_ids.end(), image_id) !=
         image_ids.end();
}

bool IsAcceptedDualPnP(const OnlineDualPnPDecisionResult& result) {
  if (!result.IsDualCandidate() || !IsValidOnlineDualSE3(result.delta_init) ||
      result.probes.size() != 2) {
    return false;
  }
  std::set<OnlineDualClusterId> clusters;
  const OnlineDualResolvedPolicy policy = GetOnlineDualResolvedPolicy();
  for (const OnlineDualPnPProbeAudit& probe : result.probes) {
    if (probe.status != OnlineDualPnPProbeStatus::SUCCESS ||
        !probe.pose_is_valid_se3 ||
        !IsValidOnlineDualSE3(probe.probe_T_cw) ||
        probe.correspondence_count == 0 ||
        probe.inlier_count < policy.minimum_pnp_inliers_per_side ||
        probe.inlier_count > probe.correspondence_count ||
        probe.inlier_ratio < policy.minimum_pnp_inlier_ratio ||
        !std::isfinite(probe.inlier_ratio) ||
        !clusters.insert(probe.cluster_id).second) {
      return false;
    }
  }
  return clusters.count(OnlineDualClusterId::FIRST) == 1 &&
         clusters.count(OnlineDualClusterId::SECOND) == 1;
}

std::string JoinDetail(const std::string& prefix, const std::string& detail) {
  if (detail.empty()) return prefix;
  return prefix + ": " + detail;
}

}  // namespace

const char* ToString(const OnlineI3dgsMapperMode value) {
  switch (value) {
    case OnlineI3dgsMapperMode::NONE:
      return "NONE";
    case OnlineI3dgsMapperMode::BOOTSTRAP:
      return "BOOTSTRAP";
    case OnlineI3dgsMapperMode::SINGLE:
      return "SINGLE";
    case OnlineI3dgsMapperMode::CATCHUP:
      return "CATCHUP";
    case OnlineI3dgsMapperMode::DUAL:
      return "DUAL";
    case OnlineI3dgsMapperMode::END_OF_SEQUENCE_FLUSH:
      return "END_OF_SEQUENCE_FLUSH";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineI3dgsMapperStatus value) {
  switch (value) {
    case OnlineI3dgsMapperStatus::RUNNING:
      return "RUNNING";
    case OnlineI3dgsMapperStatus::READY_TO_FLUSH:
      return "READY_TO_FLUSH";
    case OnlineI3dgsMapperStatus::COMPLETE:
      return "COMPLETE";
    case OnlineI3dgsMapperStatus::INCOMPLETE:
      return "INCOMPLETE";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineI3dgsBaDisposition value) {
  switch (value) {
    case OnlineI3dgsBaDisposition::SUCCESS:
      return "SUCCESS";
    case OnlineI3dgsBaDisposition::RECOVERABLE_FAILURE:
      return "RECOVERABLE_FAILURE";
    case OnlineI3dgsBaDisposition::FATAL_FAILURE:
      return "FATAL_FAILURE";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineI3dgsBaFailureReason value) {
  switch (value) {
    case OnlineI3dgsBaFailureReason::NONE:
      return "NONE";
    case OnlineI3dgsBaFailureReason::
        TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM:
      return "TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM";
    case OnlineI3dgsBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED:
      return "RETRY_EVIDENCE_NOT_INCREASED";
    case OnlineI3dgsBaFailureReason::PREPARE_FAILED:
      return "PREPARE_FAILED";
    case OnlineI3dgsBaFailureReason::SOLVE_NATIVE_FAILED:
      return "SOLVE_NATIVE_FAILED";
    case OnlineI3dgsBaFailureReason::POSTPROCESS_FAILED:
      return "POSTPROCESS_FAILED";
    case OnlineI3dgsBaFailureReason::PROPAGATION_FAILED:
      return "PROPAGATION_FAILED";
    case OnlineI3dgsBaFailureReason::TRANSACTION_FAILED:
      return "TRANSACTION_FAILED";
    case OnlineI3dgsBaFailureReason::INVALID_RESULT:
      return "INVALID_RESULT";
  }
  return "UNKNOWN";
}

bool OnlineI3dgsMapperOptions::Check() const {
  return expected_frame_count > 0 && ba_window_size >= 2 &&
         ba_window_size <= ActiveCovisibilityGraph::kMaxWindowSize;
}

bool OnlineI3dgsMapperDependencies::Check() const {
  return frontend != nullptr && lidar != nullptr && pnp != nullptr &&
         local_ba != nullptr && state != nullptr;
}

OnlineI3dgsMapper::OnlineI3dgsMapper(
    const OnlineI3dgsMapperOptions& options,
    OnlineI3dgsMapperDependencies dependencies)
    : options_(options),
      dependencies_(dependencies),
      control_thread_id_(std::this_thread::get_id()) {
  summary_.status = status_;
  summary_.last_consistent_stage = "CONSTRUCTED";
  if (!options_.Check()) {
    FailRun("CONSTRUCTION", "invalid online mapper options");
  } else if (!dependencies_.Check()) {
    FailRun("CONSTRUCTION", "missing online mapper dependency");
  }
}

OnlineI3dgsMapperStatus OnlineI3dgsMapper::Status() const noexcept {
  return status_;
}

const OnlineI3dgsRunSummary& OnlineI3dgsMapper::Summary() const noexcept {
  return summary_;
}

OnlineI3dgsMapper::EdgeKey OnlineI3dgsMapper::MakeEdgeKey(
    const image_t image_id1, const image_t image_id2) {
  return image_id1 < image_id2 ? EdgeKey(image_id1, image_id2)
                              : EdgeKey(image_id2, image_id1);
}

void OnlineI3dgsMapper::FailRun(const std::string& stage,
                                const std::string& detail) {
  status_ = OnlineI3dgsMapperStatus::INCOMPLETE;
  summary_.status = status_;
  summary_.last_consistent_stage = stage;
  summary_.incomplete_reason = detail;
  if (dependencies_.state != nullptr) {
    summary_.global_ba_call_count = dependencies_.state->GlobalBaCallCount();
  }
}

OnlineI3dgsFrameResult OnlineI3dgsMapper::FailFrame(
    OnlineI3dgsFrameAudit audit,
    const std::string& stage,
    const std::string& detail) {
  FailRun(stage, detail);
  audit.termination = "INCOMPLETE";
  audit.detail = detail;
  audit.global_ba_call_count = summary_.global_ba_call_count;
  return {status_, std::move(audit)};
}

bool OnlineI3dgsMapper::CheckGlobalBaCount(std::string* error) {
  const uint64_t count = dependencies_.state->GlobalBaCallCount();
  summary_.global_ba_call_count = count;
  if (count != 0) {
    if (error != nullptr) {
      *error = "online run observed a global BA call";
    }
    return false;
  }
  return true;
}

bool OnlineI3dgsMapper::ValidateFrameInput(
    const OnlineI3dgsFrameInput& frame, std::string* error) const {
  if (frame.frame_index != next_frame_index_) {
    if (error != nullptr) *error = "ARRIVED frame index is not contiguous";
    return false;
  }
  if (frame.frame_index == 0 ||
      frame.frame_index > options_.expected_frame_count) {
    if (error != nullptr) *error = "ARRIVED frame index is outside run range";
    return false;
  }
  if (frame.image_id == kInvalidImageId || !IsValidPose(frame.fastlio_T_cw)) {
    if (error != nullptr) *error = "ARRIVED frame has invalid image or pose";
    return false;
  }
  if (frame.image_path.empty() || frame.camera_path.empty() ||
      frame.odometry_path.empty() || frame.scan_path.empty() ||
      frame.image_sha256.empty() || frame.camera_sha256.empty() ||
      frame.odometry_sha256.empty() || frame.scan_sha256.empty() ||
      frame.scan_size_bytes == 0) {
    if (error != nullptr) *error = "ARRIVED frame is missing sealed input data";
    return false;
  }
  return true;
}

bool OnlineI3dgsMapper::MatchBatch(
    const OnlineI3dgsFrameInput& frame,
    const std::vector<image_t>& references,
    std::vector<OnlineI3dgsPairMatch>* matches,
    std::string* error) {
  if (references.empty()) return true;
  if (references.size() > kFirstMatchBatchSize) {
    if (error != nullptr) *error = "matching batch exceeds five references";
    return false;
  }
  const OnlineI3dgsMatchBatchResult result =
      dependencies_.frontend->MatchExplicitReferences(
          frame.image_id, references, frame.frame_index);
  if (!CheckGlobalBaCount(error)) return false;
  if (!result.success) {
    if (error != nullptr) *error = JoinDetail("explicit matching failed", result.detail);
    return false;
  }
  if (result.pairs.size() != references.size()) {
    if (error != nullptr) *error = "matcher returned a non-bijective batch";
    return false;
  }
  for (size_t index = 0; index < references.size(); ++index) {
    const OnlineI3dgsPairMatch& pair = result.pairs[index];
    if (pair.reference_image_id != references[index] ||
        pair.verified_inlier_count > pair.raw_match_count) {
      if (error != nullptr) *error = "matcher audit does not match request order";
      return false;
    }
    matches->push_back(pair);
  }
  return true;
}

std::vector<image_t> OnlineI3dgsMapper::EffectiveActiveReferences(
    const std::vector<OnlineI3dgsPairMatch>& matches) const {
  std::vector<image_t> result;
  for (const OnlineI3dgsPairMatch& pair : matches) {
    if (pair.verified_inlier_count > 0 &&
        dependencies_.state->ActiveGraph().HasNode(pair.reference_image_id)) {
      result.push_back(pair.reference_image_id);
    }
  }
  return result;
}

bool OnlineI3dgsMapper::ComputeMaximumFiniteHop(
    const std::vector<image_t>& image_ids,
    bool* defined,
    size_t* maximum_hop,
    std::string* error) const {
  *defined = false;
  *maximum_hop = 0;
  if (image_ids.size() < 2) return true;
  const HopDistanceMatrixResult matrix =
      dependencies_.state->ActiveGraph().HopDistanceMatrix(image_ids);
  if (!matrix.IsSuccess()) {
    if (error != nullptr) *error = JoinDetail("active hop query failed", matrix.detail);
    return false;
  }
  for (size_t row = 0; row < matrix.distances.size(); ++row) {
    for (size_t column = row + 1; column < matrix.distances[row].size();
         ++column) {
      const HopDistance& distance = matrix.distances[row][column];
      if (!distance.IsFinite()) {
        if (error != nullptr) *error = "canonical active graph is disconnected";
        return false;
      }
      *defined = true;
      *maximum_hop = std::max(*maximum_hop, distance.hops);
    }
  }
  return true;
}

bool OnlineI3dgsMapper::UpdatePendingEdges(
    const image_t current_image_id,
    const size_t frame_index,
    const std::vector<OnlineI3dgsPairMatch>& matches,
    std::vector<ActiveCovisibilityEdgeInput>* accepted,
    std::string* error) {
  std::vector<OrdinaryPairSupport> supports;
  supports.reserve(matches.size());
  for (const OnlineI3dgsPairMatch& pair : matches) {
    const KnownPoseRecordQueryResult record =
        dependencies_.state->KnownPoses().GetByImageId(pair.reference_image_id);
    if (!record.IsSuccess()) {
      if (error != nullptr) *error = "matched reference is absent from registry";
      return false;
    }
    supports.emplace_back(pair.reference_image_id,
                          record.record.registration_sequence,
                          pair.verified_inlier_count);
  }
  const OrdinaryPairAcceptanceResult decisions =
      SelectOrdinaryPairEdges(current_image_id, frame_index, supports);
  if (!decisions.IsSuccess()) {
    if (error != nullptr) {
      *error = JoinDetail("ordinary edge selection failed", decisions.detail);
    }
    return false;
  }
  *accepted = decisions.accepted_edges;
  for (const ActiveCovisibilityEdgeInput& edge : decisions.accepted_edges) {
    if (dependencies_.state->ActiveGraph().HasEdge(edge.image_id1,
                                                   edge.image_id2)) {
      continue;
    }
    const EdgeKey key = MakeEdgeKey(edge.image_id1, edge.image_id2);
    auto existing = pending_edges_.find(key);
    if (existing == pending_edges_.end()) {
      PendingEdge pending;
      pending.edge = edge;
      pending.evidence_version = next_edge_evidence_version_++;
      pending_edges_.emplace(key, std::move(pending));
    } else if (edge.strength > existing->second.edge.strength) {
      existing->second.edge = edge;
      existing->second.evidence_version = next_edge_evidence_version_++;
    }
  }
  return true;
}

std::vector<OnlineI3dgsMapper::PendingEdge>
OnlineI3dgsMapper::DirectPendingEdgesToActive(const image_t image_id) {
  std::vector<PendingEdge> result;
  for (auto& entry : pending_edges_) {
    PendingEdge& pending = entry.second;
    const image_t other = OtherEndpoint(pending.edge, image_id);
    if (other != kInvalidImageId &&
        dependencies_.state->ActiveGraph().HasNode(other) &&
        !dependencies_.state->ActiveGraph().HasEdge(image_id, other)) {
      if (pending.direct_active_evidence_version == 0) {
        pending.direct_active_evidence_version = next_edge_evidence_version_++;
      }
      pending.evidence_version =
          std::max(pending.evidence_version,
                   pending.direct_active_evidence_version);
      result.push_back(pending);
    }
  }
  std::sort(result.begin(), result.end(), [](const PendingEdge& lhs,
                                             const PendingEdge& rhs) {
    if (lhs.edge.strength != rhs.edge.strength) {
      return lhs.edge.strength > rhs.edge.strength;
    }
    return MakeEdgeKey(lhs.edge.image_id1, lhs.edge.image_id2) <
           MakeEdgeKey(rhs.edge.image_id1, rhs.edge.image_id2);
  });
  return result;
}

std::vector<image_t> OnlineI3dgsMapper::BuildBootstrapWindow(
    const image_t trigger_image_id) const {
  std::vector<image_t> selected{trigger_image_id};
  std::set<image_t> selected_set{trigger_image_id};
  while (selected.size() < options_.ba_window_size) {
    bool found = false;
    image_t best_image_id = kInvalidImageId;
    uint64_t best_strength = 0;
    size_t best_sequence = std::numeric_limits<size_t>::max();
    for (const auto& entry : pending_edges_) {
      const ActiveCovisibilityEdgeInput& edge = entry.second.edge;
      const bool first_selected = selected_set.count(edge.image_id1) != 0;
      const bool second_selected = selected_set.count(edge.image_id2) != 0;
      if (first_selected == second_selected) continue;
      const image_t candidate = first_selected ? edge.image_id2 : edge.image_id1;
      const KnownPoseRecordQueryResult record =
          dependencies_.state->KnownPoses().GetByImageId(candidate);
      if (!record.IsSuccess() ||
          record.record.visual_state != KnownPoseVisualState::POSE_ONLY) {
        continue;
      }
      uint64_t support = 0;
      for (const auto& support_entry : pending_edges_) {
        const ActiveCovisibilityEdgeInput& support_edge =
            support_entry.second.edge;
        const image_t other = OtherEndpoint(support_edge, candidate);
        if (other != kInvalidImageId && selected_set.count(other) != 0) {
          const uint64_t strength = support_edge.strength;
          support = std::numeric_limits<uint64_t>::max() - support < strength
                        ? std::numeric_limits<uint64_t>::max()
                        : support + strength;
        }
      }
      const auto key = std::make_tuple(std::numeric_limits<uint64_t>::max() - support,
                                       record.record.registration_sequence,
                                       candidate);
      const auto best_key = std::make_tuple(
          std::numeric_limits<uint64_t>::max() - best_strength,
          best_sequence,
          best_image_id);
      if (!found || key < best_key) {
        found = true;
        best_image_id = candidate;
        best_strength = support;
        best_sequence = record.record.registration_sequence;
      }
    }
    if (!found) break;
    selected.push_back(best_image_id);
    selected_set.insert(best_image_id);
  }
  return selected;
}

bool OnlineI3dgsMapper::BuildActiveWindow(
    const image_t trigger_image_id,
    const std::vector<PendingEdge>& connecting_edges,
    std::vector<image_t>* window,
    std::string* error) const {
  const KnownPoseRecordQueryResult trigger =
      dependencies_.state->KnownPoses().GetByImageId(trigger_image_id);
  if (!trigger.IsSuccess() ||
      trigger.record.visual_state != KnownPoseVisualState::POSE_ONLY) {
    if (error != nullptr) *error = "window trigger is not POSE_ONLY";
    return false;
  }
  ActiveCovisibilityOverlay overlay =
      dependencies_.state->ActiveGraph().CreateTentativeOverlay();
  overlay.temporary_nodes.emplace_back(trigger_image_id,
                                       trigger.record.registration_sequence);
  for (const PendingEdge& pending : connecting_edges) {
    overlay.ordinary_edges.push_back(pending.edge);
  }
  const ActiveCovisibilityResult validation =
      dependencies_.state->ActiveGraph().ValidateTentativeOverlay(overlay);
  if (!validation.IsSuccess()) {
    if (error != nullptr) *error = JoinDetail("invalid active overlay", validation.detail);
    return false;
  }
  const WindowExpansionResult expansion =
      dependencies_.state->ActiveGraph().ExpandWindow(
          {trigger_image_id}, options_.ba_window_size, &overlay);
  if (!expansion.IsSuccess()) {
    if (error != nullptr) *error = JoinDetail("active window failed", expansion.detail);
    return false;
  }
  window->clear();
  for (const WindowSelection& selection : expansion.selection_order) {
    window->push_back(selection.image_id);
  }
  if (window->empty() || !ContainsImage(*window, trigger_image_id) ||
      window->size() > options_.ba_window_size) {
    if (error != nullptr) *error = "active window violates trigger or size bound";
    return false;
  }
  return true;
}

bool OnlineI3dgsMapper::ValidateSuccessfulBa(
    const OnlineI3dgsBaRequest& request,
    const OnlineI3dgsBaExecution& execution,
    std::string* error) const {
  if (!request.all_camera_poses_variable || !request.intrinsics_fixed ||
      request.pose_prior_enabled || request.fallback_allowed ||
      request.requested_backend != "custom_cuda" ||
      request.requested_problem_source != "native_graph" ||
      request.solve_api != "BundleAdjuster::SolveNative" ||
      request.expected_canonical_version.known_pose_registry !=
          request.expected_registry_version ||
      request.expected_canonical_version.active_covisibility_graph !=
          request.expected_graph_version ||
      request.lidar_map_version == 0 ||
      request.max_scan_index != request.max_visible_frame_index ||
      request.lidar_snapshot_sha256.empty() ||
      request.lidar_geometry_sha256.empty() ||
      request.frozen_image_ids.empty() ||
      request.frozen_image_ids.size() > options_.ba_window_size ||
      !ContainsImage(request.frozen_image_ids, request.trigger_image_id)) {
    if (error != nullptr) *error = "BA request violates online invariants";
    return false;
  }
  if (execution.disposition != OnlineI3dgsBaDisposition::SUCCESS ||
      execution.failure_reason != OnlineI3dgsBaFailureReason::NONE ||
      execution.candidate == nullptr ||
      execution.passes.size() != request.required_pass_count ||
      execution.total_postprocess_count != 1) {
    if (error != nullptr) *error = "successful BA result is incomplete";
    return false;
  }
  for (size_t index = 0; index < execution.passes.size(); ++index) {
    const OnlineI3dgsBaPassAudit& pass = execution.passes[index];
    if (pass.pass_index != index + 1 || !pass.solve_native_invoked ||
        !pass.solve_native_success || !pass.finite_costs ||
        !pass.finite_output ||
        pass.trigger_submitted_lidar_constraint_count <
            request.minimum_trigger_submitted_lidar_constraints ||
        pass.constant_camera_pose_count != 0 ||
        pass.requested_backend != "custom_cuda" ||
        pass.executed_backend != "custom_cuda" ||
        pass.requested_problem_source != "native_graph" ||
        pass.executed_problem_source != "native_graph" ||
        pass.fallback_used ||
        (pass.termination != "CONVERGENCE" &&
         pass.termination != "NO_CONVERGENCE")) {
      if (error != nullptr) *error = "BA pass violates native online contract";
      return false;
    }
    if (request.mode == OnlineI3dgsMapperMode::DUAL) {
      const size_t expected_postprocess = index + 1 == execution.passes.size();
      if (pass.postprocess_count != expected_postprocess) {
        if (error != nullptr) *error = "DUAL postprocess count is not 0 then 1";
        return false;
      }
    } else if (pass.postprocess_count != 1) {
      if (error != nullptr) *error = "single-pass BA must postprocess once";
      return false;
    }
  }
  if (request.mode == OnlineI3dgsMapperMode::DUAL &&
      (!execution.propagation_applied || request.required_pass_count != 2)) {
    if (error != nullptr) *error = "DUAL result lacks two-pass propagation";
    return false;
  }
  std::set<image_t> pose_ids;
  bool has_trigger_pose = false;
  for (const OnlineI3dgsImagePoseUpdate& update : execution.pose_updates) {
    if (update.image_id == kInvalidImageId ||
        !IsValidPose(update.latest_T_cw) ||
        !pose_ids.insert(update.image_id).second) {
      if (error != nullptr) *error = "BA pose updates are invalid or duplicated";
      return false;
    }
    has_trigger_pose = has_trigger_pose || update.image_id == request.trigger_image_id;
  }
  if (!has_trigger_pose) {
    if (error != nullptr) *error = "BA result omits trigger pose";
    return false;
  }
  std::set<image_t> lidar_count_ids;
  bool has_trigger_lidar_count = false;
  for (const OnlineI3dgsPerImageSubmittedLidarConstraintCount& count :
       execution.per_image_submitted_lidar_constraint_counts) {
    if (!ContainsImage(request.frozen_image_ids, count.image_id) ||
        !lidar_count_ids.insert(count.image_id).second) {
      if (error != nullptr) *error = "per-image LiDAR counts are invalid";
      return false;
    }
    if (count.image_id == request.trigger_image_id &&
        count.submitted_lidar_constraint_count !=
            execution.passes.back()
                .trigger_submitted_lidar_constraint_count) {
      if (error != nullptr) *error = "trigger LiDAR accounting is inconsistent";
      return false;
    }
    has_trigger_lidar_count =
        has_trigger_lidar_count || count.image_id == request.trigger_image_id;
  }
  if (!has_trigger_lidar_count ||
      lidar_count_ids.size() != request.frozen_image_ids.size()) {
    if (error != nullptr) *error = "BA result omits per-image LiDAR counts";
    return false;
  }
  return true;
}

OnlineI3dgsCommitRequest OnlineI3dgsMapper::BuildCommitRequest(
    const OnlineI3dgsBaRequest& request,
    const std::vector<image_t>& promoted_image_ids,
    const std::vector<ActiveCovisibilityEdgeInput>& ordinary_edges,
    const std::vector<ActiveCovisibilityEdgeInput>& loop_edges,
    const OnlineI3dgsBaExecution& execution,
    const bool bootstrap) const {
  OnlineI3dgsCommitRequest commit;
  commit.mode = request.mode;
  commit.attempt_id = request.attempt_id;
  commit.frame_index = request.frame_index;
  commit.trigger_image_id = request.trigger_image_id;
  commit.expected_canonical_version = request.expected_canonical_version;
  commit.expected_registry_version = request.expected_registry_version;
  commit.expected_graph_version = request.expected_graph_version;
  commit.promoted_image_ids = promoted_image_ids;
  commit.ordinary_edges = ordinary_edges;
  commit.loop_edges = loop_edges;
  const std::set<image_t> promoted(promoted_image_ids.begin(),
                                   promoted_image_ids.end());
  for (const OnlineI3dgsImagePoseUpdate& update : execution.pose_updates) {
    if (promoted.count(update.image_id) != 0 ||
        dependencies_.state->ActiveGraph().HasNode(update.image_id)) {
      commit.pose_updates.push_back(update);
    }
  }
  commit.logical_write_order = {
      OnlineI3dgsCommitWriteKind::STATE_AND_ACTIVE_NODE,
      OnlineI3dgsCommitWriteKind::ORDINARY_EDGES,
      OnlineI3dgsCommitWriteKind::RECONSTRUCTION_AND_OWNERS};
  if (request.mode == OnlineI3dgsMapperMode::DUAL) {
    commit.logical_write_order.push_back(
        OnlineI3dgsCommitWriteKind::PROPAGATION);
    commit.logical_write_order.push_back(
        OnlineI3dgsCommitWriteKind::LOOP_EDGES);
  }
  if (bootstrap) {
    std::sort(commit.promoted_image_ids.begin(),
              commit.promoted_image_ids.end());
  }
  return commit;
}

bool OnlineI3dgsMapper::ExecuteAndCommit(
    const OnlineI3dgsBaRequest& request,
    const std::vector<image_t>& promoted_image_ids,
    const std::vector<ActiveCovisibilityEdgeInput>& ordinary_edges,
    const std::vector<ActiveCovisibilityEdgeInput>& loop_edges,
    const bool bootstrap,
    OnlineI3dgsBaExecution* execution,
    OnlineI3dgsCommitResult* commit,
    std::string* error) {
  *execution = dependencies_.local_ba->Execute(request);
  if (!CheckGlobalBaCount(error)) {
    execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
    execution->failure_reason = OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
    return false;
  }
  if (dependencies_.state->CanonicalVersion() !=
          request.expected_canonical_version ||
      dependencies_.state->KnownPoses().Version() !=
          request.expected_registry_version ||
      dependencies_.state->ActiveGraph().Version() !=
          request.expected_graph_version) {
    execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
    execution->failure_reason = OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
    execution->detail = "BA Execute exposed candidate state before commit";
    if (error != nullptr) *error = execution->detail;
    return false;
  }
  if (execution->disposition != OnlineI3dgsBaDisposition::SUCCESS) {
    if (error != nullptr) *error = execution->detail;
    return false;
  }
  if (request.mode == OnlineI3dgsMapperMode::CATCHUP &&
      request.has_previous_catchup_failure &&
      request.active_edge_evidence_version <=
          request.previous_catchup_failure
              .active_edge_evidence_version) {
    const uint64_t previous_count =
        request.previous_catchup_failure
            .trigger_submitted_lidar_constraint_count;
    const uint64_t current_count =
        TriggerSubmittedConstraintCount(*execution);
    if (current_count < previous_count ||
        current_count - previous_count <
            kMinimumTriggerSubmittedLidarConstraints) {
      execution->disposition = OnlineI3dgsBaDisposition::RECOVERABLE_FAILURE;
      execution->failure_reason =
          OnlineI3dgsBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED;
      execution->detail =
          "CATCHUP retry has neither new active edge evidence nor 50 new "
          "trigger residuals";
      execution->candidate.reset();
      if (error != nullptr) *error = execution->detail;
      return false;
    }
  }
  if (!ValidateSuccessfulBa(request, *execution, error)) {
    execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
    execution->failure_reason = OnlineI3dgsBaFailureReason::INVALID_RESULT;
    return false;
  }
  const OnlineI3dgsCommitRequest commit_request = BuildCommitRequest(
      request, promoted_image_ids, ordinary_edges, loop_edges, *execution,
      bootstrap);
  *commit = dependencies_.local_ba->Commit(commit_request, execution);
  const uint64_t actual_registry_version =
      dependencies_.state->KnownPoses().Version();
  const uint64_t actual_graph_version =
      dependencies_.state->ActiveGraph().Version();
  if (!commit->success) {
    execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
    execution->failure_reason = OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
    if (dependencies_.state->CanonicalVersion() !=
            request.expected_canonical_version ||
        commit->canonical_version_before !=
            request.expected_canonical_version ||
        commit->canonical_version_after !=
            request.expected_canonical_version ||
        actual_registry_version != request.expected_registry_version ||
        actual_graph_version != request.expected_graph_version ||
        commit->registry_version_before != request.expected_registry_version ||
        commit->registry_version_after != request.expected_registry_version ||
        commit->graph_version_before != request.expected_graph_version ||
        commit->graph_version_after != request.expected_graph_version) {
      ++summary_.partial_commit_detection_count;
      if (error != nullptr) *error = "failed commit exposed partial canonical state";
    } else if (error != nullptr) {
      *error = JoinDetail("atomic commit failed", commit->detail);
    }
    return false;
  }
  if (!commit->atomic_publication ||
      commit->canonical_version_before != request.expected_canonical_version ||
      dependencies_.state->CanonicalVersion() !=
          commit->canonical_version_after ||
      commit->registry_version_before != request.expected_registry_version ||
      commit->registry_version_after != request.expected_registry_version + 1 ||
      commit->graph_version_before != request.expected_graph_version ||
      commit->graph_version_after != request.expected_graph_version + 1 ||
      actual_registry_version != commit->registry_version_after ||
      actual_graph_version != commit->graph_version_after) {
    ++summary_.partial_commit_detection_count;
    execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
    execution->failure_reason = OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
    if (error != nullptr) *error = "commit did not publish exactly one atomic version";
    return false;
  }
  for (const image_t image_id : promoted_image_ids) {
    const KnownPoseRecordQueryResult record =
        dependencies_.state->KnownPoses().GetByImageId(image_id);
    if (!record.IsSuccess() ||
        record.record.visual_state != KnownPoseVisualState::VISUAL_ACTIVE ||
        !dependencies_.state->ActiveGraph().HasNode(image_id)) {
      ++summary_.partial_commit_detection_count;
      execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      execution->failure_reason =
          OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
      if (error != nullptr) *error = "commit omitted promoted canonical state";
      return false;
    }
  }
  for (const ActiveCovisibilityEdgeInput& edge : ordinary_edges) {
    if (!dependencies_.state->ActiveGraph().HasEdge(edge.image_id1,
                                                    edge.image_id2)) {
      ++summary_.partial_commit_detection_count;
      execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      execution->failure_reason =
          OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
      if (error != nullptr) *error = "commit omitted an ordinary edge";
      return false;
    }
  }
  for (const ActiveCovisibilityEdgeInput& edge : loop_edges) {
    if (!dependencies_.state->ActiveGraph().HasEdge(edge.image_id1,
                                                    edge.image_id2)) {
      ++summary_.partial_commit_detection_count;
      execution->disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      execution->failure_reason =
          OnlineI3dgsBaFailureReason::TRANSACTION_FAILED;
      if (error != nullptr) *error = "commit omitted a loop edge";
      return false;
    }
  }
  for (const ActiveCovisibilityEdgeInput& edge : ordinary_edges) {
    pending_edges_.erase(MakeEdgeKey(edge.image_id1, edge.image_id2));
  }
  for (const ActiveCovisibilityEdgeInput& edge : loop_edges) {
    pending_edges_.erase(MakeEdgeKey(edge.image_id1, edge.image_id2));
  }
  return true;
}

OnlineI3dgsCatchupAudit OnlineI3dgsMapper::RunCatchup(
    const image_t image_id,
    const size_t frame_index,
    const bool is_flush,
    bool* fatal) {
  OnlineI3dgsCatchupAudit audit;
  audit.image_id = image_id;
  audit.canonical_version_before = dependencies_.state->CanonicalVersion();
  audit.canonical_version_after = audit.canonical_version_before;
  audit.registry_version_before = dependencies_.state->KnownPoses().Version();
  audit.graph_version_before = dependencies_.state->ActiveGraph().Version();
  const KnownPoseRecordQueryResult record =
      dependencies_.state->KnownPoses().GetByImageId(image_id);
  if (!record.IsSuccess() ||
      record.record.visual_state != KnownPoseVisualState::POSE_ONLY) {
    audit.detail = "catchup candidate is no longer POSE_ONLY";
    return audit;
  }
  const OnlineI3dgsCatchupFailureQueryResult previous_failure =
      dependencies_.state->GetCatchupFailure(image_id);
  if (!previous_failure.success) {
    audit.detail = JoinDetail("CATCHUP evidence query failed",
                              previous_failure.detail);
    *fatal = true;
    return audit;
  }
  audit.registration_sequence = record.record.registration_sequence;
  audit.had_previous_failure = previous_failure.has_evidence;
  const std::vector<PendingEdge> connecting_edges =
      DirectPendingEdgesToActive(image_id);
  if (connecting_edges.empty()) {
    audit.detail = "no pending edge directly connects canonical active state";
    return audit;
  }
  for (const PendingEdge& edge : connecting_edges) {
    audit.active_edge_evidence_version =
        std::max(audit.active_edge_evidence_version, edge.evidence_version);
  }
  audit.retry_evidence_sufficient =
      !previous_failure.has_evidence ||
      audit.active_edge_evidence_version >
          previous_failure.evidence.active_edge_evidence_version;
  std::string window_error;
  if (!BuildActiveWindow(image_id, connecting_edges,
                         &audit.frozen_image_ids, &window_error)) {
    audit.detail = window_error;
    *fatal = true;
    return audit;
  }

  OnlineI3dgsBaRequest request;
  request.mode = OnlineI3dgsMapperMode::CATCHUP;
  request.attempt_id = next_ba_attempt_id_++;
  request.frame_index = frame_index;
  request.max_visible_frame_index = frame_index;
  request.trigger_image_id = image_id;
  request.expected_canonical_version =
      dependencies_.state->CanonicalVersion();
  request.expected_registry_version = audit.registry_version_before;
  request.expected_graph_version = audit.graph_version_before;
  request.lidar_map_version = last_lidar_map_version_;
  request.max_scan_index = frame_index;
  request.lidar_snapshot_sha256 = last_lidar_snapshot_sha256_;
  request.lidar_geometry_sha256 = last_lidar_geometry_sha256_;
  request.frozen_image_ids = audit.frozen_image_ids;
  for (const PendingEdge& edge : connecting_edges) {
    request.ordinary_edges.push_back(edge.edge);
  }
  request.required_pass_count = 1;
  request.minimum_trigger_submitted_lidar_constraints =
      kMinimumTriggerSubmittedLidarConstraints;
  request.is_flush = is_flush;
  request.has_previous_catchup_failure = previous_failure.has_evidence;
  request.previous_catchup_failure = previous_failure.evidence;
  request.active_edge_evidence_version = audit.active_edge_evidence_version;

  ++summary_.catchup_call_count;
  audit.attempted = true;
  OnlineI3dgsBaExecution execution;
  OnlineI3dgsCommitResult commit;
  std::string execution_error;
  const bool committed = ExecuteAndCommit(
      request, {image_id}, request.ordinary_edges, {}, false, &execution,
      &commit, &execution_error);
  audit.trigger_submitted_lidar_constraint_count =
      TriggerSubmittedConstraintCount(execution);
  audit.failure_reason = execution.failure_reason;
  audit.frozen_image_ids = request.frozen_image_ids;
  summary_.ba_pass_count += execution.passes.size();
  if (committed) {
    audit.promoted = true;
    audit.retry_evidence_sufficient = true;
    audit.registry_version_after = commit.registry_version_after;
    audit.graph_version_after = commit.graph_version_after;
    audit.canonical_version_after = commit.canonical_version_after;
    ++summary_.catchup_commit_count;
    audit.detail = "CATCHUP committed";
    return audit;
  }

  audit.registry_version_after = dependencies_.state->KnownPoses().Version();
  audit.graph_version_after = dependencies_.state->ActiveGraph().Version();
  audit.canonical_version_after = dependencies_.state->CanonicalVersion();
  if (summary_.partial_commit_detection_count != 0 ||
      execution.disposition == OnlineI3dgsBaDisposition::FATAL_FAILURE) {
    audit.detail = execution_error;
    *fatal = true;
    return audit;
  }
  if (execution.failure_reason ==
      OnlineI3dgsBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED) {
    audit.retry_evidence_sufficient = false;
    audit.detail = execution_error;
    return audit;
  }
  OnlineI3dgsCatchupFailureEvidence evidence;
  evidence.active_edge_evidence_version =
      audit.active_edge_evidence_version;
  evidence.trigger_submitted_lidar_constraint_count =
      audit.trigger_submitted_lidar_constraint_count;
  evidence.lidar_map_version = last_lidar_map_version_;
  const OnlineI3dgsDependencyResult recorded =
      dependencies_.state->RecordCatchupFailure(image_id, evidence);
  if (!recorded.success) {
    audit.detail = JoinDetail("failed to record CATCHUP evidence", recorded.detail);
    *fatal = true;
    return audit;
  }
  audit.registry_version_after = dependencies_.state->KnownPoses().Version();
  audit.graph_version_after = dependencies_.state->ActiveGraph().Version();
  audit.canonical_version_after = dependencies_.state->CanonicalVersion();
  audit.detail = execution_error;
  return audit;
}

bool OnlineI3dgsMapper::RunCatchups(
    const size_t frame_index,
    const image_t excluded_current_image_id,
    const bool is_flush,
    std::vector<OnlineI3dgsCatchupAudit>* audits,
    size_t* promotion_count,
    std::string* error) {
  *promotion_count = 0;
  const KnownPoseImageIdsResult pose_only =
      dependencies_.state->KnownPoses().GetPoseOnlyImageIds();
  if (!pose_only.IsSuccess()) {
    if (error != nullptr) *error = JoinDetail("POSE_ONLY query failed", pose_only.detail);
    return false;
  }
  for (const image_t image_id : pose_only.image_ids) {
    if (image_id == excluded_current_image_id) continue;
    bool fatal = false;
    OnlineI3dgsCatchupAudit audit =
        RunCatchup(image_id, frame_index, is_flush, &fatal);
    if (audit.attempted || !audit.detail.empty()) audits->push_back(audit);
    if (audit.promoted) ++*promotion_count;
    if (fatal) {
      if (error != nullptr) *error = audit.detail;
      return false;
    }
    if (!CheckGlobalBaCount(error)) return false;
  }
  return true;
}

std::vector<ActiveCovisibilityEdgeInput> OnlineI3dgsMapper::EdgesWithin(
    const std::vector<image_t>& image_ids) const {
  const std::set<image_t> selected(image_ids.begin(), image_ids.end());
  std::vector<ActiveCovisibilityEdgeInput> result;
  for (const auto& entry : pending_edges_) {
    const ActiveCovisibilityEdgeInput& edge = entry.second.edge;
    if (selected.count(edge.image_id1) != 0 &&
        selected.count(edge.image_id2) != 0) {
      result.push_back(edge);
    }
  }
  return result;
}

std::vector<image_t> OnlineI3dgsMapper::BootstrapPromotedSet(
    const image_t trigger_image_id,
    const std::vector<image_t>& window,
    const OnlineI3dgsBaExecution& execution) const {
  std::map<image_t, uint64_t> counts;
  for (const OnlineI3dgsPerImageSubmittedLidarConstraintCount& item :
       execution.per_image_submitted_lidar_constraint_counts) {
    counts[item.image_id] = item.submitted_lidar_constraint_count;
  }
  std::set<image_t> eligible;
  for (const image_t image_id : window) {
    if (counts[image_id] >= kMinimumTriggerSubmittedLidarConstraints) {
      eligible.insert(image_id);
    }
  }
  if (eligible.count(trigger_image_id) == 0) return {};
  const std::vector<ActiveCovisibilityEdgeInput> edges = EdgesWithin(window);
  std::set<image_t> reached{trigger_image_id};
  std::queue<image_t> queue;
  queue.push(trigger_image_id);
  while (!queue.empty()) {
    const image_t current = queue.front();
    queue.pop();
    for (const ActiveCovisibilityEdgeInput& edge : edges) {
      const image_t other = OtherEndpoint(edge, current);
      if (other != kInvalidImageId && eligible.count(other) != 0 &&
          reached.insert(other).second) {
        queue.push(other);
      }
    }
  }
  std::vector<image_t> result(reached.begin(), reached.end());
  std::sort(result.begin(), result.end(), [&](const image_t lhs,
                                               const image_t rhs) {
    const KnownPoseRecordQueryResult lhs_record =
        dependencies_.state->KnownPoses().GetByImageId(lhs);
    const KnownPoseRecordQueryResult rhs_record =
        dependencies_.state->KnownPoses().GetByImageId(rhs);
    if (lhs_record.IsSuccess() && rhs_record.IsSuccess() &&
        lhs_record.record.registration_sequence !=
            rhs_record.record.registration_sequence) {
      return lhs_record.record.registration_sequence <
             rhs_record.record.registration_sequence;
    }
    return lhs < rhs;
  });
  return result;
}

bool OnlineI3dgsMapper::RunBootstrap(const OnlineI3dgsFrameInput& frame,
                                     OnlineI3dgsFrameAudit* audit,
                                     std::string* error) {
  audit->mode = OnlineI3dgsMapperMode::BOOTSTRAP;
  const std::vector<image_t> window = BuildBootstrapWindow(frame.image_id);
  audit->frozen_image_ids = window;
  if (window.size() < 2) {
    audit->detail = "BOOTSTRAP has no connected visual seed component";
    return true;
  }
  OnlineI3dgsBaRequest request;
  request.mode = OnlineI3dgsMapperMode::BOOTSTRAP;
  request.attempt_id = next_ba_attempt_id_++;
  request.frame_index = frame.frame_index;
  request.max_visible_frame_index = frame.frame_index;
  request.trigger_image_id = frame.image_id;
  request.expected_canonical_version =
      dependencies_.state->CanonicalVersion();
  request.expected_registry_version =
      dependencies_.state->KnownPoses().Version();
  request.expected_graph_version =
      dependencies_.state->ActiveGraph().Version();
  request.lidar_map_version = last_lidar_map_version_;
  request.max_scan_index = frame.frame_index;
  request.lidar_snapshot_sha256 = last_lidar_snapshot_sha256_;
  request.lidar_geometry_sha256 = last_lidar_geometry_sha256_;
  request.frozen_image_ids = window;
  request.ordinary_edges = EdgesWithin(window);
  request.required_pass_count = 1;
  ++summary_.bootstrap_call_count;
  OnlineI3dgsBaExecution execution = dependencies_.local_ba->Execute(request);
  summary_.ba_pass_count += execution.passes.size();
  audit->ba_passes = execution.passes;
  if (!CheckGlobalBaCount(error)) return false;
  if (dependencies_.state->CanonicalVersion() !=
          request.expected_canonical_version ||
      dependencies_.state->KnownPoses().Version() !=
          request.expected_registry_version ||
      dependencies_.state->ActiveGraph().Version() !=
          request.expected_graph_version) {
    if (error != nullptr) {
      *error = "BOOTSTRAP Execute exposed candidate state before commit";
    }
    return false;
  }
  if (execution.disposition != OnlineI3dgsBaDisposition::SUCCESS) {
    if (execution.disposition == OnlineI3dgsBaDisposition::FATAL_FAILURE) {
      if (error != nullptr) *error = execution.detail;
      return false;
    }
    audit->detail = execution.detail;
    return true;
  }
  std::string validation_error;
  if (!ValidateSuccessfulBa(request, execution, &validation_error)) {
    if (error != nullptr) *error = validation_error;
    return false;
  }
  const std::vector<image_t> promoted =
      BootstrapPromotedSet(frame.image_id, window, execution);
  if (promoted.size() < 2) {
    audit->detail = "BOOTSTRAP promoted component contains fewer than two images";
    return true;
  }
  const std::vector<ActiveCovisibilityEdgeInput> edges = EdgesWithin(promoted);
  if (edges.empty()) {
    if (error != nullptr) *error = "BOOTSTRAP promoted component is disconnected";
    return false;
  }
  const OnlineI3dgsCommitRequest commit_request = BuildCommitRequest(
      request, promoted, edges, {}, execution, true);
  audit->commit_attempted = true;
  const OnlineI3dgsCommitResult commit =
      dependencies_.local_ba->Commit(commit_request, &execution);
  const uint64_t actual_registry_version =
      dependencies_.state->KnownPoses().Version();
  const uint64_t actual_graph_version =
      dependencies_.state->ActiveGraph().Version();
  if (!commit.success) {
    if (dependencies_.state->CanonicalVersion() !=
            request.expected_canonical_version ||
        commit.canonical_version_before !=
            request.expected_canonical_version ||
        commit.canonical_version_after !=
            request.expected_canonical_version ||
        actual_registry_version != request.expected_registry_version ||
        actual_graph_version != request.expected_graph_version ||
        commit.registry_version_after != request.expected_registry_version ||
        commit.graph_version_after != request.expected_graph_version) {
      ++summary_.partial_commit_detection_count;
      if (error != nullptr) *error = "BOOTSTRAP commit partially published";
      return false;
    }
    if (error != nullptr) {
      *error = JoinDetail("BOOTSTRAP atomic commit failed", commit.detail);
    }
    return false;
  }
  if (!commit.atomic_publication ||
      commit.canonical_version_before != request.expected_canonical_version ||
      dependencies_.state->CanonicalVersion() !=
          commit.canonical_version_after ||
      commit.registry_version_before != request.expected_registry_version ||
      commit.registry_version_after != request.expected_registry_version + 1 ||
      commit.graph_version_before != request.expected_graph_version ||
      commit.graph_version_after != request.expected_graph_version + 1 ||
      actual_registry_version != commit.registry_version_after ||
      actual_graph_version != commit.graph_version_after) {
    ++summary_.partial_commit_detection_count;
    if (error != nullptr) *error = "BOOTSTRAP commit was not one atomic publication";
    return false;
  }
  for (const image_t image_id : promoted) {
    const KnownPoseRecordQueryResult record =
        dependencies_.state->KnownPoses().GetByImageId(image_id);
    if (!record.IsSuccess() ||
        record.record.visual_state != KnownPoseVisualState::VISUAL_ACTIVE ||
        !dependencies_.state->ActiveGraph().HasNode(image_id)) {
      ++summary_.partial_commit_detection_count;
      if (error != nullptr) {
        *error = "BOOTSTRAP commit omitted promoted canonical state";
      }
      return false;
    }
  }
  for (const ActiveCovisibilityEdgeInput& edge : edges) {
    if (!dependencies_.state->ActiveGraph().HasEdge(edge.image_id1,
                                                    edge.image_id2)) {
      ++summary_.partial_commit_detection_count;
      if (error != nullptr) *error = "BOOTSTRAP commit omitted an edge";
      return false;
    }
  }
  for (const ActiveCovisibilityEdgeInput& edge : edges) {
    pending_edges_.erase(MakeEdgeKey(edge.image_id1, edge.image_id2));
  }
  audit->ba_succeeded = true;
  audit->commit_succeeded = true;
  audit->current_visual_active = true;
  audit->detail = "BOOTSTRAP committed";
  return true;
}

bool OnlineI3dgsMapper::RunSingle(
    const OnlineI3dgsFrameInput& frame,
    const std::vector<ActiveCovisibilityEdgeInput>& accepted_current_edges,
    OnlineI3dgsFrameAudit* audit,
    std::string* error) {
  audit->mode = OnlineI3dgsMapperMode::SINGLE;
  std::vector<PendingEdge> connecting_edges;
  for (const ActiveCovisibilityEdgeInput& edge : accepted_current_edges) {
    const image_t other = OtherEndpoint(edge, frame.image_id);
    if (other == kInvalidImageId ||
        !dependencies_.state->ActiveGraph().HasNode(other)) {
      continue;
    }
    const auto pending = pending_edges_.find(
        MakeEdgeKey(edge.image_id1, edge.image_id2));
    if (pending != pending_edges_.end()) connecting_edges.push_back(pending->second);
  }
  if (connecting_edges.empty()) {
    audit->detail = "SINGLE has no direct canonical active connection";
    return true;
  }
  std::vector<image_t> window;
  if (!BuildActiveWindow(frame.image_id, connecting_edges, &window, error)) {
    return false;
  }
  audit->frozen_image_ids = window;
  OnlineI3dgsBaRequest request;
  request.mode = OnlineI3dgsMapperMode::SINGLE;
  request.attempt_id = next_ba_attempt_id_++;
  request.frame_index = frame.frame_index;
  request.max_visible_frame_index = frame.frame_index;
  request.trigger_image_id = frame.image_id;
  request.expected_canonical_version =
      dependencies_.state->CanonicalVersion();
  request.expected_registry_version =
      dependencies_.state->KnownPoses().Version();
  request.expected_graph_version =
      dependencies_.state->ActiveGraph().Version();
  request.lidar_map_version = last_lidar_map_version_;
  request.max_scan_index = frame.frame_index;
  request.lidar_snapshot_sha256 = last_lidar_snapshot_sha256_;
  request.lidar_geometry_sha256 = last_lidar_geometry_sha256_;
  request.frozen_image_ids = window;
  for (const PendingEdge& edge : connecting_edges) {
    request.ordinary_edges.push_back(edge.edge);
  }
  ++summary_.single_call_count;
  OnlineI3dgsBaExecution execution;
  OnlineI3dgsCommitResult commit;
  std::string execution_error;
  const bool committed = ExecuteAndCommit(
      request, {frame.image_id}, request.ordinary_edges, {}, false, &execution,
      &commit, &execution_error);
  summary_.ba_pass_count += execution.passes.size();
  audit->ba_passes = execution.passes;
  if (committed) {
    audit->ba_succeeded = true;
    audit->commit_attempted = true;
    audit->commit_succeeded = true;
    audit->current_visual_active = true;
    audit->detail = "SINGLE committed";
    return true;
  }
  audit->commit_attempted = execution.disposition ==
                            OnlineI3dgsBaDisposition::SUCCESS;
  if (summary_.partial_commit_detection_count != 0 ||
      execution.disposition == OnlineI3dgsBaDisposition::FATAL_FAILURE) {
    if (error != nullptr) *error = execution_error;
    return false;
  }
  audit->detail = execution_error;
  return true;
}

bool OnlineI3dgsMapper::RunDual(
    const OnlineI3dgsFrameInput& frame,
    const std::vector<OnlineI3dgsPairMatch>& matches,
    const std::vector<ActiveCovisibilityEdgeInput>& accepted_current_edges,
    OnlineI3dgsFrameAudit* audit,
    bool* fallback_to_single,
    std::string* error) {
  *fallback_to_single = false;
  audit->dual_probe_attempted = true;
  ++summary_.pnp_probe_call_count;
  const OnlineMapperCanonicalVersion canonical_version_before_probe =
      dependencies_.state->CanonicalVersion();
  OnlineI3dgsPnPRequest pnp_request;
  pnp_request.current_image_id = frame.image_id;
  pnp_request.frame_index = frame.frame_index;
  pnp_request.max_visible_frame_index = frame.frame_index;
  pnp_request.canonical_graph_version =
      dependencies_.state->ActiveGraph().Version();
  pnp_request.classification = audit->dual_classification;
  pnp_request.matched_pairs = matches;
  audit->dual_pnp = dependencies_.pnp->ProbeTwoSides(pnp_request);
  if (!CheckGlobalBaCount(error)) return false;
  if (dependencies_.state->CanonicalVersion() !=
      canonical_version_before_probe) {
    if (error != nullptr) *error = "DUAL PnP probe mutated canonical state";
    return false;
  }
  if (!IsAcceptedDualPnP(audit->dual_pnp)) {
    *fallback_to_single = true;
    audit->dual_fell_back_to_single = true;
    return true;
  }

  std::map<image_t, OnlineDualVisualSide> side_by_reference;
  for (const OnlineDualReferenceAssignment& assignment :
       audit->dual_classification.assignments) {
    for (const OnlineDualClusterSummary& cluster :
         audit->dual_classification.clusters) {
      if (cluster.cluster_id == assignment.cluster_id) {
        side_by_reference[assignment.reference.image_id] = cluster.visual_side;
      }
    }
  }
  std::vector<ActiveCovisibilityEdgeInput> ordinary_edges;
  for (const ActiveCovisibilityEdgeInput& edge : accepted_current_edges) {
    const image_t reference = OtherEndpoint(edge, frame.image_id);
    if (side_by_reference[reference] == OnlineDualVisualSide::MAIN) {
      ordinary_edges.push_back(edge);
    }
  }
  std::vector<ActiveCovisibilityEdgeInput> loop_edges;
  for (const OnlineI3dgsPairMatch& match : matches) {
    if (side_by_reference[match.reference_image_id] ==
            OnlineDualVisualSide::LOOP &&
        match.verified_inlier_count >=
            ActiveCovisibilityGraph::kRobustOrdinaryMinStrength) {
      loop_edges.emplace_back(frame.image_id,
                              match.reference_image_id,
                              match.verified_inlier_count,
                              frame.frame_index);
    }
  }
  if (ordinary_edges.empty() || loop_edges.empty()) {
    *fallback_to_single = true;
    audit->dual_fell_back_to_single = true;
    return true;
  }
  const KnownPoseRecordQueryResult current =
      dependencies_.state->KnownPoses().GetByImageId(frame.image_id);
  if (!current.IsSuccess()) {
    if (error != nullptr) *error = "DUAL current is absent from registry";
    return false;
  }
  ActiveCovisibilityOverlay overlay =
      dependencies_.state->ActiveGraph().CreateTentativeOverlay();
  overlay.temporary_nodes.emplace_back(frame.image_id,
                                       current.record.registration_sequence);
  overlay.ordinary_edges = ordinary_edges;
  overlay.loop_edges = loop_edges;
  const OnlineDualWindowResult dual_window = BuildOnlineDualFrozenWindow(
      dependencies_.state->ActiveGraph(), overlay, frame.image_id,
      current.record.registration_sequence, audit->dual_classification,
      options_.ba_window_size);
  if (!dual_window.IsSuccess()) {
    if (dual_window.result_class == OnlineDualResultClass::SINGLE_FALLBACK) {
      *fallback_to_single = true;
      audit->dual_fell_back_to_single = true;
      return true;
    }
    if (error != nullptr) *error = dual_window.detail;
    return false;
  }
  OnlineI3dgsBaRequest request;
  request.mode = OnlineI3dgsMapperMode::DUAL;
  request.attempt_id = next_ba_attempt_id_++;
  request.frame_index = frame.frame_index;
  request.max_visible_frame_index = frame.frame_index;
  request.trigger_image_id = frame.image_id;
  request.expected_canonical_version =
      dependencies_.state->CanonicalVersion();
  request.expected_registry_version =
      dependencies_.state->KnownPoses().Version();
  request.expected_graph_version =
      dependencies_.state->ActiveGraph().Version();
  request.lidar_map_version = last_lidar_map_version_;
  request.max_scan_index = frame.frame_index;
  request.lidar_snapshot_sha256 = last_lidar_snapshot_sha256_;
  request.lidar_geometry_sha256 = last_lidar_geometry_sha256_;
  for (const OnlineDualWindowImage& image : dual_window.images) {
    request.frozen_image_ids.push_back(image.image_id);
  }
  request.ordinary_edges = ordinary_edges;
  request.loop_edges = loop_edges;
  request.required_pass_count = 2;
  request.has_dual_context = true;
  request.dual_classification = audit->dual_classification;
  request.dual_pnp = audit->dual_pnp;
  request.dual_window = dual_window;
  audit->mode = OnlineI3dgsMapperMode::DUAL;
  audit->frozen_image_ids = request.frozen_image_ids;
  ++summary_.dual_call_count;
  OnlineI3dgsBaExecution execution;
  OnlineI3dgsCommitResult commit;
  std::string execution_error;
  const bool committed = ExecuteAndCommit(
      request, {frame.image_id}, ordinary_edges, loop_edges, false, &execution,
      &commit, &execution_error);
  summary_.ba_pass_count += execution.passes.size();
  audit->ba_passes = execution.passes;
  audit->commit_attempted = execution.disposition ==
                            OnlineI3dgsBaDisposition::SUCCESS;
  if (!committed) {
    if (error != nullptr) {
      *error = execution_error.empty() ? "DUAL execution failed"
                                      : execution_error;
    }
    return false;
  }
  audit->ba_succeeded = true;
  audit->commit_succeeded = true;
  audit->current_visual_active = true;
  ++summary_.dual_atomic_commit_count;
  audit->detail = "DUAL committed";
  return true;
}

bool OnlineI3dgsMapper::RunCurrent(
    const OnlineI3dgsFrameInput& frame,
    const std::vector<OnlineI3dgsPairMatch>& matches,
    const std::vector<ActiveCovisibilityEdgeInput>& accepted_current_edges,
    OnlineI3dgsFrameAudit* audit,
    std::string* error) {
  if (dependencies_.state->ActiveGraph().NumNodes() == 0) {
    return RunBootstrap(frame, audit, error);
  }
  std::vector<OnlineDualReference> references;
  for (const OnlineI3dgsPairMatch& match : matches) {
    if (match.verified_inlier_count == 0 ||
        !dependencies_.state->ActiveGraph().HasNode(match.reference_image_id)) {
      continue;
    }
    const KnownPoseRecordQueryResult record =
        dependencies_.state->KnownPoses().GetByImageId(match.reference_image_id);
    if (!record.IsSuccess() ||
        record.record.visual_state != KnownPoseVisualState::VISUAL_ACTIVE) {
      if (error != nullptr) *error = "active graph and registry disagree";
      return false;
    }
    references.emplace_back(match.reference_image_id,
                            record.record.registration_sequence,
                            match.verified_inlier_count,
                            ActiveCovisibilityNodeState::VISUAL_ACTIVE,
                            true,
                            true);
  }
  audit->dual_classification = ClassifyOnlineDualReferences(
      dependencies_.state->ActiveGraph(), references);
  if (audit->dual_classification.result_class ==
          OnlineDualResultClass::INPUT_ERROR ||
      audit->dual_classification.result_class ==
          OnlineDualResultClass::DUAL_FATAL) {
    if (error != nullptr) *error = audit->dual_classification.detail;
    return false;
  }
  if (audit->dual_classification.IsDualCandidate()) {
    bool fallback_to_single = false;
    if (!RunDual(frame, matches, accepted_current_edges, audit,
                 &fallback_to_single, error)) {
      return false;
    }
    if (!fallback_to_single) return true;
  }
  return RunSingle(frame, accepted_current_edges, audit, error);
}

OnlineI3dgsFrameResult OnlineI3dgsMapper::ProcessArrivedFrame(
    const OnlineI3dgsFrameInput& frame) {
  OnlineI3dgsFrameAudit audit;
  audit.frame_index = frame.frame_index;
  audit.image_id = frame.image_id;
  audit.max_visible_frame_index = frame.frame_index;
  if (std::this_thread::get_id() != control_thread_id_) {
    return FailFrame(std::move(audit), "ARRIVAL", "wrong controller thread");
  }
  if (status_ != OnlineI3dgsMapperStatus::RUNNING) {
    return FailFrame(std::move(audit), "ARRIVAL",
                     "ARRIVED event received outside RUNNING state");
  }
  std::string error;
  if (!ValidateFrameInput(frame, &error)) {
    return FailFrame(std::move(audit), "VALIDATE_INPUT", error);
  }
  if (!CheckGlobalBaCount(&error)) {
    return FailFrame(std::move(audit), "GLOBAL_BA_GUARD", error);
  }

  Clock::time_point stage_start = Clock::now();
  const OnlineI3dgsLidarIngestResult lidar =
      dependencies_.lidar->IngestScan(frame, frame.frame_index);
  audit.stage_milliseconds["lidar_ingest"] = ElapsedMilliseconds(stage_start);
  if (!lidar.success || lidar.opened_scan_index != frame.frame_index ||
      lidar.max_scan_index != frame.frame_index ||
      lidar.map_version_before != last_lidar_map_version_ ||
      lidar.map_version_before == std::numeric_limits<uint64_t>::max() ||
      lidar.map_version_after != lidar.map_version_before + 1 ||
      lidar.point_transform_count_min != 1 ||
      lidar.point_transform_count_max != 1 ||
      lidar.normal_transform_count_min != 1 ||
      lidar.normal_transform_count_max != 1 ||
      lidar.snapshot_sha256.empty() || lidar.geometry_sha256.empty()) {
    return FailFrame(std::move(audit), "LIDAR_INGEST",
                     JoinDetail("causal LiDAR ingestion failed", lidar.detail));
  }
  last_lidar_map_version_ = lidar.map_version_after;
  last_lidar_snapshot_sha256_ = lidar.snapshot_sha256;
  last_lidar_geometry_sha256_ = lidar.geometry_sha256;
  audit.lidar_map_version = last_lidar_map_version_;
  audit.max_scan_index = lidar.max_scan_index;
  if (!CheckGlobalBaCount(&error)) {
    return FailFrame(std::move(audit), "GLOBAL_BA_GUARD", error);
  }

  stage_start = Clock::now();
  const OnlineI3dgsDependencyResult frontend =
      dependencies_.frontend->IngestFrame(frame, frame.frame_index);
  audit.stage_milliseconds["frontend_ingest"] = ElapsedMilliseconds(stage_start);
  if (!frontend.success) {
    return FailFrame(std::move(audit), "FRONTEND_INGEST", frontend.detail);
  }
  if (!CheckGlobalBaCount(&error)) {
    return FailFrame(std::move(audit), "GLOBAL_BA_GUARD", error);
  }

  stage_start = Clock::now();
  const OnlineI3dgsKnownPoseRegistrationResult registration =
      dependencies_.state->RegisterKnownPose(frame);
  audit.stage_milliseconds["known_pose_registration"] =
      ElapsedMilliseconds(stage_start);
  if (!registration.success) {
    return FailFrame(std::move(audit), "KNOWN_POSE_REGISTRATION",
                     registration.detail);
  }
  const KnownPoseRecordQueryResult current_record =
      dependencies_.state->KnownPoses().GetByImageId(frame.image_id);
  if (!current_record.IsSuccess() ||
      current_record.record.frame_index != frame.frame_index ||
      current_record.record.registration_sequence !=
          registration.registration_sequence ||
      current_record.record.visual_state != KnownPoseVisualState::POSE_ONLY) {
    return FailFrame(std::move(audit), "KNOWN_POSE_REGISTRATION",
                     "registered pose is not immediately visible as POSE_ONLY");
  }
  audit.registration_sequence = registration.registration_sequence;
  ++summary_.arrived_event_count;
  ++summary_.known_pose_registered_count;
  if (!CheckGlobalBaCount(&error)) {
    return FailFrame(std::move(audit), "GLOBAL_BA_GUARD", error);
  }

  stage_start = Clock::now();
  const ConfirmedNormalizedSE3ReferenceSelector selector;
  const ConfirmedNormalizedSE3SelectionResult selection =
      selector.Select(dependencies_.state->KnownPoses(), frame.image_id);
  audit.stage_milliseconds["candidate_selection"] =
      ElapsedMilliseconds(stage_start);
  if (!selection.IsSuccess()) {
    return FailFrame(std::move(audit), "CANDIDATE_SELECTION", selection.detail);
  }
  audit.candidates = selection.references;
  for (size_t index = 0;
       index < std::min(kFirstMatchBatchSize, selection.references.size());
       ++index) {
    audit.first_batch_reference_image_ids.push_back(
        selection.references[index].image_id);
  }
  stage_start = Clock::now();
  if (!MatchBatch(frame, audit.first_batch_reference_image_ids,
                  &audit.matched_pairs, &error)) {
    return FailFrame(std::move(audit), "FIRST_MATCH_BATCH", error);
  }
  audit.initial_effective_active_reference_image_ids =
      EffectiveActiveReferences(audit.matched_pairs);
  if (!ComputeMaximumFiniteHop(
          audit.initial_effective_active_reference_image_ids,
          &audit.maximum_finite_hop_defined,
          &audit.maximum_finite_hop,
          &error)) {
    return FailFrame(std::move(audit), "MATCH_EXPANSION_GATE", error);
  }
  if (audit.maximum_finite_hop_defined &&
      audit.maximum_finite_hop > kDualHopThreshold) {
    for (size_t index = kFirstMatchBatchSize;
         index < std::min(kMaximumMatchedReferences,
                          selection.references.size());
         ++index) {
      audit.second_batch_reference_image_ids.push_back(
          selection.references[index].image_id);
    }
    if (!MatchBatch(frame, audit.second_batch_reference_image_ids,
                    &audit.matched_pairs, &error)) {
      return FailFrame(std::move(audit), "SECOND_MATCH_BATCH", error);
    }
    audit.matching_extended_to_ten =
        !audit.second_batch_reference_image_ids.empty();
  }
  audit.stage_milliseconds["matching"] = ElapsedMilliseconds(stage_start);

  std::vector<ActiveCovisibilityEdgeInput> accepted_current_edges;
  if (!UpdatePendingEdges(frame.image_id, frame.frame_index,
                          audit.matched_pairs, &accepted_current_edges,
                          &error)) {
    return FailFrame(std::move(audit), "PENDING_EVIDENCE", error);
  }
  audit.event_start_registry_version =
      dependencies_.state->KnownPoses().Version();
  audit.event_start_graph_version =
      dependencies_.state->ActiveGraph().Version();
  audit.event_start_canonical_version =
      dependencies_.state->CanonicalVersion();

  stage_start = Clock::now();
  size_t catchup_promotions = 0;
  if (!RunCatchups(frame.frame_index, frame.image_id, false, &audit.catchup,
                   &catchup_promotions, &error)) {
    return FailFrame(std::move(audit), "CATCHUP", error);
  }
  audit.stage_milliseconds["catchup"] = ElapsedMilliseconds(stage_start);
  static_cast<void>(catchup_promotions);
  audit.current_recomputed_registry_version =
      dependencies_.state->KnownPoses().Version();
  audit.current_recomputed_graph_version =
      dependencies_.state->ActiveGraph().Version();
  audit.current_recomputed_canonical_version =
      dependencies_.state->CanonicalVersion();
  audit.recomputed_effective_active_reference_image_ids =
      EffectiveActiveReferences(audit.matched_pairs);
  audit.current_recomputed_after_catchup = true;

  stage_start = Clock::now();
  if (!RunCurrent(frame, audit.matched_pairs, accepted_current_edges, &audit,
                  &error)) {
    return FailFrame(std::move(audit), "CURRENT", error);
  }
  audit.stage_milliseconds["current"] = ElapsedMilliseconds(stage_start);
  if (!CheckGlobalBaCount(&error)) {
    return FailFrame(std::move(audit), "GLOBAL_BA_GUARD", error);
  }
  audit.registry_version_after = dependencies_.state->KnownPoses().Version();
  audit.graph_version_after = dependencies_.state->ActiveGraph().Version();
  audit.canonical_version_after = dependencies_.state->CanonicalVersion();
  audit.global_ba_call_count = summary_.global_ba_call_count;
  const KnownPoseRecordQueryResult final_record =
      dependencies_.state->KnownPoses().GetByImageId(frame.image_id);
  if (!final_record.IsSuccess()) {
    return FailFrame(std::move(audit), "CURRENT_FINAL_STATE",
                     "current image disappeared from registry");
  }
  audit.current_visual_active =
      final_record.record.visual_state == KnownPoseVisualState::VISUAL_ACTIVE;
  const KnownPoseImageIdsResult active =
      dependencies_.state->KnownPoses().GetVisualActiveImageIds();
  if (!active.IsSuccess()) {
    return FailFrame(std::move(audit), "CURRENT_FINAL_STATE", active.detail);
  }
  summary_.visual_active_online_count = active.image_ids.size();
  summary_.last_consistent_stage = "ARRIVED_" +
                                   std::to_string(frame.frame_index) +
                                   "_COMMITTED";
  ++next_frame_index_;
  if (next_frame_index_ > options_.expected_frame_count) {
    status_ = OnlineI3dgsMapperStatus::READY_TO_FLUSH;
  }
  summary_.status = status_;
  audit.termination = audit.current_visual_active ? "VISUAL_ACTIVE"
                                                  : "POSE_ONLY";
  return {status_, std::move(audit)};
}

OnlineI3dgsFinishResult OnlineI3dgsMapper::Finish() {
  OnlineI3dgsFinishResult result;
  if (std::this_thread::get_id() != control_thread_id_) {
    FailRun("END_OF_SEQUENCE_FLUSH", "wrong controller thread");
  } else if (status_ != OnlineI3dgsMapperStatus::READY_TO_FLUSH ||
             next_frame_index_ != options_.expected_frame_count + 1 ||
             last_lidar_map_version_ == 0) {
    FailRun("END_OF_SEQUENCE_FLUSH",
            "flush requires every configured ARRIVED event");
  }
  if (status_ == OnlineI3dgsMapperStatus::INCOMPLETE) {
    result.status = status_;
    result.summary = summary_;
    return result;
  }

  for (size_t round = 1; round <= options_.expected_frame_count + 1; ++round) {
    std::vector<OnlineI3dgsCatchupAudit> audits;
    size_t promotions = 0;
    std::string error;
    if (!RunCatchups(options_.expected_frame_count, kInvalidImageId, true,
                     &audits, &promotions, &error)) {
      FailRun("END_OF_SEQUENCE_FLUSH", error);
      break;
    }
    for (OnlineI3dgsCatchupAudit& audit : audits) {
      OnlineI3dgsFlushAttemptAudit flush_audit;
      flush_audit.round = round;
      flush_audit.catchup = std::move(audit);
      result.flush_attempts.push_back(std::move(flush_audit));
    }
    if (promotions == 0) break;
    if (round == options_.expected_frame_count + 1) {
      FailRun("END_OF_SEQUENCE_FLUSH",
              "flush did not converge within bounded rounds");
    }
  }
  if (status_ != OnlineI3dgsMapperStatus::INCOMPLETE) {
    std::string error;
    if (!CheckGlobalBaCount(&error)) {
      FailRun("END_OF_SEQUENCE_FLUSH", error);
    }
  }
  if (status_ != OnlineI3dgsMapperStatus::INCOMPLETE) {
    const KnownPoseImageIdsResult registered =
        dependencies_.state->KnownPoses().GetRegisteredImageIds();
    const KnownPoseImageIdsResult active =
        dependencies_.state->KnownPoses().GetVisualActiveImageIds();
    const KnownPoseImageIdsResult pose_only =
        dependencies_.state->KnownPoses().GetPoseOnlyImageIds();
    if (!registered.IsSuccess() || !active.IsSuccess() ||
        !pose_only.IsSuccess() ||
        registered.image_ids.size() != options_.expected_frame_count) {
      FailRun("END_OF_SEQUENCE_FLUSH",
              "final registry cardinality is inconsistent");
    } else {
      summary_.known_pose_registered_count = registered.image_ids.size();
      summary_.visual_active_after_flush_count = active.image_ids.size();
      summary_.pose_only_final_count = pose_only.image_ids.size();
      summary_.last_consistent_stage = "END_OF_SEQUENCE_FLUSH_COMPLETE";
      status_ = OnlineI3dgsMapperStatus::COMPLETE;
      summary_.status = status_;
    }
  }
  result.status = status_;
  result.summary = summary_;
  return result;
}

}  // namespace colmap
