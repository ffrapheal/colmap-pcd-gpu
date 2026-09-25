#ifndef COLMAP_SRC_SFM_ACTIVE_COVISIBILITY_GRAPH_H_
#define COLMAP_SRC_SFM_ACTIVE_COVISIBILITY_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "util/types.h"

namespace colmap {

enum class ActiveCovisibilityStatus {
  SUCCESS,
  WRONG_THREAD,
  INVALID_IMAGE_ID,
  REQUIRES_VISUAL_ACTIVE,
  DUPLICATE_NODE,
  NODE_NOT_FOUND,
  MISSING_ENDPOINT,
  SELF_EDGE,
  ZERO_STRENGTH,
  DUPLICATE_EDGE,
  STALE_CANONICAL_VERSION,
  CANONICAL_VERSION_EXHAUSTED,
  EMPTY_COMMIT,
  EMPTY_QUERY,
  DUPLICATE_QUERY_NODE,
  INVALID_MAX_TOTAL,
  SEED_LIMIT_EXCEEDED,
  EMPTY_SOURCES,
  DUPLICATE_SOURCE,
  UNREACHABLE,
  INVALID_REGISTRATION_SEQUENCE,
  DUPLICATE_REGISTRATION_SEQUENCE,
  REGISTRATION_SEQUENCE_MISMATCH,
  DUPLICATE_PAIR_SUPPORT,
  STRENGTH_SUM_OVERFLOW,
  INVALID_PREPARED_COMMIT,
};

enum class ActiveCovisibilityNodeState {
  POSE_ONLY,
  VISUAL_ACTIVE,
};

enum class ActiveCovisibilityEdgeKind {
  ORDINARY,
  LOOP,
};

struct ActiveCovisibilityResult {
  ActiveCovisibilityStatus status = ActiveCovisibilityStatus::SUCCESS;
  std::string detail;
  uint64_t canonical_version = 0;

  bool IsSuccess() const {
    return status == ActiveCovisibilityStatus::SUCCESS;
  }
};

struct ActiveCovisibilityNodeInput {
  ActiveCovisibilityNodeInput() = default;
  ActiveCovisibilityNodeInput(image_t image_id,
                              size_t registration_sequence,
                              ActiveCovisibilityNodeState state);

  image_t image_id = kInvalidImageId;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  ActiveCovisibilityNodeState state =
      ActiveCovisibilityNodeState::POSE_ONLY;
};

struct ActiveCovisibilityNode {
  ActiveCovisibilityNode() = default;
  ActiveCovisibilityNode(image_t image_id, size_t registration_sequence);

  image_t image_id = kInvalidImageId;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
};

struct ActiveCovisibilityEdgeInput {
  ActiveCovisibilityEdgeInput() = default;
  ActiveCovisibilityEdgeInput(image_t image_id1,
                              image_t image_id2,
                              size_t strength,
                              size_t birth_frame = 0);

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  size_t strength = 0;
  size_t birth_frame = 0;
};

struct ActiveCovisibilityEdge {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  size_t strength = 0;
  size_t birth_frame = 0;
  ActiveCovisibilityEdgeKind kind = ActiveCovisibilityEdgeKind::ORDINARY;

  double Cost() const;
};

struct ActiveCovisibilityCommit {
  uint64_t expected_version = 0;
  std::vector<ActiveCovisibilityNodeInput> nodes;
  std::vector<ActiveCovisibilityEdgeInput> ordinary_edges;
  std::vector<ActiveCovisibilityEdgeInput> loop_edges;
};

// Tentative nodes must not already be canonical. Edges may join any canonical
// or declared tentative endpoints, but may not duplicate each other or a
// canonical edge, including across the ordinary and loop lists. Query methods
// consume this value without modifying either it or the canonical graph.
struct ActiveCovisibilityOverlay {
  uint64_t base_version = 0;
  std::vector<ActiveCovisibilityNode> temporary_nodes;
  std::vector<ActiveCovisibilityEdgeInput> ordinary_edges;
  std::vector<ActiveCovisibilityEdgeInput> loop_edges;
};

struct ActiveCovisibilitySnapshot {
  uint64_t version = 0;
  std::vector<ActiveCovisibilityNode> nodes;
  std::vector<image_t> image_ids;
  std::vector<ActiveCovisibilityEdge> edges;
};

class ActiveCovisibilityGraph;

class PreparedActiveCovisibilityCommit {
 public:
  PreparedActiveCovisibilityCommit() = default;
  PreparedActiveCovisibilityCommit(
      PreparedActiveCovisibilityCommit&& other);
  PreparedActiveCovisibilityCommit& operator=(
      PreparedActiveCovisibilityCommit&& other);

  PreparedActiveCovisibilityCommit(
      const PreparedActiveCovisibilityCommit&) = delete;
  PreparedActiveCovisibilityCommit& operator=(
      const PreparedActiveCovisibilityCommit&) = delete;

  bool IsPrepared() const noexcept;

 private:
  friend class ActiveCovisibilityGraph;

  using EdgeKey = std::pair<image_t, image_t>;

  void Reset() noexcept;

  const ActiveCovisibilityGraph* graph_ = nullptr;
  std::thread::id owner_thread_id_;
  uint64_t expected_version_ = 0;
  uint64_t committed_version_ = 0;
  bool prepared_ = false;
  std::map<image_t, ActiveCovisibilityNode> nodes_;
  std::map<size_t, image_t> registration_sequences_;
  std::map<EdgeKey, ActiveCovisibilityEdge> edges_;
};

enum class HopReachability {
  FINITE,
  UNREACHABLE,
};

struct HopDistance {
  HopReachability reachability = HopReachability::UNREACHABLE;
  size_t hops = std::numeric_limits<size_t>::max();

  bool IsFinite() const { return reachability == HopReachability::FINITE; }
};

struct HopDistancesResult : ActiveCovisibilityResult {
  image_t source_image_id = kInvalidImageId;
  std::map<image_t, HopDistance> distances;
};

struct HopDistanceMatrixResult : ActiveCovisibilityResult {
  std::vector<image_t> image_ids;
  std::vector<std::vector<HopDistance>> distances;
};

enum class WindowExpansionStopReason {
  MAX_TOTAL_REACHED,
  NO_ADJACENT_ACTIVE_NODE,
};

struct WindowSelection {
  image_t image_id = kInvalidImageId;
  uint64_t support_strength = 0;
  bool is_seed = false;
};

struct WindowExpansionResult : ActiveCovisibilityResult {
  size_t max_total = 0;
  size_t unfilled_slots = 0;
  size_t candidate_evaluations = 0;
  size_t adjacency_visits = 0;
  WindowExpansionStopReason stop_reason =
      WindowExpansionStopReason::NO_ADJACENT_ACTIVE_NODE;
  std::vector<WindowSelection> selection_order;

  bool ReachedMaxTotal() const {
    return IsSuccess() &&
           stop_reason == WindowExpansionStopReason::MAX_TOTAL_REACHED;
  }
};

struct ShortestBackboneResult : ActiveCovisibilityResult {
  bool reachable = false;
  double total_cost = std::numeric_limits<double>::infinity();
  size_t hops = std::numeric_limits<size_t>::max();
  std::vector<image_t> path;
  std::vector<double> edge_costs;
  size_t queue_pops = 0;
  size_t edge_relaxations = 0;
};

struct MultiSourceOwnership {
  image_t node_image_id = kInvalidImageId;
  bool reachable = false;
  image_t source_image_id = kInvalidImageId;
  double distance = std::numeric_limits<double>::infinity();
};

struct MultiSourceOwnershipResult : ActiveCovisibilityResult {
  std::vector<image_t> source_image_ids;
  std::vector<MultiSourceOwnership> ownership;
  size_t queue_pops = 0;
  size_t edge_relaxations = 0;
};

struct OrdinaryPairSupport {
  OrdinaryPairSupport() = default;
  OrdinaryPairSupport(image_t reference_image_id,
                      size_t reference_registration_sequence,
                      size_t verified_inliers);

  image_t reference_image_id = kInvalidImageId;
  size_t reference_registration_sequence =
      std::numeric_limits<size_t>::max();
  size_t verified_inliers = 0;
};

struct OrdinaryPairDecision {
  OrdinaryPairSupport support;
  bool is_robust = false;
  bool is_strongest_positive = false;
  bool accepted = false;
};

struct OrdinaryPairAcceptanceResult : ActiveCovisibilityResult {
  image_t current_image_id = kInvalidImageId;
  size_t strongest_positive_support = 0;
  std::vector<image_t> strongest_positive_reference_image_ids;
  // Set only when the strongest-positive set has exactly one member.
  image_t strongest_positive_reference_image_id = kInvalidImageId;
  std::vector<OrdinaryPairDecision> decisions;
  std::vector<ActiveCovisibilityEdgeInput> accepted_edges;
};

// This helper is intentionally ordinary-edge-only. Loop acceptance is a
// separate controller decision and cannot be requested through this API.
OrdinaryPairAcceptanceResult SelectOrdinaryPairEdges(
    image_t current_image_id,
    size_t edge_birth_frame,
    const std::vector<OrdinaryPairSupport>& pair_supports);

class ActiveCovisibilityGraph {
 public:
  static constexpr size_t kMaxWindowSize = 20;
  static constexpr size_t kRobustOrdinaryMinStrength = 15;

  ActiveCovisibilityGraph();

  ActiveCovisibilityGraph(const ActiveCovisibilityGraph&) = delete;
  ActiveCovisibilityGraph& operator=(const ActiveCovisibilityGraph&) = delete;
  ActiveCovisibilityGraph(ActiveCovisibilityGraph&&) = delete;
  ActiveCovisibilityGraph& operator=(ActiveCovisibilityGraph&&) = delete;

  uint64_t Version() const;
  bool IsOwnerThread() const noexcept;
  size_t NumNodes() const;
  size_t NumEdges() const;
  bool HasNode(image_t image_id) const;
  bool HasEdge(image_t image_id1, image_t image_id2) const;

  ActiveCovisibilitySnapshot Snapshot() const;
  ActiveCovisibilityOverlay CreateTentativeOverlay() const;

  ActiveCovisibilityResult AddNode(image_t image_id,
                                   size_t registration_sequence,
                                   ActiveCovisibilityNodeState state);
  ActiveCovisibilityResult AddEdge(
      const ActiveCovisibilityEdgeInput& edge,
      ActiveCovisibilityEdgeKind kind =
          ActiveCovisibilityEdgeKind::ORDINARY);

  ActiveCovisibilityResult ValidateCommit(
      const ActiveCovisibilityCommit& commit) const;
  ActiveCovisibilityResult PrepareCommit(
      const ActiveCovisibilityCommit& commit,
      PreparedActiveCovisibilityCommit* prepared) const;
  ActiveCovisibilityResult ValidatePreparedCommit(
      const PreparedActiveCovisibilityCommit& prepared) const;
  ActiveCovisibilityResult Commit(const ActiveCovisibilityCommit& commit);

  // Precondition: ValidatePreparedCommit succeeded and the owner thread has not
  // touched the graph afterwards. This path only swaps prepared maps.
  void CommitPrepared(
      PreparedActiveCovisibilityCommit* prepared) noexcept;

  ActiveCovisibilityResult ValidateTentativeOverlay(
      const ActiveCovisibilityOverlay& overlay) const;

  // Commits every overlay edge in one all-or-nothing canonical transaction.
  // Every temporary endpoint used by an edge must be explicitly repeated in
  // nodes_to_add with VISUAL_ACTIVE state.
  ActiveCovisibilityResult CommitTentativeOverlay(
      const ActiveCovisibilityOverlay& overlay,
      const std::vector<ActiveCovisibilityNodeInput>& nodes_to_add);

  HopDistancesResult HopDistances(
      image_t source_image_id,
      const ActiveCovisibilityOverlay* overlay = nullptr) const;
  // The returned row/column IDs are the sorted unique requested IDs. Paths
  // may traverse other nodes in the canonical-plus-overlay query view.
  HopDistanceMatrixResult HopDistanceMatrix(
      const std::vector<image_t>& image_ids,
      const ActiveCovisibilityOverlay* overlay = nullptr) const;

  // Seeds may include temporary overlay nodes. Expansion candidates are only
  // canonical VISUAL_ACTIVE nodes; both overlay edge lists contribute support.
  // Seeds are emitted first in image_id order, followed by greedy selections.
  WindowExpansionResult ExpandWindow(
      const std::vector<image_t>& seed_image_ids,
      size_t max_total,
      const ActiveCovisibilityOverlay* overlay = nullptr) const;

  // Backbone and ownership intentionally operate on canonical state only.
  // Exact rational costs resolve backbone ties by the full image_id path, and
  // ownership ties by distance, source image_id and node image_id.
  ShortestBackboneResult ShortestBackbone(image_t source_image_id,
                                          image_t target_image_id) const;
  MultiSourceOwnershipResult ComputeMultiSourceOwnership(
      const std::vector<image_t>& source_image_ids) const;

 private:
  using EdgeKey = std::pair<image_t, image_t>;

  static EdgeKey MakeEdgeKey(image_t image_id1, image_t image_id2);
  ActiveCovisibilityResult ValidateCommitAgainst(
      const ActiveCovisibilityCommit& commit,
      const std::map<image_t, ActiveCovisibilityNode>& nodes,
      const std::map<size_t, image_t>& registration_sequences,
      const std::map<EdgeKey, ActiveCovisibilityEdge>& edges) const;

  uint64_t version_ = 0;
  const std::thread::id owner_thread_id_;
  std::map<image_t, ActiveCovisibilityNode> nodes_;
  std::map<size_t, image_t> registration_sequences_;
  std::map<EdgeKey, ActiveCovisibilityEdge> edges_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_ACTIVE_COVISIBILITY_GRAPH_H_
