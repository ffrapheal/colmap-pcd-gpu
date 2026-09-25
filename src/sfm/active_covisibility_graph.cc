#include "sfm/active_covisibility_graph.h"

#include <algorithm>
#include <deque>
#include <queue>
#include <tuple>

#include <boost/multiprecision/cpp_int.hpp>

namespace colmap {
namespace {

using EdgeKey = std::pair<image_t, image_t>;
using boost::multiprecision::cpp_int;

cpp_int GreatestCommonDivisor(cpp_int lhs, cpp_int rhs) {
  while (rhs != 0) {
    const cpp_int remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

struct QueryView {
  std::set<image_t> nodes;
  std::set<image_t> active_nodes;
  std::map<image_t, size_t> registration_sequences;
  std::map<image_t, std::map<image_t, size_t>> adjacency;
};

struct ExactCost {
  cpp_int numerator = 0;
  cpp_int denominator = 1;
  long double value = 0.0L;

  void AddInverse(const size_t strength) {
    const cpp_int exact_strength = strength;
    numerator = numerator * exact_strength + denominator;
    denominator *= exact_strength;
    const cpp_int common_divisor =
        GreatestCommonDivisor(numerator, denominator);
    numerator /= common_divisor;
    denominator /= common_divisor;
    value += 1.0L / static_cast<long double>(strength);
  }
};

int CompareCost(const ExactCost& lhs, const ExactCost& rhs) {
  const cpp_int lhs_scaled = lhs.numerator * rhs.denominator;
  const cpp_int rhs_scaled = rhs.numerator * lhs.denominator;
  if (lhs_scaled < rhs_scaled) {
    return -1;
  }
  if (lhs_scaled > rhs_scaled) {
    return 1;
  }
  return 0;
}

struct PathLabel {
  ExactCost cost;
  std::vector<image_t> path;
};

struct PathLabelGreater {
  bool operator()(const PathLabel& lhs, const PathLabel& rhs) const {
    const int cost_order = CompareCost(lhs.cost, rhs.cost);
    if (cost_order != 0) {
      return cost_order > 0;
    }
    return lhs.path > rhs.path;
  }
};

struct OwnershipLabel {
  ExactCost distance;
  image_t source_image_id = kInvalidImageId;
  image_t node_image_id = kInvalidImageId;
};

struct OwnershipLabelGreater {
  bool operator()(const OwnershipLabel& lhs,
                  const OwnershipLabel& rhs) const {
    const int distance_order = CompareCost(lhs.distance, rhs.distance);
    if (distance_order != 0) {
      return distance_order > 0;
    }
    if (lhs.source_image_id != rhs.source_image_id) {
      return lhs.source_image_id > rhs.source_image_id;
    }
    return lhs.node_image_id > rhs.node_image_id;
  }
};

template <typename Result>
void SetResultStatus(Result* result,
                     const ActiveCovisibilityStatus status,
                     const std::string& detail,
                     const uint64_t version) {
  result->status = status;
  result->detail = detail;
  result->canonical_version = version;
}

ActiveCovisibilityResult MakeResult(const ActiveCovisibilityStatus status,
                                    const std::string& detail,
                                    const uint64_t version) {
  ActiveCovisibilityResult result;
  SetResultStatus(&result, status, detail, version);
  return result;
}

bool IsValidImageId(const image_t image_id) {
  return image_id != kInvalidImageId;
}

bool IsValidRegistrationSequence(const size_t registration_sequence) {
  return registration_sequence > 0 &&
         registration_sequence != std::numeric_limits<size_t>::max();
}

EdgeKey NormalizeEdgeKey(const image_t image_id1, const image_t image_id2) {
  return image_id1 < image_id2 ? EdgeKey(image_id1, image_id2)
                               : EdgeKey(image_id2, image_id1);
}

void AddViewEdge(const ActiveCovisibilityEdgeInput& edge, QueryView* view) {
  view->adjacency[edge.image_id1].emplace(edge.image_id2, edge.strength);
  view->adjacency[edge.image_id2].emplace(edge.image_id1, edge.strength);
}

QueryView BuildQueryView(const ActiveCovisibilitySnapshot& snapshot,
                         const ActiveCovisibilityOverlay* overlay) {
  QueryView view;
  for (const ActiveCovisibilityNode& node : snapshot.nodes) {
    view.nodes.insert(node.image_id);
    view.active_nodes.insert(node.image_id);
    view.registration_sequences.emplace(node.image_id,
                                        node.registration_sequence);
    view.adjacency.emplace(node.image_id, std::map<image_t, size_t>());
  }
  for (const ActiveCovisibilityEdge& edge : snapshot.edges) {
    AddViewEdge(ActiveCovisibilityEdgeInput(edge.image_id1,
                                           edge.image_id2,
                                           edge.strength,
                                           edge.birth_frame),
                &view);
  }

  if (overlay != nullptr) {
    for (const ActiveCovisibilityNode& node : overlay->temporary_nodes) {
      view.nodes.insert(node.image_id);
      view.registration_sequences.emplace(node.image_id,
                                          node.registration_sequence);
      view.adjacency.emplace(node.image_id, std::map<image_t, size_t>());
    }
    for (const ActiveCovisibilityEdgeInput& edge : overlay->ordinary_edges) {
      AddViewEdge(edge, &view);
    }
    for (const ActiveCovisibilityEdgeInput& edge : overlay->loop_edges) {
      AddViewEdge(edge, &view);
    }
  }
  return view;
}

std::map<image_t, HopDistance> ComputeHopDistances(
    const image_t source_image_id, const QueryView& view) {
  std::map<image_t, HopDistance> distances;
  for (const image_t image_id : view.nodes) {
    distances.emplace(image_id, HopDistance());
  }

  distances.at(source_image_id).reachability = HopReachability::FINITE;
  distances.at(source_image_id).hops = 0;
  std::deque<image_t> queue;
  queue.push_back(source_image_id);

  while (!queue.empty()) {
    const image_t image_id = queue.front();
    queue.pop_front();
    const size_t next_hops = distances.at(image_id).hops + 1;
    for (const auto& neighbor : view.adjacency.at(image_id)) {
      HopDistance& distance = distances.at(neighbor.first);
      if (distance.IsFinite()) {
        continue;
      }
      distance.reachability = HopReachability::FINITE;
      distance.hops = next_hops;
      queue.push_back(neighbor.first);
    }
  }
  return distances;
}

bool IsBetterPath(const PathLabel& candidate, const PathLabel& current) {
  const int cost_order = CompareCost(candidate.cost, current.cost);
  if (cost_order != 0) {
    return cost_order < 0;
  }
  return candidate.path < current.path;
}

bool IsSamePath(const PathLabel& lhs, const PathLabel& rhs) {
  return CompareCost(lhs.cost, rhs.cost) == 0 && lhs.path == rhs.path;
}

bool IsBetterOwner(const ExactCost& candidate_distance,
                   const image_t candidate_source,
                   const OwnershipLabel& current) {
  const int distance_order = CompareCost(candidate_distance, current.distance);
  return distance_order < 0 ||
         (distance_order == 0 &&
          candidate_source < current.source_image_id);
}

}  // namespace

constexpr size_t ActiveCovisibilityGraph::kMaxWindowSize;
constexpr size_t ActiveCovisibilityGraph::kRobustOrdinaryMinStrength;

PreparedActiveCovisibilityCommit::PreparedActiveCovisibilityCommit(
    PreparedActiveCovisibilityCommit&& other) {
  *this = std::move(other);
}

PreparedActiveCovisibilityCommit&
PreparedActiveCovisibilityCommit::operator=(
    PreparedActiveCovisibilityCommit&& other) {
  if (this != &other) {
    Reset();
    graph_ = other.graph_;
    owner_thread_id_ = other.owner_thread_id_;
    expected_version_ = other.expected_version_;
    committed_version_ = other.committed_version_;
    prepared_ = other.prepared_;
    nodes_ = std::move(other.nodes_);
    registration_sequences_ = std::move(other.registration_sequences_);
    edges_ = std::move(other.edges_);
    other.Reset();
  }
  return *this;
}

bool PreparedActiveCovisibilityCommit::IsPrepared() const noexcept {
  return prepared_;
}

void PreparedActiveCovisibilityCommit::Reset() noexcept {
  graph_ = nullptr;
  owner_thread_id_ = std::thread::id();
  expected_version_ = 0;
  committed_version_ = 0;
  prepared_ = false;
  nodes_.clear();
  registration_sequences_.clear();
  edges_.clear();
}

ActiveCovisibilityGraph::ActiveCovisibilityGraph()
    : owner_thread_id_(std::this_thread::get_id()) {}

ActiveCovisibilityNodeInput::ActiveCovisibilityNodeInput(
    const image_t image_id,
    const size_t registration_sequence,
    const ActiveCovisibilityNodeState state)
    : image_id(image_id),
      registration_sequence(registration_sequence),
      state(state) {}

ActiveCovisibilityNode::ActiveCovisibilityNode(
    const image_t image_id, const size_t registration_sequence)
    : image_id(image_id), registration_sequence(registration_sequence) {}

ActiveCovisibilityEdgeInput::ActiveCovisibilityEdgeInput(
    const image_t image_id1,
    const image_t image_id2,
    const size_t strength,
    const size_t birth_frame)
    : image_id1(image_id1),
      image_id2(image_id2),
      strength(strength),
      birth_frame(birth_frame) {}

double ActiveCovisibilityEdge::Cost() const {
  if (strength == 0) {
    return std::numeric_limits<double>::infinity();
  }
  return 1.0 / static_cast<double>(strength);
}

OrdinaryPairSupport::OrdinaryPairSupport(
    const image_t reference_image_id,
    const size_t reference_registration_sequence,
    const size_t verified_inliers)
    : reference_image_id(reference_image_id),
      reference_registration_sequence(reference_registration_sequence),
      verified_inliers(verified_inliers) {}

OrdinaryPairAcceptanceResult SelectOrdinaryPairEdges(
    const image_t current_image_id,
    const size_t edge_birth_frame,
    const std::vector<OrdinaryPairSupport>& pair_supports) {
  OrdinaryPairAcceptanceResult result;
  result.current_image_id = current_image_id;
  if (!IsValidImageId(current_image_id)) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                    "current image_id is invalid",
                    0);
    return result;
  }

  std::set<image_t> reference_image_ids;
  for (const OrdinaryPairSupport& support : pair_supports) {
    if (!IsValidImageId(support.reference_image_id)) {
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                      "reference image_id is invalid",
                      0);
      return result;
    }
    if (support.reference_image_id == current_image_id) {
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::SELF_EDGE,
                      "ordinary pair cannot reference the current image",
                      0);
      return result;
    }
    if (!IsValidRegistrationSequence(
            support.reference_registration_sequence)) {
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE,
                      "reference registration_sequence must be one-based",
                      0);
      return result;
    }
    if (!reference_image_ids.insert(support.reference_image_id).second) {
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::DUPLICATE_PAIR_SUPPORT,
                      "reference image appears more than once in the batch",
                      0);
      return result;
    }
  }

  std::vector<OrdinaryPairSupport> sorted_supports = pair_supports;
  std::sort(sorted_supports.begin(),
            sorted_supports.end(),
            [](const OrdinaryPairSupport& lhs,
               const OrdinaryPairSupport& rhs) {
              return std::tie(lhs.reference_registration_sequence,
                              lhs.reference_image_id) <
                     std::tie(rhs.reference_registration_sequence,
                              rhs.reference_image_id);
            });

  size_t strongest_positive_support = 0;
  for (const OrdinaryPairSupport& support : sorted_supports) {
    strongest_positive_support =
        std::max(strongest_positive_support, support.verified_inliers);
  }
  result.strongest_positive_support = strongest_positive_support;

  for (const OrdinaryPairSupport& support : sorted_supports) {
    OrdinaryPairDecision decision;
    decision.support = support;
    decision.is_robust =
        support.verified_inliers >=
        ActiveCovisibilityGraph::kRobustOrdinaryMinStrength;
    decision.is_strongest_positive =
        strongest_positive_support > 0 &&
        support.verified_inliers == strongest_positive_support;
    decision.accepted =
        support.verified_inliers > 0 &&
        (decision.is_robust || decision.is_strongest_positive);
    result.decisions.push_back(decision);
    if (decision.is_strongest_positive) {
      result.strongest_positive_reference_image_ids.push_back(
          support.reference_image_id);
    }
    if (decision.accepted) {
      result.accepted_edges.emplace_back(current_image_id,
                                         support.reference_image_id,
                                         support.verified_inliers,
                                         edge_birth_frame);
    }
  }
  if (result.strongest_positive_reference_image_ids.size() == 1) {
    result.strongest_positive_reference_image_id =
        result.strongest_positive_reference_image_ids.front();
  }

  SetResultStatus(
      &result, ActiveCovisibilityStatus::SUCCESS, std::string(), 0);
  return result;
}

uint64_t ActiveCovisibilityGraph::Version() const { return version_; }

bool ActiveCovisibilityGraph::IsOwnerThread() const noexcept {
  return std::this_thread::get_id() == owner_thread_id_;
}

size_t ActiveCovisibilityGraph::NumNodes() const { return nodes_.size(); }

size_t ActiveCovisibilityGraph::NumEdges() const { return edges_.size(); }

bool ActiveCovisibilityGraph::HasNode(const image_t image_id) const {
  return nodes_.count(image_id) != 0;
}

bool ActiveCovisibilityGraph::HasEdge(const image_t image_id1,
                                      const image_t image_id2) const {
  if (image_id1 == image_id2) {
    return false;
  }
  return edges_.count(MakeEdgeKey(image_id1, image_id2)) != 0;
}

ActiveCovisibilitySnapshot ActiveCovisibilityGraph::Snapshot() const {
  ActiveCovisibilitySnapshot snapshot;
  snapshot.version = version_;
  snapshot.nodes.reserve(nodes_.size());
  snapshot.image_ids.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    snapshot.nodes.push_back(node.second);
    snapshot.image_ids.push_back(node.first);
  }
  snapshot.edges.reserve(edges_.size());
  for (const auto& edge : edges_) {
    snapshot.edges.push_back(edge.second);
  }
  return snapshot;
}

ActiveCovisibilityOverlay
ActiveCovisibilityGraph::CreateTentativeOverlay() const {
  ActiveCovisibilityOverlay overlay;
  overlay.base_version = version_;
  return overlay;
}

ActiveCovisibilityResult ActiveCovisibilityGraph::AddNode(
    const image_t image_id,
    const size_t registration_sequence,
    const ActiveCovisibilityNodeState state) {
  ActiveCovisibilityCommit commit;
  commit.expected_version = version_;
  commit.nodes.emplace_back(image_id, registration_sequence, state);
  return Commit(commit);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::AddEdge(
    const ActiveCovisibilityEdgeInput& edge,
    const ActiveCovisibilityEdgeKind kind) {
  ActiveCovisibilityCommit commit;
  commit.expected_version = version_;
  if (kind == ActiveCovisibilityEdgeKind::ORDINARY) {
    commit.ordinary_edges.push_back(edge);
  } else {
    commit.loop_edges.push_back(edge);
  }
  return Commit(commit);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::ValidateCommit(
    const ActiveCovisibilityCommit& commit) const {
  if (!IsOwnerThread()) {
    return MakeResult(ActiveCovisibilityStatus::WRONG_THREAD,
                      "commit validation must run on the graph owner thread",
                      0);
  }
  return ValidateCommitAgainst(
      commit, nodes_, registration_sequences_, edges_);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::PrepareCommit(
    const ActiveCovisibilityCommit& commit,
    PreparedActiveCovisibilityCommit* prepared) const {
  if (prepared == nullptr) {
    return MakeResult(ActiveCovisibilityStatus::INVALID_PREPARED_COMMIT,
                      "prepared commit output is null",
                      version_);
  }
  prepared->Reset();
  const ActiveCovisibilityResult validation = ValidateCommit(commit);
  if (!validation.IsSuccess()) {
    return validation;
  }

  PreparedActiveCovisibilityCommit next;
  next.graph_ = this;
  next.owner_thread_id_ = owner_thread_id_;
  next.expected_version_ = commit.expected_version;
  next.committed_version_ = commit.expected_version + 1;
  next.nodes_ = nodes_;
  next.registration_sequences_ = registration_sequences_;
  next.edges_ = edges_;
  for (const ActiveCovisibilityNodeInput& node : commit.nodes) {
    next.nodes_.emplace(
        node.image_id,
        ActiveCovisibilityNode(node.image_id, node.registration_sequence));
    next.registration_sequences_.emplace(node.registration_sequence,
                                         node.image_id);
  }

  const auto add_edges = [&next](
                             const std::vector<ActiveCovisibilityEdgeInput>&
                                 edge_inputs,
                             const ActiveCovisibilityEdgeKind kind) {
    for (const ActiveCovisibilityEdgeInput& input : edge_inputs) {
      const EdgeKey key = NormalizeEdgeKey(input.image_id1, input.image_id2);
      ActiveCovisibilityEdge edge;
      edge.image_id1 = key.first;
      edge.image_id2 = key.second;
      edge.strength = input.strength;
      edge.birth_frame = input.birth_frame;
      edge.kind = kind;
      next.edges_.emplace(key, edge);
    }
  };
  add_edges(commit.ordinary_edges, ActiveCovisibilityEdgeKind::ORDINARY);
  add_edges(commit.loop_edges, ActiveCovisibilityEdgeKind::LOOP);

  next.prepared_ = true;
  *prepared = std::move(next);
  return validation;
}

ActiveCovisibilityResult ActiveCovisibilityGraph::ValidatePreparedCommit(
    const PreparedActiveCovisibilityCommit& prepared) const {
  if (!IsOwnerThread() ||
      std::this_thread::get_id() != prepared.owner_thread_id_) {
    return MakeResult(ActiveCovisibilityStatus::WRONG_THREAD,
                      "prepared commit must remain on the graph owner thread",
                      0);
  }
  if (!prepared.prepared_ || prepared.graph_ != this ||
      prepared.committed_version_ != prepared.expected_version_ + 1) {
    return MakeResult(ActiveCovisibilityStatus::INVALID_PREPARED_COMMIT,
                      "prepared commit does not belong to this graph",
                      version_);
  }
  if (prepared.expected_version_ != version_) {
    return MakeResult(ActiveCovisibilityStatus::STALE_CANONICAL_VERSION,
                      "prepared commit expected_version is stale",
                      version_);
  }
  return MakeResult(
      ActiveCovisibilityStatus::SUCCESS, std::string(), version_);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::Commit(
    const ActiveCovisibilityCommit& commit) {
  PreparedActiveCovisibilityCommit prepared;
  ActiveCovisibilityResult result = PrepareCommit(commit, &prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  result = ValidatePreparedCommit(prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  CommitPrepared(&prepared);
  return MakeResult(
      ActiveCovisibilityStatus::SUCCESS, std::string(), version_);
}

void ActiveCovisibilityGraph::CommitPrepared(
    PreparedActiveCovisibilityCommit* prepared) noexcept {
  nodes_.swap(prepared->nodes_);
  registration_sequences_.swap(prepared->registration_sequences_);
  edges_.swap(prepared->edges_);
  version_ = prepared->committed_version_;
  prepared->graph_ = nullptr;
  prepared->owner_thread_id_ = std::thread::id();
  prepared->expected_version_ = 0;
  prepared->committed_version_ = 0;
  prepared->prepared_ = false;
}

ActiveCovisibilityResult ActiveCovisibilityGraph::ValidateTentativeOverlay(
    const ActiveCovisibilityOverlay& overlay) const {
  if (!IsOwnerThread()) {
    return MakeResult(ActiveCovisibilityStatus::WRONG_THREAD,
                      "overlay validation must run on the graph owner thread",
                      0);
  }
  if (overlay.base_version != version_) {
    return MakeResult(ActiveCovisibilityStatus::STALE_CANONICAL_VERSION,
                      "overlay base_version does not match canonical state",
                      version_);
  }

  std::set<image_t> all_nodes;
  for (const auto& node : nodes_) {
    all_nodes.insert(node.first);
  }
  std::map<size_t, image_t> all_registration_sequences =
      registration_sequences_;
  for (const ActiveCovisibilityNode& node : overlay.temporary_nodes) {
    if (!IsValidImageId(node.image_id)) {
      return MakeResult(ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                        "overlay temporary image_id is invalid",
                        version_);
    }
    if (!IsValidRegistrationSequence(node.registration_sequence)) {
      return MakeResult(
          ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE,
          "overlay registration_sequence must be one-based",
          version_);
    }
    if (!all_nodes.insert(node.image_id).second) {
      return MakeResult(ActiveCovisibilityStatus::DUPLICATE_NODE,
                        "overlay node duplicates canonical or tentative node",
                        version_);
    }
    if (!all_registration_sequences
             .emplace(node.registration_sequence, node.image_id)
             .second) {
      return MakeResult(
          ActiveCovisibilityStatus::DUPLICATE_REGISTRATION_SEQUENCE,
          "overlay registration_sequence duplicates canonical or tentative node",
          version_);
    }
  }

  std::set<EdgeKey> all_edges;
  for (const auto& edge : edges_) {
    all_edges.insert(edge.first);
  }
  const auto validate_edges =
      [this, &all_nodes, &all_edges](
          const std::vector<ActiveCovisibilityEdgeInput>& edge_inputs)
      -> ActiveCovisibilityResult {
    for (const ActiveCovisibilityEdgeInput& edge : edge_inputs) {
      if (!IsValidImageId(edge.image_id1) ||
          !IsValidImageId(edge.image_id2)) {
        return MakeResult(ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                          "overlay edge endpoint image_id is invalid",
                          version_);
      }
      if (edge.image_id1 == edge.image_id2) {
        return MakeResult(ActiveCovisibilityStatus::SELF_EDGE,
                          "overlay self-edge is not allowed",
                          version_);
      }
      if (edge.strength == 0) {
        return MakeResult(ActiveCovisibilityStatus::ZERO_STRENGTH,
                          "overlay edge strength must be positive",
                          version_);
      }
      if (all_nodes.count(edge.image_id1) == 0 ||
          all_nodes.count(edge.image_id2) == 0) {
        return MakeResult(ActiveCovisibilityStatus::MISSING_ENDPOINT,
                          "overlay edge endpoint is not declared",
                          version_);
      }
      if (!all_edges.insert(MakeEdgeKey(edge.image_id1, edge.image_id2))
               .second) {
        return MakeResult(
            ActiveCovisibilityStatus::DUPLICATE_EDGE,
            "overlay edge duplicates canonical, ordinary, or loop edge",
            version_);
      }
    }
    return MakeResult(
        ActiveCovisibilityStatus::SUCCESS, std::string(), version_);
  };

  ActiveCovisibilityResult validation =
      validate_edges(overlay.ordinary_edges);
  if (!validation.IsSuccess()) {
    return validation;
  }
  return validate_edges(overlay.loop_edges);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::CommitTentativeOverlay(
    const ActiveCovisibilityOverlay& overlay,
    const std::vector<ActiveCovisibilityNodeInput>& nodes_to_add) {
  const ActiveCovisibilityResult overlay_validation =
      ValidateTentativeOverlay(overlay);
  if (!overlay_validation.IsSuccess()) {
    return overlay_validation;
  }

  std::map<image_t, size_t> temporary_nodes;
  for (const ActiveCovisibilityNode& node : overlay.temporary_nodes) {
    temporary_nodes.emplace(node.image_id, node.registration_sequence);
  }
  for (const ActiveCovisibilityNodeInput& node : nodes_to_add) {
    const auto temporary_node = temporary_nodes.find(node.image_id);
    if (temporary_node == temporary_nodes.end()) {
      return MakeResult(
          ActiveCovisibilityStatus::NODE_NOT_FOUND,
          "overlay commit node was not declared as temporary",
          version_);
    }
    if (node.registration_sequence != temporary_node->second) {
      return MakeResult(
          ActiveCovisibilityStatus::REGISTRATION_SEQUENCE_MISMATCH,
          "overlay commit registration_sequence differs from its declaration",
          version_);
    }
  }

  ActiveCovisibilityCommit commit;
  commit.expected_version = overlay.base_version;
  commit.nodes = nodes_to_add;
  commit.ordinary_edges = overlay.ordinary_edges;
  commit.loop_edges = overlay.loop_edges;
  return Commit(commit);
}

HopDistancesResult ActiveCovisibilityGraph::HopDistances(
    const image_t source_image_id,
    const ActiveCovisibilityOverlay* overlay) const {
  HopDistancesResult result;
  result.source_image_id = source_image_id;
  if (!IsOwnerThread()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::WRONG_THREAD,
                    "hop query must run on the graph owner thread",
                    0);
    return result;
  }
  if (overlay != nullptr) {
    const ActiveCovisibilityResult validation =
        ValidateTentativeOverlay(*overlay);
    if (!validation.IsSuccess()) {
      SetResultStatus(&result,
                      validation.status,
                      validation.detail,
                      validation.canonical_version);
      return result;
    }
  }

  const ActiveCovisibilitySnapshot snapshot = Snapshot();
  const QueryView view = BuildQueryView(snapshot, overlay);
  if (view.nodes.count(source_image_id) == 0) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::NODE_NOT_FOUND,
                    "hop source is not in the query view",
                    snapshot.version);
    return result;
  }

  result.distances = ComputeHopDistances(source_image_id, view);
  SetResultStatus(&result,
                  ActiveCovisibilityStatus::SUCCESS,
                  std::string(),
                  snapshot.version);
  return result;
}

HopDistanceMatrixResult ActiveCovisibilityGraph::HopDistanceMatrix(
    const std::vector<image_t>& image_ids,
    const ActiveCovisibilityOverlay* overlay) const {
  HopDistanceMatrixResult result;
  if (!IsOwnerThread()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::WRONG_THREAD,
                    "hop matrix query must run on the graph owner thread",
                    0);
    return result;
  }
  if (image_ids.empty()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::EMPTY_QUERY,
                    "hop matrix image set must not be empty",
                    version_);
    return result;
  }
  if (overlay != nullptr) {
    const ActiveCovisibilityResult validation =
        ValidateTentativeOverlay(*overlay);
    if (!validation.IsSuccess()) {
      SetResultStatus(&result,
                      validation.status,
                      validation.detail,
                      validation.canonical_version);
      return result;
    }
  }

  const ActiveCovisibilitySnapshot snapshot = Snapshot();
  const QueryView view = BuildQueryView(snapshot, overlay);
  result.image_ids = image_ids;
  std::sort(result.image_ids.begin(), result.image_ids.end());
  if (std::adjacent_find(result.image_ids.begin(), result.image_ids.end()) !=
      result.image_ids.end()) {
    result.image_ids.clear();
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::DUPLICATE_QUERY_NODE,
                    "hop matrix image set contains a duplicate",
                    snapshot.version);
    return result;
  }
  for (const image_t image_id : result.image_ids) {
    if (view.nodes.count(image_id) == 0) {
      result.image_ids.clear();
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::NODE_NOT_FOUND,
                      "hop matrix image is not in the query view",
                      snapshot.version);
      return result;
    }
  }

  result.distances.reserve(result.image_ids.size());
  for (const image_t source_image_id : result.image_ids) {
    const std::map<image_t, HopDistance> distances =
        ComputeHopDistances(source_image_id, view);
    std::vector<HopDistance> row;
    row.reserve(result.image_ids.size());
    for (const image_t target_image_id : result.image_ids) {
      row.push_back(distances.at(target_image_id));
    }
    result.distances.push_back(row);
  }
  SetResultStatus(&result,
                  ActiveCovisibilityStatus::SUCCESS,
                  std::string(),
                  snapshot.version);
  return result;
}

WindowExpansionResult ActiveCovisibilityGraph::ExpandWindow(
    const std::vector<image_t>& seed_image_ids,
    const size_t max_total,
    const ActiveCovisibilityOverlay* overlay) const {
  WindowExpansionResult result;
  result.max_total = max_total;
  if (!IsOwnerThread()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::WRONG_THREAD,
                    "window expansion must run on the graph owner thread",
                    0);
    return result;
  }
  if (max_total == 0 || max_total > kMaxWindowSize) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::INVALID_MAX_TOTAL,
                    "max_total must be in [1, 20]",
                    version_);
    return result;
  }
  if (seed_image_ids.empty()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::EMPTY_QUERY,
                    "window seed set must not be empty",
                    version_);
    return result;
  }
  if (seed_image_ids.size() > max_total) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::SEED_LIMIT_EXCEEDED,
                    "window seed count exceeds max_total",
                    version_);
    return result;
  }
  if (overlay != nullptr) {
    const ActiveCovisibilityResult validation =
        ValidateTentativeOverlay(*overlay);
    if (!validation.IsSuccess()) {
      SetResultStatus(&result,
                      validation.status,
                      validation.detail,
                      validation.canonical_version);
      return result;
    }
  }

  const ActiveCovisibilitySnapshot snapshot = Snapshot();
  const QueryView view = BuildQueryView(snapshot, overlay);
  std::vector<image_t> sorted_seeds = seed_image_ids;
  std::sort(sorted_seeds.begin(), sorted_seeds.end());
  if (std::adjacent_find(sorted_seeds.begin(), sorted_seeds.end()) !=
      sorted_seeds.end()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::DUPLICATE_QUERY_NODE,
                    "window seed set contains a duplicate",
                    snapshot.version);
    return result;
  }
  for (const image_t image_id : sorted_seeds) {
    if (view.nodes.count(image_id) == 0) {
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::NODE_NOT_FOUND,
                      "window seed is not in the query view",
                      snapshot.version);
      return result;
    }
  }

  std::set<image_t> selected(sorted_seeds.begin(), sorted_seeds.end());
  for (const image_t image_id : sorted_seeds) {
    WindowSelection selection;
    selection.image_id = image_id;
    selection.is_seed = true;
    result.selection_order.push_back(selection);
  }

  while (result.selection_order.size() < max_total) {
    image_t best_image_id = kInvalidImageId;
    uint64_t best_support = 0;
    for (const image_t candidate_image_id : view.active_nodes) {
      if (selected.count(candidate_image_id) != 0) {
        continue;
      }
      ++result.candidate_evaluations;
      uint64_t support = 0;
      for (const auto& neighbor : view.adjacency.at(candidate_image_id)) {
        ++result.adjacency_visits;
        if (selected.count(neighbor.first) == 0) {
          continue;
        }
        if (std::numeric_limits<uint64_t>::max() - support <
            neighbor.second) {
          result.selection_order.clear();
          SetResultStatus(&result,
                          ActiveCovisibilityStatus::STRENGTH_SUM_OVERFLOW,
                          "window support strength sum overflowed",
                          snapshot.version);
          return result;
        }
        support += static_cast<uint64_t>(neighbor.second);
      }
      if (support > best_support ||
          (support == best_support && support > 0 &&
           std::tie(view.registration_sequences.at(candidate_image_id),
                    candidate_image_id) <
               std::tie(view.registration_sequences.at(best_image_id),
                        best_image_id))) {
        best_image_id = candidate_image_id;
        best_support = support;
      }
    }

    if (best_image_id == kInvalidImageId) {
      result.stop_reason =
          WindowExpansionStopReason::NO_ADJACENT_ACTIVE_NODE;
      break;
    }
    selected.insert(best_image_id);
    WindowSelection selection;
    selection.image_id = best_image_id;
    selection.support_strength = best_support;
    result.selection_order.push_back(selection);
  }

  if (result.selection_order.size() == max_total) {
    result.stop_reason = WindowExpansionStopReason::MAX_TOTAL_REACHED;
  }
  result.unfilled_slots = max_total - result.selection_order.size();
  SetResultStatus(&result,
                  ActiveCovisibilityStatus::SUCCESS,
                  std::string(),
                  snapshot.version);
  return result;
}

ShortestBackboneResult ActiveCovisibilityGraph::ShortestBackbone(
    const image_t source_image_id, const image_t target_image_id) const {
  ShortestBackboneResult result;
  if (!IsOwnerThread()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::WRONG_THREAD,
                    "backbone query must run on the graph owner thread",
                    0);
    return result;
  }
  const ActiveCovisibilitySnapshot snapshot = Snapshot();
  const QueryView view = BuildQueryView(snapshot, nullptr);
  if (view.nodes.count(source_image_id) == 0 ||
      view.nodes.count(target_image_id) == 0) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::NODE_NOT_FOUND,
                    "backbone endpoint is not canonical",
                    snapshot.version);
    return result;
  }

  std::map<image_t, PathLabel> best_paths;
  std::priority_queue<PathLabel,
                      std::vector<PathLabel>,
                      PathLabelGreater>
      queue;
  PathLabel source;
  source.path.push_back(source_image_id);
  best_paths.emplace(source_image_id, source);
  queue.push(source);

  while (!queue.empty()) {
    const PathLabel current = queue.top();
    queue.pop();
    ++result.queue_pops;
    const image_t current_image_id = current.path.back();
    const auto current_best = best_paths.find(current_image_id);
    if (current_best == best_paths.end() ||
        !IsSamePath(current, current_best->second)) {
      continue;
    }

    for (const auto& neighbor : view.adjacency.at(current_image_id)) {
      ++result.edge_relaxations;
      PathLabel candidate = current;
      candidate.cost.AddInverse(neighbor.second);
      candidate.path.push_back(neighbor.first);
      const auto existing = best_paths.find(neighbor.first);
      if (existing == best_paths.end() ||
          IsBetterPath(candidate, existing->second)) {
        best_paths[neighbor.first] = candidate;
        queue.push(candidate);
      }
    }
  }

  const auto target = best_paths.find(target_image_id);
  if (target == best_paths.end()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::UNREACHABLE,
                    "no canonical backbone connects the endpoints",
                    snapshot.version);
    return result;
  }

  result.reachable = true;
  result.total_cost = static_cast<double>(target->second.cost.value);
  result.path = target->second.path;
  result.hops = result.path.size() - 1;
  result.edge_costs.reserve(result.hops);
  for (size_t index = 1; index < result.path.size(); ++index) {
    const size_t strength =
        view.adjacency.at(result.path[index - 1]).at(result.path[index]);
    result.edge_costs.push_back(1.0 / static_cast<double>(strength));
  }
  SetResultStatus(&result,
                  ActiveCovisibilityStatus::SUCCESS,
                  std::string(),
                  snapshot.version);
  return result;
}

MultiSourceOwnershipResult
ActiveCovisibilityGraph::ComputeMultiSourceOwnership(
    const std::vector<image_t>& source_image_ids) const {
  MultiSourceOwnershipResult result;
  if (!IsOwnerThread()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::WRONG_THREAD,
                    "ownership query must run on the graph owner thread",
                    0);
    return result;
  }
  const ActiveCovisibilitySnapshot snapshot = Snapshot();
  if (source_image_ids.empty()) {
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::EMPTY_SOURCES,
                    "ownership source set must not be empty",
                    snapshot.version);
    return result;
  }

  result.source_image_ids = source_image_ids;
  std::sort(result.source_image_ids.begin(), result.source_image_ids.end());
  if (std::adjacent_find(result.source_image_ids.begin(),
                         result.source_image_ids.end()) !=
      result.source_image_ids.end()) {
    result.source_image_ids.clear();
    SetResultStatus(&result,
                    ActiveCovisibilityStatus::DUPLICATE_SOURCE,
                    "ownership source set contains a duplicate",
                    snapshot.version);
    return result;
  }
  for (const image_t source_image_id : result.source_image_ids) {
    if (nodes_.count(source_image_id) == 0) {
      result.source_image_ids.clear();
      SetResultStatus(&result,
                      ActiveCovisibilityStatus::NODE_NOT_FOUND,
                      "ownership source is not canonical",
                      snapshot.version);
      return result;
    }
  }

  const QueryView view = BuildQueryView(snapshot, nullptr);
  std::map<image_t, OwnershipLabel> best_owners;
  std::priority_queue<OwnershipLabel,
                      std::vector<OwnershipLabel>,
                      OwnershipLabelGreater>
      queue;
  for (const image_t source_image_id : result.source_image_ids) {
    OwnershipLabel source;
    source.source_image_id = source_image_id;
    source.node_image_id = source_image_id;
    best_owners.emplace(source_image_id, source);
    queue.push(source);
  }

  while (!queue.empty()) {
    const OwnershipLabel current = queue.top();
    queue.pop();
    ++result.queue_pops;
    const auto current_best = best_owners.find(current.node_image_id);
    if (current_best == best_owners.end() ||
        CompareCost(current.distance, current_best->second.distance) != 0 ||
        current.source_image_id != current_best->second.source_image_id) {
      continue;
    }

    for (const auto& neighbor : view.adjacency.at(current.node_image_id)) {
      ++result.edge_relaxations;
      ExactCost candidate_distance = current.distance;
      candidate_distance.AddInverse(neighbor.second);
      const auto existing = best_owners.find(neighbor.first);
      if (existing == best_owners.end() ||
          IsBetterOwner(candidate_distance,
                        current.source_image_id,
                        existing->second)) {
        OwnershipLabel candidate;
        candidate.distance = candidate_distance;
        candidate.source_image_id = current.source_image_id;
        candidate.node_image_id = neighbor.first;
        best_owners[neighbor.first] = candidate;
        queue.push(candidate);
      }
    }
  }

  result.ownership.reserve(snapshot.image_ids.size());
  for (const image_t node_image_id : snapshot.image_ids) {
    MultiSourceOwnership ownership;
    ownership.node_image_id = node_image_id;
    const auto best = best_owners.find(node_image_id);
    if (best != best_owners.end()) {
      ownership.reachable = true;
      ownership.source_image_id = best->second.source_image_id;
      ownership.distance = static_cast<double>(best->second.distance.value);
    }
    result.ownership.push_back(ownership);
  }
  SetResultStatus(&result,
                  ActiveCovisibilityStatus::SUCCESS,
                  std::string(),
                  snapshot.version);
  return result;
}

ActiveCovisibilityGraph::EdgeKey ActiveCovisibilityGraph::MakeEdgeKey(
    const image_t image_id1, const image_t image_id2) {
  return NormalizeEdgeKey(image_id1, image_id2);
}

ActiveCovisibilityResult ActiveCovisibilityGraph::ValidateCommitAgainst(
    const ActiveCovisibilityCommit& commit,
    const std::map<image_t, ActiveCovisibilityNode>& nodes,
    const std::map<size_t, image_t>& registration_sequences,
    const std::map<EdgeKey, ActiveCovisibilityEdge>& edges) const {
  if (commit.expected_version != version_) {
    return MakeResult(ActiveCovisibilityStatus::STALE_CANONICAL_VERSION,
                      "commit expected_version does not match canonical state",
                      version_);
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return MakeResult(
        ActiveCovisibilityStatus::CANONICAL_VERSION_EXHAUSTED,
        "canonical version cannot be incremented",
        version_);
  }
  if (commit.nodes.empty() && commit.ordinary_edges.empty() &&
      commit.loop_edges.empty()) {
    return MakeResult(ActiveCovisibilityStatus::EMPTY_COMMIT,
                      "canonical commit has no writes",
                      version_);
  }

  std::set<image_t> all_nodes;
  for (const auto& node : nodes) {
    all_nodes.insert(node.first);
  }
  std::map<size_t, image_t> all_registration_sequences =
      registration_sequences;
  for (const ActiveCovisibilityNodeInput& node : commit.nodes) {
    if (!IsValidImageId(node.image_id)) {
      return MakeResult(ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                        "canonical node image_id is invalid",
                        version_);
    }
    if (!IsValidRegistrationSequence(node.registration_sequence)) {
      return MakeResult(
          ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE,
          "canonical registration_sequence must be one-based",
          version_);
    }
    if (node.state != ActiveCovisibilityNodeState::VISUAL_ACTIVE) {
      return MakeResult(ActiveCovisibilityStatus::REQUIRES_VISUAL_ACTIVE,
                        "canonical nodes must be VISUAL_ACTIVE",
                        version_);
    }
    if (!all_nodes.insert(node.image_id).second) {
      return MakeResult(ActiveCovisibilityStatus::DUPLICATE_NODE,
                        "canonical node already exists or repeats in commit",
                        version_);
    }
    if (!all_registration_sequences
             .emplace(node.registration_sequence, node.image_id)
             .second) {
      return MakeResult(
          ActiveCovisibilityStatus::DUPLICATE_REGISTRATION_SEQUENCE,
          "canonical registration_sequence already exists or repeats in commit",
          version_);
    }
  }

  std::set<EdgeKey> all_edges;
  for (const auto& edge : edges) {
    all_edges.insert(edge.first);
  }
  const auto validate_edges =
      [this, &all_nodes, &all_edges](
          const std::vector<ActiveCovisibilityEdgeInput>& edge_inputs)
      -> ActiveCovisibilityResult {
    for (const ActiveCovisibilityEdgeInput& edge : edge_inputs) {
      if (!IsValidImageId(edge.image_id1) ||
          !IsValidImageId(edge.image_id2)) {
        return MakeResult(ActiveCovisibilityStatus::INVALID_IMAGE_ID,
                          "canonical edge endpoint image_id is invalid",
                          version_);
      }
      if (edge.image_id1 == edge.image_id2) {
        return MakeResult(ActiveCovisibilityStatus::SELF_EDGE,
                          "canonical self-edge is not allowed",
                          version_);
      }
      if (edge.strength == 0) {
        return MakeResult(ActiveCovisibilityStatus::ZERO_STRENGTH,
                          "canonical edge strength must be positive",
                          version_);
      }
      if (all_nodes.count(edge.image_id1) == 0 ||
          all_nodes.count(edge.image_id2) == 0) {
        return MakeResult(ActiveCovisibilityStatus::MISSING_ENDPOINT,
                          "canonical edge endpoint is not active",
                          version_);
      }
      if (!all_edges.insert(MakeEdgeKey(edge.image_id1, edge.image_id2))
               .second) {
        return MakeResult(
            ActiveCovisibilityStatus::DUPLICATE_EDGE,
            "canonical edge duplicates existing or pending edge",
            version_);
      }
    }
    return MakeResult(
        ActiveCovisibilityStatus::SUCCESS, std::string(), version_);
  };

  ActiveCovisibilityResult validation =
      validate_edges(commit.ordinary_edges);
  if (!validation.IsSuccess()) {
    return validation;
  }
  return validate_edges(commit.loop_edges);
}

}  // namespace colmap
