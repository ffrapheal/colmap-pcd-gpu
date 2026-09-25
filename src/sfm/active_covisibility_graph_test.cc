#define TEST_NAME "sfm/active_covisibility_graph"
#include "util/testing.h"

#include "sfm/active_covisibility_graph.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using namespace colmap;

namespace {

void AddActiveNodes(ActiveCovisibilityGraph* graph,
                    const std::initializer_list<image_t> image_ids) {
  for (const image_t image_id : image_ids) {
    BOOST_REQUIRE(
        graph
            ->AddNode(image_id,
                      static_cast<size_t>(image_id),
                      ActiveCovisibilityNodeState::VISUAL_ACTIVE)
            .IsSuccess());
  }
}

void AddActiveRange(ActiveCovisibilityGraph* graph,
                    const image_t first,
                    const image_t last) {
  for (image_t image_id = first; image_id <= last; ++image_id) {
    BOOST_REQUIRE(
        graph
            ->AddNode(image_id,
                      static_cast<size_t>(image_id),
                      ActiveCovisibilityNodeState::VISUAL_ACTIVE)
            .IsSuccess());
  }
}

void AddEdge(ActiveCovisibilityGraph* graph,
             const image_t image_id1,
             const image_t image_id2,
             const size_t strength,
             const ActiveCovisibilityEdgeKind kind =
                 ActiveCovisibilityEdgeKind::ORDINARY) {
  BOOST_REQUIRE(
      graph
          ->AddEdge(ActiveCovisibilityEdgeInput(
                        image_id1, image_id2, strength, 1),
                    kind)
          .IsSuccess());
}

std::vector<image_t> SelectionImageIds(
    const WindowExpansionResult& result) {
  std::vector<image_t> image_ids;
  image_ids.reserve(result.selection_order.size());
  for (const WindowSelection& selection : result.selection_order) {
    image_ids.push_back(selection.image_id);
  }
  return image_ids;
}

std::vector<image_t> DecisionReferenceImageIds(
    const OrdinaryPairAcceptanceResult& result) {
  std::vector<image_t> image_ids;
  image_ids.reserve(result.decisions.size());
  for (const OrdinaryPairDecision& decision : result.decisions) {
    image_ids.push_back(decision.support.reference_image_id);
  }
  return image_ids;
}

std::vector<image_t> AcceptedReferenceImageIds(
    const OrdinaryPairAcceptanceResult& result) {
  std::vector<image_t> image_ids;
  image_ids.reserve(result.accepted_edges.size());
  for (const ActiveCovisibilityEdgeInput& edge : result.accepted_edges) {
    image_ids.push_back(edge.image_id2);
  }
  return image_ids;
}

const ActiveCovisibilityEdge* FindEdge(
    const ActiveCovisibilitySnapshot& snapshot,
    const image_t image_id1,
    const image_t image_id2) {
  const image_t smaller = std::min(image_id1, image_id2);
  const image_t larger = std::max(image_id1, image_id2);
  for (const ActiveCovisibilityEdge& edge : snapshot.edges) {
    if (edge.image_id1 == smaller && edge.image_id2 == larger) {
      return &edge;
    }
  }
  return nullptr;
}

const ActiveCovisibilityNode* FindNode(
    const ActiveCovisibilitySnapshot& snapshot, const image_t image_id) {
  for (const ActiveCovisibilityNode& node : snapshot.nodes) {
    if (node.image_id == image_id) {
      return &node;
    }
  }
  return nullptr;
}

const OrdinaryPairDecision* FindDecision(
    const OrdinaryPairAcceptanceResult& result,
    const image_t reference_image_id) {
  for (const OrdinaryPairDecision& decision : result.decisions) {
    if (decision.support.reference_image_id == reference_image_id) {
      return &decision;
    }
  }
  return nullptr;
}

const MultiSourceOwnership* FindOwnership(
    const MultiSourceOwnershipResult& result, const image_t node_image_id) {
  for (const MultiSourceOwnership& ownership : result.ownership) {
    if (ownership.node_image_id == node_image_id) {
      return &ownership;
    }
  }
  return nullptr;
}

}  // namespace

static_assert(
    noexcept(std::declval<ActiveCovisibilityGraph&>().CommitPrepared(
        std::declval<PreparedActiveCovisibilityCommit*>())),
    "prepared graph publication must be noexcept");

BOOST_AUTO_TEST_CASE(PreparedCommitPublishesWithoutRevalidationFailure) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2});

  ActiveCovisibilityCommit commit;
  commit.expected_version = graph.Version();
  commit.nodes.emplace_back(
      3, 3, ActiveCovisibilityNodeState::VISUAL_ACTIVE);
  commit.ordinary_edges.emplace_back(1, 3, 7, 11);
  PreparedActiveCovisibilityCommit prepared;
  BOOST_REQUIRE(graph.PrepareCommit(commit, &prepared).IsSuccess());
  BOOST_REQUIRE(prepared.IsPrepared());
  BOOST_REQUIRE(graph.ValidatePreparedCommit(prepared).IsSuccess());
  graph.CommitPrepared(&prepared);

  BOOST_CHECK(!prepared.IsPrepared());
  BOOST_CHECK_EQUAL(graph.Version(), commit.expected_version + 1);
  BOOST_CHECK(graph.HasNode(3));
  BOOST_CHECK(graph.HasEdge(1, 3));
}

BOOST_AUTO_TEST_CASE(PreparedCommitDetectsDriftAndWrongThread) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2});

  ActiveCovisibilityCommit commit;
  commit.expected_version = graph.Version();
  commit.nodes.emplace_back(
      3, 3, ActiveCovisibilityNodeState::VISUAL_ACTIVE);
  PreparedActiveCovisibilityCommit prepared;
  BOOST_REQUIRE(graph.PrepareCommit(commit, &prepared).IsSuccess());

  ActiveCovisibilityResult wrong_thread;
  std::thread worker(
      [&]() { wrong_thread = graph.ValidatePreparedCommit(prepared); });
  worker.join();
  BOOST_CHECK(wrong_thread.status == ActiveCovisibilityStatus::WRONG_THREAD);
  BOOST_CHECK_EQUAL(graph.Version(), commit.expected_version);

  BOOST_REQUIRE(
      graph.AddNode(4, 4, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  BOOST_CHECK(graph.ValidatePreparedCommit(prepared).status ==
              ActiveCovisibilityStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(!graph.HasNode(3));
}

BOOST_AUTO_TEST_CASE(CanonicalRejectsInvalidStructureWithoutPartialMutation) {
  ActiveCovisibilityGraph graph;

  const ActiveCovisibilityResult pose_only = graph.AddNode(
      7, 1, ActiveCovisibilityNodeState::POSE_ONLY);
  BOOST_CHECK(pose_only.status ==
              ActiveCovisibilityStatus::REQUIRES_VISUAL_ACTIVE);
  BOOST_CHECK_EQUAL(graph.Version(), 0);
  BOOST_CHECK_EQUAL(graph.NumNodes(), 0);

  BOOST_REQUIRE(
      graph.AddNode(7, 1, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  BOOST_CHECK_EQUAL(graph.Version(), 1);
  BOOST_CHECK(
      graph.AddNode(7, 2, ActiveCovisibilityNodeState::VISUAL_ACTIVE).status ==
      ActiveCovisibilityStatus::DUPLICATE_NODE);
  BOOST_CHECK(
      graph.AddNode(kInvalidImageId,
                    2,
                    ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .status == ActiveCovisibilityStatus::INVALID_IMAGE_ID);
  BOOST_CHECK(
      graph.AddNode(9, 0, ActiveCovisibilityNodeState::VISUAL_ACTIVE).status ==
      ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE);
  BOOST_CHECK(
      graph
          .AddNode(9,
                   std::numeric_limits<size_t>::max(),
                   ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .status == ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE);
  BOOST_CHECK_EQUAL(graph.Version(), 1);

  BOOST_REQUIRE(
      graph.AddNode(8, 2, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  BOOST_CHECK(
      graph.AddNode(9, 2, ActiveCovisibilityNodeState::VISUAL_ACTIVE).status ==
      ActiveCovisibilityStatus::DUPLICATE_REGISTRATION_SEQUENCE);
  BOOST_CHECK(graph.AddEdge(ActiveCovisibilityEdgeInput(7, 9, 4)).status ==
              ActiveCovisibilityStatus::MISSING_ENDPOINT);
  BOOST_CHECK(graph.AddEdge(ActiveCovisibilityEdgeInput(7, 7, 4)).status ==
              ActiveCovisibilityStatus::SELF_EDGE);
  BOOST_CHECK(graph.AddEdge(ActiveCovisibilityEdgeInput(7, 8, 0)).status ==
              ActiveCovisibilityStatus::ZERO_STRENGTH);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
  BOOST_CHECK_EQUAL(graph.Version(), 2);

  BOOST_REQUIRE(
      graph.AddEdge(ActiveCovisibilityEdgeInput(8, 7, 5, 12)).IsSuccess());
  BOOST_CHECK(
      graph
          .AddEdge(ActiveCovisibilityEdgeInput(7, 8, 9, 13),
                   ActiveCovisibilityEdgeKind::LOOP)
          .status == ActiveCovisibilityStatus::DUPLICATE_EDGE);
  const ActiveCovisibilitySnapshot snapshot = graph.Snapshot();
  BOOST_REQUIRE_EQUAL(snapshot.edges.size(), 1);
  BOOST_CHECK_EQUAL(snapshot.edges.front().image_id1, 7);
  BOOST_CHECK_EQUAL(snapshot.edges.front().image_id2, 8);
  BOOST_CHECK_EQUAL(snapshot.edges.front().strength, 5);
  BOOST_CHECK_EQUAL(snapshot.edges.front().birth_frame, 12);
  BOOST_CHECK(snapshot.edges.front().kind ==
              ActiveCovisibilityEdgeKind::ORDINARY);
  BOOST_CHECK_SMALL(std::abs(snapshot.edges.front().Cost() - 0.2), 1e-15);
  BOOST_REQUIRE(FindNode(snapshot, 7) != nullptr);
  BOOST_REQUIRE(FindNode(snapshot, 8) != nullptr);
  BOOST_CHECK_EQUAL(FindNode(snapshot, 7)->registration_sequence, 1);
  BOOST_CHECK_EQUAL(FindNode(snapshot, 8)->registration_sequence, 2);

  const uint64_t version_before_failed_commit = graph.Version();
  ActiveCovisibilityCommit commit;
  commit.expected_version = version_before_failed_commit;
  commit.nodes.emplace_back(
      9, 3, ActiveCovisibilityNodeState::VISUAL_ACTIVE);
  commit.ordinary_edges.emplace_back(8, 9, 4, 20);
  commit.loop_edges.emplace_back(9, 99, 3, 20);
  const ActiveCovisibilityResult failed_commit = graph.Commit(commit);
  BOOST_CHECK(failed_commit.status ==
              ActiveCovisibilityStatus::MISSING_ENDPOINT);
  BOOST_CHECK_EQUAL(graph.Version(), version_before_failed_commit);
  BOOST_CHECK_EQUAL(graph.NumNodes(), 2);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 1);
  BOOST_CHECK(!graph.HasNode(9));
  BOOST_CHECK(!graph.HasEdge(8, 9));
}

BOOST_AUTO_TEST_CASE(TentativeOverlayIsValidatedReadOnlyAndAtomicallyCommitted) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2});
  AddEdge(&graph, 1, 2, 5);

  const ActiveCovisibilitySnapshot canonical_before = graph.Snapshot();
  ActiveCovisibilityOverlay overlay = graph.CreateTentativeOverlay();
  overlay.temporary_nodes.emplace_back(3, 3);
  overlay.ordinary_edges.emplace_back(3, 1, 9, 30);
  overlay.loop_edges.emplace_back(2, 3, 7, 30);
  BOOST_REQUIRE(graph.ValidateTentativeOverlay(overlay).IsSuccess());

  const HopDistancesResult overlay_hops = graph.HopDistances(3, &overlay);
  BOOST_REQUIRE(overlay_hops.IsSuccess());
  BOOST_CHECK_EQUAL(overlay_hops.distances.at(1).hops, 1);
  BOOST_CHECK_EQUAL(overlay_hops.distances.at(2).hops, 1);

  const WindowExpansionResult overlay_window =
      graph.ExpandWindow({3}, 3, &overlay);
  BOOST_REQUIRE(overlay_window.IsSuccess());
  const std::vector<image_t> expected_window = {3, 1, 2};
  BOOST_CHECK(SelectionImageIds(overlay_window) == expected_window);
  BOOST_CHECK_EQUAL(overlay_window.selection_order[1].support_strength, 9);
  BOOST_CHECK_EQUAL(overlay_window.selection_order[2].support_strength, 12);

  ActiveCovisibilityOverlay extra_temporary = overlay;
  extra_temporary.temporary_nodes.emplace_back(4, 4);
  extra_temporary.ordinary_edges.emplace_back(3, 4, 100, 30);
  const WindowExpansionResult active_only_window =
      graph.ExpandWindow({3}, 3, &extra_temporary);
  BOOST_REQUIRE(active_only_window.IsSuccess());
  BOOST_CHECK(SelectionImageIds(active_only_window) == expected_window);

  const ActiveCovisibilitySnapshot canonical_after_queries = graph.Snapshot();
  BOOST_CHECK_EQUAL(canonical_after_queries.version, canonical_before.version);
  BOOST_CHECK(canonical_after_queries.image_ids == canonical_before.image_ids);
  BOOST_CHECK_EQUAL(canonical_after_queries.edges.size(),
                    canonical_before.edges.size());
  BOOST_CHECK(!graph.HasNode(3));
  BOOST_CHECK(!graph.HasEdge(1, 3));

  ActiveCovisibilityOverlay duplicate_node = overlay;
  duplicate_node.temporary_nodes.emplace_back(1, 5);
  BOOST_CHECK(graph.ValidateTentativeOverlay(duplicate_node).status ==
              ActiveCovisibilityStatus::DUPLICATE_NODE);

  ActiveCovisibilityOverlay canonical_sequence_conflict = overlay;
  canonical_sequence_conflict.temporary_nodes.emplace_back(5, 1);
  BOOST_CHECK(
      graph.ValidateTentativeOverlay(canonical_sequence_conflict).status ==
      ActiveCovisibilityStatus::DUPLICATE_REGISTRATION_SEQUENCE);

  ActiveCovisibilityOverlay tentative_sequence_conflict = overlay;
  tentative_sequence_conflict.temporary_nodes.emplace_back(5, 3);
  BOOST_CHECK(
      graph.ValidateTentativeOverlay(tentative_sequence_conflict).status ==
      ActiveCovisibilityStatus::DUPLICATE_REGISTRATION_SEQUENCE);

  ActiveCovisibilityOverlay invalid_sequence =
      graph.CreateTentativeOverlay();
  invalid_sequence.temporary_nodes.emplace_back(5, 0);
  BOOST_CHECK(graph.ValidateTentativeOverlay(invalid_sequence).status ==
              ActiveCovisibilityStatus::INVALID_REGISTRATION_SEQUENCE);

  ActiveCovisibilityOverlay missing_endpoint =
      graph.CreateTentativeOverlay();
  missing_endpoint.temporary_nodes.emplace_back(4, 4);
  missing_endpoint.ordinary_edges.emplace_back(4, 99, 1);
  BOOST_CHECK(graph.ValidateTentativeOverlay(missing_endpoint).status ==
              ActiveCovisibilityStatus::MISSING_ENDPOINT);

  ActiveCovisibilityOverlay duplicate_edge = graph.CreateTentativeOverlay();
  duplicate_edge.temporary_nodes.emplace_back(4, 4);
  duplicate_edge.ordinary_edges.emplace_back(4, 1, 2);
  duplicate_edge.loop_edges.emplace_back(1, 4, 3);
  BOOST_CHECK(graph.ValidateTentativeOverlay(duplicate_edge).status ==
              ActiveCovisibilityStatus::DUPLICATE_EDGE);

  ActiveCovisibilityOverlay duplicate_canonical =
      graph.CreateTentativeOverlay();
  duplicate_canonical.loop_edges.emplace_back(2, 1, 8);
  BOOST_CHECK(graph.ValidateTentativeOverlay(duplicate_canonical).status ==
              ActiveCovisibilityStatus::DUPLICATE_EDGE);

  const ActiveCovisibilityResult sequence_mismatch =
      graph.CommitTentativeOverlay(
          overlay,
          {ActiveCovisibilityNodeInput(
              3, 4, ActiveCovisibilityNodeState::VISUAL_ACTIVE)});
  BOOST_CHECK(sequence_mismatch.status ==
              ActiveCovisibilityStatus::REGISTRATION_SEQUENCE_MISMATCH);
  BOOST_CHECK_EQUAL(graph.Version(), canonical_before.version);
  BOOST_CHECK(!graph.HasNode(3));
  BOOST_CHECK(!graph.HasEdge(1, 3));

  const ActiveCovisibilityResult inactive_commit =
      graph.CommitTentativeOverlay(
          overlay,
          {ActiveCovisibilityNodeInput(
              3, 3, ActiveCovisibilityNodeState::POSE_ONLY)});
  BOOST_CHECK(inactive_commit.status ==
              ActiveCovisibilityStatus::REQUIRES_VISUAL_ACTIVE);
  BOOST_CHECK_EQUAL(graph.Version(), canonical_before.version);
  BOOST_CHECK(!graph.HasNode(3));

  const ActiveCovisibilityResult committed =
      graph.CommitTentativeOverlay(
          overlay,
          {ActiveCovisibilityNodeInput(
              3, 3, ActiveCovisibilityNodeState::VISUAL_ACTIVE)});
  BOOST_REQUIRE(committed.IsSuccess());
  BOOST_CHECK_EQUAL(graph.Version(), canonical_before.version + 1);
  BOOST_CHECK(graph.HasNode(3));
  BOOST_CHECK(graph.HasEdge(1, 3));
  BOOST_CHECK(graph.HasEdge(2, 3));
  const ActiveCovisibilitySnapshot committed_snapshot = graph.Snapshot();
  BOOST_REQUIRE(FindNode(committed_snapshot, 3) != nullptr);
  BOOST_CHECK_EQUAL(FindNode(committed_snapshot, 3)->registration_sequence, 3);
  BOOST_REQUIRE(FindEdge(committed_snapshot, 1, 3) != nullptr);
  BOOST_REQUIRE(FindEdge(committed_snapshot, 2, 3) != nullptr);
  BOOST_CHECK(FindEdge(committed_snapshot, 1, 3)->kind ==
              ActiveCovisibilityEdgeKind::ORDINARY);
  BOOST_CHECK(FindEdge(committed_snapshot, 2, 3)->kind ==
              ActiveCovisibilityEdgeKind::LOOP);
  BOOST_CHECK(graph.ValidateTentativeOverlay(overlay).status ==
              ActiveCovisibilityStatus::STALE_CANONICAL_VERSION);
}

BOOST_AUTO_TEST_CASE(HopQueriesDistinguishFiniteAndUnreachable) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2, 3, 4, 5});
  AddEdge(&graph, 1, 2, 100);
  AddEdge(&graph, 2, 3, 1);
  AddEdge(&graph, 4, 5, 50);

  const HopDistancesResult distances = graph.HopDistances(1);
  BOOST_REQUIRE(distances.IsSuccess());
  BOOST_CHECK(distances.distances.at(1).IsFinite());
  BOOST_CHECK_EQUAL(distances.distances.at(1).hops, 0);
  BOOST_CHECK(distances.distances.at(3).IsFinite());
  BOOST_CHECK_EQUAL(distances.distances.at(3).hops, 2);
  BOOST_CHECK(!distances.distances.at(4).IsFinite());
  BOOST_CHECK(distances.distances.at(4).reachability ==
              HopReachability::UNREACHABLE);
  BOOST_CHECK_EQUAL(distances.distances.at(4).hops,
                    std::numeric_limits<size_t>::max());

  const HopDistanceMatrixResult matrix =
      graph.HopDistanceMatrix({4, 3, 1});
  BOOST_REQUIRE(matrix.IsSuccess());
  const std::vector<image_t> expected_ids = {1, 3, 4};
  BOOST_CHECK(matrix.image_ids == expected_ids);
  BOOST_REQUIRE_EQUAL(matrix.distances.size(), 3);
  BOOST_CHECK_EQUAL(matrix.distances[0][1].hops, 2);
  BOOST_CHECK(!matrix.distances[0][2].IsFinite());
  BOOST_CHECK(!matrix.distances[2][0].IsFinite());
  BOOST_CHECK(matrix.distances[2][2].IsFinite());
  BOOST_CHECK_EQUAL(matrix.distances[2][2].hops, 0);
}

BOOST_AUTO_TEST_CASE(WindowExpansionUsesStrengthCapsAtTwentyAndReportsShortfall) {
  ActiveCovisibilityGraph graph;
  AddActiveRange(&graph, 1, 22);
  AddEdge(&graph, 1, 2, 10);
  AddEdge(&graph, 1, 3, 9);
  AddEdge(&graph, 2, 3, 5);
  for (image_t image_id = 4; image_id <= 22; ++image_id) {
    AddEdge(&graph, 1, image_id, 1);
  }

  const WindowExpansionResult full = graph.ExpandWindow({1}, 20);
  BOOST_REQUIRE(full.IsSuccess());
  BOOST_CHECK(full.ReachedMaxTotal());
  BOOST_CHECK_EQUAL(full.unfilled_slots, 0);
  BOOST_REQUIRE_EQUAL(full.selection_order.size(), 20);
  BOOST_CHECK_EQUAL(full.selection_order[0].image_id, 1);
  BOOST_CHECK(full.selection_order[0].is_seed);
  BOOST_CHECK_EQUAL(full.selection_order[1].image_id, 2);
  BOOST_CHECK_EQUAL(full.selection_order[1].support_strength, 10);
  BOOST_CHECK_EQUAL(full.selection_order[2].image_id, 3);
  BOOST_CHECK_EQUAL(full.selection_order[2].support_strength, 14);
  for (size_t index = 3; index < full.selection_order.size(); ++index) {
    BOOST_CHECK_EQUAL(full.selection_order[index].image_id,
                      static_cast<image_t>(index + 1));
    BOOST_CHECK_EQUAL(full.selection_order[index].support_strength, 1);
  }
  const std::vector<image_t> full_image_ids = SelectionImageIds(full);
  BOOST_CHECK(std::find(full_image_ids.begin(), full_image_ids.end(), 21) ==
              full_image_ids.end());
  BOOST_CHECK(graph.ExpandWindow({1}, 21).status ==
              ActiveCovisibilityStatus::INVALID_MAX_TOTAL);

  ActiveCovisibilityGraph sparse;
  AddActiveNodes(&sparse, {100, 101, 102});
  AddEdge(&sparse, 100, 101, 6);
  const WindowExpansionResult short_window =
      sparse.ExpandWindow({100}, 20);
  BOOST_REQUIRE(short_window.IsSuccess());
  BOOST_CHECK(!short_window.ReachedMaxTotal());
  BOOST_CHECK(short_window.stop_reason ==
              WindowExpansionStopReason::NO_ADJACENT_ACTIVE_NODE);
  BOOST_CHECK_EQUAL(short_window.selection_order.size(), 2);
  BOOST_CHECK_EQUAL(short_window.unfilled_slots, 18);
  BOOST_CHECK_EQUAL(short_window.selection_order.back().image_id, 101);
}

BOOST_AUTO_TEST_CASE(WindowExpansionTiesUseRegistrationSequenceThenImageId) {
  ActiveCovisibilityGraph graph;
  BOOST_REQUIRE(
      graph.AddNode(100, 1, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  BOOST_REQUIRE(
      graph.AddNode(10, 3, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  BOOST_REQUIRE(
      graph.AddNode(20, 2, ActiveCovisibilityNodeState::VISUAL_ACTIVE)
          .IsSuccess());
  AddEdge(&graph, 100, 10, 7);
  AddEdge(&graph, 100, 20, 7);

  const WindowExpansionResult result = graph.ExpandWindow({100}, 2);
  BOOST_REQUIRE(result.IsSuccess());
  const std::vector<image_t> expected = {100, 20};
  BOOST_CHECK(SelectionImageIds(result) == expected);
}

BOOST_AUTO_TEST_CASE(OrdinaryPairAcceptanceUsesThresholdAndAllStrongestTies) {
  const OrdinaryPairAcceptanceResult robust = SelectOrdinaryPairEdges(
      100,
      44,
      {OrdinaryPairSupport(30, 3, 14),
       OrdinaryPairSupport(20, 2, 15),
       OrdinaryPairSupport(10, 1, 20),
       OrdinaryPairSupport(40, 4, 0)});
  BOOST_REQUIRE(robust.IsSuccess());
  BOOST_CHECK_EQUAL(robust.strongest_positive_support, 20);
  const std::vector<image_t> expected_robust_strongest = {10};
  BOOST_CHECK(robust.strongest_positive_reference_image_ids ==
              expected_robust_strongest);
  BOOST_CHECK_EQUAL(robust.strongest_positive_reference_image_id, 10);
  BOOST_REQUIRE_EQUAL(robust.accepted_edges.size(), 2);
  BOOST_CHECK_EQUAL(robust.accepted_edges[0].image_id2, 10);
  BOOST_CHECK_EQUAL(robust.accepted_edges[1].image_id2, 20);
  BOOST_CHECK_EQUAL(robust.accepted_edges[0].birth_frame, 44);
  BOOST_REQUIRE(FindDecision(robust, 10) != nullptr);
  BOOST_REQUIRE(FindDecision(robust, 20) != nullptr);
  BOOST_REQUIRE(FindDecision(robust, 30) != nullptr);
  BOOST_REQUIRE(FindDecision(robust, 40) != nullptr);
  BOOST_CHECK(FindDecision(robust, 10)->is_robust);
  BOOST_CHECK(FindDecision(robust, 10)->is_strongest_positive);
  BOOST_CHECK(FindDecision(robust, 20)->is_robust);
  BOOST_CHECK(FindDecision(robust, 20)->accepted);
  BOOST_CHECK(!FindDecision(robust, 30)->is_robust);
  BOOST_CHECK(!FindDecision(robust, 30)->accepted);
  BOOST_CHECK(!FindDecision(robust, 40)->accepted);

  const OrdinaryPairAcceptanceResult weak = SelectOrdinaryPairEdges(
      100,
      45,
      {OrdinaryPairSupport(40, 2, 14),
       OrdinaryPairSupport(30, 2, 14),
       OrdinaryPairSupport(50, 3, 14),
       OrdinaryPairSupport(60, 4, 13),
       OrdinaryPairSupport(70, 5, 0)});
  BOOST_REQUIRE(weak.IsSuccess());
  BOOST_CHECK_EQUAL(weak.strongest_positive_support, 14);
  BOOST_CHECK_EQUAL(weak.strongest_positive_reference_image_id,
                    kInvalidImageId);
  const std::vector<image_t> expected_weak_strongest = {30, 40, 50};
  const std::vector<image_t> expected_weak_accepted = {30, 40, 50};
  BOOST_CHECK(weak.strongest_positive_reference_image_ids ==
              expected_weak_strongest);
  BOOST_CHECK(AcceptedReferenceImageIds(weak) == expected_weak_accepted);
  size_t strongest_count = 0;
  for (const OrdinaryPairDecision& decision : weak.decisions) {
    strongest_count += decision.is_strongest_positive ? 1 : 0;
  }
  BOOST_CHECK_EQUAL(strongest_count, 3);
  BOOST_CHECK(FindDecision(weak, 30)->accepted);
  BOOST_CHECK(FindDecision(weak, 40)->accepted);
  BOOST_CHECK(FindDecision(weak, 50)->accepted);
  BOOST_CHECK(!FindDecision(weak, 60)->accepted);
  BOOST_CHECK(!FindDecision(weak, 70)->accepted);

  const OrdinaryPairAcceptanceResult sequence_tie = SelectOrdinaryPairEdges(
      100,
      46,
      {OrdinaryPairSupport(80, 8, 9),
       OrdinaryPairSupport(90, 7, 9)});
  BOOST_REQUIRE(sequence_tie.IsSuccess());
  BOOST_CHECK_EQUAL(sequence_tie.strongest_positive_reference_image_id,
                    kInvalidImageId);
  const std::vector<image_t> expected_sequence_tie = {90, 80};
  BOOST_CHECK(sequence_tie.strongest_positive_reference_image_ids ==
              expected_sequence_tie);
  BOOST_CHECK(AcceptedReferenceImageIds(sequence_tie) ==
              expected_sequence_tie);

  const OrdinaryPairAcceptanceResult zero_only = SelectOrdinaryPairEdges(
      100, 47, {OrdinaryPairSupport(10, 1, 0)});
  BOOST_REQUIRE(zero_only.IsSuccess());
  BOOST_CHECK_EQUAL(zero_only.strongest_positive_support, 0);
  BOOST_CHECK(zero_only.strongest_positive_reference_image_ids.empty());
  BOOST_CHECK(zero_only.accepted_edges.empty());
  BOOST_CHECK_EQUAL(zero_only.strongest_positive_reference_image_id,
                    kInvalidImageId);

  const OrdinaryPairAcceptanceResult duplicate = SelectOrdinaryPairEdges(
      100,
      48,
      {OrdinaryPairSupport(10, 1, 5),
       OrdinaryPairSupport(10, 2, 6)});
  BOOST_CHECK(duplicate.status ==
              ActiveCovisibilityStatus::DUPLICATE_PAIR_SUPPORT);
  BOOST_CHECK(duplicate.accepted_edges.empty());
}

BOOST_AUTO_TEST_CASE(OrdinaryPairAcceptanceIsInputPermutationInvariant) {
  const std::vector<OrdinaryPairSupport> supports = {
      OrdinaryPairSupport(90, 4, 20),
      OrdinaryPairSupport(10, 1, 0),
      OrdinaryPairSupport(70, 3, 15),
      OrdinaryPairSupport(50, 2, 14),
      OrdinaryPairSupport(30, 2, 20)};
  const std::vector<image_t> expected_decisions = {10, 30, 50, 70, 90};
  const std::vector<image_t> expected_strongest = {30, 90};
  const std::vector<image_t> expected_accepted = {30, 70, 90};

  std::vector<size_t> permutation = {0, 1, 2, 3, 4};
  size_t permutation_count = 0;
  do {
    std::vector<OrdinaryPairSupport> permuted_supports;
    permuted_supports.reserve(supports.size());
    for (const size_t index : permutation) {
      permuted_supports.push_back(supports[index]);
    }

    const OrdinaryPairAcceptanceResult result =
        SelectOrdinaryPairEdges(100, 49, permuted_supports);
    BOOST_REQUIRE(result.IsSuccess());
    BOOST_CHECK_EQUAL(result.strongest_positive_support, 20);
    BOOST_CHECK_EQUAL(result.strongest_positive_reference_image_id,
                      kInvalidImageId);
    BOOST_CHECK(DecisionReferenceImageIds(result) == expected_decisions);
    BOOST_CHECK(result.strongest_positive_reference_image_ids ==
                expected_strongest);
    BOOST_CHECK(AcceptedReferenceImageIds(result) == expected_accepted);
    BOOST_REQUIRE(FindDecision(result, 10) != nullptr);
    BOOST_REQUIRE(FindDecision(result, 30) != nullptr);
    BOOST_REQUIRE(FindDecision(result, 50) != nullptr);
    BOOST_REQUIRE(FindDecision(result, 70) != nullptr);
    BOOST_REQUIRE(FindDecision(result, 90) != nullptr);
    BOOST_CHECK(!FindDecision(result, 10)->accepted);
    BOOST_CHECK(FindDecision(result, 30)->is_strongest_positive);
    BOOST_CHECK(!FindDecision(result, 50)->accepted);
    BOOST_CHECK(FindDecision(result, 70)->is_robust);
    BOOST_CHECK(FindDecision(result, 90)->is_strongest_positive);
    ++permutation_count;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  BOOST_CHECK_EQUAL(permutation_count, 120);
}

BOOST_AUTO_TEST_CASE(ShortestBackboneUsesCostThenFullLexicographicPath) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2, 3, 4, 6, 7, 8, 9});
  AddEdge(&graph, 1, 2, 4);
  AddEdge(&graph, 2, 6, 4);
  AddEdge(&graph, 1, 3, 4);
  AddEdge(&graph, 3, 6, 4);
  AddEdge(&graph, 1, 4, 3);
  AddEdge(&graph, 4, 6, 6);
  AddEdge(&graph, 1, 7, 10);
  AddEdge(&graph, 7, 8, 10);
  AddEdge(&graph, 1, 8, 4);

  const ShortestBackboneResult lex_tie = graph.ShortestBackbone(1, 6);
  BOOST_REQUIRE(lex_tie.IsSuccess());
  const std::vector<image_t> expected_lex_path = {1, 2, 6};
  BOOST_CHECK(lex_tie.path == expected_lex_path);
  BOOST_CHECK_EQUAL(lex_tie.hops, 2);
  BOOST_CHECK_SMALL(std::abs(lex_tie.total_cost - 0.5), 1e-15);

  BOOST_REQUIRE(
      graph.AddEdge(ActiveCovisibilityEdgeInput(1, 6, 2)).IsSuccess());
  const ShortestBackboneResult lex_over_shorter =
      graph.ShortestBackbone(1, 6);
  BOOST_REQUIRE(lex_over_shorter.IsSuccess());
  const std::vector<image_t> expected_lex_over_shorter_path = {1, 2, 6};
  BOOST_CHECK(lex_over_shorter.path == expected_lex_over_shorter_path);
  BOOST_CHECK_EQUAL(lex_over_shorter.hops, 2);
  BOOST_CHECK_SMALL(std::abs(lex_over_shorter.total_cost - 0.5), 1e-15);

  const ShortestBackboneResult weighted = graph.ShortestBackbone(1, 8);
  BOOST_REQUIRE(weighted.IsSuccess());
  const std::vector<image_t> expected_weighted_path = {1, 7, 8};
  BOOST_CHECK(weighted.path == expected_weighted_path);
  BOOST_CHECK_SMALL(std::abs(weighted.total_cost - 0.2), 1e-15);
  BOOST_REQUIRE_EQUAL(weighted.edge_costs.size(), 2);
  BOOST_CHECK_SMALL(std::abs(weighted.edge_costs[0] - 0.1), 1e-15);

  const ShortestBackboneResult unreachable = graph.ShortestBackbone(1, 9);
  BOOST_CHECK(unreachable.status == ActiveCovisibilityStatus::UNREACHABLE);
  BOOST_CHECK(!unreachable.reachable);
  BOOST_CHECK(std::isinf(unreachable.total_cost));
  BOOST_CHECK(unreachable.path.empty());
}

BOOST_AUTO_TEST_CASE(MultiSourceOwnershipUsesDistanceThenSourceTie) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {10, 20, 30, 40, 50, 60});
  AddEdge(&graph, 10, 30, 4);
  AddEdge(&graph, 20, 30, 4);
  AddEdge(&graph, 30, 40, 4);
  AddEdge(&graph, 20, 50, 10);

  const MultiSourceOwnershipResult ownership =
      graph.ComputeMultiSourceOwnership({20, 10});
  BOOST_REQUIRE(ownership.IsSuccess());
  const std::vector<image_t> expected_sources = {10, 20};
  BOOST_CHECK(ownership.source_image_ids == expected_sources);
  BOOST_REQUIRE(FindOwnership(ownership, 10) != nullptr);
  BOOST_REQUIRE(FindOwnership(ownership, 20) != nullptr);
  BOOST_REQUIRE(FindOwnership(ownership, 30) != nullptr);
  BOOST_REQUIRE(FindOwnership(ownership, 40) != nullptr);
  BOOST_REQUIRE(FindOwnership(ownership, 50) != nullptr);
  BOOST_REQUIRE(FindOwnership(ownership, 60) != nullptr);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 10)->source_image_id, 10);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 20)->source_image_id, 20);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 30)->source_image_id, 10);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 40)->source_image_id, 10);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 50)->source_image_id, 20);
  BOOST_CHECK_SMALL(std::abs(FindOwnership(ownership, 30)->distance - 0.25),
                    1e-15);
  BOOST_CHECK(!FindOwnership(ownership, 60)->reachable);
  BOOST_CHECK_EQUAL(FindOwnership(ownership, 60)->source_image_id,
                    kInvalidImageId);
  BOOST_CHECK(std::isinf(FindOwnership(ownership, 60)->distance));

  const MultiSourceOwnershipResult duplicate_sources =
      graph.ComputeMultiSourceOwnership({10, 10});
  BOOST_CHECK(duplicate_sources.status ==
              ActiveCovisibilityStatus::DUPLICATE_SOURCE);
}

BOOST_AUTO_TEST_CASE(AlgorithmsAreDeterministicAcrossRepeatedRuns) {
  ActiveCovisibilityGraph graph;
  AddActiveNodes(&graph, {1, 2, 3, 4, 5, 6});
  AddEdge(&graph, 1, 2, 5);
  AddEdge(&graph, 1, 3, 5);
  AddEdge(&graph, 2, 4, 5);
  AddEdge(&graph, 3, 4, 5);
  AddEdge(&graph, 4, 5, 5);
  AddEdge(&graph, 4, 6, 5);

  const ShortestBackboneResult expected_path =
      graph.ShortestBackbone(1, 4);
  const WindowExpansionResult expected_window = graph.ExpandWindow({1}, 6);
  const MultiSourceOwnershipResult expected_ownership =
      graph.ComputeMultiSourceOwnership({6, 5});
  const OrdinaryPairAcceptanceResult expected_pairs =
      SelectOrdinaryPairEdges(
          100,
          1,
          {OrdinaryPairSupport(3, 2, 14),
           OrdinaryPairSupport(2, 2, 14)});
  BOOST_REQUIRE(expected_path.IsSuccess());
  BOOST_REQUIRE(expected_window.IsSuccess());
  BOOST_REQUIRE(expected_ownership.IsSuccess());
  BOOST_REQUIRE(expected_pairs.IsSuccess());

  for (size_t run = 0; run < 50; ++run) {
    const ShortestBackboneResult path = graph.ShortestBackbone(1, 4);
    const WindowExpansionResult window = graph.ExpandWindow({1}, 6);
    const MultiSourceOwnershipResult ownership =
        graph.ComputeMultiSourceOwnership({6, 5});
    const OrdinaryPairAcceptanceResult pairs = SelectOrdinaryPairEdges(
        100,
        1,
        {OrdinaryPairSupport(3, 2, 14),
         OrdinaryPairSupport(2, 2, 14)});
    BOOST_CHECK(path.path == expected_path.path);
    BOOST_CHECK_EQUAL(path.total_cost, expected_path.total_cost);
    BOOST_CHECK(SelectionImageIds(window) ==
                SelectionImageIds(expected_window));
    BOOST_REQUIRE_EQUAL(ownership.ownership.size(),
                        expected_ownership.ownership.size());
    for (size_t index = 0; index < ownership.ownership.size(); ++index) {
      BOOST_CHECK_EQUAL(ownership.ownership[index].node_image_id,
                        expected_ownership.ownership[index].node_image_id);
      BOOST_CHECK_EQUAL(ownership.ownership[index].reachable,
                        expected_ownership.ownership[index].reachable);
      BOOST_CHECK_EQUAL(ownership.ownership[index].source_image_id,
                        expected_ownership.ownership[index].source_image_id);
      BOOST_CHECK_EQUAL(ownership.ownership[index].distance,
                        expected_ownership.ownership[index].distance);
    }
    BOOST_CHECK_EQUAL(pairs.strongest_positive_reference_image_id,
                      expected_pairs.strongest_positive_reference_image_id);
    BOOST_CHECK(pairs.strongest_positive_reference_image_ids ==
                expected_pairs.strongest_positive_reference_image_ids);
    BOOST_REQUIRE_EQUAL(pairs.accepted_edges.size(),
                        expected_pairs.accepted_edges.size());
    BOOST_CHECK(AcceptedReferenceImageIds(pairs) ==
                AcceptedReferenceImageIds(expected_pairs));
  }
}

BOOST_AUTO_TEST_CASE(AlgorithmsHandleFullSessionScaleWithBoundedWork) {
  constexpr size_t kNodeCount = 246;
  constexpr image_t kImageIdBase = 1000;
  constexpr size_t kStrength = 10;

  ActiveCovisibilityCommit commit;
  commit.expected_version = 0;
  commit.nodes.reserve(kNodeCount);
  commit.ordinary_edges.reserve(kNodeCount - 1);
  for (size_t registration_sequence = 1;
       registration_sequence <= kNodeCount;
       ++registration_sequence) {
    const image_t image_id =
        kImageIdBase - static_cast<image_t>(registration_sequence);
    commit.nodes.emplace_back(image_id,
                              registration_sequence,
                              ActiveCovisibilityNodeState::VISUAL_ACTIVE);
    if (registration_sequence > 1) {
      commit.ordinary_edges.emplace_back(
          image_id + 1, image_id, kStrength, registration_sequence);
    }
  }

  ActiveCovisibilityGraph graph;
  BOOST_REQUIRE(graph.Commit(commit).IsSuccess());
  BOOST_CHECK_EQUAL(graph.NumNodes(), kNodeCount);
  BOOST_CHECK_EQUAL(graph.NumEdges(), kNodeCount - 1);

  const image_t first_image_id = kImageIdBase - 1;
  const image_t last_image_id =
      kImageIdBase - static_cast<image_t>(kNodeCount);
  const HopDistancesResult hops = graph.HopDistances(first_image_id);
  BOOST_REQUIRE(hops.IsSuccess());
  BOOST_CHECK_EQUAL(hops.distances.size(), kNodeCount);
  BOOST_CHECK_EQUAL(hops.distances.at(last_image_id).hops, kNodeCount - 1);

  const WindowExpansionResult window =
      graph.ExpandWindow({first_image_id},
                         ActiveCovisibilityGraph::kMaxWindowSize);
  BOOST_REQUIRE(window.IsSuccess());
  BOOST_REQUIRE_EQUAL(window.selection_order.size(),
                      ActiveCovisibilityGraph::kMaxWindowSize);
  for (size_t index = 0; index < window.selection_order.size(); ++index) {
    BOOST_CHECK_EQUAL(window.selection_order[index].image_id,
                      first_image_id - static_cast<image_t>(index));
  }
  const size_t expansion_rounds = window.selection_order.size() - 1;
  BOOST_CHECK(window.candidate_evaluations <=
              expansion_rounds * graph.NumNodes());
  BOOST_CHECK(window.adjacency_visits <=
              expansion_rounds * 2 * graph.NumEdges());

  const ShortestBackboneResult backbone =
      graph.ShortestBackbone(first_image_id, last_image_id);
  BOOST_REQUIRE(backbone.IsSuccess());
  BOOST_CHECK_EQUAL(backbone.hops, kNodeCount - 1);
  BOOST_CHECK_EQUAL(backbone.path.size(), kNodeCount);
  BOOST_CHECK(backbone.queue_pops <= graph.NumNodes());
  BOOST_CHECK(backbone.edge_relaxations <= 2 * graph.NumEdges());

  const MultiSourceOwnershipResult ownership =
      graph.ComputeMultiSourceOwnership({first_image_id, last_image_id});
  BOOST_REQUIRE(ownership.IsSuccess());
  BOOST_CHECK_EQUAL(ownership.ownership.size(), kNodeCount);
  BOOST_CHECK(ownership.queue_pops <= graph.NumNodes());
  BOOST_CHECK(ownership.edge_relaxations <= 2 * graph.NumEdges());
}
