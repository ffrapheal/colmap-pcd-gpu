#define TEST_NAME "sfm/online_dual_selection"
#include "util/testing.h"

#include "sfm/online_dual_selection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Geometry>

using namespace colmap;

namespace {

void AddChain(ActiveCovisibilityGraph* graph,
              const image_t first,
              const image_t last,
              const size_t strength = 20) {
  for (image_t image_id = first; image_id <= last; ++image_id) {
    BOOST_REQUIRE(
        graph
            ->AddNode(image_id,
                      static_cast<size_t>(image_id),
                      ActiveCovisibilityNodeState::VISUAL_ACTIVE)
            .IsSuccess());
    if (image_id > first) {
      BOOST_REQUIRE(
          graph
              ->AddEdge(ActiveCovisibilityEdgeInput(
                  image_id - 1, image_id, strength, 1))
              .IsSuccess());
    }
  }
}

OnlineDualReference ActiveReference(const image_t image_id,
                                    const size_t verified_inliers) {
  return OnlineDualReference(image_id,
                             static_cast<size_t>(image_id),
                             verified_inliers,
                             ActiveCovisibilityNodeState::VISUAL_ACTIVE,
                             true,
                             true);
}

std::vector<OnlineDualReference> SeparatedReferences(
    const std::vector<size_t>& older_support,
    const std::vector<size_t>& newer_support) {
  std::vector<OnlineDualReference> references;
  const image_t older_ids[] = {1, 2, 3};
  const image_t newer_ids[] = {10, 11, 12};
  for (size_t index = 0; index < older_support.size(); ++index) {
    references.push_back(ActiveReference(older_ids[index],
                                         older_support[index]));
  }
  for (size_t index = 0; index < newer_support.size(); ++index) {
    references.push_back(ActiveReference(newer_ids[index],
                                         newer_support[index]));
  }
  return references;
}

OnlineDualClassificationResult DualCandidateClassification(
    const ActiveCovisibilityGraph& graph,
    const std::vector<size_t>& older_support = {20, 18, 16},
    const std::vector<size_t>& newer_support = {24, 22, 20}) {
  const OnlineDualClassificationResult result = ClassifyOnlineDualReferences(
      graph, SeparatedReferences(older_support, newer_support));
  BOOST_REQUIRE_MESSAGE(result.IsDualCandidate(), result.detail);
  return result;
}

OnlineDualPnPCorrespondenceSet Correspondences(
    const OnlineDualClusterId cluster_id,
    const size_t count = 120) {
  OnlineDualPnPCorrespondenceSet set;
  set.cluster_id = cluster_id;
  for (size_t index = 0; index < count; ++index) {
    set.points2D.emplace_back(static_cast<double>(index),
                              static_cast<double>(index % 7));
    set.points3D.emplace_back(static_cast<double>(index),
                              static_cast<double>(index % 5),
                              1.0 + static_cast<double>(index % 3));
  }
  return set;
}

OnlineDualPnPSolverOutput SuccessfulProbe(
    const OnlineDualPnPProbeRequest& request,
    const OnlineDualSE3& pose,
    const size_t inlier_count = 30) {
  OnlineDualPnPSolverOutput output;
  output.success = true;
  output.probe_T_cw = pose;
  output.inlier_mask.assign(request.points2D.size(), 0);
  std::fill_n(output.inlier_mask.begin(), inlier_count, 1);
  return output;
}

Eigen::Vector4d RotationZ(const double radians) {
  const Eigen::Quaterniond quaternion(
      Eigen::AngleAxisd(radians, Eigen::Vector3d::UnitZ()));
  return Eigen::Vector4d(quaternion.w(),
                         quaternion.x(),
                         quaternion.y(),
                         quaternion.z());
}

void CheckVectorNear(const Eigen::Vector3d& actual,
                     const Eigen::Vector3d& expected,
                     const double tolerance = 1e-12) {
  BOOST_CHECK_SMALL((actual - expected).norm(), tolerance);
}

const OnlineDualWindowImage* FindWindowImage(
    const OnlineDualWindowResult& window, const image_t image_id) {
  for (const OnlineDualWindowImage& image : window.images) {
    if (image.image_id == image_id) {
      return &image;
    }
  }
  return nullptr;
}

const OnlineDualImageCorrection* FindImageCorrection(
    const OnlineDualPropagationPlan& plan, const image_t image_id) {
  for (const OnlineDualImageCorrection& correction : plan.image_corrections) {
    if (correction.image_id == image_id) {
      return &correction;
    }
  }
  return nullptr;
}

const OnlineDualPointCorrection* FindPointCorrection(
    const OnlineDualPointCorrectionPlan& plan, const point3D_t point3D_id) {
  for (const OnlineDualPointCorrection& correction : plan.corrections) {
    if (correction.point3D_id == point3D_id) {
      return &correction;
    }
  }
  return nullptr;
}

OnlineDualWindowImage WindowImage(const image_t image_id,
                                  const OnlineDualTemporalSide temporal_side,
                                  const bool is_current = false) {
  OnlineDualWindowImage image;
  image.image_id = image_id;
  image.temporal_side = temporal_side;
  image.is_current = is_current;
  return image;
}

OnlineDualImageDelta ImageDelta(const image_t image_id,
                                const Eigen::Vector4d& qvec,
                                const Eigen::Vector3d& tvec) {
  OnlineDualImageDelta image_delta;
  image_delta.image_id = image_id;
  image_delta.delta = OnlineDualSE3(qvec, tvec);
  return image_delta;
}

OnlineDualPointState VisualPoint(
    const point3D_t point3D_id,
    const image_t owner_image_id,
    const ActiveCovisibilityNodeState owner_state,
    const bool directly_optimized = false) {
  OnlineDualPointState point;
  point.point3D_id = point3D_id;
  point.owner_image_id = owner_image_id;
  point.owner_registration_sequence = owner_image_id;
  point.owner_visual_state = owner_state;
  point.directly_optimized = directly_optimized;
  return point;
}

OnlineDualPointLineageAncestor LineageAncestor(
    const image_t owner_image_id,
    const size_t owner_registration_sequence,
    const bool directly_optimized) {
  OnlineDualPointLineageAncestor ancestor;
  ancestor.owner_image_id = owner_image_id;
  ancestor.owner_registration_sequence = owner_registration_sequence;
  ancestor.directly_optimized = directly_optimized;
  return ancestor;
}

}  // namespace

BOOST_AUTO_TEST_CASE(MaximumHopBoundaryIsStrictAtTen) {
  ActiveCovisibilityGraph ten_hop_graph;
  AddChain(&ten_hop_graph, 1, 11);
  std::vector<OnlineDualReference> ten_hop_references = {
      ActiveReference(1, 20), ActiveReference(11, 20)};
  const OnlineDualClassificationResult ten_hops =
      ClassifyOnlineDualReferences(ten_hop_graph, ten_hop_references);
  BOOST_CHECK(ten_hops.result_class ==
              OnlineDualResultClass::SINGLE_FALLBACK);
  BOOST_CHECK(ten_hops.reason ==
              OnlineDualReason::MAX_HOP_NOT_GREATER_THAN_TEN);
  BOOST_CHECK_EQUAL(ten_hops.maximum_hops, 10);

  ActiveCovisibilityGraph eleven_hop_graph;
  AddChain(&eleven_hop_graph, 1, 12);
  const OnlineDualClassificationResult eleven_hops =
      DualCandidateClassification(eleven_hop_graph);
  BOOST_CHECK_EQUAL(eleven_hops.maximum_hops, 11);
  BOOST_CHECK(eleven_hops.decision == OnlineDualDecision::DUAL_CANDIDATE);
}

BOOST_AUTO_TEST_CASE(RejectsPoseOnlyAndNonCurrentPositiveReferences) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  std::vector<OnlineDualReference> references =
      SeparatedReferences({20, 18, 16}, {24, 22, 20});
  references[0].visual_state = ActiveCovisibilityNodeState::POSE_ONLY;
  OnlineDualClassificationResult result =
      ClassifyOnlineDualReferences(graph, references);
  BOOST_CHECK(result.result_class == OnlineDualResultClass::INPUT_ERROR);
  BOOST_CHECK(result.reason ==
              OnlineDualReason::REFERENCE_NOT_VISUAL_ACTIVE);

  references[0].visual_state = ActiveCovisibilityNodeState::VISUAL_ACTIVE;
  references[0].matched_this_event = false;
  result = ClassifyOnlineDualReferences(graph, references);
  BOOST_CHECK(result.reason ==
              OnlineDualReason::REFERENCE_NOT_MATCHED_THIS_EVENT);

  references[0].matched_this_event = true;
  references[0].positive_verified = false;
  result = ClassifyOnlineDualReferences(graph, references);
  BOOST_CHECK(result.reason ==
              OnlineDualReason::REFERENCE_NOT_POSITIVE_VERIFIED);
}

BOOST_AUTO_TEST_CASE(WeakBestExactlyHalfFallsBackToSingle) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult result = ClassifyOnlineDualReferences(
      graph, SeparatedReferences({20, 15, 15}, {40, 15, 15}));
  BOOST_CHECK(result.result_class ==
              OnlineDualResultClass::SINGLE_FALLBACK);
  BOOST_CHECK(result.reason ==
              OnlineDualReason::WEAK_SUPPORT_NOT_STRICTLY_GREATER_THAN_HALF);
  BOOST_REQUIRE_EQUAL(result.clusters.size(), 2);
  BOOST_CHECK_EQUAL(result.clusters[0].robust_pair_count, 3);
  BOOST_CHECK_EQUAL(result.clusters[1].robust_pair_count, 3);
}

BOOST_AUTO_TEST_CASE(MainAndOlderLabelsAreIndependent) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult result =
      DualCandidateClassification(graph, {16, 16, 16}, {30, 30, 30});
  BOOST_CHECK(result.older_cluster_id == OnlineDualClusterId::FIRST);
  BOOST_CHECK(result.main_cluster_id == OnlineDualClusterId::SECOND);
  BOOST_CHECK(result.main_cluster_id != result.older_cluster_id);
  BOOST_CHECK_EQUAL(std::string(ToString(result.main_cluster_id)), "SECOND");
}

BOOST_AUTO_TEST_CASE(MainSupportTieUsesEarlierCluster) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult result =
      DualCandidateClassification(graph, {20, 18, 16}, {20, 18, 16});
  BOOST_CHECK(result.main_selected_by_tie_break);
  BOOST_CHECK(result.main_cluster_id == OnlineDualClusterId::FIRST);
  BOOST_CHECK(result.older_cluster_id == OnlineDualClusterId::FIRST);
  const OnlineDualResolvedPolicy policy = GetOnlineDualResolvedPolicy();
  BOOST_CHECK_EQUAL(policy.dual_hop_threshold, 10);
  BOOST_CHECK_EQUAL(policy.robust_pair_minimum_inliers, 15);
  BOOST_CHECK_EQUAL(policy.minimum_robust_pairs_per_side, 3);
  BOOST_CHECK_EQUAL(policy.minimum_pnp_inliers_per_side, 30);
  BOOST_CHECK_EQUAL(policy.maximum_window_size, 20);
  BOOST_CHECK_EQUAL(policy.maximum_window_size_per_side, 10);
}

BOOST_AUTO_TEST_CASE(ClusterDistanceTieUsesEarlierSeed) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 13);
  const std::vector<OnlineDualReference> references = {
      ActiveReference(1, 20),  ActiveReference(2, 20),
      ActiveReference(3, 20),  ActiveReference(7, 20),
      ActiveReference(11, 20), ActiveReference(12, 20),
      ActiveReference(13, 20)};
  const OnlineDualClassificationResult result =
      ClassifyOnlineDualReferences(graph, references);
  BOOST_REQUIRE(result.IsDualCandidate());
  const auto assignment = std::find_if(
      result.assignments.begin(),
      result.assignments.end(),
      [](const OnlineDualReferenceAssignment& value) {
        return value.reference.image_id == 7;
      });
  BOOST_REQUIRE(assignment != result.assignments.end());
  BOOST_CHECK(assignment->assigned_by_seed_tie_break);
  BOOST_CHECK(assignment->cluster_id == OnlineDualClusterId::FIRST);
  BOOST_CHECK_EQUAL(assignment->hops_to_first_seed, 6);
  BOOST_CHECK_EQUAL(assignment->hops_to_second_seed, 6);
}

BOOST_AUTO_TEST_CASE(TwoSideProbeFailureFallsBackWithoutRegistrationEffects) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult classification =
      DualCandidateClassification(graph);
  size_t calls = 0;
  const OnlineDualPnPProbeAdapter adapter(
      [&calls](const OnlineDualPnPProbeRequest& request) {
        ++calls;
        if (request.cluster_id == OnlineDualClusterId::SECOND) {
          OnlineDualPnPSolverOutput output;
          output.detail = "second side intentionally failed";
          return output;
        }
        return SuccessfulProbe(request, OnlineDualSE3());
      });
  const OnlineDualPnPDecisionResult result = RunOnlineDualTwoSidePnPProbe(
      classification,
      {Correspondences(OnlineDualClusterId::FIRST),
       Correspondences(OnlineDualClusterId::SECOND)},
      adapter);
  BOOST_CHECK_EQUAL(calls, 2);
  BOOST_CHECK(result.result_class ==
              OnlineDualResultClass::SINGLE_FALLBACK);
  BOOST_CHECK(result.reason == OnlineDualReason::PNP_SOLVER_FAILED);
  BOOST_CHECK(result.failed_cluster_id == OnlineDualClusterId::SECOND);

  OnlineDualClassificationResult single = classification;
  single.result_class = OnlineDualResultClass::SINGLE_FALLBACK;
  single.decision = OnlineDualDecision::SINGLE;
  const OnlineDualPnPDecisionResult not_called =
      RunOnlineDualTwoSidePnPProbe(single, {}, adapter);
  BOOST_CHECK(not_called.result_class == OnlineDualResultClass::INPUT_ERROR);
  BOOST_CHECK_EQUAL(calls, 2);
}

BOOST_AUTO_TEST_CASE(PnPRequiresValidSE3AndComputesPnPOnlyDelta) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult classification =
      DualCandidateClassification(graph, {16, 16, 16}, {30, 30, 30});
  const OnlineDualPnPProbeAdapter invalid_adapter(
      [](const OnlineDualPnPProbeRequest& request) {
        OnlineDualSE3 pose;
        if (request.cluster_id == OnlineDualClusterId::SECOND) {
          pose.qvec.setZero();
        }
        return SuccessfulProbe(request, pose);
      });
  OnlineDualPnPDecisionResult result = RunOnlineDualTwoSidePnPProbe(
      classification,
      {Correspondences(OnlineDualClusterId::FIRST),
       Correspondences(OnlineDualClusterId::SECOND)},
      invalid_adapter);
  BOOST_CHECK(result.result_class ==
              OnlineDualResultClass::SINGLE_FALLBACK);
  BOOST_CHECK(result.reason == OnlineDualReason::PNP_INVALID_SE3);

  const OnlineDualPnPProbeAdapter valid_adapter(
      [](const OnlineDualPnPProbeRequest& request) {
        const double x = request.cluster_id == OnlineDualClusterId::FIRST
                             ? 1.0
                             : 1001.0;
        return SuccessfulProbe(
            request,
            OnlineDualSE3(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                          Eigen::Vector3d(x, 0.0, 0.0)));
      });
  result = RunOnlineDualTwoSidePnPProbe(
      classification,
      {Correspondences(OnlineDualClusterId::FIRST),
       Correspondences(OnlineDualClusterId::SECOND)},
      valid_adapter);
  BOOST_REQUIRE(result.IsDualCandidate());
  CheckVectorNear(result.delta_init.tvec, Eigen::Vector3d(1000.0, 0.0, 0.0));
  BOOST_CHECK_CLOSE(result.delta_translation_norm, 1000.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(DualWindowGivesOlderOverlapPriorityWithoutBorrowing) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 12);
  const OnlineDualClassificationResult classification =
      DualCandidateClassification(graph, {16, 16, 16}, {30, 30, 30});
  ActiveCovisibilityOverlay overlay = graph.CreateTentativeOverlay();
  overlay.temporary_nodes.emplace_back(100, 100);
  overlay.ordinary_edges.emplace_back(100, 1, 25, 100);
  overlay.loop_edges.emplace_back(100, 12, 25, 100);
  const OnlineDualWindowResult window = BuildOnlineDualFrozenWindow(
      graph, overlay, 100, 100, classification);
  BOOST_REQUIRE_MESSAGE(window.IsSuccess(), window.detail);
  BOOST_CHECK_LE(window.images.size(), 20);
  BOOST_CHECK_EQUAL(window.older_side.selected_image_ids.size(), 10);
  BOOST_CHECK_EQUAL(window.newer_side.selected_image_ids.size(), 3);
  const OnlineDualWindowResult reduced_window = BuildOnlineDualFrozenWindow(
      graph, overlay, 100, 100, classification, 10);
  BOOST_REQUIRE_MESSAGE(reduced_window.IsSuccess(), reduced_window.detail);
  BOOST_CHECK_EQUAL(reduced_window.older_side.selected_image_ids.size(), 5);
  BOOST_CHECK_EQUAL(reduced_window.newer_side.selected_image_ids.size(), 5);
  BOOST_CHECK_EQUAL(reduced_window.images.size(), 10);
  BOOST_REQUIRE(FindWindowImage(reduced_window, 100) != nullptr);
  const OnlineDualWindowResult limited_seed_window = BuildOnlineDualFrozenWindow(
      graph, overlay, 100, 100, classification, 6);
  BOOST_REQUIRE_MESSAGE(limited_seed_window.IsSuccess(), limited_seed_window.detail);
  BOOST_CHECK_LE(limited_seed_window.images.size(), 6);
  BOOST_REQUIRE(FindWindowImage(limited_seed_window, 100) != nullptr);
  BOOST_CHECK(!window.newer_side.skipped_overlap_image_ids.empty());
  BOOST_CHECK_EQUAL(static_cast<size_t>(std::count_if(
                        window.images.begin(),
                        window.images.end(),
                        [](const OnlineDualWindowImage& image) {
                          return image.is_current;
                        })),
                    1);
  const OnlineDualWindowImage* current = FindWindowImage(window, 100);
  BOOST_REQUIRE(current != nullptr);
  BOOST_CHECK(current->temporal_side == OnlineDualTemporalSide::NEWER);
  for (const image_t overlap :
       window.newer_side.skipped_overlap_image_ids) {
    const OnlineDualWindowImage* owner = FindWindowImage(window, overlap);
    BOOST_REQUIRE(owner != nullptr);
    BOOST_CHECK(owner->temporal_side == OnlineDualTemporalSide::OLDER);
  }
}

BOOST_AUTO_TEST_CASE(BackboneEndpointAndPathTiesAreDeterministic) {
  ActiveCovisibilityGraph endpoint_graph;
  for (image_t image_id : {1, 2, 4, 5}) {
    BOOST_REQUIRE(endpoint_graph
                      .AddNode(image_id,
                               image_id,
                               ActiveCovisibilityNodeState::VISUAL_ACTIVE)
                      .IsSuccess());
  }
  for (image_t older : {1, 2}) {
    for (image_t newer : {4, 5}) {
      BOOST_REQUIRE(endpoint_graph
                        .AddEdge(ActiveCovisibilityEdgeInput(
                            older, newer, 20, 1))
                        .IsSuccess());
    }
  }
  OnlineDualWindowResult endpoint_window;
  endpoint_window.result_class = OnlineDualResultClass::SUCCESS;
  endpoint_window.frozen_graph_version = endpoint_graph.Version();
  endpoint_window.images = {
      WindowImage(1, OnlineDualTemporalSide::OLDER),
      WindowImage(2, OnlineDualTemporalSide::OLDER),
      WindowImage(4, OnlineDualTemporalSide::NEWER),
      WindowImage(5, OnlineDualTemporalSide::NEWER),
      WindowImage(100, OnlineDualTemporalSide::OLDER, true)};
  const OnlineDualBackboneResult endpoint_backbone =
      SelectOnlineDualBackbone(endpoint_graph, endpoint_window);
  BOOST_REQUIRE(endpoint_backbone.IsDualReady());
  BOOST_CHECK_EQUAL(endpoint_backbone.older_endpoint_image_id, 1);
  BOOST_CHECK_EQUAL(endpoint_backbone.newer_endpoint_image_id, 4);
  BOOST_CHECK_EQUAL(endpoint_backbone.minimum_cost_endpoint_pair_count, 4);

  ActiveCovisibilityGraph path_graph;
  for (image_t image_id = 1; image_id <= 4; ++image_id) {
    BOOST_REQUIRE(path_graph
                      .AddNode(image_id,
                               image_id,
                               ActiveCovisibilityNodeState::VISUAL_ACTIVE)
                      .IsSuccess());
  }
  BOOST_REQUIRE(path_graph.AddEdge({1, 2, 20, 1}).IsSuccess());
  BOOST_REQUIRE(path_graph.AddEdge({2, 4, 20, 1}).IsSuccess());
  BOOST_REQUIRE(path_graph.AddEdge({1, 3, 20, 1}).IsSuccess());
  BOOST_REQUIRE(path_graph.AddEdge({3, 4, 20, 1}).IsSuccess());
  OnlineDualWindowResult path_window;
  path_window.result_class = OnlineDualResultClass::SUCCESS;
  path_window.frozen_graph_version = path_graph.Version();
  path_window.images = {
      WindowImage(1, OnlineDualTemporalSide::OLDER),
      WindowImage(4, OnlineDualTemporalSide::NEWER),
      WindowImage(100, OnlineDualTemporalSide::OLDER, true)};
  const OnlineDualBackboneResult path_backbone =
      SelectOnlineDualBackbone(path_graph, path_window);
  BOOST_REQUIRE(path_backbone.IsDualReady());
  const std::vector<image_t> expected_path = {1, 2, 4};
  BOOST_CHECK(path_backbone.path == expected_path);
}

BOOST_AUTO_TEST_CASE(BackboneNeverUsesTentativeLoopEdge) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 4);
  ActiveCovisibilityOverlay tentative = graph.CreateTentativeOverlay();
  tentative.loop_edges.emplace_back(1, 4, 1000, 10);
  BOOST_REQUIRE(graph.ValidateTentativeOverlay(tentative).IsSuccess());

  OnlineDualWindowResult window;
  window.result_class = OnlineDualResultClass::SUCCESS;
  window.frozen_graph_version = graph.Version();
  window.images = {
      WindowImage(1, OnlineDualTemporalSide::OLDER),
      WindowImage(4, OnlineDualTemporalSide::NEWER),
      WindowImage(100, OnlineDualTemporalSide::OLDER, true)};
  const OnlineDualBackboneResult backbone =
      SelectOnlineDualBackbone(graph, window);
  BOOST_REQUIRE(backbone.IsDualReady());
  const std::vector<image_t> expected_path = {1, 2, 3, 4};
  BOOST_CHECK(backbone.path == expected_path);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 3);
}

BOOST_AUTO_TEST_CASE(EvenMedianSelectsOneCompleteDeltaAndPropagatesByCost) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 6, 20);
  OnlineDualWindowResult window;
  window.result_class = OnlineDualResultClass::SUCCESS;
  window.frozen_graph_version = graph.Version();
  window.current_image_id = 100;
  window.images = {
      WindowImage(1, OnlineDualTemporalSide::OLDER),
      WindowImage(100, OnlineDualTemporalSide::OLDER, true),
      WindowImage(4, OnlineDualTemporalSide::NEWER),
      WindowImage(5, OnlineDualTemporalSide::NEWER)};
  const OnlineDualBackboneResult backbone =
      SelectOnlineDualBackbone(graph, window);
  BOOST_REQUIRE(backbone.IsDualReady());

  const Eigen::Vector4d identity(1.0, 0.0, 0.0, 0.0);
  const Eigen::Vector4d quarter_turn = RotationZ(std::acos(-1.0) / 2.0);
  const OnlineDualImageDeltaVector deltas = {
      ImageDelta(1, identity, Eigen::Vector3d::Zero()),
      ImageDelta(100, identity, Eigen::Vector3d::Zero()),
      ImageDelta(4, quarter_turn, Eigen::Vector3d(0.0, 2.0, 0.0)),
      ImageDelta(5, identity, Eigen::Vector3d(2.0, 0.0, 0.0))};
  const OnlineDualPropagationPlan plan = BuildOnlineDualPropagationPlan(
      graph, window, backbone, deltas, {99});
  BOOST_REQUIRE_MESSAGE(plan.IsSuccess(), plan.detail);
  CheckVectorNear(plan.componentwise_translation_median,
                  Eigen::Vector3d(1.0, 1.0, 0.0));
  BOOST_CHECK_EQUAL(plan.representative_image_id, 4);
  BOOST_CHECK_EQUAL(plan.minimum_distance_representative_count, 2);
  CheckVectorNear(plan.representative_delta.tvec,
                  Eigen::Vector3d(0.0, 2.0, 0.0));
  BOOST_CHECK_SMALL(
      (plan.representative_delta.qvec - quarter_turn).norm(), 1e-12);

  const OnlineDualImageCorrection* direct = FindImageCorrection(plan, 4);
  BOOST_REQUIRE(direct != nullptr);
  BOOST_CHECK(direct->kind ==
              OnlineDualImageCorrectionKind::DIRECT_WINDOW_RESULT);
  BOOST_CHECK(!direct->apply_correction);
  const OnlineDualImageCorrection* interpolated =
      FindImageCorrection(plan, 2);
  BOOST_REQUIRE(interpolated != nullptr);
  BOOST_CHECK(interpolated->kind ==
              OnlineDualImageCorrectionKind::PROPAGATED);
  BOOST_CHECK_CLOSE(interpolated->backbone_alpha, 1.0 / 3.0, 1e-10);
  CheckVectorNear(interpolated->delta.tvec,
                  Eigen::Vector3d(0.0, 2.0 / 3.0, 0.0));
  BOOST_CHECK_SMALL(
      (interpolated->delta.qvec -
       RotationZ(std::acos(-1.0) / 6.0))
          .norm(),
      1e-12);
  const OnlineDualImageCorrection* propagated = FindImageCorrection(plan, 6);
  BOOST_REQUIRE(propagated != nullptr);
  BOOST_CHECK(propagated->kind ==
              OnlineDualImageCorrectionKind::PROPAGATED);
  BOOST_CHECK(propagated->apply_correction);
  BOOST_CHECK_CLOSE(propagated->backbone_alpha, 1.0, 1e-12);
  CheckVectorNear(propagated->delta.tvec,
                  Eigen::Vector3d(0.0, 2.0, 0.0));
  const OnlineDualImageCorrection* pose_only = FindImageCorrection(plan, 99);
  BOOST_REQUIRE(pose_only != nullptr);
  BOOST_CHECK(pose_only->kind == OnlineDualImageCorrectionKind::POSE_ONLY);

  OnlineDualImageDeltaVector invalid_deltas = deltas;
  invalid_deltas[2].delta.qvec.setZero();
  const OnlineDualPropagationPlan invalid = BuildOnlineDualPropagationPlan(
      graph, window, backbone, invalid_deltas, {99});
  BOOST_CHECK(invalid.result_class == OnlineDualResultClass::DUAL_FATAL);
  BOOST_CHECK(invalid.reason == OnlineDualReason::INVALID_WINDOW_DELTA);
}

BOOST_AUTO_TEST_CASE(PointPlanUsesDirectOwnerAndPropagationPrecedence) {
  ActiveCovisibilityGraph graph;
  AddChain(&graph, 1, 6, 20);
  OnlineDualWindowResult window;
  window.result_class = OnlineDualResultClass::SUCCESS;
  window.frozen_graph_version = graph.Version();
  window.current_image_id = 100;
  window.images = {
      WindowImage(1, OnlineDualTemporalSide::OLDER),
      WindowImage(100, OnlineDualTemporalSide::OLDER, true),
      WindowImage(4, OnlineDualTemporalSide::NEWER),
      WindowImage(5, OnlineDualTemporalSide::NEWER)};
  const OnlineDualBackboneResult backbone =
      SelectOnlineDualBackbone(graph, window);
  BOOST_REQUIRE(backbone.IsDualReady());
  const Eigen::Vector4d identity(1.0, 0.0, 0.0, 0.0);
  const OnlineDualImageDeltaVector deltas = {
      ImageDelta(1, identity, Eigen::Vector3d::Zero()),
      ImageDelta(100, identity, Eigen::Vector3d::Zero()),
      ImageDelta(4, identity, Eigen::Vector3d(0.0, 2.0, 0.0)),
      ImageDelta(5, identity, Eigen::Vector3d(2.0, 0.0, 0.0))};
  const OnlineDualPropagationPlan propagation =
      BuildOnlineDualPropagationPlan(graph, window, backbone, deltas, {99});
  BOOST_REQUIRE(propagation.IsSuccess());

  std::vector<OnlineDualPointState> points = {
      VisualPoint(1, 6, ActiveCovisibilityNodeState::VISUAL_ACTIVE, true),
      VisualPoint(2, 5, ActiveCovisibilityNodeState::VISUAL_ACTIVE),
      VisualPoint(3, 6, ActiveCovisibilityNodeState::VISUAL_ACTIVE),
      VisualPoint(4, 1, ActiveCovisibilityNodeState::VISUAL_ACTIVE),
      VisualPoint(5, 99, ActiveCovisibilityNodeState::POSE_ONLY)};
  OnlineDualPointState lidar_point;
  lidar_point.point3D_id = 6;
  lidar_point.point_kind = OnlineDualPointKind::LIDAR;
  points.push_back(lidar_point);
  const OnlineDualPointCorrectionPlan plan =
      BuildOnlineDualPointCorrectionPlan(window, deltas, propagation, points);
  BOOST_REQUIRE_MESSAGE(plan.IsSuccess(), plan.detail);
  const OnlineDualPointCorrection* direct = FindPointCorrection(plan, 1);
  BOOST_REQUIRE(direct != nullptr);
  BOOST_CHECK(direct->kind == OnlineDualPointCorrectionKind::DIRECT_OPTIMIZED);
  const OnlineDualPointCorrection* own_delta = FindPointCorrection(plan, 2);
  BOOST_REQUIRE(own_delta != nullptr);
  BOOST_CHECK(own_delta->kind ==
              OnlineDualPointCorrectionKind::OWNER_NEWER_WINDOW_DELTA);
  CheckVectorNear(own_delta->delta.tvec, Eigen::Vector3d(2.0, 0.0, 0.0));
  const OnlineDualPointCorrection* propagated = FindPointCorrection(plan, 3);
  const OnlineDualPointCorrection* older = FindPointCorrection(plan, 4);
  const OnlineDualPointCorrection* pose_only = FindPointCorrection(plan, 5);
  const OnlineDualPointCorrection* lidar = FindPointCorrection(plan, 6);
  BOOST_REQUIRE(propagated != nullptr);
  BOOST_REQUIRE(older != nullptr);
  BOOST_REQUIRE(pose_only != nullptr);
  BOOST_REQUIRE(lidar != nullptr);
  BOOST_CHECK(propagated->kind ==
              OnlineDualPointCorrectionKind::OWNER_PROPAGATED_DELTA);
  BOOST_CHECK(older->kind ==
              OnlineDualPointCorrectionKind::OWNER_OLDER_WINDOW);
  BOOST_CHECK(pose_only->kind ==
              OnlineDualPointCorrectionKind::OWNER_POSE_ONLY);
  BOOST_CHECK(lidar->kind == OnlineDualPointCorrectionKind::LIDAR_UNCHANGED);
}

BOOST_AUTO_TEST_CASE(LineagePrefersDirectInputsThenEarlierOwner) {
  OnlineDualPointLineage lineage;
  lineage.final_point3D_id = 50;
  lineage.ancestors = {
      LineageAncestor(1, 1, false),
      LineageAncestor(9, 9, true),
      LineageAncestor(7, 7, true),
      LineageAncestor(2, 2, false)};
  const OnlineDualPointLineageResult result =
      ResolveOnlineDualPointLineage({lineage});
  BOOST_REQUIRE(result.IsSuccess());
  BOOST_REQUIRE_EQUAL(result.owners.size(), 1);
  BOOST_CHECK(result.owners.front().directly_optimized);
  BOOST_CHECK_EQUAL(result.owners.front().owner_image_id, 7);
  BOOST_CHECK_EQUAL(result.owners.front().owner_registration_sequence, 7);
  BOOST_CHECK_EQUAL(std::string(ToString(OnlineDualResultClass::DUAL_FATAL)),
                    "DUAL_FATAL");
  BOOST_CHECK_EQUAL(
      std::string(ToString(
          OnlineDualPointCorrectionKind::OWNER_PROPAGATED_DELTA)),
      "OWNER_PROPAGATED_DELTA");
}
