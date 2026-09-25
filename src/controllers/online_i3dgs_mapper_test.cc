#define TEST_NAME "controllers/online_i3dgs_mapper_test"
#include "util/testing.h"

#include "controllers/online_i3dgs_mapper.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace colmap;

namespace {

using PairKey = std::pair<image_t, image_t>;

PairKey MakePairKey(const image_t image_id1, const image_t image_id2) {
  return image_id1 < image_id2 ? PairKey(image_id1, image_id2)
                              : PairKey(image_id2, image_id1);
}

KnownPoseSE3 PoseFromCenter(const Eigen::Vector3d& center) {
  return KnownPoseSE3(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), -center);
}

OnlineI3dgsFrameInput Frame(const size_t frame_index,
                            const Eigen::Vector3d& center =
                                Eigen::Vector3d::Zero()) {
  OnlineI3dgsFrameInput frame;
  frame.frame_index = frame_index;
  frame.image_id = static_cast<image_t>(frame_index);
  const std::string suffix = std::to_string(frame_index);
  frame.image_path = "imgs_" + suffix + ".jpg";
  frame.camera_path = "imgs_" + suffix + ".CAM";
  frame.odometry_path = "odoms_" + suffix + ".txt";
  frame.scan_path = "scans_" + suffix + ".pcd";
  frame.image_sha256 = "image-" + suffix;
  frame.camera_sha256 = "camera-" + suffix;
  frame.odometry_sha256 = "odom-" + suffix;
  frame.scan_sha256 = "scan-" + suffix;
  frame.scan_size_bytes = 1;
  frame.fastlio_T_cw = PoseFromCenter(center);
  return frame;
}

class FakeState : public OnlineI3dgsStateDependency {
 public:
  const KnownPoseRegistry& KnownPoses() const override { return registry; }
  const ActiveCovisibilityGraph& ActiveGraph() const override { return graph; }
  OnlineMapperCanonicalVersion CanonicalVersion() const override {
    OnlineMapperCanonicalVersion version;
    version.known_pose_registry = registry.Version();
    version.active_covisibility_graph = graph.Version();
    return version;
  }

  OnlineI3dgsKnownPoseRegistrationResult RegisterKnownPose(
      const OnlineI3dgsFrameInput& frame) override {
    OnlineI3dgsKnownPoseRegistrationResult result;
    const KnownPoseRegistryResult registration = registry.AddKnownPose(
        frame.image_id, frame.frame_index, frame.fastlio_T_cw);
    result.success = registration.IsSuccess();
    result.detail = registration.detail;
    result.registration_sequence = registration.registration_sequence;
    return result;
  }

  OnlineI3dgsCatchupFailureQueryResult GetCatchupFailure(
      const image_t image_id) const override {
    OnlineI3dgsCatchupFailureQueryResult result;
    const KnownPoseRecordQueryResult record = registry.GetByImageId(image_id);
    result.success = record.IsSuccess();
    result.detail = record.detail;
    const auto found = catchup_failures.find(image_id);
    result.has_evidence = found != catchup_failures.end();
    if (result.has_evidence) result.evidence = found->second;
    return result;
  }

  OnlineI3dgsDependencyResult RecordCatchupFailure(
      const image_t image_id,
      const OnlineI3dgsCatchupFailureEvidence& evidence) override {
    const CatchupFailureEvidence registry_evidence(
        evidence.active_edge_evidence_version,
        evidence.trigger_submitted_lidar_constraint_count,
        evidence.lidar_map_version);
    const KnownPoseRegistryResult result =
        registry.RecordCatchupFailure(image_id, registry_evidence);
    if (result.IsSuccess()) catchup_failures[image_id] = evidence;
    return {result.IsSuccess(), result.detail};
  }

  uint64_t GlobalBaCallCount() const override { return global_ba_calls; }

  KnownPoseRegistry registry;
  ActiveCovisibilityGraph graph;
  std::map<image_t, OnlineI3dgsCatchupFailureEvidence> catchup_failures;
  uint64_t global_ba_calls = 0;
};

class FakeFrontend : public OnlineI3dgsFrontendDependency {
 public:
  OnlineI3dgsDependencyResult IngestFrame(
      const OnlineI3dgsFrameInput& frame,
      const size_t max_visible_frame_index) override {
    ingested_frames.push_back(frame.frame_index);
    visibility_limits.push_back(max_visible_frame_index);
    observed_paths.emplace_back(frame.frame_index, frame.image_path);
    return {true, ""};
  }

  OnlineI3dgsMatchBatchResult MatchExplicitReferences(
      const image_t current_image_id,
      const std::vector<image_t>& ordered_reference_image_ids,
      const size_t max_visible_frame_index) override {
    match_batches.push_back(ordered_reference_image_ids);
    visibility_limits.push_back(max_visible_frame_index);
    OnlineI3dgsMatchBatchResult result;
    result.success = true;
    for (const image_t reference_image_id : ordered_reference_image_ids) {
      OnlineI3dgsPairMatch pair;
      pair.reference_image_id = reference_image_id;
      pair.verified_inlier_count =
          supports[MakePairKey(current_image_id, reference_image_id)];
      pair.raw_match_count = pair.verified_inlier_count;
      result.pairs.push_back(pair);
    }
    return result;
  }

  void SetSupport(const image_t image_id1,
                  const image_t image_id2,
                  const size_t verified_inliers = 20) {
    supports[MakePairKey(image_id1, image_id2)] = verified_inliers;
  }

  std::map<PairKey, size_t> supports;
  std::vector<size_t> ingested_frames;
  std::vector<size_t> visibility_limits;
  std::vector<std::pair<size_t, std::string>> observed_paths;
  std::vector<std::vector<image_t>> match_batches;
};

class FakeLidar : public OnlineI3dgsLidarDependency {
 public:
  OnlineI3dgsLidarIngestResult IngestScan(
      const OnlineI3dgsFrameInput& frame,
      const size_t max_visible_frame_index) override {
    visibility_limits.push_back(max_visible_frame_index);
    opened_paths.emplace_back(frame.frame_index, frame.scan_path);
    OnlineI3dgsLidarIngestResult result;
    result.success = true;
    result.opened_scan_index = frame.frame_index;
    result.map_version_before = map_version;
    result.map_version_after = ++map_version;
    result.max_scan_index = frame.frame_index;
    result.point_transform_count_min = 1;
    result.point_transform_count_max = 1;
    result.normal_transform_count_min = 1;
    result.normal_transform_count_max = 1;
    result.snapshot_sha256 = "snapshot-" + std::to_string(map_version);
    result.geometry_sha256 = "geometry-" + std::to_string(map_version);
    return result;
  }

  uint64_t map_version = 0;
  std::vector<size_t> visibility_limits;
  std::vector<std::pair<size_t, std::string>> opened_paths;
};

class FakePnP : public OnlineI3dgsPnPDependency {
 public:
  OnlineDualPnPDecisionResult ProbeTwoSides(
      const OnlineI3dgsPnPRequest& request) override {
    requests.push_back(request);
    if (callback) return callback(request);
    OnlineDualPnPDecisionResult result;
    result.result_class = OnlineDualResultClass::SINGLE_FALLBACK;
    result.decision = OnlineDualDecision::SINGLE;
    result.reason = OnlineDualReason::PNP_SOLVER_FAILED;
    result.detail = "injected PnP failure";
    return result;
  }

  static OnlineDualPnPDecisionResult Success() {
    OnlineDualPnPDecisionResult result;
    result.result_class = OnlineDualResultClass::SUCCESS;
    result.decision = OnlineDualDecision::DUAL_CANDIDATE;
    result.reason = OnlineDualReason::NONE;
    result.delta_init = OnlineDualSE3(
        Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
        Eigen::Vector3d(0.1, 0.0, 0.0));
    for (const OnlineDualClusterId cluster_id :
         {OnlineDualClusterId::FIRST, OnlineDualClusterId::SECOND}) {
      OnlineDualPnPProbeAudit probe;
      probe.cluster_id = cluster_id;
      probe.status = OnlineDualPnPProbeStatus::SUCCESS;
      probe.correspondence_count = 120;
      probe.inlier_count = 30;
      probe.inlier_ratio = 0.25;
      probe.pose_is_valid_se3 = true;
      Eigen::Vector3d translation = Eigen::Vector3d::Zero();
      if (cluster_id == OnlineDualClusterId::SECOND) {
        translation = Eigen::Vector3d(0.1, 0.0, 0.0);
      }
      probe.probe_T_cw = OnlineDualSE3(
          Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
          translation);
      result.probes.push_back(probe);
    }
    return result;
  }

  std::function<OnlineDualPnPDecisionResult(const OnlineI3dgsPnPRequest&)>
      callback;
  std::vector<OnlineI3dgsPnPRequest> requests;
};

class FakeCandidate : public OnlineI3dgsBaCandidate {};

class FakeBa : public OnlineI3dgsLocalBaDependency {
 public:
  explicit FakeBa(FakeState* state) : state_(state) {}

  OnlineI3dgsBaExecution Execute(
      const OnlineI3dgsBaRequest& request) override {
    requests.push_back(request);
    if (execute_observer) execute_observer(request);
    if (callback) return callback(request);
    return SuccessfulExecution(request);
  }

  OnlineI3dgsCommitResult Commit(
      const OnlineI3dgsCommitRequest& request,
      OnlineI3dgsBaExecution* execution) override {
    commit_requests.push_back(request);
    OnlineI3dgsCommitResult result;
    result.registry_version_before = state_->registry.Version();
    result.registry_version_after = result.registry_version_before;
    result.graph_version_before = state_->graph.Version();
    result.graph_version_after = result.graph_version_before;
    result.canonical_version_before = state_->CanonicalVersion();
    result.canonical_version_after = result.canonical_version_before;
    if (commit_callback) return commit_callback(request, execution, result);
    if (execution == nullptr || execution->candidate == nullptr ||
        request.expected_canonical_version !=
            result.canonical_version_before ||
        request.expected_registry_version != result.registry_version_before ||
        request.expected_graph_version != result.graph_version_before) {
      result.detail = "invalid fake commit input";
      return result;
    }

    KnownPoseRegistryCommit registry_commit;
    registry_commit.expected_version = request.expected_registry_version;
    const std::set<image_t> promoted(request.promoted_image_ids.begin(),
                                     request.promoted_image_ids.end());
    for (const image_t image_id : request.promoted_image_ids) {
      const auto pose = std::find_if(
          request.pose_updates.begin(),
          request.pose_updates.end(),
          [image_id](const OnlineI3dgsImagePoseUpdate& update) {
            return update.image_id == image_id;
          });
      if (pose == request.pose_updates.end()) {
        result.detail = "missing promoted pose";
        return result;
      }
      registry_commit.promotions.emplace_back(image_id, pose->latest_T_cw);
    }
    for (const OnlineI3dgsImagePoseUpdate& pose : request.pose_updates) {
      if (promoted.count(pose.image_id) == 0) {
        registry_commit.latest_pose_updates.emplace_back(pose.image_id,
                                                         pose.latest_T_cw);
      }
    }
    ActiveCovisibilityCommit graph_commit;
    graph_commit.expected_version = request.expected_graph_version;
    for (const image_t image_id : request.promoted_image_ids) {
      const KnownPoseRecordQueryResult record =
          state_->registry.GetByImageId(image_id);
      if (!record.IsSuccess()) {
        result.detail = "missing promoted registry record";
        return result;
      }
      graph_commit.nodes.emplace_back(
          image_id,
          record.record.registration_sequence,
          ActiveCovisibilityNodeState::VISUAL_ACTIVE);
    }
    graph_commit.ordinary_edges = request.ordinary_edges;
    graph_commit.loop_edges = request.loop_edges;
    const KnownPoseRegistryResult registry_validation =
        state_->registry.ValidateCommit(registry_commit);
    const ActiveCovisibilityResult graph_validation =
        state_->graph.ValidateCommit(graph_commit);
    if (!registry_validation.IsSuccess() || !graph_validation.IsSuccess()) {
      result.detail = registry_validation.IsSuccess()
                          ? graph_validation.detail
                          : registry_validation.detail;
      return result;
    }
    const KnownPoseRegistryResult registry_result =
        state_->registry.Commit(registry_commit);
    const ActiveCovisibilityResult graph_result =
        state_->graph.Commit(graph_commit);
    result.success = registry_result.IsSuccess() && graph_result.IsSuccess();
    result.detail = result.success ? "" : "fake publication failed";
    result.atomic_publication = result.success;
    result.registry_version_after = state_->registry.Version();
    result.graph_version_after = state_->graph.Version();
    result.canonical_version_after = state_->CanonicalVersion();
    return result;
  }

  OnlineI3dgsBaExecution SuccessfulExecution(
      const OnlineI3dgsBaRequest& request) const {
    OnlineI3dgsBaExecution execution;
    execution.disposition = OnlineI3dgsBaDisposition::SUCCESS;
    execution.failure_reason = OnlineI3dgsBaFailureReason::NONE;
    for (size_t pass_index = 0; pass_index < request.required_pass_count;
         ++pass_index) {
      OnlineI3dgsBaPassAudit pass;
      pass.pass_index = static_cast<uint32_t>(pass_index + 1);
      pass.solve_native_invoked = true;
      pass.solve_native_success = true;
      pass.finite_costs = true;
      pass.finite_output = true;
      pass.submitted_visual_residual_count = 100;
      pass.submitted_lidar_constraint_count = 100;
      pass.trigger_submitted_lidar_constraint_count = 50;
      pass.executed_backend = "custom_cuda";
      pass.executed_problem_source = "native_graph";
      pass.termination = "CONVERGENCE";
      pass.postprocess_count =
          pass_index + 1 == request.required_pass_count ? 1 : 0;
      execution.passes.push_back(pass);
    }
    for (const image_t image_id : request.frozen_image_ids) {
      const KnownPoseRecordQueryResult record =
          state_->registry.GetByImageId(image_id);
      if (!record.IsSuccess()) {
        execution.disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
        execution.failure_reason = OnlineI3dgsBaFailureReason::INVALID_RESULT;
        execution.detail = "fake BA window image is absent from registry";
        return execution;
      }
      execution.pose_updates.push_back(
          {image_id, record.record.latest_T_cw});
      execution.per_image_submitted_lidar_constraint_counts.push_back(
          {image_id, 50});
    }
    execution.total_postprocess_count = 1;
    execution.propagation_applied =
        request.mode == OnlineI3dgsMapperMode::DUAL;
    execution.candidate.reset(new FakeCandidate());
    return execution;
  }

  static OnlineI3dgsBaExecution RecoverableFailure(
      const OnlineI3dgsBaFailureReason reason,
      const uint64_t trigger_count = 0) {
    OnlineI3dgsBaExecution execution;
    execution.disposition = OnlineI3dgsBaDisposition::RECOVERABLE_FAILURE;
    execution.failure_reason = reason;
    execution.detail = ToString(reason);
    OnlineI3dgsBaPassAudit pass;
    pass.pass_index = 1;
    pass.trigger_submitted_lidar_constraint_count = trigger_count;
    execution.passes.push_back(pass);
    return execution;
  }

  FakeState* state_;
  std::function<OnlineI3dgsBaExecution(const OnlineI3dgsBaRequest&)> callback;
  std::function<void(const OnlineI3dgsBaRequest&)> execute_observer;
  std::function<OnlineI3dgsCommitResult(
      const OnlineI3dgsCommitRequest&,
      OnlineI3dgsBaExecution*,
      OnlineI3dgsCommitResult)>
      commit_callback;
  std::vector<OnlineI3dgsBaRequest,
              Eigen::aligned_allocator<OnlineI3dgsBaRequest>>
      requests;
  std::vector<OnlineI3dgsCommitRequest> commit_requests;
};

struct Harness {
  explicit Harness(const size_t frame_count, const size_t ba_window_size = 20)
      : ba(&state) {
    OnlineI3dgsMapperOptions options;
    options.expected_frame_count = frame_count;
    options.ba_window_size = ba_window_size;
    OnlineI3dgsMapperDependencies dependencies;
    dependencies.frontend = &frontend;
    dependencies.lidar = &lidar;
    dependencies.pnp = &pnp;
    dependencies.local_ba = &ba;
    dependencies.state = &state;
    mapper.reset(new OnlineI3dgsMapper(options, dependencies));
  }

  FakeState state;
  FakeFrontend frontend;
  FakeLidar lidar;
  FakePnP pnp;
  FakeBa ba;
  std::unique_ptr<OnlineI3dgsMapper> mapper;
};

void RequireProcessed(const OnlineI3dgsFrameResult& result) {
  BOOST_REQUIRE_MESSAGE(result.IsSuccess(), result.audit.detail);
}

Eigen::Vector3d CircleCenter(const size_t image_id) {
  const double angle =
      2.0 * std::acos(-1.0) * static_cast<double>(image_id - 1) / 12.0;
  return Eigen::Vector3d(std::cos(angle), std::sin(angle), 0.0);
}

void BuildThirteenImageActiveChain(Harness* harness) {
  for (image_t image_id = 2; image_id <= 13; ++image_id) {
    harness->frontend.SetSupport(image_id, image_id - 1);
  }
  for (size_t frame_index = 1; frame_index <= 13; ++frame_index) {
    const OnlineI3dgsFrameResult result = harness->mapper->ProcessArrivedFrame(
        Frame(frame_index, CircleCenter(frame_index)));
    RequireProcessed(result);
    if (frame_index > 1) {
      BOOST_REQUIRE_MESSAGE(result.audit.current_visual_active,
                            "failed to activate chain image " << frame_index);
    }
  }
}

}  // namespace

BOOST_AUTO_TEST_CASE(TenImageWindowCapsSingleAndCatchupRequests) {
  Harness harness(14, 10);
  for (image_t image_id = 2; image_id <= 14; ++image_id) {
    harness.frontend.SetSupport(image_id, image_id - 1);
  }
  harness.frontend.SetSupport(14, 12);
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    if (request.trigger_image_id == 13 && request.frame_index == 13) {
      return FakeBa::RecoverableFailure(
          OnlineI3dgsBaFailureReason::
              TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM,
          0);
    }
    return harness.ba.SuccessfulExecution(request);
  };
  for (size_t frame_index = 1; frame_index <= 14; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(
        Frame(frame_index, Eigen::Vector3d(frame_index, 0.0, 0.0))));
  }
  bool saw_full_single = false;
  bool saw_full_catchup = false;
  for (const OnlineI3dgsBaRequest& request : harness.ba.requests) {
    BOOST_CHECK_LE(request.frozen_image_ids.size(), 10);
    if (request.frozen_image_ids.size() != 10) continue;
    saw_full_single |= request.mode == OnlineI3dgsMapperMode::SINGLE;
    saw_full_catchup |= request.mode == OnlineI3dgsMapperMode::CATCHUP;
  }
  BOOST_CHECK(saw_full_single);
  BOOST_CHECK(saw_full_catchup);
  BOOST_CHECK_EQUAL(harness.mapper->Summary().global_ba_call_count, 0);
}

BOOST_AUTO_TEST_CASE(TenImageWindowCapsDelayedBootstrap) {
  Harness harness(12, 10);
  for (image_t image_id = 2; image_id <= 12; ++image_id) {
    harness.frontend.SetSupport(image_id, image_id - 1);
  }
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    if (request.frame_index < 12) {
      return FakeBa::RecoverableFailure(
          OnlineI3dgsBaFailureReason::
              TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM,
          0);
    }
    return harness.ba.SuccessfulExecution(request);
  };
  for (size_t frame_index = 1; frame_index <= 12; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(
        Frame(frame_index, Eigen::Vector3d(frame_index, 0.0, 0.0))));
  }
  BOOST_REQUIRE(!harness.ba.requests.empty());
  BOOST_CHECK(harness.ba.requests.back().mode == OnlineI3dgsMapperMode::BOOTSTRAP);
  BOOST_CHECK_EQUAL(harness.ba.requests.back().frozen_image_ids.size(), 10);
  BOOST_CHECK_EQUAL(harness.state.graph.NumNodes(), 10);
}

BOOST_AUTO_TEST_CASE(CausalInputsPoseOnlyFailureAndGlobalBaRemainExplicit) {
  Harness harness(4);
  harness.frontend.SetSupport(2, 1);
  harness.frontend.SetSupport(3, 2);
  harness.frontend.SetSupport(4, 2);
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    if ((request.mode == OnlineI3dgsMapperMode::SINGLE ||
         request.mode == OnlineI3dgsMapperMode::CATCHUP) &&
        request.trigger_image_id == 3) {
      return FakeBa::RecoverableFailure(
          OnlineI3dgsBaFailureReason::
              TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM,
          0);
    }
    return harness.ba.SuccessfulExecution(request);
  };

  for (size_t frame_index = 1; frame_index <= 4; ++frame_index) {
    const OnlineI3dgsFrameResult result =
        harness.mapper->ProcessArrivedFrame(Frame(frame_index));
    RequireProcessed(result);
    BOOST_CHECK_EQUAL(result.audit.max_visible_frame_index, frame_index);
    BOOST_CHECK_EQUAL(result.audit.max_scan_index, frame_index);
  }
  const KnownPoseRecordQueryResult third =
      harness.state.registry.GetByImageId(3);
  BOOST_REQUIRE(third.IsSuccess());
  BOOST_CHECK(third.record.visual_state == KnownPoseVisualState::POSE_ONLY);
  const KnownPoseRecordQueryResult fourth =
      harness.state.registry.GetByImageId(4);
  BOOST_REQUIRE(fourth.IsSuccess());
  BOOST_CHECK(fourth.record.visual_state ==
              KnownPoseVisualState::VISUAL_ACTIVE);
  BOOST_CHECK_EQUAL(harness.frontend.ingested_frames.size(), 4);
  BOOST_CHECK_EQUAL(harness.lidar.opened_paths.size(), 4);
  for (size_t index = 0; index < 4; ++index) {
    BOOST_CHECK_EQUAL(harness.frontend.ingested_frames[index], index + 1);
    BOOST_CHECK_EQUAL(harness.lidar.opened_paths[index].first, index + 1);
    BOOST_CHECK_EQUAL(harness.lidar.visibility_limits[index], index + 1);
  }
  BOOST_CHECK_EQUAL(harness.mapper->Summary().global_ba_call_count, 0);
  BOOST_CHECK_EQUAL(harness.mapper->Summary().pnp_registration_call_count, 0);
}

BOOST_AUTO_TEST_CASE(MatchingExtendsFromFiveToTenOnlyBeyondTenFiniteHops) {
  Harness harness(14);
  BuildThirteenImageActiveChain(&harness);
  for (image_t reference_image_id = 1; reference_image_id <= 13;
       ++reference_image_id) {
    harness.frontend.SetSupport(14, reference_image_id);
  }
  const OnlineI3dgsFrameResult result = harness.mapper->ProcessArrivedFrame(
      Frame(14, CircleCenter(1)));
  RequireProcessed(result);
  BOOST_CHECK(result.audit.maximum_finite_hop_defined);
  BOOST_CHECK_GT(result.audit.maximum_finite_hop, 10);
  BOOST_CHECK(result.audit.matching_extended_to_ten);
  BOOST_CHECK_EQUAL(result.audit.first_batch_reference_image_ids.size(), 5);
  BOOST_CHECK_EQUAL(result.audit.second_batch_reference_image_ids.size(), 5);
  BOOST_CHECK_EQUAL(result.audit.matched_pairs.size(), 10);
}

BOOST_AUTO_TEST_CASE(CatchupRunsOldestFirstAndCurrentUsesLatestCanonicalState) {
  Harness harness(5);
  harness.frontend.SetSupport(2, 1);
  harness.frontend.SetSupport(3, 2);
  harness.frontend.SetSupport(4, 2);
  harness.frontend.SetSupport(5, 3);
  harness.frontend.SetSupport(5, 4);
  std::map<std::tuple<size_t, OnlineI3dgsMapperMode, image_t>, size_t> calls;
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    const auto key = std::make_tuple(request.frame_index,
                                     request.mode,
                                     request.trigger_image_id);
    ++calls[key];
    if ((request.frame_index == 3 &&
         request.mode == OnlineI3dgsMapperMode::SINGLE) ||
        (request.frame_index == 4 &&
         request.mode == OnlineI3dgsMapperMode::CATCHUP) ||
        (request.frame_index == 4 &&
         request.mode == OnlineI3dgsMapperMode::SINGLE)) {
      return FakeBa::RecoverableFailure(
          OnlineI3dgsBaFailureReason::
              TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM,
          0);
    }
    return harness.ba.SuccessfulExecution(request);
  };

  for (size_t frame_index = 1; frame_index <= 4; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(frame_index)));
  }
  const OnlineI3dgsFrameResult fifth =
      harness.mapper->ProcessArrivedFrame(Frame(5));
  RequireProcessed(fifth);
  std::vector<image_t> promoted;
  for (const OnlineI3dgsCatchupAudit& catchup : fifth.audit.catchup) {
    if (catchup.promoted) promoted.push_back(catchup.image_id);
  }
  const std::vector<image_t> expected{3, 4};
  BOOST_CHECK(promoted == expected);
  BOOST_CHECK(fifth.audit.initial_effective_active_reference_image_ids.empty());
  BOOST_CHECK(fifth.audit.recomputed_effective_active_reference_image_ids ==
              expected);
  BOOST_CHECK(fifth.audit.current_recomputed_after_catchup);
  BOOST_CHECK_GT(fifth.audit.current_recomputed_graph_version,
                 fifth.audit.event_start_graph_version);
  BOOST_CHECK(fifth.audit.mode == OnlineI3dgsMapperMode::SINGLE);
  BOOST_CHECK(fifth.audit.current_visual_active);
  BOOST_CHECK_EQUAL(
      calls[std::make_tuple(5, OnlineI3dgsMapperMode::CATCHUP, 3)], 1);
  BOOST_CHECK_EQUAL(
      calls[std::make_tuple(5, OnlineI3dgsMapperMode::CATCHUP, 4)], 1);
}

BOOST_AUTO_TEST_CASE(CatchupFailureRetainsPreflightEvidenceWithoutSubmission) {
  Harness harness(4);
  harness.frontend.SetSupport(2, 1);
  harness.frontend.SetSupport(3, 2);
  harness.frontend.SetSupport(4, 2);
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    if (request.trigger_image_id == 3) {
      auto failure = FakeBa::RecoverableFailure(
          OnlineI3dgsBaFailureReason::
              TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM, 0);
      failure.passes.front().trigger_preflight_checked = true;
      failure.passes.front().trigger_preflight_lidar_constraint_count = 49;
      return failure;
    }
    return harness.ba.SuccessfulExecution(request);
  };
  for (size_t frame_index = 1; frame_index <= 4; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(frame_index)));
  }
  const auto failure = harness.state.GetCatchupFailure(3);
  BOOST_REQUIRE(failure.success);
  BOOST_REQUIRE(failure.has_evidence);
  BOOST_CHECK_EQUAL(failure.evidence.trigger_submitted_lidar_constraint_count, 49);
  BOOST_CHECK(!harness.state.graph.HasNode(3));
}

BOOST_AUTO_TEST_CASE(DualPnPFailureDiscardsDualAndFallsBackOnceToSingle) {
  Harness harness(14);
  BuildThirteenImageActiveChain(&harness);
  for (image_t reference_image_id = 1; reference_image_id <= 13;
       ++reference_image_id) {
    harness.frontend.SetSupport(14, reference_image_id);
  }
  const OnlineI3dgsFrameResult result = harness.mapper->ProcessArrivedFrame(
      Frame(14, CircleCenter(1)));
  RequireProcessed(result);
  BOOST_REQUIRE_EQUAL(harness.pnp.requests.size(), 1);
  BOOST_CHECK(result.audit.dual_probe_attempted);
  BOOST_CHECK(result.audit.dual_fell_back_to_single);
  BOOST_CHECK(result.audit.mode == OnlineI3dgsMapperMode::SINGLE);
  size_t current_single_calls = 0;
  size_t current_dual_calls = 0;
  for (const OnlineI3dgsBaRequest& request : harness.ba.requests) {
    if (request.trigger_image_id != 14) continue;
    if (request.mode == OnlineI3dgsMapperMode::SINGLE) ++current_single_calls;
    if (request.mode == OnlineI3dgsMapperMode::DUAL) ++current_dual_calls;
  }
  BOOST_CHECK_EQUAL(current_single_calls, 1);
  BOOST_CHECK_EQUAL(current_dual_calls, 0);
}

BOOST_AUTO_TEST_CASE(DualPassFailureIsFatalAndPublishesNothing) {
  Harness harness(14);
  BuildThirteenImageActiveChain(&harness);
  for (image_t reference_image_id = 1; reference_image_id <= 13;
       ++reference_image_id) {
    harness.frontend.SetSupport(14, reference_image_id);
  }
  harness.pnp.callback = [](const OnlineI3dgsPnPRequest&) {
    return FakePnP::Success();
  };
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    if (request.mode != OnlineI3dgsMapperMode::DUAL) {
      return harness.ba.SuccessfulExecution(request);
    }
    OnlineI3dgsBaExecution failure = FakeBa::RecoverableFailure(
        OnlineI3dgsBaFailureReason::SOLVE_NATIVE_FAILED, 50);
    failure.passes.front().solve_native_invoked = true;
    failure.passes.front().solve_native_success = false;
    return failure;
  };
  const size_t commit_count_before = harness.ba.commit_requests.size();
  const OnlineI3dgsFrameResult result = harness.mapper->ProcessArrivedFrame(
      Frame(14, CircleCenter(1)));
  BOOST_CHECK(result.status == OnlineI3dgsMapperStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(harness.ba.commit_requests.size(), commit_count_before);
  BOOST_CHECK(!harness.state.graph.HasNode(14));
  const KnownPoseRecordQueryResult current =
      harness.state.registry.GetByImageId(14);
  BOOST_REQUIRE(current.IsSuccess());
  BOOST_CHECK(current.record.visual_state == KnownPoseVisualState::POSE_ONLY);
}

BOOST_AUTO_TEST_CASE(DualUsesOneCandidateRequestAndOneLoopLastCommit) {
  Harness harness(14);
  BuildThirteenImageActiveChain(&harness);
  for (image_t reference_image_id = 1; reference_image_id <= 13;
       ++reference_image_id) {
    harness.frontend.SetSupport(14, reference_image_id);
  }
  harness.pnp.callback = [](const OnlineI3dgsPnPRequest&) {
    return FakePnP::Success();
  };
  const size_t commit_count_before = harness.ba.commit_requests.size();
  const OnlineI3dgsFrameResult result = harness.mapper->ProcessArrivedFrame(
      Frame(14, CircleCenter(1)));
  RequireProcessed(result);
  BOOST_CHECK(result.audit.mode == OnlineI3dgsMapperMode::DUAL);
  BOOST_REQUIRE_EQUAL(result.audit.ba_passes.size(), 2);
  BOOST_CHECK_EQUAL(result.audit.ba_passes[0].postprocess_count, 0);
  BOOST_CHECK_EQUAL(result.audit.ba_passes[1].postprocess_count, 1);
  BOOST_REQUIRE_EQUAL(harness.ba.commit_requests.size(),
                      commit_count_before + 1);
  const OnlineI3dgsCommitRequest& commit = harness.ba.commit_requests.back();
  BOOST_REQUIRE(!commit.ordinary_edges.empty());
  BOOST_REQUIRE(!commit.loop_edges.empty());
  BOOST_REQUIRE_EQUAL(commit.logical_write_order.size(), 5);
  BOOST_CHECK(commit.logical_write_order.back() ==
              OnlineI3dgsCommitWriteKind::LOOP_EDGES);
  const OnlineI3dgsBaRequest& request = harness.ba.requests.back();
  BOOST_CHECK(request.mode == OnlineI3dgsMapperMode::DUAL);
  BOOST_CHECK_EQUAL(request.required_pass_count, 2);
  BOOST_CHECK_EQUAL(request.solve_api, "BundleAdjuster::SolveNative");
  BOOST_CHECK_EQUAL(harness.mapper->Summary().dual_atomic_commit_count, 1);
}

BOOST_AUTO_TEST_CASE(FlushPromotesDeferredCurrentEdgesWithoutRunningDual) {
  Harness harness(4);
  harness.frontend.SetSupport(2, 1);
  harness.frontend.SetSupport(4, 2);
  harness.frontend.SetSupport(4, 3);
  for (size_t frame_index = 1; frame_index <= 4; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(frame_index)));
  }
  BOOST_CHECK(!harness.state.graph.HasNode(3));
  BOOST_CHECK(harness.state.graph.HasNode(4));
  const OnlineI3dgsFinishResult finish = harness.mapper->Finish();
  BOOST_REQUIRE_MESSAGE(finish.IsComplete(), finish.summary.incomplete_reason);
  BOOST_CHECK(harness.state.graph.HasNode(3));
  BOOST_CHECK_EQUAL(finish.summary.visual_active_after_flush_count, 4);
  BOOST_CHECK_EQUAL(finish.summary.pose_only_final_count, 0);
  BOOST_CHECK_EQUAL(finish.summary.global_ba_call_count, 0);
  BOOST_REQUIRE(!finish.flush_attempts.empty());
  BOOST_CHECK_EQUAL(finish.flush_attempts.front().catchup.image_id, 3);
  BOOST_CHECK(finish.flush_attempts.front().catchup.promoted);
  for (const OnlineI3dgsBaRequest& request : harness.ba.requests) {
    if (request.is_flush) {
      BOOST_CHECK(request.mode == OnlineI3dgsMapperMode::CATCHUP);
      BOOST_CHECK_EQUAL(request.required_pass_count, 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(CanonicalStateChangesOnlyAtUnifiedCommitBoundary) {
  Harness harness(2);
  harness.frontend.SetSupport(2, 1);
  bool observed_detached_candidate = false;
  harness.ba.execute_observer = [&](const OnlineI3dgsBaRequest& request) {
    if (request.mode != OnlineI3dgsMapperMode::BOOTSTRAP) return;
    const KnownPoseRecordQueryResult first =
        harness.state.registry.GetByImageId(1);
    const KnownPoseRecordQueryResult second =
        harness.state.registry.GetByImageId(2);
    BOOST_REQUIRE(first.IsSuccess());
    BOOST_REQUIRE(second.IsSuccess());
    BOOST_CHECK(first.record.visual_state == KnownPoseVisualState::POSE_ONLY);
    BOOST_CHECK(second.record.visual_state == KnownPoseVisualState::POSE_ONLY);
    BOOST_CHECK_EQUAL(harness.state.graph.NumNodes(), 0);
    observed_detached_candidate = true;
  };
  RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(1)));
  const OnlineI3dgsFrameResult second =
      harness.mapper->ProcessArrivedFrame(Frame(2));
  RequireProcessed(second);
  BOOST_CHECK(observed_detached_candidate);
  BOOST_REQUIRE_EQUAL(harness.ba.commit_requests.size(), 1);
  const OnlineI3dgsCommitRequest& commit = harness.ba.commit_requests.front();
  BOOST_REQUIRE_EQUAL(commit.logical_write_order.size(), 3);
  BOOST_CHECK(commit.logical_write_order.front() ==
              OnlineI3dgsCommitWriteKind::STATE_AND_ACTIVE_NODE);
  BOOST_CHECK(commit.logical_write_order.back() ==
              OnlineI3dgsCommitWriteKind::RECONSTRUCTION_AND_OWNERS);
  BOOST_CHECK(harness.state.graph.HasNode(1));
  BOOST_CHECK(harness.state.graph.HasNode(2));
  BOOST_CHECK_EQUAL(harness.state.registry.Version(),
                    commit.expected_registry_version + 1);
  BOOST_CHECK_EQUAL(harness.state.graph.Version(),
                    commit.expected_graph_version + 1);
}

BOOST_AUTO_TEST_CASE(UsableIterationLimitedBaCommitsBootstrapAndSingle) {
  Harness harness(3);
  harness.frontend.SetSupport(2, 1);
  harness.frontend.SetSupport(3, 2);
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    OnlineI3dgsBaExecution execution = harness.ba.SuccessfulExecution(request);
    for (OnlineI3dgsBaPassAudit& pass : execution.passes) {
      pass.termination = "NO_CONVERGENCE";
    }
    return execution;
  };
  for (size_t frame_index = 1; frame_index <= 3; ++frame_index) {
    RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(frame_index)));
  }
  BOOST_CHECK(harness.state.graph.HasNode(1));
  BOOST_CHECK(harness.state.graph.HasNode(2));
  BOOST_CHECK(harness.state.graph.HasNode(3));
  BOOST_REQUIRE_EQUAL(harness.ba.commit_requests.size(), 2);
  BOOST_CHECK(harness.ba.requests.back().mode == OnlineI3dgsMapperMode::SINGLE);
}

BOOST_AUTO_TEST_CASE(UnusableBaTerminationCannotPublishCandidate) {
  Harness harness(2);
  harness.frontend.SetSupport(2, 1);
  harness.ba.callback = [&](const OnlineI3dgsBaRequest& request) {
    OnlineI3dgsBaExecution execution = harness.ba.SuccessfulExecution(request);
    execution.passes.front().termination = "FAILURE";
    return execution;
  };
  RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(1)));
  const OnlineI3dgsFrameResult second =
      harness.mapper->ProcessArrivedFrame(Frame(2));
  BOOST_CHECK(!second.IsSuccess());
  BOOST_CHECK(harness.ba.commit_requests.empty());
  BOOST_CHECK_EQUAL(harness.state.graph.NumNodes(), 0);
}

BOOST_AUTO_TEST_CASE(GlobalBaGuardTerminatesRunBeforeReadingNextFrame) {
  Harness harness(2);
  RequireProcessed(harness.mapper->ProcessArrivedFrame(Frame(1)));
  harness.state.global_ba_calls = 1;
  const OnlineI3dgsFrameResult second =
      harness.mapper->ProcessArrivedFrame(Frame(2));
  BOOST_CHECK(second.status == OnlineI3dgsMapperStatus::INCOMPLETE);
  BOOST_CHECK_EQUAL(harness.frontend.ingested_frames.size(), 1);
  BOOST_CHECK_EQUAL(harness.lidar.opened_paths.size(), 1);
  BOOST_CHECK_EQUAL(harness.mapper->Summary().global_ba_call_count, 1);
}
