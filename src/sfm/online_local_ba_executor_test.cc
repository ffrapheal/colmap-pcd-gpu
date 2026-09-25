#define TEST_NAME "sfm/online_local_ba_executor"
#include "util/testing.h"

#ifdef GPU_BA_CUDA_ENABLED

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "sfm/incremental_triangulator.h"
#include "sfm/online_local_ba_executor.h"

namespace colmap {
namespace {

constexpr uint64_t kOwnerEpoch = 24001;
constexpr image_t kTriggerImageId = 1;
constexpr camera_t kCameraId = 1;
constexpr size_t kPointCount = 64;

class ExecutorTempPcd {
 public:
  ExecutorTempPcd() {
    static std::atomic<uint64_t> next_id{0};
    path_ = "/tmp/online_local_ba_executor_test_" +
            std::to_string(static_cast<uint64_t>(getpid())) + "_" +
            std::to_string(next_id.fetch_add(1)) + ".pcd";
  }

  ~ExecutorTempPcd() { std::remove(path_.c_str()); }

  ExecutorTempPcd(const ExecutorTempPcd&) = delete;
  ExecutorTempPcd& operator=(const ExecutorTempPcd&) = delete;

  const std::string& Path() const { return path_; }

  void Write() const {
    std::ofstream file(path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("cannot create test PCD");
    file << "VERSION 0.7\n"
         << "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
         << "SIZE 4 4 4 4 4 4 4 4\n"
         << "TYPE F F F F F F F F\n"
         << "COUNT 1 1 1 1 1 1 1 1\n"
         << "WIDTH 1\n"
         << "HEIGHT 1\n"
         << "POINTS 1\n"
         << "DATA ascii\n"
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << "3 -1 -2 1 0 0 0 0\n";
    if (!file) throw std::runtime_error("cannot write test PCD");
  }

 private:
  std::string path_;
};

lidar::IncrementalCausalLidarMapDependencies MapDependencies() {
  lidar::IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [](const std::vector<float>& support_xyz,
         const std::vector<float>& query_xyz,
         const float outer_radius,
         const float inner_radius,
         std::vector<float>* outer_normals,
         std::vector<float>* inner_normals,
         lidar::CudaNormalEstimationTiming* timing,
         std::string* error) {
        (void)support_xyz;
        (void)outer_radius;
        (void)inner_radius;
        if (outer_normals == nullptr || inner_normals == nullptr ||
            timing == nullptr || error == nullptr ||
            query_xyz.size() % 3 != 0) {
          return false;
        }
        outer_normals->clear();
        inner_normals->clear();
        for (size_t index = 0; index < query_xyz.size() / 3; ++index) {
          outer_normals->insert(outer_normals->end(),
                                {1.0f, 0.0f, 0.0f, 0.1f});
          inner_normals->insert(inner_normals->end(),
                                {0.0f, 1.0f, 0.0f, 0.2f});
        }
        *timing = lidar::CudaNormalEstimationTiming();
        error->clear();
        return true;
      };
  return dependencies;
}

std::shared_ptr<const lidar::LidarMapSnapshot> MakeSnapshot() {
  ExecutorTempPcd pcd;
  pcd.Write();
  lidar::IncrementalCausalLidarMap map(MapDependencies());
  lidar::ScanSource source;
  source.scan_index = 1;
  source.pcd_path = pcd.Path();
  source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
  std::string error;
  if (!map.AppendScan(source, nullptr, &error)) {
    throw std::runtime_error(error);
  }
  return map.GetSnapshot();
}

std::array<uint8_t, 32> DecodeSha256(const std::string& encoded) {
  if (encoded.size() != 64) throw std::runtime_error("invalid test SHA256");
  std::array<uint8_t, 32> decoded;
  decoded.fill(0);
  for (size_t index = 0; index < decoded.size(); ++index) {
    const auto nibble = [](const char value) -> uint8_t {
      if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
      }
      if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(value - 'a' + 10);
      }
      throw std::runtime_error("invalid test SHA256 digit");
    };
    decoded[index] = static_cast<uint8_t>(
        (nibble(encoded[index * 2]) << 4) | nibble(encoded[index * 2 + 1]));
  }
  return decoded;
}

BundleAdjustmentOptions StrictOptions() {
  BundleAdjustmentOptions options;
  options.ba_backend = "custom_cuda";
  options.ba_fallback_to_ceres = false;
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.refine_extrinsics = true;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.if_add_lidar_constraint = true;
  options.print_summary = false;
  return options;
}

struct CanonicalState {
  ReconstructionCanonicalVersion version;
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  std::vector<double> intrinsics;
  std::vector<std::array<double, 3>> points;
};

CanonicalState CaptureCanonicalState(const Reconstruction& reconstruction) {
  CanonicalState state;
  state.version = reconstruction.CanonicalVersion();
  state.qvec = reconstruction.Image(kTriggerImageId).Qvec();
  state.tvec = reconstruction.Image(kTriggerImageId).Tvec();
  state.intrinsics = reconstruction.Camera(kCameraId).Params();
  for (point3D_t point3D_id = 1; point3D_id <= kPointCount; ++point3D_id) {
    const Eigen::Vector3d& point = reconstruction.Point3D(point3D_id).XYZ();
    state.points.push_back({{point(0), point(1), point(2)}});
  }
  return state;
}

void CheckCanonicalState(const Reconstruction& reconstruction,
                         const CanonicalState& expected) {
  BOOST_CHECK(reconstruction.CanonicalVersion() == expected.version);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      reconstruction.Image(kTriggerImageId).Qvec().data(),
      reconstruction.Image(kTriggerImageId).Qvec().data() + 4,
      expected.qvec.data(),
      expected.qvec.data() + 4);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      reconstruction.Image(kTriggerImageId).Tvec().data(),
      reconstruction.Image(kTriggerImageId).Tvec().data() + 3,
      expected.tvec.data(),
      expected.tvec.data() + 3);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      reconstruction.Camera(kCameraId).Params().begin(),
      reconstruction.Camera(kCameraId).Params().end(),
      expected.intrinsics.begin(),
      expected.intrinsics.end());
  BOOST_REQUIRE_EQUAL(reconstruction.NumPoints3D(), expected.points.size());
  for (point3D_t point3D_id = 1; point3D_id <= kPointCount; ++point3D_id) {
    const Eigen::Vector3d& actual = reconstruction.Point3D(point3D_id).XYZ();
    const std::array<double, 3>& expected_point =
        expected.points[point3D_id - 1];
    BOOST_CHECK_EQUAL_COLLECTIONS(actual.data(), actual.data() + 3,
                                  expected_point.begin(), expected_point.end());
  }
}

class ExecutorFixture {
 public:
  ExecutorFixture() : snapshot(MakeSnapshot()) {
    Camera camera;
    camera.SetCameraId(kCameraId);
    camera.InitializeWithId(OpenCVCameraModel::model_id, 100.0, 100, 80);
    camera.SetParams({100.0, 100.0, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0});
    reconstruction.AddCamera(std::move(camera));

    Image image;
    image.SetImageId(kTriggerImageId);
    image.SetCameraId(kCameraId);
    image.SetName("trigger.jpg");
    image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    image.SetTvec(Eigen::Vector3d::Zero());
    image.SetPoints2D(std::vector<Eigen::Vector2d>(
        kPointCount, Eigen::Vector2d(50.0, 40.0)));
    reconstruction.AddImage(std::move(image));
    reconstruction.RegisterImage(kTriggerImageId);

    for (size_t index = 0; index < kPointCount; ++index) {
      Track track;
      point3D_ids.push_back(reconstruction.AddPoint3D(
          Eigen::Vector3d(static_cast<double>(index), 0.0, 5.0),
          std::move(track),
          Eigen::Vector3ub(1, 2, 3)));
    }
    reconstruction.BeginStructureJournal(kOwnerEpoch, 128);
  }

  ~ExecutorFixture() { reconstruction.EndStructureJournal(); }

  ReconstructionTransaction NewTransaction() {
    ReconstructionTransaction transaction;
    const ReconstructionTransactionResult created =
        reconstruction.CreateTransactionSnapshot(&transaction);
    if (!created.IsSuccess()) throw std::runtime_error(created.detail);
    return transaction;
  }

  OnlineLocalBaRequest Request(ReconstructionTransaction* transaction,
                               const OnlineLocalBaMode mode,
                               const uint32_t pass_index,
                               const uint64_t selection_revision,
                               const bool allow_postprocess) const {
    const Reconstruction* candidate = transaction->Candidate();
    if (candidate == nullptr) throw std::runtime_error("missing candidate");
    OnlineLocalBaRequest request;
    request.mode = mode;
    request.attempt_id = 7001;
    request.pass_index = pass_index;
    request.trigger_image_id = kTriggerImageId;
    request.ordered_frozen_image_ids = {kTriggerImageId};
    request.point3D_ids = point3D_ids;
    request.expected_candidate_version = candidate->CanonicalVersion();
    request.owner_epoch = candidate->StructureOwnerEpoch();
    request.topology_revision = candidate->StructureRevision();
    request.selection_revision = selection_revision;
    request.expected_map_version = snapshot->Version();
    request.expected_max_scan_index = snapshot->MaxScanIndex();
    request.expected_snapshot_sha256 = snapshot->SnapshotSha256();
    request.expected_geometry_sha256 = snapshot->GeometrySha256();
    request.map_snapshot = snapshot;
    request.allow_postprocess = allow_postprocess;
    std::string error;
    if (!CaptureOnlineLocalBaFixedIntrinsics(
            *candidate,
            request.ordered_frozen_image_ids,
            &request.fixed_intrinsics,
            &error)) {
      throw std::runtime_error(error);
    }
    return request;
  }

  std::shared_ptr<const lidar::LidarMapSnapshot> snapshot;
  Reconstruction reconstruction;
  std::vector<point3D_t> point3D_ids;
};

class FakeIntentBuilder final : public OnlineLocalBaIntentBuilder {
 public:
  explicit FakeIntentBuilder(std::vector<uint64_t> trigger_counts)
      : trigger_counts_(std::move(trigger_counts)) {}

  bool Build(const Reconstruction& candidate,
             const BundleAdjustmentOptions& options,
             const OnlineLocalBaRequest& request,
             OnlineLocalBaIntentBuildResult* result,
             std::string* error) override {
    (void)options;
    candidate_addresses.push_back(&candidate);
    observed_trigger_tx.push_back(
        candidate.Image(request.trigger_image_id).Tvec(0));
    const size_t call_index = call_count++;
    if (call_index >= trigger_counts_.size()) {
      *error = "unexpected fake intent build";
      return false;
    }
    const uint64_t trigger_count = trigger_counts_[call_index];
    *result = OnlineLocalBaIntentBuildResult();
    result->association_audit.attempt_id = request.attempt_id;
    result->association_audit.pass_index = request.pass_index;
    result->association_audit.trigger_image_id = request.trigger_image_id;
    result->association_audit.map_version = request.expected_map_version;
    result->association_audit.max_scan_index = request.expected_max_scan_index;
    result->association_audit.snapshot_sha256 =
        request.expected_snapshot_sha256;
    result->association_audit.geometry_sha256 =
        request.expected_geometry_sha256;
    result->association_audit.selected_association_count = trigger_count;
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      result->association_audit.preliminary_selected_count_by_image.emplace(
          image_id, image_id == request.trigger_image_id ? trigger_count : 0);
    }
    result->association_audit.trigger_preliminary_selected_count =
        trigger_count;
    const char association_digit = static_cast<char>('a' + (call_index % 6));
    result->association_audit.association_sha256 =
        std::string(64, association_digit);

    gpu_ba::NativeBaSolveIntent& intent = result->intent;
    intent.owner_epoch = request.owner_epoch;
    intent.reconstruction_identity = reinterpret_cast<uintptr_t>(&candidate);
    intent.expected_topology_revision = request.topology_revision;
    intent.selection_revision = request.selection_revision;
    intent.kind = gpu_ba::BaKind::kLocal;
    intent.visual_observation_scope =
        gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly;
    intent.config.resolved = true;
    intent.config.config_generation = request.selection_revision;
    intent.active_image_ids = request.ordered_frozen_image_ids;
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      gpu_ba::NativeBaTranslationPolicy policy;
      policy.image_id = image_id;
      policy.constant_mask = 0;
      intent.translation_policies.push_back(policy);
    }
    for (const OnlineLocalBaFixedIntrinsicsSnapshot& intrinsics :
         request.fixed_intrinsics) {
      gpu_ba::NativeBaCameraPolicy policy;
      policy.camera_id = intrinsics.camera_id;
      policy.constant = true;
      intent.camera_policies.push_back(policy);
    }
    intent.explicit_variable_point_ids = request.point3D_ids;
    for (const point3D_t point3D_id : request.point3D_ids) {
      gpu_ba::NativeBaPointPolicy policy;
      policy.point3D_id = point3D_id;
      policy.constant = false;
      intent.point_policies.push_back(policy);
    }
    intent.lidar_map_generation = request.expected_map_version;
    intent.lidar_match_config_generation = request.selection_revision;
    intent.online_lidar_identity.valid = true;
    intent.online_lidar_identity.trigger_image_id = request.trigger_image_id;
    intent.online_lidar_identity.map_version = request.expected_map_version;
    intent.online_lidar_identity.max_scan_index =
        request.expected_max_scan_index;
    intent.online_lidar_identity.snapshot_sha256 =
        DecodeSha256(request.expected_snapshot_sha256);
    intent.online_lidar_identity.geometry_sha256 =
        DecodeSha256(request.expected_geometry_sha256);
    intent.online_lidar_identity.association_sha256 =
        DecodeSha256(result->association_audit.association_sha256);
    for (uint64_t index = 0; index < trigger_count; ++index) {
      gpu_ba::NativeBaLidarConstraint constraint;
      constraint.point3D_id = request.point3D_ids.at(index);
      constraint.constraint_slot = static_cast<uint32_t>(index);
      constraint.physical_identity = index + 1;
      constraint.association_id = index;
      constraint.owner_image_id = request.trigger_image_id;
      constraint.owner_point2D_idx = static_cast<uint32_t>(index);
      constraint.weight = 1.0;
      intent.lidar_constraints.push_back(constraint);
    }
    error->clear();
    return true;
  }

  size_t call_count = 0;
  std::vector<const Reconstruction*> candidate_addresses;
  std::vector<double> observed_trigger_tx;

 private:
  std::vector<uint64_t> trigger_counts_;
};

struct FakeRunStep {
  bool adapter_success = true;
  bool solve_returned = true;
  bool execution_success = true;
  ceres::TerminationType termination = ceres::CONVERGENCE;
  bool finite = true;
  bool mutate_candidate = true;
};

class FakeNativeRunner final : public OnlineLocalBaNativeRunner {
 public:
  explicit FakeNativeRunner(std::vector<FakeRunStep> steps)
      : steps_(std::move(steps)) {}

  bool Run(const BundleAdjustmentOptions& options,
           Reconstruction* candidate,
           const OnlineLocalBaIntentBuildResult& build_result,
           const double intent_build_milliseconds,
           OnlineLocalBaNativeRunOutput* output,
           std::string* error) override {
    (void)intent_build_milliseconds;
    candidate_addresses.push_back(candidate);
    const size_t call_index = call_count++;
    if (call_index >= steps_.size()) {
      *error = "unexpected fake native run";
      return false;
    }
    const FakeRunStep& step = steps_[call_index];
    *output = OnlineLocalBaNativeRunOutput();
    output->solver_invoked = true;
    output->solve_returned = step.solve_returned;
    if (step.mutate_candidate) {
      candidate->Image(kTriggerImageId).Tvec(0) += 1.0;
    }
    output->execution.requested_backend = options.ba_backend;
    output->execution.executed_backend = "custom_cuda";
    output->execution.problem_source_requested = "native_graph";
    output->execution.problem_source_effective = "native_graph";
    output->execution.success = step.execution_success;
    output->execution.termination =
        step.termination == ceres::CONVERGENCE ? "function_tolerance"
                                                : "maximum_trial_iterations";
    output->execution.diagnostic_message =
        step.solve_returned ? std::string() : "injected solve failure";
    output->execution.initial_cost = 10.0;
    output->execution.final_cost =
        step.finite ? 4.0 : std::numeric_limits<double>::infinity();
    output->execution.max_backward_error = 0.0;
    output->execution.wall_seconds = 0.01;
    output->execution.residual_blocks =
        build_result.intent.lidar_constraints.size() + 4;
    output->summary.termination_type = step.termination;
    output->summary.initial_cost = 10.0;
    output->summary.final_cost = output->execution.final_cost;
    output->summary.total_time_in_seconds = 0.01;
    output->summary.minimizer_time_in_seconds = 0.01;
    output->summary.num_residual_blocks =
        static_cast<int>(output->execution.residual_blocks);

    if (!step.adapter_success) {
      *error = "injected native runner failure";
    } else {
      error->clear();
    }
    return step.adapter_success;
  }

  size_t call_count = 0;
  std::vector<const Reconstruction*> candidate_addresses;

 private:
  std::vector<FakeRunStep> steps_;
};

enum class PostprocessFailure { NONE, MERGE, COMPLETE, FILTER };

class FakePostprocess final : public OnlineLocalBaPostprocessAdapter {
 public:
  explicit FakePostprocess(
      const PostprocessFailure failure = PostprocessFailure::NONE)
      : failure_(failure) {}

  bool Merge(Reconstruction* candidate,
             const OnlineLocalBaRequest& request,
             uint64_t* count,
             std::string* error) override {
    (void)candidate;
    modes.push_back(request.mode);
    ++merge_calls;
    *count = 3;
    if (failure_ == PostprocessFailure::MERGE) {
      *error = "injected merge failure";
      return false;
    }
    return true;
  }

  bool Complete(Reconstruction* candidate,
                const OnlineLocalBaRequest& request,
                uint64_t* count,
                std::string* error) override {
    (void)candidate;
    modes.push_back(request.mode);
    ++complete_calls;
    *count = 5;
    if (failure_ == PostprocessFailure::COMPLETE) {
      *error = "injected complete failure";
      return false;
    }
    return true;
  }

  bool Filter(Reconstruction* candidate,
              const OnlineLocalBaRequest& request,
              uint64_t* count,
              std::string* error) override {
    modes.push_back(request.mode);
    ++filter_calls;
    candidate->Image(request.trigger_image_id).Tvec(0) += 0.25;
    *count = 7;
    if (failure_ == PostprocessFailure::FILTER) {
      *error = "injected filter failure";
      return false;
    }
    return true;
  }

  size_t merge_calls = 0;
  size_t complete_calls = 0;
  size_t filter_calls = 0;
  std::vector<OnlineLocalBaMode> modes;

 private:
  PostprocessFailure failure_;
};

OnlineLocalBaExecutorDependencies Dependencies(
    FakeIntentBuilder* builder,
    FakeNativeRunner* runner,
    FakePostprocess* postprocess,
    const bool audit_visual_state_sha256 = false) {
  OnlineLocalBaExecutorDependencies dependencies;
  dependencies.intent_builder = builder;
  dependencies.native_runner = runner;
  dependencies.postprocess = postprocess;
  dependencies.audit_visual_state_sha256 = audit_visual_state_sha256;
  return dependencies;
}

BOOST_AUTO_TEST_CASE(TriggerBelowFiftyDoesNotCallSolver) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 11, true);
  FakeIntentBuilder builder({49});
  FakeNativeRunner runner({FakeRunStep()});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM);
  BOOST_CHECK(!result.solver_called);
  BOOST_CHECK_EQUAL(result.trigger_submitted_lidar_constraint_count, 49);
  BOOST_CHECK(!result.residual_evidence.materialized_counts_available);
  BOOST_CHECK(!result.residual_evidence.evaluated_counts_available);
  BOOST_CHECK_EQUAL(builder.call_count, 1);
  BOOST_CHECK_EQUAL(runner.call_count, 0);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  BOOST_CHECK(result.candidate_discarded);
  BOOST_CHECK(!transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(CatchupIncreaseIsCheckedBeforeSolveAndPostprocess) {
  for (const uint64_t count : {uint64_t(49), uint64_t(51), uint64_t(52)}) {
    ExecutorFixture fixture;
    const CanonicalState canonical = CaptureCanonicalState(fixture.reconstruction);
    ReconstructionTransaction transaction = fixture.NewTransaction();
    OnlineLocalBaRequest request = fixture.Request(
        &transaction, OnlineLocalBaMode::CATCHUP, 1, 11, true);
    request.require_trigger_residual_increase = true;
    request.previous_trigger_lidar_constraint_count = 2;
    FakeIntentBuilder builder({count});
    FakeNativeRunner runner({FakeRunStep()});
    FakePostprocess postprocess;
    const auto result = ExecuteOnlineLocalBa(
        StrictOptions(), request, &transaction,
        Dependencies(&builder, &runner, &postprocess));
    BOOST_CHECK_EQUAL(result.success, count == 52);
    BOOST_CHECK_EQUAL(runner.call_count, count == 52 ? 1 : 0);
    BOOST_CHECK_EQUAL(postprocess.merge_calls, count == 52 ? 1 : 0);
    if (count < 52) {
      BOOST_CHECK(result.failure_reason ==
          (count < 50 ? OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM
                      : OnlineLocalBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED));
      BOOST_CHECK(result.candidate_visual_state_sha256_before.empty());
      BOOST_CHECK(result.candidate_visual_state_sha256_after.empty());
      BOOST_CHECK(!transaction.HasCandidate());
    }
    CheckCanonicalState(fixture.reconstruction, canonical);
  }
}

BOOST_AUTO_TEST_CASE(DefaultBuilderPreflightRejectsWithoutMaterializingIntent) {
  ExecutorFixture fixture;
  for (size_t index = 0; index < fixture.point3D_ids.size(); ++index) {
    const point3D_t point_id = fixture.point3D_ids[index];
    fixture.reconstruction.Point3D(point_id).SetXYZ(
        index < 49 ? Eigen::Vector3d(1.0, 2.0, 3.0)
                   : Eigen::Vector3d(100.0, 100.0, 100.0));
    fixture.reconstruction.AddObservation(
        point_id, TrackElement(kTriggerImageId, static_cast<point2D_t>(index)));
  }
  const CanonicalState canonical = CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 11, true);
  request.association_options.local_lidar_kdtree_only = true;
  FakeNativeRunner runner({FakeRunStep()});
  FakePostprocess postprocess;
  const auto result = ExecuteOnlineLocalBa(
      StrictOptions(), request, &transaction,
      Dependencies(nullptr, &runner, &postprocess));
  BOOST_CHECK_MESSAGE(result.failure_reason ==
      OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM, result.failure_detail);
  BOOST_CHECK(result.trigger_gate.checked);
  BOOST_CHECK(!result.trigger_gate.passed);
  BOOST_CHECK_EQUAL(result.trigger_gate.actual_count, 49);
  BOOST_CHECK_EQUAL(result.submitted_lidar_constraint_count, 0);
  BOOST_CHECK(!result.submitted_constraint_audit_passed);
  BOOST_CHECK_EQUAL(runner.call_count, 0);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  BOOST_CHECK(result.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(!transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(SingleSuccessRunsPostprocessExactlyOnce) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 12, true);
  FakeIntentBuilder builder({50});
  FakeNativeRunner runner({FakeRunStep()});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_REQUIRE(result.success);
  BOOST_CHECK(result.candidate_ready_for_commit);
  BOOST_CHECK(!result.candidate_discarded);
  BOOST_CHECK(result.solver_called);
  BOOST_CHECK(result.backend_contract_passed);
  BOOST_CHECK(result.termination_converged);
  BOOST_CHECK(result.finite_costs);
  BOOST_CHECK(result.submitted_constraint_audit_passed);
  BOOST_CHECK(result.solve_acceptance_passed);
  BOOST_CHECK_EQUAL(result.trigger_submitted_lidar_constraint_count, 50);
  BOOST_CHECK_EQUAL(result.submitted_lidar_constraint_count, 50);
  BOOST_CHECK(!result.residual_evidence.materialized_counts_available);
  BOOST_CHECK(!result.residual_evidence.evaluated_counts_available);
  BOOST_CHECK(
      result.residual_evidence.per_image_materialized_lidar_count.empty());
  BOOST_CHECK(result.residual_evidence.per_image_evaluated_lidar_count.empty());
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.complete_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.filter_calls, 1);
  BOOST_CHECK_EQUAL(result.postprocess.merge_call_count, 1);
  BOOST_CHECK_EQUAL(result.postprocess.complete_call_count, 1);
  BOOST_CHECK_EQUAL(result.postprocess.filter_call_count, 1);
  BOOST_REQUIRE(transaction.HasCandidate());
  BOOST_CHECK_EQUAL(transaction.Candidate()->Image(kTriggerImageId).Tvec(0),
                    1.25);
  BOOST_CHECK(result.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(result.candidate_visual_state_sha256_after.empty());
  BOOST_CHECK_EQUAL(result.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_GT(result.timing.request_validation_milliseconds, 0.0);
  BOOST_CHECK_EQUAL(result.timing.transaction_discard_milliseconds, 0.0);
  BOOST_REQUIRE_EQUAL(builder.candidate_addresses.size(), 1);
  BOOST_CHECK(builder.candidate_addresses.front() != &fixture.reconstruction);
  BOOST_CHECK(runner.candidate_addresses.front() != &fixture.reconstruction);
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(UsableIterationLimitedSolveRetainsCandidateForCommit) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 13, true);
  FakeIntentBuilder builder({50});
  FakeRunStep step;
  step.termination = ceres::NO_CONVERGENCE;
  FakeNativeRunner runner({step});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(result.success);
  BOOST_CHECK(result.solve_acceptance_passed);
  BOOST_CHECK(!result.termination_converged);
  BOOST_CHECK(result.finite_costs);
  BOOST_CHECK_EQUAL(result.execution.termination, "maximum_trial_iterations");
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.complete_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.filter_calls, 1);
  BOOST_CHECK(transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(UnusableTerminationIsRejectedAndDiscarded) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 13, true);
  FakeIntentBuilder builder({50});
  FakeRunStep step;
  step.termination = ceres::FAILURE;
  FakeNativeRunner runner({step});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess, true));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::TERMINATION_UNUSABLE);
  BOOST_CHECK(result.solver_called);
  BOOST_CHECK(!result.solve_acceptance_passed);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_before.size(), 64);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_after.size(), 64);
  BOOST_CHECK_NE(result.candidate_visual_state_sha256_before,
                 result.candidate_visual_state_sha256_after);
  BOOST_CHECK_GT(result.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_GT(result.timing.request_validation_milliseconds, 0.0);
  BOOST_CHECK_GT(result.timing.transaction_discard_milliseconds, 0.0);
  BOOST_CHECK(!transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(FusedSolveFailureDropsMutatedCandidate) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 14, true);
  FakeIntentBuilder builder({50});
  FakeRunStep step;
  step.solve_returned = false;
  step.execution_success = false;
  FakeNativeRunner runner({step});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::SOLVE_FAILED);
  BOOST_CHECK(result.solver_called);
  BOOST_CHECK(result.candidate_discarded);
  BOOST_CHECK(!transaction.HasCandidate());
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(NonFiniteCostIsRejectedAndDiscarded) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 16, true);
  FakeIntentBuilder builder({50});
  FakeRunStep step;
  step.finite = false;
  FakeNativeRunner runner({step});
  FakePostprocess postprocess;

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::NONFINITE_SOLVER_RESULT);
  BOOST_CHECK(result.solver_called);
  BOOST_CHECK(!result.finite_costs);
  BOOST_CHECK(!result.solve_acceptance_passed);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  BOOST_CHECK(!transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(PostprocessFailureDropsSolvedCandidate) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest request = fixture.Request(
      &transaction, OnlineLocalBaMode::SINGLE, 1, 15, true);
  FakeIntentBuilder builder({50});
  FakeNativeRunner runner({FakeRunStep()});
  FakePostprocess postprocess(PostprocessFailure::FILTER);

  const OnlineLocalBaPassResult result = ExecuteOnlineLocalBa(
      StrictOptions(),
      request,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::POSTPROCESS_FILTER_FAILED);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.complete_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.filter_calls, 1);
  BOOST_CHECK(result.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(result.candidate_visual_state_sha256_after.empty());
  BOOST_CHECK_EQUAL(result.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_GT(result.timing.request_validation_milliseconds, 0.0);
  BOOST_CHECK_GT(result.timing.transaction_discard_milliseconds, 0.0);
  BOOST_CHECK(!transaction.HasCandidate());
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(DualPassesReassociateOnOneCandidateAndPostprocessOnce) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest pass1 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS1, 1, 21, false);
  const OnlineLocalBaRequest pass2 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS2, 2, 22, true);
  FakeIntentBuilder builder({50, 50});
  FakeNativeRunner runner({FakeRunStep(), FakeRunStep()});
  FakePostprocess postprocess;

  const OnlineDualLocalBaResult result = ExecuteOnlineDualLocalBa(
      StrictOptions(),
      pass1,
      pass2,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_REQUIRE(result.success);
  BOOST_CHECK(!result.fatal);
  BOOST_CHECK(result.candidate_ready_for_commit);
  BOOST_REQUIRE(result.pass1.success);
  BOOST_REQUIRE(result.pass2.success);
  BOOST_CHECK(!result.pass1.candidate_ready_for_commit);
  BOOST_CHECK(result.pass2.candidate_ready_for_commit);
  BOOST_CHECK_EQUAL(result.pass1.postprocess.merge_call_count, 0);
  BOOST_CHECK_EQUAL(result.pass1.postprocess.complete_call_count, 0);
  BOOST_CHECK_EQUAL(result.pass1.postprocess.filter_call_count, 0);
  BOOST_CHECK_EQUAL(result.pass2.postprocess.merge_call_count, 1);
  BOOST_CHECK_EQUAL(result.pass2.postprocess.complete_call_count, 1);
  BOOST_CHECK_EQUAL(result.pass2.postprocess.filter_call_count, 1);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.complete_calls, 1);
  BOOST_CHECK_EQUAL(postprocess.filter_calls, 1);
  BOOST_REQUIRE_EQUAL(postprocess.modes.size(), 3);
  BOOST_CHECK(postprocess.modes[0] == OnlineLocalBaMode::DUAL_PASS2);
  BOOST_CHECK(postprocess.modes[1] == OnlineLocalBaMode::DUAL_PASS2);
  BOOST_CHECK(postprocess.modes[2] == OnlineLocalBaMode::DUAL_PASS2);
  BOOST_REQUIRE_EQUAL(builder.call_count, 2);
  BOOST_REQUIRE_EQUAL(builder.candidate_addresses.size(), 2);
  BOOST_CHECK_EQUAL(builder.candidate_addresses[0],
                    builder.candidate_addresses[1]);
  BOOST_REQUIRE_EQUAL(builder.observed_trigger_tx.size(), 2);
  BOOST_CHECK_EQUAL(builder.observed_trigger_tx[0], 0.0);
  BOOST_CHECK_EQUAL(builder.observed_trigger_tx[1], 1.0);
  BOOST_REQUIRE_EQUAL(runner.candidate_addresses.size(), 2);
  BOOST_CHECK_EQUAL(runner.candidate_addresses[0],
                    runner.candidate_addresses[1]);
  BOOST_CHECK_NE(result.pass1.association_audit.association_sha256,
                 result.pass2.association_audit.association_sha256);
  BOOST_CHECK(result.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(result.candidate_visual_state_sha256_after.empty());
  BOOST_CHECK(result.pass1.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(result.pass1.candidate_visual_state_sha256_after.empty());
  BOOST_CHECK(result.pass2.candidate_visual_state_sha256_before.empty());
  BOOST_CHECK(result.pass2.candidate_visual_state_sha256_after.empty());
  BOOST_CHECK_EQUAL(result.pass1.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_EQUAL(result.pass2.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_REQUIRE(transaction.HasCandidate());
  BOOST_CHECK_EQUAL(transaction.Candidate()->Image(kTriggerImageId).Tvec(0),
                    2.25);
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(TriangulationSkipsOnlyCompletedObservations) {
  Reconstruction baseline;
  CorrespondenceGraph graph;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(PinholeCameraModel::model_id, 100.0, 100, 80);
  baseline.AddCamera(camera);
  for (const image_t image_id : {1, 2}) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName(std::to_string(image_id) + ".jpg");
    image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    image.SetTvec(Eigen::Vector3d(image_id == 1 ? 0.0 : -1.0, 0.0, 0.0));
    image.SetPoints2D(std::vector<Eigen::Vector2d>(
        4, Eigen::Vector2d(image_id == 1 ? 50.0 : 30.0, 40.0)));
    baseline.AddImage(image);
    graph.AddImage(image_id, 4);
  }
  graph.AddCorrespondences(1, 2, {FeatureMatch(0, 0), FeatureMatch(1, 1),
                                 FeatureMatch(2, 2)});
  baseline.SetUp(&graph);
  BOOST_REQUIRE(baseline.AddImagePairFromCorrespondenceGraph(1, 2));
  baseline.RegisterImage(1);
  baseline.RegisterImage(2);
  Track complete_track;
  complete_track.AddElement(1, 0);
  complete_track.AddElement(2, 0);
  baseline.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), complete_track);
  Track partial_track;
  partial_track.AddElement(2, 1);
  baseline.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), partial_track);
  Reconstruction optimized = baseline;
  IncrementalTriangulator reference_triangulator(&graph, &baseline);
  IncrementalTriangulator optimized_triangulator(&graph, &optimized);
  IncrementalTriangulator::Options reference_options;
  reference_options.ignore_two_view_tracks = false;
  reference_options.max_transitivity = 2;
  auto optimized_options = reference_options;
  optimized_options.max_transitivity = 1;
  BOOST_CHECK_EQUAL(reference_triangulator.TriangulateImage(reference_options, 1),
                    3);
  BOOST_CHECK_EQUAL(optimized_triangulator.TriangulateImage(optimized_options, 1),
                    3);
  BOOST_REQUIRE_EQUAL(optimized.NumPoints3D(), baseline.NumPoints3D());
  for (const auto& entry : baseline.Points3D()) {
    const auto& actual = optimized.Point3D(entry.first);
    BOOST_CHECK_LT((actual.XYZ() - entry.second.XYZ()).norm(), 1e-12);
    BOOST_CHECK_EQUAL(actual.Track().Length(), entry.second.Track().Length());
  }
  BOOST_CHECK(!optimized.Image(1).Point2D(3).HasPoint3D());
  optimized_triangulator.ClearModifiedPoints3D();
  BOOST_CHECK_EQUAL(optimized_triangulator.TriangulateImage(optimized_options, 1),
                    0);
  BOOST_CHECK(optimized_triangulator.GetModifiedPoints3D().empty());
}

BOOST_AUTO_TEST_CASE(TriggerObservationBoundCountsExistingAndNewRegisteredEvidence) {
  ExecutorFixture fixture;
  CorrespondenceGraph graph;
  graph.AddImage(kTriggerImageId, kPointCount);
  for (const image_t image_id : {2, 3}) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(kCameraId);
    image.SetName(std::to_string(image_id) + ".jpg");
    image.SetPoints2D(std::vector<Eigen::Vector2d>(
        kPointCount, Eigen::Vector2d(50.0, 40.0)));
    fixture.reconstruction.AddImage(image);
    graph.AddImage(image_id, kPointCount);
  }
  fixture.reconstruction.RegisterImage(2);
  fixture.reconstruction.AddObservation(fixture.point3D_ids.front(),
                                        TrackElement(kTriggerImageId, 0));
  FeatureMatches matches;
  for (point2D_t feature_index = 0; feature_index < 49; ++feature_index) {
    matches.emplace_back(feature_index, feature_index);
  }
  graph.AddCorrespondences(kTriggerImageId, 2, matches);
  graph.AddCorrespondences(kTriggerImageId, 3,
                           {FeatureMatch(1, 1), FeatureMatch(49, 49)});
  uint64_t upper_bound = 0;
  std::string error;
  BOOST_REQUIRE_MESSAGE(CountOnlineLocalBaPotentialTriggerObservations(
      fixture.reconstruction, graph, kTriggerImageId, &upper_bound, &error),
      error);
  BOOST_CHECK_EQUAL(upper_bound, 49);
  fixture.reconstruction.RegisterImage(3);
  BOOST_REQUIRE_MESSAGE(CountOnlineLocalBaPotentialTriggerObservations(
      fixture.reconstruction, graph, kTriggerImageId, &upper_bound, &error),
      error);
  BOOST_CHECK_EQUAL(upper_bound, 50);
  BOOST_CHECK_EQUAL(fixture.reconstruction.Image(kTriggerImageId).NumPoints3D(),
                    1);
  BOOST_CHECK(!CountOnlineLocalBaPotentialTriggerObservations(
      fixture.reconstruction, graph, 999, &upper_bound, &error));
  BOOST_CHECK_EQUAL(upper_bound, 0);
}

BOOST_AUTO_TEST_CASE(DualPass1FailureIsFatalAndDropsWholeCandidate) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest pass1 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS1, 1, 31, false);
  const OnlineLocalBaRequest pass2 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS2, 2, 32, true);
  FakeIntentBuilder builder({49, 50});
  FakeNativeRunner runner({FakeRunStep(), FakeRunStep()});
  FakePostprocess postprocess;

  const OnlineDualLocalBaResult result = ExecuteOnlineDualLocalBa(
      StrictOptions(),
      pass1,
      pass2,
      &transaction,
      Dependencies(&builder, &runner, &postprocess));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.fatal);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::DUAL_PASS1_FAILED);
  BOOST_CHECK(result.candidate_discarded);
  BOOST_CHECK(!transaction.HasCandidate());
  BOOST_CHECK_EQUAL(builder.call_count, 1);
  BOOST_CHECK_EQUAL(runner.call_count, 0);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  CheckCanonicalState(fixture.reconstruction, canonical);
}

BOOST_AUTO_TEST_CASE(DualPass2FailureIsFatalAndDropsPass1Delta) {
  ExecutorFixture fixture;
  const CanonicalState canonical =
      CaptureCanonicalState(fixture.reconstruction);
  ReconstructionTransaction transaction = fixture.NewTransaction();
  const OnlineLocalBaRequest pass1 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS1, 1, 41, false);
  const OnlineLocalBaRequest pass2 = fixture.Request(
      &transaction, OnlineLocalBaMode::DUAL_PASS2, 2, 42, true);
  FakeIntentBuilder builder({50, 50});
  FakeRunStep pass2_failure;
  pass2_failure.solve_returned = false;
  pass2_failure.execution_success = false;
  FakeNativeRunner runner({FakeRunStep(), pass2_failure});
  FakePostprocess postprocess;

  const OnlineDualLocalBaResult result = ExecuteOnlineDualLocalBa(
      StrictOptions(),
      pass1,
      pass2,
      &transaction,
      Dependencies(&builder, &runner, &postprocess, true));

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.fatal);
  BOOST_CHECK(result.failure_reason ==
              OnlineLocalBaFailureReason::DUAL_PASS2_FAILED);
  BOOST_CHECK(result.candidate_discarded);
  BOOST_CHECK(!transaction.HasCandidate());
  BOOST_CHECK_EQUAL(builder.call_count, 2);
  BOOST_CHECK_EQUAL(runner.call_count, 2);
  BOOST_CHECK_EQUAL(postprocess.merge_calls, 0);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_before.size(), 64);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_after.size(), 64);
  BOOST_CHECK_EQUAL(result.pass1.candidate_visual_state_sha256_before.size(),
                    64);
  BOOST_CHECK_EQUAL(result.pass1.candidate_visual_state_sha256_after.size(),
                    64);
  BOOST_CHECK_EQUAL(result.pass2.candidate_visual_state_sha256_before.size(),
                    64);
  BOOST_CHECK_EQUAL(result.pass2.candidate_visual_state_sha256_after.size(),
                    64);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_before,
                    result.pass1.candidate_visual_state_sha256_before);
  BOOST_CHECK_EQUAL(result.pass1.candidate_visual_state_sha256_after,
                    result.pass2.candidate_visual_state_sha256_before);
  BOOST_CHECK_EQUAL(result.candidate_visual_state_sha256_after,
                    result.pass2.candidate_visual_state_sha256_after);
  BOOST_CHECK_NE(result.candidate_visual_state_sha256_before,
                 result.candidate_visual_state_sha256_after);
  BOOST_CHECK_GT(result.pass1.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_GT(result.pass2.timing.visual_state_hash_milliseconds, 0.0);
  BOOST_CHECK_GT(result.pass1.timing.request_validation_milliseconds, 0.0);
  BOOST_CHECK_GT(result.pass2.timing.request_validation_milliseconds, 0.0);
  BOOST_CHECK_EQUAL(result.pass1.timing.transaction_discard_milliseconds, 0.0);
  BOOST_CHECK_GT(result.pass2.timing.transaction_discard_milliseconds, 0.0);
  CheckCanonicalState(fixture.reconstruction, canonical);
}

}  // namespace
}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED
