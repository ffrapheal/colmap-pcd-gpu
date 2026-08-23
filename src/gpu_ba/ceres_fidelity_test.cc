#define TEST_NAME "gpu_ba/ceres_fidelity"
#include "util/testing.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <unistd.h>

#include "base/camera_models.h"
#include "gpu_ba/ceres_fidelity.h"
#include "optim/bundle_adjustment.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string TempRoot() {
  return "/tmp/colmap_gpu_ba_ceres_fidelity_test_" +
         std::to_string(getpid());
}

struct Fixture {
  Fixture() { boost::filesystem::remove_all(TempRoot()); }
  ~Fixture() { boost::filesystem::remove_all(TempRoot()); }
};

Snapshot MakeSnapshot() {
  Snapshot snapshot;
  snapshot.metadata.snapshot_id =
      "global-reg2-call1-refine0-trigger35-phraseglobal";
  snapshot.metadata.ba_kind = BaKind::kGlobal;
  snapshot.metadata.registered_image_count = 2;
  snapshot.metadata.ba_call_index = 1;
  snapshot.metadata.trigger_image_id = 35;
  snapshot.metadata.optimize_phrase = "global";
  snapshot.metadata.loss_function = "trivial";

  CameraSnapshot camera;
  camera.camera_id = 1;
  camera.model_id = OpenCVCameraModel::kModelId;
  camera.width = 640;
  camera.height = 480;
  camera.constant = true;
  camera.params = {500.0, 505.0, 320.0, 240.0,
                   -0.02, 0.003, 0.001, -0.0004};
  snapshot.cameras.push_back(camera);

  ImageSnapshot variable_image;
  variable_image.image_id = 35;
  variable_image.camera_id = 1;
  variable_image.selected = true;
  variable_image.pose_constant = false;
  variable_image.has_pose_parameter_blocks = true;
  variable_image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  variable_image.tvec = {{0.1, -0.2, 0.3}};
  snapshot.images.push_back(variable_image);

  ImageSnapshot constant_image = variable_image;
  constant_image.image_id = 30;
  constant_image.pose_constant = true;
  constant_image.has_pose_parameter_blocks = false;
  constant_image.tvec = {{0.0, 0.0, 0.0}};
  snapshot.images.push_back(constant_image);

  PointSnapshot point;
  point.point3D_id = 7;
  point.xyz = {{0.2, -0.1, 3.0}};
  snapshot.points.push_back(point);

  ObservationSnapshot variable_observation;
  variable_observation.source_index = 1;
  variable_observation.image_id = 35;
  variable_observation.point2D_idx = 11;
  variable_observation.point3D_id = 7;
  variable_observation.xy = {{355.0, 225.0}};
  snapshot.observations.push_back(variable_observation);

  ObservationSnapshot constant_observation = variable_observation;
  constant_observation.source_index = 2;
  constant_observation.image_id = 30;
  constant_observation.point2D_idx = 9;
  constant_observation.pose_constant = true;
  constant_observation.xy = {{352.0, 222.0}};
  snapshot.observations.push_back(constant_observation);
  snapshot.tracks.push_back({7, 35, 11});
  snapshot.tracks.push_back({7, 30, 9});

  LidarSnapshot lidar;
  lidar.source_index = 0;
  lidar.point3D_id = 7;
  lidar.weight = 10.0;
  lidar.lidar_xyz = {{0.2, -0.1, 2.99}};
  lidar.plane = {{0.0, 0.0, 1.0, -2.99}};
  snapshot.lidar.push_back(lidar);

  snapshot.source_insertion_order.push_back(
      {0, ResidualKind::kLidar, std::numeric_limits<uint32_t>::max(),
       std::numeric_limits<uint32_t>::max(), 7});
  snapshot.source_insertion_order.push_back(
      {1, ResidualKind::kVisual, 35, 11, 7});
  snapshot.source_insertion_order.push_back(
      {2, ResidualKind::kVisual, 30, 9, 7});
  snapshot.canonical_order = snapshot.source_insertion_order;
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                              lhs.point3D_id, lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                              rhs.point3D_id, rhs.source_index);
            });

  snapshot.parameter_blocks_source_order.push_back(
      {0, ParameterKind::kPoint3D, 7, 3, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {1, ParameterKind::kQuaternion, 35, 4, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {2, ParameterKind::kTranslation, 35, 3, 3, false});
  snapshot.parameter_blocks_source_order.push_back(
      {3, ParameterKind::kCamera, 1, 8, 8, true});
  snapshot.parameter_blocks_canonical_order = {1, 2, 0, 3};
  return snapshot;
}

std::vector<ParameterConstraintSnapshot> MakeConstraints() {
  return {
      {ParameterKind::kPoint3D, 7, false, "euclidean", {}},
      {ParameterKind::kQuaternion, 35, false, "quaternion", {}},
      {ParameterKind::kTranslation, 35, false, "euclidean", {}},
      {ParameterKind::kCamera, 1, true, "euclidean", {}}};
}

EffectiveCeresOptionsSnapshot MakeEffectiveOptions(
    int max_invalid = 10,
    int max_linear_iterations = 123) {
  BundleAdjustmentOptions options;
  options.solver_options.max_num_consecutive_invalid_steps = max_invalid;
  options.solver_options.max_linear_solver_iterations = max_linear_iterations;
  options.solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = 1;
#endif
  const ceres::Solver::Options effective =
      CreateEffectiveBundleAdjustmentSolverOptions(options, 2, 5);
  return CaptureEffectiveCeresOptions(
      options.solver_options, effective,
      options.min_num_residuals_for_multi_threading);
}

struct OracleFixtureData {
  Snapshot initial;
  SnapshotWriteResult initial_write;
  SnapshotWriteResult post_write;
  CeresFidelityRecord record;
  CeresFidelityWriteResult record_write;
};

OracleFixtureData WriteOracleFixture(
    const std::vector<LossSpecificationSnapshot>& losses,
    const std::string& directory,
    const std::string& run_id = "oracle") {
  OracleFixtureData data;
  data.initial = MakeSnapshot();
  std::string error;
  BOOST_REQUIRE(WriteSnapshot(data.initial, directory + "/initial",
                              &data.initial_write, &error));
  Snapshot post = data.initial;
  post.metadata.snapshot_id += "-post";
  BOOST_REQUIRE(WriteSnapshot(post, directory + "/post", &data.post_write,
                              &error));

  setenv("COLMAP_FIDELITY_LIBCERES_PATH", "/proc/self/exe", 1);
  setenv("COLMAP_FIDELITY_GIT_HEAD", "unit-test-head", 1);
  setenv("COLMAP_FIDELITY_DIRTY_DIFF_SHA256", "unit-test-diff", 1);
  data.record.record_kind = "original_bundle_adjuster";
  data.record.run_id = run_id;
  BOOST_REQUIRE(CaptureFidelityProvenance(&data.record.provenance, &error));
  data.record.snapshot_id = data.initial.metadata.snapshot_id;
  data.record.snapshot_manifest_path = data.initial_write.manifest_path;
  data.record.snapshot_payload_path = data.initial_write.payload_path;
  data.record.snapshot_payload_sha256 =
      data.initial_write.integrity.payload_sha256;
  data.record.snapshot_manifest_sha256 =
      data.initial_write.integrity.manifest_sha256;
  data.record.post_state_manifest_path = data.post_write.manifest_path;
  data.record.post_state_payload_path = data.post_write.payload_path;
  data.record.post_state_payload_sha256 =
      data.post_write.integrity.payload_sha256;
  data.record.final_state_sha256 = CanonicalStateSha256(post);
  data.record.effective_options = MakeEffectiveOptions();
  data.record.problem = BuildFidelityProblemSnapshot(
      data.initial, data.initial_write.integrity,
      data.record.effective_options, 2, 5, 4, 18, MakeConstraints(), losses,
      false);
  BOOST_REQUIRE(WriteCeresFidelityRecord(data.record, directory + "/oracle",
                                        &data.record_write, &error));
  return data;
}

std::vector<LossSpecificationSnapshot> TrivialLosses() {
  return {{"visual", "TRIVIAL", 1.0, 2},
          {"lidar", "TRIVIAL", 1.0, 1}};
}

}  // namespace

BOOST_AUTO_TEST_CASE(DefaultAndNondefaultInvalidLimitArePreserved) {
  BundleAdjustmentOptions defaults;
  BOOST_CHECK_EQUAL(
      defaults.solver_options.max_num_consecutive_invalid_steps, 10);
  for (const int expected : {10, 7}) {
    const EffectiveCeresOptionsSnapshot captured =
        MakeEffectiveOptions(expected, 123);
    BOOST_CHECK_EQUAL(captured.max_num_consecutive_invalid_steps, expected);
    ceres::Solver::Options replay;
    std::string error;
    BOOST_REQUIRE(ApplyEffectiveCeresOptions(captured, &replay, &error));
    BOOST_CHECK_EQUAL(replay.max_num_consecutive_invalid_steps, expected);
  }
}

BOOST_AUTO_TEST_CASE(MaxLinearSolverIterationsRoundTrip) {
  const EffectiveCeresOptionsSnapshot captured = MakeEffectiveOptions(7, 137);
  BOOST_CHECK_EQUAL(captured.max_linear_solver_iterations, 137);
  ceres::Solver::Options replay;
  std::string error;
  BOOST_REQUIRE(ApplyEffectiveCeresOptions(captured, &replay, &error));
  BOOST_CHECK_EQUAL(replay.max_linear_solver_iterations, 137);
  BOOST_CHECK_EQUAL(EffectiveCeresOptionsSha256(captured),
                    EffectiveCeresOptionsSha256(
                        CaptureEffectiveCeresOptions(replay, replay,
                                                     captured.min_num_residuals_for_multi_threading)));
}

BOOST_AUTO_TEST_CASE(ThreadThresholdBoundaryUsesScalarResidualCount) {
  BundleAdjustmentOptions options;
  options.solver_options.num_threads = 8;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = 8;
#endif
  options.min_num_residuals_for_multi_threading = 50000;
  const ceres::Solver::Options below =
      CreateEffectiveBundleAdjustmentSolverOptions(options, 51, 49999);
  const ceres::Solver::Options boundary =
      CreateEffectiveBundleAdjustmentSolverOptions(options, 51, 50000);
  BOOST_CHECK_EQUAL(below.num_threads, 1);
  BOOST_CHECK_EQUAL(boundary.num_threads, 8);
#if CERES_VERSION_MAJOR < 2
  BOOST_CHECK_EQUAL(below.num_linear_solver_threads, 1);
  BOOST_CHECK_EQUAL(boundary.num_linear_solver_threads, 8);
#endif
}

BOOST_AUTO_TEST_CASE(SolverSelectionUsesConfigImageCount) {
  BundleAdjustmentOptions options;
  options.solver_options.num_threads = 8;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = 8;
#endif
  const ceres::Solver::Options dense =
      CreateEffectiveBundleAdjustmentSolverOptions(options, 50, 50000);
  const ceres::Solver::Options sparse =
      CreateEffectiveBundleAdjustmentSolverOptions(options, 51, 50000);
  BOOST_CHECK(dense.linear_solver_type == ceres::DENSE_SCHUR);
  BOOST_CHECK(sparse.linear_solver_type == ceres::SPARSE_SCHUR);
}

BOOST_AUTO_TEST_CASE(SourceAndCanonicalOrdersAreDistinctAndHashed) {
  const Snapshot snapshot = MakeSnapshot();
  BOOST_REQUIRE(snapshot.source_insertion_order[0].source_index !=
                snapshot.canonical_order[0].source_index);
  SnapshotIntegrity integrity;
  integrity.lidar_correspondence_sha256 = "lidar";
  const FidelityProblemSnapshot problem = BuildFidelityProblemSnapshot(
      snapshot, integrity, MakeEffectiveOptions(), 2, 5, 4, 18,
      MakeConstraints(), TrivialLosses(), false);
  BOOST_CHECK_NE(problem.source_residual_order_sha256,
                 problem.canonical_residual_order_sha256);
  BOOST_CHECK_NE(problem.source_parameter_order_sha256,
                 problem.canonical_parameter_order_sha256);
}

BOOST_FIXTURE_TEST_CASE(LossTypeAndScaleRoundTrip, Fixture) {
  const std::vector<LossSpecificationSnapshot> losses = {
      {"visual", "TRIVIAL", 1.0, 2},
      {"lidar", "SOFT_L1", 1.0, 1},
      {"diagnostic", "SOFT_L1", 0.7, 0}};
  const OracleFixtureData data =
      WriteOracleFixture(losses, TempRoot() + "/loss-roundtrip");
  CeresFidelityRecord loaded;
  CeresFidelityWriteResult read;
  std::string error;
  BOOST_REQUIRE(ReadCeresFidelityRecord(data.record_write.manifest_path,
                                       &loaded, &read, &error));
  BOOST_REQUIRE_EQUAL(loaded.problem.loss_specifications.size(), 3);
  BOOST_CHECK_EQUAL(loaded.problem.loss_specifications[0].type, "TRIVIAL");
  BOOST_CHECK_EQUAL(loaded.problem.loss_specifications[1].type, "SOFT_L1");
  BOOST_CHECK_EQUAL(loaded.problem.loss_specifications[1].scale, 1.0);
  BOOST_CHECK_EQUAL(loaded.problem.loss_specifications[2].scale, 0.7);
  BOOST_CHECK_EQUAL(loaded.problem.loss_specification_sha256,
                    data.record.problem.loss_specification_sha256);
}

BOOST_FIXTURE_TEST_CASE(PostSolveStateSerializationPreservesHash, Fixture) {
  Snapshot original = MakeSnapshot();
  original.images[0].qvec[2] = 0.012345678901234567;
  original.images[0].tvec[1] = -9.8765432109876543;
  original.points[0].xyz[2] = 3.1415926535897931;
  original.cameras[0].params[0] = 500.00000000000006;
  SnapshotWriteResult written;
  std::string error;
  BOOST_REQUIRE(WriteSnapshot(original, TempRoot() + "/state", &written,
                              &error));
  Snapshot loaded;
  SnapshotReadResult read;
  BOOST_REQUIRE(ReadSnapshot(written.manifest_path, &loaded, &read, &error));
  BOOST_CHECK_EQUAL(CanonicalStateSha256(original),
                    CanonicalStateSha256(loaded));
}

BOOST_FIXTURE_TEST_CASE(StaleOracleIsRefused, Fixture) {
  const OracleFixtureData data =
      WriteOracleFixture(TrivialLosses(), TempRoot() + "/stale");
  CeresFidelityWriteResult duplicate;
  std::string error;
  BOOST_CHECK(!WriteCeresFidelityRecord(data.record,
                                       TempRoot() + "/stale/oracle",
                                       &duplicate, &error));
  BOOST_CHECK(error.find("stale") != std::string::npos);
}

BOOST_FIXTURE_TEST_CASE(OptionLossAndOrderPerturbationsFailClosed, Fixture) {
  const OracleFixtureData data =
      WriteOracleFixture(TrivialLosses(), TempRoot() + "/perturb");
  struct Case {
    std::string perturbation;
    std::string expected_mismatch;
    bool option;
  };
  const std::vector<Case> cases = {
      {"option_invalid_plus_one", "max_num_consecutive_invalid_steps", true},
      {"option_max_linear_plus_one", "max_linear_solver_iterations", true},
      {"loss_scale_plus_0_1", "loss_specification_sha256", false},
      {"swap_first_two_residuals", "source_residual_order_sha256", false}};
  for (size_t i = 0; i < cases.size(); ++i) {
    CeresFidelityReplayResult replay;
    std::string error;
    BOOST_REQUIRE(RunOriginalFidelityReplay(
        data.initial_write.manifest_path, data.record_write.manifest_path,
        TempRoot() + "/perturb/replay-" + std::to_string(i),
        "negative-" + std::to_string(i), cases[i].perturbation, &replay,
        &error));
    BOOST_CHECK(!replay.pass);
    if (cases[i].option) {
      BOOST_CHECK_EQUAL(replay.comparison.first_option_mismatch,
                        cases[i].expected_mismatch);
    } else {
      BOOST_CHECK_EQUAL(replay.comparison.first_problem_mismatch,
                        cases[i].expected_mismatch);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(UnsupportedLossFailsClosed, Fixture) {
  const std::vector<LossSpecificationSnapshot> losses = {
      {"visual", "UNSUPPORTED_TEST_LOSS", 1.0, 2},
      {"lidar", "TRIVIAL", 1.0, 1}};
  const OracleFixtureData data =
      WriteOracleFixture(losses, TempRoot() + "/unsupported-loss");
  CeresFidelityReplayResult replay;
  std::string error;
  BOOST_CHECK(!RunOriginalFidelityReplay(
      data.initial_write.manifest_path, data.record_write.manifest_path,
      TempRoot() + "/unsupported-loss/replay", "unsupported", "none",
      &replay, &error));
  BOOST_CHECK(error.find("Unsupported fidelity loss type") !=
              std::string::npos);
}

}  // namespace gpu_ba
}  // namespace colmap
