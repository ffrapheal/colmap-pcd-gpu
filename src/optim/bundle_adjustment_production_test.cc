#define TEST_NAME "gpu_ba/production_integration"
#include "util/testing.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "base/database_cache.h"
#include "base/projection.h"
#ifdef GPU_BA_CUDA_ENABLED
#include "gpu_ba/native_graph_problem_store.h"
#endif
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_mapper.h"

namespace colmap {
namespace {

struct SyntheticProblem {
  Reconstruction reconstruction;
  CorrespondenceGraph graph;
  BundleAdjustmentConfig config;
};

SyntheticProblem MakeSyntheticProblem(size_t unrelated_points = 0) {
  SyntheticProblem problem;
  constexpr size_t kNumPoints = 12;
  for (size_t i = 0; i < kNumPoints; ++i) {
    const double x = -0.8 + 0.14 * static_cast<double>(i);
    const double y = -0.3 + 0.05 * static_cast<double>(i % 5);
    problem.reconstruction.AddPoint3D(Eigen::Vector3d(x, y, 0.2), Track());
  }
  for (size_t i = 0; i < unrelated_points; ++i) {
    problem.reconstruction.AddPoint3D(
        Eigen::Vector3d(100.0 + i, 100.0, 100.0), Track());
  }

  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 800.0, 640, 480);
  camera.SetCameraId(0);
  problem.reconstruction.AddCamera(camera);

  for (image_t image_id = 0; image_id < 2; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(0);
    image.SetName(std::to_string(image_id));
    image.Qvec() = image_id == 1
        ? Eigen::Vector4d(1.0001, 0.0, 0.0, 0.0)
        : ComposeIdentityQuaternion();
    image.Tvec() = Eigen::Vector3d(0.2 * image_id, 0.0, 8.0);
    image.SetRegistered(true);
    const Eigen::Matrix3x4d projection = image.ProjectionMatrix();
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(kNumPoints);
    size_t point_index = 0;
    for (const auto& point : problem.reconstruction.Points3D()) {
      if (point_index++ == kNumPoints) break;
      Eigen::Vector2d xy =
          ProjectPointToImage(point.second.XYZ(), projection, camera);
      xy += Eigen::Vector2d(0.03 * image_id, -0.02 * image_id);
      points2D.push_back(xy);
    }
    problem.graph.AddImage(image_id, kNumPoints);
    image.SetPoints2D(points2D);
    problem.reconstruction.AddImage(image);
  }
  problem.reconstruction.SetUp(&problem.graph);
  for (image_t image_id = 0; image_id < 2; ++image_id) {
    point3D_t point_id = 1;
    for (point2D_t point2D_idx = 0; point2D_idx < kNumPoints;
         ++point2D_idx, ++point_id) {
      problem.reconstruction.AddObservation(
          point_id, TrackElement{image_id, point2D_idx});
    }
  }

  problem.config.AddImage(0);
  problem.config.AddImage(1);
  problem.config.SetConstantCamera(0);
  problem.config.SetConstantPose(0);
  problem.config.SetConstantTvec(1, {0});
  return problem;
}

void AddThirdTrackObservationImage(SyntheticProblem* problem) {
  CHECK_NOTNULL(problem);
  constexpr image_t kImageId = 2;
  constexpr size_t kNumPoints = 12;
  Image image;
  image.SetImageId(kImageId);
  image.SetCameraId(0);
  image.SetName(std::to_string(kImageId));
  image.Qvec() = ComposeIdentityQuaternion();
  image.Tvec() = Eigen::Vector3d(0.4, 0.0, 8.0);
  std::vector<Eigen::Vector2d> points2D;
  points2D.reserve(kNumPoints);
  for (point2D_t point2D_idx = 0; point2D_idx < kNumPoints;
       ++point2D_idx) {
    points2D.push_back(
        problem->reconstruction.Image(0).Point2D(point2D_idx).XY());
  }
  problem->graph.AddImage(kImageId, kNumPoints);
  image.SetPoints2D(points2D);
  image.SetUp(problem->reconstruction.Camera(0));
  problem->reconstruction.AddImage(image);
  problem->reconstruction.RegisterImage(kImageId);
  point3D_t point3D_id = 1;
  for (point2D_t point2D_idx = 0; point2D_idx < kNumPoints;
       ++point2D_idx, ++point3D_id) {
    problem->reconstruction.AddObservation(
        point3D_id, TrackElement{kImageId, point2D_idx});
  }
}

void AddZeroObservationSelectedImage(SyntheticProblem* problem) {
  CHECK_NOTNULL(problem);
  Camera camera = problem->reconstruction.Camera(0);
  camera.SetCameraId(1);
  problem->reconstruction.AddCamera(camera);
  Image image;
  image.SetImageId(2);
  image.SetCameraId(1);
  image.SetName("zero-observation-selected");
  image.SetQvec(ComposeIdentityQuaternion());
  image.SetTvec(Eigen::Vector3d(0.4, 0.0, 8.0));
  problem->graph.AddImage(2, 0);
  image.SetPoints2D(std::vector<Eigen::Vector2d>{});
  image.SetUp(problem->reconstruction.Camera(1));
  problem->reconstruction.AddImage(image);
  problem->reconstruction.RegisterImage(2);
  problem->config.AddImage(2);
}

BundleAdjustmentOptions CudaOptions(bool fallback) {
  BundleAdjustmentOptions options;
  options.ba_backend = "custom_cuda";
  options.ba_fallback_to_ceres = fallback;
  options.ba_snapshot_capture = "none";
  options.if_add_lidar_constraint = false;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.print_summary = false;
  options.solver_options.max_num_iterations = 0;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 0.0;
  options.solver_options.parameter_tolerance = 0.0;
  return options;
}

#ifdef GPU_BA_ENABLED
ActiveBaProblemSourceComparisonForTesting CompareProblemSources(
    SyntheticProblem* problem,
    const BundleAdjustmentOptions& options,
    const BundleAdjuster::OptimazePhrase phrase) {
  ActiveBaProblemSourceComparisonForTesting comparison;
  BundleAdjuster adjuster(options, problem->config);
  adjuster.SetOptimazePhrase(phrase);
  std::string error;
  BOOST_REQUIRE_MESSAGE(adjuster.CompareProblemSourcesForTesting(
                            &problem->reconstruction, &comparison, &error),
                        error);
  BOOST_CHECK(comparison.active.problem.canonical_order.empty());
  BOOST_CHECK_EQUAL(comparison.active.residual_block_count,
                    comparison.legacy.observations.size() +
                        comparison.legacy.lidar.size());
  BOOST_CHECK_EQUAL(comparison.active.scalar_residual_count,
                    2 * comparison.legacy.observations.size() +
                        comparison.legacy.lidar.size());
  return comparison;
}
#endif

#ifdef GPU_BA_CUDA_ENABLED
std::pair<int, int> ExpectedSyntheticReducedDimensions(
    SyntheticProblem* synthetic) {
  ceres::Problem problem;
  Image& image = synthetic->reconstruction.Image(1);
  problem.AddParameterBlock(image.Qvec().data(), 4,
                            new ceres::QuaternionParameterization());
  problem.AddParameterBlock(
      image.Tvec().data(), 3,
      new ceres::SubsetParameterization(3, std::vector<int>{0}));
  for (const auto& point : synthetic->reconstruction.Points3D()) {
    if (point.first <= 12) {
      problem.AddParameterBlock(
          synthetic->reconstruction.Point3D(point.first).XYZ().data(), 3);
    }
  }

  int ambient = 0;
  int tangent = 0;
  std::vector<double*> blocks;
  problem.GetParameterBlocks(&blocks);
  for (double* block : blocks) {
    ambient += problem.ParameterBlockSize(block);
    tangent += problem.ParameterBlockLocalSize(block);
  }
  return {ambient, tangent};
}
#endif

void CheckStateEqual(const Reconstruction& expected,
                     const Reconstruction& actual) {
  BOOST_REQUIRE_EQUAL(expected.NumImages(), actual.NumImages());
  BOOST_REQUIRE_EQUAL(expected.NumPoints3D(), actual.NumPoints3D());
  for (const auto& image : expected.Images()) {
    BOOST_REQUIRE(actual.ExistsImage(image.first));
    BOOST_CHECK_EQUAL(image.second.Qvec(), actual.Image(image.first).Qvec());
    BOOST_CHECK_EQUAL(image.second.Tvec(), actual.Image(image.first).Tvec());
  }
  for (const auto& point : expected.Points3D()) {
    BOOST_REQUIRE(actual.ExistsPoint3D(point.first));
    BOOST_CHECK_EQUAL(point.second.XYZ(), actual.Point3D(point.first).XYZ());
  }
  for (const auto& camera : expected.Cameras()) {
    BOOST_REQUIRE(actual.ExistsCamera(camera.first));
    BOOST_CHECK_EQUAL_COLLECTIONS(
        camera.second.Params().begin(), camera.second.Params().end(),
        actual.Camera(camera.first).Params().begin(),
        actual.Camera(camera.first).Params().end());
  }
}

struct FailureModeReset {
  ~FailureModeReset() {
    BundleAdjuster::SetFailureModeForTesting(
        BundleAdjuster::FailureModeForTesting::kNone);
  }
};

BOOST_AUTO_TEST_CASE(CompiledBackendDispatchAndIterationZero) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  const Reconstruction entry = problem.reconstruction;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_audit_profile = "production";
  BundleAdjuster adjuster(options, problem.config);
#ifdef GPU_BA_CUDA_ENABLED
  const auto expected_dimensions =
      ExpectedSyntheticReducedDimensions(&problem);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().executed_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().execution_profile,
                    "compact_control");
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().audit_profile_requested,
                    "production");
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().audit_profile_effective,
                    "production");
  BOOST_CHECK(
      adjuster.ExecutionResult().production_audit_invariants_checked);
  BOOST_CHECK(adjuster.ExecutionResult().production_audit_invariants_pass);
  BOOST_CHECK_EQUAL(
      adjuster.ExecutionResult().production_audit_violation_count, 0);
  BOOST_CHECK(!adjuster.ExecutionResult().capture_state_trace_effective);
  BOOST_CHECK(!adjuster.ExecutionResult().instrumentation_effective);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().state_hash_computations, 0);
  BOOST_CHECK_EQUAL(
      adjuster.ExecutionResult().topology_fingerprint_computations, 0);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().audit_mirror_bytes, 0);
  BOOST_CHECK_EQUAL(
      adjuster.ExecutionResult().optional_full_array_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(
      adjuster.ExecutionResult().mixed_double_edge_materialization_bytes, 0);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().accepted_commits, 0);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().initial_cost,
                    adjuster.ExecutionResult().final_cost);
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  BOOST_CHECK_EQUAL(adjuster.Summary().num_parameters_reduced,
                    expected_dimensions.first);
  BOOST_CHECK_EQUAL(adjuster.Summary().num_effective_parameters_reduced,
                    expected_dimensions.second);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      entry.Camera(0).Params().begin(), entry.Camera(0).Params().end(),
      problem.reconstruction.Camera(0).Params().begin(),
      problem.reconstruction.Camera(0).Params().end());
  BOOST_CHECK_EQUAL(entry.Image(0).Qvec(),
                    problem.reconstruction.Image(0).Qvec());
  BOOST_CHECK_EQUAL(entry.Image(1).Tvec(0),
                    problem.reconstruction.Image(1).Tvec(0));
#else
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().stable_error,
                    "UNSUPPORTED_CONFIGURATION");
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  CheckStateEqual(entry, problem.reconstruction);
#endif
}

BOOST_AUTO_TEST_CASE(TypedExecutionProfileParsesAndDispatches) {
  BOOST_CHECK(ParseBundleAdjustmentCudaExecutionProfile("baseline") ==
              BundleAdjustmentOptions::CudaExecutionProfile::BASELINE);
  BOOST_CHECK(ParseBundleAdjustmentCudaExecutionProfile("compact_control") ==
              BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL);
  BOOST_CHECK(ParseBundleAdjustmentCudaExecutionProfile("invalid") ==
              BundleAdjustmentOptions::CudaExecutionProfile::INVALID);
#ifdef GPU_BA_CUDA_ENABLED
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::BASELINE;
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().execution_profile, "baseline");
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().executed_backend, "custom_cuda");
#endif
}

BOOST_AUTO_TEST_CASE(TypedPrecisionHessianAndSchurSelectorsDispatch) {
#ifdef GPU_BA_CUDA_ENABLED
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_arithmetic_precision = "fp32_core";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  const auto& execution = adjuster.ExecutionResult();
  BOOST_CHECK_EQUAL(execution.requested_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(execution.executed_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(execution.arithmetic_precision_requested, "fp32_core");
  BOOST_CHECK_EQUAL(execution.arithmetic_precision_effective, "fp32_core");
  BOOST_CHECK_EQUAL(execution.hessian_backend_requested,
                    "observation_segmented");
  BOOST_CHECK_EQUAL(execution.hessian_backend_effective,
                    "observation_segmented");
  BOOST_CHECK_EQUAL(execution.hot_kernel_requested, "transformed");
  BOOST_CHECK_EQUAL(execution.hot_kernel_effective, "transformed");
  BOOST_CHECK_EQUAL(execution.schur_contribution_backend_requested,
                    "segmented");
  BOOST_CHECK_EQUAL(execution.schur_contribution_backend_effective,
                    "segmented");
  BOOST_CHECK_EQUAL(execution.factorization_routine,
                    "cusolverDnSpotrf/Spotrs");
  BOOST_CHECK_EQUAL(execution.cost_precision, "fp64");
  BOOST_CHECK_EQUAL(execution.controller_precision, "fp64");
  BOOST_CHECK(!execution.fallback_used);
  BOOST_CHECK_EQUAL(execution.ceres_solve_calls, 0);
#endif
}

BOOST_AUTO_TEST_CASE(TypedMixedPrecisionUsesStableMainControllerDispatch) {
#ifdef GPU_BA_CUDA_ENABLED
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp32_mixed";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  const auto& execution = adjuster.ExecutionResult();
  BOOST_CHECK_EQUAL(execution.requested_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(execution.executed_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(execution.arithmetic_precision_requested, "fp32_mixed");
  BOOST_CHECK_EQUAL(execution.arithmetic_precision_effective, "fp32_mixed");
  BOOST_CHECK_EQUAL(execution.audit_profile_requested, "production");
  BOOST_CHECK_EQUAL(execution.audit_profile_effective, "production");
  BOOST_CHECK(execution.production_audit_invariants_checked);
  BOOST_CHECK(execution.production_audit_invariants_pass);
  BOOST_CHECK_EQUAL(execution.production_audit_violation_count, 0);
  BOOST_CHECK(!execution.capture_state_trace_effective);
  BOOST_CHECK(!execution.instrumentation_effective);
  BOOST_CHECK_EQUAL(execution.state_hash_computations, 0);
  BOOST_CHECK_EQUAL(execution.topology_fingerprint_computations, 0);
  BOOST_CHECK_EQUAL(execution.audit_mirror_bytes, 0);
  BOOST_CHECK_EQUAL(execution.optional_full_array_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(execution.mixed_double_edge_materialization_bytes, 0);
  BOOST_CHECK_EQUAL(execution.hessian_backend_effective,
                    "observation_segmented");
  BOOST_CHECK_EQUAL(execution.hot_kernel_effective, "transformed");
  BOOST_CHECK_EQUAL(execution.schur_contribution_backend_effective,
                    "segmented");
  BOOST_CHECK_EQUAL(execution.factorization_routine,
                    "cusolverDnDpotrf/Dpotrs");
  BOOST_CHECK_EQUAL(execution.cost_precision, "fp64");
  BOOST_CHECK_EQUAL(execution.controller_precision, "fp64");
  BOOST_CHECK(!execution.fallback_used);
  BOOST_CHECK_EQUAL(execution.ceres_solve_calls, 0);
#endif
}

BOOST_AUTO_TEST_CASE(UnsupportedFallbackUsesOneCeresSolve) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(true);
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::CAUCHY;
  options.solver_options.max_num_iterations = 2;
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK(adjuster.ExecutionResult().fallback_used);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().executed_backend, "ceres_cpu");
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().ceres_solve_calls, 1);
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 1);
}

BOOST_AUTO_TEST_CASE(FallbackFailureRestoresEntryState) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  const Reconstruction entry = problem.reconstruction;
  BundleAdjustmentOptions options = CudaOptions(true);
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::CAUCHY;
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCeresFailure);
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().stable_error,
                    "CERES_FALLBACK_FAILED");
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 1);
  BOOST_CHECK_NE(adjuster.ExecutionResult().diagnostic_message.find(
                     "non-qvec parameter mutation"),
                 std::string::npos);
  CheckStateEqual(entry, problem.reconstruction);
}

#ifdef GPU_BA_CUDA_ENABLED
BOOST_AUTO_TEST_CASE(UnsupportedStopsBeforeCudaAndCeres) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  const Reconstruction entry = problem.reconstruction;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::CAUCHY;
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().stable_error,
                    "UNSUPPORTED_CONFIGURATION");
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  CheckStateEqual(entry, problem.reconstruction);
}

BOOST_AUTO_TEST_CASE(SupportedSoftL1PreservesFixedAndSubsetBlocks) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  problem.reconstruction.Image(1).Tvec(1) += 0.15;
  const Eigen::Vector4d fixed_qvec = problem.reconstruction.Image(0).Qvec();
  const double fixed_translation_x = problem.reconstruction.Image(1).Tvec(0);
  const std::vector<double> fixed_camera =
      problem.reconstruction.Camera(0).Params();
  const Eigen::Vector3d variable_tvec = problem.reconstruction.Image(1).Tvec();
  const Eigen::Vector3d variable_point = problem.reconstruction.Point3D(1).XYZ();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  options.solver_options.max_num_iterations = 5;
  BundleAdjuster adjuster(options, problem.config);
  adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().executed_backend, "custom_cuda");
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  BOOST_CHECK_GT(adjuster.ExecutionResult().accepted_commits, 0);
  BOOST_CHECK_EQUAL(fixed_qvec, problem.reconstruction.Image(0).Qvec());
  BOOST_CHECK_EQUAL(fixed_translation_x,
                    problem.reconstruction.Image(1).Tvec(0));
  BOOST_CHECK_EQUAL_COLLECTIONS(
      fixed_camera.begin(), fixed_camera.end(),
      problem.reconstruction.Camera(0).Params().begin(),
      problem.reconstruction.Camera(0).Params().end());
  BOOST_CHECK((variable_tvec - problem.reconstruction.Image(1).Tvec()).norm() >
                  0.0 ||
              (variable_point - problem.reconstruction.Point3D(1).XYZ()).norm() >
                  0.0);
}

BOOST_AUTO_TEST_CASE(NoFallbackRollbackAndScopedCheckpoint) {
  FailureModeReset reset;
  BundleAdjustmentOptions options = CudaOptions(false);
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);

  SyntheticProblem small = MakeSyntheticProblem();
  const Reconstruction entry = small.reconstruction;
  BundleAdjuster small_adjuster(options, small.config);
  BOOST_CHECK(!small_adjuster.Solve(&small.reconstruction));
  CheckStateEqual(entry, small.reconstruction);
  BOOST_CHECK_EQUAL(small_adjuster.ExecutionResult().ceres_solve_calls, 0);

  SyntheticProblem large = MakeSyntheticProblem(500);
  BundleAdjuster large_adjuster(options, large.config);
  BOOST_CHECK(!large_adjuster.Solve(&large.reconstruction));
  BOOST_CHECK_EQUAL(
      small_adjuster.ExecutionResult().transaction_parameter_block_count,
      large_adjuster.ExecutionResult().transaction_parameter_block_count);
  BOOST_CHECK_EQUAL(small_adjuster.ExecutionResult().transaction_backup_bytes,
                    large_adjuster.ExecutionResult().transaction_backup_bytes);
  BOOST_CHECK_EQUAL(
      small_adjuster.ExecutionResult().transaction_identity_index_entries,
      large_adjuster.ExecutionResult().transaction_identity_index_entries);
  BOOST_CHECK_EQUAL(
      small_adjuster.ExecutionResult().transaction_parameter_lookup_count,
      small_adjuster.ExecutionResult().transaction_parameter_block_count);
  BOOST_CHECK_EQUAL(
      large_adjuster.ExecutionResult().transaction_parameter_lookup_count,
      large_adjuster.ExecutionResult().transaction_parameter_block_count);
}

BOOST_AUTO_TEST_CASE(RestoreValidationFailureDoesNotPartiallyWrite) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  Reconstruction expected_after_failed_validation = problem.reconstruction;
  for (const image_t image_id : problem.config.Images()) {
    expected_after_failed_validation.Image(image_id).NormalizeQvec();
  }
  BundleAdjustmentOptions options = CudaOptions(false);
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kRestoreValidationFailure);
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().stable_error,
                    "TRANSACTION_RESTORE_FAILED");
  CheckStateEqual(expected_after_failed_validation, problem.reconstruction);
}

BOOST_AUTO_TEST_CASE(CommitIntegrityFailureNeverFallsBack) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  const Reconstruction entry = problem.reconstruction;
  BundleAdjustmentOptions options = CudaOptions(true);
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCudaTopologyMismatch);
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().stable_error,
                    "COMMIT_INTEGRITY_ERROR");
  BOOST_CHECK(!adjuster.ExecutionResult().fallback_used);
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  CheckStateEqual(entry, problem.reconstruction);
}

BOOST_AUTO_TEST_CASE(IterationZeroGradientConvergenceIsValid) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.solver_options.max_num_iterations = 3;
  options.solver_options.gradient_tolerance = 1e100;
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().accepted_commits, 0);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().initial_cost,
                    adjuster.ExecutionResult().final_cost);
}

BOOST_AUTO_TEST_CASE(TelemetryEscapesDiagnosticsAndUsesStableError) {
  FailureModeReset reset;
  const std::string path = "/tmp/colmap-pcd-phase8-r1-telemetry.jsonl";
  std::remove(path.c_str());
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_telemetry_path = path;
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  std::ifstream file(path);
  BOOST_REQUIRE(file.good());
  std::string line;
  BOOST_REQUIRE(static_cast<bool>(std::getline(file, line)));
  std::istringstream json_stream(line);
  boost::property_tree::ptree json;
  BOOST_REQUIRE_NO_THROW(boost::property_tree::read_json(json_stream, json));
  BOOST_CHECK_EQUAL(json.get<std::string>("stable_error"),
                    "CUSTOM_CUDA_FAILED");
  BOOST_CHECK_EQUAL(json.get<std::string>("ceres_solve_calls_scope"),
                    "bundle_adjuster_only");
  BOOST_CHECK_NE(line.find("\"stable_error\":\"CUSTOM_CUDA_FAILED\""),
                 std::string::npos);
  BOOST_CHECK_NE(line.find("forced \\\"custom CUDA\\\" failure\\n"),
                 std::string::npos);
  BOOST_CHECK_NE(line.find(
                     "\"ceres_solve_calls_scope\":\"bundle_adjuster_only\""),
                 std::string::npos);
  BOOST_CHECK_NE(line.find("\"initial_cost\":null"), std::string::npos);
  BOOST_CHECK_NE(line.find("\"final_cost\":null"), std::string::npos);
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreColdDeltaExactAndRollback) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11001;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  BOOST_REQUIRE_EQUAL(problem.reconstruction.Image(0).NumPoints2D(), 12);
  BOOST_REQUIRE_EQUAL(problem.reconstruction.Image(1).NumPoints2D(), 12);
  uint64_t cold_builder_calls = 0;
  uint64_t cold_builder_traversals = 0;

  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    const bool cold_ok = cold.Solve(&problem.reconstruction);
    BOOST_TEST_MESSAGE(cold.ExecutionResult().diagnostic_message);
    BOOST_REQUIRE(cold_ok);
    const auto& execution = cold.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_mode_effective,
                      "host_prepared_store");
    BOOST_CHECK_EQUAL(execution.host_store_view_action,
                      "solve_view_rebuild");
    BOOST_CHECK_EQUAL(execution.host_store_misses, 1);
    cold_builder_calls = execution.host_store_builder_calls_executed;
    cold_builder_traversals =
        execution.host_store_builder_traversals_executed;
    BOOST_CHECK_GT(cold_builder_calls, 0);
    BOOST_CHECK_EQUAL(
        cold_builder_calls,
        execution.host_store_static_binding_builder_calls +
            execution.host_store_cost_layout_builder_calls +
            execution.host_store_hessian_topology_builder_calls +
            execution.host_store_hessian_segment_plan_builder_calls +
            execution.host_store_schur_topology_builder_calls +
            execution.host_store_schur_segment_plan_builder_calls);
    BOOST_CHECK_GT(cold_builder_traversals, 0);
    BOOST_CHECK_EQUAL(execution.host_store_dynamic_state_refresh_calls, 1);
    BOOST_CHECK_GT(execution.host_store_descriptor_identity, 0);
    BOOST_CHECK_GT(execution.host_store_descriptor_items, 0);
    BOOST_CHECK_GT(execution.host_store_descriptor_hash_updates, 0);
    BOOST_CHECK_GT(execution.host_store_catalog_generation, 0);
    BOOST_CHECK_GT(execution.host_store_dynamic_refresh_milliseconds, 0.0);
    BOOST_CHECK_GE(execution.host_store_preparation_total_milliseconds,
                   execution.host_store_dynamic_refresh_milliseconds);
    BOOST_CHECK_EQUAL(execution.host_store_journal_cursor_after,
                      store.LifetimeRuntimeInfo().journal_cursor_after);
  }

  // The new point is outside the selected BA. The catalog must consume the
  // journal delta while the local SolveView is prepared from catalog fragments.
  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(100.0, 100.0, 100.0), Track());
  const uint64_t revision_after_delta =
      problem.reconstruction.StructureRevision();
  {
    BundleAdjuster patched(options, problem.config);
    patched.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    patched.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(patched.Solve(&problem.reconstruction));
    const auto& execution = patched.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_catalog_delta_updates, 1);
    BOOST_CHECK_EQUAL(execution.host_store_hits, 1);
    BOOST_CHECK_EQUAL(execution.host_store_exact_view_reuses, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_patches, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_action, "exact_view_reuse");
    BOOST_CHECK_EQUAL(execution.host_store_builder_calls_executed, 0);
    BOOST_CHECK_EQUAL(execution.host_store_builder_calls_saved,
                      cold_builder_calls);
    BOOST_CHECK_EQUAL(execution.host_store_builder_traversals_saved,
                      cold_builder_traversals);
  }
  BOOST_CHECK_EQUAL(store.LifetimeRuntimeInfo().journal_cursor_after,
                    revision_after_delta);

  {
    BundleAdjuster exact(options, problem.config);
    exact.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    exact.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(exact.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(exact.ExecutionResult().host_store_hits, 1);
    BOOST_CHECK_EQUAL(exact.ExecutionResult().host_store_exact_view_reuses, 1);
    BOOST_CHECK_EQUAL(exact.ExecutionResult().host_store_view_action,
                      "exact_view_reuse");
  }

  // A failed pending view with a different fixed mask must not replace the
  // already-published exact view.
  BundleAdjustmentConfig fixed_point_config = problem.config;
  fixed_point_config.AddConstantPoint(1);
  const uint64_t cursor_before_fixed_mask_failure =
      store.LifetimeRuntimeInfo().journal_cursor_after;
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);
  {
    BundleAdjuster failed_fixed_mask(options, fixed_point_config);
    failed_fixed_mask.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    failed_fixed_mask.SetCudaHostStoreBinding(binding);
    BOOST_CHECK(!failed_fixed_mask.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(failed_fixed_mask.ExecutionResult().host_store_hits, 0);
    BOOST_CHECK_EQUAL(
        failed_fixed_mask.ExecutionResult().host_store_view_rebuilds, 1);
  }
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kNone);
  BOOST_CHECK_EQUAL(store.LifetimeRuntimeInfo().journal_cursor_after,
                    cursor_before_fixed_mask_failure);
  {
    BundleAdjuster retained_exact(options, problem.config);
    retained_exact.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    retained_exact.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(retained_exact.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(retained_exact.ExecutionResult().host_store_hits, 1);
    BOOST_CHECK_EQUAL(
        retained_exact.ExecutionResult().host_store_exact_view_reuses, 1);
  }

  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(101.0, 100.0, 100.0), Track());
  const uint64_t revision_before_failure =
      store.LifetimeRuntimeInfo().journal_cursor_after;
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);
  {
    BundleAdjuster failed(options, problem.config);
    failed.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    failed.SetCudaHostStoreBinding(binding);
    BOOST_CHECK(!failed.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(failed.ExecutionResult().host_store_catalog_delta_updates,
                      1);
  }
  BOOST_CHECK_EQUAL(store.LifetimeRuntimeInfo().journal_cursor_after,
                    revision_before_failure);
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kNone);
  {
    BundleAdjuster retry(options, problem.config);
    retry.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    retry.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(retry.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(retry.ExecutionResult().host_store_catalog_delta_updates,
                      1);
  }
  BOOST_CHECK_GT(store.LifetimeRuntimeInfo().journal_cursor_after,
                 revision_before_failure);
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreObservationAndFixedMaskRebuildFromCatalog) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  AddThirdTrackObservationImage(&problem);
  constexpr uint64_t kOwnerEpoch = 11005;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";

  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
  }

  BundleAdjustmentConfig fixed_point_config = problem.config;
  fixed_point_config.AddConstantPoint(1);
  {
    BundleAdjuster fixed_mask(options, fixed_point_config);
    fixed_mask.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    fixed_mask.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(fixed_mask.Solve(&problem.reconstruction));
    const auto& execution = fixed_mask.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_hits, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_patches, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_action,
                      "solve_view_rebuild_from_catalog");
  }

  problem.reconstruction.DeleteObservation(0, 0);
  {
    BundleAdjuster deleted_observation(options, problem.config);
    deleted_observation.SetOptimazePhrase(
        BundleAdjuster::OptimazePhrase::Local);
    deleted_observation.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(deleted_observation.Solve(&problem.reconstruction));
    const auto& execution = deleted_observation.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_catalog_delta_updates, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_patches, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_action,
                      "solve_view_rebuild_from_catalog");
  }

  problem.reconstruction.AddObservation(1, TrackElement{0, 0});
  {
    BundleAdjuster restored_observation(options, problem.config);
    restored_observation.SetOptimazePhrase(
        BundleAdjuster::OptimazePhrase::Local);
    restored_observation.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(restored_observation.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(
        restored_observation.ExecutionResult().host_store_catalog_delta_updates,
        1);
    BOOST_CHECK_EQUAL(
        restored_observation.ExecutionResult().host_store_hits, 1);
    BOOST_CHECK_EQUAL(
        restored_observation.ExecutionResult().host_store_exact_view_reuses,
        1);
    BOOST_CHECK_EQUAL(
        restored_observation.ExecutionResult().host_store_view_rebuilds, 0);
  }

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreUnknownAndGapRebuildTransactionally) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11006;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 1);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";

  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
  }

  problem.reconstruction.MarkStructureUnknown("focused-unknown");
  {
    BundleAdjuster unknown(options, problem.config);
    unknown.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    unknown.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(unknown.Solve(&problem.reconstruction));
    const auto& execution = unknown.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_view_patches, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 1);
    const auto lifetime = store.LifetimeRuntimeInfo();
    BOOST_CHECK_EQUAL(lifetime.journal_unknown_events, 1);
    BOOST_CHECK_EQUAL(lifetime.catalog_full_rebuilds, 1);
    BOOST_CHECK_EQUAL(lifetime.fallback_rebuilds, 1);
  }

  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(201.0, 100.0, 100.0), Track());
  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(202.0, 100.0, 100.0), Track());
  {
    BundleAdjuster gap(options, problem.config);
    gap.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    gap.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(gap.Solve(&problem.reconstruction));
    const auto& execution = gap.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_view_patches, 0);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 1);
    const auto lifetime = store.LifetimeRuntimeInfo();
    BOOST_CHECK_EQUAL(lifetime.journal_gaps, 1);
    BOOST_CHECK_EQUAL(lifetime.catalog_full_rebuilds, 2);
    BOOST_CHECK_EQUAL(lifetime.fallback_rebuilds, 2);
  }

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(IndexedCatalogGapRebuildPublishesOnlyAfterCommit) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11009;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 1);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kIndexedCatalog;

  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(
        cold.ExecutionResult().host_store_indexed_catalog_full_graph_build_calls,
        1);
  }
  const uint64_t cursor_before =
      store.LifetimeRuntimeInfo().journal_cursor_after;
  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(301.0, 100.0, 100.0), Track());
  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(302.0, 100.0, 100.0), Track());
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);
  {
    BundleAdjuster failed(options, problem.config);
    failed.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    failed.SetCudaHostStoreBinding(binding);
    BOOST_CHECK(!failed.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(failed.ExecutionResult().host_store_journal_gaps, 1);
    BOOST_CHECK_EQUAL(
        store.LifetimeRuntimeInfo().journal_cursor_after, cursor_before);
  }
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kNone);
  {
    BundleAdjuster retry(options, problem.config);
    retry.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    retry.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(retry.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(retry.ExecutionResult().host_store_journal_gaps, 1);
    BOOST_CHECK_EQUAL(
        retry.ExecutionResult().host_store_catalog_full_rebuilds, 1);
    BOOST_CHECK_GT(store.LifetimeRuntimeInfo().journal_cursor_after,
                   cursor_before);
  }

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(IndexedCatalogConsumesKnownJournalDelta) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11010;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kIndexedCatalog;
  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
  }
  const uint64_t cursor_before =
      store.LifetimeRuntimeInfo().journal_cursor_after;
  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(401.0, 100.0, 100.0), Track());
  {
    BundleAdjuster delta(options, problem.config);
    delta.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    delta.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(delta.Solve(&problem.reconstruction));
    const auto& execution = delta.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_catalog_delta_updates, 1);
    BOOST_CHECK_EQUAL(
        execution.host_store_indexed_catalog_journal_apply_calls, 1);
    BOOST_CHECK_EQUAL(
        execution.host_store_indexed_catalog_full_graph_build_calls, 0);
    BOOST_CHECK_GT(execution.host_store_indexed_full_graph_records_scanned, 0);
    BOOST_CHECK_GT(execution.host_store_indexed_estimated_impl_copy_bytes, 0);
  }
  BOOST_CHECK_GT(store.LifetimeRuntimeInfo().journal_cursor_after,
                 cursor_before);

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(IndexedCatalogStaleMergedPointRebuildsTransactionally) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11011;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kIndexedCatalog;

  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
  }
  const uint64_t cursor_before =
      store.LifetimeRuntimeInfo().journal_cursor_after;
  const point3D_t merged = problem.reconstruction.MergePoints3D(1, 2);
  problem.reconstruction.DeletePoint3D(merged);

  {
    BundleAdjuster rebuilt(options, problem.config);
    rebuilt.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    rebuilt.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE_MESSAGE(rebuilt.Solve(&problem.reconstruction),
                          rebuilt.ExecutionResult().diagnostic_message);
    const BundleAdjustmentExecutionResult& execution =
        rebuilt.ExecutionResult();
    BOOST_CHECK_EQUAL(
        execution.host_store_indexed_catalog_journal_apply_calls, 1);
    BOOST_CHECK_EQUAL(
        execution.host_store_indexed_catalog_full_graph_build_calls, 1);
    BOOST_CHECK_EQUAL(execution.host_store_catalog_delta_updates, 0);
    BOOST_CHECK_EQUAL(execution.host_store_catalog_full_rebuilds, 1);
    BOOST_CHECK_EQUAL(execution.host_store_fallback_rebuilds, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_action,
                      "indexed_catalog_full_rebuild");
    BOOST_CHECK_EQUAL(execution.host_store_rebuild_reason,
                      "catalog_delta_apply_failed");
  }
  BOOST_CHECK_GT(store.LifetimeRuntimeInfo().journal_cursor_after,
                 cursor_before);

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(HostPreparedStorePrecisionIdentityDoesNotLeak) {
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11002;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  for (const std::string& precision : {std::string("fp64"),
                                      std::string("fp32_mixed"),
                                      std::string("fp64")}) {
    BundleAdjustmentOptions options = CudaOptions(false);
    options.ba_cuda_host_problem_store = "host_prepared_store";
    options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kIndexedCatalog;
    options.ba_cuda_arithmetic_precision = precision;
    if (precision == "fp32_mixed") {
      options.ba_cuda_hessian_assembly_backend = "observation_segmented";
      options.ba_cuda_schur_contribution_backend = "segmented";
    }
    BundleAdjuster adjuster(options, problem.config);
    adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    adjuster.SetCudaHostStoreBinding(binding);
    const bool solve_ok = adjuster.Solve(&problem.reconstruction);
    BOOST_TEST_MESSAGE(adjuster.ExecutionResult().diagnostic_message);
    BOOST_REQUIRE(solve_ok);
    BOOST_CHECK_EQUAL(adjuster.ExecutionResult().arithmetic_precision_effective,
                      precision);
    BOOST_CHECK(!adjuster.ExecutionResult().fallback_used);
    const BundleAdjustmentExecutionResult& execution =
        adjuster.ExecutionResult();
    BOOST_CHECK_GT(execution.indexed_device_catalog_lookup_calls, 0);
    BOOST_CHECK_GT(execution.indexed_device_catalog_prefix_bytes, 0);
    BOOST_CHECK_GT(execution.indexed_device_catalog_full_upload_calls +
                       execution.indexed_device_catalog_reuse_calls,
                   0);
    BOOST_CHECK_EQUAL(execution.cuda_host_build_cuda_layer_a_inputs_calls, 1);
    BOOST_CHECK_EQUAL(execution.cuda_host_build_static_layout_calls, 1);
    BOOST_CHECK_EQUAL(execution.cuda_host_build_cost_layout_calls, 1);
    BOOST_CHECK_EQUAL(execution.cuda_host_build_layer_b_topology_calls, 1);
    BOOST_CHECK_EQUAL(execution.cuda_host_build_layer_c_topology_calls, 1);
  }
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(StructureJournalNestedAbortAndGapAreStable) {
  Reconstruction reconstruction;
  constexpr uint64_t kOwnerEpoch = 11003;
  reconstruction.BeginStructureJournal(kOwnerEpoch, 1);
  const uint64_t initial_revision = reconstruction.StructureRevision();
  {
    Reconstruction::StructureMutationBatch outer(&reconstruction);
    reconstruction.MarkStructureUnknown("nested-event");
    Reconstruction::StructureMutationBatch nested(&reconstruction);
    nested.Cancel();
  }
  BOOST_CHECK_EQUAL(reconstruction.StructureRevision(), initial_revision);

  reconstruction.MarkStructureUnknown("first");
  reconstruction.MarkStructureUnknown("second");
  const ReconstructionStructureReadResult gap =
      reconstruction.ReadStructureEventsSince(kOwnerEpoch, initial_revision);
  BOOST_CHECK(gap.gap);
  BOOST_CHECK(!gap.complete);
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreBusyAndSharedControlLifetime) {
  gpu_ba::CudaHostProblemStoreLeaseTestResult result;
  std::string error;
  BOOST_REQUIRE(
      gpu_ba::RunCudaHostProblemStoreLeaseLifetimeForTesting(&result, &error));
  BOOST_CHECK(result.shutdown_reported_store_busy);
  BOOST_CHECK(result.control_survived_owner_destruction);
  BOOST_CHECK(result.lease_completion_succeeded);
  BOOST_CHECK(result.deferred_shutdown_completed);
  BOOST_CHECK(result.mismatch_released_matching_active_lease);
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreDisabledAndEnabledMapperLifecycle) {
  DatabaseCache database_cache;
  IncrementalMapper mapper(&database_cache);

  Reconstruction disabled;
  mapper.BeginReconstruction(
      &disabled, gpu_ba::CudaHostProblemStoreMode::kDisabled);
  BOOST_CHECK(mapper.CudaHostStoreModeForTesting() ==
              gpu_ba::CudaHostProblemStoreMode::kDisabled);
  BOOST_CHECK_EQUAL(mapper.CudaHostStoreOwnerEpochForTesting(), 0);
  BOOST_CHECK(!mapper.HasCudaHostProblemStoreForTesting());
  BOOST_CHECK(!disabled.StructureJournalEnabled());
  mapper.EndReconstruction(false);
  BOOST_CHECK(!disabled.StructureJournalEnabled());

  Reconstruction enabled;
  mapper.BeginReconstruction(
      &enabled, gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore);
  BOOST_CHECK(mapper.CudaHostStoreModeForTesting() ==
              gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore);
  BOOST_CHECK(mapper.HasCudaHostProblemStoreForTesting());
  BOOST_CHECK(enabled.StructureJournalEnabled());
  const gpu_ba::CudaHostStoreBinding local =
      mapper.CudaHostStoreBindingForTesting();
  const gpu_ba::CudaHostStoreBinding global =
      mapper.CudaHostStoreBindingForTesting();
  const gpu_ba::CudaHostStoreBinding whole =
      mapper.CudaHostStoreBindingForTesting();
  BOOST_CHECK_GT(local.owner_epoch, 0);
  BOOST_CHECK_EQUAL(local.owner_epoch, global.owner_epoch);
  BOOST_CHECK_EQUAL(local.owner_epoch, whole.owner_epoch);
  BOOST_CHECK_EQUAL(local.store, global.store);
  BOOST_CHECK_EQUAL(local.store, whole.store);
  const uint64_t first_owner = local.owner_epoch;
  mapper.EndReconstruction(false);
  BOOST_CHECK(!enabled.StructureJournalEnabled());
  BOOST_CHECK(!mapper.HasCudaHostProblemStoreForTesting());
  BOOST_CHECK_EQUAL(mapper.CudaHostStoreOwnerEpochForTesting(), 0);

  Reconstruction next;
  mapper.BeginReconstruction(
      &next, gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore);
  BOOST_CHECK_NE(mapper.CudaHostStoreOwnerEpochForTesting(), first_owner);
  mapper.EndReconstruction(false);
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreDescriptorIsConditional) {
  FailureModeReset reset;
  {
    SyntheticProblem problem = MakeSyntheticProblem();
    BundleAdjustmentOptions options = CudaOptions(false);
    options.ba_cuda_host_problem_store = "disabled";
    BundleAdjuster disabled(options, problem.config);
    disabled.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    BOOST_REQUIRE(disabled.Solve(&problem.reconstruction));
    const auto& execution = disabled.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_descriptor_identity, 0);
    BOOST_CHECK_EQUAL(execution.host_store_descriptor_items, 0);
    BOOST_CHECK_EQUAL(execution.host_store_descriptor_hash_updates, 0);
    BOOST_CHECK_EQUAL(execution.host_store_lookup_calls, 0);
    BOOST_CHECK_GT(execution.host_store_prepare_and_cuda_wall_seconds, 0.0);
  }

  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11007;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  BundleAdjuster enabled(options, problem.config);
  enabled.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  enabled.SetCudaHostStoreBinding(binding);
  BOOST_REQUIRE(enabled.Solve(&problem.reconstruction));
  BOOST_CHECK_GT(enabled.ExecutionResult().host_store_descriptor_identity, 0);
  BOOST_CHECK_GT(enabled.ExecutionResult().host_store_descriptor_items, 0);
  BOOST_CHECK_GT(
      enabled.ExecutionResult().host_store_descriptor_hash_updates, 0);
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(HostPreparedStoreHashCollisionRequiresCanonicalMatch) {
  FailureModeReset reset;
  struct HashOverrideReset {
    ~HashOverrideReset() {
      gpu_ba::SetCudaHostProblemStoreLookupHashForTesting(0);
    }
  } reset_hash;
  gpu_ba::SetCudaHostProblemStoreLookupHashForTesting(0x1100a11u);

  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11008;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  {
    BundleAdjuster cold(options, problem.config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
  }
  BundleAdjustmentConfig fixed = problem.config;
  fixed.AddConstantPoint(1);
  {
    BundleAdjuster collision(options, fixed);
    collision.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    collision.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(collision.Solve(&problem.reconstruction));
    const auto& execution = collision.ExecutionResult();
    BOOST_CHECK_EQUAL(execution.host_store_hits, 0);
    BOOST_CHECK_EQUAL(execution.host_store_misses, 1);
    BOOST_CHECK_EQUAL(execution.host_store_view_rebuilds, 1);
    BOOST_CHECK_EQUAL(execution.host_store_rebuild_reason,
                      "canonical_descriptor_mismatch");
  }
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(ActiveSpecFastPathAvoidsLegacyProblemAndMatchesLegacy) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem legacy_problem = MakeSyntheticProblem();
  SyntheticProblem active_problem = MakeSyntheticProblem();
  BundleAdjustmentOptions legacy_options = CudaOptions(false);
  legacy_options.ba_cuda_audit_profile = "production";
  BundleAdjustmentOptions active_options = legacy_options;
  active_options.ba_cuda_problem_source =
      gpu_ba::CudaProblemSource::kActiveSpec;

  BundleAdjuster legacy(legacy_options, legacy_problem.config);
  legacy.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE(legacy.Solve(&legacy_problem.reconstruction));
  BundleAdjuster active(active_options, active_problem.config);
  active.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE_MESSAGE(active.Solve(&active_problem.reconstruction),
                        active.ExecutionResult().diagnostic_message);

  const auto& reference = legacy.ExecutionResult();
  const auto& candidate = active.ExecutionResult();
  BOOST_CHECK_EQUAL(candidate.problem_source_requested, "active_spec");
  BOOST_CHECK_EQUAL(candidate.problem_source_effective, "active_spec");
  BOOST_CHECK(!candidate.ceres_problem_created);
  BOOST_CHECK_EQUAL(candidate.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(candidate.ceres_add_residual_calls, 0);
  BOOST_CHECK(!candidate.snapshot_recorder_created);
  BOOST_CHECK_EQUAL(candidate.snapshot_materialization_calls, 0);
  BOOST_CHECK_EQUAL(candidate.active_spec_build_calls, 1);
  BOOST_CHECK_EQUAL(candidate.residual_enumerator_passes, 1);
  BOOST_CHECK_GT(candidate.residual_enumerator_items, 0);
  BOOST_CHECK_EQUAL(candidate.residuals, reference.residuals);
  BOOST_CHECK_EQUAL(candidate.residual_blocks, reference.residual_blocks);
  BOOST_CHECK_EQUAL(candidate.parameter_blocks, reference.parameter_blocks);
  BOOST_CHECK_EQUAL(candidate.parameters, reference.parameters);
  BOOST_CHECK_EQUAL(candidate.effective_parameters,
                    reference.effective_parameters);
  BOOST_CHECK_EQUAL(candidate.initial_cost, reference.initial_cost);
  BOOST_CHECK_EQUAL(candidate.final_cost, reference.final_cost);
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 0);
  CheckStateEqual(legacy_problem.reconstruction,
                  active_problem.reconstruction);
}

BOOST_AUTO_TEST_CASE(ActiveSpecRejectsCaptureWithoutCreatingLegacyProblem) {
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(true);
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
  options.ba_snapshot_capture = "local";
  BundleAdjuster adjuster(options, problem.config);
  BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
  const auto& execution = adjuster.ExecutionResult();
  BOOST_CHECK_EQUAL(execution.problem_source_effective, "invalid");
  BOOST_CHECK_EQUAL(execution.stable_error, "UNSUPPORTED_CONFIGURATION");
  BOOST_CHECK(!execution.ceres_problem_created);
  BOOST_CHECK_EQUAL(execution.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(execution.snapshot_materialization_calls, 0);
}

BOOST_AUTO_TEST_CASE(ActiveSpecCudaFailureBuildsLazyCeresFallback) {
  FailureModeReset reset;
  BundleAdjuster::ResetCeresSolveCallCountForTesting();
  SyntheticProblem problem = MakeSyntheticProblem();
  BundleAdjustmentOptions options = CudaOptions(true);
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kCustomCudaFailure);
  BundleAdjuster adjuster(options, problem.config);
  BOOST_REQUIRE(adjuster.Solve(&problem.reconstruction));
  const auto& execution = adjuster.ExecutionResult();
  BOOST_CHECK(execution.fallback_used);
  BOOST_CHECK_EQUAL(execution.problem_source_requested, "active_spec");
  BOOST_CHECK_EQUAL(execution.problem_source_effective, "legacy_snapshot");
  BOOST_CHECK_EQUAL(execution.problem_source_fallback_reason,
                    "CUSTOM_CUDA_FAILURE_LAZY_CERES");
  BOOST_CHECK(execution.ceres_problem_created);
  BOOST_CHECK_GT(execution.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(execution.ceres_cost_function_creations,
                    execution.ceres_add_residual_calls);
  BOOST_CHECK_EQUAL(execution.snapshot_materialization_calls, 0);
  BOOST_CHECK_EQUAL(BundleAdjuster::CeresSolveCallCountForTesting(), 1);
}

BOOST_AUTO_TEST_CASE(ActiveSpecExactComparatorRepresentativeMatrix) {
  BundleAdjustmentOptions base = CudaOptions(false);

  SyntheticProblem local = MakeSyntheticProblem();
  const auto local_comparison = CompareProblemSources(
      &local, base, BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE_EQUAL(local_comparison.active.problem.images.size(), 2);
  const auto translation = std::find_if(
      local_comparison.active.parameter_blocks.begin(),
      local_comparison.active.parameter_blocks.end(),
      [](const gpu_ba::ActiveBaParameterBlockSpec& value) {
        return value.kind == gpu_ba::ParameterKind::kTranslation &&
               value.entity_id == 1;
      });
  BOOST_REQUIRE(translation != local_comparison.active.parameter_blocks.end());
  BOOST_CHECK_EQUAL(translation->translation_subset_mask, 1u);

  SyntheticProblem partial = MakeSyntheticProblem();
  BundleAdjustmentConfig partial_config;
  partial_config.AddImage(0);
  partial_config.SetConstantCamera(0);
  partial_config.SetConstantPose(0);
  partial.config = partial_config;
  const auto partial_comparison = CompareProblemSources(
      &partial, base, BundleAdjuster::OptimazePhrase::Local);
  BOOST_CHECK_GT(partial_comparison.legacy.images.size(),
                 partial_comparison.active.problem.images.size());
  BOOST_REQUIRE(!partial_comparison.active.problem.points.empty());
  BOOST_CHECK(partial_comparison.active.problem.points.front().constant);

  SyntheticProblem external = MakeSyntheticProblem();
  BundleAdjustmentConfig external_config;
  external_config.AddImage(0);
  external_config.SetConstantCamera(0);
  external_config.SetConstantPose(0);
  external_config.AddVariablePoint(1);
  external.config = external_config;
  const auto external_comparison = CompareProblemSources(
      &external, base, BundleAdjuster::OptimazePhrase::Local);
  BOOST_CHECK(std::any_of(
      external_comparison.active.problem.observations.begin(),
      external_comparison.active.problem.observations.end(),
      [](const gpu_ba::ObservationSnapshot& value) {
        return value.image_id == 1 && value.point3D_id == 1;
      }));

  SyntheticProblem global = MakeSyntheticProblem();
  for (const auto& point : global.reconstruction.Points3D()) {
    global.reconstruction.Point3D(point.first).IfInSphere() =
        (point.first % 2) == 0;
  }
  BundleAdjustmentOptions lidar_mode = base;
  lidar_mode.if_add_lidar_constraint = true;
  const auto global_comparison = CompareProblemSources(
      &global, lidar_mode, BundleAdjuster::OptimazePhrase::Global);
  BOOST_CHECK_LT(global_comparison.active.problem.observations.size(), 24);

  SyntheticProblem whole = MakeSyntheticProblem();
  CompareProblemSources(&whole, lidar_mode,
                        BundleAdjuster::OptimazePhrase::WholeMap);

  for (int group = 0; group < 3; ++group) {
    SyntheticProblem camera_problem = MakeSyntheticProblem();
    camera_problem.config.SetVariableCamera(0);
    BundleAdjustmentOptions camera_options = base;
    camera_options.refine_focal_length = group == 0;
    camera_options.refine_principal_point = group == 1;
    camera_options.refine_extra_params = group == 2;
    const auto comparison = CompareProblemSources(
        &camera_problem, camera_options,
        BundleAdjuster::OptimazePhrase::Local);
    const auto camera_parameter = std::find_if(
        comparison.active.parameter_blocks.begin(),
        comparison.active.parameter_blocks.end(),
        [](const gpu_ba::ActiveBaParameterBlockSpec& value) {
          return value.kind == gpu_ba::ParameterKind::kCamera;
        });
    BOOST_REQUIRE(camera_parameter != comparison.active.parameter_blocks.end());
    BOOST_CHECK_GT(camera_parameter->tangent_size, 0);
    BOOST_CHECK_LT(camera_parameter->tangent_size,
                   camera_parameter->ambient_size);
  }

  SyntheticProblem lidar = MakeSyntheticProblem();
  Eigen::Vector3d lidar_xyz(0.1, 0.2, 0.3);
  Eigen::Vector4d lidar_plane(0.0, 1.0, 0.0, -0.2);
  LidarPoint lidar_point(LidarPointType::Icp, lidar_xyz, lidar_plane);
  lidar.config.AddLidarPoint(1, lidar_point);
  lidar.config.SetLidarSearchRange(1, 0.25);
  const auto lidar_comparison = CompareProblemSources(
      &lidar, lidar_mode, BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE_EQUAL(lidar_comparison.active.problem.lidar.size(), 1);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      lidar_comparison.active.problem.lidar[0].plane.begin(),
      lidar_comparison.active.problem.lidar[0].plane.end(),
      lidar_comparison.legacy.lidar[0].plane.begin(),
      lidar_comparison.legacy.lidar[0].plane.end());
  BOOST_CHECK_EQUAL(lidar_comparison.active.problem.lidar[0].weight,
                    lidar_comparison.legacy.lidar[0].weight);
  BOOST_CHECK_EQUAL(lidar_comparison.active.problem.lidar[0].search_range,
                    0.25);

  SyntheticProblem zero_observation = MakeSyntheticProblem();
  AddZeroObservationSelectedImage(&zero_observation);
  const auto zero_comparison = CompareProblemSources(
      &zero_observation, base, BundleAdjuster::OptimazePhrase::Local);
  const auto zero_camera = std::find_if(
      zero_comparison.active.problem.cameras.begin(),
      zero_comparison.active.problem.cameras.end(),
      [](const gpu_ba::CameraSnapshot& value) { return value.camera_id == 1; });
  BOOST_REQUIRE(zero_camera != zero_comparison.active.problem.cameras.end());
  BOOST_CHECK(zero_camera->constant);
  BOOST_CHECK(std::none_of(
      zero_comparison.active.parameter_blocks.begin(),
      zero_comparison.active.parameter_blocks.end(),
      [](const gpu_ba::ActiveBaParameterBlockSpec& value) {
        return value.kind == gpu_ba::ParameterKind::kCamera &&
               value.entity_id == 1;
      }));
}

BOOST_AUTO_TEST_CASE(ActiveSpecTransactionsRestoreEntryState) {
  FailureModeReset reset;
  const std::vector<std::pair<BundleAdjuster::FailureModeForTesting, bool>>
      cases = {
          {BundleAdjuster::FailureModeForTesting::kActiveSpecBuildFailure,
           false},
          {BundleAdjuster::FailureModeForTesting::kCustomCudaFailure, false},
          {BundleAdjuster::FailureModeForTesting::kCudaTopologyMismatch,
           false},
          {BundleAdjuster::FailureModeForTesting::
               kCustomCudaThenCeresFailure,
           true},
      };
  for (const auto& test : cases) {
    SyntheticProblem problem = MakeSyntheticProblem();
    const Reconstruction entry = problem.reconstruction;
    BundleAdjustmentOptions options = CudaOptions(test.second);
    options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
    BundleAdjuster::SetFailureModeForTesting(test.first);
    BundleAdjuster adjuster(options, problem.config);
    adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    BOOST_CHECK(!adjuster.Solve(&problem.reconstruction));
    CheckStateEqual(entry, problem.reconstruction);
  }
  BundleAdjuster::SetFailureModeForTesting(
      BundleAdjuster::FailureModeForTesting::kNone);
}

BOOST_AUTO_TEST_CASE(ActiveSpecPreparedIdentityRejectsChangedLidarAndCamera) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 11101;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.if_add_lidar_constraint = true;

  const auto make_config = [&](const Eigen::Vector4d& plane,
                               const double range) {
    BundleAdjustmentConfig config = problem.config;
    Eigen::Vector3d xyz(0.1, 0.2, 0.3);
    Eigen::Vector4d mutable_plane = plane;
    LidarPoint lidar_point(LidarPointType::Icp, xyz, mutable_plane);
    config.AddLidarPoint(1, lidar_point);
    config.SetLidarSearchRange(1, range);
    return config;
  };
  {
    BundleAdjustmentConfig config =
        make_config(Eigen::Vector4d(0.0, 1.0, 0.0, -0.2), 0.25);
    BundleAdjuster cold(options, config);
    cold.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    cold.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(cold.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(cold.ExecutionResult().host_store_misses, 1);
  }
  {
    BundleAdjustmentConfig config =
        make_config(Eigen::Vector4d(0.1, 0.9, 0.0, -0.3), 0.5);
    BundleAdjustmentOptions changed = options;
    changed.icp_lidar_constraint_weight = 125.0;
    BundleAdjuster lidar_changed(changed, config);
    lidar_changed.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    lidar_changed.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(lidar_changed.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(lidar_changed.ExecutionResult().host_store_hits, 0);
    BOOST_CHECK_EQUAL(lidar_changed.ExecutionResult().host_store_misses, 1);
    BOOST_CHECK_EQUAL(lidar_changed.ExecutionResult().host_store_view_rebuilds,
                      1);
  }
  problem.reconstruction.Camera(0).Params()[0] += 1.0;
  {
    BundleAdjustmentConfig config =
        make_config(Eigen::Vector4d(0.1, 0.9, 0.0, -0.3), 0.5);
    BundleAdjustmentOptions changed = options;
    changed.icp_lidar_constraint_weight = 125.0;
    BundleAdjuster camera_changed(changed, config);
    camera_changed.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    camera_changed.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE(camera_changed.Solve(&problem.reconstruction));
    BOOST_CHECK_EQUAL(camera_changed.ExecutionResult().host_store_hits, 0);
    BOOST_CHECK_EQUAL(camera_changed.ExecutionResult().host_store_misses, 1);
    BOOST_CHECK_EQUAL(camera_changed.ExecutionResult().host_store_view_rebuilds,
                      1);
  }
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(ActiveSpecZeroObservationSelectedCameraRunsFastPath) {
  SyntheticProblem problem = MakeSyntheticProblem();
  AddZeroObservationSelectedImage(&problem);
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kActiveSpec;
  BundleAdjuster adjuster(options, problem.config);
  adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  BOOST_REQUIRE_MESSAGE(adjuster.Solve(&problem.reconstruction),
                        adjuster.ExecutionResult().diagnostic_message);
  BOOST_CHECK(!adjuster.ExecutionResult().ceres_problem_created);
  BOOST_CHECK_EQUAL(adjuster.ExecutionResult().snapshot_materialization_calls,
                    0);
}

BOOST_AUTO_TEST_CASE(NativeGraphActiveSpecAdapterRunsWithoutLegacySnapshot) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 12001;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 32);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp64";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  BundleAdjuster adjuster(options, problem.config);
  adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  adjuster.SetCudaHostStoreBinding(binding);
  BOOST_REQUIRE_MESSAGE(adjuster.Solve(&problem.reconstruction),
                        adjuster.ExecutionResult().diagnostic_message);
  const BundleAdjustmentExecutionResult& execution =
      adjuster.ExecutionResult();
  BOOST_CHECK_EQUAL(execution.problem_source_requested, "native_graph");
  BOOST_CHECK_EQUAL(execution.problem_source_effective, "native_graph");
  BOOST_CHECK(!execution.ceres_problem_created);
  BOOST_CHECK_EQUAL(execution.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(execution.ceres_add_residual_calls, 0);
  BOOST_CHECK_EQUAL(execution.snapshot_materialization_calls, 0);
  BOOST_CHECK_EQUAL(execution.active_spec_build_calls, 1);
  BOOST_CHECK_EQUAL(execution.native_legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK_EQUAL(execution.native_repeated_residual_state_packing_bytes, 0);
  BOOST_CHECK_GT(execution.native_device_store_lookup_calls, 0);
  BOOST_CHECK_GT(execution.native_variable_state_d2h_calls, 0);
  BOOST_CHECK(!execution.fallback_used);
  BOOST_CHECK(std::isfinite(execution.initial_cost));
  BOOST_CHECK(std::isfinite(execution.final_cost));
  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(DirectNativeIntentMatchesSelectionAndRunsWithoutLegacy) {
  FailureModeReset reset;
  const auto configure = [](SyntheticProblem* problem) {
    AddThirdTrackObservationImage(problem);
    BundleAdjustmentConfig config;
    config.AddImage(0);
    config.AddImage(0);
    config.SetConstantCamera(0);
    config.SetConstantPose(0);
    config.AddVariablePoint(2);
    config.AddConstantPoint(1);
    Eigen::Vector3d lidar_xyz(0.1, 0.2, 0.3);
    Eigen::Vector4d lidar_plane(0.0, 1.0, 0.0, -0.2);
    LidarPoint lidar(LidarPointType::Icp, lidar_xyz, lidar_plane);
    config.AddLidarPoint(2, lidar);
    config.SetLidarSearchRange(2, 0.25);
    problem->config = std::move(config);
  };
  SyntheticProblem reference = MakeSyntheticProblem();
  SyntheticProblem direct = MakeSyntheticProblem();
  configure(&reference);
  configure(&direct);
  for (const image_t image_id : {image_t{0}, image_t{1}, image_t{2}}) {
    reference.reconstruction.Image(image_id).NormalizeQvec();
    direct.reconstruction.Image(image_id).NormalizeQvec();
  }
  BOOST_REQUIRE_EQUAL(direct.config.OrderedImages().size(), 1);
  BOOST_CHECK_EQUAL(direct.config.OrderedImages().front(), 0);

  BundleAdjustmentOptions options = CudaOptions(false);
  options.if_add_lidar_constraint = true;
  options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  options.loss_function_scale = 0.0;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp64";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";

  constexpr uint64_t kReferenceOwner = 12002;
  constexpr uint64_t kDirectOwner = 12003;
  reference.reconstruction.BeginStructureJournal(kReferenceOwner, 64);
  direct.reconstruction.BeginStructureJournal(kDirectOwner, 64);
  const auto comparison = CompareProblemSources(
      &reference, options, BundleAdjuster::OptimazePhrase::Local);

  gpu_ba::NativeBaSolveIntent intent;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildNativeBaSolveIntent(
          options, direct.config, direct.reconstruction, kDirectOwner,
          direct.reconstruction.StructureRevision(), 7,
          gpu_ba::BaKind::kLocal, &intent, &error),
      error);
  BOOST_REQUIRE_EQUAL(intent.explicit_variable_point_ids.size(), 1);
  BOOST_REQUIRE_EQUAL(intent.explicit_constant_point_ids.size(), 1);
  BOOST_CHECK_EQUAL(intent.explicit_variable_point_ids.front(), 2);
  BOOST_CHECK_EQUAL(intent.explicit_constant_point_ids.front(), 1);
  BOOST_CHECK(intent.config.loss_mode == gpu_ba::CudaLossMode::kTrivial);
  BOOST_CHECK_EQUAL(intent.config.loss_scale, 1.0);

  const ceres::Solver::Options effective =
      CreateEffectiveBundleAdjustmentSolverOptions(
          options, direct.config.NumImages(),
          static_cast<int>(direct.config.NumResiduals(direct.reconstruction)));
  BOOST_CHECK_EQUAL(intent.config.max_num_iterations,
                    effective.max_num_iterations);
  BOOST_CHECK_EQUAL(intent.config.max_consecutive_invalid_steps,
                    effective.max_num_consecutive_invalid_steps);
  BOOST_CHECK_EQUAL(intent.config.function_tolerance,
                    effective.function_tolerance);
  BOOST_CHECK_EQUAL(intent.config.gradient_tolerance,
                    effective.gradient_tolerance);
  BOOST_CHECK_EQUAL(intent.config.parameter_tolerance,
                    effective.parameter_tolerance);
  BOOST_CHECK_EQUAL(intent.config.max_solver_time_in_seconds,
                    effective.max_solver_time_in_seconds);
  BOOST_CHECK_EQUAL(intent.config.initial_trust_region_radius,
                    effective.initial_trust_region_radius);
  BOOST_CHECK_EQUAL(intent.config.min_trust_region_radius,
                    effective.min_trust_region_radius);
  BOOST_CHECK_EQUAL(intent.config.max_trust_region_radius,
                    effective.max_trust_region_radius);
  BOOST_CHECK_EQUAL(intent.config.min_relative_decrease,
                    effective.min_relative_decrease);
  BOOST_CHECK_EQUAL(intent.config.min_lm_diagonal,
                    effective.min_lm_diagonal);
  BOOST_CHECK_EQUAL(intent.config.max_lm_diagonal,
                    effective.max_lm_diagonal);

  BundleAdjustmentOptions invalid_soft_l1 = options;
  invalid_soft_l1.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  invalid_soft_l1.loss_function_scale = 0.0;
  gpu_ba::CudaFullLmOptions invalid_resolved_options;
  gpu_ba::NativeCudaResolvedConfig invalid_resolved_config;
  BOOST_CHECK(!ResolveNativeBundleAdjustmentCudaConfiguration(
      invalid_soft_l1, effective, 8, &invalid_resolved_options,
      &invalid_resolved_config, &error));

  gpu_ba::CudaFullLmOptions resolved_options;
  gpu_ba::NativeCudaResolvedConfig resolved_config;
  BOOST_REQUIRE_MESSAGE(ResolveNativeBundleAdjustmentCudaConfiguration(
                            options, effective, 7, &resolved_options,
                            &resolved_config, &error),
                        error);
  gpu_ba::GpuBaHostProblemStore direct_store(&direct.reconstruction,
                                              kDirectOwner);
  gpu_ba::CudaHostStoreBinding direct_binding;
  direct_binding.store = &direct_store;
  direct_binding.owner_epoch = kDirectOwner;
  direct_binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  gpu_ba::PreparedNativeActiveSolve prepared;
  BOOST_REQUIRE_MESSAGE(gpu_ba::PrepareCudaNativeBaSolve(
                            intent, &direct.reconstruction, resolved_options,
                            direct_binding, &prepared, &error),
                        error);
  const gpu_ba::NativeHostSolveView* view = prepared.view();
  BOOST_REQUIRE(view != nullptr);

  using PhysicalResidual = std::pair<uint8_t, uint64_t>;
  std::vector<PhysicalResidual> direct_physical;
  for (const gpu_ba::ResidualOrdinal& ordinal : view->residual_ordinals) {
    direct_physical.emplace_back(static_cast<uint8_t>(ordinal.kind),
                                 ordinal.physical_identity);
  }
  std::vector<PhysicalResidual> reference_physical;
  for (const gpu_ba::ObservationSnapshot& observation :
       comparison.active.problem.observations) {
    reference_physical.emplace_back(
        static_cast<uint8_t>(gpu_ba::ResidualKind::kVisual),
        (static_cast<uint64_t>(observation.image_id) << 32) |
            observation.point2D_idx);
  }
  for (const gpu_ba::LidarSnapshot& lidar : comparison.active.problem.lidar) {
    reference_physical.emplace_back(
        static_cast<uint8_t>(gpu_ba::ResidualKind::kLidar),
        (static_cast<uint64_t>(lidar.point3D_id) << 2) ^ lidar.lidar_type);
  }
  std::sort(direct_physical.begin(), direct_physical.end());
  std::sort(reference_physical.begin(), reference_physical.end());
  BOOST_CHECK(direct_physical == reference_physical);
  BOOST_CHECK_EQUAL(view->scalar_residual_count,
                    comparison.active.scalar_residual_count);
  BOOST_CHECK_EQUAL(view->ambient_parameter_count,
                    comparison.active.ambient_parameter_count);
  BOOST_CHECK_EQUAL(view->effective_parameter_count,
                    comparison.active.effective_parameter_count);

  std::vector<const gpu_ba::ResidualOrdinal*> source_order(
      view->residual_ordinals.size(), nullptr);
  for (const gpu_ba::ResidualOrdinal& ordinal : view->residual_ordinals) {
    BOOST_REQUIRE_LT(ordinal.source_insertion_index, source_order.size());
    source_order[ordinal.source_insertion_index] = &ordinal;
  }
  BOOST_REQUIRE_EQUAL(source_order.size(), 17);
  const auto catalog_observations = view->catalog.observations();
  const auto catalog_images = view->catalog.images();
  const auto catalog_points = view->catalog.points();
  const auto visual_ids = [&](const gpu_ba::ResidualOrdinal* ordinal) {
    const gpu_ba::HostBaObservationSlot& observation =
        catalog_observations[ordinal->source_slot];
    return std::make_pair(catalog_images[observation.image_slot].image_id,
                          catalog_points[observation.point_slot].point3D_id);
  };
  for (size_t index = 0; index < 12; ++index) {
    BOOST_REQUIRE(source_order[index] != nullptr);
    BOOST_REQUIRE(source_order[index]->kind ==
                  gpu_ba::ResidualKind::kVisual);
    BOOST_CHECK_EQUAL(visual_ids(source_order[index]).first, 0);
  }
  for (size_t index = 12; index < 14; ++index) {
    BOOST_REQUIRE(source_order[index] != nullptr);
    BOOST_REQUIRE(source_order[index]->kind ==
                  gpu_ba::ResidualKind::kVisual);
    const auto ids = visual_ids(source_order[index]);
    BOOST_CHECK_EQUAL(ids.second, 2);
    BOOST_CHECK(ids.first == 1 || ids.first == 2);
  }
  BOOST_REQUIRE(source_order[14] != nullptr);
  BOOST_CHECK(source_order[14]->kind == gpu_ba::ResidualKind::kLidar);
  for (size_t index = 15; index < 17; ++index) {
    BOOST_REQUIRE(source_order[index] != nullptr);
    BOOST_REQUIRE(source_order[index]->kind ==
                  gpu_ba::ResidualKind::kVisual);
    const auto ids = visual_ids(source_order[index]);
    BOOST_CHECK_EQUAL(ids.second, 1);
    BOOST_CHECK(ids.first == 1 || ids.first == 2);
  }
  const gpu_ba::HostBaPointSlot* constant_slot =
      view->catalog.FindPointById(1);
  BOOST_REQUIRE(constant_slot != nullptr);
  const auto constant_policy = std::find_if(
      view->fixed.points.begin(), view->fixed.points.end(),
      [&](const gpu_ba::PointFixedPolicyResult& value) {
        return value.point_slot == constant_slot->header.slot;
      });
  BOOST_REQUIRE(constant_policy != view->fixed.points.end());
  BOOST_CHECK_EQUAL(constant_policy->constant, 1);
  const gpu_ba::HostBaPointSlot* variable_slot =
      view->catalog.FindPointById(2);
  BOOST_REQUIRE(variable_slot != nullptr);
  const auto variable_policy = std::find_if(
      view->fixed.points.begin(), view->fixed.points.end(),
      [&](const gpu_ba::PointFixedPolicyResult& value) {
        return value.point_slot == variable_slot->header.slot;
      });
  BOOST_REQUIRE(variable_policy != view->fixed.points.end());
  BOOST_CHECK_EQUAL(variable_policy->constant, 0);
  BOOST_REQUIRE_EQUAL(view->boundary_image_slots.size(), 2);
  BOOST_CHECK(std::all_of(
      view->fixed.images.begin(), view->fixed.images.end(),
      [](const gpu_ba::ImageFixedPolicyResult& value) {
        return value.pose_constant != 0;
      }));
  BOOST_REQUIRE_EQUAL(view->fixed.cameras.size(), 1);
  BOOST_CHECK_EQUAL(view->fixed.cameras.front().constant, 1);

  const std::vector<gpu_ba::ResidualOrdinal> first_ordinals =
      view->residual_ordinals;
  prepared.Release();
  gpu_ba::PreparedNativeActiveSolve repeated;
  BOOST_REQUIRE_MESSAGE(gpu_ba::PrepareCudaNativeBaSolve(
                            intent, &direct.reconstruction, resolved_options,
                            direct_binding, &repeated, &error),
                        error);
  BOOST_REQUIRE_EQUAL(repeated.view()->residual_ordinals.size(),
                      first_ordinals.size());
  for (size_t index = 0; index < first_ordinals.size(); ++index) {
    const auto& lhs = first_ordinals[index];
    const auto& rhs = repeated.view()->residual_ordinals[index];
    BOOST_CHECK(lhs.kind == rhs.kind);
    BOOST_CHECK_EQUAL(lhs.source_insertion_index,
                      rhs.source_insertion_index);
    BOOST_CHECK_EQUAL(lhs.physical_identity, rhs.physical_identity);
  }
  repeated.Release();

  gpu_ba::GpuBaHostProblemStore reference_store(&reference.reconstruction,
                                                 kReferenceOwner);
  gpu_ba::CudaHostStoreBinding reference_binding;
  reference_binding.store = &reference_store;
  reference_binding.owner_epoch = kReferenceOwner;
  reference_binding.mode =
      gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  BundleAdjuster reference_adjuster(options, reference.config);
  reference_adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  reference_adjuster.SetCudaHostStoreBinding(reference_binding);
  BOOST_REQUIRE_MESSAGE(reference_adjuster.Solve(&reference.reconstruction),
                        reference_adjuster.ExecutionResult().diagnostic_message);
  BundleAdjuster direct_adjuster(options, direct.config);
  direct_adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  direct_adjuster.SetCudaHostStoreBinding(direct_binding);
  BOOST_REQUIRE_MESSAGE(direct_adjuster.SolveNative(&direct.reconstruction,
                                                    intent),
                        direct_adjuster.ExecutionResult().diagnostic_message);
  const auto& reference_execution = reference_adjuster.ExecutionResult();
  const auto& direct_execution = direct_adjuster.ExecutionResult();
  BOOST_CHECK_EQUAL(direct_execution.active_spec_build_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.ceres_problem_created, false);
  BOOST_CHECK_EQUAL(direct_execution.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(direct_execution.ceres_add_residual_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.snapshot_materialization_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.bundle_adjuster_setup_milliseconds, 0.0);
  BOOST_CHECK_EQUAL(direct_execution.native_legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.native_plan_prepare_requests, 0);
  BOOST_CHECK_EQUAL(direct_execution.native_device_selection_hits, 0);
  BOOST_CHECK_EQUAL(direct_execution.native_device_selection_misses, 0);
  BOOST_CHECK_GT(direct_execution.native_variable_state_d2h_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.fallback_used, false);
  BOOST_CHECK_EQUAL(direct_execution.trial_steps,
                    reference_execution.trial_steps);
  BOOST_CHECK_EQUAL(direct_execution.accepted_commits,
                    reference_execution.accepted_commits);
  BOOST_CHECK_EQUAL(direct_execution.rejected_steps,
                    reference_execution.rejected_steps);
  BOOST_CHECK_EQUAL(direct_execution.invalid_steps,
                    reference_execution.invalid_steps);
  BOOST_CHECK_SMALL(direct_execution.initial_cost -
                        reference_execution.initial_cost,
                    1e-12);
  BOOST_CHECK_SMALL(direct_execution.final_cost - reference_execution.final_cost,
                    1e-12);
  CheckStateEqual(reference.reconstruction, direct.reconstruction);

  std::string shutdown_error;
  BOOST_REQUIRE(reference_store.Shutdown(&shutdown_error));
  BOOST_REQUIRE(direct_store.Shutdown(&shutdown_error));
  reference.reconstruction.EndStructureJournal();
  direct.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(DirectNativeAcceptedCommitWritesReconstructionState) {
  FailureModeReset reset;
  SyntheticProblem reference = MakeSyntheticProblem();
  SyntheticProblem direct = MakeSyntheticProblem();
  reference.reconstruction.Image(1).Tvec(1) += 0.15;
  direct.reconstruction.Image(1).Tvec(1) += 0.15;
  constexpr uint64_t kReferenceOwner = 12004;
  constexpr uint64_t kDirectOwner = 12005;
  reference.reconstruction.BeginStructureJournal(kReferenceOwner, 64);
  direct.reconstruction.BeginStructureJournal(kDirectOwner, 64);

  BundleAdjustmentOptions options = CudaOptions(false);
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  options.solver_options.max_num_iterations = 5;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp64";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";

  gpu_ba::GpuBaHostProblemStore reference_store(&reference.reconstruction,
                                                 kReferenceOwner);
  gpu_ba::GpuBaHostProblemStore direct_store(&direct.reconstruction,
                                              kDirectOwner);
  gpu_ba::CudaHostStoreBinding reference_binding;
  reference_binding.store = &reference_store;
  reference_binding.owner_epoch = kReferenceOwner;
  reference_binding.mode =
      gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
  gpu_ba::CudaHostStoreBinding direct_binding;
  direct_binding.store = &direct_store;
  direct_binding.owner_epoch = kDirectOwner;
  direct_binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;

  gpu_ba::NativeBaSolveIntent intent;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      BuildNativeBaSolveIntent(
          options, direct.config, direct.reconstruction, kDirectOwner,
          direct.reconstruction.StructureRevision(), 9,
          gpu_ba::BaKind::kLocal, &intent, &error),
      error);
  const Eigen::Vector4d fixed_qvec = direct.reconstruction.Image(0).Qvec();
  const double fixed_translation_x = direct.reconstruction.Image(1).Tvec(0);
  const std::vector<double> fixed_camera =
      direct.reconstruction.Camera(0).Params();

  BundleAdjuster reference_adjuster(options, reference.config);
  reference_adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  reference_adjuster.SetCudaHostStoreBinding(reference_binding);
  BOOST_REQUIRE_MESSAGE(reference_adjuster.Solve(&reference.reconstruction),
                        reference_adjuster.ExecutionResult().diagnostic_message);
  BundleAdjuster direct_adjuster(options, direct.config);
  direct_adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
  direct_adjuster.SetCudaHostStoreBinding(direct_binding);
  BOOST_REQUIRE_MESSAGE(direct_adjuster.SolveNative(&direct.reconstruction,
                                                    intent),
                        direct_adjuster.ExecutionResult().diagnostic_message);

  const auto& reference_execution = reference_adjuster.ExecutionResult();
  const auto& direct_execution = direct_adjuster.ExecutionResult();
  BOOST_TEST_MESSAGE(
      "direct accepted trace: trials=" << direct_execution.trial_steps
      << " commits=" << direct_execution.accepted_commits
      << " rejects=" << direct_execution.rejected_steps
      << " invalid=" << direct_execution.invalid_steps
      << " initial_cost=" << direct_execution.initial_cost
      << " final_cost=" << direct_execution.final_cost);
  BOOST_REQUIRE_GE(reference_execution.accepted_commits, 1);
  BOOST_REQUIRE_GE(direct_execution.accepted_commits, 1);
  BOOST_CHECK_EQUAL(direct_execution.accepted_commits,
                    reference_execution.accepted_commits);
  BOOST_CHECK_EQUAL(direct_execution.trial_steps,
                    reference_execution.trial_steps);
  BOOST_CHECK_EQUAL(direct_execution.rejected_steps,
                    reference_execution.rejected_steps);
  BOOST_CHECK_EQUAL(direct_execution.invalid_steps,
                    reference_execution.invalid_steps);
  BOOST_CHECK_SMALL(direct_execution.initial_cost -
                        reference_execution.initial_cost,
                    1e-12);
  BOOST_CHECK_SMALL(direct_execution.final_cost - reference_execution.final_cost,
                    1e-10);
  BOOST_CHECK_EQUAL(direct_execution.active_spec_build_calls, 0);
  BOOST_CHECK(!direct_execution.ceres_problem_created);
  BOOST_CHECK_EQUAL(direct_execution.ceres_cost_function_creations, 0);
  BOOST_CHECK_EQUAL(direct_execution.ceres_add_residual_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.snapshot_materialization_calls, 0);
  BOOST_CHECK_EQUAL(direct_execution.native_legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK(!direct_execution.fallback_used);
  BOOST_CHECK_GT(direct_execution.native_variable_state_d2h_calls, 0);
  BOOST_CHECK_EQUAL(fixed_qvec, direct.reconstruction.Image(0).Qvec());
  BOOST_CHECK_EQUAL(fixed_translation_x,
                    direct.reconstruction.Image(1).Tvec(0));
  BOOST_CHECK_EQUAL_COLLECTIONS(
      fixed_camera.begin(), fixed_camera.end(),
      direct.reconstruction.Camera(0).Params().begin(),
      direct.reconstruction.Camera(0).Params().end());
  for (const auto& image : reference.reconstruction.Images()) {
    BOOST_REQUIRE(direct.reconstruction.ExistsImage(image.first));
    BOOST_CHECK_SMALL((image.second.Qvec() -
                       direct.reconstruction.Image(image.first).Qvec()).norm(),
                      1e-10);
    BOOST_CHECK_SMALL((image.second.Tvec() -
                       direct.reconstruction.Image(image.first).Tvec()).norm(),
                      1e-9);
  }
  for (const auto& point : reference.reconstruction.Points3D()) {
    BOOST_REQUIRE(direct.reconstruction.ExistsPoint3D(point.first));
    BOOST_CHECK_SMALL(
        (point.second.XYZ() -
         direct.reconstruction.Point3D(point.first).XYZ()).norm(),
        1e-8);
  }

  std::string shutdown_error;
  BOOST_REQUIRE(reference_store.Shutdown(&shutdown_error));
  BOOST_REQUIRE(direct_store.Shutdown(&shutdown_error));
  reference.reconstruction.EndStructureJournal();
  direct.reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(DirectNativePreparedSelectionHostAndDeviceHit) {
  FailureModeReset reset;
  SyntheticProblem problem = MakeSyntheticProblem();
  constexpr uint64_t kOwnerEpoch = 12006;
  problem.reconstruction.BeginStructureJournal(kOwnerEpoch, 64);
  gpu_ba::GpuBaHostProblemStore store(&problem.reconstruction, kOwnerEpoch);
  gpu_ba::CudaHostStoreBinding binding;
  binding.store = &store;
  binding.owner_epoch = kOwnerEpoch;
  binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;

  BundleAdjustmentOptions options = CudaOptions(false);
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp32_mixed";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  options.ba_cuda_prepared_selection_cache =
      gpu_ba::CudaPreparedSelectionCacheMode::kEnabled;

  const auto run = [&](const uint64_t selection_revision) {
    gpu_ba::NativeBaSolveIntent intent;
    std::string error;
    BOOST_REQUIRE_MESSAGE(
        BuildNativeBaSolveIntent(
            options, problem.config, problem.reconstruction, kOwnerEpoch,
            problem.reconstruction.StructureRevision(), selection_revision,
            gpu_ba::BaKind::kLocal, &intent, &error),
        error);
    BundleAdjuster adjuster(options, problem.config);
    adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
    adjuster.SetCudaHostStoreBinding(binding);
    BOOST_REQUIRE_MESSAGE(adjuster.SolveNative(&problem.reconstruction, intent),
                          adjuster.ExecutionResult().diagnostic_message);
    return adjuster.ExecutionResult();
  };

  const BundleAdjustmentExecutionResult cold = run(1);
  BOOST_CHECK_EQUAL(cold.native_plan_prepare_requests, 1);
  BOOST_CHECK_EQUAL(cold.native_host_plan_hits, 0);
  BOOST_CHECK_EQUAL(cold.native_host_plan_misses, 1);
  BOOST_CHECK_EQUAL(cold.native_host_plan_build_calls, 1);
  BOOST_CHECK_EQUAL(cold.native_static_materialize_calls, 1);
  BOOST_CHECK_EQUAL(cold.native_dynamic_state_gather_calls, 1);
  BOOST_CHECK_GT(cold.native_intent_incidence_traversal_visits, 0);
  BOOST_CHECK_EQUAL(cold.native_materializer_incidence_traversal_visits, 0);
  BOOST_CHECK_EQUAL(cold.native_indexed_plan_build_calls, 1);
  BOOST_CHECK_EQUAL(cold.native_device_selection_hits, 0);
  BOOST_CHECK_EQUAL(cold.native_device_selection_misses, 1);
  BOOST_CHECK_EQUAL(cold.native_device_selection_lookup_calls, 1);
  BOOST_CHECK_EQUAL(cold.native_device_selection_upload_calls, 1);
  BOOST_CHECK_EQUAL(cold.native_device_selection_cached_workspace_bytes, 0);
  BOOST_CHECK_GT(cold.native_device_selection_static_h2d_calls, 0);
  BOOST_CHECK_GT(cold.native_device_selection_static_h2d_bytes, 0);

  std::string pool_error;
  BOOST_REQUIRE_MESSAGE(gpu_ba::ResetCudaRuntimePoolForTesting(&pool_error),
                        pool_error);
  problem.reconstruction.Image(1).Tvec(2) += 1e-6;
  const double dynamic_translation = problem.reconstruction.Image(1).Tvec(2);
  const BundleAdjustmentExecutionResult hot = run(2);
  BOOST_CHECK_EQUAL(hot.native_plan_prepare_requests, 1);
  BOOST_CHECK_EQUAL(hot.native_host_plan_hits, 1);
  BOOST_CHECK_EQUAL(hot.native_host_plan_misses, 0);
  BOOST_CHECK_EQUAL(hot.native_host_plan_bypasses, 0);
  BOOST_CHECK_EQUAL(hot.native_host_plan_build_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_static_materialize_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_dynamic_state_gather_calls, 1);
  BOOST_CHECK_EQUAL(hot.native_plan_vector_copy_bytes_on_hit, 0);
  BOOST_CHECK_EQUAL(hot.native_indexed_plan_build_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_hits, 1);
  BOOST_CHECK_EQUAL(hot.native_device_selection_misses, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_bypasses, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_lookup_calls, 1);
  BOOST_CHECK_EQUAL(hot.native_device_selection_upload_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_evictions, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_context_invalidations, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_poison_events, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_static_h2d_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_static_h2d_bytes, 0);
  BOOST_CHECK_GT(hot.native_device_selection_static_h2d_saved_calls, 0);
  BOOST_CHECK_GT(hot.native_device_selection_static_h2d_saved_bytes, 0);
  BOOST_CHECK_EQUAL(hot.native_device_selection_cached_workspace_bytes, 0);
  BOOST_CHECK_EQUAL(hot.native_legacy_kernel_input_bundle_calls, 0);
  BOOST_CHECK_EQUAL(hot.native_repeated_residual_state_packing_bytes, 0);
  BOOST_CHECK(!hot.fallback_used);
  BOOST_CHECK_EQUAL(problem.reconstruction.Image(1).Tvec(2),
                    dynamic_translation);

  problem.reconstruction.AddPoint3D(
      Eigen::Vector3d(100.0, 100.0, 100.0), Track());
  const BundleAdjustmentExecutionResult unrelated_append = run(3);
  BOOST_CHECK_EQUAL(unrelated_append.native_host_plan_hits, 1);
  BOOST_CHECK_EQUAL(unrelated_append.native_host_plan_misses, 0);
  BOOST_CHECK_EQUAL(unrelated_append.native_host_plan_dependency_misses, 0);
  BOOST_CHECK_EQUAL(unrelated_append.native_static_materialize_calls, 0);
  BOOST_CHECK_EQUAL(unrelated_append.native_device_selection_hits, 1);
  BOOST_CHECK_EQUAL(
      unrelated_append.native_device_selection_static_h2d_calls, 0);
  BOOST_CHECK_EQUAL(
      unrelated_append.native_device_selection_static_h2d_bytes, 0);

  AddThirdTrackObservationImage(&problem);
  const BundleAdjustmentExecutionResult selected_adjacency_change = run(4);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_host_plan_hits, 0);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_host_plan_misses, 1);
  BOOST_CHECK_EQUAL(
      selected_adjacency_change.native_host_plan_dependency_misses, 1);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_host_plan_build_calls, 1);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_static_materialize_calls,
                    1);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_device_selection_hits, 0);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_device_selection_misses,
                    1);
  BOOST_CHECK_EQUAL(selected_adjacency_change.native_device_selection_upload_calls,
                    1);
  BOOST_CHECK_GT(
      selected_adjacency_change.native_device_selection_static_h2d_calls, 0);

  std::string shutdown_error;
  BOOST_REQUIRE(store.Shutdown(&shutdown_error));
  problem.reconstruction.EndStructureJournal();
}
#endif

}  // namespace
}  // namespace colmap
