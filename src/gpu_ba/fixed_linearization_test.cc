#define TEST_NAME "gpu_ba/fixed_linearization"
#include "util/testing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <tuple>

#include <unistd.h>

#include "base/camera_models.h"
#include "gpu_ba/fixed_linearization.h"
#include "util/misc.h"

namespace colmap {
namespace gpu_ba {
namespace {

Snapshot SyntheticFixedProblem() {
  Snapshot snapshot;
  snapshot.metadata.loss_function = "trivial";
  snapshot.metadata.lidar_residual_mode = "legacy_exact";

  CameraSnapshot camera;
  camera.camera_id = 7;
  camera.model_id = OpenCVCameraModel::kModelId;
  camera.constant = true;
  camera.params = {500.0, 505.0, 320.0, 240.0,
                   -0.02, 0.003, 0.001, -0.0004};
  snapshot.cameras.push_back(camera);

  ImageSnapshot variable_image;
  variable_image.image_id = 11;
  variable_image.camera_id = camera.camera_id;
  variable_image.selected = true;
  variable_image.pose_constant = false;
  variable_image.has_pose_parameter_blocks = true;
  variable_image.constant_tvec_mask = 1u << 0;
  variable_image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  variable_image.tvec = {{0.1, -0.2, 0.3}};
  snapshot.images.push_back(variable_image);

  ImageSnapshot constant_image;
  constant_image.image_id = 12;
  constant_image.camera_id = camera.camera_id;
  constant_image.selected = true;
  constant_image.pose_constant = true;
  constant_image.has_pose_parameter_blocks = false;
  constant_image.qvec = {{0.9999500004166653, 0.0, 0.009999833334166664,
                          0.0}};
  constant_image.tvec = {{0.4, 0.0, 0.2}};
  snapshot.images.push_back(constant_image);

  PointSnapshot variable_point;
  variable_point.point3D_id = 101;
  variable_point.constant = false;
  variable_point.xyz = {{0.2, -0.1, 3.0}};
  snapshot.points.push_back(variable_point);
  PointSnapshot constant_point;
  constant_point.point3D_id = 102;
  constant_point.constant = true;
  constant_point.xyz = {{-0.3, 0.4, 4.0}};
  snapshot.points.push_back(constant_point);

  ObservationSnapshot observation0;
  observation0.source_index = 0;
  observation0.image_id = variable_image.image_id;
  observation0.point2D_idx = 4;
  observation0.point3D_id = variable_point.point3D_id;
  observation0.xy = {{370.0, 180.0}};
  snapshot.observations.push_back(observation0);
  ObservationSnapshot observation1 = observation0;
  observation1.source_index = 1;
  observation1.point2D_idx = 5;
  observation1.point3D_id = constant_point.point3D_id;
  observation1.xy = {{290.0, 270.0}};
  snapshot.observations.push_back(observation1);
  ObservationSnapshot observation2 = observation0;
  observation2.source_index = 2;
  observation2.image_id = constant_image.image_id;
  observation2.point2D_idx = 2;
  observation2.xy = {{410.0, 220.0}};
  observation2.pose_constant = true;
  snapshot.observations.push_back(observation2);

  LidarSnapshot lidar;
  lidar.source_index = 3;
  lidar.point3D_id = variable_point.point3D_id;
  lidar.weight = 10.0;
  lidar.plane = {{1.0, 0.0, 0.0, -0.25}};
  snapshot.lidar.push_back(lidar);

  for (const ObservationSnapshot& observation : snapshot.observations) {
    OrderEntrySnapshot entry;
    entry.source_index = observation.source_index;
    entry.residual_kind = ResidualKind::kVisual;
    entry.image_id = observation.image_id;
    entry.point2D_idx = observation.point2D_idx;
    entry.point3D_id = observation.point3D_id;
    snapshot.source_insertion_order.push_back(entry);
    snapshot.canonical_order.push_back(entry);
  }
  OrderEntrySnapshot lidar_entry;
  lidar_entry.source_index = lidar.source_index;
  lidar_entry.residual_kind = ResidualKind::kLidar;
  lidar_entry.image_id = std::numeric_limits<uint32_t>::max();
  lidar_entry.point2D_idx = std::numeric_limits<uint32_t>::max();
  lidar_entry.point3D_id = lidar.point3D_id;
  snapshot.source_insertion_order.push_back(lidar_entry);
  snapshot.canonical_order.push_back(lidar_entry);
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                              lhs.point3D_id, lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                              rhs.point3D_id, rhs.source_index);
            });

  ParameterBlockSnapshot quaternion;
  quaternion.source_index = 0;
  quaternion.kind = ParameterKind::kQuaternion;
  quaternion.entity_id = variable_image.image_id;
  quaternion.ambient_size = 4;
  quaternion.tangent_size = 3;
  quaternion.constant = false;
  snapshot.parameter_blocks_source_order.push_back(quaternion);
  ParameterBlockSnapshot translation;
  translation.source_index = 1;
  translation.kind = ParameterKind::kTranslation;
  translation.entity_id = variable_image.image_id;
  translation.ambient_size = 3;
  translation.tangent_size = 2;
  translation.constant = false;
  snapshot.parameter_blocks_source_order.push_back(translation);
  ParameterBlockSnapshot point;
  point.source_index = 2;
  point.kind = ParameterKind::kPoint3D;
  point.entity_id = variable_point.point3D_id;
  point.ambient_size = 3;
  point.tangent_size = 3;
  point.constant = false;
  snapshot.parameter_blocks_source_order.push_back(point);
  return snapshot;
}

}  // namespace

BOOST_AUTO_TEST_CASE(CanonicalSubsetAndConstantBlocks) {
  const Snapshot snapshot = SyntheticFixedProblem();
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_REQUIRE(RunFixedLinearizationComparison(snapshot, options, &result,
                                                &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.variable_pose_blocks, 1);
  BOOST_CHECK_EQUAL(result.pose_tangent_dimension, 5);
  BOOST_CHECK_EQUAL(result.variable_point_blocks, 1);
  BOOST_CHECK_EQUAL(result.point_tangent_dimension, 3);
  BOOST_CHECK_EQUAL(result.pose_point_edges, 1);
  BOOST_CHECK_EQUAL(result.constant_pose_residuals, 1);
  BOOST_CHECK_EQUAL(result.constant_point_residuals, 1);
  BOOST_CHECK_EQUAL(result.fully_constant_residuals, 0);
  BOOST_CHECK(result.reference_backward_error <= 1e-10);
  BOOST_CHECK(result.candidate_backward_error <= 1e-10);
  BOOST_CHECK_EQUAL(result.reference_accept, result.candidate_accept);
  for (const ErrorSummary& metric : result.metrics) {
    BOOST_CHECK_MESSAGE(metric.pass, metric.name);
  }
  const std::string json = FixedLinearizationJson(result, options);
  BOOST_CHECK(json.find("shared_frozen_reference_residual\": true") !=
              std::string::npos);
  BOOST_CHECK(
      json.find("shared_frozen_reference_gradient_for_schur\": true") !=
      std::string::npos);
}

BOOST_AUTO_TEST_CASE(Ceres14FixedSingleStepUsesFullLmMath) {
  const Snapshot snapshot = SyntheticFixedProblem();
  FixedLinearizationOptions options;
  options.step_semantics = FixedStepSemantics::kCeres14;
  FixedLinearizationResult result;
  std::string error;
  BOOST_REQUIRE(RunFixedLinearizationComparison(snapshot, options, &result,
                                                &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.step_semantics, "ceres14");
  BOOST_CHECK_EQUAL(result.reference_accept, result.candidate_accept);
  const std::string json = FixedLinearizationJson(result, options);
  BOOST_CHECK(json.find("\"step_semantics\": \"ceres14\"") !=
              std::string::npos);
}

BOOST_AUTO_TEST_CASE(VariableIntrinsicsAreExplicitlyUnsupported) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.cameras.front().constant = false;
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_CHECK(!RunFixedLinearizationComparison(snapshot, options, &result,
                                               &error));
  BOOST_CHECK(error.find("variable intrinsics") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(NonLegacyLidarIsNotStrictParity) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.lidar_residual_mode = "legacy_guarded";
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_CHECK(!RunFixedLinearizationComparison(snapshot, options, &result,
                                               &error));
  BOOST_CHECK(error.find("legacy_exact") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(ProjectedGradientExcludesFixedAndSubsetComponents) {
  const Snapshot snapshot = SyntheticFixedProblem();
  CpuGradientNorms norms;
  std::string error;
  BOOST_REQUIRE(EvaluateCustomCpuGradientNorms(snapshot, &norms, &error));
  BOOST_CHECK_MESSAGE(std::isfinite(norms.projected_max_norm), error);
  BOOST_CHECK_EQUAL(norms.variable_rotation_blocks, 1);
  BOOST_CHECK_EQUAL(norms.variable_translation_components, 2);
  BOOST_CHECK_EQUAL(norms.variable_point_blocks, 1);
  BOOST_CHECK(!norms.worst_ambient_id.empty());
}

BOOST_AUTO_TEST_CASE(ProjectedGradientEuclideanPointMatchesRawGradient) {
  Snapshot snapshot = SyntheticFixedProblem();
  for (ImageSnapshot& image : snapshot.images) {
    image.pose_constant = true;
    image.has_pose_parameter_blocks = false;
  }
  for (ObservationSnapshot& observation : snapshot.observations) {
    observation.pose_constant = true;
  }
  CpuGradientNorms norms;
  std::string error;
  BOOST_REQUIRE(EvaluateCustomCpuGradientNorms(snapshot, &norms, &error));
  BOOST_CHECK_EQUAL(norms.variable_rotation_blocks, 0);
  BOOST_CHECK_EQUAL(norms.variable_translation_components, 0);
  BOOST_CHECK_EQUAL(norms.variable_point_blocks, 1);
  BOOST_CHECK_CLOSE_FRACTION(norms.projected_max_norm,
                             norms.raw_tangent_max_norm, 1e-15);
}

BOOST_AUTO_TEST_CASE(ProjectedGradientQuaternionUsesAmbientPlusDifference) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.images.front().constant_tvec_mask = 0x7;
  for (PointSnapshot& point : snapshot.points) point.constant = true;
  for (ParameterBlockSnapshot& parameter :
       snapshot.parameter_blocks_source_order) {
    if (parameter.kind == ParameterKind::kTranslation) {
      parameter.tangent_size = 0;
    } else if (parameter.kind == ParameterKind::kPoint3D) {
      parameter.constant = true;
      parameter.tangent_size = 0;
    }
  }
  CpuGradientNorms norms;
  std::string error;
  BOOST_REQUIRE(EvaluateCustomCpuGradientNorms(snapshot, &norms, &error));
  BOOST_CHECK_EQUAL(norms.variable_rotation_blocks, 1);
  BOOST_CHECK_EQUAL(norms.variable_translation_components, 0);
  BOOST_CHECK_EQUAL(norms.variable_point_blocks, 0);
  BOOST_CHECK(norms.projected_max_norm <= 2.0);
  BOOST_CHECK(norms.raw_tangent_max_norm > norms.projected_max_norm);
}

BOOST_AUTO_TEST_CASE(CustomLmChecksProjectedGradientAtIterationZero) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.max_num_iterations = 5;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 1e100;
  snapshot.metadata.parameter_tolerance = 0.0;
  CustomCpuSolveResult result;
  Snapshot final_state;
  std::string error;
  BOOST_REQUIRE(
      RunCustomCpuSolve(snapshot, 1e-4, CustomCpuSolveMode::kFull,
                        &result, &final_state, &error));
  BOOST_CHECK_MESSAGE(result.success, error);
  BOOST_CHECK(result.termination_type == CpuTerminationType::kConvergence);
  BOOST_CHECK_EQUAL(result.termination_reason,
                    "gradient_tolerance_at_iteration_0");
  BOOST_CHECK_EQUAL(result.trial_iterations, 0);
  BOOST_CHECK_EQUAL(result.accepted_steps, 1);
  BOOST_CHECK_EQUAL(result.rejected_steps, 0);
  BOOST_REQUIRE_EQUAL(result.trace.size(), 1);
  BOOST_CHECK_EQUAL(result.trace.front().iteration, 0);
  BOOST_CHECK_CLOSE_FRACTION(result.initial_cost, result.final_cost, 1e-15);
  BOOST_CHECK_EQUAL(result.gradient_tolerance, 1e100);
  BOOST_CHECK_EQUAL(result.max_trial_iterations, 5);
}

BOOST_AUTO_TEST_CASE(CustomLmCountsAcceptedAndRejectedTrialsAgainstLimit) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CustomCpuSolveResult result;
  Snapshot final_state;
  std::string error;
  BOOST_REQUIRE(
      RunCustomCpuSolve(snapshot, 1e-4, CustomCpuSolveMode::kFull,
                        &result, &final_state, &error));
  BOOST_CHECK_MESSAGE(result.success, error);
  BOOST_CHECK(result.termination_type == CpuTerminationType::kNoConvergence);
  BOOST_CHECK_EQUAL(result.termination_reason, "maximum_trial_iterations");
  BOOST_CHECK_EQUAL(result.trial_iterations, 1);
  BOOST_CHECK_EQUAL(result.accepted_steps + result.rejected_steps, 2);
  BOOST_REQUIRE_EQUAL(result.trace.size(), 2);
  BOOST_CHECK_EQUAL(result.trace.front().iteration, 0);
  BOOST_CHECK_EQUAL(result.trace.back().iteration, 1);
}

BOOST_AUTO_TEST_CASE(CustomFullLmSyntheticReplayIsDeterministic) {
  Snapshot snapshot = SyntheticFixedProblem();
  // Keep the full-LM determinism fixture away from the intentionally
  // non-differentiable d=0 boundary of legacy_exact. That boundary has
  // dedicated residual tests; the real replay snapshots report zero hits.
  snapshot.lidar.clear();
  snapshot.source_insertion_order.erase(
      std::remove_if(snapshot.source_insertion_order.begin(),
                     snapshot.source_insertion_order.end(),
                     [](const OrderEntrySnapshot& entry) {
                       return entry.residual_kind == ResidualKind::kLidar;
                     }),
      snapshot.source_insertion_order.end());
  snapshot.canonical_order.erase(
      std::remove_if(snapshot.canonical_order.begin(),
                     snapshot.canonical_order.end(),
                     [](const OrderEntrySnapshot& entry) {
                       return entry.residual_kind == ResidualKind::kLidar;
                     }),
      snapshot.canonical_order.end());
  snapshot.metadata.max_num_iterations = 8;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CustomCpuSolveResult first_result;
  CustomCpuSolveResult second_result;
  Snapshot first_state;
  Snapshot second_state;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCpuSolve(snapshot, 1e-4,
                                          CustomCpuSolveMode::kFull,
                                          &first_result, &first_state, &error),
                        error);
  BOOST_REQUIRE_MESSAGE(RunCustomCpuSolve(snapshot, 1e-4,
                                          CustomCpuSolveMode::kFull,
                                          &second_result, &second_state,
                                          &error),
                        error);
  BOOST_CHECK(first_result.success);
  BOOST_CHECK(second_result.success);
  BOOST_CHECK_GT(first_result.trial_iterations, 0);
  BOOST_CHECK_LE(first_result.trial_iterations, 8);
  BOOST_CHECK_EQUAL(first_result.trial_iterations,
                    second_result.trial_iterations);
  BOOST_CHECK_EQUAL(first_result.termination_reason,
                    second_result.termination_reason);
  BOOST_CHECK_EQUAL(first_result.accepted_steps, second_result.accepted_steps);
  BOOST_CHECK_EQUAL(first_result.rejected_steps, second_result.rejected_steps);
  BOOST_CHECK_EQUAL(first_result.final_cost, second_result.final_cost);
  BOOST_CHECK_EQUAL(first_result.final_lambda, second_result.final_lambda);
  BOOST_REQUIRE_EQUAL(first_state.images.size(), second_state.images.size());
  BOOST_REQUIRE_EQUAL(first_state.points.size(), second_state.points.size());
  for (size_t i = 0; i < first_state.images.size(); ++i) {
    BOOST_CHECK(first_state.images[i].qvec == second_state.images[i].qvec);
    BOOST_CHECK(first_state.images[i].tvec == second_state.images[i].tvec);
  }
  for (size_t i = 0; i < first_state.points.size(); ++i) {
    BOOST_CHECK(first_state.points[i].xyz == second_state.points[i].xyz);
  }
}

BOOST_AUTO_TEST_CASE(SoftL1CustomLinearizationMatchesPublicCeresCost) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.loss_function = "soft_l1";
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  snapshot.metadata.max_consecutive_invalid_steps = 10;

  CustomCpuSolverOptions options;
  options.loss_type = "SOFT_L1";
  options.loss_scale = 1.0;
  options.max_num_iterations = 1;
  options.max_num_consecutive_invalid_steps = 10;
  CustomCpuSolveResult custom_result;
  Snapshot custom_state;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCpuSolve(snapshot, options,
                        CustomCpuSolveMode::kCeres14SingleStep,
                        &custom_result, &custom_state, &error),
      error);

  CeresCpuSolveResult ceres_result;
  Snapshot ceres_state;
  BOOST_REQUIRE_MESSAGE(
      RunCeresCpuSolve(snapshot, true, "", &ceres_result, &ceres_state,
                       &error),
      error);
  BOOST_CHECK_CLOSE_FRACTION(custom_result.initial_cost,
                             ceres_result.initial_cost, 1e-12);
  BOOST_CHECK_CLOSE_FRACTION(
      custom_result.initial_projected_gradient_max_norm,
      ceres_result.trace.front().projected_gradient_max_norm, 1e-9);
  BOOST_CHECK_EQUAL(custom_result.effective_options.loss_type, "SOFT_L1");
  BOOST_CHECK_EQUAL(
      custom_result.effective_options.max_num_consecutive_invalid_steps, 10);
}

BOOST_AUTO_TEST_CASE(CustomAndCeresSingleStepUseIndependentDeepCopies) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.max_num_iterations = 5;
  snapshot.metadata.function_tolerance = 1e-12;
  snapshot.metadata.gradient_tolerance = 1e-12;
  snapshot.metadata.parameter_tolerance = 1e-12;
  CustomCpuSolveResult custom_result;
  CeresCpuSolveResult ceres_result;
  Snapshot custom_state;
  Snapshot ceres_state;
  std::string error;
  BOOST_REQUIRE(RunCustomCpuSolve(snapshot, 1e-4,
                                  CustomCpuSolveMode::kLegacySingleStep,
                                  &custom_result,
                                  &custom_state, &error));
  BOOST_CHECK_MESSAGE(custom_result.success, error);
  BOOST_CHECK_EQUAL(custom_result.trial_iterations, 1);
  BOOST_CHECK_GT(custom_result.trace.back().lm_diagonal_min, 0.0);
  BOOST_CHECK_GE(custom_result.trace.back().lm_diagonal_max,
                 custom_result.trace.back().lm_diagonal_min);
  BOOST_REQUIRE(
      RunCeresCpuSolve(snapshot, true, "", &ceres_result, &ceres_state,
                       &error));
  BOOST_CHECK_MESSAGE(ceres_result.success, error);
  BOOST_CHECK_EQUAL(ceres_result.ba_image_count, 2);
  BOOST_CHECK_EQUAL(ceres_result.requested_linear_solver_type,
                    "DENSE_SCHUR");
  BOOST_CHECK_EQUAL(ceres_result.actual_linear_solver_type, "DENSE_SCHUR");
  BOOST_CHECK_EQUAL(ceres_result.requested_num_threads, 1);
  BOOST_CHECK_EQUAL(ceres_result.effective_num_threads, 1);
  BOOST_CHECK(ceres_result.jacobi_scaling);
  BOOST_CHECK(ceres_result.deterministic_oracle);
  BOOST_CHECK_EQUAL(ceres_result.residual_order,
                    "source_insertion_order");
  BOOST_REQUIRE_EQUAL(ceres_result.trace.size(), 2);
  BOOST_CHECK_EQUAL(ceres_result.trace.front().iteration, 0);
  BOOST_CHECK_EQUAL(ceres_result.trace.back().iteration, 1);
  BOOST_CHECK_CLOSE_FRACTION(custom_result.initial_cost,
                             ceres_result.initial_cost, 1e-10);
  BOOST_CHECK_CLOSE_FRACTION(
      custom_result.initial_projected_gradient_max_norm,
      ceres_result.trace.front().projected_gradient_max_norm, 1e-7);
  BOOST_CHECK_EQUAL(snapshot.images.front().qvec[0], 1.0);
  BOOST_CHECK_EQUAL(snapshot.points.front().xyz[0], 0.2);

  CpuSolveComparisonResult comparison;
  BOOST_REQUIRE(CompareCpuSolveResults(ceres_state, custom_state,
                                       ceres_result, custom_result,
                                       &comparison, &error));
  BOOST_CHECK_EQUAL(comparison.compared_poses, snapshot.images.size());
  BOOST_CHECK_EQUAL(comparison.compared_points, snapshot.points.size());
  const std::string json = CustomCpuSolveJson(custom_result);
  BOOST_CHECK(json.find("\"stopping_config\"") != std::string::npos);
  BOOST_CHECK(json.find("\"max_trial_iterations\": 1") !=
              std::string::npos);
  BOOST_CHECK(json.find("\"lidar_near_zero\"") != std::string::npos);
  BOOST_CHECK(json.find("\"rejected\"") != std::string::npos);
  const std::string comparison_json = CpuSolveComparisonJson(comparison);
  BOOST_CHECK(comparison_json.find("\"phase5_strict_gate\"") !=
              std::string::npos);
  BOOST_CHECK(comparison_json.find("\"quality_diagnostic_thresholds\"") !=
              std::string::npos);
  BOOST_CHECK(comparison_json.find("\"trace_structure\"") !=
              std::string::npos);
  BOOST_CHECK(comparison_json.find("\"trace_numeric_diagnostic\"") !=
              std::string::npos);
}

BOOST_AUTO_TEST_CASE(TraceStructureScansPastFirstNumericDivergence) {
  const Snapshot snapshot = SyntheticFixedProblem();
  CeresCpuSolveResult ceres_result;
  CustomCpuSolveResult custom_result;
  ceres_result.success = true;
  custom_result.success = true;
  custom_result.solve_mode = CustomCpuSolveMode::kFull;
  ceres_result.termination_type = CpuTerminationType::kConvergence;
  custom_result.termination_type = CpuTerminationType::kConvergence;
  ceres_result.termination_reason = "Gradient tolerance reached.";
  custom_result.termination_reason = "gradient_tolerance";
  ceres_result.final_cost = 10.0;
  custom_result.final_cost = 10.0;
  ceres_result.final_gradient_max_norm = 0.0;
  custom_result.final_projected_gradient_max_norm = 0.0;
  for (int iteration = 0; iteration < 3; ++iteration) {
    CustomCpuIteration reference;
    reference.iteration = iteration;
    reference.cost_before = 10.0 - iteration;
    reference.trial_cost = reference.cost_before;
    reference.cost_after = reference.cost_before;
    reference.step_valid = true;
    reference.trial_finite = true;
    reference.accepted = true;
    CustomCpuIteration candidate = reference;
    if (iteration == 1) candidate.cost_before += 1.0;
    if (iteration == 2) candidate.accepted = false;
    ceres_result.trace.push_back(reference);
    custom_result.trace.push_back(candidate);
  }
  ceres_result.accepted_steps = 3;
  custom_result.accepted_steps = 2;
  custom_result.rejected_steps = 1;
  CpuSolveComparisonResult comparison;
  std::string error;
  BOOST_REQUIRE(CompareCpuSolveResults(snapshot, snapshot, ceres_result,
                                       custom_result, &comparison, &error));
  BOOST_CHECK(!comparison.trace_numeric_within_tolerance);
  BOOST_CHECK_EQUAL(comparison.first_trace_divergence_iteration, 1);
  BOOST_CHECK_EQUAL(comparison.first_trace_divergence_field, "cost_before");
  BOOST_CHECK(!comparison.trace_structure_pass);
  BOOST_CHECK_EQUAL(comparison.trace_structure_iterations_checked, 3);
  BOOST_CHECK_EQUAL(comparison.first_trace_structure_mismatch_iteration, 2);
  BOOST_CHECK_EQUAL(comparison.first_trace_structure_mismatch_field,
                    "accepted");
  BOOST_CHECK(!comparison.pass);
}

BOOST_AUTO_TEST_CASE(Ceres14SingleStepUsesIndependentCeresDampingDump) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.max_num_iterations = 5;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CustomCpuSolveResult custom_result;
  CeresCpuSolveResult ceres_result;
  Snapshot custom_state;
  Snapshot ceres_state;
  std::string error;
  BOOST_REQUIRE(RunCustomCpuSolve(snapshot, 1e-4,
                                  CustomCpuSolveMode::kCeres14SingleStep,
                                  &custom_result, &custom_state, &error));
  const std::string dump_directory =
      "/tmp/colmap_gpu_ba_ceres14_damping_" + std::to_string(getpid());
  CreateDirIfNotExists(dump_directory, true);
  BOOST_REQUIRE(RunCeresCpuSolve(snapshot, true, dump_directory, &ceres_result,
                                &ceres_state, &error));
  BOOST_CHECK_MESSAGE(ceres_result.damping_oracle_observed, error);
  BOOST_CHECK(!ceres_result.damping_oracle_lm_diagonal.empty());
  BOOST_CHECK(ExistsFile(ceres_result.damping_oracle_path));
  BOOST_REQUIRE_EQUAL(ceres_result.trace.size(), 2);
  BOOST_CHECK(!ceres_result.trace.front().lm_diagonal_observed);
  BOOST_CHECK(ceres_result.trace.back().lm_diagonal_observed);
  const auto observed_range =
      std::minmax_element(ceres_result.damping_oracle_lm_diagonal.begin(),
                          ceres_result.damping_oracle_lm_diagonal.end());
  BOOST_CHECK_EQUAL(ceres_result.trace.back().lm_diagonal_min,
                    *observed_range.first);
  BOOST_CHECK_EQUAL(ceres_result.trace.back().lm_diagonal_max,
                    *observed_range.second);
  CpuSolveComparisonResult comparison;
  BOOST_REQUIRE(CompareCpuSolveResults(ceres_state, custom_state, ceres_result,
                                       custom_result, &comparison, &error));
  BOOST_CHECK(comparison.trace_structure_pass);
  BOOST_CHECK(comparison.damping_oracle_required);
  BOOST_CHECK(comparison.damping_oracle_available);
  BOOST_CHECK_MESSAGE(comparison.damping_oracle_pass,
                      comparison.damping_oracle_worst_id);
  BOOST_CHECK(comparison.pass);
  const std::string ceres_json = CeresCpuSolveJson(ceres_result);
  BOOST_CHECK(ceres_json.find(
                  "\"trace_lm_diagonal_source\": "
                  "\"actual_ceres_dump_iteration_1_or_unobserved\"") !=
              std::string::npos);
  BOOST_CHECK(ceres_json.find("reconstructed_from_callback_state") ==
              std::string::npos);
  std::remove(ceres_result.damping_oracle_path.c_str());
  rmdir(dump_directory.c_str());
}

}  // namespace gpu_ba
}  // namespace colmap
