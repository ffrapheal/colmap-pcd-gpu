#define TEST_NAME "gpu_ba/ceres_envelope"
#include "util/testing.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <unistd.h>

#include "gpu_ba/ceres_envelope.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string TempRoot() {
  return "/tmp/colmap_gpu_ba_ceres_envelope_test_" +
         std::to_string(getpid());
}

struct Fixture {
  Fixture() { boost::filesystem::remove_all(TempRoot()); }
  ~Fixture() { boost::filesystem::remove_all(TempRoot()); }
};

FidelityProvenanceSnapshot Provenance() {
  FidelityProvenanceSnapshot value;
  value.executable_path = "/test/colmap";
  value.executable_sha256 = std::string(64, 'a');
  value.libceres_path = "/test/libceres.so";
  value.libceres_sha256 = std::string(64, 'b');
  value.git_head = "test-head";
  value.dirty_diff_sha256 = std::string(64, 'c');
  return value;
}

Snapshot MakeState(double variation = 0.0) {
  Snapshot state;
  state.metadata.snapshot_id = "local-reg50-call118-refine0-trigger79-phraselocal";
  CameraSnapshot camera;
  camera.camera_id = 1;
  camera.model_id = 4;
  camera.constant = true;
  camera.params = {500.0};
  state.cameras.push_back(camera);
  ImageSnapshot image;
  image.image_id = 30;
  image.camera_id = 1;
  image.selected = true;
  image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  image.tvec = {{variation, 0.0, 0.0}};
  state.images.push_back(image);
  PointSnapshot point;
  point.point3D_id = 7;
  point.xyz = {{0.0, 0.0, 3.0 + variation}};
  state.points.push_back(point);
  state.parameter_blocks_source_order.push_back(
      {0, ParameterKind::kPoint3D, 7, 3, 3, false});
  state.parameter_blocks_canonical_order.push_back(0);
  return state;
}

EffectiveCeresOptionsSnapshot Options() {
  EffectiveCeresOptionsSnapshot value;
  value.minimizer_type = ceres::TRUST_REGION;
  value.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  value.linear_solver_type = ceres::DENSE_SCHUR;
  value.preconditioner_type = ceres::JACOBI;
  value.num_threads = 8;
  value.num_linear_solver_threads = 8;
  value.max_num_iterations = 25;
  value.max_linear_solver_iterations = 100;
  value.max_num_consecutive_invalid_steps = 10;
  value.initial_trust_region_radius = 1e4;
  value.min_trust_region_radius = 1e-32;
  value.max_trust_region_radius = 1e16;
  value.min_lm_diagonal = 1e-6;
  value.max_lm_diagonal = 1e32;
  value.min_relative_decrease = 1e-3;
  value.jacobi_scaling = true;
  return value;
}

LoadedCeresFidelitySample MakeSample(const std::string& run_id,
                                     double variation) {
  LoadedCeresFidelitySample sample;
  sample.record.run_id = run_id;
  sample.record.provenance = Provenance();
  sample.record.snapshot_id =
      "local-reg50-call118-refine0-trigger79-phraselocal";
  sample.record.snapshot_payload_sha256 = std::string(64, 'd');
  sample.solve_pre_state = MakeState();
  sample.post_state = MakeState(variation * 1e-12);
  sample.record.final_state_sha256 = CanonicalStateSha256(sample.post_state);
  sample.record.effective_options = Options();
  FidelityProblemSnapshot& problem = sample.record.problem;
  problem.complete_fingerprint_sha256 = std::string(64, 'e');
  problem.source_residual_order_sha256 = std::string(64, 'f');
  problem.source_parameter_order_sha256 = std::string(64, '1');
  problem.effective_options_sha256 =
      EffectiveCeresOptionsSha256(sample.record.effective_options);
  problem.loss_specifications = {
      {"visual", "TRIVIAL", 1.0, 0}, {"lidar", "TRIVIAL", 1.0, 0}};
  problem.parameter_constraints = {
      {ParameterKind::kPoint3D, 7, false, "euclidean", {}}};
  CeresSummarySnapshot& summary = sample.record.summary;
  summary.minimizer_type = ceres::TRUST_REGION;
  summary.termination_type = ceres::CONVERGENCE;
  summary.termination_message = "Gradient tolerance reached.";
  summary.successful_steps = 3;
  summary.unsuccessful_steps = 0;
  summary.invalid_steps = 0;
  summary.num_linear_solves = 2;
  summary.num_threads_given = 8;
  summary.num_threads_used = 8;
  summary.num_linear_solver_threads_given = 8;
  summary.num_linear_solver_threads_used = 8;
  summary.linear_solver_type_given = ceres::DENSE_SCHUR;
  summary.linear_solver_type_used = ceres::DENSE_SCHUR;
  summary.preconditioner_type_given = ceres::JACOBI;
  summary.preconditioner_type_used = ceres::JACOBI;
  summary.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  for (int iteration = 0; iteration < 3; ++iteration) {
    CeresIterationSnapshot item;
    item.iteration = iteration;
    item.step_is_valid = true;
    item.step_is_successful = true;
    item.cost = 100.0 - iteration + variation * 1e-10;
    item.cost_change = iteration == 0 ? 0.0 : 1.0 + variation * 1e-11;
    item.gradient_max_norm = 10.0 - iteration + variation * 1e-8;
    item.gradient_norm = 20.0 - iteration + variation * 1e-8;
    item.step_norm = iteration == 0 ? 0.0 : 0.1 + variation * 1e-12;
    item.relative_decrease = iteration == 0 ? 0.0 : 0.9 + variation * 1e-9;
    item.trust_region_radius = 1e4 * (iteration + 1);
    item.eta = iteration == 0 ? 0.1 : 0.0;
    item.linear_solver_iterations = iteration == 0 ? 0 : 1;
    summary.iterations.push_back(item);
  }
  summary.initial_cost = summary.iterations.front().cost;
  summary.final_cost = summary.iterations.back().cost;
  return sample;
}

CeresEnvelopePolicy Policy() {
  CeresEnvelopePolicy policy;
  policy.schema_version = 1;
  policy.policy_id = "unit-test-policy";
  policy.required_reference_samples = 8;
  policy.empirical_diameter_multiplier = 1.05;
  policy.joint_distance = "single_reference_joint_linf";
  policy.zero_diameter_rule = "exact";
  policy.expected_reference_provenance = Provenance();
  policy.trace_absolute_caps = {
      {"cost", 1e-6}, {"cost_change", 1e-6},
      {"gradient_max_norm", 1e-3}, {"gradient_norm", 1e-3},
      {"step_norm", 1e-8}, {"relative_decrease", 1e-3},
      {"trust_region_radius", 0.0}, {"eta", 0.0}, {"step_size", 0.0}};
  policy.state_absolute_caps = {
      {"quaternion_ambient", 1e-12}, {"rotation_degrees", 1e-4},
      {"translation_m", 1e-6}, {"point_m", 1e-5}, {"camera", 1e-12}};
  return policy;
}

std::vector<LoadedCeresFidelitySample> Samples() {
  // Duplicate both extrema so every leave-one-out training set retains the
  // same support. The production data exercise the non-trivial 1.05 guard.
  const std::vector<double> values = {0, 0, 1, 2, 3, 4, 5, 5};
  std::vector<LoadedCeresFidelitySample> samples;
  for (size_t i = 0; i < values.size(); ++i) {
    samples.push_back(MakeSample(std::string("original-") +
                                     static_cast<char>('a' + i),
                                 values[i]));
  }
  return samples;
}

std::string WritePolicy() {
  boost::filesystem::create_directories(TempRoot());
  const std::string path = TempRoot() + "/policy.json";
  std::ofstream output(path);
  output << R"JSON({
  "schema_version": 1,
  "policy_id": "unit-test-policy",
  "required_reference_samples": 8,
  "empirical_diameter_multiplier": 1.05,
  "joint_distance": "single_reference_joint_linf",
  "zero_diameter_rule": "exact",
  "expected_reference_provenance": {
    "executable_path": "/test/colmap",
    "executable_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "libceres_path": "/test/libceres.so",
    "libceres_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "git_head": "test-head",
    "dirty_diff_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "trace_absolute_caps": {
    "cost": 1e-6,
    "cost_change": 1e-6,
    "gradient_max_norm": 1e-3,
    "gradient_norm": 1e-3,
    "step_norm": 1e-8,
    "relative_decrease": 1e-3,
    "trust_region_radius": 0,
    "eta": 0,
    "step_size": 0
  },
  "state_absolute_caps": {
    "quaternion_ambient": 1e-12,
    "rotation_degrees": 1e-4,
    "translation_m": 1e-6,
    "point_m": 1e-5,
    "camera": 1e-12
  }
})JSON";
  output.close();
  return path;
}

}  // namespace

BOOST_FIXTURE_TEST_CASE(PolicyIsMachineReadableAndHasStableHash, Fixture) {
  CeresEnvelopePolicy policy;
  std::string error;
  BOOST_REQUIRE(ReadCeresEnvelopePolicy(WritePolicy(), &policy, &error));
  BOOST_CHECK_EQUAL(policy.schema_version, 1);
  BOOST_CHECK_EQUAL(policy.required_reference_samples, 8);
  BOOST_CHECK_EQUAL(policy.joint_distance, "single_reference_joint_linf");
  BOOST_CHECK_EQUAL(policy.file_sha256.size(), 64);
}

BOOST_AUTO_TEST_CASE(LeaveOneOutUsesOneJointWitnessAndAllFoldsPass) {
  const auto samples = Samples();
  CeresEnvelopeLeaveOneOutResult result;
  std::string error;
  BOOST_REQUIRE(EvaluateCeresEnvelopeLeaveOneOut(samples, Policy(), &result,
                                                 &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_REQUIRE_EQUAL(result.folds.size(), 8);
  for (const auto& fold : result.folds) {
    BOOST_CHECK(fold.pass);
    BOOST_CHECK(!fold.best_reference_run_id.empty());
    BOOST_CHECK_LE(fold.best_joint_ratio, 1.0);
  }
}

BOOST_AUTO_TEST_CASE(BidirectionalStateIdentityRejectsExtraMissingAndDuplicate) {
  const auto references = Samples();
  const CeresEnvelopePolicy policy = Policy();
  for (const std::string& kind : {"extra", "missing", "duplicate"}) {
    LoadedCeresFidelitySample candidate = references.front();
    candidate.record.run_id = "candidate-" + kind;
    if (kind == "extra") {
      ImageSnapshot image = candidate.post_state.images.front();
      image.image_id = 999;
      candidate.post_state.images.push_back(image);
    } else if (kind == "missing") {
      candidate.post_state.images.clear();
    } else {
      candidate.post_state.images.push_back(candidate.post_state.images.front());
    }
    CeresEnvelopeCandidateResult result;
    std::string error;
    BOOST_REQUIRE(EvaluateCeresEnvelopeCandidate(references, candidate, policy,
                                                 &result, &error));
    BOOST_CHECK(!result.pass);
    BOOST_CHECK(!result.id_sets_pass);
    BOOST_CHECK(!result.failure_reasons.empty());
  }
}

BOOST_AUTO_TEST_CASE(ControlledPerturbationsFailWithReasons) {
  const auto references = Samples();
  const CeresEnvelopePolicy policy = Policy();
  const std::vector<std::string> perturbations = {
      "option_max_linear_plus_one", "loss_scale_plus_0_1",
      "swap_first_two_residuals", "parameter_id_plus_one",
      "trace_cost_plus_1e_3", "trace_gradient_plus_1e_2",
      "trace_radius_plus_one", "final_pose_translation_plus_1e_3"};
  for (const std::string& perturbation : perturbations) {
    LoadedCeresFidelitySample candidate = references.front();
    candidate.record.run_id = "candidate-" + perturbation;
    if (perturbation == "swap_first_two_residuals") {
      candidate.solve_pre_state.source_insertion_order = {
          {0, ResidualKind::kVisual, 30, 0, 7},
          {1, ResidualKind::kLidar, 0, 0, 7}};
    }
    std::string error;
    if (perturbation == "swap_first_two_residuals") {
      std::swap(candidate.solve_pre_state.source_insertion_order[0],
                candidate.solve_pre_state.source_insertion_order[1]);
      candidate.record.problem.source_residual_order_sha256[0] = '0';
    } else {
      BOOST_REQUIRE(ApplyCeresEnvelopePerturbation(perturbation, &candidate,
                                                   &error));
    }
    CeresEnvelopeCandidateResult result;
    BOOST_REQUIRE(EvaluateCeresEnvelopeCandidate(references, candidate, policy,
                                                 &result, &error));
    BOOST_CHECK_MESSAGE(!result.pass, perturbation);
    BOOST_CHECK_MESSAGE(!result.failure_reasons.empty(), perturbation);
  }
}

BOOST_AUTO_TEST_CASE(DirectStateComparisonAlsoRejectsExtraIds) {
  LoadedCeresFidelitySample reference = MakeSample("reference", 0.0);
  LoadedCeresFidelitySample candidate = reference;
  candidate.record.run_id = "candidate";
  ImageSnapshot extra = candidate.post_state.images.front();
  extra.image_id = 99;
  candidate.post_state.images.push_back(extra);
  CeresFidelityComparison comparison;
  std::string error;
  BOOST_CHECK(!CompareCeresFidelityRecords(
      reference.record, reference.post_state, candidate.record,
      candidate.post_state, &comparison, &error));
  BOOST_CHECK(error.find("extra image ID=99") != std::string::npos);
}

}  // namespace gpu_ba
}  // namespace colmap
