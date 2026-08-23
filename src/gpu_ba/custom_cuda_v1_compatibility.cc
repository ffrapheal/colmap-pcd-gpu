#include "gpu_ba/custom_cuda.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace colmap {
namespace gpu_ba {
namespace {

CudaFullLmOptions NormalOptions() {
  CudaFullLmOptions options;
  options.layer_c.layer_b.layer_a.residual_order =
      CudaResidualOrder::kSourceInsertion;
  options.layer_c.layer_b.cost_reduction_threads = 8;
  options.performance_mode = false;
  options.capture_state_trace = false;
  options.max_solver_time_in_seconds = 1e9;
  options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kDisabled;
  return options;
}

CudaFullLmResult SyntheticPreparationFailure(const Snapshot& snapshot) {
  CudaFullLmResult result;
  result.success = false;
  result.error = "pending accepted-state allocation failed: synthetic";
  result.error_classification =
      CudaSolveErrorClass::kAcceptedPendingPreparation;
  result.termination_type = CudaTerminationType::kFailure;
  result.termination_reason = "accepted_state_pending_preparation_failed";
  result.initial_cost = 12.5;
  result.final_cost = 12.5;
  result.initial_projected_gradient_max_norm = 4.0;
  result.final_projected_gradient_max_norm = 4.0;
  result.initial_scaled_gradient_norm = 2.0;
  result.final_scaled_gradient_norm = 2.0;
  result.final_radius = 10000.0;
  result.final_lambda = 1e-4;
  result.trial_iterations = 1;
  result.accepted_decisions = 1;
  result.accepted_pending_preparation_failures = 1;
  result.runtime.actual_trials = 1;
  result.runtime.final_internal_state_epoch = 0;
  CudaLmIteration iteration;
  iteration.iteration = 1;
  iteration.cost_before = 12.5;
  iteration.trial_cost = 12.0;
  iteration.cost_after = 12.5;
  iteration.projected_gradient_max_norm = 4.0;
  iteration.scaled_gradient_norm = 2.0;
  iteration.radius_before = 10000.0;
  iteration.radius_after = 10000.0;
  iteration.lambda_before = 1e-4;
  iteration.lambda_after = 1e-4;
  iteration.lm_diagonal_min = 0.01;
  iteration.lm_diagonal_max = 3.0;
  iteration.predicted_reduction = 0.5;
  iteration.actual_reduction = 0.5;
  iteration.rho = 1.0;
  iteration.step_norm = 0.25;
  iteration.backward_error = 1e-12;
  iteration.factorization_success = true;
  iteration.step_valid = true;
  iteration.trial_finite = true;
  iteration.accepted_decision = true;
  iteration.accepted_commit_success = false;
  iteration.accepted = false;
  iteration.termination_reason = result.termination_reason;
  result.trace.push_back(iteration);
  result.final_state = snapshot;
  return result;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: " << argv[0]
              << " SNAPSHOT normal|pending_linearization|synthetic_preparation "
                 "EXPECTED_SEMANTIC EXPECTED_DECISION OUTPUT_JSON\n";
    return EXIT_FAILURE;
  }
  using namespace colmap::gpu_ba;
  Snapshot snapshot;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read, &error)) {
    std::cerr << error << '\n';
    return EXIT_FAILURE;
  }
  const std::string mode = argv[2];
  CudaFullLmResult result;
  bool lifecycle_pass = false;
  if (mode == "normal") {
    lifecycle_pass = RunCustomCudaSolve(snapshot, NormalOptions(), &result,
                                        &error);
  } else if (mode == "pending_linearization") {
    if (!ResetCudaResourceStateForTesting(&error)) {
      std::cerr << error << '\n';
      return EXIT_FAILURE;
    }
    CudaFullLmOptions options = NormalOptions();
    options.max_num_iterations = 1;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 0.0;
    options.parameter_tolerance = 0.0;
    options.performance_mode = true;
    options.current_linearization_cache_mode =
        CudaCurrentLinearizationCacheMode::kEnabled;
    options.fault_trigger.fault_kind =
        CudaFaultInjection::kForceLinearizationFailure;
    options.fault_trigger.logical_site =
        CudaFaultLogicalSite::kAcceptedPendingLinearization;
    options.fault_trigger.target_state_epoch = 1;
    options.fault_trigger.occurrence_within_site = 1;
    const bool ran = RunCustomCudaSolve(snapshot, options, &result, &error);
    lifecycle_pass = !ran && result.accepted_decisions == 1 &&
        result.accepted_commits == 0 &&
        result.accepted_pending_linearization_failures == 1 &&
        result.runtime.final_internal_state_epoch == 0;
  } else if (mode == "synthetic_preparation") {
    result = SyntheticPreparationFailure(snapshot);
    lifecycle_pass = result.accepted_decisions == 1 &&
        result.accepted_pending_preparation_failures == 1 &&
        CudaLegacyV1ErrorClassification(result) ==
            CudaSolveErrorClass::kAcceptedPendingLinearization;
  } else {
    std::cerr << "unknown mode: " << mode << '\n';
    return EXIT_FAILURE;
  }
  const std::string semantic = CudaFullLmSemanticBitwiseSha256V1(result);
  const std::string decision = CudaFullLmDecisionBitwiseSha256(result);
  const bool semantic_pass = semantic == argv[3];
  const bool decision_pass = decision == argv[4];
  const bool available = CudaLegacyV1ComparisonAvailable(result);
  const bool pass = lifecycle_pass && semantic_pass && decision_pass &&
                    available;
  std::ofstream output(argv[5], std::ios::trunc);
  output << "{\"schema\":\"phase7p1a-r1-v1-compatibility\","
         << "\"mode\":\"" << mode << "\","
         << "\"semantic_v1\":\"" << semantic << "\","
         << "\"expected_semantic_v1\":\"" << argv[3] << "\","
         << "\"decision_v1\":\"" << decision << "\","
         << "\"expected_decision_v1\":\"" << argv[4] << "\","
         << "\"lifecycle_pass\":" << (lifecycle_pass ? "true" : "false")
         << ",\"semantic_pass\":" << (semantic_pass ? "true" : "false")
         << ",\"decision_pass\":" << (decision_pass ? "true" : "false")
         << ",\"legacy_v1_comparison_available\":"
         << (available ? "true" : "false")
         << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
  if (!output) {
    std::cerr << "failed to write " << argv[5] << '\n';
    return EXIT_FAILURE;
  }
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
