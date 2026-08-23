#include "gpu_ba/custom_cuda.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace colmap {
namespace gpu_ba {
namespace {

bool EnsureDirectory(const std::string& path, std::string* error) {
  if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) return true;
  *error = "mkdir failed for " + path + ": " + std::strerror(errno);
  return false;
}

bool WriteBinary(const std::string& path,
                 const std::string& data,
                 std::string* error) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    *error = "cannot open " + path;
    return false;
  }
  output.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!output) {
    *error = "cannot write " + path;
    return false;
  }
  return true;
}

bool WriteText(const std::string& path,
               const std::string& data,
               std::string* error) {
  return WriteBinary(path, data, error);
}

std::string Hex(const std::string& data) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const unsigned char value : data)
    output << std::setw(2) << static_cast<unsigned int>(value);
  output << '\n';
  return output.str();
}

CudaStableDiagnosticV2 Diagnostic(const CudaSolveErrorClass error_class,
                                  const CudaFailureSite site,
                                  const CudaStatusDomain domain,
                                  const int64_t status) {
  CudaStableDiagnosticV2 value;
  value.error_classification = error_class;
  value.failure_site = site;
  value.status_domain = domain;
  value.raw_status_code = status;
  return value;
}

CudaFullLmResult Fixture() {
  CudaFullLmResult result;
  result.success = false;
  result.error = "dynamic text excluded from canonical V2";
  result.error_classification = CudaSolveErrorClass::kResourceCleanup;
  result.termination_type = CudaTerminationType::kFailure;
  result.termination_reason = "resource_cleanup_failed";
  result.initial_cost = 10.5;
  result.final_cost = 7.25;
  result.initial_projected_gradient_max_norm = 4.0;
  result.final_projected_gradient_max_norm = 0.5;
  result.initial_scaled_gradient_norm = 3.0;
  result.final_scaled_gradient_norm = 0.25;
  result.final_radius = 8192.0;
  result.final_lambda = 1.0 / result.final_radius;
  result.trial_iterations = 3;
  result.accepted_steps = 2;
  result.rejected_steps = 1;
  result.invalid_steps = 1;
  result.factorization_failures = 1;
  result.accepted_decisions = 3;
  result.accepted_commits = 1;
  result.accepted_pending_preparation_failures = 1;
  result.accepted_pending_linearization_failures = 1;

  CudaLmIteration iteration;
  iteration.iteration = 1;
  iteration.cost_before = 10.5;
  iteration.trial_cost = 7.25;
  iteration.cost_after = 7.25;
  iteration.projected_gradient_max_norm = 0.5;
  iteration.scaled_gradient_norm = 0.25;
  iteration.radius_before = 4096.0;
  iteration.radius_after = 8192.0;
  iteration.lambda_before = 1.0 / 4096.0;
  iteration.lambda_after = 1.0 / 8192.0;
  iteration.lm_diagonal_min = 1e-6;
  iteration.lm_diagonal_max = 1e6;
  iteration.predicted_reduction = 4.0;
  iteration.actual_reduction = 3.25;
  iteration.rho = 0.8125;
  iteration.function_metric = 0.3;
  iteration.parameter_metric = 0.02;
  iteration.step_norm = 0.01;
  iteration.backward_error = 1e-11;
  iteration.factorization_success = true;
  iteration.step_valid = true;
  iteration.trial_finite = true;
  iteration.accepted_decision = true;
  iteration.accepted_commit_success = true;
  iteration.accepted = true;
  iteration.invalid = false;
  iteration.termination_reason = "fixture_iteration";
  result.trace.push_back(iteration);
  result.accepted_state_trace.push_back(Snapshot());

  CudaFullLmRuntimeInfo& runtime = result.runtime;
  runtime.actual_trials = 3;
  runtime.accepted_trials = 1;
  runtime.rejected_trials = 1;
  runtime.final_internal_state_epoch = 1;
  runtime.performance_mode_requested = true;
  runtime.effective_reduction_mode =
      CudaReductionMode::kParallelDeterministic;
  runtime.instrumentation_effective = true;
  runtime.current_linearization_cache_requested = true;
  runtime.current_linearization_cache_effective = true;
  runtime.single_stream_synchronous = true;
  runtime.buffers_reused_across_iterations = true;
  runtime.topology_reused_across_iterations = true;
  runtime.cache_access_serialized = true;
  runtime.cache_context_isolated = true;
  runtime.initial_resource_health = CudaResourceHealth::kClean;
  runtime.final_resource_health = CudaResourceHealth::kTainted;
  runtime.layer_a_calls = 5;
  runtime.layer_b_calls = 3;
  runtime.layer_c_calls = 2;
  runtime.cost_calls = 5;
  runtime.build_cuda_layer_a_inputs_count = 12;
  runtime.build_cost_order_count = 5;
  runtime.topology_epoch = 2;
  runtime.topology_build_count = 1;
  runtime.topology_refresh_count = 2;
  runtime.cache_lookup_count = 4;
  runtime.cache_hit_count = 2;
  runtime.cache_miss_count = 2;
  runtime.cache_invalidation_count = 1;
  runtime.topology_fingerprint = "0123456789abcdef";
  runtime.topology_fingerprint_computations = 1;
  runtime.current_linearization_logical_requests = 5;
  runtime.current_linearization_lookup_aborts = 1;
  runtime.current_linearization_cache_lookups = 4;
  runtime.current_linearization_cache_hits = 2;
  runtime.current_linearization_cache_misses = 2;
  runtime.current_linearization_build_attempts = 2;
  runtime.current_linearization_build_successes = 1;
  runtime.current_linearization_build_failures = 1;
  runtime.current_linearization_temporary_builds = 0;
  runtime.current_linearization_publishes = 1;
  runtime.current_linearization_replacements = 0;
  runtime.current_linearization_invalidations = 1;
  runtime.current_linearization_borrows = 2;
  runtime.current_linearization_copies = 0;
  runtime.current_linearization_teardowns = 1;
  runtime.current_linearization_builds_by_state_epoch = {1, 0};
  runtime.final_config_generation = 101;
  runtime.final_topology_generation = 202;
  runtime.solve_generation_nonzero_checks = 1;
  runtime.solve_generation_nonzero_violations = 0;
  runtime.solve_generation_nonzero = true;
  runtime.solve_generation_consistency_checks = 7;
  runtime.solve_generation_consistency_violations = 0;
  runtime.solve_generation_consistent = true;
  runtime.linearization_identity_checks = 8;
  runtime.linearization_identity_violations = 1;
  runtime.linearization_identity_consistent = false;
  runtime.topology_context_generation_checks = 8;
  runtime.topology_context_generation_violations = 0;
  runtime.topology_context_generation_consistent = true;
  runtime.final_linearization_reason = "new_state_relinearization";
  runtime.buffer_reuse_hits = 20;
  runtime.topology_reuse_hits = 10;
  runtime.stream_reuse_hits = 9;
  runtime.event_reuse_hits = 8;
  runtime.solver_handle_reuse_hits = 7;
  runtime.blas_handle_reuse_hits = 6;
  runtime.audit_capacity_preflight_pass = true;
  runtime.fault_record_capacity = kCudaFaultRecordCapacity;
  runtime.secondary_diagnostic_capacity =
      kCudaSecondaryDiagnosticCapacity;
  runtime.resource_registry_capacity = kCudaResourceRegistryCapacity;
  runtime.max_open_timing_interval_capacity =
      kCudaMaxOpenTimingIntervalCapacity;
  runtime.audit_overflow_count = 2;
  runtime.has_first_audit_overflow_site = true;
  runtime.first_audit_overflow_site = CudaFailureSite::kAuditRuntimeOverflow;

  CudaSynchronizationAudit& synchronization =
      runtime.timing.synchronization_audit;
  synchronization.stream_synchronize_calls = 11;
  synchronization.event_synchronize_calls = 2;
  synchronization.device_synchronize_calls = 0;
  synchronization.layer_a_final.calls = 1;
  synchronization.layer_b_final.calls = 2;
  synchronization.layer_c_factor.calls = 3;
  synchronization.layer_c_solver.calls = 1;
  synchronization.layer_c_final.calls = 1;
  synchronization.trial_cost_final.calls = 2;
  synchronization.resource_release.calls = 1;

  runtime.state_hash_audit.layout_builds = 1;
  runtime.state_hash_audit.layout_blocks = 4;
  runtime.state_hash_audit.current.requests = 5;
  runtime.state_hash_audit.current.computations = 2;
  runtime.state_hash_audit.current.cache_hits = 3;
  runtime.state_hash_audit.current.blocks_visited = 8;
  runtime.state_hash_audit.current.scalars_hashed = 32;
  runtime.state_hash_audit.trial.requests = 3;
  runtime.state_hash_audit.trial.computations = 3;
  runtime.state_hash_audit.trial.cache_hits = 0;
  runtime.state_hash_audit.trial.blocks_visited = 12;
  runtime.state_hash_audit.trial.scalars_hashed = 48;

  std::vector<CudaPhaseTiming*> phases = {
      &runtime.timing.layer_a_call,
      &runtime.timing.layer_b_call,
      &runtime.timing.layer_c_call,
      &runtime.timing.build_cuda_layer_a_inputs,
      &runtime.timing.topology_lookup,
      &runtime.timing.topology_build,
      &runtime.timing.topology_refresh,
      &runtime.timing.topology_cache,
      &runtime.timing.build_cost_order,
      &runtime.timing.allocation,
      &runtime.timing.free,
      &runtime.timing.h2d_memcpy,
      &runtime.timing.d2h_memcpy,
      &runtime.timing.d2d_memcpy,
      &runtime.timing.managed_prefetch,
      &runtime.timing.layer_a_kernel,
      &runtime.timing.layer_b_pose_kernel,
      &runtime.timing.layer_b_point_kernel,
      &runtime.timing.layer_b_edge_kernel,
      &runtime.timing.layer_b_gradient_kernel,
      &runtime.timing.layer_c_point_factor,
      &runtime.timing.schur,
      &runtime.timing.rhs,
      &runtime.timing.cusolver,
      &runtime.timing.back_substitution,
      &runtime.timing.back_substitution_host_self,
      &runtime.timing.trial_cost,
      &runtime.timing.apply_layer_c_step,
      &runtime.timing.compute_layer_c_diagnostics,
      &runtime.timing.lm_decision,
      &runtime.timing.synchronization,
      &runtime.timing.state_hash,
      &runtime.timing.topology_fingerprint_audit,
      &runtime.timing.controller_self,
      &runtime.timing.resource_release,
      &runtime.timing.other,
  };
  for (size_t i = 0; i < phases.size(); ++i) {
    phases[i]->calls = static_cast<uint64_t>(i + 1);
    phases[i]->bytes = static_cast<uint64_t>((i + 1) * 17);
    phases[i]->host_wall_milliseconds = 1000.0 + i;
    phases[i]->cuda_event_milliseconds = 2000.0 + i;
  }

  runtime.fault_consumption_count = 2;
  runtime.fault_consumptions[0].logical_site =
      CudaFaultLogicalSite::kAcceptedPendingLinearization;
  runtime.fault_consumptions[0].trigger_phase =
      CudaFaultTriggerPhase::kBeforeCacheLookup;
  runtime.fault_consumptions[0].operation = 0;
  runtime.fault_consumptions[0].configured_fault_kind =
      CudaFaultInjection::kForceLinearizationFailure;
  runtime.fault_consumptions[0].epoch = 1;
  runtime.fault_consumptions[0].occurrence = 1;
  runtime.fault_consumptions[0].triggered = true;
  runtime.fault_consumptions[1].logical_site =
      CudaFaultLogicalSite::kLayerCWorkspace;
  runtime.fault_consumptions[1].trigger_phase =
      CudaFaultTriggerPhase::kLayerCOperation;
  runtime.fault_consumptions[1].operation = 2;
  runtime.fault_consumptions[1].configured_fault_kind =
      CudaFaultInjection::kForceWorkspaceAllocationFailure;
  runtime.fault_consumptions[1].epoch = 1;
  runtime.fault_consumptions[1].occurrence = 2;
  runtime.fault_consumptions[1].triggered = false;

  runtime.resource_generation_advance_events = 2;
  runtime.resource_generation_advance_violations = 1;
  runtime.resource_generation_advanced_by_this_solve = false;
  runtime.resource_taint_count = 1;
  runtime.resource_cleanup_attempts = 1;
  runtime.resource_cleanup_successes = 0;
  runtime.resource_cleanup_failures = 1;
  runtime.resource_lookup_blocked_count = 3;
  runtime.cross_solve_cache_hits = 0;
  runtime.resource_quarantine_count = 2;
  runtime.resource_cleanup_complete = false;
  runtime.teardown_record_count = kCudaTeardownRecordCapacity;
  for (size_t i = 0; i < runtime.teardown_records.size(); ++i) {
    CudaTeardownRecordV2& record = runtime.teardown_records[i];
    record.teardown_type = static_cast<CudaTeardownType>(i);
    record.device = static_cast<int32_t>(i % 2);
    record.creation_sequence = i == 0 ? 0 : 100 + i;
    record.attempted = true;
    record.success = i != 6;
    record.status_domain = i == 2 ? CudaStatusDomain::kCusolver
                                  : i == 3 ? CudaStatusDomain::kCublas
                                           : CudaStatusDomain::kCudaRuntime;
    record.raw_status_code = i == 6 ? 999 : 0;
    record.quarantined = i == 6;
  }
  runtime.has_first_cleanup_failure = true;
  runtime.first_cleanup_failure =
      Diagnostic(CudaSolveErrorClass::kResourceCleanup,
                 CudaFailureSite::kDestroyAllocation,
                 CudaStatusDomain::kCudaRuntime, 999);

  runtime.timing_structure_v2.timing_status = CudaTimingStatus::kIncomplete;
  runtime.timing_structure_v2.event_attempts = 3;
  runtime.timing_structure_v2.event_completions = 2;
  runtime.timing_structure_v2.event_failures = 1;
  runtime.timing_structure_v2.intervals_started = 2;
  runtime.timing_structure_v2.intervals_completed = 1;
  runtime.timing_structure_v2.intervals_abandoned = 1;
  runtime.timing_structure_v2.open_intervals_at_finalize = 0;
  runtime.timing_structure_v2.has_first_incomplete_site = true;
  runtime.timing_structure_v2.first_incomplete_site =
      CudaFailureSite::kTimingEventRecord;

  CudaDiagnosticStructureV2& diagnostics =
      runtime.diagnostic_structure_v2;
  diagnostics.has_primary = true;
  diagnostics.primary = runtime.first_cleanup_failure;
  diagnostics.secondary_count = kCudaSecondaryDiagnosticCapacity;
  for (size_t i = 0; i < diagnostics.secondary.size(); ++i) {
    diagnostics.secondary[i] = Diagnostic(
        i == 0 ? CudaSolveErrorClass::kAuditCapacity
               : CudaSolveErrorClass::kCudaTiming,
        i == 0 ? CudaFailureSite::kAuditRuntimeOverflow
               : CudaFailureSite::kTimingEventRecord,
        i == 0 ? CudaStatusDomain::kInternal
               : CudaStatusDomain::kCudaRuntime,
        static_cast<int64_t>(i + 1));
  }
  diagnostics.overflow_sentinel_set = true;
  diagnostics.diagnostic_overflow_count = 3;
  diagnostics.overflow_sentinel =
      Diagnostic(CudaSolveErrorClass::kAuditCapacity,
                 CudaFailureSite::kAuditRuntimeOverflow,
                 CudaStatusDomain::kInternal, 3);
  return result;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " OUTPUT_DIRECTORY\n";
    return EXIT_FAILURE;
  }
  std::string error;
  if (!colmap::gpu_ba::EnsureDirectory(argv[1], &error)) {
    std::cerr << error << '\n';
    return EXIT_FAILURE;
  }
  const colmap::gpu_ba::CudaFullLmResult fixture =
      colmap::gpu_ba::Fixture();
  const std::string semantic =
      colmap::gpu_ba::CudaFullLmSemanticCanonicalV2(fixture);
  const std::string execution =
      colmap::gpu_ba::CudaFullLmExecutionCanonicalV2(fixture);
  if (semantic != colmap::gpu_ba::CudaFullLmSemanticCanonicalV2(fixture) ||
      execution != colmap::gpu_ba::CudaFullLmExecutionCanonicalV2(fixture)) {
    std::cerr << "canonical V2 serializer is not deterministic\n";
    return EXIT_FAILURE;
  }
  const std::string root = argv[1];
  if (!colmap::gpu_ba::WriteBinary(root + "/semantic_v2_canonical.bin",
                                    semantic, &error) ||
      !colmap::gpu_ba::WriteBinary(root + "/execution_v2_canonical.bin",
                                    execution, &error) ||
      !colmap::gpu_ba::WriteText(root + "/semantic_v2_canonical.hex",
                                 colmap::gpu_ba::Hex(semantic), &error) ||
      !colmap::gpu_ba::WriteText(root + "/execution_v2_canonical.hex",
                                 colmap::gpu_ba::Hex(execution), &error)) {
    std::cerr << error << '\n';
    return EXIT_FAILURE;
  }
  const std::string semantic_hash =
      colmap::gpu_ba::CudaFullLmSemanticSha256V2(fixture);
  const std::string execution_hash =
      colmap::gpu_ba::CudaFullLmExecutionSha256V2(fixture);
  std::ostringstream json;
  json << "{\n"
       << "  \"schema\":\"phase7p1a-r1-canonical-v2-golden\",\n"
       << "  \"semantic_bytes\":" << semantic.size() << ",\n"
       << "  \"execution_bytes\":" << execution.size() << ",\n"
       << "  \"semantic_sha256\":\"" << semantic_hash << "\",\n"
       << "  \"execution_sha256\":\"" << execution_hash << "\",\n"
       << "  \"coverage\":{\"lookup_abort\":true,"
          "\"fault_triggered\":true,\"double_advance\":true,"
          "\"timing_incomplete\":true,\"cleanup\":true,"
          "\"quarantine\":true,\"secondary_full_primary_promotion\":true,"
          "\"diagnostic_overflow\":true,\"audit_capacity\":true,"
          "\"optional_fields\":true},\n"
       << "  \"diagnostic_overflow_count\":3,\n"
       << "  \"audit_overflow_count\":2,\n"
       << "  \"primary_error_classification\":8\n"
       << "}\n";
  if (!colmap::gpu_ba::WriteText(root + "/fixture.json", json.str(),
                                  &error) ||
      !colmap::gpu_ba::WriteText(root + "/canonical.sha256",
          semantic_hash + "  semantic_v2_canonical.bin\n" +
          execution_hash + "  execution_v2_canonical.bin\n", &error)) {
    std::cerr << error << '\n';
    return EXIT_FAILURE;
  }
  std::cout << json.str();
  return EXIT_SUCCESS;
}
