#include "gpu_ba/custom_cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace colmap {
namespace gpu_ba {
namespace {

void AppendU8(std::string* data, uint8_t value) {
  data->push_back(static_cast<char>(value));
}

void AppendU16(std::string* data, uint16_t value) {
  for (int shift = 0; shift < 16; shift += 8) {
    data->push_back(static_cast<char>((value >> shift) & 0xffu));
  }
}

void AppendU32(std::string* data, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    data->push_back(static_cast<char>((value >> shift) & 0xffu));
  }
}

void AppendU64(std::string* data, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    data->push_back(static_cast<char>((value >> shift) & 0xffu));
  }
}

void AppendI32(std::string* data, int32_t value) {
  AppendU32(data, static_cast<uint32_t>(value));
}

void AppendI64(std::string* data, int64_t value) {
  AppendU64(data, static_cast<uint64_t>(value));
}

void AppendBool(std::string* data, bool value) {
  AppendU8(data, value ? 1u : 0u);
}

void AppendDouble(std::string* data, double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "IEEE-754 binary64 required");
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU64(data, bits);
}

void AppendString(std::string* data, const std::string& value) {
  AppendU64(data, value.size());
  data->append(value);
}

void AppendOptionalU16(std::string* data, bool present, uint16_t value) {
  AppendBool(data, present);
  if (present) AppendU16(data, value);
}

void AppendDiagnosticRecord(std::string* data,
                            const CudaStableDiagnosticV2& value) {
  AppendU8(data, static_cast<uint8_t>(value.error_classification));
  AppendU16(data, static_cast<uint16_t>(value.failure_site));
  AppendU8(data, static_cast<uint8_t>(value.status_domain));
  AppendI64(data, value.raw_status_code);
}

void AppendDiagnosticStructure(std::string* data,
                               const CudaDiagnosticStructureV2& value) {
  AppendBool(data, value.has_primary);
  if (value.has_primary) AppendDiagnosticRecord(data, value.primary);
  AppendU64(data, value.secondary_count);
  const uint64_t occupied =
      std::min<uint64_t>(value.secondary_count, value.secondary.size());
  for (uint64_t i = 0; i < occupied; ++i) {
    AppendDiagnosticRecord(data, value.secondary[static_cast<size_t>(i)]);
  }
  AppendBool(data, value.overflow_sentinel_set);
  AppendU64(data, value.diagnostic_overflow_count);
  if (value.overflow_sentinel_set) {
    AppendDiagnosticRecord(data, value.overflow_sentinel);
  }
}

void AppendSemanticIteration(std::string* data,
                             const CudaLmIteration& value) {
  AppendI32(data, value.iteration);
  AppendDouble(data, value.cost_before);
  AppendDouble(data, value.trial_cost);
  AppendDouble(data, value.cost_after);
  AppendDouble(data, value.projected_gradient_max_norm);
  AppendDouble(data, value.scaled_gradient_norm);
  AppendDouble(data, value.radius_before);
  AppendDouble(data, value.radius_after);
  AppendDouble(data, value.lambda_before);
  AppendDouble(data, value.lambda_after);
  AppendDouble(data, value.lm_diagonal_min);
  AppendDouble(data, value.lm_diagonal_max);
  AppendDouble(data, value.predicted_reduction);
  AppendDouble(data, value.actual_reduction);
  AppendDouble(data, value.rho);
  AppendDouble(data, value.function_metric);
  AppendDouble(data, value.parameter_metric);
  AppendDouble(data, value.step_norm);
  AppendDouble(data, value.backward_error);
  AppendBool(data, value.factorization_success);
  AppendBool(data, value.step_valid);
  AppendBool(data, value.trial_finite);
  AppendBool(data, value.accepted_decision);
  AppendBool(data, value.accepted_commit_success);
  AppendBool(data, value.accepted);
  AppendBool(data, value.invalid);
  AppendString(data, value.termination_reason);
}

void AppendPhase(std::string* data,
                 uint8_t phase_id,
                 const CudaPhaseTiming& phase) {
  AppendU8(data, phase_id);
  AppendU64(data, phase.calls);
  AppendU64(data, phase.bytes);
}

void AppendFixedPhases(std::string* data, const CudaTimingLedger& timing) {
  AppendU64(data, 36);
  AppendPhase(data, 0, timing.layer_a_call);
  AppendPhase(data, 1, timing.layer_b_call);
  AppendPhase(data, 2, timing.layer_c_call);
  AppendPhase(data, 3, timing.build_cuda_layer_a_inputs);
  AppendPhase(data, 4, timing.topology_lookup);
  AppendPhase(data, 5, timing.topology_build);
  AppendPhase(data, 6, timing.topology_refresh);
  AppendPhase(data, 7, timing.topology_cache);
  AppendPhase(data, 8, timing.build_cost_order);
  AppendPhase(data, 9, timing.allocation);
  AppendPhase(data, 10, timing.free);
  AppendPhase(data, 11, timing.h2d_memcpy);
  AppendPhase(data, 12, timing.d2h_memcpy);
  AppendPhase(data, 13, timing.d2d_memcpy);
  AppendPhase(data, 14, timing.managed_prefetch);
  AppendPhase(data, 15, timing.layer_a_kernel);
  AppendPhase(data, 16, timing.layer_b_pose_kernel);
  AppendPhase(data, 17, timing.layer_b_point_kernel);
  AppendPhase(data, 18, timing.layer_b_edge_kernel);
  AppendPhase(data, 19, timing.layer_b_gradient_kernel);
  AppendPhase(data, 20, timing.layer_c_point_factor);
  AppendPhase(data, 21, timing.schur);
  AppendPhase(data, 22, timing.rhs);
  AppendPhase(data, 23, timing.cusolver);
  AppendPhase(data, 24, timing.back_substitution);
  AppendPhase(data, 25, timing.back_substitution_host_self);
  AppendPhase(data, 26, timing.trial_cost);
  AppendPhase(data, 27, timing.apply_layer_c_step);
  AppendPhase(data, 28, timing.compute_layer_c_diagnostics);
  AppendPhase(data, 29, timing.lm_decision);
  AppendPhase(data, 30, timing.synchronization);
  AppendPhase(data, 31, timing.state_hash);
  AppendPhase(data, 32, timing.topology_fingerprint_audit);
  AppendPhase(data, 33, timing.controller_self);
  AppendPhase(data, 34, timing.resource_release);
  AppendPhase(data, 35, timing.other);
}

void AppendSynchronization(std::string* data,
                           const CudaSynchronizationAudit& value) {
  AppendU64(data, value.stream_synchronize_calls);
  AppendU64(data, value.event_synchronize_calls);
  AppendU64(data, value.device_synchronize_calls);
  AppendU64(data, value.layer_a_final.calls);
  AppendU64(data, value.layer_b_final.calls);
  AppendU64(data, value.layer_c_factor.calls);
  AppendU64(data, value.layer_c_solver.calls);
  AppendU64(data, value.layer_c_final.calls);
  AppendU64(data, value.trial_cost_final.calls);
  AppendU64(data, value.resource_release.calls);
}

void AppendStateHashCounter(std::string* data,
                            const CudaStateHashCounter& value) {
  AppendU64(data, value.requests);
  AppendU64(data, value.computations);
  AppendU64(data, value.cache_hits);
  AppendU64(data, value.blocks_visited);
  AppendU64(data, value.scalars_hashed);
}

void AppendStateHash(std::string* data, const CudaStateHashAudit& value) {
  AppendU64(data, value.layout_builds);
  AppendU64(data, value.layout_blocks);
  AppendStateHashCounter(data, value.current);
  AppendStateHashCounter(data, value.trial);
}

void AppendFaultConsumptions(std::string* data,
                             const CudaFullLmRuntimeInfo& runtime) {
  const uint64_t count = std::min<uint64_t>(
      runtime.fault_consumption_count, runtime.fault_consumptions.size());
  AppendU64(data, count);
  for (uint64_t i = 0; i < count; ++i) {
    const CudaFaultConsumptionV2& value =
        runtime.fault_consumptions[static_cast<size_t>(i)];
    AppendU8(data, static_cast<uint8_t>(value.logical_site));
    AppendU8(data, static_cast<uint8_t>(value.trigger_phase));
    AppendU16(data, value.operation);
    AppendU8(data, static_cast<uint8_t>(value.configured_fault_kind));
    AppendU64(data, value.epoch);
    AppendU64(data, value.occurrence);
    AppendBool(data, value.triggered);
  }
}

void AppendResource(std::string* data,
                    const CudaFullLmRuntimeInfo& runtime) {
  AppendU8(data, static_cast<uint8_t>(runtime.initial_resource_health));
  AppendU8(data, static_cast<uint8_t>(runtime.final_resource_health));
  AppendU64(data, runtime.resource_generation_advance_events);
  AppendU64(data, runtime.resource_generation_advance_violations);
  AppendBool(data, runtime.resource_generation_advanced_by_this_solve);
  AppendU64(data, runtime.resource_taint_count);
  AppendU64(data, runtime.resource_cleanup_attempts);
  AppendU64(data, runtime.resource_cleanup_successes);
  AppendU64(data, runtime.resource_cleanup_failures);
  AppendU64(data, runtime.resource_lookup_blocked_count);
  AppendU64(data, runtime.cross_solve_cache_hits);
  AppendU64(data, runtime.resource_quarantine_count);
  AppendBool(data, runtime.resource_cleanup_complete);
  AppendU64(data, runtime.teardown_record_count);
  for (size_t i = 0; i < runtime.teardown_records.size(); ++i) {
    const CudaTeardownRecordV2& value = runtime.teardown_records[i];
    AppendU8(data, static_cast<uint8_t>(value.teardown_type));
    AppendI32(data, value.device);
    AppendU64(data, value.creation_sequence);
    AppendBool(data, value.attempted);
    AppendBool(data, value.success);
    AppendU8(data, static_cast<uint8_t>(value.status_domain));
    AppendI64(data, value.raw_status_code);
    AppendBool(data, value.quarantined);
  }
  AppendBool(data, runtime.has_first_cleanup_failure);
  if (runtime.has_first_cleanup_failure) {
    AppendDiagnosticRecord(data, runtime.first_cleanup_failure);
  }
}

void AppendTiming(std::string* data, const CudaTimingStructureV2& value) {
  AppendU8(data, static_cast<uint8_t>(value.timing_status));
  AppendU64(data, value.event_attempts);
  AppendU64(data, value.event_completions);
  AppendU64(data, value.event_failures);
  AppendU64(data, value.intervals_started);
  AppendU64(data, value.intervals_completed);
  AppendU64(data, value.intervals_abandoned);
  AppendU64(data, value.open_intervals_at_finalize);
  AppendOptionalU16(data, value.has_first_incomplete_site,
                    static_cast<uint16_t>(value.first_incomplete_site));
}

}  // namespace

std::string CudaFullLmSemanticCanonicalV2(
    const CudaFullLmResult& result) {
  std::string data;
  AppendString(&data, "CudaFullLmSemanticV2");
  AppendBool(&data, result.success);
  AppendU8(&data, static_cast<uint8_t>(result.error_classification));
  AppendU8(&data, static_cast<uint8_t>(result.termination_type));
  AppendString(&data, result.termination_reason);
  AppendDouble(&data, result.initial_cost);
  AppendDouble(&data, result.final_cost);
  AppendDouble(&data, result.initial_projected_gradient_max_norm);
  AppendDouble(&data, result.final_projected_gradient_max_norm);
  AppendDouble(&data, result.initial_scaled_gradient_norm);
  AppendDouble(&data, result.final_scaled_gradient_norm);
  AppendDouble(&data, result.final_radius);
  AppendDouble(&data, result.final_lambda);
  AppendI32(&data, result.trial_iterations);
  AppendI32(&data, result.accepted_steps);
  AppendI32(&data, result.rejected_steps);
  AppendI32(&data, result.invalid_steps);
  AppendI32(&data, result.factorization_failures);
  AppendI32(&data, result.accepted_decisions);
  AppendI32(&data, result.accepted_commits);
  AppendI32(&data, result.accepted_pending_preparation_failures);
  AppendI32(&data, result.accepted_pending_linearization_failures);
  AppendU64(&data, result.runtime.actual_trials);
  AppendU64(&data, result.runtime.accepted_trials);
  AppendU64(&data, result.runtime.rejected_trials);
  AppendU64(&data, result.runtime.final_internal_state_epoch);
  AppendU64(&data, result.trace.size());
  for (const CudaLmIteration& iteration : result.trace) {
    AppendSemanticIteration(&data, iteration);
  }
  AppendString(&data, CudaFinalParametersBitwiseSha256(result.final_state));
  AppendString(&data, CudaFinalTopologyBitwiseSha256(result.final_state));
  AppendU64(&data, result.accepted_state_trace.size());
  for (const Snapshot& state : result.accepted_state_trace) {
    AppendString(&data, CudaFinalParametersBitwiseSha256(state));
    AppendString(&data, CudaFinalTopologyBitwiseSha256(state));
  }
  return data;
}

std::string CudaFullLmExecutionCanonicalV2(
    const CudaFullLmResult& result) {
  const CudaFullLmRuntimeInfo& runtime = result.runtime;
  std::string data;
  AppendString(&data, "CudaFullLmExecutionStructureV2");
  AppendBool(&data, runtime.performance_mode_requested);
  AppendU8(&data, static_cast<uint8_t>(runtime.effective_reduction_mode));
  AppendBool(&data, runtime.instrumentation_effective);
  AppendBool(&data, runtime.current_linearization_cache_requested);
  AppendBool(&data, runtime.current_linearization_cache_effective);
  AppendBool(&data, runtime.single_stream_synchronous);
  AppendBool(&data, runtime.buffers_reused_across_iterations);
  AppendBool(&data, runtime.topology_reused_across_iterations);
  AppendBool(&data, runtime.cache_access_serialized);
  AppendBool(&data, runtime.cache_context_isolated);
  AppendU8(&data, static_cast<uint8_t>(runtime.initial_resource_health));
  AppendU8(&data, static_cast<uint8_t>(runtime.final_resource_health));
  AppendU64(&data, runtime.layer_a_calls);
  AppendU64(&data, runtime.layer_b_calls);
  AppendU64(&data, runtime.layer_c_calls);
  AppendU64(&data, runtime.cost_calls);
  AppendU64(&data, runtime.build_cuda_layer_a_inputs_count);
  AppendU64(&data, runtime.build_cost_order_count);
  AppendU64(&data, runtime.topology_epoch);
  AppendU64(&data, runtime.topology_build_count);
  AppendU64(&data, runtime.topology_refresh_count);
  AppendU64(&data, runtime.cache_lookup_count);
  AppendU64(&data, runtime.cache_hit_count);
  AppendU64(&data, runtime.cache_miss_count);
  AppendU64(&data, runtime.cache_invalidation_count);
  AppendString(&data, runtime.topology_fingerprint);
  AppendU64(&data, runtime.topology_fingerprint_computations);
  AppendU64(&data, runtime.current_linearization_logical_requests);
  AppendU64(&data, runtime.current_linearization_lookup_aborts);
  AppendU64(&data, runtime.current_linearization_cache_lookups);
  AppendU64(&data, runtime.current_linearization_cache_hits);
  AppendU64(&data, runtime.current_linearization_cache_misses);
  AppendU64(&data, runtime.current_linearization_build_attempts);
  AppendU64(&data, runtime.current_linearization_build_successes);
  AppendU64(&data, runtime.current_linearization_build_failures);
  AppendU64(&data, runtime.current_linearization_temporary_builds);
  AppendU64(&data, runtime.current_linearization_publishes);
  AppendU64(&data, runtime.current_linearization_replacements);
  AppendU64(&data, runtime.current_linearization_invalidations);
  AppendU64(&data, runtime.current_linearization_borrows);
  AppendU64(&data, runtime.current_linearization_copies);
  AppendU64(&data, runtime.current_linearization_teardowns);
  AppendU64(&data, runtime.current_linearization_builds_by_state_epoch.size());
  for (uint64_t value : runtime.current_linearization_builds_by_state_epoch) {
    AppendU64(&data, value);
  }
  AppendU64(&data, runtime.final_config_generation);
  AppendU64(&data, runtime.final_topology_generation);
  AppendU64(&data, runtime.solve_generation_nonzero_checks);
  AppendU64(&data, runtime.solve_generation_nonzero_violations);
  AppendBool(&data, runtime.solve_generation_nonzero);
  AppendU64(&data, runtime.solve_generation_consistency_checks);
  AppendU64(&data, runtime.solve_generation_consistency_violations);
  AppendBool(&data, runtime.solve_generation_consistent);
  AppendU64(&data, runtime.linearization_identity_checks);
  AppendU64(&data, runtime.linearization_identity_violations);
  AppendBool(&data, runtime.linearization_identity_consistent);
  AppendU64(&data, runtime.topology_context_generation_checks);
  AppendU64(&data, runtime.topology_context_generation_violations);
  AppendBool(&data, runtime.topology_context_generation_consistent);
  AppendString(&data, runtime.final_linearization_reason);
  AppendU64(&data, runtime.buffer_reuse_hits);
  AppendU64(&data, runtime.topology_reuse_hits);
  AppendU64(&data, runtime.stream_reuse_hits);
  AppendU64(&data, runtime.event_reuse_hits);
  AppendU64(&data, runtime.solver_handle_reuse_hits);
  AppendU64(&data, runtime.blas_handle_reuse_hits);
  AppendBool(&data, runtime.audit_capacity_preflight_pass);
  AppendU64(&data, runtime.fault_record_capacity);
  AppendU64(&data, runtime.secondary_diagnostic_capacity);
  AppendU64(&data, runtime.resource_registry_capacity);
  AppendU64(&data, runtime.max_open_timing_interval_capacity);
  AppendU64(&data, runtime.audit_overflow_count);
  AppendOptionalU16(&data, runtime.has_first_audit_overflow_site,
                    static_cast<uint16_t>(runtime.first_audit_overflow_site));
  AppendSynchronization(&data, runtime.timing.synchronization_audit);
  AppendStateHash(&data, runtime.state_hash_audit);
  AppendFixedPhases(&data, runtime.timing);
  AppendFaultConsumptions(&data, runtime);
  AppendResource(&data, runtime);
  AppendTiming(&data, runtime.timing_structure_v2);
  AppendDiagnosticStructure(&data, runtime.diagnostic_structure_v2);
  return data;
}

std::string CudaFullLmDiagnosticCanonicalV2(
    const CudaFullLmResult& result) {
  std::string data;
  AppendString(&data, "CudaFullLmDiagnosticStructureV2");
  AppendDiagnosticStructure(&data, result.runtime.diagnostic_structure_v2);
  return data;
}

std::string CudaFullLmSemanticSha256V2(const CudaFullLmResult& result) {
  return Sha256Hex(CudaFullLmSemanticCanonicalV2(result));
}

std::string CudaFullLmExecutionSha256V2(const CudaFullLmResult& result) {
  return Sha256Hex(CudaFullLmExecutionCanonicalV2(result));
}

std::string CudaFullLmDiagnosticSha256V2(const CudaFullLmResult& result) {
  return Sha256Hex(CudaFullLmDiagnosticCanonicalV2(result));
}

}  // namespace gpu_ba
}  // namespace colmap
