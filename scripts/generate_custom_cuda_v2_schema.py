#!/usr/bin/env python3
"""Validate canonical CUDA BA V2 goldens and emit schema/offset artifacts."""

import argparse
import json
import struct
from pathlib import Path


ENUMS = {
    "CudaSolveErrorClass": {
        "None": 0, "InvalidOptions": 1, "InitialLinearization": 2,
        "CurrentLinearization": 3, "CudaStep": 4, "TrialStateHash": 5,
        "NonfiniteModel": 6, "AcceptedPendingLinearization": 7,
        "ResourceCleanup": 8, "FinalStateHash": 9,
        "AcceptedPendingPreparation": 10, "CudaTiming": 11,
        "AuditCapacity": 12,
    },
    "TimingStatus": {"disabled": 0, "complete": 1, "incomplete": 2},
    "StatusDomain": {
        "none": 0, "cuda_runtime": 1, "cusolver": 2, "cublas": 3,
        "injected": 4, "cxx_exception": 5, "internal": 6,
    },
    "FaultTriggerPhase": {
        "none": 0, "accepted_pending_preparation": 1,
        "before_cache_lookup": 2, "after_cache_miss": 3,
        "after_topology_refresh_before_publish": 4,
        "layer_c_operation": 5,
    },
    "TeardownType": {
        "get_device": 0, "set_device": 1, "solver": 2, "blas": 3,
        "event": 4, "stream": 5, "allocation": 6,
    },
    "FaultInjection": {
        "none": 0, "insufficient_memory": 1,
        "point_factorization_failure": 2, "nonfinite_trial": 3,
        "workspace_allocation_failure": 4, "linearization_failure": 5,
        "resource_teardown_failure": 6,
        "pending_preparation_failure": 7, "timing_api_failure": 8,
        "identity_mismatch": 9, "resource_double_advance": 10,
    },
    "FailureSite": {
        "None": 0, "TimingEventAcquire": 1, "TimingEventRecord": 2,
        "TimingElapsedQuery": 3, "CleanupGetDevice": 4,
        "CleanupSetDevice": 5, "DestroySolver": 6, "DestroyBlas": 7,
        "DestroyEvent": 8, "DestroyStream": 9,
        "DestroyAllocation": 10, "PendingPreparation": 11,
        "InitialLinearization": 12, "CurrentLinearization": 13,
        "AcceptedPendingLinearization": 14, "LayerCPointFactor": 15,
        "LayerCWorkspace": 16, "TrialCost": 17, "StateHash": 18,
        "AuditPreflightCapacity": 19, "AuditRuntimeOverflow": 20,
        "ResourceGenerationDoubleAdvance": 21,
    },
}

SEMANTIC_ITERATION_FIELDS = [
    ("iteration", "i32"),
    ("cost_before", "f64"), ("trial_cost", "f64"),
    ("cost_after", "f64"), ("projected_gradient", "f64"),
    ("scaled_gradient", "f64"), ("radius_before", "f64"),
    ("radius_after", "f64"), ("lambda_before", "f64"),
    ("lambda_after", "f64"), ("lm_diagonal_min", "f64"),
    ("lm_diagonal_max", "f64"), ("predicted_reduction", "f64"),
    ("actual_reduction", "f64"), ("rho", "f64"),
    ("function_metric", "f64"), ("parameter_metric", "f64"),
    ("step_norm", "f64"), ("backward_error", "f64"),
    ("factorization_success", "bool"), ("step_valid", "bool"),
    ("trial_finite", "bool"), ("accepted_decision", "bool"),
    ("accepted_commit_success", "bool"), ("accepted", "bool"),
    ("invalid", "bool"), ("termination_reason", "string"),
]

SEMANTIC_FIELDS = [
    ("magic", "string"), ("success", "bool"),
    ("error_classification", "u8"), ("termination_type", "u8"),
    ("termination_reason", "string"), ("initial_cost", "f64"),
    ("final_cost", "f64"), ("initial_projected_gradient", "f64"),
    ("final_projected_gradient", "f64"),
    ("initial_scaled_gradient", "f64"),
    ("final_scaled_gradient", "f64"), ("final_radius", "f64"),
    ("final_lambda", "f64"), ("trial_iterations", "i32"),
    ("accepted_steps", "i32"), ("rejected_steps", "i32"),
    ("invalid_steps", "i32"), ("factorization_failures", "i32"),
    ("accepted_decisions", "i32"), ("accepted_commits", "i32"),
    ("accepted_pending_preparation_failures", "i32"),
    ("accepted_pending_linearization_failures", "i32"),
    ("actual_trials", "u64"), ("accepted_trials", "u64"),
    ("rejected_trials", "u64"), ("final_internal_state_epoch", "u64"),
    ("trace", "vector<SemanticIterationV2>"),
    ("final_parameter_hash", "string"),
    ("final_topology_hash", "string"),
    ("accepted_states", "vector<AcceptedStateHashV2>"),
]

EXECUTION_FIELDS = [
    ("magic", "string"), ("performance_mode_requested", "bool"),
    ("effective_reduction_mode", "u8"),
    ("instrumentation_effective", "bool"),
    ("current_linearization_cache_requested", "bool"),
    ("current_linearization_cache_effective", "bool"),
    ("single_stream_synchronous", "bool"),
    ("buffers_reused_across_iterations", "bool"),
    ("topology_reused_across_iterations", "bool"),
    ("cache_access_serialized", "bool"),
    ("cache_context_isolated", "bool"),
    ("initial_resource_health", "u8"), ("final_resource_health", "u8"),
    ("layer_a_calls", "u64"), ("layer_b_calls", "u64"),
    ("layer_c_calls", "u64"), ("cost_calls", "u64"),
    ("build_cuda_layer_a_inputs_count", "u64"),
    ("build_cost_order_count", "u64"), ("topology_epoch", "u64"),
    ("topology_build_count", "u64"), ("topology_refresh_count", "u64"),
    ("topology_cache_lookup_count", "u64"),
    ("topology_cache_hit_count", "u64"),
    ("topology_cache_miss_count", "u64"),
    ("topology_cache_invalidation_count", "u64"),
    ("topology_fingerprint", "string"),
    ("topology_fingerprint_computations", "u64"),
    ("logical_requests", "u64"), ("lookup_aborts", "u64"),
    ("cache_lookups", "u64"), ("linearization_cache_hits", "u64"),
    ("linearization_cache_misses", "u64"), ("build_attempts", "u64"),
    ("build_successes", "u64"), ("build_failures", "u64"),
    ("temporary_builds", "u64"), ("publishes", "u64"),
    ("replacements", "u64"), ("invalidations", "u64"),
    ("borrows", "u64"), ("copies", "u64"), ("teardowns", "u64"),
    ("builds_by_state_epoch", "vector<u64>"),
    ("final_config_generation", "u64"),
    ("final_topology_generation", "u64"),
    ("solve_generation_nonzero_checks", "u64"),
    ("solve_generation_nonzero_violations", "u64"),
    ("solve_generation_nonzero", "bool"),
    ("solve_generation_consistency_checks", "u64"),
    ("solve_generation_consistency_violations", "u64"),
    ("solve_generation_consistent", "bool"),
    ("linearization_identity_checks", "u64"),
    ("linearization_identity_violations", "u64"),
    ("linearization_identity_consistent", "bool"),
    ("topology_context_generation_checks", "u64"),
    ("topology_context_generation_violations", "u64"),
    ("topology_context_generation_consistent", "bool"),
    ("final_linearization_reason", "string"),
    ("buffer_reuse_hits", "u64"), ("topology_reuse_hits", "u64"),
    ("stream_reuse_hits", "u64"), ("event_reuse_hits", "u64"),
    ("solver_handle_reuse_hits", "u64"), ("blas_handle_reuse_hits", "u64"),
    ("audit_capacity_preflight_pass", "bool"),
    ("fault_record_capacity", "u64"),
    ("secondary_diagnostic_capacity", "u64"),
    ("resource_registry_capacity", "u64"),
    ("max_open_timing_interval_capacity", "u64"),
    ("audit_overflow_count", "u64"),
    ("first_audit_overflow_site", "optional<u16>"),
    ("synchronization", "SynchronizationStructureV2"),
    ("state_hash", "StateHashStructureV2"),
    ("fixed_phases", "FixedPhaseStructureV2"),
    ("fault_consumptions", "vector<FaultConsumptionV2>"),
    ("resource", "ResourceStructureV2"),
    ("timing", "TimingStructureV2"),
    ("diagnostics", "DiagnosticStructureV2"),
]

PHASES = [
    "layer_a_call", "layer_b_call", "layer_c_call",
    "build_cuda_layer_a_inputs", "topology_lookup", "topology_build",
    "topology_refresh", "topology_cache", "build_cost_order", "allocation",
    "free", "h2d", "d2h", "d2d", "managed_prefetch", "layer_a_kernel",
    "layer_b_pose_kernel", "layer_b_point_kernel", "layer_b_edge_kernel",
    "layer_b_gradient_kernel", "layer_c_point_factor", "schur", "rhs",
    "cusolver", "back_substitution", "back_substitution_host_self",
    "trial_cost", "apply_step", "diagnostics", "lm_decision",
    "synchronization", "state_hash", "topology_fingerprint_audit",
    "controller_self", "resource_release", "other",
]


class Reader:
    def __init__(self, data):
        self.data = data
        self.offset = 0
        self.offsets = []

    def take(self, size):
        if self.offset + size > len(self.data):
            raise ValueError(f"truncated at {self.offset}, need {size}")
        start = self.offset
        self.offset += size
        return self.data[start:self.offset]

    def primitive(self, kind):
        sizes = {"u8": 1, "bool": 1, "u16": 2, "u32": 4,
                 "i32": 4, "u64": 8, "i64": 8, "f64": 8}
        raw = self.take(sizes[kind])
        if kind == "f64":
            return struct.unpack("<d", raw)[0]
        if kind.startswith("i"):
            return int.from_bytes(raw, "little", signed=True)
        return int.from_bytes(raw, "little", signed=False)

    def string(self):
        length = self.primitive("u64")
        return self.take(length).decode("utf-8")

    def field(self, path, callback):
        start = self.offset
        value = callback()
        self.offsets.append({"path": path, "offset": start,
                             "length": self.offset - start})
        return value

    def finish(self):
        if self.offset != len(self.data):
            raise ValueError(f"trailing bytes: {len(self.data) - self.offset}")


def parse_diag(reader, prefix):
    reader.field(prefix + ".error_classification", lambda: reader.primitive("u8"))
    reader.field(prefix + ".failure_site", lambda: reader.primitive("u16"))
    reader.field(prefix + ".status_domain", lambda: reader.primitive("u8"))
    reader.field(prefix + ".raw_status_code", lambda: reader.primitive("i64"))


def parse_semantic(data):
    r = Reader(data)
    for name, kind in SEMANTIC_FIELDS:
        path = "semantic." + name
        if kind == "string":
            value = r.field(path, r.string)
            if name == "magic" and value != "CudaFullLmSemanticV2":
                raise ValueError(f"wrong semantic magic: {value}")
        elif kind == "vector<SemanticIterationV2>":
            start = r.offset
            count = r.primitive("u64")
            for i in range(count):
                for child, child_kind in SEMANTIC_ITERATION_FIELDS:
                    callback = r.string if child_kind == "string" else (
                        lambda k=child_kind: r.primitive(k))
                    r.field(f"semantic.trace[{i}].{child}", callback)
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start, "count": count})
        elif kind == "vector<AcceptedStateHashV2>":
            start = r.offset
            count = r.primitive("u64")
            for i in range(count):
                r.field(f"semantic.accepted_states[{i}].parameter_hash", r.string)
                r.field(f"semantic.accepted_states[{i}].topology_hash", r.string)
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start, "count": count})
        else:
            r.field(path, lambda k=kind: r.primitive(k))
    r.finish()
    return r.offsets


def parse_execution(data):
    r = Reader(data)
    simple_until = "first_audit_overflow_site"
    for name, kind in EXECUTION_FIELDS:
        path = "execution." + name
        if name == "synchronization":
            start = r.offset
            for child in ["stream_sync_calls", "event_sync_calls",
                          "device_sync_calls", "layer_a_final", "layer_b_final",
                          "layer_c_factor", "layer_c_solver", "layer_c_final",
                          "trial_cost_final", "resource_release"]:
                r.field(path + "." + child, lambda: r.primitive("u64"))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start})
        elif name == "state_hash":
            start = r.offset
            for child in ["layout_builds", "layout_blocks"] + [
                    f"{group}.{field}" for group in ["current", "trial"]
                    for field in ["requests", "computations", "cache_hits",
                                  "blocks_visited", "scalars_hashed"]]:
                r.field(path + "." + child, lambda: r.primitive("u64"))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start})
        elif name == "fixed_phases":
            start = r.offset
            count = r.primitive("u64")
            if count != 36:
                raise ValueError(f"fixed phase count is {count}, expected 36")
            for i in range(count):
                phase_id = r.field(f"{path}[{i}].phase_id",
                                   lambda: r.primitive("u8"))
                if phase_id != i:
                    raise ValueError(f"phase id {phase_id} at index {i}")
                r.field(f"{path}[{i}].calls", lambda: r.primitive("u64"))
                r.field(f"{path}[{i}].bytes", lambda: r.primitive("u64"))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start, "count": count})
        elif name == "fault_consumptions":
            start = r.offset
            count = r.primitive("u64")
            children = [("logical_site", "u8"), ("trigger_phase", "u8"),
                        ("operation", "u16"), ("configured_fault_kind", "u8"),
                        ("epoch", "u64"), ("occurrence", "u64"),
                        ("triggered", "bool")]
            for i in range(count):
                for child, child_kind in children:
                    r.field(f"{path}[{i}].{child}",
                            lambda k=child_kind: r.primitive(k))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start, "count": count})
        elif name == "resource":
            start = r.offset
            for child, child_kind in [
                    ("initial_health", "u8"), ("final_health", "u8"),
                    ("advance_events", "u64"), ("advance_violations", "u64"),
                    ("advanced", "bool"), ("taint_count", "u64"),
                    ("cleanup_attempts", "u64"), ("cleanup_successes", "u64"),
                    ("cleanup_failures", "u64"), ("blocked_lookups", "u64"),
                    ("cross_solve_hits", "u64"), ("quarantine_count", "u64"),
                    ("cleanup_complete", "bool")]:
                r.field(path + "." + child,
                        lambda k=child_kind: r.primitive(k))
            count = r.field(path + ".teardown_record_count",
                            lambda: r.primitive("u64"))
            if count != 7:
                raise ValueError(f"teardown record count is {count}")
            record_fields = [("type", "u8"), ("device", "i32"),
                             ("creation_sequence", "u64"), ("attempted", "bool"),
                             ("success", "bool"), ("status_domain", "u8"),
                             ("raw_status_code", "i64"), ("quarantined", "bool")]
            for i in range(7):
                for child, child_kind in record_fields:
                    r.field(f"{path}.teardown[{i}].{child}",
                            lambda k=child_kind: r.primitive(k))
            present = r.field(path + ".first_cleanup_failure.present",
                              lambda: r.primitive("bool"))
            if present:
                parse_diag(r, path + ".first_cleanup_failure")
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start})
        elif name == "timing":
            start = r.offset
            for child, child_kind in [
                    ("status", "u8"), ("event_attempts", "u64"),
                    ("event_completions", "u64"), ("event_failures", "u64"),
                    ("intervals_started", "u64"),
                    ("intervals_completed", "u64"),
                    ("intervals_abandoned", "u64"),
                    ("open_intervals_at_finalize", "u64")]:
                r.field(path + "." + child,
                        lambda k=child_kind: r.primitive(k))
            present = r.field(path + ".first_incomplete_site.present",
                              lambda: r.primitive("bool"))
            if present:
                r.field(path + ".first_incomplete_site.value",
                        lambda: r.primitive("u16"))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start})
        elif name == "diagnostics":
            start = r.offset
            present = r.field(path + ".has_primary",
                              lambda: r.primitive("bool"))
            if present:
                parse_diag(r, path + ".primary")
            count = r.field(path + ".secondary_count",
                            lambda: r.primitive("u64"))
            for i in range(count):
                parse_diag(r, f"{path}.secondary[{i}]")
            sentinel = r.field(path + ".overflow_sentinel_set",
                               lambda: r.primitive("bool"))
            r.field(path + ".diagnostic_overflow_count",
                    lambda: r.primitive("u64"))
            if sentinel:
                parse_diag(r, path + ".overflow_sentinel")
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start})
        elif kind == "string":
            value = r.field(path, r.string)
            if name == "magic" and value != "CudaFullLmExecutionStructureV2":
                raise ValueError(f"wrong execution magic: {value}")
        elif kind == "vector<u64>":
            start = r.offset
            count = r.primitive("u64")
            for i in range(count):
                r.field(f"{path}[{i}]", lambda: r.primitive("u64"))
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start, "count": count})
        elif kind == "optional<u16>":
            start = r.offset
            present = r.primitive("bool")
            if present:
                r.primitive("u16")
            r.offsets.append({"path": path, "offset": start,
                              "length": r.offset - start,
                              "present": bool(present)})
        elif name == simple_until or kind in {
                "u8", "bool", "u16", "u32", "i32", "u64", "i64", "f64"}:
            r.field(path, lambda k=kind: r.primitive(k))
        else:
            raise ValueError(f"unhandled field {name}: {kind}")
    r.finish()
    diagnostic_count_paths = [entry for entry in r.offsets
                              if entry["path"] ==
                              "execution.diagnostics.diagnostic_overflow_count"]
    if len(diagnostic_count_paths) != 1:
        raise ValueError("diagnostic_overflow_count is not encoded exactly once")
    return r.offsets


def schema_document():
    return {
        "schema": "CudaFullLmCanonicalV2Schema",
        "version": 2,
        "encoding": {
            "byte_order": "little-endian",
            "signed_integer": "two's-complement fixed-width",
            "bool": "one byte, 0 or 1",
            "f64": "IEEE-754 binary64 raw bits",
            "string": "u64 byte length followed by UTF-8 bytes",
            "vector": "u64 count followed by ordered elements",
            "optional": "u8 presence followed by payload when present",
            "hash": "lowercase ASCII hexadecimal",
            "struct_layout": "never serialized",
        },
        "enums": ENUMS,
        "semantic_v2": {
            "magic": "CudaFullLmSemanticV2",
            "fields": [{"ordinal": i + 1, "name": name, "type": kind}
                       for i, (name, kind) in enumerate(SEMANTIC_FIELDS)],
            "SemanticIterationV2": [
                {"ordinal": i + 1, "name": name, "type": kind}
                for i, (name, kind) in enumerate(SEMANTIC_ITERATION_FIELDS)],
            "AcceptedStateHashV2": [
                {"ordinal": 1, "name": "parameter_hash", "type": "string"},
                {"ordinal": 2, "name": "topology_hash", "type": "string"},
            ],
        },
        "execution_v2": {
            "magic": "CudaFullLmExecutionStructureV2",
            "fields": [{"ordinal": i + 1, "name": name, "type": kind}
                       for i, (name, kind) in enumerate(EXECUTION_FIELDS)],
            "fixed_phases": [{"phase_id": i, "name": name}
                             for i, name in enumerate(PHASES)],
            "excluded_from_hash": [
                "topology_generation", "final_solve_generation",
                "resource_generation", "raw_linearization_id",
                "allocator_actual_capacity", "pointer_or_handle_address",
                "timing_duration", "memory_free_peak_active_cached_resident",
                "temperature_power_frequency", "dynamic_cuda_error_text",
            ],
            "diagnostic_overflow_count_encoding_count": 1,
        },
    }


def markdown(schema):
    lines = [
        "# custom_cuda Canonical V2 Schema", "",
        "All integers are fixed-width little-endian. Signed integers use "
        "two's-complement bit patterns, booleans are one byte, and f64 values "
        "are raw IEEE-754 binary64 bits. C++ layout and padding are never used.",
        "", "## Semantic V2", "", "| # | Field | Type |", "|---:|---|---|",
    ]
    for field in schema["semantic_v2"]["fields"]:
        lines.append(f"| {field['ordinal']} | `{field['name']}` | `{field['type']}` |")
    lines += ["", "## Execution V2", "", "| # | Field | Type |",
              "|---:|---|---|"]
    for field in schema["execution_v2"]["fields"]:
        lines.append(f"| {field['ordinal']} | `{field['name']}` | `{field['type']}` |")
    lines += ["", "## Fixed Phases", "", "| ID | Name |", "|---:|---|"]
    for phase in schema["execution_v2"]["fixed_phases"]:
        lines.append(f"| {phase['phase_id']} | `{phase['name']}` |")
    lines += ["", "## Enums", ""]
    for enum_name, values in schema["enums"].items():
        lines.append(f"### {enum_name}")
        lines.append("")
        lines.append("| Name | Value |")
        lines.append("|---|---:|")
        for name, value in values.items():
            lines.append(f"| `{name}` | {value} |")
        lines.append("")
    lines += [
        "## Diagnostic Encoding", "",
        "`diagnostic_overflow_count` is encoded exactly once, inside "
        "`DiagnosticStructureV2`. `audit_overflow_count` is a distinct "
        "Execution audit field.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--schema-json", required=True)
    parser.add_argument("--schema-md", required=True)
    args = parser.parse_args()
    golden = Path(args.golden_dir)
    semantic = (golden / "semantic_v2_canonical.bin").read_bytes()
    execution = (golden / "execution_v2_canonical.bin").read_bytes()
    offsets = {
        "schema": "phase7p1a-r1-canonical-v2-offsets",
        "semantic_bytes": len(semantic),
        "execution_bytes": len(execution),
        "semantic": parse_semantic(semantic),
        "execution": parse_execution(execution),
        "diagnostic_overflow_count_encoding_count": 1,
    }
    (golden / "offsets.json").write_text(
        json.dumps(offsets, indent=2) + "\n", encoding="utf-8")
    schema = schema_document()
    Path(args.schema_json).write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    Path(args.schema_md).write_text(markdown(schema), encoding="utf-8")
    print(json.dumps({"semantic_bytes": len(semantic),
                      "execution_bytes": len(execution),
                      "diagnostic_overflow_count_encoding_count": 1}))


if __name__ == "__main__":
    main()
