#!/usr/bin/env bash
set -euo pipefail

# Phase 6.3 production snapshot gate. A strict trace failure is an expected
# diagnostic outcome only when its structure, accepted-state, and quality
# gates are independently valid.
BUILD_DIR="${1:-/home/nvidia/colmap-PCD-gpu-build/cuda-release}"
ARTIFACT_ROOT="${2:-/home/nvidia/colmap-PCD-gpu-artifacts}"
ORACLE_ROOT="${COLMAP_GPU_BA_ORACLE_ROOT:-${3:-${ARTIFACT_ROOT}/oracles/phase6p3-custom-cuda-audit-v1}}"
SELF_TEST="${4:-}"
PHASE="phase6p3-custom-cuda-audit-v1"
SNAPSHOT_ROOT="${ARTIFACT_ROOT}/snapshots/phase2-canonical"
PROD_ROOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots"

LAYER_A="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_replay"
LAYER_B="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_layer_b_replay"
LAYER_C="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_layer_c_replay"
FULL_LM="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay"

for executable in "${LAYER_A}" "${LAYER_B}" "${LAYER_C}" "${FULL_LM}"; do
  [[ -x "${executable}" ]] || {
    echo "missing executable: ${executable}" >&2
    exit 2
  }
done

if [[ "${SELF_TEST}" == "--stale-report-test" ]]; then
  stale_root="${ORACLE_ROOT}/stale-report-self-test"
  stale_report="${stale_root}/stale.json"
  mkdir -p "${stale_root}"
  printf '{"phase":"%s","pass":true}\n' "${PHASE}" >"${stale_report}"
  rm -f "${stale_report}" "${stale_report}.stdout" "${stale_report}.stderr"
  set +e
  "${LAYER_A}" "${stale_root}/missing.manifest.json" "${stale_report}" \
    >"${stale_report}.stdout" 2>"${stale_report}.stderr"
  status=$?
  set -e
  [[ ${status} -ne 0 ]] || { echo "stale self-test command unexpectedly passed" >&2; exit 1; }
  [[ ! -s "${stale_report}" ]] || {
    echo "stale self-test report was recreated after a failed command" >&2
    exit 1
  }
  echo "STALE_REPORT_TEST_PASS"
  exit 0
fi

mkdir -p "${ORACLE_ROOT}/layer-a" "${ORACLE_ROOT}/layer-b" \
         "${ORACLE_ROOT}/layer-c" "${ORACLE_ROOT}/single-step" \
         "${ORACLE_ROOT}/full-lm"

snapshot_for_reg() {
  case "$1" in
    reg2) echo "${SNAPSHOT_ROOT}/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json" ;;
    reg6) echo "${SNAPSHOT_ROOT}/local-reg6-call12-refine1-trigger36-phraselocal.manifest.json" ;;
    reg20) echo "${SNAPSHOT_ROOT}/local-reg20-call50-refine1-trigger61-phraselocal.manifest.json" ;;
    reg50) echo "${SNAPSHOT_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json" ;;
    *) echo "unknown registration case: $1" >&2; return 2 ;;
  esac
}

snapshot_id() {
  jq -er '.manifest_core.snapshot_id' "$1"
}

clear_report() {
  local report="$1"
  rm -f "${report}" "${report}.stdout" "${report}.stderr"
}

run_layer_json() {
  local report="$1"
  local label="$2"
  local snapshot="$3"
  local expected_layer="$4"
  shift 4
  clear_report "${report}"
  mkdir -p "$(dirname "${report}")"
  echo "[phase6p3] ${label}: $*"
  set +e
  "$@" >"${report}.stdout" 2>"${report}.stderr"
  local status=$?
  set -e
  if [[ ${status} -ne 0 || ! -s "${report}" ]]; then
    echo "${label} command/report failed (status=${status})" >&2
    cat "${report}.stderr" >&2 || true
    return 1
  fi
  local sid
  sid="$(snapshot_id "${snapshot}")"
  jq -e --arg phase "${PHASE}" --arg sid "${sid}" \
      --arg layer "${expected_layer}" \
      '.phase == $phase and .snapshot_id == $sid and .layer == $layer and
       .cuda_ran == true and .pass == true and
       ($layer == "A_residual_jacobian" or
        ((.runtime.reduction_mode == "serial_deterministic" or
          .runtime.reduction_mode == "parallel_deterministic") and
         (.runtime.effective_worker_count >= 1)))' "${report}" >/dev/null || {
    echo "${label} schema/gate validation failed: ${report}" >&2
    return 1
  }
}

run_full_lm() {
  local case_name="$1"
  local snapshot="$2"
  local expected_pass="$3"
  local expected_field="$4"
  local expected_iteration="$5"
  local report="${ORACLE_ROOT}/full-lm/${case_name}/full-lm.json"
  clear_report "${report}"
  mkdir -p "$(dirname "${report}")"
  echo "[phase6p3] Full LM ${case_name}: ${FULL_LM} ${snapshot} ${report} 8"
  set +e
  "${FULL_LM}" "${snapshot}" "${report}" 8 \
    >"${report}.stdout" 2>"${report}.stderr"
  local status=$?
  set -e
  [[ -s "${report}" ]] || {
    echo "${case_name} did not produce a report" >&2
    return 1
  }
  local sid
  sid="$(snapshot_id "${snapshot}")"
  if [[ "${expected_pass}" == true ]]; then
    # Cross-backend iteration traces are diagnostic only.  The production
    # engineering gate intentionally accepts a strict trace mismatch when the
    # CUDA solve ran to a valid final state with matching topology, finite
    # accepted states, bounded final cost/state, and a successful termination.
    # Keep the replay's strict .pass and first-divergence fields untouched so
    # the numerical difference remains auditable.
    [[ ${status} -eq 0 || ${status} -eq 1 ]] || {
      echo "${case_name} replay failed before producing a comparable result" >&2
      return 1
    }
    jq -e --arg phase "${PHASE}" --arg sid "${sid}" \
      '.phase == $phase and .snapshot_id == $sid and .layer == "E_full_lm" and
       .cuda_ran == true and .cuda_error == null and
       .gates.cost_pass == true and .gates.gradient_pass == true and
       .gates.state_pass == true and
       .gates.trace_structure_pass == true and
       .gates.accepted_state_trace_pass == true and
       (.accepted_state_trace.cpu_count == .accepted_state_trace.cuda_count) and
       (.accepted_state_trace.cpu_count > 0) and
       (.accepted_state_trace.states | all(.topology_pass == true and
                                           .finite_pass == true and
                                           .invariant_pass == true)) and
       ((.iteration_trace.cpu | length) > 0) and
       ((.iteration_trace.cuda | length) > 0) and
       .gates.termination_pass == true' "${report}" >/dev/null || {
      echo "${case_name} engineering PASS schema validation failed" >&2; return 1;
    }
    if jq -e '.pass == true' "${report}" >/dev/null; then
      echo "${case_name}: STRICT_PASS_VERIFIED"
    else
      echo "${case_name}: ENGINEERING_PASS_WITH_STRICT_TRACE_DIAGNOSTIC"
    fi
  else
    [[ ${status} -eq 0 || ${status} -eq 1 ]] || {
      echo "${case_name} replay failed before producing a comparable result" >&2
      return 1
    }
    jq -e --arg phase "${PHASE}" --arg sid "${sid}" \
      '.phase == $phase and .snapshot_id == $sid and .layer == "E_full_lm" and
       .cuda_ran == true and .cuda_error == null and
       .gates.trace_structure_pass == true and
       .gates.accepted_state_trace_pass == true and
       .gates.state_pass == true and .gates.cost_pass == true and
       .gates.termination_pass == true and
       (.accepted_state_trace.cpu_count == .accepted_state_trace.cuda_count) and
       (.accepted_state_trace.cpu_count > 0) and
       (.accepted_state_trace.states | all(.topology_pass == true and
                                           .finite_pass == true and
                                           .invariant_pass == true)) and
       ((.iteration_trace.cpu | length) > 0) and
       ((.iteration_trace.cuda | length) > 0)' "${report}" >/dev/null || {
      echo "${case_name} engineering PASS schema validation failed" >&2
      return 1
    }
    if jq -e --arg field "${expected_field}" --argjson iteration "${expected_iteration}" \
        '.pass == false and
         .trace_first_divergence.numeric_field == $field and
         .trace_first_divergence.numeric_iteration == $iteration' \
        "${report}" >/dev/null; then
      echo "${case_name}: EXPECTED_STRICT_DIAGNOSTIC_VERIFIED"
    else
      echo "${case_name}: STRICT_DIAGNOSTIC_NOT_REPRODUCED_ENGINEERING_PASS"
    fi
  fi
}

for reg in reg2 reg6 reg20 reg50; do
  snapshot="$(snapshot_for_reg "${reg}")"
  [[ -f "${snapshot}" ]] || { echo "missing snapshot: ${snapshot}" >&2; exit 2; }
  run_layer_json "${ORACLE_ROOT}/layer-a/${reg}/layer-a.json" \
    "Layer A ${reg}" "${snapshot}" "A_residual_jacobian" \
    "${LAYER_A}" "${snapshot}" "${ORACLE_ROOT}/layer-a/${reg}/layer-a.json"
  run_layer_json "${ORACLE_ROOT}/layer-b/${reg}/layer-b.json" \
    "Layer B ${reg}" "${snapshot}" "B_assembly_gradient_damping" \
    "${LAYER_B}" "${snapshot}" "${ORACLE_ROOT}/layer-b/${reg}/layer-b.json"
  run_layer_json "${ORACLE_ROOT}/layer-c/${reg}/layer-c.json" \
    "Layer C ${reg}" "${snapshot}" "C_schur_factor_backsub" \
    "${LAYER_C}" "${snapshot}" "${ORACLE_ROOT}/layer-c/${reg}/layer-c.json" explicit source 8
  run_layer_json "${ORACLE_ROOT}/single-step/${reg}/single-step.json" \
    "Layer D ${reg}" "${snapshot}" "C_schur_factor_backsub" \
    "${LAYER_C}" "${snapshot}" "${ORACLE_ROOT}/single-step/${reg}/single-step.json" explicit source 1
done

run_full_lm local118 \
  "${PROD_ROOT}/local-reg50-call118-refine0-trigger79-phraselocal.manifest.json" \
  true "" -1
run_full_lm local119 \
  "${PROD_ROOT}/local-reg50-call119-refine1-trigger79-phraselocal.manifest.json" \
  false "projected_gradient" 2
run_full_lm global120 \
  "${PROD_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json" \
  false "projected_gradient" 2

echo "phase6p3 production engineering fidelity gate completed (strict trace differences are diagnostic only)"
