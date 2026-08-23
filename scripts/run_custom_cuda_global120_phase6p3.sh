#!/usr/bin/env bash
set -euo pipefail

readonly PHASE="phase6p3-custom-cuda-audit-v1"
BUILD_DIR="${1:-/home/nvidia/colmap-PCD-gpu-build/cuda-release}"
ARTIFACT_ROOT="${2:-/home/nvidia/colmap-PCD-gpu-artifacts}"
OUT_ROOT="${3:-${ARTIFACT_ROOT}/oracles/${PHASE}/diagnostics}"
SNAPSHOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json"
FULL_LM="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay"
DIAGNOSTIC="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_global120_diagnostic"
LAMBDA="3.7037037037037037e-6"
RUN_ID="$(date +%Y%m%dT%H%M%S%N)-$$-${RANDOM}"
RUN_ROOT="${OUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_ROOT}/capture" "${RUN_ROOT}/common_fixed_state" "${RUN_ROOT}/trajectory_separate_accepted_states" "${RUN_ROOT}/meta"
[[ -x "${FULL_LM}" && -x "${DIAGNOSTIC}" && -s "${SNAPSHOT}" ]] || {
  echo "missing phase6p3 diagnostic input or executable" >&2; exit 2;
}
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
env | sort > "${RUN_ROOT}/meta/environment.txt"
printf '%q ' "$0" "$@" > "${RUN_ROOT}/meta/command.txt"
printf '\n' >> "${RUN_ROOT}/meta/command.txt"

capture_report="${RUN_ROOT}/capture/full-lm.json"
capture_cmd="${RUN_ROOT}/capture/command.txt"
printf '%q ' "${FULL_LM}" "${SNAPSHOT}" "${capture_report}" 8 capture > "${capture_cmd}"
printf '\n' >> "${capture_cmd}"
set +e
"${FULL_LM}" "${SNAPSHOT}" "${capture_report}" 8 capture   >"${RUN_ROOT}/capture/stdout.log" 2>"${RUN_ROOT}/capture/stderr.log"
capture_status=$?
set -e
[[ ${capture_status} -eq 1 && -s "${capture_report}" ]] || {
  echo "global120 capture did not return expected strict-fail status" >&2; exit 1;
}
expected_snapshot_id="$(jq -er '.manifest_core.snapshot_id' "${SNAPSHOT}")"
jq -e --arg phase "${PHASE}" --arg sid "${expected_snapshot_id}" \
  '.phase==$phase and .snapshot_id==$sid and .layer=="E_full_lm" and
   .cuda_ran==true and .accepted_state_trace.cpu_count > 4 and
   .accepted_state_trace.cuda_count > 4' "${capture_report}" >/dev/null
capture_state_root="${RUN_ROOT}/capture/debug-states"
cpu_state="$(find "${capture_state_root}/cpu" -type f -name '*accepted4-cpu.manifest.json' | sort | head -n1)"
cuda_state="$(find "${capture_state_root}/cuda" -type f -name '*accepted4-cuda.manifest.json' | sort | head -n1)"
[[ -s "${cpu_state}" && -s "${cuda_state}" ]] || {
  echo "accepted4 CPU/CUDA state was not captured by current phase6p3 binary" >&2; exit 1;
}

run_diag() {
  local mode="$1" lhs="$2" rhs="$3" dir="$4"
  local report="${dir}/diagnostic.json"
  mkdir -p "${dir}"
  printf '%q ' "${DIAGNOSTIC}" "${lhs}" "${rhs}" "${LAMBDA}" "${report}" 8 "${mode}"     > "${dir}/command.txt"
  printf '\n' >> "${dir}/command.txt"
  set +e
  "${DIAGNOSTIC}" "${lhs}" "${rhs}" "${LAMBDA}" "${report}" 8 "${mode}"     >"${dir}/stdout.log" 2>"${dir}/stderr.log"
  local status=$?
  set -e
  [[ ${status} -eq 0 && -s "${report}" ]] || {
    echo "diagnostic failed for ${mode}: status=${status}" >&2; return 1;
  }
  jq -e --arg phase "${PHASE}" --arg mode "${mode}"     '.phase==$phase and .state_mode==$mode and
     (.structural_pass|type=="boolean") and
     (.finite_pass|type=="boolean") and
     (.numeric_pass|type=="boolean") and
     .attribution=="UNRESOLVED"' "${report}" >/dev/null
  echo "${report}"
}

common_report="$(run_diag common_fixed_state "${cpu_state}" "${cpu_state}" "${RUN_ROOT}/common_fixed_state")"
trajectory_report="$(run_diag trajectory_separate_accepted_states "${cpu_state}" "${cuda_state}" "${RUN_ROOT}/trajectory_separate_accepted_states")"

common_hash="$(sha256_file "${common_report}")"
trajectory_hash="$(sha256_file "${trajectory_report}")"
capture_hash="$(sha256_file "${capture_report}")"
cpu_hash="$(sha256_file "${cpu_state}")"
cuda_hash="$(sha256_file "${cuda_state}")"
jq empty "${common_report}" "${trajectory_report}" "${capture_report}"
jq -n --arg phase "${PHASE}" --arg run_id "${RUN_ID}" \
  --arg run_root "${RUN_ROOT}" --arg snapshot "${SNAPSHOT}" \
  --arg cpu_state "${cpu_state}" --arg cuda_state "${cuda_state}" \
  --arg cpu_hash "${cpu_hash}" --arg cuda_hash "${cuda_hash}" \
  --arg capture "${capture_report}" --arg capture_hash "${capture_hash}" \
  --arg capture_cmd "${capture_cmd}" \
  --arg capture_stdout "${RUN_ROOT}/capture/stdout.log" \
  --arg capture_stderr "${RUN_ROOT}/capture/stderr.log" \
  --arg common "${common_report}" --arg common_hash "${common_hash}" \
  --arg trajectory "${trajectory_report}" --arg trajectory_hash "${trajectory_hash}" \
  --arg lambda "${LAMBDA}" --arg full_lm "${FULL_LM}" \
  --arg diagnostic "${DIAGNOSTIC}" --slurpfile c "${common_report}" \
  --slurpfile t "${trajectory_report}" \
  '{phase:$phase,run_id:$run_id,run_root:$run_root,attribution:"UNRESOLVED",
    capture:{command_file:$capture_cmd,executable:$full_lm,snapshot:$snapshot,
             report:$capture,report_sha256:$capture_hash,stdout:$capture_stdout,
             stderr:$capture_stderr},
    accepted4:{cpu_state:$cpu_state,cuda_state:$cuda_state,
               cpu_state_sha256:$cpu_hash,cuda_state_sha256:$cuda_hash},
    lambda:($lambda|tonumber),diagnostic_executable:$diagnostic,
    common_fixed_state:{report:$common,report_sha256:$common_hash,
                        structural_pass:$c[0].structural_pass,
                        finite_pass:$c[0].finite_pass,
                        numeric_pass:$c[0].numeric_pass,
                        pass:$c[0].pass},
    trajectory_separate_accepted_states:{report:$trajectory,report_sha256:$trajectory_hash,
                        structural_pass:$t[0].structural_pass,
                        finite_pass:$t[0].finite_pass,
                        numeric_pass:$t[0].numeric_pass,
                        pass:$t[0].pass},
    statement:"Common mode uses current CPU accepted4 for both arithmetic paths; trajectory mode uses current CPU accepted4 for CPU and current CUDA accepted4 for CUDA. Attribution remains UNRESOLVED."}' \
  > "${OUT_ROOT}/global120-layered-${RUN_ID}.json"
jq empty "${OUT_ROOT}/global120-layered-${RUN_ID}.json"
echo "GLOBAL120_DIAGNOSTIC_RUN_ROOT=${RUN_ROOT}"
echo "GLOBAL120_DIAGNOSTIC_JSON=${OUT_ROOT}/global120-layered-${RUN_ID}.json"
