#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-/home/nvidia/colmap-PCD-gpu-build/cuda-release}"
ARTIFACT_ROOT="${2:-/home/nvidia/colmap-PCD-gpu-artifacts}"
ORACLE_ROOT="${COLMAP_GPU_BA_ORACLE_ROOT:-${3:-${ARTIFACT_ROOT}/oracles/phase6p3-custom-cuda-audit-v1}}"
SNAPSHOT_ROOT="${ARTIFACT_ROOT}/snapshots/phase2-canonical"
OUT_ROOT="${ORACLE_ROOT}/parallel-layers"
mkdir -p "${OUT_ROOT}"

LAYER_B="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_layer_b_replay"
LAYER_C="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_layer_c_replay"
for executable in "${LAYER_B}" "${LAYER_C}"; do
  [[ -x "${executable}" ]] || { echo "missing executable: ${executable}" >&2; exit 2; }
done

snapshot_for_reg() {
  case "$1" in
    reg2) echo "${SNAPSHOT_ROOT}/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json" ;;
    reg6) echo "${SNAPSHOT_ROOT}/local-reg6-call12-refine1-trigger36-phraselocal.manifest.json" ;;
    reg20) echo "${SNAPSHOT_ROOT}/local-reg20-call50-refine1-trigger61-phraselocal.manifest.json" ;;
    reg50) echo "${SNAPSHOT_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json" ;;
    *) return 2 ;;
  esac
}

run_checked() {
  local report="$1"; shift
  rm -f "${report}" "${report}.stdout" "${report}.stderr"
  mkdir -p "$(dirname "${report}")"
  "$@" >"${report}.stdout" 2>"${report}.stderr"
  [[ -s "${report}" ]] || { echo "missing report ${report}" >&2; return 1; }
  jq empty "${report}"
  jq -e '.phase == "phase6p3-custom-cuda-audit-v1" and
         .cuda_ran == true and .pass == true' "${report}" >/dev/null
}

for reg in reg2 reg6 reg20 reg50; do
  snapshot="$(snapshot_for_reg "${reg}")"
  sid="$(jq -er '.manifest_core.snapshot_id' "${snapshot}")"
  b_report="${OUT_ROOT}/${reg}/layer-b-parallel.json"
  c_report="${OUT_ROOT}/${reg}/layer-c-parallel.json"
  d_report="${OUT_ROOT}/${reg}/layer-d-parallel.json"
  run_checked "${b_report}" "${LAYER_B}" "${snapshot}" "${b_report}" explicit parallel
  run_checked "${c_report}" "${LAYER_C}" "${snapshot}" "${c_report}" explicit source 8 parallel
  run_checked "${d_report}" "${LAYER_C}" "${snapshot}" "${d_report}" explicit source 1 parallel
  jq -e --arg sid "${sid}" \
    '.phase == "phase6p3-custom-cuda-audit-v1" and .snapshot_id == $sid and
     .layer == "B_assembly_gradient_damping" and .cuda_ran == true and
     .pass == true and
     .runtime.reduction_mode == "parallel_deterministic" and
     .runtime.cost_reduction_parallel == true and
     .runtime.gradient_reduction_parallel == true and
     (.runtime.effective_worker_count >= 2)' "${b_report}" >/dev/null
  jq -e --arg sid "${sid}" \
    '.phase == "phase6p3-custom-cuda-audit-v1" and .snapshot_id == $sid and
     .layer == "C_schur_factor_backsub" and .cuda_ran == true and
     .pass == true and
     .runtime.reduction_mode == "parallel_deterministic" and
     .runtime.cost_reduction_parallel == true and
     .runtime.gradient_reduction_parallel == true and
     (.runtime.effective_worker_count >= 2)' "${c_report}" >/dev/null
  jq -e --arg sid "${sid}" \
    '.phase == "phase6p3-custom-cuda-audit-v1" and .snapshot_id == $sid and
     .layer == "C_schur_factor_backsub" and .cuda_ran == true and
     .pass == true and
     .runtime.reduction_mode == "parallel_deterministic" and
     .runtime.cost_reduction_parallel == true and
     .runtime.gradient_reduction_parallel == true and
     (.runtime.effective_worker_count >= 2)' "${d_report}" >/dev/null
  jq -e '.runtime.reduction_mode == "parallel_deterministic" and
         .runtime.cost_reduction_parallel == true and
         .runtime.gradient_reduction_parallel == true and
         (.runtime.effective_worker_count >= 2)' "${b_report}" >/dev/null
  jq -e '.phase == "phase6p3-custom-cuda-audit-v1" and
         .pass == true and .runtime.reduction_mode == "parallel_deterministic" and
         (.runtime.effective_worker_count >= 2)' "${c_report}" >/dev/null
  jq -e '.phase == "phase6p3-custom-cuda-audit-v1" and
         .pass == true and .runtime.reduction_mode == "parallel_deterministic" and
         (.runtime.effective_worker_count >= 2)' "${d_report}" >/dev/null
done
echo "PARALLEL_LAYER_FIDELITY_PASS (strict-failure semantics remain separate)"
