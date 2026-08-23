#!/usr/bin/env bash
set -euo pipefail

readonly BUILD=/home/nvidia/colmap-PCD-gpu-build/cuda-release
readonly ARTIFACTS=/home/nvidia/colmap-PCD-gpu-artifacts
readonly EXE=${BUILD}/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay
readonly ROOT=${ARTIFACTS}/oracles/phase7p1a-current-linearization-cache-r1/correctness
readonly SNAP_ROOT=${ARTIFACTS}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots
readonly GLOBAL120=${SNAP_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json
readonly REG2=${ARTIFACTS}/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json

[[ ! -e ${ROOT} ]] || {
  echo "refusing to overwrite ${ROOT}" >&2
  exit 2
}
mkdir -p "${ROOT}/runs" "${ROOT}/pairs"

run_one() {
  local case_name=$1 snapshot=$2 mode=$3 cache=$4
  local run_id=${case_name}-${mode}-cache${cache}
  local run_dir=${ROOT}/runs/${run_id}
  local report=${run_dir}/replay.json
  local payload=${snapshot%.manifest.json}.payload.bin
  local -a command=(env COLMAP_PCD_GPU_BA_INSTRUMENTATION=1
    "COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE=${cache}"
    "${EXE}" "${snapshot}" "${report}" 8 instrumentation_ab)
  [[ ${mode} == parallel ]] && command+=(parallel)
  command+=(forced_reject_probe)
  mkdir -p "${run_dir}"
  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  {
    date -u +%FT%T.%NZ
    uname -a
    nvpmodel -q 2>&1 || true
    jetson_clocks --show 2>&1 || true
  } > "${run_dir}/environment.txt"
  date -u +%FT%T.%NZ > "${run_dir}/started_at.txt"
  set +e
  "${command[@]}" > "${run_dir}/stdout.txt" 2> "${run_dir}/stderr.txt"
  local rc=$?
  set -e
  date -u +%FT%T.%NZ > "${run_dir}/finished_at.txt"
  printf '%s\n' "${rc}" > "${run_dir}/exit-code.txt"
  [[ ${rc} -eq 0 && -s ${report} ]] || return 1
  jq empty "${report}"
  sha256sum "${EXE}" "${snapshot}" "${payload}" "${report}" \
    "${run_dir}/command.txt" "${run_dir}/environment.txt" \
    > "${run_dir}/payloads.sha256"
  jq -n --arg run_id "${run_id}" --arg case_name "${case_name}" \
    --arg mode "${mode}" --argjson cache "${cache}" \
    --arg command "$(tr '\n' ' ' < "${run_dir}/command.txt")" \
    --arg executable_sha256 "$(sha256sum "${EXE}" | awk '{print $1}')" \
    --arg snapshot_sha256 "$(sha256sum "${snapshot}" | awk '{print $1}')" \
    --arg payload_sha256 "$(sha256sum "${payload}" | awk '{print $1}')" \
    --arg output_path "${report}" \
    --arg output_sha256 "$(sha256sum "${report}" | awk '{print $1}')" \
    --arg start "$(cat "${run_dir}/started_at.txt")" \
    --arg finish "$(cat "${run_dir}/finished_at.txt")" \
    '{schema:"phase7p1a-r1-correctness-run-v1",run_id:$run_id,
      case:$case_name,mode:$mode,cache:$cache,instrumentation:1,
      scenario:"forced_reject",command:$command,start_utc:$start,
      finish_utc:$finish,exit_code:0,executable_sha256:$executable_sha256,
      snapshot_manifest_sha256:$snapshot_sha256,
      snapshot_payload_sha256:$payload_sha256,output_path:$output_path,
      output_sha256:$output_sha256}' > "${run_dir}/run-manifest.json"
}

validate_pair() {
  local case_name=$1 mode=$2
  local off=${ROOT}/runs/${case_name}-${mode}-cache0/replay.json
  local on=${ROOT}/runs/${case_name}-${mode}-cache1/replay.json
  jq -n --slurpfile off "${off}" --slurpfile on "${on}" \
    --arg case_name "${case_name}" --arg mode "${mode}" '
    def counts($r;$cache):
      if $cache==0 then
        ($r.lifecycle.layer_a_calls==9 and $r.lifecycle.layer_b_calls==5 and
         $r.lifecycle.layer_c_calls==4 and $r.lifecycle.cost_calls==9 and
         $r.lifecycle.BuildCudaLayerAInputs_count==22 and
         $r.lifecycle.current_linearization.logical_requests==5 and
         $r.lifecycle.current_linearization.cache_hits==0 and
         $r.lifecycle.current_linearization.cache_misses==5 and
         $r.lifecycle.current_linearization.build_attempts==5 and
         $r.lifecycle.current_linearization.build_successes==5 and
         $r.lifecycle.current_linearization.temporary_builds==4 and
         $r.lifecycle.current_linearization.publishes==1)
      else
        ($r.lifecycle.layer_a_calls==5 and $r.lifecycle.layer_b_calls==1 and
         $r.lifecycle.layer_c_calls==4 and $r.lifecycle.cost_calls==5 and
         $r.lifecycle.BuildCudaLayerAInputs_count==14 and
         $r.lifecycle.current_linearization.logical_requests==5 and
         $r.lifecycle.current_linearization.cache_hits==4 and
         $r.lifecycle.current_linearization.cache_misses==1 and
         $r.lifecycle.current_linearization.build_attempts==1 and
         $r.lifecycle.current_linearization.build_successes==1 and
         $r.lifecycle.current_linearization.temporary_builds==0 and
         $r.lifecycle.current_linearization.publishes==1)
      end and $r.lifecycle.final_internal_state_epoch==0 and
      $r.forced_reject_probe.probe_valid==true and
      $r.forced_reject_probe.actual_trials==4 and
      $r.forced_reject_probe.accepted_trials==0 and
      $r.forced_reject_probe.rejected_trials==4;
    {case:$case_name,mode:$mode,
     semantic_v1_pass:($off[0].runtime.semantic_bitwise_sha256_v1==
                       $on[0].runtime.semantic_bitwise_sha256_v1),
     semantic_v2_pass:($off[0].runtime.semantic_sha256_v2==
                       $on[0].runtime.semantic_sha256_v2),
     final_parameters_pass:($off[0].runtime.final_parameters_bitwise_sha256==
                            $on[0].runtime.final_parameters_bitwise_sha256),
     final_topology_pass:($off[0].runtime.final_topology_bitwise_sha256==
                          $on[0].runtime.final_topology_bitwise_sha256),
     fidelity_status_preserved:($off[0].pass==$on[0].pass),
     off_counts_pass:counts($off[0];0),on_counts_pass:counts($on[0];1)} |
    .pass=(.semantic_v1_pass and .semantic_v2_pass and
      .final_parameters_pass and .final_topology_pass and
      .fidelity_status_preserved and .off_counts_pass and .on_counts_pass)' \
    > "${ROOT}/pairs/${case_name}-${mode}.json"
  jq -e .pass "${ROOT}/pairs/${case_name}-${mode}.json" >/dev/null
}

for mode in serial parallel; do
  run_one global120 "${GLOBAL120}" "${mode}" 0
  run_one global120 "${GLOBAL120}" "${mode}" 1
  validate_pair global120 "${mode}"
done
run_one reg2 "${REG2}" serial 0
run_one reg2 "${REG2}" serial 1
validate_pair reg2 serial

jq -s '.' "${ROOT}"/pairs/*.json > "${ROOT}/pairs.json"
jq -n --slurpfile pairs "${ROOT}/pairs.json" \
  '{schema:"phase7p1a-r1-correctness-summary-v1",pairs:$pairs[0],
    pass:($pairs[0]|all(.pass))}' > "${ROOT}/summary.json"
jq -e .pass "${ROOT}/summary.json" >/dev/null
find "${ROOT}" -type f ! -name integrity.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${ROOT}/integrity.sha256"
