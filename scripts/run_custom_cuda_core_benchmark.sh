#!/usr/bin/env bash
set -euo pipefail

readonly PHASE="phase6p3-custom-cuda-audit-v1"
BUILD_DIR="${1:-/home/nvidia/colmap-PCD-gpu-build/cuda-release}"
ARTIFACT_ROOT="${2:-/home/nvidia/colmap-PCD-gpu-artifacts}"
OUT_ROOT="${3:-${ARTIFACT_ROOT}/oracles/${PHASE}/performance}"
SNAPSHOT_ROOT="${ARTIFACT_ROOT}/snapshots/phase2-canonical"
PRODUCTION_SNAPSHOT_ROOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots"
FULL_LM="${BUILD_DIR}/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay"
WARMUP_RUNS=1
MEASURED_REPEATS=3

[[ -x "${FULL_LM}" ]] || { echo "missing ${FULL_LM}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}"

readonly BENCHMARK_START="$(date -Iseconds)"
readonly RUN_ID="$(date +%Y%m%dT%H%M%N)-$$-${RANDOM}"
readonly RUN_ROOT="${OUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_ROOT}/warmup" "${RUN_ROOT}/measured" "${RUN_ROOT}/meta"
env | sort > "${RUN_ROOT}/meta/benchmark-environment.txt"
printf '%q ' "$0" "$@" > "${RUN_ROOT}/meta/benchmark-command.txt"
printf '\n' >> "${RUN_ROOT}/meta/benchmark-command.txt"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

snapshot_for_case() {
  case "$1" in
    local118) echo "${PRODUCTION_SNAPSHOT_ROOT}/local-reg50-call118-refine0-trigger79-phraselocal.manifest.json" ;;
    local119) echo "${PRODUCTION_SNAPSHOT_ROOT}/local-reg50-call119-refine1-trigger79-phraselocal.manifest.json" ;;
    global120) echo "${PRODUCTION_SNAPSHOT_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json" ;;
    reg2) echo "${SNAPSHOT_ROOT}/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json" ;;
    *) return 2 ;;
  esac
}

expected_exit_for_case() {
  case "$1" in
    local118) echo 0 ;;
    reg2|local119|global120) echo 1 ;;
    *) return 2 ;;
  esac
}

validate_full_lm_report() {
  local report="$1" case_name="$2" expected_snapshot_id="$3" role="$4"
  jq -e --arg phase "${PHASE}" --arg expected "${expected_snapshot_id}" \
    '.phase == $phase and .snapshot_id == $expected and
     .layer == "E_full_lm" and .cuda_ran == true and
     .reference_backend == "custom_cpu" and .candidate_backend == "custom_cuda" and
     (.cpu.wall_milliseconds | type == "number" and isfinite) and
     (.runtime.host_wall_milliseconds | type == "number" and isfinite) and
     (.iteration_trace.cpu | type == "array" and length > 0) and
     (.iteration_trace.cuda | type == "array" and length > 0) and
     (.accepted_state_trace.structure_pass == true)' "${report}" >/dev/null
  if [[ "${role}" == measured ]]; then
    jq -e '.accepted_state_trace.values_pass == true' "${report}" >/dev/null
  fi
  if [[ "${case_name}" == local118 ]]; then
    jq -e '.pass == true and .trace_first_divergence.numeric_iteration == -1' \
      "${report}" >/dev/null
  else
    jq -e '.pass == false and
           (.trace_first_divergence.numeric_iteration >= 0) and
           (.trace_first_divergence.numeric_field != null)' "${report}" >/dev/null
  fi
}

run_one() {
  local case_name="$1" mode="$2" role="$3" repeat="$4" snapshot="$5"
  local run_dir="${RUN_ROOT}/${role}/${case_name}-${mode}-r${repeat}"
  local report="${run_dir}/full-lm.json"
  local stdout_path="${run_dir}/stdout.log"
  local stderr_path="${run_dir}/stderr.log"
  local command_path="${run_dir}/command.txt"
  local environment_path="${run_dir}/environment.txt"
  local run_manifest="${run_dir}/run-manifest.json"
  mkdir -p "${run_dir}"
  rm -f "${report}" "${stdout_path}" "${stderr_path}" "${run_manifest}"
  local snapshot_hash executable_hash expected_snapshot_id payload payload_hash
  snapshot_hash="$(sha256_file "${snapshot}")"
  payload="${snapshot%.manifest.json}.payload.bin"
  [[ -s "${payload}" ]] || { echo "missing snapshot payload ${payload}" >&2; return 1; }
  payload_hash="$(sha256_file "${payload}")"
  [[ "${payload_hash}" == "$(jq -r '.payload_sha256' "${snapshot}")" ]] || {
    echo "snapshot payload hash mismatch: ${snapshot}" >&2; return 1;
  }
  executable_hash="$(sha256_file "${FULL_LM}")"
  expected_snapshot_id="$(jq -r '.manifest_core.snapshot_id' "${snapshot}")"
  env | sort > "${environment_path}"
  local mode_args=()
  if [[ "${mode}" == parallel ]]; then mode_args=(parallel); fi
  printf '%q ' "${FULL_LM}" "${snapshot}" "${report}" 8 "${mode_args[@]}" \
    > "${command_path}"
  printf '\n' >> "${command_path}"
  local started_at ended_at exit_code
  started_at="$(date -Iseconds)"
  set +e
  "${FULL_LM}" "${snapshot}" "${report}" 8 "${mode_args[@]}" \
    >"${stdout_path}" 2>"${stderr_path}"
  exit_code=$?
  set -e
  ended_at="$(date -Iseconds)"
  [[ -s "${report}" ]] || {
    echo "missing fresh report: ${report}" >&2
    return 1
  }
  jq empty "${report}"
  validate_full_lm_report "${report}" "${case_name}" "${expected_snapshot_id}" "${role}"
  local expected_exit
  expected_exit="$(expected_exit_for_case "${case_name}")"
  [[ "${exit_code}" == "${expected_exit}" ]] || {
    echo "unexpected exit ${exit_code} for ${case_name}/${mode}; expected ${expected_exit}" >&2
    return 1
  }
  local report_hash_before report_hash_after
  report_hash_before="$(sha256_file "${report}")"
  jq empty "${report}"
  report_hash_after="$(sha256_file "${report}")"
  [[ "${report_hash_before}" == "${report_hash_after}" ]] || {
    echo "report changed during validation: ${report}" >&2
    return 1
  }
  local environment_hash command_hash stdout_hash stderr_hash
  environment_hash="$(sha256_file "${environment_path}")"
  command_hash="$(sha256_file "${command_path}")"
  stdout_hash="$(sha256_file "${stdout_path}")"
  stderr_hash="$(sha256_file "${stderr_path}")"
  jq -n --arg phase "${PHASE}" --arg run_id "${RUN_ID}" \
    --arg case_name "${case_name}" --arg mode "${mode}" --arg role "${role}" \
    --argjson repeat "${repeat}" --arg started_at "${started_at}" \
    --arg ended_at "${ended_at}" --arg snapshot "${snapshot}" \
    --arg snapshot_hash "${snapshot_hash}" --arg payload "${payload}" \
    --arg payload_hash "${payload_hash}" --arg executable "${FULL_LM}" \
    --arg executable_hash "${executable_hash}" --arg report "${report}" \
    --arg report_hash "${report_hash_after}" --arg stdout "${stdout_path}" \
    --arg stdout_hash "${stdout_hash}" --arg stderr "${stderr_path}" \
    --arg stderr_hash "${stderr_hash}" --arg command "${command_path}" \
    --arg command_hash "${command_hash}" --arg environment "${environment_path}" \
    --arg environment_hash "${environment_hash}" --argjson exit_code "${exit_code}" \
    --slurpfile output "${report}" \
    '{phase:$phase,run_id:$run_id,case:$case_name,mode:$mode,role:$role,repeat:$repeat,
      started_at:$started_at,ended_at:$ended_at,exit_code:$exit_code,
      command_file:$command,command_sha256:$command_hash,
      environment_file:$environment,environment_sha256:$environment_hash,
      executable:{path:$executable,sha256:$executable_hash},
      input_snapshot:{manifest_path:$snapshot,manifest_sha256:$snapshot_hash,
                      payload_path:$payload,payload_sha256:$payload_hash},
      output:{json:$report,sha256:$report_hash,hash_before_validation:$report_hash,
              hash_after_validation:$report_hash,hash_match:true},
      logs:{stdout:$stdout,stdout_sha256:$stdout_hash,stderr:$stderr,stderr_sha256:$stderr_hash},
      timings:{cpu_wall_milliseconds:$output[0].cpu.wall_milliseconds,
               cuda_host_wall_milliseconds:$output[0].runtime.host_wall_milliseconds,
               layer_a_kernel_milliseconds:$output[0].runtime.layer_a_kernel_milliseconds,
               layer_b_kernel_milliseconds:$output[0].runtime.layer_b_kernel_milliseconds,
               point_kernel_milliseconds:$output[0].runtime.point_kernel_milliseconds,
               schur_kernel_milliseconds:$output[0].runtime.schur_kernel_milliseconds,
               factorization_milliseconds:$output[0].runtime.factorization_milliseconds,
               back_substitution_milliseconds:$output[0].runtime.back_substitution_milliseconds,
               trial_cost_kernel_milliseconds:$output[0].runtime.trial_cost_kernel_milliseconds,
               cost_reduction_kernel_milliseconds:$output[0].runtime.cost_reduction_kernel_milliseconds,
               gradient_reduction_kernel_milliseconds:$output[0].runtime.gradient_reduction_kernel_milliseconds,
               allocation_milliseconds:$output[0].runtime.allocation_milliseconds,
               copy_milliseconds:$output[0].runtime.copy_milliseconds,
               synchronization_milliseconds:$output[0].runtime.synchronization_milliseconds,
               peak_resident_bytes:$output[0].runtime.peak_resident_bytes,
               peak_predicted_bytes:$output[0].runtime.peak_predicted_bytes,
               buffers_reused:$output[0].runtime.buffers_reused_across_iterations,
               topology_reused:$output[0].runtime.topology_reused_across_iterations},
      report_snapshot_id:$output[0].snapshot_id,
      report_phase:$output[0].phase,
      report_pass:$output[0].pass,
      accepted_state_count_cpu:$output[0].accepted_state_trace.cpu_count,
      accepted_state_count_cuda:$output[0].accepted_state_trace.cuda_count}' \
    > "${run_manifest}"
  jq empty "${run_manifest}"
  echo "FRESH_RUN ${role} ${case_name} ${mode} ${repeat} ${report}"
}

warmup_snapshot="$(snapshot_for_case reg2)"
run_one reg2 serial warmup 0 "${warmup_snapshot}"
run_one reg2 parallel warmup 0 "${warmup_snapshot}"

cases=(local118 local119 global120)
modes=(serial parallel)
for case_name in "${cases[@]}"; do
  snapshot="$(snapshot_for_case "${case_name}")"
  [[ -s "${snapshot}" ]] || { echo "missing snapshot ${snapshot}" >&2; exit 1; }
  for mode in "${modes[@]}"; do
    for ((repeat = 1; repeat <= MEASURED_REPEATS; ++repeat)); do
      run_one "${case_name}" "${mode}" measured "${repeat}" "${snapshot}"
    done
  done
done

mapfile -t run_manifests < <(find "${RUN_ROOT}" -type f -name run-manifest.json | sort)
[[ "${#run_manifests[@]}" -eq $((2 + 6 * MEASURED_REPEATS)) ]] || {
  echo "unexpected run manifest count: ${#run_manifests[@]}" >&2; exit 1;
}

csv="${OUT_ROOT}/core-times-${RUN_ID}.csv"
summary_json="${OUT_ROOT}/core-times-${RUN_ID}.json"
benchmark_manifest="${OUT_ROOT}/benchmark-manifest-${RUN_ID}.json"
printf 'case,backend,mode,role,repeat,cpu_wall_milliseconds,cuda_host_wall_milliseconds,layer_a_kernel_milliseconds,layer_b_kernel_milliseconds,point_kernel_milliseconds,schur_kernel_milliseconds,factorization_milliseconds,back_substitution_milliseconds,trial_cost_kernel_milliseconds,cost_reduction_kernel_milliseconds,gradient_reduction_kernel_milliseconds,reported_kernel_components_sum_milliseconds,allocation_milliseconds,copy_milliseconds,synchronization_milliseconds,peak_resident_bytes,peak_predicted_bytes,buffers_reused,topology_reused,fmad,run_manifest,output_json,output_sha256\n' > "${csv}"

for manifest in "${run_manifests[@]}"; do
  jq -e '.role == "measured"' "${manifest}" >/dev/null || continue
  jq -r --arg manifest "${manifest}" \
    '[.case,"custom_cpu",.mode,.role,.repeat,.timings.cpu_wall_milliseconds,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.timings.peak_resident_bytes,.timings.peak_predicted_bytes,.timings.buffers_reused,.timings.topology_reused,"--fmad=false",$manifest,.output.json,.output.sha256] | @csv' \
    "${manifest}" >> "${csv}"
  jq -r --arg manifest "${manifest}" \
    '[.case,"custom_cuda",.mode,.role,.repeat,.timings.cpu_wall_milliseconds,.timings.cuda_host_wall_milliseconds,.timings.layer_a_kernel_milliseconds,.timings.layer_b_kernel_milliseconds,.timings.point_kernel_milliseconds,.timings.schur_kernel_milliseconds,.timings.factorization_milliseconds,.timings.back_substitution_milliseconds,.timings.trial_cost_kernel_milliseconds,.timings.cost_reduction_kernel_milliseconds,.timings.gradient_reduction_kernel_milliseconds,(.timings.layer_a_kernel_milliseconds+.timings.layer_b_kernel_milliseconds+.timings.point_kernel_milliseconds+.timings.schur_kernel_milliseconds+.timings.factorization_milliseconds+.timings.back_substitution_milliseconds+.timings.trial_cost_kernel_milliseconds+.timings.cost_reduction_kernel_milliseconds+.timings.gradient_reduction_kernel_milliseconds),.timings.allocation_milliseconds,.timings.copy_milliseconds,.timings.synchronization_milliseconds,.timings.peak_resident_bytes,.timings.peak_predicted_bytes,.timings.buffers_reused,.timings.topology_reused,"--fmad=false",$manifest,.output.json,.output.sha256] | @csv' \
    "${manifest}" >> "${csv}"
done

mapfile -t measured_manifests < <(for manifest in "${run_manifests[@]}"; do jq -e '.role == "measured"' "${manifest}" >/dev/null && echo "${manifest}"; done)
jq -s --arg phase "${PHASE}" --arg run_id "${RUN_ID}" \
  --arg csv "${csv}" --arg summary "${summary_json}" --arg manifest "${benchmark_manifest}" \
  --argjson warmups "${WARMUP_RUNS}" --argjson repeats "${MEASURED_REPEATS}" \
  'def median: sort | if length % 2 == 1 then .[length/2|floor] else ((.[length/2-1|floor] + .[length/2|floor]) / 2) end;
   . as $runs |
   ["local118","local119","global120"] as $cases |
   ["serial","parallel"] as $modes |
   [ $cases[] as $case | $modes[] as $mode |
     ([$runs[] | select(.case==$case and .mode==$mode)] ) as $selected |
     {case:$case,mode:$mode,repeats:($selected|length),
      custom_cpu:{min:([$selected[].timings.cpu_wall_milliseconds]|min),median:([$selected[].timings.cpu_wall_milliseconds]|median),max:([$selected[].timings.cpu_wall_milliseconds]|max)},
      custom_cuda:{min:([$selected[].timings.cuda_host_wall_milliseconds]|min),median:([$selected[].timings.cuda_host_wall_milliseconds]|median),max:([$selected[].timings.cuda_host_wall_milliseconds]|max)},
      run_manifests:[$selected[].output.json]} ] as $aggregates |
   {phase:$phase,run_id:$run_id,warmup_runs:$warmups,measured_repeats:$repeats,
    fmad:"--fmad=false",external_wall_time_primary:false,acceleration_claim:false,
    timing_source:{custom_cpu:"cpu.wall_milliseconds",custom_cuda:"runtime.host_wall_milliseconds"},
    kernel_time_semantics:"reported_components_sum_only; not a total kernel time",
    csv:$csv,summary_json:$summary,benchmark_manifest:$manifest,aggregates:$aggregates}' \
  "${measured_manifests[@]}" > "${summary_json}"

while IFS= read -r manifest; do
  output_path="$(jq -r '.output.json' "${manifest}")"
  current_hash="$(sha256_file "${output_path}")"
  expected_hash="$(jq -r '.output.sha256' "${manifest}")"
  [[ "${current_hash}" == "${expected_hash}" ]] || {
    echo "stale output hash in ${manifest}" >&2; exit 1;
  }
done < <(printf '%s\n' "${run_manifests[@]}")

csv_hash="$(sha256_file "${csv}")"
summary_hash="$(sha256_file "${summary_json}")"
benchmark_start="${BENCHMARK_START}"
benchmark_end="$(date -Iseconds)"
jq -s --arg phase "${PHASE}" --arg run_id "${RUN_ID}" \
  --arg started_at "${benchmark_start}" --arg ended_at "${benchmark_end}" \
  --arg run_root "${RUN_ROOT}" --arg csv "${csv}" --arg csv_hash "${csv_hash}" \
  --arg summary "${summary_json}" --arg summary_hash "${summary_hash}" \
  --arg command_file "${RUN_ROOT}/meta/benchmark-command.txt" \
  --arg command_hash "$(sha256_file "${RUN_ROOT}/meta/benchmark-command.txt")" \
  --arg environment_file "${RUN_ROOT}/meta/benchmark-environment.txt" \
  --arg environment_hash "$(sha256_file "${RUN_ROOT}/meta/benchmark-environment.txt")" \
  --argjson warmups "${WARMUP_RUNS}" --argjson repeats "${MEASURED_REPEATS}" \
  '. as $runs | {phase:$phase,run_id:$run_id,started_at:$started_at,ended_at:$ended_at,
    warmup_runs:$warmups,measured_repeats:$repeats,run_root:$run_root,
    command_file:$command_file,command_sha256:$command_hash,
    environment_file:$environment_file,environment_sha256:$environment_hash,
    executable:{path:$runs[0].executable.path,sha256:$runs[0].executable.sha256},
    runs:$runs,csv:{path:$csv,sha256:$csv_hash},summary:{path:$summary,sha256:$summary_hash},
    stale_artifact_check:{all_output_hashes_verified:true,rerun_directory_unique:true}}' \
  "${run_manifests[@]}" > "${benchmark_manifest}"
jq empty "${summary_json}" "${benchmark_manifest}"
echo "FRESH_BENCHMARK_RUN_ROOT=${RUN_ROOT}"
echo "FRESH_BENCHMARK_SUMMARY=${summary_json}"
echo "FRESH_BENCHMARK_CSV=${csv}"
echo "FRESH_BENCHMARK_MANIFEST=${benchmark_manifest}"
