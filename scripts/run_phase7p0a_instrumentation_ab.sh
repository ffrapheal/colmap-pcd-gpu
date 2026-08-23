#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/nvidia/colmap-PCD-gpu-artifacts}"
EXE="${2:-/home/nvidia/colmap-PCD-gpu-build/cuda-release/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay}"
RUN_ROOT="${3:-${ARTIFACT_ROOT}/oracles/phase7p0a-instrumentation-ab-v1}"
SNAP_ROOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots"
REG2_SNAPSHOT="${ARTIFACT_ROOT}/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json"
GLOBAL120_SNAPSHOT="${SNAP_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json"
HISTORICAL_PARALLEL_MEDIAN_MS="30712.681"
PARALLEL_LIMIT_MS="33783.9491"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite existing Phase 7.0a oracle root: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${EXE}" ]]; then
  echo "missing executable: ${EXE}" >&2
  exit 2
fi
for snapshot in "${REG2_SNAPSHOT}" "${GLOBAL120_SNAPSHOT}"; do
  if [[ ! -f "${snapshot}" ]]; then
    echo "missing snapshot manifest: ${snapshot}" >&2
    exit 2
  fi
  payload="${snapshot%.manifest.json}.payload.bin"
  if [[ ! -f "${payload}" ]]; then
    echo "missing snapshot payload: ${payload}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/runs" "${RUN_ROOT}/pairs"
sha256sum "${EXE}" > "${RUN_ROOT}/binary.sha256"
PAIR_FAILURE=0

capture_environment() {
  local output="$1"
  {
    date -u +%FT%T.%NZ
    uname -a
    printf 'nproc='; nproc || true
    printf 'nvpmodel='; nvpmodel -q 2>&1 || true
    printf 'jetson_clocks='; jetson_clocks --show 2>&1 || true
    printf 'cpu_freq='; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 || true
    printf 'gpu_freq='; cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>&1 || true
    printf 'emc_freq='; cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>&1 || true
    printf 'temperature='; cat /sys/class/thermal/thermal_zone*/temp 2>&1 || true
    printf 'power='; cat /sys/bus/i2c/drivers/ina3221x/*/iio_device/*/in_power*_input 2>&1 || true
    printf 'tegrastats='; timeout 3 tegrastats --interval 500 --count 1 2>&1 || true
  } > "${output}"
}

make_run_id() {
  printf '%s-%s-%s\n' "$(date -u +%Y%m%dT%H%M%S%N)" "$$" "${RANDOM}"
}

LAST_RUN_MANIFEST=""
run_one() {
  local case_name="$1" snapshot="$2" mode="$3" scenario="$4"
  local instrumentation="$5" measured="$6" repeat="$7" pair_id="$8"
  local run_id run_dir report payload started_ns finished_ns rc
  local -a command
  run_id="$(make_run_id)"
  run_dir="${RUN_ROOT}/runs/${run_id}"
  report="${run_dir}/replay.json"
  payload="${snapshot%.manifest.json}.payload.bin"
  mkdir -p "${run_dir}"
  command=(env "COLMAP_PCD_GPU_BA_INSTRUMENTATION=${instrumentation}"
           "${EXE}" "${snapshot}" "${report}" 8 instrumentation_ab)
  if [[ "${mode}" == parallel ]]; then command+=(parallel); fi
  if [[ "${scenario}" == forced_reject ]]; then
    command+=(forced_reject_probe)
  fi
  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  jq -n --arg case_name "${case_name}" --arg mode "${mode}" \
        --arg scenario "${scenario}" --arg instrumentation "${instrumentation}" \
        --arg measured "${measured}" --argjson repeat "${repeat}" \
        --arg pair_id "${pair_id}" \
        '{case:$case_name,mode:$mode,scenario:$scenario,
          instrumentation:$instrumentation,measured:($measured=="true"),
          repeat:$repeat,pair_id:(if $pair_id=="" then null else $pair_id end),
          capture_state_trace:false,cost_reduction_threads:8,
          cuda_memory_mode:"explicit_device_copy",
          max_solver_time_in_seconds:1000000000}' > "${run_dir}/config.json"
  capture_environment "${run_dir}/environment.before.txt"
  sha256sum "${run_dir}/environment.before.txt" > \
    "${run_dir}/environment.before.sha256"
  sha256sum "${EXE}" > "${run_dir}/binary.sha256"
  sha256sum "${snapshot}" > "${run_dir}/snapshot-manifest.sha256"
  sha256sum "${payload}" > "${run_dir}/snapshot-payload.sha256"
  date -u +%FT%T.%NZ > "${run_dir}/started_at.txt"
  started_ns="$(date +%s%N)"
  set +e
  "${command[@]}" > "${run_dir}/stdout.txt" 2> "${run_dir}/stderr.txt"
  rc=$?
  set -e
  finished_ns="$(date +%s%N)"
  date -u +%FT%T.%NZ > "${run_dir}/finished_at.txt"
  capture_environment "${run_dir}/environment.after.txt"
  sha256sum "${run_dir}/environment.after.txt" > \
    "${run_dir}/environment.after.sha256"
  printf '%s\n' "${rc}" > "${run_dir}/exit_code.txt"
  printf '%s\n' "$(( (finished_ns - started_ns) / 1000000 ))" > \
    "${run_dir}/process_wall_milliseconds.txt"
  if [[ "${rc}" -ne 0 || ! -s "${report}" ]]; then
    echo "run failed: ${run_dir} rc=${rc}" >&2
    return 1
  fi
  jq empty "${report}"
  sha256sum "${report}" > "${run_dir}/output.sha256"
  sha256sum "${run_dir}/stdout.txt" "${run_dir}/stderr.txt" \
    "${run_dir}/command.txt" "${run_dir}/config.json" > \
    "${run_dir}/supporting-files.sha256"
  jq -n \
    --arg run_id "${run_id}" --arg run_dir "${run_dir}" \
    --arg case_name "${case_name}" --arg mode "${mode}" \
    --arg scenario "${scenario}" --arg instrumentation "${instrumentation}" \
    --arg measured "${measured}" --argjson repeat "${repeat}" \
    --arg pair_id "${pair_id}" --arg snapshot "${snapshot}" \
    --arg payload "${payload}" --arg report "${report}" \
    --arg command "$(tr '\n' ' ' < "${run_dir}/command.txt")" \
    --arg started "$(cat "${run_dir}/started_at.txt")" \
    --arg finished "$(cat "${run_dir}/finished_at.txt")" \
    --arg binary_sha "$(awk '{print $1}' "${run_dir}/binary.sha256")" \
    --arg manifest_sha "$(awk '{print $1}' "${run_dir}/snapshot-manifest.sha256")" \
    --arg payload_sha "$(awk '{print $1}' "${run_dir}/snapshot-payload.sha256")" \
    --arg output_sha "$(awk '{print $1}' "${run_dir}/output.sha256")" \
    --arg env_before_sha "$(awk '{print $1}' "${run_dir}/environment.before.sha256")" \
    --arg env_after_sha "$(awk '{print $1}' "${run_dir}/environment.after.sha256")" \
    --argjson process_wall "$(cat "${run_dir}/process_wall_milliseconds.txt")" \
    --argjson rc "${rc}" \
    '{run_id:$run_id,run_dir:$run_dir,case:$case_name,mode:$mode,
      scenario:$scenario,instrumentation:$instrumentation,
      measured:($measured=="true"),repeat:$repeat,
      pair_id:(if $pair_id=="" then null else $pair_id end),
      command:$command,start_utc:$started,finish_utc:$finished,
      process_exit_code:$rc,process_wall_milliseconds:$process_wall,
      executable_sha256:$binary_sha,snapshot_manifest_path:$snapshot,
      snapshot_manifest_sha256:$manifest_sha,snapshot_payload_path:$payload,
      snapshot_payload_sha256:$payload_sha,output_path:$report,
      output_sha256:$output_sha,environment_before_sha256:$env_before_sha,
      environment_after_sha256:$env_after_sha}' > "${run_dir}/run-manifest.json"
  sha256sum "${run_dir}/run-manifest.json" > \
    "${run_dir}/run-manifest.sha256"
  LAST_RUN_MANIFEST="${run_dir}/run-manifest.json"
}

validate_pair() {
  local pair_id="$1" off_manifest="$2" on_manifest="$3" scenario="$4"
  local pair_path off_report on_report first_difference
  pair_path="${RUN_ROOT}/pairs/${pair_id}.json"
  off_report="$(jq -r .output_path "${off_manifest}")"
  on_report="$(jq -r .output_path "${on_manifest}")"
  first_difference=""
  if [[ "$(jq -r .runtime.decision_bitwise_sha256 "${off_report}")" != \
        "$(jq -r .runtime.decision_bitwise_sha256 "${on_report}")" ]]; then
    first_difference="decision_bitwise_sha256"
  elif [[ "$(jq -r .runtime.final_parameters_bitwise_sha256 "${off_report}")" != \
          "$(jq -r .runtime.final_parameters_bitwise_sha256 "${on_report}")" ]]; then
    first_difference="final_parameters_bitwise_sha256"
  elif [[ "$(jq -r .runtime.final_topology_bitwise_sha256 "${off_report}")" != \
          "$(jq -r .runtime.final_topology_bitwise_sha256 "${on_report}")" ]]; then
    first_difference="final_topology_bitwise_sha256"
  fi
  jq -n --slurpfile off_manifest "${off_manifest}" \
        --slurpfile on_manifest "${on_manifest}" \
        --slurpfile off "${off_report}" --slurpfile on "${on_report}" \
        --arg pair_id "${pair_id}" --arg scenario "${scenario}" \
        --arg first_difference "${first_difference}" \
    '{pair_id:$pair_id,scenario:$scenario,
      off_run_id:$off_manifest[0].run_id,on_run_id:$on_manifest[0].run_id,
      same_binary:($off_manifest[0].executable_sha256==$on_manifest[0].executable_sha256),
      same_manifest:($off_manifest[0].snapshot_manifest_sha256==$on_manifest[0].snapshot_manifest_sha256),
      same_payload:($off_manifest[0].snapshot_payload_sha256==$on_manifest[0].snapshot_payload_sha256),
      same_config:(($off[0].snapshot_id==$on[0].snapshot_id) and
        ($off[0].performance_mode==$on[0].performance_mode) and
        ($off[0].capture_state_trace==$on[0].capture_state_trace) and
        ($off[0].cuda_memory_mode==$on[0].cuda_memory_mode) and
        ($off[0].cost_reduction_threads==$on[0].cost_reduction_threads) and
        ($off[0].max_solver_time_in_seconds==$on[0].max_solver_time_in_seconds)),
      decision_bitwise_pass:($off[0].runtime.decision_bitwise_sha256==$on[0].runtime.decision_bitwise_sha256),
      parameters_bitwise_pass:($off[0].runtime.final_parameters_bitwise_sha256==$on[0].runtime.final_parameters_bitwise_sha256),
      topology_bitwise_pass:($off[0].runtime.final_topology_bitwise_sha256==$on[0].runtime.final_topology_bitwise_sha256),
      first_difference:(if $first_difference=="" then null else $first_difference end),
      off_runtime_ms:$off[0].runtime.host_wall_milliseconds,
      on_runtime_ms:$on[0].runtime.host_wall_milliseconds,
      runtime_ratio:($on[0].runtime.host_wall_milliseconds/$off[0].runtime.host_wall_milliseconds),
      off_solve_call_ms:$off[0].runtime.cuda_solve_call_wall_milliseconds,
      on_solve_call_ms:$on[0].runtime.cuda_solve_call_wall_milliseconds,
      solve_call_ratio:($on[0].runtime.cuda_solve_call_wall_milliseconds/$off[0].runtime.cuda_solve_call_wall_milliseconds),
      off_probe_valid:$off[0].forced_reject_probe.probe_valid,
      on_probe_valid:$on[0].forced_reject_probe.probe_valid,
      expected_global120_strict_failure:(($off[0].pass|not) and ($on[0].pass|not)),
      pass:(($off_manifest[0].executable_sha256==$on_manifest[0].executable_sha256) and
        ($off_manifest[0].snapshot_manifest_sha256==$on_manifest[0].snapshot_manifest_sha256) and
        ($off_manifest[0].snapshot_payload_sha256==$on_manifest[0].snapshot_payload_sha256) and
        ($off[0].snapshot_id==$on[0].snapshot_id) and
        ($off[0].performance_mode==$on[0].performance_mode) and
        ($off[0].capture_state_trace==$on[0].capture_state_trace) and
        ($off[0].cuda_memory_mode==$on[0].cuda_memory_mode) and
        ($off[0].cost_reduction_threads==$on[0].cost_reduction_threads) and
        ($off[0].max_solver_time_in_seconds==$on[0].max_solver_time_in_seconds) and
        ($off[0].instrumentation.requested=="0") and
        ($off[0].instrumentation.effective|not) and
        ($on[0].instrumentation.requested=="1") and
        $on[0].instrumentation.effective and
        ($off[0].lifecycle.state_hash_audit.layout_builds==0) and
        ($off[0].lifecycle.topology_fingerprint_computations==0) and
        ($on[0].lifecycle.topology_fingerprint_computations==1) and
        ($off[0].runtime.decision_bitwise_sha256==$on[0].runtime.decision_bitwise_sha256) and
        ($off[0].runtime.final_parameters_bitwise_sha256==$on[0].runtime.final_parameters_bitwise_sha256) and
        ($off[0].runtime.final_topology_bitwise_sha256==$on[0].runtime.final_topology_bitwise_sha256) and
        ($off[0].capture_state_trace==$on[0].capture_state_trace) and
        (if $scenario=="forced_reject" then
           ($off[0].forced_reject_probe.probe_valid and $on[0].forced_reject_probe.probe_valid)
         else true end))}' > "${pair_path}"
  if ! jq -e '.pass' "${pair_path}" >/dev/null; then
    PAIR_FAILURE=1
  fi
}

run_pair() {
  local case_name="$1" snapshot="$2" mode="$3" scenario="$4"
  local measured="$5" repeat="$6" order="$7" pair_id off_manifest on_manifest
  pair_id="${case_name}-${mode}-${scenario}-${measured}-r${repeat}"
  if [[ "${order}" == off_on ]]; then
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" 0 \
      "${measured}" "${repeat}" "${pair_id}"
    off_manifest="${LAST_RUN_MANIFEST}"
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" 1 \
      "${measured}" "${repeat}" "${pair_id}"
    on_manifest="${LAST_RUN_MANIFEST}"
  else
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" 1 \
      "${measured}" "${repeat}" "${pair_id}"
    on_manifest="${LAST_RUN_MANIFEST}"
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" 0 \
      "${measured}" "${repeat}" "${pair_id}"
    off_manifest="${LAST_RUN_MANIFEST}"
  fi
  validate_pair "${pair_id}" "${off_manifest}" "${on_manifest}" "${scenario}"
}

for mode in serial parallel; do
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" normal false 0 off_on
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" normal true 1 off_on
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" normal true 2 on_off
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" normal true 3 off_on
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" forced_reject true 1 off_on
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" forced_reject true 2 on_off
  run_pair global120 "${GLOBAL120_SNAPSHOT}" "${mode}" forced_reject true 3 off_on
  run_pair reg2 "${REG2_SNAPSHOT}" "${mode}" normal false 0 off_on
done

find "${RUN_ROOT}/runs" -mindepth 2 -name run-manifest.json -print0 | \
  sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/runs.json"
records_ndjson="${RUN_ROOT}/run-records.ndjson"
: > "${records_ndjson}"
while IFS= read -r manifest; do
  run_dir="$(dirname "${manifest}")"
  jq -c -s '.[0] + {config:.[1],result:.[2]}' \
    "${manifest}" "${run_dir}/config.json" "${run_dir}/replay.json" >> \
    "${records_ndjson}"
done < <(find "${RUN_ROOT}/runs" -mindepth 2 -name run-manifest.json | sort)
jq -s '.' "${records_ndjson}" > "${RUN_ROOT}/run-records.json"
rm -f "${records_ndjson}"
find "${RUN_ROOT}/pairs" -mindepth 1 -name '*.json' -print0 | \
  sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/pairs.json"

jq -s --arg historical "${HISTORICAL_PARALLEL_MEDIAN_MS}" \
      --arg limit "${PARALLEL_LIMIT_MS}" '
  def median: sort | if length%2==1 then .[length/2|floor]
    else (.[length/2-1]+.[length/2])/2 end;
  .[0] as $runs | .[1] as $pairs |
  ($pairs | map(select(.pair_id|test("global120-parallel-normal-true")))) as $parallel |
  ($pairs | map(select(.pair_id|test("global120-serial-normal-true")))) as $serial |
  {schema:"phase7p0a-instrumentation-ab-summary-v1",
   historical_parallel_median_ms:($historical|tonumber),
   parallel_limit_ms:($limit|tonumber),runs:$runs,pairs:$pairs,
   statistics:{
     parallel_off:{min:([$parallel[].off_runtime_ms]|min),
       median:([$parallel[].off_runtime_ms]|median),max:([$parallel[].off_runtime_ms]|max)},
     parallel_on:{min:([$parallel[].on_runtime_ms]|min),
       median:([$parallel[].on_runtime_ms]|median),max:([$parallel[].on_runtime_ms]|max)},
     parallel_runtime_ratio:{min:([$parallel[].runtime_ratio]|min),
       median:([$parallel[].runtime_ratio]|median),max:([$parallel[].runtime_ratio]|max)},
     serial_runtime_ratio:{min:([$serial[].runtime_ratio]|min),
       median:([$serial[].runtime_ratio]|median),max:([$serial[].runtime_ratio]|max)}},
   gates:{all_pairs_bitwise:($pairs|all(.pass)),
     all_pairs_same_config:($pairs|all(.same_config)),
     all_global120_normal_expected_strict_failure:
       ($pairs|map(select(.pair_id|startswith("global120-") and contains("-normal-")))|
        all(.expected_global120_strict_failure)),
     parallel_off_non_regression:(([$parallel[].off_runtime_ms]|median)<=($limit|tonumber)),
     parallel_instrumentation_overhead:(([$parallel[].runtime_ratio]|median)<=1.05),
     serial_instrumentation_overhead:(([$serial[].runtime_ratio]|median)<=1.05)}}' \
  "${RUN_ROOT}/run-records.json" "${RUN_ROOT}/pairs.json" > \
  "${RUN_ROOT}/summary.json"

gate_rc=0
if ! jq -e '.gates.all_pairs_bitwise and .gates.all_pairs_same_config and
            .gates.all_global120_normal_expected_strict_failure and
            .gates.parallel_off_non_regression and
            .gates.parallel_instrumentation_overhead and
            .gates.serial_instrumentation_overhead' \
     "${RUN_ROOT}/summary.json" >/dev/null; then
  gate_rc=1
fi
if [[ "${PAIR_FAILURE}" -ne 0 ]]; then gate_rc=1; fi
find "${RUN_ROOT}" -type f ! -name integrity.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RUN_ROOT}/integrity.sha256"
echo "Phase 7.0a A/B artifacts written to ${RUN_ROOT}"
exit "${gate_rc}"
