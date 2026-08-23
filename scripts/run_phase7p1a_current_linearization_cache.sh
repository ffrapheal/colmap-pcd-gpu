#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/nvidia/colmap-PCD-gpu-artifacts}"
EXE="${2:-/home/nvidia/colmap-PCD-gpu-build/cuda-release/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay}"
RUN_ROOT="${3:-${ARTIFACT_ROOT}/oracles/phase7p1a-current-linearization-cache-v1}"
SNAP_ROOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots"
REG2="${ARTIFACT_ROOT}/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json"
LOCAL118="${SNAP_ROOT}/local-reg50-call118-refine0-trigger79-phraselocal.manifest.json"
LOCAL119="${SNAP_ROOT}/local-reg50-call119-refine1-trigger79-phraselocal.manifest.json"
GLOBAL120="${SNAP_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite Phase 7.1a oracle root: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${EXE}" ]]; then
  echo "missing executable: ${EXE}" >&2
  exit 2
fi
for snapshot in "${REG2}" "${LOCAL118}" "${LOCAL119}" "${GLOBAL120}"; do
  [[ -f "${snapshot}" ]] || { echo "missing snapshot: ${snapshot}" >&2; exit 2; }
  [[ -f "${snapshot%.manifest.json}.payload.bin" ]] || {
    echo "missing payload for ${snapshot}" >&2; exit 2;
  }
done

mkdir -p "${RUN_ROOT}/runs" "${RUN_ROOT}/pairs" "${RUN_ROOT}/fidelity"
sha256sum "${EXE}" > "${RUN_ROOT}/binary.sha256"

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
    printf 'nsys_used=false\n'
    printf 'ncu_used=false\n'
  } > "${output}"
}

make_run_id() {
  printf '%s-%s-%s\n' "$(date -u +%Y%m%dT%H%M%S%N)" "$$" "${RANDOM}"
}

LAST_MANIFEST=""
run_one() {
  local case_name="$1" snapshot="$2" mode="$3" scenario="$4"
  local instrumentation="$5" cache="$6" measured="$7" repeat="$8"
  local pair_id="$9"
  local run_id run_dir report payload started_ns finished_ns rc
  local -a command
  run_id="$(make_run_id)"
  run_dir="${RUN_ROOT}/runs/${run_id}"
  report="${run_dir}/replay.json"
  payload="${snapshot%.manifest.json}.payload.bin"
  mkdir -p "${run_dir}"
  command=(env "COLMAP_PCD_GPU_BA_INSTRUMENTATION=${instrumentation}"
           "COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE=${cache}"
           "${EXE}" "${snapshot}" "${report}" 8 instrumentation_ab)
  [[ "${mode}" == parallel ]] && command+=(parallel)
  [[ "${scenario}" == forced_reject ]] && command+=(forced_reject_probe)
  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  jq -n --arg case_name "${case_name}" --arg mode "${mode}" \
    --arg scenario "${scenario}" --arg instrumentation "${instrumentation}" \
    --arg cache "${cache}" --arg measured "${measured}" \
    --argjson repeat "${repeat}" --arg pair_id "${pair_id}" \
    '{case:$case_name,mode:$mode,scenario:$scenario,
      instrumentation:($instrumentation|tonumber),cache:($cache|tonumber),
      measured:($measured=="true"),repeat:$repeat,
      pair_id:(if $pair_id=="" then null else $pair_id end),
      capture_state_trace:false,cost_reduction_threads:8,
      cuda_memory_mode:"explicit_device_copy",max_solver_time_in_seconds:1e9}' \
    > "${run_dir}/config.json"
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
  jq -n --arg run_id "${run_id}" --arg run_dir "${run_dir}" \
    --arg case_name "${case_name}" --arg mode "${mode}" \
    --arg scenario "${scenario}" --arg instrumentation "${instrumentation}" \
    --arg cache "${cache}" --arg measured "${measured}" \
    --argjson repeat "${repeat}" --arg pair_id "${pair_id}" \
    --arg snapshot "${snapshot}" --arg payload "${payload}" \
    --arg report "${report}" \
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
      scenario:$scenario,instrumentation:($instrumentation|tonumber),
      cache:($cache|tonumber),measured:($measured=="true"),repeat:$repeat,
      pair_id:(if $pair_id=="" then null else $pair_id end),command:$command,
      start_utc:$started,finish_utc:$finished,process_exit_code:$rc,
      process_wall_milliseconds:$process_wall,executable_sha256:$binary_sha,
      snapshot_manifest_path:$snapshot,snapshot_manifest_sha256:$manifest_sha,
      snapshot_payload_path:$payload,snapshot_payload_sha256:$payload_sha,
      output_path:$report,output_sha256:$output_sha,
      environment_before_sha256:$env_before_sha,
      environment_after_sha256:$env_after_sha}' > "${run_dir}/run-manifest.json"
  sha256sum "${run_dir}/run-manifest.json" > \
    "${run_dir}/run-manifest.sha256"
  LAST_MANIFEST="${run_dir}/run-manifest.json"
}

validate_cache_pair() {
  local pair_id="$1" off_manifest="$2" on_manifest="$3" scenario="$4"
  local off_report on_report output
  off_report="$(jq -r .output_path "${off_manifest}")"
  on_report="$(jq -r .output_path "${on_manifest}")"
  output="${RUN_ROOT}/pairs/${pair_id}.json"
  jq -n --slurpfile om "${off_manifest}" --slurpfile nm "${on_manifest}" \
    --slurpfile off "${off_report}" --slurpfile on "${on_report}" \
    --arg pair_id "${pair_id}" --arg scenario "${scenario}" '
    def normal_counts($r;$cache):
      if $cache==0 then
        ($r.lifecycle.layer_a_calls==31 and $r.lifecycle.layer_b_calls==21 and
         $r.lifecycle.layer_c_calls==10 and $r.lifecycle.cost_calls==31 and
         $r.lifecycle.BuildCudaLayerAInputs_count==72 and
         $r.lifecycle.final_internal_state_epoch==10 and
         $r.lifecycle.current_linearization.build_attempts==21 and
         $r.lifecycle.current_linearization.temporary_builds==10 and
         $r.lifecycle.current_linearization.cache_hits==0)
      else
        ($r.lifecycle.layer_a_calls==21 and $r.lifecycle.layer_b_calls==11 and
         $r.lifecycle.layer_c_calls==10 and $r.lifecycle.cost_calls==21 and
         $r.lifecycle.BuildCudaLayerAInputs_count==52 and
         $r.lifecycle.final_internal_state_epoch==10 and
         $r.lifecycle.current_linearization.build_attempts==11 and
         $r.lifecycle.current_linearization.temporary_builds==0 and
         $r.lifecycle.current_linearization.cache_hits==10)
      end;
    def reject_counts($r;$cache):
      if $cache==0 then
        ($r.lifecycle.layer_a_calls==9 and $r.lifecycle.layer_b_calls==5 and
         $r.lifecycle.layer_c_calls==4 and $r.lifecycle.cost_calls==9 and
         $r.lifecycle.BuildCudaLayerAInputs_count==22 and
         $r.lifecycle.current_linearization.build_attempts==5 and
         $r.lifecycle.current_linearization.temporary_builds==4 and
         $r.lifecycle.current_linearization.cache_hits==0)
      else
        ($r.lifecycle.layer_a_calls==5 and $r.lifecycle.layer_b_calls==1 and
         $r.lifecycle.layer_c_calls==4 and $r.lifecycle.cost_calls==5 and
         $r.lifecycle.BuildCudaLayerAInputs_count==14 and
         $r.lifecycle.current_linearization.build_attempts==1 and
         $r.lifecycle.current_linearization.temporary_builds==0 and
         $r.lifecycle.current_linearization.cache_hits==4)
      end and ($r.lifecycle.final_internal_state_epoch==0) and
        ($r.forced_reject_probe.probe_valid==true);
    ($off[0].runtime.host_wall_milliseconds) as $off_ms |
    ($on[0].runtime.host_wall_milliseconds) as $on_ms |
    {pair_id:$pair_id,scenario:$scenario,
     off_run_id:$om[0].run_id,on_run_id:$nm[0].run_id,
     same_binary:($om[0].executable_sha256==$nm[0].executable_sha256),
     same_snapshot:($om[0].snapshot_manifest_sha256==$nm[0].snapshot_manifest_sha256 and
                    $om[0].snapshot_payload_sha256==$nm[0].snapshot_payload_sha256),
     semantic_v1_pass:($off[0].runtime.semantic_bitwise_sha256_v1==
                       $on[0].runtime.semantic_bitwise_sha256_v1),
     final_parameters_pass:($off[0].runtime.final_parameters_bitwise_sha256==
                            $on[0].runtime.final_parameters_bitwise_sha256),
     final_topology_pass:($off[0].runtime.final_topology_bitwise_sha256==
                          $on[0].runtime.final_topology_bitwise_sha256),
     fidelity_status_preserved:($off[0].pass==$on[0].pass),
     off_counts_pass:(if $scenario=="forced_reject" then reject_counts($off[0];0)
                      else normal_counts($off[0];0) end),
     on_counts_pass:(if $scenario=="forced_reject" then reject_counts($on[0];1)
                     else normal_counts($on[0];1) end),
     off_runtime_ms:$off_ms,on_runtime_ms:$on_ms,
     ratio:($on_ms/$off_ms),log_ratio:(($on_ms/$off_ms)|log)} |
    .pass=(.same_binary and .same_snapshot and .semantic_v1_pass and
      .final_parameters_pass and .final_topology_pass and
      .fidelity_status_preserved and .off_counts_pass and .on_counts_pass)' \
    > "${output}"
  jq -e .pass "${output}" >/dev/null
}

run_cache_pair() {
  local case_name="$1" snapshot="$2" mode="$3" scenario="$4"
  local instrumentation="$5" measured="$6" repeat="$7" order="$8"
  local pair_id off_manifest on_manifest
  pair_id="${case_name}-${mode}-${scenario}-i${instrumentation}-${measured}-r${repeat}"
  if [[ "${order}" == off_on ]]; then
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" \
      "${instrumentation}" 0 "${measured}" "${repeat}" "${pair_id}"
    off_manifest="${LAST_MANIFEST}"
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" \
      "${instrumentation}" 1 "${measured}" "${repeat}" "${pair_id}"
    on_manifest="${LAST_MANIFEST}"
  else
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" \
      "${instrumentation}" 1 "${measured}" "${repeat}" "${pair_id}"
    on_manifest="${LAST_MANIFEST}"
    run_one "${case_name}" "${snapshot}" "${mode}" "${scenario}" \
      "${instrumentation}" 0 "${measured}" "${repeat}" "${pair_id}"
    off_manifest="${LAST_MANIFEST}"
  fi
  validate_cache_pair "${pair_id}" "${off_manifest}" "${on_manifest}" \
    "${scenario}"
}

for mode in serial parallel; do
  run_cache_pair global120 "${GLOBAL120}" "${mode}" normal 0 false 0 off_on
  for repeat in 1 2 3 4 5; do
    if (( repeat % 2 == 1 )); then order=off_on; else order=on_off; fi
    run_cache_pair global120 "${GLOBAL120}" "${mode}" normal 0 true \
      "${repeat}" "${order}"
  done
  for repeat in 1 2 3; do
    if (( repeat % 2 == 1 )); then order=off_on; else order=on_off; fi
    run_cache_pair global120 "${GLOBAL120}" "${mode}" forced_reject 1 true \
      "${repeat}" "${order}"
  done
done

# Instrumentation-on normal pair supplies the exact lifecycle/packing audit.
run_cache_pair global120 "${GLOBAL120}" serial normal 1 false 0 off_on
# Fast forced-reject sanity uses an independent canonical snapshot.
run_cache_pair reg2 "${REG2}" serial forced_reject 1 false 0 off_on

for entry in "local118:${LOCAL118}:pass" "local119:${LOCAL119}:fail"; do
  IFS=: read -r name snapshot expected <<< "${entry}"
  run_one "${name}" "${snapshot}" parallel normal 1 1 false 0 ""
  cp "${LAST_MANIFEST}" "${RUN_ROOT}/fidelity/${name}.run-manifest.json"
  report="$(jq -r .output_path "${LAST_MANIFEST}")"
  actual="$(jq -r .pass "${report}")"
  if [[ "${expected}" == pass && "${actual}" != true ]]; then exit 1; fi
  if [[ "${expected}" == fail && "${actual}" != false ]]; then exit 1; fi
done

find "${RUN_ROOT}/runs" -mindepth 2 -name run-manifest.json -print0 | \
  sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/runs.json"
find "${RUN_ROOT}/pairs" -mindepth 1 -name '*.json' -print0 | \
  sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/pairs.json"

jq '
  def median: sort | if length % 2 == 1 then .[length/2|floor]
    else (.[length/2-1] + .[length/2]) / 2 end;
  def mean: add / length;
  def sample_stddev:
    . as $v | ($v|mean) as $m |
    if ($v|length) < 2 then 0
    else (([$v[] | (. - $m) * (. - $m)] | add) / (($v|length)-1) | sqrt)
    end;
  def stats:
    . as $v | ($v|mean) as $mean | ($v|sample_stddev) as $sd |
    {min:($v|min),median:($v|median),max:($v|max),mean:$mean,
     sample_stddev:$sd,cv:($sd/$mean)};
  def mode_summary($pairs):
    ([$pairs[].off_runtime_ms]) as $off |
    ([$pairs[].on_runtime_ms]) as $on |
    ([$pairs[].ratio]) as $ratios |
    ([$pairs[].log_ratio]) as $logs |
    ($logs|median) as $median_log |
    ([$logs[] | (. - $median_log | fabs)]|median) as $mad |
    ($off|stats) as $off_stats | ($on|stats) as $on_stats |
    (($off_stats.cv * $off_stats.cv + 1)|log|sqrt) as $noise_cv_off |
    (($on_stats.cv * $on_stats.cv + 1)|log|sqrt) as $noise_cv_on |
    (1.4826 * $mad) as $noise_mad |
    ([$noise_cv_off,$noise_cv_on,$noise_mad]|max) as $noise |
    (-$median_log) as $signal |
    ([$ratios[] | select(. < 1)]|length) as $faster |
    {pairs:[$pairs[]|{pair_id,off_runtime_ms,on_runtime_ms,ratio,log_ratio,
      semantic_v1_pass,final_parameters_pass,final_topology_pass,
      off_counts_pass,on_counts_pass,pass}],
     off:$off_stats,on:$on_stats,ratios:$ratios,log_ratios:$logs,
     median_log_ratio:$median_log,mad_log_ratio:$mad,
     noise_cv_off:$noise_cv_off,noise_cv_on:$noise_cv_on,
     noise_mad:$noise_mad,noise:$noise,signal:$signal,
     faster_pair_count:$faster,
     correctness_pass:($pairs|all(.pass)),
     semantic_v1_pass:($pairs|all(.semantic_v1_pass)),
     final_state_pass:($pairs|all(.final_parameters_pass and .final_topology_pass)),
     execution_counts_pass:($pairs|all(.off_counts_pass and .on_counts_pass))} |
    .acceleration_pass=(.signal > .noise and .faster_pair_count >= 4 and
      .correctness_pass and .semantic_v1_pass and .final_state_pass and
      .execution_counts_pass);
  . as $all |
  ($all | map(select(.pair_id|test("^global120-serial-normal-i0-true")))) as $serial |
  ($all | map(select(.pair_id|test("^global120-parallel-normal-i0-true")))) as $parallel |
  {schema:"phase7p1a-cache-benchmark-summary-v1",measured_pairs_per_mode:5,
   formula:{ratio:"cache_on_i/cache_off_i",log_ratio:"ln(ratio)",
    cv:"sample_stddev/mean",noise_cv:"sqrt(ln(1+CV^2))",
    noise_mad:"1.4826*MAD(log_ratio)",signal:"-median(log_ratio)",
    noise:"max(noise_cv_off,noise_cv_on,noise_mad)",
    pass:"signal>noise and faster_pair_count>=4 and all correctness/semantic/final/count gates"},
   serial:mode_summary($serial),parallel:mode_summary($parallel)} |
  .overall_acceleration_pass=(.serial.acceleration_pass and
                              .parallel.acceleration_pass)' \
  "${RUN_ROOT}/pairs.json" > "${RUN_ROOT}/benchmark-summary.json"

jq -n --slurpfile runs "${RUN_ROOT}/runs.json" \
  --slurpfile pairs "${RUN_ROOT}/pairs.json" \
  --slurpfile benchmark "${RUN_ROOT}/benchmark-summary.json" \
  '{schema:"phase7p1a-current-linearization-cache-oracle-v1",
    runs:$runs[0],pairs:$pairs[0],benchmark:$benchmark[0]}' \
  > "${RUN_ROOT}/summary.json"

find "${RUN_ROOT}" -type f ! -name integrity.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RUN_ROOT}/integrity.sha256"
jq -e '.serial.correctness_pass and .parallel.correctness_pass and
       .serial.semantic_v1_pass and .parallel.semantic_v1_pass and
       .serial.final_state_pass and .parallel.final_state_pass and
       .serial.execution_counts_pass and .parallel.execution_counts_pass' \
  "${RUN_ROOT}/benchmark-summary.json" >/dev/null
echo "Phase 7.1a artifacts written under ${RUN_ROOT}"
