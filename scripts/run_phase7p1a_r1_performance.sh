#!/usr/bin/env bash
set -euo pipefail

REPO=/home/nvidia/colmap-PCD-gpu
BUILD=/home/nvidia/colmap-PCD-gpu-build/cuda-release
ARTIFACTS=/home/nvidia/colmap-PCD-gpu-artifacts
RUN_ROOT=${ARTIFACTS}/oracles/phase7p1a-current-linearization-cache-r1/performance
EXE=${BUILD}/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay
V1_ROOT=${ARTIFACTS}/oracles/phase7p1a-current-linearization-cache-v1
FROZEN_ROOT=${ARTIFACTS}/oracles/phase7p1a-current-linearization-cache-r1/baseline-v1
FROZEN_EXE=${FROZEN_ROOT}/bin/gpu_ba_custom_cuda_full_lm_replay
FROZEN_WRAPPER=${FROZEN_ROOT}/run-frozen.sh
GLOBAL120=${ARTIFACTS}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json
PAYLOAD=${GLOBAL120%.manifest.json}.payload.bin
EXPECTED_V1_SHA=a4f6bbb7cb30847b559c8c8efff648f174b879240d1b791cc9af7361b3f508e6
BOUNDARY_ID=cuda_solve_call_wall_milliseconds.caller_inclusive.v1
BOUNDARY_TEXT='caller interval around RunCustomCudaSolve including all RAII teardown'

[[ ! -e ${RUN_ROOT} ]] || {
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 2
}
for path in "${EXE}" "${FROZEN_EXE}" "${FROZEN_WRAPPER}" \
            "${GLOBAL120}" "${PAYLOAD}" "${V1_ROOT}/pairs.json"; do
  [[ -e ${path} ]] || { echo "missing ${path}" >&2; exit 2; }
done
[[ $(sha256sum "${FROZEN_EXE}" | awk '{print $1}') == ${EXPECTED_V1_SHA} ]] || {
  echo "frozen V1 executable hash mismatch" >&2
  exit 2
}

mkdir -p "${RUN_ROOT}/runs" "${RUN_ROOT}/pairs" \
         "${RUN_ROOT}/historical-v1" "${RUN_ROOT}/absolute-pairs" \
         "${RUN_ROOT}/same-batch"

cat > "${RUN_ROOT}/measurement-order.json" <<'JSON'
{
  "schema":"phase7p1a-r1-prefrozen-order-v1",
  "measured_pairs_per_mode":6,
  "serial":["AB","BA","AB","BA","AB","BA"],
  "parallel":["AB","BA","AB","BA","AB","BA"],
  "A":"cache_off",
  "B":"cache_on"
}
JSON
sha256sum "${RUN_ROOT}/measurement-order.json" > \
  "${RUN_ROOT}/measurement-order.sha256"

capture_environment() {
  local output=$1
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

LAST_MANIFEST=
run_one() {
  local binary_kind=$1 mode=$2 cache=$3 measured=$4 repeat=$5
  local pair_id=$6 role=$7 scenario=${8:-normal}
  local run_id run_dir report started_ns finished_ns rc exe_sha output_sha
  local -a command
  run_id=$(make_run_id)
  run_dir=${RUN_ROOT}/runs/${run_id}
  report=${run_dir}/replay.json
  mkdir -p "${run_dir}"
  if [[ ${binary_kind} == r1 ]]; then
    command=(env COLMAP_PCD_GPU_BA_INSTRUMENTATION=0
      "COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE=${cache}"
      "${EXE}" "${GLOBAL120}" "${report}" 8 instrumentation_ab)
  else
    command=(env COLMAP_PCD_GPU_BA_INSTRUMENTATION=0
      "COLMAP_PCD_GPU_BA_CURRENT_LINEARIZATION_CACHE=${cache}"
      "${FROZEN_WRAPPER}" "${FROZEN_EXE}" "${GLOBAL120}" "${report}" 8
      instrumentation_ab)
  fi
  [[ ${mode} == parallel ]] && command+=(parallel)
  [[ ${scenario} == forced_reject ]] && command+=(forced_reject_probe)
  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  jq -n --arg binary_kind "${binary_kind}" --arg mode "${mode}" \
    --argjson cache "${cache}" --argjson measured "${measured}" \
    --argjson repeat "${repeat}" --arg pair_id "${pair_id}" \
    --arg role "${role}" --arg scenario "${scenario}" \
    '{case:"global120",binary_kind:$binary_kind,mode:$mode,scenario:$scenario,
      instrumentation:0,cache:$cache,measured:$measured,repeat:$repeat,
      pair_id:(if $pair_id=="" then null else $pair_id end),role:$role,
      capture_state_trace:false,cost_reduction_threads:8}' \
    > "${run_dir}/config.json"
  capture_environment "${run_dir}/environment.before.txt"
  date -u +%FT%T.%NZ > "${run_dir}/started_at.txt"
  started_ns=$(date +%s%N)
  set +e
  "${command[@]}" > "${run_dir}/stdout.txt" 2> "${run_dir}/stderr.txt"
  rc=$?
  set -e
  finished_ns=$(date +%s%N)
  date -u +%FT%T.%NZ > "${run_dir}/finished_at.txt"
  capture_environment "${run_dir}/environment.after.txt"
  printf '%s\n' "${rc}" > "${run_dir}/exit-code.txt"
  printf '%s\n' "$(((finished_ns - started_ns) / 1000000))" > \
    "${run_dir}/process-wall-milliseconds.txt"
  [[ ${rc} -eq 0 && -s ${report} ]] || {
    echo "run failed: ${run_dir} rc=${rc}" >&2
    return 1
  }
  jq -e --arg boundary "${BOUNDARY_TEXT}" '
    (.runtime.cuda_solve_call_wall_milliseconds|type)=="number" and
    .runtime.cuda_solve_call_wall_milliseconds>0 and
    .runtime.timing_boundaries.solve_call==$boundary and .cuda_ran==true' \
    "${report}" >/dev/null
  if [[ ${binary_kind} == r1 ]]; then
    exe_sha=$(sha256sum "${EXE}" | awk '{print $1}')
  else
    exe_sha=$(sha256sum "${FROZEN_EXE}" | awk '{print $1}')
  fi
  output_sha=$(sha256sum "${report}" | awk '{print $1}')
  sha256sum "${GLOBAL120}" > "${run_dir}/snapshot-manifest.sha256"
  sha256sum "${PAYLOAD}" > "${run_dir}/snapshot-payload.sha256"
  sha256sum "${report}" > "${run_dir}/output.sha256"
  sha256sum "${run_dir}/environment.before.txt" \
            "${run_dir}/environment.after.txt" > "${run_dir}/environment.sha256"
  jq -n --slurpfile config "${run_dir}/config.json" \
    --arg run_id "${run_id}" --arg run_dir "${run_dir}" \
    --arg command "$(tr '\n' ' ' < "${run_dir}/command.txt")" \
    --arg start "$(cat "${run_dir}/started_at.txt")" \
    --arg finish "$(cat "${run_dir}/finished_at.txt")" \
    --arg exe_sha "${exe_sha}" --arg output "${report}" \
    --arg output_sha "${output_sha}" \
    --arg snapshot_sha "$(sha256sum "${GLOBAL120}" | awk '{print $1}')" \
    --arg payload_sha "$(sha256sum "${PAYLOAD}" | awk '{print $1}')" \
    --arg boundary_id "${BOUNDARY_ID}" --arg boundary_text "${BOUNDARY_TEXT}" \
    --argjson rc "${rc}" \
    --argjson process_wall "$(cat "${run_dir}/process-wall-milliseconds.txt")" \
    '{schema:"phase7p1a-r1-run-manifest-v1",run_id:$run_id,run_dir:$run_dir,
      config:$config[0],command:$command,start_utc:$start,finish_utc:$finish,
      process_exit_code:$rc,process_wall_milliseconds:$process_wall,
      executable_sha256:$exe_sha,snapshot_manifest_sha256:$snapshot_sha,
      snapshot_payload_sha256:$payload_sha,output_path:$output,
      output_sha256:$output_sha,timing_boundary_id:$boundary_id,
      timing_boundary_description:$boundary_text}' > "${run_dir}/run-manifest.json"
  sha256sum "${run_dir}/run-manifest.json" > "${run_dir}/run-manifest.sha256"
  LAST_MANIFEST=${run_dir}/run-manifest.json
}

normal_counts_filter='
  def counts($r;$cache):
    if $cache==0 then
      ($r.lifecycle.layer_a_calls==31 and $r.lifecycle.layer_b_calls==21 and
       $r.lifecycle.layer_c_calls==10 and $r.lifecycle.cost_calls==31 and
       $r.lifecycle.BuildCudaLayerAInputs_count==72 and
       $r.lifecycle.current_linearization.logical_requests==21 and
       $r.lifecycle.current_linearization.lookup_aborts==0 and
       $r.lifecycle.current_linearization.cache_lookups==21 and
       $r.lifecycle.current_linearization.cache_hits==0 and
       $r.lifecycle.current_linearization.cache_misses==21 and
       $r.lifecycle.current_linearization.build_attempts==21 and
       $r.lifecycle.current_linearization.build_successes==21 and
       $r.lifecycle.current_linearization.build_failures==0 and
       $r.lifecycle.current_linearization.temporary_builds==10 and
       $r.lifecycle.current_linearization.publishes==11)
    else
      ($r.lifecycle.layer_a_calls==21 and $r.lifecycle.layer_b_calls==11 and
       $r.lifecycle.layer_c_calls==10 and $r.lifecycle.cost_calls==21 and
       $r.lifecycle.BuildCudaLayerAInputs_count==52 and
       $r.lifecycle.current_linearization.logical_requests==21 and
       $r.lifecycle.current_linearization.lookup_aborts==0 and
       $r.lifecycle.current_linearization.cache_lookups==21 and
       $r.lifecycle.current_linearization.cache_hits==10 and
       $r.lifecycle.current_linearization.cache_misses==11 and
       $r.lifecycle.current_linearization.build_attempts==11 and
       $r.lifecycle.current_linearization.build_successes==11 and
       $r.lifecycle.current_linearization.build_failures==0 and
       $r.lifecycle.current_linearization.temporary_builds==0 and
       $r.lifecycle.current_linearization.publishes==11)
    end and $r.lifecycle.final_internal_state_epoch==10 and
    $r.v2_audit.resource.advance_events==1 and
    $r.v2_audit.resource.advance_violations==0 and
    $r.v2_audit.resource.advanced_by_this_solve==true;
'

validate_pair() {
  local pair_id=$1 off_manifest=$2 on_manifest=$3
  local off_report on_report output
  off_report=$(jq -r .output_path "${off_manifest}")
  on_report=$(jq -r .output_path "${on_manifest}")
  output=${RUN_ROOT}/pairs/${pair_id}.json
  jq -n --slurpfile om "${off_manifest}" --slurpfile nm "${on_manifest}" \
    --slurpfile off "${off_report}" --slurpfile on "${on_report}" \
    --arg pair_id "${pair_id}" --arg boundary_id "${BOUNDARY_ID}" \
    "${normal_counts_filter}"'
    ($off[0].runtime.cuda_solve_call_wall_milliseconds) as $off_ms |
    ($on[0].runtime.cuda_solve_call_wall_milliseconds) as $on_ms |
    {pair_id:$pair_id,off_run_id:$om[0].run_id,on_run_id:$nm[0].run_id,
     timing_boundary_id:$boundary_id,
     same_binary:($om[0].executable_sha256==$nm[0].executable_sha256),
     same_snapshot:($om[0].snapshot_manifest_sha256==$nm[0].snapshot_manifest_sha256 and
                    $om[0].snapshot_payload_sha256==$nm[0].snapshot_payload_sha256),
     semantic_v1_pass:($off[0].runtime.semantic_bitwise_sha256_v1==
                       $on[0].runtime.semantic_bitwise_sha256_v1),
     semantic_v2_pass:($off[0].runtime.semantic_sha256_v2==
                       $on[0].runtime.semantic_sha256_v2),
     final_parameters_pass:($off[0].runtime.final_parameters_bitwise_sha256==
                            $on[0].runtime.final_parameters_bitwise_sha256),
     final_topology_pass:($off[0].runtime.final_topology_bitwise_sha256==
                          $on[0].runtime.final_topology_bitwise_sha256),
     fidelity_status_preserved:($off[0].pass==$on[0].pass),
     off_counts_pass:counts($off[0];0),on_counts_pass:counts($on[0];1),
     off_caller_wall_ms:$off_ms,on_caller_wall_ms:$on_ms,
     ratio:($on_ms/$off_ms),log_ratio:(($on_ms/$off_ms)|log)} |
    .pass=(.same_binary and .same_snapshot and .semantic_v1_pass and
      .semantic_v2_pass and .final_parameters_pass and .final_topology_pass and
      .fidelity_status_preserved and .off_counts_pass and .on_counts_pass)' \
    > "${output}"
  jq -e .pass "${output}" >/dev/null
}

run_pair() {
  local mode=$1 repeat=$2 order=$3 measured=$4
  local pair_id off_manifest on_manifest
  pair_id=global120-${mode}-normal-r1-${repeat}
  if [[ ${order} == AB ]]; then
    run_one r1 "${mode}" 0 "${measured}" "${repeat}" "${pair_id}" A
    off_manifest=${LAST_MANIFEST}
    run_one r1 "${mode}" 1 "${measured}" "${repeat}" "${pair_id}" B
    on_manifest=${LAST_MANIFEST}
  else
    run_one r1 "${mode}" 1 "${measured}" "${repeat}" "${pair_id}" B
    on_manifest=${LAST_MANIFEST}
    run_one r1 "${mode}" 0 "${measured}" "${repeat}" "${pair_id}" A
    off_manifest=${LAST_MANIFEST}
  fi
  validate_pair "${pair_id}" "${off_manifest}" "${on_manifest}"
}

validate_absolute_pair() {
  local pair_id=$1 v1_manifest=$2 r1_manifest=$3
  local v1_report r1_report output
  v1_report=$(jq -r .output_path "${v1_manifest}")
  r1_report=$(jq -r .output_path "${r1_manifest}")
  output=${RUN_ROOT}/absolute-pairs/${pair_id}.json
  jq -n --slurpfile vm "${v1_manifest}" --slurpfile rm "${r1_manifest}" \
    --slurpfile v1 "${v1_report}" --slurpfile r1 "${r1_report}" \
    --arg pair_id "${pair_id}" --arg boundary_id "${BOUNDARY_ID}" \
    --arg expected_v1_sha "${EXPECTED_V1_SHA}" \
    --arg expected_r1_sha "$(sha256sum "${EXE}" | awk '{print $1}')" \
    "${normal_counts_filter}"'
    ($v1[0].runtime.cuda_solve_call_wall_milliseconds) as $v1_ms |
    ($r1[0].runtime.cuda_solve_call_wall_milliseconds) as $r1_ms |
    {pair_id:$pair_id,v1_run_id:$vm[0].run_id,r1_run_id:$rm[0].run_id,
     timing_boundary_id:$boundary_id,
     binary_identity_pass:($vm[0].executable_sha256==$expected_v1_sha and
                           $rm[0].executable_sha256==$expected_r1_sha),
     same_snapshot:($vm[0].snapshot_manifest_sha256==
                    $rm[0].snapshot_manifest_sha256 and
                    $vm[0].snapshot_payload_sha256==
                    $rm[0].snapshot_payload_sha256),
     boundary_pass:($vm[0].timing_boundary_id==$boundary_id and
                    $rm[0].timing_boundary_id==$boundary_id),
     semantic_v1_pass:($v1[0].runtime.semantic_bitwise_sha256_v1==
                       $r1[0].runtime.semantic_bitwise_sha256_v1),
     decision_v1_pass:($v1[0].runtime.decision_bitwise_sha256==
                       $r1[0].runtime.decision_bitwise_sha256),
     final_parameters_pass:($v1[0].runtime.final_parameters_bitwise_sha256==
                            $r1[0].runtime.final_parameters_bitwise_sha256),
     final_topology_pass:($v1[0].runtime.final_topology_bitwise_sha256==
                          $r1[0].runtime.final_topology_bitwise_sha256),
     fidelity_status_preserved:($v1[0].pass==$r1[0].pass),
     v1_counts_pass:($v1[0].lifecycle.layer_a_calls==21 and
       $v1[0].lifecycle.layer_b_calls==11 and
       $v1[0].lifecycle.layer_c_calls==10 and
       $v1[0].lifecycle.cost_calls==21 and
       $v1[0].lifecycle.BuildCudaLayerAInputs_count==52 and
       $v1[0].lifecycle.current_linearization.requests==21 and
       $v1[0].lifecycle.current_linearization.cache_hits==10 and
       $v1[0].lifecycle.current_linearization.cache_misses==11 and
       $v1[0].lifecycle.current_linearization.build_attempts==11 and
       $v1[0].lifecycle.current_linearization.build_successes==11 and
       $v1[0].lifecycle.current_linearization.build_failures==0 and
       $v1[0].lifecycle.current_linearization.temporary_builds==0 and
       $v1[0].lifecycle.current_linearization.publishes==11 and
       $v1[0].lifecycle.final_internal_state_epoch==10),
     r1_counts_pass:counts($r1[0];1),
     v1_caller_wall_ms:$v1_ms,r1_caller_wall_ms:$r1_ms,
     ratio:($r1_ms/$v1_ms),log_ratio:(($r1_ms/$v1_ms)|log)} |
    .pass=(.binary_identity_pass and .same_snapshot and .boundary_pass and
      .semantic_v1_pass and .decision_v1_pass and .final_parameters_pass and
      .final_topology_pass and .fidelity_status_preserved and
      .v1_counts_pass and .r1_counts_pass)' > "${output}"
  jq -e .pass "${output}" >/dev/null
}

run_absolute_pair() {
  local mode=$1 repeat=$2 order=$3
  local pair_id v1_manifest r1_manifest
  pair_id=global120-${mode}-absolute-r1-${repeat}
  if [[ ${order} == AB ]]; then
    run_one v1 "${mode}" 1 true "${repeat}" "${pair_id}" A
    v1_manifest=${LAST_MANIFEST}
    run_one r1 "${mode}" 1 true "${repeat}" "${pair_id}" B
    r1_manifest=${LAST_MANIFEST}
  else
    run_one r1 "${mode}" 1 true "${repeat}" "${pair_id}" B
    r1_manifest=${LAST_MANIFEST}
    run_one v1 "${mode}" 1 true "${repeat}" "${pair_id}" A
    v1_manifest=${LAST_MANIFEST}
  fi
  validate_absolute_pair "${pair_id}" "${v1_manifest}" "${r1_manifest}"
}

derive_historical() {
  local mode=$1 output=${RUN_ROOT}/historical-v1/${mode}.json lines
  lines=${RUN_ROOT}/historical-v1/${mode}.jsonl
  : > "${lines}"
  mapfile -t pair_rows < <(jq -c --arg mode "${mode}" '
    [.[] | select(.pair_id|test("^global120-"+$mode+"-normal-i0-true-r[1-5]$"))]
    | sort_by(.pair_id) | .[]' "${V1_ROOT}/pairs.json")
  [[ ${#pair_rows[@]} -eq 5 ]] || return 1
  local row run_id run_dir config manifest report valid caller
  for row in "${pair_rows[@]}"; do
    run_id=$(jq -r .on_run_id <<<"${row}")
    run_dir=${V1_ROOT}/runs/${run_id}
    config=${run_dir}/config.json
    manifest=${run_dir}/run-manifest.json
    report=${run_dir}/replay.json
    [[ -s ${config} && -s ${manifest} && -s ${report} ]] || return 1
    valid=$(jq -n --slurpfile c "${config}" --slurpfile m "${manifest}" \
      --slurpfile r "${report}" --arg mode "${mode}" \
      --arg sha "${EXPECTED_V1_SHA}" --arg boundary "${BOUNDARY_TEXT}" '
      ($c[0].case=="global120" and $c[0].scenario=="normal" and
       $c[0].cache==1 and $c[0].instrumentation==0 and
       $c[0].measured==true and $c[0].mode==$mode and
       $m[0].executable_sha256==$sha and
       ($r[0].runtime.cuda_solve_call_wall_milliseconds|type)=="number" and
       $r[0].runtime.cuda_solve_call_wall_milliseconds>0 and
       $r[0].runtime.timing_boundaries.solve_call==$boundary)')
    [[ ${valid} == true ]] || return 1
    caller=$(jq -r .runtime.cuda_solve_call_wall_milliseconds "${report}")
    jq -n --arg pair_id "$(jq -r .pair_id <<<"${row}")" \
      --arg run_id "${run_id}" --arg mode "${mode}" \
      --arg config_path "${config}" --arg manifest_path "${manifest}" \
      --arg report_path "${report}" --arg sha "${EXPECTED_V1_SHA}" \
      --arg boundary_id "${BOUNDARY_ID}" --argjson caller "${caller}" \
      '{pair_id:$pair_id,on_run_id:$run_id,mode:$mode,
        selection_predicate:{case_global120:true,scenario_normal:true,cache_on:true,
          instrumentation_off:true,measured:true,mode_match:true,binary_match:true,
          caller_wall_valid:true,boundary_match:true},config_path:$config_path,
        run_manifest_path:$manifest_path,report_path:$report_path,
        executable_sha256:$sha,timing_boundary_id:$boundary_id,
        caller_wall_milliseconds:$caller}' >> "${lines}"
  done
  jq -s --arg mode "${mode}" '
    def median: sort | .[length/2|floor];
    {schema:"phase7p1a-r1-historical-v1-selection-v1",mode:$mode,
     source:"exact_on_run_ids_from_phase7p1a_v1_pairs",runs:.,
     selected_run_ids:[.[].on_run_id],
     caller_wall_values:[.[].caller_wall_milliseconds]} |
    .frozen_v1_caller_wall_median=(.caller_wall_values|median) |
    .pass=((.runs|length)==5 and (.runs|all(.selection_predicate|all(.==true))))' \
    "${lines}" > "${output}"
  jq -e .pass "${output}" >/dev/null
}

summarize_pairs() {
  find "${RUN_ROOT}/pairs" -maxdepth 1 -name '*.json' -print0 | sort -z | \
    xargs -0 jq -s '.' > "${RUN_ROOT}/pairs.json"
  jq '
    def median: sort | if length%2==1 then .[length/2|floor]
      else (.[length/2-1]+.[length/2])/2 end;
    def mean: add/length;
    def sample_stddev: . as $v | ($v|mean) as $m |
      if length<2 then 0 else
      (([$v[]|(.-$m)*(.-$m)]|add)/(length-1)|sqrt) end;
    def stats: . as $v | ($v|mean) as $m | ($v|sample_stddev) as $s |
      {values:$v,min:($v|min),median:($v|median),max:($v|max),mean:$m,
       sample_stddev:$s,cv:($s/$m)};
    def mode($pairs):
      ([$pairs[].off_caller_wall_ms]) as $off |
      ([$pairs[].on_caller_wall_ms]) as $on |
      ([$pairs[].ratio]) as $ratios | ([$pairs[].log_ratio]) as $logs |
      ($logs|median) as $medlog |
      ([$logs[]|(.-$medlog|fabs)]|median) as $mad |
      ($off|stats) as $offs | ($on|stats) as $ons |
      ((1+$offs.cv*$offs.cv)|log|sqrt) as $nco |
      ((1+$ons.cv*$ons.cv)|log|sqrt) as $ncn |
      (1.4826*$mad) as $nm |
      ([$nco,$ncn,$nm]|max) as $noise |
      (-$medlog) as $signal |
      {pairs:$pairs,off:$offs,on:$ons,ratios:$ratios,log_ratios:$logs,
       median_log_ratio:$medlog,mad_log_ratio:$mad,noise_cv_off:$nco,
       noise_cv_on:$ncn,noise_mad:$nm,noise:$noise,signal:$signal,
       faster_pair_count:([$ratios[]|select(.<1)]|length),required_faster_pairs:5,
       correctness_pass:($pairs|all(.pass)),
       semantic_v1_pass:($pairs|all(.semantic_v1_pass)),
       semantic_v2_pass:($pairs|all(.semantic_v2_pass)),
       final_state_pass:($pairs|all(.final_parameters_pass and .final_topology_pass)),
       execution_counts_pass:($pairs|all(.off_counts_pass and .on_counts_pass))} |
      .acceleration_pass=(.signal>.noise and .faster_pair_count>=5 and
        .correctness_pass and .semantic_v1_pass and .semantic_v2_pass and
        .final_state_pass and .execution_counts_pass);
    . as $all |
    {schema:"phase7p1a-r1-caller-wall-paired-gate-v1",algorithm_version:1,
     N:6,timing_boundary_id:"cuda_solve_call_wall_milliseconds.caller_inclusive.v1",
     serial:mode([$all[]|select(.pair_id|test("-serial-"))]),
     parallel:mode([$all[]|select(.pair_id|test("-parallel-"))])} |
    .pass=(.serial.acceleration_pass and .parallel.acceleration_pass)' \
    "${RUN_ROOT}/pairs.json" > "${RUN_ROOT}/paired-summary.json"
}

for mode in serial parallel; do
  run_one r1 "${mode}" 0 false 0 "" warmup_off
  run_one r1 "${mode}" 1 false 0 "" warmup_on
  for repeat in 1 2 3 4 5 6; do
    if ((repeat % 2 == 1)); then order=AB; else order=BA; fi
    run_pair "${mode}" "${repeat}" "${order}" true
  done
done
summarize_pairs

historical_available=true
for mode in serial parallel; do
  if ! derive_historical "${mode}"; then
    historical_available=false
    jq -n --arg mode "${mode}" \
      '{schema:"phase7p1a-r1-historical-v1-selection-v1",mode:$mode,
        pass:false,status:"UNAVAILABLE",runs:[],selected_run_ids:[],
        reason:"exact historical V1 selection failed validation"}' \
      > "${RUN_ROOT}/historical-v1/${mode}.json"
  fi
done

jq -n \
  --slurpfile paired "${RUN_ROOT}/paired-summary.json" \
  --slurpfile serial "${RUN_ROOT}/historical-v1/serial.json" \
  --slurpfile parallel "${RUN_ROOT}/historical-v1/parallel.json" '
  def mode($r1;$v1):
    if $v1.pass then
      ($r1.on.median/$v1.frozen_v1_caller_wall_median) as $ratio |
      {source:"historical_exact_run_id_selection",r1_on_median:$r1.on.median,
       frozen_v1_on_median:$v1.frozen_v1_caller_wall_median,
       median_ratio:$ratio,threshold:1.05,
       paired_same_batch_required:($ratio>1.05),
       status:(if $ratio<=1.05 then "PASS" else "PENDING_SAME_BATCH" end)}
    else
      {source:"historical_selection_unavailable",r1_on_median:$r1.on.median,
       frozen_v1_on_median:null,median_ratio:null,threshold:1.05,
       paired_same_batch_required:true,status:"PENDING_SAME_BATCH"}
    end;
  {schema:"phase7p1a-r1-absolute-caller-wall-gate-v1",
   serial:mode($paired[0].serial;$serial[0]),
   parallel:mode($paired[0].parallel;$parallel[0])} |
  .same_batch_required=(.serial.paired_same_batch_required or
                        .parallel.paired_same_batch_required) |
  .status=(if .same_batch_required then "PENDING_SAME_BATCH" else "PASS" end)' \
  > "${RUN_ROOT}/absolute-summary.json"

same_batch_mode() {
  local mode=$1
  run_one v1 "${mode}" 1 false 0 "" same_batch_warmup_v1
  run_one r1 "${mode}" 1 false 0 "" same_batch_warmup_r1
  local repeat order
  for repeat in 1 2 3 4 5 6; do
    if ((repeat % 2 == 1)); then order=AB; else order=BA; fi
    run_absolute_pair "${mode}" "${repeat}" "${order}"
  done
}

for mode in serial parallel; do
  if jq -e --arg mode "${mode}" '.[$mode].paired_same_batch_required' \
      "${RUN_ROOT}/absolute-summary.json" >/dev/null; then
    same_batch_mode "${mode}"
  fi
done

mapfile -d '' absolute_pair_files < <(
  find "${RUN_ROOT}/absolute-pairs" -maxdepth 1 -name '*.json' -print0 | sort -z
)
if ((${#absolute_pair_files[@]} == 0)); then
  printf '[]\n' > "${RUN_ROOT}/absolute-pairs.json"
else
  jq -s '.' "${absolute_pair_files[@]}" > "${RUN_ROOT}/absolute-pairs.json"
fi

jq -n --slurpfile absolute "${RUN_ROOT}/absolute-summary.json" \
  --slurpfile pairs "${RUN_ROOT}/absolute-pairs.json" '
  def median: sort | if length%2==1 then .[length/2|floor]
    else (.[length/2-1]+.[length/2])/2 end;
  def mean: add/length;
  def sample_stddev: . as $v | ($v|mean) as $m |
    if length<2 then 0 else
    (([$v[]|(.-$m)*(.-$m)]|add)/(length-1)|sqrt) end;
  def evaluate($mode;$base;$all):
    if ($base.paired_same_batch_required|not) then
      $base + {same_batch:null,final_status:"PASS"}
    else
      ([$all[]|select(.pair_id|test("-"+$mode+"-absolute-"))]) as $p |
      ([$p[].v1_caller_wall_ms]) as $v1 |
      ([$p[].r1_caller_wall_ms]) as $r1 |
      ([$p[].ratio]) as $ratios | ([$p[].log_ratio]) as $logs |
      ($v1|mean) as $v1_mean | ($r1|mean) as $r1_mean |
      ($v1|sample_stddev) as $v1_sd | ($r1|sample_stddev) as $r1_sd |
      ($v1_sd/$v1_mean) as $v1_cv | ($r1_sd/$r1_mean) as $r1_cv |
      ($logs|median) as $median_log |
      ([$logs[]|(.-$median_log|fabs)]|median) as $mad |
      ((1+$v1_cv*$v1_cv)|log|sqrt) as $noise_v1 |
      ((1+$r1_cv*$r1_cv)|log|sqrt) as $noise_r1 |
      (1.4826*$mad) as $noise_mad |
      ([$noise_v1,$noise_r1,$noise_mad]|max) as $noise |
      ($ratios|median) as $median_ratio |
      ($median_log) as $signal |
      ([$ratios[]|select(.>1)]|length) as $slower |
      ($p|all(.pass)) as $correct |
      (if ($correct and $median_ratio<=1.05) then "PASS"
       elif ($correct and $median_ratio>1.05 and $signal>$noise and $slower>=5)
       then "FAIL" else "INCONCLUSIVE" end) as $status |
      $base + {same_batch:{algorithm_version:1,N:6,pairs:$p,
        v1_values:$v1,r1_values:$r1,ratios:$ratios,log_ratios:$logs,
        v1_mean:$v1_mean,r1_mean:$r1_mean,
        v1_sample_stddev:$v1_sd,r1_sample_stddev:$r1_sd,
        cv_v1:$v1_cv,cv_r1:$r1_cv,median_ratio:$median_ratio,
        median_log_ratio:$median_log,mad_log_ratio:$mad,
        noise_cv_v1:$noise_v1,noise_cv_r1:$noise_r1,
        noise_mad:$noise_mad,regression_noise:$noise,
        regression_signal:$signal,slower_pair_count:$slower,
        required_slower_pairs:5,correctness_pass:$correct,
        status:$status},final_status:$status}
    end;
  ($pairs[0] // []) as $all |
  $absolute[0] |
  .serial=evaluate("serial";.serial;$all) |
  .parallel=evaluate("parallel";.parallel;$all) |
  .same_batch_completed=((.serial.paired_same_batch_required|not) or
    .serial.same_batch!=null) and ((.parallel.paired_same_batch_required|not) or
    .parallel.same_batch!=null) |
  .status=(if .serial.final_status=="FAIL" or .parallel.final_status=="FAIL"
    then "FAIL" elif .serial.final_status=="INCONCLUSIVE" or
    .parallel.final_status=="INCONCLUSIVE" then "INCONCLUSIVE" else "PASS" end)' \
  > "${RUN_ROOT}/absolute-summary.final.json"
mv "${RUN_ROOT}/absolute-summary.final.json" "${RUN_ROOT}/absolute-summary.json"

find "${RUN_ROOT}/runs" -mindepth 2 -name run-manifest.json -print0 | \
  sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/runs.json"
jq -n --slurpfile paired "${RUN_ROOT}/paired-summary.json" \
  --slurpfile absolute "${RUN_ROOT}/absolute-summary.json" \
  --slurpfile hs "${RUN_ROOT}/historical-v1/serial.json" \
  --slurpfile hp "${RUN_ROOT}/historical-v1/parallel.json" \
  '{schema:"phase7p1a-r1-performance-summary-v1",paired:$paired[0],
    absolute:$absolute[0],historical_v1:{serial:$hs[0],parallel:$hp[0]},
    acceleration_pass:$paired[0].pass,absolute_status:$absolute[0].status}' \
  > "${RUN_ROOT}/summary.json"
find "${RUN_ROOT}" -type f ! -name integrity.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RUN_ROOT}/integrity.sha256"
jq empty "${RUN_ROOT}"/*.json "${RUN_ROOT}"/historical-v1/*.json
echo "phase7p1a-r1 performance complete: ${RUN_ROOT}"
