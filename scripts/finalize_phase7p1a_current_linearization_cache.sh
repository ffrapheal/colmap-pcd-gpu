#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/home/nvidia/colmap-PCD-gpu-artifacts/oracles/phase7p1a-current-linearization-cache-v1}"
SOURCE_ROOT="${2:-/home/nvidia/colmap-PCD-gpu}"
[[ -d "${RUN_ROOT}/runs" && -d "${RUN_ROOT}/pairs" ]] || {
  echo "incomplete Phase 7.1a oracle root: ${RUN_ROOT}" >&2; exit 2;
}
[[ "$(find "${RUN_ROOT}/runs" -name run-manifest.json | wc -l)" -eq 42 ]]
[[ "$(find "${RUN_ROOT}/pairs" -name '*.json' | wc -l)" -eq 20 ]]

while IFS= read -r manifest; do
  output="$(jq -r .output_path "${manifest}")"
  expected="$(jq -r .output_sha256 "${manifest}")"
  [[ -f "${output}" ]]
  [[ "$(sha256sum "${output}" | awk '{print $1}')" == "${expected}" ]]
  jq empty "${manifest}" "${output}"
done < <(find "${RUN_ROOT}/runs" -name run-manifest.json | sort)
for pair in "${RUN_ROOT}"/pairs/*.json; do
  jq -e .pass "${pair}" >/dev/null
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
     faster_pair_count:$faster,correctness_pass:($pairs|all(.pass)),
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

jq -e '.measured_pairs_per_mode==5 and ((.serial.pairs|length)==5)' \
  "${RUN_ROOT}/benchmark-summary.json" >/dev/null
jq -e '((.parallel.pairs|length)==5)' \
  "${RUN_ROOT}/benchmark-summary.json" >/dev/null
jq -n --slurpfile runs "${RUN_ROOT}/runs.json" \
  --slurpfile pairs "${RUN_ROOT}/pairs.json" \
  --slurpfile benchmark "${RUN_ROOT}/benchmark-summary.json" \
  '{schema:"phase7p1a-current-linearization-cache-oracle-v1",
    runs:$runs[0],pairs:$pairs[0],benchmark:$benchmark[0]}' \
  > "${RUN_ROOT}/summary.json"

cp "$0" "${RUN_ROOT}/finalizer.executed.sh"
cp "${SOURCE_ROOT}/scripts/run_phase7p1a_current_linearization_cache.sh" \
  "${RUN_ROOT}/runner.corrected.sh"
sha256sum "${RUN_ROOT}/finalizer.executed.sh" \
  "${RUN_ROOT}/runner.corrected.sh" > "${RUN_ROOT}/executed-scripts.sha256"
find "${RUN_ROOT}" -type f ! -name integrity.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > "${RUN_ROOT}/integrity.sha256"
jq empty "${RUN_ROOT}"/*.json
echo "Phase 7.1a summary finalized under ${RUN_ROOT}"
