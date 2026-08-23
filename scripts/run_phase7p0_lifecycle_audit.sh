#!/usr/bin/env bash
set -u

ARTIFACT_ROOT="${1:-/home/nvidia/colmap-PCD-gpu-artifacts}"
EXE="${2:-/home/nvidia/colmap-PCD-gpu-build/cuda-release/src/gpu_ba/gpu_ba_custom_cuda_full_lm_replay}"
RUN_ROOT="${3:-${ARTIFACT_ROOT}/oracles/phase7p0-lm-lifecycle-timing-v1/runs}"
SNAP_ROOT="${ARTIFACT_ROOT}/oracles/phase5p5-ceres-replay-fidelity-v1/prod50-envelope8/snapshots"
REG2_SNAP="${ARTIFACT_ROOT}/snapshots/phase2-canonical/global-reg2-call1-refine0-trigger35-phraseglobal.manifest.json"
GLOBAL120_SNAP="${SNAP_ROOT}/global-reg50-call120-refine0-trigger79-phraseglobal.manifest.json"

mkdir -p "${RUN_ROOT}"
if [[ ! -x "${EXE}" ]]; then
  echo "missing executable: ${EXE}" >&2
  exit 2
fi
if [[ ! -f "${REG2_SNAP}" || ! -f "${GLOBAL120_SNAP}" ]]; then
  echo "missing production snapshot" >&2
  exit 2
fi

sha256sum "${EXE}" > "${RUN_ROOT}/binary.sha256"
{
  date -u +%FT%TZ
  uname -a
  printf 'nproc='; nproc || true
  printf 'nvpmodel='; nvpmodel -q 2>&1 || true
  printf 'jetson_clocks='; jetson_clocks --show 2>&1 || true
  printf 'cpu_freq='; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>&1 || true
  printf 'gpu_freq='; cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>&1 || true
  printf 'emc_freq='; cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>&1 || true
  printf 'tegrastats='; timeout 5 tegrastats --interval 1000 --count 1 2>&1 || true
  printf 'temperature='; cat /sys/class/thermal/thermal_zone*/temp 2>&1 || true
  printf 'power='; cat /sys/bus/i2c/drivers/ina3221x/*/iio_device/*/in_power*_input 2>&1 || true
  printf 'nvcc='; /usr/local/cuda-11.4/bin/nvcc --version 2>&1 || true
  printf 'driver='; cat /proc/driver/nvidia/version 2>&1 || true
  printf 'gpu='; cat /sys/devices/soc0/machine 2>&1 || true
  printf 'nvidia_smi='; nvidia-smi 2>&1 || true
  printf 'nsys='; nsys --version 2>&1 || true
  printf 'ncu='; ncu --version 2>&1 || true
} > "${RUN_ROOT}/environment.txt"
sha256sum "${RUN_ROOT}/environment.txt" > "${RUN_ROOT}/environment.sha256"

run_one() {
  local label="$1" snapshot="$2" mode="$3" probe="$4"
  local id dir report rc start_ns end_ns args
  id="$(date -u +%Y%m%dT%H%M%S%N)-$$-${RANDOM}"
  dir="${RUN_ROOT}/${id}"
  mkdir -p "${dir}"
  report="${dir}/replay.json"
  args=("${EXE}" "${snapshot}" "${report}" 8)
  [[ "${mode}" == parallel ]] && args+=(parallel)
  [[ "${probe}" == probe ]] && args+=(forced_reject_probe)
  printf '%q ' "${args[@]}" > "${dir}/command.txt"
  printf '\n' >> "${dir}/command.txt"
  printf 'label=%s\nmode=%s\nprobe=%s\n' "${label}" "${mode}" "${probe}" > "${dir}/run-config.txt"
  date -u +%FT%T.%NZ > "${dir}/started_at.txt"
  cp -p "${RUN_ROOT}/environment.txt" "${dir}/environment.txt"
  cp -p "${RUN_ROOT}/environment.sha256" "${dir}/environment.sha256"
  sha256sum "${snapshot}" > "${dir}/snapshot.sha256"
  sha256sum "${EXE}" > "${dir}/binary.sha256"
  start_ns="$(date +%s%N)"
  "${args[@]}" >"${dir}/stdout.txt" 2>"${dir}/stderr.txt"
  rc=$?
  end_ns="$(date +%s%N)"
  date -u +%FT%T.%NZ > "${dir}/finished_at.txt"
  printf '%s\n' "${rc}" > "${dir}/exit_code.txt"
  printf '%s\n' "$(( (end_ns - start_ns) / 1000000 ))" > "${dir}/host_wall_milliseconds.txt"
  if [[ -s "${report}" ]]; then
    jq empty "${report}" > "${dir}/jq-validation.txt" 2>&1 || true
    sha256sum "${report}" > "${dir}/report.sha256"
    jq --arg label "${label}" --arg mode "${mode}" --arg probe "${probe}" \
       --argjson rc "${rc}" \
       --argjson wall "$(cat "${dir}/host_wall_milliseconds.txt")" \
       '. + {run_label:$label,run_mode:$mode,run_probe:$probe,process_exit_code:$rc,measured_host_wall_milliseconds:$wall}' \
       "${report}" > "${dir}/report.with-run.json" && mv "${dir}/report.with-run.json" "${report}"
    sha256sum "${report}" > "${dir}/report.sha256"
  fi
  local report_hash=""
  if [[ -s "${dir}/report.sha256" ]]; then
    report_hash="$(awk '{print $1}' "${dir}/report.sha256")"
  fi
  jq -n --arg id "${id}" --arg label "${label}" --arg mode "${mode}" \
        --arg probe "${probe}" --arg snapshot "${snapshot}" --arg report "${report}" \
        --arg command "$(tr '\n' ' ' < "${dir}/command.txt")" \
        --arg start "$(cat "${dir}/started_at.txt")" --arg finish "$(cat "${dir}/finished_at.txt")" \
        --arg binary_hash "$(awk '{print $1}' "${dir}/binary.sha256")" \
        --arg snapshot_hash "$(awk '{print $1}' "${dir}/snapshot.sha256")" \
        --arg environment_hash "$(awk '{print $1}' "${dir}/environment.sha256")" \
        --arg report_hash "${report_hash}" \
        --argjson rc "${rc}" --argjson wall "$(cat "${dir}/host_wall_milliseconds.txt")" \
        '{run_id:$id,label:$label,mode:$mode,probe:$probe,command:$command,start_utc:$start,finish_utc:$finish,process_exit_code:$rc,host_wall_milliseconds:$wall,binary_hash:$binary_hash,environment_hash:$environment_hash,snapshot_path:$snapshot,snapshot_hash:$snapshot_hash,report_path:$report,report_hash:$report_hash}' \
        > "${dir}/run-manifest.json"
  printf '%s\n' "${dir}/run-manifest.json"
}

run_group() {
  local label="$1" snapshot="$2" mode="$3" probe="$4" repeats="$5" i
  for ((i=1; i<=repeats; ++i)); do
    run_one "${label}-${i}" "${snapshot}" "${mode}" "${probe}"
  done
}

# Probe runs: the requested settings are encoded by the replay binary and the
# JSON reports whether the actual trajectory really rejected every trial.
run_group global120 "${GLOBAL120_SNAP}" serial probe 3
run_group global120 "${GLOBAL120_SNAP}" parallel probe 3
run_group reg2 "${REG2_SNAP}" serial probe 1
run_group reg2 "${REG2_SNAP}" parallel probe 1

# Normal runs: one uncounted warm-up per mode, followed by three measured runs.
run_group global120-warmup "${GLOBAL120_SNAP}" serial normal 1
run_group global120-warmup "${GLOBAL120_SNAP}" parallel normal 1
run_group global120 "${GLOBAL120_SNAP}" serial normal 3
run_group global120 "${GLOBAL120_SNAP}" parallel normal 3

find "${RUN_ROOT}" -mindepth 2 -name run-manifest.json -print0 | sort -z | xargs -0 jq -s '.' > "${RUN_ROOT}/run-manifests.json"
sha256sum "${RUN_ROOT}/run-manifests.json" > "${RUN_ROOT}/run-manifests.sha256"
echo "phase7p0 lifecycle audit runs written under ${RUN_ROOT}"
