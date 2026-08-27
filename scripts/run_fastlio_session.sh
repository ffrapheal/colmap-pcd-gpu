#!/usr/bin/env bash
# Optimize an indexed FAST-LIO session and emit a complete pose session.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_BIN="${COLMAP_PCD_FRONTEND_BIN:-/home/nvidia/colmap_test/local/colmap/bin/colmap}"
MAPPER_BIN="${COLMAP_PCD_MAPPER_BIN:-/home/nvidia/colmap-PCD-gpu-build/cuda-release/src/exe/colmap}"
CUDA_NORMAL_BIN="${COLMAP_PCD_CUDA_NORMAL_BIN:-/home/nvidia/colmap-PCD-gpu-build/cuda-release/src/lidar/colmap_cuda_lidar_normals}"
MESH_DEPTH_GENERATOR="${COLMAP_PCD_MESH_DEPTH_GENERATOR:-/home/nvidia/intensity_opt/build/generate_mesh_depth}"
INTRINSICS="${COLMAP_PCD_INTRINSICS:-$REPO_ROOT/config/fastlio_camera_1224x1024.json}"
LOCK_FILE="${COLMAP_PCD_LOCK_FILE:-/home/nvidia/.codex/colmap-pcd-gpu-build-test.lock}"
SESSION_DIR=""
OUTPUT_ROOT=""
MESH_PATH=""

usage() {
  printf 'Usage: %s --session-dir DIR --output-root DIR [--intrinsics FILE] [--mesh-path FILE]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir) SESSION_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --intrinsics) INTRINSICS="$2"; shift 2 ;;
    --mesh-path) MESH_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SESSION_DIR" && -n "$OUTPUT_ROOT" ]] || { usage >&2; exit 2; }
SESSION_DIR="$(readlink -f "$SESSION_DIR")"
INTRINSICS="$(readlink -f "$INTRINSICS")"
if [[ -z "$MESH_PATH" && -f "$SESSION_DIR/mesh_0.ply" ]]; then
  MESH_PATH="$SESSION_DIR/mesh_0.ply"
fi
[[ -n "$MESH_PATH" ]] || {
  printf 'Mesh initialization requires --mesh-path FILE.\n' >&2
  exit 2
}
MESH_PATH="$(readlink -f "$MESH_PATH")"
OUTPUT_ROOT="$(mkdir -p "$OUTPUT_ROOT" && cd "$OUTPUT_ROOT" && pwd)"

if [[ "${COLMAP_PCD_LOCK_HELD:-0}" != 1 ]]; then
  exec flock -n "$LOCK_FILE" env COLMAP_PCD_LOCK_HELD=1 "$0" \
    --session-dir "$SESSION_DIR" --output-root "$OUTPUT_ROOT" \
    --intrinsics "$INTRINSICS" --mesh-path "$MESH_PATH"
fi

for path in "$SESSION_DIR" "$INTRINSICS" "$MESH_PATH" "$FRONTEND_BIN" \
            "$MAPPER_BIN" "$MESH_DEPTH_GENERATOR"; do
  [[ -e "$path" ]] || { printf 'Missing required path: %s\n' "$path" >&2; exit 2; }
done
LIDAR_NORMAL_BACKEND="${COLMAP_PCD_LIDAR_NORMAL_BACKEND:-cuda}"
case "$LIDAR_NORMAL_BACKEND" in
  cuda) [[ -x "$CUDA_NORMAL_BIN" ]] || {
          printf 'Missing CUDA normal estimator: %s\n' "$CUDA_NORMAL_BIN" >&2
          exit 2
        } ;;
  pcl) ;;
  *) printf 'Unsupported LiDAR normal backend: %s\n' \
       "$LIDAR_NORMAL_BACKEND" >&2; exit 2 ;;
esac
for command in sqlite3 pcl_concatenate_points_pcd pcl_voxel_grid \
               pcl_pcd2ply python3; do
  command -v "$command" >/dev/null || {
    printf 'Missing required command: %s\n' "$command" >&2
    exit 2
  }
done
if [[ "$LIDAR_NORMAL_BACKEND" == pcl ]]; then
  command -v pcl_normal_estimation >/dev/null || {
    printf 'Missing required command: pcl_normal_estimation\n' >&2
    exit 2
  }
fi

IMAGE_DIR="$OUTPUT_ROOT/images"
IMAGE_LIST="$OUTPUT_ROOT/image-list.txt"
DATABASE="$OUTPUT_ROOT/database.db"
LIDAR_DIR="$OUTPUT_ROOT/lidar"
LIDAR_MAP="$LIDAR_DIR/lidar-map.ply"
LIDAR_BA_MAP="$LIDAR_DIR/lidar-ba-map.ply"
LIDAR_BUILD_CONFIG="$LIDAR_DIR/build-config.txt"
LIDAR_PROJECTION_NORMAL_RADIUS="${COLMAP_PCD_LIDAR_PROJECTION_NORMAL_RADIUS:-0.15}"
LIDAR_BA_NORMAL_RADIUS="${COLMAP_PCD_LIDAR_BA_NORMAL_RADIUS:-0.05}"
POSE_PRIOR="$OUTPUT_ROOT/pose-prior.ply"
SPARSE="$OUTPUT_ROOT/sparse"
OPTIMIZED_SESSION="$OUTPUT_ROOT/optimized-session"
OPTIMIZED_REPORT="$OUTPUT_ROOT/optimized-session-report.json"
MESH_DEPTH_INTRINSICS="$OUTPUT_ROOT/initial-mesh-depth-intrinsics.json"
MESH_DEPTH_ROOT="$OUTPUT_ROOT/initial-mesh-depth"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$IMAGE_DIR" "$LIDAR_DIR" "$SPARSE" "$LOG_DIR" "$MESH_DEPTH_ROOT"

mapfile -t IMAGES < <(find "$SESSION_DIR" -maxdepth 1 -type f \
  \( -name 'imgs_*.jpg' -o -name 'imgs_*.jpeg' -o -name 'imgs_*.png' \) \
  -print | sort -V)
mapfile -t ODOMS < <(find "$SESSION_DIR" -maxdepth 1 -type f \
  -name 'odoms_*.txt' -print | sort -V)
mapfile -t SCANS < <(find "$SESSION_DIR" -maxdepth 1 -type f \
  -name 'scans_*.pcd' -print | sort -V)
FRAME_COUNT="${#IMAGES[@]}"
if (( FRAME_COUNT < 2 || ${#ODOMS[@]} != FRAME_COUNT ||
      ${#SCANS[@]} != FRAME_COUNT )); then
  printf 'Expected equal non-zero imgs/odoms/scans counts; got %d/%d/%d\n' \
    "$FRAME_COUNT" "${#ODOMS[@]}" "${#SCANS[@]}" >&2
  exit 2
fi

if [[ -f "$OUTPUT_ROOT/source-session.txt" ]]; then
  [[ "$(<"$OUTPUT_ROOT/source-session.txt")" == "$SESSION_DIR" ]] || {
    printf 'Output root belongs to another session.\n' >&2
    exit 2
  }
else
  printf '%s\n' "$SESSION_DIR" > "$OUTPUT_ROOT/source-session.txt"
fi

if [[ ! -s "$IMAGE_LIST" ]]; then
  : > "$IMAGE_LIST"
  for source in "${IMAGES[@]}"; do
    index="$(basename "$source" | sed -E 's/^imgs_([0-9]+)\..*$/\1/')"
    name="frame_$(printf '%06d' "$index").jpg"
    ln -s "$source" "$IMAGE_DIR/$name"
    printf '%s\n' "$name" >> "$IMAGE_LIST"
  done
fi
[[ "$(wc -l < "$IMAGE_LIST")" -eq "$FRAME_COUNT" ]]

CAMERA_PARAMS="$(python3 - "$INTRINSICS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
k, dist = d['K'], d['D']
print(','.join(str(value) for value in (
    k['fx'], k['fy'], k['cx'], k['cy'],
    dist[0], dist[1], dist[2], dist[3])))
PY
)"

MESH_DEPTH_PARAMS="$(python3 - "$INTRINSICS" "$MESH_DEPTH_INTRINSICS" <<'PY'
import json, sys
source = json.load(open(sys.argv[1], encoding='utf-8'))
calibration = source.get('P') or source['K']
output = {
    'image': source['image'],
    'K': {name: float(calibration[name]) for name in ('fx', 'fy', 'cx', 'cy')},
    'P': {name: float(calibration[name]) for name in ('fx', 'fy', 'cx', 'cy')},
    'distortion_model': 'none',
    'D': [0.0, 0.0, 0.0, 0.0, 0.0],
}
with open(sys.argv[2], 'w', encoding='utf-8') as handle:
    json.dump(output, handle, indent=2)
    handle.write('\n')
print(*(output['K'][name] for name in ('fx', 'fy', 'cx', 'cy')))
PY
)"
read -r MESH_DEPTH_FX MESH_DEPTH_FY MESH_DEPTH_CX MESH_DEPTH_CY \
  <<< "$MESH_DEPTH_PARAMS"
MESH_DEPTH_ID="$({
  sha256sum "$MESH_PATH" "$MESH_DEPTH_INTRINSICS" "$MESH_DEPTH_GENERATOR"
  printf 'pnp_max_error=12\n'
} | sha256sum | cut -c1-16)"
MESH_DEPTH_CACHE="$MESH_DEPTH_ROOT/$MESH_DEPTH_ID"
mkdir -p "$MESH_DEPTH_CACHE"

run_stage() {
  local name="$1"
  shift
  local start end status
  start="$(date +%s%N)"
  set +e
  /usr/bin/time -v -o "$LOG_DIR/$name.time.txt" \
    "$@" > "$LOG_DIR/$name.log" 2>&1
  status=$?
  set -e
  end="$(date +%s%N)"
  awk -v start="$start" -v end="$end" \
    'BEGIN { printf "%.9f\n", (end-start)/1000000000.0 }' \
    > "$LOG_DIR/$name.wall-seconds.txt"
  printf '%d\n' "$status" > "$LOG_DIR/$name.exit-code.txt"
  return "$status"
}

database_ready() {
  [[ -f "$DATABASE" ]] || return 1
  [[ "$(sqlite3 "$DATABASE" 'SELECT count(*) FROM images;')" -eq "$FRAME_COUNT" ]] || return 1
  [[ "$(sqlite3 "$DATABASE" 'SELECT count(*) FROM keypoints WHERE rows > 0;')" -eq "$FRAME_COUNT" ]] || return 1
  [[ "$(sqlite3 "$DATABASE" 'SELECT count(*) FROM two_view_geometries WHERE rows > 0;')" -gt 0 ]]
}

if ! database_ready; then
  [[ ! -e "$DATABASE" ]] || {
    printf 'Existing database is incomplete: %s\n' "$DATABASE" >&2
    exit 2
  }
  run_stage feature-extraction "$FRONTEND_BIN" feature_extractor \
    --database_path "$DATABASE" --image_path "$IMAGE_DIR" \
    --image_list_path "$IMAGE_LIST" --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --ImageReader.camera_params "$CAMERA_PARAMS" \
    --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0 \
    --SiftExtraction.num_threads 8 --SiftExtraction.max_image_size 612
  run_stage feature-matching "$FRONTEND_BIN" sequential_matcher \
    --database_path "$DATABASE" --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0 --SiftMatching.num_threads 8 \
    --SequentialMatching.overlap 15 \
    --SequentialMatching.quadratic_overlap 1 \
    --SequentialMatching.loop_detection 0
fi

LIDAR_BUILD_ID="voxel_leaf=0.01 normal_backend=$LIDAR_NORMAL_BACKEND normal_schema=v1 projection_normal_radius=$LIDAR_PROJECTION_NORMAL_RADIUS ba_normal_radius=$LIDAR_BA_NORMAL_RADIUS"
lidar_map_ready() {
  [[ -s "$LIDAR_MAP" && -s "$LIDAR_BA_MAP" &&
     -f "$LIDAR_BUILD_CONFIG" ]] || return 1
  [[ "$(<"$LIDAR_BUILD_CONFIG")" == "$LIDAR_BUILD_ID" ]]
}

if ! lidar_map_ready; then
  printf '%s\n' "${SCANS[@]}" > "$LIDAR_DIR/scans-manifest.txt"
  (
    cd "$LIDAR_DIR"
    run_stage lidar-merge pcl_concatenate_points_pcd "${SCANS[@]}"
    mv output.pcd merged-raw.pcd
  )
  run_stage lidar-voxel pcl_voxel_grid "$LIDAR_DIR/merged-raw.pcd" \
    "$LIDAR_DIR/global-1cm.pcd" -leaf 0.01,0.01,0.01
  if [[ "$LIDAR_NORMAL_BACKEND" == cuda ]]; then
    run_stage lidar-cuda-dual-normals "$CUDA_NORMAL_BIN" \
      --input "$LIDAR_DIR/global-1cm.pcd" \
      --outer_output "$LIDAR_DIR/global-1cm-projection-normals.pcd" \
      --inner_output "$LIDAR_DIR/global-1cm-ba-normals.pcd" \
      --outer_radius "$LIDAR_PROJECTION_NORMAL_RADIUS" \
      --inner_radius "$LIDAR_BA_NORMAL_RADIUS"
  else
    run_stage lidar-projection-normals pcl_normal_estimation \
      "$LIDAR_DIR/global-1cm.pcd" \
      "$LIDAR_DIR/global-1cm-projection-normals.pcd" \
      -radius "$LIDAR_PROJECTION_NORMAL_RADIUS"
    run_stage lidar-ba-normals pcl_normal_estimation \
      "$LIDAR_DIR/global-1cm.pcd" "$LIDAR_DIR/global-1cm-ba-normals.pcd" \
      -radius "$LIDAR_BA_NORMAL_RADIUS"
  fi
  run_stage lidar-projection-ply pcl_pcd2ply -format 1 -use_camera 0 \
    "$LIDAR_DIR/global-1cm-projection-normals.pcd" "$LIDAR_MAP"
  run_stage lidar-ba-ply pcl_pcd2ply -format 1 -use_camera 0 \
    "$LIDAR_DIR/global-1cm-ba-normals.pcd" "$LIDAR_BA_MAP"
  printf '%s\n' "$LIDAR_BUILD_ID" > "$LIDAR_BUILD_CONFIG"
  rm -f "$LIDAR_DIR/merged-raw.pcd" "$LIDAR_DIR/global-1cm.pcd" \
    "$LIDAR_DIR/global-1cm-projection-normals.pcd" \
    "$LIDAR_DIR/global-1cm-ba-normals.pcd"
fi

if [[ ! -s "$POSE_PRIOR" ]]; then
  run_stage pose-prior python3 \
    "$REPO_ROOT/scripts/python/fastlio_segmented_session.py" pose-prior \
    --session-dir "$SESSION_DIR" --output "$POSE_PRIOR"
fi

models_ready() {
  find -L "$SPARSE" -mindepth 1 -maxdepth 2 -type f -name images.bin \
    -print -quit | grep -q .
}

if ! models_ready; then
  run_stage mapper "$MAPPER_BIN" mapper \
    --database_path "$DATABASE" --image_path "$IMAGE_DIR" \
    --image_list_path "$IMAGE_LIST" --output_path "$SPARSE" \
    --Mapper.init_image_id1 -1 --Mapper.init_image_id2 -1 \
    --Mapper.if_import_pose_prior 1 \
    --Mapper.image_pose_prior_path "$POSE_PRIOR" \
    --Mapper.if_add_lidar_constraint 1 \
    --Mapper.lidar_pointcloud_path "$LIDAR_MAP" \
    --Mapper.lidar_ba_pointcloud_path "$LIDAR_BA_MAP" \
    --Mapper.initial_mesh_depth_path "$MESH_DEPTH_CACHE" \
    --Mapper.initial_mesh_depth_generator_path "$MESH_DEPTH_GENERATOR" \
    --Mapper.initial_mesh_path "$MESH_PATH" \
    --Mapper.initial_mesh_depth_dataset_path "$SESSION_DIR" \
    --Mapper.initial_mesh_depth_intrinsics_path "$MESH_DEPTH_INTRINSICS" \
    --Mapper.initial_mesh_depth_fx "$MESH_DEPTH_FX" \
    --Mapper.initial_mesh_depth_fy "$MESH_DEPTH_FY" \
    --Mapper.initial_mesh_depth_cx "$MESH_DEPTH_CX" \
    --Mapper.initial_mesh_depth_cy "$MESH_DEPTH_CY" \
    --Mapper.initial_mesh_depth_pnp_max_error 12 \
    --Mapper.if_add_lidar_corresponding 1 \
    --Mapper.if_add_lidar_display 0 --Mapper.multiple_models 1 \
    --Mapper.max_num_models 20 --Mapper.min_model_size 10 \
    --Mapper.extract_colors 1 --Mapper.num_threads 8 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.icp_lidar_constraint_weight 10 \
    --Mapper.icp_ground_lidar_constraint_weight 10 \
    --Mapper.kdtree_max_search_range 0.15 \
    --Mapper.kdtree_min_search_range 0.05 \
    --Mapper.search_range_drop_speed 0.01 \
    --Mapper.local_lidar_kdtree_only 1 \
    --Mapper.ba_local_max_refinements 1 \
    --Mapper.ba_global_images_freq 10 \
    --Mapper.non_ba_profile 1 \
    --Mapper.non_ba_profile_path "$OUTPUT_ROOT/nonba-profile.json" \
    --Mapper.ba_backend custom_cuda --Mapper.ba_fallback_to_ceres 0 \
    --Mapper.ba_snapshot_capture none \
    --Mapper.ba_lidar_residual legacy_exact \
    --Mapper.ba_cuda_execution_profile compact_control \
    --Mapper.ba_cuda_audit_profile production \
    --Mapper.ba_cuda_arithmetic_precision fp32_mixed \
    --Mapper.ba_cuda_hessian_assembly_backend observation_segmented \
    --Mapper.ba_cuda_hot_kernel_mode transformed \
    --Mapper.ba_cuda_schur_contribution_backend segmented \
    --Mapper.ba_cuda_host_problem_store host_prepared_store \
    --Mapper.ba_cuda_problem_source native_graph \
    --Mapper.ba_cuda_prepared_selection_cache 0 \
    --Mapper.ba_telemetry_path "$OUTPUT_ROOT/telemetry.jsonl"
fi

optimized_ready() {
  [[ -f "$OPTIMIZED_REPORT" ]] || return 1
  [[ "$(find -L "$OPTIMIZED_SESSION" -maxdepth 1 -name 'imgs_*.jpg' | wc -l)" -eq "$FRAME_COUNT" ]] || return 1
  [[ "$(find -L "$OPTIMIZED_SESSION" -maxdepth 1 -name 'odoms_*.txt' | wc -l)" -eq "$FRAME_COUNT" ]]
}

if ! optimized_ready; then
  [[ ! -e "$OPTIMIZED_SESSION" ]] || {
    printf 'Existing optimized session is incomplete: %s\n' "$OPTIMIZED_SESSION" >&2
    exit 2
  }
  run_stage stage-session python3 \
    "$REPO_ROOT/scripts/python/fastlio_segmented_session.py" stage-session \
    --session-dir "$SESSION_DIR" --models-root "$SPARSE" \
    --output-dir "$OPTIMIZED_SESSION" --report "$OPTIMIZED_REPORT" \
    --intrinsics "$INTRINSICS"
fi

python3 - "$OUTPUT_ROOT" "$FRAME_COUNT" "$FRONTEND_BIN" "$MAPPER_BIN" \
  "$MESH_PATH" "$MESH_DEPTH_GENERATOR" "$MESH_DEPTH_CACHE" <<'PY'
import hashlib, json, sys
from pathlib import Path

root, frame_count = Path(sys.argv[1]), int(sys.argv[2])
report = json.load(open(root / 'optimized-session-report.json', encoding='utf-8'))
def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def seconds(name):
    path = root / 'logs' / (name + '.wall-seconds.txt')
    return float(path.read_text()) if path.is_file() else None
summary = {
    'schema': 'fastlio_colmap_pcd_session_run_v1',
    'source_session': str(Path(root / 'source-session.txt').read_text().strip()),
    'frame_count': frame_count,
    'optimized_frame_count': report['optimized_frame_count'],
    'fallback_frame_count': report['fallback_frame_count'],
    'optimized_ranges': report['optimized_ranges'],
    'fallback_ranges': report['fallback_ranges'],
    'model_validation_policy': report['model_validation_policy'],
    'model_validation': report['model_validation'],
    'pose_update': report['pose_update'],
    'pose_update_gate': report['pose_update_gate'],
    'stage_seconds': {name: seconds(name) for name in (
        'feature-extraction', 'feature-matching', 'lidar-merge',
        'lidar-voxel', 'lidar-projection-normals',
        'lidar-projection-ply', 'lidar-ba-normals', 'lidar-ba-ply',
        'pose-prior',
        'mapper', 'stage-session')},
    'frontend_binary': {'path': sys.argv[3], 'sha256': sha256(sys.argv[3])},
    'mapper_binary': {'path': sys.argv[4], 'sha256': sha256(sys.argv[4])},
    'initial_mesh': {'path': sys.argv[5], 'sha256': sha256(sys.argv[5])},
    'mesh_depth_generator': {
        'path': sys.argv[6], 'sha256': sha256(sys.argv[6])},
    'initial_mesh_depth_cache': sys.argv[7],
    'initial_depth_usage': 'initial_pair_only',
    'optimized_session': str((root / 'optimized-session').resolve()),
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
PY

touch "$OUTPUT_ROOT/COMPLETE"
printf 'OPTIMIZED_SESSION=%s\n' "$OPTIMIZED_SESSION"
