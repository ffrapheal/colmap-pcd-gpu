#!/usr/bin/env bash
# Prepare or execute a sealed online i3dgs replay.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPARE_SCRIPT="$REPO_ROOT/scripts/python/prepare_online_mapper_replay.py"
SESSION_DIR="$REPO_ROOT/Data/20260820-111812"
ARTIFACT_DIR=""
SEALED_ARTIFACT=""
OUTPUT_ARTIFACT=""
INTRINSICS="$REPO_ROOT/config/fastlio_camera_1224x1024.json"
resolve_command() {
  local command_name="$1"
  local fallback="$2"
  command -v "$command_name" 2>/dev/null || printf '%s' "$fallback"
}
FRONTEND_BINARY="${COLMAP_PCD_FRONTEND_BIN:-$(resolve_command colmap "$REPO_ROOT/build/src/exe/colmap")}"
MAPPER_BINARY="${COLMAP_PCD_MAPPER_BIN:-$REPO_ROOT/build/src/exe/colmap}"
TEXRECON_BINARY="${COLMAP_PCD_TEXRECON_BIN:-$(resolve_command texrecon texrecon)}"
I3DGS_DIR="$REPO_ROOT/thirdpart/i3dgs"
EXPECTED_I3DGS_COMMIT="cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42"
EXPECTED_FRAME_COUNT=""
PREPARE_ONLY=0
EXECUTE=0
MAX_FRAMES=""
BA_WINDOW_SIZE=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --prepare-only --artifact-dir NEW_PHASE1_DIR [options]
  $(basename "$0") --execute --sealed-artifact PHASE1_DIR \\
    --output-artifact NEW_OUTPUT_DIR [--max-frames N]

Options:
  --session-dir DIR              Input session (default: $SESSION_DIR)
  --artifact-dir DIR             Required; parent must exist, DIR must not exist
  --intrinsics FILE              Final camera intrinsics JSON
  --frontend-binary FILE         COLMAP feature frontend binary
  --mapper-binary FILE           COLMAP mapper binary to archive
  --texrecon-binary FILE         Texrecon binary to archive
  --i3dgs-dir DIR                Read-only i3dgs reference checkout
  --expected-frame-count N       Required frame count (default session: 246)
  --expected-i3dgs-commit SHA    Required i3dgs reference commit
  --prepare-only                 Seal inputs and write causal ARRIVED events
  --execute                      Replay an existing sealed Phase 1 artifact
  --sealed-artifact DIR          Existing Phase 1 artifact for --execute
  --output-artifact DIR          Required new replay artifact for --execute
  --max-frames N                 Positive smoke limit; omitted means all frames
  --ba-window-size N             BA window cap, 2..20 (default: 20)
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    printf 'Missing value for %s\n' "$1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir)
      require_value "$@"
      SESSION_DIR="$2"
      shift 2
      ;;
    --artifact-dir)
      require_value "$@"
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --sealed-artifact)
      require_value "$@"
      SEALED_ARTIFACT="$2"
      shift 2
      ;;
    --output-artifact|--output-path)
      require_value "$@"
      OUTPUT_ARTIFACT="$2"
      shift 2
      ;;
    --intrinsics)
      require_value "$@"
      INTRINSICS="$2"
      shift 2
      ;;
    --frontend-binary)
      require_value "$@"
      FRONTEND_BINARY="$2"
      shift 2
      ;;
    --mapper-binary)
      require_value "$@"
      MAPPER_BINARY="$2"
      shift 2
      ;;
    --texrecon-binary)
      require_value "$@"
      TEXRECON_BINARY="$2"
      shift 2
      ;;
    --i3dgs-dir)
      require_value "$@"
      I3DGS_DIR="$2"
      shift 2
      ;;
    --expected-frame-count)
      require_value "$@"
      EXPECTED_FRAME_COUNT="$2"
      shift 2
      ;;
    --expected-i3dgs-commit)
      require_value "$@"
      EXPECTED_I3DGS_COMMIT="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --max-frames)
      require_value "$@"
      MAX_FRAMES="$2"
      shift 2
      ;;
    --ba-window-size)
      require_value "$@"
      BA_WINDOW_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ $((PREPARE_ONLY + EXECUTE)) == 1 ]] || {
  printf 'Select exactly one of --prepare-only or --execute.\n' >&2
  exit 2
}
if [[ "$EXECUTE" == 1 ]]; then
  [[ -z "$ARTIFACT_DIR" ]] || {
    printf -- '--artifact-dir is only valid with --prepare-only.\n' >&2
    exit 2
  }
  [[ -n "$SEALED_ARTIFACT" && -d "$SEALED_ARTIFACT" ]] || {
    printf -- '--sealed-artifact must name an existing Phase 1 directory.\n' >&2
    exit 2
  }
  [[ -n "$OUTPUT_ARTIFACT" ]] || {
    printf -- '--output-artifact is required and must name a new path.\n' >&2
    exit 2
  }
  [[ ! -e "$OUTPUT_ARTIFACT" && ! -L "$OUTPUT_ARTIFACT" ]] || {
    printf 'Output artifact already exists; refusing to overwrite: %s\n' \
      "$OUTPUT_ARTIFACT" >&2
    exit 2
  }
  [[ -x "$MAPPER_BINARY" ]] || {
    printf 'Mapper binary is not executable: %s\n' "$MAPPER_BINARY" >&2
    exit 2
  }
  if [[ -n "$MAX_FRAMES" && ! "$MAX_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
    printf -- '--max-frames must be a positive integer.\n' >&2
    exit 2
  fi
  execute_arguments=(
    online_i3dgs_mapper
    --sealed_artifact "$SEALED_ARTIFACT"
    --output_path "$OUTPUT_ARTIFACT"
  )
  if [[ -n "$MAX_FRAMES" ]]; then
    execute_arguments+=(--max_frames "$MAX_FRAMES")
  fi
  if [[ -n "$BA_WINDOW_SIZE" ]]; then
    [[ "$BA_WINDOW_SIZE" =~ ^([2-9]|1[0-9]|20)$ ]] || {
      printf -- '--ba-window-size must be an integer in [2, 20].\n' >&2
      exit 2
    }
    execute_arguments+=(--ba_window_size "$BA_WINDOW_SIZE")
  fi
  exec "$MAPPER_BINARY" "${execute_arguments[@]}"
fi

[[ -z "$SEALED_ARTIFACT" && -z "$OUTPUT_ARTIFACT" && -z "$MAX_FRAMES" && -z "$BA_WINDOW_SIZE" ]] || {
  printf -- '--sealed-artifact, --output-artifact, --max-frames, and --ba-window-size require --execute.\n' >&2
  exit 2
}
[[ -n "$ARTIFACT_DIR" ]] || {
  printf -- '--artifact-dir is required and must name a new path.\n' >&2
  exit 2
}
[[ ! -e "$ARTIFACT_DIR" && ! -L "$ARTIFACT_DIR" ]] || {
  printf 'Artifact path already exists; refusing to overwrite: %s\n' \
    "$ARTIFACT_DIR" >&2
  exit 2
}
command -v python3 >/dev/null || {
  printf 'Missing required command: python3\n' >&2
  exit 2
}
for binary in "$FRONTEND_BINARY" "$MAPPER_BINARY" "$TEXRECON_BINARY"; do
  [[ -x "$binary" ]] || {
    printf 'Required executable is missing or not executable: %s\n' "$binary" >&2
    exit 2
  }
done

if [[ -z "$EXPECTED_FRAME_COUNT" &&
      "$(readlink -f "$SESSION_DIR")" == "$(readlink -f "$REPO_ROOT/Data/20260820-111812")" ]]; then
  EXPECTED_FRAME_COUNT=246
fi
if [[ -n "$EXPECTED_FRAME_COUNT" && ! "$EXPECTED_FRAME_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  printf -- '--expected-frame-count must be a positive integer.\n' >&2
  exit 2
fi

arguments=(
  --prepare-only
  --session-dir "$SESSION_DIR"
  --artifact-dir "$ARTIFACT_DIR"
  --intrinsics "$INTRINSICS"
  --frontend-binary "$FRONTEND_BINARY"
  --mapper-binary "$MAPPER_BINARY"
  --texrecon-binary "$TEXRECON_BINARY"
  --repo-root "$REPO_ROOT"
  --i3dgs-dir "$I3DGS_DIR"
  --expected-i3dgs-commit "$EXPECTED_I3DGS_COMMIT"
)
if [[ -n "$EXPECTED_FRAME_COUNT" ]]; then
  arguments+=(--expected-frame-count "$EXPECTED_FRAME_COUNT")
fi

python3 "$PREPARE_SCRIPT" "${arguments[@]}"
printf 'Phase 1 artifact sealed at %s\n' "$ARTIFACT_DIR"
