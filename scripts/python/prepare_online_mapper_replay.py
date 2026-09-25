#!/usr/bin/env python3
"""Seal an indexed FAST-LIO session and emit a causal Phase 1 replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from fastlio_segmented_session import (
    FASTLIO_TO_COLMAP_WORLD,
    quaternion_to_rotation,
    read_odom,
    rotation_to_yxz,
    write_pose_prior,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SESSION = REPO_ROOT / "Data/20260820-111812"
DEFAULT_INTRINSICS = REPO_ROOT / "config/fastlio_camera_1224x1024.json"


def default_executable(
    environment_variable: str, command: str, fallback: Path
) -> Path:
    configured_path = os.environ.get(environment_variable)
    if configured_path:
        return Path(configured_path).expanduser()
    executable_path = shutil.which(command)
    return Path(executable_path) if executable_path else fallback


DEFAULT_FRONTEND_BINARY = default_executable(
    "COLMAP_PCD_FRONTEND_BIN", "colmap", REPO_ROOT / "build/src/exe/colmap"
)
DEFAULT_MAPPER_BINARY = Path(
    os.environ.get(
        "COLMAP_PCD_MAPPER_BIN", str(REPO_ROOT / "build/src/exe/colmap")
    )
).expanduser()
DEFAULT_TEXRECON_BINARY = default_executable(
    "COLMAP_PCD_TEXRECON_BIN", "texrecon", Path("texrecon")
)
DEFAULT_I3DGS_DIR = REPO_ROOT / "thirdpart/i3dgs"
EXPECTED_I3DGS_COMMIT = "cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42"
DEFAULT_FRAME_COUNT = 246

FRAME_PATTERNS = {
    "jpg": re.compile(r"imgs_(\d+)\.jpg", re.IGNORECASE),
    "cam": re.compile(r"imgs_(\d+)\.CAM", re.IGNORECASE),
    "odom": re.compile(r"odoms_(\d+)\.txt", re.IGNORECASE),
    "scan": re.compile(r"scans_(\d+)\.pcd", re.IGNORECASE),
}
REQUIRED_SESSION_ASSETS = (
    "mesh_0.ply",
    "obj1/mesh_hole_filled.ply",
    "voxel_cloud_0.ply",
    "voxel_data.bin",
    "voxel_config.json",
)
KEY_SOURCE_PATHS = (
    ".gitmodules",
    "lib/SiftGPU/SiftGPU.cpp",
    "lib/SiftGPU/SiftGPU.h",
    "scripts/python/fastlio_segmented_session.py",
    "scripts/python/prepare_online_mapper_replay.py",
    "scripts/python/test_prepare_online_mapper_replay.py",
    "scripts/run_fastlio_session.sh",
    "scripts/run_online_i3dgs_mapper_session.sh",
    "src/CMakeLists.txt",
    "src/base/correspondence_graph.cc",
    "src/base/correspondence_graph.h",
    "src/base/correspondence_graph_test.cc",
    "src/base/database_cache.cc",
    "src/base/database_cache.h",
    "src/base/database_cache_test.cc",
    "src/base/point3d.h",
    "src/base/reconstruction.cc",
    "src/base/reconstruction.h",
    "src/base/reconstruction_test.cc",
    "src/controllers/CMakeLists.txt",
    "src/controllers/incremental_mapper.cc",
    "src/controllers/incremental_mapper.h",
    "src/controllers/online_i3dgs_mapper.cc",
    "src/controllers/online_i3dgs_mapper.h",
    "src/controllers/online_i3dgs_mapper_test.cc",
    "src/controllers/online_mapper_frontend.cc",
    "src/controllers/online_mapper_frontend.h",
    "src/controllers/online_mapper_frontend_test.cc",
    "src/controllers/online_mapper_state.cc",
    "src/controllers/online_mapper_state.h",
    "src/controllers/online_mapper_state_test.cc",
    "src/exe/CMakeLists.txt",
    "src/exe/colmap.cc",
    "src/exe/online_i3dgs_replay.cc",
    "src/exe/online_i3dgs_replay.h",
    "src/exe/online_i3dgs_replay_test.cc",
    "src/exe/sfm.cc",
    "src/exe/sfm.h",
    "src/feature/CMakeLists.txt",
    "src/feature/extraction.cc",
    "src/feature/extraction.h",
    "src/feature/incremental_matching_test.cc",
    "src/feature/matching.cc",
    "src/feature/matching.h",
    "src/feature/sift.cc",
    "src/feature/single_image_sift_extractor_test.cc",
    "src/gpu_ba/CMakeLists.txt",
    "src/gpu_ba/custom_cuda.cu",
    "src/gpu_ba/custom_cuda_test.cc",
    "src/gpu_ba/host_ba_graph.cc",
    "src/gpu_ba/host_ba_graph.h",
    "src/gpu_ba/host_ba_graph_test.cc",
    "src/gpu_ba/native_cuda_bridge_test.cc",
    "src/gpu_ba/native_graph_problem_store.cc",
    "src/gpu_ba/online_health.cc",
    "src/gpu_ba/online_health.h",
    "src/gpu_ba/online_health_test.cc",
    "src/lidar/CMakeLists.txt",
    "src/lidar/cuda_normal_estimation.cu",
    "src/lidar/cuda_normal_estimation.h",
    "src/lidar/cuda_normal_estimation_test.cc",
    "src/lidar/incremental_causal_lidar_map.cc",
    "src/lidar/incremental_causal_lidar_map.h",
    "src/lidar/incremental_causal_lidar_map_test.cc",
    "src/lidar/pcd_projection.cc",
    "src/lidar/pcd_projection.h",
    "src/lidar/pcd_projection_test.cc",
    "src/sfm/CMakeLists.txt",
    "src/sfm/active_covisibility_graph.cc",
    "src/sfm/active_covisibility_graph.h",
    "src/sfm/active_covisibility_graph_test.cc",
    "src/sfm/incremental_mapper.cc",
    "src/sfm/incremental_mapper.h",
    "src/sfm/incremental_mapper_test.cc",
    "src/sfm/incremental_triangulator.cc",
    "src/sfm/incremental_triangulator.h",
    "src/sfm/online_dual_selection.cc",
    "src/sfm/online_dual_selection.h",
    "src/sfm/online_dual_selection_test.cc",
    "src/sfm/online_lidar_association.cc",
    "src/sfm/online_lidar_association.h",
    "src/sfm/online_lidar_association_test.cc",
    "src/sfm/online_lidar_ba_intent.cc",
    "src/sfm/online_lidar_ba_intent.h",
    "src/sfm/online_lidar_ba_intent_test.cc",
    "src/sfm/online_local_ba_executor.cc",
    "src/sfm/online_local_ba_executor.h",
    "src/sfm/online_local_ba_executor_test.cc",
    "src/sfm/online_local_ba_postprocess.cc",
    "src/sfm/online_local_ba_postprocess.h",
    "src/sfm/online_local_ba_postprocess_test.cc",
    "src/util/option_manager.cc",
)
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


class ReplayPreparationError(RuntimeError):
    """Raised when an input cannot be sealed for replay."""


class InputChangedError(ReplayPreparationError):
    """Raised when a sealed input changes before replay completion."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    real_path: Path
    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str
    path_entry_identity: Tuple[int, ...]
    path_is_symlink: bool
    reject_symlink: bool

    def to_json(self) -> dict:
        return {
            "path": str(self.path),
            "real_path": str(self.real_path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "mtime_utc": datetime.fromtimestamp(
                self.mtime_ns / 1_000_000_000, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "ctime_ns": self.ctime_ns,
            "ctime_utc": datetime.fromtimestamp(
                self.ctime_ns / 1_000_000_000, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mode": stat.filemode(self.mode),
            "mode_octal": oct(stat.S_IMODE(self.mode)),
            "path_is_symlink": self.path_is_symlink,
        }


@dataclass(frozen=True)
class ReplayOptions:
    session_dir: Path
    artifact_dir: Path
    intrinsics_path: Path
    frontend_binary: Path
    mapper_binary: Path
    texrecon_binary: Path
    repo_root: Path = REPO_ROOT
    i3dgs_dir: Path = DEFAULT_I3DGS_DIR
    expected_frame_count: int | None = None
    expected_i3dgs_commit: str = EXPECTED_I3DGS_COMMIT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_stat_tuple(file_stat: os.stat_result) -> Tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def snapshot_file(path: Path, *, reject_symlink: bool = False) -> FileSnapshot:
    absolute_path = path.expanduser().absolute()
    try:
        entry_before = absolute_path.lstat()
        if reject_symlink and stat.S_ISLNK(entry_before.st_mode):
            raise ReplayPreparationError(
                f"Symlinks are not allowed for sealed session inputs: {absolute_path}"
            )
        real_path = absolute_path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ReplayPreparationError(f"Missing required file: {absolute_path}") from error

    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if reject_symlink:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute_path, flags)
    except OSError as error:
        if reject_symlink and absolute_path.is_symlink():
            raise ReplayPreparationError(
                f"Symlinks are not allowed for sealed session inputs: {absolute_path}"
            ) from error
        raise
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ReplayPreparationError(f"Required path is not a file: {absolute_path}")
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(handle.fileno())
    try:
        entry_after = absolute_path.lstat()
        current_real_path = absolute_path.resolve(strict=True)
        current_target = absolute_path.stat()
    except (FileNotFoundError, RuntimeError) as error:
        raise InputChangedError(f"File changed while hashing: {absolute_path}") from error
    if (
        stable_stat_tuple(before) != stable_stat_tuple(after)
        or stable_stat_tuple(entry_before) != stable_stat_tuple(entry_after)
        or stable_stat_tuple(after) != stable_stat_tuple(current_target)
        or real_path != current_real_path
    ):
        raise InputChangedError(f"File changed while hashing: {absolute_path}")
    return FileSnapshot(
        path=absolute_path,
        real_path=real_path,
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        sha256=digest.hexdigest(),
        path_entry_identity=stable_stat_tuple(entry_after),
        path_is_symlink=stat.S_ISLNK(entry_after.st_mode),
        reject_symlink=reject_symlink,
    )


def snapshot_identity(snapshot: FileSnapshot) -> dict:
    return {
        "path": str(snapshot.path),
        "real_path": str(snapshot.real_path),
        "device": snapshot.device,
        "inode": snapshot.inode,
        "mode": snapshot.mode,
        "size_bytes": snapshot.size_bytes,
        "mtime_ns": snapshot.mtime_ns,
        "ctime_ns": snapshot.ctime_ns,
        "sha256": snapshot.sha256,
        "path_entry_identity": list(snapshot.path_entry_identity),
        "path_is_symlink": snapshot.path_is_symlink,
    }


def snapshot_set_sha256(snapshots: Sequence[FileSnapshot]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                snapshot_identity(snapshot)
                for snapshot in sorted(snapshots, key=lambda item: str(item.path))
            ]
        )
    )


def verify_snapshots_unchanged(snapshots: Sequence[FileSnapshot]) -> dict:
    current_snapshots = []
    for expected in sorted(snapshots, key=lambda item: str(item.path)):
        current = snapshot_file(
            expected.path, reject_symlink=expected.reject_symlink
        )
        if snapshot_identity(current) != snapshot_identity(expected):
            raise InputChangedError(f"Sealed input changed during replay: {expected.path}")
        current_snapshots.append(current)
    return {
        "verified_file_count": len(current_snapshots),
        "snapshot_set_sha256": snapshot_set_sha256(current_snapshots),
    }


def session_inventory_paths(session_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for root, directory_names, file_names in os.walk(session_dir, followlinks=False):
        directory_names.sort()
        file_names.sort()
        root_path = Path(root)
        for directory_name in directory_names:
            directory_path = root_path / directory_name
            if directory_path.is_symlink():
                raise ReplayPreparationError(
                    f"Session contains a directory symlink: {directory_path}"
                )
        for file_name in file_names:
            path = root_path / file_name
            if path.is_symlink():
                raise ReplayPreparationError(
                    f"Session contains a file symlink: {path}"
                )
            if not path.is_file():
                raise ReplayPreparationError(f"Session entry is not a file: {path}")
            try:
                path.resolve(strict=True).relative_to(session_dir)
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                raise ReplayPreparationError(
                    f"Session file escapes the sealed session: {path}"
                ) from error
            paths.append(path.absolute())
    return sorted(paths, key=lambda path: path.relative_to(session_dir).as_posix())


def relative_inventory(paths: Iterable[Path], session_dir: Path) -> List[str]:
    return [path.relative_to(session_dir).as_posix() for path in paths]


def verify_session_inventory(session_dir: Path, expected: Sequence[str]) -> None:
    current = relative_inventory(session_inventory_paths(session_dir), session_dir)
    if current != list(expected):
        expected_set = set(expected)
        current_set = set(current)
        raise InputChangedError(
            "Session inventory changed during replay: "
            f"added={sorted(current_set - expected_set)}, "
            f"removed={sorted(expected_set - current_set)}"
        )


def discover_indexed_frames(session_dir: Path) -> List[dict]:
    indexed: Dict[str, Dict[int, Path]] = {name: {} for name in FRAME_PATTERNS}
    for path in session_dir.iterdir():
        if not path.is_file():
            continue
        for kind, pattern in FRAME_PATTERNS.items():
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            frame_index = int(match.group(1))
            previous = indexed[kind].get(frame_index)
            if previous is not None:
                raise ReplayPreparationError(
                    f"Duplicate {kind} frame index {frame_index}: {previous} and {path}"
                )
            indexed[kind][frame_index] = path.absolute()
            break

    if not indexed["jpg"]:
        raise ReplayPreparationError(f"No imgs_N.jpg files under {session_dir}")

    image_indices = set(indexed["jpg"])
    for kind in ("cam", "odom", "scan"):
        kind_indices = set(indexed[kind])
        if kind_indices != image_indices:
            raise ReplayPreparationError(
                f"Frame index mismatch for {kind}: "
                f"missing={sorted(image_indices - kind_indices)}, "
                f"unexpected={sorted(kind_indices - image_indices)}"
            )

    ordered_indices = sorted(image_indices)
    expected_indices = list(range(1, ordered_indices[-1] + 1))
    if ordered_indices != expected_indices:
        raise ReplayPreparationError(
            "Frame indices must be contiguous and start at 1: "
            f"found={ordered_indices}, "
            f"missing={sorted(set(expected_indices) - image_indices)}"
        )

    return [
        {
            "frame_index": frame_index,
            **{kind: indexed[kind][frame_index] for kind in FRAME_PATTERNS},
        }
        for frame_index in ordered_indices
    ]


def jpeg_dimensions(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ReplayPreparationError(f"Not a JPEG image: {path}")
        while True:
            prefix = handle.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue
            marker_bytes = handle.read(1)
            while marker_bytes == b"\xff":
                marker_bytes = handle.read(1)
            if not marker_bytes:
                break
            marker = marker_bytes[0]
            if marker in {0x01, *range(0xD0, 0xD9)}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ReplayPreparationError(f"Invalid JPEG segment in {path}")
            if marker in JPEG_SOF_MARKERS:
                dimensions = handle.read(5)
                if len(dimensions) != 5:
                    break
                height, width = struct.unpack(">HH", dimensions[1:5])
                if width <= 0 or height <= 0:
                    raise ReplayPreparationError(f"Invalid JPEG dimensions in {path}")
                return width, height
            if marker == 0xDA:
                break
            handle.seek(segment_length - 2, os.SEEK_CUR)
    raise ReplayPreparationError(f"JPEG dimensions not found: {path}")


def finite_floats(tokens: Sequence[str], path: Path, expected: int) -> List[float]:
    if len(tokens) != expected:
        raise ReplayPreparationError(
            f"Expected {expected} values in {path}, found {len(tokens)}"
        )
    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ReplayPreparationError(f"Non-numeric value in {path}") from error
    if not all(math.isfinite(value) for value in values):
        raise ReplayPreparationError(f"Non-finite value in {path}")
    return values


def read_cam(path: Path) -> dict:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ReplayPreparationError(f"Expected two non-empty CAM lines in {path}")
    extrinsics = finite_floats(lines[0], path, 12)
    calibration = finite_floats(lines[1], path, 6)
    return {
        "format": "MVE_CAM",
        "world_to_camera_3x4": [extrinsics[offset : offset + 4] for offset in range(0, 12, 4)],
        "calibration_values": calibration,
    }


def read_intrinsics(path: Path) -> dict:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
        width = int(source["image"]["width"])
        height = int(source["image"]["height"])
        intrinsic = source["K"]
        distortion = [float(value) for value in source["D"]]
        params = [
            float(intrinsic["fx"]),
            float(intrinsic["fy"]),
            float(intrinsic["cx"]),
            float(intrinsic["cy"]),
            *distortion[:4],
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayPreparationError(f"Invalid intrinsics JSON: {path}") from error
    if width <= 0 or height <= 0 or len(distortion) < 4:
        raise ReplayPreparationError(f"Incomplete intrinsics JSON: {path}")
    if not all(math.isfinite(value) for value in params):
        raise ReplayPreparationError(f"Non-finite intrinsics in {path}")
    return {
        "model": "OPENCV",
        "width": width,
        "height": height,
        "params": params,
        "param_order": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "source_distortion_model": source.get("distortion_model"),
        "source_distortion": distortion,
    }


def matrix4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def convert_fastlio_pose(path: Path) -> dict:
    source_tokens = path.read_text(encoding="utf-8").split()
    source_values = finite_floats(source_tokens, path, 8)
    try:
        timestamp, normalized = read_odom(path)
    except RuntimeError as error:
        raise ReplayPreparationError(f"Invalid FAST-LIO odometry: {error}") from error
    center_fastlio = normalized[1:4]
    quaternion_fastlio = normalized[4:8]
    rotation_wc_fastlio = quaternion_to_rotation(quaternion_fastlio)
    world_conversion = np.asarray(FASTLIO_TO_COLMAP_WORLD, dtype=np.float64)
    center_colmap = world_conversion @ center_fastlio
    rotation_wc_colmap = world_conversion @ rotation_wc_fastlio
    rotation_cw_colmap = rotation_wc_colmap.T
    translation_cw_colmap = -rotation_cw_colmap @ center_colmap
    fastlio_t_wc = matrix4(rotation_wc_fastlio, center_fastlio)
    colmap_t_cw = matrix4(rotation_cw_colmap, translation_cw_colmap)
    if not np.isfinite(fastlio_t_wc).all() or not np.isfinite(colmap_t_cw).all():
        raise ReplayPreparationError(f"Non-finite converted pose in {path}")

    y_angle, x_angle, z_angle = rotation_to_yxz(rotation_wc_colmap)
    return {
        "timestamp": timestamp,
        "source_values": source_values,
        "fastlio": {
            "convention": "T_wc",
            "translation_m": center_fastlio.tolist(),
            "source_quaternion_wxyz": source_values[4:8],
            "normalized_quaternion_wxyz": quaternion_fastlio.tolist(),
            "source_quaternion_norm": float(np.linalg.norm(source_values[4:8])),
            "T_wc": fastlio_t_wc.tolist(),
        },
        "colmap_prior": {
            "convention": "T_cw",
            "camera_center_world_m": center_colmap.tolist(),
            "rotation_cw": rotation_cw_colmap.tolist(),
            "translation_cw_m": translation_cw_colmap.tolist(),
            "T_cw": colmap_t_cw.tolist(),
            "pose_prior_ply_values": [
                float(normalized[1]),
                float(normalized[2]),
                float(normalized[3]),
                z_angle,
                -x_angle,
                -y_angle,
            ],
        },
    }


def run_git(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ReplayPreparationError(
            f"Git metadata query failed for {repository}: {detail or error}"
        ) from error
    return result.stdout


def parse_git_status(raw_status: bytes) -> List[dict]:
    entries = []
    for line in raw_status.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        entries.append({"status": line[:2], "path": line[3:]})
    return entries


def capture_repository(repository: Path, key_paths: Sequence[str] = ()) -> dict:
    commit = run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    tree = run_git(repository, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    status_raw = run_git(repository, "status", "--porcelain=v1", "--untracked-files=normal")
    diff_command = ["diff", "--binary", "HEAD"]
    stat_command = ["diff", "--stat", "HEAD"]
    numstat_command = ["diff", "--numstat", "HEAD"]
    key_status_raw = b""
    if key_paths:
        diff_command.extend(["--", *key_paths])
        stat_command.extend(["--", *key_paths])
        numstat_command.extend(["--", *key_paths])
        key_status_raw = run_git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *key_paths,
        )
    diff_raw = run_git(repository, *diff_command)
    diff_stat = run_git(repository, *stat_command).decode("utf-8", errors="replace")
    numstat_raw = run_git(repository, *numstat_command).decode("utf-8", errors="replace")
    numstat = []
    for line in numstat_raw.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3:
            numstat.append({"added": fields[0], "deleted": fields[1], "path": fields[2]})
    return {
        "path": str(repository),
        "commit": commit,
        "tree": tree,
        "dirty": bool(status_raw),
        "status": parse_git_status(status_raw),
        "status_sha256": sha256_bytes(status_raw),
        "key_source_status": parse_git_status(key_status_raw),
        "key_source_status_sha256": sha256_bytes(key_status_raw),
        "key_source_diff_sha256": sha256_bytes(diff_raw),
        "key_source_diff_stat": diff_stat,
        "key_source_diff_numstat": numstat,
    }


def capture_i3dgs(repository: Path) -> dict:
    metadata = capture_repository(repository)
    try:
        remote = run_git(repository, "remote", "get-url", "origin").decode("utf-8").strip()
    except ReplayPreparationError:
        remote = None
    metadata["origin"] = remote
    return metadata


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def staged_output_path(path: Path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        yield temporary_path
        descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary_path, path, follow_symlinks=False)
        published = True
        fsync_directory(path.parent)
    except BaseException:
        if published:
            try:
                path.unlink()
                fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_text_exclusive(path: Path, text: str) -> None:
    with staged_output_path(path) as temporary_path:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())


def write_json_exclusive(path: Path, value: object) -> None:
    write_text_exclusive(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def create_artifact_dir(path: Path, forbidden_roots: Sequence[Path]) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    try:
        parent = requested.parent.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ReplayPreparationError(
            f"Artifact parent must already exist: {requested.parent}"
        ) from error
    if not parent.is_dir():
        raise ReplayPreparationError(f"Artifact parent is not a directory: {parent}")
    artifact_dir = parent / requested.name
    for forbidden_root in forbidden_roots:
        if path_is_within(artifact_dir, forbidden_root):
            raise ReplayPreparationError(
                f"Artifact must not be inside sealed input root {forbidden_root}: "
                f"{artifact_dir}"
            )
    try:
        artifact_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ReplayPreparationError(
            f"Artifact path already exists; refusing to overwrite: {artifact_dir}"
        ) from error
    try:
        fsync_directory(parent)
    except OSError:
        artifact_dir.rmdir()
        raise
    return artifact_dir


def frame_file_paths(indexed_frames: Sequence[Mapping[str, object]]) -> set[Path]:
    return {
        Path(frame[kind])
        for frame in indexed_frames
        for kind in FRAME_PATTERNS
    }


def snapshot_unique(
    paths: Iterable[Path], *, reject_symlinks: Iterable[Path] = ()
) -> Dict[Path, FileSnapshot]:
    snapshots: Dict[Path, FileSnapshot] = {}
    rejected = {path.expanduser().absolute() for path in reject_symlinks}
    for path in sorted({path.expanduser().absolute() for path in paths}, key=str):
        snapshots[path] = snapshot_file(path, reject_symlink=path in rejected)
    return snapshots


def build_frame_records(
    indexed_frames: Sequence[Mapping[str, object]],
    snapshots: Mapping[Path, FileSnapshot],
    camera: Mapping[str, object],
) -> List[dict]:
    records = []
    expected_dimensions = (int(camera["width"]), int(camera["height"]))
    for indexed in indexed_frames:
        frame_index = int(indexed["frame_index"])
        jpg_path = Path(indexed["jpg"])
        dimensions = jpeg_dimensions(jpg_path)
        if dimensions != expected_dimensions:
            raise ReplayPreparationError(
                f"Image dimensions differ from final intrinsics for frame {frame_index}: "
                f"image={dimensions}, intrinsics={expected_dimensions}"
            )
        files = {
            kind: snapshots[Path(indexed[kind])].to_json() for kind in FRAME_PATTERNS
        }
        records.append(
            {
                "frame_index": frame_index,
                "image_name": jpg_path.name,
                "image": {"width": dimensions[0], "height": dimensions[1]},
                "files": files,
                "cam": read_cam(Path(indexed["cam"])),
                "pose": convert_fastlio_pose(Path(indexed["odom"])),
            }
        )
    return records


def nested_strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {
            item
            for nested in value.values()
            for item in nested_strings(nested)
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {item for nested in value for item in nested_strings(nested)}
    return set()


def frame_input_identifiers(frame: Mapping[str, object]) -> set[str]:
    files = frame["files"]
    identifiers = {str(frame["image_name"])}
    for kind in FRAME_PATTERNS:
        file_record = files[kind]
        identifiers.add(str(file_record["path"]))
        identifiers.add(str(file_record["real_path"]))
    return identifiers


def write_causal_frames(path: Path, frame_records: Sequence[dict]) -> dict:
    previous_event_sha256 = None
    visible_frame_indices: List[int] = []
    future_identifiers = set().union(
        *(frame_input_identifiers(frame) for frame in frame_records)
    )
    with staged_output_path(path) as temporary_path:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for event_sequence, frame in enumerate(frame_records, start=1):
                frame_index = int(frame["frame_index"])
                expected_frame_index = event_sequence
                if frame_index != expected_frame_index:
                    raise ReplayPreparationError(
                        "Causal replay order must be contiguous and start at frame 1"
                    )
                visible_frame_indices.append(frame_index)
                future_identifiers.difference_update(frame_input_identifiers(frame))
                event = {
                    "schema": "online_i3dgs_phase1_frame_event_v1",
                    "event_sequence": event_sequence,
                    "event_type": "ARRIVED",
                    "frame_index": frame_index,
                    "image_name": frame["image_name"],
                    "state_before": "SEALED",
                    "state_after": "ARRIVED",
                    "phase1_disposition": "PREPARED_ONLY_NO_MAPPER",
                    "attempt_no": 0,
                    "registration_attempted": False,
                    "released_inputs": {
                        kind: {
                            "path": frame["files"][kind]["path"],
                            "sha256": frame["files"][kind]["sha256"],
                        }
                        for kind in FRAME_PATTERNS
                    },
                    "released_pose": frame["pose"],
                    "causality": {
                        "visibility_rule": "CURRENT_AND_HISTORY_ONLY",
                        "visible_frame_indices": list(visible_frame_indices),
                        "released_pose_frame_indices": list(visible_frame_indices),
                        "max_visible_frame_index": frame_index,
                        "visible_feature_frame_indices": [],
                        "visible_match_pairs": [],
                        "future_paths_exposed": False,
                        "future_features_exposed": False,
                    },
                    "previous_event_sha256": previous_event_sha256,
                }
                leaked_identifiers = nested_strings(event) & future_identifiers
                if leaked_identifiers:
                    raise ReplayPreparationError(
                        "Causal event exposes future input identifiers: "
                        f"{sorted(leaked_identifiers)}"
                    )
                event_sha256 = sha256_bytes(canonical_json_bytes(event))
                event["event_sha256"] = event_sha256
                handle.write(
                    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                )
                previous_event_sha256 = event_sha256
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "event_count": len(frame_records),
        "first_frame_index": frame_records[0]["frame_index"] if frame_records else None,
        "last_frame_index": frame_records[-1]["frame_index"] if frame_records else None,
        "event_chain_tail_sha256": previous_event_sha256,
        "future_path_exposure_count": 0,
        "future_feature_exposure_count": 0,
    }


def read_pose_prior_rows(path: Path) -> List[List[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_end = lines.index("end_header")
    except ValueError as error:
        raise ReplayPreparationError(f"Invalid pose-prior PLY header: {path}") from error
    vertex_lines = [line for line in lines[:header_end] if line.startswith("element vertex ")]
    if len(vertex_lines) != 1:
        raise ReplayPreparationError(f"Invalid pose-prior vertex declaration: {path}")
    try:
        vertex_count = int(vertex_lines[0].split()[-1])
    except ValueError as error:
        raise ReplayPreparationError(f"Invalid pose-prior vertex count: {path}") from error
    rows = [finite_floats(line.split(), path, 6) for line in lines[header_end + 1 :] if line]
    if len(rows) != vertex_count:
        raise ReplayPreparationError(
            f"Pose-prior row count mismatch: header={vertex_count}, rows={len(rows)}"
        )
    return rows


def write_pose_prior_exclusive(
    session_dir: Path, path: Path, frame_records: Sequence[Mapping[str, object]]
) -> dict:
    with staged_output_path(path) as temporary_path:
        try:
            report = write_pose_prior(session_dir, temporary_path)
        except RuntimeError as error:
            raise ReplayPreparationError(f"FAST-LIO pose conversion failed: {error}") from error
        expected_rows = [
            frame["pose"]["colmap_prior"]["pose_prior_ply_values"]
            for frame in frame_records
        ]
        actual_rows = read_pose_prior_rows(temporary_path)
        if len(actual_rows) != len(expected_rows):
            raise ReplayPreparationError(
                "Pose-prior frame count does not match sealed frame records"
            )
        if expected_rows and not np.allclose(
            np.asarray(actual_rows),
            np.asarray(expected_rows),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ReplayPreparationError(
                "Pose-prior values differ from sealed FAST-LIO conversions"
            )
    return report


def output_file_record(path: Path) -> dict:
    snapshot = snapshot_file(path)
    return {
        "path": str(path),
        "size_bytes": snapshot.size_bytes,
        "sha256": snapshot.sha256,
    }


def ensure_executable(snapshot: FileSnapshot, label: str) -> None:
    if not os.access(snapshot.path, os.X_OK):
        raise ReplayPreparationError(f"{label} is not executable: {snapshot.path}")


def write_incomplete_marker(artifact_dir: Path, error: BaseException) -> None:
    marker = artifact_dir / "PREPARATION_INCOMPLETE.json"
    if marker.exists():
        return
    try:
        write_json_exclusive(
            marker,
            {
                "schema": "online_i3dgs_phase1_incomplete_v1",
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    except OSError:
        pass


def prepare_replay(options: ReplayOptions) -> dict:
    session_dir = options.session_dir.expanduser().resolve(strict=True)
    if not session_dir.is_dir():
        raise ReplayPreparationError(f"Session is not a directory: {session_dir}")
    repo_root = options.repo_root.expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise ReplayPreparationError(f"Repository is not a directory: {repo_root}")
    i3dgs_dir = options.i3dgs_dir.expanduser().resolve(strict=True)
    if not i3dgs_dir.is_dir():
        raise ReplayPreparationError(f"i3dgs checkout is not a directory: {i3dgs_dir}")
    artifact_dir = create_artifact_dir(
        options.artifact_dir, (session_dir, repo_root, i3dgs_dir)
    )

    try:
        intrinsics_path = options.intrinsics_path.expanduser().absolute()
        binaries = {
            "frontend": options.frontend_binary.expanduser().absolute(),
            "mapper": options.mapper_binary.expanduser().absolute(),
            "texrecon": options.texrecon_binary.expanduser().absolute(),
        }

        expected_frame_count = options.expected_frame_count
        try:
            fixed_session = DEFAULT_SESSION.resolve(strict=True)
        except FileNotFoundError:
            fixed_session = None
        if expected_frame_count is None and session_dir == fixed_session:
            expected_frame_count = DEFAULT_FRAME_COUNT

        indexed_frames = discover_indexed_frames(session_dir)
        if (
            expected_frame_count is not None
            and len(indexed_frames) != expected_frame_count
        ):
            raise ReplayPreparationError(
                f"Expected {expected_frame_count} frames, found {len(indexed_frames)}"
            )
        for relative_path in REQUIRED_SESSION_ASSETS:
            if not (session_dir / relative_path).is_file():
                raise ReplayPreparationError(
                    f"Missing required session asset: {session_dir / relative_path}"
                )

        inventory_paths = session_inventory_paths(session_dir)
        inventory_relative = relative_inventory(inventory_paths, session_dir)
        key_source_files = [repo_root / relative_path for relative_path in KEY_SOURCE_PATHS]
        all_snapshot_paths = [
            *inventory_paths,
            intrinsics_path,
            *binaries.values(),
            *key_source_files,
        ]
        snapshots = snapshot_unique(
            all_snapshot_paths, reject_symlinks=inventory_paths
        )
        initial_snapshot_set_sha256 = snapshot_set_sha256(list(snapshots.values()))
        binary_snapshots = {name: snapshots[path] for name, path in binaries.items()}
        for name, binary_snapshot in binary_snapshots.items():
            ensure_executable(binary_snapshot, name)

        repository_before = capture_repository(repo_root, KEY_SOURCE_PATHS)
        i3dgs_before = capture_i3dgs(i3dgs_dir)
        if i3dgs_before["commit"] != options.expected_i3dgs_commit:
            raise ReplayPreparationError(
                "Unexpected i3dgs commit: "
                f"expected={options.expected_i3dgs_commit}, "
                f"actual={i3dgs_before['commit']}"
            )
        if i3dgs_before["dirty"]:
            raise ReplayPreparationError(f"i3dgs reference must be clean: {i3dgs_dir}")

        camera = read_intrinsics(intrinsics_path)
        frame_records = build_frame_records(indexed_frames, snapshots, camera)

        pose_prior_path = artifact_dir / "pose-prior.ply"
        verify_session_inventory(session_dir, inventory_relative)
        pre_replay_verification = verify_snapshots_unchanged(
            list(snapshots.values())
        )
        if capture_repository(repo_root, KEY_SOURCE_PATHS) != repository_before:
            raise InputChangedError("Repository changed before replay manifest sealing")
        if capture_i3dgs(i3dgs_dir) != i3dgs_before:
            raise InputChangedError("i3dgs changed before replay manifest sealing")
        pre_replay_verification_at = utc_now()

        frame_paths = frame_file_paths(indexed_frames)
        session_assets = []
        for path in inventory_paths:
            if path in frame_paths:
                continue
            relative_path = path.relative_to(session_dir).as_posix()
            session_assets.append(
                {
                    "relative_path": relative_path,
                    "required_for_phase1": relative_path in REQUIRED_SESSION_ASSETS,
                    "file": snapshots[path].to_json(),
                }
            )

        conversion_matrix = np.eye(4, dtype=np.float64)
        conversion_matrix[:3, :3] = FASTLIO_TO_COLMAP_WORLD
        manifest = {
            "schema": "online_i3dgs_phase1_input_manifest_v1",
            "phase": 1,
            "run_id": artifact_dir.name,
            "generated_at": utc_now(),
            "artifact_dir": str(artifact_dir),
            "hash_algorithm": "SHA-256",
            "session": {
                "path": str(options.session_dir.expanduser().absolute()),
                "real_path": str(session_dir),
                "read_only_policy": True,
                "inventory_file_count": len(inventory_paths),
                "inventory_relative_paths_sha256": sha256_bytes(
                    canonical_json_bytes(inventory_relative)
                ),
            },
            "frame_count": len(frame_records),
            "frame_range": [frame_records[0]["frame_index"], frame_records[-1]["frame_index"]],
            "frame_order": "NUMERIC_SUFFIX_ASCENDING",
            "frames": frame_records,
            "camera": {
                **camera,
                "source_file": snapshots[intrinsics_path].to_json(),
            },
            "session_assets": session_assets,
            "required_session_assets": list(REQUIRED_SESSION_ASSETS),
            "binaries": {
                name: binary_snapshot.to_json()
                for name, binary_snapshot in binary_snapshots.items()
            },
            "repository": {
                **repository_before,
                "key_source_files": [
                    snapshots[path].to_json() for path in key_source_files
                ],
                "key_source_snapshot_set_sha256": snapshot_set_sha256(
                    [snapshots[path] for path in key_source_files]
                ),
                "untracked_key_source_files": [
                    {
                        "path": entry["path"],
                        "size_bytes": snapshots[repo_root / entry["path"]].size_bytes,
                        "sha256": snapshots[repo_root / entry["path"]].sha256,
                    }
                    for entry in repository_before["key_source_status"]
                    if entry["status"] == "??"
                ],
            },
            "i3dgs": {
                **i3dgs_before,
                "expected_commit": options.expected_i3dgs_commit,
            },
            "pose_conversion": {
                "implementation": "scripts/python/fastlio_segmented_session.py",
                "implementation_file": snapshots[
                    repo_root / "scripts/python/fastlio_segmented_session.py"
                ].to_json(),
                "reused_symbols": [
                    "FASTLIO_TO_COLMAP_WORLD",
                    "read_odom",
                    "quaternion_to_rotation",
                    "rotation_to_yxz",
                    "write_pose_prior",
                ],
                "fastlio_pose_convention": "T_wc",
                "colmap_prior_convention": "T_cw",
                "fastlio_to_colmap_world": conversion_matrix.tolist(),
                "pose_prior_path": str(pose_prior_path),
            },
            "stability_verification": {
                "method": "initial SHA-256 snapshot plus pre-replay full rehash",
                "metadata_fields": [
                    "real_path",
                    "device",
                    "inode",
                    "mode",
                    "size_bytes",
                    "mtime_ns",
                    "ctime_ns",
                    "path_entry_identity",
                ],
                "initial_snapshot_set_sha256": initial_snapshot_set_sha256,
                "pre_replay_snapshot_set_sha256": pre_replay_verification[
                    "snapshot_set_sha256"
                ],
                "pre_replay_verified_file_count": pre_replay_verification[
                    "verified_file_count"
                ],
                "pre_replay_verified_at": pre_replay_verification_at,
                "post_replay_result": "RECORDED_IN_RUN_SUMMARY",
                "status": "SEALED_FOR_REPLAY",
            },
        }
        manifest_path = artifact_dir / "input-manifest.json"
        write_json_exclusive(manifest_path, manifest)
        manifest_output = output_file_record(manifest_path)

        resolved_config = {
            "schema": "online_i3dgs_phase1_resolved_config_v1",
            "phase": 1,
            "run_mode": "PREPARE_ONLY_DRY_RUN",
            "session_dir": str(session_dir),
            "artifact_dir": str(artifact_dir),
            "expected_frame_count": expected_frame_count,
            "resolved_frame_count": len(frame_records),
            "input_manifest_sha256": manifest_output["sha256"],
            "input_policy": {
                "artifact_must_not_exist": True,
                "overwrite_allowed": False,
                "input_mutation_detection": "FULL_REHASH_BEFORE_AND_AFTER_REPLAY",
                "frame_sets": ["imgs_N.jpg", "imgs_N.CAM", "odoms_N.txt", "scans_N.pcd"],
                "indices": "CONTIGUOUS_FROM_1",
            },
            "replay": {
                "event_order": "ARRIVED_BY_NUMERIC_FRAME_INDEX",
                "visibility": "CURRENT_AND_HISTORY_ONLY",
                "release_fastlio_pose_on_arrival": True,
                "release_future_paths": False,
                "release_future_features": False,
                "feature_extraction_enabled": False,
                "matching_enabled": False,
                "mapper_enabled": False,
                "terminal_registration_state_emitted": False,
            },
            "camera": camera,
            "frontend_baseline_not_executed": {
                "binary": binary_snapshots["frontend"].to_json(),
                "ImageReader.single_camera": 1,
                "ImageReader.camera_model": "OPENCV",
                "ImageReader.camera_params": camera["params"],
                "SiftExtraction.use_gpu": 1,
                "SiftExtraction.gpu_index": 0,
                "SiftExtraction.num_threads": 8,
                "SiftExtraction.max_image_size": 612,
                "SiftMatching.use_gpu": 1,
                "SiftMatching.gpu_index": 0,
                "SiftMatching.num_threads": 8,
            },
            "mapper_baseline_not_executed": {
                "binary": binary_snapshots["mapper"].to_json(),
                "Mapper.online_mode": 1,
                "Mapper.online_reference_count": 5,
                "Mapper.online_min_loop_size": 10,
                "Mapper.online_graph_min_inliers": None,
                "Mapper.online_graph_min_inliers_status": "UNFROZEN_NOT_USED_IN_PHASE1",
                "Mapper.ba_local_num_images": 20,
                "Mapper.online_loop_ba_passes": 2,
                "Mapper.online_localba_fixed_ratio": 0.3,
                "Mapper.online_retry_policy": "none",
                "Mapper.max_reg_trials": None,
                "Mapper.online_se3_propagation": 1,
                "Mapper.online_point_ownership": 1,
                "Mapper.if_import_pose_prior": 1,
                "Mapper.known_pose_registration": 0,
                "Mapper.ba_local_max_refinements": 1,
                "Mapper.ba_global_enabled": 0,
                "Mapper.ba_refine_focal_length": 0,
                "Mapper.ba_refine_principal_point": 0,
                "Mapper.ba_refine_extra_params": 0,
                "Mapper.num_threads": 8,
            },
            "texrecon_not_executed": {
                "binary": binary_snapshots["texrecon"].to_json()
            },
            "pose_conversion": manifest["pose_conversion"],
            "i3dgs_reference": manifest["i3dgs"],
        }
        config_path = artifact_dir / "resolved-config.json"
        write_json_exclusive(config_path, resolved_config)

        logs_dir = artifact_dir / "logs"
        logs_dir.mkdir(mode=0o700)
        pose_prior_report = write_pose_prior_exclusive(
            session_dir, pose_prior_path, frame_records
        )
        if pose_prior_report["frame_count"] != len(frame_records):
            raise ReplayPreparationError("Pose-prior frame count does not match sealed frames")
        if pose_prior_report["frame_range"] != manifest["frame_range"]:
            raise ReplayPreparationError("Pose-prior frame range does not match sealed frames")
        frames_path = logs_dir / "frames.jsonl"
        causal_summary = write_causal_frames(frames_path, frame_records)

        verify_session_inventory(session_dir, inventory_relative)
        final_verification = verify_snapshots_unchanged(list(snapshots.values()))
        if capture_repository(repo_root, KEY_SOURCE_PATHS) != repository_before:
            raise InputChangedError("Repository changed during final artifact seal")
        if capture_i3dgs(i3dgs_dir) != i3dgs_before:
            raise InputChangedError("i3dgs changed during final artifact seal")
        final_verification_at = utc_now()

        outputs = {
            "input_manifest": manifest_output,
            "resolved_config": output_file_record(config_path),
            "frames": output_file_record(frames_path),
            "pose_prior": output_file_record(pose_prior_path),
        }
        summary = {
            "schema": "online_i3dgs_phase1_run_summary_v1",
            "status": "COMPLETED",
            "phase": 1,
            "mode": "PREPARE_ONLY_DRY_RUN",
            "completed_at": utc_now(),
            "artifact_dir": str(artifact_dir),
            "session_dir": str(session_dir),
            "frame_count": len(frame_records),
            "frame_range": manifest["frame_range"],
            "causal_event_count": causal_summary["event_count"],
            "arrived_event_count": causal_summary["event_count"],
            "future_path_exposure_count": causal_summary[
                "future_path_exposure_count"
            ],
            "future_feature_exposure_count": causal_summary[
                "future_feature_exposure_count"
            ],
            "event_chain_tail_sha256": causal_summary["event_chain_tail_sha256"],
            "input_stability": {
                "status": "VERIFIED",
                "pre_replay_verification_at": pre_replay_verification_at,
                "final_verification_at": final_verification_at,
                "verified_file_count": final_verification["verified_file_count"],
                "initial_snapshot_set_sha256": initial_snapshot_set_sha256,
                "pre_replay_snapshot_set_sha256": pre_replay_verification[
                    "snapshot_set_sha256"
                ],
                "final_snapshot_set_sha256": final_verification[
                    "snapshot_set_sha256"
                ],
                "session_inventory_relative_paths_sha256": manifest["session"][
                    "inventory_relative_paths_sha256"
                ],
                "repository_status_sha256": repository_before["status_sha256"],
                "i3dgs_status_sha256": i3dgs_before["status_sha256"],
            },
            "outputs": outputs,
        }
        write_json_exclusive(artifact_dir / "run-summary.json", summary)
        return summary
    except BaseException as error:
        write_incomplete_marker(artifact_dir, error)
        raise


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--frontend-binary", type=Path, default=DEFAULT_FRONTEND_BINARY)
    parser.add_argument("--mapper-binary", type=Path, default=DEFAULT_MAPPER_BINARY)
    parser.add_argument("--texrecon-binary", type=Path, default=DEFAULT_TEXRECON_BINARY)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--i3dgs-dir", type=Path, default=DEFAULT_I3DGS_DIR)
    parser.add_argument("--expected-frame-count", type=positive_integer)
    parser.add_argument("--expected-i3dgs-commit", default=EXPECTED_I3DGS_COMMIT)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Explicitly select the Phase 1 prepare-only dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.prepare_only:
        raise ReplayPreparationError(
            "Phase 1 requires explicit --prepare-only mode"
        )
    summary = prepare_replay(
        ReplayOptions(
            session_dir=args.session_dir,
            artifact_dir=args.artifact_dir,
            intrinsics_path=args.intrinsics,
            frontend_binary=args.frontend_binary,
            mapper_binary=args.mapper_binary,
            texrecon_binary=args.texrecon_binary,
            repo_root=args.repo_root,
            i3dgs_dir=args.i3dgs_dir,
            expected_frame_count=args.expected_frame_count,
            expected_i3dgs_commit=args.expected_i3dgs_commit,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReplayPreparationError, ValueError, np.linalg.LinAlgError) as error:
        print(f"prepare_online_mapper_replay.py: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
