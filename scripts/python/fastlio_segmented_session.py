#!/usr/bin/env python3
"""Bridge indexed FAST-LIO sessions and segmented COLMAP-PCD models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import read_write_model


IMAGE_RE = re.compile(r"imgs_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
MODEL_IMAGE_RE = re.compile(
    r"(?:frame_|imgs_)(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE
)
ODOM_RE = re.compile(r"odoms_(\d+)\.txt$")

# PointCloudDirectionTrans() uses this transform for FAST-LIO world data.
FASTLIO_TO_COLMAP_WORLD = np.array(
    ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)


def indexed_files(directory: Path, pattern: re.Pattern[str]) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in result:
            raise RuntimeError(f"Duplicate frame {index}: {path} and {result[index]}")
        result[index] = path
    return result


def discover_session(directory: Path) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    images = indexed_files(directory, IMAGE_RE)
    odoms = indexed_files(directory, ODOM_RE)
    if not images:
        raise RuntimeError(f"No imgs_N image files under {directory}")
    if set(images) != set(odoms):
        missing_images = sorted(set(odoms) - set(images))
        missing_odoms = sorted(set(images) - set(odoms))
        raise RuntimeError(
            "Image/odom index mismatch: "
            f"missing_images={missing_images}, missing_odoms={missing_odoms}"
        )
    expected = list(range(1, max(images) + 1))
    if sorted(images) != expected:
        raise RuntimeError("Frame indices must be contiguous and start at 1")
    return images, odoms


def read_odom(path: Path) -> Tuple[str, np.ndarray]:
    tokens = path.read_text(encoding="utf-8").split()
    values = np.asarray(tokens, dtype=np.float64)
    if values.size != 8 or not np.isfinite(values).all():
        raise RuntimeError(f"Expected 8 finite values in {path}")
    quaternion = values[4:8]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise RuntimeError(f"Invalid quaternion in {path}")
    values[4:8] = quaternion / norm
    return tokens[0], values


def quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
             2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
             2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x),
             1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def rotation_to_yxz(rotation: np.ndarray) -> Tuple[float, float, float]:
    """Return y, x, z for R = Ry(y) * Rx(x) * Rz(z)."""
    x_angle = math.asin(float(np.clip(-rotation[1, 2], -1.0, 1.0)))
    cosine_x = math.cos(x_angle)
    if abs(cosine_x) < 1e-10:
        y_angle = math.atan2(float(-rotation[2, 0]), float(rotation[0, 0]))
        z_angle = 0.0
    else:
        y_angle = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        z_angle = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    return y_angle, x_angle, z_angle


def write_pose_prior(session: Path, output: Path) -> dict:
    images, odoms = discover_session(session)
    rows: List[Tuple[float, ...]] = []
    for index in sorted(images):
        _, values = read_odom(odoms[index])
        rotation_fastlio = quaternion_to_rotation(values[4:8])
        rotation_colmap = FASTLIO_TO_COLMAP_WORLD @ rotation_fastlio
        y_angle, x_angle, z_angle = rotation_to_yxz(rotation_colmap)
        # LoadPose() maps these fields back with:
        # t=(-y,-z,x), R=Ry(-yaw)*Rx(-pitch)*Rz(roll).
        rows.append(
            (
                float(values[1]),
                float(values[2]),
                float(values[3]),
                z_angle,
                -x_angle,
                -y_angle,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        for field in ("x", "y", "z", "roll", "pitch", "yaw"):
            handle.write(f"property double {field}\n")
        handle.write("end_header\n")
        for row in rows:
            handle.write(" ".join(f"{value:.17g}" for value in row) + "\n")
    return {
        "schema": "fastlio_mapper_pose_prior_v1",
        "session": str(session.resolve()),
        "output": str(output.resolve()),
        "frame_count": len(rows),
        "frame_range": [min(images), max(images)],
    }


def model_directories(root: Path) -> List[Path]:
    candidates = [root] if (root / "images.bin").is_file() else []
    candidates.extend(
        path for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "images.bin").is_file()
    )
    return sorted(
        candidates,
        key=lambda path: -1 if path == root else int(path.name),
    )


def model_frame(name: str) -> int:
    match = MODEL_IMAGE_RE.fullmatch(Path(name).name)
    if match is None:
        raise RuntimeError(f"Unsupported COLMAP image name: {name}")
    return int(match.group(1))


def optimized_fastlio_pose(image) -> Tuple[np.ndarray, np.ndarray]:
    rotation_cw = image.qvec2rotmat()
    rotation_wc_colmap = rotation_cw.T
    center_colmap = -rotation_wc_colmap @ image.tvec
    rotation_wc_fastlio = FASTLIO_TO_COLMAP_WORLD.T @ rotation_wc_colmap
    center_fastlio = FASTLIO_TO_COLMAP_WORLD.T @ center_colmap
    quaternion = read_write_model.rotmat2qvec(rotation_wc_fastlio)
    return center_fastlio, quaternion


def ranges(indices: Iterable[int]) -> List[List[int]]:
    ordered = sorted(indices)
    if not ordered:
        return []
    result: List[List[int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            result.append([start, previous])
            start = index
        previous = index
    result.append([start, previous])
    return result


def summary(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def rotation_error_degrees(q0: np.ndarray, q1: np.ndarray) -> float:
    r0 = quaternion_to_rotation(q0)
    r1 = quaternion_to_rotation(q1)
    cosine = float(np.clip((np.trace(r0.T @ r1) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def robust_similarity_scale(
    optimized_centers: np.ndarray,
    original_centers: np.ndarray,
    frames: List[int],
) -> dict:
    """Estimate the Sim3 scale from optimized centers back to odometry.

    The fit is used only as a model-usability diagnostic.  Iterative MAD
    trimming prevents one isolated bad pose from making an otherwise usable
    model fail the model-level scale gate.
    """
    if optimized_centers.shape != original_centers.shape:
        raise ValueError("Similarity inputs must have identical shapes")
    if optimized_centers.ndim != 2 or optimized_centers.shape[1] != 3:
        raise ValueError("Similarity inputs must have shape Nx3")
    if optimized_centers.shape[0] < 3:
        return {
            "optimized_to_odometry": None,
            "inlier_count": 0,
            "outlier_frames": list(frames),
            "aligned_position_residual_m": summary([]),
        }

    mask = np.ones(optimized_centers.shape[0], dtype=bool)

    def fit(current_mask: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        optimized = optimized_centers[current_mask]
        original = original_centers[current_mask]
        optimized_mean = np.mean(optimized, axis=0)
        original_mean = np.mean(original, axis=0)
        optimized_centered = optimized - optimized_mean
        original_centered = original - original_mean
        denominator = float(np.sum(optimized_centered * optimized_centered))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("Degenerate optimized camera centers for Sim3 scale")
        u, singular_values, vt = np.linalg.svd(
            optimized_centered.T @ original_centered
        )
        rotation = vt.T @ u.T
        if float(np.linalg.det(rotation)) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
            singular_values[-1] *= -1.0
        scale = float(np.sum(singular_values) / denominator)
        translation = original_mean - scale * (rotation @ optimized_mean)
        return scale, rotation, translation

    for _ in range(8):
        scale, rotation, translation = fit(mask)
        aligned = (
            scale * (rotation @ optimized_centers.T)
        ).T + translation
        residuals = np.linalg.norm(aligned - original_centers, axis=1)
        median = float(np.median(residuals[mask]))
        mad = float(np.median(np.abs(residuals[mask] - median)))
        limit = median + max(4.5 * 1.4826 * mad, 1e-6)
        next_mask = residuals <= limit
        if np.count_nonzero(next_mask) < 3 or np.array_equal(next_mask, mask):
            break
        mask = next_mask

    scale, rotation, translation = fit(mask)
    aligned = (scale * (rotation @ optimized_centers.T)).T + translation
    residuals = np.linalg.norm(aligned - original_centers, axis=1)
    return {
        "optimized_to_odometry": scale,
        "inlier_count": int(np.count_nonzero(mask)),
        "outlier_frames": [
            int(frames[index]) for index in range(len(frames)) if not mask[index]
        ],
        "aligned_position_residual_m": summary(residuals.tolist()),
    }


def stage_complete_session(
    session: Path,
    models_root: Path,
    output: Path,
    report_path: Path,
    intrinsics: Path | None,
    max_translation_update_m: float,
    max_rotation_update_deg: float,
    min_model_scale: float,
    max_model_scale: float,
) -> dict:
    if min_model_scale <= 0.0 or max_model_scale < min_model_scale:
        raise ValueError(
            "Model scale bounds must satisfy 0 < min_model_scale <= max_model_scale"
        )
    images, odoms = discover_session(session)
    models = model_directories(models_root)
    if not models:
        raise RuntimeError(f"No COLMAP models under {models_root}")

    optimized = {}
    source_model = {}
    all_candidate_frames = set()
    model_validation = []
    for model in models:
        loaded = read_write_model.read_model(str(model), ext=".bin")
        if loaded is None:
            raise RuntimeError(f"Cannot read COLMAP model {model}")
        _, model_images, _ = loaded
        model_translation_updates = []
        model_rotation_updates = []
        model_frames = []
        model_frame_records = []
        model_original_centers = []
        model_optimized_centers = []
        for image in model_images.values():
            frame = model_frame(image.name)
            if frame not in odoms:
                raise RuntimeError(
                    f"Model {model} contains frame outside source session: {frame}"
                )
            all_candidate_frames.add(frame)
            model_frames.append(frame)
            _, original = read_odom(odoms[frame])
            center, quaternion = optimized_fastlio_pose(image)
            model_original_centers.append(original[1:4].copy())
            model_optimized_centers.append(center.copy())
            if float(np.dot(quaternion, original[4:8])) < 0.0:
                quaternion = -quaternion
            translation_update = float(np.linalg.norm(center - original[1:4]))
            rotation_update = rotation_error_degrees(
                original[4:8], quaternion
            )
            model_translation_updates.append(translation_update)
            model_rotation_updates.append(rotation_update)
            model_frame_records.append(
                (frame, image, translation_update, rotation_update)
            )

        translation_summary = summary(model_translation_updates)
        rotation_summary = summary(model_rotation_updates)
        scale_validation = robust_similarity_scale(
            np.asarray(model_optimized_centers, dtype=np.float64),
            np.asarray(model_original_centers, dtype=np.float64),
            model_frames,
        )
        model_scale = scale_validation["optimized_to_odometry"]
        accepted = (
            len(model_frames) >= 3 and
            model_scale is not None and
            min_model_scale <= model_scale <= max_model_scale and
            translation_summary["p95"] <= max_translation_update_m and
            rotation_summary["p95"] <= max_rotation_update_deg
        )
        reasons = []
        if len(model_frames) < 3:
            reasons.append("insufficient_model_frames")
        if model_scale is None:
            reasons.append("model_scale_unavailable")
        elif not min_model_scale <= model_scale <= max_model_scale:
            reasons.append("model_scale")
        if (translation_summary["p95"] is not None and
                translation_summary["p95"] > max_translation_update_m):
            reasons.append("translation_p95")
        if (rotation_summary["p95"] is not None and
                rotation_summary["p95"] > max_rotation_update_deg):
            reasons.append("rotation_p95")
        model_validation.append({
            "model": str(model.resolve()),
            "status": "accepted" if accepted else "rejected",
            "reasons": reasons,
            "frame_count": len(model_frames),
            "frame_ranges": ranges(model_frames),
            "pose_update": {
                "translation_m": translation_summary,
                "rotation_deg": rotation_summary,
            },
            "similarity_scale": scale_validation,
            "isolated_frame_gate_failures": sum(
                translation > max_translation_update_m or
                rotation > max_rotation_update_deg
                for _, _, translation, rotation in model_frame_records
            ),
        })
        if not accepted:
            continue

        for frame, image, _, _ in model_frame_records:
            score = int(np.count_nonzero(image.point3D_ids >= 0))
            previous = optimized.get(frame)
            if previous is None or score > previous[0]:
                optimized[frame] = (score, image)
                source_model[frame] = model

    unknown = sorted(all_candidate_frames - set(images))
    if unknown:
        raise RuntimeError(f"Models contain frames outside source session: {unknown}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    translation_updates: List[float] = []
    rotation_updates: List[float] = []
    applied_optimized_frames: List[int] = []
    rejected_updates = {}
    try:
        for frame in sorted(images):
            image_destination = output / f"imgs_{frame}.jpg"
            os.symlink(str(images[frame].resolve()), str(image_destination))
            timestamp, original = read_odom(odoms[frame])
            if frame in optimized:
                center, quaternion = optimized_fastlio_pose(optimized[frame][1])
                if float(np.dot(quaternion, original[4:8])) < 0.0:
                    quaternion = -quaternion
                translation_update = float(np.linalg.norm(center - original[1:4]))
                rotation_update = rotation_error_degrees(
                    original[4:8], quaternion
                )
                if (translation_update <= max_translation_update_m and
                        rotation_update <= max_rotation_update_deg):
                    translation_updates.append(translation_update)
                    rotation_updates.append(rotation_update)
                    applied_optimized_frames.append(frame)
                    values = [*center.tolist(), *quaternion.tolist()]
                    contents = timestamp + "\n" + "\n".join(
                        f"{value:.15g}" for value in values
                    ) + "\n"
                else:
                    rejected_updates[str(frame)] = {
                        "translation_m": translation_update,
                        "rotation_deg": rotation_update,
                        "reason": "pose_update_gate",
                    }
                    contents = odoms[frame].read_text(encoding="utf-8")
            else:
                contents = odoms[frame].read_text(encoding="utf-8")
                if not contents.endswith("\n"):
                    contents += "\n"
            (output / f"odoms_{frame}.txt").write_text(contents, encoding="utf-8")

            source_scan = session / f"scans_{frame}.pcd"
            if source_scan.is_file():
                os.symlink(
                    str(source_scan.resolve()),
                    str(output / source_scan.name),
                )

        for optional_name in ("rtk_data.json",):
            source = session / optional_name
            if source.is_file():
                os.symlink(str(source.resolve()), str(output / optional_name))
        if intrinsics is not None:
            os.symlink(
                str(intrinsics.resolve()),
                str(output / "camera_0_intrinsics.json"),
            )
    except BaseException:
        (output / "PREPARATION_INCOMPLETE").touch(exist_ok=True)
        raise

    candidate_optimized_frames = sorted(optimized)
    optimized_frames = sorted(applied_optimized_frames)
    fallback_frames = sorted(set(images) - set(optimized_frames))
    report = {
        "schema": "fastlio_segmented_optimized_session_v3",
        "source_session": str(session.resolve()),
        "models_root": str(models_root.resolve()),
        "models": [str(path.resolve()) for path in models],
        "output_session": str(output.resolve()),
        "intrinsics": str(intrinsics.resolve()) if intrinsics else None,
        "frame_count": len(images),
        "raw_candidate_frame_count": len(all_candidate_frames),
        "candidate_optimized_frame_count": len(candidate_optimized_frames),
        "optimized_frame_count": len(optimized_frames),
        "fallback_frame_count": len(fallback_frames),
        "optimized_ranges": ranges(optimized_frames),
        "fallback_ranges": ranges(fallback_frames),
        "optimized_frames": optimized_frames,
        "fallback_frames": fallback_frames,
        "pose_update_gate": {
            "max_translation_m": max_translation_update_m,
            "max_rotation_deg": max_rotation_update_deg,
            "rejected_frame_count": len(rejected_updates),
            "rejected_updates": rejected_updates,
        },
        "model_validation_policy": {
            "minimum_frames": 3,
            "minimum_scale_optimized_to_odometry": min_model_scale,
            "maximum_scale_optimized_to_odometry": max_model_scale,
            "translation_p95_m": max_translation_update_m,
            "rotation_p95_deg": max_rotation_update_deg,
        },
        "model_validation": model_validation,
        "source_model_by_frame": {
            str(frame): str(source_model[frame].resolve()) for frame in optimized_frames
        },
        "pose_update": {
            "translation_m": summary(translation_updates),
            "rotation_deg": summary(rotation_updates),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prior = subparsers.add_parser("pose-prior")
    prior.add_argument("--session-dir", required=True, type=Path)
    prior.add_argument("--output", required=True, type=Path)

    stage = subparsers.add_parser("stage-session")
    stage.add_argument("--session-dir", required=True, type=Path)
    stage.add_argument("--models-root", required=True, type=Path)
    stage.add_argument("--output-dir", required=True, type=Path)
    stage.add_argument("--report", required=True, type=Path)
    stage.add_argument("--intrinsics", type=Path)
    stage.add_argument("--max-translation-update-m", type=float, default=0.5)
    stage.add_argument("--max-rotation-update-deg", type=float, default=15.0)
    stage.add_argument("--min-model-scale", type=float, default=0.9)
    stage.add_argument("--max-model-scale", type=float, default=1.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "pose-prior":
        report = write_pose_prior(args.session_dir.resolve(), args.output.resolve())
    else:
        report = stage_complete_session(
            args.session_dir.resolve(),
            args.models_root.resolve(),
            args.output_dir.resolve(),
            args.report.resolve(),
            args.intrinsics.resolve() if args.intrinsics else None,
            args.max_translation_update_m,
            args.max_rotation_update_deg,
            args.min_model_scale,
            args.max_model_scale,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"fastlio_segmented_session.py: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
