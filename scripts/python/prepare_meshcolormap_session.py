#!/usr/bin/env python3
"""Prepare an isolated meshcolormap session from a COLMAP-PCD model."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

import read_write_model


FRAME_PATTERN = re.compile(r"(?:frame_|imgs_)(\d+)\.(?:jpg|jpeg|png)$", re.I)
LIDAR_TO_COLMAP_WORLD = np.array(
    ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--mesh-dir", required=True, type=Path)
    return parser.parse_args()


def resolve_source_image_dir(dataset):
    image_dir = dataset / "image"
    return image_dir if image_dir.is_dir() else dataset


def read_images(model):
    binary = model / "images.bin"
    text = model / "images.txt"
    if binary.is_file():
        return read_write_model.read_images_binary(str(binary)), ".bin"
    if text.is_file():
        return read_write_model.read_images_text(str(text)), ".txt"
    raise FileNotFoundError(f"No images.bin or images.txt under {model}")


def extract_frame(name):
    match = FRAME_PATTERN.search(Path(name).name)
    if not match:
        raise ValueError(f"Unsupported model image name: {name}")
    return int(match.group(1))


def find_image(image_dir, frame):
    stems = (f"imgs_{frame}", f"frame_{frame:06d}", f"frame_{frame}")
    suffixes = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    for stem in stems:
        for suffix in suffixes:
            path = image_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No image for frame {frame} under {image_dir}")


def read_odom(path):
    tokens = path.read_text(encoding="utf-8").split()
    values = np.asarray(tokens, dtype=np.float64)
    if values.size != 8 or not np.isfinite(values).all():
        raise ValueError(f"Expected 8 finite values in {path}")
    return tokens[0], values


def optimized_fastlio_pose(image):
    rotation_cw_internal = image.qvec2rotmat()
    rotation_wc_internal = rotation_cw_internal.T
    center_internal = -rotation_wc_internal @ image.tvec
    rotation_wc_fastlio = LIDAR_TO_COLMAP_WORLD.T @ rotation_wc_internal
    center_fastlio = LIDAR_TO_COLMAP_WORLD.T @ center_internal
    quaternion_wxyz = read_write_model.rotmat2qvec(rotation_wc_fastlio)
    return center_fastlio, rotation_wc_fastlio, quaternion_wxyz


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
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


def rotation_error_degrees(reference, estimate):
    cosine = np.clip((np.trace(reference.T @ estimate) - 1.0) * 0.5,
                     -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symlink(source, destination):
    os.symlink(str(source.resolve()), str(destination))


def main():
    args = parse_args()
    model = args.model.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    source_image_dir = resolve_source_image_dir(source_dataset)
    image_dir = args.image_dir.expanduser().resolve()
    intrinsics = args.intrinsics.expanduser().resolve()
    mesh = args.mesh.expanduser().resolve()
    session_dir = args.session_dir.expanduser().resolve()
    mesh_dir = args.mesh_dir.expanduser().resolve()

    for path in (model, source_dataset, source_image_dir, image_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (intrinsics, mesh):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (session_dir, mesh_dir):
        if path.exists():
            raise FileExistsError(path)

    calibration = json.loads(intrinsics.read_text(encoding="utf-8"))
    distortion = np.asarray(calibration.get("D", []), dtype=np.float64)
    images_are_undistorted = bool(calibration.get("undistorted")) or (
        distortion.size > 0 and np.all(np.abs(distortion) <= 1e-12)
    )
    if not images_are_undistorted:
        raise ValueError(
            "meshcolormap .CAM files cannot represent lens distortion; provide "
            "undistorted images and their zero-distortion intrinsics"
        )

    images, model_format = read_images(model)
    indexed = {}
    for image in images.values():
        frame = extract_frame(image.name)
        if frame in indexed:
            raise ValueError(f"Duplicate model frame: {frame}")
        indexed[frame] = image
    if not indexed:
        raise ValueError("The COLMAP model has no registered images")

    prepared = []
    translation_corrections = []
    rotation_corrections = []
    for frame, image in sorted(indexed.items()):
        source_image = find_image(image_dir, frame)
        source_odom = source_image_dir / f"odoms_{frame}.txt"
        if not source_odom.is_file():
            raise FileNotFoundError(source_odom)
        timestamp, original = read_odom(source_odom)
        center, rotation, quaternion = optimized_fastlio_pose(image)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError(f"Frame {frame} has a non-orthogonal rotation")
        translation_corrections.append(float(np.linalg.norm(center - original[1:4])))
        rotation_corrections.append(
            rotation_error_degrees(quaternion_matrix(original[4:8]), rotation)
        )
        prepared.append((frame, source_image, timestamp, center, quaternion))

    session_dir.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir.parent.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir()
    mesh_dir.mkdir()
    try:
        for frame, source_image, timestamp, center, quaternion in prepared:
            symlink(source_image, session_dir / f"imgs_{frame}.jpg")
            values = [*center.tolist(), *quaternion.tolist()]
            contents = timestamp + "\n" + "\n".join(
                f"{value:.15g}" for value in values
            )
            (session_dir / f"odoms_{frame}.txt").write_text(
                contents + "\n", encoding="utf-8"
            )

        symlink(intrinsics, session_dir / "camera_0_intrinsics.json")
        rtk = source_image_dir / "rtk_data.json"
        if rtk.is_file():
            symlink(rtk, session_dir / "rtk_data.json")
        symlink(mesh, mesh_dir / "mesh_origin.ply")

        manifest = {
            "status": "prepared",
            "model": str(model),
            "model_format": model_format,
            "source_dataset": str(source_dataset),
            "image_directory": str(image_dir),
            "images_are_undistorted": images_are_undistorted,
            "intrinsics": str(intrinsics),
            "mesh": str(mesh),
            "mesh_sha256": sha256(mesh),
            "session_directory": str(session_dir),
            "mesh_directory": str(mesh_dir),
            "registered_frame_count": len(prepared),
            "registered_frames": [item[0] for item in prepared],
            "pose_output_format": (
                "timestamp, xyz, quaternion wxyz; one value per line"
            ),
            "pose_convention": "T_wc in FAST_LIO world coordinates",
            "coordinate_conversion": (
                "FAST_LIO_world = transpose(B) * Colmap-PCD_world"
            ),
            "B_lidar_to_colmap_world": LIDAR_TO_COLMAP_WORLD.tolist(),
            "correction_from_original_fastlio": {
                "translation_m": summarize(translation_corrections),
                "rotation_deg": summarize(rotation_corrections),
            },
        }
        (session_dir / "meshcolormap_input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        (session_dir / "PREPARATION_INCOMPLETE").touch(exist_ok=True)
        raise

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
