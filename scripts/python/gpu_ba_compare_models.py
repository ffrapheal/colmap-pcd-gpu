#!/usr/bin/env python3
"""Compare two COLMAP models by stable identifiers and numeric state.

This driver is intentionally read-only. It avoids treating binary serialization
order or harmless floating-point noise as a model mismatch while still requiring
the camera, image, point, observation, and track identities to agree exactly.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from read_write_model import read_model


def error_stats(values, ids):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "max": 0.0,
            "mean": 0.0,
            "rms": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "worst_id": None,
        }
    worst_index = int(np.argmax(values))
    return {
        "count": int(values.size),
        "max": float(values[worst_index]),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "worst_id": ids[worst_index],
    }


def quaternion_angle_deg(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    reference /= np.linalg.norm(reference)
    candidate /= np.linalg.norm(candidate)
    cosine = float(np.clip(abs(np.dot(reference, candidate)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def point_track_key(point):
    return tuple(sorted(zip(point.image_ids.tolist(), point.point2D_idxs.tolist())))


def compare_models(reference_path, candidate_path):
    ref_cameras, ref_images, ref_points = read_model(reference_path, ext=".bin")
    cand_cameras, cand_images, cand_points = read_model(candidate_path, ext=".bin")

    structural = {
        "camera_ids_equal": set(ref_cameras) == set(cand_cameras),
        "image_ids_equal": set(ref_images) == set(cand_images),
        "point3D_ids_equal": set(ref_points) == set(cand_points),
        "camera_metadata_equal": True,
        "image_metadata_equal": True,
        "keypoints_equal": True,
        "observation_masks_equal": True,
        "observation_point3D_ids_equal": True,
        "tracks_equal_by_point3D_id": True,
        "track_sets_equal": True,
        "track_keys_unique": True,
    }

    camera_param_errors = []
    camera_param_ids = []
    for camera_id in sorted(set(ref_cameras) & set(cand_cameras)):
        ref = ref_cameras[camera_id]
        cand = cand_cameras[camera_id]
        if (ref.model, ref.width, ref.height) != (cand.model, cand.width, cand.height):
            structural["camera_metadata_equal"] = False
        if ref.params.shape != cand.params.shape:
            structural["camera_metadata_equal"] = False
            continue
        for param_index, error in enumerate(np.abs(ref.params - cand.params)):
            camera_param_errors.append(float(error))
            camera_param_ids.append(f"camera:{camera_id}:param:{param_index}")

    rotation_errors = []
    translation_errors = []
    pose_ids = []
    for image_id in sorted(set(ref_images) & set(cand_images)):
        ref = ref_images[image_id]
        cand = cand_images[image_id]
        if (ref.camera_id, ref.name) != (cand.camera_id, cand.name):
            structural["image_metadata_equal"] = False
        if not np.array_equal(ref.xys, cand.xys):
            structural["keypoints_equal"] = False
        if not np.array_equal(ref.point3D_ids >= 0, cand.point3D_ids >= 0):
            structural["observation_masks_equal"] = False
        if not np.array_equal(ref.point3D_ids, cand.point3D_ids):
            structural["observation_point3D_ids_equal"] = False
        rotation_errors.append(quaternion_angle_deg(ref.qvec, cand.qvec))
        translation_errors.append(float(np.linalg.norm(ref.tvec - cand.tvec)))
        pose_ids.append(int(image_id))

    point_errors_by_id = []
    point_ids = []
    for point_id in sorted(set(ref_points) & set(cand_points)):
        ref = ref_points[point_id]
        cand = cand_points[point_id]
        ref_track = point_track_key(ref)
        cand_track = point_track_key(cand)
        if ref_track != cand_track:
            structural["tracks_equal_by_point3D_id"] = False
        point_errors_by_id.append(float(np.linalg.norm(ref.xyz - cand.xyz)))
        point_ids.append(int(point_id))

    ref_points_by_track = {point_track_key(point): point
                           for point in ref_points.values()}
    cand_points_by_track = {point_track_key(point): point
                            for point in cand_points.values()}
    if (len(ref_points_by_track) != len(ref_points)
            or len(cand_points_by_track) != len(cand_points)):
        structural["track_keys_unique"] = False
    ref_track_keys = set(ref_points_by_track)
    cand_track_keys = set(cand_points_by_track)
    if ref_track_keys != cand_track_keys:
        structural["track_sets_equal"] = False

    point_errors_by_track = []
    point_track_ids = []
    point_reprojection_error_differences = []
    point_reprojection_ids = []
    remapped_point_ids = 0
    for track in sorted(ref_track_keys & cand_track_keys):
        ref = ref_points_by_track[track]
        cand = cand_points_by_track[track]
        if ref.id != cand.id:
            remapped_point_ids += 1
        point_errors_by_track.append(float(np.linalg.norm(ref.xyz - cand.xyz)))
        point_track_ids.append({
            "reference_point3D_id": int(ref.id),
            "candidate_point3D_id": int(cand.id),
            "track_first_observation": list(track[0]) if track else None,
            "track_length": len(track),
        })
        point_reprojection_error_differences.append(abs(float(ref.error - cand.error)))
        point_reprojection_ids.append({
            "reference_point3D_id": int(ref.id),
            "candidate_point3D_id": int(cand.id),
        })

    return {
        "reference_path": os.path.abspath(reference_path),
        "candidate_path": os.path.abspath(candidate_path),
        "counts": {
            "reference": {
                "cameras": len(ref_cameras),
                "images": len(ref_images),
                "points3D": len(ref_points),
                "observations": int(sum(np.count_nonzero(i.point3D_ids >= 0)
                                        for i in ref_images.values())),
            },
            "candidate": {
                "cameras": len(cand_cameras),
                "images": len(cand_images),
                "points3D": len(cand_points),
                "observations": int(sum(np.count_nonzero(i.point3D_ids >= 0)
                                        for i in cand_images.values())),
            },
            "shared_track_keys": len(ref_track_keys & cand_track_keys),
            "reference_only_track_keys": len(ref_track_keys - cand_track_keys),
            "candidate_only_track_keys": len(cand_track_keys - ref_track_keys),
            "remapped_point3D_ids": remapped_point_ids,
        },
        "structural": structural,
        "errors": {
            "camera_parameter_abs": error_stats(camera_param_errors, camera_param_ids),
            "rotation_deg": error_stats(rotation_errors, pose_ids),
            "translation_m": error_stats(translation_errors, pose_ids),
            "point3D_m_by_id_diagnostic": error_stats(
                point_errors_by_id, point_ids),
            "point3D_m_by_track": error_stats(
                point_errors_by_track, point_track_ids),
            "point_reprojection_error_abs_px_by_track": error_stats(
                point_reprojection_error_differences, point_reprojection_ids),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--rotation_max_deg", type=float, default=1e-4)
    parser.add_argument("--translation_max_m", type=float, default=1e-6)
    parser.add_argument("--point_max_m", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    report = compare_models(args.reference_path, args.candidate_path)
    required_structural_fields = [
        "camera_ids_equal",
        "image_ids_equal",
        "camera_metadata_equal",
        "image_metadata_equal",
        "keypoints_equal",
        "observation_masks_equal",
        "track_sets_equal",
        "track_keys_unique",
    ]
    structural_pass = all(
        report["structural"][field] for field in required_structural_fields)
    numeric_pass = (
        report["errors"]["rotation_deg"]["max"] <= args.rotation_max_deg
        and report["errors"]["translation_m"]["max"] <= args.translation_max_m
        and report["errors"]["point3D_m_by_track"]["max"] <= args.point_max_m
    )
    report["thresholds"] = {
        "rotation_max_deg": args.rotation_max_deg,
        "translation_max_m": args.translation_max_m,
        "point_max_m": args.point_max_m,
    }
    report["required_structural_fields"] = required_structural_fields
    report["pass"] = bool(structural_pass and numeric_pass)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as output_file:
            output_file.write(output)
            output_file.write("\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
