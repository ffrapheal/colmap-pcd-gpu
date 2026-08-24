#!/usr/bin/env python3
"""Analyze one hybrid vs KD-tree-only 50-frame directional screening pair.

The strict COLMAP model comparator remains the equivalence authority. This
analyzer adds experiment-specific route, performance, health, camera-center,
and track-signature reporting. Relative model differences are not absolute
ground-truth accuracy.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shlex

import numpy as np

from read_write_model import read_model


EXPECTED_IMAGES = 50
EXPECTED_BA_CALLS = 120
EXPECTED_LOCAL_BA_CALLS = 96
EXPECTED_GLOBAL_BA_CALLS = 24
STRICT_THRESHOLDS = {
    "rotation_max_deg": 1e-4,
    "translation_max_m": 1e-6,
    "point_max_m": 1e-5,
}

TELEMETRY_CALL_STRUCTURE_FIELDS = (
    "call_index",
    "ba_kind",
    "registered_images",
    "trigger_image_id",
    "refinement_index",
    "requested_backend",
    "executed_backend",
    "problem_source_requested",
    "problem_source_effective",
    "selected_schur",
    "execution_profile",
    "audit_profile_requested",
    "audit_profile_effective",
    "arithmetic_precision_requested",
    "arithmetic_precision_effective",
    "hessian_backend_effective",
    "schur_contribution_backend_effective",
    "host_store_mode_requested",
    "host_store_mode_effective",
)

RUN_SPECIFIC_OPTIONS = {
    "--database_path",
    "--output_path",
    "--Mapper.ba_telemetry_path",
    "--Mapper.non_ba_profile_path",
    "--Mapper.local_lidar_kdtree_only",
}

VALIDATION_CONTRACT = (
    "Use one binary for one hybrid and one kdtree_only run.",
    "Run both with the frozen 50-frame command and profiler enabled.",
    "Use independent database copies and output directories.",
    "Require identical recorded pre-run database SHA256 values.",
    "Keep every option equal except run paths and local_lidar_kdtree_only.",
    "Require 50 images and 120 BA calls (96 local, 24 global) per run.",
    "Require BA success, fallback=0, production audit pass, and termination.",
    "Require profiler closure, external remainder, and BA wrapper cross-check.",
    "Treat the single pair as directional screening only.",
    "Run the unchanged strict comparator at 1e-4 deg, 1e-6 m, and 1e-5 m.",
    "Strict comparator failure is STRICT_MODEL_EQUIVALENCE_FAIL.",
    "A strict numeric difference does not invalidate complete run evidence.",
    "Strictly compare the new hybrid result with the Phase N1 hybrid model.",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-time", required=True)
    parser.add_argument("--kdtree-time", required=True)
    parser.add_argument("--hybrid-telemetry", required=True)
    parser.add_argument("--kdtree-telemetry", required=True)
    parser.add_argument("--hybrid-profile", required=True)
    parser.add_argument("--kdtree-profile", required=True)
    parser.add_argument("--hybrid-model", required=True)
    parser.add_argument("--kdtree-model", required=True)
    parser.add_argument("--hybrid-command", required=True)
    parser.add_argument("--kdtree-command", required=True)
    parser.add_argument("--hybrid-database-sha256", required=True)
    parser.add_argument("--kdtree-database-sha256", required=True)
    parser.add_argument("--hybrid-binary-sha256", required=True)
    parser.add_argument("--kdtree-binary-sha256", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--strict-model-compare", required=True)
    parser.add_argument("--n1-hybrid-model-compare", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_json_lines(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_recorded_sha256(path):
    with open(path, "r", encoding="utf-8") as input_file:
        fields = input_file.readline().split()
    if not fields:
        raise ValueError("missing SHA256 in {}".format(path))
    return fields[0]


def elapsed_seconds(path):
    pattern = re.compile(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)$")
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            match = pattern.search(line.strip())
            if not match:
                continue
            fields = [float(field) for field in match.group(1).split(":")]
            if len(fields) == 2:
                return fields[0] * 60.0 + fields[1]
            if len(fields) == 3:
                return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError("missing GNU time elapsed wall in {}".format(path))


def nullable_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def stage_wall_ns(stage, field):
    return int(round(float(stage[field]) * 1000000.0))


def format_optional(value, digits=6):
    return "null" if value is None else ("{:.%df}" % digits).format(value)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def metric_stats(values, ids, worst_key):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "max": None,
            "median": None,
            "p95": None,
            "rms": None,
            worst_key: None,
        }
    worst_index = int(np.argmax(values))
    return {
        "count": int(values.size),
        "max": float(values[worst_index]),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        worst_key: ids[worst_index],
    }


def quaternion_angle_deg(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    reference /= np.linalg.norm(reference)
    candidate /= np.linalg.norm(candidate)
    cosine = float(np.clip(abs(np.dot(reference, candidate)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def camera_center(image):
    return -image.qvec2rotmat().T.dot(image.tvec)


def point_track_key(point):
    return tuple(sorted(zip(point.image_ids.tolist(),
                            point.point2D_idxs.tolist())))


def model_counts(cameras, images, points):
    return {
        "cameras": len(cameras),
        "images": len(images),
        "points3D": len(points),
        "observations": int(sum(
            np.count_nonzero(image.point3D_ids >= 0)
            for image in images.values())),
    }


def points_by_track(points):
    result = {}
    duplicate_track_keys = 0
    for point in points.values():
        key = point_track_key(point)
        if key in result:
            duplicate_track_keys += 1
        else:
            result[key] = point
    return result, duplicate_track_keys


def relative_model_report(hybrid_path, kdtree_path):
    hybrid_cameras, hybrid_images, hybrid_points = read_model(
        hybrid_path, ext=".bin")
    kdtree_cameras, kdtree_images, kdtree_points = read_model(
        kdtree_path, ext=".bin")

    common_image_ids = sorted(set(hybrid_images) & set(kdtree_images))
    orientation = []
    raw_tvec = []
    centers = []
    for image_id in common_image_ids:
        hybrid = hybrid_images[image_id]
        kdtree = kdtree_images[image_id]
        orientation.append(quaternion_angle_deg(hybrid.qvec, kdtree.qvec))
        raw_tvec.append(float(np.linalg.norm(hybrid.tvec - kdtree.tvec)))
        centers.append(float(np.linalg.norm(
            camera_center(hybrid) - camera_center(kdtree))))

    hybrid_by_track, hybrid_duplicate_tracks = points_by_track(hybrid_points)
    kdtree_by_track, kdtree_duplicate_tracks = points_by_track(kdtree_points)
    hybrid_tracks = set(hybrid_by_track)
    kdtree_tracks = set(kdtree_by_track)
    shared_tracks = sorted(hybrid_tracks & kdtree_tracks)
    point_errors = []
    point_error_ids = []
    for track in shared_tracks:
        hybrid = hybrid_by_track[track]
        kdtree = kdtree_by_track[track]
        point_errors.append(float(np.linalg.norm(hybrid.xyz - kdtree.xyz)))
        point_error_ids.append({
            "hybrid_point3D_id": int(hybrid.id),
            "kdtree_point3D_id": int(kdtree.id),
            "track_first_observation": list(track[0]) if track else None,
            "track_length": len(track),
        })

    hybrid_counts = model_counts(
        hybrid_cameras, hybrid_images, hybrid_points)
    kdtree_counts = model_counts(
        kdtree_cameras, kdtree_images, kdtree_points)
    return {
        "ground_truth_accuracy": "NOT_EVALUATED",
        "interpretation": (
            "All metrics are relative hybrid-to-kdtree_only differences, "
            "not absolute reconstruction accuracy."),
        "counts": {"hybrid": hybrid_counts, "kdtree_only": kdtree_counts},
        "structure": {
            "camera_ids_equal": set(hybrid_cameras) == set(kdtree_cameras),
            "image_ids_equal": set(hybrid_images) == set(kdtree_images),
            "image_names_equal": all(
                hybrid_images[image_id].name == kdtree_images[image_id].name
                for image_id in common_image_ids),
            "common_images": len(common_image_ids),
            "hybrid_only_images": len(set(hybrid_images) - set(kdtree_images)),
            "kdtree_only_images": len(set(kdtree_images) - set(hybrid_images)),
            "hybrid_common_image_coverage": nullable_ratio(
                len(common_image_ids), len(hybrid_images)),
            "kdtree_only_common_image_coverage": nullable_ratio(
                len(common_image_ids), len(kdtree_images)),
            "track_sets_equal": hybrid_tracks == kdtree_tracks,
            "hybrid_duplicate_track_keys": hybrid_duplicate_tracks,
            "kdtree_only_duplicate_track_keys": kdtree_duplicate_tracks,
        },
        "common_image_metrics": {
            "orientation_angle_deg": metric_stats(
                orientation, common_image_ids, "worst_image_id"),
            "raw_tvec_parameter_space_m": metric_stats(
                raw_tvec, common_image_ids, "worst_image_id"),
            "camera_center_m": metric_stats(
                centers, common_image_ids, "worst_image_id"),
        },
        "shared_point3D_by_track_signature": {
            "shared_tracks": len(shared_tracks),
            "hybrid_only_tracks": len(hybrid_tracks - kdtree_tracks),
            "kdtree_only_tracks": len(kdtree_tracks - hybrid_tracks),
            "hybrid_coverage": nullable_ratio(
                len(shared_tracks), len(hybrid_tracks)),
            "kdtree_only_coverage": nullable_ratio(
                len(shared_tracks), len(kdtree_tracks)),
            "position_difference_m": metric_stats(
                point_errors, point_error_ids, "worst_track"),
        },
    }


def parse_mapper_command(path):
    run_dir = None
    binary = None
    options = {}
    with open(path, "r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line.startswith("run_dir="):
                run_dir = line.split("=", 1)[1].strip("\"'")
                continue
            if run_dir is not None:
                line = line.replace("${run_dir}", run_dir)
            line = line.rstrip("\\").strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line)
            if not tokens:
                continue
            if tokens[0].startswith("--"):
                options[tokens[0]] = tokens[1] if len(tokens) > 1 else ""
            if "mapper" in tokens:
                for token in tokens:
                    if token.endswith("/colmap"):
                        binary = token
                        break
    return {"path": os.path.abspath(path),
            "run_dir": run_dir, "binary": binary, "options": options}


def command_contract_report(hybrid_path, kdtree_path):
    hybrid = parse_mapper_command(hybrid_path)
    kdtree = parse_mapper_command(kdtree_path)
    hybrid_options = hybrid["options"]
    kdtree_options = kdtree["options"]
    common_keys = set(hybrid_options) & set(kdtree_options)
    all_keys_equal = set(hybrid_options) == set(kdtree_options)
    unexpected_differences = {
        key: {"hybrid": hybrid_options.get(key),
              "kdtree_only": kdtree_options.get(key)}
        for key in sorted(common_keys - RUN_SPECIFIC_OPTIONS)
        if hybrid_options[key] != kdtree_options[key]
    }
    required_paths = (
        "--database_path",
        "--output_path",
        "--Mapper.ba_telemetry_path",
        "--Mapper.non_ba_profile_path",
    )
    independent_paths = all(
        hybrid_options.get(key) and kdtree_options.get(key)
        and hybrid_options.get(key) != kdtree_options.get(key)
        for key in required_paths)
    checks = {
        "same_binary_path": hybrid["binary"] == kdtree["binary"]
        and hybrid["binary"] is not None,
        "option_key_sets_equal": all_keys_equal,
        "other_options_equal": not unexpected_differences,
        "independent_database_output_telemetry_profile_paths":
            independent_paths,
        "profiler_enabled_both":
            hybrid_options.get("--Mapper.non_ba_profile") == "1"
            and kdtree_options.get("--Mapper.non_ba_profile") == "1",
        "hybrid_mode_selected":
            hybrid_options.get("--Mapper.local_lidar_kdtree_only") == "0",
        "kdtree_only_mode_selected":
            kdtree_options.get("--Mapper.local_lidar_kdtree_only") == "1",
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "unexpected_option_differences": unexpected_differences,
        "hybrid": hybrid,
        "kdtree_only": kdtree,
    }


def telemetry_structure(rows):
    return [tuple(row.get(field) for field in TELEMETRY_CALL_STRUCTURE_FIELDS)
            for row in rows]


def telemetry_summary(rows):
    kind_counts = dict(collections.Counter(
        row.get("ba_kind") for row in rows))
    termination_counts = dict(collections.Counter(
        row.get("termination") for row in rows))
    fixed_shape = (
        len(rows) == EXPECTED_BA_CALLS
        and kind_counts == {
            "local": EXPECTED_LOCAL_BA_CALLS,
            "global": EXPECTED_GLOBAL_BA_CALLS,
        })
    health = (
        all(bool(row.get("success")) for row in rows)
        and all(not bool(row.get("fallback_used")) for row in rows)
        and all(bool(row.get("production_audit_invariants_pass"))
                for row in rows)
        and all(bool(row.get("termination")) for row in rows))
    solver_totals = {
        field: sum(int(row.get(field, 0) or 0) for row in rows)
        for field in (
            "trial_steps",
            "accepted_steps",
            "accepted_commits",
            "rejected_steps",
            "invalid_steps",
        )
    }
    return {
        "calls": len(rows),
        "kind_counts": kind_counts,
        "success_calls": sum(bool(row.get("success")) for row in rows),
        "fallback_calls": sum(bool(row.get("fallback_used")) for row in rows),
        "production_audit_failures": sum(
            not bool(row.get("production_audit_invariants_pass"))
            for row in rows),
        "termination_counts": termination_counts,
        "termination_valid_calls": sum(
            bool(row.get("termination")) for row in rows),
        "solver_totals": solver_totals,
        "wall_seconds_sum": sum(float(row["wall_seconds"]) for row in rows),
        "fixed_shape_pass": fixed_shape,
        "health_pass": health,
    }


def profile_stages(profile):
    return {stage["name"]: stage for stage in profile["stages"]}


def item_sum(stage, key):
    return int(stage[key]["sum"])


def slim_stage(stage):
    return {
        "name": stage["name"],
        "calls": stage["calls"],
        "exclusive_wall_ms": stage["exclusive_wall_ms"],
        "inclusive_wall_diagnostic_ms":
            stage["inclusive_wall_diagnostic_ms"],
        "input_items": stage["input_items"],
        "output_items": stage["output_items"],
        "input_item_unit": stage["input_item_unit"],
        "output_item_unit": stage["output_item_unit"],
    }


def routing_report(profile, mode):
    stages = profile_stages(profile)
    preparation = stages["local_lidar_projection_preparation"]
    projection = stages["local_lidar_projection_matching"]
    project_loop = stages["local_project2image_loop"]
    set_new_image = stages["local_pcd_set_new_image"]
    set_new_image_children = [
        stages["local_pcd_feature_collection_index"],
        stages["local_pcd_search_submap"],
        stages["local_pcd_image_map_proj"],
        stages["local_pcd_association_extraction"],
    ]
    match_loop = stages["local_match_variable_point_to_lidar_loop"]
    kdtree = stages["local_kd_queries"]
    set_new_image_closure_error_ns = (
        stage_wall_ns(set_new_image, "inclusive_wall_diagnostic_ms") -
        stage_wall_ns(set_new_image, "exclusive_wall_ms") -
        sum(stage_wall_ns(stage, "inclusive_wall_diagnostic_ms")
            for stage in set_new_image_children))
    project2image_closure_error_ns = (
        stage_wall_ns(project_loop, "inclusive_wall_diagnostic_ms") -
        stage_wall_ns(project_loop, "exclusive_wall_ms") -
        stage_wall_ns(set_new_image, "inclusive_wall_diagnostic_ms"))
    projection_parent_closure_error_ns = (
        stage_wall_ns(projection, "inclusive_wall_diagnostic_ms") -
        stage_wall_ns(projection, "exclusive_wall_ms") -
        stage_wall_ns(project_loop, "inclusive_wall_diagnostic_ms") -
        stage_wall_ns(match_loop, "inclusive_wall_diagnostic_ms"))
    values = {
        "preparation_output_candidates": item_sum(preparation, "output_items"),
        "projection_input_candidates": item_sum(projection, "input_items"),
        "projection_output_constraints": item_sum(projection, "output_items"),
        "project2image_input_candidates": item_sum(project_loop, "input_items"),
        "project2image_output_new_images": item_sum(project_loop, "output_items"),
        "set_new_image_calls": int(set_new_image["calls"]),
        "match_projection_input_candidates": item_sum(match_loop, "input_items"),
        "match_projection_output_constraints": item_sum(
            match_loop, "output_items"),
        "local_kdtree_input_candidates": item_sum(kdtree, "input_items"),
        "local_kdtree_output_constraints": item_sum(kdtree, "output_items"),
        "local_kdtree_calls": int(kdtree["calls"]),
        "empty_wrapper_calls": {
            "projection_parent": int(projection["calls"]),
            "project2image_loop": int(project_loop["calls"]),
            "match_projection_loop": int(match_loop["calls"]),
        },
        "hierarchy_closure_error_ns": {
            "set_new_image": set_new_image_closure_error_ns,
            "project2image_loop": project2image_closure_error_ns,
            "projection_parent": projection_parent_closure_error_ns,
        },
    }
    common_checks = {
        "projection_stage_inputs_agree":
            values["projection_input_candidates"] ==
            values["project2image_input_candidates"] ==
            values["match_projection_input_candidates"],
        "projection_outputs_agree":
            values["projection_output_constraints"] ==
            values["match_projection_output_constraints"],
        "new_images_equal_set_new_image_calls":
            values["project2image_output_new_images"] ==
            values["set_new_image_calls"],
        "lidar_hierarchy_closure_exact_ns": all(
            error == 0
            for error in values["hierarchy_closure_error_ns"].values()),
        "local_kdtree_wrapper_calls":
            values["local_kdtree_calls"] == EXPECTED_LOCAL_BA_CALLS,
    }
    if mode == "hybrid":
        mode_checks = {
            "candidate_route_partition":
                values["projection_input_candidates"] +
                values["local_kdtree_input_candidates"] ==
                values["preparation_output_candidates"],
        }
    else:
        mode_checks = {
            "projection_input_zero":
                values["projection_input_candidates"] == 0,
            "projection_output_zero":
                values["projection_output_constraints"] == 0,
            "project2image_input_zero":
                values["project2image_input_candidates"] == 0,
            "project2image_output_zero":
                values["project2image_output_new_images"] == 0,
            "set_new_image_calls_zero": values["set_new_image_calls"] == 0,
            "match_projection_input_zero":
                values["match_projection_input_candidates"] == 0,
            "match_projection_output_zero":
                values["match_projection_output_constraints"] == 0,
            "all_candidates_routed_to_kdtree":
                values["local_kdtree_input_candidates"] ==
                values["preparation_output_candidates"],
            "empty_wrappers_preserved": all(
                calls == EXPECTED_LOCAL_BA_CALLS
                for calls in values["empty_wrapper_calls"].values()),
        }
    checks = dict(common_checks)
    checks.update(mode_checks)
    return {"values": values, "checks": checks,
            "pass": all(checks.values())}


def partition_report(profile, external_wall, telemetry_wall):
    partition = profile["partition"]
    root_seconds = partition["mapper_root_inclusive_wall_ms"] / 1000.0
    ba_wrapper_seconds = (
        partition["ba_solve_excluded_exclusive_wall_ms"] / 1000.0)
    closure_error_ms = abs(partition["closure_error_ms"])
    closure_limit_ms = max(
        1.0, partition["mapper_root_inclusive_wall_ms"] * 0.001)
    external_remainder_seconds = external_wall - root_seconds
    external_remainder_limit_seconds = max(2.0, external_wall * 0.02)
    ba_difference_seconds = abs(ba_wrapper_seconds - telemetry_wall)
    ba_difference_limit_seconds = max(0.1, telemetry_wall * 0.005)
    return {
        "mapper_root_seconds": root_seconds,
        "nonba_exclusive_seconds":
            partition["nonba_exclusive_wall_ms"] / 1000.0,
        "ba_wrapper_seconds": ba_wrapper_seconds,
        "telemetry_ba_seconds": telemetry_wall,
        "ba_wrapper_difference_seconds": ba_difference_seconds,
        "ba_wrapper_difference_limit_seconds": ba_difference_limit_seconds,
        "ba_wrapper_crosscheck_pass":
            ba_difference_seconds <= ba_difference_limit_seconds,
        "internal_closure_error_ms": closure_error_ms,
        "internal_closure_limit_ms": closure_limit_ms,
        "internal_closure_pass": closure_error_ms <= closure_limit_ms,
        "external_remainder_seconds": external_remainder_seconds,
        "external_remainder_percent":
            100.0 * external_remainder_seconds / external_wall,
        "external_remainder_limit_seconds": external_remainder_limit_seconds,
        "external_remainder_pass":
            abs(external_remainder_seconds) <=
            external_remainder_limit_seconds,
    }


def run_performance_report(profile, external_wall, telemetry_wall):
    stages = profile_stages(profile)
    projection = stages["local_lidar_projection_matching"]
    project_loop = stages["local_project2image_loop"]
    set_new_image = stages["local_pcd_set_new_image"]
    kdtree = stages["local_kd_queries"]
    global_kdtree = stages["global_kd_queries"]
    projection_candidates = item_sum(projection, "input_items")
    projection_constraints = item_sum(projection, "output_items")
    kdtree_candidates = item_sum(kdtree, "input_items")
    kdtree_constraints = item_sum(kdtree, "output_items")
    global_kdtree_candidates = item_sum(global_kdtree, "input_items")
    global_kdtree_constraints = item_sum(global_kdtree, "output_items")
    local_matching_nonoverlap_ms = (
        projection["inclusive_wall_diagnostic_ms"] +
        kdtree["inclusive_wall_diagnostic_ms"])
    return {
        "external_wall_seconds": external_wall,
        "partition": partition_report(profile, external_wall, telemetry_wall),
        "stages": {
            "projection_parent": slim_stage(projection),
            "project2image_loop": slim_stage(project_loop),
            "set_new_image": slim_stage(set_new_image),
            "local_kdtree": slim_stage(kdtree),
            "global_kdtree_diagnostic": slim_stage(global_kdtree),
        },
        "work": {
            "projection_candidates": projection_candidates,
            "projection_constraints": projection_constraints,
            "kdtree_candidates": kdtree_candidates,
            "kdtree_constraints": kdtree_constraints,
            "accepted_constraints_total":
                projection_constraints + kdtree_constraints,
            "global_kdtree_candidates_diagnostic": global_kdtree_candidates,
            "global_kdtree_constraints_diagnostic":
                global_kdtree_constraints,
        },
        "local_matching_nonoverlap_ms": local_matching_nonoverlap_ms,
        "unit_costs": {
            "projection_parent_ms_per_candidate": nullable_ratio(
                projection["inclusive_wall_diagnostic_ms"],
                projection_candidates),
            "project2image_ms_per_candidate": nullable_ratio(
                project_loop["inclusive_wall_diagnostic_ms"],
                item_sum(project_loop, "input_items")),
            "set_new_image_ms_per_call": nullable_ratio(
                set_new_image["inclusive_wall_diagnostic_ms"],
                set_new_image["calls"]),
            "local_kdtree_ms_per_candidate": nullable_ratio(
                kdtree["inclusive_wall_diagnostic_ms"], kdtree_candidates),
            "local_kdtree_ms_per_accepted_constraint": nullable_ratio(
                kdtree["inclusive_wall_diagnostic_ms"], kdtree_constraints),
            "local_matching_nonoverlap_ms_per_prepared_candidate":
                nullable_ratio(
                    local_matching_nonoverlap_ms,
                    projection_candidates + kdtree_candidates),
            "global_kdtree_ms_per_candidate_diagnostic": nullable_ratio(
                global_kdtree["inclusive_wall_diagnostic_ms"],
                global_kdtree_candidates),
        },
    }


def strict_thresholds_match(report):
    thresholds = report.get("thresholds", {})
    return all(math.isclose(float(thresholds.get(name, float("nan"))), value,
                            rel_tol=0.0, abs_tol=1e-15)
               for name, value in STRICT_THRESHOLDS.items())


def strict_report_summary(report):
    return {
        "pass": bool(report.get("pass")),
        "thresholds": report.get("thresholds"),
        "thresholds_match_contract": strict_thresholds_match(report),
        "counts": report.get("counts"),
        "structural": report.get("structural"),
        "max_errors": {
            "orientation_angle_deg": report.get("errors", {}).get(
                "rotation_deg", {}).get("max"),
            "raw_tvec_parameter_space_m": report.get("errors", {}).get(
                "translation_m", {}).get("max"),
            "point3D_m_by_track": report.get("errors", {}).get(
                "point3D_m_by_track", {}).get("max"),
        },
    }


def main():
    args = parse_args()
    hybrid_profile = load_json(args.hybrid_profile)
    kdtree_profile = load_json(args.kdtree_profile)
    hybrid_rows = load_json_lines(args.hybrid_telemetry)
    kdtree_rows = load_json_lines(args.kdtree_telemetry)
    hybrid_telemetry = telemetry_summary(hybrid_rows)
    kdtree_telemetry = telemetry_summary(kdtree_rows)
    hybrid_wall = elapsed_seconds(args.hybrid_time)
    kdtree_wall = elapsed_seconds(args.kdtree_time)
    hybrid_performance = run_performance_report(
        hybrid_profile, hybrid_wall, hybrid_telemetry["wall_seconds_sum"])
    kdtree_performance = run_performance_report(
        kdtree_profile, kdtree_wall, kdtree_telemetry["wall_seconds_sum"])
    hybrid_routing = routing_report(hybrid_profile, "hybrid")
    kdtree_routing = routing_report(kdtree_profile, "kdtree_only")
    commands = command_contract_report(
        args.hybrid_command, args.kdtree_command)
    relative_model = relative_model_report(
        args.hybrid_model, args.kdtree_model)
    strict_compare = load_json(args.strict_model_compare)
    n1_hybrid_compare = load_json(args.n1_hybrid_model_compare)
    strict_summary = strict_report_summary(strict_compare)
    n1_hybrid_summary = strict_report_summary(n1_hybrid_compare)

    current_binary_sha = sha256_file(args.binary)
    hybrid_binary_sha = read_recorded_sha256(args.hybrid_binary_sha256)
    kdtree_binary_sha = read_recorded_sha256(args.kdtree_binary_sha256)
    hybrid_database_sha = read_recorded_sha256(
        args.hybrid_database_sha256)
    kdtree_database_sha = read_recorded_sha256(
        args.kdtree_database_sha256)
    same_binary = (
        current_binary_sha == hybrid_binary_sha == kdtree_binary_sha)
    same_input_database = hybrid_database_sha == kdtree_database_sha
    ba_call_structure_equal = (
        telemetry_structure(hybrid_rows) == telemetry_structure(kdtree_rows))
    hybrid_counts = relative_model["counts"]["hybrid"]
    kdtree_counts = relative_model["counts"]["kdtree_only"]
    profile_schema_pass = (
        hybrid_profile.get("schema") == "colmap_nonba_stage_profile_v1"
        and kdtree_profile.get("schema") ==
        "colmap_nonba_stage_profile_v1")
    partition_pass = all((
        hybrid_performance["partition"]["internal_closure_pass"],
        hybrid_performance["partition"]["external_remainder_pass"],
        hybrid_performance["partition"]["ba_wrapper_crosscheck_pass"],
        kdtree_performance["partition"]["internal_closure_pass"],
        kdtree_performance["partition"]["external_remainder_pass"],
        kdtree_performance["partition"]["ba_wrapper_crosscheck_pass"],
    ))
    strict_equivalence_pass = bool(strict_summary["pass"])
    n1_compatibility_pass = bool(n1_hybrid_summary["pass"])

    topology = relative_model["structure"]
    registration_components = {
        "kdtree_only_fewer_than_50_images":
            kdtree_counts["images"] < EXPECTED_IMAGES,
        "image_ids_changed": not topology["image_ids_equal"],
        "image_names_changed": not topology["image_names_equal"],
        "camera_count_changed":
            hybrid_counts["cameras"] != kdtree_counts["cameras"],
        "points3D_count_changed":
            hybrid_counts["points3D"] != kdtree_counts["points3D"],
        "observation_count_changed":
            hybrid_counts["observations"] != kdtree_counts["observations"],
        "track_set_changed": not topology["track_sets_equal"],
    }
    registration_regression = {
        "detected": any(registration_components.values()),
        "components": registration_components,
        "hybrid_counts": hybrid_counts,
        "kdtree_only_counts": kdtree_counts,
        "common_images": topology["common_images"],
        "hybrid_only_images": topology["hybrid_only_images"],
        "kdtree_only_images": topology["kdtree_only_images"],
        "hybrid_common_image_coverage":
            topology["hybrid_common_image_coverage"],
        "kdtree_only_common_image_coverage":
            topology["kdtree_only_common_image_coverage"],
    }

    solver_total_delta = {
        field: (kdtree_telemetry["solver_totals"][field] -
                hybrid_telemetry["solver_totals"][field])
        for field in hybrid_telemetry["solver_totals"]
    }
    convergence_change_components = {
        "ba_call_structure_changed": not ba_call_structure_equal,
        "termination_counts_changed":
            hybrid_telemetry["termination_counts"] !=
            kdtree_telemetry["termination_counts"],
        "solver_totals_changed": any(
            value != 0 for value in solver_total_delta.values()),
    }
    convergence_regression_components = {
        "kdtree_only_fixed_shape_invalid":
            not kdtree_telemetry["fixed_shape_pass"],
        "kdtree_only_health_invalid":
            not kdtree_telemetry["health_pass"],
        "kdtree_only_success_shortfall":
            kdtree_telemetry["success_calls"] != EXPECTED_BA_CALLS,
        "kdtree_only_fallback_present":
            kdtree_telemetry["fallback_calls"] != 0,
        "kdtree_only_audit_failure_present":
            kdtree_telemetry["production_audit_failures"] != 0,
        "kdtree_only_termination_invalid":
            kdtree_telemetry["termination_valid_calls"] !=
            kdtree_telemetry["calls"],
    }
    convergence_regression = {
        "detected": any(convergence_regression_components.values()),
        "components": convergence_regression_components,
        "convergence_changed": any(
            convergence_change_components.values()),
        "change_components": convergence_change_components,
        "solver_totals_delta_kdtree_only_minus_hybrid": solver_total_delta,
        "termination_counts": {
            "hybrid": hybrid_telemetry["termination_counts"],
            "kdtree_only": kdtree_telemetry["termination_counts"],
        },
    }

    evidence_gates = {
        "same_binary": same_binary,
        "same_input_database": same_input_database,
        "frozen_command_contract": commands["pass"],
        "profile_schema": profile_schema_pass,
        "hybrid_registered_50_images":
            hybrid_counts["images"] == EXPECTED_IMAGES,
        "kdtree_only_registered_50_images":
            kdtree_counts["images"] == EXPECTED_IMAGES,
        "hybrid_ba_fixed_shape_and_health":
            hybrid_telemetry["fixed_shape_pass"]
            and hybrid_telemetry["health_pass"],
        "kdtree_only_ba_fixed_shape_and_health":
            kdtree_telemetry["fixed_shape_pass"]
            and kdtree_telemetry["health_pass"],
        "hybrid_route_invariants": hybrid_routing["pass"],
        "kdtree_only_route_invariants": kdtree_routing["pass"],
        "profile_partition_health": partition_pass,
        "strict_comparator_threshold_contract":
            strict_summary["thresholds_match_contract"],
        "n1_comparator_threshold_contract":
            n1_hybrid_summary["thresholds_match_contract"],
    }
    execution_valid = all(evidence_gates.values())
    result_codes = []
    if not strict_equivalence_pass:
        result_codes.append("STRICT_MODEL_EQUIVALENCE_FAIL")
    if not n1_compatibility_pass:
        result_codes.append("N1_HYBRID_DEFAULT_COMPATIBILITY_FAIL")
    if registration_regression["detected"]:
        result_codes.append("REGISTRATION_OR_TOPOLOGY_REGRESSION")
    if convergence_regression["detected"]:
        result_codes.append("CONVERGENCE_REGRESSION")
    result_codes.extend(
        "EVIDENCE_CONTRACT_FAIL:{}".format(name)
        for name, passed in evidence_gates.items() if not passed)

    wall_delta_seconds = kdtree_wall - hybrid_wall
    wall_delta_percent = nullable_ratio(
        100.0 * wall_delta_seconds, hybrid_wall)
    if wall_delta_seconds < 0:
        speed_direction = "KDTREE_ONLY_FASTER"
    elif wall_delta_seconds > 0:
        speed_direction = "KDTREE_ONLY_SLOWER"
    else:
        speed_direction = "NO_OBSERVED_DIFFERENCE"
    directional_speed_result = {
        "direction": speed_direction,
        "kdtree_only_minus_hybrid_seconds": wall_delta_seconds,
        "kdtree_only_minus_hybrid_percent": wall_delta_percent,
        "interpretation": (
            "Directional result from one serialized pair; not statistically "
            "stable."),
    }
    performance = {
        "hybrid": hybrid_performance,
        "kdtree_only": kdtree_performance,
        "kdtree_only_minus_hybrid": {
            "external_wall_seconds": wall_delta_seconds,
            "external_wall_percent": wall_delta_percent,
            "nonba_exclusive_seconds":
                kdtree_performance["partition"]["nonba_exclusive_seconds"] -
                hybrid_performance["partition"]["nonba_exclusive_seconds"],
            "ba_wrapper_seconds":
                kdtree_performance["partition"]["ba_wrapper_seconds"] -
                hybrid_performance["partition"]["ba_wrapper_seconds"],
            "local_matching_nonoverlap_seconds":
                (kdtree_performance["local_matching_nonoverlap_ms"] -
                 hybrid_performance["local_matching_nonoverlap_ms"]) /
                1000.0,
            "global_kdtree_diagnostic_seconds":
                (kdtree_performance["stages"][
                    "global_kdtree_diagnostic"][
                        "inclusive_wall_diagnostic_ms"] -
                 hybrid_performance["stages"][
                    "global_kdtree_diagnostic"][
                        "inclusive_wall_diagnostic_ms"]) / 1000.0,
        },
    }

    observed_difference = any((
        not strict_equivalence_pass,
        not n1_compatibility_pass,
        registration_regression["detected"],
        convergence_regression["detected"],
        convergence_regression["convergence_changed"],
    ))
    if not execution_valid:
        status = "INVALID_EXECUTION"
    elif observed_difference:
        status = "VALID_WITH_DIFFERENCE"
    else:
        status = "VALID_EQUIVALENT"
    if not strict_summary["thresholds_match_contract"]:
        strict_model_status = "INVALID_STRICT_COMPARATOR_CONTRACT"
    elif strict_equivalence_pass:
        strict_model_status = "PASS"
    else:
        strict_model_status = "STRICT_MODEL_EQUIVALENCE_FAIL"

    summary = {
        "schema": "colmap_nonba_phase_n2_kdtree_screening_v1",
        "status": status,
        "execution_valid": execution_valid,
        "strict_model_equivalence": strict_model_status,
        "registration_or_topology_regression": registration_regression,
        "convergence_regression": convergence_regression,
        "directional_speed_result": directional_speed_result,
        "ground_truth_accuracy": "NOT_EVALUATED",
        "result_codes": result_codes,
        "screening_note": (
            "One serialized hybrid/kdtree_only pair is directional "
            "screening, not a statistically stable performance claim."),
        "validation_contract": list(VALIDATION_CONTRACT),
        "evidence_contract": {
            "pass": execution_valid,
            "gates": evidence_gates,
        },
        "binary": {
            "path": os.path.abspath(args.binary),
            "current_sha256": current_binary_sha,
            "hybrid_recorded_sha256": hybrid_binary_sha,
            "kdtree_only_recorded_sha256": kdtree_binary_sha,
        },
        "input_database": {
            "hybrid_pre_run_sha256": hybrid_database_sha,
            "kdtree_only_pre_run_sha256": kdtree_database_sha,
            "same_input_database": same_input_database,
        },
        "commands": commands,
        "performance": performance,
        "routing": {
            "hybrid": hybrid_routing,
            "kdtree_only": kdtree_routing,
        },
        "ba": {
            "expected_per_run": {
                "calls": EXPECTED_BA_CALLS,
                "kind_counts": {
                    "local": EXPECTED_LOCAL_BA_CALLS,
                    "global": EXPECTED_GLOBAL_BA_CALLS,
                },
            },
            "hybrid": hybrid_telemetry,
            "kdtree_only": kdtree_telemetry,
            "call_structure_equal": ba_call_structure_equal,
            "convergence_regression": convergence_regression,
        },
        "models": {
            "ground_truth_accuracy": "NOT_EVALUATED",
            "strict_hybrid_vs_kdtree_only": strict_summary,
            "strict_n1_vs_new_hybrid_default_compatibility":
                n1_hybrid_summary,
            "relative_hybrid_vs_kdtree_only": relative_model,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    gate_rows = [(name, "PASS" if passed else "FAIL")
                 for name, passed in evidence_gates.items()]
    performance_rows = []
    for mode, report in (("hybrid", hybrid_performance),
                         ("kdtree_only", kdtree_performance)):
        partition = report["partition"]
        performance_rows.append((
            mode,
            format_optional(report["external_wall_seconds"], 3),
            format_optional(partition["nonba_exclusive_seconds"], 3),
            format_optional(partition["ba_wrapper_seconds"], 3),
            format_optional(
                report["stages"]["projection_parent"][
                    "inclusive_wall_diagnostic_ms"] / 1000.0, 3),
            format_optional(
                report["stages"]["local_kdtree"][
                    "inclusive_wall_diagnostic_ms"] / 1000.0, 3),
            format_optional(
                report["local_matching_nonoverlap_ms"] / 1000.0, 3),
            format_optional(
                report["stages"]["global_kdtree_diagnostic"][
                    "inclusive_wall_diagnostic_ms"] / 1000.0, 3),
            report["work"]["projection_constraints"],
            report["work"]["kdtree_constraints"],
            report["work"]["global_kdtree_candidates_diagnostic"],
            report["work"]["global_kdtree_constraints_diagnostic"],
        ))
    routing_rows = []
    for mode, report in (("hybrid", hybrid_routing),
                         ("kdtree_only", kdtree_routing)):
        values = report["values"]
        routing_rows.append((
            mode,
            values["preparation_output_candidates"],
            values["projection_input_candidates"],
            values["local_kdtree_input_candidates"],
            values["set_new_image_calls"],
            values["projection_output_constraints"],
            values["local_kdtree_output_constraints"],
            "PASS" if report["pass"] else "FAIL",
        ))
    unit_rows = []
    for mode, report in (("hybrid", hybrid_performance),
                         ("kdtree_only", kdtree_performance)):
        costs = report["unit_costs"]
        unit_rows.append((
            mode,
            format_optional(costs["projection_parent_ms_per_candidate"]),
            format_optional(costs["project2image_ms_per_candidate"]),
            format_optional(costs["set_new_image_ms_per_call"]),
            format_optional(costs["local_kdtree_ms_per_candidate"]),
            format_optional(
                costs["local_kdtree_ms_per_accepted_constraint"]),
            format_optional(
                costs[
                    "local_matching_nonoverlap_ms_per_prepared_candidate"]),
            format_optional(
                costs["global_kdtree_ms_per_candidate_diagnostic"]),
        ))
    image_metric_rows = []
    for name, unit in (("orientation_angle_deg", "deg"),
                       ("raw_tvec_parameter_space_m", "m (parameter-space)"),
                       ("camera_center_m", "m")):
        stats = relative_model["common_image_metrics"][name]
        image_metric_rows.append((
            name, unit, stats["count"], format_optional(stats["max"], 9),
            format_optional(stats["median"], 9),
            format_optional(stats["p95"], 9),
            format_optional(stats["rms"], 9), stats["worst_image_id"],
        ))
    point_report = relative_model["shared_point3D_by_track_signature"]
    point_stats = point_report["position_difference_m"]
    ba_rows = []
    for mode, report in (("hybrid", hybrid_telemetry),
                         ("kdtree_only", kdtree_telemetry)):
        ba_rows.append((
            mode,
            report["calls"],
            report["kind_counts"].get("local", 0),
            report["kind_counts"].get("global", 0),
            report["success_calls"],
            report["fallback_calls"],
            report["production_audit_failures"],
            json.dumps(report["termination_counts"], sort_keys=True),
            json.dumps(report["solver_totals"], sort_keys=True),
        ))
    model_count_rows = [
        (mode, counts["cameras"], counts["images"], counts["points3D"],
         counts["observations"])
        for mode, counts in (
            ("hybrid", relative_model["counts"]["hybrid"]),
            ("kdtree_only", relative_model["counts"]["kdtree_only"]))
    ]
    model_structure_rows = [
        (name, value)
        for name, value in relative_model["structure"].items()
    ]
    registration_component_rows = [
        (name, "true" if detected else "false")
        for name, detected in registration_components.items()
    ]
    convergence_regression_rows = [
        (name, "true" if detected else "false")
        for name, detected in convergence_regression_components.items()
    ]
    convergence_change_rows = [
        (name, "true" if changed else "false")
        for name, changed in convergence_change_components.items()
    ]

    markdown = [
        "# Phase N2 all-KD-tree directional screening",
        "",
        "Result: **{}**. {}".format(
            summary["status"], summary["screening_note"]),
        "",
        "- Execution valid: **{}**.".format(
            "true" if execution_valid else "false"),
        "- Strict model equivalence: **{}**.".format(strict_model_status),
        "- Registration/topology regression: **{}**.".format(
            "DETECTED" if registration_regression["detected"] else "none"),
        "- Convergence regression: **{}**; convergence changed: **{}**.".format(
            "DETECTED" if convergence_regression["detected"] else "none",
            "yes" if convergence_regression["convergence_changed"] else "no"),
        "- Directional speed result: **{}**, KD-only minus hybrid {} s / "
        "{}%.".format(
            speed_direction, format_optional(wall_delta_seconds, 3),
            format_optional(wall_delta_percent, 3)),
        "",
        "Ground-truth accuracy: **NOT_EVALUATED**. Relative differences "
        "must not be described as absolute accuracy.",
        "",
        "## Execution evidence gates",
        "",
        markdown_table(("Gate", "Result"), gate_rows),
        "",
        "Result codes: `{}`".format(
            ",".join(result_codes) if result_codes else "none"),
        "",
        "Pre-run database SHA256 hybrid/kdtree_only: `{}` / `{}`; same "
        "input: **{}**.".format(
            hybrid_database_sha, kdtree_database_sha,
            "PASS" if same_input_database else "FAIL"),
        "",
        "## Validation contract",
        "",
    ]
    markdown.extend("- " + item for item in VALIDATION_CONTRACT)
    markdown.extend([
        "",
        "## Performance",
        "",
        markdown_table(("Mode", "External s", "non-BA s", "BA wrapper s",
                        "Projection s", "Local KD s",
                        "Local matching non-overlap s", "Global KD diag s",
                        "Projection constraints", "KD constraints",
                        "Global KD input", "Global KD output"),
                       performance_rows),
        "",
        "KD-only minus hybrid external wall: {} s / {}%.".format(
            format_optional(wall_delta_seconds, 3),
            format_optional(wall_delta_percent, 3)),
        "",
        "## Route accounting",
        "",
        markdown_table(("Mode", "Prepared", "Projection input", "KD input",
                        "SetNewImage calls", "Projection constraints",
                        "KD constraints", "Invariant"), routing_rows),
        "",
        "Empty local wrappers may still report 96 calls in kdtree_only mode; "
        "their route inputs and outputs must be zero.",
        "",
        "## Unit costs",
        "",
        markdown_table(("Mode", "Projection ms/candidate",
                        "Project2Image ms/candidate", "SetNewImage ms/call",
                        "Local KD ms/candidate",
                        "Local KD ms/accepted constraint",
                        "Local matching ms/prepared candidate",
                        "Global KD ms/candidate (diag)"), unit_rows),
        "",
        "A zero denominator is emitted as `null`.",
        "",
        "## BA health and termination",
        "",
        markdown_table(("Mode", "Calls", "Local", "Global", "Success",
                        "Fallback", "Audit failures", "Termination counts",
                        "Solver totals"),
                       ba_rows),
        "",
        "Convergence changed: **{}**; regression: **{}**; solver total "
        "delta KD-only minus hybrid: `{}`.".format(
            "yes" if convergence_regression["convergence_changed"] else "no",
            "yes" if convergence_regression["detected"] else "no",
            json.dumps(solver_total_delta, sort_keys=True)),
        "",
        markdown_table(("Convergence regression component", "Detected"),
                       convergence_regression_rows),
        "",
        markdown_table(("Convergence change component", "Changed"),
                       convergence_change_rows),
        "",
        "## Model counts and structure",
        "",
        markdown_table(("Mode", "Cameras", "Images", "Points3D",
                        "Observations"), model_count_rows),
        "",
        markdown_table(("Structural field", "Value"), model_structure_rows),
        "",
        markdown_table(("Registration/topology component", "Detected"),
                       registration_component_rows),
        "",
        "Common images: {}; hybrid-only: {}; kdtree_only-only: {}; common "
        "coverage hybrid/kdtree_only: {}/{}.".format(
            topology["common_images"], topology["hybrid_only_images"],
            topology["kdtree_only_images"],
            format_optional(topology["hybrid_common_image_coverage"]),
            format_optional(topology[
                "kdtree_only_common_image_coverage"])),
        "",
        "## Relative camera differences",
        "",
        markdown_table(("Metric", "Unit", "Count", "Max", "Median", "P95",
                        "RMS", "Worst image"), image_metric_rows),
        "",
        "Raw tvec is explicitly parameter-space. Camera centers are computed "
        "as `C = -R^T t` from each qvec/tvec pair.",
        "",
        "## Shared Point3D track signatures",
        "",
        "Shared tracks: {}; hybrid-only: {}; kdtree_only-only: {}; "
        "coverage hybrid/kdtree_only: {}/{}.".format(
            point_report["shared_tracks"], point_report["hybrid_only_tracks"],
            point_report["kdtree_only_tracks"],
            format_optional(point_report["hybrid_coverage"]),
            format_optional(point_report["kdtree_only_coverage"])),
        "",
        "Position difference (m): max {}, median {}, P95 {}, RMS {}; "
        "worst track `{}`.".format(
            format_optional(point_stats["max"], 9),
            format_optional(point_stats["median"], 9),
            format_optional(point_stats["p95"], 9),
            format_optional(point_stats["rms"], 9),
            json.dumps(point_stats["worst_track"], sort_keys=True)),
        "",
        "## Strict comparators",
        "",
        "- Hybrid vs kdtree_only strict result: **{}** at fixed thresholds "
        "{}/{}/{}.".format(
            "PASS" if strict_equivalence_pass else
            "STRICT_MODEL_EQUIVALENCE_FAIL",
            STRICT_THRESHOLDS["rotation_max_deg"],
            STRICT_THRESHOLDS["translation_max_m"],
            STRICT_THRESHOLDS["point_max_m"]),
        "- Phase N1 hybrid vs new default hybrid: **{}**.".format(
            "PASS" if n1_compatibility_pass else
            "N1_HYBRID_DEFAULT_COMPATIBILITY_FAIL"),
        "- Strict numeric difference is reported independently and does not "
        "invalidate otherwise complete experiment evidence.",
        "",
    ])
    with open(args.output_markdown, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(markdown))

    print(json.dumps({
        "execution_valid": execution_valid,
        "status": summary["status"],
        "strict_model_equivalence": strict_model_status,
        "result_codes": result_codes,
        "output_json": os.path.abspath(args.output_json),
        "output_markdown": os.path.abspath(args.output_markdown),
    }, sort_keys=True))
    return 0 if execution_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
