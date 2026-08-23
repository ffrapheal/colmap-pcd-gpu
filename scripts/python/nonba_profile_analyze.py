#!/usr/bin/env python3
"""Summarize one disabled/enabled non-BA profiler screening pair."""

import argparse
import collections
import hashlib
import json
import os
import re


TELEMETRY_STRUCTURE_FIELDS = (
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
    "residuals",
    "residual_blocks",
    "parameter_blocks",
    "parameters",
    "effective_parameters",
    "termination",
    "success",
    "fallback_used",
)

REQUESTED_STAGE_NAMES = (
    "initial_lidar_projection",
    "register_next_image",
    "register_collect_2d3d",
    "register_absolute_pose_ransac",
    "register_pose_refine",
    "register_commit",
    "triangulate_image",
    "local_lidar_projection_preparation",
    "local_lidar_projection_matching",
    "local_kd_queries",
    "local_merge_tracks",
    "local_complete_tracks",
    "local_complete_image",
    "local_filter_in_images",
    "local_filter_modified",
    "local_lidar_outlier",
    "global_pre_complete",
    "global_pre_merge",
    "global_retriangulate",
    "global_points3d_copy_access",
    "global_pre_negative_depth_scan",
    "global_image_config_selection",
    "global_variable_point_collection",
    "global_lidar_kd_preparation",
    "global_kd_queries",
    "global_ba_config",
    "global_post_complete",
    "global_post_merge",
    "global_post_filter_all",
    "global_post_filter_images",
    "write_model",
    "cuda_runtime_shutdown",
)

OWNER_BOUNDARY_STAGE_NAMES = (
    "mapper_root",
    "lidar_ply_load_transform_submap_kdtree_index",
    "initial_lidar_projection",
    "local_ba_config",
    "local_lidar_projection_preparation",
    "local_variable_point_collection",
    "local_lidar_projection_matching",
    "local_kd_queries",
    "local_ba_solve",
    "global_points3d_copy_access",
    "global_lidar_kd_preparation",
    "global_kd_queries",
    "global_ba_config",
    "global_ba_solve",
    "global_post_filter_all",
    "write_model",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-time", required=True)
    parser.add_argument("--on-time", required=True)
    parser.add_argument("--off-telemetry", required=True)
    parser.add_argument("--on-telemetry", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model-compare", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--off-profile", required=True)
    parser.add_argument("--expected-images", type=int, default=50)
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


def telemetry_structure(rows):
    return [tuple(row.get(field) for field in TELEMETRY_STRUCTURE_FIELDS)
            for row in rows]


def telemetry_summary(rows):
    return {
        "calls": len(rows),
        "kind_counts": dict(collections.Counter(
            row["ba_kind"] for row in rows)),
        "success_calls": sum(bool(row["success"]) for row in rows),
        "fallback_calls": sum(bool(row["fallback_used"]) for row in rows),
        "production_audit_failures": sum(
            not bool(row["production_audit_invariants_pass"]) for row in rows),
        "wall_seconds_sum": sum(float(row["wall_seconds"]) for row in rows),
        "custom_cuda_caller_wall_seconds_sum": sum(
            float(row["custom_cuda_caller_wall_seconds"]) for row in rows),
        "termination_counts": dict(collections.Counter(
            row["termination"] for row in rows)),
    }


def slim_stage(stage):
    return {
        "name": stage["name"],
        "calls": stage["calls"],
        "exclusive_wall_ms": stage["exclusive_wall_ms"],
        "inclusive_wall_diagnostic_ms":
            stage["inclusive_wall_diagnostic_ms"],
        "zero_yield_calls": stage["zero_yield_calls"],
        "zero_yield_exclusive_wall_ms":
            stage["zero_yield_exclusive_wall_ms"],
        "input_items": stage["input_items"],
        "output_items": stage["output_items"],
    }


def status_label(value, pass_limit, warn_limit=None):
    if value <= pass_limit:
        return "PASS"
    if warn_limit is not None and value <= warn_limit:
        return "WARN"
    return "FAIL"


def format_float(value, digits=3):
    return ("{:.%df}" % digits).format(value)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def main():
    args = parse_args()
    profile = load_json(args.profile)
    model = load_json(args.model_compare)
    off_rows = load_json_lines(args.off_telemetry)
    on_rows = load_json_lines(args.on_telemetry)
    off_telemetry = telemetry_summary(off_rows)
    on_telemetry = telemetry_summary(on_rows)

    off_wall = elapsed_seconds(args.off_time)
    on_wall = elapsed_seconds(args.on_time)
    partition = profile["partition"]
    root_seconds = partition["mapper_root_inclusive_wall_ms"] / 1000.0
    nonba_seconds = partition["nonba_exclusive_wall_ms"] / 1000.0
    ba_wrapper_seconds = (
        partition["ba_solve_excluded_exclusive_wall_ms"] / 1000.0)
    telemetry_ba_seconds = on_telemetry["wall_seconds_sum"]
    ba_difference_seconds = abs(ba_wrapper_seconds - telemetry_ba_seconds)
    ba_difference_percent = (
        100.0 * ba_difference_seconds / telemetry_ba_seconds
        if telemetry_ba_seconds else 0.0)
    ba_difference_limit_seconds = max(0.1, telemetry_ba_seconds * 0.005)
    closure_error_ms = abs(partition["closure_error_ms"])
    closure_limit_ms = max(
        1.0, partition["mapper_root_inclusive_wall_ms"] * 0.001)
    external_remainder_seconds = on_wall - root_seconds
    external_remainder_limit_seconds = max(2.0, on_wall * 0.02)
    overhead_percent = 100.0 * (on_wall - off_wall) / off_wall

    stages = profile["stages"]
    by_name = {stage["name"]: stage for stage in stages}
    top_nonba = sorted(
        (stage for stage in stages
         if stage["partition"] == "nonba"
         and stage["name"] != "mapper_root" and stage["calls"]),
        key=lambda stage: stage["exclusive_wall_ms"], reverse=True)[:10]
    top_zero_yield = sorted(
        (stage for stage in stages
         if stage["yield_defined"] and stage["zero_yield_calls"]),
        key=lambda stage: stage["zero_yield_exclusive_wall_ms"],
        reverse=True)[:10]
    invalid_zero_yield = [
        stage["name"] for stage in stages
        if not stage["yield_defined"] and stage["zero_yield_calls"]]

    reference_counts = model["counts"]["reference"]
    candidate_counts = model["counts"]["candidate"]
    image_count_pass = (
        reference_counts["images"] == args.expected_images
        and candidate_counts["images"] == args.expected_images)
    telemetry_structure_equal = (
        telemetry_structure(off_rows) == telemetry_structure(on_rows))
    telemetry_pass = (
        telemetry_structure_equal
        and off_telemetry["success_calls"] == off_telemetry["calls"]
        and on_telemetry["success_calls"] == on_telemetry["calls"]
        and off_telemetry["fallback_calls"] == 0
        and on_telemetry["fallback_calls"] == 0
        and off_telemetry["production_audit_failures"] == 0
        and on_telemetry["production_audit_failures"] == 0)

    gates = {
        "off_profile_absent": not os.path.exists(args.off_profile),
        "registered_images": image_count_pass,
        "model_compare": bool(model["pass"]),
        "ba_call_structure_and_health": telemetry_pass,
        "yield_defined_semantics": not invalid_zero_yield,
        "internal_closure": closure_error_ms <= closure_limit_ms,
        "ba_wrapper_vs_telemetry":
            ba_difference_seconds <= ba_difference_limit_seconds,
        "external_partition_remainder":
            abs(external_remainder_seconds) <= external_remainder_limit_seconds,
        "instrumentation_overhead": overhead_percent <= 5.0,
    }
    overall_pass = all(gates.values())
    overhead_status = status_label(overhead_percent, 2.0, 5.0)
    overall_status = "FAIL" if not overall_pass else overhead_status

    summary = {
        "schema": "colmap_nonba_phase_n0_summary_v1",
        "screening_note": (
            "One serialized off/on pair is directional instrumentation "
            "screening, not a statistically stable performance claim."),
        "pass": overall_pass,
        "status": overall_status,
        "gates": gates,
        "binary": {
            "path": os.path.abspath(args.binary),
            "sha256": sha256_file(args.binary),
        },
        "external_wall_seconds": {"off": off_wall, "on": on_wall},
        "instrumentation_overhead_percent": overhead_percent,
        "instrumentation_overhead_status": overhead_status,
        "partition": {
            "mapper_root_seconds": root_seconds,
            "nonba_exclusive_seconds": nonba_seconds,
            "ba_solve_excluded_wrapper_seconds": ba_wrapper_seconds,
            "uninstrumented_remainder_exclusive_seconds":
                partition["uninstrumented_remainder_exclusive_wall_ms"]
                / 1000.0,
            "internal_closure_error_ms": closure_error_ms,
            "internal_closure_limit_ms": closure_limit_ms,
            "external_remainder_seconds": external_remainder_seconds,
            "external_remainder_percent":
                100.0 * external_remainder_seconds / on_wall,
            "external_remainder_limit_seconds":
                external_remainder_limit_seconds,
        },
        "ba": {
            "off": off_telemetry,
            "on": on_telemetry,
            "call_structure_equal": telemetry_structure_equal,
            "profile_wrapper_seconds": ba_wrapper_seconds,
            "telemetry_wall_seconds": telemetry_ba_seconds,
            "wrapper_difference_seconds": ba_difference_seconds,
            "wrapper_difference_percent": ba_difference_percent,
            "wrapper_difference_limit_seconds":
                ba_difference_limit_seconds,
        },
        "model": {
            "pass": model["pass"],
            "counts": model["counts"],
            "structural": model["structural"],
            "max_errors": {
                "rotation_deg": model["errors"]["rotation_deg"]["max"],
                "translation_m": model["errors"]["translation_m"]["max"],
                "point3D_m_by_track":
                    model["errors"]["point3D_m_by_track"]["max"],
            },
        },
        "top_nonba_exclusive": [slim_stage(stage) for stage in top_nonba],
        "top_zero_yield": [slim_stage(stage) for stage in top_zero_yield],
        "requested_stages": [slim_stage(by_name[name])
                             for name in REQUESTED_STAGE_NAMES],
        "owner_boundary_buckets": [
            {
                "name": by_name[name]["name"],
                "exclusive_bucket": by_name[name]["exclusive_bucket"],
                "boundary_contents": by_name[name]["boundary_contents"],
            }
            for name in OWNER_BOUNDARY_STAGE_NAMES
        ],
        "invalid_zero_yield_stages": invalid_zero_yield,
    }

    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    model_counts = model["counts"]["reference"]
    gate_rows = [(name, "PASS" if passed else "FAIL")
                 for name, passed in gates.items()]
    top_rows = [
        (index, stage["name"], stage["calls"],
         format_float(stage["exclusive_wall_ms"] / 1000.0),
         format_float(stage["inclusive_wall_diagnostic_ms"] / 1000.0))
        for index, stage in enumerate(top_nonba, 1)
    ]
    zero_rows = [
        (index, stage["name"], stage["zero_yield_calls"], stage["calls"],
         format_float(stage["zero_yield_exclusive_wall_ms"] / 1000.0))
        for index, stage in enumerate(top_zero_yield, 1)
    ]
    requested_rows = [
        (stage["name"], stage["calls"],
         format_float(stage["exclusive_wall_ms"] / 1000.0),
         format_float(stage["inclusive_wall_diagnostic_ms"] / 1000.0),
         stage["zero_yield_calls"])
        for stage in (by_name[name] for name in REQUESTED_STAGE_NAMES)
    ]
    boundary_rows = [
        (by_name[name]["name"], by_name[name]["exclusive_bucket"],
         by_name[name]["boundary_contents"])
        for name in OWNER_BOUNDARY_STAGE_NAMES
    ]

    markdown = [
        "# Phase N0 non-BA profiler screening",
        "",
        "Result: **{}**. {}".format(
            overall_status, summary["screening_note"]),
        "",
        "Binary SHA256: `{}`".format(summary["binary"]["sha256"]),
        "",
        "## Gates",
        "",
        markdown_table(("Gate", "Result"), gate_rows),
        "",
        "## Totals",
        "",
        "- External wall: off {} s, on {} s; observed delta {}% ({}).".format(
            format_float(off_wall), format_float(on_wall),
            format_float(overhead_percent),
            summary["instrumentation_overhead_status"]),
        "- ON partition: non-BA exclusive {} s, BA wrapper {} s, "
        "root {} s.".format(format_float(nonba_seconds),
                             format_float(ba_wrapper_seconds),
                             format_float(root_seconds)),
        "- Internal closure error: {} ms (limit {} ms).".format(
            format_float(closure_error_ms, 6),
            format_float(closure_limit_ms, 6)),
        "- External partition remainder: {} s / {}% (limit {} s).".format(
            format_float(external_remainder_seconds),
            format_float(100.0 * external_remainder_seconds / on_wall),
            format_float(external_remainder_limit_seconds)),
        "- BA telemetry: off/on calls {}/{}, local/global {}/{}, all success, "
        "fallback=0; ON cumulative wall {} s.".format(
            off_telemetry["calls"], on_telemetry["calls"],
            on_telemetry["kind_counts"].get("local", 0),
            on_telemetry["kind_counts"].get("global", 0),
            format_float(telemetry_ba_seconds)),
        "- BA wrapper vs telemetry: difference {} s / {}% "
        "(limit {} s).".format(format_float(ba_difference_seconds),
                                format_float(ba_difference_percent),
                                format_float(ba_difference_limit_seconds)),
        "- Model: {} cameras, {} images, {} points3D, {} observations; "
        "rotation/translation/point maxima are {}/{}/{}.".format(
            model_counts["cameras"], model_counts["images"],
            model_counts["points3D"], model_counts["observations"],
            model["errors"]["rotation_deg"]["max"],
            model["errors"]["translation_m"]["max"],
            model["errors"]["point3D_m_by_track"]["max"]),
        "",
        "## Top non-BA exclusive wall",
        "",
        markdown_table(("Rank", "Stage", "Calls", "Exclusive s",
                        "Inclusive diagnostic s"), top_rows),
        "",
        "## Top zero-yield wall",
        "",
        markdown_table(("Rank", "Stage", "Zero calls", "Calls",
                        "Exclusive s"), zero_rows),
        "",
        "## Requested stage detail",
        "",
        markdown_table(("Stage", "Calls", "Exclusive s",
                        "Inclusive diagnostic s", "Zero calls"),
                       requested_rows),
        "",
        "## Owner-boundary bucket semantics",
        "",
        markdown_table(("Stage", "Exclusive bucket", "Contents"),
                       boundary_rows),
        "",
    ]
    with open(args.output_markdown, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(markdown))

    print(json.dumps({
        "pass": overall_pass,
        "gates": gates,
        "output_json": os.path.abspath(args.output_json),
        "output_markdown": os.path.abspath(args.output_markdown),
    }, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
