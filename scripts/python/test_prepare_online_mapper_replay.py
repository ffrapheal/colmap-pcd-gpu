#!/usr/bin/env python3
"""Focused tests for the online mapper Phase 1 replay preparation."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_online_mapper_replay as replay


def write_test_jpeg(path: Path, width: int = 16, height: int = 12) -> None:
    components = bytes(
        (
            3,
            1,
            0x11,
            0,
            2,
            0x11,
            0,
            3,
            0x11,
            0,
        )
    )
    payload = bytes((8,)) + struct.pack(">HH", height, width) + components
    path.write_bytes(
        b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"
    )


def write_intrinsics(path: Path, width: int = 16, height: int = 12) -> None:
    path.write_text(
        json.dumps(
            {
                "image": {"width": width, "height": height},
                "K": {"fx": 10.0, "fy": 10.5, "cx": 8.0, "cy": 6.0},
                "distortion_model": "plumb_bob",
                "D": [0.1, -0.2, 0.01, -0.01, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_frame(session: Path, frame_index: int) -> None:
    write_test_jpeg(session / f"imgs_{frame_index}.jpg")
    (session / f"imgs_{frame_index}.CAM").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0.5 0.5\n",
        encoding="utf-8",
    )
    (session / f"odoms_{frame_index}.txt").write_text(
        f"{frame_index}.0 {frame_index} {frame_index * 2} {-frame_index} 1 0 0 0\n",
        encoding="utf-8",
    )
    (session / f"scans_{frame_index}.pcd").write_text(
        "# .PCD v0.7\nDATA ascii\n",
        encoding="ascii",
    )


def make_session(root: Path, frame_count: int) -> Path:
    session = root / "session"
    session.mkdir()
    for frame_index in range(1, frame_count + 1):
        write_frame(session, frame_index)
    (session / "obj1").mkdir()
    for relative_path in replay.REQUIRED_SESSION_ASSETS:
        path = session / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative_path}\n", encoding="utf-8")
    return session


def make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    path.chmod(0o755)
    return path


class ReplayFixture:
    def __init__(self, root: Path, frame_count: int) -> None:
        self.root = root
        self.session = make_session(root, frame_count)
        self.intrinsics = root / "intrinsics.json"
        write_intrinsics(self.intrinsics)
        self.binary = make_executable(root / "tool")

    def options(self, artifact: Path, expected_frame_count: int | None = None) -> replay.ReplayOptions:
        return replay.ReplayOptions(
            session_dir=self.session,
            artifact_dir=artifact,
            intrinsics_path=self.intrinsics,
            frontend_binary=self.binary,
            mapper_binary=self.binary,
            texrecon_binary=self.binary,
            expected_frame_count=expected_frame_count,
        )

    def shell_command(self, artifact: Path, expected_frame_count: int) -> list[str]:
        return [
            str(replay.REPO_ROOT / "scripts/run_online_i3dgs_mapper_session.sh"),
            "--prepare-only",
            "--session-dir",
            str(self.session),
            "--artifact-dir",
            str(artifact),
            "--intrinsics",
            str(self.intrinsics),
            "--frontend-binary",
            str(self.binary),
            "--mapper-binary",
            str(self.binary),
            "--texrecon-binary",
            str(self.binary),
            "--expected-frame-count",
            str(expected_frame_count),
        ]


class PrepareOnlineMapperReplayTest(unittest.TestCase):
    def test_numeric_sorting_places_two_before_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayFixture(Path(temporary), 10)
            frames = replay.discover_indexed_frames(fixture.session)
            self.assertEqual([frame["frame_index"] for frame in frames], list(range(1, 11)))
            self.assertEqual(Path(frames[1]["jpg"]).name, "imgs_2.jpg")
            self.assertEqual(Path(frames[9]["jpg"]).name, "imgs_10.jpg")

    def test_missing_indexed_companion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayFixture(Path(temporary), 2)
            (fixture.session / "imgs_2.CAM").unlink()
            with self.assertRaisesRegex(replay.ReplayPreparationError, "mismatch"):
                replay.discover_indexed_frames(fixture.session)

    def test_duplicate_numeric_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayFixture(Path(temporary), 1)
            shutil.copyfile(
                fixture.session / "imgs_1.jpg", fixture.session / "imgs_01.jpg"
            )
            with self.assertRaisesRegex(replay.ReplayPreparationError, "Duplicate"):
                replay.discover_indexed_frames(fixture.session)

    def test_noncontiguous_indices_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayFixture(Path(temporary), 3)
            for name in ("imgs_2.jpg", "imgs_2.CAM", "odoms_2.txt", "scans_2.pcd"):
                (fixture.session / name).unlink()
            with self.assertRaisesRegex(replay.ReplayPreparationError, "contiguous"):
                replay.discover_indexed_frames(fixture.session)

    def test_existing_artifact_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            artifact = root / "artifact"
            artifact.mkdir()
            sentinel = artifact / "sentinel"
            sentinel.write_text("keep\n", encoding="ascii")
            with self.assertRaisesRegex(replay.ReplayPreparationError, "refusing to overwrite"):
                replay.prepare_replay(fixture.options(artifact, 1))
            self.assertEqual(sentinel.read_text(encoding="ascii"), "keep\n")
            self.assertEqual(sorted(path.name for path in artifact.iterdir()), ["sentinel"])

    def test_artifact_validation_does_not_create_inside_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReplayFixture(Path(temporary), 1)
            escaped_parent = fixture.session / "must-not-be-created"
            artifact = escaped_parent / "run"
            with self.assertRaisesRegex(
                replay.ReplayPreparationError, "parent must already exist"
            ):
                replay.prepare_replay(fixture.options(artifact, 1))
            self.assertFalse(escaped_parent.exists())

    def test_artifact_symlink_parent_cannot_escape_into_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            alias = root / "session-alias"
            alias.symlink_to(fixture.session, target_is_directory=True)
            artifact = alias / "run"
            with self.assertRaisesRegex(
                replay.ReplayPreparationError, "inside sealed input root"
            ):
                replay.prepare_replay(fixture.options(artifact, 1))
            self.assertFalse((fixture.session / "run").exists())

    def test_artifact_cannot_be_created_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "session"
            repository = root / "repository"
            session.mkdir()
            repository.mkdir()
            artifact = repository / "run"
            with self.assertRaisesRegex(
                replay.ReplayPreparationError, "inside sealed input root"
            ):
                replay.create_artifact_dir(artifact, (session, repository))
            self.assertFalse(artifact.exists())

    def test_session_file_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            external_scan = root / "external.pcd"
            external_scan.write_text("external\n", encoding="ascii")
            scan = fixture.session / "scans_1.pcd"
            scan.unlink()
            scan.symlink_to(external_scan)
            artifact = root / "phase1"
            with self.assertRaisesRegex(
                replay.ReplayPreparationError, "file symlink"
            ):
                replay.prepare_replay(fixture.options(artifact, 1))
            self.assertTrue((artifact / "PREPARATION_INCOMPLETE.json").is_file())
            self.assertFalse((artifact / "run-summary.json").exists())

    def test_exclusive_output_failure_leaves_no_partial_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "run-summary.json"
            with mock.patch.object(replay.os, "link", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    replay.write_json_exclusive(target, {"status": "COMPLETED"})
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".run-summary.json.tmp-*")), [])

    def test_pose_conversion_reuses_fastlio_world_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            pose = replay.convert_fastlio_pose(fixture.session / "odoms_1.txt")
            center = np.array([1.0, 2.0, -1.0])
            expected_center = replay.FASTLIO_TO_COLMAP_WORLD @ center
            np.testing.assert_allclose(pose["fastlio"]["T_wc"], replay.matrix4(np.eye(3), center))
            np.testing.assert_allclose(
                pose["colmap_prior"]["camera_center_world_m"], expected_center
            )
            expected_rotation_cw = replay.FASTLIO_TO_COLMAP_WORLD.T
            expected_translation_cw = -expected_rotation_cw @ expected_center
            np.testing.assert_allclose(
                pose["colmap_prior"]["T_cw"],
                replay.matrix4(expected_rotation_cw, expected_translation_cw),
            )

            prior_path = root / "pose-prior.ply"
            replay.write_pose_prior(fixture.session, prior_path)
            prior_values = [float(value) for value in prior_path.read_text().splitlines()[-1].split()]
            np.testing.assert_allclose(
                prior_values, pose["colmap_prior"]["pose_prior_ply_values"]
            )

    def test_causal_events_never_expose_future_paths_or_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 3)
            artifact = root / "phase1"
            with mock.patch.object(
                replay, "write_pose_prior", wraps=replay.write_pose_prior
            ) as pose_writer:
                summary = replay.prepare_replay(fixture.options(artifact, 3))
            pose_writer.assert_called_once()
            self.assertEqual(summary["status"], "COMPLETED")
            manifest = json.loads((artifact / "input-manifest.json").read_text())
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertEqual(set(manifest["binaries"]), {"frontend", "mapper", "texrecon"})
            self.assertEqual(
                manifest["pose_conversion"]["implementation"],
                "scripts/python/fastlio_segmented_session.py",
            )
            self.assertIn(
                "write_pose_prior", manifest["pose_conversion"]["reused_symbols"]
            )
            source_names = {
                Path(record["path"]).name
                for record in manifest["repository"]["key_source_files"]
            }
            self.assertIn("test_prepare_online_mapper_replay.py", source_names)
            changed_key_paths = {
                entry["path"]
                for entry in manifest["repository"]["key_source_status"]
            }
            self.assertIn(
                "scripts/python/prepare_online_mapper_replay.py", changed_key_paths
            )
            untracked_key_paths = {
                entry["path"]
                for entry in manifest["repository"]["untracked_key_source_files"]
            }
            self.assertIn(
                "scripts/python/test_prepare_online_mapper_replay.py",
                untracked_key_paths,
            )
            self.assertEqual(
                len(manifest["repository"]["key_source_snapshot_set_sha256"]),
                64,
            )
            fingerprints = {
                summary["input_stability"][name]
                for name in (
                    "initial_snapshot_set_sha256",
                    "pre_replay_snapshot_set_sha256",
                    "final_snapshot_set_sha256",
                )
            }
            self.assertEqual(len(fingerprints), 1)
            pose_rows = replay.read_pose_prior_rows(artifact / "pose-prior.ply")
            self.assertEqual(
                pose_rows,
                [
                    frame["pose"]["colmap_prior"]["pose_prior_ply_values"]
                    for frame in manifest["frames"]
                ],
            )
            events = [
                json.loads(line)
                for line in (artifact / "logs/frames.jsonl").read_text().splitlines()
            ]
            previous_hash = None
            for frame_index, event in enumerate(events, start=1):
                self.assertEqual(event["event_type"], "ARRIVED")
                self.assertEqual(event["frame_index"], frame_index)
                self.assertEqual(
                    event["causality"]["visible_frame_indices"],
                    list(range(1, frame_index + 1)),
                )
                self.assertEqual(event["causality"]["visible_feature_frame_indices"], [])
                self.assertFalse(event["causality"]["future_paths_exposed"])
                self.assertFalse(event["causality"]["future_features_exposed"])
                future_identifiers = set().union(
                    *(
                        replay.frame_input_identifiers(frame)
                        for frame in manifest["frames"][frame_index:]
                    )
                )
                self.assertFalse(replay.nested_strings(event) & future_identifiers)
                path_indices = [
                    int(match)
                    for match in re.findall(
                        r"(?:imgs_|odoms_|scans_)(\d+)", json.dumps(event)
                    )
                ]
                self.assertTrue(path_indices)
                self.assertLessEqual(max(path_indices), frame_index)
                self.assertEqual(event["previous_event_sha256"], previous_hash)
                event_hash = event.pop("event_sha256")
                self.assertEqual(event_hash, replay.sha256_bytes(replay.canonical_json_bytes(event)))
                previous_hash = event_hash

    def test_input_change_during_run_fails_and_marks_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            artifact = root / "phase1"
            original_verify = replay.verify_snapshots_unchanged
            verification_count = 0

            def mutate_then_verify(snapshots):
                nonlocal verification_count
                verification_count += 1
                if verification_count == 2:
                    with (fixture.session / "scans_1.pcd").open("a", encoding="ascii") as handle:
                        handle.write("changed\n")
                return original_verify(snapshots)

            with mock.patch.object(
                replay, "verify_snapshots_unchanged", side_effect=mutate_then_verify
            ):
                with self.assertRaises(replay.InputChangedError):
                    replay.prepare_replay(fixture.options(artifact, 1))
            self.assertTrue((artifact / "PREPARATION_INCOMPLETE.json").is_file())
            self.assertFalse((artifact / "run-summary.json").exists())

    def test_metadata_change_after_replay_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            artifact = root / "phase1"
            original_verify = replay.verify_snapshots_unchanged
            verification_count = 0

            def chmod_then_verify(snapshots):
                nonlocal verification_count
                verification_count += 1
                if verification_count == 2:
                    scan = fixture.session / "scans_1.pcd"
                    current_mode = stat.S_IMODE(scan.stat().st_mode)
                    scan.chmod(0o644 if current_mode != 0o644 else 0o600)
                return original_verify(snapshots)

            with mock.patch.object(
                replay, "verify_snapshots_unchanged", side_effect=chmod_then_verify
            ):
                with self.assertRaises(replay.InputChangedError):
                    replay.prepare_replay(fixture.options(artifact, 1))
            self.assertTrue((artifact / "PREPARATION_INCOMPLETE.json").is_file())
            self.assertFalse((artifact / "run-summary.json").exists())

    def test_shell_prepare_only_uses_explicit_new_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root with spaces"
            root.mkdir()
            fixture = ReplayFixture(root, 2)
            artifact = root / "shell phase1"
            command = fixture.shell_command(artifact, 2)
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((artifact / "run-summary.json").is_file())
            second = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)

            dangling = root / "dangling-artifact"
            dangling.symlink_to(root / "missing-target", target_is_directory=True)
            dangling_result = subprocess.run(
                fixture.shell_command(dangling, 2),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(dangling_result.returncode, 2)
            self.assertIn("already exists", dangling_result.stderr)
            self.assertTrue(dangling.is_symlink())

    def test_cli_and_shell_fail_with_stable_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ReplayFixture(root, 1)
            direct_artifact = root / "direct"
            direct = subprocess.run(
                [
                    sys.executable,
                    str(replay.SCRIPT_PATH),
                    "--artifact-dir",
                    str(direct_artifact),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(direct.returncode, 2)
            self.assertIn("requires explicit --prepare-only", direct.stderr)
            self.assertFalse(direct_artifact.exists())

            (fixture.session / "odoms_1.txt").write_text(
                "1.0 0 0 0 0 0 0 0\n", encoding="ascii"
            )
            shell_artifact = root / "shell-invalid"
            shell = subprocess.run(
                fixture.shell_command(shell_artifact, 1),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(shell.returncode, 2)
            self.assertIn("Invalid FAST-LIO odometry", shell.stderr)
            self.assertNotIn("artifact sealed", shell.stdout)
            self.assertTrue(
                (shell_artifact / "PREPARATION_INCOMPLETE.json").is_file()
            )
            self.assertFalse((shell_artifact / "run-summary.json").exists())

    @unittest.skipUnless(
        replay.DEFAULT_SESSION.is_dir()
        and replay.DEFAULT_INTRINSICS.is_file()
        and replay.DEFAULT_FRONTEND_BINARY.is_file()
        and replay.DEFAULT_MAPPER_BINARY.is_file()
        and replay.DEFAULT_TEXRECON_BINARY.is_file(),
        "fixed 20260820-111812 replay inputs are unavailable",
    )
    def test_fixed_session_246_frame_dry_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="online-mapper-phase1-test-") as temporary:
            artifact = Path(temporary) / "phase1-fixed-session"
            summary = replay.prepare_replay(
                replay.ReplayOptions(
                    session_dir=replay.DEFAULT_SESSION,
                    artifact_dir=artifact,
                    intrinsics_path=replay.DEFAULT_INTRINSICS,
                    frontend_binary=replay.DEFAULT_FRONTEND_BINARY,
                    mapper_binary=replay.DEFAULT_MAPPER_BINARY,
                    texrecon_binary=replay.DEFAULT_TEXRECON_BINARY,
                )
            )
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["frame_count"], 246)
            self.assertEqual(summary["arrived_event_count"], 246)
            manifest = json.loads((artifact / "input-manifest.json").read_text())
            self.assertEqual(manifest["frame_range"], [1, 246])
            self.assertEqual(
                [frame["frame_index"] for frame in manifest["frames"]],
                list(range(1, 247)),
            )
            with (artifact / "logs/frames.jsonl").open(encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertEqual(len(events), 246)
            self.assertEqual(events[-1]["causality"]["max_visible_frame_index"], 246)
            config = json.loads((artifact / "resolved-config.json").read_text())
            self.assertEqual(config["expected_frame_count"], 246)


if __name__ == "__main__":
    unittest.main()
