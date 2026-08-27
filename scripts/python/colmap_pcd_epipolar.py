#!/home/nvidia/anaconda3/envs/scanner/bin/python
"""Interactively compare FAST_LIO and Colmap-PCD epipolar lines.

The left and right image points are selected manually, as in
``/home/nvidia/colormap/epipolar.py``. The initial geometry comes from the
FAST_LIO ``odoms_N.txt`` files, while the optimized geometry comes from a
COLMAP sparse model.

Controls:

* click a point in the left image;
* click the same physical point in the right image;
* ``r`` resets the current measurement;
* ``s`` saves the full-resolution comparison;
* ``q`` or Escape exits.

The source images use an OPENCV distortion model. Both images are therefore
undistorted before display so that their epipolar loci are straight lines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
HOME_ROOT = REPO_ROOT.parent
REFERENCE_SCRIPT = HOME_ROOT / "colormap" / "epipolar.py"
MODEL_READER_DIR = SCRIPT_ROOT

DEFAULT_DATASET = (
    HOME_ROOT / "colormap" / "data" / "20260717_105806_color_filtered"
)
DEFAULT_MODEL = (
    HOME_ROOT
    / "colmap-pcd-test"
    / "20260717_105806_full270_1cm_firstpose_i30_i35_lock"
    / "sparse"
    / "0"
)
DEFAULT_LEFT_INDEX = 90
DEFAULT_RIGHT_INDEX = 100

FRAME_PATTERN = re.compile(
    r"(?:frame_|imgs_)(\d+)(?:_cam\d+)?\.(?:jpg|jpeg|png)$", re.IGNORECASE
)


def load_module(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load Python module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


# Load the exact reference script instead of /home/nvidia/epipolar.py, which is
# a different viewer with the same import name.
reference = load_module(REFERENCE_SCRIPT, "colormap_epipolar_reference")

if not MODEL_READER_DIR.is_dir():
    raise FileNotFoundError(MODEL_READER_DIR)
sys.path.insert(0, str(MODEL_READER_DIR))
import read_write_model  # noqa: E402


@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    model_name: str


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion is not finite and non-zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def read_fastlio_twc(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(path.read_text(encoding="utf-8").split(), dtype=np.float64)
    if values.size != 8 or not np.isfinite(values).all():
        raise ValueError(
            f"{path} must contain timestamp, xyz, and quaternion wxyz (8 values)"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix(values[4:8])
    transform[:3, 3] = values[1:4]
    return transform


def colmap_image_twc(image) -> np.ndarray:
    rotation_cw = image.qvec2rotmat()
    rotation_wc = rotation_cw.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_wc
    transform[:3, 3] = -rotation_wc @ image.tvec
    return transform


def camera_calibration(camera) -> Calibration:
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model.upper()
    if model == "OPENCV" and params.size == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    elif model == "FULL_OPENCV" and params.size == 12:
        fx, fy, cx, cy = params[:4]
        distortion = params[4:12].copy()
    elif model == "PINHOLE" and params.size == 4:
        fx, fy, cx, cy = params
        distortion = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_PINHOLE" and params.size == 3:
        fx, cx, cy = params
        fy = fx
        distortion = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL" and params.size == 4:
        fx, cx, cy, k1 = params
        fy = fx
        distortion = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL" and params.size == 5:
        fx, cx, cy, k1, k2 = params
        fy = fx
        distortion = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {camera.model} with {params.size} params. "
            "This viewer supports OPENCV, FULL_OPENCV, PINHOLE, "
            "SIMPLE_PINHOLE, SIMPLE_RADIAL, and RADIAL."
        )
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    if not np.isfinite(intrinsic).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Camera {camera.id} has invalid intrinsics")
    return Calibration(
        intrinsic=intrinsic,
        distortion=distortion,
        image_size=(int(camera.width), int(camera.height)),
        model_name=camera.model,
    )


def json_calibration(path: Path) -> Calibration:
    path = path.expanduser().resolve()
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid calibration JSON in {path}: {error}") from error
    try:
        image = metadata["image"]
        values = metadata["K"]
        width, height = int(image["width"]), int(image["height"])
        fx, fy = float(values["fx"]), float(values["fy"])
        cx, cy = float(values["cx"]), float(values["cy"])
        distortion = np.asarray(metadata.get("D", []), dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid image/K/D calibration in {path}") from error
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid image dimensions or focal length in {path}")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(
        distortion
    ).all():
        raise ValueError(f"Unsupported OpenCV distortion vector in {path}")
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return Calibration(
        intrinsic=intrinsic,
        distortion=distortion,
        image_size=(width, height),
        model_name=f"JSON:{path.name}",
    )


def resolve_image_directory(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    candidates = (dataset / "image", dataset)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("imgs_*.*")):
            return candidate
    raise FileNotFoundError(f"No imgs_N images found under {dataset}")


def resolve_image_path(image_dir: Path, frame: int) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        for stem in (f"imgs_{frame}", f"frame_{frame:06d}"):
            path = image_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No image for frame {frame} under {image_dir}")


def read_image(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    actual_size = (image.shape[1], image.shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f"{path} is {actual_size[0]}x{actual_size[1]}, but its COLMAP "
            f"camera is {expected_size[0]}x{expected_size[1]}"
        )
    return image


def frame_number(image_name: str) -> int | None:
    match = FRAME_PATTERN.search(Path(image_name).name)
    return int(match.group(1)) if match else None


def images_by_frame(images: dict) -> dict[int, object]:
    indexed: dict[int, object] = {}
    for image in images.values():
        frame = frame_number(image.name)
        if frame is None:
            continue
        if frame in indexed:
            raise ValueError(f"COLMAP model contains duplicate frame {frame}")
        indexed[frame] = image
    return indexed


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def compute_fundamental_matrix(
    left_k: np.ndarray,
    right_k: np.ndarray,
    left_twc: np.ndarray,
    right_twc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return F for x_right.T F x_left = 0."""
    right_from_left = np.linalg.inv(right_twc) @ left_twc
    rotation = right_from_left[:3, :3]
    translation = right_from_left[:3, 3]
    baseline = float(np.linalg.norm(translation))
    if baseline <= 1e-9:
        raise ValueError("The selected frames have a near-zero translation baseline")
    essential = skew(translation) @ rotation
    fundamental = (
        np.linalg.inv(right_k).T @ essential @ np.linalg.inv(left_k)
    )
    norm = float(np.linalg.norm(fundamental))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("Computed a degenerate fundamental matrix")
    return fundamental / norm, right_from_left, baseline


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help="Override the COLMAP camera calibration with an image/K/D JSON file",
    )
    parser.add_argument("--left-index", type=int, default=DEFAULT_LEFT_INDEX)
    parser.add_argument("--right-index", type=int, default=DEFAULT_RIGHT_INDEX)
    parser.add_argument(
        "--left-point",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        help="Optional point in the undistorted left image",
    )
    parser.add_argument(
        "--right-point",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        help="Optional matching point in the undistorted right image",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.left_index == args.right_index:
        raise ValueError("--left-index and --right-index must differ")
    if args.right_point is not None and args.left_point is None:
        raise ValueError("--right-point requires --left-point")

    dataset = args.dataset.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    image_dir = resolve_image_directory(dataset)
    cameras, model_images, _ = read_write_model.read_model(str(model_path))
    model_frames = images_by_frame(model_images)
    missing = [
        frame
        for frame in (args.left_index, args.right_index)
        if frame not in model_frames
    ]
    if missing:
        raise KeyError(f"Frames are not registered in the COLMAP model: {missing}")

    left_model_image = model_frames[args.left_index]
    right_model_image = model_frames[args.right_index]
    left_calibration = camera_calibration(cameras[left_model_image.camera_id])
    right_calibration = camera_calibration(cameras[right_model_image.camera_id])
    if args.intrinsics is not None:
        override = json_calibration(args.intrinsics)
        model_sizes = {
            left_calibration.image_size,
            right_calibration.image_size,
        }
        if model_sizes != {override.image_size}:
            raise ValueError(
                f"Calibration override is {override.image_size[0]}x"
                f"{override.image_size[1]}, but the selected model images use "
                f"{sorted(model_sizes)}"
            )
        left_calibration = override
        right_calibration = override

    left_path = resolve_image_path(image_dir, args.left_index)
    right_path = resolve_image_path(image_dir, args.right_index)
    left_raw = read_image(left_path, left_calibration.image_size)
    right_raw = read_image(right_path, right_calibration.image_size)
    left_image = cv2.undistort(
        left_raw,
        left_calibration.intrinsic,
        left_calibration.distortion,
        None,
        left_calibration.intrinsic,
    )
    right_image = cv2.undistort(
        right_raw,
        right_calibration.intrinsic,
        right_calibration.distortion,
        None,
        right_calibration.intrinsic,
    )

    initial_left_twc = read_fastlio_twc(image_dir / f"odoms_{args.left_index}.txt")
    initial_right_twc = read_fastlio_twc(image_dir / f"odoms_{args.right_index}.txt")
    optimized_left_twc = colmap_image_twc(left_model_image)
    optimized_right_twc = colmap_image_twc(right_model_image)

    initial_f, initial_relative, initial_baseline = compute_fundamental_matrix(
        left_calibration.intrinsic,
        right_calibration.intrinsic,
        initial_left_twc,
        initial_right_twc,
    )
    optimized_f, optimized_relative, optimized_baseline = compute_fundamental_matrix(
        left_calibration.intrinsic,
        right_calibration.intrinsic,
        optimized_left_twc,
        optimized_right_twc,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else HOME_ROOT
        / "colmap_pcd_epipolar"
        / f"frame_{args.left_index:06d}_to_{args.right_index:06d}.jpg"
    )
    viewer = reference.EpipolarComparisonViewer(
        left_image, right_image, initial_f, optimized_f, output_path
    )
    if args.left_point is not None:
        viewer.set_left_point((float(args.left_point[0]), float(args.left_point[1])))
    if args.right_point is not None:
        viewer.set_right_point((float(args.right_point[0]), float(args.right_point[1])))
    if args.output is not None:
        viewer.save()

    relative_rotation_change = (
        optimized_relative[:3, :3] @ initial_relative[:3, :3].T
    )
    print(f"[colmap-pcd-epipolar] dataset: {dataset}")
    print(f"[colmap-pcd-epipolar] model: {model_path}")
    print(f"[colmap-pcd-epipolar] images: {left_path} -> {right_path}")
    print(
        "[colmap-pcd-epipolar] calibration: "
        f"left={left_calibration.model_name}, right={right_calibration.model_name}; "
        "display images are undistorted with the same K"
    )
    print(
        "[colmap-pcd-epipolar] pose sources: initial=FAST_LIO odoms_N.txt (T_wc), "
        "optimized=COLMAP images model (T_cw converted to T_wc)"
    )
    print(
        f"[colmap-pcd-epipolar] baseline: initial={initial_baseline:.6f} m, "
        f"optimized={optimized_baseline:.6f} m"
    )
    print(
        "[colmap-pcd-epipolar] relative rotation change: "
        f"{rotation_angle_degrees(relative_rotation_change):.6f} deg"
    )
    print(f"[colmap-pcd-epipolar] initial F:\n{initial_f}")
    print(f"[colmap-pcd-epipolar] optimized F:\n{optimized_f}")
    if viewer.measurements:
        initial_error = viewer.measurements["initial"][0]
        optimized_error = viewer.measurements["optimized"][0]
        print(
            f"[colmap-pcd-epipolar] measured error: initial={initial_error:.3f} px, "
            f"optimized={optimized_error:.3f} px, "
            f"delta={optimized_error - initial_error:+.3f} px"
        )

    if args.no_gui:
        return

    left_window = "Colormap epipolar - left"
    right_window = "Colormap epipolar - right"
    cv2.namedWindow(left_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(right_window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(left_window, viewer.on_left_click)
    cv2.setMouseCallback(right_window, viewer.on_right_click)
    viewer.show()
    print(
        "[colmap-pcd-epipolar] controls: click left, click matching right; "
        "r=reset, s=save, q/Esc=quit"
    )
    while True:
        if cv2.getWindowProperty(left_window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty(right_window, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            viewer.reset()
            viewer.show()
        if key == ord("s"):
            viewer.save()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
