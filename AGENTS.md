# COLMAP-PCD-GPU Project Paths

This file extends `/home/nvidia/AGENTS.md`.

## Main paths

- Repository: `/home/nvidia/colmap-PCD-gpu`
- Datasets: `/home/nvidia/colmap-PCD-gpu/Data`
- Build: `/home/nvidia/colmap-PCD-gpu-build`
- Artifacts: `/home/nvidia/colmap-PCD-gpu-artifacts`
- Reports: `/home/nvidia/colmap-PCD-gpu-artifacts/reports`

## Visualization server

Serve `/home/nvidia` as the HTTP root:

```bash
python3 -m http.server <PORT> --bind 0.0.0.0 --directory /home/nvidia
Reuse an existing server when possible.
Textured mesh
- Pipeline: /home/nvidia/meshcolormap/launch.sh
- Texrecon:
  /home/nvidia/new_code/mvs-texturing/build/apps/texrecon/texrecon
- Viewer:
  /home/nvidia/scanner_pipeline/new_code/mvs-texturing/web/interactive_mesh_source_viewer.html
Viewer URL:
http://127.0.0.1:<PORT>/scanner_pipeline/new_code/mvs-texturing/web/interactive_mesh_source_viewer.html?data=/<label_debug-path>&source=/<session-path>&mode=glb
Final texturing normally uses optimized poses with original-resolution images.
COLMAP and LiDAR point clouds
Current comparison-viewer reference:
/home/nvidia/colmap-PCD-gpu-artifacts/integration/
pointcloud-comparison-20260818-160549-dual15-5-v1/
It contains index.html, app.js, prepare_data.py, colored COLMAP points,
LiDAR points, and point-to-source-feature provenance.
Viewer URL:
http://127.0.0.1:<PORT>/colmap-PCD-gpu-artifacts/integration/<comparison-directory>/
prepare_data.py is currently dataset-specific; use it as a template.
Artifact rules
- Create a new artifact directory for every experiment.
- Do not overwrite frozen reports or previous visualization results.
- After context compaction, consult this file before searching for these tools.

实验过程中相机 内参 应当使用  /home/nvidia/colmap-PCD-gpu/Data/camera_0_intrinsics.json