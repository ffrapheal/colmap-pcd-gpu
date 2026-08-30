是的，那个版本过于精简。它只能让人“找到工具”，但不足以保证正确使用，尤其容易弄错：

- HTTP 根目录和端口
- `data`、`source` 参数
- `label_debug` 与 `result.glb` 的目录关系
- 点云工具是直接打开还是需要重新准备数据
- 是否应该覆盖旧结果

建议保留下面这种“路径 + 数据要求 + URL 模板 + 使用边界”的写法：

```markdown
## 可视化服务

所有项目产物统一从 `/home/nvidia` 提供 HTTP 服务。优先复用已有服务：

```bash
python3 -m http.server <PORT> --bind 0.0.0.0 --directory /home/nvidia
```

生成查看链接前，必须确认页面、GLB、OBJ 和相关元数据均返回 HTTP 200。

## 贴图查看

查看器：

`/home/nvidia/scanner_pipeline/new_code/mvs-texturing/web/interactive_mesh_source_viewer.html`

查看器需要：

- `data`：Texrecon 结果中的 `obj/label_debug/` 目录
- `source`：包含原始 `imgs_N.jpg` 的 session 目录
- `mode=glb`：默认显示 GLB，OBJ 作为面拾取代理
- `obj/` 中必须存在匹配的 `result.glb`、`result.obj`、`result.mtl` 和纹理文件
- `label_debug/atlas_face_label_map.csv` 必须与同一份 OBJ 对应

URL 模板：

```text
http://127.0.0.1:<PORT>/scanner_pipeline/new_code/mvs-texturing/web/interactive_mesh_source_viewer.html?data=/<artifact>/obj/label_debug/&source=/<artifact>/session/&mode=glb
```

不要把其他实验的 `label_debug` 与当前 OBJ/GLB 混用。

贴图主流程：

`/home/nvidia/meshcolormap/launch.sh`

Texrecon：

`/home/nvidia/new_code/mvs-texturing/build/apps/texrecon/texrecon`

正常贴图基线使用：

- `--image_resolution=high`
- `--smoothness_term=potts`
- `--outlier_removal=gauss_damping`
- `--num_threads=8`
- `--keep_unseen_faces`
- 开启 global/local seam leveling
- 建议增加 `--write_timings`

最终贴图通常使用优化位姿和原始分辨率图像。直接调用 Texrecon 时，输入依次为：

```text
<图像和CAM目录> <mesh_hole_filled.ply> <输出前缀>
```

## 点云对比

当前参考工具：

`/home/nvidia/colmap-PCD-gpu-artifacts/integration/pointcloud-comparison-20260818-160549-dual15-5-v1/`

目录包含：

- `index.html`：查看入口
- `app.js`：交互逻辑
- `prepare_data.py`：数据准备脚本
- COLMAP 彩色点云
- LiDAR 点云
- 3D 点到源图像特征的来源信息

查看 URL：

```text
http://127.0.0.1:<PORT>/colmap-PCD-gpu-artifacts/integration/<对比目录>/
```

`prepare_data.py` 目前与 0818 数据集相关。测试其他数据集时，应将它作为模板，在新的 artifact 目录中适配；不得覆盖原参考结果。

## 实验产物规则

- 每次实验必须创建新的 artifact 目录。
- 不得覆盖已有报告、贴图、点云数据或可视化结果。
- 可视化派生产物也应放入新的 artifact。
- 创建链接前必须核对输入位姿、图像数量、内参和实际 Texrecon 输入。
- “位姿报告包含 fallback”不代表“fallback 图像实际参加了贴图”，必须检查 Texrecon 的 session 图像数量。
- 上下文压缩后，应先查看本文件，再搜索工具位置。
```

这个粒度更合适：仍然精炼，但足以让后续 agent 在没有历史上下文时正确打开贴图和点云工具。当前仍未修改文件。