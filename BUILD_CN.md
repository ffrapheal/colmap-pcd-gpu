# Colmap-PCD-GPU 编译教程

本文面向 Ubuntu 20.04 / 22.04，介绍本仓库当前 `gpu-ba` 分支的三种构建方式：

1. 基础 COLMAP-PCD（CPU、可选 GUI）
2. 自研 GPU BA 框架的 CPU 后端
3. 自研 GPU BA 框架的 CUDA 后端（推荐用于本仓库的 GPU BA 功能）

> 本仓库同时存在旧版 COLMAP CUDA 路径和自研 GPU BA CUDA 路径。二者使用不同的 CMake 开关，不能只看到 `CUDA_ENABLED=OFF` 就判断自研 CUDA BA 没有启用。

## 1. 获取源码

当前开发分支为 `gpu-ba`：

```bash
git clone --branch gpu-ba https://github.com/ffrapheal/colmap-pcd-gpu.git
cd colmap-pcd-gpu
```

文档站点是可选 Git 子模块，不影响主程序编译。如需完整拉取：

```bash
git submodule update --init --recursive
```

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    ccache \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    liblz4-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libceres-dev \
    libsuitesparse-dev \
    libpcl-dev \
    libopencv-dev \
    libssl-dev
```

说明：

- `libpcl-dev` 和 `libopencv-dev` 是当前仓库相对上游 COLMAP 增加的重要依赖。
- 启用 `GPU_BA_ENABLED` 时需要 OpenSSL，因此需要 `libssl-dev`。
- Ubuntu 仓库中的 Ceres 可以直接使用；如果替换为自行编译的 Ceres，请在配置时通过 `Ceres_DIR` 指向其 CMake 配置目录。
- 只构建无界面命令行程序时仍可保留 Qt/OpenGL 依赖，以避免不同构建模式切换时重新补包。

## 3. CUDA 环境准备

仅基础 CPU 构建或 GPU BA 的 CPU 后端可以跳过本节。

先确认 NVIDIA 驱动和 CUDA 编译器可用：

```bash
nvidia-smi
nvcc --version
```

如果 `nvcc` 没有加入 `PATH`，可以在 CMake 时显式指定，例如：

```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

还需要设置 GPU 的 CUDA Compute Capability。可通过 NVIDIA 官方 GPU 列表查询，也可在支持该字段的驱动上执行：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

常见示例：

- RTX 30 系列通常为 `86`
- Jetson AGX Orin 为 `87`
- RTX 40 系列通常为 `89`

本文后续使用 `<CUDA_ARCH>` 占位，必须替换为实际数值。当前项目在本机使用过 CUDA 11.4 和架构 `87`。

## 4. 推荐：构建自研 CUDA GPU BA

建议使用源码目录之外的独立构建目录，避免污染仓库和覆盖其他实验构建：

```bash
cmake -S . -B ../colmap-PCD-gpu-build/release-cuda \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=OFF \
    -DTESTS_ENABLED=OFF \
    -DGPU_BA_TESTS_ENABLED=OFF \
    -DCUDA_ENABLED=OFF \
    -DGPU_BA_ENABLED=ON \
    -DGPU_BA_CUDA_ENABLED=ON \
    -DGPU_BA_CUDA_PERFORMANCE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=<CUDA_ARCH> \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build ../colmap-PCD-gpu-build/release-cuda --parallel "$(nproc)"
```

这里 `CUDA_ENABLED=OFF` 是有意设置的：

- `CUDA_ENABLED` 控制旧版 COLMAP CUDA/SIFT/MVS 路径。
- `GPU_BA_CUDA_ENABLED` 控制本仓库独立的 FP64 自研 CUDA BA。
- 分离两套路径可以避免旧版 CUDA 目标和 `--use_fast_math` 影响自研 BA 的数值正确性。
- `GPU_BA_CUDA_PERFORMANCE=OFF` 是正常构建建议；该开关为性能实验模式，不是正确性基线。

若机器内存有限，不要直接使用全部 CPU 核心，可改为：

```bash
cmake --build ../colmap-PCD-gpu-build/release-cuda --parallel 4
```

## 5. 构建 GPU BA 的 CPU 后端

该模式启用 GPU BA 框架和命令入口，但不编译自研 CUDA 实现，适合无 CUDA 环境、接口调试和 CPU 对照：

```bash
cmake -S . -B ../colmap-PCD-gpu-build/release-gpu-ba-cpu \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=OFF \
    -DTESTS_ENABLED=OFF \
    -DGPU_BA_TESTS_ENABLED=OFF \
    -DCUDA_ENABLED=OFF \
    -DGPU_BA_ENABLED=ON \
    -DGPU_BA_CUDA_ENABLED=OFF

cmake --build ../colmap-PCD-gpu-build/release-gpu-ba-cpu --parallel "$(nproc)"
```

## 6. 构建基础 CPU / GUI 版本

### 无界面命令行版本

```bash
cmake -S . -B ../colmap-PCD-gpu-build/release-cpu \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=OFF \
    -DCUDA_ENABLED=OFF \
    -DGPU_BA_ENABLED=OFF

cmake --build ../colmap-PCD-gpu-build/release-cpu --parallel "$(nproc)"
```

### GUI 版本

```bash
cmake -S . -B ../colmap-PCD-gpu-build/release-gui \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=ON \
    -DCUDA_ENABLED=OFF \
    -DGPU_BA_ENABLED=OFF

cmake --build ../colmap-PCD-gpu-build/release-gui --parallel "$(nproc)"
```

如果要同时启用上游 COLMAP 的旧版 CUDA 特征提取/MVS 路径，可将 `CUDA_ENABLED` 改为 `ON`，并设置 `CMAKE_CUDA_ARCHITECTURES`。这不是自研 GPU BA CUDA 后端的必要条件。

## 7. 验证编译结果

以推荐的 CUDA GPU BA 构建目录为例：

```bash
../colmap-PCD-gpu-build/release-cuda/src/exe/colmap -h
```

启用 `GPU_BA_ENABLED=ON` 后，帮助信息中应包含：

```text
gpu_ba_replay
```

检查编译缓存中的关键开关：

```bash
grep -E '^(CUDA_ENABLED|GPU_BA_ENABLED|GPU_BA_CUDA_ENABLED|GPU_BA_CUDA_PERFORMANCE|CMAKE_CUDA_ARCHITECTURES):' \
    ../colmap-PCD-gpu-build/release-cuda/CMakeCache.txt
```

推荐结果应类似：

```text
CUDA_ENABLED:BOOL=OFF
GPU_BA_ENABLED:BOOL=ON
GPU_BA_CUDA_ENABLED:BOOL=ON
GPU_BA_CUDA_PERFORMANCE:BOOL=OFF
CMAKE_CUDA_ARCHITECTURES:STRING=<CUDA_ARCH>
```

## 8. 可选安装

直接使用构建目录中的二进制最安全，不会覆盖系统中已有的 COLMAP。确认需要系统级安装后再执行：

```bash
sudo cmake --install ../colmap-PCD-gpu-build/release-cuda
colmap -h
```

如需自定义安装位置：

```bash
cmake -S . -B ../colmap-PCD-gpu-build/release-cuda \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
    -DGUI_ENABLED=OFF \
    -DCUDA_ENABLED=OFF \
    -DGPU_BA_ENABLED=ON \
    -DGPU_BA_CUDA_ENABLED=ON \
    -DGPU_BA_CUDA_PERFORMANCE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=<CUDA_ARCH> \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build ../colmap-PCD-gpu-build/release-cuda --parallel "$(nproc)"
cmake --install ../colmap-PCD-gpu-build/release-cuda
```

## 9. 启用测试

普通 COLMAP 测试和 GPU BA 测试使用不同开关：

```bash
cmake -S . -B ../colmap-PCD-gpu-build/test-cuda \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGUI_ENABLED=OFF \
    -DTESTS_ENABLED=ON \
    -DGPU_BA_ENABLED=ON \
    -DGPU_BA_TESTS_ENABLED=ON \
    -DGPU_BA_CUDA_ENABLED=ON \
    -DGPU_BA_CUDA_PERFORMANCE=OFF \
    -DCUDA_ENABLED=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=<CUDA_ARCH> \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build ../colmap-PCD-gpu-build/test-cuda --parallel 4
ctest --test-dir ../colmap-PCD-gpu-build/test-cuda --output-on-failure
```

测试和编译可能占用较多 CPU、内存和 GPU 资源，生产机器上应与正在运行的重建任务错峰执行。

## 10. 常见问题

### CMake 找不到 CUDA

显式指定 CUDA 编译器：

```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

同时检查该文件是否存在，并确认 CUDA 版本支持当前 GCC/G++。

### `unsupported GNU version`

CUDA 与系统 GCC 版本不匹配。安装 CUDA 支持的 GCC 版本后显式指定：

```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
```

然后删除失败的构建目录并重新配置，不要在旧 CMakeCache 上反复修改编译器。

### CMake 找不到 Ceres、PCL 或 OpenCV

先确认开发包已安装：

```bash
dpkg -l | grep -E 'libceres-dev|libpcl-dev|libopencv-dev'
```

自行安装的依赖可通过 `CMAKE_PREFIX_PATH` 或对应的 `<Package>_DIR` 指定。

如果 CMake 优先找到了 Conda 中不兼容的 Boost、Qt 或其他动态库，先退出 Conda 环境再重新配置：

```bash
conda deactivate
```

### GUI 启动失败或服务器没有显示器

服务器构建使用：

```bash
-DGUI_ENABLED=OFF
```

并直接运行命令行子命令，不要执行 `colmap gui`。

### 修改开关后结果异常

CMake 会缓存编译器和功能检测结果。不同模式使用不同构建目录；如必须复用目录，先彻底删除旧目录再重新配置：

```bash
rm -rf ../colmap-PCD-gpu-build/release-cuda
```

不要删除源码目录，也不要覆盖仍需保留的实验构建或结果目录。
