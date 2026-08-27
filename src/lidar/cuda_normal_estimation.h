#ifndef COLMAP_LIDAR_CUDA_NORMAL_ESTIMATION_H
#define COLMAP_LIDAR_CUDA_NORMAL_ESTIMATION_H

#include <cstddef>
#include <string>
#include <vector>

namespace colmap {
namespace lidar {

struct CudaNormalEstimationTiming {
  double upload_ms = 0.0;
  double index_ms = 0.0;
  double normals_ms = 0.0;
  double download_ms = 0.0;
};

// xyz contains three floats per point. Each output contains nx, ny, nz and
// curvature for every input point, preserving input order.
bool EstimateDualRadiusNormalsCuda(
    const std::vector<float>& xyz,
    float outer_radius,
    float inner_radius,
    std::vector<float>* outer_normals,
    std::vector<float>* inner_normals,
    CudaNormalEstimationTiming* timing,
    std::string* error);

}  // namespace lidar
}  // namespace colmap

#endif  // COLMAP_LIDAR_CUDA_NORMAL_ESTIMATION_H
