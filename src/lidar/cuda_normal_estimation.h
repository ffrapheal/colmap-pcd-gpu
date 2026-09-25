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

// Builds the spatial index over support_xyz and estimates normals only for
// query_xyz. Both arrays contain three floats per point. Each output contains
// nx, ny, nz and curvature for every query point, preserving query input order.
bool EstimateDualRadiusNormalsForQueriesCuda(
    const std::vector<float>& support_xyz,
    const std::vector<float>& query_xyz,
    float outer_radius,
    float inner_radius,
    std::vector<float>* outer_normals,
    std::vector<float>* inner_normals,
    CudaNormalEstimationTiming* timing,
    std::string* error);

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
