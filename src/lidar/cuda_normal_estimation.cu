#include "lidar/cuda_normal_estimation.h"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

namespace colmap {
namespace lidar {
namespace {

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);
constexpr uint64_t kCoordMask = (uint64_t{1} << kCoordBits) - 1;

inline bool CheckCuda(cudaError_t status,
                      const char* site,
                      std::string* error) {
  if (status == cudaSuccess) return true;
  if (error != nullptr) {
    std::ostringstream stream;
    stream << site << ": " << cudaGetErrorString(status);
    *error = stream.str();
  }
  return false;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool Allocate(size_t bytes, std::string* error) {
    if (bytes == 0) return true;
    return CheckCuda(cudaMalloc(&data_, bytes), "cudaMalloc", error);
  }

  template <typename T>
  T* As() {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
};

class EventPair {
 public:
  EventPair() {
    cudaEventCreate(&begin_);
    cudaEventCreate(&end_);
  }
  ~EventPair() {
    cudaEventDestroy(begin_);
    cudaEventDestroy(end_);
  }
  void Begin() { cudaEventRecord(begin_); }
  double End() {
    cudaEventRecord(end_);
    cudaEventSynchronize(end_);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, begin_, end_);
    return milliseconds;
  }

 private:
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
};

__host__ __device__ inline uint64_t EncodeCell(int x, int y, int z) {
  const uint64_t ux = static_cast<uint64_t>(x + kCoordBias) & kCoordMask;
  const uint64_t uy = static_cast<uint64_t>(y + kCoordBias) & kCoordMask;
  const uint64_t uz = static_cast<uint64_t>(z + kCoordBias) & kCoordMask;
  return (ux << (2 * kCoordBits)) | (uy << kCoordBits) | uz;
}

__global__ void BuildKeysKernel(const float3* points,
                                size_t count,
                                float inverse_cell_size,
                                uint64_t* keys,
                                uint32_t* point_ids) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float3 point = points[index];
  const int x = __float2int_rd(point.x * inverse_cell_size);
  const int y = __float2int_rd(point.y * inverse_cell_size);
  const int z = __float2int_rd(point.z * inverse_cell_size);
  keys[index] = EncodeCell(x, y, z);
  point_ids[index] = static_cast<uint32_t>(index);
}

__device__ inline uint32_t LowerBound(const uint64_t* keys,
                                      uint32_t count,
                                      uint64_t key) {
  uint32_t first = 0;
  uint32_t length = count;
  while (length > 0) {
    const uint32_t half = length >> 1;
    const uint32_t middle = first + half;
    if (keys[middle] < key) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first;
}

__device__ inline uint32_t UpperBound(const uint64_t* keys,
                                      uint32_t count,
                                      uint64_t key) {
  uint32_t first = 0;
  uint32_t length = count;
  while (length > 0) {
    const uint32_t half = length >> 1;
    const uint32_t middle = first + half;
    if (key < keys[middle]) {
      length = half;
    } else {
      first = middle + 1;
      length -= half + 1;
    }
  }
  return first;
}

struct Statistics {
  int count = 0;
  float sx = 0.0f;
  float sy = 0.0f;
  float sz = 0.0f;
  float sxx = 0.0f;
  float sxy = 0.0f;
  float sxz = 0.0f;
  float syy = 0.0f;
  float syz = 0.0f;
  float szz = 0.0f;
};

__device__ inline void Accumulate(float dx,
                                  float dy,
                                  float dz,
                                  Statistics* statistics) {
  ++statistics->count;
  statistics->sx += dx;
  statistics->sy += dy;
  statistics->sz += dz;
  statistics->sxx += dx * dx;
  statistics->sxy += dx * dy;
  statistics->sxz += dx * dz;
  statistics->syy += dy * dy;
  statistics->syz += dy * dz;
  statistics->szz += dz * dz;
}

__device__ inline void WarpReduce(Statistics* statistics) {
  constexpr unsigned kMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    statistics->count += __shfl_down_sync(kMask, statistics->count, offset);
    statistics->sx += __shfl_down_sync(kMask, statistics->sx, offset);
    statistics->sy += __shfl_down_sync(kMask, statistics->sy, offset);
    statistics->sz += __shfl_down_sync(kMask, statistics->sz, offset);
    statistics->sxx += __shfl_down_sync(kMask, statistics->sxx, offset);
    statistics->sxy += __shfl_down_sync(kMask, statistics->sxy, offset);
    statistics->sxz += __shfl_down_sync(kMask, statistics->sxz, offset);
    statistics->syy += __shfl_down_sync(kMask, statistics->syy, offset);
    statistics->syz += __shfl_down_sync(kMask, statistics->syz, offset);
    statistics->szz += __shfl_down_sync(kMask, statistics->szz, offset);
  }
}

__device__ inline float3 InvalidNormal() {
  const float value = nanf("");
  return make_float3(value, value, value);
}

__device__ float3 SmallestEigenvector(const Statistics& statistics,
                                      const float3 query,
                                      float* curvature) {
  if (statistics.count < 3) {
    *curvature = nanf("");
    return InvalidNormal();
  }

  const float inverse_count = 1.0f / statistics.count;
  const float mx = statistics.sx * inverse_count;
  const float my = statistics.sy * inverse_count;
  const float mz = statistics.sz * inverse_count;
  float matrix[3][3];
  matrix[0][0] = statistics.sxx * inverse_count - mx * mx;
  matrix[0][1] = statistics.sxy * inverse_count - mx * my;
  matrix[0][2] = statistics.sxz * inverse_count - mx * mz;
  matrix[1][0] = matrix[0][1];
  matrix[1][1] = statistics.syy * inverse_count - my * my;
  matrix[1][2] = statistics.syz * inverse_count - my * mz;
  matrix[2][0] = matrix[0][2];
  matrix[2][1] = matrix[1][2];
  matrix[2][2] = statistics.szz * inverse_count - mz * mz;

  float vectors[3][3] = {{1.0f, 0.0f, 0.0f},
                         {0.0f, 1.0f, 0.0f},
                         {0.0f, 0.0f, 1.0f}};
  for (int iteration = 0; iteration < 8; ++iteration) {
    int p = 0;
    int q = 1;
    float maximum = fabsf(matrix[0][1]);
    if (fabsf(matrix[0][2]) > maximum) {
      p = 0;
      q = 2;
      maximum = fabsf(matrix[0][2]);
    }
    if (fabsf(matrix[1][2]) > maximum) {
      p = 1;
      q = 2;
      maximum = fabsf(matrix[1][2]);
    }
    if (maximum <= 1e-12f) break;

    const float app = matrix[p][p];
    const float aqq = matrix[q][q];
    const float apq = matrix[p][q];
    const float tau = (aqq - app) / (2.0f * apq);
    const float t = copysignf(1.0f, tau) /
                    (fabsf(tau) + sqrtf(1.0f + tau * tau));
    const float cosine = rsqrtf(1.0f + t * t);
    const float sine = t * cosine;

    matrix[p][p] = app - t * apq;
    matrix[q][q] = aqq + t * apq;
    matrix[p][q] = 0.0f;
    matrix[q][p] = 0.0f;
    for (int k = 0; k < 3; ++k) {
      if (k == p || k == q) continue;
      const float akp = matrix[k][p];
      const float akq = matrix[k][q];
      matrix[k][p] = matrix[p][k] = cosine * akp - sine * akq;
      matrix[k][q] = matrix[q][k] = sine * akp + cosine * akq;
    }
    for (int k = 0; k < 3; ++k) {
      const float vkp = vectors[k][p];
      const float vkq = vectors[k][q];
      vectors[k][p] = cosine * vkp - sine * vkq;
      vectors[k][q] = sine * vkp + cosine * vkq;
    }
  }

  int minimum_index = 0;
  if (matrix[1][1] < matrix[minimum_index][minimum_index]) minimum_index = 1;
  if (matrix[2][2] < matrix[minimum_index][minimum_index]) minimum_index = 2;
  float3 normal = make_float3(vectors[0][minimum_index],
                              vectors[1][minimum_index],
                              vectors[2][minimum_index]);
  const float norm_squared = normal.x * normal.x + normal.y * normal.y +
                             normal.z * normal.z;
  const float trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
  if (!isfinite(norm_squared) || norm_squared <= 1e-12f ||
      !isfinite(trace) || trace <= 1e-20f) {
    *curvature = nanf("");
    return InvalidNormal();
  }
  const float inverse_norm = rsqrtf(norm_squared);
  normal.x *= inverse_norm;
  normal.y *= inverse_norm;
  normal.z *= inverse_norm;

  // Match PCL NormalEstimation's default viewpoint at the sensor origin.
  const float toward_origin = -query.x * normal.x - query.y * normal.y -
                              query.z * normal.z;
  if (toward_origin < 0.0f) {
    normal.x = -normal.x;
    normal.y = -normal.y;
    normal.z = -normal.z;
  }
  *curvature = fmaxf(0.0f, matrix[minimum_index][minimum_index]) / trace;
  return normal;
}

__device__ inline void StoreNormal(const Statistics& statistics,
                                   const float3 query,
                                   float4* output) {
  float curvature = nanf("");
  const float3 normal = SmallestEigenvector(statistics, query, &curvature);
  *output = make_float4(normal.x, normal.y, normal.z, curvature);
}

__global__ void DualRadiusNormalsKernel(const float3* points,
                                        uint32_t count,
                                        const uint64_t* sorted_keys,
                                        const uint32_t* sorted_point_ids,
                                        float cell_size,
                                        float outer_radius_squared,
                                        float inner_radius_squared,
                                        float4* outer_normals,
                                        float4* inner_normals) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t query_index = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  if (query_index >= count) return;

  const float3 query = points[query_index];
  const float inverse_cell_size = 1.0f / cell_size;
  const int cell_x = __float2int_rd(query.x * inverse_cell_size);
  const int cell_y = __float2int_rd(query.y * inverse_cell_size);
  const int cell_z = __float2int_rd(query.z * inverse_cell_size);
  Statistics outer;
  Statistics inner;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        uint32_t begin = 0;
        uint32_t end = 0;
        if (lane == 0) {
          const uint64_t key = EncodeCell(cell_x + dx, cell_y + dy,
                                          cell_z + dz);
          begin = LowerBound(sorted_keys, count, key);
          end = UpperBound(sorted_keys, count, key);
        }
        begin = __shfl_sync(0xffffffffu, begin, 0);
        end = __shfl_sync(0xffffffffu, end, 0);
        for (uint32_t position = begin + lane; position < end;
             position += 32) {
          const float3 neighbor = points[sorted_point_ids[position]];
          const float rx = neighbor.x - query.x;
          const float ry = neighbor.y - query.y;
          const float rz = neighbor.z - query.z;
          const float distance_squared = rx * rx + ry * ry + rz * rz;
          if (distance_squared <= outer_radius_squared) {
            Accumulate(rx, ry, rz, &outer);
            if (distance_squared <= inner_radius_squared) {
              Accumulate(rx, ry, rz, &inner);
            }
          }
        }
      }
    }
  }

  WarpReduce(&outer);
  WarpReduce(&inner);
  if (lane == 0) {
    StoreNormal(outer, query, &outer_normals[query_index]);
    StoreNormal(inner, query, &inner_normals[query_index]);
  }
}

}  // namespace

bool EstimateDualRadiusNormalsCuda(
    const std::vector<float>& xyz,
    float outer_radius,
    float inner_radius,
    std::vector<float>* outer_normals,
    std::vector<float>* inner_normals,
    CudaNormalEstimationTiming* timing,
    std::string* error) {
  if (outer_normals == nullptr || inner_normals == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  if (xyz.empty() || xyz.size() % 3 != 0) {
    *error = "xyz must contain three floats per non-empty point set";
    return false;
  }
  if (!(outer_radius > 0.0f) || !(inner_radius > 0.0f) ||
      inner_radius > outer_radius) {
    *error = "radii must satisfy 0 < inner_radius <= outer_radius";
    return false;
  }
  const size_t count = xyz.size() / 3;
  if (count > std::numeric_limits<uint32_t>::max()) {
    *error = "point count exceeds uint32 index capacity";
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float coordinate = xyz[3 * i + axis];
      if (!std::isfinite(coordinate)) {
        *error = "input contains non-finite XYZ";
        return false;
      }
      const double cell = std::floor(coordinate / outer_radius);
      if (cell <= -kCoordBias || cell >= kCoordBias) {
        *error = "point coordinate exceeds packed cell-key range";
        return false;
      }
    }
  }

  std::vector<float3> host_points(count);
  for (size_t i = 0; i < count; ++i) {
    host_points[i] = make_float3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
  }
  DeviceBuffer device_points;
  DeviceBuffer device_keys;
  DeviceBuffer device_point_ids;
  DeviceBuffer device_outer_normals;
  DeviceBuffer device_inner_normals;
  if (!device_points.Allocate(count * sizeof(float3), error) ||
      !device_keys.Allocate(count * sizeof(uint64_t), error) ||
      !device_point_ids.Allocate(count * sizeof(uint32_t), error) ||
      !device_outer_normals.Allocate(count * sizeof(float4), error) ||
      !device_inner_normals.Allocate(count * sizeof(float4), error)) {
    return false;
  }

  EventPair timer;
  timer.Begin();
  if (!CheckCuda(cudaMemcpy(device_points.As<float3>(), host_points.data(),
                            count * sizeof(float3), cudaMemcpyHostToDevice),
                 "upload XYZ", error)) {
    return false;
  }
  if (timing != nullptr) timing->upload_ms = timer.End();

  constexpr int kThreads = 256;
  const int key_blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  timer.Begin();
  BuildKeysKernel<<<key_blocks, kThreads>>>(
      device_points.As<float3>(), count, 1.0f / outer_radius,
      device_keys.As<uint64_t>(), device_point_ids.As<uint32_t>());
  if (!CheckCuda(cudaGetLastError(), "BuildKeysKernel launch", error)) {
    return false;
  }
  thrust::device_ptr<uint64_t> key_begin(device_keys.As<uint64_t>());
  thrust::device_ptr<uint32_t> id_begin(device_point_ids.As<uint32_t>());
  try {
    thrust::sort_by_key(key_begin, key_begin + count, id_begin);
  } catch (const std::exception& exception) {
    *error = std::string("thrust::sort_by_key: ") + exception.what();
    return false;
  }
  if (!CheckCuda(cudaDeviceSynchronize(), "spatial index synchronize", error)) {
    return false;
  }
  if (timing != nullptr) timing->index_ms = timer.End();

  const uint64_t warp_count = count;
  const uint64_t thread_count = warp_count * 32;
  const int normal_blocks = static_cast<int>((thread_count + kThreads - 1) /
                                             kThreads);
  timer.Begin();
  DualRadiusNormalsKernel<<<normal_blocks, kThreads>>>(
      device_points.As<float3>(), static_cast<uint32_t>(count),
      device_keys.As<uint64_t>(), device_point_ids.As<uint32_t>(), outer_radius,
      outer_radius * outer_radius, inner_radius * inner_radius,
      device_outer_normals.As<float4>(), device_inner_normals.As<float4>());
  if (!CheckCuda(cudaGetLastError(), "DualRadiusNormalsKernel launch", error) ||
      !CheckCuda(cudaDeviceSynchronize(), "normal estimation synchronize", error)) {
    return false;
  }
  if (timing != nullptr) timing->normals_ms = timer.End();

  std::vector<float4> host_outer(count);
  std::vector<float4> host_inner(count);
  timer.Begin();
  if (!CheckCuda(cudaMemcpy(host_outer.data(), device_outer_normals.As<float4>(),
                            count * sizeof(float4), cudaMemcpyDeviceToHost),
                 "download outer normals", error) ||
      !CheckCuda(cudaMemcpy(host_inner.data(), device_inner_normals.As<float4>(),
                            count * sizeof(float4), cudaMemcpyDeviceToHost),
                 "download inner normals", error)) {
    return false;
  }
  if (timing != nullptr) timing->download_ms = timer.End();

  outer_normals->resize(count * 4);
  inner_normals->resize(count * 4);
  for (size_t i = 0; i < count; ++i) {
    (*outer_normals)[4 * i] = host_outer[i].x;
    (*outer_normals)[4 * i + 1] = host_outer[i].y;
    (*outer_normals)[4 * i + 2] = host_outer[i].z;
    (*outer_normals)[4 * i + 3] = host_outer[i].w;
    (*inner_normals)[4 * i] = host_inner[i].x;
    (*inner_normals)[4 * i + 1] = host_inner[i].y;
    (*inner_normals)[4 * i + 2] = host_inner[i].z;
    (*inner_normals)[4 * i + 3] = host_inner[i].w;
  }
  return true;
}

}  // namespace lidar
}  // namespace colmap
