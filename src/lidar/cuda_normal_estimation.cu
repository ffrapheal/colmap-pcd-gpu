#include "lidar/cuda_normal_estimation.h"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <vector>

namespace colmap {
namespace lidar {
namespace {

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);
constexpr uint64_t kCoordMask = (uint64_t{1} << kCoordBits) - 1;
constexpr int kThreadsPerBlock = 256;
constexpr int kThreadsPerQuery = 32;

static_assert(kThreadsPerBlock % kThreadsPerQuery == 0,
              "normal kernel blocks must contain complete warps");
static_assert(
    static_cast<uint64_t>(std::numeric_limits<int>::max()) +
            kThreadsPerQuery - 1 <=
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
    "INT_MAX support points plus a warp stride must fit uint32_t");
static_assert(
    (static_cast<uint64_t>(std::numeric_limits<int>::max()) *
             kThreadsPerQuery +
         kThreadsPerBlock - 1) /
            kThreadsPerBlock <=
        static_cast<uint64_t>(std::numeric_limits<int>::max()),
    "INT_MAX query points must fit the int CUDA grid block count");

inline bool CheckCuda(cudaError_t status,
                      const char* site,
                      std::string* error) {
  if (status == cudaSuccess) return true;
  if (error != nullptr) {
    std::ostringstream stream;
    if (!error->empty()) stream << *error << "; ";
    stream << site << ": " << cudaGetErrorString(status);
    *error = stream.str();
  }
  return false;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
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
  EventPair() = default;
  ~EventPair() {
    if (end_ != nullptr) cudaEventDestroy(end_);
    if (begin_ != nullptr) cudaEventDestroy(begin_);
  }

  bool Initialize(std::string* error) {
    if (!CheckCuda(cudaEventCreate(&begin_), "cudaEventCreate(begin)", error)) {
      return false;
    }
    if (!CheckCuda(cudaEventCreate(&end_), "cudaEventCreate(end)", error)) {
      CheckCuda(cudaEventDestroy(begin_),
                "cudaEventDestroy(begin) after create failure", error);
      begin_ = nullptr;
      return false;
    }
    return true;
  }

  bool Begin(std::string* error) {
    return CheckCuda(cudaEventRecord(begin_), "cudaEventRecord(begin)", error);
  }

  bool End(double* milliseconds, std::string* error) {
    if (!CheckCuda(cudaEventRecord(end_), "cudaEventRecord(end)", error) ||
        !CheckCuda(cudaEventSynchronize(end_), "cudaEventSynchronize(end)",
                   error)) {
      return false;
    }
    float elapsed_ms = 0.0f;
    if (!CheckCuda(cudaEventElapsedTime(&elapsed_ms, begin_, end_),
                   "cudaEventElapsedTime", error)) {
      return false;
    }
    *milliseconds = elapsed_ms;
    return true;
  }

  bool Destroy(std::string* error) {
    bool success = true;
    if (end_ != nullptr) {
      if (!CheckCuda(cudaEventDestroy(end_), "cudaEventDestroy(end)", error)) {
        success = false;
      }
      end_ = nullptr;
    }
    if (begin_ != nullptr) {
      if (!CheckCuda(cudaEventDestroy(begin_), "cudaEventDestroy(begin)",
                     error)) {
        success = false;
      }
      begin_ = nullptr;
    }
    return success;
  }

 private:
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t end_ = nullptr;
};

bool ValidatePointSet(const std::vector<float>& xyz,
                      const char* name,
                      float inverse_cell_size,
                      size_t* count,
                      std::string* error) {
  if (xyz.empty() || xyz.size() % 3 != 0) {
    *error = std::string(name) +
             " must contain three floats per non-empty point set";
    return false;
  }
  *count = xyz.size() / 3;
  if (*count >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    *error = std::string(name) +
             " point count exceeds INT_MAX kernel signed-count capacity";
    return false;
  }
  for (size_t i = 0; i < *count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const float coordinate = xyz[3 * i + axis];
      if (!std::isfinite(coordinate)) {
        *error = std::string(name) + " contains non-finite XYZ";
        return false;
      }
      const float scaled_coordinate = coordinate * inverse_cell_size;
      if (!std::isfinite(scaled_coordinate)) {
        *error = std::string(name) +
                 " coordinate exceeds packed cell-key range";
        return false;
      }
      const double cell = std::floor(static_cast<double>(scaled_coordinate));
      if (cell <= -kCoordBias || cell >= kCoordBias) {
        *error = std::string(name) +
                 " coordinate exceeds packed cell-key range";
        return false;
      }
    }
  }
  return true;
}

bool ComputeKernelBlockCount(size_t item_count,
                             uint64_t threads_per_item,
                             const char* kernel_name,
                             int* block_count,
                             std::string* error) {
  const uint64_t thread_count =
      static_cast<uint64_t>(item_count) * threads_per_item;
  const uint64_t blocks =
      (thread_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    *error = std::string(kernel_name) +
             " grid exceeds INT_MAX kernel block capacity";
    return false;
  }
  *block_count = static_cast<int>(blocks);
  return true;
}

std::vector<float3> PackPoints(const std::vector<float>& xyz) {
  std::vector<float3> points(xyz.size() / 3);
  for (size_t i = 0; i < points.size(); ++i) {
    points[i] = make_float3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
  }
  return points;
}

__host__ __device__ inline uint64_t EncodeCell(int x, int y, int z) {
  const uint64_t ux = static_cast<uint64_t>(x + kCoordBias) & kCoordMask;
  const uint64_t uy = static_cast<uint64_t>(y + kCoordBias) & kCoordMask;
  const uint64_t uz = static_cast<uint64_t>(z + kCoordBias) & kCoordMask;
  return (ux << (2 * kCoordBits)) | (uy << kCoordBits) | uz;
}

__device__ inline bool IsEncodableCell(int x, int y, int z) {
  return x > -kCoordBias && x < kCoordBias && y > -kCoordBias &&
         y < kCoordBias && z > -kCoordBias && z < kCoordBias;
}

__global__ void BuildKeysKernel(const float3* points,
                                uint32_t count,
                                float inverse_cell_size,
                                uint64_t* keys,
                                uint32_t* point_ids) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
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

__global__ void DualRadiusNormalsKernel(const float3* support_points,
                                        uint32_t support_count,
                                        const float3* query_points,
                                        uint32_t query_count,
                                        const uint64_t* sorted_keys,
                                        const uint32_t* sorted_point_ids,
                                        float cell_size,
                                        float outer_radius_squared,
                                        float inner_radius_squared,
                                        float4* outer_normals,
                                        float4* inner_normals) {
  const uint64_t global_thread =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t query_index = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  if (query_index >= query_count) return;

  const float3 query = query_points[query_index];
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
        const int neighbor_x = cell_x + dx;
        const int neighbor_y = cell_y + dy;
        const int neighbor_z = cell_z + dz;
        if (lane == 0 &&
            IsEncodableCell(neighbor_x, neighbor_y, neighbor_z)) {
          const uint64_t key =
              EncodeCell(neighbor_x, neighbor_y, neighbor_z);
          begin = LowerBound(sorted_keys, support_count, key);
          end = UpperBound(sorted_keys, support_count, key);
        }
        begin = __shfl_sync(0xffffffffu, begin, 0);
        end = __shfl_sync(0xffffffffu, end, 0);
        for (uint32_t position = begin + lane; position < end;
             position += 32) {
          const float3 neighbor =
              support_points[sorted_point_ids[position]];
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

bool EstimateDualRadiusNormalsForQueriesCuda(
    const std::vector<float>& support_xyz,
    const std::vector<float>& query_xyz,
    float outer_radius,
    float inner_radius,
    std::vector<float>* outer_normals,
    std::vector<float>* inner_normals,
    CudaNormalEstimationTiming* timing,
    std::string* error) {
  if (outer_normals != nullptr) outer_normals->clear();
  if (inner_normals != nullptr) inner_normals->clear();
  if (timing != nullptr) *timing = CudaNormalEstimationTiming{};
  if (error != nullptr) error->clear();

  if (outer_normals == nullptr || inner_normals == nullptr || error == nullptr) {
    if (error != nullptr) {
      *error = "outer_normals, inner_normals and error must not be null";
    }
    return false;
  }
  if (!std::isfinite(outer_radius) || !std::isfinite(inner_radius) ||
      !(outer_radius > 0.0f) || !(inner_radius > 0.0f) ||
      inner_radius > outer_radius) {
    *error = "radii must be finite and satisfy 0 < inner_radius <= "
             "outer_radius";
    return false;
  }
  const float inverse_cell_size = 1.0f / outer_radius;
  const float outer_radius_squared = outer_radius * outer_radius;
  const float inner_radius_squared = inner_radius * inner_radius;
  if (!std::isfinite(inverse_cell_size) ||
      !std::isfinite(outer_radius_squared) ||
      !std::isfinite(inner_radius_squared) ||
      !(outer_radius_squared > 0.0f) || !(inner_radius_squared > 0.0f)) {
    *error = "radii exceed supported CUDA float arithmetic range";
    return false;
  }
  size_t support_count = 0;
  size_t query_count = 0;
  if (!ValidatePointSet(support_xyz, "support_xyz", inverse_cell_size,
                        &support_count, error) ||
      !ValidatePointSet(query_xyz, "query_xyz", inverse_cell_size,
                        &query_count, error)) {
    return false;
  }
  const uint32_t support_count_u32 = static_cast<uint32_t>(support_count);
  const uint32_t query_count_u32 = static_cast<uint32_t>(query_count);
  int key_blocks = 0;
  int normal_blocks = 0;
  if (!ComputeKernelBlockCount(support_count, 1, "BuildKeysKernel",
                               &key_blocks, error) ||
      !ComputeKernelBlockCount(query_count, kThreadsPerQuery,
                               "DualRadiusNormalsKernel", &normal_blocks,
                               error)) {
    return false;
  }

  const bool queries_are_support = &support_xyz == &query_xyz;
  const std::vector<float3> host_support_points = PackPoints(support_xyz);
  const std::vector<float3> host_query_points =
      queries_are_support ? std::vector<float3>() : PackPoints(query_xyz);
  DeviceBuffer device_support_points;
  DeviceBuffer device_query_points;
  DeviceBuffer device_keys;
  DeviceBuffer device_point_ids;
  DeviceBuffer device_outer_normals;
  DeviceBuffer device_inner_normals;
  if (!device_support_points.Allocate(support_count * sizeof(float3), error) ||
      (!queries_are_support &&
       !device_query_points.Allocate(query_count * sizeof(float3), error)) ||
      !device_keys.Allocate(support_count * sizeof(uint64_t), error) ||
      !device_point_ids.Allocate(support_count * sizeof(uint32_t), error) ||
      !device_outer_normals.Allocate(query_count * sizeof(float4), error) ||
      !device_inner_normals.Allocate(query_count * sizeof(float4), error)) {
    return false;
  }
  const float3* device_queries =
      queries_are_support ? device_support_points.As<float3>()
                          : device_query_points.As<float3>();

  EventPair timer;
  if (!timer.Initialize(error)) {
    return false;
  }
  CudaNormalEstimationTiming measured_timing;
  std::vector<float4> host_outer(query_count);
  std::vector<float4> host_inner(query_count);
  const bool computation_succeeded = [&]() {
    if (!timer.Begin(error) ||
        !CheckCuda(cudaMemcpy(device_support_points.As<float3>(),
                              host_support_points.data(),
                              support_count * sizeof(float3),
                              cudaMemcpyHostToDevice),
                   "upload support XYZ", error) ||
        (!queries_are_support &&
         !CheckCuda(cudaMemcpy(device_query_points.As<float3>(),
                               host_query_points.data(),
                               query_count * sizeof(float3),
                               cudaMemcpyHostToDevice),
                    "upload query XYZ", error)) ||
        !timer.End(&measured_timing.upload_ms, error)) {
      return false;
    }

    if (!timer.Begin(error)) return false;
    BuildKeysKernel<<<key_blocks, kThreadsPerBlock>>>(
        device_support_points.As<float3>(), support_count_u32, inverse_cell_size,
        device_keys.As<uint64_t>(), device_point_ids.As<uint32_t>());
    if (!CheckCuda(cudaGetLastError(), "BuildKeysKernel launch", error)) {
      return false;
    }
    thrust::device_ptr<uint64_t> key_begin(device_keys.As<uint64_t>());
    thrust::device_ptr<uint32_t> id_begin(device_point_ids.As<uint32_t>());
    try {
      thrust::stable_sort_by_key(key_begin, key_begin + support_count, id_begin);
    } catch (const std::exception& exception) {
      *error = std::string("thrust::stable_sort_by_key: ") + exception.what();
      return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "spatial index synchronize", error) ||
        !timer.End(&measured_timing.index_ms, error)) {
      return false;
    }

    if (!timer.Begin(error)) return false;
    DualRadiusNormalsKernel<<<normal_blocks, kThreadsPerBlock>>>(
        device_support_points.As<float3>(), support_count_u32, device_queries,
        query_count_u32, device_keys.As<uint64_t>(),
        device_point_ids.As<uint32_t>(), outer_radius, outer_radius_squared,
        inner_radius_squared, device_outer_normals.As<float4>(),
        device_inner_normals.As<float4>());
    if (!CheckCuda(cudaGetLastError(), "DualRadiusNormalsKernel launch", error) ||
        !CheckCuda(cudaDeviceSynchronize(), "normal estimation synchronize",
                   error) ||
        !timer.End(&measured_timing.normals_ms, error)) {
      return false;
    }

    if (!timer.Begin(error) ||
        !CheckCuda(cudaMemcpy(host_outer.data(),
                              device_outer_normals.As<float4>(),
                              query_count * sizeof(float4),
                              cudaMemcpyDeviceToHost),
                   "download outer normals", error) ||
        !CheckCuda(cudaMemcpy(host_inner.data(),
                              device_inner_normals.As<float4>(),
                              query_count * sizeof(float4),
                              cudaMemcpyDeviceToHost),
                   "download inner normals", error) ||
        !timer.End(&measured_timing.download_ms, error)) {
      return false;
    }
    return true;
  }();
  const bool timer_destroyed = timer.Destroy(error);
  if (!computation_succeeded || !timer_destroyed) return false;

  std::vector<float> outer_result(query_count * 4);
  std::vector<float> inner_result(query_count * 4);
  for (size_t i = 0; i < query_count; ++i) {
    outer_result[4 * i] = host_outer[i].x;
    outer_result[4 * i + 1] = host_outer[i].y;
    outer_result[4 * i + 2] = host_outer[i].z;
    outer_result[4 * i + 3] = host_outer[i].w;
    inner_result[4 * i] = host_inner[i].x;
    inner_result[4 * i + 1] = host_inner[i].y;
    inner_result[4 * i + 2] = host_inner[i].z;
    inner_result[4 * i + 3] = host_inner[i].w;
  }
  outer_normals->swap(outer_result);
  inner_normals->swap(inner_result);
  if (timing != nullptr) *timing = measured_timing;
  return true;
}

bool EstimateDualRadiusNormalsCuda(
    const std::vector<float>& xyz,
    float outer_radius,
    float inner_radius,
    std::vector<float>* outer_normals,
    std::vector<float>* inner_normals,
    CudaNormalEstimationTiming* timing,
    std::string* error) {
  return EstimateDualRadiusNormalsForQueriesCuda(
      xyz, xyz, outer_radius, inner_radius, outer_normals, inner_normals,
      timing, error);
}

}  // namespace lidar
}  // namespace colmap
