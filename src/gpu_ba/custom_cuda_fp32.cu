#include "gpu_ba/custom_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace colmap {
namespace gpu_ba {
namespace {

constexpr uint32_t kInvalidTarget = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kSegmentSize = 64;
constexpr uint32_t kPosePartialScalars = 42;
constexpr uint32_t kPointPartialScalars = 12;
constexpr uint32_t kEdgePartialScalars = 18;
constexpr uint32_t kSchurPartialScalars = 36;
constexpr uint32_t kBlockSize = 128;

template <typename T>
bool CheckedBytes(const size_t count, uint64_t* bytes) {
  if (bytes == nullptr ||
      count > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
    return false;
  }
  *bytes = static_cast<uint64_t>(count) * sizeof(T);
  return true;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { Reset(); }

  bool Allocate(const size_t count,
                CudaPersistentDeviceRuntimeInfo* runtime,
                std::string* error) {
    Reset();
    count_ = count;
    if (count == 0) return true;
    uint64_t bytes = 0;
    if (!CheckedBytes<T>(count, &bytes)) {
      *error = "FP32 buffer byte size overflow";
      return false;
    }
    const cudaError_t status = cudaMalloc(&data_, static_cast<size_t>(bytes));
    if (status != cudaSuccess) {
      *error = std::string("FP32 cudaMalloc failed: ") +
               cudaGetErrorString(status);
      data_ = nullptr;
      count_ = 0;
      return false;
    }
    if (runtime != nullptr) {
      ++runtime->float_buffer_allocation_calls;
      runtime->float_buffer_allocation_bytes += bytes;
      runtime->float_arena_reserved_bytes += bytes;
    }
    return true;
  }

  bool Upload(const std::vector<T>& host,
              cudaStream_t stream,
              CudaPersistentDeviceRuntimeInfo* runtime,
              std::string* error) {
    if (host.size() != count_) {
      *error = "FP32 upload size mismatch";
      return false;
    }
    if (host.empty()) return true;
    const size_t bytes = host.size() * sizeof(T);
    const cudaError_t status = cudaMemcpyAsync(
        data_, host.data(), bytes, cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
      *error = std::string("FP32 static upload failed: ") +
               cudaGetErrorString(status);
      return false;
    }
    if (runtime != nullptr) {
      ++runtime->float_static_upload_calls;
      runtime->float_static_upload_bytes += bytes;
    }
    return true;
  }

  bool Download(std::vector<T>* host,
                cudaStream_t stream,
                std::string* error) const {
    host->resize(count_);
    if (count_ == 0) return true;
    const cudaError_t status = cudaMemcpyAsync(
        host->data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      *error = std::string("FP32 download failed: ") +
               cudaGetErrorString(status);
      return false;
    }
    return true;
  }

  void Reset() {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
    count_ = 0;
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

struct FImageState {
  double quaternion[4];
  double translation[3];
};

struct FPointState {
  double xyz[3];
};

struct FCameraState {
  double params[8];
};

struct FVisualStatic {
  uint32_t image_entity;
  uint32_t point_entity;
  uint32_t camera_entity;
  uint32_t reserved;
  float observation[2];
};

struct FVisualCostStatic {
  uint32_t image_entity;
  uint32_t point_entity;
  uint32_t camera_entity;
  uint32_t reserved;
  double observation[2];
};

struct FLidarStatic {
  uint32_t point_entity;
  uint8_t mode;
  uint8_t reserved[3];
  float plane[4];
  float weight;
  float near_zero_threshold;
};

struct FLidarCostStatic {
  uint32_t point_entity;
  uint8_t mode;
  uint8_t reserved[3];
  double plane[4];
  double weight;
  double near_zero_threshold;
};

struct FVisualOutput {
  float residual[2];
  float pose_jacobian[12];
  float point_jacobian[6];
  uint32_t finite;
};

struct FLidarOutput {
  float residual;
  float point_jacobian[3];
  uint32_t finite;
};

struct FRecordStatusSummary {
  uint64_t visual_failure_count;
  uint64_t lidar_failure_count;
  int64_t first_visual_failure;
  int64_t first_lidar_failure;
};

struct FPoseMeta {
  uint32_t image_id;
  uint32_t image_entity;
  uint32_t dimension;
  int32_t free_translation_indices[3];
  uint32_t adjacency_begin;
  uint32_t adjacency_end;
};

struct FPointMeta {
  uint64_t point_id;
  uint32_t point_entity;
  uint32_t adjacency_begin;
  uint32_t adjacency_end;
};

struct FPointAdjacency {
  uint32_t residual_index;
  uint8_t residual_kind;
  uint8_t reserved[3];
};

struct FEdgeMeta {
  uint32_t pose_index;
  uint32_t point_index;
  uint32_t pose_dimension;
  uint32_t adjacency_begin;
  uint32_t adjacency_end;
};

struct FSegment {
  uint32_t target_index;
  uint32_t adjacency_begin;
  uint32_t adjacency_end;
  uint32_t reserved;
};

struct FRange {
  uint32_t begin;
  uint32_t end;
};

struct FPoseBlock {
  uint32_t image_id;
  uint32_t dimension;
  int32_t free_translation_indices[3];
  float hessian[36];
  float gradient[6];
  float jacobi_scaling[6];
  float damping[6];
};

struct FPointBlock {
  uint64_t point_id;
  float hessian[9];
  float gradient[3];
  float jacobi_scaling[3];
  float damping[3];
};

struct FEdgeBlock {
  uint32_t pose_index;
  uint32_t point_index;
  uint32_t pose_dimension;
  uint32_t reserved;
  float value[18];
};

struct FSolvePoseMeta {
  uint32_t offset;
  uint32_t dimension;
  uint32_t edge_begin;
  uint32_t edge_end;
};

struct FSolvePointMeta {
  uint32_t edge_begin;
  uint32_t edge_end;
};

struct FPairMeta {
  uint32_t lhs_pose;
  uint32_t rhs_pose;
  uint32_t adjacency_begin;
  uint32_t adjacency_end;
};

struct FPairContribution {
  uint32_t point_index;
  uint32_t lhs_edge;
  uint32_t rhs_edge;
};

struct FSchurSegment {
  uint32_t pair_index;
  uint32_t contribution_begin;
  uint32_t contribution_end;
  uint32_t reserved;
};

struct FCostEntry {
  uint32_t output_index;
  uint8_t residual_kind;
  uint8_t reserved[3];
};

struct FImageUpdateMeta {
  uint32_t variable_pose_index;
};

struct FPointUpdateMeta {
  uint32_t variable_point_index;
};

struct FGradientSummary {
  float projected;
  float raw;
  float scaled;
  uint32_t finite;
};

struct FStepSummary {
  float predicted_reduction;
  float backward_error;
  float step_norm;
  uint32_t finite;
};

struct Fp32Topology {
  std::vector<FVisualStatic> visual;
  std::vector<FVisualCostStatic> visual_cost;
  std::vector<FLidarStatic> lidar;
  std::vector<FLidarCostStatic> lidar_cost;
  std::vector<FCostEntry> cost_order;
  std::vector<FPoseMeta> poses;
  std::vector<FPointMeta> points;
  std::vector<FEdgeMeta> edges;
  std::vector<uint32_t> pose_adjacency;
  std::vector<FPointAdjacency> point_adjacency;
  std::vector<uint32_t> edge_adjacency;
  std::vector<FSegment> pose_segments;
  std::vector<FSegment> point_segments;
  std::vector<FSegment> edge_segments;
  std::vector<FRange> pose_ranges;
  std::vector<FRange> point_ranges;
  std::vector<FRange> edge_ranges;
  std::vector<FSolvePoseMeta> solve_poses;
  std::vector<FSolvePointMeta> solve_points;
  std::vector<uint32_t> pose_edges;
  std::vector<uint32_t> point_edges;
  std::vector<FPairMeta> pairs;
  std::vector<FPairContribution> contributions;
  std::vector<FSchurSegment> schur_segments;
  std::vector<FRange> pair_segment_ranges;
  std::vector<FImageUpdateMeta> image_updates;
  std::vector<FPointUpdateMeta> point_updates;
  uint32_t pose_dimension = 0;
};

template <typename T>
bool FitsUint32(const T value) {
  return value <= static_cast<T>(std::numeric_limits<uint32_t>::max());
}

bool BuildSegments(const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
                   std::vector<FSegment>* segments,
                   std::vector<FRange>* target_ranges,
                   std::string* error) {
  target_ranges->resize(ranges.size());
  for (size_t target = 0; target < ranges.size(); ++target) {
    if (!FitsUint32(target) || !FitsUint32(segments->size())) {
      *error = "FP32 assembly segment index overflow";
      return false;
    }
    FRange range{static_cast<uint32_t>(segments->size()), 0};
    uint32_t cursor = ranges[target].first;
    while (cursor < ranges[target].second) {
      const uint32_t end = std::min<uint32_t>(
          ranges[target].second, cursor + kSegmentSize);
      segments->push_back(
          {static_cast<uint32_t>(target), cursor, end, 0});
      cursor = end;
    }
    if (!FitsUint32(segments->size())) {
      *error = "FP32 assembly segment count overflow";
      return false;
    }
    range.end = static_cast<uint32_t>(segments->size());
    (*target_ranges)[target] = range;
  }
  return true;
}

bool BuildFp32Topology(const Snapshot& snapshot,
                       Fp32Topology* topology,
                       std::string* error) {
  if (topology == nullptr || error == nullptr) return false;
  *topology = Fp32Topology();
  if (!ValidateSnapshot(snapshot, error)) return false;
  if (snapshot.metadata.refine_focal_length ||
      snapshot.metadata.refine_principal_point ||
      snapshot.metadata.refine_extra_params) {
    *error = "FP32 experiment requires fixed intrinsics";
    return false;
  }

  std::unordered_map<uint32_t, uint32_t> image_entity;
  std::unordered_map<uint64_t, uint32_t> point_entity;
  std::unordered_map<uint32_t, uint32_t> camera_entity;
  for (size_t i = 0; i < snapshot.images.size(); ++i) {
    if (!FitsUint32(i) ||
        !image_entity.emplace(snapshot.images[i].image_id,
                              static_cast<uint32_t>(i)).second) {
      *error = "FP32 image entity index is invalid";
      return false;
    }
  }
  for (size_t i = 0; i < snapshot.points.size(); ++i) {
    if (!FitsUint32(i) ||
        !point_entity.emplace(snapshot.points[i].point3D_id,
                              static_cast<uint32_t>(i)).second) {
      *error = "FP32 point entity index is invalid";
      return false;
    }
  }
  for (size_t i = 0; i < snapshot.cameras.size(); ++i) {
    if (!FitsUint32(i) || snapshot.cameras[i].params.size() != 8 ||
        !camera_entity.emplace(snapshot.cameras[i].camera_id,
                               static_cast<uint32_t>(i)).second) {
      *error = "FP32 camera entity layout is invalid";
      return false;
    }
  }

  std::vector<const ObservationSnapshot*> observations;
  observations.reserve(snapshot.observations.size());
  for (const auto& value : snapshot.observations) observations.push_back(&value);
  std::sort(observations.begin(), observations.end(),
            [](const ObservationSnapshot* lhs,
               const ObservationSnapshot* rhs) {
              return lhs->source_index < rhs->source_index;
            });
  std::unordered_map<uint64_t, uint32_t> visual_by_source;
  for (size_t i = 0; i < observations.size(); ++i) {
    const auto& value = *observations[i];
    const auto image = image_entity.find(value.image_id);
    const auto point = point_entity.find(value.point3D_id);
    if (!FitsUint32(i) || image == image_entity.end() ||
        point == point_entity.end()) {
      *error = "FP32 visual residual references a missing entity";
      return false;
    }
    const uint32_t camera = camera_entity.at(
        snapshot.images[image->second].camera_id);
    FVisualStatic fixed{};
    fixed.image_entity = image->second;
    fixed.point_entity = point->second;
    fixed.camera_entity = camera;
    fixed.observation[0] = static_cast<float>(value.xy[0]);
    fixed.observation[1] = static_cast<float>(value.xy[1]);
    topology->visual.push_back(fixed);
    FVisualCostStatic cost{};
    cost.image_entity = fixed.image_entity;
    cost.point_entity = fixed.point_entity;
    cost.camera_entity = fixed.camera_entity;
    cost.observation[0] = value.xy[0];
    cost.observation[1] = value.xy[1];
    topology->visual_cost.push_back(cost);
    visual_by_source.emplace(value.source_index, static_cast<uint32_t>(i));
  }

  std::vector<const LidarSnapshot*> lidar;
  lidar.reserve(snapshot.lidar.size());
  for (const auto& value : snapshot.lidar) lidar.push_back(&value);
  std::sort(lidar.begin(), lidar.end(),
            [](const LidarSnapshot* lhs, const LidarSnapshot* rhs) {
              return lhs->source_index < rhs->source_index;
            });
  std::unordered_map<uint64_t, uint32_t> lidar_by_source;
  LidarResidualMode lidar_mode = LidarResidualMode::kLegacyExact;
  if (!ParseLidarResidualMode(snapshot.metadata.lidar_residual_mode,
                              &lidar_mode)) {
    *error = "FP32 LiDAR residual mode is invalid";
    return false;
  }
  for (size_t i = 0; i < lidar.size(); ++i) {
    const auto& value = *lidar[i];
    const auto point = point_entity.find(value.point3D_id);
    if (!FitsUint32(i) || point == point_entity.end()) {
      *error = "FP32 LiDAR residual references a missing point";
      return false;
    }
    FLidarStatic fixed{};
    fixed.point_entity = point->second;
    fixed.mode = static_cast<uint8_t>(lidar_mode);
    for (size_t j = 0; j < 4; ++j)
      fixed.plane[j] = static_cast<float>(value.plane[j]);
    fixed.weight = static_cast<float>(value.weight);
    fixed.near_zero_threshold = 1e-12f;
    topology->lidar.push_back(fixed);
    FLidarCostStatic cost{};
    cost.point_entity = fixed.point_entity;
    cost.mode = fixed.mode;
    for (size_t j = 0; j < 4; ++j) cost.plane[j] = value.plane[j];
    cost.weight = value.weight;
    cost.near_zero_threshold = 1e-12;
    topology->lidar_cost.push_back(cost);
    lidar_by_source.emplace(value.source_index, static_cast<uint32_t>(i));
  }

  std::vector<OrderEntrySnapshot> order = snapshot.source_insertion_order;
  if (order.empty()) {
    order.reserve(observations.size() + lidar.size());
    for (const auto* value : observations)
      order.push_back({value->source_index, ResidualKind::kVisual,
                       value->image_id, value->point2D_idx,
                       value->point3D_id});
    for (const auto* value : lidar)
      order.push_back({value->source_index, ResidualKind::kLidar, 0, 0,
                       value->point3D_id});
    std::sort(order.begin(), order.end(),
              [](const OrderEntrySnapshot& lhs,
                 const OrderEntrySnapshot& rhs) {
                return lhs.source_index < rhs.source_index;
              });
  }
  for (const auto& value : order) {
    FCostEntry entry{};
    entry.residual_kind = value.residual_kind == ResidualKind::kVisual ? 0 : 1;
    if (entry.residual_kind == 0) {
      const auto it = visual_by_source.find(value.source_index);
      if (it == visual_by_source.end()) {
        *error = "FP32 cost order misses a visual residual";
        return false;
      }
      entry.output_index = it->second;
    } else {
      const auto it = lidar_by_source.find(value.source_index);
      if (it == lidar_by_source.end()) {
        *error = "FP32 cost order misses a LiDAR residual";
        return false;
      }
      entry.output_index = it->second;
    }
    topology->cost_order.push_back(entry);
  }

  std::unordered_map<uint32_t, uint32_t> pose_index;
  std::unordered_map<uint64_t, uint32_t> variable_point_index;
  std::set<uint32_t> pose_seen;
  std::set<uint64_t> point_seen;
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    if (parameter.kind == ParameterKind::kQuaternion) {
      const uint32_t id = static_cast<uint32_t>(parameter.entity_id);
      const auto entity = image_entity.find(id);
      if (entity != image_entity.end() &&
          !snapshot.images[entity->second].pose_constant &&
          pose_seen.insert(id).second) {
        const ImageSnapshot& image = snapshot.images[entity->second];
        FPoseMeta pose{};
        pose.image_id = id;
        pose.image_entity = entity->second;
        pose.dimension = 3;
        for (size_t i = 0; i < 3; ++i) pose.free_translation_indices[i] = -1;
        for (int t = 0; t < 3; ++t) {
          if ((image.constant_tvec_mask & (1u << t)) == 0) {
            pose.free_translation_indices[pose.dimension - 3] = t;
            ++pose.dimension;
          }
        }
        pose_index.emplace(id, static_cast<uint32_t>(topology->poses.size()));
        topology->poses.push_back(pose);
      }
    } else if (parameter.kind == ParameterKind::kPoint3D) {
      const auto entity = point_entity.find(parameter.entity_id);
      if (entity != point_entity.end() &&
          !snapshot.points[entity->second].constant &&
          point_seen.insert(parameter.entity_id).second) {
        FPointMeta point{};
        point.point_id = parameter.entity_id;
        point.point_entity = entity->second;
        variable_point_index.emplace(
            parameter.entity_id,
            static_cast<uint32_t>(topology->points.size()));
        topology->points.push_back(point);
      }
    }
  }
  const size_t variable_pose_count = std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const ImageSnapshot& value) { return !value.pose_constant; });
  const size_t variable_point_count = std::count_if(
      snapshot.points.begin(), snapshot.points.end(),
      [](const PointSnapshot& value) { return !value.constant; });
  if (topology->poses.size() != variable_pose_count ||
      topology->points.size() != variable_point_count) {
    *error = "FP32 parameter order does not cover every variable entity";
    return false;
  }

  std::vector<std::vector<uint32_t>> pose_entries(topology->poses.size());
  std::vector<std::vector<FPointAdjacency>> point_entries(
      topology->points.size());
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edge_entries;
  for (size_t i = 0; i < observations.size(); ++i) {
    const auto pose = pose_index.find(observations[i]->image_id);
    const auto point = variable_point_index.find(observations[i]->point3D_id);
    if (pose != pose_index.end())
      pose_entries[pose->second].push_back(static_cast<uint32_t>(i));
    if (pose != pose_index.end() && point != variable_point_index.end()) {
      edge_entries[{pose->second, point->second}].push_back(
          static_cast<uint32_t>(i));
    }
  }
  for (const FCostEntry& entry : topology->cost_order) {
    const uint64_t point_id = entry.residual_kind == 0
        ? observations[entry.output_index]->point3D_id
        : lidar[entry.output_index]->point3D_id;
    const auto point = variable_point_index.find(point_id);
    if (point != variable_point_index.end()) {
      point_entries[point->second].push_back(
          {entry.output_index, entry.residual_kind, {0, 0, 0}});
    }
  }
  std::vector<std::pair<uint32_t, uint32_t>> pose_ranges;
  for (size_t i = 0; i < topology->poses.size(); ++i) {
    auto& pose = topology->poses[i];
    pose.adjacency_begin = topology->pose_adjacency.size();
    topology->pose_adjacency.insert(topology->pose_adjacency.end(),
                                    pose_entries[i].begin(),
                                    pose_entries[i].end());
    pose.adjacency_end = topology->pose_adjacency.size();
    pose_ranges.emplace_back(pose.adjacency_begin, pose.adjacency_end);
  }
  std::vector<std::pair<uint32_t, uint32_t>> point_ranges;
  for (size_t i = 0; i < topology->points.size(); ++i) {
    auto& point = topology->points[i];
    point.adjacency_begin = topology->point_adjacency.size();
    topology->point_adjacency.insert(topology->point_adjacency.end(),
                                     point_entries[i].begin(),
                                     point_entries[i].end());
    point.adjacency_end = topology->point_adjacency.size();
    point_ranges.emplace_back(point.adjacency_begin, point.adjacency_end);
  }
  std::vector<std::pair<uint32_t, uint32_t>> edge_ranges;
  for (const auto& entry : edge_entries) {
    FEdgeMeta edge{};
    edge.pose_index = entry.first.first;
    edge.point_index = entry.first.second;
    edge.pose_dimension = topology->poses[edge.pose_index].dimension;
    edge.adjacency_begin = topology->edge_adjacency.size();
    topology->edge_adjacency.insert(topology->edge_adjacency.end(),
                                    entry.second.begin(), entry.second.end());
    edge.adjacency_end = topology->edge_adjacency.size();
    edge_ranges.emplace_back(edge.adjacency_begin, edge.adjacency_end);
    topology->edges.push_back(edge);
  }
  if (!BuildSegments(pose_ranges, &topology->pose_segments,
                     &topology->pose_ranges, error) ||
      !BuildSegments(point_ranges, &topology->point_segments,
                     &topology->point_ranges, error) ||
      !BuildSegments(edge_ranges, &topology->edge_segments,
                     &topology->edge_ranges, error)) {
    return false;
  }

  topology->solve_poses.resize(topology->poses.size());
  topology->solve_points.resize(topology->points.size());
  std::vector<std::vector<uint32_t>> pose_edges(topology->poses.size());
  std::vector<std::vector<uint32_t>> point_edges(topology->points.size());
  for (size_t i = 0; i < topology->poses.size(); ++i) {
    topology->solve_poses[i].offset = topology->pose_dimension;
    topology->solve_poses[i].dimension = topology->poses[i].dimension;
    topology->pose_dimension += topology->poses[i].dimension;
  }
  for (size_t i = 0; i < topology->edges.size(); ++i) {
    pose_edges[topology->edges[i].pose_index].push_back(i);
    point_edges[topology->edges[i].point_index].push_back(i);
  }
  for (size_t i = 0; i < pose_edges.size(); ++i) {
    topology->solve_poses[i].edge_begin = topology->pose_edges.size();
    topology->pose_edges.insert(topology->pose_edges.end(),
                                pose_edges[i].begin(), pose_edges[i].end());
    topology->solve_poses[i].edge_end = topology->pose_edges.size();
  }
  for (size_t i = 0; i < point_edges.size(); ++i) {
    std::sort(point_edges[i].begin(), point_edges[i].end(),
              [topology](uint32_t lhs, uint32_t rhs) {
                return topology->edges[lhs].pose_index <
                       topology->edges[rhs].pose_index;
              });
    topology->solve_points[i].edge_begin = topology->point_edges.size();
    topology->point_edges.insert(topology->point_edges.end(),
                                 point_edges[i].begin(), point_edges[i].end());
    topology->solve_points[i].edge_end = topology->point_edges.size();
  }
  std::map<std::pair<uint32_t, uint32_t>,
           std::vector<FPairContribution>> pair_entries;
  for (size_t point = 0; point < point_edges.size(); ++point) {
    for (size_t lhs = 0; lhs < point_edges[point].size(); ++lhs) {
      for (size_t rhs = lhs; rhs < point_edges[point].size(); ++rhs) {
        const uint32_t lhs_edge = point_edges[point][lhs];
        const uint32_t rhs_edge = point_edges[point][rhs];
        pair_entries[{topology->edges[lhs_edge].pose_index,
                      topology->edges[rhs_edge].pose_index}]
            .push_back({static_cast<uint32_t>(point), lhs_edge, rhs_edge});
      }
    }
  }
  for (const auto& entry : pair_entries) {
    FPairMeta pair{};
    pair.lhs_pose = entry.first.first;
    pair.rhs_pose = entry.first.second;
    pair.adjacency_begin = topology->contributions.size();
    topology->contributions.insert(topology->contributions.end(),
                                   entry.second.begin(), entry.second.end());
    pair.adjacency_end = topology->contributions.size();
    topology->pairs.push_back(pair);
  }
  topology->pair_segment_ranges.resize(topology->pairs.size());
  for (size_t pair_index = 0; pair_index < topology->pairs.size();
       ++pair_index) {
    const FPairMeta& pair = topology->pairs[pair_index];
    FRange range{static_cast<uint32_t>(topology->schur_segments.size()), 0};
    for (uint32_t begin = pair.adjacency_begin; begin < pair.adjacency_end;) {
      const uint32_t end = std::min<uint32_t>(pair.adjacency_end,
                                              begin + kSegmentSize);
      topology->schur_segments.push_back(
          {static_cast<uint32_t>(pair_index), begin, end, 0});
      begin = end;
    }
    range.end = topology->schur_segments.size();
    topology->pair_segment_ranges[pair_index] = range;
  }

  topology->image_updates.assign(snapshot.images.size(), {kInvalidTarget});
  for (size_t i = 0; i < topology->poses.size(); ++i)
    topology->image_updates[topology->poses[i].image_entity] =
        {static_cast<uint32_t>(i)};
  topology->point_updates.assign(snapshot.points.size(), {kInvalidTarget});
  for (size_t i = 0; i < topology->points.size(); ++i)
    topology->point_updates[topology->points[i].point_entity] =
        {static_cast<uint32_t>(i)};
  return true;
}

__device__ inline bool FiniteF(const float value) { return isfinite(value); }

__device__ inline float LossScaleF(const float squared_norm,
                                   const uint8_t loss_mode,
                                   const float loss_scale) {
  if (loss_mode != static_cast<uint8_t>(CudaLossMode::kSoftL1)) return 1.0f;
  const float root = sqrtf(1.0f + squared_norm / (loss_scale * loss_scale));
  return sqrtf(fmaxf(FLT_MIN, 1.0f / root));
}

__device__ inline double LossCostD(const double squared_norm,
                                   const uint8_t loss_mode,
                                   const double loss_scale) {
  if (loss_mode != static_cast<uint8_t>(CudaLossMode::kSoftL1))
    return 0.5 * squared_norm;
  const double scale2 = loss_scale * loss_scale;
  return scale2 * (sqrt(1.0 + squared_norm / scale2) - 1.0);
}

__device__ inline void EvaluateVisualF(const FVisualStatic& fixed,
                                       const FImageState& image,
                                       const FPointState& point,
                                       const FCameraState& camera,
                                       FVisualOutput* output) {
  *output = FVisualOutput{};
  const float w = static_cast<float>(image.quaternion[0]);
  const float x = static_cast<float>(image.quaternion[1]);
  const float y = static_cast<float>(image.quaternion[2]);
  const float z = static_cast<float>(image.quaternion[3]);
  const float tx = static_cast<float>(image.translation[0]);
  const float ty = static_cast<float>(image.translation[1]);
  const float tz = static_cast<float>(image.translation[2]);
  const float px = static_cast<float>(point.xyz[0]);
  const float py = static_cast<float>(point.xyz[1]);
  const float pz = static_cast<float>(point.xyz[2]);
  float c[8];
  for (uint32_t i = 0; i < 8; ++i)
    c[i] = static_cast<float>(camera.params[i]);
  if (!FiniteF(w) || !FiniteF(x) || !FiniteF(y) || !FiniteF(z) ||
      !FiniteF(tx) || !FiniteF(ty) || !FiniteF(tz) || !FiniteF(px) ||
      !FiniteF(py) || !FiniteF(pz)) return;
  for (float value : c) if (!FiniteF(value)) return;

  const float t2 = w * x;
  const float t3 = w * y;
  const float t4 = w * z;
  const float t5 = -x * x;
  const float t6 = x * y;
  const float t7 = x * z;
  const float t8 = -y * y;
  const float t9 = y * z;
  const float t1 = -z * z;
  const float rotation[9] = {
      2.0f * (t8 + t1) + 1.0f, 2.0f * (t6 - t4),
      2.0f * (t3 + t7), 2.0f * (t4 + t6),
      2.0f * (t5 + t1) + 1.0f, 2.0f * (t9 - t2),
      2.0f * (t7 - t3), 2.0f * (t2 + t9),
      2.0f * (t5 + t8) + 1.0f};
  const float X = rotation[0] * px + rotation[1] * py +
                  rotation[2] * pz + tx;
  const float Y = rotation[3] * px + rotation[4] * py +
                  rotation[5] * pz + ty;
  const float Z = rotation[6] * px + rotation[7] * py +
                  rotation[8] * pz + tz;
  if (!FiniteF(Z) || fabsf(Z) <= FLT_MIN) return;
  const float u = X / Z;
  const float v = Y / Z;
  const float u2 = u * u;
  const float uv = u * v;
  const float v2 = v * v;
  const float r2 = u2 + v2;
  const float radial = c[4] * r2 + c[5] * r2 * r2;
  const float du = u * radial + 2.0f * c[6] * uv +
                   c[7] * (r2 + 2.0f * u2);
  const float dv = v * radial + 2.0f * c[7] * uv +
                   c[6] * (r2 + 2.0f * v2);
  const float distorted_u = u + du;
  const float distorted_v = v + dv;
  output->residual[0] = c[0] * distorted_u + c[2] - fixed.observation[0];
  output->residual[1] = c[1] * distorted_v + c[3] - fixed.observation[1];

  const float radial_u = 2.0f * u * (c[4] + 2.0f * c[5] * r2);
  const float radial_v = 2.0f * v * (c[4] + 2.0f * c[5] * r2);
  const float ddu_du = 1.0f + radial + u * radial_u + 2.0f * c[6] * v +
                       6.0f * c[7] * u;
  const float ddu_dv = u * radial_v + 2.0f * c[6] * u + 2.0f * c[7] * v;
  const float ddv_du = v * radial_u + 2.0f * c[7] * v + 2.0f * c[6] * u;
  const float ddv_dv = 1.0f + radial + v * radial_v + 2.0f * c[7] * u +
                       6.0f * c[6] * v;
  const float inverse_z = 1.0f / Z;
  const float normalized[6] =
      {inverse_z, 0.0f, -u * inverse_z,
       0.0f, inverse_z, -v * inverse_z};
  const float distortion[4] =
      {c[0] * ddu_du, c[0] * ddu_dv,
       c[1] * ddv_du, c[1] * ddv_dv};
  float camera_point_jacobian[6];
  for (uint32_t row = 0; row < 2; ++row) {
    for (uint32_t col = 0; col < 3; ++col) {
      camera_point_jacobian[row * 3 + col] =
          distortion[row * 2] * normalized[col] +
          distortion[row * 2 + 1] * normalized[3 + col];
    }
  }
  for (uint32_t row = 0; row < 2; ++row) {
    for (uint32_t col = 0; col < 3; ++col) {
      output->point_jacobian[row * 3 + col] =
          camera_point_jacobian[row * 3] * rotation[col] +
          camera_point_jacobian[row * 3 + 1] * rotation[3 + col] +
          camera_point_jacobian[row * 3 + 2] * rotation[6 + col];
    }
  }
  const float rotated_q_jacobian[12] = {
      2.0f * (-z * py + y * pz), 2.0f * (y * py + z * pz),
      2.0f * (-2.0f * y * px + x * py + w * pz),
      2.0f * (-2.0f * z * px - w * py + x * pz),
      2.0f * (z * px - x * pz),
      2.0f * (y * px - 2.0f * x * py - w * pz),
      2.0f * (x * px + z * pz),
      2.0f * (w * px - 2.0f * z * py + y * pz),
      2.0f * (-y * px + x * py),
      2.0f * (z * px + w * py - 2.0f * x * pz),
      2.0f * (-w * px + z * py - 2.0f * y * pz),
      2.0f * (x * px + y * py)};
  float ambient[8];
  for (uint32_t row = 0; row < 2; ++row) {
    for (uint32_t col = 0; col < 4; ++col) {
      ambient[row * 4 + col] =
          camera_point_jacobian[row * 3] * rotated_q_jacobian[col] +
          camera_point_jacobian[row * 3 + 1] * rotated_q_jacobian[4 + col] +
          camera_point_jacobian[row * 3 + 2] * rotated_q_jacobian[8 + col];
    }
  }
  const float plus[12] =
      {-x, -y, -z, w, z, -y, -z, w, x, y, -x, w};
  for (uint32_t row = 0; row < 2; ++row) {
    for (uint32_t col = 0; col < 3; ++col) {
      float value = 0.0f;
      for (uint32_t ambient_index = 0; ambient_index < 4;
           ++ambient_index) {
        value += ambient[row * 4 + ambient_index] *
                 plus[ambient_index * 3 + col];
      }
      output->pose_jacobian[row * 6 + col] = value;
      output->pose_jacobian[row * 6 + 3 + col] =
          camera_point_jacobian[row * 3 + col];
    }
  }
  bool finite = true;
  for (float value : output->residual) finite &= FiniteF(value);
  for (float value : output->pose_jacobian) finite &= FiniteF(value);
  for (float value : output->point_jacobian) finite &= FiniteF(value);
  output->finite = finite ? 1u : 0u;
}

__global__ void EvaluateVisualKernelF(const FVisualStatic* fixed,
                                      const FImageState* images,
                                      const FPointState* points,
                                      const FCameraState* cameras,
                                      FVisualOutput* output,
                                      size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const FVisualStatic value = fixed[index];
  EvaluateVisualF(value, images[value.image_entity],
                  points[value.point_entity], cameras[value.camera_entity],
                  &output[index]);
}

__global__ void EvaluateLidarKernelF(const FLidarStatic* fixed,
                                     const FPointState* points,
                                     FLidarOutput* output,
                                     size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const FLidarStatic value = fixed[index];
  const FPointState point = points[value.point_entity];
  const float x = static_cast<float>(point.xyz[0]);
  const float y = static_cast<float>(point.xyz[1]);
  const float z = static_cast<float>(point.xyz[2]);
  const float d = fmaf(
      value.plane[0], x,
      fmaf(value.plane[1], y,
           fmaf(value.plane[2], z, value.plane[3])));
  float derivative = 0.0f;
  float residual = 0.0f;
  if (value.mode == static_cast<uint8_t>(LidarResidualMode::kSigned)) {
    residual = value.weight * d;
    derivative = value.weight;
  } else {
    const float magnitude = sqrtf(d * d);
    residual = value.weight * magnitude;
    if (value.mode ==
            static_cast<uint8_t>(LidarResidualMode::kLegacyGuarded) &&
        fabsf(d) <= value.near_zero_threshold) {
      derivative = 0.0f;
    } else {
      derivative = value.weight * d / magnitude;
    }
  }
  output[index].residual = residual;
  for (uint32_t i = 0; i < 3; ++i)
    output[index].point_jacobian[i] = derivative * value.plane[i];
  output[index].finite =
      FiniteF(residual) && FiniteF(output[index].point_jacobian[0]) &&
              FiniteF(output[index].point_jacobian[1]) &&
              FiniteF(output[index].point_jacobian[2])
          ? 1u
          : 0u;
}

__global__ void RecordStatusSummaryKernelF(const FVisualOutput* visual,
                                           size_t visual_count,
                                           const FLidarOutput* lidar,
                                           size_t lidar_count,
                                           FRecordStatusSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  FRecordStatusSummary value{0, 0, -1, -1};
  for (size_t i = 0; i < visual_count; ++i) {
    if (visual[i].finite == 0) {
      if (value.first_visual_failure < 0)
        value.first_visual_failure = static_cast<int64_t>(i);
      ++value.visual_failure_count;
    }
  }
  for (size_t i = 0; i < lidar_count; ++i) {
    if (lidar[i].finite == 0) {
      if (value.first_lidar_failure < 0)
        value.first_lidar_failure = static_cast<int64_t>(i);
      ++value.lidar_failure_count;
    }
  }
  *summary = value;
}

__device__ inline float PoseJacobianValue(const FVisualOutput& visual,
                                          const FPoseMeta& pose,
                                          const uint32_t row,
                                          const uint32_t column) {
  if (column < 3) return visual.pose_jacobian[row * 6 + column];
  return visual.pose_jacobian[
      row * 6 + 3 + pose.free_translation_indices[column - 3]];
}

__global__ void PoseSegmentPartialKernelF(
    const FSegment* segments,
    const uint32_t* adjacency,
    const FPoseMeta* poses,
    const FVisualOutput* visual,
    uint8_t loss_mode,
    float loss_scale,
    float* partials,
    size_t segment_count) {
  const size_t segment_index = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (segment_index >= segment_count || scalar >= kPosePartialScalars) return;
  const FSegment segment = segments[segment_index];
  const FPoseMeta pose = poses[segment.target_index];
  const bool gradient = scalar >= 36;
  const uint32_t row_index = gradient ? scalar - 36 : scalar / 6;
  const uint32_t col_index = gradient ? 0 : scalar % 6;
  float sum = 0.0f;
  if (row_index < pose.dimension && (gradient || col_index < pose.dimension)) {
    for (uint32_t entry = segment.adjacency_begin;
         entry < segment.adjacency_end; ++entry) {
      const FVisualOutput value = visual[adjacency[entry]];
      const float scale = LossScaleF(
          value.residual[0] * value.residual[0] +
              value.residual[1] * value.residual[1],
          loss_mode, loss_scale);
      for (uint32_t residual_row = 0; residual_row < 2; ++residual_row) {
        const float ji = PoseJacobianValue(value, pose, residual_row,
                                           row_index) * scale;
        sum += gradient
            ? ji * (value.residual[residual_row] * scale)
            : ji * (PoseJacobianValue(value, pose, residual_row,
                                      col_index) * scale);
      }
    }
  }
  partials[segment_index * kPosePartialScalars + scalar] = sum;
}

__global__ void PointSegmentPartialKernelF(
    const FSegment* segments,
    const FPointAdjacency* adjacency,
    const FVisualOutput* visual,
    const FLidarOutput* lidar,
    uint8_t loss_mode,
    float loss_scale,
    float* partials,
    size_t segment_count) {
  const size_t segment_index = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (segment_index >= segment_count || scalar >= kPointPartialScalars) return;
  const FSegment segment = segments[segment_index];
  const bool gradient = scalar >= 9;
  const uint32_t row_index = gradient ? scalar - 9 : scalar / 3;
  const uint32_t col_index = gradient ? 0 : scalar % 3;
  float sum = 0.0f;
  for (uint32_t entry = segment.adjacency_begin;
       entry < segment.adjacency_end; ++entry) {
    const FPointAdjacency binding = adjacency[entry];
    if (binding.residual_kind == 0) {
      const FVisualOutput value = visual[binding.residual_index];
      const float scale = LossScaleF(
          value.residual[0] * value.residual[0] +
              value.residual[1] * value.residual[1],
          loss_mode, loss_scale);
      for (uint32_t residual_row = 0; residual_row < 2; ++residual_row) {
        const float ji = value.point_jacobian[residual_row * 3 + row_index] *
                         scale;
        sum += gradient
            ? ji * (value.residual[residual_row] * scale)
            : ji * (value.point_jacobian[residual_row * 3 + col_index] *
                    scale);
      }
    } else {
      const FLidarOutput value = lidar[binding.residual_index];
      const float scale = LossScaleF(value.residual * value.residual,
                                     loss_mode, loss_scale);
      const float ji = value.point_jacobian[row_index] * scale;
      sum += gradient ? ji * (value.residual * scale)
                      : ji * (value.point_jacobian[col_index] * scale);
    }
  }
  partials[segment_index * kPointPartialScalars + scalar] = sum;
}

__global__ void EdgeSegmentPartialKernelF(
    const FSegment* segments,
    const uint32_t* adjacency,
    const FEdgeMeta* edges,
    const FPoseMeta* poses,
    const FVisualOutput* visual,
    uint8_t loss_mode,
    float loss_scale,
    float* partials,
    size_t segment_count) {
  const size_t segment_index = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (segment_index >= segment_count || scalar >= kEdgePartialScalars) return;
  const FSegment segment = segments[segment_index];
  const FEdgeMeta edge = edges[segment.target_index];
  const FPoseMeta pose = poses[edge.pose_index];
  const uint32_t row_index = scalar / 3;
  const uint32_t col_index = scalar % 3;
  float sum = 0.0f;
  if (row_index < pose.dimension) {
    for (uint32_t entry = segment.adjacency_begin;
         entry < segment.adjacency_end; ++entry) {
      const FVisualOutput value = visual[adjacency[entry]];
      const float scale = LossScaleF(
          value.residual[0] * value.residual[0] +
              value.residual[1] * value.residual[1],
          loss_mode, loss_scale);
      for (uint32_t residual_row = 0; residual_row < 2; ++residual_row) {
        sum += (PoseJacobianValue(value, pose, residual_row, row_index) *
                scale) *
               (value.point_jacobian[residual_row * 3 + col_index] * scale);
      }
    }
  }
  partials[segment_index * kEdgePartialScalars + scalar] = sum;
}

__global__ void PoseSegmentMergeKernelF(
    const FPoseMeta* metadata,
    const FRange* ranges,
    const float* partials,
    const float* frozen_scaling,
    float min_lm_diagonal,
    float max_lm_diagonal,
    FPoseBlock* output,
    size_t count) {
  const size_t target = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (target >= count) return;
  const FPoseMeta pose = metadata[target];
  const FRange range = ranges[target];
  if (scalar == 0) {
    output[target].image_id = pose.image_id;
    output[target].dimension = pose.dimension;
    for (uint32_t i = 0; i < 3; ++i)
      output[target].free_translation_indices[i] =
          pose.free_translation_indices[i];
  }
  if (scalar >= kPosePartialScalars) return;
  float sum = 0.0f;
  for (uint32_t segment = range.begin; segment < range.end; ++segment)
    sum += partials[segment * kPosePartialScalars + scalar];
  if (scalar < 36) {
    const uint32_t row = scalar / 6;
    const uint32_t col = scalar % 6;
    if (row < pose.dimension && col < pose.dimension)
      output[target].hessian[row * pose.dimension + col] = sum;
    if (row == col && row < pose.dimension) {
      const float diagonal = fmaxf(0.0f, sum);
      const float scale = frozen_scaling == nullptr
          ? 1.0f / (1.0f + sqrtf(diagonal))
          : frozen_scaling[target * 6 + row];
      const float scaled = sum * scale * scale;
      const float clamped =
          fminf(fmaxf(scaled, min_lm_diagonal), max_lm_diagonal);
      output[target].jacobi_scaling[row] = scale;
      output[target].damping[row] = clamped / (scale * scale);
    }
  } else {
    const uint32_t row = scalar - 36;
    if (row < pose.dimension) output[target].gradient[row] = sum;
  }
}

__global__ void PointSegmentMergeKernelF(
    const FPointMeta* metadata,
    const FRange* ranges,
    const float* partials,
    const float* frozen_scaling,
    float min_lm_diagonal,
    float max_lm_diagonal,
    FPointBlock* output,
    size_t count) {
  const size_t target = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (target >= count) return;
  const FRange range = ranges[target];
  if (scalar == 0) output[target].point_id = metadata[target].point_id;
  if (scalar >= kPointPartialScalars) return;
  float sum = 0.0f;
  for (uint32_t segment = range.begin; segment < range.end; ++segment)
    sum += partials[segment * kPointPartialScalars + scalar];
  if (scalar < 9) {
    output[target].hessian[scalar] = sum;
    const uint32_t row = scalar / 3;
    const uint32_t col = scalar % 3;
    if (row == col) {
      const float diagonal = fmaxf(0.0f, sum);
      const float scale = frozen_scaling == nullptr
          ? 1.0f / (1.0f + sqrtf(diagonal))
          : frozen_scaling[target * 3 + row];
      const float scaled = sum * scale * scale;
      const float clamped =
          fminf(fmaxf(scaled, min_lm_diagonal), max_lm_diagonal);
      output[target].jacobi_scaling[row] = scale;
      output[target].damping[row] = clamped / (scale * scale);
    }
  } else {
    output[target].gradient[scalar - 9] = sum;
  }
}

__global__ void EdgeSegmentMergeKernelF(
    const FEdgeMeta* metadata,
    const FRange* ranges,
    const float* partials,
    FEdgeBlock* output,
    size_t count) {
  const size_t target = blockIdx.x;
  const uint32_t scalar = threadIdx.x;
  if (target >= count) return;
  const FEdgeMeta edge = metadata[target];
  if (scalar == 0) {
    output[target].pose_index = edge.pose_index;
    output[target].point_index = edge.point_index;
    output[target].pose_dimension = edge.pose_dimension;
  }
  if (scalar >= kEdgePartialScalars) return;
  float sum = 0.0f;
  const FRange range = ranges[target];
  for (uint32_t segment = range.begin; segment < range.end; ++segment)
    sum += partials[segment * kEdgePartialScalars + scalar];
  if (scalar < edge.pose_dimension * 3) output[target].value[scalar] = sum;
}

__global__ void CaptureScalingKernelF(const FPoseBlock* poses,
                                      size_t pose_count,
                                      const FPointBlock* points,
                                      size_t point_count,
                                      float* pose_scaling,
                                      float* point_scaling) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < pose_count * 6) {
    const size_t pose = index / 6;
    const size_t row = index % 6;
    pose_scaling[index] = row < poses[pose].dimension
        ? poses[pose].jacobi_scaling[row] : 0.0f;
  }
  if (index < point_count * 3) {
    const size_t point = index / 3;
    const size_t row = index % 3;
    point_scaling[index] = points[point].jacobi_scaling[row];
  }
}

__device__ inline void QuaternionPlusF(const float* q,
                                       const float* delta,
                                       float* result) {
  const float norm =
      sqrtf(delta[0] * delta[0] + delta[1] * delta[1] +
            delta[2] * delta[2]);
  if (norm == 0.0f) {
    for (uint32_t i = 0; i < 4; ++i) result[i] = q[i];
    return;
  }
  const float sin_delta_by_delta = sinf(norm) / norm;
  const float d[4] = {cosf(norm), sin_delta_by_delta * delta[0],
                      sin_delta_by_delta * delta[1],
                      sin_delta_by_delta * delta[2]};
  result[0] = d[0] * q[0] - d[1] * q[1] - d[2] * q[2] - d[3] * q[3];
  result[1] = d[0] * q[1] + d[1] * q[0] + d[2] * q[3] - d[3] * q[2];
  result[2] = d[0] * q[2] - d[1] * q[3] + d[2] * q[0] + d[3] * q[1];
  result[3] = d[0] * q[3] + d[1] * q[2] - d[2] * q[1] + d[3] * q[0];
}

__global__ void GradientSummaryKernelF(const FPoseMeta* metadata,
                                       const FPoseBlock* poses,
                                       size_t pose_count,
                                       const FPointBlock* points,
                                       size_t point_count,
                                       const FImageState* images,
                                       FGradientSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  FGradientSummary value{0.0f, 0.0f, 0.0f, 1u};
  for (size_t pose_index = 0; pose_index < pose_count; ++pose_index) {
    const FPoseBlock pose = poses[pose_index];
    float q[4];
    for (uint32_t i = 0; i < 4; ++i)
      q[i] = static_cast<float>(images[metadata[pose_index].image_entity]
                                    .quaternion[i]);
    const float negative_gradient[3] =
        {-pose.gradient[0], -pose.gradient[1], -pose.gradient[2]};
    float plus[4];
    QuaternionPlusF(q, negative_gradient, plus);
    for (uint32_t i = 0; i < 4; ++i)
      value.projected = fmaxf(value.projected, fabsf(q[i] - plus[i]));
    for (uint32_t i = 0; i < pose.dimension; ++i) {
      value.raw = fmaxf(value.raw, fabsf(pose.gradient[i]));
      value.scaled = fmaxf(
          value.scaled,
          fabsf(pose.gradient[i]) /
              fmaxf(sqrtf(fmaxf(pose.hessian[i * pose.dimension + i],
                                 1e-12f)),
                    1e-12f));
      if (i >= 3)
        value.projected = fmaxf(value.projected, fabsf(pose.gradient[i]));
    }
  }
  for (size_t point = 0; point < point_count; ++point) {
    for (uint32_t i = 0; i < 3; ++i) {
      value.projected = fmaxf(value.projected,
                              fabsf(points[point].gradient[i]));
      value.raw = fmaxf(value.raw, fabsf(points[point].gradient[i]));
      value.scaled = fmaxf(
          value.scaled,
          fabsf(points[point].gradient[i]) /
              fmaxf(sqrtf(fmaxf(points[point].hessian[i * 3 + i],
                                 1e-12f)),
                    1e-12f));
    }
  }
  value.finite = FiniteF(value.projected) && FiniteF(value.raw) &&
                         FiniteF(value.scaled)
                     ? 1u
                     : 0u;
  *summary = value;
}

__device__ inline bool FactorPoint3x3F(const float* matrix, float* inverse) {
  const float l00_squared = matrix[0];
  if (!(l00_squared > 0.0f) || !FiniteF(l00_squared)) return false;
  const float l00 = sqrtf(l00_squared);
  const float l10 = matrix[3] / l00;
  const float l20 = matrix[6] / l00;
  const float l11_squared = matrix[4] - l10 * l10;
  if (!(l11_squared > 0.0f) || !FiniteF(l11_squared)) return false;
  const float l11 = sqrtf(l11_squared);
  const float l21 = (matrix[7] - l20 * l10) / l11;
  const float l22_squared = matrix[8] - l20 * l20 - l21 * l21;
  if (!(l22_squared > 0.0f) || !FiniteF(l22_squared)) return false;
  const float l22 = sqrtf(l22_squared);
  for (uint32_t column = 0; column < 3; ++column) {
    const float b0 = column == 0 ? 1.0f : 0.0f;
    const float b1 = column == 1 ? 1.0f : 0.0f;
    const float b2 = column == 2 ? 1.0f : 0.0f;
    const float y0 = b0 / l00;
    const float y1 = (b1 - l10 * y0) / l11;
    const float y2 = (b2 - l20 * y0 - l21 * y1) / l22;
    const float x2 = y2 / l22;
    const float x1 = (y1 - l21 * x2) / l11;
    const float x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    inverse[column] = x0;
    inverse[3 + column] = x1;
    inverse[6 + column] = x2;
  }
  for (uint32_t i = 0; i < 9; ++i)
    if (!FiniteF(inverse[i])) return false;
  return true;
}

__global__ void PointFactorKernelF(const FPointBlock* points,
                                   float lambda,
                                   float* inverse,
                                   float* inverse_gradient,
                                   uint8_t* status,
                                   size_t count) {
  const size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= count) return;
  float damped[9];
  for (uint32_t i = 0; i < 9; ++i) damped[i] = points[point].hessian[i];
  for (uint32_t i = 0; i < 3; ++i)
    damped[i * 3 + i] += lambda * points[point].damping[i];
  float local_inverse[9];
  const bool success = FactorPoint3x3F(damped, local_inverse);
  status[point] = success ? 0u : 1u;
  for (uint32_t i = 0; i < 9; ++i)
    inverse[point * 9 + i] = success ? local_inverse[i] : nanf("");
  for (uint32_t row = 0; row < 3; ++row) {
    inverse_gradient[point * 3 + row] = success
        ? local_inverse[row * 3] * points[point].gradient[0] +
              local_inverse[row * 3 + 1] * points[point].gradient[1] +
              local_inverse[row * 3 + 2] * points[point].gradient[2]
        : nanf("");
  }
}

__global__ void TransformEdgeKernelF(const FEdgeBlock* edges,
                                     const float* point_inverse,
                                     float* transformed,
                                     size_t count) {
  const size_t edge_index = blockIdx.x;
  const uint32_t element = threadIdx.x;
  if (edge_index >= count || element >= 18) return;
  const FEdgeBlock edge = edges[edge_index];
  const uint32_t row = element / 3;
  const uint32_t col = element % 3;
  float value = 0.0f;
  if (row < edge.pose_dimension) {
    const float* inverse = point_inverse + edge.point_index * 9;
    value = edge.value[row * 3] * inverse[col] +
            edge.value[row * 3 + 1] * inverse[3 + col] +
            edge.value[row * 3 + 2] * inverse[6 + col];
  }
  transformed[edge_index * 18 + element] = value;
}

__global__ void InitializeSchurKernelF(const FSolvePoseMeta* metadata,
                                       const FPoseBlock* poses,
                                       float lambda,
                                       float* schur,
                                       float* rhs,
                                       size_t dimension,
                                       size_t pose_count) {
  const size_t pose_index = blockIdx.x;
  const uint32_t element = threadIdx.x;
  if (pose_index >= pose_count) return;
  const FSolvePoseMeta meta = metadata[pose_index];
  const FPoseBlock pose = poses[pose_index];
  if (element < meta.dimension)
    rhs[meta.offset + element] = -pose.gradient[element];
  if (element < meta.dimension * meta.dimension) {
    const uint32_t row = element / meta.dimension;
    const uint32_t col = element % meta.dimension;
    float value = pose.hessian[element];
    if (row == col) value += lambda * pose.damping[row];
    schur[(meta.offset + row) * dimension + meta.offset + col] = value;
  }
}

__global__ void SchurSegmentPartialKernelF(
    const FSchurSegment* segments,
    const FPairMeta* pairs,
    const FPairContribution* contributions,
    const FSolvePoseMeta* poses,
    const FEdgeBlock* edges,
    const float* transformed,
    float* partials,
    size_t segment_count) {
  const size_t segment_index = blockIdx.x;
  const uint32_t element = threadIdx.x;
  if (segment_index >= segment_count || element >= kSchurPartialScalars)
    return;
  const FSchurSegment segment = segments[segment_index];
  const FPairMeta pair = pairs[segment.pair_index];
  const FSolvePoseMeta lhs_pose = poses[pair.lhs_pose];
  const FSolvePoseMeta rhs_pose = poses[pair.rhs_pose];
  const uint32_t row = element / rhs_pose.dimension;
  const uint32_t col = element % rhs_pose.dimension;
  float sum = 0.0f;
  if (row < lhs_pose.dimension && col < rhs_pose.dimension) {
    for (uint32_t entry = segment.contribution_begin;
         entry < segment.contribution_end; ++entry) {
      const FPairContribution contribution = contributions[entry];
      const float* lhs = transformed + contribution.lhs_edge * 18;
      const FEdgeBlock rhs = edges[contribution.rhs_edge];
      float value = lhs[row * 3] * rhs.value[col * 3] +
                    lhs[row * 3 + 1] * rhs.value[col * 3 + 1] +
                    lhs[row * 3 + 2] * rhs.value[col * 3 + 2];
      if (pair.lhs_pose == pair.rhs_pose) {
        const float transpose =
            lhs[col * 3] * rhs.value[row * 3] +
            lhs[col * 3 + 1] * rhs.value[row * 3 + 1] +
            lhs[col * 3 + 2] * rhs.value[row * 3 + 2];
        value = 0.5f * (value + transpose);
      }
      sum += value;
    }
  }
  partials[segment_index * kSchurPartialScalars + element] = sum;
}

__global__ void SchurSegmentMergeKernelF(
    const FPairMeta* pairs,
    const FRange* ranges,
    const FSolvePoseMeta* poses,
    const float* partials,
    float* schur,
    size_t dimension,
    size_t pair_count) {
  const size_t pair_index = blockIdx.x;
  const uint32_t element = threadIdx.x;
  if (pair_index >= pair_count || element >= kSchurPartialScalars) return;
  const FPairMeta pair = pairs[pair_index];
  const FSolvePoseMeta lhs_pose = poses[pair.lhs_pose];
  const FSolvePoseMeta rhs_pose = poses[pair.rhs_pose];
  if (element >= lhs_pose.dimension * rhs_pose.dimension) return;
  float sum = 0.0f;
  const FRange range = ranges[pair_index];
  for (uint32_t segment = range.begin; segment < range.end; ++segment)
    sum += partials[segment * kSchurPartialScalars + element];
  const uint32_t row = element / rhs_pose.dimension;
  const uint32_t col = element % rhs_pose.dimension;
  schur[(lhs_pose.offset + row) * dimension + rhs_pose.offset + col] -= sum;
  if (pair.lhs_pose != pair.rhs_pose) {
    schur[(rhs_pose.offset + col) * dimension + lhs_pose.offset + row] -= sum;
  }
}

__global__ void SchurRhsKernelF(const FSolvePoseMeta* poses,
                                const uint32_t* pose_edges,
                                const FEdgeBlock* edges,
                                const float* inverse_gradient,
                                float* rhs,
                                size_t pose_count) {
  const size_t pose_index = blockIdx.x;
  const uint32_t row = threadIdx.x;
  if (pose_index >= pose_count || row >= poses[pose_index].dimension) return;
  const FSolvePoseMeta pose = poses[pose_index];
  float value = 0.0f;
  for (uint32_t entry = pose.edge_begin; entry < pose.edge_end; ++entry) {
    const FEdgeBlock edge = edges[pose_edges[entry]];
    const float* point = inverse_gradient + edge.point_index * 3;
    value += edge.value[row * 3] * point[0] +
             edge.value[row * 3 + 1] * point[1] +
             edge.value[row * 3 + 2] * point[2];
  }
  rhs[pose.offset + row] += value;
}

__global__ void BackSubstitutionKernelF(
    const FSolvePointMeta* points,
    const uint32_t* point_edges,
    const FSolvePoseMeta* poses,
    const FEdgeBlock* edges,
    const FPointBlock* point_blocks,
    const float* point_inverse,
    const float* camera_delta,
    float* point_delta,
    size_t point_count) {
  const size_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= point_count) return;
  float right[3] = {point_blocks[point_index].gradient[0],
                    point_blocks[point_index].gradient[1],
                    point_blocks[point_index].gradient[2]};
  const FSolvePointMeta point = points[point_index];
  for (uint32_t entry = point.edge_begin; entry < point.edge_end; ++entry) {
    const FEdgeBlock edge = edges[point_edges[entry]];
    const FSolvePoseMeta pose = poses[edge.pose_index];
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t camera = 0; camera < pose.dimension; ++camera) {
        right[row] += edge.value[camera * 3 + row] *
                      camera_delta[pose.offset + camera];
      }
    }
  }
  const float* inverse = point_inverse + point_index * 9;
  for (uint32_t row = 0; row < 3; ++row) {
    point_delta[point_index * 3 + row] =
        -(inverse[row * 3] * right[0] +
          inverse[row * 3 + 1] * right[1] +
          inverse[row * 3 + 2] * right[2]);
  }
}

__device__ inline void QuaternionPlusD(const double* q,
                                       const double* delta,
                                       double* result) {
  const double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                           delta[2] * delta[2]);
  if (norm == 0.0) {
    for (uint32_t i = 0; i < 4; ++i) result[i] = q[i];
    return;
  }
  const double sin_delta_by_delta = sin(norm) / norm;
  const double d[4] = {cos(norm), sin_delta_by_delta * delta[0],
                       sin_delta_by_delta * delta[1],
                       sin_delta_by_delta * delta[2]};
  result[0] = d[0] * q[0] - d[1] * q[1] - d[2] * q[2] - d[3] * q[3];
  result[1] = d[0] * q[1] + d[1] * q[0] + d[2] * q[3] - d[3] * q[2];
  result[2] = d[0] * q[2] - d[1] * q[3] + d[2] * q[0] + d[3] * q[1];
  result[3] = d[0] * q[3] + d[1] * q[2] - d[2] * q[1] + d[3] * q[0];
}

__global__ void UpdateImageStateKernelF(
    const FImageState* current,
    const FImageUpdateMeta* updates,
    const FPoseMeta* poses,
    const FSolvePoseMeta* solve_poses,
    const float* camera_delta,
    FImageState* trial,
    uint32_t* status,
    size_t count) {
  const size_t image_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (image_index >= count) return;
  trial[image_index] = current[image_index];
  const uint32_t pose_index = updates[image_index].variable_pose_index;
  if (pose_index == kInvalidTarget) return;
  const FPoseMeta pose = poses[pose_index];
  const FSolvePoseMeta solve = solve_poses[pose_index];
  const double rotation[3] = {
      static_cast<double>(camera_delta[solve.offset]),
      static_cast<double>(camera_delta[solve.offset + 1]),
      static_cast<double>(camera_delta[solve.offset + 2])};
  QuaternionPlusD(current[image_index].quaternion, rotation,
                  trial[image_index].quaternion);
  for (uint32_t row = 3; row < pose.dimension; ++row) {
    const int32_t component = pose.free_translation_indices[row - 3];
    if (component < 0 || component >= 3) {
      atomicOr(status, 1u);
      return;
    }
    trial[image_index].translation[component] +=
        static_cast<double>(camera_delta[solve.offset + row]);
  }
  for (double value : trial[image_index].quaternion)
    if (!isfinite(value)) atomicOr(status, 2u);
  for (double value : trial[image_index].translation)
    if (!isfinite(value)) atomicOr(status, 2u);
}

__global__ void UpdatePointStateKernelF(
    const FPointState* current,
    const FPointUpdateMeta* updates,
    const float* point_delta,
    FPointState* trial,
    uint32_t* status,
    size_t count) {
  const size_t point_entity = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_entity >= count) return;
  trial[point_entity] = current[point_entity];
  const uint32_t point_index = updates[point_entity].variable_point_index;
  if (point_index == kInvalidTarget) return;
  for (uint32_t i = 0; i < 3; ++i) {
    trial[point_entity].xyz[i] +=
        static_cast<double>(point_delta[point_index * 3 + i]);
    if (!isfinite(trial[point_entity].xyz[i])) atomicOr(status, 2u);
  }
}

__global__ void QuantizeStateKernelF(FImageState* images,
                                     size_t image_count,
                                     FPointState* points,
                                     size_t point_count,
                                     FCameraState* cameras,
                                     size_t camera_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < image_count) {
    for (uint32_t i = 0; i < 4; ++i)
      images[index].quaternion[i] =
          static_cast<double>(static_cast<float>(images[index].quaternion[i]));
    for (uint32_t i = 0; i < 3; ++i)
      images[index].translation[i] =
          static_cast<double>(static_cast<float>(images[index].translation[i]));
  }
  if (index < point_count) {
    for (uint32_t i = 0; i < 3; ++i)
      points[index].xyz[i] =
          static_cast<double>(static_cast<float>(points[index].xyz[i]));
  }
  if (index < camera_count) {
    for (uint32_t i = 0; i < 8; ++i)
      cameras[index].params[i] =
          static_cast<double>(static_cast<float>(cameras[index].params[i]));
  }
}

__device__ inline double VisualCostD(const FVisualCostStatic& fixed,
                                     const FImageState& image,
                                     const FPointState& point,
                                     const FCameraState& camera,
                                     const uint8_t loss_mode,
                                     const double loss_scale) {
  const double w = image.quaternion[0];
  const double x = image.quaternion[1];
  const double y = image.quaternion[2];
  const double z = image.quaternion[3];
  const double px = point.xyz[0];
  const double py = point.xyz[1];
  const double pz = point.xyz[2];
  const double t2 = w * x;
  const double t3 = w * y;
  const double t4 = w * z;
  const double t5 = -x * x;
  const double t6 = x * y;
  const double t7 = x * z;
  const double t8 = -y * y;
  const double t9 = y * z;
  const double t1 = -z * z;
  const double X =
      2.0 * ((t8 + t1) * px + (t6 - t4) * py + (t3 + t7) * pz) +
      px + image.translation[0];
  const double Y =
      2.0 * ((t4 + t6) * px + (t5 + t1) * py + (t9 - t2) * pz) +
      py + image.translation[1];
  const double Z =
      2.0 * ((t7 - t3) * px + (t2 + t9) * py + (t5 + t8) * pz) +
      pz + image.translation[2];
  if (!isfinite(Z) || fabs(Z) <= DBL_MIN) return nan("");
  const double u = X / Z;
  const double v = Y / Z;
  const double r2 = u * u + v * v;
  const double radial = camera.params[4] * r2 +
                        camera.params[5] * r2 * r2;
  const double du = u * radial + 2.0 * camera.params[6] * u * v +
                    camera.params[7] * (r2 + 2.0 * u * u);
  const double dv = v * radial + 2.0 * camera.params[7] * u * v +
                    camera.params[6] * (r2 + 2.0 * v * v);
  const double residual0 = camera.params[0] * (u + du) + camera.params[2] -
                           fixed.observation[0];
  const double residual1 = camera.params[1] * (v + dv) + camera.params[3] -
                           fixed.observation[1];
  return LossCostD(residual0 * residual0 + residual1 * residual1,
                   loss_mode, loss_scale);
}

__device__ inline double LidarCostD(const FLidarCostStatic& fixed,
                                    const FPointState& point,
                                    const uint8_t loss_mode,
                                    const double loss_scale) {
  const double d = fixed.plane[0] * point.xyz[0] +
                   fixed.plane[1] * point.xyz[1] +
                   fixed.plane[2] * point.xyz[2] + fixed.plane[3];
  double residual = 0.0;
  if (fixed.mode == static_cast<uint8_t>(LidarResidualMode::kSigned)) {
    residual = fixed.weight * d;
  } else {
    residual = fixed.weight * sqrt(d * d);
  }
  return LossCostD(residual * residual, loss_mode, loss_scale);
}

__global__ void CostValuesKernelD(const FCostEntry* order,
                                  size_t count,
                                  const FVisualCostStatic* visual,
                                  const FLidarCostStatic* lidar,
                                  const FImageState* images,
                                  const FPointState* points,
                                  const FCameraState* cameras,
                                  uint8_t loss_mode,
                                  double loss_scale,
                                  double* values) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const FCostEntry entry = order[index];
  if (entry.residual_kind == 0) {
    const FVisualCostStatic fixed = visual[entry.output_index];
    values[index] = VisualCostD(
        fixed, images[fixed.image_entity], points[fixed.point_entity],
        cameras[fixed.camera_entity], loss_mode, loss_scale);
  } else {
    const FLidarCostStatic fixed = lidar[entry.output_index];
    values[index] = LidarCostD(fixed, points[fixed.point_entity], loss_mode,
                               loss_scale);
  }
}

constexpr uint32_t kCostReductionThreads = 256;

__global__ void CostReductionKernelD(const double* input,
                                     size_t count,
                                     double* output) {
  __shared__ double values[kCostReductionThreads];
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  values[threadIdx.x] = index < count ? input[index] : 0.0;
  __syncthreads();
  for (uint32_t stride = kCostReductionThreads / 2; stride > 0;
       stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = values[0];
}

__global__ void StateNormKernelD(const FImageState* current_images,
                                 const FImageState* trial_images,
                                 const FPoseMeta* poses,
                                 size_t pose_count,
                                 const FPointState* current_points,
                                 const FPointState* trial_points,
                                 const FPointMeta* points,
                                 size_t point_count,
                                 double* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double state_squared = 0.0;
  double difference_squared = 0.0;
  for (size_t i = 0; i < pose_count; ++i) {
    const FImageState current = current_images[poses[i].image_entity];
    const FImageState trial = trial_images[poses[i].image_entity];
    for (uint32_t j = 0; j < 4; ++j) {
      state_squared += current.quaternion[j] * current.quaternion[j];
      const double difference = current.quaternion[j] - trial.quaternion[j];
      difference_squared += difference * difference;
    }
    for (uint32_t j = 0; j < 3; ++j) {
      state_squared += current.translation[j] * current.translation[j];
      const double difference = current.translation[j] - trial.translation[j];
      difference_squared += difference * difference;
    }
  }
  for (size_t i = 0; i < point_count; ++i) {
    const FPointState current = current_points[points[i].point_entity];
    const FPointState trial = trial_points[points[i].point_entity];
    for (uint32_t j = 0; j < 3; ++j) {
      state_squared += current.xyz[j] * current.xyz[j];
      const double difference = current.xyz[j] - trial.xyz[j];
      difference_squared += difference * difference;
    }
  }
  output[0] = sqrt(state_squared);
  output[1] = sqrt(difference_squared);
}

__global__ void StepDiagnosticsKernelF(
    const FSolvePoseMeta* solve_poses,
    const uint32_t* pose_edges,
    const FPoseBlock* poses,
    size_t pose_count,
    const FSolvePointMeta* solve_points,
    const uint32_t* point_edges,
    const FPointBlock* points,
    size_t point_count,
    const FEdgeBlock* edges,
    const float* camera_delta,
    const float* point_delta,
    float lambda,
    FStepSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float damping_quadratic = 0.0f;
  float gradient_dot_delta = 0.0f;
  float gradient_norm = 0.0f;
  float delta_norm = 0.0f;
  float residual_norm = 0.0f;
  float hessian_norm = 0.0f;
  for (size_t pose_index = 0; pose_index < pose_count; ++pose_index) {
    const FSolvePoseMeta meta = solve_poses[pose_index];
    const FPoseBlock pose = poses[pose_index];
    for (uint32_t row = 0; row < pose.dimension; ++row) {
      const float delta = camera_delta[meta.offset + row];
      float residual = pose.gradient[row];
      float row_sum = 0.0f;
      gradient_norm = fmaxf(gradient_norm, fabsf(pose.gradient[row]));
      delta_norm = fmaxf(delta_norm, fabsf(delta));
      damping_quadratic += pose.damping[row] * delta * delta;
      gradient_dot_delta += pose.gradient[row] * delta;
      for (uint32_t col = 0; col < pose.dimension; ++col) {
        float value = pose.hessian[row * pose.dimension + col];
        if (row == col) value += lambda * pose.damping[row];
        residual += value * camera_delta[meta.offset + col];
        row_sum += fabsf(value);
      }
      for (uint32_t edge_entry = solve_poses[pose_index].edge_begin;
           edge_entry < solve_poses[pose_index].edge_end; ++edge_entry) {
        const FEdgeBlock edge = edges[pose_edges[edge_entry]];
        for (uint32_t col = 0; col < 3; ++col) {
          const float value = edge.value[row * 3 + col];
          residual += value * point_delta[edge.point_index * 3 + col];
          row_sum += fabsf(value);
        }
      }
      residual_norm = fmaxf(residual_norm, fabsf(residual));
      hessian_norm = fmaxf(hessian_norm, row_sum);
    }
  }
  for (size_t point_index = 0; point_index < point_count; ++point_index) {
    const FPointBlock point = points[point_index];
    for (uint32_t row = 0; row < 3; ++row) {
      const float delta = point_delta[point_index * 3 + row];
      float residual = point.gradient[row];
      float row_sum = 0.0f;
      gradient_norm = fmaxf(gradient_norm, fabsf(point.gradient[row]));
      delta_norm = fmaxf(delta_norm, fabsf(delta));
      damping_quadratic += point.damping[row] * delta * delta;
      gradient_dot_delta += point.gradient[row] * delta;
      for (uint32_t col = 0; col < 3; ++col) {
        float value = point.hessian[row * 3 + col];
        if (row == col) value += lambda * point.damping[row];
        residual += value * point_delta[point_index * 3 + col];
        row_sum += fabsf(value);
      }
      for (uint32_t edge_entry = solve_points[point_index].edge_begin;
           edge_entry < solve_points[point_index].edge_end; ++edge_entry) {
        const FEdgeBlock edge = edges[point_edges[edge_entry]];
        const FSolvePoseMeta pose = solve_poses[edge.pose_index];
        for (uint32_t camera = 0; camera < pose.dimension; ++camera) {
          const float value = edge.value[camera * 3 + row];
          residual += value * camera_delta[pose.offset + camera];
          row_sum += fabsf(value);
        }
      }
      residual_norm = fmaxf(residual_norm, fabsf(residual));
      hessian_norm = fmaxf(hessian_norm, row_sum);
    }
  }
  const float denominator = hessian_norm * delta_norm + gradient_norm;
  FStepSummary value;
  value.predicted_reduction =
      0.5f * (lambda * damping_quadratic - gradient_dot_delta);
  value.backward_error = denominator == 0.0f
      ? 0.0f : residual_norm / denominator;
  value.step_norm = delta_norm;
  value.finite = FiniteF(value.predicted_reduction) &&
                         FiniteF(value.backward_error) &&
                         FiniteF(value.step_norm)
                     ? 1u
                     : 0u;
  *summary = value;
}

struct FFactorStatusSummary {
  uint64_t failure_count;
  int64_t first_failure;
};

__global__ void FactorStatusSummaryKernelF(const uint8_t* status,
                                           size_t count,
                                           FFactorStatusSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  FFactorStatusSummary value{0, -1};
  for (size_t i = 0; i < count; ++i) {
    if (status[i] != 0) {
      if (value.first_failure < 0)
        value.first_failure = static_cast<int64_t>(i);
      ++value.failure_count;
    }
  }
  *summary = value;
}

struct Fp32Capture {
  std::vector<FVisualOutput> visual;
  std::vector<FLidarOutput> lidar;
  std::vector<FPoseBlock> poses;
  std::vector<FPointBlock> points;
  std::vector<FEdgeBlock> edges;
  std::vector<float> pose_scaling;
  std::vector<float> point_scaling;
  std::vector<float> point_inverse;
  std::vector<float> inverse_gradient;
  std::vector<float> transformed_edges;
  std::vector<float> schur;
  std::vector<float> rhs;
  std::vector<float> factor;
  std::vector<float> camera_delta;
  std::vector<float> point_delta;
  FGradientSummary gradient{};
  FStepSummary step{};
  double trial_cost = 0.0;
  Snapshot trial_state;
  int solver_info = 0;
};

class Fp32ExperimentContext {
 public:
  Fp32ExperimentContext() = default;
  Fp32ExperimentContext(const Fp32ExperimentContext&) = delete;
  Fp32ExperimentContext& operator=(const Fp32ExperimentContext&) = delete;
  ~Fp32ExperimentContext() { Close(); }

  bool Initialize(const Snapshot& snapshot,
                  const CudaFullLmOptions& options,
                  const CudaArithmeticPrecision precision,
                  CudaFullLmResult* result,
                  std::string* error) {
    snapshot_ = &snapshot;
    options_ = &options;
    runtime_ = &result->runtime.persistent_device;
    precision_ = precision;
    if (options.layer_c.layer_b.hessian_assembly_backend !=
            CudaHessianAssemblyBackend::kObservationSegmented ||
        options.hot_kernel_mode != CudaHotKernelMode::kTransformed ||
        options.layer_c.schur_contribution_backend !=
            CudaSchurContributionBackend::kSegmentedTransformed) {
      *error = "FP32 experiment requires observation_segmented, transformed, "
               "and segmented selectors";
      return false;
    }
    if (!BuildFp32Topology(snapshot, &topology_, error)) return false;
    if (topology_.pose_dimension == 0 ||
        topology_.pose_dimension > static_cast<uint32_t>(
            std::numeric_limits<int>::max())) {
      *error = "FP32 experiment requires a non-empty bounded pose system";
      return false;
    }
    loss_mode_ = options.layer_c.layer_b.loss_mode;
    if (loss_mode_ == CudaLossMode::kFromSnapshot) {
      std::string loss = snapshot.metadata.loss_function;
      std::transform(loss.begin(), loss.end(), loss.begin(), ::toupper);
      loss_mode_ = loss == "SOFT_L1" ? CudaLossMode::kSoftL1
                                      : CudaLossMode::kTrivial;
    }
    loss_scale_ = loss_mode_ == CudaLossMode::kTrivial
        ? 1.0 : options.layer_c.layer_b.loss_scale;
    if ((loss_mode_ != CudaLossMode::kTrivial &&
         loss_mode_ != CudaLossMode::kSoftL1) ||
        !std::isfinite(loss_scale_) || loss_scale_ <= 0.0) {
      *error = "FP32 experiment loss configuration is invalid";
      return false;
    }
    cudaError_t cuda_status = cudaSetDevice(
        options.layer_c.layer_b.layer_a.device);
    if (cuda_status == cudaSuccess)
      cuda_status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      *error = std::string("FP32 stream initialization failed: ") +
               cudaGetErrorString(cuda_status);
      return false;
    }
    cusolverStatus_t solver_status = cusolverDnCreate(&solver_);
    if (solver_status == CUSOLVER_STATUS_SUCCESS)
      solver_status = cusolverDnSetStream(solver_, stream_);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      *error = "FP32 cuSOLVER initialization failed";
      return false;
    }

    std::vector<FImageState> images(snapshot.images.size());
    std::vector<FPointState> points(snapshot.points.size());
    std::vector<FCameraState> cameras(snapshot.cameras.size());
    for (size_t i = 0; i < snapshot.images.size(); ++i) {
      std::copy(snapshot.images[i].qvec.begin(), snapshot.images[i].qvec.end(),
                images[i].quaternion);
      std::copy(snapshot.images[i].tvec.begin(), snapshot.images[i].tvec.end(),
                images[i].translation);
    }
    for (size_t i = 0; i < snapshot.points.size(); ++i)
      std::copy(snapshot.points[i].xyz.begin(), snapshot.points[i].xyz.end(),
                points[i].xyz);
    for (size_t i = 0; i < snapshot.cameras.size(); ++i)
      std::copy(snapshot.cameras[i].params.begin(),
                snapshot.cameras[i].params.end(), cameras[i].params);

    if (!AllocateAndUpload(&d_images_[0], images, false, error) ||
        !d_images_[1].Allocate(images.size(), nullptr, error) ||
        !AllocateAndUpload(&d_points_[0], points, false, error) ||
        !d_points_[1].Allocate(points.size(), nullptr, error) ||
        !AllocateAndUpload(&d_cameras_, cameras, false, error) ||
        !AllocateAndUpload(&d_visual_static_, topology_.visual, true, error) ||
        !AllocateAndUpload(&d_visual_cost_static_, topology_.visual_cost,
                           false, error) ||
        !AllocateAndUpload(&d_lidar_static_, topology_.lidar, true, error) ||
        !AllocateAndUpload(&d_lidar_cost_static_, topology_.lidar_cost,
                           false, error) ||
        !AllocateAndUpload(&d_cost_order_, topology_.cost_order, false, error) ||
        !AllocateTopology(error) || !AllocateWorkspaces(error)) {
      return false;
    }
    if (precision == CudaArithmeticPrecision::kFp32StateQuantizedMixed) {
      const size_t maximum = std::max(
          images.size(), std::max(points.size(), cameras.size()));
      QuantizeStateKernelF<<<Blocks(maximum), kBlockSize, 0, stream_>>>(
          d_images_[0].get(), images.size(), d_points_[0].get(), points.size(),
          d_cameras_.get(), cameras.size());
      ++runtime_->state_quantization_calls;
    }
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
      *error = std::string("FP32 initialization synchronization failed: ") +
               cudaGetErrorString(cuda_status);
      return false;
    }
    runtime_->requested = true;
    runtime_->effective = true;
    runtime_->arithmetic_precision_requested =
        precision == CudaArithmeticPrecision::kFp32Core
            ? "fp32_core" : "fp32_state_quantized";
    runtime_->arithmetic_precision_effective =
        runtime_->arithmetic_precision_requested;
    runtime_->state_storage_precision = "fp64_device_state";
    runtime_->residual_jacobian_precision = "fp32";
    runtime_->hessian_schur_precision = "fp32";
    runtime_->factorization_routine = "cusolverDnSpotrf/Spotrs";
    runtime_->delta_precision = "fp32";
    runtime_->quaternion_plus_precision = "fp64";
    runtime_->cost_precision = "fp64";
    runtime_->controller_precision = "fp64";
    runtime_->hessian_assembly_backend_requested = "observation_segmented";
    runtime_->hessian_assembly_backend_effective = "observation_segmented";
    runtime_->schur_contribution_backend_requested = "segmented";
    runtime_->schur_contribution_backend_effective = "segmented";
    runtime_->hot_kernel_transformed = true;
    runtime_->hessian_segment_size = kSegmentSize;
    runtime_->schur_segment_size = kSegmentSize;
    runtime_->hessian_segment_count_pose = topology_.pose_segments.size();
    runtime_->hessian_segment_count_point = topology_.point_segments.size();
    runtime_->hessian_segment_count_edge = topology_.edge_segments.size();
    runtime_->segment_count = topology_.schur_segments.size();
    runtime_->segmented_pair_count = topology_.pairs.size();
    runtime_->hessian_partial_workspace_bytes =
        (d_pose_partials_.size() + d_point_partials_.size() +
         d_edge_partials_.size()) * sizeof(float);
    runtime_->schur_partial_workspace_bytes =
        d_schur_partials_.size() * sizeof(float);
    runtime_->float_workspace_bytes =
        runtime_->hessian_partial_workspace_bytes +
        runtime_->schur_partial_workspace_bytes +
        d_solver_workspace_.size() * sizeof(float);
    runtime_->peak_resident_bytes = runtime_->float_arena_reserved_bytes;
    return true;
  }

  bool Linearize(const bool initial, std::string* error) {
    const FImageState* images = d_images_[current_slot_].get();
    const FPointState* points = d_points_[current_slot_].get();
    cudaError_t status = cudaMemsetAsync(d_pose_blocks_.get(), 0,
        d_pose_blocks_.size() * sizeof(FPoseBlock), stream_);
    if (status == cudaSuccess)
      status = cudaMemsetAsync(d_point_blocks_.get(), 0,
          d_point_blocks_.size() * sizeof(FPointBlock), stream_);
    if (status == cudaSuccess)
      status = cudaMemsetAsync(d_edge_blocks_.get(), 0,
          d_edge_blocks_.size() * sizeof(FEdgeBlock), stream_);
    if (status != cudaSuccess) {
      *error = "FP32 assembly output initialization failed";
      return false;
    }
    if (!topology_.visual.empty()) {
      EvaluateVisualKernelF<<<Blocks(topology_.visual.size()), kBlockSize, 0,
                              stream_>>>(
          d_visual_static_.get(), images, points, d_cameras_.get(),
          d_visual_output_.get(), topology_.visual.size());
    }
    if (!topology_.lidar.empty()) {
      EvaluateLidarKernelF<<<Blocks(topology_.lidar.size()), kBlockSize, 0,
                             stream_>>>(
          d_lidar_static_.get(), points, d_lidar_output_.get(),
          topology_.lidar.size());
    }
    RecordStatusSummaryKernelF<<<1, 1, 0, stream_>>>(
        d_visual_output_.get(), topology_.visual.size(),
        d_lidar_output_.get(), topology_.lidar.size(),
        d_record_status_summary_.get());
    FRecordStatusSummary record_status{};
    status = cudaMemcpyAsync(&record_status, d_record_status_summary_.get(),
                             sizeof(record_status), cudaMemcpyDeviceToHost,
                             stream_);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
      *error = "FP32 residual/Jacobian finite status transfer failed";
      return false;
    }
    if (record_status.visual_failure_count != 0 ||
        record_status.lidar_failure_count != 0) {
      std::ostringstream stream;
      stream << "FP32 non-finite residual/Jacobian: visual_failures="
             << record_status.visual_failure_count
             << " first_visual=" << record_status.first_visual_failure
             << " lidar_failures=" << record_status.lidar_failure_count
             << " first_lidar=" << record_status.first_lidar_failure;
      if (record_status.first_lidar_failure >= 0) {
        const size_t index =
            static_cast<size_t>(record_status.first_lidar_failure);
        const uint32_t entity = topology_.lidar[index].point_entity;
        const auto& point = snapshot_->points[entity].xyz;
        const FLidarCostStatic& exact = topology_.lidar_cost[index];
        const FLidarStatic& rounded = topology_.lidar[index];
        const double distance_double = exact.plane[0] * point[0] +
            exact.plane[1] * point[1] + exact.plane[2] * point[2] +
            exact.plane[3];
        const float x = static_cast<float>(point[0]);
        const float y = static_cast<float>(point[1]);
        const float z = static_cast<float>(point[2]);
        const float distance_float = rounded.plane[0] * x +
            rounded.plane[1] * y + rounded.plane[2] * z +
            rounded.plane[3];
        const float distance_float_fma = std::fma(
            rounded.plane[0], x,
            std::fma(rounded.plane[1], y,
                     std::fma(rounded.plane[2], z, rounded.plane[3])));
        const float distance_float_pairwise =
            (rounded.plane[0] * x + rounded.plane[1] * y) +
            (rounded.plane[2] * z + rounded.plane[3]);
        stream << " first_lidar_point3D_id="
               << snapshot_->points[entity].point3D_id
               << std::setprecision(17)
               << " signed_distance_fp64=" << distance_double
               << " signed_distance_fp32=" << distance_float
               << " signed_distance_fp32_fma=" << distance_float_fma
               << " signed_distance_fp32_pairwise="
               << distance_float_pairwise;
      }
      *error = stream.str();
      return false;
    }
    if (!topology_.pose_segments.empty()) {
      PoseSegmentPartialKernelF<<<topology_.pose_segments.size(), 64, 0,
                                  stream_>>>(
          d_pose_segments_.get(), d_pose_adjacency_.get(), d_pose_meta_.get(),
          d_visual_output_.get(), static_cast<uint8_t>(loss_mode_),
          static_cast<float>(loss_scale_), d_pose_partials_.get(),
          topology_.pose_segments.size());
    }
    if (!topology_.point_segments.empty()) {
      PointSegmentPartialKernelF<<<topology_.point_segments.size(), 64, 0,
                                   stream_>>>(
          d_point_segments_.get(), d_point_adjacency_.get(),
          d_visual_output_.get(), d_lidar_output_.get(),
          static_cast<uint8_t>(loss_mode_), static_cast<float>(loss_scale_),
          d_point_partials_.get(), topology_.point_segments.size());
    }
    if (!topology_.edge_segments.empty()) {
      EdgeSegmentPartialKernelF<<<topology_.edge_segments.size(), 64, 0,
                                  stream_>>>(
          d_edge_segments_.get(), d_edge_adjacency_.get(), d_edge_meta_.get(),
          d_pose_meta_.get(), d_visual_output_.get(),
          static_cast<uint8_t>(loss_mode_), static_cast<float>(loss_scale_),
          d_edge_partials_.get(), topology_.edge_segments.size());
    }
    if (!topology_.poses.empty()) {
      PoseSegmentMergeKernelF<<<topology_.poses.size(), 64, 0, stream_>>>(
          d_pose_meta_.get(), d_pose_ranges_.get(), d_pose_partials_.get(),
          initial ? nullptr : d_frozen_pose_scaling_.get(),
          static_cast<float>(options_->layer_c.layer_b.min_lm_diagonal),
          static_cast<float>(options_->layer_c.layer_b.max_lm_diagonal),
          d_pose_blocks_.get(), topology_.poses.size());
    }
    if (!topology_.points.empty()) {
      PointSegmentMergeKernelF<<<topology_.points.size(), 32, 0, stream_>>>(
          d_point_meta_.get(), d_point_ranges_.get(), d_point_partials_.get(),
          initial ? nullptr : d_frozen_point_scaling_.get(),
          static_cast<float>(options_->layer_c.layer_b.min_lm_diagonal),
          static_cast<float>(options_->layer_c.layer_b.max_lm_diagonal),
          d_point_blocks_.get(), topology_.points.size());
    }
    if (!topology_.edges.empty()) {
      EdgeSegmentMergeKernelF<<<topology_.edges.size(), 32, 0, stream_>>>(
          d_edge_meta_.get(), d_edge_ranges_.get(), d_edge_partials_.get(),
          d_edge_blocks_.get(), topology_.edges.size());
    }
    if (initial) {
      const size_t scaling_count = std::max(topology_.poses.size() * 6,
                                            topology_.points.size() * 3);
      CaptureScalingKernelF<<<Blocks(scaling_count), kBlockSize, 0, stream_>>>(
          d_pose_blocks_.get(), topology_.poses.size(), d_point_blocks_.get(),
          topology_.points.size(), d_frozen_pose_scaling_.get(),
          d_frozen_point_scaling_.get());
      ++runtime_->scaling_slot_publishes;
    }
    GradientSummaryKernelF<<<1, 1, 0, stream_>>>(
        d_pose_meta_.get(), d_pose_blocks_.get(), topology_.poses.size(),
        d_point_blocks_.get(), topology_.points.size(), images,
        d_gradient_summary_.get());
    status = cudaGetLastError();
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(&gradient_summary_, d_gradient_summary_.get(),
                               sizeof(gradient_summary_),
                               cudaMemcpyDeviceToHost, stream_);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess || gradient_summary_.finite == 0) {
      *error = "FP32 residual/Jacobian or Hessian assembly is non-finite";
      return false;
    }
    ++runtime_->hessian_gradient_assembly_calls;
    ++runtime_->pose_block_assembly_calls;
    ++runtime_->point_block_assembly_calls;
    ++runtime_->edge_block_assembly_calls;
    ++runtime_->jacobi_damping_finalize_calls;
    ++runtime_->gradient_summary_calls;
    runtime_->observation_segment_partial_launches +=
        (!topology_.pose_segments.empty()) +
        (!topology_.point_segments.empty()) +
        (!topology_.edge_segments.empty());
    runtime_->observation_segment_merge_launches +=
        (!topology_.poses.empty()) + (!topology_.points.empty()) +
        (!topology_.edges.empty());
    return true;
  }

  bool ComputeCurrentCost(double* cost, std::string* error) {
    return ComputeCost(current_slot_, cost, error);
  }

  bool SolveStep(const float lambda,
                 double* trial_cost,
                 double* state_norm,
                 double* difference_norm,
                 bool* factorization_success,
                 std::string* error) {
    *factorization_success = false;
    if (!topology_.points.empty()) {
      PointFactorKernelF<<<Blocks(topology_.points.size()), kBlockSize, 0,
                           stream_>>>(
          d_point_blocks_.get(), lambda, d_point_inverse_.get(),
          d_inverse_gradient_.get(), d_point_factor_status_.get(),
          topology_.points.size());
      FactorStatusSummaryKernelF<<<1, 1, 0, stream_>>>(
          d_point_factor_status_.get(), topology_.points.size(),
          d_factor_status_summary_.get());
    } else {
      const FFactorStatusSummary empty{0, -1};
      cudaMemcpyAsync(d_factor_status_summary_.get(), &empty, sizeof(empty),
                      cudaMemcpyHostToDevice, stream_);
    }
    FFactorStatusSummary factor_status{};
    cudaError_t cuda_status = cudaMemcpyAsync(
        &factor_status, d_factor_status_summary_.get(), sizeof(factor_status),
        cudaMemcpyDeviceToHost, stream_);
    if (cuda_status == cudaSuccess)
      cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
      *error = "FP32 point factor status transfer failed";
      return false;
    }
    if (factor_status.failure_count != 0) {
      FPointBlock failed{};
      cuda_status = cudaMemcpyAsync(
          &failed, d_point_blocks_.get() + factor_status.first_failure,
          sizeof(failed), cudaMemcpyDeviceToHost, stream_);
      if (cuda_status == cudaSuccess)
        cuda_status = cudaStreamSynchronize(stream_);
      std::ostringstream stream;
      stream << std::setprecision(9)
             << "FP32 point factorization failed at point "
             << factor_status.first_failure;
      if (cuda_status == cudaSuccess) {
        stream << " point3D_id=" << failed.point_id << " lambda=" << lambda
               << " hessian=[";
        for (uint32_t i = 0; i < 9; ++i)
          stream << (i == 0 ? "" : ",") << failed.hessian[i];
        stream << "] damping=[" << failed.damping[0] << ','
               << failed.damping[1] << ',' << failed.damping[2] << ']';
      }
      *error = stream.str();
      return true;
    }
    if (!topology_.edges.empty()) {
      TransformEdgeKernelF<<<topology_.edges.size(), 18, 0, stream_>>>(
          d_edge_blocks_.get(), d_point_inverse_.get(),
          d_transformed_edges_.get(), topology_.edges.size());
      ++runtime_->transform_kernel_calls;
    }
    cuda_status = cudaMemsetAsync(d_schur_.get(), 0,
        d_schur_.size() * sizeof(float), stream_);
    if (cuda_status == cudaSuccess)
      cuda_status = cudaMemsetAsync(d_rhs_.get(), 0,
          d_rhs_.size() * sizeof(float), stream_);
    if (cuda_status != cudaSuccess) {
      *error = "FP32 Schur initialization failed";
      return false;
    }
    if (!topology_.solve_poses.empty()) {
      InitializeSchurKernelF<<<topology_.solve_poses.size(), 36, 0, stream_>>>(
          d_solve_pose_meta_.get(), d_pose_blocks_.get(), lambda,
          d_schur_.get(), d_rhs_.get(), topology_.pose_dimension,
          topology_.solve_poses.size());
    }
    if (!topology_.schur_segments.empty()) {
      SchurSegmentPartialKernelF<<<topology_.schur_segments.size(), 36, 0,
                                   stream_>>>(
          d_schur_segments_.get(), d_pair_meta_.get(),
          d_pair_contributions_.get(), d_solve_pose_meta_.get(),
          d_edge_blocks_.get(), d_transformed_edges_.get(),
          d_schur_partials_.get(), topology_.schur_segments.size());
      SchurSegmentMergeKernelF<<<topology_.pairs.size(), 36, 0, stream_>>>(
          d_pair_meta_.get(), d_pair_ranges_.get(), d_solve_pose_meta_.get(),
          d_schur_partials_.get(), d_schur_.get(), topology_.pose_dimension,
          topology_.pairs.size());
      ++runtime_->schur_partial_kernel_launches;
      ++runtime_->schur_merge_kernel_launches;
    }
    if (!topology_.solve_poses.empty()) {
      SchurRhsKernelF<<<topology_.solve_poses.size(), 6, 0, stream_>>>(
          d_solve_pose_meta_.get(), d_pose_edges_.get(), d_edge_blocks_.get(),
          d_inverse_gradient_.get(), d_rhs_.get(),
          topology_.solve_poses.size());
      ++runtime_->schur_rhs_calls;
    }
    ++runtime_->schur_contribution_calls;
    cuda_status = cudaMemcpyAsync(
        d_factor_.get(), d_schur_.get(), d_schur_.size() * sizeof(float),
        cudaMemcpyDeviceToDevice, stream_);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpyAsync(
          d_camera_delta_.get(), d_rhs_.get(), d_rhs_.size() * sizeof(float),
          cudaMemcpyDeviceToDevice, stream_);
    }
    if (cuda_status == cudaSuccess)
      cuda_status = cudaMemsetAsync(d_solver_info_.get(), 0, sizeof(int),
                                    stream_);
    if (cuda_status != cudaSuccess) {
      *error = "FP32 Schur factor preparation failed";
      return false;
    }
    cusolverStatus_t solver_status = cusolverDnSpotrf(
        solver_, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(topology_.pose_dimension), d_factor_.get(),
        static_cast<int>(topology_.pose_dimension), d_solver_workspace_.get(),
        solver_workspace_elements_, d_solver_info_.get());
    ++runtime_->spotrf_calls;
    ++runtime_->factorization_calls;
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnSpotrs(
          solver_, CUBLAS_FILL_MODE_LOWER,
          static_cast<int>(topology_.pose_dimension), 1, d_factor_.get(),
          static_cast<int>(topology_.pose_dimension), d_camera_delta_.get(),
          static_cast<int>(topology_.pose_dimension), d_solver_info_.get());
      ++runtime_->spotrs_calls;
    }
    cuda_status = cudaMemcpyAsync(&solver_info_, d_solver_info_.get(),
                                  sizeof(int), cudaMemcpyDeviceToHost, stream_);
    if (cuda_status == cudaSuccess) cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess || solver_status != CUSOLVER_STATUS_SUCCESS) {
      *error = "FP32 Spotrf/Spotrs API failure";
      return false;
    }
    if (solver_info_ != 0) {
      *error = "FP32 Spotrf/Spotrs devInfo=" + std::to_string(solver_info_);
      return true;
    }
    if (!topology_.points.empty()) {
      BackSubstitutionKernelF<<<Blocks(topology_.points.size()), kBlockSize, 0,
                                stream_>>>(
          d_solve_point_meta_.get(), d_point_edges_.get(),
          d_solve_pose_meta_.get(), d_edge_blocks_.get(),
          d_point_blocks_.get(), d_point_inverse_.get(),
          d_camera_delta_.get(), d_point_delta_.get(), topology_.points.size());
    }
    StepDiagnosticsKernelF<<<1, 1, 0, stream_>>>(
        d_solve_pose_meta_.get(), d_pose_edges_.get(), d_pose_blocks_.get(),
        topology_.poses.size(), d_solve_point_meta_.get(), d_point_edges_.get(),
        d_point_blocks_.get(), topology_.points.size(), d_edge_blocks_.get(),
        d_camera_delta_.get(), d_point_delta_.get(), lambda,
        d_step_summary_.get());
    const int trial_slot = 1 - current_slot_;
    cuda_status = cudaMemsetAsync(d_state_status_.get(), 0, sizeof(uint32_t),
                                  stream_);
    if (!snapshot_->images.empty()) {
      UpdateImageStateKernelF<<<Blocks(snapshot_->images.size()), kBlockSize,
                                0, stream_>>>(
          d_images_[current_slot_].get(), d_image_updates_.get(),
          d_pose_meta_.get(), d_solve_pose_meta_.get(), d_camera_delta_.get(),
          d_images_[trial_slot].get(), d_state_status_.get(),
          snapshot_->images.size());
    }
    if (!snapshot_->points.empty()) {
      UpdatePointStateKernelF<<<Blocks(snapshot_->points.size()), kBlockSize,
                                0, stream_>>>(
          d_points_[current_slot_].get(), d_point_updates_.get(),
          d_point_delta_.get(), d_points_[trial_slot].get(),
          d_state_status_.get(), snapshot_->points.size());
    }
    if (precision_ == CudaArithmeticPrecision::kFp32StateQuantizedMixed) {
      const size_t maximum =
          std::max(snapshot_->images.size(), snapshot_->points.size());
      QuantizeStateKernelF<<<Blocks(maximum), kBlockSize, 0, stream_>>>(
          d_images_[trial_slot].get(), snapshot_->images.size(),
          d_points_[trial_slot].get(), snapshot_->points.size(), nullptr, 0);
      ++runtime_->state_quantization_calls;
    }
    uint32_t state_status = 0;
    cuda_status = cudaMemcpyAsync(&step_summary_, d_step_summary_.get(),
                                  sizeof(step_summary_),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status == cudaSuccess)
      cuda_status = cudaMemcpyAsync(&state_status, d_state_status_.get(),
                                    sizeof(state_status),
                                    cudaMemcpyDeviceToHost, stream_);
    if (cuda_status == cudaSuccess) cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess || state_status != 0 ||
        step_summary_.finite == 0) {
      *error = "FP32 delta diagnostics or double-state update failed";
      return false;
    }
    if (!ComputeCost(trial_slot, trial_cost, error)) return false;
    StateNormKernelD<<<1, 1, 0, stream_>>>(
        d_images_[current_slot_].get(), d_images_[trial_slot].get(),
        d_pose_meta_.get(), topology_.poses.size(),
        d_points_[current_slot_].get(), d_points_[trial_slot].get(),
        d_point_meta_.get(), topology_.points.size(), d_state_norm_.get());
    double norms[2] = {};
    cuda_status = cudaMemcpyAsync(norms, d_state_norm_.get(), sizeof(norms),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status == cudaSuccess) cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess || !std::isfinite(norms[0]) ||
        !std::isfinite(norms[1])) {
      *error = "FP32 state norm computation failed";
      return false;
    }
    *state_norm = norms[0];
    *difference_norm = norms[1];
    *factorization_success = true;
    return true;
  }

  void AcceptTrial() { current_slot_ = 1 - current_slot_; }

  const FGradientSummary& gradient_summary() const { return gradient_summary_; }
  const FStepSummary& step_summary() const { return step_summary_; }
  int solver_info() const { return solver_info_; }

  bool MaterializeCurrent(Snapshot* output, std::string* error) {
    std::vector<FImageState> images;
    std::vector<FPointState> points;
    if (!d_images_[current_slot_].Download(&images, stream_, error) ||
        !d_points_[current_slot_].Download(&points, stream_, error)) {
      return false;
    }
    const cudaError_t status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
      *error = "FP32 final state materialization failed";
      return false;
    }
    *output = *snapshot_;
    for (const FPoseMeta& pose : topology_.poses) {
      auto& image = output->images[pose.image_entity];
      for (uint32_t i = 0; i < 4; ++i)
        image.qvec[i] = images[pose.image_entity].quaternion[i];
      for (uint32_t row = 3; row < pose.dimension; ++row) {
        const int32_t component = pose.free_translation_indices[row - 3];
        image.tvec[component] =
            images[pose.image_entity].translation[component];
      }
    }
    for (const FPointMeta& point : topology_.points) {
      for (uint32_t i = 0; i < 3; ++i)
        output->points[point.point_entity].xyz[i] =
            points[point.point_entity].xyz[i];
    }
    ++runtime_->final_state_materialization_operations;
    return true;
  }

  bool Capture(Fp32Capture* capture, std::string* error) {
    if (!d_visual_output_.Download(&capture->visual, stream_, error) ||
        !d_lidar_output_.Download(&capture->lidar, stream_, error) ||
        !d_pose_blocks_.Download(&capture->poses, stream_, error) ||
        !d_point_blocks_.Download(&capture->points, stream_, error) ||
        !d_edge_blocks_.Download(&capture->edges, stream_, error) ||
        !d_frozen_pose_scaling_.Download(&capture->pose_scaling, stream_, error) ||
        !d_frozen_point_scaling_.Download(&capture->point_scaling, stream_, error) ||
        !d_point_inverse_.Download(&capture->point_inverse, stream_, error) ||
        !d_inverse_gradient_.Download(&capture->inverse_gradient, stream_, error) ||
        !d_transformed_edges_.Download(&capture->transformed_edges, stream_, error) ||
        !d_schur_.Download(&capture->schur, stream_, error) ||
        !d_rhs_.Download(&capture->rhs, stream_, error) ||
        !d_factor_.Download(&capture->factor, stream_, error) ||
        !d_camera_delta_.Download(&capture->camera_delta, stream_, error) ||
        !d_point_delta_.Download(&capture->point_delta, stream_, error) ||
        !MaterializeTrial(&capture->trial_state, error)) {
      return false;
    }
    const cudaError_t status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
      *error = "FP32 component capture synchronization failed";
      return false;
    }
    capture->gradient = gradient_summary_;
    capture->step = step_summary_;
    capture->solver_info = solver_info_;
    return true;
  }

 private:
  static uint32_t Blocks(const size_t count) {
    return static_cast<uint32_t>((count + kBlockSize - 1) / kBlockSize);
  }

  template <typename T>
  bool AllocateAndUpload(DeviceBuffer<T>* device,
                         const std::vector<T>& host,
                         const bool float_payload,
                         std::string* error) {
    CudaPersistentDeviceRuntimeInfo* accounting =
        float_payload ? runtime_ : nullptr;
    return device->Allocate(host.size(), accounting, error) &&
           device->Upload(host, stream_, accounting, error);
  }

  bool AllocateTopology(std::string* error) {
    return AllocateAndUpload(&d_pose_meta_, topology_.poses, false, error) &&
           AllocateAndUpload(&d_point_meta_, topology_.points, false, error) &&
           AllocateAndUpload(&d_edge_meta_, topology_.edges, false, error) &&
           AllocateAndUpload(&d_pose_adjacency_, topology_.pose_adjacency,
                             false, error) &&
           AllocateAndUpload(&d_point_adjacency_, topology_.point_adjacency,
                             false, error) &&
           AllocateAndUpload(&d_edge_adjacency_, topology_.edge_adjacency,
                             false, error) &&
           AllocateAndUpload(&d_pose_segments_, topology_.pose_segments,
                             false, error) &&
           AllocateAndUpload(&d_point_segments_, topology_.point_segments,
                             false, error) &&
           AllocateAndUpload(&d_edge_segments_, topology_.edge_segments,
                             false, error) &&
           AllocateAndUpload(&d_pose_ranges_, topology_.pose_ranges, false,
                             error) &&
           AllocateAndUpload(&d_point_ranges_, topology_.point_ranges, false,
                             error) &&
           AllocateAndUpload(&d_edge_ranges_, topology_.edge_ranges, false,
                             error) &&
           AllocateAndUpload(&d_solve_pose_meta_, topology_.solve_poses,
                             false, error) &&
           AllocateAndUpload(&d_solve_point_meta_, topology_.solve_points,
                             false, error) &&
           AllocateAndUpload(&d_pose_edges_, topology_.pose_edges, false,
                             error) &&
           AllocateAndUpload(&d_point_edges_, topology_.point_edges, false,
                             error) &&
           AllocateAndUpload(&d_pair_meta_, topology_.pairs, false, error) &&
           AllocateAndUpload(&d_pair_contributions_, topology_.contributions,
                             false, error) &&
           AllocateAndUpload(&d_schur_segments_, topology_.schur_segments,
                             false, error) &&
           AllocateAndUpload(&d_pair_ranges_, topology_.pair_segment_ranges,
                             false, error) &&
           AllocateAndUpload(&d_image_updates_, topology_.image_updates,
                             false, error) &&
           AllocateAndUpload(&d_point_updates_, topology_.point_updates,
                             false, error);
  }

  bool AllocateWorkspaces(std::string* error) {
    const size_t schur_elements =
        static_cast<size_t>(topology_.pose_dimension) *
        topology_.pose_dimension;
    const size_t cost_count = topology_.cost_order.size();
    const size_t cost_partials =
        std::max<size_t>(1, (cost_count + kCostReductionThreads - 1) /
                                kCostReductionThreads);
    if (!d_visual_output_.Allocate(topology_.visual.size(), runtime_, error) ||
        !d_lidar_output_.Allocate(topology_.lidar.size(), runtime_, error) ||
        !d_record_status_summary_.Allocate(1, nullptr, error) ||
        !d_pose_blocks_.Allocate(topology_.poses.size(), runtime_, error) ||
        !d_point_blocks_.Allocate(topology_.points.size(), runtime_, error) ||
        !d_edge_blocks_.Allocate(topology_.edges.size(), runtime_, error) ||
        !d_pose_partials_.Allocate(topology_.pose_segments.size() *
                                       kPosePartialScalars,
                                   runtime_, error) ||
        !d_point_partials_.Allocate(topology_.point_segments.size() *
                                        kPointPartialScalars,
                                    runtime_, error) ||
        !d_edge_partials_.Allocate(topology_.edge_segments.size() *
                                       kEdgePartialScalars,
                                   runtime_, error) ||
        !d_frozen_pose_scaling_.Allocate(topology_.poses.size() * 6,
                                         runtime_, error) ||
        !d_frozen_point_scaling_.Allocate(topology_.points.size() * 3,
                                          runtime_, error) ||
        !d_gradient_summary_.Allocate(1, runtime_, error) ||
        !d_point_inverse_.Allocate(topology_.points.size() * 9, runtime_,
                                   error) ||
        !d_inverse_gradient_.Allocate(topology_.points.size() * 3, runtime_,
                                      error) ||
        !d_point_factor_status_.Allocate(topology_.points.size(), nullptr,
                                         error) ||
        !d_factor_status_summary_.Allocate(1, nullptr, error) ||
        !d_transformed_edges_.Allocate(topology_.edges.size() * 18, runtime_,
                                       error) ||
        !d_schur_partials_.Allocate(topology_.schur_segments.size() *
                                        kSchurPartialScalars,
                                    runtime_, error) ||
        !d_schur_.Allocate(schur_elements, runtime_, error) ||
        !d_factor_.Allocate(schur_elements, runtime_, error) ||
        !d_rhs_.Allocate(topology_.pose_dimension, runtime_, error) ||
        !d_camera_delta_.Allocate(topology_.pose_dimension, runtime_, error) ||
        !d_point_delta_.Allocate(topology_.points.size() * 3, runtime_, error) ||
        !d_solver_info_.Allocate(1, nullptr, error) ||
        !d_step_summary_.Allocate(1, runtime_, error) ||
        !d_state_status_.Allocate(1, nullptr, error) ||
        !d_state_norm_.Allocate(2, nullptr, error) ||
        !d_cost_values_.Allocate(cost_count, nullptr, error) ||
        !d_cost_partial_a_.Allocate(cost_partials, nullptr, error) ||
        !d_cost_partial_b_.Allocate(cost_partials, nullptr, error)) {
      return false;
    }
    cusolverStatus_t status = cusolverDnSpotrf_bufferSize(
        solver_, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(topology_.pose_dimension), d_factor_.get(),
        static_cast<int>(topology_.pose_dimension), &solver_workspace_elements_);
    if (status != CUSOLVER_STATUS_SUCCESS || solver_workspace_elements_ <= 0) {
      *error = "FP32 cusolverDnSpotrf_bufferSize failed";
      return false;
    }
    return d_solver_workspace_.Allocate(
        static_cast<size_t>(solver_workspace_elements_), runtime_, error);
  }

  bool ComputeCost(const int slot, double* cost, std::string* error) {
    if (topology_.cost_order.empty()) {
      *cost = 0.0;
      return true;
    }
    CostValuesKernelD<<<Blocks(topology_.cost_order.size()), kBlockSize, 0,
                        stream_>>>(
        d_cost_order_.get(), topology_.cost_order.size(),
        d_visual_cost_static_.get(), d_lidar_cost_static_.get(),
        d_images_[slot].get(), d_points_[slot].get(), d_cameras_.get(),
        static_cast<uint8_t>(loss_mode_), loss_scale_, d_cost_values_.get());
    const double* input = d_cost_values_.get();
    size_t count = topology_.cost_order.size();
    double* output = d_cost_partial_a_.get();
    bool use_a = true;
    while (count > 1) {
      const size_t blocks = (count + kCostReductionThreads - 1) /
                            kCostReductionThreads;
      CostReductionKernelD<<<blocks, kCostReductionThreads, 0, stream_>>>(
          input, count, output);
      count = blocks;
      input = output;
      use_a = !use_a;
      output = use_a ? d_cost_partial_a_.get() : d_cost_partial_b_.get();
    }
    const cudaError_t copy_status = cudaMemcpyAsync(
        cost, input, sizeof(double), cudaMemcpyDeviceToHost, stream_);
    const cudaError_t sync_status = copy_status == cudaSuccess
        ? cudaStreamSynchronize(stream_) : copy_status;
    ++runtime_->fp64_cost_calls;
    if (sync_status != cudaSuccess || !std::isfinite(*cost)) {
      *error = "FP32 independent FP64 cost oracle failed";
      return false;
    }
    return true;
  }

  bool MaterializeTrial(Snapshot* output, std::string* error) {
    const int saved = current_slot_;
    current_slot_ = 1 - current_slot_;
    const bool ok = MaterializeCurrent(output, error);
    --runtime_->final_state_materialization_operations;
    current_slot_ = saved;
    return ok;
  }

  void Close() {
    if (stream_ != nullptr) cudaStreamSynchronize(stream_);
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    solver_ = nullptr;
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  const Snapshot* snapshot_ = nullptr;
  const CudaFullLmOptions* options_ = nullptr;
  CudaPersistentDeviceRuntimeInfo* runtime_ = nullptr;
  CudaArithmeticPrecision precision_ = CudaArithmeticPrecision::kFp32Core;
  Fp32Topology topology_;
  CudaLossMode loss_mode_ = CudaLossMode::kTrivial;
  double loss_scale_ = 1.0;
  cudaStream_t stream_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  int solver_workspace_elements_ = 0;
  int solver_info_ = 0;
  int current_slot_ = 0;
  FGradientSummary gradient_summary_{};
  FStepSummary step_summary_{};

  DeviceBuffer<FImageState> d_images_[2];
  DeviceBuffer<FPointState> d_points_[2];
  DeviceBuffer<FCameraState> d_cameras_;
  DeviceBuffer<FVisualStatic> d_visual_static_;
  DeviceBuffer<FVisualCostStatic> d_visual_cost_static_;
  DeviceBuffer<FLidarStatic> d_lidar_static_;
  DeviceBuffer<FLidarCostStatic> d_lidar_cost_static_;
  DeviceBuffer<FCostEntry> d_cost_order_;
  DeviceBuffer<FPoseMeta> d_pose_meta_;
  DeviceBuffer<FPointMeta> d_point_meta_;
  DeviceBuffer<FEdgeMeta> d_edge_meta_;
  DeviceBuffer<uint32_t> d_pose_adjacency_;
  DeviceBuffer<FPointAdjacency> d_point_adjacency_;
  DeviceBuffer<uint32_t> d_edge_adjacency_;
  DeviceBuffer<FSegment> d_pose_segments_;
  DeviceBuffer<FSegment> d_point_segments_;
  DeviceBuffer<FSegment> d_edge_segments_;
  DeviceBuffer<FRange> d_pose_ranges_;
  DeviceBuffer<FRange> d_point_ranges_;
  DeviceBuffer<FRange> d_edge_ranges_;
  DeviceBuffer<FSolvePoseMeta> d_solve_pose_meta_;
  DeviceBuffer<FSolvePointMeta> d_solve_point_meta_;
  DeviceBuffer<uint32_t> d_pose_edges_;
  DeviceBuffer<uint32_t> d_point_edges_;
  DeviceBuffer<FPairMeta> d_pair_meta_;
  DeviceBuffer<FPairContribution> d_pair_contributions_;
  DeviceBuffer<FSchurSegment> d_schur_segments_;
  DeviceBuffer<FRange> d_pair_ranges_;
  DeviceBuffer<FImageUpdateMeta> d_image_updates_;
  DeviceBuffer<FPointUpdateMeta> d_point_updates_;
  DeviceBuffer<FVisualOutput> d_visual_output_;
  DeviceBuffer<FLidarOutput> d_lidar_output_;
  DeviceBuffer<FRecordStatusSummary> d_record_status_summary_;
  DeviceBuffer<FPoseBlock> d_pose_blocks_;
  DeviceBuffer<FPointBlock> d_point_blocks_;
  DeviceBuffer<FEdgeBlock> d_edge_blocks_;
  DeviceBuffer<float> d_pose_partials_;
  DeviceBuffer<float> d_point_partials_;
  DeviceBuffer<float> d_edge_partials_;
  DeviceBuffer<float> d_frozen_pose_scaling_;
  DeviceBuffer<float> d_frozen_point_scaling_;
  DeviceBuffer<FGradientSummary> d_gradient_summary_;
  DeviceBuffer<float> d_point_inverse_;
  DeviceBuffer<float> d_inverse_gradient_;
  DeviceBuffer<uint8_t> d_point_factor_status_;
  DeviceBuffer<FFactorStatusSummary> d_factor_status_summary_;
  DeviceBuffer<float> d_transformed_edges_;
  DeviceBuffer<float> d_schur_partials_;
  DeviceBuffer<float> d_schur_;
  DeviceBuffer<float> d_factor_;
  DeviceBuffer<float> d_rhs_;
  DeviceBuffer<float> d_camera_delta_;
  DeviceBuffer<float> d_point_delta_;
  DeviceBuffer<float> d_solver_workspace_;
  DeviceBuffer<int> d_solver_info_;
  DeviceBuffer<FStepSummary> d_step_summary_;
  DeviceBuffer<uint32_t> d_state_status_;
  DeviceBuffer<double> d_state_norm_;
  DeviceBuffer<double> d_cost_values_;
  DeviceBuffer<double> d_cost_partial_a_;
  DeviceBuffer<double> d_cost_partial_b_;
};

}  // namespace

bool RunCustomCudaFp32Experimental(const Snapshot& snapshot,
                                   const CudaFullLmOptions& options,
                                   const CudaArithmeticPrecision precision,
                                   CudaFullLmResult* result,
                                   std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = CudaFullLmResult();
  error->clear();
  if (options.device_context_mode != CudaDeviceContextMode::kDeviceControl) {
    *error = "FP32 experiment requires typed device_control context";
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "invalid_fp32_device_context";
    result->error_classification = CudaSolveErrorClass::kInvalidOptions;
    return false;
  }
  result->runtime.device_context_backend = "device_control";
  result->runtime.current_linearization_cache_effective =
      options.current_linearization_cache_mode ==
      CudaCurrentLinearizationCacheMode::kEnabled;
  const auto start = std::chrono::steady_clock::now();
  Fp32ExperimentContext context;
  if (!context.Initialize(snapshot, options, precision, result, error)) {
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "fp32_initialization_failed";
    result->error_classification = CudaSolveErrorClass::kInitialLinearization;
    return false;
  }
  const int32_t maximum_iterations = options.max_num_iterations < 0
      ? snapshot.metadata.max_num_iterations : options.max_num_iterations;
  const int32_t invalid_limit = options.max_num_consecutive_invalid_steps < 0
      ? (snapshot.metadata.max_consecutive_invalid_steps > 0
             ? snapshot.metadata.max_consecutive_invalid_steps
             : 10)
      : options.max_num_consecutive_invalid_steps;
  const double function_tolerance = options.function_tolerance < 0.0
      ? snapshot.metadata.function_tolerance : options.function_tolerance;
  const double gradient_tolerance = options.gradient_tolerance < 0.0
      ? snapshot.metadata.gradient_tolerance : options.gradient_tolerance;
  const double parameter_tolerance = options.parameter_tolerance < 0.0
      ? snapshot.metadata.parameter_tolerance : options.parameter_tolerance;
  if (maximum_iterations < 0 || invalid_limit <= 0 ||
      !std::isfinite(function_tolerance) || function_tolerance < 0.0 ||
      !std::isfinite(gradient_tolerance) || gradient_tolerance < 0.0 ||
      !std::isfinite(parameter_tolerance) || parameter_tolerance < 0.0 ||
      !std::isfinite(options.initial_trust_region_radius) ||
      options.initial_trust_region_radius <= 0.0 ||
      !std::isfinite(options.min_trust_region_radius) ||
      options.min_trust_region_radius < 0.0 ||
      !std::isfinite(options.max_trust_region_radius) ||
      options.max_trust_region_radius < options.initial_trust_region_radius ||
      !std::isfinite(options.min_relative_decrease) ||
      options.min_relative_decrease < 0.0) {
    *error = "FP32 full-LM options are invalid";
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "invalid_fp32_options";
    result->error_classification = CudaSolveErrorClass::kInvalidOptions;
    return false;
  }
  double current_cost = 0.0;
  if (!context.ComputeCurrentCost(&current_cost, error)) {
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "fp32_initial_cost_failed";
    result->error_classification = CudaSolveErrorClass::kInitialLinearization;
    return false;
  }
  result->initial_cost = current_cost;
  result->final_cost = current_cost;
  if (!context.Linearize(true, error)) {
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "fp32_initial_linearization_failed";
    result->error_classification = CudaSolveErrorClass::kInitialLinearization;
    return false;
  }
  result->accepted_steps = 1;  // Preserve the established iteration-0 field.
  result->initial_projected_gradient_max_norm =
      context.gradient_summary().projected;
  result->initial_scaled_gradient_norm = context.gradient_summary().scaled;
  double radius = options.initial_trust_region_radius;
  double decrease_factor = 2.0;
  int consecutive_invalid = 0;
  result->termination_type = CudaTerminationType::kNoConvergence;
  result->termination_reason = maximum_iterations == 0
      ? "maximum_trial_iterations" : "maximum_trial_iterations";

  if (context.gradient_summary().projected <= gradient_tolerance) {
    result->termination_type = CudaTerminationType::kConvergence;
    result->termination_reason = "gradient_tolerance";
  }
  for (int32_t trial = 0;
       trial < maximum_iterations &&
       result->termination_type == CudaTerminationType::kNoConvergence;
       ++trial) {
    CudaLmIteration iteration;
    iteration.iteration = trial + 1;
    iteration.cost_before = current_cost;
    iteration.cost_after = current_cost;
    iteration.trial_cost = current_cost;
    iteration.projected_gradient_max_norm =
        context.gradient_summary().projected;
    iteration.scaled_gradient_norm = context.gradient_summary().scaled;
    iteration.radius_before = radius;
    iteration.radius_after = radius;
    iteration.lambda_before = 1.0 / radius;
    iteration.lambda_after = iteration.lambda_before;
    bool factorization_success = false;
    double trial_cost = current_cost;
    double state_norm = 0.0;
    double difference_norm = 0.0;
    std::string step_error;
    const bool step_ok = context.SolveStep(
        static_cast<float>(iteration.lambda_before), &trial_cost, &state_norm,
        &difference_norm, &factorization_success, &step_error);
    ++result->trial_iterations;
    iteration.factorization_success = factorization_success;
    if (!step_ok) {
      *error = step_error;
      result->error = step_error;
      result->termination_type = CudaTerminationType::kFailure;
      result->termination_reason = "fp32_cuda_step_failed";
      result->error_classification = CudaSolveErrorClass::kCudaStep;
      result->trace.push_back(iteration);
      break;
    }
    if (!factorization_success) {
      ++result->factorization_failures;
      ++result->invalid_steps;
      ++consecutive_invalid;
      iteration.invalid = true;
      radius /= decrease_factor;
      decrease_factor *= 2.0;
      iteration.radius_after = radius;
      iteration.lambda_after = 1.0 / radius;
      iteration.termination_reason = "recoverable_fp32_factorization_failure";
      result->trace.push_back(iteration);
      if (consecutive_invalid >= invalid_limit) {
        *error = step_error;
        result->error = step_error;
        result->termination_type = CudaTerminationType::kFailure;
        result->termination_reason = "maximum_consecutive_invalid_steps";
      } else if (radius <= options.min_trust_region_radius) {
        result->termination_type = CudaTerminationType::kConvergence;
        result->termination_reason = "minimum_trust_region_radius";
      }
      continue;
    }
    if (options.layer_c.fault_injection ==
        CudaFaultInjection::kForceNonfiniteTrial) {
      *error = "FP32 injected post-trial failure";
      result->error = *error;
      result->termination_type = CudaTerminationType::kFailure;
      result->termination_reason = "fp32_injected_post_trial_failure";
      result->error_classification = CudaSolveErrorClass::kCudaStep;
      result->trace.push_back(iteration);
      break;
    }
    consecutive_invalid = 0;
    iteration.predicted_reduction =
        static_cast<double>(context.step_summary().predicted_reduction);
    iteration.backward_error =
        static_cast<double>(context.step_summary().backward_error);
    iteration.step_norm = difference_norm;
    iteration.step_valid =
        std::isfinite(iteration.predicted_reduction) &&
        iteration.predicted_reduction > 0.0;
    iteration.trial_cost = trial_cost;
    iteration.trial_finite = std::isfinite(trial_cost);
    if (!iteration.step_valid || !iteration.trial_finite ||
        !std::isfinite(iteration.backward_error)) {
      *error = "FP32 step produced a non-finite or non-positive model";
      result->error = *error;
      result->termination_type = CudaTerminationType::kFailure;
      result->termination_reason = "fp32_nonfinite_model";
      result->error_classification = CudaSolveErrorClass::kNonfiniteModel;
      result->trace.push_back(iteration);
      break;
    }
    iteration.actual_reduction = current_cost - trial_cost;
    iteration.rho =
        iteration.actual_reduction / iteration.predicted_reduction;
    iteration.parameter_metric =
        difference_norm / (state_norm + parameter_tolerance);
    iteration.function_metric = fabs(iteration.actual_reduction) /
                                std::max(current_cost, 1e-300);
    if (difference_norm <=
        parameter_tolerance * (state_norm + parameter_tolerance)) {
      result->termination_type = CudaTerminationType::kConvergence;
      result->termination_reason = "parameter_tolerance";
      result->trace.push_back(iteration);
      break;
    }
    if (fabs(iteration.actual_reduction) <=
        function_tolerance * current_cost) {
      result->termination_type = CudaTerminationType::kConvergence;
      result->termination_reason = "function_tolerance";
      result->trace.push_back(iteration);
      break;
    }
    iteration.accepted_decision = std::isfinite(iteration.rho) &&
        iteration.rho > options.min_relative_decrease;
    iteration.accepted = iteration.accepted_decision;
    if (iteration.accepted_decision) {
      context.AcceptTrial();
      ++result->accepted_decisions;
      ++result->accepted_commits;
      ++result->accepted_steps;
      iteration.accepted_commit_success = true;
      iteration.cost_after = trial_cost;
      current_cost = trial_cost;
      const double radius_update = std::max(
          1.0 / 3.0, 1.0 - std::pow(2.0 * iteration.rho - 1.0, 3.0));
      radius = std::min(options.max_trust_region_radius,
                        radius / radius_update);
      decrease_factor = 2.0;
      if (!context.Linearize(false, error)) {
        result->error = *error;
        result->termination_type = CudaTerminationType::kFailure;
        result->termination_reason = "fp32_accepted_relinearization_failed";
        result->error_classification =
            CudaSolveErrorClass::kCurrentLinearization;
        result->trace.push_back(iteration);
        break;
      }
      iteration.projected_gradient_max_norm =
          context.gradient_summary().projected;
      iteration.scaled_gradient_norm = context.gradient_summary().scaled;
    } else {
      ++result->rejected_steps;
      radius /= decrease_factor;
      decrease_factor *= 2.0;
    }
    iteration.radius_after = radius;
    iteration.lambda_after = 1.0 / radius;
    result->trace.push_back(iteration);
    if (context.gradient_summary().projected <= gradient_tolerance &&
        iteration.accepted_decision) {
      result->termination_type = CudaTerminationType::kConvergence;
      result->termination_reason = "gradient_tolerance";
    } else if (radius <= options.min_trust_region_radius) {
      result->termination_type = CudaTerminationType::kConvergence;
      result->termination_reason = "minimum_trust_region_radius";
    } else if (trial + 1 >= maximum_iterations) {
      result->termination_type = CudaTerminationType::kNoConvergence;
      result->termination_reason = "maximum_trial_iterations";
    }
  }
  if (!context.MaterializeCurrent(&result->final_state, error)) {
    result->error = *error;
    result->termination_type = CudaTerminationType::kFailure;
    result->termination_reason = "fp32_final_materialization_failed";
    result->error_classification = CudaSolveErrorClass::kCudaStep;
  }
  result->final_cost = current_cost;
  result->final_projected_gradient_max_norm =
      context.gradient_summary().projected;
  result->final_scaled_gradient_norm = context.gradient_summary().scaled;
  result->final_radius = radius;
  result->final_lambda = 1.0 / radius;
  result->runtime.layer_a_calls =
      result->runtime.persistent_device.hessian_gradient_assembly_calls;
  result->runtime.layer_b_calls = result->runtime.layer_a_calls;
  result->runtime.layer_c_calls =
      result->runtime.persistent_device.schur_contribution_calls;
  result->runtime.cost_calls = result->runtime.persistent_device.fp64_cost_calls;
  result->runtime.persistent_device.close_succeeded = true;
  result->runtime.persistent_device.final_state_materialization_operations = 1;
  result->runtime.host_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
  result->success = result->termination_type != CudaTerminationType::kFailure;
  if (!result->trace.empty())
    result->trace.back().termination_reason = result->termination_reason;
  return result->success;
}

namespace {

double PercentileValue(const std::vector<double>& sorted,
                       const double fraction) {
  if (sorted.empty()) return 0.0;
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - lower;
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

bool ComputeErrorStatistics(const std::vector<double>& reference,
                            const std::vector<double>& candidate,
                            const std::string& kind,
                            CudaPrecisionErrorStatistics* statistics,
                            std::string* error) {
  if (reference.size() != candidate.size()) {
    *error = "precision component size mismatch for " + kind;
    return false;
  }
  *statistics = CudaPrecisionErrorStatistics();
  statistics->count = reference.size();
  statistics->worst_entity_kind = kind;
  if (reference.empty()) return true;
  std::vector<double> absolute;
  std::vector<double> relative;
  absolute.reserve(reference.size());
  relative.reserve(reference.size());
  long double absolute_sum = 0.0;
  long double squared_sum = 0.0;
  long double reference_squared = 0.0;
  long double difference_squared = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    statistics->reference_max_magnitude = std::max(
        statistics->reference_max_magnitude, std::abs(reference[i]));
    statistics->candidate_max_magnitude = std::max(
        statistics->candidate_max_magnitude, std::abs(candidate[i]));
  }
  const double floor = 1e-12 * statistics->reference_max_magnitude;
  for (size_t i = 0; i < reference.size(); ++i) {
    if (std::isfinite(reference[i]) && std::isfinite(candidate[i]))
      ++statistics->finite_count;
    const double difference = std::abs(candidate[i] - reference[i]);
    const double relative_error =
        difference / std::max(std::abs(reference[i]), floor);
    absolute.push_back(difference);
    relative.push_back(relative_error);
    absolute_sum += difference;
    squared_sum += static_cast<long double>(difference) * difference;
    reference_squared +=
        static_cast<long double>(reference[i]) * reference[i];
    difference_squared +=
        static_cast<long double>(candidate[i] - reference[i]) *
        (candidate[i] - reference[i]);
    if (difference > statistics->max_abs_error) {
      statistics->max_abs_error = difference;
      statistics->worst_index = i;
      statistics->worst_reference = reference[i];
      statistics->worst_candidate = candidate[i];
    }
    statistics->max_relative_error =
        std::max(statistics->max_relative_error, relative_error);
  }
  std::sort(absolute.begin(), absolute.end());
  std::sort(relative.begin(), relative.end());
  statistics->mean_abs_error =
      static_cast<double>(absolute_sum / reference.size());
  statistics->rms_abs_error =
      std::sqrt(static_cast<double>(squared_sum / reference.size()));
  statistics->p50_abs_error = PercentileValue(absolute, 0.50);
  statistics->p95_abs_error = PercentileValue(absolute, 0.95);
  statistics->p99_abs_error = PercentileValue(absolute, 0.99);
  statistics->p50_relative_error = PercentileValue(relative, 0.50);
  statistics->p95_relative_error = PercentileValue(relative, 0.95);
  statistics->p99_relative_error = PercentileValue(relative, 0.99);
  statistics->normalized_max_error = statistics->max_abs_error /
      std::max(1.0, statistics->reference_max_magnitude);
  statistics->normalized_frobenius_error =
      std::sqrt(static_cast<double>(difference_squared)) /
      std::max(1.0, std::sqrt(static_cast<double>(reference_squared)));
  return statistics->finite_count == statistics->count;
}

bool FactorPointHostD(const double* matrix, double* inverse) {
  const double l00_squared = matrix[0];
  if (!(l00_squared > 0.0) || !std::isfinite(l00_squared)) return false;
  const double l00 = std::sqrt(l00_squared);
  const double l10 = matrix[3] / l00;
  const double l20 = matrix[6] / l00;
  const double l11_squared = matrix[4] - l10 * l10;
  if (!(l11_squared > 0.0) || !std::isfinite(l11_squared)) return false;
  const double l11 = std::sqrt(l11_squared);
  const double l21 = (matrix[7] - l20 * l10) / l11;
  const double l22_squared = matrix[8] - l20 * l20 - l21 * l21;
  if (!(l22_squared > 0.0) || !std::isfinite(l22_squared)) return false;
  const double l22 = std::sqrt(l22_squared);
  for (uint32_t column = 0; column < 3; ++column) {
    const double b0 = column == 0 ? 1.0 : 0.0;
    const double b1 = column == 1 ? 1.0 : 0.0;
    const double b2 = column == 2 ? 1.0 : 0.0;
    const double y0 = b0 / l00;
    const double y1 = (b1 - l10 * y0) / l11;
    const double y2 = (b2 - l20 * y0 - l21 * y1) / l22;
    const double x2 = y2 / l22;
    const double x1 = (y1 - l21 * x2) / l11;
    const double x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    inverse[column] = x0;
    inverse[3 + column] = x1;
    inverse[6 + column] = x2;
  }
  return true;
}

void AppendVariableState(const Snapshot& state,
                         const Fp32Topology& topology,
                         std::vector<double>* values) {
  for (const FPoseMeta& pose : topology.poses) {
    const ImageSnapshot& image = state.images[pose.image_entity];
    values->insert(values->end(), image.qvec.begin(), image.qvec.end());
    for (uint32_t row = 3; row < pose.dimension; ++row)
      values->push_back(
          image.tvec[pose.free_translation_indices[row - 3]]);
  }
  for (const FPointMeta& point : topology.points) {
    const auto& xyz = state.points[point.point_entity].xyz;
    values->insert(values->end(), xyz.begin(), xyz.end());
  }
}

}  // namespace

bool RunCudaPrecisionComponentForTesting(
    const Snapshot& snapshot,
    const double lambda,
    const size_t fp32_repeats,
    CudaPrecisionComponentResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr || !std::isfinite(lambda) ||
      lambda <= 0.0 || fp32_repeats == 0) {
    if (error != nullptr) *error = "precision component arguments are invalid";
    return false;
  }
  *result = CudaPrecisionComponentResult();
  error->clear();
  CudaLayerCOptions reference_options;
  reference_options.lambda = lambda;
  reference_options.layer_b.layer_a.residual_order =
      CudaResidualOrder::kSourceInsertion;
  reference_options.layer_b.hessian_assembly_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  reference_options.layer_b.hessian_segment_size_for_testing = kSegmentSize;
  reference_options.schur_contribution_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  reference_options.schur_segment_size_for_testing = kSegmentSize;
  reference_options.layer_b.reduction_mode =
      CudaReductionMode::kParallelDeterministic;
  reference_options.layer_b.cost_reduction_threads = 128;
  reference_options.layer_b.loss_mode = CudaLossMode::kFromSnapshot;
  CudaLayerCResult reference;
  if (!RunCudaSnapshotLayerC(snapshot, reference_options, &reference, error)) {
    result->error = *error;
    return false;
  }

  CudaFullLmOptions fp32_options;
  fp32_options.arithmetic_precision = CudaArithmeticPrecision::kFp32Core;
  fp32_options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  fp32_options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  fp32_options.execution_profile = CudaExecutionProfile::kCompactControl;
  fp32_options.layer_c = reference_options;
  CudaFullLmResult runtime_result;
  Fp32ExperimentContext context;
  if (!context.Initialize(snapshot, fp32_options,
                          CudaArithmeticPrecision::kFp32Core,
                          &runtime_result, error) ||
      !context.Linearize(true, error)) {
    result->error = *error;
    return false;
  }
  double current_cost = 0.0;
  double trial_cost = 0.0;
  double state_norm = 0.0;
  double difference_norm = 0.0;
  bool factorization_success = false;
  if (!context.ComputeCurrentCost(&current_cost, error) ||
      !context.SolveStep(static_cast<float>(lambda), &trial_cost, &state_norm,
                         &difference_norm, &factorization_success, error) ||
      !factorization_success) {
    result->error = error->empty() ? "FP32 component factorization failed"
                                   : *error;
    return false;
  }
  Fp32Capture candidate;
  if (!context.Capture(&candidate, error)) {
    result->error = *error;
    return false;
  }
  for (size_t repeat = 1; repeat < fp32_repeats; ++repeat) {
    double repeat_cost = 0.0;
    double repeat_state_norm = 0.0;
    double repeat_difference_norm = 0.0;
    bool repeat_factorization = false;
    if (!context.SolveStep(static_cast<float>(lambda), &repeat_cost,
                           &repeat_state_norm, &repeat_difference_norm,
                           &repeat_factorization, error) ||
        !repeat_factorization) {
      result->error = *error;
      return false;
    }
    Fp32Capture repeated;
    if (!context.Capture(&repeated, error) ||
        repeated.schur != candidate.schur ||
        repeated.camera_delta != candidate.camera_delta ||
        repeated.point_delta != candidate.point_delta ||
        repeat_cost != trial_cost) {
      *error = "FP32 segmented component is not deterministic";
      result->error = *error;
      return false;
    }
  }

  Fp32Topology topology;
  if (!BuildFp32Topology(snapshot, &topology, error)) return false;
  std::vector<double> input_reference;
  std::vector<double> input_candidate;
  for (const auto& image : snapshot.images) {
    for (double value : image.qvec) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
    for (double value : image.tvec) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
  }
  for (const auto& point : snapshot.points) {
    for (double value : point.xyz) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
  }
  for (const auto& camera : snapshot.cameras) {
    for (double value : camera.params) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
  }
  for (const auto& observation : snapshot.observations) {
    for (double value : observation.xy) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
  }
  for (const auto& lidar : snapshot.lidar) {
    for (double value : lidar.plane) {
      input_reference.push_back(value);
      input_candidate.push_back(static_cast<float>(value));
    }
    input_reference.push_back(lidar.weight);
    input_candidate.push_back(static_cast<float>(lidar.weight));
  }
  if (!ComputeErrorStatistics(input_reference, input_candidate, "input_cast",
                              &result->input_cast, error)) {
    result->error = *error;
    return false;
  }

  std::vector<double> reference_values;
  std::vector<double> candidate_values;
  for (size_t i = 0; i < reference.layer_b.layer_a.visual.size(); ++i) {
    const auto& lhs = reference.layer_b.layer_a.visual[i];
    const auto& rhs = candidate.visual[i];
    for (uint32_t row = 0; row < 2; ++row) {
      reference_values.push_back(lhs.residual[row]);
      candidate_values.push_back(rhs.residual[row]);
      for (uint32_t col = 0; col < 3; ++col) {
        reference_values.push_back(lhs.local_rotation_jacobian[row * 3 + col]);
        candidate_values.push_back(rhs.pose_jacobian[row * 6 + col]);
        reference_values.push_back(lhs.translation_jacobian[row * 3 + col]);
        candidate_values.push_back(rhs.pose_jacobian[row * 6 + 3 + col]);
        reference_values.push_back(lhs.point_jacobian[row * 3 + col]);
        candidate_values.push_back(rhs.point_jacobian[row * 3 + col]);
      }
    }
  }
  for (size_t i = 0; i < reference.layer_b.layer_a.lidar.size(); ++i) {
    reference_values.push_back(reference.layer_b.layer_a.lidar[i].residual);
    candidate_values.push_back(candidate.lidar[i].residual);
    for (uint32_t col = 0; col < 3; ++col) {
      reference_values.push_back(
          reference.layer_b.layer_a.lidar[i].point_jacobian[col]);
      candidate_values.push_back(candidate.lidar[i].point_jacobian[col]);
    }
  }
  if (!ComputeErrorStatistics(reference_values, candidate_values,
                              "residual_jacobian",
                              &result->residual_jacobian, error)) {
    result->error = *error;
    return false;
  }

  CudaLossMode loss_mode = CudaLossMode::kTrivial;
  std::string loss_name = snapshot.metadata.loss_function;
  std::transform(loss_name.begin(), loss_name.end(), loss_name.begin(),
                 ::toupper);
  if (loss_name == "SOFT_L1") loss_mode = CudaLossMode::kSoftL1;
  reference_values.clear();
  candidate_values.clear();
  for (size_t i = 0; i < reference.layer_b.layer_a.visual.size(); ++i) {
    const auto& lhs = reference.layer_b.layer_a.visual[i];
    const auto& rhs = candidate.visual[i];
    const double lhs_squared = lhs.residual[0] * lhs.residual[0] +
                               lhs.residual[1] * lhs.residual[1];
    const float rhs_squared = rhs.residual[0] * rhs.residual[0] +
                              rhs.residual[1] * rhs.residual[1];
    const double lhs_scale = loss_mode == CudaLossMode::kSoftL1
        ? std::sqrt(std::max(DBL_MIN,
            1.0 / std::sqrt(1.0 + lhs_squared))) : 1.0;
    const float rhs_scale = loss_mode == CudaLossMode::kSoftL1
        ? std::sqrt(std::max(FLT_MIN,
            1.0f / std::sqrt(1.0f + rhs_squared))) : 1.0f;
    reference_values.push_back(lhs_scale);
    candidate_values.push_back(rhs_scale);
  }
  for (size_t i = 0; i < reference.layer_b.layer_a.lidar.size(); ++i) {
    const double lhs_residual = reference.layer_b.layer_a.lidar[i].residual;
    const float rhs_residual = candidate.lidar[i].residual;
    const double lhs_scale = loss_mode == CudaLossMode::kSoftL1
        ? std::sqrt(std::max(DBL_MIN,
            1.0 / std::sqrt(1.0 + lhs_residual * lhs_residual))) : 1.0;
    const float rhs_scale = loss_mode == CudaLossMode::kSoftL1
        ? std::sqrt(std::max(FLT_MIN,
            1.0f / std::sqrt(1.0f + rhs_residual * rhs_residual))) : 1.0f;
    reference_values.push_back(lhs_scale);
    candidate_values.push_back(rhs_scale);
  }
  if (!ComputeErrorStatistics(reference_values, candidate_values,
                              "robust_scale", &result->robust_scale, error)) {
    result->error = *error;
    return false;
  }

  const auto compare_blocks = [&](const char* kind,
                                  CudaPrecisionErrorStatistics* statistics,
                                  const bool pose_blocks,
                                  const bool edge_blocks) {
    reference_values.clear();
    candidate_values.clear();
    if (pose_blocks) {
      for (size_t i = 0; i < reference.layer_b.poses.size(); ++i) {
        const auto& lhs = reference.layer_b.poses[i];
        const auto& rhs = candidate.poses[i];
        for (uint32_t row = 0; row < lhs.dimension; ++row) {
          reference_values.push_back(lhs.gradient[row]);
          candidate_values.push_back(rhs.gradient[row]);
          for (uint32_t col = 0; col < lhs.dimension; ++col) {
            reference_values.push_back(
                lhs.hessian[row * lhs.dimension + col]);
            candidate_values.push_back(
                rhs.hessian[row * rhs.dimension + col]);
          }
        }
      }
    } else if (edge_blocks) {
      for (size_t i = 0; i < reference.layer_b.edges.size(); ++i) {
        const auto& lhs = reference.layer_b.edges[i];
        const auto& rhs = candidate.edges[i];
        for (uint32_t scalar = 0; scalar < lhs.pose_dimension * 3; ++scalar) {
          reference_values.push_back(lhs.value[scalar]);
          candidate_values.push_back(rhs.value[scalar]);
        }
      }
    } else {
      for (size_t i = 0; i < reference.layer_b.points.size(); ++i) {
        const auto& lhs = reference.layer_b.points[i];
        const auto& rhs = candidate.points[i];
        for (uint32_t scalar = 0; scalar < 9; ++scalar) {
          reference_values.push_back(lhs.hessian[scalar]);
          candidate_values.push_back(rhs.hessian[scalar]);
        }
        for (uint32_t scalar = 0; scalar < 3; ++scalar) {
          reference_values.push_back(lhs.gradient[scalar]);
          candidate_values.push_back(rhs.gradient[scalar]);
        }
      }
    }
    return ComputeErrorStatistics(reference_values, candidate_values, kind,
                                  statistics, error);
  };
  if (!compare_blocks("pose_hessian_gradient",
                      &result->pose_hessian_gradient, true, false) ||
      !compare_blocks("point_hessian_gradient",
                      &result->point_hessian_gradient, false, false) ||
      !compare_blocks("edge_blocks", &result->edge_blocks, false, true)) {
    result->error = *error;
    return false;
  }

  reference_values.clear();
  candidate_values.clear();
  result->pose_diagonal_min = std::numeric_limits<double>::infinity();
  result->pose_diagonal_max = 0.0;
  for (size_t i = 0; i < reference.layer_b.poses.size(); ++i) {
    const auto& lhs = reference.layer_b.poses[i];
    const auto& rhs = candidate.poses[i];
    for (uint32_t row = 0; row < lhs.dimension; ++row) {
      const double diagonal = lhs.hessian[row * lhs.dimension + row];
      result->pose_diagonal_min = std::min(result->pose_diagonal_min,
                                           diagonal);
      result->pose_diagonal_max = std::max(result->pose_diagonal_max,
                                           diagonal);
      reference_values.push_back(lhs.jacobi_scaling[row]);
      candidate_values.push_back(rhs.jacobi_scaling[row]);
      reference_values.push_back(lhs.damping[row]);
      candidate_values.push_back(rhs.damping[row]);
    }
  }
  result->point_positive_diagonal_min =
      std::numeric_limits<double>::infinity();
  result->point_diagonal_max = 0.0;
  result->point_determinant_min_abs =
      std::numeric_limits<double>::infinity();
  std::vector<double> reference_inverse(reference.layer_b.points.size() * 9);
  for (size_t i = 0; i < reference.layer_b.points.size(); ++i) {
    const auto& lhs = reference.layer_b.points[i];
    const auto& rhs = candidate.points[i];
    double damped[9];
    for (uint32_t scalar = 0; scalar < 9; ++scalar)
      damped[scalar] = lhs.hessian[scalar];
    for (uint32_t row = 0; row < 3; ++row) {
      const double diagonal = lhs.hessian[row * 3 + row];
      if (diagonal > 0.0)
        result->point_positive_diagonal_min = std::min(
            result->point_positive_diagonal_min, diagonal);
      result->point_diagonal_max =
          std::max(result->point_diagonal_max, diagonal);
      damped[row * 3 + row] += lambda * lhs.damping[row];
      reference_values.push_back(lhs.jacobi_scaling[row]);
      candidate_values.push_back(rhs.jacobi_scaling[row]);
      reference_values.push_back(lhs.damping[row]);
      candidate_values.push_back(rhs.damping[row]);
    }
    const double determinant =
        damped[0] * (damped[4] * damped[8] - damped[5] * damped[7]) -
        damped[1] * (damped[3] * damped[8] - damped[5] * damped[6]) +
        damped[2] * (damped[3] * damped[7] - damped[4] * damped[6]);
    result->point_determinant_min_abs = std::min(
        result->point_determinant_min_abs, std::abs(determinant));
    if (!FactorPointHostD(damped, reference_inverse.data() + i * 9)) {
      *error = "reference point factorization failed in precision component";
      result->error = *error;
      return false;
    }
  }
  if (!ComputeErrorStatistics(reference_values, candidate_values,
                              "jacobi_damping", &result->jacobi_damping,
                              error)) {
    result->error = *error;
    return false;
  }
  candidate_values.assign(candidate.point_inverse.begin(),
                          candidate.point_inverse.end());
  if (!ComputeErrorStatistics(reference_inverse, candidate_values,
                              "point_inverse", &result->point_inverse,
                              error)) {
    result->error = *error;
    return false;
  }

  std::vector<double> reference_transformed(
      reference.layer_b.edges.size() * 18, 0.0);
  for (size_t edge_index = 0; edge_index < reference.layer_b.edges.size();
       ++edge_index) {
    const auto& edge = reference.layer_b.edges[edge_index];
    const double* inverse = reference_inverse.data() + edge.point_index * 9;
    for (uint32_t row = 0; row < edge.pose_dimension; ++row) {
      for (uint32_t col = 0; col < 3; ++col) {
        reference_transformed[edge_index * 18 + row * 3 + col] =
            edge.value[row * 3] * inverse[col] +
            edge.value[row * 3 + 1] * inverse[3 + col] +
            edge.value[row * 3 + 2] * inverse[6 + col];
      }
    }
  }
  candidate_values.assign(candidate.transformed_edges.begin(),
                          candidate.transformed_edges.end());
  if (!ComputeErrorStatistics(reference_transformed, candidate_values,
                              "transformed_edge",
                              &result->transformed_edge, error)) {
    result->error = *error;
    return false;
  }
  candidate_values.assign(candidate.schur.begin(), candidate.schur.end());
  if (!ComputeErrorStatistics(reference.schur, candidate_values, "schur",
                              &result->schur, error)) {
    result->error = *error;
    return false;
  }
  candidate_values.assign(candidate.rhs.begin(), candidate.rhs.end());
  if (!ComputeErrorStatistics(reference.rhs, candidate_values, "rhs",
                              &result->rhs, error)) {
    result->error = *error;
    return false;
  }
  candidate_values.assign(candidate.camera_delta.begin(),
                          candidate.camera_delta.end());
  if (!ComputeErrorStatistics(reference.camera_delta, candidate_values,
                              "camera_delta", &result->camera_delta, error)) {
    result->error = *error;
    return false;
  }
  candidate_values.assign(candidate.point_delta.begin(),
                          candidate.point_delta.end());
  if (!ComputeErrorStatistics(reference.point_delta, candidate_values,
                              "point_delta", &result->point_delta, error)) {
    result->error = *error;
    return false;
  }
  reference_values.clear();
  candidate_values.clear();
  AppendVariableState(reference.trial_state, topology, &reference_values);
  AppendVariableState(candidate.trial_state, topology, &candidate_values);
  if (!ComputeErrorStatistics(reference_values, candidate_values, "trial_state",
                              &result->trial_state, error)) {
    result->error = *error;
    return false;
  }

  const auto build_expected_trial = [&snapshot, &topology](
      const std::vector<double>& camera_delta,
      const std::vector<double>& point_delta,
      Snapshot* expected) {
    *expected = snapshot;
    for (size_t pose_index = 0; pose_index < topology.poses.size();
         ++pose_index) {
      const FPoseMeta& pose = topology.poses[pose_index];
      const FSolvePoseMeta& solve = topology.solve_poses[pose_index];
      std::array<double, 3> rotation{{camera_delta[solve.offset],
                                     camera_delta[solve.offset + 1],
                                     camera_delta[solve.offset + 2]}};
      std::array<double, 4> updated;
      if (!QuaternionPlusCeres14(
              expected->images[pose.image_entity].qvec, rotation, &updated)) {
        return false;
      }
      expected->images[pose.image_entity].qvec = updated;
      for (uint32_t row = 3; row < pose.dimension; ++row) {
        expected->images[pose.image_entity]
            .tvec[pose.free_translation_indices[row - 3]] +=
            camera_delta[solve.offset + row];
      }
    }
    for (size_t point_index = 0; point_index < topology.points.size();
         ++point_index) {
      for (uint32_t component = 0; component < 3; ++component) {
        expected->points[topology.points[point_index].point_entity]
            .xyz[component] += point_delta[point_index * 3 + component];
      }
    }
    return true;
  };
  Snapshot expected_fp64;
  Snapshot expected_fp32;
  std::vector<double> candidate_camera_delta(candidate.camera_delta.begin(),
                                             candidate.camera_delta.end());
  std::vector<double> candidate_point_delta(candidate.point_delta.begin(),
                                            candidate.point_delta.end());
  if (!build_expected_trial(reference.camera_delta, reference.point_delta,
                            &expected_fp64) ||
      !build_expected_trial(candidate_camera_delta, candidate_point_delta,
                            &expected_fp32)) {
    *error = "precision component host trial update failed";
    result->error = *error;
    return false;
  }
  reference_values.clear();
  candidate_values.clear();
  AppendVariableState(expected_fp64, topology, &reference_values);
  AppendVariableState(reference.trial_state, topology, &candidate_values);
  if (!ComputeErrorStatistics(reference_values, candidate_values,
                              "fp64_trial_update_consistency",
                              &result->fp64_trial_update_consistency, error)) {
    result->error = *error;
    return false;
  }
  reference_values.clear();
  candidate_values.clear();
  AppendVariableState(expected_fp32, topology, &reference_values);
  AppendVariableState(candidate.trial_state, topology, &candidate_values);
  if (!ComputeErrorStatistics(reference_values, candidate_values,
                              "fp32_trial_update_consistency",
                              &result->fp32_trial_update_consistency, error)) {
    result->error = *error;
    return false;
  }
  result->fp64_predicted_reduction = reference.predicted_reduction;
  result->fp32_predicted_reduction = candidate.step.predicted_reduction;
  result->fp64_backward_error = reference.backward_error;
  result->fp32_backward_error = candidate.step.backward_error;
  result->fp64_trial_cost = reference.trial_cost;
  result->fp32_trial_cost = trial_cost;
  result->fp64_factorization_routine = "cusolverDnDpotrf/Dpotrs";
  result->fp32_factorization_routine = "cusolverDnSpotrf/Spotrs";
  result->fp64_solver_info = reference.runtime.cusolver_dev_info;
  result->fp32_solver_info = candidate.solver_info;
  result->dpotrf_calls = reference.runtime.factorization_calls;
  result->dpotrs_calls = reference.runtime.factorization_calls;
  result->spotrf_calls = runtime_result.runtime.persistent_device.spotrf_calls;
  result->spotrs_calls = runtime_result.runtime.persistent_device.spotrs_calls;
  result->schur_diagonal_min = std::numeric_limits<double>::infinity();
  result->schur_diagonal_max = 0.0;
  for (uint32_t i = 0; i < topology.pose_dimension; ++i) {
    const double diagonal =
        reference.schur[static_cast<size_t>(i) * topology.pose_dimension + i];
    result->schur_diagonal_min =
        std::min(result->schur_diagonal_min, diagonal);
    result->schur_diagonal_max =
        std::max(result->schur_diagonal_max, diagonal);
  }
  result->schur_spd_reference = reference.runtime.cusolver_dev_info == 0;
  result->schur_condition_metric = "diagonal_dynamic_range_proxy";
  result->schur_condition_number =
      result->schur_diagonal_min > 0.0
          ? result->schur_diagonal_max / result->schur_diagonal_min
          : std::numeric_limits<double>::infinity();
  if (!topology.poses.empty()) {
    const FPoseMeta& pose = topology.poses.front();
    const FSolvePoseMeta& solve = topology.solve_poses.front();
    result->diagnostic_pose_image_id = pose.image_id;
    result->diagnostic_initial_qvec = snapshot.images[pose.image_entity].qvec;
    result->diagnostic_fp64_trial_qvec =
        reference.trial_state.images[pose.image_entity].qvec;
    result->diagnostic_fp32_trial_qvec =
        candidate.trial_state.images[pose.image_entity].qvec;
    for (uint32_t row = 0; row < pose.dimension; ++row) {
      result->diagnostic_fp64_camera_delta[row] =
          reference.camera_delta[solve.offset + row];
      result->diagnostic_fp32_camera_delta[row] =
          candidate.camera_delta[solve.offset + row];
    }
  }
  result->success = true;
  return true;
}

}  // namespace gpu_ba
}  // namespace colmap
