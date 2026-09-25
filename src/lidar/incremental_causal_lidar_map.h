#ifndef COLMAP_LIDAR_INCREMENTAL_CAUSAL_LIDAR_MAP_H
#define COLMAP_LIDAR_INCREMENTAL_CAUSAL_LIDAR_MAP_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lidar/cuda_normal_estimation.h"

namespace colmap {
namespace lidar {

enum class LidarCoordinateFrame : uint8_t {
  FASTLIO_WORLD = 0,
  COLMAP_WORLD = 1,
};

enum class LidarNormalScale : uint8_t {
  OUTER_0_15_M = 0,
  INNER_0_05_M = 1,
  PROJECTION = OUTER_0_15_M,
  BA = INNER_0_05_M,
};

struct VoxelKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;

  bool operator==(const VoxelKey& other) const noexcept;
  bool operator!=(const VoxelKey& other) const noexcept;
  bool operator<(const VoxelKey& other) const noexcept;
};

struct VoxelBlockKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;

  bool operator==(const VoxelBlockKey& other) const noexcept;
  bool operator!=(const VoxelBlockKey& other) const noexcept;
  bool operator<(const VoxelBlockKey& other) const noexcept;
};

VoxelBlockKey VoxelKeyToBlockKey(const VoxelKey& key) noexcept;

struct ScanSource {
  uint64_t scan_index = 0;
  std::string pcd_path;
  LidarCoordinateFrame input_frame = LidarCoordinateFrame::FASTLIO_WORLD;
  uint32_t point_transform_count = 0;
  uint32_t normal_transform_count = 0;

  // A zero size or empty digest means that the corresponding expectation is
  // not supplied. PCD inputs are non-empty, so zero is unambiguous here.
  uint64_t expected_size_bytes = 0;
  std::string expected_sha256;
};

struct LidarNormalValue {
  std::array<float, 3> normal{{0.0f, 0.0f, 0.0f}};
  float curvature = 0.0f;
  bool valid = false;
  uint64_t revision = 0;
};

struct LidarVoxelRecord {
  VoxelKey key;
  LidarCoordinateFrame frame = LidarCoordinateFrame::COLMAP_WORLD;
  uint64_t count = 0;
  std::array<double, 3> xyz_sum{{0.0, 0.0, 0.0}};
  std::array<float, 3> centroid{{0.0f, 0.0f, 0.0f}};
  uint64_t first_scan_index = 0;
  uint64_t last_scan_index = 0;
  LidarNormalValue outer_normal;
  LidarNormalValue inner_normal;
};

struct PlaneSample {
  VoxelKey key;
  LidarCoordinateFrame frame = LidarCoordinateFrame::COLMAP_WORLD;
  LidarNormalScale scale = LidarNormalScale::OUTER_0_15_M;
  std::array<float, 3> point{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> normal{{0.0f, 0.0f, 0.0f}};
  float curvature = 0.0f;
  uint64_t normal_revision = 0;
  uint64_t voxel_count = 0;
};

struct NearestPlaneResult {
  bool ok = false;
  bool found = false;
  PlaneSample plane;
  double squared_distance = 0.0;
  uint64_t visited_block_count = 0;
  std::string error;
};

struct LidarAabb {
  std::array<double, 3> min{{0.0, 0.0, 0.0}};
  std::array<double, 3> max{{0.0, 0.0, 0.0}};
  LidarCoordinateFrame frame = LidarCoordinateFrame::COLMAP_WORLD;
};

struct PlaneCollectionResult {
  bool ok = false;
  std::vector<PlaneSample> planes;
  uint64_t visited_block_count = 0;
  std::string error;
};

struct NeighborhoodSearchAudit {
  uint64_t query_count = 0;
  uint64_t block_probe_count = 0;
  uint64_t visited_block_count = 0;
  uint64_t visited_voxel_count = 0;
  uint64_t hit_count = 0;
  uint64_t inserted_key_count = 0;
};

struct AppendScanAudit {
  uint64_t scan_index = 0;
  uint64_t opened_scan_index = 0;
  uint64_t version_before = 0;
  uint64_t version_after = 0;
  uint64_t max_scan_index_before = 0;
  uint64_t max_scan_index_after = 0;
  LidarCoordinateFrame source_frame = LidarCoordinateFrame::FASTLIO_WORLD;
  LidarCoordinateFrame output_frame = LidarCoordinateFrame::COLMAP_WORLD;
  LidarCoordinateFrame derived_normal_frame =
      LidarCoordinateFrame::COLMAP_WORLD;
  uint32_t point_transform_count_min = 0;
  uint32_t point_transform_count_max = 0;
  uint32_t source_normal_transform_count_min = 0;
  uint32_t source_normal_transform_count_max = 0;
  uint32_t derived_normal_transform_count = 0;
  uint64_t file_size_bytes = 0;
  std::string file_sha256;
  uint64_t input_record_count = 0;
  uint64_t accepted_record_count = 0;
  uint64_t dropped_non_finite_record_count = 0;
  uint64_t zero_source_normal_count = 0;
  uint64_t nonzero_source_normal_count = 0;
  bool has_first_accepted_record = false;
  std::array<float, 3> first_transformed_point{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> first_transformed_source_normal{{0.0f, 0.0f,
                                                        0.0f}};
  uint64_t touched_voxel_count = 0;
  uint64_t new_voxel_count = 0;
  uint64_t count_only_voxel_count = 0;
  uint64_t geometry_changed_voxel_count = 0;
  uint64_t affected_query_count = 0;
  uint64_t support_voxel_count = 0;
  uint64_t outer_valid_count = 0;
  uint64_t outer_invalid_count = 0;
  uint64_t inner_valid_count = 0;
  uint64_t inner_invalid_count = 0;
  bool all_voxels_queried = false;
  uint64_t map_voxel_count = 0;
  uint64_t map_block_count = 0;
  uint64_t mutable_block_count = 0;
  NeighborhoodSearchAudit affected_search;
  NeighborhoodSearchAudit support_search;
  std::map<std::string, double> stage_milliseconds;
  double total_milliseconds = 0.0;
  CudaNormalEstimationTiming cuda_timing;
  std::string geometry_sha256_before;
  std::string geometry_sha256_after;
  std::string snapshot_sha256_before;
  std::string snapshot_sha256_after;
};

using LidarFileStatFunction = std::function<bool(
    const std::string&, uint64_t*, std::string*)>;
using LidarFileReadFunction = std::function<bool(
    const std::string&, std::vector<uint8_t>*, std::string*)>;
using LidarSha256Function = std::function<bool(
    const std::vector<uint8_t>&, std::string*, std::string*)>;
using LidarNormalEstimator = std::function<bool(
    const std::vector<float>&,
    const std::vector<float>&,
    float,
    float,
    std::vector<float>*,
    std::vector<float>*,
    CudaNormalEstimationTiming*,
    std::string*)>;

struct IncrementalCausalLidarMapDependencies {
  LidarFileStatFunction stat_file;
  LidarFileReadFunction read_file;
  LidarSha256Function sha256;
  LidarNormalEstimator estimate_normals;
};

namespace internal {
struct IncrementalCausalLidarMapState;
struct IncrementalCausalLidarMapImpl;
}  // namespace internal

class LidarMapSnapshot {
 public:
  uint64_t Version() const noexcept;
  uint64_t MaxScanIndex() const noexcept;
  LidarCoordinateFrame Frame() const noexcept;
  size_t VoxelCount() const noexcept;
  size_t BlockCount() const noexcept;
  const std::string& GeometrySha256() const noexcept;
  const std::string& SnapshotSha256() const noexcept;

  bool FindVoxel(const VoxelKey& key, LidarVoxelRecord* record) const noexcept;
  std::vector<VoxelKey> VoxelKeys() const;

  NearestPlaneResult FindNearestPlane(
      const std::array<double, 3>& query_world,
      LidarCoordinateFrame query_frame,
      LidarNormalScale scale,
      double max_distance) const noexcept;
  NearestPlaneResult FindNearestPlane(
      const std::array<double, 3>& query_world,
      LidarNormalScale scale,
      double max_distance) const noexcept;

  PlaneCollectionResult CollectPlanesInAabb(
      const LidarAabb& bounds, LidarNormalScale scale) const noexcept;

 private:
  friend class IncrementalCausalLidarMap;
  friend struct internal::IncrementalCausalLidarMapImpl;
  explicit LidarMapSnapshot(
      std::shared_ptr<const internal::IncrementalCausalLidarMapState> state);

  std::shared_ptr<const internal::IncrementalCausalLidarMapState> state_;
};

class IncrementalCausalLidarMap {
 public:
  static constexpr float kVoxelLeafMeters = 0.01f;
  static constexpr int64_t kVoxelsPerBlockAxis = 32;
  static constexpr float kOuterNormalRadiusMeters = 0.15f;
  static constexpr float kInnerNormalRadiusMeters = 0.05f;

  IncrementalCausalLidarMap();
  explicit IncrementalCausalLidarMap(
      IncrementalCausalLidarMapDependencies dependencies);
  ~IncrementalCausalLidarMap();

  IncrementalCausalLidarMap(const IncrementalCausalLidarMap&) = delete;
  IncrementalCausalLidarMap& operator=(
      const IncrementalCausalLidarMap&) = delete;

  std::shared_ptr<const LidarMapSnapshot> GetSnapshot() const noexcept;

  bool AppendScan(const ScanSource& source,
                  AppendScanAudit* audit,
                  std::string* error) noexcept;

 private:
  std::unique_ptr<internal::IncrementalCausalLidarMapImpl> impl_;
};

}  // namespace lidar
}  // namespace colmap

#endif  // COLMAP_LIDAR_INCREMENTAL_CAUSAL_LIDAR_MAP_H
