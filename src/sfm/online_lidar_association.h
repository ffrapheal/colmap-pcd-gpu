#ifndef COLMAP_SRC_SFM_ONLINE_LIDAR_ASSOCIATION_H_
#define COLMAP_SRC_SFM_ONLINE_LIDAR_ASSOCIATION_H_

#ifdef GPU_BA_CUDA_ENABLED

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/reconstruction.h"
#include "lidar/incremental_causal_lidar_map.h"
#include "lidar/lidar_point.h"
#include "lidar/pcd_projection.h"

namespace colmap {

enum class OnlineLidarAssociationRoute : uint8_t {
  PROJECTION = 0,
  KDTREE = 1,
};

struct OnlineLidarAssociationOptions {
  bool local_lidar_kdtree_only = false;
  int min_proj_num = 1;
  double kdtree_max_search_range = 1.5;
  double kdtree_min_search_range = 0.2;
  double search_range_drop_speed = 0.1;
  int ba_match_features_threshold = 200;
};

struct OnlineLidarAssociationRequest {
  uint64_t attempt_id = 0;
  uint32_t pass_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  std::vector<image_t> ordered_frozen_image_ids;
  std::vector<point3D_t> point3D_ids;
  uint64_t expected_map_version = 0;
  uint64_t expected_max_scan_index = 0;
  std::string expected_snapshot_sha256;
  std::string expected_geometry_sha256;
  OnlineLidarAssociationOptions options;
  lidar::PcdProjectionOptions projection_options;
  std::shared_ptr<const lidar::LidarMapSnapshot> snapshot;
};

struct OnlineLidarAssociation {
  uint64_t association_id = 0;
  uint64_t attempt_id = 0;
  uint32_t pass_index = 0;
  point3D_t point3D_id = kInvalidPoint3DId;
  std::array<double, 3> point3D_xyz{{0.0, 0.0, 0.0}};
  image_t owner_image_id = kInvalidImageId;
  point2D_t owner_point2D_idx = kInvalidPoint2DIdx;
  uint64_t map_version = 0;
  uint64_t max_scan_index = 0;
  std::string snapshot_sha256;
  std::string geometry_sha256;
  OnlineLidarAssociationRoute route =
      OnlineLidarAssociationRoute::PROJECTION;
  lidar::PlaneSample plane;
  lidar::VoxelKey plane_key;
  std::array<double, 4> plane_abcd{{0.0, 0.0, 0.0, 0.0}};
  bool has_search_range = false;
  double search_range = 0.0;
  LidarPointType lidar_point_type = LidarPointType::Proj;
  double projection_camera_distance = 0.0;
  double projection_angle_score = 0.0;
};

struct OnlineLidarProjectionCallAudit {
  image_t image_id = kInvalidImageId;
  lidar::SnapshotProjectionAudit projection;
};

struct OnlineLidarAssociationAudit {
  uint64_t attempt_id = 0;
  uint32_t pass_index = 0;
  image_t trigger_image_id = kInvalidImageId;
  uint64_t map_version = 0;
  uint64_t max_scan_index = 0;
  std::string snapshot_sha256;
  std::string geometry_sha256;

  uint64_t input_point_count = 0;
  uint64_t selected_association_count = 0;
  uint64_t skipped_point_count = 0;
  uint64_t skipped_no_window_observation_count = 0;
  uint64_t skipped_pair_threshold_point_count = 0;
  uint64_t skipped_no_plane_count = 0;
  uint64_t pair_threshold_skipped_observation_count = 0;

  uint64_t projection_route_point_count = 0;
  uint64_t kdtree_route_point_count = 0;
  uint64_t projection_selected_association_count = 0;
  uint64_t kdtree_selected_association_count = 0;

  uint64_t projection_call_count = 0;
  std::vector<OnlineLidarProjectionCallAudit> projection_calls;
  std::map<image_t, uint64_t> preliminary_selected_count_by_image;
  uint64_t trigger_preliminary_selected_count = 0;
  std::string association_sha256;
};

struct OnlineLidarAssociationOutput {
  std::vector<OnlineLidarAssociation> associations;
};

struct OnlineLidarAssociationTriggerGate {
  uint64_t minimum_count = 0;
  uint64_t previous_count = 0;
  uint64_t minimum_increase = 0;
  bool checked = false;
  bool passed = false;
  uint64_t actual_count = 0;
  uint64_t nearest_query_count = 0;

  bool Accepts(const uint64_t count) const {
    return count >= minimum_count &&
           (minimum_increase == 0 ||
            (count >= previous_count &&
             count - previous_count >= minimum_increase));
  }
};

bool SelectOnlineLidarAssociationRoute(
    const OnlineLidarAssociationOptions& options,
    size_t track_length,
    OnlineLidarAssociationRoute* route,
    std::string* error) noexcept;

bool BuildOnlineLidarAssociations(
    const Reconstruction& reconstruction,
    const OnlineLidarAssociationRequest& request,
    OnlineLidarAssociationOutput* output,
    OnlineLidarAssociationAudit* audit,
    std::string* error,
    OnlineLidarAssociationTriggerGate* trigger_gate = nullptr) noexcept;

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED

#endif  // COLMAP_SRC_SFM_ONLINE_LIDAR_ASSOCIATION_H_
