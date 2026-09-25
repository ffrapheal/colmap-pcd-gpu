#ifndef COLMAP_SRC_SFM_ONLINE_LOCAL_BA_POSTPROCESS_H_
#define COLMAP_SRC_SFM_ONLINE_LOCAL_BA_POSTPROCESS_H_

#include <cstddef>
#include <string>
#include <unordered_set>

#include "base/reconstruction.h"

namespace colmap {

struct OnlineLocalBaTouchedFilterResult {
  bool success = false;
  std::string detail;
  size_t input_touched_point_count = 0;
  size_t existing_input_point_count = 0;
  size_t missing_input_point_count = 0;
  size_t filtered_observation_count = 0;
  size_t reprojection_or_angle_filtered_observation_count = 0;
  size_t negative_depth_filtered_observation_count = 0;
};

OnlineLocalBaTouchedFilterResult FilterOnlineLocalBaTouchedPoints(
    Reconstruction* reconstruction,
    const std::unordered_set<point3D_t>& touched_point3D_ids,
    double max_reproj_error,
    double min_tri_angle);

}  // namespace colmap

#endif  // COLMAP_SRC_SFM_ONLINE_LOCAL_BA_POSTPROCESS_H_
