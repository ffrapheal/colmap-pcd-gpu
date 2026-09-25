#include "sfm/online_local_ba_postprocess.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>
#include <vector>

#include "base/projection.h"

namespace colmap {
namespace {

std::unordered_set<point3D_t> ExistingTouchedPoints(
    const Reconstruction& reconstruction,
    const std::unordered_set<point3D_t>& touched_point3D_ids) {
  std::unordered_set<point3D_t> existing;
  existing.reserve(touched_point3D_ids.size());
  for (const point3D_t point3D_id : touched_point3D_ids) {
    if (reconstruction.ExistsPoint3D(point3D_id)) {
      existing.insert(point3D_id);
    }
  }
  return existing;
}

std::vector<std::pair<image_t, point2D_t>> TouchedTrackObservations(
    const Reconstruction& reconstruction,
    const std::unordered_set<point3D_t>& touched_point3D_ids) {
  std::vector<std::pair<image_t, point2D_t>> observations;
  observations.reserve(touched_point3D_ids.size() * 3);
  for (const point3D_t point3D_id : touched_point3D_ids) {
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      continue;
    }
    for (const TrackElement& track_element :
         reconstruction.Point3D(point3D_id).Track().Elements()) {
      observations.emplace_back(track_element.image_id,
                                 track_element.point2D_idx);
    }
  }
  std::sort(observations.begin(), observations.end());
  observations.erase(std::unique(observations.begin(), observations.end()),
                      observations.end());
  return observations;
}

size_t FilterTouchedNegativeDepthObservations(
    Reconstruction* reconstruction,
    const std::vector<std::pair<image_t, point2D_t>>& observations) {
  size_t filtered_observation_count = 0;
  image_t previous_image_id = kInvalidImageId;
  const Image* image = nullptr;
  Eigen::Matrix3x4d projection_matrix;
  for (const auto& observation : observations) {
    if (observation.first != previous_image_id) {
      previous_image_id = observation.first;
      image = nullptr;
      if (reconstruction->ExistsImage(previous_image_id) &&
          reconstruction->Image(previous_image_id).IsRegistered()) {
        image = &reconstruction->Image(previous_image_id);
        projection_matrix = image->ProjectionMatrix();
      }
    }
    if (image == nullptr || observation.second >= image->NumPoints2D()) continue;
    const Point2D& point2D = image->Point2D(observation.second);
    if (!point2D.HasPoint3D() ||
        !reconstruction->ExistsPoint3D(point2D.Point3DId()) ||
        HasPointPositiveDepth(
            projection_matrix,
            reconstruction->Point3D(point2D.Point3DId()).XYZ())) {
      continue;
    }
    reconstruction->DeleteObservation(observation.first, observation.second);
    ++filtered_observation_count;
  }
  return filtered_observation_count;
}

}  // namespace

OnlineLocalBaTouchedFilterResult FilterOnlineLocalBaTouchedPoints(
    Reconstruction* reconstruction,
    const std::unordered_set<point3D_t>& touched_point3D_ids,
    const double max_reproj_error,
    const double min_tri_angle) {
  OnlineLocalBaTouchedFilterResult result;
  result.input_touched_point_count = touched_point3D_ids.size();
  if (reconstruction == nullptr || !std::isfinite(max_reproj_error) ||
      max_reproj_error < 0.0 || !std::isfinite(min_tri_angle) ||
      min_tri_angle < 0.0) {
    result.detail = "invalid touched-point filter input";
    return result;
  }

  try {
    const std::unordered_set<point3D_t> existing_before =
        ExistingTouchedPoints(*reconstruction, touched_point3D_ids);
    result.existing_input_point_count = existing_before.size();
    result.missing_input_point_count =
        touched_point3D_ids.size() - existing_before.size();
    result.reprojection_or_angle_filtered_observation_count =
        reconstruction->FilterPoints3D(max_reproj_error, min_tri_angle,
                                       existing_before);

    const auto observations =
        TouchedTrackObservations(*reconstruction, existing_before);
    result.negative_depth_filtered_observation_count =
        FilterTouchedNegativeDepthObservations(reconstruction, observations);
    result.filtered_observation_count =
        result.reprojection_or_angle_filtered_observation_count +
        result.negative_depth_filtered_observation_count;
    result.success = true;
    return result;
  } catch (const std::exception& exception) {
    result.detail = std::string("touched-point filter failed: ") +
                    exception.what();
    return result;
  }
}

}  // namespace colmap
