#include "sfm/online_lidar_association.h"

#ifdef GPU_BA_CUDA_ENABLED

#include <openssl/sha.h>

#include "base/camera_models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace colmap {
namespace {

bool Fail(std::string* error, const std::string& message) noexcept {
  if (error != nullptr) {
    try {
      *error = message;
    } catch (...) {
      error->clear();
    }
  }
  return false;
}

bool FailWithException(std::string* error,
                       const char* context,
                       const char* what) noexcept {
  try {
    std::string message(context);
    if (what != nullptr && what[0] != '\0') {
      message += ": ";
      message += what;
    }
    return Fail(error, message);
  } catch (...) {
    return Fail(error, context);
  }
}

constexpr char kAssociationCanonicalSchema[] =
    "colmap.online_lidar_association.canonical.v3";

static_assert(SHA256_DIGEST_LENGTH == 32,
              "canonical association hashes require SHA256");
static_assert(sizeof(float) == sizeof(uint32_t) &&
                  std::numeric_limits<float>::is_iec559,
              "canonical association hashes require IEEE-754 float32");
static_assert(sizeof(double) == sizeof(uint64_t) &&
                  std::numeric_limits<double>::is_iec559,
              "canonical association hashes require IEEE-754 float64");

bool SizeToUint64(const size_t value,
                  const char* field,
                  uint64_t* converted,
                  std::string* error) {
  if (converted == nullptr) {
    return Fail(error, "canonical size output is null");
  }
  if (static_cast<std::uintmax_t>(value) >
      std::numeric_limits<uint64_t>::max()) {
    return Fail(error, std::string(field) + " exceeds uint64_t");
  }
  *converted = static_cast<uint64_t>(value);
  return true;
}

bool IntToInt32(const int value,
                const char* field,
                int32_t* converted,
                std::string* error) {
  if (converted == nullptr) {
    return Fail(error, "canonical int32 output is null");
  }
  const std::intmax_t wide_value = static_cast<std::intmax_t>(value);
  if (wide_value < std::numeric_limits<int32_t>::min() ||
      wide_value > std::numeric_limits<int32_t>::max()) {
    return Fail(error, std::string(field) + " exceeds int32_t");
  }
  *converted = static_cast<int32_t>(value);
  return true;
}

class CanonicalWriter {
 public:
  void Uint8(const uint8_t value) { bytes_.push_back(value); }

  void Bool(const bool value) { Uint8(value ? uint8_t{1} : uint8_t{0}); }

  void Uint32(const uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      bytes_.push_back(
          static_cast<uint8_t>((value >> shift) & uint32_t{0xff}));
    }
  }

  void Int32(const int32_t value) {
    Uint32(static_cast<uint32_t>(value));
  }

  void Uint64(const uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      bytes_.push_back(
          static_cast<uint8_t>((value >> shift) & uint64_t{0xff}));
    }
  }

  void Int64(const int64_t value) {
    Uint64(static_cast<uint64_t>(value));
  }

  bool Size(const size_t value, const char* field, std::string* error) {
    uint64_t converted = 0;
    if (!SizeToUint64(value, field, &converted, error)) {
      return false;
    }
    Uint64(converted);
    return true;
  }

  bool Float(const float value, const char* field, std::string* error) {
    if (!std::isfinite(value)) {
      return Fail(error, std::string(field) + " is non-finite");
    }
    const float canonical_value = value == 0.0f ? 0.0f : value;
    uint32_t bits = 0;
    std::memcpy(&bits, &canonical_value, sizeof(bits));
    Uint32(bits);
    return true;
  }

  bool Double(const double value, const char* field, std::string* error) {
    if (!std::isfinite(value)) {
      return Fail(error, std::string(field) + " is non-finite");
    }
    const double canonical_value = value == 0.0 ? 0.0 : value;
    uint64_t bits = 0;
    std::memcpy(&bits, &canonical_value, sizeof(bits));
    Uint64(bits);
    return true;
  }

  bool String(const std::string& value,
              const char* field,
              std::string* error) {
    if (!Size(value.size(), field, error)) {
      return false;
    }
    bytes_.insert(bytes_.end(), value.begin(), value.end());
    return true;
  }

  const std::vector<uint8_t>& Bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

bool WriteInt32(const int value,
                const char* field,
                CanonicalWriter* writer,
                std::string* error) {
  int32_t converted = 0;
  if (!IntToInt32(value, field, &converted, error)) {
    return false;
  }
  writer->Int32(converted);
  return true;
}

bool WriteAssociationRoute(const OnlineLidarAssociationRoute route,
                           CanonicalWriter* writer,
                           std::string* error) {
  switch (route) {
    case OnlineLidarAssociationRoute::PROJECTION:
      writer->Uint8(0);
      return true;
    case OnlineLidarAssociationRoute::KDTREE:
      writer->Uint8(1);
      return true;
  }
  return Fail(error, "canonical association route is invalid");
}

bool WriteCoordinateFrame(const lidar::LidarCoordinateFrame frame,
                          CanonicalWriter* writer,
                          std::string* error) {
  switch (frame) {
    case lidar::LidarCoordinateFrame::FASTLIO_WORLD:
      writer->Uint8(0);
      return true;
    case lidar::LidarCoordinateFrame::COLMAP_WORLD:
      writer->Uint8(1);
      return true;
  }
  return Fail(error, "canonical LiDAR coordinate frame is invalid");
}

bool WriteNormalScale(const lidar::LidarNormalScale scale,
                      CanonicalWriter* writer,
                      std::string* error) {
  switch (scale) {
    case lidar::LidarNormalScale::OUTER_0_15_M:
      writer->Uint8(0);
      return true;
    case lidar::LidarNormalScale::INNER_0_05_M:
      writer->Uint8(1);
      return true;
  }
  return Fail(error, "canonical LiDAR normal scale is invalid");
}

bool WriteLidarPointType(const LidarPointType type,
                         CanonicalWriter* writer,
                         std::string* error) {
  switch (type) {
    case LidarPointType::Proj:
      writer->Uint8(0);
      return true;
    case LidarPointType::Icp:
      writer->Uint8(1);
      return true;
    case LidarPointType::IcpGround:
      writer->Uint8(2);
      return true;
  }
  return Fail(error, "canonical LiDAR point type is invalid");
}

void WriteVoxelKey(const lidar::VoxelKey& key, CanonicalWriter* writer) {
  writer->Int64(key.x);
  writer->Int64(key.y);
  writer->Int64(key.z);
}

bool WritePlaneSample(const lidar::PlaneSample& plane,
                      CanonicalWriter* writer,
                      std::string* error) {
  WriteVoxelKey(plane.key, writer);
  if (!WriteCoordinateFrame(plane.frame, writer, error) ||
      !WriteNormalScale(plane.scale, writer, error)) {
    return false;
  }
  for (const float value : plane.point) {
    if (!writer->Float(value, "association.plane.point", error)) {
      return false;
    }
  }
  for (const float value : plane.normal) {
    if (!writer->Float(value, "association.plane.normal", error)) {
      return false;
    }
  }
  if (!writer->Float(
          plane.curvature, "association.plane.curvature", error)) {
    return false;
  }
  writer->Uint64(plane.normal_revision);
  writer->Uint64(plane.voxel_count);
  return true;
}

bool WriteProjectionOptions(const lidar::PcdProjectionOptions& options,
                            CanonicalWriter* writer,
                            std::string* error) {
  if (!writer->String(options.ba_pointcloud_path,
                      "projection_options.ba_pointcloud_path",
                      error) ||
      !writer->String(options.initial_mesh_depth_path,
                      "projection_options.initial_mesh_depth_path",
                      error) ||
      !writer->String(options.initial_mesh_depth_generator_path,
                      "projection_options.initial_mesh_depth_generator_path",
                      error) ||
      !writer->String(options.initial_mesh_path,
                      "projection_options.initial_mesh_path",
                      error) ||
      !writer->String(options.initial_mesh_depth_dataset_path,
                      "projection_options.initial_mesh_depth_dataset_path",
                      error) ||
      !writer->String(options.initial_mesh_depth_intrinsics_path,
                      "projection_options.initial_mesh_depth_intrinsics_path",
                      error) ||
      !writer->Double(options.initial_mesh_depth_fx,
                      "projection_options.initial_mesh_depth_fx",
                      error) ||
      !writer->Double(options.initial_mesh_depth_fy,
                      "projection_options.initial_mesh_depth_fy",
                      error) ||
      !writer->Double(options.initial_mesh_depth_cx,
                      "projection_options.initial_mesh_depth_cx",
                      error) ||
      !writer->Double(options.initial_mesh_depth_cy,
                      "projection_options.initial_mesh_depth_cy",
                      error) ||
      !writer->Double(options.initial_mesh_depth_pnp_max_error,
                      "projection_options.initial_mesh_depth_pnp_max_error",
                      error) ||
      !writer->Double(options.depth_image_scale,
                      "projection_options.depth_image_scale",
                      error)) {
    return false;
  }
  writer->Bool(options.if_save_depth_image);
  if (!writer->String(options.depth_image_folder,
                      "projection_options.depth_image_folder",
                      error) ||
      !writer->String(options.original_image_folder,
                      "projection_options.original_image_folder",
                      error)) {
    return false;
  }
  writer->Bool(options.if_save_lidar_frame);
  if (!writer->String(options.lidar_frame_folder,
                      "projection_options.lidar_frame_folder",
                      error) ||
      !WriteInt32(options.max_proj_scale,
                  "projection_options.max_proj_scale",
                  writer,
                  error) ||
      !WriteInt32(options.min_proj_scale,
                  "projection_options.min_proj_scale",
                  writer,
                  error) ||
      !writer->Double(options.min_proj_dist,
                      "projection_options.min_proj_dist",
                      error) ||
      !writer->Float(options.submap_length,
                     "projection_options.submap_length",
                     error) ||
      !writer->Float(options.submap_width,
                     "projection_options.submap_width",
                     error) ||
      !writer->Float(options.submap_height,
                     "projection_options.submap_height",
                     error) ||
      !writer->Float(options.choose_meter,
                     "projection_options.choose_meter",
                     error) ||
      !writer->Double(options.min_lidar_proj_dist,
                      "projection_options.min_lidar_proj_dist",
                      error)) {
    return false;
  }
  return true;
}

bool ValidateProjectionOptionsForCanonicalHash(
    const lidar::PcdProjectionOptions& options,
    std::string* error) {
  if (!std::isfinite(options.initial_mesh_depth_fx) ||
      !std::isfinite(options.initial_mesh_depth_fy) ||
      !std::isfinite(options.initial_mesh_depth_cx) ||
      !std::isfinite(options.initial_mesh_depth_cy) ||
      !std::isfinite(options.initial_mesh_depth_pnp_max_error) ||
      !std::isfinite(options.depth_image_scale) ||
      !std::isfinite(options.min_proj_dist) ||
      !std::isfinite(options.submap_length) ||
      !std::isfinite(options.submap_width) ||
      !std::isfinite(options.submap_height) ||
      !std::isfinite(options.choose_meter) ||
      !std::isfinite(options.min_lidar_proj_dist)) {
    return Fail(error,
                "PcdProjectionOptions contains a non-finite numeric value");
  }
  int32_t converted = 0;
  return IntToInt32(options.max_proj_scale,
                    "projection_options.max_proj_scale",
                    &converted,
                    error) &&
         IntToInt32(options.min_proj_scale,
                    "projection_options.min_proj_scale",
                    &converted,
                    error);
}

bool WriteAssociationOptions(const OnlineLidarAssociationOptions& options,
                             CanonicalWriter* writer,
                             std::string* error) {
  writer->Bool(options.local_lidar_kdtree_only);
  if (!WriteInt32(options.min_proj_num,
                  "options.min_proj_num",
                  writer,
                  error) ||
      !writer->Double(options.kdtree_max_search_range,
                      "options.kdtree_max_search_range",
                      error) ||
      !writer->Double(options.kdtree_min_search_range,
                      "options.kdtree_min_search_range",
                      error) ||
      !writer->Double(options.search_range_drop_speed,
                      "options.search_range_drop_speed",
                      error) ||
      !WriteInt32(options.ba_match_features_threshold,
                  "options.ba_match_features_threshold",
                  writer,
                  error)) {
    return false;
  }
  return true;
}

bool WriteProjectionAudit(const lidar::SnapshotProjectionAudit& audit,
                          CanonicalWriter* writer,
                          std::string* error) {
  writer->Uint64(audit.map_version);
  if (!writer->String(audit.snapshot_sha256,
                      "projection_audit.snapshot_sha256",
                      error) ||
      !writer->String(audit.geometry_sha256,
                      "projection_audit.geometry_sha256",
                      error) ||
      !writer->Size(audit.snapshot_voxel_count,
                    "projection_audit.snapshot_voxel_count",
                    error) ||
      !writer->Size(audit.snapshot_block_count,
                    "projection_audit.snapshot_block_count",
                    error)) {
    return false;
  }
  writer->Bool(audit.full_snapshot_due_to_distortion);
  for (const double value : audit.projection_aabb.min) {
    if (!writer->Double(
            value, "projection_audit.projection_aabb.min", error)) {
      return false;
    }
  }
  for (const double value : audit.projection_aabb.max) {
    if (!writer->Double(
            value, "projection_audit.projection_aabb.max", error)) {
      return false;
    }
  }
  if (!WriteCoordinateFrame(audit.projection_aabb.frame, writer, error)) {
    return false;
  }
  writer->Uint64(audit.input_feature_count);
  writer->Uint64(audit.in_image_feature_count);
  writer->Uint64(audit.aabb_visited_block_count);
  writer->Uint64(audit.candidate_plane_count);
  writer->Uint64(audit.positive_depth_hit_count);
  writer->Uint64(audit.axial_depth_hit_count);
  writer->Uint64(audit.pixel_hit_count);
  writer->Uint64(audit.coverage_pixel_visit_count);
  writer->Uint64(audit.feature_coverage_hit_count);
  writer->Uint64(audit.feature_pixel_hit_count);
  writer->Uint64(audit.feature_hit_count);
  return true;
}

bool WriteAssociationAudit(const OnlineLidarAssociationAudit& audit,
                           CanonicalWriter* writer,
                           std::string* error) {
  writer->Uint64(audit.attempt_id);
  writer->Uint32(audit.pass_index);
  writer->Uint32(audit.trigger_image_id);
  writer->Uint64(audit.map_version);
  writer->Uint64(audit.max_scan_index);
  if (!writer->String(
          audit.snapshot_sha256, "audit.snapshot_sha256", error) ||
      !writer->String(
          audit.geometry_sha256, "audit.geometry_sha256", error)) {
    return false;
  }
  writer->Uint64(audit.input_point_count);
  writer->Uint64(audit.selected_association_count);
  writer->Uint64(audit.skipped_point_count);
  writer->Uint64(audit.skipped_no_window_observation_count);
  writer->Uint64(audit.skipped_pair_threshold_point_count);
  writer->Uint64(audit.skipped_no_plane_count);
  writer->Uint64(audit.pair_threshold_skipped_observation_count);
  writer->Uint64(audit.projection_route_point_count);
  writer->Uint64(audit.kdtree_route_point_count);
  writer->Uint64(audit.projection_selected_association_count);
  writer->Uint64(audit.kdtree_selected_association_count);
  writer->Uint64(audit.projection_call_count);
  if (!writer->Size(audit.projection_calls.size(),
                    "audit.projection_calls.size",
                    error)) {
    return false;
  }
  for (const OnlineLidarProjectionCallAudit& call : audit.projection_calls) {
    writer->Uint32(call.image_id);
    if (!WriteProjectionAudit(call.projection, writer, error)) {
      return false;
    }
  }
  if (!writer->Size(audit.preliminary_selected_count_by_image.size(),
                    "audit.preliminary_selected_count_by_image.size",
                    error)) {
    return false;
  }
  for (const auto& image_count :
       audit.preliminary_selected_count_by_image) {
    writer->Uint32(image_count.first);
    writer->Uint64(image_count.second);
  }
  writer->Uint64(audit.trigger_preliminary_selected_count);
  return true;
}

bool WriteAssociation(const OnlineLidarAssociation& association,
                      CanonicalWriter* writer,
                      std::string* error) {
  writer->Uint64(association.association_id);
  writer->Uint64(association.attempt_id);
  writer->Uint32(association.pass_index);
  writer->Uint64(association.point3D_id);
  for (const double value : association.point3D_xyz) {
    if (!writer->Double(value, "association.point3D_xyz", error)) {
      return false;
    }
  }
  writer->Uint32(association.owner_image_id);
  writer->Uint32(association.owner_point2D_idx);
  writer->Uint64(association.map_version);
  writer->Uint64(association.max_scan_index);
  if (!writer->String(association.snapshot_sha256,
                      "association.snapshot_sha256",
                      error) ||
      !writer->String(association.geometry_sha256,
                      "association.geometry_sha256",
                      error) ||
      !WriteAssociationRoute(association.route, writer, error) ||
      !WritePlaneSample(association.plane, writer, error)) {
    return false;
  }
  WriteVoxelKey(association.plane_key, writer);
  for (const double value : association.plane_abcd) {
    if (!writer->Double(value, "association.plane_abcd", error)) {
      return false;
    }
  }
  writer->Bool(association.has_search_range);
  if (!writer->Double(
          association.search_range, "association.search_range", error) ||
      !WriteLidarPointType(
          association.lidar_point_type, writer, error) ||
      !writer->Double(association.projection_camera_distance,
                      "association.projection_camera_distance",
                      error) ||
      !writer->Double(association.projection_angle_score,
                      "association.projection_angle_score",
                      error)) {
    return false;
  }
  return true;
}

bool ValidatePlaneFiniteForCanonicalHash(const lidar::PlaneSample& plane,
                                         std::string* error) {
  for (const float value : plane.point) {
    if (!std::isfinite(value)) {
      return Fail(error, "association plane point is non-finite");
    }
  }
  for (const float value : plane.normal) {
    if (!std::isfinite(value)) {
      return Fail(error, "association plane normal is non-finite");
    }
  }
  if (!std::isfinite(plane.curvature)) {
    return Fail(error, "association plane curvature is non-finite");
  }
  return true;
}

bool ValidateFloatingPointForCanonicalHash(
    const OnlineLidarAssociationRequest& request,
    const OnlineLidarAssociationOutput& output,
    const OnlineLidarAssociationAudit& audit,
    std::string* error) {
  if (!ValidateProjectionOptionsForCanonicalHash(
          request.projection_options, error)) {
    return false;
  }
  for (const OnlineLidarProjectionCallAudit& call : audit.projection_calls) {
    for (const double value : call.projection.projection_aabb.min) {
      if (!std::isfinite(value)) {
        return Fail(error, "projection audit AABB minimum is non-finite");
      }
    }
    for (const double value : call.projection.projection_aabb.max) {
      if (!std::isfinite(value)) {
        return Fail(error, "projection audit AABB maximum is non-finite");
      }
    }
  }
  for (const OnlineLidarAssociation& association : output.associations) {
    for (const double value : association.point3D_xyz) {
      if (!std::isfinite(value)) {
        return Fail(error, "association Point3D XYZ is non-finite");
      }
    }
    if (!ValidatePlaneFiniteForCanonicalHash(association.plane, error)) {
      return false;
    }
    for (const double value : association.plane_abcd) {
      if (!std::isfinite(value)) {
        return Fail(error, "association plane equation is non-finite");
      }
    }
    if (!std::isfinite(association.search_range) ||
        !std::isfinite(association.projection_camera_distance) ||
        !std::isfinite(association.projection_angle_score)) {
      return Fail(error, "association score or range is non-finite");
    }
  }
  return true;
}

bool ComputeSha256Hex(const CanonicalWriter& writer,
                      std::string* sha256,
                      std::string* error) {
  if (sha256 == nullptr) {
    return Fail(error, "association SHA256 output is null");
  }
  const std::vector<uint8_t>& bytes = writer.Bytes();
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  const unsigned char* data =
      bytes.empty()
          ? nullptr
          : reinterpret_cast<const unsigned char*>(bytes.data());
  if (::SHA256(data, bytes.size(), digest.data()) == nullptr) {
    return Fail(error, "OpenSSL SHA256 failed for canonical associations");
  }

  static const char kHexDigits[] = "0123456789abcdef";
  std::string encoded(SHA256_DIGEST_LENGTH * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    encoded[2 * index] = kHexDigits[digest[index] >> 4];
    encoded[2 * index + 1] = kHexDigits[digest[index] & 0x0f];
  }
  *sha256 = std::move(encoded);
  return true;
}

void ResetBuildResults(OnlineLidarAssociationOutput* output,
                       OnlineLidarAssociationAudit* audit,
                       std::string* error) noexcept {
  if (output != nullptr) {
    output->associations.clear();
  }
  if (audit != nullptr) {
    audit->attempt_id = 0;
    audit->pass_index = 0;
    audit->trigger_image_id = kInvalidImageId;
    audit->map_version = 0;
    audit->max_scan_index = 0;
    audit->snapshot_sha256.clear();
    audit->geometry_sha256.clear();
    audit->input_point_count = 0;
    audit->selected_association_count = 0;
    audit->skipped_point_count = 0;
    audit->skipped_no_window_observation_count = 0;
    audit->skipped_pair_threshold_point_count = 0;
    audit->skipped_no_plane_count = 0;
    audit->pair_threshold_skipped_observation_count = 0;
    audit->projection_route_point_count = 0;
    audit->kdtree_route_point_count = 0;
    audit->projection_selected_association_count = 0;
    audit->kdtree_selected_association_count = 0;
    audit->projection_call_count = 0;
    audit->projection_calls.clear();
    audit->preliminary_selected_count_by_image.clear();
    audit->trigger_preliminary_selected_count = 0;
    audit->association_sha256.clear();
  }
  if (error != nullptr) {
    error->clear();
  }
}

bool ValidateOptions(const OnlineLidarAssociationOptions& options,
                     std::string* error) {
  if (options.min_proj_num < 0) {
    return Fail(error, "min_proj_num must be non-negative");
  }
  const size_t maximum_size = std::numeric_limits<size_t>::max();
  if (maximum_size < 3 ||
      static_cast<std::uintmax_t>(options.min_proj_num) >
          static_cast<std::uintmax_t>(maximum_size - 3)) {
    return Fail(error, "min_proj_num + 3 overflows size_t");
  }

  if (!std::isfinite(options.kdtree_min_search_range) ||
      options.kdtree_min_search_range < 0.0) {
    return Fail(error,
                "kdtree_min_search_range must be finite and non-negative");
  }
  if (!std::isfinite(options.kdtree_max_search_range) ||
      options.kdtree_max_search_range < 0.0) {
    return Fail(error,
                "kdtree_max_search_range must be finite and non-negative");
  }
  if (options.kdtree_min_search_range >
      options.kdtree_max_search_range) {
    return Fail(error,
                "kdtree_min_search_range exceeds kdtree_max_search_range");
  }
  const double maximum_query_distance =
      std::sqrt(std::numeric_limits<double>::max());
  if (options.kdtree_max_search_range > maximum_query_distance) {
    return Fail(error, "kdtree_max_search_range is too large");
  }
  if (!std::isfinite(options.search_range_drop_speed) ||
      options.search_range_drop_speed < 0.0) {
    return Fail(error,
                "search_range_drop_speed must be finite and non-negative");
  }
  if (options.ba_match_features_threshold < 0) {
    return Fail(error,
                "ba_match_features_threshold must be non-negative");
  }
  int32_t converted = 0;
  if (!IntToInt32(
          options.min_proj_num, "options.min_proj_num", &converted, error) ||
      !IntToInt32(options.ba_match_features_threshold,
                  "options.ba_match_features_threshold",
                  &converted,
                  error)) {
    return false;
  }
  return true;
}

bool ComputeSearchRange(const OnlineLidarAssociationOptions& options,
                        const int global_opt_num,
                        double* search_range,
                        std::string* error) {
  if (search_range == nullptr) {
    return Fail(error, "search range output is null");
  }
  if (global_opt_num < 0) {
    return Fail(error, "Point3D GlobalOptNum is negative");
  }
  const double dropped_range =
      static_cast<double>(global_opt_num) * options.search_range_drop_speed;
  if (!std::isfinite(dropped_range)) {
    return Fail(error, "KD search range drop is non-finite");
  }
  const double reduced_range =
      options.kdtree_max_search_range - dropped_range;
  if (!std::isfinite(reduced_range)) {
    return Fail(error, "KD reduced search range is non-finite");
  }
  const double computed_range =
      std::max(options.kdtree_min_search_range, reduced_range);
  if (!std::isfinite(computed_range) || computed_range < 0.0 ||
      computed_range > std::sqrt(std::numeric_limits<double>::max())) {
    return Fail(error, "KD search range is invalid");
  }
  *search_range = computed_range;
  return true;
}

bool ValidateKdPlane(const lidar::LidarMapSnapshot& snapshot,
                     const lidar::PlaneSample& plane,
                     std::string* error) {
  if (plane.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD) {
    return Fail(error, "nearest-plane result is not in COLMAP_WORLD");
  }
  if (plane.scale != lidar::LidarNormalScale::BA) {
    return Fail(error, "nearest-plane result does not use BA normal scale");
  }
  if (plane.voxel_count == 0) {
    return Fail(error, "nearest-plane result has zero voxel count");
  }
  if (plane.normal_revision == 0 ||
      plane.normal_revision > snapshot.Version()) {
    return Fail(error, "nearest-plane result has invalid normal revision");
  }
  if (!std::isfinite(plane.curvature)) {
    return Fail(error, "nearest-plane result has non-finite curvature");
  }

  double normal_squared_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(plane.point[axis]) ||
        !std::isfinite(plane.normal[axis])) {
      return Fail(error, "nearest-plane result contains non-finite values");
    }
    normal_squared_norm += static_cast<double>(plane.normal[axis]) *
                           static_cast<double>(plane.normal[axis]);
  }
  if (!std::isfinite(normal_squared_norm) ||
      !(normal_squared_norm > 0.0)) {
    return Fail(error, "nearest-plane result has a zero or invalid normal");
  }

  lidar::LidarVoxelRecord record;
  if (!snapshot.FindVoxel(plane.key, &record)) {
    return Fail(error, "nearest-plane result references a missing voxel key");
  }
  if (record.key != plane.key ||
      record.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD ||
      record.count != plane.voxel_count || record.centroid != plane.point ||
      !record.inner_normal.valid ||
      record.inner_normal.normal != plane.normal ||
      record.inner_normal.curvature != plane.curvature ||
      record.inner_normal.revision != plane.normal_revision) {
    return Fail(error, "nearest-plane result metadata does not match snapshot");
  }
  return true;
}

bool ValidateProjectionMatch(
    const lidar::LidarMapSnapshot& snapshot,
    const point3D_t expected_point3D_id,
    const point2D_t expected_point2D_idx,
    const point3D_t match_key,
    const lidar::SnapshotProjectionMatch& match,
    std::string* error) {
  if (match_key != expected_point3D_id ||
      match.point3D_id != expected_point3D_id) {
    return Fail(error, "snapshot projection returned the wrong point3D ID");
  }
  if (match.point2D_idx != expected_point2D_idx) {
    return Fail(error, "snapshot projection returned the wrong point2D index");
  }
  if (match.map_version != snapshot.Version()) {
    return Fail(error, "snapshot projection match has the wrong map version");
  }
  if (match.plane_key != match.plane.key) {
    return Fail(error,
                "snapshot projection match has inconsistent plane keys");
  }
  if (match.plane.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD) {
    return Fail(error, "snapshot projection plane is not in COLMAP_WORLD");
  }
  if (match.plane.scale != lidar::LidarNormalScale::PROJECTION) {
    return Fail(error,
                "snapshot projection plane does not use PROJECTION scale");
  }
  if (match.plane.voxel_count == 0) {
    return Fail(error, "snapshot projection plane has zero voxel count");
  }
  if (match.plane.normal_revision == 0 ||
      match.plane.normal_revision > snapshot.Version()) {
    return Fail(error, "snapshot projection plane has invalid revision");
  }
  if (!std::isfinite(match.plane.curvature)) {
    return Fail(error, "snapshot projection plane has non-finite curvature");
  }
  if (!std::isfinite(static_cast<double>(match.camera_distance)) ||
      match.camera_distance < 0.0f) {
    return Fail(error, "snapshot projection camera distance is invalid");
  }

  double normal_squared_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(match.plane.point[axis]) ||
        !std::isfinite(match.plane.normal[axis])) {
      return Fail(error,
                  "snapshot projection plane contains non-finite values");
    }
    normal_squared_norm +=
        static_cast<double>(match.plane.normal[axis]) *
        static_cast<double>(match.plane.normal[axis]);
  }
  if (!std::isfinite(normal_squared_norm) ||
      !(normal_squared_norm > 0.0)) {
    return Fail(error,
                "snapshot projection plane has a zero or invalid normal");
  }

  lidar::LidarVoxelRecord record;
  if (!snapshot.FindVoxel(match.plane_key, &record)) {
    return Fail(error,
                "snapshot projection plane references a missing voxel key");
  }
  if (record.key != match.plane_key ||
      record.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD ||
      record.count != match.plane.voxel_count ||
      record.centroid != match.plane.point || !record.outer_normal.valid ||
      record.outer_normal.normal != match.plane.normal ||
      record.outer_normal.curvature != match.plane.curvature ||
      record.outer_normal.revision != match.plane.normal_revision) {
    return Fail(error,
                "snapshot projection plane metadata does not match snapshot");
  }
  return true;
}

bool ComputeProjectionAngleScore(
    const std::array<double, 3>& point_xyz,
    const lidar::PlaneSample& plane,
    double* score,
    std::string* error) {
  if (score == nullptr) {
    return Fail(error, "projection angle score output is null");
  }
  const Eigen::Vector3d point(point_xyz[0], point_xyz[1], point_xyz[2]);
  const Eigen::Vector3d plane_point(
      static_cast<double>(plane.point[0]),
      static_cast<double>(plane.point[1]),
      static_cast<double>(plane.point[2]));
  const Eigen::Vector3d normal(
      static_cast<double>(plane.normal[0]),
      static_cast<double>(plane.normal[1]),
      static_cast<double>(plane.normal[2]));
  const Eigen::Vector3d vector = point - plane_point;
  const double vector_norm = vector.norm();
  const double normal_norm = normal.norm();
  if (!vector.allFinite() || !normal.allFinite() ||
      !std::isfinite(vector_norm) || !std::isfinite(normal_norm)) {
    return Fail(error, "projection angle score input is non-finite");
  }
  if (normal_norm == 0.0) {
    return Fail(error, "projection angle score normal is zero");
  }
  if (vector_norm == 0.0) {
    *score = 0.0;
    return true;
  }

  const double dot_product = vector.dot(normal);
  const double denominator = vector_norm * normal_norm;
  const double computed_score = std::abs(dot_product) / denominator;
  if (!std::isfinite(dot_product) || !std::isfinite(denominator) ||
      !std::isfinite(computed_score)) {
    return Fail(error, "projection angle score is non-finite");
  }
  *score = computed_score;
  return true;
}

struct WindowObservation {
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;
};

struct ProjectionObservationGate {
  WindowObservation observation;
  bool trigger_bypass = false;
  bool pair_exists = false;
  size_t num_total_corrs = 0;
  bool eligible = false;
};

struct PreparedPoint {
  point3D_t point3D_id = kInvalidPoint3DId;
  std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
  size_t full_track_length = 0;
  int global_opt_num = 0;
  OnlineLidarAssociationRoute route =
      OnlineLidarAssociationRoute::PROJECTION;
  std::vector<WindowObservation> window_observations;
  std::vector<ProjectionObservationGate> projection_observation_gates;
  bool has_owner = false;
  image_t owner_image_id = kInvalidImageId;
  point2D_t owner_point2D_idx = kInvalidPoint2DIdx;
  double search_range = 0.0;
};

struct ProjectionEligibleObservation {
  size_t prepared_point_index = 0;
  point3D_t point3D_id = kInvalidPoint3DId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;
};

struct ProjectionImageWork {
  image_t image_id = kInvalidImageId;
  std::vector<ProjectionEligibleObservation> eligible_observations;
};

struct FrozenProjectionBinding {
  point2D_t point2D_idx = kInvalidPoint2DIdx;
  point3D_t point3D_id = kInvalidPoint3DId;
  std::array<double, 2> xy{{0.0, 0.0}};
};

struct FrozenProjectionInput {
  image_t image_id = kInvalidImageId;
  std::string image_name;
  std::array<double, 4> qvec{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> tvec{{0.0, 0.0, 0.0}};
  camera_t camera_id = kInvalidCameraId;
  int camera_model_id = kInvalidCameraModelId;
  size_t camera_width = 0;
  size_t camera_height = 0;
  std::vector<double> camera_params;
  bool camera_is_undistorted = false;
  size_t original_point2D_slot_count = 0;
  std::vector<FrozenProjectionBinding> retained_bindings;
};

bool FreezeProjectionInput(const Image& projection_image,
                           const Camera& projection_camera,
                           FrozenProjectionInput* frozen,
                           std::string* error) {
  if (frozen == nullptr) {
    return Fail(error, "frozen projection input output is null");
  }
  if (projection_image.ImageId() == kInvalidImageId) {
    return Fail(error, "projection input has an invalid image ID");
  }
  if (!projection_image.HasCamera() ||
      projection_image.CameraId() == kInvalidCameraId) {
    return Fail(error, "projection input has an invalid camera ID");
  }
  if (projection_image.Points2D().size() >
      static_cast<size_t>(kInvalidPoint2DIdx)) {
    return Fail(error, "projection input has too many point2D slots");
  }
  if (!ExistsCameraModelWithId(projection_camera.ModelId()) ||
      !projection_camera.VerifyParams()) {
    return Fail(error,
                "projection input camera model or parameters are invalid");
  }

  FrozenProjectionInput local;
  local.image_id = projection_image.ImageId();
  local.image_name = projection_image.Name();
  for (size_t index = 0; index < local.qvec.size(); ++index) {
    local.qvec[index] = projection_image.Qvec(index);
    if (!std::isfinite(local.qvec[index])) {
      return Fail(error, "projection input image quaternion is non-finite");
    }
  }
  for (size_t index = 0; index < local.tvec.size(); ++index) {
    local.tvec[index] = projection_image.Tvec(index);
    if (!std::isfinite(local.tvec[index])) {
      return Fail(error, "projection input image translation is non-finite");
    }
  }
  local.camera_id = projection_image.CameraId();
  local.camera_model_id = projection_camera.ModelId();
  local.camera_width = projection_camera.Width();
  local.camera_height = projection_camera.Height();
  local.camera_params = projection_camera.Params();
  for (const double parameter : local.camera_params) {
    if (!std::isfinite(parameter)) {
      return Fail(error, "projection input camera parameter is non-finite");
    }
  }
  local.camera_is_undistorted = projection_camera.IsUndistorted();
  local.original_point2D_slot_count = projection_image.Points2D().size();
  local.retained_bindings.reserve(projection_image.Points2D().size());
  for (size_t point2D_index = 0;
       point2D_index < projection_image.Points2D().size();
       ++point2D_index) {
    const class Point2D& point2D =
        projection_image.Points2D()[point2D_index];
    if (!point2D.HasPoint3D()) {
      continue;
    }
    if (!point2D.XY().allFinite()) {
      return Fail(error, "projection input retained observation is non-finite");
    }
    FrozenProjectionBinding binding;
    binding.point2D_idx = static_cast<point2D_t>(point2D_index);
    binding.point3D_id = point2D.Point3DId();
    binding.xy = {{point2D.X(), point2D.Y()}};
    local.retained_bindings.push_back(binding);
  }
  *frozen = std::move(local);
  return true;
}

struct ProjectionCandidate {
  image_t owner_image_id = kInvalidImageId;
  point2D_t owner_point2D_idx = kInvalidPoint2DIdx;
  lidar::SnapshotProjectionMatch match;
  double angle_score = 0.0;
};

bool ProjectionCandidateLess(const ProjectionCandidate& first,
                             const ProjectionCandidate& second) {
  if (first.angle_score != second.angle_score) {
    return first.angle_score < second.angle_score;
  }
  if (first.owner_image_id != second.owner_image_id) {
    return first.owner_image_id < second.owner_image_id;
  }
  if (first.owner_point2D_idx != second.owner_point2D_idx) {
    return first.owner_point2D_idx < second.owner_point2D_idx;
  }
  return first.match.plane_key < second.match.plane_key;
}

bool PreparedPointLess(const PreparedPoint& first,
                       const PreparedPoint& second) {
  return first.point3D_id < second.point3D_id;
}

bool PreparePoint(const Reconstruction& reconstruction,
                  const OnlineLidarAssociationRequest& request,
                  const std::set<image_t>& window_image_ids,
                  const std::vector<image_t>& owner_priority,
                  const point3D_t point3D_id,
                  PreparedPoint* prepared,
                  std::string* error) {
  if (prepared == nullptr) {
    return Fail(error, "prepared point output is null");
  }
  const Point3D& point3D = reconstruction.Point3D(point3D_id);
  const Eigen::Vector3d& xyz = point3D.XYZ();
  if (!xyz.allFinite()) {
    return Fail(error, "Point3D XYZ is non-finite");
  }

  PreparedPoint local;
  local.point3D_id = point3D_id;
  local.xyz = {{xyz.x(), xyz.y(), xyz.z()}};
  local.full_track_length = point3D.Track().Length();
  local.global_opt_num = point3D.GlobalOptNum();
  std::string route_error;
  if (!SelectOnlineLidarAssociationRoute(request.options,
                                         local.full_track_length,
                                         &local.route,
                                         &route_error)) {
    return Fail(error, "cannot select association route: " + route_error);
  }

  std::map<image_t, point2D_t> window_observations;
  std::set<std::pair<image_t, point2D_t>> unique_observations;
  for (const TrackElement& element : point3D.Track().Elements()) {
    if (window_image_ids.count(element.image_id) == 0) {
      continue;
    }
    if (!unique_observations
             .emplace(element.image_id, element.point2D_idx).second) {
      return Fail(error,
                  "Point3D track contains the same window observation twice");
    }
    if (!reconstruction.ExistsImage(element.image_id)) {
      return Fail(error, "Point3D track references a missing window image");
    }
    const Image& image = reconstruction.Image(element.image_id);
    if (element.point2D_idx == kInvalidPoint2DIdx ||
        element.point2D_idx >= image.NumPoints2D()) {
      return Fail(error, "Point3D track has an invalid point2D index");
    }
    const class Point2D& point2D = image.Point2D(element.point2D_idx);
    if (!point2D.HasPoint3D() || point2D.Point3DId() != point3D_id) {
      return Fail(error, "Point3D track observation does not point back");
    }
    const auto inserted =
        window_observations.emplace(element.image_id, element.point2D_idx);
    if (!inserted.second) {
      inserted.first->second =
          std::min(inserted.first->second, element.point2D_idx);
    }
  }

  local.window_observations.reserve(window_observations.size());
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    const auto observation = window_observations.find(image_id);
    if (observation == window_observations.end()) {
      continue;
    }
    WindowObservation ordered_observation;
    ordered_observation.image_id = observation->first;
    ordered_observation.point2D_idx = observation->second;
    local.window_observations.push_back(ordered_observation);
  }

  for (const image_t image_id : owner_priority) {
    const auto observation = window_observations.find(image_id);
    if (observation == window_observations.end()) {
      continue;
    }
    local.has_owner = true;
    local.owner_image_id = observation->first;
    local.owner_point2D_idx = observation->second;
    break;
  }

  if (local.route == OnlineLidarAssociationRoute::KDTREE &&
      !ComputeSearchRange(request.options,
                          local.global_opt_num,
                          &local.search_range,
                          error)) {
    return false;
  }
  *prepared = std::move(local);
  return true;
}

void WriteWindowObservation(const WindowObservation& observation,
                            CanonicalWriter* writer) {
  writer->Uint32(observation.image_id);
  writer->Uint32(observation.point2D_idx);
}

bool WritePreparedPointInputs(
    const OnlineLidarAssociationRequest& request,
    const std::set<point3D_t>& sorted_point3D_ids,
    const std::vector<PreparedPoint>& prepared_points,
    CanonicalWriter* writer,
    std::string* error) {
  if (prepared_points.size() != sorted_point3D_ids.size()) {
    return Fail(error, "prepared Point3D count does not match request");
  }
  if (!writer->String(
          "prepared_points", "canonical prepared points tag", error) ||
      !writer->Size(prepared_points.size(),
                    "prepared_points.size",
                    error)) {
    return false;
  }

  auto expected_point3D_id = sorted_point3D_ids.begin();
  const size_t pair_threshold =
      static_cast<size_t>(request.options.ba_match_features_threshold);
  for (const PreparedPoint& prepared : prepared_points) {
    if (expected_point3D_id == sorted_point3D_ids.end() ||
        prepared.point3D_id != *expected_point3D_id) {
      return Fail(error, "prepared Point3D records are not canonical");
    }
    ++expected_point3D_id;

    writer->Uint64(prepared.point3D_id);
    for (const double value : prepared.xyz) {
      if (!writer->Double(value, "prepared_point.xyz", error)) {
        return false;
      }
    }
    if (!writer->Size(prepared.full_track_length,
                      "prepared_point.full_track_length",
                      error) ||
        !WriteInt32(prepared.global_opt_num,
                    "prepared_point.global_opt_num",
                    writer,
                    error) ||
        !WriteAssociationRoute(prepared.route, writer, error) ||
        !writer->Size(prepared.window_observations.size(),
                      "prepared_point.window_observations.size",
                      error)) {
      return false;
    }
    if (prepared.window_observations.size() > prepared.full_track_length) {
      return Fail(error, "prepared window observation count exceeds track");
    }

    size_t next_frozen_position = 0;
    for (const WindowObservation& observation :
         prepared.window_observations) {
      while (next_frozen_position <
                 request.ordered_frozen_image_ids.size() &&
             request.ordered_frozen_image_ids[next_frozen_position] !=
                 observation.image_id) {
        ++next_frozen_position;
      }
      if (next_frozen_position ==
              request.ordered_frozen_image_ids.size() ||
          observation.point2D_idx == kInvalidPoint2DIdx) {
        return Fail(error,
                    "prepared window observations are not in frozen order");
      }
      ++next_frozen_position;
      WriteWindowObservation(observation, writer);
    }

    if (prepared.route == OnlineLidarAssociationRoute::PROJECTION) {
      if (prepared.projection_observation_gates.size() !=
          prepared.window_observations.size()) {
        return Fail(error,
                    "prepared projection gates do not match observations");
      }
      if (!writer->Size(prepared.projection_observation_gates.size(),
                        "prepared_point.projection_gates.size",
                        error)) {
        return false;
      }
      for (size_t index = 0;
           index < prepared.projection_observation_gates.size();
           ++index) {
        const ProjectionObservationGate& gate =
            prepared.projection_observation_gates[index];
        const WindowObservation& observation =
            prepared.window_observations[index];
        if (gate.observation.image_id != observation.image_id ||
            gate.observation.point2D_idx != observation.point2D_idx) {
          return Fail(error,
                      "prepared projection gate observation is inconsistent");
        }
        const bool expected_trigger_bypass =
            observation.image_id == request.trigger_image_id;
        if (gate.trigger_bypass != expected_trigger_bypass ||
            (gate.trigger_bypass && gate.pair_exists) ||
            (!gate.pair_exists && gate.num_total_corrs != 0)) {
          return Fail(error, "prepared projection gate input is inconsistent");
        }
        const bool expected_eligible =
            gate.trigger_bypass || !gate.pair_exists ||
            gate.num_total_corrs > pair_threshold;
        if (gate.eligible != expected_eligible) {
          return Fail(error,
                      "prepared projection gate result is inconsistent");
        }

        WriteWindowObservation(gate.observation, writer);
        writer->Bool(gate.trigger_bypass);
        writer->Bool(gate.pair_exists);
        if (gate.pair_exists &&
            !writer->Size(gate.num_total_corrs,
                          "prepared_point.projection_gate.num_total_corrs",
                          error)) {
          return false;
        }
        writer->Bool(gate.eligible);
      }
      continue;
    }

    if (prepared.route != OnlineLidarAssociationRoute::KDTREE) {
      return Fail(error, "prepared association route is invalid");
    }
    if (!prepared.projection_observation_gates.empty()) {
      return Fail(error, "prepared KD point has projection gates");
    }
    const WindowObservation* expected_owner = nullptr;
    for (const WindowObservation& observation :
         prepared.window_observations) {
      if (observation.image_id == request.trigger_image_id) {
        expected_owner = &observation;
        break;
      }
    }
    if (expected_owner == nullptr &&
        !prepared.window_observations.empty()) {
      expected_owner = &prepared.window_observations.front();
    }
    if (prepared.has_owner != (expected_owner != nullptr) ||
        (expected_owner != nullptr &&
         (prepared.owner_image_id != expected_owner->image_id ||
          prepared.owner_point2D_idx != expected_owner->point2D_idx))) {
      return Fail(error, "prepared KD owner is inconsistent");
    }
    writer->Bool(prepared.has_owner);
    if (prepared.has_owner) {
      writer->Uint32(prepared.owner_image_id);
      writer->Uint32(prepared.owner_point2D_idx);
    }
    if (!writer->Double(
            prepared.search_range, "prepared_point.search_range", error)) {
      return false;
    }
  }
  return expected_point3D_id == sorted_point3D_ids.end();
}

bool WriteFrozenProjectionInputs(
    const OnlineLidarAssociationRequest& request,
    const std::vector<FrozenProjectionInput>& frozen_projection_inputs,
    const OnlineLidarAssociationAudit& audit,
    CanonicalWriter* writer,
    std::string* error) {
  if (frozen_projection_inputs.size() != audit.projection_calls.size()) {
    return Fail(error,
                "frozen projection input count does not match actual calls");
  }
  if (!writer->String("projection_inputs",
                      "canonical projection inputs tag",
                      error) ||
      !writer->Size(frozen_projection_inputs.size(),
                    "projection_inputs.size",
                    error)) {
    return false;
  }

  size_t next_frozen_position = 0;
  for (size_t input_index = 0;
       input_index < frozen_projection_inputs.size();
       ++input_index) {
    const FrozenProjectionInput& input =
        frozen_projection_inputs[input_index];
    if (input.image_id != audit.projection_calls[input_index].image_id) {
      return Fail(error,
                  "frozen projection input does not match actual call");
    }
    while (next_frozen_position <
               request.ordered_frozen_image_ids.size() &&
           request.ordered_frozen_image_ids[next_frozen_position] !=
               input.image_id) {
      ++next_frozen_position;
    }
    if (next_frozen_position == request.ordered_frozen_image_ids.size()) {
      return Fail(error,
                  "frozen projection inputs are not in frozen window order");
    }
    ++next_frozen_position;
    if (input.image_id == kInvalidImageId ||
        input.camera_id == kInvalidCameraId ||
        input.camera_model_id == kInvalidCameraModelId ||
        input.camera_width == 0 || input.camera_height == 0 ||
        input.original_point2D_slot_count >
            static_cast<size_t>(kInvalidPoint2DIdx) ||
        input.retained_bindings.size() >
            input.original_point2D_slot_count) {
      return Fail(error, "frozen projection input is invalid");
    }

    writer->Uint32(input.image_id);
    if (!writer->String(
            input.image_name, "projection_input.image_name", error)) {
      return false;
    }
    for (const double value : input.qvec) {
      if (!writer->Double(value, "projection_input.qvec", error)) {
        return false;
      }
    }
    for (const double value : input.tvec) {
      if (!writer->Double(value, "projection_input.tvec", error)) {
        return false;
      }
    }
    writer->Uint32(input.camera_id);
    if (!WriteInt32(input.camera_model_id,
                    "projection_input.camera_model_id",
                    writer,
                    error) ||
        !writer->Size(
            input.camera_width, "projection_input.camera_width", error) ||
        !writer->Size(
            input.camera_height, "projection_input.camera_height", error) ||
        !writer->Size(input.camera_params.size(),
                      "projection_input.camera_params.size",
                      error)) {
      return false;
    }
    for (const double parameter : input.camera_params) {
      if (!writer->Double(
              parameter, "projection_input.camera_param", error)) {
        return false;
      }
    }
    writer->Bool(input.camera_is_undistorted);
    if (!writer->Size(input.original_point2D_slot_count,
                      "projection_input.original_point2D_slot_count",
                      error)) {
      return false;
    }

    uint64_t retained_binding_count = 0;
    if (!SizeToUint64(input.retained_bindings.size(),
                      "projection_input.retained_bindings.size",
                      &retained_binding_count,
                      error)) {
      return false;
    }
    if (retained_binding_count !=
        audit.projection_calls[input_index].projection.input_feature_count) {
      return Fail(error,
                  "frozen projection bindings do not match actual call");
    }
    writer->Uint64(retained_binding_count);

    bool has_previous_point2D_idx = false;
    point2D_t previous_point2D_idx = kInvalidPoint2DIdx;
    std::set<point3D_t> retained_point3D_ids;
    for (const FrozenProjectionBinding& binding : input.retained_bindings) {
      if (binding.point2D_idx == kInvalidPoint2DIdx ||
          binding.point2D_idx >= input.original_point2D_slot_count ||
          binding.point3D_id == kInvalidPoint3DId ||
          (has_previous_point2D_idx &&
           binding.point2D_idx <= previous_point2D_idx) ||
          !retained_point3D_ids.insert(binding.point3D_id).second) {
        return Fail(error,
                    "frozen projection bindings are not canonical");
      }
      has_previous_point2D_idx = true;
      previous_point2D_idx = binding.point2D_idx;
      writer->Uint32(binding.point2D_idx);
      writer->Uint64(binding.point3D_id);
      for (const double value : binding.xy) {
        if (!writer->Double(value, "projection_input.binding.xy", error)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool AssociationLess(const OnlineLidarAssociation& first,
                     const OnlineLidarAssociation& second) {
  if (first.point3D_id != second.point3D_id) {
    return first.point3D_id < second.point3D_id;
  }
  if (first.route != second.route) {
    return static_cast<uint8_t>(first.route) <
           static_cast<uint8_t>(second.route);
  }
  if (first.owner_image_id != second.owner_image_id) {
    return first.owner_image_id < second.owner_image_id;
  }
  if (first.owner_point2D_idx != second.owner_point2D_idx) {
    return first.owner_point2D_idx < second.owner_point2D_idx;
  }
  return first.plane_key < second.plane_key;
}

bool AddUint64(const uint64_t first,
               const uint64_t second,
               const char* field,
               uint64_t* sum,
               std::string* error) {
  if (sum == nullptr) {
    return Fail(error, "audit sum output is null");
  }
  if (first > std::numeric_limits<uint64_t>::max() - second) {
    return Fail(error, std::string(field) + " overflows uint64_t");
  }
  *sum = first + second;
  return true;
}

bool IncrementUint64(uint64_t* value,
                     const char* field,
                     std::string* error) {
  if (value == nullptr) {
    return Fail(error, "audit counter is null");
  }
  if (*value == std::numeric_limits<uint64_t>::max()) {
    return Fail(error, std::string(field) + " overflows uint64_t");
  }
  ++(*value);
  return true;
}

bool RequireUint64Sum(const uint64_t first,
                      const uint64_t second,
                      const uint64_t expected,
                      const char* field,
                      std::string* error) {
  uint64_t sum = 0;
  if (!AddUint64(first, second, field, &sum, error)) {
    return false;
  }
  if (sum != expected) {
    return Fail(error, std::string(field) + " invariant failed");
  }
  return true;
}

bool ValidateAssociationAuditInvariants(
    const OnlineLidarAssociationRequest& request,
    const std::set<point3D_t>& sorted_point3D_ids,
    const OnlineLidarAssociationOutput& output,
    const OnlineLidarAssociationAudit& audit,
    std::string* error) {
  uint64_t request_point_count = 0;
  uint64_t selected_count = 0;
  uint64_t projection_call_count = 0;
  if (!SizeToUint64(request.point3D_ids.size(),
                    "request.point3D_ids.size",
                    &request_point_count,
                    error) ||
      !SizeToUint64(output.associations.size(),
                    "output.associations.size",
                    &selected_count,
                    error) ||
      !SizeToUint64(audit.projection_calls.size(),
                    "audit.projection_calls.size",
                    &projection_call_count,
                    error)) {
    return false;
  }
  if (audit.attempt_id != request.attempt_id ||
      audit.pass_index != request.pass_index ||
      audit.trigger_image_id != request.trigger_image_id ||
      audit.map_version != request.expected_map_version ||
      audit.max_scan_index != request.expected_max_scan_index ||
      audit.snapshot_sha256 != request.expected_snapshot_sha256 ||
      audit.geometry_sha256 != request.expected_geometry_sha256) {
    return Fail(error, "association audit identity does not match request");
  }
  if (audit.input_point_count != request_point_count ||
      sorted_point3D_ids.size() != request.point3D_ids.size()) {
    return Fail(error, "association audit input count invariant failed");
  }
  if (audit.selected_association_count != selected_count) {
    return Fail(error, "association audit selected count invariant failed");
  }
  if (!RequireUint64Sum(audit.projection_route_point_count,
                        audit.kdtree_route_point_count,
                        audit.input_point_count,
                        "association route count",
                        error) ||
      !RequireUint64Sum(audit.selected_association_count,
                        audit.skipped_point_count,
                        audit.input_point_count,
                        "association selected and skipped count",
                        error) ||
      !RequireUint64Sum(audit.projection_selected_association_count,
                        audit.kdtree_selected_association_count,
                        audit.selected_association_count,
                        "association selected route count",
                        error)) {
    return false;
  }

  uint64_t known_skipped_count = 0;
  if (!AddUint64(audit.skipped_no_window_observation_count,
                 audit.skipped_pair_threshold_point_count,
                 "association categorized skipped count",
                 &known_skipped_count,
                 error) ||
      !RequireUint64Sum(known_skipped_count,
                        audit.skipped_no_plane_count,
                        audit.skipped_point_count,
                        "association categorized skipped count",
                        error)) {
    return false;
  }
  if (audit.projection_call_count != projection_call_count) {
    return Fail(error, "association projection call count invariant failed");
  }

  if (audit.preliminary_selected_count_by_image.size() !=
      request.ordered_frozen_image_ids.size()) {
    return Fail(error, "association per-image count map has wrong size");
  }
  std::map<image_t, uint64_t> actual_selected_count_by_image;
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    if (audit.preliminary_selected_count_by_image.count(image_id) != 1) {
      return Fail(error,
                  "association per-image count map does not match window");
    }
    actual_selected_count_by_image.emplace(image_id, 0);
  }

  uint64_t per_image_selected_count = 0;
  for (const auto& image_count :
       audit.preliminary_selected_count_by_image) {
    if (!AddUint64(per_image_selected_count,
                   image_count.second,
                   "association per-image selected count",
                   &per_image_selected_count,
                   error)) {
      return false;
    }
  }
  if (per_image_selected_count != audit.selected_association_count) {
    return Fail(error, "association per-image selected count invariant failed");
  }
  const auto trigger_count =
      audit.preliminary_selected_count_by_image.find(
          request.trigger_image_id);
  if (trigger_count == audit.preliminary_selected_count_by_image.end() ||
      audit.trigger_preliminary_selected_count != trigger_count->second) {
    return Fail(error, "association trigger count invariant failed");
  }

  std::set<image_t> projection_call_images;
  size_t next_frozen_position = 0;
  for (const OnlineLidarProjectionCallAudit& call : audit.projection_calls) {
    if (!projection_call_images.insert(call.image_id).second) {
      return Fail(error, "association projection call image is duplicated");
    }
    while (next_frozen_position < request.ordered_frozen_image_ids.size() &&
           request.ordered_frozen_image_ids[next_frozen_position] !=
               call.image_id) {
      ++next_frozen_position;
    }
    if (next_frozen_position == request.ordered_frozen_image_ids.size()) {
      return Fail(error,
                  "association projection calls are not in frozen order");
    }
    ++next_frozen_position;
    if (call.projection.map_version != audit.map_version ||
        call.projection.snapshot_sha256 != audit.snapshot_sha256 ||
        call.projection.geometry_sha256 != audit.geometry_sha256) {
      return Fail(error,
                  "association projection call identity is inconsistent");
    }
  }

  uint64_t actual_projection_selected_count = 0;
  uint64_t actual_kdtree_selected_count = 0;
  std::set<point3D_t> associated_point3D_ids;
  for (size_t index = 0; index < output.associations.size(); ++index) {
    const OnlineLidarAssociation& association = output.associations[index];
    uint64_t expected_association_id = 0;
    if (!SizeToUint64(index,
                      "association index",
                      &expected_association_id,
                      error)) {
      return false;
    }
    if (association.association_id != expected_association_id) {
      return Fail(error, "association IDs are not canonical");
    }
    if (index > 0 &&
        AssociationLess(association, output.associations[index - 1])) {
      return Fail(error, "associations are not in canonical order");
    }
    if (sorted_point3D_ids.count(association.point3D_id) != 1 ||
        !associated_point3D_ids.insert(association.point3D_id).second) {
      return Fail(error, "association Point3D IDs are invalid or duplicated");
    }
    if (association.attempt_id != audit.attempt_id ||
        association.pass_index != audit.pass_index ||
        association.map_version != audit.map_version ||
        association.max_scan_index != audit.max_scan_index ||
        association.snapshot_sha256 != audit.snapshot_sha256 ||
        association.geometry_sha256 != audit.geometry_sha256 ||
        association.plane_key != association.plane.key) {
      return Fail(error, "association identity or plane key is inconsistent");
    }
    auto owner_count =
        actual_selected_count_by_image.find(association.owner_image_id);
    if (owner_count == actual_selected_count_by_image.end()) {
      return Fail(error, "association owner is outside frozen window");
    }
    if (owner_count->second == std::numeric_limits<uint64_t>::max()) {
      return Fail(error, "association owner count overflows uint64_t");
    }
    ++owner_count->second;

    switch (association.route) {
      case OnlineLidarAssociationRoute::PROJECTION:
        if (actual_projection_selected_count ==
            std::numeric_limits<uint64_t>::max()) {
          return Fail(error,
                      "projection association count overflows uint64_t");
        }
        ++actual_projection_selected_count;
        break;
      case OnlineLidarAssociationRoute::KDTREE:
        if (actual_kdtree_selected_count ==
            std::numeric_limits<uint64_t>::max()) {
          return Fail(error, "KD association count overflows uint64_t");
        }
        ++actual_kdtree_selected_count;
        break;
      default:
        return Fail(error, "association route is invalid");
    }
  }
  if (actual_projection_selected_count !=
          audit.projection_selected_association_count ||
      actual_kdtree_selected_count !=
          audit.kdtree_selected_association_count ||
      actual_selected_count_by_image !=
          audit.preliminary_selected_count_by_image) {
    return Fail(error, "association selected audit data is inconsistent");
  }
  return true;
}

bool BuildCanonicalAssociationSha256(
    const OnlineLidarAssociationRequest& request,
    const std::set<point3D_t>& sorted_point3D_ids,
    const std::vector<PreparedPoint>& prepared_points,
    const std::vector<FrozenProjectionInput>& frozen_projection_inputs,
    const OnlineLidarAssociationOutput& output,
    const OnlineLidarAssociationAudit& audit,
    std::string* association_sha256,
    std::string* error) {
  if (association_sha256 == nullptr) {
    return Fail(error, "association SHA256 output is null");
  }
  association_sha256->clear();
  if (!ValidateAssociationAuditInvariants(
          request, sorted_point3D_ids, output, audit, error) ||
      !ValidateOptions(request.options, error) ||
      !ValidateFloatingPointForCanonicalHash(
          request, output, audit, error)) {
    return false;
  }

  CanonicalWriter writer;
  if (!writer.String(kAssociationCanonicalSchema,
                     "canonical schema tag",
                     error) ||
      !writer.String("request", "canonical request tag", error)) {
    return false;
  }
  writer.Uint64(request.attempt_id);
  writer.Uint32(request.pass_index);
  writer.Uint32(request.trigger_image_id);
  if (!writer.Size(request.ordered_frozen_image_ids.size(),
                   "request.ordered_frozen_image_ids.size",
                   error)) {
    return false;
  }
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    writer.Uint32(image_id);
  }
  if (!WritePreparedPointInputs(request,
                                sorted_point3D_ids,
                                prepared_points,
                                &writer,
                                error) ||
      !WriteFrozenProjectionInputs(request,
                                   frozen_projection_inputs,
                                   audit,
                                   &writer,
                                   error)) {
    return false;
  }
  writer.Uint64(request.expected_map_version);
  writer.Uint64(request.expected_max_scan_index);
  if (!writer.String(request.expected_snapshot_sha256,
                     "request.expected_snapshot_sha256",
                     error) ||
      !writer.String(request.expected_geometry_sha256,
                     "request.expected_geometry_sha256",
                     error) ||
      !WriteAssociationOptions(request.options, &writer, error) ||
      !WriteProjectionOptions(request.projection_options, &writer, error) ||
      !writer.String("audit", "canonical audit tag", error) ||
      !WriteAssociationAudit(audit, &writer, error) ||
      !writer.String(
          "associations", "canonical associations tag", error) ||
      !writer.Size(output.associations.size(),
                   "output.associations.size",
                   error)) {
    return false;
  }
  for (const OnlineLidarAssociation& association : output.associations) {
    if (!WriteAssociation(association, &writer, error)) {
      return false;
    }
  }
  return ComputeSha256Hex(writer, association_sha256, error);
}

}  // namespace

bool SelectOnlineLidarAssociationRoute(
    const OnlineLidarAssociationOptions& options,
    const size_t track_length,
    OnlineLidarAssociationRoute* route,
    std::string* error) noexcept {
  try {
    if (error != nullptr) {
      error->clear();
    }
    if (error == nullptr) {
      return false;
    }
    if (route == nullptr) {
      return Fail(error, "association route output is null");
    }
    if (!ValidateOptions(options, error)) {
      return false;
    }
    const size_t projection_track_limit =
        static_cast<size_t>(options.min_proj_num) + 3;
    const OnlineLidarAssociationRoute selected_route =
        !options.local_lidar_kdtree_only &&
                track_length < projection_track_limit
            ? OnlineLidarAssociationRoute::PROJECTION
            : OnlineLidarAssociationRoute::KDTREE;
    *route = selected_route;
    return true;
  } catch (const std::exception& exception) {
    return FailWithException(error,
                             "online LiDAR association route selection threw",
                             exception.what());
  } catch (...) {
    return Fail(error,
                "online LiDAR association route selection threw an unknown "
                "exception");
  }
}

bool BuildOnlineLidarAssociations(
    const Reconstruction& reconstruction,
    const OnlineLidarAssociationRequest& request,
    OnlineLidarAssociationOutput* output,
    OnlineLidarAssociationAudit* audit,
    std::string* error,
    OnlineLidarAssociationTriggerGate* trigger_gate) noexcept {
  try {
    ResetBuildResults(output, audit, error);
    if (trigger_gate != nullptr) {
      trigger_gate->checked = false;
      trigger_gate->passed = false;
      trigger_gate->actual_count = 0;
      trigger_gate->nearest_query_count = 0;
    }
    if (error == nullptr) {
      return false;
    }
    if (output == nullptr) {
      return Fail(error, "association output is null");
    }
    if (audit == nullptr) {
      return Fail(error, "association audit is null");
    }
    if (request.attempt_id == 0) {
      return Fail(error, "attempt_id must be non-zero");
    }
    if (request.pass_index == 0) {
      return Fail(error, "pass_index must be non-zero");
    }
    if (request.trigger_image_id == kInvalidImageId) {
      return Fail(error, "trigger_image_id is invalid");
    }
    if (!ValidateOptions(request.options, error)) {
      return false;
    }
    if (!ValidateProjectionOptionsForCanonicalHash(
            request.projection_options, error)) {
      return false;
    }
    if (request.snapshot == nullptr) {
      return Fail(error, "LiDAR snapshot is null");
    }

    const lidar::LidarMapSnapshot& snapshot = *request.snapshot;
    if (snapshot.Frame() != lidar::LidarCoordinateFrame::COLMAP_WORLD) {
      return Fail(error, "LiDAR snapshot frame must be COLMAP_WORLD");
    }
    if (snapshot.Version() != snapshot.MaxScanIndex()) {
      return Fail(error,
                  "LiDAR snapshot version does not equal max scan index");
    }
    if (snapshot.Version() != request.expected_map_version) {
      return Fail(error, "LiDAR snapshot version does not match request");
    }
    if (snapshot.MaxScanIndex() != request.expected_max_scan_index) {
      return Fail(error,
                  "LiDAR snapshot max scan index does not match request");
    }
    if (snapshot.SnapshotSha256() != request.expected_snapshot_sha256) {
      return Fail(error, "LiDAR snapshot SHA256 does not match request");
    }
    if (snapshot.GeometrySha256() != request.expected_geometry_sha256) {
      return Fail(error, "LiDAR geometry SHA256 does not match request");
    }

    if (request.ordered_frozen_image_ids.empty() ||
        request.ordered_frozen_image_ids.size() > 20) {
      return Fail(error, "frozen image window size must be in [1, 20]");
    }
    std::set<image_t> window_image_ids;
    size_t trigger_count = 0;
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      if (image_id == kInvalidImageId) {
        return Fail(error, "frozen image window contains an invalid image ID");
      }
      if (!window_image_ids.insert(image_id).second) {
        return Fail(error, "frozen image window contains duplicate images");
      }
      if (image_id == request.trigger_image_id) {
        ++trigger_count;
      }
      if (!reconstruction.ExistsImage(image_id)) {
        return Fail(error, "frozen image window contains a missing image");
      }
      const Image& image = reconstruction.Image(image_id);
      if (!image.IsRegistered()) {
        return Fail(error,
                    "frozen image window contains an unregistered image");
      }
      if (!image.HasCamera() ||
          !reconstruction.ExistsCamera(image.CameraId())) {
        return Fail(error,
                    "frozen image window image has no existing camera");
      }
    }
    if (trigger_count != 1) {
      return Fail(error,
                  "trigger image must occur exactly once in frozen window");
    }

    std::vector<image_t> owner_priority;
    owner_priority.reserve(request.ordered_frozen_image_ids.size());
    owner_priority.push_back(request.trigger_image_id);
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      if (image_id != request.trigger_image_id) {
        owner_priority.push_back(image_id);
      }
    }

    OnlineLidarAssociationAudit local_audit;
    local_audit.attempt_id = request.attempt_id;
    local_audit.pass_index = request.pass_index;
    local_audit.trigger_image_id = request.trigger_image_id;
    local_audit.map_version = snapshot.Version();
    local_audit.max_scan_index = snapshot.MaxScanIndex();
    local_audit.snapshot_sha256 = snapshot.SnapshotSha256();
    local_audit.geometry_sha256 = snapshot.GeometrySha256();
    if (!SizeToUint64(request.point3D_ids.size(),
                      "request.point3D_ids.size",
                      &local_audit.input_point_count,
                      error)) {
      return false;
    }
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      local_audit.preliminary_selected_count_by_image.emplace(image_id, 0);
    }

    std::set<point3D_t> unique_point3D_ids;
    std::vector<PreparedPoint> prepared_points;
    prepared_points.reserve(request.point3D_ids.size());
    for (const point3D_t point3D_id : request.point3D_ids) {
      if (point3D_id == kInvalidPoint3DId) {
        return Fail(error, "input contains an invalid point3D ID");
      }
      if (!unique_point3D_ids.insert(point3D_id).second) {
        return Fail(error, "input contains duplicate point3D IDs");
      }
      if (!reconstruction.ExistsPoint3D(point3D_id)) {
        return Fail(error, "input contains a missing Point3D");
      }
    }

    std::set<point3D_t> trigger_point3D_ids;
    std::map<point3D_t, lidar::NearestPlaneResult> trigger_nearest;
    if (trigger_gate != nullptr && request.options.local_lidar_kdtree_only) {
      for (const Point2D& point2D :
           reconstruction.Image(request.trigger_image_id).Points2D()) {
        if (point2D.HasPoint3D() &&
            unique_point3D_ids.count(point2D.Point3DId()) != 0) {
          trigger_point3D_ids.insert(point2D.Point3DId());
        }
      }
      for (const point3D_t point3D_id : trigger_point3D_ids) {
        PreparedPoint prepared;
        if (!PreparePoint(reconstruction, request, window_image_ids,
                          owner_priority, point3D_id, &prepared, error)) {
          return false;
        }
        prepared_points.push_back(std::move(prepared));
      }
      for (const PreparedPoint& prepared : prepared_points) {
        if (!prepared.has_owner ||
            prepared.owner_image_id != request.trigger_image_id) {
          continue;
        }
        lidar::NearestPlaneResult nearest = snapshot.FindNearestPlane(
            prepared.xyz, lidar::LidarCoordinateFrame::COLMAP_WORLD,
            lidar::LidarNormalScale::BA, prepared.search_range);
        ++trigger_gate->nearest_query_count;
        if (!nearest.ok) {
          return Fail(error, "trigger nearest-plane query failed: " +
                                 nearest.error);
        }
        if (nearest.found) {
          if (!std::isfinite(nearest.squared_distance) ||
              nearest.squared_distance < 0.0 ||
              !ValidateKdPlane(snapshot, nearest.plane, error)) {
            if (error->empty()) *error = "trigger nearest-plane result is invalid";
            return false;
          }
          double plane_offset = 0.0;
          for (size_t axis = 0; axis < 3; ++axis) {
            plane_offset -= static_cast<double>(nearest.plane.normal[axis]) *
                            nearest.plane.point[axis];
          }
          if (!std::isfinite(plane_offset)) {
            return Fail(error, "trigger nearest-plane equation is non-finite");
          }
          ++trigger_gate->actual_count;
        }
        trigger_nearest.emplace(prepared.point3D_id, std::move(nearest));
      }
      trigger_gate->checked = true;
      trigger_gate->passed = trigger_gate->Accepts(trigger_gate->actual_count);
      if (!trigger_gate->passed) return true;
    }

    for (const point3D_t point3D_id : request.point3D_ids) {
      if (trigger_point3D_ids.count(point3D_id) != 0) continue;
      PreparedPoint prepared;
      if (!PreparePoint(reconstruction, request, window_image_ids,
                        owner_priority, point3D_id, &prepared, error)) {
        return false;
      }
      prepared_points.push_back(std::move(prepared));
    }
    for (const PreparedPoint& prepared : prepared_points) {
      if (prepared.route == OnlineLidarAssociationRoute::PROJECTION) {
        if (!IncrementUint64(&local_audit.projection_route_point_count,
                             "audit.projection_route_point_count", error)) {
          return false;
        }
      } else if (prepared.route == OnlineLidarAssociationRoute::KDTREE) {
        if (!IncrementUint64(&local_audit.kdtree_route_point_count,
                             "audit.kdtree_route_point_count", error)) {
          return false;
        }
      } else {
        return Fail(error, "association route is invalid");
      }
    }
    std::sort(prepared_points.begin(), prepared_points.end(), PreparedPointLess);

    std::vector<ProjectionImageWork> projection_image_work;
    projection_image_work.reserve(request.ordered_frozen_image_ids.size());
    std::map<image_t, size_t> projection_image_indices;
    for (const image_t image_id : request.ordered_frozen_image_ids) {
      ProjectionImageWork image_work;
      image_work.image_id = image_id;
      projection_image_indices.emplace(image_id, projection_image_work.size());
      projection_image_work.push_back(std::move(image_work));
    }

    const size_t pair_threshold =
        static_cast<size_t>(request.options.ba_match_features_threshold);
    for (size_t prepared_index = 0; prepared_index < prepared_points.size();
         ++prepared_index) {
      PreparedPoint& prepared = prepared_points[prepared_index];
      if (prepared.route != OnlineLidarAssociationRoute::PROJECTION) {
        continue;
      }
      prepared.projection_observation_gates.reserve(
          prepared.window_observations.size());
      for (const WindowObservation& observation :
           prepared.window_observations) {
        ProjectionObservationGate gate;
        gate.observation = observation;
        gate.trigger_bypass =
            observation.image_id == request.trigger_image_id;
        gate.eligible = true;
        if (!gate.trigger_bypass) {
          gate.pair_exists = reconstruction.ExistsImagePair(
              request.trigger_image_id, observation.image_id);
          if (gate.pair_exists) {
            gate.num_total_corrs =
                reconstruction
                    .ImagePair(request.trigger_image_id,
                               observation.image_id)
                    .num_total_corrs;
            gate.eligible = gate.num_total_corrs > pair_threshold;
          }
          if (!gate.eligible) {
            if (!IncrementUint64(
                    &local_audit.pair_threshold_skipped_observation_count,
                    "audit.pair_threshold_skipped_observation_count",
                    error)) {
              return false;
            }
            prepared.projection_observation_gates.push_back(gate);
            continue;
          }
        }

        prepared.projection_observation_gates.push_back(gate);
        const auto image_index =
            projection_image_indices.find(observation.image_id);
        if (image_index == projection_image_indices.end()) {
          return Fail(error,
                      "projection observation is outside frozen window");
        }
        ProjectionEligibleObservation eligible;
        eligible.prepared_point_index = prepared_index;
        eligible.point3D_id = prepared.point3D_id;
        eligible.point2D_idx = observation.point2D_idx;
        projection_image_work[image_index->second]
            .eligible_observations.push_back(eligible);
      }
    }

    std::vector<ProjectionCandidate> best_projection_candidates(
        prepared_points.size());
    std::vector<bool> has_projection_candidate(prepared_points.size(), false);
    lidar::PcdProj pass_projector(request.projection_options);
    local_audit.projection_calls.reserve(projection_image_work.size());
    std::vector<FrozenProjectionInput> frozen_projection_inputs;
    frozen_projection_inputs.reserve(projection_image_work.size());
    for (const ProjectionImageWork& image_work : projection_image_work) {
      if (image_work.eligible_observations.empty()) {
        continue;
      }

      const Image& source_image = reconstruction.Image(image_work.image_id);
      if (source_image.Points2D().size() >
          static_cast<size_t>(kInvalidPoint2DIdx)) {
        return Fail(error, "projection image has too many point2D slots");
      }
      Image projection_image = source_image;
      std::map<point3D_t, ProjectionEligibleObservation> eligible_by_point;
      for (const ProjectionEligibleObservation& eligible :
           image_work.eligible_observations) {
        if (!eligible_by_point.emplace(eligible.point3D_id, eligible).second) {
          return Fail(error,
                      "projection image has duplicate eligible point3D IDs");
        }
        if (eligible.point2D_idx == kInvalidPoint2DIdx ||
            eligible.point2D_idx >= source_image.NumPoints2D()) {
          return Fail(error,
                      "projection image has an invalid eligible point2D index");
        }
        const class Point2D& point2D =
            source_image.Point2D(eligible.point2D_idx);
        if (!point2D.HasPoint3D() ||
            point2D.Point3DId() != eligible.point3D_id) {
          return Fail(error,
                      "projection image eligible binding does not match source");
        }
      }

      for (size_t point2D_index = 0;
           point2D_index < projection_image.Points2D().size();
           ++point2D_index) {
        const point2D_t point2D_idx =
            static_cast<point2D_t>(point2D_index);
        const class Point2D& point2D =
            projection_image.Point2D(point2D_idx);
        if (!point2D.HasPoint3D()) {
          continue;
        }
        const auto eligible = eligible_by_point.find(point2D.Point3DId());
        if (eligible == eligible_by_point.end() ||
            eligible->second.point2D_idx != point2D_idx) {
          projection_image.ResetPoint3DForPoint2D(point2D_idx);
        }
      }

      const Camera projection_camera =
          reconstruction.Camera(source_image.CameraId());
      FrozenProjectionInput frozen_projection_input;
      if (!FreezeProjectionInput(projection_image,
                                 projection_camera,
                                 &frozen_projection_input,
                                 error)) {
        return false;
      }

      std::map<point3D_t, lidar::SnapshotProjectionMatch> matches;
      lidar::SnapshotProjectionAudit projection_audit;
      std::string projection_error;
      if (!pass_projector.ProjectImageToSnapshot(projection_image,
                                                 projection_camera,
                                                 snapshot,
                                                 &matches,
                                                 &projection_audit,
                                                 &projection_error)) {
        std::string message =
            "snapshot projection failed for image " +
            std::to_string(static_cast<uint64_t>(image_work.image_id));
        if (!projection_error.empty()) {
          message += ": ";
          message += projection_error;
        }
        return Fail(error, message);
      }
      if (projection_audit.map_version != snapshot.Version() ||
          projection_audit.snapshot_sha256 != snapshot.SnapshotSha256() ||
          projection_audit.geometry_sha256 != snapshot.GeometrySha256() ||
          projection_audit.snapshot_voxel_count != snapshot.VoxelCount() ||
          projection_audit.snapshot_block_count != snapshot.BlockCount()) {
        return Fail(error,
                    "snapshot projection audit does not match snapshot");
      }
      uint64_t eligible_point_count = 0;
      if (!SizeToUint64(eligible_by_point.size(),
                        "eligible_by_point.size",
                        &eligible_point_count,
                        error)) {
        return false;
      }
      if (projection_audit.input_feature_count != eligible_point_count) {
        return Fail(error,
                    "snapshot projection input was not the exact eligible set");
      }

      for (const auto& match_entry : matches) {
        const auto eligible = eligible_by_point.find(match_entry.first);
        if (eligible == eligible_by_point.end()) {
          return Fail(error,
                      "snapshot projection returned an ineligible point");
        }
        if (!ValidateProjectionMatch(snapshot,
                                     eligible->second.point3D_id,
                                     eligible->second.point2D_idx,
                                     match_entry.first,
                                     match_entry.second,
                                     error)) {
          return false;
        }
        double angle_score = 0.0;
        if (!ComputeProjectionAngleScore(
                prepared_points[eligible->second.prepared_point_index].xyz,
                match_entry.second.plane,
                &angle_score,
                error)) {
          return false;
        }

        ProjectionCandidate candidate;
        candidate.owner_image_id = image_work.image_id;
        candidate.owner_point2D_idx = eligible->second.point2D_idx;
        candidate.match = match_entry.second;
        candidate.angle_score = angle_score;
        const size_t prepared_index =
            eligible->second.prepared_point_index;
        if (!has_projection_candidate[prepared_index] ||
            ProjectionCandidateLess(
                candidate, best_projection_candidates[prepared_index])) {
          best_projection_candidates[prepared_index] = std::move(candidate);
          has_projection_candidate[prepared_index] = true;
        }
      }

      OnlineLidarProjectionCallAudit call_audit;
      call_audit.image_id = image_work.image_id;
      call_audit.projection = std::move(projection_audit);
      local_audit.projection_calls.push_back(std::move(call_audit));
      frozen_projection_inputs.push_back(std::move(frozen_projection_input));
    }
    if (!SizeToUint64(local_audit.projection_calls.size(),
                      "audit.projection_calls.size",
                      &local_audit.projection_call_count,
                      error)) {
      return false;
    }

    OnlineLidarAssociationOutput local_output;
    local_output.associations.reserve(prepared_points.size());
    for (size_t prepared_index = 0; prepared_index < prepared_points.size();
         ++prepared_index) {
      const PreparedPoint& prepared = prepared_points[prepared_index];
      if (prepared.route == OnlineLidarAssociationRoute::PROJECTION) {
        if (prepared.window_observations.empty()) {
          if (!IncrementUint64(&local_audit.skipped_point_count,
                               "audit.skipped_point_count",
                               error) ||
              !IncrementUint64(
                  &local_audit.skipped_no_window_observation_count,
                  "audit.skipped_no_window_observation_count",
                  error)) {
            return false;
          }
          continue;
        }
        if (std::none_of(prepared.projection_observation_gates.begin(),
                         prepared.projection_observation_gates.end(),
                         [](const ProjectionObservationGate& gate) {
                           return gate.eligible;
                         })) {
          if (!IncrementUint64(&local_audit.skipped_point_count,
                               "audit.skipped_point_count",
                               error) ||
              !IncrementUint64(
                  &local_audit.skipped_pair_threshold_point_count,
                  "audit.skipped_pair_threshold_point_count",
                  error)) {
            return false;
          }
          continue;
        }
        if (!has_projection_candidate[prepared_index]) {
          if (!IncrementUint64(&local_audit.skipped_point_count,
                               "audit.skipped_point_count",
                               error) ||
              !IncrementUint64(&local_audit.skipped_no_plane_count,
                               "audit.skipped_no_plane_count",
                               error)) {
            return false;
          }
          continue;
        }

        const ProjectionCandidate& candidate =
            best_projection_candidates[prepared_index];
        OnlineLidarAssociation association;
        association.attempt_id = request.attempt_id;
        association.pass_index = request.pass_index;
        association.point3D_id = prepared.point3D_id;
        association.point3D_xyz = prepared.xyz;
        association.owner_image_id = candidate.owner_image_id;
        association.owner_point2D_idx = candidate.owner_point2D_idx;
        association.map_version = snapshot.Version();
        association.max_scan_index = snapshot.MaxScanIndex();
        association.snapshot_sha256 = snapshot.SnapshotSha256();
        association.geometry_sha256 = snapshot.GeometrySha256();
        association.route = OnlineLidarAssociationRoute::PROJECTION;
        association.plane = candidate.match.plane;
        association.plane_key = candidate.match.plane_key;
        for (size_t axis = 0; axis < 3; ++axis) {
          association.plane_abcd[axis] =
              static_cast<double>(candidate.match.plane.normal[axis]);
        }
        association.plane_abcd[3] =
            -(association.plane_abcd[0] * candidate.match.plane.point[0] +
              association.plane_abcd[1] * candidate.match.plane.point[1] +
              association.plane_abcd[2] * candidate.match.plane.point[2]);
        if (!std::isfinite(association.plane_abcd[3])) {
          return Fail(error,
                      "snapshot projection plane equation is non-finite");
        }
        association.has_search_range = false;
        association.search_range = 0.0;
        association.lidar_point_type = LidarPointType::Proj;
        association.projection_camera_distance =
            static_cast<double>(candidate.match.camera_distance);
        association.projection_angle_score = candidate.angle_score;
        local_output.associations.push_back(std::move(association));
        if (!IncrementUint64(
                &local_audit.projection_selected_association_count,
                "audit.projection_selected_association_count",
                error)) {
          return false;
        }

        auto owner_count =
            local_audit.preliminary_selected_count_by_image.find(
                candidate.owner_image_id);
        if (owner_count ==
            local_audit.preliminary_selected_count_by_image.end()) {
          return Fail(error, "association owner is outside frozen window");
        }
        if (!IncrementUint64(&owner_count->second,
                             "audit preliminary per-image count",
                             error)) {
          return false;
        }
        continue;
      }

      if (!prepared.has_owner) {
        if (!IncrementUint64(&local_audit.skipped_point_count,
                             "audit.skipped_point_count",
                             error) ||
            !IncrementUint64(
                &local_audit.skipped_no_window_observation_count,
                "audit.skipped_no_window_observation_count",
                error)) {
          return false;
        }
        continue;
      }

      const auto cached_nearest = trigger_nearest.find(prepared.point3D_id);
      const lidar::NearestPlaneResult nearest =
          cached_nearest != trigger_nearest.end()
              ? cached_nearest->second
              : snapshot.FindNearestPlane(
                    prepared.xyz, lidar::LidarCoordinateFrame::COLMAP_WORLD,
                    lidar::LidarNormalScale::BA, prepared.search_range);
      if (!nearest.ok) {
        std::string message = "nearest-plane query failed";
        if (!nearest.error.empty()) {
          message += ": ";
          message += nearest.error;
        }
        return Fail(error, message);
      }
      if (!nearest.found) {
        if (!IncrementUint64(&local_audit.skipped_point_count,
                             "audit.skipped_point_count",
                             error) ||
            !IncrementUint64(&local_audit.skipped_no_plane_count,
                             "audit.skipped_no_plane_count",
                             error)) {
          return false;
        }
        continue;
      }
      if (!std::isfinite(nearest.squared_distance) ||
          nearest.squared_distance < 0.0) {
        return Fail(error, "nearest-plane query returned invalid distance");
      }
      if (!ValidateKdPlane(snapshot, nearest.plane, error)) {
        return false;
      }

      OnlineLidarAssociation association;
      association.attempt_id = request.attempt_id;
      association.pass_index = request.pass_index;
      association.point3D_id = prepared.point3D_id;
      association.point3D_xyz = prepared.xyz;
      association.owner_image_id = prepared.owner_image_id;
      association.owner_point2D_idx = prepared.owner_point2D_idx;
      association.map_version = snapshot.Version();
      association.max_scan_index = snapshot.MaxScanIndex();
      association.snapshot_sha256 = snapshot.SnapshotSha256();
      association.geometry_sha256 = snapshot.GeometrySha256();
      association.route = OnlineLidarAssociationRoute::KDTREE;
      association.plane = nearest.plane;
      association.plane_key = nearest.plane.key;
      for (size_t axis = 0; axis < 3; ++axis) {
        association.plane_abcd[axis] =
            static_cast<double>(nearest.plane.normal[axis]);
      }
      association.plane_abcd[3] =
          -(association.plane_abcd[0] * nearest.plane.point[0] +
            association.plane_abcd[1] * nearest.plane.point[1] +
            association.plane_abcd[2] * nearest.plane.point[2]);
      if (!std::isfinite(association.plane_abcd[3])) {
        return Fail(error, "nearest-plane equation is non-finite");
      }
      association.has_search_range = true;
      association.search_range = prepared.search_range;
      const double absolute_normal_x =
          std::abs(static_cast<double>(nearest.plane.normal[0]));
      const double absolute_normal_y =
          std::abs(static_cast<double>(nearest.plane.normal[1]));
      const double absolute_normal_z =
          std::abs(static_cast<double>(nearest.plane.normal[2]));
      association.lidar_point_type =
          absolute_normal_y > 10.0 * absolute_normal_x &&
                  absolute_normal_y > 10.0 * absolute_normal_z
              ? LidarPointType::IcpGround
              : LidarPointType::Icp;
      association.projection_camera_distance = 0.0;
      association.projection_angle_score = 0.0;
      local_output.associations.push_back(std::move(association));
      if (!IncrementUint64(&local_audit.kdtree_selected_association_count,
                           "audit.kdtree_selected_association_count",
                           error)) {
        return false;
      }

      auto owner_count = local_audit.preliminary_selected_count_by_image.find(
          prepared.owner_image_id);
      if (owner_count ==
          local_audit.preliminary_selected_count_by_image.end()) {
        return Fail(error, "association owner is outside frozen window");
      }
      if (!IncrementUint64(&owner_count->second,
                           "audit preliminary per-image count",
                           error)) {
        return false;
      }
    }

    std::stable_sort(local_output.associations.begin(),
                     local_output.associations.end(),
                     AssociationLess);
    for (size_t index = 0; index < local_output.associations.size(); ++index) {
      if (!SizeToUint64(index,
                        "association index",
                        &local_output.associations[index].association_id,
                        error)) {
        return false;
      }
    }
    if (!SizeToUint64(local_output.associations.size(),
                      "output.associations.size",
                      &local_audit.selected_association_count,
                      error)) {
      return false;
    }
    local_audit.trigger_preliminary_selected_count =
        local_audit.preliminary_selected_count_by_image
            .at(request.trigger_image_id);
    std::string association_sha256;
    if (!BuildCanonicalAssociationSha256(request,
                                         unique_point3D_ids,
                                         prepared_points,
                                         frozen_projection_inputs,
                                         local_output,
                                         local_audit,
                                         &association_sha256,
                                         error)) {
      ResetBuildResults(output, audit, nullptr);
      return false;
    }
    local_audit.association_sha256 = std::move(association_sha256);

    output->associations = std::move(local_output.associations);
    *audit = std::move(local_audit);
    return true;
  } catch (const std::exception& exception) {
    ResetBuildResults(output, audit, error);
    return FailWithException(
        error, "online LiDAR association build threw", exception.what());
  } catch (...) {
    ResetBuildResults(output, audit, error);
    return Fail(error,
                "online LiDAR association build threw an unknown exception");
  }
}

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED
