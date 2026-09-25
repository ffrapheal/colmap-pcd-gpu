#include "sfm/online_lidar_ba_intent.h"

#ifdef GPU_BA_CUDA_ENABLED

#include "gpu_ba/custom_cuda.h"
#include "optim/bundle_adjustment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace colmap {

OnlineLidarNativeBaSolveIntentBuildOutput::
    OnlineLidarNativeBaSolveIntentBuildOutput(
        OnlineLidarNativeBaSolveIntentBuildOutput&& other) noexcept {
  *this = std::move(other);
}

OnlineLidarNativeBaSolveIntentBuildOutput&
OnlineLidarNativeBaSolveIntentBuildOutput::operator=(
    OnlineLidarNativeBaSolveIntentBuildOutput&& other) noexcept {
  if (this != &other) {
    Reset();
    built_ = other.built_;
    association_output_ = std::move(other.association_output_);
    association_audit_ = std::move(other.association_audit_);
    intent_ = std::move(other.intent_);
    resolved_cuda_options_ = std::move(other.resolved_cuda_options_);
    other.Reset();
  }
  return *this;
}

bool OnlineLidarNativeBaSolveIntentBuildOutput::built() const noexcept {
  return built_;
}

const OnlineLidarAssociationOutput&
OnlineLidarNativeBaSolveIntentBuildOutput::association_output()
    const noexcept {
  return association_output_;
}

const OnlineLidarAssociationAudit&
OnlineLidarNativeBaSolveIntentBuildOutput::association_audit() const noexcept {
  return association_audit_;
}

const gpu_ba::NativeBaSolveIntent&
OnlineLidarNativeBaSolveIntentBuildOutput::intent() const noexcept {
  return intent_;
}

const gpu_ba::CudaFullLmOptions&
OnlineLidarNativeBaSolveIntentBuildOutput::resolved_cuda_options()
    const noexcept {
  return resolved_cuda_options_;
}

void OnlineLidarNativeBaSolveIntentBuildOutput::Reset() noexcept {
  static_assert(
      std::is_nothrow_move_assignable<OnlineLidarAssociationOutput>::value,
      "association output reset must not throw");
  static_assert(
      std::is_nothrow_move_assignable<OnlineLidarAssociationAudit>::value,
      "association audit reset must not throw");
  static_assert(
      std::is_nothrow_move_assignable<gpu_ba::NativeBaSolveIntent>::value,
      "native intent reset must not throw");
  static_assert(
      std::is_nothrow_move_assignable<gpu_ba::CudaFullLmOptions>::value,
      "resolved CUDA options reset must not throw");
  built_ = false;
  association_output_ = OnlineLidarAssociationOutput();
  association_audit_ = OnlineLidarAssociationAudit();
  intent_ = gpu_ba::NativeBaSolveIntent();
  resolved_cuda_options_ = gpu_ba::CudaFullLmOptions();
}

void OnlineLidarNativeBaSolveIntentBuildOutput::Seal(
    OnlineLidarAssociationOutput association_output,
    OnlineLidarAssociationAudit association_audit,
    gpu_ba::NativeBaSolveIntent intent,
    gpu_ba::CudaFullLmOptions resolved_cuda_options) {
  Reset();
  association_output_ = std::move(association_output);
  association_audit_ = std::move(association_audit);
  intent_ = std::move(intent);
  resolved_cuda_options_ = std::move(resolved_cuda_options);
  built_ = true;
}

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

void ResetMaterializationSummary(
    OnlineLidarMaterializationSummary* summary) noexcept {
  if (summary != nullptr) {
    try {
      *summary = OnlineLidarMaterializationSummary();
    } catch (...) {
      summary->per_image_materialized_count.clear();
      summary->trigger_materialized_count = 0;
      summary->materialized_association_count = 0;
      summary->materialized_visual_residual_count = 0;
      summary->materialized_lidar_residual_count = 0;
      summary->materialized_association_ids.clear();
      summary->materialized_online_lidar_identity =
          gpu_ba::NativeBaOnlineLidarIdentity();
      summary->solver_evaluation_pending = false;
    }
  }
}

bool SizeToUint64(const size_t value,
                  const char* field,
                  uint64_t* converted,
                  std::string* error) {
  if (converted == nullptr) {
    return Fail(error, std::string(field) + " output is null");
  }
  if (static_cast<std::uintmax_t>(value) >
      static_cast<std::uintmax_t>(std::numeric_limits<uint64_t>::max())) {
    return Fail(error, std::string(field) + " exceeds uint64_t");
  }
  *converted = static_cast<uint64_t>(value);
  return true;
}

bool IncrementUint64(uint64_t* value,
                     const char* field,
                     std::string* error) {
  if (value == nullptr) {
    return Fail(error, std::string(field) + " is null");
  }
  if (*value == std::numeric_limits<uint64_t>::max()) {
    return Fail(error, std::string(field) + " overflows uint64_t");
  }
  ++(*value);
  return true;
}

bool DecodeLowercaseSha256(const std::string& encoded,
                           const char* field,
                           std::array<uint8_t, 32>* decoded,
                           std::string* error) {
  if (decoded == nullptr) {
    return Fail(error, std::string(field) + " output is null");
  }
  decoded->fill(0);
  if (encoded.size() != decoded->size() * 2) {
    return Fail(error, std::string(field) +
                           " must contain 64 lowercase hexadecimal digits");
  }
  bool has_nonzero_byte = false;
  for (size_t index = 0; index < decoded->size(); ++index) {
    const auto nibble = [](const char value, uint8_t* result) {
      if (value >= '0' && value <= '9') {
        *result = static_cast<uint8_t>(value - '0');
        return true;
      }
      if (value >= 'a' && value <= 'f') {
        *result = static_cast<uint8_t>(value - 'a' + 10);
        return true;
      }
      return false;
    };
    uint8_t high = 0;
    uint8_t low = 0;
    if (!nibble(encoded[index * 2], &high) ||
        !nibble(encoded[index * 2 + 1], &low)) {
      decoded->fill(0);
      return Fail(error, std::string(field) +
                             " must contain 64 lowercase hexadecimal digits");
    }
    (*decoded)[index] = static_cast<uint8_t>((high << 4) | low);
    has_nonzero_byte = has_nonzero_byte || (*decoded)[index] != 0;
  }
  if (!has_nonzero_byte) {
    decoded->fill(0);
    return Fail(error, std::string(field) + " must be non-zero");
  }
  return true;
}

template <typename Value, size_t Size>
bool IsFiniteArray(const std::array<Value, Size>& values) {
  for (const Value value : values) {
    if (!std::isfinite(static_cast<double>(value))) {
      return false;
    }
  }
  return true;
}

bool ResolveLidarWeight(const BundleAdjustmentOptions& options,
                        const LidarPointType type,
                        double* weight,
                        std::string* error) {
  if (weight == nullptr) {
    return Fail(error, "LiDAR weight output is null");
  }
  switch (type) {
    case LidarPointType::Proj:
      *weight = options.proj_lidar_constraint_weight;
      break;
    case LidarPointType::Icp:
      *weight = options.icp_lidar_constraint_weight;
      break;
    case LidarPointType::IcpGround:
      *weight = options.icp_ground_lidar_constraint_weight;
      break;
    default:
      return Fail(error, "online LiDAR association type is invalid");
  }
  if (!std::isfinite(*weight)) {
    return Fail(error, "online LiDAR association weight is non-finite");
  }
  return true;
}

bool ValidateStrictOnlineOptions(const BundleAdjustmentOptions& options,
                                 std::string* error) {
  if (options.ba_backend != "custom_cuda") {
    return Fail(error, "online native BA requires ba_backend=custom_cuda");
  }
  if (options.ba_fallback_to_ceres) {
    return Fail(error, "online native BA requires Ceres fallback to be disabled");
  }
  if (options.ba_cuda_problem_source !=
      gpu_ba::CudaProblemSource::kNativeGraph) {
    return Fail(error, "online native BA requires native_graph problem source");
  }
  if (!options.refine_extrinsics || options.refine_focal_length ||
      options.refine_principal_point || options.refine_extra_params) {
    return Fail(error,
                "online native BA requires variable extrinsics and fixed "
                "intrinsics");
  }
  if (!options.if_add_lidar_constraint) {
    return Fail(error, "online native BA requires LiDAR constraints");
  }
  return true;
}

bool ValidateFrozenWindowAndPoints(
    const Reconstruction& reconstruction,
    const OnlineLidarAssociationRequest& request,
    const bool validate_point_observations,
    std::string* error) {
  if (request.attempt_id == 0 || request.pass_index == 0) {
    return Fail(error, "online association attempt and pass must be non-zero");
  }
  if (request.trigger_image_id == 0 ||
      request.trigger_image_id == kInvalidImageId) {
    return Fail(error, "online association trigger image must be non-zero");
  }
  if (request.expected_map_version == 0 ||
      request.expected_max_scan_index == 0 ||
      request.expected_map_version != request.expected_max_scan_index) {
    return Fail(error, "online association map identity is invalid");
  }
  std::array<uint8_t, 32> decoded;
  if (!DecodeLowercaseSha256(request.expected_snapshot_sha256,
                            "snapshot SHA256",
                            &decoded,
                            error) ||
      !DecodeLowercaseSha256(request.expected_geometry_sha256,
                            "geometry SHA256",
                            &decoded,
                            error)) {
    return false;
  }
  if (request.snapshot == nullptr) {
    return Fail(error, "online association snapshot is null");
  }

  if (request.ordered_frozen_image_ids.empty()) {
    return Fail(error, "online association frozen window is empty");
  }
  std::set<image_t> frozen_image_ids;
  size_t trigger_count = 0;
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    if (image_id == kInvalidImageId ||
        !frozen_image_ids.insert(image_id).second) {
      return Fail(error, "online association frozen window is not unique");
    }
    trigger_count += image_id == request.trigger_image_id ? 1 : 0;
    if (!reconstruction.ExistsImage(image_id)) {
      return Fail(error, "online association frozen image is missing");
    }
    const Image& image = reconstruction.Image(image_id);
    if (!image.IsRegistered() || !image.HasCamera() ||
        !reconstruction.ExistsCamera(image.CameraId())) {
      return Fail(error, "online association frozen image is not usable");
    }
  }
  if (trigger_count != 1) {
    return Fail(error,
                "online association trigger must occur once in frozen window");
  }

  point3D_t previous_point3D_id = 0;
  bool has_previous_point3D_id = false;
  for (const point3D_t point3D_id : request.point3D_ids) {
    if (point3D_id == kInvalidPoint3DId ||
        (has_previous_point3D_id && point3D_id <= previous_point3D_id)) {
      return Fail(error,
                  "online association Point3D IDs must be strictly increasing");
    }
    previous_point3D_id = point3D_id;
    has_previous_point3D_id = true;
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      return Fail(error, "online association selected Point3D is missing");
    }
    if (!validate_point_observations) continue;
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (!point3D.XYZ().allFinite()) {
      return Fail(error, "online association selected Point3D is non-finite");
    }
    bool has_window_observation = false;
    std::set<std::pair<image_t, point2D_t>> observed_window_observations;
    for (const TrackElement& element : point3D.Track().Elements()) {
      if (frozen_image_ids.count(element.image_id) == 0) {
        continue;
      }
      if (!observed_window_observations
               .emplace(element.image_id, element.point2D_idx).second ||
          !reconstruction.ExistsImage(element.image_id)) {
        return Fail(error,
                    "online association window observation is duplicated or "
                    "missing");
      }
      const Image& image = reconstruction.Image(element.image_id);
      if (element.point2D_idx == kInvalidPoint2DIdx ||
          element.point2D_idx >= image.NumPoints2D()) {
        return Fail(error,
                    "online association window observation index is invalid");
      }
      const class Point2D& point2D = image.Point2D(element.point2D_idx);
      if (!point2D.HasPoint3D() || point2D.Point3DId() != point3D_id) {
        return Fail(error,
                    "online association window observation does not point back");
      }
      has_window_observation = true;
    }
    if (!has_window_observation) {
      return Fail(error,
                  "online association selected Point3D has no frozen-window "
                  "observation");
    }
  }
  return true;
}

bool ComputeExpectedKdSearchRange(const OnlineLidarAssociationRequest& request,
                                  const Point3D& point3D,
                                  double* search_range,
                                  std::string* error) {
  if (search_range == nullptr) {
    return Fail(error, "KD search range output is null");
  }
  if (point3D.GlobalOptNum() < 0) {
    return Fail(error, "online association Point3D optimization count is invalid");
  }
  const double dropped = static_cast<double>(point3D.GlobalOptNum()) *
                         request.options.search_range_drop_speed;
  const double reduced = request.options.kdtree_max_search_range - dropped;
  const double expected =
      std::max(request.options.kdtree_min_search_range, reduced);
  if (!std::isfinite(dropped) || !std::isfinite(reduced) ||
      !std::isfinite(expected) || expected < 0.0) {
    return Fail(error, "online association KD search range is invalid");
  }
  *search_range = expected;
  return true;
}

bool ValidateAssociationPlaneMetadata(
    const OnlineLidarAssociation& association,
    std::string* error) {
  if (association.route != OnlineLidarAssociationRoute::PROJECTION &&
      association.route != OnlineLidarAssociationRoute::KDTREE) {
    return Fail(error, "online association route is invalid");
  }
  const lidar::LidarNormalScale expected_scale =
      association.route == OnlineLidarAssociationRoute::PROJECTION
          ? lidar::LidarNormalScale::PROJECTION
          : lidar::LidarNormalScale::BA;
  if (association.plane_key != association.plane.key ||
      association.plane.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD ||
      association.plane.scale != expected_scale ||
      association.plane.voxel_count == 0 ||
      association.plane.normal_revision == 0 ||
      association.plane.normal_revision > association.map_version ||
      !IsFiniteArray(association.plane.point) ||
      !IsFiniteArray(association.plane.normal) ||
      !std::isfinite(static_cast<double>(association.plane.curvature)) ||
      !IsFiniteArray(association.plane_abcd)) {
    return Fail(error, "online association plane is invalid");
  }
  double normal_squared_norm = 0.0;
  for (const float value : association.plane.normal) {
    normal_squared_norm += static_cast<double>(value) *
                           static_cast<double>(value);
  }
  if (!std::isfinite(normal_squared_norm) || !(normal_squared_norm > 0.0)) {
    return Fail(error, "online association plane normal is invalid");
  }

  for (size_t axis = 0; axis < 3; ++axis) {
    if (association.plane_abcd[axis] !=
        static_cast<double>(association.plane.normal[axis])) {
      return Fail(error, "online association plane equation is inconsistent");
    }
  }
  const double expected_offset =
      -(association.plane_abcd[0] * association.plane.point[0] +
        association.plane_abcd[1] * association.plane.point[1] +
        association.plane_abcd[2] * association.plane.point[2]);
  if (!std::isfinite(expected_offset) ||
      association.plane_abcd[3] != expected_offset) {
    return Fail(error, "online association plane offset is inconsistent");
  }
  return true;
}

bool ValidateAssociationPlane(
    const OnlineLidarAssociation& association,
    const lidar::LidarMapSnapshot& snapshot,
    std::string* error) {
  if (!ValidateAssociationPlaneMetadata(association, error)) {
    return false;
  }

  lidar::LidarVoxelRecord voxel;
  if (!snapshot.FindVoxel(association.plane_key, &voxel) ||
      voxel.key != association.plane_key ||
      voxel.frame != lidar::LidarCoordinateFrame::COLMAP_WORLD ||
      voxel.count != association.plane.voxel_count ||
      voxel.centroid != association.plane.point) {
    return Fail(error, "online association plane does not match LiDAR map");
  }
  const lidar::LidarNormalValue* normal = nullptr;
  if (association.route == OnlineLidarAssociationRoute::PROJECTION) {
    normal = &voxel.outer_normal;
  } else if (association.route == OnlineLidarAssociationRoute::KDTREE) {
    normal = &voxel.inner_normal;
  } else {
    return Fail(error, "online association route is invalid");
  }
  if (!normal->valid || normal->normal != association.plane.normal ||
      normal->curvature != association.plane.curvature ||
      normal->revision != association.plane.normal_revision) {
    return Fail(error, "online association normal does not match LiDAR map");
  }

  return true;
}

bool BuildOnlineIdentity(const OnlineLidarAssociationAudit& audit,
                         gpu_ba::NativeBaOnlineLidarIdentity* identity,
                         std::string* error) {
  if (identity == nullptr) {
    return Fail(error, "online LiDAR identity output is null");
  }
  *identity = gpu_ba::NativeBaOnlineLidarIdentity();
  if (audit.trigger_image_id == 0 ||
      audit.trigger_image_id == kInvalidImageId || audit.map_version == 0 ||
      audit.max_scan_index == 0 || audit.map_version != audit.max_scan_index) {
    return Fail(error, "online LiDAR association identity is invalid");
  }
  gpu_ba::NativeBaOnlineLidarIdentity local;
  local.valid = true;
  local.trigger_image_id = audit.trigger_image_id;
  local.map_version = audit.map_version;
  local.max_scan_index = audit.max_scan_index;
  if (!DecodeLowercaseSha256(audit.snapshot_sha256,
                            "snapshot SHA256",
                            &local.snapshot_sha256,
                            error) ||
      !DecodeLowercaseSha256(audit.geometry_sha256,
                            "geometry SHA256",
                            &local.geometry_sha256,
                            error) ||
      !DecodeLowercaseSha256(audit.association_sha256,
                            "association SHA256",
                            &local.association_sha256,
                            error)) {
    return false;
  }
  if (!gpu_ba::IsValidNativeBaOnlineLidarIdentity(local)) {
    return Fail(error, "online LiDAR identity is not canonical");
  }
  *identity = local;
  return true;
}

bool ValidateAssociationBundle(
    const Reconstruction& reconstruction,
    const BundleAdjustmentOptions& options,
    const OnlineLidarAssociationRequest& request,
    const OnlineLidarAssociationOutput& output,
    const OnlineLidarAssociationAudit& audit,
    gpu_ba::NativeBaOnlineLidarIdentity* identity,
    std::string* error) {
  if (request.snapshot == nullptr ||
      request.snapshot->Version() != request.expected_map_version ||
      request.snapshot->MaxScanIndex() != request.expected_max_scan_index ||
      request.snapshot->SnapshotSha256() != request.expected_snapshot_sha256 ||
      request.snapshot->GeometrySha256() != request.expected_geometry_sha256) {
    return Fail(error, "online association snapshot identity changed");
  }
  if (audit.attempt_id != request.attempt_id ||
      audit.pass_index != request.pass_index ||
      audit.trigger_image_id != request.trigger_image_id ||
      audit.map_version != request.expected_map_version ||
      audit.max_scan_index != request.expected_max_scan_index ||
      audit.snapshot_sha256 != request.expected_snapshot_sha256 ||
      audit.geometry_sha256 != request.expected_geometry_sha256) {
    return Fail(error, "online association audit identity is inconsistent");
  }
  if (!BuildOnlineIdentity(audit, identity, error)) {
    return false;
  }

  uint64_t input_count = 0;
  uint64_t association_count = 0;
  if (!SizeToUint64(request.point3D_ids.size(),
                    "online association input count",
                    &input_count,
                    error) ||
      !SizeToUint64(output.associations.size(),
                    "online association selected count",
                    &association_count,
                    error)) {
    return false;
  }
  if (audit.input_point_count != input_count ||
      audit.selected_association_count != association_count ||
      output.associations.size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      audit.selected_association_count >
          std::numeric_limits<uint64_t>::max() - audit.skipped_point_count ||
      audit.selected_association_count + audit.skipped_point_count !=
          audit.input_point_count ||
      audit.projection_selected_association_count >
          std::numeric_limits<uint64_t>::max() -
              audit.kdtree_selected_association_count ||
      audit.projection_selected_association_count +
              audit.kdtree_selected_association_count !=
          audit.selected_association_count) {
    return Fail(error, "online association audit counts are inconsistent");
  }

  std::map<image_t, uint64_t> selected_count_by_image;
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    selected_count_by_image.emplace(image_id, 0);
  }
  if (selected_count_by_image.size() !=
          request.ordered_frozen_image_ids.size() ||
      audit.preliminary_selected_count_by_image.size() !=
          selected_count_by_image.size()) {
    return Fail(error, "online association per-image audit is inconsistent");
  }
  for (const auto& image_count : audit.preliminary_selected_count_by_image) {
    if (selected_count_by_image.count(image_count.first) != 1) {
      return Fail(error, "online association audit image is outside window");
    }
  }

  uint64_t projection_count = 0;
  uint64_t kdtree_count = 0;
  point3D_t previous_point3D_id = 0;
  bool has_previous_point3D_id = false;
  std::set<point3D_t> associated_point3D_ids;
  for (size_t index = 0; index < output.associations.size(); ++index) {
    const OnlineLidarAssociation& association = output.associations[index];
    if (index > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        association.association_id != static_cast<uint64_t>(index) ||
        association.association_id == gpu_ba::kBaGraphInvalidAssociationId) {
      return Fail(error, "online association IDs are not dense uint32 values");
    }
    if ((has_previous_point3D_id &&
         association.point3D_id <= previous_point3D_id) ||
        !std::binary_search(request.point3D_ids.begin(),
                            request.point3D_ids.end(),
                            association.point3D_id) ||
        !associated_point3D_ids.insert(association.point3D_id).second) {
      return Fail(error,
                  "online association Point3D order or identity is invalid");
    }
    previous_point3D_id = association.point3D_id;
    has_previous_point3D_id = true;
    if (!reconstruction.ExistsPoint3D(association.point3D_id)) {
      return Fail(error, "online association Point3D is missing");
    }
    const Point3D& point3D = reconstruction.Point3D(association.point3D_id);
    for (size_t axis = 0; axis < 3; ++axis) {
      if (association.point3D_xyz[axis] != point3D.XYZ(axis)) {
        return Fail(error, "online association Point3D snapshot is stale");
      }
    }
    if (!IsFiniteArray(association.point3D_xyz) ||
        !std::isfinite(association.search_range) ||
        !std::isfinite(association.projection_camera_distance) ||
        !std::isfinite(association.projection_angle_score) ||
        association.attempt_id != audit.attempt_id ||
        association.pass_index != audit.pass_index ||
        association.map_version != audit.map_version ||
        association.max_scan_index != audit.max_scan_index ||
        association.snapshot_sha256 != audit.snapshot_sha256 ||
        association.geometry_sha256 != audit.geometry_sha256) {
      return Fail(error, "online association payload identity is inconsistent");
    }

    auto owner_count = selected_count_by_image.find(association.owner_image_id);
    if (owner_count == selected_count_by_image.end() ||
        !reconstruction.ExistsImage(association.owner_image_id)) {
      return Fail(error, "online association owner is outside frozen window");
    }
    const Image& owner = reconstruction.Image(association.owner_image_id);
    if (association.owner_point2D_idx == kInvalidPoint2DIdx ||
        association.owner_point2D_idx >= owner.NumPoints2D()) {
      return Fail(error, "online association owner observation is invalid");
    }
    const class Point2D& owner_point2D =
        owner.Point2D(association.owner_point2D_idx);
    if (!owner_point2D.HasPoint3D() ||
        owner_point2D.Point3DId() != association.point3D_id) {
      return Fail(error, "online association owner does not point back");
    }
    if (!IncrementUint64(&owner_count->second,
                         "online association owner count",
                         error) ||
        !ValidateAssociationPlane(association, *request.snapshot, error)) {
      return false;
    }

    double weight = 0.0;
    if (!ResolveLidarWeight(
            options, association.lidar_point_type, &weight, error) ||
        !(weight > 0.0)) {
      return Fail(error,
                  "online association uses a non-positive LiDAR weight");
    }
    if (association.route == OnlineLidarAssociationRoute::PROJECTION) {
      if (association.lidar_point_type != LidarPointType::Proj ||
          association.has_search_range || association.search_range != 0.0 ||
          association.projection_camera_distance < 0.0) {
        return Fail(error, "online projection association is inconsistent");
      }
      if (!IncrementUint64(
              &projection_count, "online projection count", error)) {
        return false;
      }
    } else if (association.route == OnlineLidarAssociationRoute::KDTREE) {
      double expected_search_range = 0.0;
      if (!ComputeExpectedKdSearchRange(
              request, point3D, &expected_search_range, error)) {
        return false;
      }
      const double absolute_normal_x =
          std::abs(static_cast<double>(association.plane.normal[0]));
      const double absolute_normal_y =
          std::abs(static_cast<double>(association.plane.normal[1]));
      const double absolute_normal_z =
          std::abs(static_cast<double>(association.plane.normal[2]));
      const LidarPointType expected_type =
          absolute_normal_y > 10.0 * absolute_normal_x &&
                  absolute_normal_y > 10.0 * absolute_normal_z
              ? LidarPointType::IcpGround
              : LidarPointType::Icp;
      if (!association.has_search_range ||
          association.search_range != expected_search_range ||
          association.lidar_point_type != expected_type ||
          association.projection_camera_distance != 0.0 ||
          association.projection_angle_score != 0.0) {
        return Fail(error, "online KD association is inconsistent");
      }
      if (!IncrementUint64(&kdtree_count, "online KD count", error)) {
        return false;
      }
    } else {
      return Fail(error, "online association route is invalid");
    }
  }

  const auto trigger_count =
      selected_count_by_image.find(request.trigger_image_id);
  if (selected_count_by_image !=
          audit.preliminary_selected_count_by_image ||
      trigger_count == selected_count_by_image.end() ||
      trigger_count->second != audit.trigger_preliminary_selected_count ||
      projection_count != audit.projection_selected_association_count ||
      kdtree_count != audit.kdtree_selected_association_count) {
    return Fail(error, "online association selected counts are inconsistent");
  }
  return true;
}

bool SameNativeConstraint(
    const gpu_ba::NativeBaLidarConstraint& expected,
    const gpu_ba::LidarConstraintRecord& materialized) {
  return expected.constraint_slot == materialized.constraint_slot &&
         expected.physical_identity == materialized.physical_identity &&
         expected.association_id == materialized.association_id &&
         expected.owner_image_id == materialized.owner_image_id &&
         expected.owner_point2D_idx == materialized.owner_point2D_idx &&
         expected.lidar_type == materialized.lidar_type &&
         expected.frozen_point3D_xyz == materialized.frozen_point3D_xyz &&
         expected.plane == materialized.plane &&
         expected.lidar_xyz == materialized.lidar_xyz &&
         expected.weight == materialized.weight &&
         expected.search_range == materialized.search_range;
}

bool ResolvedCudaOptionsMatchConfig(
    const gpu_ba::CudaFullLmOptions& options,
    const gpu_ba::NativeCudaResolvedConfig& config) {
  return config.resolved && config.config_generation != 0 &&
         options.performance_mode == config.performance_mode &&
         options.arithmetic_precision == config.arithmetic_precision &&
         options.device_context_mode == config.device_context &&
         options.layer_c.layer_b.layer_a.memory_mode == config.memory_mode &&
         options.layer_c.layer_b.reduction_mode == config.reduction_mode &&
         options.audit_profile == config.audit_profile &&
         options.current_linearization_cache_mode ==
             config.linearization_cache &&
         options.layer_c.layer_b.hessian_assembly_backend ==
             config.hessian_backend &&
         options.layer_c.schur_contribution_backend == config.schur_backend &&
         options.hot_kernel_mode == config.hot_kernel &&
         options.execution_profile == config.execution_profile &&
         options.layer_c.layer_b.layer_a.residual_order ==
             config.residual_order &&
         options.prepared_selection_cache_mode ==
             config.prepared_selection_cache &&
         options.layer_c.layer_b.loss_mode == config.loss_mode &&
         options.layer_c.layer_b.loss_scale == config.loss_scale &&
         options.layer_c.layer_b.layer_a.device == config.device &&
         options.layer_c.layer_b.layer_a.block_size == config.block_size &&
         options.layer_c.layer_b.cost_reduction_threads ==
             config.cost_reduction_threads &&
         options.pair_chunk_limit_bytes_for_testing ==
             config.pair_chunk_limit_bytes &&
         options.layer_c.layer_b.hessian_segment_size_for_testing ==
             config.hessian_segment_size &&
         options.layer_c.schur_segment_size_for_testing ==
             config.schur_segment_size &&
         options.max_num_iterations == config.max_num_iterations &&
         options.max_num_consecutive_invalid_steps ==
             config.max_consecutive_invalid_steps &&
         options.function_tolerance == config.function_tolerance &&
         options.gradient_tolerance == config.gradient_tolerance &&
         options.parameter_tolerance == config.parameter_tolerance &&
         options.max_solver_time_in_seconds ==
             config.max_solver_time_in_seconds &&
         options.initial_trust_region_radius ==
             config.initial_trust_region_radius &&
         options.min_trust_region_radius == config.min_trust_region_radius &&
         options.max_trust_region_radius == config.max_trust_region_radius &&
         options.min_relative_decrease == config.min_relative_decrease &&
         options.layer_c.layer_b.min_lm_diagonal == config.min_lm_diagonal &&
         options.layer_c.layer_b.max_lm_diagonal == config.max_lm_diagonal;
}

bool SameResolvedCudaConfig(const gpu_ba::NativeCudaResolvedConfig& first,
                            const gpu_ba::NativeCudaResolvedConfig& second) {
  return first.resolved == second.resolved &&
         first.performance_mode == second.performance_mode &&
         first.config_generation == second.config_generation &&
         first.arithmetic_precision == second.arithmetic_precision &&
         first.device_context == second.device_context &&
         first.memory_mode == second.memory_mode &&
         first.reduction_mode == second.reduction_mode &&
         first.audit_profile == second.audit_profile &&
         first.linearization_cache == second.linearization_cache &&
         first.hessian_backend == second.hessian_backend &&
         first.schur_backend == second.schur_backend &&
         first.hot_kernel == second.hot_kernel &&
         first.execution_profile == second.execution_profile &&
         first.residual_order == second.residual_order &&
         first.prepared_selection_cache == second.prepared_selection_cache &&
         first.loss_mode == second.loss_mode &&
         first.lidar_residual_mode == second.lidar_residual_mode &&
         first.lidar_near_zero_threshold ==
             second.lidar_near_zero_threshold &&
         first.loss_scale == second.loss_scale &&
         first.device == second.device && first.block_size == second.block_size &&
         first.cost_reduction_threads == second.cost_reduction_threads &&
         first.pair_chunk_limit_bytes == second.pair_chunk_limit_bytes &&
         first.hessian_segment_size == second.hessian_segment_size &&
         first.schur_segment_size == second.schur_segment_size &&
         first.max_num_iterations == second.max_num_iterations &&
         first.max_consecutive_invalid_steps ==
             second.max_consecutive_invalid_steps &&
         first.function_tolerance == second.function_tolerance &&
         first.gradient_tolerance == second.gradient_tolerance &&
         first.parameter_tolerance == second.parameter_tolerance &&
         first.max_solver_time_in_seconds ==
             second.max_solver_time_in_seconds &&
         first.initial_trust_region_radius ==
             second.initial_trust_region_radius &&
         first.min_trust_region_radius == second.min_trust_region_radius &&
         first.max_trust_region_radius == second.max_trust_region_radius &&
         first.min_relative_decrease == second.min_relative_decrease &&
         first.min_lm_diagonal == second.min_lm_diagonal &&
         first.max_lm_diagonal == second.max_lm_diagonal;
}

bool ValidateSummaryIdentityAndImages(
    const gpu_ba::NativeHostSolveView& view,
    const OnlineLidarNativeBaSolveIntentBuildOutput& build_output,
    gpu_ba::NativeBaOnlineLidarIdentity* expected_identity,
    std::string* error) {
  const gpu_ba::NativeBaSolveIntent& intent = build_output.intent();
  const OnlineLidarAssociationAudit& audit = build_output.association_audit();
  if (!BuildOnlineIdentity(audit, expected_identity, error)) {
    return false;
  }
  if (intent.abi_version != gpu_ba::kNativeHostSolveViewAbiVersion ||
      view.identity.abi_version != gpu_ba::kNativeHostSolveViewAbiVersion ||
      view.identity.catalog_abi_version != gpu_ba::kHostBaGraphAbiVersion ||
      intent.kind != gpu_ba::BaKind::kLocal ||
      view.kind != gpu_ba::BaKind::kLocal || intent.owner_epoch == 0 ||
      intent.reconstruction_identity == 0 ||
      intent.expected_topology_revision == 0 ||
      intent.selection_revision == 0 || !intent.config.resolved ||
      !view.config.resolved ||
      !SameResolvedCudaConfig(view.config, intent.config) ||
      !ResolvedCudaOptionsMatchConfig(build_output.resolved_cuda_options(),
                                      intent.config) ||
      intent.visual_observation_scope !=
          gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly ||
      view.identity.visual_observation_scope !=
          gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly ||
      !gpu_ba::SameNativeBaOnlineLidarIdentity(
          intent.online_lidar_identity, *expected_identity) ||
      !gpu_ba::SameNativeBaOnlineLidarIdentity(
          view.identity.online_lidar_identity, *expected_identity) ||
      !gpu_ba::SameNativeBaOnlineLidarIdentity(
          view.lidar.online_lidar_identity, *expected_identity)) {
    return Fail(error, "online materialized identity or scope is inconsistent");
  }
  if (!view.catalog.valid() ||
      view.catalog.owner_epoch() != intent.owner_epoch ||
      view.catalog.topology_revision() != intent.expected_topology_revision ||
      view.identity.catalog_generation != view.catalog.generation() ||
      view.identity.owner_epoch != intent.owner_epoch ||
      view.identity.catalog_revision != intent.expected_topology_revision ||
      view.identity.selection_revision != intent.selection_revision ||
      view.identity.config_generation != intent.config.config_generation ||
      view.config.config_generation != intent.config.config_generation ||
      intent.lidar_map_generation != expected_identity->map_version ||
      intent.lidar_match_config_generation != intent.config.config_generation ||
      view.identity.lidar_map_generation != expected_identity->map_version ||
      view.identity.lidar_match_config_generation !=
          intent.config.config_generation ||
      view.lidar.lidar_map_generation != expected_identity->map_version ||
      view.lidar.match_config_generation != intent.config.config_generation ||
      !view.BoundaryImageSlots().empty()) {
    return Fail(error, "online materialized generations are inconsistent");
  }

  const std::vector<uint32_t>& active_slots = view.ActiveImageSlots();
  const auto catalog_images = view.catalog.images();
  if (active_slots.size() != intent.active_image_ids.size() ||
      active_slots.size() !=
          build_output.association_audit()
              .preliminary_selected_count_by_image.size()) {
    return Fail(error, "online materialized active image count is inconsistent");
  }
  std::set<uint32_t> active_image_ids;
  size_t trigger_count = 0;
  for (size_t index = 0; index < active_slots.size(); ++index) {
    const uint32_t slot = active_slots[index];
    if (slot >= catalog_images.size || !catalog_images[slot].header.alive ||
        !catalog_images[slot].registered ||
        catalog_images[slot].image_id != intent.active_image_ids[index] ||
        !active_image_ids.insert(catalog_images[slot].image_id).second ||
        build_output.association_audit().preliminary_selected_count_by_image
                .count(catalog_images[slot].image_id) !=
            1) {
      return Fail(error, "online materialized active image order is invalid");
    }
    trigger_count += catalog_images[slot].image_id ==
                             expected_identity->trigger_image_id
                         ? 1
                         : 0;
  }
  if (trigger_count != 1 ||
      active_image_ids.count(expected_identity->trigger_image_id) != 1) {
    return Fail(error, "online materialized trigger image is invalid");
  }
  return true;
}

}  // namespace

bool BuildOnlineLidarNativeBaSolveIntent(
    const Reconstruction& reconstruction,
    const BundleAdjustmentOptions& options,
    const OnlineLidarAssociationRequest& association_request,
    const uint64_t owner_epoch,
    const uint64_t topology_revision,
    const uint64_t selection_revision,
    const gpu_ba::BaKind kind,
    OnlineLidarNativeBaSolveIntentBuildOutput* output,
    std::string* error,
    OnlineLidarAssociationTriggerGate* trigger_gate) noexcept {
  try {
    if (trigger_gate != nullptr) {
      trigger_gate->checked = false;
      trigger_gate->passed = false;
      trigger_gate->actual_count = 0;
      trigger_gate->nearest_query_count = 0;
    }
    if (output != nullptr) {
      output->Reset();
    }
    if (error != nullptr) {
      error->clear();
    }
    if (output == nullptr || error == nullptr) {
      return output == nullptr
                 ? Fail(error, "online native BA build output is null")
                 : false;
    }
    if (kind != gpu_ba::BaKind::kLocal || owner_epoch == 0 ||
        topology_revision == 0 || selection_revision == 0 ||
        reconstruction.StructureOwnerEpoch() != owner_epoch ||
        reconstruction.StructureRevision() != topology_revision) {
      return Fail(error, "online native BA identity or kind is invalid");
    }
    OnlineLidarAssociationRequest canonical_request = association_request;
    std::sort(canonical_request.point3D_ids.begin(),
              canonical_request.point3D_ids.end());
    if (std::adjacent_find(canonical_request.point3D_ids.begin(),
                           canonical_request.point3D_ids.end()) !=
        canonical_request.point3D_ids.end()) {
      return Fail(error, "online association Point3D IDs contain duplicates");
    }

    const bool defer_point_observations =
        trigger_gate != nullptr &&
        canonical_request.options.local_lidar_kdtree_only &&
        std::isfinite(options.icp_lidar_constraint_weight) &&
        std::isfinite(options.icp_ground_lidar_constraint_weight);
    if (!ValidateStrictOnlineOptions(options, error) ||
        !ValidateFrozenWindowAndPoints(
            reconstruction, canonical_request, !defer_point_observations,
            error)) {
      return false;
    }

    gpu_ba::CudaFullLmOptions resolved_options;
    gpu_ba::NativeCudaResolvedConfig resolved_config;
    if (!ResolveNativeBundleAdjustmentCudaConfiguration(options,
                                                        options.solver_options,
                                                        selection_revision,
                                                        &resolved_options,
                                                        &resolved_config,
                                                        error) ||
        !resolved_config.resolved ||
        resolved_config.config_generation != selection_revision) {
      if (error->empty()) {
        return Fail(error, "online native BA CUDA configuration is unresolved");
      }
      return false;
    }

    OnlineLidarAssociationOutput association_output;
    OnlineLidarAssociationAudit association_audit;
    std::string association_error;
    OnlineLidarAssociationTriggerGate* association_gate =
        std::isfinite(options.icp_lidar_constraint_weight) &&
                std::isfinite(options.icp_ground_lidar_constraint_weight)
            ? trigger_gate
            : nullptr;
    if (!BuildOnlineLidarAssociations(reconstruction,
                                      canonical_request,
                                      &association_output,
                                      &association_audit,
                                      &association_error,
                                      association_gate)) {
      return Fail(error, association_error.empty()
                             ? "online LiDAR association build failed"
                             : association_error);
    }
    if (reconstruction.StructureOwnerEpoch() != owner_epoch ||
        reconstruction.StructureRevision() != topology_revision) {
      return Fail(error, "online native BA reconstruction identity changed");
    }
    if (trigger_gate != nullptr && trigger_gate->checked &&
        !trigger_gate->passed) {
      return true;
    }
    if (defer_point_observations &&
        !ValidateFrozenWindowAndPoints(reconstruction, canonical_request, true,
                                       error)) {
      return false;
    }

    gpu_ba::NativeBaOnlineLidarIdentity online_identity;
    if (!ValidateAssociationBundle(reconstruction,
                                   options,
                                   canonical_request,
                                   association_output,
                                   association_audit,
                                   &online_identity,
                                   error)) {
      return false;
    }

    gpu_ba::NativeBaSolveIntent intent;
    intent.owner_epoch = owner_epoch;
    intent.reconstruction_identity =
        reinterpret_cast<uintptr_t>(&reconstruction);
    intent.expected_topology_revision = topology_revision;
    intent.selection_revision = selection_revision;
    intent.kind = kind;
    intent.visual_observation_scope =
        gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly;
    intent.config = resolved_config;
    intent.active_image_ids.assign(
        canonical_request.ordered_frozen_image_ids.begin(),
        canonical_request.ordered_frozen_image_ids.end());
    intent.translation_policies.reserve(intent.active_image_ids.size());
    intent.camera_policies.reserve(intent.active_image_ids.size());
    std::set<camera_t> seen_camera_ids;
    for (const image_t image_id :
         canonical_request.ordered_frozen_image_ids) {
      gpu_ba::NativeBaTranslationPolicy translation;
      translation.image_id = image_id;
      translation.constant_mask = 0;
      intent.translation_policies.push_back(translation);

      const camera_t camera_id = reconstruction.Image(image_id).CameraId();
      if (seen_camera_ids.insert(camera_id).second) {
        gpu_ba::NativeBaCameraPolicy camera;
        camera.camera_id = camera_id;
        camera.constant = true;
        intent.camera_policies.push_back(std::move(camera));
      }
    }

    intent.explicit_variable_point_ids.assign(
        canonical_request.point3D_ids.begin(),
        canonical_request.point3D_ids.end());
    intent.point_policies.reserve(canonical_request.point3D_ids.size());
    std::map<point3D_t, const OnlineLidarAssociation*> association_by_point;
    for (const OnlineLidarAssociation& association :
         association_output.associations) {
      association_by_point.emplace(association.point3D_id, &association);
    }
    for (const point3D_t point3D_id : canonical_request.point3D_ids) {
      gpu_ba::NativeBaPointPolicy policy;
      policy.point3D_id = point3D_id;
      policy.constant = false;
      policy.config_role = 1;
      const auto association = association_by_point.find(point3D_id);
      if (association != association_by_point.end() &&
          association->second->route == OnlineLidarAssociationRoute::KDTREE) {
        policy.has_search_range = true;
        policy.search_range = association->second->search_range;
      }
      intent.point_policies.push_back(policy);
    }

    intent.lidar_map_generation = association_audit.map_version;
    intent.lidar_match_config_generation = resolved_config.config_generation;
    intent.online_lidar_identity = online_identity;
    intent.lidar_constraints.reserve(association_output.associations.size());
    for (const OnlineLidarAssociation& association :
         association_output.associations) {
      double weight = 0.0;
      if (!ResolveLidarWeight(
              options, association.lidar_point_type, &weight, error) ||
          !(weight > 0.0)) {
        return Fail(error,
                    "online association uses a non-positive LiDAR weight");
      }
      gpu_ba::NativeBaLidarConstraint constraint;
      constraint.point3D_id = association.point3D_id;
      constraint.constraint_slot =
          static_cast<uint32_t>(association.association_id);
      constraint.physical_identity = association.association_id + 1;
      constraint.association_id = association.association_id;
      constraint.owner_image_id = association.owner_image_id;
      constraint.owner_point2D_idx = association.owner_point2D_idx;
      constraint.lidar_type =
          static_cast<uint8_t>(association.lidar_point_type);
      constraint.frozen_point3D_xyz = association.point3D_xyz;
      constraint.plane = association.plane_abcd;
      for (size_t axis = 0; axis < 3; ++axis) {
        constraint.lidar_xyz[axis] =
            static_cast<double>(association.plane.point[axis]);
      }
      constraint.weight = weight;
      constraint.search_range = association.search_range;
      intent.lidar_constraints.push_back(constraint);
    }

    output->Seal(std::move(association_output),
                 std::move(association_audit),
                 std::move(intent),
                 std::move(resolved_options));
    error->clear();
    return true;
  } catch (const std::exception& exception) {
    if (output != nullptr) {
      output->Reset();
    }
    return FailWithException(
        error, "online native BA intent build threw", exception.what());
  } catch (...) {
    if (output != nullptr) {
      output->Reset();
    }
    return Fail(error, "online native BA intent build threw an unknown exception");
  }
}

bool SummarizeOnlineLidarMaterialization(
    const gpu_ba::NativeHostSolveView& view,
    const OnlineLidarNativeBaSolveIntentBuildOutput& build_output,
    OnlineLidarMaterializationSummary* summary,
    std::string* error) noexcept {
  try {
    ResetMaterializationSummary(summary);
    if (error != nullptr) {
      error->clear();
    }
    if (summary == nullptr || error == nullptr) {
      return summary == nullptr
                 ? Fail(error, "online materialization summary output is null")
                 : false;
    }
    if (!build_output.built()) {
      return Fail(error, "online native BA build output is not built");
    }

    gpu_ba::NativeBaOnlineLidarIdentity expected_identity;
    if (!ValidateSummaryIdentityAndImages(
            view, build_output, &expected_identity, error)) {
      return false;
    }
    const gpu_ba::NativeBaSolveIntent& intent = build_output.intent();
    const OnlineLidarAssociationOutput& association_output =
        build_output.association_output();
    const OnlineLidarAssociationAudit& audit =
        build_output.association_audit();

    uint64_t input_point_count = 0;
    uint64_t selected_association_count = 0;
    if (!SizeToUint64(intent.explicit_variable_point_ids.size(),
                      "online selected Point3D count",
                      &input_point_count,
                      error) ||
        !SizeToUint64(association_output.associations.size(),
                      "online selected association count",
                      &selected_association_count,
                      error)) {
      return false;
    }
    if (audit.input_point_count != input_point_count ||
        audit.selected_association_count != selected_association_count ||
        association_output.associations.size() >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        association_output.associations.size() !=
            intent.lidar_constraints.size() ||
        !intent.explicit_constant_point_ids.empty() ||
        intent.point_policies.size() !=
            intent.explicit_variable_point_ids.size() ||
        !intent.fixed_pose_ids.empty() ||
        intent.translation_policies.size() != intent.active_image_ids.size()) {
      return Fail(error, "online intent selection shape is inconsistent");
    }

    for (size_t index = 0; index < intent.active_image_ids.size(); ++index) {
      if (intent.translation_policies[index].image_id !=
              intent.active_image_ids[index] ||
          intent.translation_policies[index].constant_mask != 0) {
        return Fail(error, "online intent translation policy is inconsistent");
      }
    }

    std::map<point3D_t, const gpu_ba::NativeBaPointPolicy*>
        point_policy_by_id;
    point3D_t previous_selected_point_id = 0;
    bool has_previous_selected_point_id = false;
    for (size_t index = 0;
         index < intent.explicit_variable_point_ids.size();
         ++index) {
      const point3D_t point3D_id = intent.explicit_variable_point_ids[index];
      if ((has_previous_selected_point_id &&
           point3D_id <= previous_selected_point_id) ||
          intent.point_policies[index].point3D_id != point3D_id ||
          intent.point_policies[index].constant ||
          intent.point_policies[index].config_role != 1 ||
          !point_policy_by_id.emplace(point3D_id,
                                      &intent.point_policies[index])
               .second) {
        return Fail(error, "online intent Point3D policy is inconsistent");
      }
      previous_selected_point_id = point3D_id;
      has_previous_selected_point_id = true;
    }

    std::map<uint64_t, const OnlineLidarAssociation*> association_by_id;
    std::map<point3D_t, const OnlineLidarAssociation*> association_by_point;
    point3D_t previous_associated_point_id = 0;
    bool has_previous_associated_point_id = false;
    for (size_t index = 0;
         index < association_output.associations.size();
         ++index) {
      const OnlineLidarAssociation& association =
          association_output.associations[index];
      if (!ValidateAssociationPlaneMetadata(association, error)) {
        return false;
      }
      if (association.association_id != static_cast<uint64_t>(index) ||
          association.association_id >
              static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
          (has_previous_associated_point_id &&
           association.point3D_id <= previous_associated_point_id) ||
          point_policy_by_id.count(association.point3D_id) != 1 ||
          association.attempt_id != audit.attempt_id ||
          association.pass_index != audit.pass_index ||
          association.map_version != audit.map_version ||
          association.max_scan_index != audit.max_scan_index ||
          association.snapshot_sha256 != audit.snapshot_sha256 ||
          association.geometry_sha256 != audit.geometry_sha256 ||
          !association_by_id.emplace(association.association_id, &association)
               .second ||
          !association_by_point.emplace(association.point3D_id, &association)
               .second) {
        return Fail(error, "online association build output is inconsistent");
      }
      const gpu_ba::NativeBaPointPolicy& point_policy =
          *point_policy_by_id.at(association.point3D_id);
      const bool expects_search_range =
          association.route == OnlineLidarAssociationRoute::KDTREE;
      if ((association.route != OnlineLidarAssociationRoute::PROJECTION &&
           association.route != OnlineLidarAssociationRoute::KDTREE) ||
          (association.route == OnlineLidarAssociationRoute::PROJECTION &&
           (association.lidar_point_type != LidarPointType::Proj ||
            association.has_search_range || association.search_range != 0.0)) ||
          (association.route == OnlineLidarAssociationRoute::KDTREE &&
           (association.lidar_point_type == LidarPointType::Proj ||
            !association.has_search_range)) ||
          !IsFiniteArray(association.point3D_xyz) ||
          !IsFiniteArray(association.plane_abcd) ||
          !std::isfinite(association.search_range) ||
          !std::isfinite(association.projection_camera_distance) ||
          !std::isfinite(association.projection_angle_score) ||
          point_policy.has_search_range != expects_search_range ||
          point_policy.search_range !=
              (expects_search_range ? association.search_range : 0.0)) {
        return Fail(error, "online association point range policy is inconsistent");
      }
      previous_associated_point_id = association.point3D_id;
      has_previous_associated_point_id = true;
    }
    for (const auto& point_policy : point_policy_by_id) {
      if (association_by_point.count(point_policy.first) == 0 &&
          (point_policy.second->has_search_range ||
           point_policy.second->search_range != 0.0)) {
        return Fail(error,
                    "online unassociated Point3D has a search range policy");
      }
    }

    std::map<uint64_t, const gpu_ba::NativeBaLidarConstraint*>
        intended_constraint_by_association;
    std::set<point3D_t> intended_constraint_points;
    for (const gpu_ba::NativeBaLidarConstraint& constraint :
         intent.lidar_constraints) {
      const auto association =
          association_by_id.find(constraint.association_id);
      if (association == association_by_id.end() ||
          constraint.constraint_slot != constraint.association_id ||
          constraint.physical_identity != constraint.association_id + 1 ||
          constraint.point3D_id != association->second->point3D_id ||
          constraint.owner_image_id != association->second->owner_image_id ||
          constraint.owner_point2D_idx !=
              association->second->owner_point2D_idx ||
          constraint.lidar_type !=
              static_cast<uint8_t>(association->second->lidar_point_type) ||
          constraint.frozen_point3D_xyz !=
              association->second->point3D_xyz ||
          constraint.plane != association->second->plane_abcd ||
          constraint.search_range != association->second->search_range ||
          !IsFiniteArray(constraint.frozen_point3D_xyz) ||
          !IsFiniteArray(constraint.plane) ||
          !IsFiniteArray(constraint.lidar_xyz) ||
          !std::isfinite(constraint.weight) || !(constraint.weight > 0.0) ||
          !std::isfinite(constraint.search_range) ||
          !intended_constraint_by_association
               .emplace(constraint.association_id, &constraint)
               .second ||
          !intended_constraint_points.insert(constraint.point3D_id).second) {
        return Fail(error, "online intended LiDAR constraint is inconsistent");
      }
      for (size_t axis = 0; axis < 3; ++axis) {
        if (constraint.lidar_xyz[axis] !=
            static_cast<double>(association->second->plane.point[axis])) {
          return Fail(error,
                      "online intended LiDAR plane point is inconsistent");
        }
      }
    }

    const std::vector<gpu_ba::LidarConstraintRecord>& materialized_constraints =
        view.LidarConstraints();
    if (materialized_constraints.size() != association_by_id.size()) {
      return Fail(error, "online materialized LiDAR constraint count differs");
    }
    const auto catalog_points = view.catalog.points();
    std::map<uint32_t, const gpu_ba::LidarConstraintRecord*>
        materialized_constraint_by_slot;
    std::map<uint64_t, const gpu_ba::LidarConstraintRecord*>
        materialized_constraint_by_association;
    std::set<uint64_t> materialized_constraint_points;
    for (const gpu_ba::LidarConstraintRecord& constraint :
         materialized_constraints) {
      const auto intended =
          intended_constraint_by_association.find(constraint.association_id);
      const gpu_ba::HostBaImageSlot* owner =
          view.catalog.FindImageById(constraint.owner_image_id);
      const gpu_ba::HostBaObservationSlot* owner_observation =
          view.catalog.FindObservation(constraint.owner_image_id,
                                       constraint.owner_point2D_idx);
      if (intended == intended_constraint_by_association.end() ||
          constraint.point_slot >= catalog_points.size ||
          !catalog_points[constraint.point_slot].header.alive ||
          catalog_points[constraint.point_slot].point3D_id !=
              intended->second->point3D_id ||
          owner == nullptr || !owner->header.alive || !owner->registered ||
          owner_observation == nullptr ||
          !owner_observation->header.alive ||
          owner_observation->image_slot != owner->header.slot ||
          owner_observation->point_slot != constraint.point_slot ||
          !SameNativeConstraint(*intended->second, constraint) ||
          !materialized_constraint_by_slot
               .emplace(constraint.constraint_slot, &constraint)
               .second ||
          !materialized_constraint_by_association
               .emplace(constraint.association_id, &constraint)
               .second ||
          !materialized_constraint_points
               .insert(catalog_points[constraint.point_slot].point3D_id)
               .second) {
        return Fail(error, "online materialized LiDAR constraint is inconsistent");
      }
    }

    OnlineLidarMaterializationSummary local_summary;
    local_summary.per_image_materialized_count =
        audit.preliminary_selected_count_by_image;
    for (auto& image_count : local_summary.per_image_materialized_count) {
      image_count.second = 0;
    }
    const std::vector<gpu_ba::ResidualOrdinal>& ordinals =
        view.ResidualOrdinals();
    std::vector<uint8_t> source_indices_seen(ordinals.size(), 0);
    std::set<uint32_t> materialized_lidar_slots;
    std::set<uint64_t> materialized_lidar_association_ids;
    std::set<uint64_t> materialized_lidar_point_ids;
    std::set<uint32_t> materialized_visual_slots;
    const auto catalog_observations = view.catalog.observations();
    for (size_t index = 0; index < ordinals.size(); ++index) {
      const gpu_ba::ResidualOrdinal& ordinal = ordinals[index];
      if (ordinal.execution_ordinal != static_cast<uint64_t>(index) ||
          ordinal.source_insertion_index >= ordinals.size() ||
          source_indices_seen[ordinal.source_insertion_index] != 0) {
        return Fail(error, "online materialized residual ordinals are invalid");
      }
      source_indices_seen[ordinal.source_insertion_index] = 1;
      if (ordinal.kind == gpu_ba::ResidualKind::kVisual) {
        if (ordinal.source_slot >= catalog_observations.size ||
            !catalog_observations[ordinal.source_slot].header.alive ||
            ordinal.association_id != gpu_ba::kBaGraphInvalidAssociationId ||
            ordinal.owner_image_id != gpu_ba::kBaGraphInvalidSlot ||
            ordinal.owner_point2D_idx != gpu_ba::kBaGraphInvalidSlot ||
            !materialized_visual_slots.insert(ordinal.source_slot).second) {
          return Fail(error,
                      "online materialized visual ordinal is inconsistent");
        }
        if (!IncrementUint64(
                &local_summary.materialized_visual_residual_count,
                "online materialized visual residual count",
                error)) {
          return false;
        }
        continue;
      }
      if (ordinal.kind != gpu_ba::ResidualKind::kLidar) {
        return Fail(error, "online materialized residual kind is invalid");
      }
      const auto materialized =
          materialized_constraint_by_slot.find(ordinal.source_slot);
      if (materialized == materialized_constraint_by_slot.end()) {
        return Fail(error,
                    "online LiDAR ordinal references a missing constraint");
      }
      const gpu_ba::LidarConstraintRecord& constraint = *materialized->second;
      if (ordinal.physical_identity != constraint.physical_identity ||
          ordinal.association_id != constraint.association_id ||
          ordinal.owner_image_id != constraint.owner_image_id ||
          ordinal.owner_point2D_idx != constraint.owner_point2D_idx ||
          !materialized_lidar_slots.insert(constraint.constraint_slot).second ||
          !materialized_lidar_association_ids
               .insert(constraint.association_id)
               .second ||
          !materialized_lidar_point_ids
               .insert(catalog_points[constraint.point_slot].point3D_id)
               .second) {
        return Fail(error, "online LiDAR ordinal provenance is inconsistent");
      }
      auto image_count = local_summary.per_image_materialized_count.find(
          constraint.owner_image_id);
      if (image_count == local_summary.per_image_materialized_count.end()) {
        return Fail(error,
                    "online materialized owner is outside the active window");
      }
      if (!IncrementUint64(&image_count->second,
                           "online per-image materialized count",
                           error) ||
          !IncrementUint64(
              &local_summary.materialized_lidar_residual_count,
              "online materialized LiDAR residual count",
              error)) {
        return false;
      }
      if (constraint.owner_image_id == expected_identity.trigger_image_id &&
          !IncrementUint64(&local_summary.trigger_materialized_count,
                           "online trigger materialized count",
                           error)) {
        return false;
      }
      local_summary.materialized_association_ids.push_back(
          constraint.association_id);
    }

    uint64_t ordinal_count = 0;
    if (!SizeToUint64(ordinals.size(),
                      "online materialized ordinal count",
                      &ordinal_count,
                      error) ||
        view.ResidualBlockCount() != ordinal_count ||
        local_summary.materialized_visual_residual_count !=
            view.VisualObservationSlots().size() ||
        local_summary.materialized_lidar_residual_count !=
            materialized_constraints.size() ||
        materialized_lidar_slots.size() != materialized_constraints.size() ||
        materialized_lidar_association_ids.size() != association_by_id.size() ||
        materialized_lidar_point_ids.size() != association_by_point.size()) {
      return Fail(error, "online materialized residual coverage is incomplete");
    }
    std::sort(local_summary.materialized_association_ids.begin(),
              local_summary.materialized_association_ids.end());
    for (size_t index = 0;
         index < local_summary.materialized_association_ids.size();
         ++index) {
      if (local_summary.materialized_association_ids[index] !=
          static_cast<uint64_t>(index)) {
        return Fail(error,
                    "online materialized association IDs are not canonical");
      }
    }
    local_summary.materialized_association_count =
        local_summary.materialized_lidar_residual_count;
    if (local_summary.materialized_association_count !=
            audit.selected_association_count ||
        local_summary.per_image_materialized_count !=
            audit.preliminary_selected_count_by_image ||
        local_summary.trigger_materialized_count !=
            audit.trigger_preliminary_selected_count) {
      return Fail(error, "online materialized association counts differ");
    }
    local_summary.materialized_online_lidar_identity = expected_identity;
    local_summary.solver_evaluation_pending = true;
    *summary = std::move(local_summary);
    error->clear();
    return true;
  } catch (const std::exception& exception) {
    ResetMaterializationSummary(summary);
    return FailWithException(error,
                             "online materialization summary threw",
                             exception.what());
  } catch (...) {
    ResetMaterializationSummary(summary);
    return Fail(error,
                "online materialization summary threw an unknown exception");
  }
}

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED
