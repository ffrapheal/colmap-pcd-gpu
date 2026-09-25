#include "sfm/online_local_ba_executor.h"

#ifdef GPU_BA_CUDA_ENABLED

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

#include "base/camera.h"
#include "base/correspondence_graph.h"
#include "base/image.h"
#include "base/point2d.h"
#include "base/point3d.h"
#include "gpu_ba/host_problem_store.h"
#include "gpu_ba/snapshot.h"

namespace colmap {

const char* ToString(const OnlineLocalBaMode mode) noexcept {
  switch (mode) {
    case OnlineLocalBaMode::BOOTSTRAP:
      return "BOOTSTRAP";
    case OnlineLocalBaMode::SINGLE:
      return "SINGLE";
    case OnlineLocalBaMode::CATCHUP:
      return "CATCHUP";
    case OnlineLocalBaMode::DUAL_PASS1:
      return "DUAL_PASS1";
    case OnlineLocalBaMode::DUAL_PASS2:
      return "DUAL_PASS2";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineLocalBaFailureReason reason) noexcept {
  switch (reason) {
    case OnlineLocalBaFailureReason::NONE:
      return "NONE";
    case OnlineLocalBaFailureReason::INVALID_REQUEST:
      return "INVALID_REQUEST";
    case OnlineLocalBaFailureReason::CANDIDATE_UNAVAILABLE:
      return "CANDIDATE_UNAVAILABLE";
    case OnlineLocalBaFailureReason::FIXED_INTRINSICS_MISMATCH:
      return "FIXED_INTRINSICS_MISMATCH";
    case OnlineLocalBaFailureReason::INTENT_BUILD_FAILED:
      return "INTENT_BUILD_FAILED";
    case OnlineLocalBaFailureReason::INTENT_AUDIT_FAILED:
      return "INTENT_AUDIT_FAILED";
    case OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM:
      return "TRIGGER_BELOW_MINIMUM";
    case OnlineLocalBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED:
      return "RETRY_EVIDENCE_NOT_INCREASED";
    case OnlineLocalBaFailureReason::NATIVE_RUNNER_FAILED:
      return "NATIVE_RUNNER_FAILED";
    case OnlineLocalBaFailureReason::SOLVE_FAILED:
      return "SOLVE_FAILED";
    case OnlineLocalBaFailureReason::BACKEND_CONTRACT_FAILED:
      return "BACKEND_CONTRACT_FAILED";
    case OnlineLocalBaFailureReason::TERMINATION_UNUSABLE:
      return "TERMINATION_UNUSABLE";
    case OnlineLocalBaFailureReason::NONFINITE_SOLVER_RESULT:
      return "NONFINITE_SOLVER_RESULT";
    case OnlineLocalBaFailureReason::RESIDUAL_AUDIT_MISMATCH:
      return "RESIDUAL_AUDIT_MISMATCH";
    case OnlineLocalBaFailureReason::CANDIDATE_STATE_INVALID:
      return "CANDIDATE_STATE_INVALID";
    case OnlineLocalBaFailureReason::POSTPROCESS_ADAPTER_MISSING:
      return "POSTPROCESS_ADAPTER_MISSING";
    case OnlineLocalBaFailureReason::POSTPROCESS_MERGE_FAILED:
      return "POSTPROCESS_MERGE_FAILED";
    case OnlineLocalBaFailureReason::POSTPROCESS_COMPLETE_FAILED:
      return "POSTPROCESS_COMPLETE_FAILED";
    case OnlineLocalBaFailureReason::POSTPROCESS_FILTER_FAILED:
      return "POSTPROCESS_FILTER_FAILED";
    case OnlineLocalBaFailureReason::DUAL_REQUEST_MISMATCH:
      return "DUAL_REQUEST_MISMATCH";
    case OnlineLocalBaFailureReason::DUAL_PASS1_FAILED:
      return "DUAL_PASS1_FAILED";
    case OnlineLocalBaFailureReason::DUAL_PASS2_FAILED:
      return "DUAL_PASS2_FAILED";
    case OnlineLocalBaFailureReason::INTERNAL_ERROR:
      return "INTERNAL_ERROR";
  }
  return "UNKNOWN";
}

bool OnlineLocalBaFixedIntrinsicsSnapshot::operator==(
    const OnlineLocalBaFixedIntrinsicsSnapshot& other) const {
  return camera_id == other.camera_id && model_id == other.model_id &&
         width == other.width && height == other.height &&
         has_prior_focal_length == other.has_prior_focal_length &&
         parameters == other.parameters;
}

bool OnlineLocalBaFixedIntrinsicsSnapshot::operator!=(
    const OnlineLocalBaFixedIntrinsicsSnapshot& other) const {
  return !(*this == other);
}

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

bool IsDualMode(const OnlineLocalBaMode mode) noexcept {
  return mode == OnlineLocalBaMode::DUAL_PASS1 ||
         mode == OnlineLocalBaMode::DUAL_PASS2;
}

bool IsFiniteCandidate(const Reconstruction& candidate,
                       std::string* error) {
  for (const auto& camera_item : candidate.Cameras()) {
    const Camera& camera = camera_item.second;
    if (!camera.VerifyParams() || camera.Width() == 0 || camera.Height() == 0) {
      *error = "candidate contains invalid camera intrinsics";
      return false;
    }
    for (const double parameter : camera.Params()) {
      if (!std::isfinite(parameter)) {
        *error = "candidate contains non-finite camera intrinsics";
        return false;
      }
    }
  }
  for (const auto& image_item : candidate.Images()) {
    const Image& image = image_item.second;
    if (!image.Qvec().array().isFinite().all() ||
        !image.Tvec().array().isFinite().all() ||
        !(image.Qvec().norm() > std::numeric_limits<double>::epsilon())) {
      *error = "candidate contains a non-finite or zero image pose";
      return false;
    }
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.XY().array().isFinite().all()) {
        *error = "candidate contains a non-finite image observation";
        return false;
      }
    }
  }
  for (const auto& point_item : candidate.Points3D()) {
    const Point3D& point3D = point_item.second;
    if (!point3D.XYZ().array().isFinite().all() ||
        !std::isfinite(point3D.Error())) {
      *error = "candidate contains a non-finite 3D point";
      return false;
    }
  }
  return true;
}

class CanonicalStateWriter {
 public:
  template <typename Integer>
  void IntegerValue(const Integer value) {
    typedef typename std::make_unsigned<Integer>::type Unsigned;
    const Unsigned converted = static_cast<Unsigned>(value);
    std::array<char, sizeof(Unsigned)> encoded;
    for (size_t index = 0; index < sizeof(Unsigned); ++index) {
      encoded[index] =
          static_cast<char>((converted >> (index * 8)) & Unsigned(0xff));
    }
    bytes_.append(encoded.data(), encoded.size());
  }

  void BoolValue(const bool value) { IntegerValue<uint8_t>(value ? 1 : 0); }

  void DoubleValue(const double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "double must be 64-bit");
    std::memcpy(&bits, &value, sizeof(bits));
    IntegerValue(bits);
  }

  void StringValue(const std::string& value) {
    IntegerValue<uint64_t>(value.size());
    bytes_.append(value);
  }

  const std::string& bytes() const noexcept { return bytes_; }

 private:
  std::string bytes_;
};

template <typename Map>
std::vector<typename Map::key_type> SortedKeys(const Map& values) {
  std::vector<typename Map::key_type> keys;
  keys.reserve(values.size());
  for (const auto& item : values) keys.push_back(item.first);
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string VisualStateSha256(const Reconstruction& reconstruction) {
  CanonicalStateWriter writer;
  const ReconstructionCanonicalVersion version =
      reconstruction.CanonicalVersion();
  writer.IntegerValue(version.structure_owner_epoch);
  writer.IntegerValue(version.structure_revision);
  writer.IntegerValue(version.publish_version);
  writer.IntegerValue(version.correspondence_graph_identity);

  const std::vector<camera_t> camera_ids = SortedKeys(reconstruction.Cameras());
  writer.IntegerValue<uint64_t>(camera_ids.size());
  for (const camera_t camera_id : camera_ids) {
    const Camera& camera = reconstruction.Camera(camera_id);
    writer.IntegerValue(camera_id);
    writer.IntegerValue(camera.ModelId());
    writer.IntegerValue<uint64_t>(camera.Width());
    writer.IntegerValue<uint64_t>(camera.Height());
    writer.BoolValue(camera.HasPriorFocalLength());
    writer.IntegerValue<uint64_t>(camera.Params().size());
    for (const double parameter : camera.Params()) {
      writer.DoubleValue(parameter);
    }
  }

  const std::vector<image_t> image_ids = SortedKeys(reconstruction.Images());
  writer.IntegerValue<uint64_t>(image_ids.size());
  for (const image_t image_id : image_ids) {
    const Image& image = reconstruction.Image(image_id);
    writer.IntegerValue(image_id);
    writer.IntegerValue(image.CameraId());
    writer.StringValue(image.Name());
    writer.BoolValue(image.IsRegistered());
    for (int index = 0; index < image.Qvec().size(); ++index) {
      writer.DoubleValue(image.Qvec()(index));
    }
    for (int index = 0; index < image.Tvec().size(); ++index) {
      writer.DoubleValue(image.Tvec()(index));
    }
    writer.IntegerValue<uint64_t>(image.Points2D().size());
    for (const Point2D& point2D : image.Points2D()) {
      writer.DoubleValue(point2D.X());
      writer.DoubleValue(point2D.Y());
      writer.IntegerValue(point2D.Point3DId());
    }
  }

  const std::vector<point3D_t> point3D_ids =
      SortedKeys(reconstruction.Points3D());
  writer.IntegerValue<uint64_t>(point3D_ids.size());
  for (const point3D_t point3D_id : point3D_ids) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    writer.IntegerValue(point3D_id);
    for (int index = 0; index < point3D.XYZ().size(); ++index) {
      writer.DoubleValue(point3D.XYZ()(index));
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      writer.IntegerValue(point3D.Color(channel));
    }
    writer.DoubleValue(point3D.Error());
    writer.IntegerValue<uint64_t>(point3D.Track().Length());
    for (const TrackElement& element : point3D.Track().Elements()) {
      writer.IntegerValue(element.image_id);
      writer.IntegerValue(element.point2D_idx);
    }
  }

  writer.IntegerValue<uint64_t>(reconstruction.RegImageIds().size());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    writer.IntegerValue(image_id);
  }
  return gpu_ba::Sha256Hex(writer.bytes());
}

bool DecodeLowercaseSha256(const std::string& encoded,
                           std::array<uint8_t, 32>* decoded) {
  decoded->fill(0);
  if (encoded.size() != decoded->size() * 2) return false;
  bool nonzero = false;
  for (size_t index = 0; index < decoded->size(); ++index) {
    const auto decode_nibble = [](const char value, uint8_t* nibble) {
      if (value >= '0' && value <= '9') {
        *nibble = static_cast<uint8_t>(value - '0');
        return true;
      }
      if (value >= 'a' && value <= 'f') {
        *nibble = static_cast<uint8_t>(value - 'a' + 10);
        return true;
      }
      return false;
    };
    uint8_t high = 0;
    uint8_t low = 0;
    if (!decode_nibble(encoded[index * 2], &high) ||
        !decode_nibble(encoded[index * 2 + 1], &low)) {
      decoded->fill(0);
      return false;
    }
    (*decoded)[index] = static_cast<uint8_t>((high << 4) | low);
    nonzero = nonzero || (*decoded)[index] != 0;
  }
  return nonzero;
}

bool SameAssociationOptions(const OnlineLidarAssociationOptions& first,
                            const OnlineLidarAssociationOptions& second) {
  return first.local_lidar_kdtree_only == second.local_lidar_kdtree_only &&
         first.min_proj_num == second.min_proj_num &&
         first.kdtree_max_search_range == second.kdtree_max_search_range &&
         first.kdtree_min_search_range == second.kdtree_min_search_range &&
         first.search_range_drop_speed == second.search_range_drop_speed &&
         first.ba_match_features_threshold ==
             second.ba_match_features_threshold;
}

bool SameProjectionOptions(const lidar::PcdProjectionOptions& first,
                           const lidar::PcdProjectionOptions& second) {
  return first.ba_pointcloud_path == second.ba_pointcloud_path &&
         first.initial_mesh_depth_path == second.initial_mesh_depth_path &&
         first.initial_mesh_depth_generator_path ==
             second.initial_mesh_depth_generator_path &&
         first.initial_mesh_path == second.initial_mesh_path &&
         first.initial_mesh_depth_dataset_path ==
             second.initial_mesh_depth_dataset_path &&
         first.initial_mesh_depth_intrinsics_path ==
             second.initial_mesh_depth_intrinsics_path &&
         first.initial_mesh_depth_fx == second.initial_mesh_depth_fx &&
         first.initial_mesh_depth_fy == second.initial_mesh_depth_fy &&
         first.initial_mesh_depth_cx == second.initial_mesh_depth_cx &&
         first.initial_mesh_depth_cy == second.initial_mesh_depth_cy &&
         first.initial_mesh_depth_pnp_max_error ==
             second.initial_mesh_depth_pnp_max_error &&
         first.depth_image_scale == second.depth_image_scale &&
         first.if_save_depth_image == second.if_save_depth_image &&
         first.depth_image_folder == second.depth_image_folder &&
         first.original_image_folder == second.original_image_folder &&
         first.if_save_lidar_frame == second.if_save_lidar_frame &&
         first.lidar_frame_folder == second.lidar_frame_folder &&
         first.max_proj_scale == second.max_proj_scale &&
         first.min_proj_scale == second.min_proj_scale &&
         first.min_proj_dist == second.min_proj_dist &&
         first.submap_length == second.submap_length &&
         first.submap_width == second.submap_width &&
         first.submap_height == second.submap_height &&
         first.choose_meter == second.choose_meter &&
         first.min_lidar_proj_dist == second.min_lidar_proj_dist;
}

bool ValidateModePolicy(const OnlineLocalBaRequest& request,
                        std::string* error) {
  switch (request.mode) {
    case OnlineLocalBaMode::BOOTSTRAP:
    case OnlineLocalBaMode::SINGLE:
    case OnlineLocalBaMode::CATCHUP:
      if (request.pass_index != 1 || !request.allow_postprocess) {
        *error = "single-pass local BA requires pass_index=1 and postprocess";
        return false;
      }
      return true;
    case OnlineLocalBaMode::DUAL_PASS1:
      if (request.pass_index != 1 || request.allow_postprocess) {
        *error = "DUAL_PASS1 requires pass_index=1 and forbids postprocess";
        return false;
      }
      return true;
    case OnlineLocalBaMode::DUAL_PASS2:
      if (request.pass_index != 2 || !request.allow_postprocess) {
        *error = "DUAL_PASS2 requires pass_index=2 and postprocess";
        return false;
      }
      return true;
  }
  *error = "local BA mode is invalid";
  return false;
}

bool ValidateRequest(const BundleAdjustmentOptions& options,
                     const OnlineLocalBaRequest& request,
                     const Reconstruction& candidate,
                     const OnlineLocalBaExecutorDependencies& dependencies,
                     OnlineLocalBaFailureReason* failure_reason,
                     std::string* error) {
  *failure_reason = OnlineLocalBaFailureReason::INVALID_REQUEST;
  if (!ValidateModePolicy(request, error)) return false;
  if (request.attempt_id == 0 || request.owner_epoch == 0 ||
      request.topology_revision == 0 || request.selection_revision == 0) {
    *error = "local BA request generations must be non-zero";
    return false;
  }
  if (request.trigger_image_id == 0 ||
      request.trigger_image_id == kInvalidImageId ||
      request.ordered_frozen_image_ids.empty() ||
      request.ordered_frozen_image_ids.size() >
          kOnlineLocalBaMaximumImageCount) {
    *error = "local BA frozen window or trigger is invalid";
    return false;
  }
  std::set<image_t> image_ids;
  size_t trigger_count = 0;
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    if (image_id == 0 || image_id == kInvalidImageId ||
        !image_ids.insert(image_id).second ||
        !candidate.ExistsImage(image_id) ||
        !candidate.Image(image_id).IsRegistered()) {
      *error = "local BA frozen window contains an invalid image";
      return false;
    }
    trigger_count += image_id == request.trigger_image_id ? 1 : 0;
  }
  if (trigger_count != 1) {
    *error = "local BA trigger must occur exactly once in the frozen window";
    return false;
  }
  if (request.point3D_ids.empty()) {
    *error = "local BA point selection is empty";
    return false;
  }
  std::set<point3D_t> point3D_ids;
  for (const point3D_t point3D_id : request.point3D_ids) {
    if (point3D_id == kInvalidPoint3DId ||
        !point3D_ids.insert(point3D_id).second ||
        !candidate.ExistsPoint3D(point3D_id)) {
      *error = "local BA point selection contains an invalid point";
      return false;
    }
  }
  if (candidate.CanonicalVersion() != request.expected_candidate_version ||
      candidate.StructureOwnerEpoch() != request.owner_epoch ||
      candidate.StructureRevision() != request.topology_revision) {
    *error = "local BA candidate version differs from the frozen request";
    return false;
  }
  if (request.map_snapshot == nullptr || request.expected_map_version == 0 ||
      request.expected_max_scan_index == 0 ||
      request.expected_map_version != request.expected_max_scan_index ||
      request.map_snapshot->Version() != request.expected_map_version ||
      request.map_snapshot->MaxScanIndex() != request.expected_max_scan_index ||
      request.map_snapshot->Frame() !=
          lidar::LidarCoordinateFrame::COLMAP_WORLD ||
      request.map_snapshot->SnapshotSha256() !=
          request.expected_snapshot_sha256 ||
      request.map_snapshot->GeometrySha256() !=
          request.expected_geometry_sha256) {
    *error = "local BA LiDAR snapshot identity is invalid";
    return false;
  }
  std::array<uint8_t, 32> decoded;
  if (!DecodeLowercaseSha256(request.expected_snapshot_sha256, &decoded) ||
      !DecodeLowercaseSha256(request.expected_geometry_sha256, &decoded)) {
    *error = "local BA LiDAR snapshot hashes are invalid";
    return false;
  }
  if (!request.all_camera_poses_variable ||
      !request.constant_camera_pose_ids.empty()) {
    *error = "local BA requires every frozen camera pose to be variable";
    return false;
  }
  if (request.requested_backend != "custom_cuda" ||
      request.requested_problem_source !=
          gpu_ba::CudaProblemSource::kNativeGraph ||
      request.fallback_allowed || options.ba_backend != "custom_cuda" ||
      options.ba_fallback_to_ceres ||
      options.ba_cuda_problem_source !=
          gpu_ba::CudaProblemSource::kNativeGraph ||
      options.ba_cuda_host_problem_store != "host_prepared_store" ||
      options.refine_focal_length || options.refine_principal_point ||
      options.refine_extra_params || !options.refine_extrinsics ||
      !options.if_add_lidar_constraint) {
    *error = "local BA requires custom_cuda/native_graph, fixed intrinsics, "
             "variable poses, and no fallback";
    return false;
  }
  if (request.require_trigger_residual_increase &&
      request.mode != OnlineLocalBaMode::CATCHUP) {
    *error = "residual-increase gate requires CATCHUP mode";
    return false;
  }
  if (request.allow_postprocess && dependencies.postprocess == nullptr) {
    *failure_reason =
        OnlineLocalBaFailureReason::POSTPROCESS_ADAPTER_MISSING;
    *error = "local BA postprocess adapter is missing";
    return false;
  }
  std::vector<OnlineLocalBaFixedIntrinsicsSnapshot> current_intrinsics;
  if (!CaptureOnlineLocalBaFixedIntrinsics(candidate,
                                           request.ordered_frozen_image_ids,
                                           &current_intrinsics,
                                           error) ||
      current_intrinsics != request.fixed_intrinsics) {
    *failure_reason = OnlineLocalBaFailureReason::FIXED_INTRINSICS_MISMATCH;
    if (error->empty()) {
      *error = "local BA fixed intrinsics differ from the frozen snapshot";
    }
    return false;
  }
  return true;
}

class DefaultOnlineLocalBaIntentBuilder final
    : public OnlineLocalBaIntentBuilder {
 public:
  bool Build(const Reconstruction& candidate,
             const BundleAdjustmentOptions& options,
             const OnlineLocalBaRequest& request,
             OnlineLocalBaIntentBuildResult* result,
             std::string* error) override {
    if (result == nullptr || error == nullptr) return false;
    *result = OnlineLocalBaIntentBuildResult();
    error->clear();
    OnlineLidarAssociationRequest association_request;
    association_request.attempt_id = request.attempt_id;
    association_request.pass_index = request.pass_index;
    association_request.trigger_image_id = request.trigger_image_id;
    association_request.ordered_frozen_image_ids =
        request.ordered_frozen_image_ids;
    association_request.point3D_ids = request.point3D_ids;
    association_request.expected_map_version = request.expected_map_version;
    association_request.expected_max_scan_index =
        request.expected_max_scan_index;
    association_request.expected_snapshot_sha256 =
        request.expected_snapshot_sha256;
    association_request.expected_geometry_sha256 =
        request.expected_geometry_sha256;
    association_request.options = request.association_options;
    association_request.projection_options = request.projection_options;
    association_request.snapshot = request.map_snapshot;

    OnlineLidarNativeBaSolveIntentBuildOutput build_output;
    result->trigger_gate.minimum_count =
        kOnlineLocalBaMinimumTriggerLidarResidualCount;
    if (request.require_trigger_residual_increase) {
      result->trigger_gate.previous_count =
          request.previous_trigger_lidar_constraint_count;
      result->trigger_gate.minimum_increase =
          kOnlineLocalBaMinimumTriggerLidarResidualCount;
    }
    if (!BuildOnlineLidarNativeBaSolveIntent(candidate,
                                             options,
                                             association_request,
                                             request.owner_epoch,
                                             request.topology_revision,
                                             request.selection_revision,
                                             gpu_ba::BaKind::kLocal,
                                             &build_output,
                                             error,
                                             &result->trigger_gate)) {
      return false;
    }
    if (result->trigger_gate.checked && !result->trigger_gate.passed) {
      return true;
    }
    result->association_audit = build_output.association_audit();
    result->intent = build_output.intent();
    return true;
  }
};

bool AddCount(const image_t image_id,
              std::map<image_t, uint64_t>* counts,
              std::string* error) {
  const auto item = counts->find(image_id);
  if (item == counts->end()) {
    *error = "LiDAR residual owner is outside the frozen window";
    return false;
  }
  if (item->second == std::numeric_limits<uint64_t>::max()) {
    *error = "LiDAR residual owner count overflowed";
    return false;
  }
  ++item->second;
  return true;
}

bool CountSubmittedLidarConstraints(
    const OnlineLocalBaRequest& request,
    const gpu_ba::NativeBaSolveIntent& intent,
    std::map<image_t, uint64_t>* per_image,
    uint64_t* total,
    std::string* error) {
  per_image->clear();
  for (const image_t image_id : request.ordered_frozen_image_ids) {
    per_image->emplace(image_id, 0);
  }
  for (const gpu_ba::NativeBaLidarConstraint& constraint :
       intent.lidar_constraints) {
    if (!AddCount(constraint.owner_image_id, per_image, error)) return false;
  }
  *total = static_cast<uint64_t>(intent.lidar_constraints.size());
  return true;
}

bool ValidateAndSummarizeIntent(
    const Reconstruction& candidate,
    const OnlineLocalBaRequest& request,
    const OnlineLocalBaIntentBuildResult& build_result,
    OnlineLocalBaIntentSummary* summary,
    std::map<image_t, uint64_t>* per_image_submitted,
    uint64_t* submitted_total,
    std::string* error) {
  const OnlineLidarAssociationAudit& association =
      build_result.association_audit;
  const gpu_ba::NativeBaSolveIntent& intent = build_result.intent;
  if (association.attempt_id != request.attempt_id ||
      association.pass_index != request.pass_index ||
      association.trigger_image_id != request.trigger_image_id ||
      association.map_version != request.expected_map_version ||
      association.max_scan_index != request.expected_max_scan_index ||
      association.snapshot_sha256 != request.expected_snapshot_sha256 ||
      association.geometry_sha256 != request.expected_geometry_sha256) {
    *error = "association audit identity differs from the local BA request";
    return false;
  }
  std::array<uint8_t, 32> snapshot_sha256;
  std::array<uint8_t, 32> geometry_sha256;
  std::array<uint8_t, 32> association_sha256;
  if (!DecodeLowercaseSha256(request.expected_snapshot_sha256,
                             &snapshot_sha256) ||
      !DecodeLowercaseSha256(request.expected_geometry_sha256,
                             &geometry_sha256) ||
      !DecodeLowercaseSha256(association.association_sha256,
                             &association_sha256)) {
    *error = "association audit contains an invalid SHA256 identity";
    return false;
  }
  if (intent.owner_epoch != request.owner_epoch ||
      intent.reconstruction_identity !=
          reinterpret_cast<uintptr_t>(&candidate) ||
      intent.expected_topology_revision != request.topology_revision ||
      intent.selection_revision != request.selection_revision ||
      intent.kind != gpu_ba::BaKind::kLocal ||
      intent.visual_observation_scope !=
          gpu_ba::NativeBaVisualObservationScope::kActiveImagesOnly ||
      !intent.config.resolved ||
      intent.config.config_generation != request.selection_revision ||
      intent.active_image_ids != request.ordered_frozen_image_ids ||
      !intent.fixed_pose_ids.empty() ||
      intent.translation_policies.size() != intent.active_image_ids.size() ||
      intent.lidar_map_generation != request.expected_map_version ||
      intent.lidar_match_config_generation != request.selection_revision ||
      !intent.online_lidar_identity.valid ||
      intent.online_lidar_identity.trigger_image_id !=
          request.trigger_image_id ||
      intent.online_lidar_identity.map_version !=
          request.expected_map_version ||
      intent.online_lidar_identity.max_scan_index !=
          request.expected_max_scan_index ||
      intent.online_lidar_identity.snapshot_sha256 != snapshot_sha256 ||
      intent.online_lidar_identity.geometry_sha256 != geometry_sha256 ||
      intent.online_lidar_identity.association_sha256 != association_sha256) {
    *error = "native intent identity or fixed policy differs from the request";
    return false;
  }
  for (size_t index = 0; index < intent.active_image_ids.size(); ++index) {
    if (intent.translation_policies[index].image_id !=
            intent.active_image_ids[index] ||
        intent.translation_policies[index].constant_mask != 0) {
      *error = "native intent contains a fixed or malformed camera pose";
      return false;
    }
  }
  std::set<camera_t> expected_camera_ids;
  for (const OnlineLocalBaFixedIntrinsicsSnapshot& intrinsics :
       request.fixed_intrinsics) {
    expected_camera_ids.insert(intrinsics.camera_id);
  }
  std::set<camera_t> intent_camera_ids;
  for (const gpu_ba::NativeBaCameraPolicy& camera : intent.camera_policies) {
    if (!camera.constant || !camera.fixed_parameter_indices.empty() ||
        !intent_camera_ids.insert(camera.camera_id).second) {
      *error = "native intent contains variable or malformed intrinsics";
      return false;
    }
  }
  if (intent_camera_ids != expected_camera_ids) {
    *error = "native intent fixed camera set differs from the snapshot";
    return false;
  }
  if (!CountSubmittedLidarConstraints(
          request, intent, per_image_submitted, submitted_total, error)) {
    return false;
  }
  const auto trigger = per_image_submitted->find(request.trigger_image_id);
  if (trigger == per_image_submitted->end() ||
      association.selected_association_count != *submitted_total ||
      association.preliminary_selected_count_by_image != *per_image_submitted ||
      association.trigger_preliminary_selected_count != trigger->second) {
    *error = "association and native intent residual counts differ";
    return false;
  }

  summary->owner_epoch = intent.owner_epoch;
  summary->reconstruction_identity = intent.reconstruction_identity;
  summary->topology_revision = intent.expected_topology_revision;
  summary->selection_revision = intent.selection_revision;
  summary->config_generation = intent.config.config_generation;
  summary->lidar_map_generation = intent.lidar_map_generation;
  summary->lidar_match_config_generation =
      intent.lidar_match_config_generation;
  summary->active_image_ids.assign(intent.active_image_ids.begin(),
                                   intent.active_image_ids.end());
  summary->fixed_pose_count = intent.fixed_pose_ids.size();
  summary->translation_policy_count = intent.translation_policies.size();
  summary->fixed_camera_count = intent.camera_policies.size();
  summary->variable_point_count = intent.explicit_variable_point_ids.size();
  summary->constant_point_count = intent.explicit_constant_point_ids.size();
  summary->lidar_constraint_count = intent.lidar_constraints.size();
  summary->local_kind = true;
  summary->active_images_only = true;
  summary->all_poses_variable = true;
  summary->all_intrinsics_fixed = true;
  summary->config_resolved = true;
  return true;
}

bool BuildBundleAdjustmentConfig(const gpu_ba::NativeBaSolveIntent& intent,
                                 BundleAdjustmentConfig* config,
                                 std::string* error) {
  *config = BundleAdjustmentConfig();
  for (const image_t image_id : intent.active_image_ids) {
    config->AddImage(image_id);
    config->SetVariablePose(image_id);
  }
  for (const gpu_ba::NativeBaCameraPolicy& camera : intent.camera_policies) {
    if (!camera.constant) {
      *error = "native runner received variable camera intrinsics";
      return false;
    }
    config->SetConstantCamera(camera.camera_id);
  }
  for (const point3D_t point3D_id : intent.explicit_variable_point_ids) {
    config->AddVariablePoint(point3D_id);
  }
  for (const point3D_t point3D_id : intent.explicit_constant_point_ids) {
    config->AddConstantPoint(point3D_id);
  }
  if (config->NumConstantPoses() != 0 || config->NumConstantTvecs() != 0) {
    *error = "native runner constructed a fixed camera pose";
    return false;
  }
  return true;
}

class DefaultOnlineLocalBaNativeRunner final
    : public OnlineLocalBaNativeRunner {
 public:
  bool Run(const BundleAdjustmentOptions& options,
           Reconstruction* candidate,
           const OnlineLocalBaIntentBuildResult& build_result,
           const double intent_build_milliseconds,
           OnlineLocalBaNativeRunOutput* output,
           std::string* error) override {
    if (candidate == nullptr || output == nullptr || error == nullptr) {
      return false;
    }
    *output = OnlineLocalBaNativeRunOutput();
    error->clear();
    BundleAdjustmentConfig config;
    if (!BuildBundleAdjustmentConfig(build_result.intent, &config, error)) {
      return false;
    }

    gpu_ba::GpuBaHostProblemStore store(candidate,
                                        build_result.intent.owner_epoch);
    gpu_ba::CudaHostStoreBinding binding;
    binding.store = &store;
    binding.owner_epoch = build_result.intent.owner_epoch;
    binding.mode = gpu_ba::CudaHostProblemStoreMode::kHostPreparedStore;
    {
      BundleAdjuster adjuster(options, config);
      adjuster.SetOptimazePhrase(BundleAdjuster::OptimazePhrase::Local);
      adjuster.SetCudaHostStoreBinding(binding);
      output->solver_invoked = true;
      output->solve_returned = adjuster.SolveNative(
          candidate, build_result.intent, intent_build_milliseconds);
      output->execution = adjuster.ExecutionResult();
      output->summary = adjuster.Summary();
    }
    std::string shutdown_error;
    if (!store.Shutdown(&shutdown_error)) {
      *error = shutdown_error.empty() ? "native host store shutdown failed"
                                      : shutdown_error;
      return false;
    }

    return true;
  }
};

bool ValidateBackendContract(const OnlineLocalBaNativeRunOutput& run) {
  const BundleAdjustmentExecutionResult& execution = run.execution;
  return execution.requested_backend == "custom_cuda" &&
         execution.executed_backend == "custom_cuda" &&
         execution.problem_source_requested == "native_graph" &&
         execution.problem_source_effective == "native_graph" &&
         !execution.fallback_used && execution.fallback_reason.empty() &&
         execution.ceres_solve_calls == 0 &&
         !execution.ceres_problem_created &&
         execution.ceres_cost_function_creations == 0 &&
         execution.ceres_add_residual_calls == 0;
}

bool ValidateFiniteSolverResult(const OnlineLocalBaNativeRunOutput& run) {
  const BundleAdjustmentExecutionResult& execution = run.execution;
  const ceres::Solver::Summary& summary = run.summary;
  return std::isfinite(execution.wall_seconds) &&
         execution.wall_seconds >= 0.0 &&
         std::isfinite(execution.initial_cost) &&
         execution.initial_cost >= 0.0 && std::isfinite(execution.final_cost) &&
         execution.final_cost >= 0.0 &&
         std::isfinite(execution.max_backward_error) &&
         execution.max_backward_error >= 0.0 &&
         std::isfinite(summary.initial_cost) && summary.initial_cost >= 0.0 &&
         std::isfinite(summary.final_cost) && summary.final_cost >= 0.0 &&
         std::isfinite(summary.total_time_in_seconds) &&
         summary.total_time_in_seconds >= 0.0 &&
         std::isfinite(summary.minimizer_time_in_seconds) &&
         summary.minimizer_time_in_seconds >= 0.0;
}

bool CheckFixedIntrinsics(const Reconstruction& candidate,
                          const OnlineLocalBaRequest& request,
                          std::string* error) {
  std::vector<OnlineLocalBaFixedIntrinsicsSnapshot> current;
  if (!CaptureOnlineLocalBaFixedIntrinsics(candidate,
                                           request.ordered_frozen_image_ids,
                                           &current,
                                           error)) {
    return false;
  }
  if (current != request.fixed_intrinsics) {
    *error = "candidate camera intrinsics changed during local BA";
    return false;
  }
  return true;
}

void CaptureAfterState(const Reconstruction* candidate,
                       OnlineLocalBaPassResult* result) {
  if (candidate == nullptr) return;
  result->candidate_version_after = candidate->CanonicalVersion();
  if (!result->candidate_visual_state_sha256_before.empty()) {
    const Clock::time_point hash_start = Clock::now();
    result->candidate_visual_state_sha256_after = VisualStateSha256(*candidate);
    result->timing.visual_state_hash_milliseconds +=
        ElapsedMilliseconds(hash_start);
  }
}

OnlineLocalBaPassResult FinishFailure(
    OnlineLocalBaPassResult result,
    const OnlineLocalBaFailureReason reason,
    const std::string& detail,
    const Clock::time_point total_start,
    Reconstruction* candidate,
    ReconstructionTransaction* transaction) {
  result.success = false;
  result.fatal = IsDualMode(result.mode);
  result.failure_reason = reason;
  result.failure_detail = detail.empty() ? ToString(reason) : detail;
  try {
    CaptureAfterState(candidate, &result);
  } catch (...) {
    result.candidate_visual_state_sha256_after.clear();
  }
  if (transaction != nullptr) {
    const Clock::time_point discard_start = Clock::now();
    *transaction = ReconstructionTransaction();
    result.timing.transaction_discard_milliseconds +=
        ElapsedMilliseconds(discard_start);
    result.candidate_discarded = true;
  }
  result.candidate_ready_for_commit = false;
  result.timing.total_milliseconds = ElapsedMilliseconds(total_start);
  return result;
}

bool SameDualFrozenRequest(const OnlineLocalBaRequest& pass1,
                           const OnlineLocalBaRequest& pass2,
                           std::string* error) {
  if (pass1.mode != OnlineLocalBaMode::DUAL_PASS1 || pass1.pass_index != 1 ||
      pass1.allow_postprocess ||
      pass2.mode != OnlineLocalBaMode::DUAL_PASS2 || pass2.pass_index != 2 ||
      !pass2.allow_postprocess) {
    *error = "dual local BA requires DUAL_PASS1 then DUAL_PASS2 policies";
    return false;
  }
  if (pass1.attempt_id != pass2.attempt_id ||
      pass1.trigger_image_id != pass2.trigger_image_id ||
      pass1.ordered_frozen_image_ids != pass2.ordered_frozen_image_ids ||
      pass1.expected_candidate_version != pass2.expected_candidate_version ||
      pass1.owner_epoch != pass2.owner_epoch ||
      pass1.topology_revision != pass2.topology_revision ||
      pass1.selection_revision == pass2.selection_revision ||
      pass1.expected_map_version != pass2.expected_map_version ||
      pass1.expected_max_scan_index != pass2.expected_max_scan_index ||
      pass1.expected_snapshot_sha256 != pass2.expected_snapshot_sha256 ||
      pass1.expected_geometry_sha256 != pass2.expected_geometry_sha256 ||
      pass1.map_snapshot.get() != pass2.map_snapshot.get() ||
      !SameAssociationOptions(pass1.association_options,
                              pass2.association_options) ||
      !SameProjectionOptions(pass1.projection_options,
                             pass2.projection_options) ||
      pass1.fixed_intrinsics != pass2.fixed_intrinsics ||
      pass1.all_camera_poses_variable != pass2.all_camera_poses_variable ||
      pass1.constant_camera_pose_ids != pass2.constant_camera_pose_ids ||
      pass1.requested_backend != pass2.requested_backend ||
      pass1.requested_problem_source != pass2.requested_problem_source ||
      pass1.fallback_allowed != pass2.fallback_allowed) {
    *error = "dual local BA requests do not share one frozen window/map/config";
    return false;
  }
  return true;
}

}  // namespace

bool CountOnlineLocalBaPotentialTriggerObservations(
    const Reconstruction& reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    const image_t trigger_image_id,
    uint64_t* upper_bound,
    std::string* error) noexcept {
  if (upper_bound == nullptr || error == nullptr) return false;
  *upper_bound = 0;
  error->clear();
  try {
    if (!reconstruction.ExistsImage(trigger_image_id) ||
        !reconstruction.Image(trigger_image_id).IsRegistered() ||
        !correspondence_graph.ExistsImage(trigger_image_id)) {
      *error = "trigger observation bound requires a registered graph image";
      return false;
    }
    const Image& trigger = reconstruction.Image(trigger_image_id);
    for (point2D_t point2D_idx = 0; point2D_idx < trigger.NumPoints2D();
         ++point2D_idx) {
      if (trigger.Point2D(point2D_idx).HasPoint3D()) {
        ++*upper_bound;
        continue;
      }
      for (const auto& correspondence :
           correspondence_graph.FindCorrespondences(trigger_image_id,
                                                    point2D_idx)) {
        if (reconstruction.ExistsImage(correspondence.image_id) &&
            reconstruction.Image(correspondence.image_id).IsRegistered()) {
          ++*upper_bound;
          break;
        }
      }
    }
    return true;
  } catch (const std::exception& exception) {
    *upper_bound = 0;
    *error = std::string("trigger observation bound failed: ") +
             exception.what();
    return false;
  } catch (...) {
    *upper_bound = 0;
    *error = "trigger observation bound failed";
    return false;
  }
}

bool CaptureOnlineLocalBaFixedIntrinsics(
    const Reconstruction& reconstruction,
    const std::vector<image_t>& ordered_frozen_image_ids,
    std::vector<OnlineLocalBaFixedIntrinsicsSnapshot>* fixed_intrinsics,
    std::string* error) noexcept {
  try {
    if (fixed_intrinsics == nullptr || error == nullptr) return false;
    fixed_intrinsics->clear();
    error->clear();
    if (ordered_frozen_image_ids.empty() ||
        ordered_frozen_image_ids.size() > kOnlineLocalBaMaximumImageCount) {
      *error = "cannot capture intrinsics for an invalid frozen window";
      return false;
    }
    std::set<image_t> seen_images;
    std::set<camera_t> camera_ids;
    for (const image_t image_id : ordered_frozen_image_ids) {
      if (!seen_images.insert(image_id).second ||
          !reconstruction.ExistsImage(image_id)) {
        *error = "cannot capture intrinsics for a missing or duplicate image";
        fixed_intrinsics->clear();
        return false;
      }
      const Image& image = reconstruction.Image(image_id);
      if (!reconstruction.ExistsCamera(image.CameraId())) {
        *error = "cannot capture intrinsics for a missing camera";
        fixed_intrinsics->clear();
        return false;
      }
      camera_ids.insert(image.CameraId());
    }
    fixed_intrinsics->reserve(camera_ids.size());
    for (const camera_t camera_id : camera_ids) {
      const Camera& camera = reconstruction.Camera(camera_id);
      OnlineLocalBaFixedIntrinsicsSnapshot snapshot;
      snapshot.camera_id = camera_id;
      snapshot.model_id = camera.ModelId();
      snapshot.width = camera.Width();
      snapshot.height = camera.Height();
      snapshot.has_prior_focal_length = camera.HasPriorFocalLength();
      snapshot.parameters = camera.Params();
      if (!camera.VerifyParams() || snapshot.width == 0 ||
          snapshot.height == 0 ||
          std::find_if(snapshot.parameters.begin(),
                       snapshot.parameters.end(),
                       [](const double value) {
                         return !std::isfinite(value);
                       }) !=
              snapshot.parameters.end()) {
        *error = "cannot freeze invalid camera intrinsics";
        fixed_intrinsics->clear();
        return false;
      }
      fixed_intrinsics->push_back(std::move(snapshot));
    }
    return true;
  } catch (const std::exception& exception) {
    if (fixed_intrinsics != nullptr) fixed_intrinsics->clear();
    if (error != nullptr) {
      *error = std::string("fixed intrinsics capture failed: ") +
               exception.what();
    }
    return false;
  } catch (...) {
    if (fixed_intrinsics != nullptr) fixed_intrinsics->clear();
    if (error != nullptr) *error = "fixed intrinsics capture failed";
    return false;
  }
}

OnlineLocalBaPassResult ExecuteOnlineLocalBa(
    const BundleAdjustmentOptions& options,
    const OnlineLocalBaRequest& request,
    ReconstructionTransaction* transaction,
    const OnlineLocalBaExecutorDependencies& dependencies) {
  const Clock::time_point total_start = Clock::now();
  OnlineLocalBaPassResult result;
  result.mode = request.mode;
  result.attempt_id = request.attempt_id;
  result.pass_index = request.pass_index;
  result.trigger_image_id = request.trigger_image_id;
  Reconstruction* candidate = nullptr;
  try {
    if (transaction == nullptr || !transaction->HasCandidate() ||
        (candidate = transaction->MutableCandidate()) == nullptr) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::CANDIDATE_UNAVAILABLE,
                           "local BA transaction candidate is unavailable",
                           total_start,
                           nullptr,
                           transaction);
    }
    result.candidate_version_before = candidate->CanonicalVersion();

    std::string error;
    OnlineLocalBaFailureReason request_failure =
        OnlineLocalBaFailureReason::INVALID_REQUEST;
    const Clock::time_point request_validation_start = Clock::now();
    const bool request_valid = ValidateRequest(options,
                                               request,
                                               *candidate,
                                               dependencies,
                                               &request_failure,
                                               &error);
    result.timing.request_validation_milliseconds =
        ElapsedMilliseconds(request_validation_start);
    if (!request_valid) {
      return FinishFailure(std::move(result),
                           request_failure,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }

    DefaultOnlineLocalBaIntentBuilder default_intent_builder;
    OnlineLocalBaIntentBuilder* intent_builder = dependencies.intent_builder;
    if (intent_builder == nullptr) intent_builder = &default_intent_builder;
    OnlineLocalBaIntentBuildResult build_result;
    const Clock::time_point intent_start = Clock::now();
    result.intent_builder_called = true;
    if (!intent_builder->Build(
            *candidate, options, request, &build_result, &error)) {
      result.timing.intent_build_milliseconds =
          ElapsedMilliseconds(intent_start);
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::INTENT_BUILD_FAILED,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }
    result.timing.intent_build_milliseconds = ElapsedMilliseconds(intent_start);
    result.trigger_gate = build_result.trigger_gate;
    if (result.trigger_gate.checked && !result.trigger_gate.passed) {
      const bool below_minimum = result.trigger_gate.actual_count <
                                kOnlineLocalBaMinimumTriggerLidarResidualCount;
      return FinishFailure(
          std::move(result),
          below_minimum
              ? OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM
              : OnlineLocalBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED,
          below_minimum
              ? "trigger preflight has fewer than 50 valid LiDAR constraints"
              : "CATCHUP retry has neither new active edge evidence nor 50 new "
                "trigger residuals",
          total_start, candidate, transaction);
    }
    result.association_audit = build_result.association_audit;
    std::map<image_t, uint64_t>* submitted_counts =
        &result.per_image_submitted_lidar_constraint_count;
    if (!ValidateAndSummarizeIntent(*candidate,
                                    request,
                                    build_result,
                                    &result.intent_summary,
                                    submitted_counts,
                                    &result.submitted_lidar_constraint_count,
                                    &error)) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::INTENT_AUDIT_FAILED,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }
    result.trigger_submitted_lidar_constraint_count =
        result.per_image_submitted_lidar_constraint_count.at(
            request.trigger_image_id);
    result.submitted_constraint_audit_passed = true;
    if (result.trigger_submitted_lidar_constraint_count <
        kOnlineLocalBaMinimumTriggerLidarResidualCount) {
      return FinishFailure(
          std::move(result),
          OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM,
          "trigger has fewer than 50 submitted LiDAR constraints",
          total_start,
          candidate,
          transaction);
    }

    if (request.require_trigger_residual_increase &&
        (result.trigger_submitted_lidar_constraint_count <
             request.previous_trigger_lidar_constraint_count ||
         result.trigger_submitted_lidar_constraint_count -
                 request.previous_trigger_lidar_constraint_count <
             kOnlineLocalBaMinimumTriggerLidarResidualCount)) {
      return FinishFailure(
          std::move(result),
          OnlineLocalBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED,
          "CATCHUP retry has neither new active edge evidence nor 50 new "
          "trigger residuals",
          total_start, candidate, transaction);
    }
    if (result.trigger_gate.checked &&
        result.trigger_gate.actual_count !=
            result.trigger_submitted_lidar_constraint_count) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::INTENT_AUDIT_FAILED,
                           "trigger preflight count differs from submitted count",
                           total_start, candidate, transaction);
    }

    if (dependencies.audit_visual_state_sha256) {
      const Clock::time_point hash_start = Clock::now();
      result.candidate_visual_state_sha256_before =
          VisualStateSha256(*candidate);
      result.timing.visual_state_hash_milliseconds +=
          ElapsedMilliseconds(hash_start);
    }

    DefaultOnlineLocalBaNativeRunner default_native_runner;
    OnlineLocalBaNativeRunner* native_runner = dependencies.native_runner;
    if (native_runner == nullptr) native_runner = &default_native_runner;
    OnlineLocalBaNativeRunOutput native_output;
    const Clock::time_point native_start = Clock::now();
    const bool runner_ok = native_runner->Run(options,
                                              candidate,
                                              build_result,
                                              result.timing
                                                  .intent_build_milliseconds,
                                              &native_output,
                                              &error);
    result.timing.native_run_milliseconds = ElapsedMilliseconds(native_start);
    result.solver_called = native_output.solver_invoked;
    result.residual_evidence = native_output.residual_evidence;
    result.execution = native_output.execution;
    result.solver_summary = native_output.summary;
    if (!runner_ok || !native_output.solver_invoked) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::NATIVE_RUNNER_FAILED,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }
    if (!native_output.solve_returned || !native_output.execution.success) {
      const std::string detail =
          !native_output.execution.diagnostic_message.empty()
              ? native_output.execution.diagnostic_message
              : native_output.execution.stable_error;
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::SOLVE_FAILED,
                           detail,
                           total_start,
                           candidate,
                           transaction);
    }

    result.backend_contract_passed = ValidateBackendContract(native_output);
    if (!result.backend_contract_passed) {
      return FinishFailure(
          std::move(result),
          OnlineLocalBaFailureReason::BACKEND_CONTRACT_FAILED,
          "native solve did not preserve custom_cuda/native_graph/no-fallback",
          total_start,
          candidate,
          transaction);
    }
    result.termination_converged =
        native_output.summary.termination_type == ceres::CONVERGENCE;
    if (!native_output.summary.IsSolutionUsable()) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::
                               TERMINATION_UNUSABLE,
                           native_output.execution.termination,
                           total_start,
                           candidate,
                           transaction);
    }
    result.finite_costs = ValidateFiniteSolverResult(native_output);
    if (!result.finite_costs) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::NONFINITE_SOLVER_RESULT,
                           "native solve reported a non-finite cost or timing",
                           total_start,
                           candidate,
                           transaction);
    }
    result.solve_acceptance_passed = true;

    if (!CheckFixedIntrinsics(*candidate, request, &error)) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::
                               FIXED_INTRINSICS_MISMATCH,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }
    if (!IsFiniteCandidate(*candidate, &error)) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::CANDIDATE_STATE_INVALID,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }

    if (request.allow_postprocess) {
      const Clock::time_point merge_start = Clock::now();
      ++result.postprocess.merge_call_count;
      if (!dependencies.postprocess->Merge(
              candidate,
              request,
              &result.postprocess.merged_observation_count,
              &error)) {
        result.timing.merge_milliseconds = ElapsedMilliseconds(merge_start);
        return FinishFailure(std::move(result),
                             OnlineLocalBaFailureReason::
                                 POSTPROCESS_MERGE_FAILED,
                             error,
                             total_start,
                             candidate,
                             transaction);
      }
      result.timing.merge_milliseconds = ElapsedMilliseconds(merge_start);

      const Clock::time_point complete_start = Clock::now();
      ++result.postprocess.complete_call_count;
      if (!dependencies.postprocess->Complete(
              candidate,
              request,
              &result.postprocess.completed_observation_count,
              &error)) {
        result.timing.complete_milliseconds =
            ElapsedMilliseconds(complete_start);
        return FinishFailure(std::move(result),
                             OnlineLocalBaFailureReason::
                                 POSTPROCESS_COMPLETE_FAILED,
                             error,
                             total_start,
                             candidate,
                             transaction);
      }
      result.timing.complete_milliseconds =
          ElapsedMilliseconds(complete_start);

      const Clock::time_point filter_start = Clock::now();
      ++result.postprocess.filter_call_count;
      if (!dependencies.postprocess->Filter(
              candidate,
              request,
              &result.postprocess.filtered_observation_count,
              &error)) {
        result.timing.filter_milliseconds = ElapsedMilliseconds(filter_start);
        return FinishFailure(std::move(result),
                             OnlineLocalBaFailureReason::
                                 POSTPROCESS_FILTER_FAILED,
                             error,
                             total_start,
                             candidate,
                             transaction);
      }
      result.timing.filter_milliseconds = ElapsedMilliseconds(filter_start);
    }

    if (!CheckFixedIntrinsics(*candidate, request, &error)) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::
                               FIXED_INTRINSICS_MISMATCH,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }
    if (!IsFiniteCandidate(*candidate, &error)) {
      return FinishFailure(std::move(result),
                           OnlineLocalBaFailureReason::CANDIDATE_STATE_INVALID,
                           error,
                           total_start,
                           candidate,
                           transaction);
    }

    CaptureAfterState(candidate, &result);
    result.success = true;
    result.fatal = false;
    result.candidate_discarded = false;
    result.candidate_ready_for_commit =
        request.mode != OnlineLocalBaMode::DUAL_PASS1;
    result.failure_reason = OnlineLocalBaFailureReason::NONE;
    result.failure_detail.clear();
    result.timing.total_milliseconds = ElapsedMilliseconds(total_start);
    return result;
  } catch (const std::exception& exception) {
    return FinishFailure(
        std::move(result),
        OnlineLocalBaFailureReason::INTERNAL_ERROR,
        std::string("local BA executor threw: ") + exception.what(),
        total_start,
        candidate,
        transaction);
  } catch (...) {
    return FinishFailure(std::move(result),
                         OnlineLocalBaFailureReason::INTERNAL_ERROR,
                         "local BA executor threw an unknown exception",
                         total_start,
                         candidate,
                         transaction);
  }
}

OnlineDualLocalBaResult ExecuteOnlineDualLocalBa(
    const BundleAdjustmentOptions& options,
    const OnlineLocalBaRequest& pass1_request,
    const OnlineLocalBaRequest& pass2_request,
    ReconstructionTransaction* transaction,
    const OnlineLocalBaExecutorDependencies& dependencies) {
  OnlineDualLocalBaResult result;
  result.pass1.mode = pass1_request.mode;
  result.pass1.attempt_id = pass1_request.attempt_id;
  result.pass1.pass_index = pass1_request.pass_index;
  result.pass1.trigger_image_id = pass1_request.trigger_image_id;
  result.pass2.mode = pass2_request.mode;
  result.pass2.attempt_id = pass2_request.attempt_id;
  result.pass2.pass_index = pass2_request.pass_index;
  result.pass2.trigger_image_id = pass2_request.trigger_image_id;
  Reconstruction* candidate =
      transaction == nullptr ? nullptr : transaction->MutableCandidate();
  try {
    if (candidate != nullptr) {
      result.candidate_version_before = candidate->CanonicalVersion();
      if (dependencies.audit_visual_state_sha256) {
        result.candidate_visual_state_sha256_before =
            VisualStateSha256(*candidate);
      }
    }
    std::string error;
    if (candidate == nullptr ||
        !SameDualFrozenRequest(pass1_request, pass2_request, &error)) {
      result.fatal = true;
      result.failure_reason = candidate == nullptr
                                  ? OnlineLocalBaFailureReason::
                                        CANDIDATE_UNAVAILABLE
                                  : OnlineLocalBaFailureReason::
                                        DUAL_REQUEST_MISMATCH;
      result.failure_detail = error.empty()
                                  ? "dual local BA candidate is unavailable"
                                  : error;
      if (transaction != nullptr) {
        *transaction = ReconstructionTransaction();
        result.candidate_discarded = true;
      }
      return result;
    }

    result.pass1 = ExecuteOnlineLocalBa(
        options, pass1_request, transaction, dependencies);
    if (!result.pass1.success) {
      result.fatal = true;
      result.candidate_discarded = true;
      result.failure_reason = OnlineLocalBaFailureReason::DUAL_PASS1_FAILED;
      result.failure_detail =
          std::string(ToString(result.pass1.failure_reason)) + ": " +
          result.pass1.failure_detail;
      result.candidate_version_after = result.pass1.candidate_version_after;
      result.candidate_visual_state_sha256_after =
          result.pass1.candidate_visual_state_sha256_after;
      return result;
    }
    if (result.pass1.postprocess.merge_call_count != 0 ||
        result.pass1.postprocess.complete_call_count != 0 ||
        result.pass1.postprocess.filter_call_count != 0 ||
        result.pass1.candidate_ready_for_commit) {
      if (transaction != nullptr) *transaction = ReconstructionTransaction();
      result.fatal = true;
      result.candidate_discarded = true;
      result.failure_reason = OnlineLocalBaFailureReason::DUAL_PASS1_FAILED;
      result.failure_detail =
          "DUAL_PASS1 executed postprocess or became committable";
      return result;
    }

    result.pass2 = ExecuteOnlineLocalBa(
        options, pass2_request, transaction, dependencies);
    if (!result.pass2.success) {
      result.fatal = true;
      result.candidate_discarded = true;
      result.failure_reason = OnlineLocalBaFailureReason::DUAL_PASS2_FAILED;
      result.failure_detail =
          std::string(ToString(result.pass2.failure_reason)) + ": " +
          result.pass2.failure_detail;
      result.candidate_version_after = result.pass2.candidate_version_after;
      result.candidate_visual_state_sha256_after =
          result.pass2.candidate_visual_state_sha256_after;
      return result;
    }
    if (result.pass2.postprocess.merge_call_count != 1 ||
        result.pass2.postprocess.complete_call_count != 1 ||
        result.pass2.postprocess.filter_call_count != 1 ||
        !result.pass2.candidate_ready_for_commit) {
      candidate = transaction == nullptr ? nullptr
                                         : transaction->MutableCandidate();
      if (candidate != nullptr) {
        result.candidate_version_after = candidate->CanonicalVersion();
        if (dependencies.audit_visual_state_sha256) {
          result.candidate_visual_state_sha256_after =
              VisualStateSha256(*candidate);
        }
      }
      if (transaction != nullptr) *transaction = ReconstructionTransaction();
      result.fatal = true;
      result.candidate_discarded = true;
      result.failure_reason = OnlineLocalBaFailureReason::DUAL_PASS2_FAILED;
      result.failure_detail =
          "DUAL_PASS2 did not execute exactly one postprocess";
      return result;
    }

    candidate = transaction->MutableCandidate();
    if (candidate == nullptr) {
      result.fatal = true;
      result.candidate_discarded = true;
      result.failure_reason = OnlineLocalBaFailureReason::CANDIDATE_UNAVAILABLE;
      result.failure_detail = "dual local BA candidate disappeared";
      *transaction = ReconstructionTransaction();
      return result;
    }
    result.candidate_version_after = candidate->CanonicalVersion();
    if (dependencies.audit_visual_state_sha256) {
      result.candidate_visual_state_sha256_after =
          VisualStateSha256(*candidate);
    }
    result.success = true;
    result.fatal = false;
    result.candidate_discarded = false;
    result.candidate_ready_for_commit = true;
    result.failure_reason = OnlineLocalBaFailureReason::NONE;
    return result;
  } catch (const std::exception& exception) {
    if (transaction != nullptr) *transaction = ReconstructionTransaction();
    result.success = false;
    result.fatal = true;
    result.candidate_discarded = true;
    result.failure_reason = OnlineLocalBaFailureReason::INTERNAL_ERROR;
    result.failure_detail = std::string("dual local BA executor threw: ") +
                            exception.what();
    return result;
  } catch (...) {
    if (transaction != nullptr) *transaction = ReconstructionTransaction();
    result.success = false;
    result.fatal = true;
    result.candidate_discarded = true;
    result.failure_reason = OnlineLocalBaFailureReason::INTERNAL_ERROR;
    result.failure_detail = "dual local BA executor threw an unknown exception";
    return result;
  }
}

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED
