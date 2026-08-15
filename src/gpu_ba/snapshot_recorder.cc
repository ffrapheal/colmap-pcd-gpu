#include "gpu_ba/snapshot_recorder.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <ceres/problem.h>

#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::string SanitizeIdToken(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '-' || ch == '_') {
      result.push_back(static_cast<char>(ch));
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "unknown" : result;
}

std::string LossFunctionName(
    BundleAdjustmentOptions::LossFunctionType loss_function_type) {
  switch (loss_function_type) {
    case BundleAdjustmentOptions::LossFunctionType::TRIVIAL: return "trivial";
    case BundleAdjustmentOptions::LossFunctionType::SOFT_L1: return "soft_l1";
    case BundleAdjustmentOptions::LossFunctionType::CAUCHY: return "cauchy";
  }
  return "unknown";
}

struct ParameterState {
  bool present = false;
  bool constant = true;
  uint32_t ambient_size = 0;
  uint32_t tangent_size = 0;
};

uint64_t ParameterKey(ParameterKind kind, uint64_t entity_id) {
  return (static_cast<uint64_t>(kind) << 60) ^ entity_id;
}

}  // namespace

bool ShouldCaptureSnapshot(const std::string& capture_mode,
                           const std::string& registered_images,
                           BaKind ba_kind,
                           uint64_t registered_image_count,
                           std::string* error) {
  if (error == nullptr) return false;
  error->clear();
  const std::string mode = Trim(capture_mode);
  if (mode == "none" || mode.empty()) return false;
  if (mode != "all" && mode != BaKindName(ba_kind)) return false;

  const std::string list = Trim(registered_images);
  if (list.empty()) return true;
  std::istringstream stream(list);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = Trim(token);
    if (token.empty()) {
      *error = "Empty entry in ba_snapshot_registered_images";
      return false;
    }
    size_t consumed = 0;
    uint64_t value = 0;
    try {
      value = std::stoull(token, &consumed);
    } catch (const std::exception&) {
      *error = "Invalid registered-image count: " + token;
      return false;
    }
    if (consumed != token.size()) {
      *error = "Invalid registered-image count: " + token;
      return false;
    }
    if (value == registered_image_count) return true;
  }
  return false;
}

void SnapshotRecorder::RecordVisualResidual(
    uint32_t image_id,
    uint32_t point2D_idx,
    uint64_t point3D_id,
    const std::array<double, 2>& xy,
    bool pose_constant) {
  const uint64_t source_index = source_order_.size();
  ObservationSnapshot observation;
  observation.source_index = source_index;
  observation.image_id = image_id;
  observation.point2D_idx = point2D_idx;
  observation.point3D_id = point3D_id;
  observation.pose_constant = pose_constant;
  observation.xy = xy;
  observations_.push_back(observation);

  OrderEntrySnapshot entry;
  entry.source_index = source_index;
  entry.residual_kind = ResidualKind::kVisual;
  entry.image_id = image_id;
  entry.point2D_idx = point2D_idx;
  entry.point3D_id = point3D_id;
  source_order_.push_back(entry);
}

void SnapshotRecorder::RecordLidarResidual(
    uint64_t point3D_id,
    uint8_t lidar_type,
    const std::array<double, 3>& lidar_xyz,
    const std::array<double, 4>& plane,
    double weight,
    bool has_search_range,
    double search_range) {
  const uint64_t source_index = source_order_.size();
  LidarSnapshot lidar;
  lidar.source_index = source_index;
  lidar.point3D_id = point3D_id;
  lidar.lidar_type = lidar_type;
  lidar.has_search_range = has_search_range;
  lidar.search_range = search_range;
  lidar.weight = weight;
  lidar.lidar_xyz = lidar_xyz;
  lidar.plane = plane;
  lidar_.push_back(lidar);

  OrderEntrySnapshot entry;
  entry.source_index = source_index;
  entry.residual_kind = ResidualKind::kLidar;
  entry.image_id = std::numeric_limits<uint32_t>::max();
  entry.point2D_idx = std::numeric_limits<uint32_t>::max();
  entry.point3D_id = point3D_id;
  source_order_.push_back(entry);
}

void SnapshotRecorder::RecordParameterBlock(ParameterKind kind,
                                            uint64_t entity_id,
                                            double* values) {
  if (parameter_indices_.count(values) != 0) return;
  parameter_indices_.emplace(values, parameters_.size());
  RecordedParameter parameter;
  parameter.kind = kind;
  parameter.entity_id = entity_id;
  parameter.values = values;
  parameters_.push_back(parameter);
}

bool SnapshotRecorder::FinalizeAndWrite(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction,
    const ceres::Problem& problem,
    BaKind ba_kind,
    uint64_t ba_call_index,
    SnapshotWriteResult* result,
    std::string* error) const {
  Snapshot snapshot;
  snapshot.metadata.ba_kind = ba_kind;
  snapshot.metadata.registered_image_count = reconstruction.NumRegImages();
  snapshot.metadata.ba_call_index = ba_call_index;
  snapshot.metadata.refinement_index = options.ba_refinement_index;
  snapshot.metadata.trigger_image_id = options.ba_trigger_image_id;
  snapshot.metadata.optimize_phrase = BaKindName(ba_kind);
  snapshot.metadata.backend = options.ba_backend;
  snapshot.metadata.loss_function = LossFunctionName(options.loss_function_type);
  snapshot.metadata.lidar_residual_mode = options.ba_lidar_residual;
  snapshot.metadata.lidar_correspondence_version =
      "colmap-pcd-cpu-correspondence-v1";
  snapshot.metadata.schur_mode = options.ba_cuda_schur_mode;
  snapshot.metadata.refine_focal_length = options.refine_focal_length;
  snapshot.metadata.refine_principal_point = options.refine_principal_point;
  snapshot.metadata.refine_extra_params = options.refine_extra_params;
  snapshot.metadata.refine_extrinsics = options.refine_extrinsics;
  snapshot.metadata.proj_lidar_weight = options.proj_lidar_constraint_weight;
  snapshot.metadata.icp_lidar_weight = options.icp_lidar_constraint_weight;
  snapshot.metadata.icp_ground_lidar_weight =
      options.icp_ground_lidar_constraint_weight;
  snapshot.metadata.function_tolerance =
      options.solver_options.function_tolerance;
  snapshot.metadata.gradient_tolerance =
      options.solver_options.gradient_tolerance;
  snapshot.metadata.parameter_tolerance =
      options.solver_options.parameter_tolerance;
  snapshot.metadata.max_num_iterations =
      options.solver_options.max_num_iterations;
  snapshot.metadata.max_linear_solver_iterations =
      options.solver_options.max_linear_solver_iterations;
  snapshot.metadata.max_consecutive_invalid_steps =
      options.solver_options.max_num_consecutive_invalid_steps;

  std::ostringstream snapshot_id;
  snapshot_id << BaKindName(ba_kind) << "-reg"
              << snapshot.metadata.registered_image_count << "-call"
              << ba_call_index << "-refine" << options.ba_refinement_index
              << "-trigger" << options.ba_trigger_image_id << "-phrase"
              << SanitizeIdToken(snapshot.metadata.optimize_phrase);
  snapshot.metadata.snapshot_id = snapshot_id.str();

  std::unordered_map<uint64_t, ParameterState> parameter_states;
  snapshot.parameter_blocks_source_order.reserve(parameters_.size());
  for (size_t index = 0; index < parameters_.size(); ++index) {
    const auto& recorded = parameters_[index];
    if (!problem.HasParameterBlock(recorded.values)) {
      *error = "Recorded parameter block is missing from Ceres problem";
      return false;
    }
    ParameterBlockSnapshot parameter;
    parameter.source_index = index;
    parameter.kind = recorded.kind;
    parameter.entity_id = recorded.entity_id;
    parameter.ambient_size = problem.ParameterBlockSize(recorded.values);
    parameter.tangent_size = problem.ParameterBlockLocalSize(recorded.values);
    parameter.constant = problem.IsParameterBlockConstant(recorded.values);
    snapshot.parameter_blocks_source_order.push_back(parameter);

    ParameterState state;
    state.present = true;
    state.constant = parameter.constant;
    state.ambient_size = parameter.ambient_size;
    state.tangent_size = parameter.tangent_size;
    parameter_states[ParameterKey(recorded.kind, recorded.entity_id)] = state;
  }
  snapshot.parameter_blocks_canonical_order.resize(parameters_.size());
  for (size_t i = 0; i < parameters_.size(); ++i) {
    snapshot.parameter_blocks_canonical_order[i] = i;
  }
  std::sort(snapshot.parameter_blocks_canonical_order.begin(),
            snapshot.parameter_blocks_canonical_order.end(),
            [&snapshot](uint64_t lhs, uint64_t rhs) {
              const auto& a = snapshot.parameter_blocks_source_order[lhs];
              const auto& b = snapshot.parameter_blocks_source_order[rhs];
              return std::tie(a.kind, a.entity_id) <
                     std::tie(b.kind, b.entity_id);
            });

  std::set<uint32_t> image_ids(config.Images().begin(), config.Images().end());
  std::set<uint64_t> point_ids;
  std::set<uint32_t> camera_ids;
  for (const auto& observation : observations_) {
    image_ids.insert(observation.image_id);
    point_ids.insert(observation.point3D_id);
  }
  for (const auto& lidar : lidar_) point_ids.insert(lidar.point3D_id);
  for (const auto& parameter : parameters_) {
    if (parameter.kind == ParameterKind::kPoint3D) {
      point_ids.insert(parameter.entity_id);
    } else if (parameter.kind == ParameterKind::kCamera) {
      camera_ids.insert(static_cast<uint32_t>(parameter.entity_id));
    }
  }

  // Tracks are part of the snapshot identity even when a track observation is
  // outside the selected local bundle and does not create a residual. Preserve
  // the state of every image referenced by those complete tracks.
  for (const uint64_t point_id : point_ids) {
    const Point3D& point = reconstruction.Point3D(point_id);
    for (const auto& track : point.Track().Elements()) {
      image_ids.insert(track.image_id);
    }
  }

  for (const uint32_t image_id : image_ids) {
    const Image& image = reconstruction.Image(image_id);
    camera_ids.insert(image.CameraId());
    const auto q_state = parameter_states.find(
        ParameterKey(ParameterKind::kQuaternion, image_id));
    const auto t_state = parameter_states.find(
        ParameterKey(ParameterKind::kTranslation, image_id));
    ImageSnapshot output;
    output.image_id = image_id;
    output.camera_id = image.CameraId();
    output.selected = config.HasImage(image_id);
    output.has_pose_parameter_blocks =
        q_state != parameter_states.end() && t_state != parameter_states.end();
    output.pose_constant = !output.has_pose_parameter_blocks ||
                           (q_state->second.constant && t_state->second.constant);
    if (config.HasConstantTvec(image_id)) {
      for (const int index : config.ConstantTvec(image_id)) {
        output.constant_tvec_mask |= static_cast<uint8_t>(1u << index);
      }
    }
    for (size_t i = 0; i < 4; ++i) output.qvec[i] = image.Qvec()[i];
    for (size_t i = 0; i < 3; ++i) output.tvec[i] = image.Tvec()[i];
    snapshot.images.push_back(output);
  }

  for (const uint32_t camera_id : camera_ids) {
    const Camera& camera = reconstruction.Camera(camera_id);
    const auto state = parameter_states.find(
        ParameterKey(ParameterKind::kCamera, camera_id));
    CameraSnapshot output;
    output.camera_id = camera_id;
    output.model_id = camera.ModelId();
    output.width = camera.Width();
    output.height = camera.Height();
    output.constant = state == parameter_states.end() || state->second.constant;
    output.params.assign(camera.Params().begin(), camera.Params().end());
    snapshot.cameras.push_back(output);
  }

  for (const uint64_t point_id : point_ids) {
    const Point3D& point = reconstruction.Point3D(point_id);
    const auto state = parameter_states.find(
        ParameterKey(ParameterKind::kPoint3D, point_id));
    PointSnapshot output;
    output.point3D_id = point_id;
    output.constant = state == parameter_states.end() || state->second.constant;
    if (config.HasVariablePoint(point_id)) {
      output.config_role = 1;
    } else if (config.HasConstantPoint(point_id)) {
      output.config_role = 2;
    }
    const auto range = config.LidarSearchRanges().find(point_id);
    if (range != config.LidarSearchRanges().end()) {
      output.has_search_range = true;
      output.search_range = range->second;
    }
    for (size_t i = 0; i < 3; ++i) output.xyz[i] = point.XYZ()[i];
    snapshot.points.push_back(output);

    for (const auto& track : point.Track().Elements()) {
      TrackElementSnapshot element;
      element.point3D_id = point_id;
      element.image_id = track.image_id;
      element.point2D_idx = track.point2D_idx;
      snapshot.tracks.push_back(element);
    }
  }
  std::sort(snapshot.tracks.begin(), snapshot.tracks.end(),
            [](const TrackElementSnapshot& lhs,
               const TrackElementSnapshot& rhs) {
              return std::tie(lhs.point3D_id, lhs.image_id, lhs.point2D_idx) <
                     std::tie(rhs.point3D_id, rhs.image_id, rhs.point2D_idx);
            });

  snapshot.observations = observations_;
  snapshot.lidar = lidar_;
  snapshot.source_insertion_order = source_order_;
  snapshot.canonical_order = source_order_;
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs, const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                              lhs.point3D_id, lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                              rhs.point3D_id, rhs.source_index);
            });

  return WriteSnapshot(snapshot, options.ba_snapshot_dir, result, error);
}

}  // namespace gpu_ba
}  // namespace colmap
