#include "gpu_ba/snapshot_recorder.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
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

void DescriptorHashByte(uint64_t* state, const uint8_t value) {
  *state ^= value;
  *state *= 1099511628211ull;
}

void DescriptorHashWord(uint64_t* state, uint64_t value) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    DescriptorHashByte(state, static_cast<uint8_t>(value & 0xffu));
    value >>= 8;
  }
}

void DescriptorHashDouble(uint64_t* state, const double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double descriptor size");
  std::memcpy(&bits, &value, sizeof(bits));
  DescriptorHashWord(state, bits);
}

void DescriptorHashString(uint64_t* state, const std::string& value) {
  DescriptorHashWord(state, value.size());
  for (const unsigned char ch : value) DescriptorHashByte(state, ch);
}

}  // namespace

SnapshotRecorder::SnapshotRecorder(
    const bool compute_prepared_host_descriptor)
    : compute_prepared_host_descriptor_(compute_prepared_host_descriptor) {}

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

  if (compute_prepared_host_descriptor_) {
    DescriptorHashWord(&residual_descriptor_identity_, 0);
    DescriptorHashWord(&residual_descriptor_identity_, source_index);
    DescriptorHashWord(&residual_descriptor_identity_, image_id);
    DescriptorHashWord(&residual_descriptor_identity_, point2D_idx);
    DescriptorHashWord(&residual_descriptor_identity_, point3D_id);
    DescriptorHashWord(&residual_descriptor_identity_, pose_constant);
    DescriptorHashDouble(&residual_descriptor_identity_, xy[0]);
    DescriptorHashDouble(&residual_descriptor_identity_, xy[1]);
    residual_descriptor_hash_updates_ += 8;
    ++residual_descriptor_items_;
  }
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

  if (compute_prepared_host_descriptor_) {
    DescriptorHashWord(&residual_descriptor_identity_, 1);
    DescriptorHashWord(&residual_descriptor_identity_, source_index);
    DescriptorHashWord(&residual_descriptor_identity_, point3D_id);
    DescriptorHashWord(&residual_descriptor_identity_, lidar_type);
    DescriptorHashWord(&residual_descriptor_identity_, has_search_range);
    DescriptorHashDouble(&residual_descriptor_identity_, search_range);
    DescriptorHashDouble(&residual_descriptor_identity_, weight);
    for (const double value : lidar_xyz) {
      DescriptorHashDouble(&residual_descriptor_identity_, value);
    }
    for (const double value : plane) {
      DescriptorHashDouble(&residual_descriptor_identity_, value);
    }
    residual_descriptor_hash_updates_ += 13;
    ++residual_descriptor_items_;
  }
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

bool SnapshotRecorder::Finalize(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_solver_options,
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction,
    const ceres::Problem& problem,
    BaKind ba_kind,
    uint64_t ba_call_index,
    Snapshot* snapshot_output,
    std::string* error) const {
  if (snapshot_output == nullptr) {
    *error = "Snapshot recorder requires an output snapshot";
    return false;
  }
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
      effective_solver_options.function_tolerance;
  snapshot.metadata.gradient_tolerance =
      effective_solver_options.gradient_tolerance;
  snapshot.metadata.parameter_tolerance =
      effective_solver_options.parameter_tolerance;
  snapshot.metadata.max_num_iterations =
      effective_solver_options.max_num_iterations;
  snapshot.metadata.max_linear_solver_iterations =
      effective_solver_options.max_linear_solver_iterations;
  snapshot.metadata.max_consecutive_invalid_steps =
      effective_solver_options.max_num_consecutive_invalid_steps;

  std::ostringstream snapshot_id;
  snapshot_id << BaKindName(ba_kind) << "-reg"
              << snapshot.metadata.registered_image_count << "-call"
              << ba_call_index << "-refine" << options.ba_refinement_index
              << "-trigger" << options.ba_trigger_image_id << "-phrase"
              << SanitizeIdToken(snapshot.metadata.optimize_phrase);
  snapshot.metadata.snapshot_id = snapshot_id.str();

  std::unordered_map<ParameterIdentityKey, ParameterState,
                     ParameterIdentityKeyHash> parameter_states;
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
    parameter_states[ParameterIdentityKey{recorded.kind, recorded.entity_id}] =
        state;
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
        ParameterIdentityKey{ParameterKind::kQuaternion, image_id});
    const auto t_state = parameter_states.find(
        ParameterIdentityKey{ParameterKind::kTranslation, image_id});
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
        ParameterIdentityKey{ParameterKind::kCamera, camera_id});
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
        ParameterIdentityKey{ParameterKind::kPoint3D, point_id});
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

  if (compute_prepared_host_descriptor_) {
    // Residual identity is accumulated inside SetUp. This final pass only
    // covers the active entities and parameter blocks already materialized by
    // Finalize, and is entirely absent when the host store is disabled.
    const auto descriptor_start = std::chrono::steady_clock::now();
    uint64_t descriptor = residual_descriptor_identity_;
    uint64_t descriptor_hash_updates = residual_descriptor_hash_updates_;
    const auto hash_word = [&](const uint64_t value) {
      DescriptorHashWord(&descriptor, value);
      ++descriptor_hash_updates;
    };
    const auto hash_double = [&](const double value) {
      DescriptorHashDouble(&descriptor, value);
      ++descriptor_hash_updates;
    };
    const auto hash_string = [&](const std::string& value) {
      DescriptorHashString(&descriptor, value);
      ++descriptor_hash_updates;
    };
    hash_word(0x4853545657455631ull);
    hash_word(static_cast<uint64_t>(ba_kind));
    hash_word(options.refine_focal_length);
    hash_word(options.refine_principal_point);
    hash_word(options.refine_extra_params);
    hash_word(options.refine_extrinsics);
    hash_string(snapshot.metadata.loss_function);
    hash_string(snapshot.metadata.lidar_residual_mode);
    for (const CameraSnapshot& value : snapshot.cameras) {
      hash_word(value.camera_id);
      hash_word(static_cast<uint64_t>(value.model_id));
      hash_word(value.width);
      hash_word(value.height);
      hash_word(value.constant);
      hash_word(value.params.size());
      for (const double parameter : value.params) hash_double(parameter);
    }
    for (const ImageSnapshot& value : snapshot.images) {
      hash_word(value.image_id);
      hash_word(value.camera_id);
      hash_word(value.selected);
      hash_word(value.pose_constant);
      hash_word(value.has_pose_parameter_blocks);
      hash_word(value.constant_tvec_mask);
    }
    for (const PointSnapshot& value : snapshot.points) {
      hash_word(value.point3D_id);
      hash_word(value.constant);
      hash_word(value.config_role);
      hash_word(value.has_search_range);
      if (value.has_search_range) hash_double(value.search_range);
    }
    for (const ParameterBlockSnapshot& value :
         snapshot.parameter_blocks_source_order) {
      hash_word(value.source_index);
      hash_word(static_cast<uint64_t>(value.kind));
      hash_word(value.entity_id);
      hash_word(value.ambient_size);
      hash_word(value.tangent_size);
      hash_word(value.constant);
    }
    snapshot.prepared_host_topology_identity = descriptor == 0 ? 1 : descriptor;
    snapshot.prepared_host_descriptor_items =
        residual_descriptor_items_ + snapshot.cameras.size() +
        snapshot.images.size() + snapshot.points.size() +
        snapshot.parameter_blocks_source_order.size();
    snapshot.prepared_host_descriptor_hash_updates = descriptor_hash_updates;
    snapshot.prepared_host_descriptor_wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - descriptor_start).count();
  }

  *snapshot_output = std::move(snapshot);
  return true;
}

bool SnapshotRecorder::FinalizeAndWrite(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_solver_options,
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction,
    const ceres::Problem& problem,
    BaKind ba_kind,
    uint64_t ba_call_index,
    Snapshot* snapshot_output,
    SnapshotWriteResult* result,
    std::string* error) const {
  if (!Finalize(options, effective_solver_options, config, reconstruction,
                problem, ba_kind, ba_call_index, snapshot_output, error)) {
    return false;
  }
  return WriteSnapshot(*snapshot_output, options.ba_snapshot_dir, result, error);
}

}  // namespace gpu_ba
}  // namespace colmap
