#include "gpu_ba/active_ba_solve_spec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <tuple>
#include <unordered_set>

#include "base/reconstruction.h"
#include "base/camera_models.h"
#include "optim/bundle_adjustment.h"

namespace colmap {
namespace gpu_ba {
namespace {

void HashByte(uint64_t* state, const uint8_t value) {
  *state ^= value;
  *state *= 1099511628211ull;
}

void HashWord(uint64_t* state, uint64_t value) {
  for (size_t i = 0; i < sizeof(value); ++i) {
    HashByte(state, static_cast<uint8_t>(value & 0xffu));
    value >>= 8;
  }
}

void HashDouble(uint64_t* state, const double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "double hash size");
  std::memcpy(&bits, &value, sizeof(bits));
  HashWord(state, bits);
}

void HashString(uint64_t* state, const std::string& value) {
  HashWord(state, value.size());
  for (const unsigned char ch : value) HashByte(state, ch);
}

std::string LossFunctionName(
    const BundleAdjustmentOptions::LossFunctionType type) {
  switch (type) {
    case BundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      return "trivial";
    case BundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      return "soft_l1";
    case BundleAdjustmentOptions::LossFunctionType::CAUCHY:
      return "cauchy";
  }
  return "unknown";
}

uint32_t PopCount3(const uint8_t value) {
  return static_cast<uint32_t>((value & 1u) != 0) +
         static_cast<uint32_t>((value & 2u) != 0) +
         static_cast<uint32_t>((value & 4u) != 0);
}

bool SameOrderEntry(const OrderEntrySnapshot& lhs,
                    const OrderEntrySnapshot& rhs) {
  return lhs.source_index == rhs.source_index &&
         lhs.residual_kind == rhs.residual_kind &&
         lhs.image_id == rhs.image_id &&
         lhs.point2D_idx == rhs.point2D_idx &&
         lhs.point3D_id == rhs.point3D_id;
}

bool SameObservation(const ObservationSnapshot& lhs,
                     const ObservationSnapshot& rhs) {
  return lhs.source_index == rhs.source_index &&
         lhs.image_id == rhs.image_id &&
         lhs.point2D_idx == rhs.point2D_idx &&
         lhs.point3D_id == rhs.point3D_id &&
         lhs.pose_constant == rhs.pose_constant && lhs.xy == rhs.xy;
}

bool SameLidar(const LidarSnapshot& lhs, const LidarSnapshot& rhs) {
  return lhs.source_index == rhs.source_index &&
         lhs.point3D_id == rhs.point3D_id &&
         lhs.lidar_type == rhs.lidar_type &&
         lhs.has_search_range == rhs.has_search_range &&
         lhs.search_range == rhs.search_range && lhs.weight == rhs.weight &&
         lhs.lidar_xyz == rhs.lidar_xyz && lhs.plane == rhs.plane;
}

bool SameCamera(const CameraSnapshot& lhs, const CameraSnapshot& rhs) {
  return lhs.camera_id == rhs.camera_id && lhs.model_id == rhs.model_id &&
         lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.constant == rhs.constant && lhs.params == rhs.params;
}

bool SameImage(const ImageSnapshot& lhs, const ImageSnapshot& rhs) {
  return lhs.image_id == rhs.image_id && lhs.camera_id == rhs.camera_id &&
         lhs.selected == rhs.selected &&
         lhs.pose_constant == rhs.pose_constant &&
         lhs.has_pose_parameter_blocks == rhs.has_pose_parameter_blocks &&
         lhs.constant_tvec_mask == rhs.constant_tvec_mask &&
         lhs.qvec == rhs.qvec && lhs.tvec == rhs.tvec;
}

bool SamePoint(const PointSnapshot& lhs, const PointSnapshot& rhs) {
  return lhs.point3D_id == rhs.point3D_id &&
         lhs.constant == rhs.constant && lhs.config_role == rhs.config_role &&
         lhs.has_search_range == rhs.has_search_range &&
         lhs.search_range == rhs.search_range && lhs.xyz == rhs.xyz;
}

bool SameParameter(const ParameterBlockSnapshot& lhs,
                   const ParameterBlockSnapshot& rhs) {
  return lhs.source_index == rhs.source_index && lhs.kind == rhs.kind &&
         lhs.entity_id == rhs.entity_id &&
         lhs.ambient_size == rhs.ambient_size &&
         lhs.tangent_size == rhs.tangent_size &&
         lhs.constant == rhs.constant;
}

bool SameMetadataSemantics(const SnapshotMetadata& lhs,
                           const SnapshotMetadata& rhs) {
  return lhs.ba_kind == rhs.ba_kind &&
         lhs.registered_image_count == rhs.registered_image_count &&
         lhs.ba_call_index == rhs.ba_call_index &&
         lhs.refinement_index == rhs.refinement_index &&
         lhs.trigger_image_id == rhs.trigger_image_id &&
         lhs.optimize_phrase == rhs.optimize_phrase &&
         lhs.backend == rhs.backend &&
         lhs.loss_function == rhs.loss_function &&
         lhs.lidar_residual_mode == rhs.lidar_residual_mode &&
         lhs.lidar_correspondence_version ==
             rhs.lidar_correspondence_version &&
         lhs.schur_mode == rhs.schur_mode &&
         lhs.refine_focal_length == rhs.refine_focal_length &&
         lhs.refine_principal_point == rhs.refine_principal_point &&
         lhs.refine_extra_params == rhs.refine_extra_params &&
         lhs.refine_extrinsics == rhs.refine_extrinsics &&
         lhs.proj_lidar_weight == rhs.proj_lidar_weight &&
         lhs.icp_lidar_weight == rhs.icp_lidar_weight &&
         lhs.icp_ground_lidar_weight == rhs.icp_ground_lidar_weight &&
         lhs.function_tolerance == rhs.function_tolerance &&
         lhs.gradient_tolerance == rhs.gradient_tolerance &&
         lhs.parameter_tolerance == rhs.parameter_tolerance &&
         lhs.max_num_iterations == rhs.max_num_iterations &&
         lhs.max_linear_solver_iterations ==
             rhs.max_linear_solver_iterations &&
         lhs.max_consecutive_invalid_steps ==
             rhs.max_consecutive_invalid_steps;
}

template <typename T, typename Predicate>
bool SameVector(const std::vector<T>& lhs,
                const std::vector<T>& rhs,
                Predicate predicate) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!predicate(lhs[i], rhs[i])) return false;
  }
  return true;
}

}  // namespace

struct ActiveBaSolveSpecBuilder::Impl {
  std::vector<ObservationSnapshot> observations;
  std::vector<LidarSnapshot> lidar;
  std::vector<OrderEntrySnapshot> source_order;
  std::vector<std::pair<ParameterKind, uint64_t>> parameters;
  std::unordered_set<ParameterIdentityKey, ParameterIdentityKeyHash>
      parameter_keys;
  uint64_t descriptor_identity = 1469598103934665603ull;
  uint64_t descriptor_items = 0;
  uint64_t descriptor_hash_updates = 0;
};

bool ParseCudaProblemSource(const std::string& value,
                            CudaProblemSource* source) {
  if (source == nullptr) return false;
  if (value.empty() || value == "legacy_snapshot") {
    *source = CudaProblemSource::kLegacySnapshot;
    return true;
  }
  if (value == "active_spec") {
    *source = CudaProblemSource::kActiveSpec;
    return true;
  }
  if (value == "indexed_catalog") {
    *source = CudaProblemSource::kIndexedCatalog;
    return true;
  }
  if (value == "native_graph") {
    *source = CudaProblemSource::kNativeGraph;
    return true;
  }
  return false;
}

const char* CudaProblemSourceName(const CudaProblemSource source) {
  switch (source) {
    case CudaProblemSource::kLegacySnapshot: return "legacy_snapshot";
    case CudaProblemSource::kActiveSpec: return "active_spec";
    case CudaProblemSource::kIndexedCatalog: return "indexed_catalog";
    case CudaProblemSource::kNativeGraph: return "native_graph";
  }
  return "invalid";
}

ActiveBaSolveSpecBuilder::ActiveBaSolveSpecBuilder()
    : impl_(new Impl()) {}

ActiveBaSolveSpecBuilder::~ActiveBaSolveSpecBuilder() = default;

uint64_t ActiveBaSolveSpecBuilder::ResidualBlockCount() const noexcept {
  return impl_->source_order.size();
}

uint64_t ActiveBaSolveSpecBuilder::ScalarResidualCount() const noexcept {
  return 2 * impl_->observations.size() + impl_->lidar.size();
}

void ActiveBaSolveSpecBuilder::RecordVisualResidual(
    const uint32_t image_id,
    const uint32_t point2D_idx,
    const uint64_t point3D_id,
    const std::array<double, 2>& xy,
    const bool pose_constant) {
  const uint64_t source_index = impl_->source_order.size();
  ObservationSnapshot observation;
  observation.source_index = source_index;
  observation.image_id = image_id;
  observation.point2D_idx = point2D_idx;
  observation.point3D_id = point3D_id;
  observation.pose_constant = pose_constant;
  observation.xy = xy;
  impl_->observations.push_back(observation);

  OrderEntrySnapshot order;
  order.source_index = source_index;
  order.residual_kind = ResidualKind::kVisual;
  order.image_id = image_id;
  order.point2D_idx = point2D_idx;
  order.point3D_id = point3D_id;
  impl_->source_order.push_back(order);

  HashWord(&impl_->descriptor_identity, 0);
  HashWord(&impl_->descriptor_identity, source_index);
  HashWord(&impl_->descriptor_identity, image_id);
  HashWord(&impl_->descriptor_identity, point2D_idx);
  HashWord(&impl_->descriptor_identity, point3D_id);
  HashWord(&impl_->descriptor_identity, pose_constant);
  HashDouble(&impl_->descriptor_identity, xy[0]);
  HashDouble(&impl_->descriptor_identity, xy[1]);
  impl_->descriptor_hash_updates += 8;
  ++impl_->descriptor_items;
}

void ActiveBaSolveSpecBuilder::RecordLidarResidual(
    const uint64_t point3D_id,
    const uint8_t lidar_type,
    const std::array<double, 3>& lidar_xyz,
    const std::array<double, 4>& plane,
    const double weight,
    const bool has_search_range,
    const double search_range) {
  const uint64_t source_index = impl_->source_order.size();
  LidarSnapshot value;
  value.source_index = source_index;
  value.point3D_id = point3D_id;
  value.lidar_type = lidar_type;
  value.lidar_xyz = lidar_xyz;
  value.plane = plane;
  value.weight = weight;
  value.has_search_range = has_search_range;
  value.search_range = search_range;
  impl_->lidar.push_back(value);

  OrderEntrySnapshot order;
  order.source_index = source_index;
  order.residual_kind = ResidualKind::kLidar;
  order.image_id = std::numeric_limits<uint32_t>::max();
  order.point2D_idx = std::numeric_limits<uint32_t>::max();
  order.point3D_id = point3D_id;
  impl_->source_order.push_back(order);

  HashWord(&impl_->descriptor_identity, 1);
  HashWord(&impl_->descriptor_identity, source_index);
  HashWord(&impl_->descriptor_identity, point3D_id);
  HashWord(&impl_->descriptor_identity, lidar_type);
  HashWord(&impl_->descriptor_identity, has_search_range);
  HashDouble(&impl_->descriptor_identity, search_range);
  HashDouble(&impl_->descriptor_identity, weight);
  for (const double value : lidar_xyz) {
    HashDouble(&impl_->descriptor_identity, value);
  }
  for (const double value : plane) {
    HashDouble(&impl_->descriptor_identity, value);
  }
  impl_->descriptor_hash_updates += 13;
  ++impl_->descriptor_items;
}

void ActiveBaSolveSpecBuilder::RecordParameterBlock(
    const ParameterKind kind, const uint64_t entity_id) {
  const ParameterIdentityKey key{kind, entity_id};
  if (impl_->parameter_keys.insert(key).second) {
    impl_->parameters.emplace_back(kind, entity_id);
  }
}

bool ActiveBaSolveSpecBuilder::Finalize(
    const BundleAdjustmentOptions& options,
    const ceres::Solver::Options& effective_solver_options,
    const BundleAdjustmentConfig& config,
    const Reconstruction& reconstruction,
    const std::unordered_set<camera_t>& active_camera_ids,
    const std::unordered_map<point3D_t, size_t>& point_observation_counts,
    const BaKind ba_kind,
    const uint64_t ba_call_index,
    const uint64_t owner_epoch,
      const uint64_t catalog_revision,
      ActiveBaSolveSpec* output,
      std::string* error) {
  if (output == nullptr || error == nullptr) return false;
  error->clear();
  const auto descriptor_start = std::chrono::steady_clock::now();
  ActiveBaSolveSpec spec;
  spec.owner_epoch = owner_epoch;
  spec.catalog_revision = catalog_revision;
  spec.problem.metadata.ba_kind = ba_kind;
  spec.problem.metadata.registered_image_count = reconstruction.NumRegImages();
  spec.problem.metadata.ba_call_index = ba_call_index;
  spec.problem.metadata.refinement_index = options.ba_refinement_index;
  spec.problem.metadata.trigger_image_id = options.ba_trigger_image_id;
  spec.problem.metadata.optimize_phrase = BaKindName(ba_kind);
  spec.problem.metadata.backend = options.ba_backend;
  spec.problem.metadata.loss_function =
      LossFunctionName(options.loss_function_type);
  spec.problem.metadata.lidar_residual_mode = options.ba_lidar_residual;
  spec.problem.metadata.lidar_correspondence_version =
      "colmap-pcd-cpu-correspondence-v1";
  spec.problem.metadata.schur_mode = options.ba_cuda_schur_mode;
  spec.problem.metadata.refine_focal_length = options.refine_focal_length;
  spec.problem.metadata.refine_principal_point =
      options.refine_principal_point;
  spec.problem.metadata.refine_extra_params = options.refine_extra_params;
  spec.problem.metadata.refine_extrinsics = options.refine_extrinsics;
  spec.problem.metadata.proj_lidar_weight =
      options.proj_lidar_constraint_weight;
  spec.problem.metadata.icp_lidar_weight =
      options.icp_lidar_constraint_weight;
  spec.problem.metadata.icp_ground_lidar_weight =
      options.icp_ground_lidar_constraint_weight;
  spec.problem.metadata.function_tolerance =
      effective_solver_options.function_tolerance;
  spec.problem.metadata.gradient_tolerance =
      effective_solver_options.gradient_tolerance;
  spec.problem.metadata.parameter_tolerance =
      effective_solver_options.parameter_tolerance;
  spec.problem.metadata.max_num_iterations =
      effective_solver_options.max_num_iterations;
  spec.problem.metadata.max_linear_solver_iterations =
      effective_solver_options.max_linear_solver_iterations;
  spec.problem.metadata.max_consecutive_invalid_steps =
      effective_solver_options.max_num_consecutive_invalid_steps;
  spec.problem.metadata.snapshot_id =
      "active-spec-call" + std::to_string(ba_call_index);

  spec.residual_block_count = impl_->source_order.size();
  spec.scalar_residual_count =
      2 * impl_->observations.size() + impl_->lidar.size();
  spec.residual_enumerator_passes = 1;
  spec.residual_enumerator_items = impl_->source_order.size();

  std::set<uint32_t> image_ids(config.Images().begin(), config.Images().end());
  std::set<uint64_t> point_ids;
  std::set<uint32_t> camera_ids(active_camera_ids.begin(),
                                active_camera_ids.end());
  for (const ObservationSnapshot& value : impl_->observations) {
    image_ids.insert(value.image_id);
    point_ids.insert(value.point3D_id);
  }
  for (const LidarSnapshot& value : impl_->lidar) {
    point_ids.insert(value.point3D_id);
  }
  for (const auto& recorded : impl_->parameters) {
    if (recorded.first == ParameterKind::kPoint3D) {
      point_ids.insert(recorded.second);
    } else if (recorded.first == ParameterKind::kCamera) {
      camera_ids.insert(static_cast<uint32_t>(recorded.second));
    }
  }

  std::unordered_map<ParameterIdentityKey, ActiveBaParameterBlockSpec,
                     ParameterIdentityKeyHash> parameter_lookup;
  parameter_lookup.reserve(impl_->parameters.size());
  for (size_t index = 0; index < impl_->parameters.size(); ++index) {
    const ParameterKind kind = impl_->parameters[index].first;
    const uint64_t entity_id = impl_->parameters[index].second;
    ActiveBaParameterBlockSpec parameter;
    parameter.kind = kind;
    parameter.entity_id = entity_id;
    switch (kind) {
      case ParameterKind::kQuaternion:
        parameter.ambient_size = 4;
        parameter.tangent_size = 3;
        break;
      case ParameterKind::kTranslation: {
        parameter.ambient_size = 3;
        uint8_t mask = 0;
        if (config.HasConstantTvec(static_cast<image_t>(entity_id))) {
          for (const int component :
               config.ConstantTvec(static_cast<image_t>(entity_id))) {
            if (component >= 0 && component < 3) {
              mask |= static_cast<uint8_t>(1u << component);
            }
          }
        }
        parameter.translation_subset_mask = mask;
        parameter.tangent_size = 3 - PopCount3(mask);
        parameter.constant = parameter.tangent_size == 0;
        break;
      }
      case ParameterKind::kPoint3D: {
        parameter.ambient_size = 3;
        parameter.tangent_size = 3;
        const point3D_t point_id = static_cast<point3D_t>(entity_id);
        const auto count = point_observation_counts.find(point_id);
        const size_t selected_count =
            count == point_observation_counts.end() ? 0 : count->second;
        parameter.constant = config.HasConstantPoint(point_id) ||
            reconstruction.Point3D(point_id).Track().Length() > selected_count;
        break;
      }
      case ParameterKind::kCamera: {
        const Camera& camera =
            reconstruction.Camera(static_cast<camera_t>(entity_id));
        parameter.ambient_size = static_cast<uint32_t>(camera.NumParams());
        std::set<size_t> fixed_indices;
        if (!options.refine_focal_length) {
          fixed_indices.insert(camera.FocalLengthIdxs().begin(),
                               camera.FocalLengthIdxs().end());
        }
        if (!options.refine_principal_point) {
          fixed_indices.insert(camera.PrincipalPointIdxs().begin(),
                               camera.PrincipalPointIdxs().end());
        }
        if (!options.refine_extra_params) {
          fixed_indices.insert(camera.ExtraParamsIdxs().begin(),
                               camera.ExtraParamsIdxs().end());
        }
        parameter.tangent_size = parameter.ambient_size -
            static_cast<uint32_t>(fixed_indices.size());
        parameter.fixed_camera_parameter_indices.assign(
            fixed_indices.begin(), fixed_indices.end());
        parameter.constant =
            config.IsConstantCamera(static_cast<camera_t>(entity_id)) ||
            parameter.tangent_size == 0;
        // Ceres keeps the ambient local size when a camera block is marked
        // constant; constancy, rather than a zero tangent size, removes it
        // from the effective parameter count.
        if (parameter.constant) parameter.tangent_size = parameter.ambient_size;
        break;
      }
    }
    ParameterBlockSnapshot snapshot;
    snapshot.source_index = index;
    snapshot.kind = kind;
    snapshot.entity_id = entity_id;
    snapshot.ambient_size = parameter.ambient_size;
    snapshot.tangent_size = parameter.tangent_size;
    snapshot.constant = parameter.constant;
    spec.problem.parameter_blocks_source_order.push_back(snapshot);
    spec.parameter_blocks.push_back(parameter);
    spec.ambient_parameter_count += parameter.ambient_size;
    if (!parameter.constant) {
      spec.effective_parameter_count += parameter.tangent_size;
    }
    parameter_lookup.emplace(ParameterIdentityKey{kind, entity_id}, parameter);
  }

  for (const uint32_t image_id : image_ids) {
    if (!reconstruction.ExistsImage(image_id)) {
      *error = "active spec references a missing image";
      return false;
    }
    const Image& image = reconstruction.Image(image_id);
    camera_ids.insert(image.CameraId());
    const auto q = parameter_lookup.find(
        ParameterIdentityKey{ParameterKind::kQuaternion, image_id});
    const auto t = parameter_lookup.find(
        ParameterIdentityKey{ParameterKind::kTranslation, image_id});
    ImageSnapshot value;
    value.image_id = image_id;
    value.camera_id = image.CameraId();
    value.selected = config.HasImage(image_id);
    value.has_pose_parameter_blocks =
        q != parameter_lookup.end() && t != parameter_lookup.end();
    value.pose_constant = !value.has_pose_parameter_blocks;
    if (t != parameter_lookup.end()) {
      value.constant_tvec_mask = t->second.translation_subset_mask;
    }
    for (size_t i = 0; i < 4; ++i) value.qvec[i] = image.Qvec()[i];
    for (size_t i = 0; i < 3; ++i) value.tvec[i] = image.Tvec()[i];
    spec.problem.images.push_back(value);
  }

  for (const uint32_t camera_id : camera_ids) {
    if (!reconstruction.ExistsCamera(camera_id)) {
      *error = "active spec references a missing camera";
      return false;
    }
    const Camera& camera = reconstruction.Camera(camera_id);
    const auto parameter = parameter_lookup.find(
        ParameterIdentityKey{ParameterKind::kCamera, camera_id});
    CameraSnapshot value;
    value.camera_id = camera_id;
    value.model_id = camera.ModelId();
    value.width = camera.Width();
    value.height = camera.Height();
    value.constant = parameter == parameter_lookup.end() ||
                     parameter->second.constant;
    value.params.assign(camera.Params().begin(), camera.Params().end());
    spec.problem.cameras.push_back(std::move(value));
  }

  for (const uint64_t point_id : point_ids) {
    if (!reconstruction.ExistsPoint3D(point_id)) {
      *error = "active spec references a missing point";
      return false;
    }
    const Point3D& point = reconstruction.Point3D(point_id);
    const auto parameter = parameter_lookup.find(
        ParameterIdentityKey{ParameterKind::kPoint3D, point_id});
    PointSnapshot value;
    value.point3D_id = point_id;
    value.constant = parameter == parameter_lookup.end() ||
                     parameter->second.constant;
    value.config_role = config.HasVariablePoint(point_id)
                            ? 1
                            : (config.HasConstantPoint(point_id) ? 2 : 0);
    const auto range = config.LidarSearchRanges().find(point_id);
    if (range != config.LidarSearchRanges().end()) {
      value.has_search_range = true;
      value.search_range = range->second;
    }
    for (size_t i = 0; i < 3; ++i) value.xyz[i] = point.XYZ()[i];
    spec.problem.points.push_back(value);
  }

  uint64_t descriptor = impl_->descriptor_identity;
  uint64_t descriptor_updates = impl_->descriptor_hash_updates;
  const auto hash_word = [&](const uint64_t value) {
    HashWord(&descriptor, value);
    ++descriptor_updates;
  };
  const auto hash_double = [&](const double value) {
    HashDouble(&descriptor, value);
    ++descriptor_updates;
  };
  const auto hash_string = [&](const std::string& value) {
    HashString(&descriptor, value);
    ++descriptor_updates;
  };
  hash_word(0x4853545657455631ull);
  hash_word(static_cast<uint64_t>(ba_kind));
  hash_word(options.refine_focal_length);
  hash_word(options.refine_principal_point);
  hash_word(options.refine_extra_params);
  hash_word(options.refine_extrinsics);
  hash_string(spec.problem.metadata.loss_function);
  hash_string(spec.problem.metadata.lidar_residual_mode);
  for (const CameraSnapshot& value : spec.problem.cameras) {
    hash_word(value.camera_id);
    hash_word(static_cast<uint64_t>(value.model_id));
    hash_word(value.width);
    hash_word(value.height);
    hash_word(value.constant);
    hash_word(value.params.size());
    for (const double parameter : value.params) hash_double(parameter);
  }
  for (const ImageSnapshot& value : spec.problem.images) {
    hash_word(value.image_id);
    hash_word(value.camera_id);
    hash_word(value.selected);
    hash_word(value.pose_constant);
    hash_word(value.has_pose_parameter_blocks);
    hash_word(value.constant_tvec_mask);
  }
  for (const PointSnapshot& value : spec.problem.points) {
    hash_word(value.point3D_id);
    hash_word(value.constant);
    hash_word(value.config_role);
    hash_word(value.has_search_range);
    if (value.has_search_range) hash_double(value.search_range);
  }
  for (const ParameterBlockSnapshot& value :
       spec.problem.parameter_blocks_source_order) {
    hash_word(value.source_index);
    hash_word(static_cast<uint64_t>(value.kind));
    hash_word(value.entity_id);
    hash_word(value.ambient_size);
    hash_word(value.tangent_size);
    hash_word(value.constant);
  }
  spec.problem.prepared_host_topology_identity = descriptor == 0 ? 1 : descriptor;
  spec.problem.prepared_host_descriptor_items =
      impl_->descriptor_items + spec.problem.cameras.size() +
      spec.problem.images.size() + spec.problem.points.size() +
      spec.problem.parameter_blocks_source_order.size();
  spec.problem.prepared_host_descriptor_hash_updates = descriptor_updates;
  spec.problem.prepared_host_descriptor_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - descriptor_start).count();
  spec.problem.observations = std::move(impl_->observations);
  spec.problem.lidar = std::move(impl_->lidar);
  spec.problem.source_insertion_order = std::move(impl_->source_order);
  *output = std::move(spec);
  return true;
}

bool CompareActiveBaSolveSpecToSnapshot(const ActiveBaSolveSpec& active,
                                        const Snapshot& legacy,
                                        std::string* error) {
  if (error == nullptr) return false;
  error->clear();
  uint64_t legacy_ambient_parameters = 0;
  uint64_t legacy_effective_parameters = 0;
  for (const ParameterBlockSnapshot& parameter :
       legacy.parameter_blocks_source_order) {
    legacy_ambient_parameters += parameter.ambient_size;
    if (!parameter.constant) legacy_effective_parameters += parameter.tangent_size;
  }
  std::unordered_set<uint32_t> relevant_image_ids;
  std::unordered_set<uint64_t> relevant_point_ids;
  std::unordered_set<uint32_t> relevant_camera_ids;
  std::unordered_set<ParameterIdentityKey, ParameterIdentityKeyHash>
      parameter_keys;
  parameter_keys.reserve(legacy.parameter_blocks_source_order.size());
  for (const ParameterBlockSnapshot& parameter :
       legacy.parameter_blocks_source_order) {
    parameter_keys.insert(
        ParameterIdentityKey{parameter.kind, parameter.entity_id});
    if (parameter.kind == ParameterKind::kQuaternion ||
        parameter.kind == ParameterKind::kTranslation) {
      relevant_image_ids.insert(static_cast<uint32_t>(parameter.entity_id));
    } else if (parameter.kind == ParameterKind::kPoint3D) {
      relevant_point_ids.insert(parameter.entity_id);
    } else if (parameter.kind == ParameterKind::kCamera) {
      relevant_camera_ids.insert(static_cast<uint32_t>(parameter.entity_id));
    }
  }
  for (const ObservationSnapshot& observation : legacy.observations) {
    relevant_image_ids.insert(observation.image_id);
    relevant_point_ids.insert(observation.point3D_id);
  }
  for (const LidarSnapshot& lidar : legacy.lidar) {
    relevant_point_ids.insert(lidar.point3D_id);
  }
  for (const ImageSnapshot& image : legacy.images) {
    if (image.selected) relevant_image_ids.insert(image.image_id);
  }
  for (const ImageSnapshot& image : legacy.images) {
    if (relevant_image_ids.count(image.image_id) != 0) {
      relevant_camera_ids.insert(image.camera_id);
    }
  }

  std::vector<CameraSnapshot> legacy_cameras;
  std::vector<ImageSnapshot> legacy_images;
  std::vector<PointSnapshot> legacy_points;
  legacy_cameras.reserve(relevant_camera_ids.size());
  legacy_images.reserve(relevant_image_ids.size());
  legacy_points.reserve(relevant_point_ids.size());
  for (const CameraSnapshot& camera : legacy.cameras) {
    if (relevant_camera_ids.count(camera.camera_id) != 0) {
      legacy_cameras.push_back(camera);
      continue;
    }
    if (parameter_keys.count(
            ParameterIdentityKey{ParameterKind::kCamera, camera.camera_id}) !=
        0) {
      *error = "discarded legacy camera remains solver-relevant";
      return false;
    }
  }
  for (const ImageSnapshot& image : legacy.images) {
    if (relevant_image_ids.count(image.image_id) != 0) {
      legacy_images.push_back(image);
      continue;
    }
    const bool has_parameter =
        parameter_keys.count(
            ParameterIdentityKey{ParameterKind::kQuaternion, image.image_id}) !=
            0 ||
        parameter_keys.count(
            ParameterIdentityKey{ParameterKind::kTranslation, image.image_id}) !=
            0;
    const bool has_residual = std::any_of(
        legacy.observations.begin(), legacy.observations.end(),
        [&](const ObservationSnapshot& observation) {
          return observation.image_id == image.image_id;
        });
    if (image.selected || image.has_pose_parameter_blocks || has_parameter ||
        has_residual) {
      *error = "discarded legacy track-only image remains solver-relevant";
      return false;
    }
  }
  for (const PointSnapshot& point : legacy.points) {
    if (relevant_point_ids.count(point.point3D_id) != 0) {
      legacy_points.push_back(point);
      continue;
    }
    const bool has_parameter = parameter_keys.count(
        ParameterIdentityKey{ParameterKind::kPoint3D, point.point3D_id}) != 0;
    const bool has_visual = std::any_of(
        legacy.observations.begin(), legacy.observations.end(),
        [&](const ObservationSnapshot& observation) {
          return observation.point3D_id == point.point3D_id;
        });
    const bool has_lidar = std::any_of(
        legacy.lidar.begin(), legacy.lidar.end(),
        [&](const LidarSnapshot& lidar) {
          return lidar.point3D_id == point.point3D_id;
        });
    if (has_parameter || has_visual || has_lidar) {
      *error = "discarded legacy track-only point remains solver-relevant";
      return false;
    }
  }

  if (!SameMetadataSemantics(active.problem.metadata, legacy.metadata) ||
      active.residual_block_count !=
          legacy.observations.size() + legacy.lidar.size() ||
      active.scalar_residual_count !=
          2 * legacy.observations.size() + legacy.lidar.size() ||
      active.ambient_parameter_count != legacy_ambient_parameters ||
      active.effective_parameter_count != legacy_effective_parameters ||
      active.problem.cameras.size() != legacy_cameras.size() ||
      active.problem.images.size() != legacy_images.size() ||
      active.problem.points.size() != legacy_points.size() ||
      active.problem.observations.size() != legacy.observations.size() ||
      active.problem.lidar.size() != legacy.lidar.size() ||
      active.problem.source_insertion_order.size() !=
          legacy.source_insertion_order.size() ||
      active.problem.parameter_blocks_source_order.size() !=
          legacy.parameter_blocks_source_order.size()) {
    *error = "active/legacy metadata or count mismatch";
    return false;
  }
  if (!SameVector(active.problem.cameras, legacy_cameras, SameCamera) ||
      !SameVector(active.problem.images, legacy_images, SameImage) ||
      !SameVector(active.problem.points, legacy_points, SamePoint) ||
      !SameVector(active.problem.observations, legacy.observations,
                  SameObservation) ||
      !SameVector(active.problem.lidar, legacy.lidar, SameLidar) ||
      !SameVector(active.problem.source_insertion_order,
                  legacy.source_insertion_order, SameOrderEntry)) {
    *error = "active/legacy entity, state, residual, or order mismatch";
    return false;
  }
  for (size_t i = 0;
       i < active.problem.parameter_blocks_source_order.size(); ++i) {
    const ParameterBlockSnapshot& lhs =
        active.problem.parameter_blocks_source_order[i];
    const ParameterBlockSnapshot& rhs =
        legacy.parameter_blocks_source_order[i];
    if (!SameParameter(lhs, rhs)) {
      *error =
          "active/legacy parameter metadata mismatch at index " +
          std::to_string(i) + " active={source=" +
          std::to_string(lhs.source_index) + ",kind=" +
          std::to_string(static_cast<uint32_t>(lhs.kind)) + ",id=" +
          std::to_string(lhs.entity_id) + ",ambient=" +
          std::to_string(lhs.ambient_size) + ",tangent=" +
          std::to_string(lhs.tangent_size) + ",constant=" +
          std::to_string(lhs.constant) + "} legacy={source=" +
          std::to_string(rhs.source_index) + ",kind=" +
          std::to_string(static_cast<uint32_t>(rhs.kind)) + ",id=" +
          std::to_string(rhs.entity_id) + ",ambient=" +
          std::to_string(rhs.ambient_size) + ",tangent=" +
          std::to_string(rhs.tangent_size) + ",constant=" +
          std::to_string(rhs.constant) + "}";
      return false;
    }
    const ActiveBaParameterBlockSpec& typed = active.parameter_blocks[i];
    if (typed.kind != lhs.kind || typed.entity_id != lhs.entity_id ||
        typed.ambient_size != lhs.ambient_size ||
        typed.tangent_size != lhs.tangent_size ||
        typed.constant != lhs.constant) {
      *error = "active typed/source parameter metadata mismatch";
      return false;
    }
    if (typed.kind == ParameterKind::kTranslation) {
      const auto image = std::find_if(
          active.problem.images.begin(), active.problem.images.end(),
          [&](const ImageSnapshot& value) {
            return value.image_id == typed.entity_id;
          });
      if (image == active.problem.images.end() ||
          typed.translation_subset_mask != image->constant_tvec_mask) {
        *error = "active translation subset identity mismatch";
        return false;
      }
    } else if (typed.kind == ParameterKind::kCamera) {
      const auto camera = std::find_if(
          active.problem.cameras.begin(), active.problem.cameras.end(),
          [&](const CameraSnapshot& value) {
            return value.camera_id == typed.entity_id;
          });
      if (camera == active.problem.cameras.end()) {
        *error = "active camera parameter references a missing camera";
        return false;
      }
      std::set<uint32_t> expected;
      const auto append = [&](const std::vector<size_t>& indices) {
        for (const size_t index : indices) {
          expected.insert(static_cast<uint32_t>(index));
        }
      };
      if (!active.problem.metadata.refine_focal_length) {
        append(CameraModelFocalLengthIdxs(camera->model_id));
      }
      if (!active.problem.metadata.refine_principal_point) {
        append(CameraModelPrincipalPointIdxs(camera->model_id));
      }
      if (!active.problem.metadata.refine_extra_params) {
        append(CameraModelExtraParamsIdxs(camera->model_id));
      }
      const std::vector<uint32_t> expected_indices(expected.begin(),
                                                   expected.end());
      if (typed.fixed_camera_parameter_indices != expected_indices) {
        *error = "active camera fixed-index identity mismatch";
        return false;
      }
    }
  }
  return true;
}

bool ValidateAndCommitActiveBaState(const ActiveBaSolveSpec& initial,
                                    const CudaSolveProblem& candidate,
                                    Reconstruction* reconstruction,
                                    std::string* error) {
  if (reconstruction == nullptr || error == nullptr) return false;
  error->clear();
  if (initial.problem.cameras.size() != candidate.cameras.size() ||
      initial.problem.images.size() != candidate.images.size() ||
      initial.problem.points.size() != candidate.points.size() ||
      initial.problem.observations.size() != candidate.observations.size() ||
      initial.problem.lidar.size() != candidate.lidar.size() ||
      initial.problem.parameter_blocks_source_order.size() !=
          candidate.parameter_blocks_source_order.size() ||
      initial.problem.source_insertion_order.size() !=
          candidate.source_insertion_order.size()) {
    *error = "ACTIVE_SPEC_COMMIT_TOPOLOGY_COUNT_MISMATCH";
    return false;
  }
  for (size_t i = 0; i < initial.problem.observations.size(); ++i) {
    if (!SameObservation(initial.problem.observations[i],
                         candidate.observations[i])) {
      *error = "ACTIVE_SPEC_COMMIT_VISUAL_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.problem.lidar.size(); ++i) {
    if (!SameLidar(initial.problem.lidar[i], candidate.lidar[i])) {
      *error = "ACTIVE_SPEC_COMMIT_LIDAR_IDENTITY_MISMATCH";
      return false;
    }
  }
  for (size_t i = 0; i < initial.problem.source_insertion_order.size(); ++i) {
    if (!SameOrderEntry(initial.problem.source_insertion_order[i],
                        candidate.source_insertion_order[i])) {
      *error = "ACTIVE_SPEC_COMMIT_RESIDUAL_ORDER_MISMATCH";
      return false;
    }
  }
  if (!SameMetadataSemantics(initial.problem.metadata, candidate.metadata) ||
      initial.parameter_blocks.size() !=
          candidate.parameter_blocks_source_order.size()) {
    *error = "ACTIVE_SPEC_COMMIT_METADATA_MISMATCH";
    return false;
  }
  for (size_t i = 0; i < initial.parameter_blocks.size(); ++i) {
    const ActiveBaParameterBlockSpec& typed = initial.parameter_blocks[i];
    const ParameterBlockSnapshot& before =
        initial.problem.parameter_blocks_source_order[i];
    const ParameterBlockSnapshot& after =
        candidate.parameter_blocks_source_order[i];
    if (!SameParameter(before, after) || typed.kind != before.kind ||
        typed.entity_id != before.entity_id ||
        typed.ambient_size != before.ambient_size ||
        typed.tangent_size != before.tangent_size ||
        typed.constant != before.constant) {
      *error = "ACTIVE_SPEC_COMMIT_PARAMETER_BLOCK_MISMATCH";
      return false;
    }
  }
  struct CameraUpdate {
    uint32_t id = 0;
    std::vector<double> params;
  };
  struct ImageUpdate {
    uint32_t id = 0;
    bool update_quaternion = false;
    std::array<double, 4> quaternion{};
    std::array<double, 3> translation{};
    uint8_t translation_mask = 0;
  };
  struct PointUpdate {
    uint64_t id = 0;
    std::array<double, 3> xyz{};
  };
  std::vector<CameraUpdate> camera_updates;
  std::vector<ImageUpdate> image_updates;
  std::vector<PointUpdate> point_updates;
  for (size_t i = 0; i < initial.problem.cameras.size(); ++i) {
    const CameraSnapshot& before = initial.problem.cameras[i];
    const CameraSnapshot& after = candidate.cameras[i];
    if (before.camera_id != after.camera_id ||
        before.model_id != after.model_id || before.width != after.width ||
        before.height != after.height || before.constant != after.constant ||
        before.params.size() != after.params.size() ||
        !reconstruction->ExistsCamera(before.camera_id)) {
      *error = "ACTIVE_SPEC_COMMIT_CAMERA_IDENTITY_MISMATCH";
      return false;
    }
    const Camera& current = reconstruction->Camera(before.camera_id);
    if (current.ModelId() != before.model_id || current.Width() != before.width ||
        current.Height() != before.height ||
        current.Params().size() != before.params.size()) {
      *error = "ACTIVE_SPEC_COMMIT_CAMERA_RECONSTRUCTION_MISMATCH";
      return false;
    }
    if (!std::all_of(after.params.begin(), after.params.end(),
                     [](const double value) { return std::isfinite(value); })) {
      *error = "ACTIVE_SPEC_COMMIT_NONFINITE_CAMERA";
      return false;
    }
    const auto parameter = std::find_if(
        initial.parameter_blocks.begin(), initial.parameter_blocks.end(),
        [&](const ActiveBaParameterBlockSpec& value) {
          return value.kind == ParameterKind::kCamera &&
                 value.entity_id == before.camera_id;
        });
    if (before.constant) {
      if (before.params != after.params) {
        *error = "ACTIVE_SPEC_COMMIT_FIXED_CAMERA_CHANGED";
        return false;
      }
    } else {
      if (parameter == initial.parameter_blocks.end()) {
        *error = "ACTIVE_SPEC_COMMIT_CAMERA_PARAMETER_MISSING";
        return false;
      }
      if (parameter->constant ||
          parameter->ambient_size != before.params.size()) {
        *error = "ACTIVE_SPEC_COMMIT_CAMERA_PARAMETER_MISMATCH";
        return false;
      }
      for (const uint32_t fixed :
           parameter->fixed_camera_parameter_indices) {
        if (fixed >= before.params.size() ||
            before.params[fixed] != after.params[fixed]) {
          *error = "ACTIVE_SPEC_COMMIT_FIXED_CAMERA_COMPONENT_CHANGED";
          return false;
        }
      }
      camera_updates.push_back({before.camera_id, after.params});
    }
  }
  for (size_t i = 0; i < initial.problem.images.size(); ++i) {
    const ImageSnapshot& before = initial.problem.images[i];
    const ImageSnapshot& after = candidate.images[i];
    if (before.image_id != after.image_id ||
        before.camera_id != after.camera_id ||
        before.selected != after.selected ||
        before.pose_constant != after.pose_constant ||
        before.has_pose_parameter_blocks != after.has_pose_parameter_blocks ||
        before.constant_tvec_mask != after.constant_tvec_mask) {
      *error = "ACTIVE_SPEC_COMMIT_IMAGE_IDENTITY_MISMATCH";
      return false;
    }
    if (!std::all_of(after.qvec.begin(), after.qvec.end(),
                     [](const double value) { return std::isfinite(value); }) ||
        !std::all_of(after.tvec.begin(), after.tvec.end(),
                     [](const double value) { return std::isfinite(value); })) {
      *error = "ACTIVE_SPEC_COMMIT_NONFINITE_IMAGE_STATE";
      return false;
    }
    if (before.pose_constant &&
        (before.qvec != after.qvec || before.tvec != after.tvec)) {
      *error = "ACTIVE_SPEC_COMMIT_FIXED_POSE_CHANGED";
      return false;
    }
    for (size_t component = 0; component < 3; ++component) {
      if ((before.constant_tvec_mask & (1u << component)) != 0 &&
          before.tvec[component] != after.tvec[component]) {
        *error = "ACTIVE_SPEC_COMMIT_FIXED_TRANSLATION_CHANGED";
        return false;
      }
    }
    if (!reconstruction->ExistsImage(before.image_id)) {
      *error = "ACTIVE_SPEC_COMMIT_IMAGE_MISSING";
      return false;
    }
    if (before.has_pose_parameter_blocks) {
      const double norm = std::sqrt(
          after.qvec[0] * after.qvec[0] + after.qvec[1] * after.qvec[1] +
          after.qvec[2] * after.qvec[2] + after.qvec[3] * after.qvec[3]);
      if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-10) {
        *error = "ACTIVE_SPEC_COMMIT_QUATERNION_NORM";
        return false;
      }
      ImageUpdate update;
      update.id = before.image_id;
      update.update_quaternion = !before.pose_constant;
      update.quaternion = after.qvec;
      update.translation = after.tvec;
      update.translation_mask = before.constant_tvec_mask;
      image_updates.push_back(update);
    }
  }
  for (size_t i = 0; i < initial.problem.points.size(); ++i) {
    const PointSnapshot& before = initial.problem.points[i];
    const PointSnapshot& after = candidate.points[i];
    if (before.point3D_id != after.point3D_id ||
        before.constant != after.constant ||
        before.config_role != after.config_role ||
        before.has_search_range != after.has_search_range ||
        before.search_range != after.search_range) {
      *error = "ACTIVE_SPEC_COMMIT_POINT_IDENTITY_MISMATCH";
      return false;
    }
    if (!std::all_of(after.xyz.begin(), after.xyz.end(),
                     [](const double value) { return std::isfinite(value); })) {
      *error = "ACTIVE_SPEC_COMMIT_NONFINITE_POINT";
      return false;
    }
    if (before.constant && before.xyz != after.xyz) {
      *error = "ACTIVE_SPEC_COMMIT_FIXED_POINT_CHANGED";
      return false;
    }
    if (!reconstruction->ExistsPoint3D(before.point3D_id)) {
      *error = "ACTIVE_SPEC_COMMIT_POINT_MISSING";
      return false;
    }
    if (!before.constant) {
      point_updates.push_back({before.point3D_id, after.xyz});
    }
  }

  for (const CameraUpdate& update : camera_updates) {
    std::copy(update.params.begin(), update.params.end(),
              reconstruction->Camera(update.id).Params().begin());
  }
  for (const ImageUpdate& update : image_updates) {
    Image& image = reconstruction->Image(update.id);
    if (update.update_quaternion) {
      image.SetQvec(Eigen::Map<const Eigen::Vector4d>(
          update.quaternion.data()));
    }
    Eigen::Vector3d translation = image.Tvec();
    for (size_t component = 0; component < 3; ++component) {
      if ((update.translation_mask & (1u << component)) == 0) {
        translation[component] = update.translation[component];
      }
    }
    image.SetTvec(translation);
  }
  for (const PointUpdate& update : point_updates) {
    reconstruction->Point3D(update.id).XYZ() =
        Eigen::Map<const Eigen::Vector3d>(update.xyz.data());
  }
  return true;
}

}  // namespace gpu_ba
}  // namespace colmap
