#include "gpu_ba/native_graph_problem_store.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Core>

#include "base/reconstruction.h"
#include "gpu_ba/host_problem_store_internal.h"
#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

struct ParameterKey {
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;
  bool operator==(const ParameterKey& other) const noexcept {
    return kind == other.kind && entity_id == other.entity_id;
  }
};

struct ParameterKeyHash {
  size_t operator()(const ParameterKey& value) const noexcept {
    return std::hash<uint64_t>()(
        (value.entity_id << 3) ^ static_cast<uint64_t>(value.kind));
  }
};

template <size_t N>
bool Finite(const std::array<double, N>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

uint64_t ObservationKey(const uint32_t image_id,
                        const uint32_t point2D_idx) noexcept {
  return (static_cast<uint64_t>(image_id) << 32) | point2D_idx;
}

bool BuildColdInput(const Reconstruction& reconstruction,
                    const uint64_t owner_epoch,
                    HostBaGraphColdInput* output,
                    std::string* error) {
  if (output == nullptr || error == nullptr || owner_epoch == 0 ||
      reconstruction.StructureOwnerEpoch() != owner_epoch) {
    if (error != nullptr) *error = "invalid native graph cold-build identity";
    return false;
  }
  const uint64_t revision = reconstruction.StructureRevision();
  if (revision == 0) {
    *error = "native graph requires an active structure journal";
    return false;
  }
  HostBaGraphColdInput result;
  result.owner_epoch = owner_epoch;
  result.topology_revision = revision;
  std::vector<uint32_t> camera_ids;
  std::vector<uint32_t> image_ids;
  std::vector<uint64_t> point_ids;
  camera_ids.reserve(reconstruction.NumCameras());
  image_ids.reserve(reconstruction.NumImages());
  point_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& item : reconstruction.Cameras()) camera_ids.push_back(item.first);
  for (const auto& item : reconstruction.Images()) image_ids.push_back(item.first);
  for (const auto& item : reconstruction.Points3D()) point_ids.push_back(item.first);
  std::sort(camera_ids.begin(), camera_ids.end());
  std::sort(image_ids.begin(), image_ids.end());
  std::sort(point_ids.begin(), point_ids.end());
  result.cameras.reserve(camera_ids.size());
  result.images.reserve(image_ids.size());
  result.points.reserve(point_ids.size());
  for (const uint32_t id : camera_ids) {
    const Camera& camera = reconstruction.Camera(id);
    result.cameras.push_back({id, camera.ModelId(), camera.Width(),
                              camera.Height(),
                              static_cast<uint32_t>(camera.Params().size())});
  }
  for (const uint32_t id : image_ids) {
    const Image& image = reconstruction.Image(id);
    if (!reconstruction.ExistsCamera(image.CameraId())) {
      *error = "native graph image references a missing camera";
      return false;
    }
    result.images.push_back({id, image.CameraId(), image.IsRegistered()});
  }
  std::unordered_set<uint64_t> observation_keys;
  for (const uint64_t id : point_ids) {
    const Point3D& point = reconstruction.Point3D(id);
    result.points.push_back({id});
    for (const TrackElement& element : point.Track().Elements()) {
      if (!reconstruction.ExistsImage(element.image_id) ||
          element.point2D_idx >=
              reconstruction.Image(element.image_id).NumPoints2D()) {
        *error = "native graph point track references a missing observation";
        return false;
      }
      const Point2D& point2D =
          reconstruction.Image(element.image_id).Point2D(element.point2D_idx);
      if (!point2D.HasPoint3D() || point2D.Point3DId() != id ||
          !observation_keys.insert(
              ObservationKey(element.image_id, element.point2D_idx)).second) {
        *error = "native graph point track is not reciprocal or is duplicated";
        return false;
      }
      result.observations.push_back(
          {element.image_id, element.point2D_idx, id,
           {{point2D.X(), point2D.Y()}}});
    }
  }
  std::sort(result.observations.begin(), result.observations.end(),
            [](const HostBaObservationRecord& lhs,
               const HostBaObservationRecord& rhs) {
              return std::tie(lhs.image_id, lhs.point2D_idx) <
                     std::tie(rhs.image_id, rhs.point2D_idx);
            });
  if (reconstruction.StructureOwnerEpoch() != owner_epoch ||
      reconstruction.StructureRevision() != revision) {
    *error = "Reconstruction changed during native graph cold build";
    return false;
  }
  *output = std::move(result);
  return true;
}

bool BuildCoalescedMutation(const Reconstruction& reconstruction,
                            const HostStructureReadResult& journal,
                            CoalescedBaGraphMutation* output,
                            bool* full_rebuild,
                            NativeGraphPrepareRuntime* runtime,
                            std::string* error) {
  if (output == nullptr || full_rebuild == nullptr || runtime == nullptr ||
      error == nullptr || !journal.complete || journal.gap) {
    return false;
  }
  *full_rebuild = false;
  CoalescedBaGraphMutation mutation;
  mutation.owner_epoch = journal.owner_epoch;
  mutation.revision_before = journal.cursor;
  mutation.revision_after = journal.current_revision;
  std::unordered_set<uint32_t> cameras;
  std::unordered_set<uint32_t> images;
  std::unordered_set<uint64_t> points;
  std::unordered_set<uint64_t> observations;
  for (const HostStructureBatch& batch : journal.batches) {
    ++runtime->journal_batches;
    for (const HostStructureEvent& event : batch.events) {
      ++runtime->journal_events;
      switch (event.kind) {
        case HostStructureEventKind::kCameraAdded:
          cameras.insert(event.new_camera_id);
          break;
        case HostStructureEventKind::kImageAdded:
        case HostStructureEventKind::kImageRegistrationChanged:
        case HostStructureEventKind::kCameraAssociationChanged:
          images.insert(event.image_id);
          break;
        case HostStructureEventKind::kPointAdded:
          points.insert(event.point3D_id);
          break;
        case HostStructureEventKind::kPointDeleted:
          points.insert(event.point3D_id);
          for (const HostCatalogTrackElement& element : event.old_track) {
            observations.insert(ObservationKey(element.image_id,
                                               element.point2D_idx));
          }
          break;
        case HostStructureEventKind::kPointMerged:
          points.insert(event.old_point3D_id1);
          points.insert(event.old_point3D_id2);
          points.insert(event.point3D_id);
          break;
        case HostStructureEventKind::kObservationAdded:
        case HostStructureEventKind::kObservationDeleted:
          observations.insert(ObservationKey(event.image_id,
                                             event.point2D_idx));
          break;
        case HostStructureEventKind::kBulkUnknown:
          *full_rebuild = true;
          return true;
      }
    }
  }
  std::vector<uint32_t> sorted_cameras(cameras.begin(), cameras.end());
  std::vector<uint32_t> sorted_images(images.begin(), images.end());
  std::vector<uint64_t> sorted_points(points.begin(), points.end());
  std::sort(sorted_cameras.begin(), sorted_cameras.end());
  std::sort(sorted_images.begin(), sorted_images.end());
  std::sort(sorted_points.begin(), sorted_points.end());
  for (const uint32_t id : sorted_cameras) {
    if (!reconstruction.ExistsCamera(id)) {
      *full_rebuild = true;
      return true;
    }
    const Camera& camera = reconstruction.Camera(id);
    mutation.camera_upserts.push_back(
        {id, camera.ModelId(), camera.Width(), camera.Height(),
         static_cast<uint32_t>(camera.Params().size())});
  }
  for (const uint32_t id : sorted_images) {
    if (!reconstruction.ExistsImage(id)) {
      *full_rebuild = true;
      return true;
    }
    const Image& image = reconstruction.Image(id);
    mutation.image_upserts.push_back(
        {id, image.CameraId(), image.IsRegistered()});
  }
  for (const uint64_t id : sorted_points) {
    if (!reconstruction.ExistsPoint3D(id)) {
      mutation.tombstone_point_ids.push_back(id);
      continue;
    }
    mutation.point_upserts.push_back({id});
    const Point3D& point = reconstruction.Point3D(id);
    for (const TrackElement& element : point.Track().Elements()) {
      observations.insert(ObservationKey(element.image_id,
                                         element.point2D_idx));
    }
  }
  std::vector<uint64_t> sorted_observations(observations.begin(),
                                            observations.end());
  std::sort(sorted_observations.begin(), sorted_observations.end());
  for (const uint64_t key : sorted_observations) {
    const uint32_t image_id = static_cast<uint32_t>(key >> 32);
    const uint32_t point2D_idx = static_cast<uint32_t>(key);
    if (!reconstruction.ExistsImage(image_id) ||
        point2D_idx >= reconstruction.Image(image_id).NumPoints2D()) {
      mutation.tombstone_observations.push_back({image_id, point2D_idx});
      continue;
    }
    const Point2D& point2D = reconstruction.Image(image_id).Point2D(point2D_idx);
    if (!point2D.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2D.Point3DId())) {
      mutation.tombstone_observations.push_back({image_id, point2D_idx});
      continue;
    }
    mutation.observation_upserts.push_back(
        {image_id, point2D_idx, point2D.Point3DId(),
         {{point2D.X(), point2D.Y()}}});
  }
  if (reconstruction.StructureOwnerEpoch() != journal.owner_epoch ||
      reconstruction.StructureRevision() != journal.current_revision) {
    *error = "Reconstruction changed during native graph delta conversion";
    return false;
  }
  *output = std::move(mutation);
  return true;
}

const ActiveBaParameterBlockSpec* FindParameter(
    const NativeActiveSolveInputs& inputs,
    const ParameterKind kind,
    const uint64_t entity_id) {
  for (size_t i = 0; i < inputs.parameter_blocks.size; ++i) {
    const ActiveBaParameterBlockSpec& parameter = inputs.parameter_blocks[i];
    if (parameter.kind == kind && parameter.entity_id == entity_id)
      return &parameter;
  }
  return nullptr;
}

template <typename SnapshotType, typename IdType, typename GetId>
const SnapshotType* FindById(const std::vector<SnapshotType>& values,
                             const IdType id,
                             const GetId& get_id) {
  const auto found = std::find_if(values.begin(), values.end(),
                                  [&](const SnapshotType& value) {
                                    return get_id(value) == id;
                                  });
  return found == values.end() ? nullptr : &*found;
}

constexpr size_t kPreparedSelectionHostMaxEntries = 8;
constexpr uint64_t kPreparedSelectionHostBudgetBytes = 256ull << 20;

template <typename T>
void AppendPlanKeyValue(const T& value, std::vector<uint8_t>* bytes) {
  static_assert(std::is_trivially_copyable<T>::value,
                "plan key values must be trivially copyable");
  const uint8_t* first = reinterpret_cast<const uint8_t*>(&value);
  bytes->insert(bytes->end(), first, first + sizeof(value));
}

template <typename T>
void AppendPlanKeyVector(const std::vector<T>& values,
                         std::vector<uint8_t>* bytes) {
  const uint64_t size = values.size();
  AppendPlanKeyValue(size, bytes);
  for (const T& value : values) AppendPlanKeyValue(value, bytes);
}

void AppendPlanStaticConfig(const NativeCudaResolvedConfig& config,
                            std::vector<uint8_t>* bytes) {
  AppendPlanKeyValue(config.hessian_backend, bytes);
  AppendPlanKeyValue(config.schur_backend, bytes);
  AppendPlanKeyValue(config.hot_kernel, bytes);
  AppendPlanKeyValue(config.execution_profile, bytes);
  AppendPlanKeyValue(config.residual_order, bytes);
  AppendPlanKeyValue(config.lidar_residual_mode, bytes);
  AppendPlanKeyValue(config.lidar_near_zero_threshold, bytes);
  AppendPlanKeyValue(config.pair_chunk_limit_bytes, bytes);
  AppendPlanKeyValue(config.hessian_segment_size, bytes);
  AppendPlanKeyValue(config.schur_segment_size, bytes);
}

void AppendPlanOnlineLidarIdentity(
    const NativeBaOnlineLidarIdentity& identity,
    std::vector<uint8_t>* bytes) {
  AppendPlanKeyValue(identity.valid, bytes);
  AppendPlanKeyValue(identity.trigger_image_id, bytes);
  AppendPlanKeyValue(identity.map_version, bytes);
  AppendPlanKeyValue(identity.max_scan_index, bytes);
  bytes->insert(bytes->end(), identity.snapshot_sha256.begin(),
                identity.snapshot_sha256.end());
  bytes->insert(bytes->end(), identity.geometry_sha256.begin(),
                identity.geometry_sha256.end());
  bytes->insert(bytes->end(), identity.association_sha256.begin(),
                identity.association_sha256.end());
}

bool ValidateLidarProvenanceIntentBeforeCache(
    const NativeBaSolveIntent& intent,
    const CatalogReadLease& lease,
    std::string* error) {
  if (!IsValidNativeBaLidarScopeIdentity(intent.visual_observation_scope,
                                         intent.online_lidar_identity)) {
    *error = "native LiDAR scope and identity are inconsistent";
    return false;
  }
  if (!intent.online_lidar_identity.valid) {
    for (const NativeBaLidarConstraint& constraint :
         intent.lidar_constraints) {
      if (!IsCanonicalNativeBaLegacyLidarConstraint(constraint)) {
        *error = "native legacy LiDAR provenance is not canonical";
        return false;
      }
    }
    return true;
  }
  if (intent.lidar_map_generation !=
          intent.online_lidar_identity.map_version ||
      intent.lidar_match_config_generation !=
          intent.config.config_generation) {
    *error = "native online LiDAR generations are inconsistent";
    return false;
  }
  if (intent.lidar_constraints.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    *error = "native online LiDAR constraint count is not representable";
    return false;
  }
  try {
    std::unordered_set<uint32_t> active_image_ids;
    active_image_ids.reserve(intent.active_image_ids.size());
    for (const uint32_t image_id : intent.active_image_ids) {
      const HostBaImageSlot* image = lease.FindImageById(image_id);
      if (image_id == kBaGraphInvalidSlot || image == nullptr ||
          !image->header.alive || !image->registered ||
          !active_image_ids.insert(image_id).second) {
        *error = "native online LiDAR active image identity is invalid";
        return false;
      }
    }
    if (active_image_ids.count(
            intent.online_lidar_identity.trigger_image_id) == 0) {
      *error = "native online LiDAR trigger is outside active images";
      return false;
    }

    const size_t constraint_count = intent.lidar_constraints.size();
    std::vector<uint8_t> constraint_slots_seen(constraint_count, 0);
    std::vector<uint8_t> association_ids_seen(constraint_count, 0);
    std::unordered_set<uint64_t> point_ids;
    std::unordered_set<uint64_t> physical_identities;
    point_ids.reserve(constraint_count);
    physical_identities.reserve(constraint_count);
    for (const NativeBaLidarConstraint& constraint :
         intent.lidar_constraints) {
      if (constraint.constraint_slot >= constraint_count ||
          constraint_slots_seen[constraint.constraint_slot] != 0) {
        *error = "native online LiDAR constraint slots are not dense";
        return false;
      }
      constraint_slots_seen[constraint.constraint_slot] = 1;
      if (constraint.association_id >= constraint_count ||
          association_ids_seen[constraint.association_id] != 0) {
        *error = "native online LiDAR association IDs are not dense";
        return false;
      }
      association_ids_seen[constraint.association_id] = 1;
      if (constraint.physical_identity == 0 ||
          constraint.physical_identity != constraint.association_id + 1 ||
          !physical_identities.insert(constraint.physical_identity).second) {
        *error = "native online LiDAR physical identity is invalid";
        return false;
      }
      if (!point_ids.insert(constraint.point3D_id).second) {
        *error = "native online LiDAR point identity is duplicated";
        return false;
      }
      if (constraint.owner_image_id == kBaGraphInvalidSlot ||
          constraint.owner_point2D_idx == kBaGraphInvalidSlot ||
          active_image_ids.count(constraint.owner_image_id) == 0) {
        *error = "native online LiDAR owner is outside active images";
        return false;
      }
      const HostBaPointSlot* point =
          lease.FindPointById(constraint.point3D_id);
      const HostBaImageSlot* owner =
          lease.FindImageById(constraint.owner_image_id);
      const HostBaObservationSlot* observation = lease.FindObservation(
          constraint.owner_image_id, constraint.owner_point2D_idx);
      if (point == nullptr || !point->header.alive || owner == nullptr ||
          !owner->header.alive || !owner->registered || observation == nullptr ||
          !observation->header.alive ||
          observation->image_slot != owner->header.slot ||
          observation->point_slot != point->header.slot) {
        *error = "native online LiDAR owner observation is invalid";
        return false;
      }
      if (!Finite(constraint.frozen_point3D_xyz) ||
          !Finite(constraint.plane) || !Finite(constraint.lidar_xyz) ||
          !std::isfinite(constraint.weight) ||
          !std::isfinite(constraint.search_range)) {
        *error = "native online LiDAR constraint contains non-finite data";
        return false;
      }
    }
  } catch (const std::bad_alloc&) {
    *error = "native online LiDAR validation allocation failed";
    return false;
  } catch (const std::length_error&) {
    *error = "native online LiDAR validation size overflow";
    return false;
  }
  return true;
}

bool BuildPlanLookupKey(const NativeBaSolveIntent& intent,
                        const uint64_t slot_namespace_epoch,
                        std::vector<uint8_t>* bytes,
                        uint64_t* hash,
                        std::string* error) {
  if (bytes == nullptr || hash == nullptr || error == nullptr ||
      intent.owner_epoch == 0 || slot_namespace_epoch == 0) {
    return false;
  }
  try {
    bytes->clear();
    bytes->reserve(256 + intent.active_image_ids.size() * sizeof(uint32_t) +
                   (intent.explicit_variable_point_ids.size() +
                    intent.explicit_constant_point_ids.size()) *
                       sizeof(uint64_t));
    AppendPlanKeyValue(intent.owner_epoch, bytes);
    AppendPlanKeyValue(slot_namespace_epoch, bytes);
    AppendPlanKeyValue(intent.kind, bytes);
    AppendPlanKeyValue(intent.visual_observation_scope, bytes);
    AppendPlanStaticConfig(intent.config, bytes);
    AppendPlanKeyVector(intent.active_image_ids, bytes);
    AppendPlanKeyVector(intent.explicit_variable_point_ids, bytes);
    AppendPlanKeyVector(intent.explicit_constant_point_ids, bytes);
    AppendPlanKeyVector(intent.fixed_pose_ids, bytes);
    const uint64_t translation_count = intent.translation_policies.size();
    AppendPlanKeyValue(translation_count, bytes);
    for (const NativeBaTranslationPolicy& value :
         intent.translation_policies) {
      AppendPlanKeyValue(value.image_id, bytes);
      AppendPlanKeyValue(value.constant_mask, bytes);
    }
    const uint64_t camera_count = intent.camera_policies.size();
    AppendPlanKeyValue(camera_count, bytes);
    for (const NativeBaCameraPolicy& value : intent.camera_policies) {
      AppendPlanKeyValue(value.camera_id, bytes);
      AppendPlanKeyValue(value.constant, bytes);
      AppendPlanKeyVector(value.fixed_parameter_indices, bytes);
    }
    const uint64_t point_count = intent.point_policies.size();
    AppendPlanKeyValue(point_count, bytes);
    for (const NativeBaPointPolicy& value : intent.point_policies) {
      AppendPlanKeyValue(value.point3D_id, bytes);
      AppendPlanKeyValue(value.constant, bytes);
      AppendPlanKeyValue(value.config_role, bytes);
      AppendPlanKeyValue(value.has_search_range, bytes);
      AppendPlanKeyValue(value.search_range, bytes);
    }
    AppendPlanOnlineLidarIdentity(intent.online_lidar_identity, bytes);
    const uint64_t lidar_count = intent.lidar_constraints.size();
    AppendPlanKeyValue(lidar_count, bytes);
    for (const NativeBaLidarConstraint& value : intent.lidar_constraints) {
      AppendPlanKeyValue(value.point3D_id, bytes);
      AppendPlanKeyValue(value.constraint_slot, bytes);
      AppendPlanKeyValue(value.physical_identity, bytes);
      AppendPlanKeyValue(value.association_id, bytes);
      AppendPlanKeyValue(value.owner_image_id, bytes);
      AppendPlanKeyValue(value.owner_point2D_idx, bytes);
      AppendPlanKeyValue(value.lidar_type, bytes);
      AppendPlanKeyValue(value.frozen_point3D_xyz, bytes);
      AppendPlanKeyValue(value.plane, bytes);
      AppendPlanKeyValue(value.lidar_xyz, bytes);
      AppendPlanKeyValue(value.weight, bytes);
      AppendPlanKeyValue(value.search_range, bytes);
    }
  } catch (const std::bad_alloc&) {
    *error = "native prepared selection key allocation failed";
    return false;
  } catch (const std::length_error&) {
    *error = "native prepared selection key size overflow";
    return false;
  }
  uint64_t value = 1469598103934665603ull;
  for (const uint8_t byte : *bytes) {
    value ^= byte;
    value *= 1099511628211ull;
  }
  *hash = value;
  return true;
}

struct PreparedSelectionDependencyStamp {
  std::vector<HostBaCameraSlot> cameras;
  std::vector<HostBaImageSlot> images;
  std::vector<HostBaPointSlot> points;
  std::vector<HostBaObservationSlot> observations;
};

bool SameCameraDependency(const HostBaCameraSlot& a,
                          const HostBaCameraSlot& b) noexcept {
  return a.header.slot == b.header.slot &&
         a.header.generation == b.header.generation &&
         a.header.alive == b.header.alive && a.camera_id == b.camera_id &&
         a.model_id == b.model_id && a.width == b.width &&
         a.height == b.height && a.parameter_count == b.parameter_count;
}

bool SameImageDependency(const HostBaImageSlot& a,
                         const HostBaImageSlot& b) noexcept {
  return a.header.slot == b.header.slot &&
         a.header.generation == b.header.generation &&
         a.header.alive == b.header.alive && a.image_id == b.image_id &&
         a.camera_slot == b.camera_slot && a.registered == b.registered &&
         a.adjacency_count == b.adjacency_count &&
         a.lifetime_incidence_count == b.lifetime_incidence_count;
}

bool SamePointDependency(const HostBaPointSlot& a,
                         const HostBaPointSlot& b) noexcept {
  return a.header.slot == b.header.slot &&
         a.header.generation == b.header.generation &&
         a.header.alive == b.header.alive && a.point3D_id == b.point3D_id &&
         a.adjacency_count == b.adjacency_count &&
         a.lifetime_incidence_count == b.lifetime_incidence_count &&
         a.track_length == b.track_length;
}

bool SameObservationDependency(const HostBaObservationSlot& a,
                               const HostBaObservationSlot& b) noexcept {
  return a.header.slot == b.header.slot &&
         a.header.generation == b.header.generation &&
         a.header.alive == b.header.alive && a.image_slot == b.image_slot &&
         a.point_slot == b.point_slot && a.point2D_idx == b.point2D_idx &&
         a.source_identity == b.source_identity &&
         a.association_generation == b.association_generation &&
         std::memcmp(a.xy.data(), b.xy.data(), sizeof(double) * 2) == 0;
}

bool BuildDependencyStamp(const CatalogReadLease& lease,
                          const PreparedSelectionPlan& plan,
                          PreparedSelectionDependencyStamp* stamp,
                          std::string* error) {
  if (stamp == nullptr || error == nullptr || !lease.valid()) return false;
  const auto cameras = lease.cameras();
  const auto images = lease.images();
  const auto points = lease.points();
  const auto observations = lease.observations();
  try {
    stamp->cameras.reserve(plan.active_camera_slots.size());
    stamp->images.reserve(plan.active_image_slots.size() +
                          plan.boundary_image_slots.size());
    stamp->points.reserve(plan.active_point_slots.size());
    stamp->observations.reserve(plan.visual_observation_slots.size());
    for (const uint32_t slot : plan.active_camera_slots) {
      if (slot >= cameras.size) {
        *error = "prepared plan camera dependency is invalid";
        return false;
      }
      stamp->cameras.push_back(cameras[slot]);
    }
    for (const uint32_t slot : plan.active_image_slots) {
      if (slot >= images.size) {
        *error = "prepared plan image dependency is invalid";
        return false;
      }
      stamp->images.push_back(images[slot]);
    }
    for (const uint32_t slot : plan.boundary_image_slots) {
      if (slot >= images.size) {
        *error = "prepared plan boundary dependency is invalid";
        return false;
      }
      stamp->images.push_back(images[slot]);
    }
    for (const uint32_t slot : plan.active_point_slots) {
      if (slot >= points.size) {
        *error = "prepared plan point dependency is invalid";
        return false;
      }
      stamp->points.push_back(points[slot]);
    }
    for (const uint32_t slot : plan.visual_observation_slots) {
      if (slot >= observations.size) {
        *error = "prepared plan observation dependency is invalid";
        return false;
      }
      stamp->observations.push_back(observations[slot]);
    }
  } catch (const std::bad_alloc&) {
    *error = "native prepared dependency allocation failed";
    return false;
  }
  return true;
}

bool ValidateDependencyStamp(const CatalogReadLease& lease,
                             const PreparedSelectionDependencyStamp& stamp) {
  const auto cameras = lease.cameras();
  const auto images = lease.images();
  const auto points = lease.points();
  const auto observations = lease.observations();
  for (const HostBaCameraSlot& value : stamp.cameras) {
    if (value.header.slot >= cameras.size ||
        !SameCameraDependency(value, cameras[value.header.slot])) return false;
  }
  for (const HostBaImageSlot& value : stamp.images) {
    if (value.header.slot >= images.size ||
        !SameImageDependency(value, images[value.header.slot])) return false;
  }
  for (const HostBaPointSlot& value : stamp.points) {
    if (value.header.slot >= points.size ||
        !SamePointDependency(value, points[value.header.slot])) return false;
  }
  for (const HostBaObservationSlot& value : stamp.observations) {
    if (value.header.slot >= observations.size ||
        !SameObservationDependency(value, observations[value.header.slot]))
      return false;
  }
  return true;
}

uint64_t EstimatePreparedSelectionPlanBytes(
    const PreparedSelectionPlan& plan) noexcept {
  uint64_t bytes = sizeof(plan);
  bytes += plan.active_camera_slots.capacity() * sizeof(uint32_t);
  bytes += plan.active_image_slots.capacity() * sizeof(uint32_t);
  bytes += plan.boundary_image_slots.capacity() * sizeof(uint32_t);
  bytes += plan.active_point_slots.capacity() * sizeof(uint32_t);
  bytes += plan.visual_observation_slots.capacity() * sizeof(uint32_t);
  bytes += plan.fixed.images.capacity() * sizeof(ImageFixedPolicyResult);
  bytes += plan.fixed.cameras.capacity() * sizeof(CameraFixedPolicyResult);
  bytes += plan.fixed.points.capacity() * sizeof(PointFixedPolicyResult);
  for (const CameraFixedPolicyResult& value : plan.fixed.cameras)
    bytes += value.fixed_parameter_indices.capacity() * sizeof(uint32_t);
  bytes += plan.lidar_constraints.capacity() * sizeof(LidarConstraintRecord);
  bytes += plan.residual_ordinals.capacity() * sizeof(ResidualOrdinal);
  bytes += plan.parameter_ordinals.capacity() * sizeof(ParameterOrdinal);
  return bytes;
}

std::shared_ptr<PreparedSelectionPlan> ExtractPreparedSelectionPlan(
    NativeHostSolveView* view,
    const uint64_t publication_id) {
  std::shared_ptr<PreparedSelectionPlan> plan =
      std::make_shared<PreparedSelectionPlan>();
  plan->publication_id = publication_id;
  plan->slot_namespace_epoch = view->catalog.slot_namespace_epoch();
  plan->visual_observation_scope = view->identity.visual_observation_scope;
  plan->online_lidar_identity = view->identity.online_lidar_identity;
  plan->active_camera_slots = std::move(view->active_camera_slots);
  plan->active_image_slots = std::move(view->active_image_slots);
  plan->boundary_image_slots = std::move(view->boundary_image_slots);
  plan->active_point_slots = std::move(view->active_point_slots);
  plan->visual_observation_slots = std::move(view->visual_observation_slots);
  plan->fixed = std::move(view->fixed);
  plan->lidar_constraints = std::move(view->lidar.constraints);
  for (LidarConstraintRecord& value : plan->lidar_constraints)
    value.point_state_generation = 0;
  plan->residual_ordinals = std::move(view->residual_ordinals);
  plan->parameter_ordinals = std::move(view->parameter_ordinals);
  plan->residual_block_count = view->residual_block_count;
  plan->scalar_residual_count = view->scalar_residual_count;
  plan->ambient_parameter_count = view->ambient_parameter_count;
  plan->effective_parameter_count = view->effective_parameter_count;
  plan->host_resident_bytes = EstimatePreparedSelectionPlanBytes(*plan);
  view->prepared_plan = plan;
  return plan;
}

struct PreparedSelectionCacheEntry {
  uint64_t hash = 0;
  uint64_t last_use = 0;
  uint64_t resident_bytes = 0;
  std::vector<uint8_t> canonical_key;
  PreparedSelectionDependencyStamp dependencies;
  std::shared_ptr<const PreparedSelectionPlan> plan;
};

static_assert(
    std::is_nothrow_move_constructible<PreparedSelectionCacheEntry>::value,
    "prepared selection cache publication must be noexcept movable");
static_assert(
    std::is_nothrow_move_assignable<PreparedSelectionCacheEntry>::value,
    "prepared selection cache replacement must be noexcept movable");

}  // namespace

struct NativeGraphStoreState {
  NativeGraphStoreState(const Reconstruction* reconstruction_in,
                        const uint64_t owner_epoch_in)
      : reconstruction(reconstruction_in),
        owner_epoch(owner_epoch_in),
        graph(owner_epoch_in),
        device_store(CreateDeviceBaProblemStore(owner_epoch_in)) {
    prepared_plans.reserve(kPreparedSelectionHostMaxEntries);
  }

  const Reconstruction* reconstruction = nullptr;
  uint64_t owner_epoch = 0;
  uint64_t journal_cursor = 0;
  uint64_t next_config_generation = 0;
  uint64_t next_selection_generation = 0;
  uint64_t next_state_generation = 0;
  uint64_t active_prepares = 0;
  bool closing = false;
  std::mutex mutex;
  HostBaGraphStore graph;
  NativeHostSolveMaterializer materializer;
  uint64_t next_plan_publication_id = 0;
  uint64_t plan_lru_clock = 0;
  uint64_t host_plan_resident_bytes = 0;
  uint64_t host_plan_peak_bytes = 0;
  std::vector<PreparedSelectionCacheEntry> prepared_plans;
  std::shared_ptr<DeviceBaProblemStoreHandle> device_store;
};

struct PreparedNativeActiveSolve::Data {
  std::shared_ptr<NativeGraphStoreState> state;
  NativeHostSolveView view;
  ActiveStateBuffer initial_state;
  CudaFullLmOptions options;
  NativeGraphPrepareRuntime runtime;
  bool active = false;
};

namespace {

bool SynchronizeNativeGraphStateLocked(
    const std::shared_ptr<NativeGraphStoreState>& state,
    NativeGraphPrepareRuntime* runtime,
    std::string* error) {
  const auto sync_start = std::chrono::steady_clock::now();
  const auto cold_build = [&](const bool rebuild) -> bool {
    HostBaGraphColdInput cold;
    if (!BuildColdInput(*state->reconstruction, state->owner_epoch, &cold,
                        error)) {
      return false;
    }
    HostBaGraphUpdateResult update;
    if (!state->graph.ColdBuild(cold, &update, error)) {
      if (update.reader_busy) ++runtime->reader_busy;
      return false;
    }
    state->journal_cursor = cold.topology_revision;
    ++runtime->catalog_cold_builds;
    if (rebuild) ++runtime->catalog_full_rebuilds;
    return true;
  };
  if (state->graph.empty()) {
    if (!cold_build(false)) return false;
  } else {
    HostStructureReadResult journal;
    if (!ReadHostStructureEventsSince(state->reconstruction,
                                      state->owner_epoch,
                                      state->journal_cursor,
                                      &journal, error)) {
      return false;
    }
    bool rebuild = journal.gap || !journal.complete;
    CoalescedBaGraphMutation mutation;
    if (!rebuild && journal.current_revision != state->journal_cursor &&
        !BuildCoalescedMutation(*state->reconstruction, journal, &mutation,
                                &rebuild, runtime, error)) {
      return false;
    }
    if (rebuild) {
      if (!cold_build(true)) return false;
    } else if (journal.current_revision != state->journal_cursor) {
      HostBaGraphUpdateResult update;
      if (!state->graph.ApplyCoalescedMutation(mutation, &update, error)) {
        if (update.reader_busy) {
          ++runtime->reader_busy;
          return false;
        }
        if (!cold_build(true)) return false;
      } else {
        state->journal_cursor = journal.current_revision;
        ++runtime->catalog_delta_updates;
      }
    }
  }
  runtime->journal_cursor_after = state->journal_cursor;
  runtime->catalog_sync_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - sync_start)
          .count();
  return true;
}

uint64_t EstimateDependencyBytes(
    const PreparedSelectionDependencyStamp& stamp) noexcept {
  return sizeof(stamp) +
         stamp.cameras.capacity() * sizeof(HostBaCameraSlot) +
         stamp.images.capacity() * sizeof(HostBaImageSlot) +
         stamp.points.capacity() * sizeof(HostBaPointSlot) +
         stamp.observations.capacity() * sizeof(HostBaObservationSlot);
}

std::shared_ptr<const PreparedSelectionPlan> LookupPreparedSelectionPlan(
    NativeGraphStoreState* state,
    const CatalogReadLease& lease,
    const std::vector<uint8_t>& canonical_key,
    const uint64_t hash,
    NativeGraphPrepareRuntime* runtime) {
  for (PreparedSelectionCacheEntry& entry : state->prepared_plans) {
    if (entry.hash != hash) continue;
    if (entry.canonical_key != canonical_key) {
      ++runtime->host_plan_hash_collisions;
      continue;
    }
    if (!ValidateDependencyStamp(lease, entry.dependencies)) {
      ++runtime->host_plan_dependency_misses;
      continue;
    }
    entry.last_use = ++state->plan_lru_clock;
    ++runtime->host_plan_hits;
    return entry.plan;
  }
  ++runtime->host_plan_misses;
  return nullptr;
}

bool PublishPreparedSelectionPlan(
    NativeGraphStoreState* state,
    const CatalogReadLease& lease,
    std::vector<uint8_t> canonical_key,
    const uint64_t hash,
    const std::shared_ptr<const PreparedSelectionPlan>& plan,
    NativeGraphPrepareRuntime* runtime,
    std::string* error) {
  PreparedSelectionDependencyStamp dependencies;
  if (!BuildDependencyStamp(lease, *plan, &dependencies, error)) return false;
  const uint64_t resident_bytes =
      plan->host_resident_bytes + canonical_key.capacity() +
      EstimateDependencyBytes(dependencies);
  if (resident_bytes > kPreparedSelectionHostBudgetBytes) {
    --runtime->host_plan_misses;
    ++runtime->host_plan_bypasses;
    runtime->host_plan_resident_bytes = state->host_plan_resident_bytes;
    runtime->host_plan_peak_bytes = state->host_plan_peak_bytes;
    return false;
  }
  PreparedSelectionCacheEntry entry;
  entry.hash = hash;
  entry.last_use = ++state->plan_lru_clock;
  entry.resident_bytes = resident_bytes;
  entry.canonical_key = std::move(canonical_key);
  entry.dependencies = std::move(dependencies);
  entry.plan = plan;
  std::array<size_t, kPreparedSelectionHostMaxEntries> victims{};
  size_t victim_count = 0;
  const auto is_victim = [&](const size_t candidate) {
    return std::find(victims.begin(), victims.begin() + victim_count,
                     candidate) != victims.begin() + victim_count;
  };
  uint64_t retained_bytes = state->host_plan_resident_bytes;
  size_t retained_entries = state->prepared_plans.size();
  while (retained_entries >= kPreparedSelectionHostMaxEntries ||
         retained_bytes + resident_bytes > kPreparedSelectionHostBudgetBytes) {
    size_t victim = state->prepared_plans.size();
    for (size_t i = 0; i < state->prepared_plans.size(); ++i) {
      if (state->prepared_plans[i].plan.use_count() != 1 ||
          is_victim(i)) {
        continue;
      }
      if (victim == state->prepared_plans.size() ||
          state->prepared_plans[i].last_use <
              state->prepared_plans[victim].last_use) {
        victim = i;
      }
    }
    if (victim == state->prepared_plans.size()) {
      --runtime->host_plan_misses;
      ++runtime->host_plan_bypasses;
      runtime->host_plan_resident_bytes = state->host_plan_resident_bytes;
      runtime->host_plan_peak_bytes = state->host_plan_peak_bytes;
      return false;
    }
    assert(victim_count < victims.size());
    victims[victim_count++] = victim;
    retained_bytes -= state->prepared_plans[victim].resident_bytes;
    --retained_entries;
  }
  std::sort(victims.begin(), victims.begin() + victim_count,
            std::greater<size_t>());
  for (size_t i = 0; i < victim_count; ++i) {
    const size_t victim = victims[i];
    state->host_plan_resident_bytes -=
        state->prepared_plans[victim].resident_bytes;
    state->prepared_plans.erase(state->prepared_plans.begin() + victim);
    ++runtime->host_plan_evictions;
  }
  assert(state->prepared_plans.size() < state->prepared_plans.capacity());
  state->prepared_plans.push_back(std::move(entry));
  state->host_plan_resident_bytes += resident_bytes;
  state->host_plan_peak_bytes =
      std::max(state->host_plan_peak_bytes, state->host_plan_resident_bytes);
  runtime->host_plan_resident_bytes = state->host_plan_resident_bytes;
  runtime->host_plan_peak_bytes = state->host_plan_peak_bytes;
  return true;
}

bool GatherDynamicStateFromPreparedPlan(
    const Reconstruction& reconstruction,
    const CatalogReadLease& lease,
    const PreparedSelectionPlan& plan,
    const uint64_t state_generation,
    ActiveStateBuffer* output,
    NativeGraphPrepareRuntime* runtime,
    std::string* error) {
  if (output == nullptr || runtime == nullptr || error == nullptr ||
      !lease.valid() || state_generation == 0) return false;
  const auto start = std::chrono::steady_clock::now();
  ActiveStateBuffer state;
  state.owner_epoch = lease.owner_epoch();
  state.catalog_generation = lease.generation();
  state.state_generation = state_generation;
  const auto cameras = lease.cameras();
  const auto images = lease.images();
  const auto points = lease.points();
  std::unordered_set<uint32_t> fixed_images;
  fixed_images.reserve(plan.fixed.images.size());
  for (const ImageFixedPolicyResult& policy : plan.fixed.images) {
    if (policy.pose_constant || policy.boundary_pose)
      fixed_images.insert(policy.image_slot);
  }
  try {
    state.cameras.reserve(plan.active_camera_slots.size());
    state.images.reserve(plan.active_image_slots.size() +
                         plan.boundary_image_slots.size());
    state.points.reserve(plan.active_point_slots.size());
    for (const uint32_t slot : plan.active_camera_slots) {
      if (slot >= cameras.size || !cameras[slot].header.alive ||
          !reconstruction.ExistsCamera(cameras[slot].camera_id)) {
        *error = "prepared plan camera state is unavailable";
        return false;
      }
      DenseCameraState value;
      value.camera_slot = slot;
      value.state_generation = state_generation;
      value.parameters = reconstruction.Camera(cameras[slot].camera_id).Params();
      if (value.parameters.size() != cameras[slot].parameter_count ||
          !std::all_of(value.parameters.begin(), value.parameters.end(),
                       [](double x) { return std::isfinite(x); })) {
        *error = "prepared plan camera state is invalid";
        return false;
      }
      state.cameras.push_back(std::move(value));
    }
    const auto gather_image = [&](const uint32_t slot) -> bool {
      if (slot >= images.size || !images[slot].header.alive ||
          !reconstruction.ExistsImage(images[slot].image_id)) {
        *error = "prepared plan image state is unavailable";
        return false;
      }
      const Image& image = reconstruction.Image(images[slot].image_id);
      DenseImageState value;
      value.image_slot = slot;
      value.state_generation = state_generation;
      std::copy(image.Qvec().data(), image.Qvec().data() + 4,
                value.quaternion.begin());
      std::copy(image.Tvec().data(), image.Tvec().data() + 3,
                value.translation.begin());
      double norm2 = 0.0;
      for (double q : value.quaternion) norm2 += q * q;
      if (!Finite(value.quaternion) || !Finite(value.translation) ||
          (fixed_images.count(slot) != 0 &&
           (!std::isfinite(norm2) ||
            std::abs(std::sqrt(norm2) - 1.0) > 1e-12)) ||
          !NormalizeQuaternion(value.quaternion, &value.quaternion)) {
        *error = "prepared plan image state normalization failed";
        return false;
      }
      state.images.push_back(value);
      return true;
    };
    for (const uint32_t slot : plan.active_image_slots)
      if (!gather_image(slot)) return false;
    for (const uint32_t slot : plan.boundary_image_slots)
      if (!gather_image(slot)) return false;
    for (const uint32_t slot : plan.active_point_slots) {
      if (slot >= points.size || !points[slot].header.alive ||
          !reconstruction.ExistsPoint3D(points[slot].point3D_id)) {
        *error = "prepared plan point state is unavailable";
        return false;
      }
      DensePointState value;
      value.point_slot = slot;
      value.state_generation = state_generation;
      const Eigen::Vector3d& xyz =
          reconstruction.Point3D(points[slot].point3D_id).XYZ();
      std::copy(xyz.data(), xyz.data() + 3, value.xyz.begin());
      if (!Finite(value.xyz)) {
        *error = "prepared plan point state is invalid";
        return false;
      }
      state.points.push_back(value);
    }
  } catch (const std::bad_alloc&) {
    *error = "prepared plan dynamic state allocation failed";
    return false;
  }
  *output = std::move(state);
  ++runtime->dynamic_state_gather_calls;
  runtime->dynamic_state_gather_milliseconds +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

}  // namespace

PreparedNativeActiveSolve::PreparedNativeActiveSolve() = default;
PreparedNativeActiveSolve::~PreparedNativeActiveSolve() { Release(); }
PreparedNativeActiveSolve::PreparedNativeActiveSolve(
    PreparedNativeActiveSolve&&) noexcept = default;
PreparedNativeActiveSolve& PreparedNativeActiveSolve::operator=(
    PreparedNativeActiveSolve&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::move(other.data_);
  }
  return *this;
}

bool PreparedNativeActiveSolve::valid() const noexcept {
  return data_ != nullptr && data_->active && data_->state != nullptr &&
         data_->view.catalog.valid();
}

NativeCudaSolveRequest PreparedNativeActiveSolve::request() const noexcept {
  NativeCudaSolveRequest request;
  if (valid()) {
    request.view = &data_->view;
    request.initial_state = &data_->initial_state;
    request.options = &data_->options;
    request.device_store = data_->state->device_store;
  }
  return request;
}

const NativeHostSolveView* PreparedNativeActiveSolve::view() const noexcept {
  return valid() ? &data_->view : nullptr;
}

const ActiveStateBuffer* PreparedNativeActiveSolve::initial_state()
    const noexcept {
  return valid() ? &data_->initial_state : nullptr;
}

const NativeGraphPrepareRuntime& PreparedNativeActiveSolve::runtime()
    const noexcept {
  static const NativeGraphPrepareRuntime empty;
  return data_ == nullptr ? empty : data_->runtime;
}

void PreparedNativeActiveSolve::Release() noexcept {
  std::shared_ptr<Data> data = std::move(data_);
  if (data == nullptr || !data->active || data->state == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(data->state->mutex);
    // Release the graph read guard while active_prepares still excludes a
    // second prepare. This closes the reader_busy window at handoff.
    data->view.catalog = CatalogReadLease();
    if (data->state->active_prepares != 0) --data->state->active_prepares;
    data->active = false;
  }
  data->view = NativeHostSolveView();
}

bool PreparedNativeActiveSolve::Complete(std::string* error) noexcept {
  if (error == nullptr) return false;
  error->clear();
  Release();
  return true;
}

bool MakeNativeActiveSolveInputs(const ActiveBaSolveSpec& spec,
                                 NativeActiveSolveInputs* inputs,
                                 std::string* error) {
  if (inputs == nullptr || error == nullptr || spec.owner_epoch == 0 ||
      spec.catalog_revision == 0) {
    if (error != nullptr) *error = "active spec lacks native graph identity";
    return false;
  }
  NativeActiveSolveInputs result;
  result.problem = &spec.problem;
  result.parameter_blocks = {spec.parameter_blocks.data(),
                             spec.parameter_blocks.size()};
  result.owner_epoch = spec.owner_epoch;
  result.catalog_revision = spec.catalog_revision;
  result.residual_block_count = spec.residual_block_count;
  result.scalar_residual_count = spec.scalar_residual_count;
  result.ambient_parameter_count = spec.ambient_parameter_count;
  result.effective_parameter_count = spec.effective_parameter_count;
  *inputs = result;
  error->clear();
  return true;
}

bool MakeNativeBaSolveIntentFromActiveSpec(
    const ActiveBaSolveSpec& spec,
    const uintptr_t reconstruction_identity,
    const NativeCudaResolvedConfig& resolved_config,
    NativeBaSolveIntent* intent,
    std::string* error) {
  if (intent == nullptr || error == nullptr || spec.owner_epoch == 0 ||
      spec.catalog_revision == 0 || reconstruction_identity == 0 ||
      !resolved_config.resolved) {
    if (error != nullptr) *error = "invalid ActiveSpec native intent adapter";
    return false;
  }
  NativeBaSolveIntent result;
  result.owner_epoch = spec.owner_epoch;
  result.reconstruction_identity = reconstruction_identity;
  result.expected_topology_revision = spec.catalog_revision;
  result.selection_revision = spec.problem.metadata.ba_call_index == 0
                                  ? 1
                                  : spec.problem.metadata.ba_call_index;
  result.kind = spec.problem.metadata.ba_kind;
  result.config = resolved_config;

  std::unordered_map<ParameterKey, const ActiveBaParameterBlockSpec*,
                     ParameterKeyHash>
      parameters;
  parameters.reserve(spec.parameter_blocks.size());
  for (const ActiveBaParameterBlockSpec& parameter : spec.parameter_blocks) {
    if (!parameters.emplace(ParameterKey{parameter.kind, parameter.entity_id},
                            &parameter)
             .second) {
      *error = "ActiveSpec native intent contains a duplicate parameter";
      return false;
    }
  }
  for (const ImageSnapshot& image : spec.problem.images) {
    if (!image.selected) continue;
    result.active_image_ids.push_back(image.image_id);
    if (image.pose_constant) result.fixed_pose_ids.push_back(image.image_id);
    NativeBaTranslationPolicy translation;
    translation.image_id = image.image_id;
    translation.constant_mask = image.constant_tvec_mask;
    result.translation_policies.push_back(translation);
  }
  for (const CameraSnapshot& camera : spec.problem.cameras) {
    NativeBaCameraPolicy policy;
    policy.camera_id = camera.camera_id;
    const auto found = parameters.find(
        ParameterKey{ParameterKind::kCamera, camera.camera_id});
    policy.constant = found == parameters.end() || found->second->constant;
    if (found != parameters.end()) {
      policy.fixed_parameter_indices =
          found->second->fixed_camera_parameter_indices;
    }
    result.camera_policies.push_back(std::move(policy));
  }
  for (const PointSnapshot& point : spec.problem.points) {
    NativeBaPointPolicy policy;
    policy.point3D_id = point.point3D_id;
    policy.constant = point.constant;
    policy.config_role = point.config_role;
    policy.has_search_range = point.has_search_range;
    policy.search_range = point.search_range;
    result.point_policies.push_back(policy);
    if (point.config_role == 1) {
      result.explicit_variable_point_ids.push_back(point.point3D_id);
    } else if (point.config_role == 2) {
      result.explicit_constant_point_ids.push_back(point.point3D_id);
    }
  }
  result.lidar_map_generation = spec.catalog_revision;
  result.lidar_match_config_generation = resolved_config.config_generation;
  result.lidar_constraints.reserve(spec.problem.lidar.size());
  for (size_t index = 0; index < spec.problem.lidar.size(); ++index) {
    const LidarSnapshot& lidar = spec.problem.lidar[index];
    NativeBaLidarConstraint constraint;
    constraint.point3D_id = lidar.point3D_id;
    constraint.constraint_slot = static_cast<uint32_t>(index);
    constraint.physical_identity =
        (static_cast<uint64_t>(lidar.point3D_id) << 2) ^
        static_cast<uint64_t>(lidar.lidar_type);
    constraint.lidar_type = lidar.lidar_type;
    constraint.plane = lidar.plane;
    constraint.lidar_xyz = lidar.lidar_xyz;
    constraint.weight = lidar.weight;
    constraint.search_range = lidar.search_range;
    result.lidar_constraints.push_back(constraint);
  }
  *intent = std::move(result);
  error->clear();
  return true;
}

std::shared_ptr<NativeGraphStoreState> CreateNativeGraphStoreState(
    const Reconstruction* reconstruction, const uint64_t owner_epoch) {
  if (reconstruction == nullptr || owner_epoch == 0) return nullptr;
  try {
    std::shared_ptr<NativeGraphStoreState> state =
        std::make_shared<NativeGraphStoreState>(reconstruction, owner_epoch);
    return state->device_store == nullptr ? nullptr : state;
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

bool ShutdownNativeGraphStoreState(
    const std::shared_ptr<NativeGraphStoreState>& state,
    std::string* error) noexcept {
  if (error == nullptr) return false;
  error->clear();
  if (state == nullptr) return true;
  std::lock_guard<std::mutex> lock(state->mutex);
  state->closing = true;
  if (state->active_prepares != 0) {
    *error = "StoreBusy: native graph has an active prepared solve";
    return false;
  }
  return true;
}

bool PrepareCudaNativeActiveSolve(
    const NativeActiveSolveInputs& inputs,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedNativeActiveSolve* prepared,
    std::string* error) {
  if (prepared != nullptr) prepared->Release();
  if (prepared == nullptr || error == nullptr || inputs.problem == nullptr) {
    if (error != nullptr) *error = "invalid native graph prepare arguments";
    return false;
  }
  error->clear();
  if (binding.mode != CudaHostProblemStoreMode::kHostPreparedStore ||
      binding.store == nullptr || binding.owner_epoch == 0 ||
      binding.owner_epoch != inputs.owner_epoch ||
      binding.store->native_graph_state_ == nullptr) {
    *error = "native_graph requires a matching enabled host store";
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  std::shared_ptr<NativeGraphStoreState> state =
      binding.store->native_graph_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  if (state->closing || state->active_prepares != 0 ||
      state->owner_epoch != binding.owner_epoch ||
      state->reconstruction == nullptr) {
    *error = state->active_prepares != 0
        ? "StoreBusy: native graph already has an active solve"
        : "native graph store is closed or has mismatched identity";
    return false;
  }
  std::shared_ptr<PreparedNativeActiveSolve::Data> data;
  try {
    data = std::make_shared<PreparedNativeActiveSolve::Data>();
  } catch (const std::bad_alloc&) {
    *error = "native graph prepared solve allocation failed";
    return false;
  }
  data->state = state;
  data->runtime.journal_cursor_before = state->journal_cursor;
  if (!SynchronizeNativeGraphStateLocked(state, &data->runtime, error))
    return false;
  if (inputs.catalog_revision != state->journal_cursor) {
    *error = "native active selection and catalog revisions differ";
    return false;
  }
  CatalogReadLease lease = state->graph.AcquireReadLease();
  if (!lease.valid()) {
    *error = "native graph catalog lease is unavailable";
    return false;
  }
  if (++state->next_config_generation == 0)
    ++state->next_config_generation;
  if (++state->next_selection_generation == 0)
    ++state->next_selection_generation;
  if (++state->next_state_generation == 0)
    ++state->next_state_generation;
  NativeCudaResolvedConfig resolved_config;
  if (!ResolveNativeCudaConfiguration(
          *inputs.problem, options, state->next_config_generation,
          &data->options, &resolved_config, error)) {
    return false;
  }

  const CudaSolveProblem& problem = *inputs.problem;
  BaSolveIntent intent;
  intent.owner_epoch = inputs.owner_epoch;
  intent.catalog_revision = lease.topology_revision();
  intent.catalog_generation = lease.generation();
  intent.selection_revision = state->next_selection_generation;
  intent.kind = problem.metadata.ba_kind;
  intent.config = resolved_config;
  intent.active_visual_observation_slots_explicit = true;
  DenseActiveState dense;
  dense.owner_epoch = inputs.owner_epoch;
  dense.state_generation = state->next_state_generation;
  std::unordered_set<uint32_t> selected_images;
  std::unordered_set<uint32_t> relevant_cameras;
  std::unordered_set<uint32_t> relevant_points;
  std::unordered_set<uint32_t> explicit_variable_points;
  std::unordered_set<uint32_t> explicit_constant_points;
  for (const ImageSnapshot& image : problem.images) {
    const HostBaImageSlot* slot = lease.FindImageById(image.image_id);
    if (slot == nullptr || !slot->header.alive ||
        slot->camera_slot >= lease.cameras().size ||
        lease.cameras()[slot->camera_slot].camera_id != image.camera_id) {
      *error = "native active image is inconsistent with the graph catalog";
      return false;
    }
    DenseImageState state_value;
    state_value.image_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.quaternion = image.qvec;
    state_value.translation = image.tvec;
    dense.images.push_back(state_value);
    if (image.selected) {
      if (!selected_images.insert(image.image_id).second) {
        *error = "native active image selection is duplicated";
        return false;
      }
      intent.active_image_slots.push_back(slot->header.slot);
      if (image.pose_constant) intent.fixed_pose_slots.push_back(slot->header.slot);
      const ActiveBaParameterBlockSpec* translation = FindParameter(
          inputs, ParameterKind::kTranslation, image.image_id);
      if (translation != nullptr) {
        intent.translation_subsets.push_back(
            {slot->header.slot, translation->translation_subset_mask,
             {0, 0, 0}});
      }
    }
    relevant_cameras.insert(image.camera_id);
  }
  for (const CameraSnapshot& camera : problem.cameras) {
    if (relevant_cameras.count(camera.camera_id) == 0) continue;
    const HostBaCameraSlot* slot = lease.FindCameraById(camera.camera_id);
    if (slot == nullptr || !slot->header.alive ||
        slot->model_id != camera.model_id || slot->width != camera.width ||
        slot->height != camera.height ||
        slot->parameter_count != camera.params.size()) {
      *error = "native active camera is inconsistent with the graph catalog";
      return false;
    }
    DenseCameraState state_value;
    state_value.camera_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.parameters = camera.params;
    dense.cameras.push_back(std::move(state_value));
    CameraParameterPolicy policy;
    policy.camera_slot = slot->header.slot;
    const ActiveBaParameterBlockSpec* parameter =
        FindParameter(inputs, ParameterKind::kCamera, camera.camera_id);
    policy.constant = parameter == nullptr || parameter->constant;
    if (parameter != nullptr) {
      policy.fixed_parameter_indices =
          parameter->fixed_camera_parameter_indices;
    }
    intent.camera_policies.push_back(std::move(policy));
  }
  for (const PointSnapshot& point : problem.points) {
    const HostBaPointSlot* slot = lease.FindPointById(point.point3D_id);
    if (slot == nullptr || !slot->header.alive) {
      *error = "native active point is missing from the graph catalog";
      return false;
    }
    DensePointState state_value;
    state_value.point_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.xyz = point.xyz;
    dense.points.push_back(state_value);
    relevant_points.insert(slot->header.slot);
    PointFixedPolicy policy;
    policy.point_slot = slot->header.slot;
    policy.constant = point.constant;
    policy.config_role = point.config_role;
    policy.has_search_range = point.has_search_range;
    policy.search_range = point.search_range;
    intent.point_policies.push_back(policy);
    if (point.config_role == 1) {
      intent.explicit_variable_point_slots.push_back(slot->header.slot);
      explicit_variable_points.insert(slot->header.slot);
    } else if (point.config_role == 2) {
      intent.explicit_constant_point_slots.push_back(slot->header.slot);
      explicit_constant_points.insert(slot->header.slot);
    }
  }
  std::unordered_map<uint64_t, uint32_t> visual_source_slots;
  std::unordered_map<uint64_t, uint32_t> lidar_source_slots;
  visual_source_slots.reserve(problem.observations.size());
  lidar_source_slots.reserve(problem.lidar.size());
  for (const ObservationSnapshot& observation : problem.observations) {
    const HostBaObservationSlot* slot =
        lease.FindObservation(observation.image_id, observation.point2D_idx);
    if (slot == nullptr || !slot->header.alive ||
        slot->point_slot >= lease.points().size ||
        lease.points()[slot->point_slot].point3D_id != observation.point3D_id ||
        slot->xy != observation.xy ||
        !visual_source_slots.emplace(observation.source_index,
                                     slot->header.slot).second) {
      *error = "native visual residual is inconsistent with the graph catalog";
      return false;
    }
    if (selected_images.count(observation.image_id) != 0) {
      intent.active_visual_observation_slots.push_back(slot->header.slot);
    } else if (explicit_variable_points.count(slot->point_slot) == 0 &&
               explicit_constant_points.count(slot->point_slot) == 0) {
      *error =
          "native boundary visual residual is not owned by an explicit point";
      return false;
    }
  }
  intent.lidar.lidar_map_generation = state->journal_cursor;
  intent.lidar.match_config_generation = resolved_config.config_generation;
  intent.lidar.constraints.reserve(problem.lidar.size());
  for (size_t index = 0; index < problem.lidar.size(); ++index) {
    const LidarSnapshot& lidar = problem.lidar[index];
    const HostBaPointSlot* point = lease.FindPointById(lidar.point3D_id);
    if (point == nullptr || !point->header.alive ||
        relevant_points.count(point->header.slot) == 0 ||
        !lidar_source_slots.emplace(lidar.source_index,
                                    static_cast<uint32_t>(index)).second) {
      *error = "native LiDAR residual is inconsistent with the active points";
      return false;
    }
    LidarConstraintRecord constraint;
    constraint.point_slot = point->header.slot;
    constraint.constraint_slot = static_cast<uint32_t>(index);
    constraint.physical_identity = lidar.source_index;
    constraint.lidar_type = lidar.lidar_type;
    constraint.plane = lidar.plane;
    constraint.lidar_xyz = lidar.lidar_xyz;
    constraint.weight = lidar.weight;
    constraint.search_range = lidar.search_range;
    constraint.point_state_generation = dense.state_generation;
    intent.lidar.constraints.push_back(constraint);
  }
  const size_t residual_count =
      problem.observations.size() + problem.lidar.size();
  if (problem.source_insertion_order.size() != residual_count) {
    *error = "native source order does not cover all residuals";
    return false;
  }
  intent.source_insertion_order.reserve(residual_count);
  for (size_t index = 0; index < residual_count; ++index) {
    const OrderEntrySnapshot& entry = problem.source_insertion_order[index];
    if (entry.source_index != index) {
      *error = "native source order is not a dense solve-local sequence";
      return false;
    }
    if (entry.residual_kind == ResidualKind::kVisual) {
      const auto found = visual_source_slots.find(index);
      if (found == visual_source_slots.end()) {
        *error = "native source order references a missing visual residual";
        return false;
      }
      intent.source_insertion_order.push_back(
          {ResidualKind::kVisual, found->second});
    } else {
      const auto found = lidar_source_slots.find(index);
      if (found == lidar_source_slots.end()) {
        *error = "native source order references a missing LiDAR residual";
        return false;
      }
      intent.source_insertion_order.push_back(
          {ResidualKind::kLidar, found->second});
    }
  }
  const auto materialize_start = std::chrono::steady_clock::now();
  NativeHostSolvePreparationRuntime materialize_runtime;
  if (!state->materializer.Materialize(
          lease, intent, dense, &data->view, &data->initial_state,
          &materialize_runtime, error)) {
    return false;
  }
  data->runtime.materialize_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - materialize_start).count();
  if (data->view.ResidualBlockCount() != inputs.residual_block_count ||
      data->view.ScalarResidualCount() != inputs.scalar_residual_count ||
      data->view.AmbientParameterCount() != inputs.ambient_parameter_count ||
      data->view.EffectiveParameterCount() !=
          inputs.effective_parameter_count ||
      data->view.VisualObservationSlots().size() !=
          problem.observations.size() ||
      data->view.LidarConstraints().size() != problem.lidar.size() ||
      data->view.ParameterOrdinals().size() != inputs.parameter_blocks.size) {
    *error = "native materialized view count contract mismatch";
    return false;
  }
  const auto graph_cameras = data->view.catalog.cameras();
  const auto graph_images = data->view.catalog.images();
  const auto graph_points = data->view.catalog.points();
  for (size_t i = 0; i < inputs.parameter_blocks.size; ++i) {
    const ActiveBaParameterBlockSpec& expected = inputs.parameter_blocks[i];
    const ParameterOrdinal& actual = data->view.ParameterOrdinals()[i];
    uint64_t entity_id = 0;
    if (actual.kind == ParameterKind::kQuaternion ||
        actual.kind == ParameterKind::kTranslation) {
      if (actual.entity_slot >= graph_images.size) return false;
      entity_id = graph_images[actual.entity_slot].image_id;
    } else if (actual.kind == ParameterKind::kPoint3D) {
      if (actual.entity_slot >= graph_points.size) return false;
      entity_id = graph_points[actual.entity_slot].point3D_id;
    } else {
      if (actual.entity_slot >= graph_cameras.size) return false;
      entity_id = graph_cameras[actual.entity_slot].camera_id;
    }
    if (actual.kind != expected.kind || entity_id != expected.entity_id ||
        actual.ambient_size != expected.ambient_size ||
        actual.tangent_size != expected.tangent_size ||
        static_cast<bool>(actual.constant) != expected.constant ||
        actual.translation_subset_mask !=
            expected.translation_subset_mask) {
      *error = "native parameter ordinal contract mismatch";
      return false;
    }
  }
  ++state->active_prepares;
  ++data->runtime.prepared_views;
  data->active = true;
  prepared->data_ = std::move(data);
  (void)start;
  return true;
}

bool PrepareCudaNativeBaSolve(
    const NativeBaSolveIntent& id_intent,
    Reconstruction* reconstruction,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedNativeActiveSolve* prepared,
    std::string* error) {
  if (prepared != nullptr) prepared->Release();
  if (prepared == nullptr || error == nullptr || reconstruction == nullptr ||
      id_intent.abi_version != kNativeHostSolveViewAbiVersion ||
      id_intent.owner_epoch == 0 ||
      id_intent.reconstruction_identity !=
          reinterpret_cast<uintptr_t>(reconstruction) ||
      id_intent.expected_topology_revision == 0 ||
      id_intent.selection_revision == 0 || !id_intent.config.resolved ||
      !IsValidNativeBaLidarScopeIdentity(
          id_intent.visual_observation_scope,
          id_intent.online_lidar_identity)) {
    if (error != nullptr) *error = "invalid ID-based native BA prepare";
    return false;
  }
  error->clear();
  if (binding.mode != CudaHostProblemStoreMode::kHostPreparedStore ||
      binding.store == nullptr || binding.owner_epoch != id_intent.owner_epoch ||
      binding.store->native_graph_state_ == nullptr) {
    *error = "native_graph requires a matching enabled host store";
    return false;
  }
  std::shared_ptr<NativeGraphStoreState> state =
      binding.store->native_graph_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  if (state->closing || state->active_prepares != 0 ||
      state->owner_epoch != id_intent.owner_epoch ||
      state->reconstruction != reconstruction) {
    *error = state->active_prepares != 0
        ? "StoreBusy: native graph already has an active solve"
        : "native graph store/reconstruction identity mismatch";
    return false;
  }
  std::shared_ptr<PreparedNativeActiveSolve::Data> data;
  try {
    data = std::make_shared<PreparedNativeActiveSolve::Data>();
  } catch (const std::bad_alloc&) {
    *error = "native graph prepared solve allocation failed";
    return false;
  }
  data->state = state;
  data->options = options;
  data->runtime.journal_cursor_before = state->journal_cursor;
  if (!SynchronizeNativeGraphStateLocked(state, &data->runtime, error))
    return false;
  if (state->journal_cursor != id_intent.expected_topology_revision ||
      reconstruction->StructureOwnerEpoch() != id_intent.owner_epoch ||
      reconstruction->StructureRevision() !=
          id_intent.expected_topology_revision) {
    *error = "native intent and committed topology revision differ";
    return false;
  }
  CatalogReadLease lease = state->graph.AcquireReadLease();
  if (!lease.valid()) {
    *error = "native graph catalog lease is unavailable";
    return false;
  }
  if (!ValidateLidarProvenanceIntentBeforeCache(id_intent, lease, error)) {
    return false;
  }
  if (++state->next_state_generation == 0) ++state->next_state_generation;

  const bool cache_enabled =
      id_intent.config.prepared_selection_cache ==
          CudaPreparedSelectionCacheMode::kEnabled &&
      options.prepared_selection_cache_mode ==
          CudaPreparedSelectionCacheMode::kEnabled;
  std::vector<uint8_t> plan_key;
  uint64_t plan_hash = 0;
  if (cache_enabled) {
    ++data->runtime.prepare_requests;
    const auto lookup_start = std::chrono::steady_clock::now();
    if (!BuildPlanLookupKey(id_intent, lease.slot_namespace_epoch(),
                            &plan_key, &plan_hash, error)) {
      return false;
    }
    std::shared_ptr<const PreparedSelectionPlan> plan =
        LookupPreparedSelectionPlan(state.get(), lease, plan_key, plan_hash,
                                    &data->runtime);
    data->runtime.host_plan_lookup_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - lookup_start).count();
    if (plan != nullptr) {
      if (plan->visual_observation_scope !=
              id_intent.visual_observation_scope ||
          !IsValidNativeBaLidarScopeIdentity(
              plan->visual_observation_scope,
              plan->online_lidar_identity) ||
          !SameNativeBaOnlineLidarIdentity(
              plan->online_lidar_identity,
              id_intent.online_lidar_identity)) {
        *error = "native prepared plan scope or LiDAR identity mismatch";
        return false;
      }
      data->view.identity.owner_epoch = lease.owner_epoch();
      data->view.identity.catalog_revision = lease.topology_revision();
      data->view.identity.catalog_generation = lease.generation();
      data->view.identity.selection_revision = id_intent.selection_revision;
      data->view.identity.config_generation =
          id_intent.config.config_generation;
      data->view.identity.lidar_map_generation =
          id_intent.lidar_map_generation;
      data->view.identity.lidar_match_config_generation =
          id_intent.lidar_match_config_generation;
      data->view.identity.visual_observation_scope =
          id_intent.visual_observation_scope;
      data->view.identity.online_lidar_identity =
          id_intent.online_lidar_identity;
      data->view.kind = id_intent.kind;
      data->view.config = id_intent.config;
      data->view.catalog = lease;
      data->view.lidar.lidar_map_generation =
          id_intent.lidar_map_generation;
      data->view.lidar.match_config_generation =
          id_intent.lidar_match_config_generation;
      data->view.lidar.online_lidar_identity =
          id_intent.online_lidar_identity;
      data->view.prepared_plan = std::move(plan);
      const auto bind_start = std::chrono::steady_clock::now();
      if (!GatherDynamicStateFromPreparedPlan(
              *reconstruction, lease, *data->view.prepared_plan,
              state->next_state_generation, &data->initial_state,
              &data->runtime, error) ||
          !ValidateNativeHostSolveView(data->view, data->initial_state,
                                       error)) {
        return false;
      }
      data->runtime.plan_bind_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - bind_start).count();
      data->runtime.host_plan_resident_bytes =
          state->host_plan_resident_bytes;
      data->runtime.host_plan_peak_bytes = state->host_plan_peak_bytes;
      ++state->active_prepares;
      ++data->runtime.prepared_views;
      data->active = true;
      prepared->data_ = std::move(data);
      return true;
    }
  }
  const auto id_resolution_start = std::chrono::steady_clock::now();

  BaSolveIntent slot_intent;
  slot_intent.owner_epoch = id_intent.owner_epoch;
  slot_intent.catalog_revision = lease.topology_revision();
  slot_intent.catalog_generation = lease.generation();
  slot_intent.selection_revision = id_intent.selection_revision;
  slot_intent.kind = id_intent.kind;
  slot_intent.visual_observation_scope =
      id_intent.visual_observation_scope;
  slot_intent.config = id_intent.config;
  slot_intent.visual_observation_slots_fully_resolved = true;
  const auto cameras = lease.cameras();
  const auto images = lease.images();
  const auto points = lease.points();
  const auto observations = lease.observations();
  const auto image_nodes = lease.image_incidence_nodes();
  const auto point_nodes = lease.point_incidence_nodes();

  std::unordered_set<uint32_t> active_images;
  std::unordered_set<uint32_t> relevant_images;
  std::unordered_set<uint32_t> relevant_points;
  std::unordered_set<uint32_t> relevant_cameras;
  std::unordered_set<uint32_t> selected_observations;
  active_images.reserve(id_intent.active_image_ids.size());
  relevant_images.reserve(id_intent.active_image_ids.size() * 2 + 1);
  relevant_points.reserve(id_intent.point_policies.size() + 1);
  selected_observations.reserve(observations.size < 1024
                                    ? observations.size : 1024);

  const auto visit_chain = [&](const uint32_t head,
                               const bool image_chain,
                               const uint32_t owner,
                               const auto& visitor) -> bool {
    const auto nodes = image_chain ? image_nodes : point_nodes;
    uint32_t node = head;
    size_t steps = 0;
    while (node != kBaGraphInvalidSlot) {
      if (node >= nodes.size || ++steps > nodes.size) {
        *error = "native intent incidence chain is corrupt";
        return false;
      }
      const HostBaIncidenceNode& incidence = nodes[node];
      if (incidence.observation_slot >= observations.size) {
        *error = "native intent incidence observation is invalid";
        return false;
      }
      const HostBaObservationSlot& observation =
          observations[incidence.observation_slot];
      const bool owns = image_chain ? observation.image_slot == owner
                                    : observation.point_slot == owner;
      if (observation.header.alive && owns &&
          incidence.association_generation ==
              observation.association_generation &&
          !visitor(incidence.observation_slot)) {
        return false;
      }
      ++data->runtime.intent_incidence_traversal_visits;
      node = incidence.next_node;
    }
    return true;
  };
  const auto add_visual = [&](const uint32_t observation_slot) -> bool {
    if (!selected_observations.insert(observation_slot).second) return true;
    const HostBaObservationSlot& observation = observations[observation_slot];
    relevant_images.insert(observation.image_slot);
    relevant_points.insert(observation.point_slot);
    slot_intent.source_insertion_order.push_back(
        {ResidualKind::kVisual, observation_slot});
    slot_intent.resolved_visual_observation_slots.push_back(observation_slot);
    return true;
  };
  for (const uint32_t image_id : id_intent.active_image_ids) {
    const HostBaImageSlot* image = lease.FindImageById(image_id);
    if (image == nullptr || !image->header.alive || !image->registered ||
        !active_images.insert(image->header.slot).second) {
      *error = "native intent active image is missing or duplicated";
      return false;
    }
    slot_intent.active_image_slots.push_back(image->header.slot);
    relevant_images.insert(image->header.slot);
    relevant_cameras.insert(image->camera_slot);
    std::vector<uint32_t> image_observations;
    if (!visit_chain(
            image->adjacency_head, true, image->header.slot,
            [&](const uint32_t slot) {
              image_observations.push_back(slot);
              return true;
            })) {
      return false;
    }
    std::sort(image_observations.begin(), image_observations.end(),
              [&](const uint32_t lhs, const uint32_t rhs) {
                return observations[lhs].point2D_idx <
                       observations[rhs].point2D_idx;
              });
    for (const uint32_t slot : image_observations)
      if (!add_visual(slot)) return false;
  }
  std::unordered_set<uint32_t> explicit_point_slots;
  explicit_point_slots.reserve(id_intent.explicit_variable_point_ids.size() +
                               id_intent.explicit_constant_point_ids.size());
  std::vector<uint32_t> deferred_constant_observations;
  const auto add_explicit_point = [&](const uint64_t point_id,
                                      const bool constant) -> bool {
    const HostBaPointSlot* point = lease.FindPointById(point_id);
    if (point == nullptr || !point->header.alive) {
      *error = constant ? "native intent explicit constant point is missing"
                        : "native intent explicit variable point is missing";
      return false;
    }
    if (!explicit_point_slots.insert(point->header.slot).second) {
      *error = "native intent explicit point selection is duplicated";
      return false;
    }
    if (constant) {
      slot_intent.explicit_constant_point_slots.push_back(point->header.slot);
    } else {
      slot_intent.explicit_variable_point_slots.push_back(point->header.slot);
    }
    relevant_points.insert(point->header.slot);
    if (id_intent.visual_observation_scope ==
        NativeBaVisualObservationScope::kActiveImagesOnly) {
      return true;
    }
    std::vector<uint32_t> point_observations;
    if (!visit_chain(
            point->adjacency_head, false, point->header.slot,
            [&](const uint32_t slot) {
              point_observations.push_back(slot);
              return true;
            })) {
      return false;
    }
    std::sort(point_observations.begin(), point_observations.end(),
              [&](const uint32_t lhs, const uint32_t rhs) {
                const HostBaObservationSlot& a = observations[lhs];
                const HostBaObservationSlot& b = observations[rhs];
                const uint32_t a_image = images[a.image_slot].image_id;
                const uint32_t b_image = images[b.image_slot].image_id;
                return std::tie(a_image, a.point2D_idx) <
                       std::tie(b_image, b.point2D_idx);
              });
    if (constant) {
      deferred_constant_observations.insert(
          deferred_constant_observations.end(), point_observations.begin(),
          point_observations.end());
    } else {
      for (const uint32_t slot : point_observations)
        if (!add_visual(slot)) return false;
    }
    return true;
  };
  for (const uint64_t point_id : id_intent.explicit_variable_point_ids) {
    if (!add_explicit_point(point_id, false)) return false;
  }
  for (const uint64_t point_id : id_intent.explicit_constant_point_ids) {
    if (!add_explicit_point(point_id, true)) return false;
  }
  for (const uint32_t image_id : id_intent.fixed_pose_ids) {
    const HostBaImageSlot* image = lease.FindImageById(image_id);
    if (image == nullptr) {
      *error = "native intent fixed pose is missing";
      return false;
    }
    slot_intent.fixed_pose_slots.push_back(image->header.slot);
  }
  for (const NativeBaTranslationPolicy& value :
       id_intent.translation_policies) {
    const HostBaImageSlot* image = lease.FindImageById(value.image_id);
    if (image == nullptr) {
      *error = "native intent translation policy image is missing";
      return false;
    }
    slot_intent.translation_subsets.push_back(
        {image->header.slot, value.constant_mask, {0, 0, 0}});
  }
  for (const NativeBaCameraPolicy& value : id_intent.camera_policies) {
    const HostBaCameraSlot* camera = lease.FindCameraById(value.camera_id);
    if (camera == nullptr) {
      *error = "native intent camera policy is missing";
      return false;
    }
    CameraParameterPolicy policy;
    policy.camera_slot = camera->header.slot;
    policy.constant = value.constant;
    policy.fixed_parameter_indices = value.fixed_parameter_indices;
    slot_intent.camera_policies.push_back(std::move(policy));
  }
  for (const NativeBaPointPolicy& value : id_intent.point_policies) {
    const HostBaPointSlot* point = lease.FindPointById(value.point3D_id);
    if (point == nullptr) {
      *error = "native intent point policy is missing";
      return false;
    }
    slot_intent.point_policies.push_back(
        {point->header.slot, value.constant, value.config_role,
         value.has_search_range, value.search_range});
  }

  data->runtime.intent_id_resolution_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - id_resolution_start).count();
  DenseActiveState dense;
  dense.owner_epoch = id_intent.owner_epoch;
  dense.state_generation = state->next_state_generation;
  slot_intent.lidar.lidar_map_generation = id_intent.lidar_map_generation;
  slot_intent.lidar.match_config_generation =
      id_intent.lidar_match_config_generation;
  slot_intent.lidar.online_lidar_identity =
      id_intent.online_lidar_identity;
  slot_intent.lidar.constraints.reserve(id_intent.lidar_constraints.size());
  for (const NativeBaLidarConstraint& value : id_intent.lidar_constraints) {
    const HostBaPointSlot* point = lease.FindPointById(value.point3D_id);
    if (point == nullptr || !point->header.alive ||
        relevant_points.count(point->header.slot) == 0) {
      *error = "native intent LiDAR point is outside the active point set";
      return false;
    }
    LidarConstraintRecord constraint;
    constraint.point_slot = point->header.slot;
    constraint.constraint_slot = value.constraint_slot;
    constraint.physical_identity = value.physical_identity;
    constraint.association_id = value.association_id;
    constraint.owner_image_id = value.owner_image_id;
    constraint.owner_point2D_idx = value.owner_point2D_idx;
    constraint.lidar_type = value.lidar_type;
    constraint.frozen_point3D_xyz = value.frozen_point3D_xyz;
    constraint.plane = value.plane;
    constraint.lidar_xyz = value.lidar_xyz;
    constraint.weight = value.weight;
    constraint.search_range = value.search_range;
    constraint.point_state_generation = dense.state_generation;
    slot_intent.lidar.constraints.push_back(constraint);
    slot_intent.source_insertion_order.push_back(
        {ResidualKind::kLidar, value.constraint_slot});
  }
  for (const uint32_t slot : deferred_constant_observations) {
    if (!add_visual(slot)) return false;
  }
  for (const uint32_t image_slot : relevant_images) {
    if (image_slot >= images.size ||
        images[image_slot].camera_slot >= cameras.size) {
      *error = "native intent relevant image/camera is invalid";
      return false;
    }
    relevant_cameras.insert(images[image_slot].camera_slot);
  }
  const auto dynamic_gather_start = std::chrono::steady_clock::now();
  for (const uint32_t slot : relevant_cameras) {
    if (slot >= cameras.size || !cameras[slot].header.alive ||
        !reconstruction->ExistsCamera(cameras[slot].camera_id)) {
      *error = "native intent camera state is unavailable";
      return false;
    }
    DenseCameraState value;
    value.camera_slot = slot;
    value.state_generation = dense.state_generation;
    value.parameters = reconstruction->Camera(cameras[slot].camera_id).Params();
    dense.cameras.push_back(std::move(value));
  }
  for (const uint32_t slot : relevant_images) {
    if (slot >= images.size || !images[slot].header.alive ||
        !reconstruction->ExistsImage(images[slot].image_id)) {
      *error = "native intent image state is unavailable";
      return false;
    }
    const Image& image = reconstruction->Image(images[slot].image_id);
    DenseImageState value;
    value.image_slot = slot;
    value.state_generation = dense.state_generation;
    std::copy(image.Qvec().data(), image.Qvec().data() + 4,
              value.quaternion.begin());
    std::copy(image.Tvec().data(), image.Tvec().data() + 3,
              value.translation.begin());
    dense.images.push_back(value);
  }
  for (const uint32_t slot : relevant_points) {
    if (slot >= points.size || !points[slot].header.alive ||
        !reconstruction->ExistsPoint3D(points[slot].point3D_id)) {
      *error = "native intent point state is unavailable";
      return false;
    }
    DensePointState value;
    value.point_slot = slot;
    value.state_generation = dense.state_generation;
    const Eigen::Vector3d& xyz =
        reconstruction->Point3D(points[slot].point3D_id).XYZ();
    std::copy(xyz.data(), xyz.data() + 3, value.xyz.begin());
    dense.points.push_back(value);
  }
  data->runtime.dynamic_state_gather_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - dynamic_gather_start).count();
  const auto materialize_start = std::chrono::steady_clock::now();
  NativeHostSolvePreparationRuntime materialize_runtime;
  if (!state->materializer.Materialize(
          lease, slot_intent, dense, &data->view, &data->initial_state,
          &materialize_runtime, error)) {
    return false;
  }
  data->runtime.materializer_incidence_traversal_visits =
      materialize_runtime.incidence_traversal_visits;
  data->runtime.materialize_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - materialize_start)
          .count();
  ++data->runtime.adjacency_static_materialize_calls;
  ++data->runtime.dynamic_state_gather_calls;
  if (cache_enabled) {
    const auto build_start = std::chrono::steady_clock::now();
    if (state->next_plan_publication_id ==
        std::numeric_limits<uint64_t>::max()) {
      *error = "native prepared plan publication identity exhausted";
      return false;
    }
    std::shared_ptr<PreparedSelectionPlan> plan;
    try {
      plan = ExtractPreparedSelectionPlan(
          &data->view, ++state->next_plan_publication_id);
    } catch (const std::bad_alloc&) {
      *error = "native prepared plan allocation failed";
      return false;
    }
    ++data->runtime.host_plan_build_calls;
    const bool published = PublishPreparedSelectionPlan(
        state.get(), lease, std::move(plan_key), plan_hash, plan,
        &data->runtime, error);
    if (!error->empty()) return false;
    if (!published) plan->publication_id = 0;
    data->runtime.host_plan_build_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start).count();
  }
  ++state->active_prepares;
  ++data->runtime.prepared_views;
  data->active = true;
  prepared->data_ = std::move(data);
  return true;
}

bool ValidateAndCommitNativeBaState(
    const NativeActiveSolveInputs& inputs,
    const PreparedNativeActiveSolve& prepared,
    const VariableStateDelta& candidate,
    Reconstruction* reconstruction,
    std::string* error) {
  if (inputs.problem == nullptr) {
    if (error != nullptr) *error = "native reference commit lacks a problem";
    return false;
  }
  NativeBaSolveIntent intent;
  intent.owner_epoch = inputs.owner_epoch;
  intent.expected_topology_revision = inputs.catalog_revision;
  intent.reconstruction_identity = reinterpret_cast<uintptr_t>(reconstruction);
  return ValidateAndCommitNativeBaDelta(intent, prepared, candidate,
                                       reconstruction, error);
}

bool ValidateAndCommitNativeBaDelta(
    const NativeBaSolveIntent& intent,
    const PreparedNativeActiveSolve& prepared,
    const VariableStateDelta& candidate,
    Reconstruction* reconstruction,
    std::string* error) {
  if (!prepared.valid() || reconstruction == nullptr || error == nullptr) {
    if (error != nullptr) *error = "invalid native graph delta commit";
    return false;
  }
  error->clear();
  const auto& data = *prepared.data_;
  const NativeHostSolveView& view = data.view;
  if (intent.owner_epoch != view.identity.owner_epoch ||
      intent.expected_topology_revision != view.identity.catalog_revision ||
      intent.reconstruction_identity !=
          reinterpret_cast<uintptr_t>(reconstruction) ||
      reconstruction->StructureOwnerEpoch() != view.identity.owner_epoch ||
      reconstruction->StructureRevision() != view.identity.catalog_revision ||
      candidate.owner_epoch != view.identity.owner_epoch ||
      candidate.catalog_revision != view.identity.catalog_revision ||
      candidate.catalog_generation != view.identity.catalog_generation ||
      candidate.view_generation != view.identity.selection_revision ||
      candidate.solve_generation == 0 || candidate.state_generation == 0 ||
      candidate.state_generation < data.initial_state.state_generation) {
    *error = "native variable delta identity mismatch";
    return false;
  }
  struct ImageUpdate {
    uint32_t id = 0;
    std::array<double, 4> q;
    std::array<double, 3> t;
    uint8_t mask = 0;
  };
  struct PointUpdate {
    uint64_t id = 0;
    std::array<double, 3> xyz;
  };
  std::vector<ImageUpdate> image_updates;
  std::vector<PointUpdate> point_updates;
  image_updates.reserve(candidate.images.size());
  point_updates.reserve(candidate.points.size());
  const auto images = view.catalog.images();
  const auto points = view.catalog.points();
  std::unordered_set<uint32_t> image_slots;
  std::unordered_set<uint32_t> point_slots;
  std::unordered_map<uint32_t, const ImageFixedPolicyResult*> image_policy;
  std::unordered_map<uint32_t, const PointFixedPolicyResult*> point_policy;
  image_policy.reserve(view.Fixed().images.size());
  point_policy.reserve(view.Fixed().points.size());
  for (const ImageFixedPolicyResult& value : view.Fixed().images)
    image_policy.emplace(value.image_slot, &value);
  for (const PointFixedPolicyResult& value : view.Fixed().points)
    point_policy.emplace(value.point_slot, &value);
  if (!candidate.cameras.empty() || !candidate.expected_camera_slots.empty()) {
    *error = "native variable camera delta is unsupported";
    return false;
  }
  for (const VariableImageStateDelta& value : candidate.images) {
    if (value.image_slot >= images.size ||
        !image_slots.insert(value.image_slot).second ||
        !Finite(value.quaternion) || !Finite(value.translation)) {
      *error = "native variable image delta is invalid";
      return false;
    }
    const auto policy = image_policy.find(value.image_slot);
    const uint32_t image_id = images[value.image_slot].image_id;
    if (policy == image_policy.end() || policy->second->pose_constant ||
        policy->second->translation_subset_mask !=
            value.translation_subset_mask ||
        !reconstruction->ExistsImage(image_id)) {
      *error = "native variable image policy/identity is invalid";
      return false;
    }
    const double norm = std::sqrt(
        value.quaternion[0] * value.quaternion[0] +
        value.quaternion[1] * value.quaternion[1] +
        value.quaternion[2] * value.quaternion[2] +
        value.quaternion[3] * value.quaternion[3]);
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-10) {
      *error = "native final quaternion norm is invalid";
      return false;
    }
    image_updates.push_back({image_id, value.quaternion, value.translation,
                             value.translation_subset_mask});
  }
  for (const VariablePointStateDelta& value : candidate.points) {
    if (value.point_slot >= points.size ||
        !point_slots.insert(value.point_slot).second || !Finite(value.xyz)) {
      *error = "native variable point delta is invalid";
      return false;
    }
    const auto policy = point_policy.find(value.point_slot);
    const uint64_t point_id = points[value.point_slot].point3D_id;
    if (policy == point_policy.end() || policy->second->constant ||
        !reconstruction->ExistsPoint3D(point_id)) {
      *error = "native variable point policy/identity is invalid";
      return false;
    }
    point_updates.push_back({point_id, value.xyz});
  }
  std::vector<uint32_t> actual_images(image_slots.begin(), image_slots.end());
  std::vector<uint32_t> actual_points(point_slots.begin(), point_slots.end());
  std::vector<uint32_t> expected_images = candidate.expected_image_slots;
  std::vector<uint32_t> expected_points = candidate.expected_point_slots;
  std::vector<uint32_t> view_expected_images;
  std::vector<uint32_t> view_expected_points;
  // The active view may retain selected entities that no longer participate
  // in any residual after Mapper filtering. They are present in Fixed() but
  // are intentionally absent from the native CUDA parameter layout and from
  // the variable-only device download. Derive commit coverage from the
  // residual-backed parameter ordinals, which are the authoritative solve
  // parameter set, rather than from every selected fixed-policy record.
  for (const ParameterOrdinal& value : view.ParameterOrdinals()) {
    if (value.constant) continue;
    if (value.kind == ParameterKind::kQuaternion) {
      view_expected_images.push_back(
          static_cast<uint32_t>(value.entity_slot));
    } else if (value.kind == ParameterKind::kPoint3D) {
      view_expected_points.push_back(
          static_cast<uint32_t>(value.entity_slot));
    }
  }
  std::sort(actual_images.begin(), actual_images.end());
  std::sort(actual_points.begin(), actual_points.end());
  std::sort(expected_images.begin(), expected_images.end());
  std::sort(expected_points.begin(), expected_points.end());
  std::sort(view_expected_images.begin(), view_expected_images.end());
  std::sort(view_expected_points.begin(), view_expected_points.end());
  if (actual_images != expected_images || actual_points != expected_points ||
      expected_images != view_expected_images ||
      expected_points != view_expected_points ||
      actual_images.size() != candidate.images.size() ||
      actual_points.size() != candidate.points.size()) {
    *error = "native variable delta slot coverage mismatch";
    return false;
  }
  for (const ImageUpdate& update : image_updates) {
    Image& image = reconstruction->Image(update.id);
    image.SetQvec(Eigen::Map<const Eigen::Vector4d>(update.q.data()));
    Eigen::Vector3d translation = image.Tvec();
    for (size_t component = 0; component < 3; ++component) {
      if ((update.mask & (1u << component)) == 0)
        translation[component] = update.t[component];
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
