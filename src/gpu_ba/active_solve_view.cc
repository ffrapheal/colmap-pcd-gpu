#include "gpu_ba/active_solve_view.h"
#include "gpu_ba/host_problem_store_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace colmap {
namespace gpu_ba {
namespace {

static_assert(std::is_standard_layout<MapperCatalogCameraSlot>::value &&
                  std::is_trivially_copyable<MapperCatalogCameraSlot>::value,
              "camera catalog slot must be upload-safe POD");
static_assert(std::is_standard_layout<MapperCatalogImageSlot>::value &&
                  std::is_trivially_copyable<MapperCatalogImageSlot>::value,
              "image catalog slot must be upload-safe POD");
static_assert(std::is_standard_layout<MapperCatalogPointSlot>::value &&
                  std::is_trivially_copyable<MapperCatalogPointSlot>::value,
              "point catalog slot must be upload-safe POD");
static_assert(
    std::is_standard_layout<MapperCatalogObservationSlot>::value &&
        std::is_trivially_copyable<MapperCatalogObservationSlot>::value,
    "observation catalog slot must be upload-safe POD");
static_assert(std::is_trivially_copyable<MapperCatalogIncidenceSlot>::value,
              "catalog incidence must be upload-safe POD");

uint64_t ObservationKey(const uint32_t image_id,
                        const uint32_t point2D_idx) noexcept {
  return (static_cast<uint64_t>(image_id) << 32) |
         static_cast<uint64_t>(point2D_idx);
}

bool CheckedAdd(const uint64_t lhs,
                const uint64_t rhs,
                uint64_t* output) noexcept {
  if (output == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *output = lhs + rhs;
  return true;
}

bool CheckedMultiply(const uint64_t lhs,
                     const uint64_t rhs,
                     uint64_t* output) noexcept {
  if (output == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *output = lhs * rhs;
  return true;
}

bool CheckedAlign(const uint64_t value,
                  const uint64_t alignment,
                  uint64_t* output) noexcept {
  if (output == nullptr || alignment == 0 ||
      (alignment & (alignment - 1)) != 0 ||
      value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    return false;
  }
  *output = (value + alignment - 1) & ~(alignment - 1);
  return true;
}

template <typename T>
bool FitsUint32(const T value) noexcept {
  return value <= static_cast<T>(std::numeric_limits<uint32_t>::max());
}

template <typename T>
uint64_t VectorBytes(const std::vector<T>& values) {
  return static_cast<uint64_t>(values.size()) * sizeof(T);
}

}  // namespace

struct MapperStaticProblemDataCatalog::Impl {
  explicit Impl(const uint64_t value) : owner_epoch(value) {}

  uint64_t owner_epoch = 0;
  uint64_t revision = 0;
  uint64_t generation = 0;
  std::vector<MapperCatalogCameraSlot> cameras;
  std::vector<MapperCatalogImageSlot> images;
  std::vector<MapperCatalogPointSlot> points;
  std::vector<MapperCatalogObservationSlot> observations;
  std::vector<MapperCatalogIncidenceSlot> image_incidences;
  std::vector<MapperCatalogIncidenceSlot> point_incidences;
  std::unordered_map<uint32_t, uint32_t> camera_slots;
  std::unordered_map<uint32_t, uint32_t> image_slots;
  std::unordered_map<uint64_t, uint32_t> point_slots;
  std::unordered_map<uint64_t, uint32_t> observation_slots;
};

namespace {

bool ValidCatalogOwner(const MapperStaticProblemDataCatalog::Impl& catalog,
                       const uint64_t owner_epoch,
                       const uint64_t revision,
                       std::string* error) {
  if (owner_epoch == 0 || owner_epoch != catalog.owner_epoch) {
    *error = "mapper static catalog owner identity mismatch";
    return false;
  }
  if (revision < catalog.revision) {
    *error = "mapper static catalog revision regressed";
    return false;
  }
  return true;
}

bool AppendIncidence(std::vector<MapperCatalogIncidenceSlot>* incidences,
                     uint32_t* head,
                     uint32_t* tail,
                     uint32_t* count,
                     const uint32_t observation_slot,
                     const uint64_t generation,
                     uint32_t* incidence_slot,
                     MapperStaticCatalogUpdateResult* result,
                     std::string* error) {
  if (!FitsUint32(incidences->size()) ||
      incidences->size() == std::numeric_limits<uint32_t>::max() ||
      *count == std::numeric_limits<uint32_t>::max()) {
    *error = "mapper catalog incidence capacity exceeds uint32";
    return false;
  }
  const uint32_t slot = static_cast<uint32_t>(incidences->size());
  MapperCatalogIncidenceSlot value;
  value.observation_slot = observation_slot;
  value.generation = generation;
  incidences->push_back(value);
  if (*tail == kIndexedInvalidSlot) {
    *head = slot;
  } else {
    if (*tail >= slot) {
      *error = "mapper catalog incidence tail is invalid";
      return false;
    }
    (*incidences)[*tail].next = slot;
    (*incidences)[*tail].generation = generation;
    ++result->updated_incidence_links;
  }
  *tail = slot;
  ++*count;
  *incidence_slot = slot;
  ++result->appended_incidences;
  return true;
}

bool ApplyProblemStaticData(MapperStaticProblemDataCatalog::Impl* catalog,
                            const CudaSolveProblem& problem,
                            const uint64_t generation,
                            MapperStaticCatalogUpdateResult* result,
                            std::string* error) {
  if (!FitsUint32(catalog->cameras.size()) ||
      !FitsUint32(catalog->images.size()) ||
      !FitsUint32(catalog->points.size()) ||
      !FitsUint32(catalog->observations.size())) {
    *error = "mapper static catalog stable slot capacity exceeds uint32";
    return false;
  }

  for (const CameraSnapshot& camera : problem.cameras) {
    const auto found = catalog->camera_slots.find(camera.camera_id);
    if (found != catalog->camera_slots.end()) {
      const MapperCatalogCameraSlot& stable = catalog->cameras[found->second];
      if (!stable.header.alive || stable.model_id != camera.model_id ||
          stable.width != camera.width || stable.height != camera.height ||
          stable.parameter_count != camera.params.size()) {
        *error = "mapper static catalog camera identity changed without delta";
        return false;
      }
      continue;
    }
    if (catalog->cameras.size() == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog camera slot capacity exhausted";
      return false;
    }
    MapperCatalogCameraSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->cameras.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.camera_id = camera.camera_id;
    stable.model_id = camera.model_id;
    stable.width = camera.width;
    stable.height = camera.height;
    stable.parameter_count = static_cast<uint32_t>(camera.params.size());
    catalog->camera_slots.emplace(camera.camera_id, stable.header.slot);
    catalog->cameras.push_back(stable);
    ++result->appended_cameras;
    result->content_changed = true;
  }

  for (const ImageSnapshot& image : problem.images) {
    const auto camera = catalog->camera_slots.find(image.camera_id);
    if (camera == catalog->camera_slots.end() ||
        !catalog->cameras[camera->second].header.alive) {
      *error = "mapper static catalog image references a missing camera";
      return false;
    }
    const auto found = catalog->image_slots.find(image.image_id);
    if (found != catalog->image_slots.end()) {
      MapperCatalogImageSlot& stable = catalog->images[found->second];
      if (!stable.header.alive || stable.camera_slot != camera->second) {
        *error = "mapper static catalog image identity changed without delta";
        return false;
      }
      continue;
    }
    if (catalog->images.size() == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog image slot capacity exhausted";
      return false;
    }
    MapperCatalogImageSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->images.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.image_id = image.image_id;
    stable.camera_slot = camera->second;
    stable.registered = 1;
    catalog->image_slots.emplace(image.image_id, stable.header.slot);
    catalog->images.push_back(stable);
    ++result->appended_images;
    result->content_changed = true;
  }

  for (const PointSnapshot& point : problem.points) {
    const auto found = catalog->point_slots.find(point.point3D_id);
    if (found != catalog->point_slots.end()) {
      if (!catalog->points[found->second].header.alive) {
        *error = "mapper static catalog point was tombstoned without add delta";
        return false;
      }
      continue;
    }
    if (catalog->points.size() == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog point slot capacity exhausted";
      return false;
    }
    MapperCatalogPointSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->points.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.point3D_id = point.point3D_id;
    catalog->point_slots.emplace(point.point3D_id, stable.header.slot);
    catalog->points.push_back(stable);
    ++result->appended_points;
    result->content_changed = true;
  }

  for (const ObservationSnapshot& observation : problem.observations) {
    const auto image = catalog->image_slots.find(observation.image_id);
    const auto point = catalog->point_slots.find(observation.point3D_id);
    if (image == catalog->image_slots.end() ||
        point == catalog->point_slots.end() ||
        !catalog->images[image->second].header.alive ||
        !catalog->points[point->second].header.alive) {
      *error = "mapper static catalog observation references a missing entity";
      return false;
    }
    const uint64_t key = ObservationKey(observation.image_id,
                                        observation.point2D_idx);
    const auto found = catalog->observation_slots.find(key);
    if (found != catalog->observation_slots.end()) {
      const MapperCatalogObservationSlot& stable =
          catalog->observations[found->second];
      if (!stable.header.alive || stable.image_slot != image->second ||
          stable.point_slot != point->second || stable.xy != observation.xy ||
          stable.source_identity != key) {
        *error =
            "mapper static catalog observation identity changed without delta";
        return false;
      }
      continue;
    }
    if (catalog->observations.size() ==
        std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog observation slot capacity exhausted";
      return false;
    }
    MapperCatalogObservationSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->observations.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.image_slot = image->second;
    stable.point_slot = point->second;
    stable.camera_slot = catalog->images[image->second].camera_slot;
    stable.point2D_idx = observation.point2D_idx;
    stable.source_identity = key;
    stable.xy = observation.xy;
    MapperCatalogImageSlot& stable_image = catalog->images[image->second];
    MapperCatalogPointSlot& stable_point = catalog->points[point->second];
    if (stable_point.track_length == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog alive track length exceeds uint32";
      return false;
    }
    if (!AppendIncidence(&catalog->image_incidences,
                         &stable_image.observation_head,
                         &stable_image.observation_tail,
                         &stable_image.observation_count, stable.header.slot,
                         generation, &stable.image_incidence, result, error) ||
        !AppendIncidence(&catalog->point_incidences,
                         &stable_point.observation_head,
                         &stable_point.observation_tail,
                         &stable_point.observation_count, stable.header.slot,
                         generation, &stable.point_incidence, result, error)) {
      return false;
    }
    stable_image.header.generation = generation;
    stable_point.header.generation = generation;
    ++stable_point.track_length;
    catalog->observation_slots.emplace(key, stable.header.slot);
    catalog->observations.push_back(stable);
    ++result->appended_observations;
    result->content_changed = true;
  }
  return true;
}

bool ProblemRequiresCatalogChange(
    const MapperStaticProblemDataCatalog::Impl& catalog,
    const CudaSolveProblem& problem,
    bool* change,
    std::string* error) {
  *change = catalog.generation == 0;
  for (const CameraSnapshot& camera : problem.cameras) {
    const auto found = catalog.camera_slots.find(camera.camera_id);
    if (found == catalog.camera_slots.end()) {
      *change = true;
      continue;
    }
    const MapperCatalogCameraSlot& stable = catalog.cameras[found->second];
    if (!stable.header.alive || stable.model_id != camera.model_id ||
        stable.width != camera.width || stable.height != camera.height ||
        stable.parameter_count != camera.params.size()) {
      *error = "mapper static catalog camera identity changed without delta";
      return false;
    }
  }
  for (const ImageSnapshot& image : problem.images) {
    const auto camera = catalog.camera_slots.find(image.camera_id);
    if (camera == catalog.camera_slots.end()) {
      *change = true;
      continue;
    }
    const auto found = catalog.image_slots.find(image.image_id);
    if (found == catalog.image_slots.end()) {
      *change = true;
      continue;
    }
    const MapperCatalogImageSlot& stable = catalog.images[found->second];
    if (!stable.header.alive || stable.camera_slot != camera->second) {
      *error = "mapper static catalog image identity changed without delta";
      return false;
    }
  }
  for (const PointSnapshot& point : problem.points) {
    const auto found = catalog.point_slots.find(point.point3D_id);
    if (found == catalog.point_slots.end()) {
      *change = true;
    } else if (!catalog.points[found->second].header.alive) {
      *error = "mapper static catalog point was tombstoned without add delta";
      return false;
    }
  }
  for (const ObservationSnapshot& observation : problem.observations) {
    const auto image = catalog.image_slots.find(observation.image_id);
    const auto point = catalog.point_slots.find(observation.point3D_id);
    if (image == catalog.image_slots.end() ||
        point == catalog.point_slots.end()) {
      *change = true;
      continue;
    }
    const auto found = catalog.observation_slots.find(
        ObservationKey(observation.image_id, observation.point2D_idx));
    if (found == catalog.observation_slots.end()) {
      *change = true;
      continue;
    }
    const MapperCatalogObservationSlot& stable =
        catalog.observations[found->second];
    if (!stable.header.alive || stable.image_slot != image->second ||
        stable.point_slot != point->second || stable.xy != observation.xy ||
        stable.source_identity != ObservationKey(observation.image_id,
                                                 observation.point2D_idx)) {
      *error =
          "mapper static catalog observation identity changed without delta";
      return false;
    }
  }
  return true;
}

template <typename Key, typename Slot>
bool TombstoneSlots(const std::vector<Key>& ids,
                    std::unordered_map<Key, uint32_t>* index,
                    std::vector<Slot>* slots,
                    const uint64_t generation,
                    uint64_t* count,
                    std::string* error) {
  std::unordered_set<Key> unique;
  unique.reserve(ids.size());
  for (const Key id : ids) {
    if (!unique.emplace(id).second) {
      *error = "mapper static catalog delta contains duplicate tombstones";
      return false;
    }
    const auto found = index->find(id);
    if (found == index->end() || found->second >= slots->size() ||
        !(*slots)[found->second].header.alive) {
      *error = "mapper static catalog tombstone references a missing slot";
      return false;
    }
    (*slots)[found->second].header.alive = 0;
    (*slots)[found->second].header.generation = generation;
    index->erase(found);
    ++*count;
  }
  return true;
}

uint64_t EstimateMapperCatalogImplBytes(
    const MapperStaticProblemDataCatalog::Impl& catalog) noexcept {
  uint64_t bytes = sizeof(catalog);
  bytes += VectorBytes(catalog.cameras) + VectorBytes(catalog.images) +
           VectorBytes(catalog.points) + VectorBytes(catalog.observations) +
           VectorBytes(catalog.image_incidences) +
           VectorBytes(catalog.point_incidences);
  bytes += static_cast<uint64_t>(catalog.camera_slots.size()) *
           (sizeof(uint32_t) * 2 + sizeof(void*));
  bytes += static_cast<uint64_t>(catalog.image_slots.size()) *
           (sizeof(uint32_t) * 2 + sizeof(void*));
  bytes += static_cast<uint64_t>(catalog.point_slots.size()) *
           (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*));
  bytes += static_cast<uint64_t>(catalog.observation_slots.size()) *
           (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*));
  return bytes;
}

bool TombstoneObservation(MapperStaticProblemDataCatalog::Impl* catalog,
                          const uint64_t key,
                          const uint64_t generation,
                          MapperStaticCatalogUpdateResult* result,
                          std::string* error) {
  const auto found = catalog->observation_slots.find(key);
  if (found == catalog->observation_slots.end() ||
      found->second >= catalog->observations.size() ||
      !catalog->observations[found->second].header.alive) {
    *error = "mapper static catalog observation tombstone is missing";
    return false;
  }
  MapperCatalogObservationSlot& observation =
      catalog->observations[found->second];
  if (observation.point_slot >= catalog->points.size() ||
      !catalog->points[observation.point_slot].header.alive) {
    *error = "mapper static catalog observation has a dangling point";
    return false;
  }
  MapperCatalogPointSlot& point = catalog->points[observation.point_slot];
  if (point.track_length == 0) {
    *error = "mapper static catalog alive track length underflow";
    return false;
  }
  observation.header.alive = 0;
  observation.header.generation = generation;
  catalog->observation_slots.erase(found);
  --point.track_length;
  point.header.generation = generation;
  ++result->tombstoned_observations;
  result->content_changed = true;
  return true;
}

bool ValidateMapperCatalog(const MapperStaticProblemDataCatalog::Impl& catalog,
                           std::string* error) {
  const auto validate_incidence_chain =
      [&](const uint32_t owner_slot,
          const uint32_t head,
          const uint32_t tail,
          const uint32_t expected_count,
          const std::vector<MapperCatalogIncidenceSlot>& incidences,
          const bool image_chain) {
        if (expected_count == 0) {
          if (head != kIndexedInvalidSlot || tail != kIndexedInvalidSlot) {
            *error = "mapper static catalog empty incidence chain is invalid";
            return false;
          }
          return true;
        }
        if (head == kIndexedInvalidSlot || tail == kIndexedInvalidSlot) {
          *error = "mapper static catalog incidence endpoints are missing";
          return false;
        }
        uint32_t current = head;
        uint32_t count = 0;
        uint32_t last = kIndexedInvalidSlot;
        while (current != kIndexedInvalidSlot) {
          if (current >= incidences.size() || count == expected_count) {
            *error = "mapper static catalog incidence chain is out of bounds";
            return false;
          }
          const MapperCatalogIncidenceSlot& incidence = incidences[current];
          if (incidence.observation_slot >= catalog.observations.size()) {
            *error = "mapper static catalog incidence observation is invalid";
            return false;
          }
          const MapperCatalogObservationSlot& observation =
              catalog.observations[incidence.observation_slot];
          if ((image_chain &&
               (observation.image_slot != owner_slot ||
                observation.image_incidence != current)) ||
              (!image_chain &&
               (observation.point_slot != owner_slot ||
                observation.point_incidence != current))) {
            *error = "mapper static catalog incidence ownership is invalid";
            return false;
          }
          last = current;
          ++count;
          if (incidence.next != kIndexedInvalidSlot &&
              incidence.next <= current) {
            *error = "mapper static catalog incidence chain is not append-only";
            return false;
          }
          current = incidence.next;
        }
        if (count != expected_count || last != tail) {
          *error = "mapper static catalog incidence count is inconsistent";
          return false;
        }
        return true;
      };
  std::vector<uint32_t> alive_point_observations(catalog.points.size(), 0);
  for (const auto& item : catalog.observation_slots) {
    if (item.second >= catalog.observations.size()) {
      *error = "mapper static catalog observation index is out of bounds";
      return false;
    }
    const MapperCatalogObservationSlot& observation =
        catalog.observations[item.second];
    if (!observation.header.alive || observation.source_identity != item.first ||
        observation.image_slot >= catalog.images.size() ||
        observation.point_slot >= catalog.points.size() ||
        observation.camera_slot >= catalog.cameras.size() ||
        !catalog.images[observation.image_slot].header.alive ||
        !catalog.points[observation.point_slot].header.alive ||
        !catalog.cameras[observation.camera_slot].header.alive ||
        catalog.images[observation.image_slot].camera_slot !=
            observation.camera_slot) {
      *error = "mapper static catalog contains a dangling observation";
      return false;
    }
    if (alive_point_observations[observation.point_slot] ==
        std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog alive observation count overflow";
      return false;
    }
    ++alive_point_observations[observation.point_slot];
  }
  for (size_t index = 0; index < catalog.points.size(); ++index) {
    const MapperCatalogPointSlot& point = catalog.points[index];
    if (point.header.alive) {
      if (point.track_length != alive_point_observations[index]) {
        *error = "mapper static catalog alive track length is inconsistent";
        return false;
      }
      if (!validate_incidence_chain(
              static_cast<uint32_t>(index), point.observation_head,
              point.observation_tail, point.observation_count,
              catalog.point_incidences, false)) {
        return false;
      }
    }
  }
  for (const auto& item : catalog.image_slots) {
    if (item.second >= catalog.images.size() ||
        !catalog.images[item.second].header.alive ||
        catalog.images[item.second].camera_slot >= catalog.cameras.size() ||
        !catalog.cameras[catalog.images[item.second].camera_slot].header.alive) {
      *error = "mapper static catalog contains a dangling image";
      return false;
    }
    const MapperCatalogImageSlot& image = catalog.images[item.second];
    if (!validate_incidence_chain(item.second, image.observation_head,
                                  image.observation_tail,
                                  image.observation_count,
                                  catalog.image_incidences, true)) {
      return false;
    }
  }
  return true;
}

bool AppendHostStaticData(MapperStaticProblemDataCatalog::Impl* catalog,
                          const HostIndexedCatalogData& data,
                          const uint64_t generation,
                          MapperStaticCatalogUpdateResult* result,
                          std::string* error) {
  for (const HostCatalogCamera& camera : data.cameras) {
    if (catalog->camera_slots.count(camera.camera_id) != 0) continue;
    if (catalog->cameras.size() == std::numeric_limits<uint32_t>::max() ||
        camera.parameter_count > std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog camera capacity exceeds uint32";
      return false;
    }
    MapperCatalogCameraSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->cameras.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.camera_id = camera.camera_id;
    stable.model_id = camera.model_id;
    stable.width = camera.width;
    stable.height = camera.height;
    stable.parameter_count = static_cast<uint32_t>(camera.parameter_count);
    catalog->camera_slots.emplace(camera.camera_id, stable.header.slot);
    catalog->cameras.push_back(stable);
    ++result->appended_cameras;
    result->content_changed = true;
  }
  for (const HostCatalogImage& image : data.images) {
    if (catalog->image_slots.count(image.image_id) != 0) continue;
    const auto camera = catalog->camera_slots.find(image.camera_id);
    if (camera == catalog->camera_slots.end() ||
        catalog->images.size() == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog host image identity is invalid";
      return false;
    }
    MapperCatalogImageSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->images.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.image_id = image.image_id;
    stable.camera_slot = camera->second;
    stable.registered = image.registered ? 1 : 0;
    catalog->image_slots.emplace(image.image_id, stable.header.slot);
    catalog->images.push_back(stable);
    ++result->appended_images;
    result->content_changed = true;
  }
  for (const HostCatalogPoint& point : data.points) {
    if (catalog->point_slots.count(point.point3D_id) != 0) continue;
    if (catalog->points.size() == std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog host point capacity exceeds uint32";
      return false;
    }
    MapperCatalogPointSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->points.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.point3D_id = point.point3D_id;
    catalog->point_slots.emplace(point.point3D_id, stable.header.slot);
    catalog->points.push_back(stable);
    ++result->appended_points;
    result->content_changed = true;
  }
  for (const HostCatalogObservation& observation : data.observations) {
    const uint64_t key = ObservationKey(observation.image_id,
                                        observation.point2D_idx);
    if (catalog->observation_slots.count(key) != 0) continue;
    const auto image = catalog->image_slots.find(observation.image_id);
    const auto point = catalog->point_slots.find(observation.point3D_id);
    if (image == catalog->image_slots.end() ||
        point == catalog->point_slots.end() ||
        catalog->observations.size() ==
            std::numeric_limits<uint32_t>::max()) {
      *error = "mapper static catalog host observation identity is invalid";
      return false;
    }
    MapperCatalogObservationSlot stable;
    stable.header.slot = static_cast<uint32_t>(catalog->observations.size());
    stable.header.generation = generation;
    stable.header.alive = 1;
    stable.image_slot = image->second;
    stable.point_slot = point->second;
    stable.camera_slot = catalog->images[image->second].camera_slot;
    stable.point2D_idx = observation.point2D_idx;
    stable.source_identity = key;
    stable.xy = observation.xy;
    MapperCatalogImageSlot& stable_image = catalog->images[image->second];
    MapperCatalogPointSlot& stable_point = catalog->points[point->second];
    if (stable_point.track_length == std::numeric_limits<uint32_t>::max() ||
        !AppendIncidence(&catalog->image_incidences,
                         &stable_image.observation_head,
                         &stable_image.observation_tail,
                         &stable_image.observation_count, stable.header.slot,
                         generation, &stable.image_incidence, result, error) ||
        !AppendIncidence(&catalog->point_incidences,
                         &stable_point.observation_head,
                         &stable_point.observation_tail,
                         &stable_point.observation_count, stable.header.slot,
                         generation, &stable.point_incidence, result, error)) {
      return false;
    }
    ++stable_point.track_length;
    stable_image.header.generation = generation;
    stable_point.header.generation = generation;
    catalog->observation_slots.emplace(key, stable.header.slot);
    catalog->observations.push_back(stable);
    ++result->appended_observations;
    result->content_changed = true;
  }
  return true;
}

}  // namespace

MapperStaticProblemDataCatalog::MapperStaticProblemDataCatalog(
    const uint64_t owner_epoch, const uint64_t generation_floor)
    : impl_(new Impl(owner_epoch)) {
  impl_->generation = generation_floor;
}

MapperStaticProblemDataCatalog::MapperStaticProblemDataCatalog(
    const MapperStaticProblemDataCatalog& other)
    : impl_(new Impl(*other.impl_)) {}

MapperStaticProblemDataCatalog::~MapperStaticProblemDataCatalog() = default;

uint64_t MapperStaticProblemDataCatalog::owner_epoch() const noexcept {
  return impl_->owner_epoch;
}

uint64_t MapperStaticProblemDataCatalog::revision() const noexcept {
  return impl_->revision;
}

uint64_t MapperStaticProblemDataCatalog::generation() const noexcept {
  return impl_->generation;
}

uint64_t MapperStaticProblemDataCatalog::camera_slot_count() const noexcept {
  return impl_->cameras.size();
}

uint64_t MapperStaticProblemDataCatalog::image_slot_count() const noexcept {
  return impl_->images.size();
}

uint64_t MapperStaticProblemDataCatalog::point_slot_count() const noexcept {
  return impl_->points.size();
}

uint64_t MapperStaticProblemDataCatalog::observation_slot_count() const noexcept {
  return impl_->observations.size();
}

uint64_t MapperStaticProblemDataCatalog::image_incidence_count() const noexcept {
  return impl_->image_incidences.size();
}

uint64_t MapperStaticProblemDataCatalog::point_incidence_count() const noexcept {
  return impl_->point_incidences.size();
}

uint64_t MapperStaticProblemDataCatalog::EstimatedResidentBytes() const noexcept {
  return EstimateMapperCatalogImplBytes(*impl_);
}

const MapperCatalogCameraSlot* MapperStaticProblemDataCatalog::FindCamera(
    const uint32_t id) const noexcept {
  const auto found = impl_->camera_slots.find(id);
  return found == impl_->camera_slots.end()
      ? nullptr : &impl_->cameras[found->second];
}

const MapperCatalogImageSlot* MapperStaticProblemDataCatalog::FindImage(
    const uint32_t id) const noexcept {
  const auto found = impl_->image_slots.find(id);
  return found == impl_->image_slots.end()
      ? nullptr : &impl_->images[found->second];
}

const MapperCatalogPointSlot* MapperStaticProblemDataCatalog::FindPoint(
    const uint64_t id) const noexcept {
  const auto found = impl_->point_slots.find(id);
  return found == impl_->point_slots.end()
      ? nullptr : &impl_->points[found->second];
}

const MapperCatalogObservationSlot*
MapperStaticProblemDataCatalog::FindObservation(
    const uint32_t image_id, const uint32_t point2D_idx) const noexcept {
  const auto found = impl_->observation_slots.find(
      ObservationKey(image_id, point2D_idx));
  return found == impl_->observation_slots.end()
      ? nullptr : &impl_->observations[found->second];
}

bool MapperStaticProblemDataCatalog::ExportStableTables(
    MapperStaticCatalogStableTables* output, std::string* error) const {
  if (output == nullptr || error == nullptr || impl_->owner_epoch == 0 ||
      impl_->generation == 0) {
    if (error != nullptr) *error = "invalid mapper static catalog export";
    return false;
  }
  try {
    MapperStaticCatalogStableTables tables;
    tables.owner_epoch = impl_->owner_epoch;
    tables.catalog_revision = impl_->revision;
    tables.catalog_generation = impl_->generation;
    tables.cameras = impl_->cameras;
    tables.images = impl_->images;
    tables.points = impl_->points;
    tables.observations = impl_->observations;
    tables.image_incidences = impl_->image_incidences;
    tables.point_incidences = impl_->point_incidences;
    *output = std::move(tables);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "mapper static catalog export allocation failed";
    return false;
  }
}

bool MapperStaticProblemDataCatalog::EnsureProblemStaticData(
    const uint64_t owner_epoch,
    const uint64_t revision,
    const CudaSolveProblem& problem,
    MapperStaticCatalogUpdateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr ||
      !ValidCatalogOwner(*impl_, owner_epoch, revision, error)) {
    return false;
  }
  error->clear();
  MapperStaticCatalogUpdateResult pending_result;
  pending_result.generation_before = impl_->generation;
  try {
    bool content_change = false;
    if (!ProblemRequiresCatalogChange(*impl_, problem, &content_change,
                                      error)) {
      return false;
    }
    if (!content_change) {
      impl_->revision = revision;
      pending_result.generation_after = impl_->generation;
      *result = pending_result;
      return true;
    }
    Impl pending = *impl_;
    pending_result.estimated_impl_copy_bytes =
        EstimateMapperCatalogImplBytes(*impl_);
    uint64_t next_generation = pending.generation;
    if (next_generation == std::numeric_limits<uint64_t>::max()) {
      *error = "mapper static catalog generation exhausted";
      return false;
    }
    ++next_generation;
    if (next_generation == 0) ++next_generation;
    if (!ApplyProblemStaticData(&pending, problem, next_generation,
                                &pending_result, error)) {
      return false;
    }
    if (!ValidateMapperCatalog(pending, error)) return false;
    pending_result.content_changed = true;
    pending.revision = revision;
    if (pending_result.content_changed) pending.generation = next_generation;
    pending_result.generation_after = pending.generation;
    *impl_ = std::move(pending);
    *result = pending_result;
    return true;
  } catch (const std::bad_alloc&) {
    *error = "mapper static catalog update allocation failed";
    return false;
  }
}

bool MapperStaticProblemDataCatalog::ApplyDelta(
    const uint64_t owner_epoch,
    const uint64_t revision,
    const CudaSolveProblem& appended_problem,
    const MapperStaticCatalogTombstones& tombstones,
    MapperStaticCatalogUpdateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr ||
      !ValidCatalogOwner(*impl_, owner_epoch, revision, error)) {
    return false;
  }
  error->clear();
  MapperStaticCatalogUpdateResult pending_result;
  pending_result.generation_before = impl_->generation;
  try {
    Impl pending = *impl_;
    pending_result.estimated_impl_copy_bytes =
        EstimateMapperCatalogImplBytes(*impl_);
    if (pending.generation == std::numeric_limits<uint64_t>::max()) {
      *error = "mapper static catalog generation exhausted";
      return false;
    }
    uint64_t next_generation = pending.generation + 1;
    if (next_generation == 0) ++next_generation;

    std::unordered_set<uint32_t> camera_ids(tombstones.cameras.begin(),
                                            tombstones.cameras.end());
    std::unordered_set<uint32_t> image_ids(tombstones.images.begin(),
                                           tombstones.images.end());
    std::unordered_set<uint64_t> point_ids(tombstones.points.begin(),
                                           tombstones.points.end());
    if (camera_ids.size() != tombstones.cameras.size() ||
        image_ids.size() != tombstones.images.size() ||
        point_ids.size() != tombstones.points.size()) {
      *error = "mapper static catalog delta contains duplicate tombstones";
      return false;
    }
    for (const uint32_t camera_id : camera_ids) {
      const auto camera = pending.camera_slots.find(camera_id);
      if (camera == pending.camera_slots.end()) {
        *error = "mapper static catalog camera tombstone is missing";
        return false;
      }
      for (const auto& image : pending.image_slots) {
        if (pending.images[image.second].camera_slot == camera->second)
          image_ids.insert(image.first);
      }
    }

    std::unordered_set<uint32_t> image_slots;
    std::unordered_set<uint32_t> point_slots;
    for (const uint32_t image_id : image_ids) {
      const auto image = pending.image_slots.find(image_id);
      if (image == pending.image_slots.end()) {
        *error = "mapper static catalog image tombstone is missing";
        return false;
      }
      image_slots.insert(image->second);
    }
    for (const uint64_t point_id : point_ids) {
      const auto point = pending.point_slots.find(point_id);
      if (point == pending.point_slots.end()) {
        *error = "mapper static catalog point tombstone is missing";
        return false;
      }
      point_slots.insert(point->second);
    }

    std::unordered_set<uint64_t> observation_keys;
    observation_keys.reserve(tombstones.observations.size());
    for (const auto& id : tombstones.observations) {
      const uint64_t key = ObservationKey(id[0], id[1]);
      if (!observation_keys.emplace(key).second) {
        *error = "mapper static catalog delta contains duplicate observations";
        return false;
      }
    }
    for (const auto& item : pending.observation_slots) {
      const MapperCatalogObservationSlot& observation =
          pending.observations[item.second];
      if (image_slots.count(observation.image_slot) != 0 ||
          point_slots.count(observation.point_slot) != 0) {
        observation_keys.insert(item.first);
      }
    }
    for (const uint64_t key : observation_keys) {
      if (!TombstoneObservation(&pending, key, next_generation,
                                &pending_result, error)) {
        return false;
      }
    }

    const std::vector<uint32_t> cascade_images(image_ids.begin(),
                                               image_ids.end());
    const std::vector<uint32_t> cascade_cameras(camera_ids.begin(),
                                                camera_ids.end());
    const std::vector<uint64_t> cascade_points(point_ids.begin(),
                                               point_ids.end());
    for (const uint64_t point_id : cascade_points) {
      const auto point = pending.point_slots.find(point_id);
      if (point == pending.point_slots.end() ||
          pending.points[point->second].track_length != 0) {
        *error = "mapper static catalog point tombstone retains observations";
        return false;
      }
    }
    if (!TombstoneSlots(cascade_images, &pending.image_slots,
                        &pending.images, next_generation,
                        &pending_result.tombstoned_images, error) ||
        !TombstoneSlots(cascade_points, &pending.point_slots,
                        &pending.points, next_generation,
                        &pending_result.tombstoned_points, error) ||
        !TombstoneSlots(cascade_cameras, &pending.camera_slots,
                        &pending.cameras, next_generation,
                        &pending_result.tombstoned_cameras, error)) {
      return false;
    }
    pending_result.content_changed = pending_result.content_changed ||
        pending_result.tombstoned_cameras != 0 ||
        pending_result.tombstoned_images != 0 ||
        pending_result.tombstoned_points != 0;
    // Replacements are appended only after the old stable slots have been
    // tombstoned, so a point merge or observation reassociation never reuses
    // an old slot.
    if (!ApplyProblemStaticData(&pending, appended_problem, next_generation,
                                &pending_result, error)) {
      return false;
    }
    if (!ValidateMapperCatalog(pending, error)) return false;
    pending.revision = revision;
    if (pending_result.content_changed) pending.generation = next_generation;
    pending_result.generation_after = pending.generation;
    *impl_ = std::move(pending);
    *result = pending_result;
    return true;
  } catch (const std::bad_alloc&) {
    *error = "mapper static catalog delta allocation failed";
    return false;
  }
}

bool MapperStaticProblemDataCatalog::ReconcileFullStaticData(
    const HostIndexedCatalogData& data,
    MapperStaticCatalogUpdateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr || data.owner_epoch == 0 ||
      data.owner_epoch != impl_->owner_epoch || data.revision < impl_->revision) {
    if (error != nullptr) *error = "invalid full indexed catalog reconcile";
    return false;
  }
  error->clear();
  MapperStaticCatalogUpdateResult pending_result;
  pending_result.generation_before = impl_->generation;
  const bool seeded_full_rebuild =
      impl_->revision == 0 && impl_->cameras.empty() && impl_->images.empty() &&
      impl_->points.empty() && impl_->observations.empty();
  pending_result.full_rebuild = impl_->generation == 0 || seeded_full_rebuild;
  pending_result.estimated_impl_copy_bytes =
      EstimateMapperCatalogImplBytes(*impl_);
  pending_result.full_graph_records_scanned =
      data.cameras.size() + data.images.size() + data.points.size() +
      data.observations.size();
  try {
    Impl pending = *impl_;
    if (pending.generation == std::numeric_limits<uint64_t>::max()) {
      *error = "mapper static catalog generation exhausted";
      return false;
    }
    uint64_t next_generation = pending.generation + 1;
    if (next_generation == 0) ++next_generation;

    std::unordered_map<uint32_t, const HostCatalogCamera*> desired_cameras;
    std::unordered_map<uint32_t, const HostCatalogImage*> desired_images;
    std::unordered_map<uint64_t, const HostCatalogPoint*> desired_points;
    std::unordered_map<uint64_t, const HostCatalogObservation*>
        desired_observations;
    desired_cameras.reserve(data.cameras.size());
    desired_images.reserve(data.images.size());
    desired_points.reserve(data.points.size());
    desired_observations.reserve(data.observations.size());
    for (const HostCatalogCamera& camera : data.cameras) {
      if (!desired_cameras.emplace(camera.camera_id, &camera).second) {
        *error = "full indexed catalog contains duplicate camera";
        return false;
      }
    }
    for (const HostCatalogImage& image : data.images) {
      if (!desired_images.emplace(image.image_id, &image).second ||
          desired_cameras.count(image.camera_id) == 0) {
        *error = "full indexed catalog image identity is invalid";
        return false;
      }
    }
    for (const HostCatalogPoint& point : data.points) {
      if (!desired_points.emplace(point.point3D_id, &point).second) {
        *error = "full indexed catalog contains duplicate point";
        return false;
      }
    }
    for (const HostCatalogObservation& observation : data.observations) {
      const uint64_t key = ObservationKey(observation.image_id,
                                          observation.point2D_idx);
      if (!desired_observations.emplace(key, &observation).second ||
          desired_images.count(observation.image_id) == 0 ||
          desired_points.count(observation.point3D_id) == 0) {
        *error = "full indexed catalog observation identity is invalid";
        return false;
      }
    }

    std::unordered_set<uint32_t> tombstone_cameras;
    std::unordered_set<uint32_t> tombstone_images;
    std::unordered_set<uint64_t> tombstone_points;
    std::unordered_set<uint64_t> tombstone_observations;
    for (const auto& current : pending.camera_slots) {
      const MapperCatalogCameraSlot& slot = pending.cameras[current.second];
      const auto desired = desired_cameras.find(current.first);
      if (desired == desired_cameras.end() ||
          slot.model_id != desired->second->model_id ||
          slot.width != desired->second->width ||
          slot.height != desired->second->height ||
          slot.parameter_count != desired->second->parameter_count) {
        tombstone_cameras.insert(current.first);
      }
    }
    for (const auto& current : pending.image_slots) {
      const MapperCatalogImageSlot& slot = pending.images[current.second];
      const auto desired = desired_images.find(current.first);
      const uint32_t current_camera_id =
          slot.camera_slot < pending.cameras.size()
              ? pending.cameras[slot.camera_slot].camera_id
              : kIndexedInvalidSlot;
      if (desired == desired_images.end() ||
          current_camera_id != desired->second->camera_id ||
          tombstone_cameras.count(current_camera_id) != 0) {
        tombstone_images.insert(current.first);
      }
    }
    for (const auto& current : pending.point_slots) {
      if (desired_points.count(current.first) == 0)
        tombstone_points.insert(current.first);
    }
    for (const auto& current : pending.observation_slots) {
      const MapperCatalogObservationSlot& slot =
          pending.observations[current.second];
      const auto desired = desired_observations.find(current.first);
      const uint32_t image_id = slot.image_slot < pending.images.size()
          ? pending.images[slot.image_slot].image_id : kIndexedInvalidSlot;
      const uint64_t point_id = slot.point_slot < pending.points.size()
          ? pending.points[slot.point_slot].point3D_id
          : std::numeric_limits<uint64_t>::max();
      if (desired == desired_observations.end() ||
          desired->second->image_id != image_id ||
          desired->second->point3D_id != point_id ||
          desired->second->xy != slot.xy ||
          tombstone_images.count(image_id) != 0 ||
          tombstone_points.count(point_id) != 0) {
        tombstone_observations.insert(current.first);
      }
    }
    for (const auto& current : pending.observation_slots) {
      const MapperCatalogObservationSlot& slot =
          pending.observations[current.second];
      const uint32_t image_id = pending.images[slot.image_slot].image_id;
      const uint64_t point_id = pending.points[slot.point_slot].point3D_id;
      if (tombstone_images.count(image_id) != 0 ||
          tombstone_points.count(point_id) != 0) {
        tombstone_observations.insert(current.first);
      }
    }
    for (const uint64_t key : tombstone_observations) {
      if (!TombstoneObservation(&pending, key, next_generation,
                                &pending_result, error)) {
        return false;
      }
    }
    for (const uint64_t point_id : tombstone_points) {
      const auto point = pending.point_slots.find(point_id);
      if (point == pending.point_slots.end() ||
          pending.points[point->second].track_length != 0) {
        *error = "full indexed catalog point retains an alive observation";
        return false;
      }
    }
    const std::vector<uint32_t> cameras(tombstone_cameras.begin(),
                                        tombstone_cameras.end());
    const std::vector<uint32_t> images(tombstone_images.begin(),
                                       tombstone_images.end());
    const std::vector<uint64_t> points(tombstone_points.begin(),
                                       tombstone_points.end());
    if (!TombstoneSlots(images, &pending.image_slots, &pending.images,
                        next_generation, &pending_result.tombstoned_images,
                        error) ||
        !TombstoneSlots(points, &pending.point_slots, &pending.points,
                        next_generation, &pending_result.tombstoned_points,
                        error) ||
        !TombstoneSlots(cameras, &pending.camera_slots, &pending.cameras,
                        next_generation, &pending_result.tombstoned_cameras,
                        error) ||
        !AppendHostStaticData(&pending, data, next_generation,
                              &pending_result, error)) {
      return false;
    }
    pending_result.content_changed = pending_result.content_changed ||
        pending_result.tombstoned_cameras != 0 ||
        pending_result.tombstoned_images != 0 ||
        pending_result.tombstoned_points != 0;
    for (const HostCatalogImage& image : data.images) {
      const auto current = pending.image_slots.find(image.image_id);
      if (current == pending.image_slots.end()) {
        *error = "full indexed catalog image publication is incomplete";
        return false;
      }
      MapperCatalogImageSlot& slot = pending.images[current->second];
      const uint8_t registered = image.registered ? 1 : 0;
      if (slot.registered != registered) {
        slot.registered = registered;
        slot.header.generation = next_generation;
        pending_result.content_changed = true;
      }
    }
    if (pending.generation == 0 || seeded_full_rebuild)
      pending_result.content_changed = true;
    pending.revision = data.revision;
    if (pending_result.content_changed) pending.generation = next_generation;
    if (!ValidateMapperCatalog(pending, error)) return false;
    pending_result.generation_after = pending.generation;
    *impl_ = std::move(pending);
    *result = pending_result;
    return true;
  } catch (const std::bad_alloc&) {
    *error = "full indexed catalog reconcile allocation failed";
    return false;
  }
}

bool ValidateIndexedCatalogSolveBinding(
    const MapperStaticCatalogStableTables& tables,
    const IndexedActiveSolveDescriptor& descriptor,
    const IndexedActiveSolveConfig& expected_config,
    std::string* error) noexcept {
  if (error == nullptr) return false;
  error->clear();
  const IndexedActiveSolveConfig& actual = descriptor.config;
  if (tables.abi_version != kMapperStaticCatalogAbiVersion ||
      descriptor.identity.abi_version != kIndexedActiveSolveAbiVersion ||
      descriptor.identity.catalog_abi_version != tables.abi_version ||
      tables.owner_epoch == 0 || tables.catalog_revision == 0 ||
      tables.catalog_generation == 0 ||
      descriptor.identity.owner_epoch != tables.owner_epoch ||
      descriptor.identity.catalog_revision != tables.catalog_revision ||
      descriptor.identity.catalog_generation != tables.catalog_generation ||
      descriptor.identity.solve_view_generation == 0 ||
      descriptor.identity.config_generation == 0 ||
      descriptor.identity.config_generation != actual.config_generation ||
      !actual.effective_config_resolved ||
      !expected_config.effective_config_resolved ||
      actual.arithmetic_precision != expected_config.arithmetic_precision ||
      actual.hessian_backend != expected_config.hessian_backend ||
      actual.schur_backend != expected_config.schur_backend ||
      actual.hot_kernel != expected_config.hot_kernel ||
      actual.execution_profile != expected_config.execution_profile ||
      actual.loss_mode != expected_config.loss_mode ||
      actual.loss_scale != expected_config.loss_scale) {
    *error = "indexed catalog solve identity or configuration mismatch";
    return false;
  }
  return true;
}

bool IndexedActiveSolveMaterializer::Materialize(
    const uint64_t owner_epoch,
    const MapperStaticProblemDataCatalog& catalog,
    const CudaSolveProblem& problem,
    const IndexedActiveSolveConfig& config,
    IndexedActiveSolveDescriptor* output,
    std::string* error) {
  if (output == nullptr || error == nullptr || owner_epoch == 0 ||
      owner_epoch != catalog.owner_epoch() || catalog.revision() == 0 ||
      catalog.generation() == 0 ||
      !config.effective_config_resolved ||
      config.config_generation == 0 ||
      (config.arithmetic_precision != CudaArithmeticPrecision::kFp64 &&
       config.arithmetic_precision !=
           CudaArithmeticPrecision::kFp32MixedStable) ||
      config.hessian_backend ==
          CudaHessianAssemblyBackend::kCompatibilityDefault ||
      config.schur_backend ==
          CudaSchurContributionBackend::kCompatibilityDefault ||
      config.hot_kernel == CudaHotKernelMode::kCompatibilityDefault ||
      config.loss_mode == CudaLossMode::kFromSnapshot ||
      !std::isfinite(config.loss_scale) || config.loss_scale <= 0.0) {
    if (error != nullptr) *error = "invalid indexed active solve request";
    return false;
  }
  error->clear();
  const auto start = std::chrono::steady_clock::now();
  try {
    IndexedActiveSolveDescriptor descriptor;
    descriptor.identity.owner_epoch = owner_epoch;
    descriptor.identity.catalog_revision = catalog.revision();
    descriptor.identity.catalog_generation = catalog.generation();
    if (next_view_generation_ == std::numeric_limits<uint64_t>::max()) {
      *error = "indexed active solve generation exhausted";
      return false;
    }
    descriptor.identity.solve_view_generation = ++next_view_generation_;
    if (descriptor.identity.solve_view_generation == 0) {
      descriptor.identity.solve_view_generation = ++next_view_generation_;
    }
    descriptor.identity.config_generation = config.config_generation;
    descriptor.config = config;
    descriptor.metadata = problem.metadata;
    descriptor.runtime.descriptor_calls = 1;
    descriptor.runtime.selection_passes = 1;
    descriptor.residual_block_count =
        problem.observations.size() + problem.lidar.size();
    descriptor.scalar_residual_count =
        2 * problem.observations.size() + problem.lidar.size();

    descriptor.cameras.reserve(problem.cameras.size());
    descriptor.images.reserve(problem.images.size());
    descriptor.points.reserve(problem.points.size());
    if (scratch_generation_ == std::numeric_limits<uint64_t>::max()) {
      std::fill(camera_slot_stamps_.begin(), camera_slot_stamps_.end(), 0);
      std::fill(image_slot_stamps_.begin(), image_slot_stamps_.end(), 0);
      std::fill(point_slot_stamps_.begin(), point_slot_stamps_.end(), 0);
      scratch_generation_ = 0;
    }
    const uint64_t scratch_generation = ++scratch_generation_;
    camera_slot_stamps_.resize(catalog.camera_slot_count(), 0);
    image_slot_stamps_.resize(catalog.image_slot_count(), 0);
    point_slot_stamps_.resize(catalog.point_slot_count(), 0);
    camera_state_by_slot_.resize(catalog.camera_slot_count());
    image_state_by_slot_.resize(catalog.image_slot_count());
    point_state_by_slot_.resize(catalog.point_slot_count());
    const auto camera_state_index = [&](const uint32_t slot) {
      return slot < camera_slot_stamps_.size() &&
                     camera_slot_stamps_[slot] == scratch_generation
                 ? camera_state_by_slot_[slot] : -1;
    };
    const auto image_state_index = [&](const uint32_t slot) {
      return slot < image_slot_stamps_.size() &&
                     image_slot_stamps_[slot] == scratch_generation
                 ? image_state_by_slot_[slot] : -1;
    };
    const auto point_state_index = [&](const uint32_t slot) {
      return slot < point_slot_stamps_.size() &&
                     point_slot_stamps_[slot] == scratch_generation
                 ? point_state_by_slot_[slot] : -1;
    };

    const auto dynamic_state_start = std::chrono::steady_clock::now();
    for (const CameraSnapshot& camera : problem.cameras) {
      const MapperCatalogCameraSlot* stable = catalog.FindCamera(camera.camera_id);
      if (stable == nullptr ||
          stable->header.slot >= camera_state_by_slot_.size() ||
          stable->model_id != camera.model_id || stable->width != camera.width ||
          stable->height != camera.height ||
          stable->parameter_count != camera.params.size()) {
        *error = "indexed camera state is inconsistent with catalog";
        return false;
      }
      IndexedCameraState state;
      state.catalog_slot = stable->header.slot;
      state.camera_id = camera.camera_id;
      state.constant = camera.constant ? 1 : 0;
      state.parameters = camera.params;
      camera_slot_stamps_[state.catalog_slot] = scratch_generation;
      camera_state_by_slot_[state.catalog_slot] =
          static_cast<int32_t>(descriptor.cameras.size());
      descriptor.cameras.push_back(std::move(state));
    }

    for (const ImageSnapshot& image : problem.images) {
      const MapperCatalogImageSlot* stable = catalog.FindImage(image.image_id);
      const MapperCatalogCameraSlot* camera = catalog.FindCamera(image.camera_id);
      if (stable == nullptr || camera == nullptr ||
          stable->header.slot >= image_state_by_slot_.size() ||
          stable->camera_slot != camera->header.slot ||
          camera_state_index(camera->header.slot) < 0) {
        *error = "indexed image state is inconsistent with catalog";
        return false;
      }
      IndexedImageState state;
      state.catalog_slot = stable->header.slot;
      state.image_id = image.image_id;
      state.camera_slot = camera->header.slot;
      state.selected = image.selected ? 1 : 0;
      state.pose_constant = image.pose_constant ? 1 : 0;
      state.has_pose_parameter_blocks =
          image.has_pose_parameter_blocks ? 1 : 0;
      state.constant_tvec_mask = image.constant_tvec_mask;
      state.quaternion = image.qvec;
      state.translation = image.tvec;
      image_slot_stamps_[state.catalog_slot] = scratch_generation;
      image_state_by_slot_[state.catalog_slot] =
          static_cast<int32_t>(descriptor.images.size());
      descriptor.images.push_back(state);
    }

    for (const PointSnapshot& point : problem.points) {
      const MapperCatalogPointSlot* stable = catalog.FindPoint(point.point3D_id);
      if (stable == nullptr ||
          stable->header.slot >= point_state_by_slot_.size()) {
        *error = "indexed point state is inconsistent with catalog";
        return false;
      }
      IndexedPointState state;
      state.catalog_slot = stable->header.slot;
      state.point3D_id = point.point3D_id;
      state.constant = point.constant ? 1 : 0;
      state.config_role = point.config_role;
      state.has_search_range = point.has_search_range ? 1 : 0;
      state.search_range = point.search_range;
      state.xyz = point.xyz;
      point_slot_stamps_[state.catalog_slot] = scratch_generation;
      point_state_by_slot_[state.catalog_slot] =
          static_cast<int32_t>(descriptor.points.size());
      descriptor.points.push_back(state);
    }
    descriptor.runtime.dynamic_state_wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - dynamic_state_start).count();

    uint32_t pose_count = 0;
    uint32_t point_count = 0;
    for (const ParameterBlockSnapshot& parameter :
         problem.parameter_blocks_source_order) {
      descriptor.ambient_parameter_count += parameter.ambient_size;
      if (!parameter.constant)
        descriptor.effective_parameter_count += parameter.tangent_size;
      if (parameter.kind == ParameterKind::kQuaternion &&
          !parameter.constant) {
        const MapperCatalogImageSlot* stable = catalog.FindImage(
            static_cast<uint32_t>(parameter.entity_id));
        if (stable == nullptr || image_state_index(stable->header.slot) < 0) {
          *error = "indexed parameter order references a missing image";
          return false;
        }
        IndexedImageState& state = descriptor.images[
            static_cast<size_t>(image_state_index(stable->header.slot))];
        if (state.pose_constant || !state.has_pose_parameter_blocks ||
            state.pose_index >= 0) {
          *error = "indexed variable pose parameter identity is invalid";
          return false;
        }
        state.pose_index = static_cast<int32_t>(pose_count++);
        state.schur_offset = descriptor.pose_dimension;
        state.pose_dimension = 3;
        for (int component = 0; component < 3; ++component) {
          if ((state.constant_tvec_mask & (1u << component)) == 0) {
            state.free_translation_indices[state.pose_dimension - 3] =
                component;
            ++state.pose_dimension;
          }
        }
        if (descriptor.pose_dimension >
            std::numeric_limits<uint32_t>::max() - state.pose_dimension) {
          *error = "indexed pose dimension exceeds uint32";
          return false;
        }
        descriptor.pose_dimension += state.pose_dimension;
      } else if (parameter.kind == ParameterKind::kPoint3D &&
                 !parameter.constant) {
        const MapperCatalogPointSlot* stable =
            catalog.FindPoint(parameter.entity_id);
        if (stable == nullptr || point_state_index(stable->header.slot) < 0) {
          *error = "indexed parameter order references a missing point";
          return false;
        }
        IndexedPointState& state = descriptor.points[
            static_cast<size_t>(point_state_index(stable->header.slot))];
        if (state.constant || state.point_index >= 0) {
          *error = "indexed variable point parameter identity is invalid";
          return false;
        }
        state.point_index = static_cast<int32_t>(point_count++);
      }
    }

    const size_t expected_poses = static_cast<size_t>(std::count_if(
        problem.images.begin(), problem.images.end(),
        [](const ImageSnapshot& value) { return !value.pose_constant; }));
    const size_t expected_points = static_cast<size_t>(std::count_if(
        problem.points.begin(), problem.points.end(),
        [](const PointSnapshot& value) { return !value.constant; }));
    if (pose_count != expected_poses || point_count != expected_points) {
      *error = "indexed parameter order does not cover variable entities";
      return false;
    }

    descriptor.visual.reserve(problem.observations.size());
    descriptor.lidar.reserve(problem.lidar.size());
    for (const ObservationSnapshot& value : problem.observations) {
      const MapperCatalogObservationSlot* stable =
          catalog.FindObservation(value.image_id, value.point2D_idx);
      const MapperCatalogPointSlot* expected_point =
          catalog.FindPoint(value.point3D_id);
      if (stable == nullptr || expected_point == nullptr ||
          point_state_index(stable->point_slot) < 0 ||
          image_state_index(stable->image_slot) < 0 ||
          camera_state_index(stable->camera_slot) < 0 ||
          expected_point->header.slot != stable->point_slot) {
        *error = "indexed visual binding is inconsistent with catalog";
        return false;
      }
      IndexedVisualResidual residual;
      residual.source_index = value.source_index;
      residual.observation_slot = stable->header.slot;
      residual.image_slot = stable->image_slot;
      residual.point_slot = stable->point_slot;
      residual.camera_slot = stable->camera_slot;
      residual.pose_index = descriptor.images[
          static_cast<size_t>(image_state_index(stable->image_slot))]
                                .pose_index;
      residual.point_index = descriptor.points[
          static_cast<size_t>(point_state_index(stable->point_slot))]
                                 .point_index;
      descriptor.visual.push_back(residual);
    }
    for (const LidarSnapshot& value : problem.lidar) {
      const MapperCatalogPointSlot* stable = catalog.FindPoint(value.point3D_id);
      if (stable == nullptr || point_state_index(stable->header.slot) < 0) {
        *error = "indexed LiDAR binding references a missing point";
        return false;
      }
      IndexedLidarConstraint residual;
      residual.source_index = value.source_index;
      residual.point3D_id = value.point3D_id;
      residual.point_slot = stable->header.slot;
      residual.point_index = descriptor.points[
          static_cast<size_t>(point_state_index(stable->header.slot))]
                                 .point_index;
      residual.lidar_type = value.lidar_type;
      residual.has_search_range = value.has_search_range ? 1 : 0;
      residual.search_range = value.search_range;
      residual.weight = value.weight;
      residual.lidar_xyz = value.lidar_xyz;
      residual.plane = value.plane;
      descriptor.lidar.push_back(residual);
    }
    if (descriptor.visual.size() != problem.observations.size() ||
        descriptor.lidar.size() != problem.lidar.size()) {
      *error = "indexed residual coverage is incomplete";
      return false;
    }

    descriptor.runtime.selected_visual_residuals = descriptor.visual.size();
    descriptor.runtime.selected_lidar_residuals = descriptor.lidar.size();
    descriptor.runtime.selected_cameras = descriptor.cameras.size();
    descriptor.runtime.selected_images = descriptor.images.size();
    descriptor.runtime.selected_points = descriptor.points.size();
    uint64_t bytes = sizeof(descriptor);
    bytes += VectorBytes(descriptor.cameras) +
             VectorBytes(descriptor.images) + VectorBytes(descriptor.points) +
             VectorBytes(descriptor.visual) + VectorBytes(descriptor.lidar);
    for (const IndexedCameraState& camera : descriptor.cameras) {
      bytes += VectorBytes(camera.parameters);
    }
    descriptor.runtime.descriptor_bytes = bytes;
    descriptor.runtime.wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    *output = std::move(descriptor);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "indexed active solve materialization allocation failed";
    return false;
  }
}

bool DeviceCatalogPhysicalIdentity::valid() const noexcept {
  return device_ordinal >= 0 && context_incarnation != 0 &&
         catalog_abi_version == kMapperStaticCatalogAbiVersion &&
         owner_epoch != 0 && catalog_generation != 0 && arena_generation != 0;
}

bool DeviceCatalogPhysicalIdentity::operator==(
    const DeviceCatalogPhysicalIdentity& other) const noexcept {
  return device_ordinal == other.device_ordinal &&
         context_incarnation == other.context_incarnation &&
         catalog_abi_version == other.catalog_abi_version &&
         owner_epoch == other.owner_epoch &&
         catalog_revision == other.catalog_revision &&
         catalog_generation == other.catalog_generation &&
         arena_generation == other.arena_generation;
}

bool PlanPersistentDeviceCatalogLayout(
    const MapperStaticCatalogStableTables& tables,
    const DeviceCatalogLayoutSpec& spec,
    DeviceCatalogLayout* output,
    std::string* error) {
  if (output == nullptr || error == nullptr || tables.owner_epoch == 0 ||
      tables.catalog_generation == 0 || spec.alignment < 256 ||
      (spec.alignment & (spec.alignment - 1)) != 0) {
    if (error != nullptr) *error = "invalid persistent device catalog layout";
    return false;
  }
  DeviceCatalogLayout layout;
  layout.alignment = spec.alignment;
  uint64_t cursor = 0;
  const uint64_t nonzero_base =
      std::max<uint64_t>(spec.base_offset, spec.alignment);
  if (!CheckedAlign(nonzero_base, spec.alignment, &cursor)) {
    *error = "persistent device catalog base alignment overflow";
    return false;
  }
  layout.begin = cursor;
  auto append = [&](const uint64_t elements,
                    const uint64_t stride,
                    DeviceCatalogSlice* slice) {
    uint64_t bytes = 0;
    uint64_t end = 0;
    if (!CheckedAlign(cursor, spec.alignment, &cursor) ||
        !CheckedMultiply(elements, stride, &bytes) ||
        !CheckedAdd(cursor, bytes, &end)) {
      return false;
    }
    *slice = {cursor, bytes, elements};
    cursor = end;
    return true;
  };
  if (!append(tables.cameras.size(), sizeof(MapperCatalogCameraSlot),
              &layout.cameras) ||
      !append(tables.images.size(), sizeof(MapperCatalogImageSlot),
              &layout.images) ||
      !append(tables.points.size(), sizeof(MapperCatalogPointSlot),
              &layout.points) ||
      !append(tables.observations.size(),
              sizeof(MapperCatalogObservationSlot), &layout.observations) ||
      !append(tables.image_incidences.size(),
              sizeof(MapperCatalogIncidenceSlot), &layout.image_incidences) ||
      !append(tables.point_incidences.size(),
              sizeof(MapperCatalogIncidenceSlot), &layout.point_incidences) ||
      !CheckedAlign(cursor, spec.alignment, &layout.end) ||
      (spec.arena_capacity != 0 && layout.end > spec.arena_capacity)) {
    *error = "persistent device catalog layout overflow or out of bounds";
    return false;
  }
  *output = layout;
  return true;
}

bool PersistentDeviceCatalogState::Publish(
    const DeviceCatalogPhysicalIdentity& identity,
    const DeviceCatalogLayout& layout,
    std::string* error) {
  if (error == nullptr || !identity.valid() || layout.alignment < 256 ||
      layout.begin > layout.end || layout.end == 0) {
    if (error != nullptr) *error = "invalid persistent device catalog publish";
    return false;
  }
  const DeviceCatalogSlice* slices[] = {
      &layout.cameras,          &layout.images, &layout.points,
      &layout.observations,     &layout.image_incidences,
      &layout.point_incidences};
  uint64_t previous_end = layout.begin;
  for (const DeviceCatalogSlice* slice : slices) {
    if ((slice->offset % layout.alignment) != 0 ||
        slice->offset < previous_end ||
        slice->bytes > layout.end - slice->offset) {
      *error = "persistent device catalog slices overlap or misalign";
      return false;
    }
    previous_end = slice->offset + slice->bytes;
  }
  identity_ = identity;
  layout_ = layout;
  valid_ = true;
  return true;
}

bool PersistentDeviceCatalogState::CanReuse(
    const DeviceCatalogPhysicalIdentity& identity) const noexcept {
  return valid_ && identity.valid() && identity_ == identity;
}

bool PersistentDeviceCatalogState::CanReuseStorage(
    const DeviceCatalogPhysicalIdentity& identity) const noexcept {
  return valid_ && identity.valid() &&
         identity_.device_ordinal == identity.device_ordinal &&
         identity_.context_incarnation == identity.context_incarnation &&
         identity_.catalog_abi_version == identity.catalog_abi_version &&
         identity_.owner_epoch == identity.owner_epoch &&
         identity_.catalog_generation == identity.catalog_generation &&
         identity_.arena_generation == identity.arena_generation;
}

void PersistentDeviceCatalogState::Invalidate() noexcept {
  valid_ = false;
  identity_ = DeviceCatalogPhysicalIdentity();
  layout_ = DeviceCatalogLayout();
}

bool PersistentDeviceCatalogState::valid() const noexcept { return valid_; }

const DeviceCatalogPhysicalIdentity&
PersistentDeviceCatalogState::identity() const noexcept {
  return identity_;
}

const DeviceCatalogLayout& PersistentDeviceCatalogState::layout() const noexcept {
  return layout_;
}

}  // namespace gpu_ba
}  // namespace colmap
