#include "gpu_ba/host_ba_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {

namespace {

static_assert(std::is_standard_layout<HostBaCameraSlot>::value &&
                  std::is_trivially_copyable<HostBaCameraSlot>::value,
              "camera slot must be upload-safe");
static_assert(std::is_standard_layout<HostBaImageSlot>::value &&
                  std::is_trivially_copyable<HostBaImageSlot>::value,
              "image slot must be upload-safe");
static_assert(std::is_standard_layout<HostBaPointSlot>::value &&
                  std::is_trivially_copyable<HostBaPointSlot>::value,
              "point slot must be upload-safe");
static_assert(std::is_standard_layout<HostBaObservationSlot>::value &&
                  std::is_trivially_copyable<HostBaObservationSlot>::value,
              "observation slot must be upload-safe");

uint64_t ObservationIdentity(const uint32_t image_id,
                             const uint32_t point2D_idx) noexcept {
  return (static_cast<uint64_t>(image_id) << 32) |
         static_cast<uint64_t>(point2D_idx);
}

bool SameDouble(const double lhs, const double rhs) noexcept {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits == rhs_bits;
}

bool SameXy(const std::array<double, 2>& lhs,
            const std::array<double, 2>& rhs) noexcept {
  return SameDouble(lhs[0], rhs[0]) && SameDouble(lhs[1], rhs[1]);
}

template <typename T>
BaArrayView<T> MakeView(const std::vector<T>& values) noexcept {
  BaArrayView<T> view;
  view.data = values.empty() ? nullptr : values.data();
  view.size = values.size();
  return view;
}

template <typename T>
bool FitsUint32(const T value) noexcept {
  return value <= static_cast<T>(std::numeric_limits<uint32_t>::max());
}

template <typename T>
uint64_t VectorBytes(const std::vector<T>& values) noexcept {
  return static_cast<uint64_t>(values.size()) * sizeof(T);
}

template <typename T>
bool IsFiniteArray(const T& values) noexcept {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool SetError(const char* message, std::string* error) {
  if (error != nullptr) *error = message;
  return false;
}

}  // namespace

struct HostBaGraphPublication {
  uint32_t abi_version = kHostBaGraphAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t topology_revision = 0;
  uint64_t generation = 0;
  std::vector<HostBaCameraSlot> cameras;
  std::vector<HostBaImageSlot> images;
  std::vector<HostBaPointSlot> points;
  std::vector<HostBaObservationSlot> observations;
  std::vector<uint32_t> image_observation_slots;
  std::vector<uint32_t> point_observation_slots;
  std::unordered_map<uint32_t, uint32_t> camera_by_id;
  std::unordered_map<uint32_t, uint32_t> image_by_id;
  std::unordered_map<uint64_t, uint32_t> point_by_id;
  std::unordered_map<uint64_t, uint32_t> observation_by_identity;
  uint64_t resident_bytes = 0;
};

namespace {

uint64_t EstimateResidentBytes(const HostBaGraphPublication& graph) noexcept {
  uint64_t bytes = sizeof(graph);
  bytes += VectorBytes(graph.cameras);
  bytes += VectorBytes(graph.images);
  bytes += VectorBytes(graph.points);
  bytes += VectorBytes(graph.observations);
  bytes += VectorBytes(graph.image_observation_slots);
  bytes += VectorBytes(graph.point_observation_slots);
  bytes += static_cast<uint64_t>(graph.camera_by_id.size()) *
           (sizeof(uint32_t) * 2 + sizeof(void*) * 2);
  bytes += static_cast<uint64_t>(graph.image_by_id.size()) *
           (sizeof(uint32_t) * 2 + sizeof(void*) * 2);
  bytes += static_cast<uint64_t>(graph.point_by_id.size()) *
           (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*) * 2);
  bytes += static_cast<uint64_t>(graph.observation_by_identity.size()) *
           (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(void*) * 2);
  return bytes;
}

bool ValidateGraphReferences(const HostBaGraphPublication& graph,
                             std::string* error) {
  if (graph.owner_epoch == 0 || graph.topology_revision == 0 ||
      graph.generation == 0) {
    return SetError("host BA graph has invalid publication identity", error);
  }
  for (const HostBaImageSlot& image : graph.images) {
    if (!image.header.alive) continue;
    if (image.camera_slot >= graph.cameras.size() ||
        !graph.cameras[image.camera_slot].header.alive) {
      return SetError("host BA graph image references a dead camera", error);
    }
  }
  for (const HostBaObservationSlot& observation : graph.observations) {
    if (!observation.header.alive) continue;
    if (observation.image_slot >= graph.images.size() ||
        observation.point_slot >= graph.points.size() ||
        observation.camera_slot >= graph.cameras.size() ||
        !graph.images[observation.image_slot].header.alive ||
        !graph.points[observation.point_slot].header.alive ||
        !graph.cameras[observation.camera_slot].header.alive ||
        graph.images[observation.image_slot].camera_slot !=
            observation.camera_slot) {
      return SetError("host BA graph contains a dangling observation", error);
    }
    if (!IsFiniteArray(observation.xy)) {
      return SetError("host BA graph observation is non-finite", error);
    }
  }
  return true;
}

bool RebuildAdjacency(HostBaGraphPublication* graph, std::string* error) {
  if (graph == nullptr) return SetError("host BA graph output is null", error);
  if (!ValidateGraphReferences(*graph, error)) return false;

  uint64_t alive_observations = 0;
  for (HostBaImageSlot& image : graph->images) {
    image.adjacency_offset = 0;
    image.adjacency_count = 0;
  }
  for (HostBaPointSlot& point : graph->points) {
    point.adjacency_offset = 0;
    point.adjacency_count = 0;
    point.track_length = 0;
  }
  for (const HostBaObservationSlot& observation : graph->observations) {
    if (!observation.header.alive) continue;
    ++alive_observations;
    HostBaImageSlot& image = graph->images[observation.image_slot];
    HostBaPointSlot& point = graph->points[observation.point_slot];
    if (image.adjacency_count == std::numeric_limits<uint32_t>::max() ||
        point.adjacency_count == std::numeric_limits<uint32_t>::max() ||
        point.track_length == std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph adjacency count overflow", error);
    }
    ++image.adjacency_count;
    ++point.adjacency_count;
    ++point.track_length;
  }
  if (!FitsUint32(alive_observations)) {
    return SetError("host BA graph adjacency exceeds uint32", error);
  }

  uint64_t image_offset = 0;
  for (HostBaImageSlot& image : graph->images) {
    if (!FitsUint32(image_offset)) {
      return SetError("host BA graph image adjacency offset overflow", error);
    }
    image.adjacency_offset = static_cast<uint32_t>(image_offset);
    image_offset += image.adjacency_count;
  }
  uint64_t point_offset = 0;
  for (HostBaPointSlot& point : graph->points) {
    if (!FitsUint32(point_offset)) {
      return SetError("host BA graph point adjacency offset overflow", error);
    }
    point.adjacency_offset = static_cast<uint32_t>(point_offset);
    point_offset += point.adjacency_count;
  }
  if (image_offset != alive_observations ||
      point_offset != alive_observations) {
    return SetError("host BA graph adjacency coverage mismatch", error);
  }

  graph->image_observation_slots.assign(alive_observations,
                                         kBaGraphInvalidSlot);
  graph->point_observation_slots.assign(alive_observations,
                                         kBaGraphInvalidSlot);
  std::vector<uint32_t> image_write(graph->images.size(), 0);
  std::vector<uint32_t> point_write(graph->points.size(), 0);
  for (const HostBaObservationSlot& observation : graph->observations) {
    if (!observation.header.alive) continue;
    const HostBaImageSlot& image = graph->images[observation.image_slot];
    const HostBaPointSlot& point = graph->points[observation.point_slot];
    const uint64_t image_index =
        static_cast<uint64_t>(image.adjacency_offset) +
        image_write[observation.image_slot]++;
    const uint64_t point_index =
        static_cast<uint64_t>(point.adjacency_offset) +
        point_write[observation.point_slot]++;
    graph->image_observation_slots[image_index] = observation.header.slot;
    graph->point_observation_slots[point_index] = observation.header.slot;
  }
  graph->resident_bytes = EstimateResidentBytes(*graph);
  return true;
}

bool SameHeaderWithoutGeneration(const HostBaGraphSlotHeader& lhs,
                                 const HostBaGraphSlotHeader& rhs) noexcept {
  return lhs.slot == rhs.slot && lhs.alive == rhs.alive;
}

bool SameGraphSemantics(const HostBaGraphPublication& lhs,
                        const HostBaGraphPublication& rhs) noexcept {
  if (lhs.cameras.size() != rhs.cameras.size() ||
      lhs.images.size() != rhs.images.size() ||
      lhs.points.size() != rhs.points.size() ||
      lhs.observations.size() != rhs.observations.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.cameras.size(); ++i) {
    const auto& a = lhs.cameras[i];
    const auto& b = rhs.cameras[i];
    if (!SameHeaderWithoutGeneration(a.header, b.header) ||
        a.camera_id != b.camera_id || a.model_id != b.model_id ||
        a.width != b.width || a.height != b.height ||
        a.parameter_count != b.parameter_count) {
      return false;
    }
  }
  for (size_t i = 0; i < lhs.images.size(); ++i) {
    const auto& a = lhs.images[i];
    const auto& b = rhs.images[i];
    if (!SameHeaderWithoutGeneration(a.header, b.header) ||
        a.image_id != b.image_id || a.camera_slot != b.camera_slot ||
        a.registered != b.registered ||
        a.lifetime_incidence_count != b.lifetime_incidence_count) {
      return false;
    }
  }
  for (size_t i = 0; i < lhs.points.size(); ++i) {
    const auto& a = lhs.points[i];
    const auto& b = rhs.points[i];
    if (!SameHeaderWithoutGeneration(a.header, b.header) ||
        a.point3D_id != b.point3D_id ||
        a.lifetime_incidence_count != b.lifetime_incidence_count) {
      return false;
    }
  }
  for (size_t i = 0; i < lhs.observations.size(); ++i) {
    const auto& a = lhs.observations[i];
    const auto& b = rhs.observations[i];
    if (!SameHeaderWithoutGeneration(a.header, b.header) ||
        a.image_slot != b.image_slot || a.point_slot != b.point_slot ||
        a.camera_slot != b.camera_slot ||
        a.point2D_idx != b.point2D_idx ||
        a.source_identity != b.source_identity || !SameXy(a.xy, b.xy)) {
      return false;
    }
  }
  return true;
}

bool AppendCamera(HostBaGraphPublication* graph,
                  const HostBaCameraRecord& input,
                  const uint64_t generation,
                  HostBaGraphUpdateResult* result,
                  std::string* error) {
  if (graph->cameras.size() >= std::numeric_limits<uint32_t>::max()) {
    return SetError("host BA graph camera slot capacity exhausted", error);
  }
  HostBaCameraSlot value;
  value.header.slot = static_cast<uint32_t>(graph->cameras.size());
  value.header.generation = generation;
  value.header.alive = 1;
  value.camera_id = input.camera_id;
  value.model_id = input.model_id;
  value.width = input.width;
  value.height = input.height;
  value.parameter_count = input.parameter_count;
  graph->camera_by_id.emplace(value.camera_id, value.header.slot);
  graph->cameras.push_back(value);
  ++result->appended_slots;
  return true;
}

bool UpsertCamera(HostBaGraphPublication* graph,
                  const HostBaCameraRecord& input,
                  const uint64_t generation,
                  HostBaGraphUpdateResult* result,
                  std::string* error) {
  const auto found = graph->camera_by_id.find(input.camera_id);
  if (found == graph->camera_by_id.end()) {
    return AppendCamera(graph, input, generation, result, error);
  }
  HostBaCameraSlot& value = graph->cameras[found->second];
  if (value.header.alive && value.model_id == input.model_id &&
      value.width == input.width && value.height == input.height &&
      value.parameter_count == input.parameter_count) {
    return true;
  }
  value.header.alive = 1;
  value.header.generation = generation;
  value.model_id = input.model_id;
  value.width = input.width;
  value.height = input.height;
  value.parameter_count = input.parameter_count;
  ++result->updated_slots;
  return true;
}

bool UpsertImage(HostBaGraphPublication* graph,
                 const HostBaImageRecord& input,
                 const uint64_t generation,
                 HostBaGraphUpdateResult* result,
                 std::string* error) {
  const auto camera = graph->camera_by_id.find(input.camera_id);
  if (camera == graph->camera_by_id.end() ||
      !graph->cameras[camera->second].header.alive) {
    return SetError("host BA graph image upsert references a dead camera",
                    error);
  }
  const auto found = graph->image_by_id.find(input.image_id);
  if (found == graph->image_by_id.end()) {
    if (graph->images.size() >= std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph image slot capacity exhausted", error);
    }
    HostBaImageSlot value;
    value.header.slot = static_cast<uint32_t>(graph->images.size());
    value.header.generation = generation;
    value.header.alive = 1;
    value.image_id = input.image_id;
    value.camera_slot = camera->second;
    value.registered = input.registered ? 1 : 0;
    graph->image_by_id.emplace(value.image_id, value.header.slot);
    graph->images.push_back(value);
    ++result->appended_slots;
    return true;
  }
  HostBaImageSlot& value = graph->images[found->second];
  if (value.header.alive && value.camera_slot == camera->second &&
      value.registered == (input.registered ? 1 : 0)) {
    return true;
  }
  const bool camera_changed = value.camera_slot != camera->second;
  value.header.alive = 1;
  value.header.generation = generation;
  value.camera_slot = camera->second;
  value.registered = input.registered ? 1 : 0;
  if (camera_changed) {
    for (HostBaObservationSlot& observation : graph->observations) {
      if (!observation.header.alive ||
          observation.image_slot != value.header.slot) {
        continue;
      }
      observation.camera_slot = camera->second;
      observation.header.generation = generation;
      ++result->updated_slots;
    }
  }
  ++result->updated_slots;
  return true;
}

bool UpsertPoint(HostBaGraphPublication* graph,
                 const HostBaPointRecord& input,
                 const uint64_t generation,
                 HostBaGraphUpdateResult* result,
                 std::string* error) {
  const auto found = graph->point_by_id.find(input.point3D_id);
  if (found == graph->point_by_id.end()) {
    if (graph->points.size() >= std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph point slot capacity exhausted", error);
    }
    HostBaPointSlot value;
    value.header.slot = static_cast<uint32_t>(graph->points.size());
    value.header.generation = generation;
    value.header.alive = 1;
    value.point3D_id = input.point3D_id;
    graph->point_by_id.emplace(value.point3D_id, value.header.slot);
    graph->points.push_back(value);
    ++result->appended_slots;
    return true;
  }
  HostBaPointSlot& value = graph->points[found->second];
  if (value.header.alive) return true;
  value.header.alive = 1;
  value.header.generation = generation;
  ++result->updated_slots;
  return true;
}

bool UpsertObservation(HostBaGraphPublication* graph,
                       const HostBaObservationRecord& input,
                       const uint64_t generation,
                       HostBaGraphUpdateResult* result,
                       std::string* error) {
  const uint64_t identity =
      ObservationIdentity(input.image_id, input.point2D_idx);
  const auto image = graph->image_by_id.find(input.image_id);
  const auto point = graph->point_by_id.find(input.point3D_id);
  if (image == graph->image_by_id.end() || point == graph->point_by_id.end() ||
      !graph->images[image->second].header.alive ||
      !graph->points[point->second].header.alive) {
    return SetError("host BA graph observation upsert references a dead entity",
                    error);
  }
  if (!IsFiniteArray(input.xy)) {
    return SetError("host BA graph observation upsert is non-finite", error);
  }
  const uint32_t camera_slot = graph->images[image->second].camera_slot;
  const auto found = graph->observation_by_identity.find(identity);
  if (found == graph->observation_by_identity.end()) {
    if (graph->observations.size() >= std::numeric_limits<uint32_t>::max() ||
        graph->images[image->second].lifetime_incidence_count ==
            std::numeric_limits<uint32_t>::max() ||
        graph->points[point->second].lifetime_incidence_count ==
            std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph observation capacity exhausted", error);
    }
    HostBaObservationSlot value;
    value.header.slot = static_cast<uint32_t>(graph->observations.size());
    value.header.generation = generation;
    value.header.alive = 1;
    value.image_slot = image->second;
    value.point_slot = point->second;
    value.camera_slot = camera_slot;
    value.point2D_idx = input.point2D_idx;
    value.source_identity = identity;
    value.xy = input.xy;
    graph->observation_by_identity.emplace(identity, value.header.slot);
    graph->observations.push_back(value);
    ++graph->images[image->second].lifetime_incidence_count;
    ++graph->points[point->second].lifetime_incidence_count;
    ++result->appended_slots;
    return true;
  }
  HostBaObservationSlot& value = graph->observations[found->second];
  if (value.header.alive && value.image_slot == image->second &&
      value.point_slot == point->second && value.camera_slot == camera_slot &&
      SameXy(value.xy, input.xy)) {
    return true;
  }
  const bool association_changed = value.point_slot != point->second;
  if (association_changed) {
    if (graph->points[point->second].lifetime_incidence_count ==
        std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph point incidence overflow", error);
    }
    ++graph->points[point->second].lifetime_incidence_count;
  }
  value.header.alive = 1;
  value.header.generation = generation;
  value.image_slot = image->second;
  value.point_slot = point->second;
  value.camera_slot = camera_slot;
  value.xy = input.xy;
  ++result->updated_slots;
  return true;
}

bool TombstoneObservation(HostBaGraphPublication* graph,
                          const uint64_t identity,
                          const uint64_t generation,
                          HostBaGraphUpdateResult* result,
                          const bool cascaded) {
  const auto found = graph->observation_by_identity.find(identity);
  if (found == graph->observation_by_identity.end()) return true;
  HostBaObservationSlot& value = graph->observations[found->second];
  if (!value.header.alive) return true;
  value.header.alive = 0;
  value.header.generation = generation;
  ++result->tombstoned_slots;
  if (cascaded) ++result->cascaded_observation_tombstones;
  return true;
}

template <typename Predicate>
void TombstoneMatchingObservations(HostBaGraphPublication* graph,
                                   const uint64_t generation,
                                   HostBaGraphUpdateResult* result,
                                   Predicate predicate) {
  for (HostBaObservationSlot& observation : graph->observations) {
    if (!observation.header.alive || !predicate(observation)) continue;
    observation.header.alive = 0;
    observation.header.generation = generation;
    ++result->tombstoned_slots;
    ++result->cascaded_observation_tombstones;
  }
}

template <typename T>
void GrowScratch(std::vector<T>* values, const size_t size, const T& value) {
  if (values->size() < size) values->resize(size, value);
}

uint32_t StateSlot(const DenseCameraState& state) noexcept {
  return state.camera_slot;
}
uint32_t StateSlot(const DenseImageState& state) noexcept {
  return state.image_slot;
}
uint32_t StateSlot(const DensePointState& state) noexcept {
  return state.point_slot;
}

template <typename State>
bool CheckUniqueStateSlots(const std::vector<State>& states,
                           const size_t slot_count,
                           std::vector<uint64_t>* stamps,
                           std::vector<int32_t>* indices,
                           const uint64_t generation,
                           const uint64_t parent_state_generation,
                           const char* label,
                           std::string* error) {
  GrowScratch(stamps, slot_count, uint64_t{0});
  GrowScratch(indices, slot_count, int32_t{-1});
  for (size_t i = 0; i < states.size(); ++i) {
    const uint32_t slot = StateSlot(states[i]);
    if (slot >= slot_count || (*stamps)[slot] == generation ||
        states[i].state_generation != parent_state_generation) {
      if (error != nullptr) *error = std::string(label) + " state slot invalid";
      return false;
    }
    (*stamps)[slot] = generation;
    (*indices)[slot] = static_cast<int32_t>(i);
  }
  return true;
}

}  // namespace

CatalogReadLease::CatalogReadLease(
    std::shared_ptr<const HostBaGraphPublication> publication)
    : publication_(std::move(publication)) {}

bool CatalogReadLease::valid() const noexcept { return publication_ != nullptr; }
uint32_t CatalogReadLease::abi_version() const noexcept {
  return publication_ == nullptr ? 0 : publication_->abi_version;
}
uint64_t CatalogReadLease::owner_epoch() const noexcept {
  return publication_ == nullptr ? 0 : publication_->owner_epoch;
}
uint64_t CatalogReadLease::topology_revision() const noexcept {
  return publication_ == nullptr ? 0 : publication_->topology_revision;
}
uint64_t CatalogReadLease::generation() const noexcept {
  return publication_ == nullptr ? 0 : publication_->generation;
}
uint64_t CatalogReadLease::resident_bytes() const noexcept {
  return publication_ == nullptr ? 0 : publication_->resident_bytes;
}
BaArrayView<HostBaCameraSlot> CatalogReadLease::cameras() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaCameraSlot>()
                                 : MakeView(publication_->cameras);
}
BaArrayView<HostBaImageSlot> CatalogReadLease::images() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaImageSlot>()
                                 : MakeView(publication_->images);
}
BaArrayView<HostBaPointSlot> CatalogReadLease::points() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaPointSlot>()
                                 : MakeView(publication_->points);
}
BaArrayView<HostBaObservationSlot>
CatalogReadLease::observations() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaObservationSlot>()
                                 : MakeView(publication_->observations);
}
BaArrayView<uint32_t>
CatalogReadLease::image_observation_slots() const noexcept {
  return publication_ == nullptr ? BaArrayView<uint32_t>()
                                 : MakeView(publication_->image_observation_slots);
}
BaArrayView<uint32_t>
CatalogReadLease::point_observation_slots() const noexcept {
  return publication_ == nullptr ? BaArrayView<uint32_t>()
                                 : MakeView(publication_->point_observation_slots);
}
const HostBaCameraSlot* CatalogReadLease::FindCameraById(
    const uint32_t id) const noexcept {
  if (publication_ == nullptr) return nullptr;
  const auto found = publication_->camera_by_id.find(id);
  return found == publication_->camera_by_id.end()
             ? nullptr
             : &publication_->cameras[found->second];
}
const HostBaImageSlot* CatalogReadLease::FindImageById(
    const uint32_t id) const noexcept {
  if (publication_ == nullptr) return nullptr;
  const auto found = publication_->image_by_id.find(id);
  return found == publication_->image_by_id.end()
             ? nullptr
             : &publication_->images[found->second];
}
const HostBaPointSlot* CatalogReadLease::FindPointById(
    const uint64_t id) const noexcept {
  if (publication_ == nullptr) return nullptr;
  const auto found = publication_->point_by_id.find(id);
  return found == publication_->point_by_id.end()
             ? nullptr
             : &publication_->points[found->second];
}
const HostBaObservationSlot* CatalogReadLease::FindObservation(
    const uint32_t image_id, const uint32_t point2D_idx) const noexcept {
  if (publication_ == nullptr) return nullptr;
  const auto found = publication_->observation_by_identity.find(
      ObservationIdentity(image_id, point2D_idx));
  return found == publication_->observation_by_identity.end()
             ? nullptr
             : &publication_->observations[found->second];
}

HostBaGraphStore::HostBaGraphStore(const uint64_t owner_epoch)
    : owner_epoch_(owner_epoch) {}
HostBaGraphStore::~HostBaGraphStore() = default;
uint64_t HostBaGraphStore::owner_epoch() const noexcept { return owner_epoch_; }
uint64_t HostBaGraphStore::topology_revision() const noexcept {
  return publication_ == nullptr ? 0 : publication_->topology_revision;
}
uint64_t HostBaGraphStore::generation() const noexcept {
  return publication_ == nullptr ? 0 : publication_->generation;
}
bool HostBaGraphStore::empty() const noexcept { return publication_ == nullptr; }

bool HostBaGraphStore::ColdBuild(const HostBaGraphColdInput& input,
                                 HostBaGraphUpdateResult* result,
                                 std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = HostBaGraphUpdateResult();
  error->clear();
  if (owner_epoch_ == 0 || input.owner_epoch != owner_epoch_ ||
      input.topology_revision == 0) {
    return SetError("host BA graph cold-build identity mismatch", error);
  }
  if (publication_ != nullptr &&
      input.topology_revision <= publication_->topology_revision) {
    return SetError("host BA graph full rebuild revision did not advance",
                    error);
  }
  if (next_generation_ == std::numeric_limits<uint64_t>::max()) {
    return SetError("host BA graph generation exhausted", error);
  }

  try {
    result->full_rebuild = publication_ != nullptr;
    result->revision_before =
        publication_ == nullptr ? 0 : publication_->topology_revision;
    result->generation_before =
        publication_ == nullptr ? 0 : publication_->generation;
    result->transaction_copy_bytes = 0;
    std::shared_ptr<HostBaGraphPublication> pending =
        std::make_shared<HostBaGraphPublication>();
    pending->owner_epoch = owner_epoch_;
    pending->topology_revision = input.topology_revision;
    pending->generation = next_generation_ + 1;
    pending->cameras.reserve(input.cameras.size());
    pending->images.reserve(input.images.size());
    pending->points.reserve(input.points.size());
    pending->observations.reserve(input.observations.size());
    pending->camera_by_id.reserve(input.cameras.size());
    pending->image_by_id.reserve(input.images.size());
    pending->point_by_id.reserve(input.points.size());
    pending->observation_by_identity.reserve(input.observations.size());

    for (const HostBaCameraRecord& camera : input.cameras) {
      if (pending->camera_by_id.count(camera.camera_id) != 0 ||
          !UpsertCamera(pending.get(), camera, pending->generation, result,
                        error)) {
        if (error->empty()) *error = "host BA graph duplicate camera";
        return false;
      }
    }
    for (const HostBaImageRecord& image : input.images) {
      if (pending->image_by_id.count(image.image_id) != 0 ||
          !UpsertImage(pending.get(), image, pending->generation, result,
                       error)) {
        if (error->empty()) *error = "host BA graph duplicate image";
        return false;
      }
    }
    for (const HostBaPointRecord& point : input.points) {
      if (pending->point_by_id.count(point.point3D_id) != 0 ||
          !UpsertPoint(pending.get(), point, pending->generation, result,
                       error)) {
        if (error->empty()) *error = "host BA graph duplicate point";
        return false;
      }
    }
    for (const HostBaObservationRecord& observation : input.observations) {
      const uint64_t identity = ObservationIdentity(
          observation.image_id, observation.point2D_idx);
      if (pending->observation_by_identity.count(identity) != 0 ||
          !UpsertObservation(pending.get(), observation, pending->generation,
                             result, error)) {
        if (error->empty()) *error = "host BA graph duplicate observation";
        return false;
      }
    }
    if (!RebuildAdjacency(pending.get(), error)) return false;
    result->revision_after = input.topology_revision;
    result->generation_after = pending->generation;
    result->published = true;
    publication_ = std::move(pending);
    next_generation_ = publication_->generation;
    return true;
  } catch (const std::bad_alloc&) {
    return SetError("host BA graph cold-build allocation failed", error);
  }
}

bool HostBaGraphStore::ApplyCoalescedMutation(
    const CoalescedBaGraphMutation& mutation,
    HostBaGraphUpdateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = HostBaGraphUpdateResult();
  error->clear();
  if (publication_ == nullptr) {
    return SetError("host BA graph mutation requires a cold publication", error);
  }
  result->revision_before = publication_->topology_revision;
  result->generation_before = publication_->generation;
  if (mutation.owner_epoch != owner_epoch_ ||
      mutation.revision_before != publication_->topology_revision ||
      mutation.revision_after <= mutation.revision_before) {
    return SetError("host BA graph mutation identity mismatch", error);
  }
  if (mutation.force_full_rebuild) {
    result->full_rebuild_required = true;
    return SetError("host BA graph mutation requires a full cold rebuild", error);
  }
  if (next_generation_ == std::numeric_limits<uint64_t>::max()) {
    return SetError("host BA graph generation exhausted", error);
  }

  try {
    // NOT_PERFORMANCE_READY correctness scaffold: VS1 currently uses
    // transactional copy-on-write publication so readers never see a partial
    // batch. transaction_copy_bytes exposes the remaining O(catalog) host cost
    // that must be removed before a Mapper performance candidate is enabled.
    std::shared_ptr<HostBaGraphPublication> pending =
        std::make_shared<HostBaGraphPublication>(*publication_);
    result->transaction_copy_bytes = publication_->resident_bytes;
    const uint64_t candidate_generation = next_generation_ + 1;

    std::unordered_set<uint64_t> observation_tombstones;
    observation_tombstones.reserve(mutation.tombstone_observations.size());
    for (const HostBaObservationKey& key :
         mutation.tombstone_observations) {
      observation_tombstones.insert(
          ObservationIdentity(key.image_id, key.point2D_idx));
    }

    for (const HostBaCameraRecord& value : mutation.camera_upserts) {
      if (!UpsertCamera(pending.get(), value, candidate_generation, result,
                        error)) {
        return false;
      }
    }
    for (const HostBaImageRecord& value : mutation.image_upserts) {
      if (!UpsertImage(pending.get(), value, candidate_generation, result,
                       error)) {
        return false;
      }
    }
    for (const HostBaPointRecord& value : mutation.point_upserts) {
      if (!UpsertPoint(pending.get(), value, candidate_generation, result,
                       error)) {
        return false;
      }
    }
    for (const HostBaObservationRecord& value : mutation.observation_upserts) {
      const uint64_t identity =
          ObservationIdentity(value.image_id, value.point2D_idx);
      // A coalesced add-then-delete of a previously absent observation is a
      // semantic no-op and must not consume a stable slot.
      if (observation_tombstones.count(identity) != 0 &&
          pending->observation_by_identity.count(identity) == 0) {
        continue;
      }
      if (!UpsertObservation(pending.get(), value, candidate_generation,
                             result, error)) {
        return false;
      }
    }
    for (const uint64_t identity : observation_tombstones) {
      TombstoneObservation(pending.get(), identity, candidate_generation,
                           result, false);
    }

    for (const uint64_t id : mutation.tombstone_point_ids) {
      const auto found = pending->point_by_id.find(id);
      if (found == pending->point_by_id.end()) continue;
      HostBaPointSlot& point = pending->points[found->second];
      if (!point.header.alive) continue;
      point.header.alive = 0;
      point.header.generation = candidate_generation;
      ++result->tombstoned_slots;
      TombstoneMatchingObservations(
          pending.get(), candidate_generation, result,
          [&](const HostBaObservationSlot& value) {
            return value.point_slot == point.header.slot;
          });
    }
    for (const uint32_t id : mutation.tombstone_image_ids) {
      const auto found = pending->image_by_id.find(id);
      if (found == pending->image_by_id.end()) continue;
      HostBaImageSlot& image = pending->images[found->second];
      if (!image.header.alive) continue;
      image.header.alive = 0;
      image.header.generation = candidate_generation;
      ++result->tombstoned_slots;
      TombstoneMatchingObservations(
          pending.get(), candidate_generation, result,
          [&](const HostBaObservationSlot& value) {
            return value.image_slot == image.header.slot;
          });
    }
    for (const uint32_t id : mutation.tombstone_camera_ids) {
      const auto found = pending->camera_by_id.find(id);
      if (found == pending->camera_by_id.end()) continue;
      HostBaCameraSlot& camera = pending->cameras[found->second];
      if (!camera.header.alive) continue;
      camera.header.alive = 0;
      camera.header.generation = candidate_generation;
      ++result->tombstoned_slots;
      for (HostBaImageSlot& image : pending->images) {
        if (!image.header.alive || image.camera_slot != camera.header.slot)
          continue;
        image.header.alive = 0;
        image.header.generation = candidate_generation;
        ++result->tombstoned_slots;
        TombstoneMatchingObservations(
            pending.get(), candidate_generation, result,
            [&](const HostBaObservationSlot& value) {
              return value.image_slot == image.header.slot;
            });
      }
    }

    if (!RebuildAdjacency(pending.get(), error)) return false;
    const bool semantic_noop = SameGraphSemantics(*publication_, *pending);
    pending->topology_revision = mutation.revision_after;
    if (semantic_noop) {
      pending->generation = publication_->generation;
      for (HostBaCameraSlot& value : pending->cameras)
        value.header.generation = std::min(value.header.generation,
                                           pending->generation);
      for (HostBaImageSlot& value : pending->images)
        value.header.generation = std::min(value.header.generation,
                                           pending->generation);
      for (HostBaPointSlot& value : pending->points)
        value.header.generation = std::min(value.header.generation,
                                           pending->generation);
      for (HostBaObservationSlot& value : pending->observations)
        value.header.generation = std::min(value.header.generation,
                                           pending->generation);
      result->semantic_noop = true;
    } else {
      pending->generation = candidate_generation;
      next_generation_ = candidate_generation;
    }
    pending->resident_bytes = EstimateResidentBytes(*pending);
    result->revision_after = pending->topology_revision;
    result->generation_after = pending->generation;
    result->published = true;
    publication_ = std::move(pending);
    return true;
  } catch (const std::bad_alloc&) {
    return SetError("host BA graph mutation allocation failed", error);
  }
}

CatalogReadLease HostBaGraphStore::AcquireReadLease() const noexcept {
  return CatalogReadLease(publication_);
}
bool HostBaGraphStore::IsCurrent(const CatalogReadLease& lease) const noexcept {
  return publication_ != nullptr && lease.publication_ == publication_;
}

namespace {

const TranslationSubsetPolicy* FindTranslationPolicy(
    const BaSolveIntent& intent, const uint32_t slot) {
  for (const TranslationSubsetPolicy& policy : intent.translation_subsets) {
    if (policy.image_slot == slot) return &policy;
  }
  return nullptr;
}

const CameraParameterPolicy* FindCameraPolicy(const BaSolveIntent& intent,
                                               const uint32_t slot) {
  for (const CameraParameterPolicy& policy : intent.camera_policies) {
    if (policy.camera_slot == slot) return &policy;
  }
  return nullptr;
}

const PointFixedPolicy* FindPointPolicy(const BaSolveIntent& intent,
                                        const uint32_t slot) {
  for (const PointFixedPolicy& policy : intent.point_policies) {
    if (policy.point_slot == slot) return &policy;
  }
  return nullptr;
}

bool ContainsSlot(const std::vector<uint32_t>& slots,
                  const uint32_t slot) noexcept {
  return std::find(slots.begin(), slots.end(), slot) != slots.end();
}

struct ResidualCandidate {
  ResidualKind kind = ResidualKind::kVisual;
  uint32_t source_slot = kBaGraphInvalidSlot;
  uint64_t source_insertion_index = 0;
  uint64_t physical_identity = 0;
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
  uint64_t point3D_id = 0;
};

bool AddParameterIfFirst(
    const ParameterOrdinal& parameter,
    std::unordered_map<ParameterIdentityKey, size_t, ParameterIdentityKeyHash>*
        seen,
    NativeHostSolveView* view,
    std::string* error) {
  const ParameterIdentityKey key{parameter.kind, parameter.entity_slot};
  if (!seen->emplace(key, view->parameter_ordinals.size()).second) return true;
  if (parameter.ambient_size < parameter.tangent_size) {
    return SetError("native parameter tangent dimension exceeds ambient",
                    error);
  }
  view->parameter_ordinals.push_back(parameter);
  view->ambient_parameter_count += parameter.ambient_size;
  if (!parameter.constant)
    view->effective_parameter_count += parameter.tangent_size;
  return true;
}

}  // namespace

bool NativeHostSolveMaterializer::Materialize(
    const CatalogReadLease& catalog,
    const BaSolveIntent& intent,
    const DenseActiveState& gathered_state,
    NativeHostSolveView* view,
    ActiveStateBuffer* active_state,
    NativeHostSolvePreparationRuntime* runtime,
    std::string* error) {
  if (view == nullptr || active_state == nullptr || runtime == nullptr ||
      error == nullptr) {
    return false;
  }
  *view = NativeHostSolveView();
  *active_state = ActiveStateBuffer();
  *runtime = NativeHostSolvePreparationRuntime();
  error->clear();
  const auto start = std::chrono::steady_clock::now();
  ++runtime->calls;

  if (!catalog.valid() || catalog.abi_version() != kHostBaGraphAbiVersion ||
      intent.abi_version != kNativeHostSolveViewAbiVersion ||
      intent.owner_epoch != catalog.owner_epoch() ||
      intent.catalog_revision != catalog.topology_revision() ||
      intent.catalog_generation != catalog.generation() ||
      gathered_state.owner_epoch != catalog.owner_epoch() ||
      intent.selection_revision == 0 || !intent.config.resolved ||
      intent.config.config_generation == 0 ||
      static_cast<uint8_t>(intent.config.arithmetic_precision) == 0 ||
      static_cast<uint8_t>(intent.config.hessian_backend) == 0 ||
      static_cast<uint8_t>(intent.config.schur_backend) == 0 ||
      static_cast<uint8_t>(intent.config.hot_kernel) == 0 ||
      static_cast<uint8_t>(intent.config.loss_mode) == 0 ||
      !std::isfinite(intent.config.loss_scale) ||
      intent.config.loss_scale <= 0.0 ||
      !std::isfinite(intent.config.lidar_near_zero_threshold) ||
      intent.config.lidar_near_zero_threshold < 0.0 ||
      intent.config.max_num_iterations < 0 ||
      intent.config.max_consecutive_invalid_steps <= 0 ||
      !std::isfinite(intent.config.function_tolerance) ||
      intent.config.function_tolerance < 0.0 ||
      !std::isfinite(intent.config.gradient_tolerance) ||
      intent.config.gradient_tolerance < 0.0 ||
      !std::isfinite(intent.config.parameter_tolerance) ||
      intent.config.parameter_tolerance < 0.0 ||
      !std::isfinite(intent.config.max_solver_time_in_seconds) ||
      intent.config.max_solver_time_in_seconds <= 0.0 ||
      !std::isfinite(intent.config.initial_trust_region_radius) ||
      !std::isfinite(intent.config.min_trust_region_radius) ||
      !std::isfinite(intent.config.max_trust_region_radius) ||
      intent.config.min_trust_region_radius < 0.0 ||
      intent.config.initial_trust_region_radius <= 0.0 ||
      intent.config.max_trust_region_radius <
          intent.config.initial_trust_region_radius ||
      intent.config.min_trust_region_radius >
          intent.config.initial_trust_region_radius ||
      !std::isfinite(intent.config.min_relative_decrease) ||
      intent.config.min_relative_decrease < 0.0 ||
      !std::isfinite(intent.config.min_lm_diagonal) ||
      !std::isfinite(intent.config.max_lm_diagonal) ||
      intent.config.min_lm_diagonal <= 0.0 ||
      intent.config.max_lm_diagonal < intent.config.min_lm_diagonal ||
      gathered_state.state_generation == 0) {
    return SetError("native solve identity or resolved config is invalid",
                    error);
  }
  const auto cameras = catalog.cameras();
  const auto images = catalog.images();
  const auto points = catalog.points();
  const auto observations = catalog.observations();
  const auto image_adjacency = catalog.image_observation_slots();
  const auto point_adjacency = catalog.point_observation_slots();

  if (++scratch_generation_ == 0) {
    std::fill(camera_active_stamps_.begin(), camera_active_stamps_.end(), 0);
    std::fill(image_active_stamps_.begin(), image_active_stamps_.end(), 0);
    std::fill(point_active_stamps_.begin(), point_active_stamps_.end(), 0);
    std::fill(observation_stamps_.begin(), observation_stamps_.end(), 0);
    std::fill(camera_state_stamps_.begin(), camera_state_stamps_.end(), 0);
    std::fill(image_state_stamps_.begin(), image_state_stamps_.end(), 0);
    std::fill(point_state_stamps_.begin(), point_state_stamps_.end(), 0);
    scratch_generation_ = 1;
  }
  const uint64_t stamp = scratch_generation_;
  GrowScratch(&camera_active_stamps_, cameras.size, uint64_t{0});
  GrowScratch(&image_active_stamps_, images.size, uint64_t{0});
  GrowScratch(&point_active_stamps_, points.size, uint64_t{0});
  GrowScratch(&observation_stamps_, observations.size, uint64_t{0});
  if (!CheckUniqueStateSlots(gathered_state.cameras, cameras.size,
                             &camera_state_stamps_, &camera_state_indices_,
                             stamp, gathered_state.state_generation, "camera",
                             error) ||
      !CheckUniqueStateSlots(gathered_state.images, images.size,
                             &image_state_stamps_, &image_state_indices_, stamp,
                             gathered_state.state_generation, "image", error) ||
      !CheckUniqueStateSlots(gathered_state.points, points.size,
                             &point_state_stamps_, &point_state_indices_, stamp,
                             gathered_state.state_generation, "point", error)) {
    return false;
  }

  view->identity.owner_epoch = catalog.owner_epoch();
  view->identity.catalog_revision = catalog.topology_revision();
  view->identity.catalog_generation = catalog.generation();
  view->identity.selection_revision = intent.selection_revision;
  view->identity.config_generation = intent.config.config_generation;
  view->identity.lidar_map_generation = intent.lidar.lidar_map_generation;
  view->identity.lidar_match_config_generation =
      intent.lidar.match_config_generation;
  view->kind = intent.kind;
  view->config = intent.config;
  view->catalog = catalog;
  view->lidar = intent.lidar;

  const auto activate_camera = [&](const uint32_t slot) -> bool {
    if (slot >= cameras.size || !cameras[slot].header.alive)
      return SetError("native solve references a dead camera slot", error);
    if (camera_active_stamps_[slot] != stamp) {
      camera_active_stamps_[slot] = stamp;
      view->active_camera_slots.push_back(slot);
    }
    return true;
  };
  const auto activate_point = [&](const uint32_t slot) -> bool {
    if (slot >= points.size || !points[slot].header.alive)
      return SetError("native solve references a dead point slot", error);
    if (point_active_stamps_[slot] != stamp) {
      point_active_stamps_[slot] = stamp;
      view->active_point_slots.push_back(slot);
    }
    return true;
  };
  const auto activate_image = [&](const uint32_t slot,
                                  const bool boundary) -> bool {
    if (slot >= images.size || !images[slot].header.alive ||
        !images[slot].registered)
      return SetError("native solve references an inactive image slot", error);
    if (image_active_stamps_[slot] != stamp) {
      image_active_stamps_[slot] = stamp;
      if (boundary)
        view->boundary_image_slots.push_back(slot);
      else
        view->active_image_slots.push_back(slot);
      if (!activate_camera(images[slot].camera_slot)) return false;
    } else if (!boundary && ContainsSlot(view->boundary_image_slots, slot)) {
      return SetError("native solve image is both active and boundary", error);
    }
    return true;
  };
  const auto select_observation = [&](const uint32_t slot,
                                      const bool boundary_source) -> bool {
    if (slot >= observations.size || !observations[slot].header.alive)
      return SetError("native solve references a dead observation slot", error);
    const HostBaObservationSlot& observation = observations[slot];
    if (!activate_image(observation.image_slot, boundary_source)) return false;
    if (!activate_point(observation.point_slot)) return false;
    if (observation_stamps_[slot] != stamp) {
      observation_stamps_[slot] = stamp;
      view->visual_observation_slots.push_back(slot);
      ++runtime->active_observation_visits;
      if (boundary_source) ++runtime->boundary_observation_visits;
    }
    return true;
  };

  for (const uint32_t image_slot : intent.active_image_slots) {
    if (image_slot >= images.size || image_active_stamps_[image_slot] == stamp) {
      return SetError("native active image selection is invalid or duplicated",
                      error);
    }
    if (!activate_image(image_slot, false)) return false;
    if (intent.active_visual_observation_slots.empty()) {
      const HostBaImageSlot& image = images[image_slot];
      const uint64_t end = static_cast<uint64_t>(image.adjacency_offset) +
                           image.adjacency_count;
      if (end > image_adjacency.size) {
        return SetError("native image adjacency is out of bounds", error);
      }
      for (uint64_t i = image.adjacency_offset; i < end; ++i) {
        ++runtime->catalog_observation_visits;
        if (!select_observation(image_adjacency[i], false)) return false;
      }
    }
  }
  if (!intent.active_visual_observation_slots.empty()) {
    for (const uint32_t observation_slot :
         intent.active_visual_observation_slots) {
      if (observation_slot >= observations.size ||
          !observations[observation_slot].header.alive ||
          observation_stamps_[observation_slot] == stamp ||
          image_active_stamps_[observations[observation_slot].image_slot] !=
              stamp) {
        return SetError(
            "native explicit visual selection is outside active images",
            error);
      }
      ++runtime->catalog_observation_visits;
      if (!select_observation(observation_slot, false)) return false;
    }
  }
  for (const uint32_t point_slot : intent.explicit_variable_point_slots) {
    if (!activate_point(point_slot)) return false;
    const HostBaPointSlot& point = points[point_slot];
    const uint64_t end = static_cast<uint64_t>(point.adjacency_offset) +
                         point.adjacency_count;
    if (end > point_adjacency.size) {
      return SetError("native point adjacency is out of bounds", error);
    }
    for (uint64_t i = point.adjacency_offset; i < end; ++i) {
      ++runtime->catalog_observation_visits;
      const uint32_t observation_slot = point_adjacency[i];
      if (observation_slot >= observations.size) {
        return SetError("native point observation slot is out of bounds", error);
      }
      const bool boundary = image_active_stamps_[
                                observations[observation_slot].image_slot] !=
                            stamp;
      if (!select_observation(observation_slot, boundary)) return false;
    }
  }

  std::unordered_set<uint32_t> lidar_constraint_slots;
  lidar_constraint_slots.reserve(view->lidar.constraints.size());
  for (const LidarConstraintRecord& lidar : view->lidar.constraints) {
    if (lidar.point_slot >= points.size ||
        point_active_stamps_[lidar.point_slot] != stamp ||
        !points[lidar.point_slot].header.alive ||
        !lidar_constraint_slots.insert(lidar.constraint_slot).second ||
        point_state_stamps_[lidar.point_slot] != stamp ||
        gathered_state.points[point_state_indices_[lidar.point_slot]]
                .state_generation != lidar.point_state_generation ||
        !IsFiniteArray(lidar.plane) || !IsFiniteArray(lidar.lidar_xyz) ||
        !std::isfinite(lidar.weight) || !std::isfinite(lidar.search_range)) {
      return SetError("native LiDAR selection is invalid or expands points",
                      error);
    }
  }

  std::unordered_set<uint32_t> fixed_pose_slots;
  fixed_pose_slots.reserve(intent.fixed_pose_slots.size());
  for (const uint32_t slot : intent.fixed_pose_slots) {
    if (slot >= images.size || !fixed_pose_slots.insert(slot).second ||
        image_active_stamps_[slot] != stamp) {
      return SetError("native fixed-pose selection is invalid", error);
    }
  }
  std::unordered_set<uint32_t> translation_policy_slots;
  translation_policy_slots.reserve(intent.translation_subsets.size());
  for (const TranslationSubsetPolicy& policy : intent.translation_subsets) {
    if (policy.image_slot >= images.size ||
        image_active_stamps_[policy.image_slot] != stamp ||
        ContainsSlot(view->boundary_image_slots, policy.image_slot) ||
        !translation_policy_slots.insert(policy.image_slot).second ||
        (policy.constant_mask & ~uint8_t{0x7}) != 0) {
      return SetError("native translation policy is invalid", error);
    }
  }
  std::unordered_set<uint32_t> camera_policy_slots;
  camera_policy_slots.reserve(intent.camera_policies.size());
  for (const CameraParameterPolicy& policy : intent.camera_policies) {
    if (policy.camera_slot >= cameras.size ||
        camera_active_stamps_[policy.camera_slot] != stamp ||
        !camera_policy_slots.insert(policy.camera_slot).second) {
      return SetError("native camera policy is invalid", error);
    }
    if (!policy.constant) {
      return SetError("native custom CUDA does not support variable cameras",
                      error);
    }
  }
  std::unordered_set<uint32_t> point_policy_slots;
  point_policy_slots.reserve(intent.point_policies.size());
  for (const PointFixedPolicy& policy : intent.point_policies) {
    if (policy.point_slot >= points.size ||
        point_active_stamps_[policy.point_slot] != stamp ||
        !point_policy_slots.insert(policy.point_slot).second ||
        (policy.has_search_range &&
         (!std::isfinite(policy.search_range) || policy.search_range < 0.0))) {
      return SetError("native point policy is invalid", error);
    }
  }
  for (const uint32_t slot : view->active_image_slots) {
    ImageFixedPolicyResult policy;
    policy.image_slot = slot;
    policy.pose_constant = fixed_pose_slots.count(slot) != 0 ? 1 : 0;
    const TranslationSubsetPolicy* translation =
        FindTranslationPolicy(intent, slot);
    if (translation != nullptr) {
      if ((translation->constant_mask & ~uint8_t{0x7}) != 0) {
        return SetError("native translation subset mask is invalid", error);
      }
      policy.translation_subset_mask = translation->constant_mask;
    }
    policy.tangent_size = policy.pose_constant
                              ? 0
                              : 3 + 3 - static_cast<uint32_t>(
                                            __builtin_popcount(
                                                policy.translation_subset_mask));
    view->fixed.images.push_back(policy);
  }
  for (const uint32_t slot : view->boundary_image_slots) {
    ImageFixedPolicyResult policy;
    policy.image_slot = slot;
    policy.pose_constant = 1;
    policy.boundary_pose = 1;
    view->fixed.images.push_back(policy);
  }
  for (const uint32_t slot : view->active_camera_slots) {
    const HostBaCameraSlot& camera = cameras[slot];
    CameraFixedPolicyResult result;
    result.camera_slot = slot;
    result.ambient_size = camera.parameter_count;
    const CameraParameterPolicy* policy = FindCameraPolicy(intent, slot);
    result.constant = policy == nullptr || policy->constant ? 1 : 0;
    if (policy != nullptr) result.fixed_parameter_indices =
                               policy->fixed_parameter_indices;
    std::sort(result.fixed_parameter_indices.begin(),
              result.fixed_parameter_indices.end());
    if (std::adjacent_find(result.fixed_parameter_indices.begin(),
                           result.fixed_parameter_indices.end()) !=
            result.fixed_parameter_indices.end() ||
        (!result.fixed_parameter_indices.empty() &&
         result.fixed_parameter_indices.back() >= camera.parameter_count)) {
      return SetError("native camera fixed-index policy is invalid", error);
    }
    result.tangent_size = result.constant
                              ? camera.parameter_count
                              : camera.parameter_count - static_cast<uint32_t>(
                                    result.fixed_parameter_indices.size());
    if (result.tangent_size == 0) {
      result.constant = 1;
      result.tangent_size = camera.parameter_count;
    }
    view->fixed.cameras.push_back(std::move(result));
  }
  std::unordered_map<uint32_t, uint32_t> selected_point_observations;
  selected_point_observations.reserve(view->active_point_slots.size());
  for (const uint32_t observation_slot : view->visual_observation_slots) {
    if (observation_slot >= observations.size) {
      return SetError("native visual selection contains an invalid slot",
                      error);
    }
    ++selected_point_observations[observations[observation_slot].point_slot];
  }
  for (const uint32_t slot : view->active_point_slots) {
    PointFixedPolicyResult result;
    result.point_slot = slot;
    const PointFixedPolicy* policy = FindPointPolicy(intent, slot);
    if (policy != nullptr) {
      result.constant = policy->constant ? 1 : 0;
      result.config_role = policy->config_role;
      result.has_search_range = policy->has_search_range ? 1 : 0;
      result.search_range = policy->search_range;
    }
    const auto selected = selected_point_observations.find(slot);
    const uint32_t selected_count =
        selected == selected_point_observations.end() ? 0 : selected->second;
    if (points[slot].track_length > selected_count) result.constant = 1;
    if (ContainsSlot(intent.explicit_variable_point_slots, slot)) {
      if (policy != nullptr && policy->constant) {
        return SetError(
            "native point is both explicitly variable and constant", error);
      }
      result.constant = 0;
    }
    view->fixed.points.push_back(result);
  }

  const std::vector<BaSolveIntent::ResidualSelection>& insertion_order =
      intent.source_insertion_order;
  const size_t expected_residual_count =
      view->visual_observation_slots.size() + view->lidar.constraints.size();
  if (insertion_order.size() != expected_residual_count) {
    return SetError(
        "native source insertion order must explicitly cover every residual",
        error);
  }
  std::unordered_map<uint32_t, const LidarConstraintRecord*> lidar_by_slot;
  lidar_by_slot.reserve(view->lidar.constraints.size());
  for (const LidarConstraintRecord& lidar : view->lidar.constraints) {
    if (!lidar_by_slot.emplace(lidar.constraint_slot, &lidar).second) {
      return SetError("native LiDAR constraint slot is duplicated", error);
    }
  }
  std::unordered_set<uint32_t> covered_visual_slots;
  std::unordered_set<uint32_t> covered_lidar_slots;
  covered_visual_slots.reserve(view->visual_observation_slots.size());
  covered_lidar_slots.reserve(view->lidar.constraints.size());
  std::vector<ResidualCandidate> residuals;
  residuals.reserve(insertion_order.size());
  for (size_t source_index = 0; source_index < insertion_order.size();
       ++source_index) {
    const BaSolveIntent::ResidualSelection& selected =
        insertion_order[source_index];
    if (selected.kind == ResidualKind::kVisual) {
      const uint32_t slot = selected.source_slot;
      if (slot >= observations.size ||
          observation_stamps_[slot] != stamp ||
          !covered_visual_slots.insert(slot).second) {
        return SetError("native source order has an invalid visual residual",
                        error);
      }
      const HostBaObservationSlot& observation = observations[slot];
      ResidualCandidate candidate;
      candidate.kind = ResidualKind::kVisual;
      candidate.source_slot = slot;
      candidate.source_insertion_index = source_index;
      candidate.physical_identity = observation.source_identity;
      candidate.image_id = images[observation.image_slot].image_id;
      candidate.point2D_idx = observation.point2D_idx;
      candidate.point3D_id = points[observation.point_slot].point3D_id;
      residuals.push_back(candidate);
    } else {
      const auto lidar_it = lidar_by_slot.find(selected.source_slot);
      if (lidar_it == lidar_by_slot.end() ||
          !covered_lidar_slots.insert(selected.source_slot).second) {
        return SetError("native source order has an invalid LiDAR residual",
                        error);
      }
      const LidarConstraintRecord& lidar = *lidar_it->second;
      ResidualCandidate candidate;
      candidate.kind = ResidualKind::kLidar;
      candidate.source_slot = lidar.constraint_slot;
      candidate.source_insertion_index = source_index;
      candidate.physical_identity = lidar.physical_identity;
      candidate.point3D_id = points[lidar.point_slot].point3D_id;
      residuals.push_back(candidate);
    }
  }
  if (covered_visual_slots.size() != view->visual_observation_slots.size() ||
      covered_lidar_slots.size() != view->lidar.constraints.size()) {
    return SetError("native source order does not cover every residual", error);
  }
  const bool source_order =
      static_cast<uint8_t>(intent.config.residual_order) == 1;
  if (!source_order) {
    std::sort(residuals.begin(), residuals.end(),
              [](const ResidualCandidate& lhs,
                 const ResidualCandidate& rhs) {
                return std::tie(lhs.kind, lhs.image_id, lhs.point2D_idx,
                                lhs.point3D_id,
                                lhs.source_insertion_index) <
                       std::tie(rhs.kind, rhs.image_id, rhs.point2D_idx,
                                rhs.point3D_id,
                                rhs.source_insertion_index);
              });
  }
  for (size_t i = 0; i < residuals.size(); ++i) {
    ResidualOrdinal ordinal;
    ordinal.execution_ordinal = i;
    ordinal.source_insertion_index = residuals[i].source_insertion_index;
    ordinal.physical_identity = residuals[i].physical_identity;
    ordinal.kind = residuals[i].kind;
    ordinal.source_slot = residuals[i].source_slot;
    view->residual_ordinals.push_back(ordinal);
  }

  std::unordered_map<ParameterIdentityKey, size_t, ParameterIdentityKeyHash>
      parameters;
  parameters.reserve(view->active_image_slots.size() * 2 +
                     view->active_point_slots.size() +
                     view->active_camera_slots.size());
  const auto image_policy = [&](const uint32_t slot)
      -> const ImageFixedPolicyResult* {
    for (const auto& value : view->fixed.images)
      if (value.image_slot == slot) return &value;
    return nullptr;
  };
  const auto point_policy = [&](const uint32_t slot)
      -> const PointFixedPolicyResult* {
    for (const auto& value : view->fixed.points)
      if (value.point_slot == slot) return &value;
    return nullptr;
  };
  const auto camera_policy = [&](const uint32_t slot)
      -> const CameraFixedPolicyResult* {
    for (const auto& value : view->fixed.cameras)
      if (value.camera_slot == slot) return &value;
    return nullptr;
  };
  for (const ResidualCandidate& residual : residuals) {
    if (residual.kind == ResidualKind::kVisual) {
      const HostBaObservationSlot& observation =
          observations[residual.source_slot];
      const ImageFixedPolicyResult* image =
          image_policy(observation.image_slot);
      const PointFixedPolicyResult* point =
          point_policy(observation.point_slot);
      const CameraFixedPolicyResult* camera =
          camera_policy(observation.camera_slot);
      if (image == nullptr || point == nullptr || camera == nullptr) {
        return SetError("native fixed policy does not cover a residual", error);
      }
      if (!image->pose_constant) {
        ParameterOrdinal quaternion;
        quaternion.ordinal = view->parameter_ordinals.size();
        quaternion.kind = ParameterKind::kQuaternion;
        quaternion.entity_slot = observation.image_slot;
        quaternion.ambient_size = 4;
        quaternion.tangent_size = 3;
        if (!AddParameterIfFirst(quaternion, &parameters, view, error))
          return false;
        ParameterOrdinal translation;
        translation.ordinal = view->parameter_ordinals.size();
        translation.kind = ParameterKind::kTranslation;
        translation.entity_slot = observation.image_slot;
        translation.ambient_size = 3;
        translation.translation_subset_mask =
            image->translation_subset_mask;
        translation.tangent_size = image->tangent_size - 3;
        translation.constant = translation.tangent_size == 0 ? 1 : 0;
        if (!AddParameterIfFirst(translation, &parameters, view, error))
          return false;
      }
      ParameterOrdinal point_parameter;
      point_parameter.ordinal = view->parameter_ordinals.size();
      point_parameter.kind = ParameterKind::kPoint3D;
      point_parameter.entity_slot = observation.point_slot;
      point_parameter.ambient_size = 3;
      point_parameter.tangent_size = 3;
      point_parameter.constant = point->constant;
      if (!AddParameterIfFirst(point_parameter, &parameters, view, error))
        return false;
      ParameterOrdinal camera_parameter;
      camera_parameter.ordinal = view->parameter_ordinals.size();
      camera_parameter.kind = ParameterKind::kCamera;
      camera_parameter.entity_slot = observation.camera_slot;
      camera_parameter.ambient_size = camera->ambient_size;
      camera_parameter.tangent_size = camera->tangent_size;
      camera_parameter.constant = camera->constant;
      if (!AddParameterIfFirst(camera_parameter, &parameters, view, error))
        return false;
    } else {
      const auto lidar = lidar_by_slot.find(residual.source_slot);
      if (lidar == lidar_by_slot.end()) {
        return SetError("native LiDAR ordinal references a missing constraint",
                        error);
      }
      const PointFixedPolicyResult* point =
          point_policy(lidar->second->point_slot);
      if (point == nullptr) {
        return SetError("native LiDAR point policy is missing", error);
      }
      ParameterOrdinal point_parameter;
      point_parameter.ordinal = view->parameter_ordinals.size();
      point_parameter.kind = ParameterKind::kPoint3D;
      point_parameter.entity_slot = lidar->second->point_slot;
      point_parameter.ambient_size = 3;
      point_parameter.tangent_size = 3;
      point_parameter.constant = point->constant;
      if (!AddParameterIfFirst(point_parameter, &parameters, view, error))
        return false;
    }
  }
  for (size_t i = 0; i < view->parameter_ordinals.size(); ++i)
    view->parameter_ordinals[i].ordinal = i;
  view->residual_block_count = residuals.size();
  view->scalar_residual_count = view->visual_observation_slots.size() * 2 +
                                view->lidar.constraints.size();

  active_state->owner_epoch = catalog.owner_epoch();
  active_state->catalog_generation = catalog.generation();
  active_state->state_generation = gathered_state.state_generation;
  for (const uint32_t slot : view->active_camera_slots) {
    if (camera_state_stamps_[slot] != stamp) {
      return SetError("native solve is missing active camera state", error);
    }
    const DenseCameraState& state =
        gathered_state.cameras[camera_state_indices_[slot]];
    if (state.parameters.size() != cameras[slot].parameter_count ||
        !IsFiniteArray(state.parameters)) {
      return SetError("native camera state is invalid", error);
    }
    active_state->cameras.push_back(state);
    runtime->state_values_copied += state.parameters.size();
  }
  const auto copy_image_state = [&](const uint32_t slot) -> bool {
    if (image_state_stamps_[slot] != stamp) {
      return SetError("native solve is missing active image state", error);
    }
    DenseImageState state = gathered_state.images[image_state_indices_[slot]];
    if (!IsFiniteArray(state.quaternion) || !IsFiniteArray(state.translation) ||
        !NormalizeQuaternion(state.quaternion, &state.quaternion)) {
      return SetError("native image state normalization failed", error);
    }
    active_state->images.push_back(state);
    runtime->state_values_copied += 7;
    ++runtime->quaternion_normalizations;
    return true;
  };
  for (const uint32_t slot : view->active_image_slots)
    if (!copy_image_state(slot)) return false;
  for (const uint32_t slot : view->boundary_image_slots)
    if (!copy_image_state(slot)) return false;
  for (const uint32_t slot : view->active_point_slots) {
    if (point_state_stamps_[slot] != stamp) {
      return SetError("native solve is missing active point state", error);
    }
    const DensePointState& state =
        gathered_state.points[point_state_indices_[slot]];
    if (!IsFiniteArray(state.xyz)) {
      return SetError("native point state is non-finite", error);
    }
    active_state->points.push_back(state);
    runtime->state_values_copied += 3;
  }

  runtime->descriptor_bytes =
      sizeof(*view) + VectorBytes(view->active_camera_slots) +
      VectorBytes(view->active_image_slots) +
      VectorBytes(view->boundary_image_slots) +
      VectorBytes(view->active_point_slots) +
      VectorBytes(view->visual_observation_slots) +
      VectorBytes(view->residual_ordinals) +
      VectorBytes(view->parameter_ordinals) +
      VectorBytes(view->fixed.images) + VectorBytes(view->fixed.points) +
      VectorBytes(view->lidar.constraints);
  for (const CameraFixedPolicyResult& camera : view->fixed.cameras)
    runtime->descriptor_bytes +=
        sizeof(camera) + VectorBytes(camera.fixed_parameter_indices);
  runtime->wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  return ValidateNativeHostSolveView(*view, *active_state, error);
}

bool ValidateNativeHostSolveView(const NativeHostSolveView& view,
                                 const ActiveStateBuffer& state,
                                 std::string* error) {
  if (error == nullptr) return false;
  if (!view.catalog.valid() ||
      view.identity.abi_version != kNativeHostSolveViewAbiVersion ||
      view.identity.catalog_abi_version != kHostBaGraphAbiVersion ||
      view.identity.owner_epoch != view.catalog.owner_epoch() ||
      view.identity.catalog_revision != view.catalog.topology_revision() ||
      view.identity.catalog_generation != view.catalog.generation() ||
      view.identity.config_generation != view.config.config_generation ||
      view.identity.lidar_map_generation !=
          view.lidar.lidar_map_generation ||
      view.identity.lidar_match_config_generation !=
          view.lidar.match_config_generation ||
      state.owner_epoch != view.identity.owner_epoch ||
      state.catalog_generation != view.identity.catalog_generation ||
      view.residual_block_count != view.residual_ordinals.size() ||
      view.scalar_residual_count !=
          view.visual_observation_slots.size() * 2 +
              view.lidar.constraints.size()) {
    return SetError("native solve view identity or count is inconsistent",
                    error);
  }
  std::vector<uint8_t> source_indices_seen(view.residual_ordinals.size(), 0);
  for (size_t i = 0; i < view.residual_ordinals.size(); ++i) {
    if (view.residual_ordinals[i].execution_ordinal != i ||
        view.residual_ordinals[i].source_insertion_index >=
            view.residual_ordinals.size() ||
        source_indices_seen[view.residual_ordinals[i]
                                .source_insertion_index] != 0) {
      return SetError("native residual ordinal sequence is invalid", error);
    }
    source_indices_seen[view.residual_ordinals[i].source_insertion_index] = 1;
  }
  for (size_t i = 0; i < view.parameter_ordinals.size(); ++i) {
    if (view.parameter_ordinals[i].ordinal != i) {
      return SetError("native parameter ordinal sequence is invalid", error);
    }
  }
  for (const DenseCameraState& value : state.cameras) {
    if (value.state_generation != state.state_generation)
      return SetError("native camera state generation is stale", error);
  }
  for (const DenseImageState& value : state.images) {
    if (value.state_generation != state.state_generation)
      return SetError("native image state generation is stale", error);
  }
  for (const DensePointState& value : state.points) {
    if (value.state_generation != state.state_generation)
      return SetError("native point state generation is stale", error);
  }
  return true;
}

}  // namespace gpu_ba
}  // namespace colmap
