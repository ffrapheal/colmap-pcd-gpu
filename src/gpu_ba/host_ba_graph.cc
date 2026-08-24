#include "gpu_ba/host_ba_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <stdexcept>
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
static_assert(std::is_standard_layout<HostBaIncidenceNode>::value &&
                  std::is_trivially_copyable<HostBaIncidenceNode>::value,
              "incidence node must be upload-safe");
static_assert(std::is_nothrow_copy_constructible<HostBaCameraSlot>::value &&
                  std::is_nothrow_copy_assignable<HostBaCameraSlot>::value &&
                  std::is_nothrow_destructible<HostBaCameraSlot>::value &&
                  std::is_nothrow_copy_constructible<HostBaImageSlot>::value &&
                  std::is_nothrow_copy_assignable<HostBaImageSlot>::value &&
                  std::is_nothrow_destructible<HostBaImageSlot>::value &&
                  std::is_nothrow_copy_constructible<HostBaPointSlot>::value &&
                  std::is_nothrow_copy_assignable<HostBaPointSlot>::value &&
                  std::is_nothrow_destructible<HostBaPointSlot>::value &&
                  std::is_nothrow_copy_constructible<HostBaObservationSlot>::value &&
                  std::is_nothrow_copy_assignable<HostBaObservationSlot>::value &&
                  std::is_nothrow_destructible<HostBaObservationSlot>::value &&
                  std::is_nothrow_copy_constructible<HostBaIncidenceNode>::value &&
                  std::is_nothrow_copy_assignable<HostBaIncidenceNode>::value &&
                  std::is_nothrow_destructible<HostBaIncidenceNode>::value,
              "in-place graph rollback records must be nothrow");

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

bool CanAppendUint32(const size_t current, const size_t additional) noexcept {
  constexpr size_t kLimit = std::numeric_limits<uint32_t>::max();
  return current <= kLimit && additional <= kLimit - current;
}

bool CheckedSizeAdd(const size_t lhs,
                    const size_t rhs,
                    size_t* result) noexcept {
  if (result == nullptr || rhs > std::numeric_limits<size_t>::max() - lhs)
    return false;
  *result = lhs + rhs;
  return true;
}

bool CheckedSizeMultiply(const size_t lhs,
                         const size_t rhs,
                         size_t* result) noexcept {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
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

bool AddCounter(const uint64_t value,
                uint64_t* counter,
                std::string* error) {
  if (counter == nullptr ||
      value > std::numeric_limits<uint64_t>::max() - *counter) {
    return SetError("host BA graph telemetry counter overflow", error);
  }
  *counter += value;
  return true;
}

bool ComputeGrowthTarget(const size_t current_capacity,
                         const size_t required,
                         const size_t max_size,
                         size_t* target) noexcept {
  if (target == nullptr || required > max_size) return false;
  if (required <= current_capacity && current_capacity != 0) {
    *target = current_capacity;
    return true;
  }
  size_t geometric = 0;
  if (current_capacity == 0) {
    const size_t headroom = required / 4 + (required % 4 != 0 ? 1 : 0);
    geometric = required > max_size - headroom ? max_size
                                                : required + headroom;
    geometric = std::max(geometric, std::min<size_t>(64, max_size));
  } else {
    const size_t headroom =
        current_capacity / 2 + (current_capacity % 2 != 0 ? 1 : 0);
    geometric = current_capacity > max_size - headroom
                    ? max_size
                    : current_capacity + headroom;
  }
  *target = std::max(required, geometric);
  return *target <= max_size;
}

template <typename T>
bool EnsureVectorCapacity(std::vector<T>* values,
                          const size_t required,
                          HostBaGraphUpdateResult* result,
                          std::string* error) {
  if (values == nullptr || result == nullptr)
    return SetError("host BA graph vector capacity output is null", error);
  size_t target = 0;
  if (!ComputeGrowthTarget(values->capacity(), required, values->max_size(),
                           &target)) {
    return SetError("host BA graph vector capacity exhausted", error);
  }
  if (target <= values->capacity()) return true;
  if (values->size() >
      std::numeric_limits<uint64_t>::max() / sizeof(T)) {
    return SetError("host BA graph capacity copy byte count overflow", error);
  }
  const uint64_t copy_bytes =
      static_cast<uint64_t>(values->size()) * sizeof(T);
  values->reserve(target);
  return AddCounter(1, &result->capacity_growth_events, error) &&
         AddCounter(copy_bytes, &result->capacity_growth_copy_bytes, error);
}

template <typename Map>
size_t HashEntryCapacity(const Map& values) noexcept {
  const long double capacity =
      static_cast<long double>(values.bucket_count()) *
      static_cast<long double>(values.max_load_factor());
  if (capacity >=
      static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(capacity);
}

template <typename Map>
bool EnsureHashCapacity(Map* values,
                        const size_t required,
                        HostBaGraphUpdateResult* result,
                        std::string* error) {
  if (values == nullptr || result == nullptr)
    return SetError("host BA graph hash capacity output is null", error);
  const size_t current_capacity = HashEntryCapacity(*values);
  if (required <= current_capacity &&
      (current_capacity >= 64 || !values->empty())) {
    return true;
  }
  const size_t growth_base =
      values->empty() && current_capacity < 64 ? 0 : current_capacity;
  size_t target = 0;
  if (!ComputeGrowthTarget(growth_base, required, values->max_size(),
                           &target)) {
    return SetError("host BA graph hash capacity exhausted", error);
  }
  if (target <= current_capacity) return true;
  const uint64_t moved_entries = static_cast<uint64_t>(values->size());
  values->reserve(target);
  return AddCounter(1, &result->hash_rehash_events, error) &&
         AddCounter(moved_entries, &result->hash_rehash_entries, error);
}

}  // namespace

using CameraIndex = std::unordered_map<uint32_t, uint32_t>;
using ImageIndex = std::unordered_map<uint32_t, uint32_t>;
using PointIndex = std::unordered_map<uint64_t, uint32_t>;
using ObservationIndex = std::unordered_map<uint64_t, uint32_t>;

struct HostBaGraphLeaseGateState {
  mutable std::shared_timed_mutex mutex;
};

struct HostBaGraphReadGuard {
  explicit HostBaGraphReadGuard(
      const std::shared_ptr<HostBaGraphLeaseGateState>& value)
      : gate(value), lock(gate->mutex, std::try_to_lock) {}

  bool owns_lock() const noexcept { return lock.owns_lock(); }

  std::shared_ptr<HostBaGraphLeaseGateState> gate;
  std::shared_lock<std::shared_timed_mutex> lock;
};

struct HostBaGraphPublication {
  uint32_t abi_version = kHostBaGraphAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t topology_revision = 0;
  uint64_t generation = 0;
  std::vector<HostBaCameraSlot> cameras;
  std::vector<HostBaImageSlot> images;
  std::vector<HostBaPointSlot> points;
  std::vector<HostBaObservationSlot> observations;
  std::vector<HostBaIncidenceNode> image_incidence_nodes;
  std::vector<HostBaIncidenceNode> point_incidence_nodes;
  CameraIndex camera_by_id;
  ImageIndex image_by_id;
  PointIndex point_by_id;
  ObservationIndex observation_by_identity;
  uint64_t resident_bytes = 0;
};

namespace {

uint64_t EstimateResidentBytes(const HostBaGraphPublication& graph) noexcept {
  uint64_t bytes = sizeof(graph);
  bytes += VectorBytes(graph.cameras);
  bytes += VectorBytes(graph.images);
  bytes += VectorBytes(graph.points);
  bytes += VectorBytes(graph.observations);
  bytes += VectorBytes(graph.image_incidence_nodes);
  bytes += VectorBytes(graph.point_incidence_nodes);
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
        !graph.images[observation.image_slot].header.alive ||
        !graph.points[observation.point_slot].header.alive ||
        graph.images[observation.image_slot].camera_slot >=
            graph.cameras.size() ||
        !graph.cameras[graph.images[observation.image_slot].camera_slot]
             .header.alive ||
        observation.association_generation == 0) {
      return SetError("host BA graph contains a dangling observation", error);
    }
    if (!IsFiniteArray(observation.xy)) {
      return SetError("host BA graph observation is non-finite", error);
    }
  }
  std::vector<uint32_t> image_counts(graph.images.size(), 0);
  std::vector<uint32_t> point_counts(graph.points.size(), 0);
  for (const HostBaObservationSlot& observation : graph.observations) {
    if (!observation.header.alive) continue;
    if (image_counts[observation.image_slot] ==
            std::numeric_limits<uint32_t>::max() ||
        point_counts[observation.point_slot] ==
            std::numeric_limits<uint32_t>::max()) {
      return SetError("host BA graph adjacency count overflow", error);
    }
    ++image_counts[observation.image_slot];
    ++point_counts[observation.point_slot];
  }
  for (size_t i = 0; i < graph.images.size(); ++i) {
    if (graph.images[i].adjacency_count != image_counts[i])
      return SetError("host BA graph image adjacency count mismatch", error);
  }
  for (size_t i = 0; i < graph.points.size(); ++i) {
    if (graph.points[i].adjacency_count != point_counts[i] ||
        graph.points[i].track_length != point_counts[i]) {
      return SetError("host BA graph point adjacency count mismatch", error);
    }
  }
  return true;
}

bool SameCamera(const HostBaCameraSlot& lhs,
                const HostBaCameraSlot& rhs) noexcept {
  return lhs.header.alive == rhs.header.alive &&
         lhs.camera_id == rhs.camera_id && lhs.model_id == rhs.model_id &&
         lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.parameter_count == rhs.parameter_count;
}

bool SameImage(const HostBaImageSlot& lhs,
               const HostBaImageSlot& rhs) noexcept {
  return lhs.header.alive == rhs.header.alive &&
         lhs.image_id == rhs.image_id && lhs.camera_slot == rhs.camera_slot &&
         lhs.adjacency_head == rhs.adjacency_head &&
         lhs.adjacency_count == rhs.adjacency_count &&
         lhs.lifetime_incidence_count == rhs.lifetime_incidence_count &&
         lhs.registered == rhs.registered;
}

bool SamePoint(const HostBaPointSlot& lhs,
               const HostBaPointSlot& rhs) noexcept {
  return lhs.header.alive == rhs.header.alive &&
         lhs.point3D_id == rhs.point3D_id &&
         lhs.adjacency_head == rhs.adjacency_head &&
         lhs.adjacency_count == rhs.adjacency_count &&
         lhs.lifetime_incidence_count == rhs.lifetime_incidence_count &&
         lhs.track_length == rhs.track_length;
}

bool SameObservation(const HostBaObservationSlot& lhs,
                     const HostBaObservationSlot& rhs) noexcept {
  return lhs.header.alive == rhs.header.alive &&
         lhs.image_slot == rhs.image_slot &&
         lhs.point_slot == rhs.point_slot &&
         lhs.point2D_idx == rhs.point2D_idx &&
         lhs.source_identity == rhs.source_identity &&
         lhs.association_generation == rhs.association_generation &&
         SameXy(lhs.xy, rhs.xy);
}

bool AppendIncidence(std::vector<HostBaIncidenceNode>* nodes,
                     uint32_t* head,
                     const uint32_t observation_slot,
                     const uint64_t association_generation,
                     std::string* error) {
  if (nodes->size() >= std::numeric_limits<uint32_t>::max())
    return SetError("host BA graph incidence capacity exhausted", error);
  HostBaIncidenceNode node;
  node.observation_slot = observation_slot;
  node.next_node = *head;
  node.association_generation = association_generation;
  *head = static_cast<uint32_t>(nodes->size());
  nodes->push_back(node);
  return true;
}

bool BuildColdPublication(const HostBaGraphColdInput& input,
                          const uint64_t generation,
                          HostBaGraphUpdateResult* result,
                          HostBaGraphPublication* graph,
                          std::string* error) {
  graph->owner_epoch = input.owner_epoch;
  graph->topology_revision = input.topology_revision;
  graph->generation = generation;
  if (!FitsUint32(input.cameras.size()) || !FitsUint32(input.images.size()) ||
      !FitsUint32(input.points.size()) ||
      !FitsUint32(input.observations.size()) ||
      input.cameras.size() > graph->cameras.max_size() ||
      input.images.size() > graph->images.max_size() ||
      input.points.size() > graph->points.max_size() ||
      input.observations.size() > graph->observations.max_size() ||
      input.observations.size() > graph->image_incidence_nodes.max_size() ||
      input.observations.size() > graph->point_incidence_nodes.max_size()) {
    return SetError("host BA graph cold input exceeds storage capacity", error);
  }
  if (!EnsureVectorCapacity(&graph->cameras, input.cameras.size(), result,
                            error) ||
      !EnsureVectorCapacity(&graph->images, input.images.size(), result,
                            error) ||
      !EnsureVectorCapacity(&graph->points, input.points.size(), result,
                            error) ||
      !EnsureVectorCapacity(&graph->observations, input.observations.size(),
                            result, error) ||
      !EnsureVectorCapacity(&graph->image_incidence_nodes,
                            input.observations.size(), result, error) ||
      !EnsureVectorCapacity(&graph->point_incidence_nodes,
                            input.observations.size(), result, error) ||
      !EnsureHashCapacity(&graph->camera_by_id, input.cameras.size(), result,
                          error) ||
      !EnsureHashCapacity(&graph->image_by_id, input.images.size(), result,
                          error) ||
      !EnsureHashCapacity(&graph->point_by_id, input.points.size(), result,
                          error) ||
      !EnsureHashCapacity(&graph->observation_by_identity,
                          input.observations.size(), result, error)) {
    return false;
  }
  for (const HostBaCameraRecord& input_camera : input.cameras) {
    HostBaCameraSlot camera;
    camera.header.slot = static_cast<uint32_t>(graph->cameras.size());
    camera.header.generation = generation;
    camera.header.alive = 1;
    camera.camera_id = input_camera.camera_id;
    camera.model_id = input_camera.model_id;
    camera.width = input_camera.width;
    camera.height = input_camera.height;
    camera.parameter_count = input_camera.parameter_count;
    if (!graph->camera_by_id.emplace(camera.camera_id, camera.header.slot)
             .second) {
      return SetError("host BA graph duplicate camera", error);
    }
    graph->cameras.push_back(camera);
    ++result->appended_slots;
  }
  for (const HostBaImageRecord& input_image : input.images) {
    const auto camera = graph->camera_by_id.find(input_image.camera_id);
    if (camera == graph->camera_by_id.end())
      return SetError("host BA graph image references a missing camera", error);
    HostBaImageSlot image;
    image.header.slot = static_cast<uint32_t>(graph->images.size());
    image.header.generation = generation;
    image.header.alive = 1;
    image.image_id = input_image.image_id;
    image.camera_slot = camera->second;
    image.registered = input_image.registered ? 1 : 0;
    if (!graph->image_by_id.emplace(image.image_id, image.header.slot).second)
      return SetError("host BA graph duplicate image", error);
    graph->images.push_back(image);
    ++result->appended_slots;
  }
  for (const HostBaPointRecord& input_point : input.points) {
    HostBaPointSlot point;
    point.header.slot = static_cast<uint32_t>(graph->points.size());
    point.header.generation = generation;
    point.header.alive = 1;
    point.point3D_id = input_point.point3D_id;
    if (!graph->point_by_id.emplace(point.point3D_id, point.header.slot).second)
      return SetError("host BA graph duplicate point", error);
    graph->points.push_back(point);
    ++result->appended_slots;
  }
  for (const HostBaObservationRecord& input_observation : input.observations) {
    const auto image = graph->image_by_id.find(input_observation.image_id);
    const auto point = graph->point_by_id.find(input_observation.point3D_id);
    const uint64_t identity = ObservationIdentity(
        input_observation.image_id, input_observation.point2D_idx);
    if (image == graph->image_by_id.end() ||
        point == graph->point_by_id.end() ||
        !IsFiniteArray(input_observation.xy)) {
      return SetError("host BA graph observation references missing data", error);
    }
    HostBaObservationSlot observation;
    observation.header.slot = static_cast<uint32_t>(graph->observations.size());
    observation.header.generation = generation;
    observation.header.alive = 1;
    observation.image_slot = image->second;
    observation.point_slot = point->second;
    observation.point2D_idx = input_observation.point2D_idx;
    observation.source_identity = identity;
    observation.association_generation = 1;
    observation.xy = input_observation.xy;
    if (!graph->observation_by_identity
             .emplace(identity, observation.header.slot).second) {
      return SetError("host BA graph duplicate observation", error);
    }
    HostBaImageSlot& image_slot = graph->images[observation.image_slot];
    HostBaPointSlot& point_slot = graph->points[observation.point_slot];
    if (image_slot.adjacency_count == std::numeric_limits<uint32_t>::max() ||
        point_slot.adjacency_count == std::numeric_limits<uint32_t>::max() ||
        point_slot.track_length == std::numeric_limits<uint32_t>::max() ||
        image_slot.lifetime_incidence_count ==
            std::numeric_limits<uint32_t>::max() ||
        point_slot.lifetime_incidence_count ==
            std::numeric_limits<uint32_t>::max() ||
        !AppendIncidence(&graph->image_incidence_nodes,
                         &image_slot.adjacency_head, observation.header.slot,
                         observation.association_generation, error) ||
        !AppendIncidence(&graph->point_incidence_nodes,
                         &point_slot.adjacency_head, observation.header.slot,
                         observation.association_generation, error)) {
      return false;
    }
    ++image_slot.adjacency_count;
    ++image_slot.lifetime_incidence_count;
    ++point_slot.adjacency_count;
    ++point_slot.lifetime_incidence_count;
    ++point_slot.track_length;
    graph->observations.push_back(observation);
    ++result->appended_slots;
    result->adjacency_nodes_appended += 2;
  }
  ++result->full_catalog_scans;
  if (!ValidateGraphReferences(*graph, error)) return false;
  graph->resident_bytes = EstimateResidentBytes(*graph);
  return true;
}

struct MapInsertionRollback {
  explicit MapInsertionRollback(HostBaGraphPublication* value) : graph(value) {}
  ~MapInsertionRollback() noexcept {
    if (keep || graph == nullptr) return;
    for (auto it = observations.rbegin(); it != observations.rend(); ++it)
      graph->observation_by_identity.erase(*it);
    for (auto it = points.rbegin(); it != points.rend(); ++it)
      graph->point_by_id.erase(*it);
    for (auto it = images.rbegin(); it != images.rend(); ++it)
      graph->image_by_id.erase(*it);
    for (auto it = cameras.rbegin(); it != cameras.rend(); ++it)
      graph->camera_by_id.erase(*it);
  }

  HostBaGraphPublication* graph = nullptr;
  std::vector<CameraIndex::iterator> cameras;
  std::vector<ImageIndex::iterator> images;
  std::vector<PointIndex::iterator> points;
  std::vector<ObservationIndex::iterator> observations;
  bool keep = false;
};

class AcceptedBatchFailureGuard {
 public:
  AcceptedBatchFailureGuard(bool* valid, HostBaGraphUpdateResult* result)
      : valid_(valid), result_(result) {}
  ~AcceptedBatchFailureGuard() noexcept {
    if (!armed_) return;
    if (valid_ != nullptr) *valid_ = false;
    if (result_ != nullptr) result_->full_rebuild_required = true;
  }

  void Disarm() noexcept { armed_ = false; }

 private:
  bool* valid_ = nullptr;
  HostBaGraphUpdateResult* result_ = nullptr;
  bool armed_ = true;
};

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
    std::shared_ptr<const HostBaGraphPublication> publication,
    std::shared_ptr<HostBaGraphReadGuard> guard)
    : publication_(std::move(publication)), guard_(std::move(guard)) {}

bool CatalogReadLease::valid() const noexcept {
  return publication_ != nullptr && guard_ != nullptr && guard_->owns_lock();
}
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
BaArrayView<HostBaIncidenceNode>
CatalogReadLease::image_incidence_nodes() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaIncidenceNode>()
                                 : MakeView(publication_->image_incidence_nodes);
}
BaArrayView<HostBaIncidenceNode>
CatalogReadLease::point_incidence_nodes() const noexcept {
  return publication_ == nullptr ? BaArrayView<HostBaIncidenceNode>()
                                 : MakeView(publication_->point_incidence_nodes);
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
    : owner_epoch_(owner_epoch),
      gate_(std::make_shared<HostBaGraphLeaseGateState>()) {}
HostBaGraphStore::~HostBaGraphStore() = default;
uint64_t HostBaGraphStore::owner_epoch() const noexcept { return owner_epoch_; }
uint64_t HostBaGraphStore::topology_revision() const noexcept {
  if (gate_ == nullptr) return 0;
  std::shared_lock<std::shared_timed_mutex> lock(gate_->mutex,
                                                 std::try_to_lock);
  return !lock.owns_lock() || publication_ == nullptr
             ? 0
             : publication_->topology_revision;
}
uint64_t HostBaGraphStore::generation() const noexcept {
  if (gate_ == nullptr) return 0;
  std::shared_lock<std::shared_timed_mutex> lock(gate_->mutex,
                                                 std::try_to_lock);
  return !lock.owns_lock() || publication_ == nullptr ? 0
                                                       : publication_->generation;
}
bool HostBaGraphStore::empty() const noexcept {
  if (gate_ == nullptr) return true;
  std::shared_lock<std::shared_timed_mutex> lock(gate_->mutex,
                                                 std::try_to_lock);
  return !lock.owns_lock() || publication_ == nullptr;
}

bool HostBaGraphStore::ColdBuild(const HostBaGraphColdInput& input,
                                 HostBaGraphUpdateResult* result,
                                 std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = HostBaGraphUpdateResult();
  error->clear();
  if (gate_ == nullptr)
    return SetError("host BA graph lease gate is unavailable", error);
  std::unique_lock<std::shared_timed_mutex> write_lock(gate_->mutex,
                                                       std::try_to_lock);
  if (!write_lock.owns_lock()) {
    result->reader_busy = true;
    return SetError("HOST_BA_GRAPH_STORE_BUSY", error);
  }
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
    valid_ = false;
    std::shared_ptr<HostBaGraphPublication> pending =
        std::make_shared<HostBaGraphPublication>();
    if (!BuildColdPublication(input, next_generation_ + 1, result,
                              pending.get(), error)) {
      return false;
    }
    result->revision_after = input.topology_revision;
    result->generation_after = pending->generation;
    result->published = true;
    publication_ = std::move(pending);
    next_generation_ = publication_->generation;
    valid_ = true;
    return true;
  } catch (const std::bad_alloc&) {
    return SetError("host BA graph cold-build allocation failed", error);
  } catch (const std::length_error&) {
    return SetError("host BA graph cold-build size overflow", error);
  }
}

bool HostBaGraphStore::ApplyCoalescedMutation(
    const CoalescedBaGraphMutation& mutation,
    HostBaGraphUpdateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = HostBaGraphUpdateResult();
  error->clear();
  if (gate_ == nullptr)
    return SetError("host BA graph lease gate is unavailable", error);
  std::unique_lock<std::shared_timed_mutex> write_lock(gate_->mutex,
                                                       std::try_to_lock);
  if (!write_lock.owns_lock()) {
    result->reader_busy = true;
    return SetError("HOST_BA_GRAPH_STORE_BUSY", error);
  }
  if (!valid_ || publication_ == nullptr) {
    return SetError("host BA graph mutation requires a cold publication", error);
  }
  result->revision_before = publication_->topology_revision;
  result->generation_before = publication_->generation;
  if (mutation.owner_epoch != owner_epoch_ ||
      mutation.revision_before != publication_->topology_revision ||
      mutation.revision_after <= mutation.revision_before) {
    return SetError("host BA graph mutation identity mismatch", error);
  }
  AcceptedBatchFailureGuard accepted_batch_failure(&valid_, result);
  if (mutation.force_full_rebuild ||
      !mutation.tombstone_camera_ids.empty()) {
    return SetError("host BA graph mutation requires a full cold rebuild", error);
  }
  if (next_generation_ == std::numeric_limits<uint64_t>::max()) {
    return SetError("host BA graph generation exhausted", error);
  }

  try {
    HostBaGraphPublication& graph = *publication_;
    const uint64_t candidate_generation = next_generation_ + 1;
    result->transaction_copy_bytes = 0;
    result->full_catalog_scans = 0;
    result->in_place_delta = true;

    using StagedCameras = std::unordered_map<uint32_t, HostBaCameraSlot>;
    using StagedImages = std::unordered_map<uint32_t, HostBaImageSlot>;
    using StagedPoints = std::unordered_map<uint32_t, HostBaPointSlot>;
    using StagedObservations =
        std::unordered_map<uint32_t, HostBaObservationSlot>;
    size_t staged_image_capacity = 0;
    size_t staged_point_capacity = 0;
    size_t staged_observation_capacity = 0;
    size_t cascade_capacity = 0;
    size_t observation_stage_owners = 0;
    size_t point_cascade_capacity = 0;
    size_t image_cascade_capacity = 0;
    if (!CheckedSizeMultiply(mutation.observation_upserts.size(), 2,
                             &observation_stage_owners) ||
        !CheckedSizeMultiply(mutation.tombstone_point_ids.size(), 4,
                             &point_cascade_capacity) ||
        !CheckedSizeMultiply(mutation.tombstone_image_ids.size(), 4,
                             &image_cascade_capacity) ||
        !CheckedSizeAdd(mutation.image_upserts.size(),
                        mutation.tombstone_image_ids.size(),
                        &staged_image_capacity) ||
        !CheckedSizeAdd(staged_image_capacity,
                        observation_stage_owners,
                        &staged_image_capacity) ||
        !CheckedSizeAdd(mutation.point_upserts.size(),
                        mutation.tombstone_point_ids.size(),
                        &staged_point_capacity) ||
        !CheckedSizeAdd(staged_point_capacity,
                        observation_stage_owners,
                        &staged_point_capacity) ||
        !CheckedSizeAdd(mutation.observation_upserts.size(),
                        mutation.tombstone_observations.size(),
                        &staged_observation_capacity) ||
        !CheckedSizeAdd(staged_observation_capacity, 8,
                        &staged_observation_capacity) ||
        !CheckedSizeAdd(point_cascade_capacity, image_cascade_capacity,
                        &cascade_capacity)) {
      return SetError("host BA graph mutation size overflow", error);
    }
    StagedCameras staged_cameras;
    StagedImages staged_images;
    StagedPoints staged_points;
    StagedObservations staged_observations;
    std::vector<uint32_t> staged_observation_order;
    staged_cameras.reserve(mutation.camera_upserts.size());
    staged_images.reserve(staged_image_capacity);
    staged_points.reserve(staged_point_capacity);
    staged_observations.reserve(staged_observation_capacity);
    staged_observation_order.reserve(staged_observation_capacity);

    std::unordered_set<uint64_t> observation_tombstones;
    observation_tombstones.reserve(mutation.tombstone_observations.size());
    for (const HostBaObservationKey& key : mutation.tombstone_observations) {
      observation_tombstones.insert(
          ObservationIdentity(key.image_id, key.point2D_idx));
    }
    std::unordered_set<uint32_t> cascaded_observations;
    cascaded_observations.reserve(cascade_capacity);
    std::unordered_set<uint32_t> tombstoned_point_slots;
    std::unordered_set<uint32_t> tombstoned_image_slots;
    tombstoned_point_slots.reserve(mutation.tombstone_point_ids.size());
    tombstoned_image_slots.reserve(mutation.tombstone_image_ids.size());

    std::unordered_set<uint32_t> new_camera_ids;
    std::unordered_set<uint32_t> new_image_ids;
    std::unordered_set<uint64_t> new_point_ids;
    std::unordered_set<uint64_t> new_observation_identities;
    new_camera_ids.reserve(mutation.camera_upserts.size());
    new_image_ids.reserve(mutation.image_upserts.size());
    new_point_ids.reserve(mutation.point_upserts.size());
    new_observation_identities.reserve(mutation.observation_upserts.size());
    for (const HostBaCameraRecord& value : mutation.camera_upserts) {
      if (graph.camera_by_id.count(value.camera_id) == 0)
        new_camera_ids.insert(value.camera_id);
    }
    for (const HostBaImageRecord& value : mutation.image_upserts) {
      if (graph.image_by_id.count(value.image_id) == 0)
        new_image_ids.insert(value.image_id);
    }
    for (const HostBaPointRecord& value : mutation.point_upserts) {
      if (graph.point_by_id.count(value.point3D_id) == 0)
        new_point_ids.insert(value.point3D_id);
    }
    for (const HostBaObservationRecord& value :
         mutation.observation_upserts) {
      const uint64_t identity =
          ObservationIdentity(value.image_id, value.point2D_idx);
      if (graph.observation_by_identity.count(identity) == 0 &&
          observation_tombstones.count(identity) == 0) {
        new_observation_identities.insert(identity);
      }
    }
    if (!CanAppendUint32(graph.cameras.size(), new_camera_ids.size()) ||
        !CanAppendUint32(graph.images.size(), new_image_ids.size()) ||
        !CanAppendUint32(graph.points.size(), new_point_ids.size()) ||
        !CanAppendUint32(graph.observations.size(),
                         new_observation_identities.size())) {
      return SetError("host BA graph stable slot capacity exhausted", error);
    }
    const size_t required_cameras = graph.cameras.size() + new_camera_ids.size();
    const size_t required_images = graph.images.size() + new_image_ids.size();
    const size_t required_points = graph.points.size() + new_point_ids.size();
    const size_t required_observations =
        graph.observations.size() + new_observation_identities.size();
    if (!EnsureVectorCapacity(&graph.cameras, required_cameras, result,
                              error) ||
        !EnsureVectorCapacity(&graph.images, required_images, result, error) ||
        !EnsureVectorCapacity(&graph.points, required_points, result, error) ||
        !EnsureVectorCapacity(&graph.observations, required_observations,
                              result, error) ||
        !EnsureHashCapacity(&graph.camera_by_id,
                            graph.camera_by_id.size() + new_camera_ids.size(),
                            result, error) ||
        !EnsureHashCapacity(&graph.image_by_id,
                            graph.image_by_id.size() + new_image_ids.size(),
                            result, error) ||
        !EnsureHashCapacity(&graph.point_by_id,
                            graph.point_by_id.size() + new_point_ids.size(),
                            result, error) ||
        !EnsureHashCapacity(
            &graph.observation_by_identity,
            graph.observation_by_identity.size() +
                new_observation_identities.size(),
            result, error)) {
      return false;
    }

    MapInsertionRollback new_keys(&graph);
    new_keys.cameras.reserve(new_camera_ids.size());
    new_keys.images.reserve(new_image_ids.size());
    new_keys.points.reserve(new_point_ids.size());
    new_keys.observations.reserve(new_observation_identities.size());

    for (const HostBaCameraRecord& value : mutation.camera_upserts) {
      if (graph.camera_by_id.count(value.camera_id) != 0) continue;
      const uint32_t slot = static_cast<uint32_t>(
          graph.cameras.size() + new_keys.cameras.size());
      const auto inserted = graph.camera_by_id.emplace(value.camera_id, slot);
      if (inserted.second) new_keys.cameras.push_back(inserted.first);
    }
    for (const HostBaImageRecord& value : mutation.image_upserts) {
      if (graph.image_by_id.count(value.image_id) != 0) continue;
      const uint32_t slot = static_cast<uint32_t>(
          graph.images.size() + new_keys.images.size());
      const auto inserted = graph.image_by_id.emplace(value.image_id, slot);
      if (inserted.second) new_keys.images.push_back(inserted.first);
    }
    for (const HostBaPointRecord& value : mutation.point_upserts) {
      if (graph.point_by_id.count(value.point3D_id) != 0) continue;
      const uint32_t slot = static_cast<uint32_t>(
          graph.points.size() + new_keys.points.size());
      const auto inserted = graph.point_by_id.emplace(value.point3D_id, slot);
      if (inserted.second) new_keys.points.push_back(inserted.first);
    }
    for (const HostBaObservationRecord& value :
         mutation.observation_upserts) {
      const uint64_t identity =
          ObservationIdentity(value.image_id, value.point2D_idx);
      if (graph.observation_by_identity.count(identity) != 0 ||
          observation_tombstones.count(identity) != 0) {
        continue;
      }
      const uint32_t slot = static_cast<uint32_t>(
          graph.observations.size() + new_keys.observations.size());
      const auto inserted =
          graph.observation_by_identity.emplace(identity, slot);
      if (inserted.second) new_keys.observations.push_back(inserted.first);
    }

    const auto stage_camera = [&](const uint32_t slot)
        -> HostBaCameraSlot& {
      const auto found = staged_cameras.find(slot);
      if (found != staged_cameras.end()) return found->second;
      HostBaCameraSlot value;
      if (slot < graph.cameras.size()) value = graph.cameras[slot];
      value.header.slot = slot;
      return staged_cameras.emplace(slot, value).first->second;
    };
    const auto stage_image = [&](const uint32_t slot) -> HostBaImageSlot& {
      const auto found = staged_images.find(slot);
      if (found != staged_images.end()) return found->second;
      HostBaImageSlot value;
      if (slot < graph.images.size()) value = graph.images[slot];
      value.header.slot = slot;
      return staged_images.emplace(slot, value).first->second;
    };
    const auto stage_point = [&](const uint32_t slot) -> HostBaPointSlot& {
      const auto found = staged_points.find(slot);
      if (found != staged_points.end()) return found->second;
      HostBaPointSlot value;
      if (slot < graph.points.size()) value = graph.points[slot];
      value.header.slot = slot;
      return staged_points.emplace(slot, value).first->second;
    };
    const auto stage_observation = [&](const uint32_t slot)
        -> HostBaObservationSlot& {
      const auto found = staged_observations.find(slot);
      if (found != staged_observations.end()) return found->second;
      HostBaObservationSlot value;
      if (slot < graph.observations.size()) value = graph.observations[slot];
      value.header.slot = slot;
      const auto inserted = staged_observations.emplace(slot, value);
      staged_observation_order.push_back(slot);
      return inserted.first->second;
    };
    const auto camera_value = [&](const uint32_t slot)
        -> const HostBaCameraSlot* {
      const auto found = staged_cameras.find(slot);
      if (found != staged_cameras.end()) return &found->second;
      return slot < graph.cameras.size() ? &graph.cameras[slot] : nullptr;
    };
    const auto image_value = [&](const uint32_t slot)
        -> const HostBaImageSlot* {
      const auto found = staged_images.find(slot);
      if (found != staged_images.end()) return &found->second;
      return slot < graph.images.size() ? &graph.images[slot] : nullptr;
    };
    const auto point_value = [&](const uint32_t slot)
        -> const HostBaPointSlot* {
      const auto found = staged_points.find(slot);
      if (found != staged_points.end()) return &found->second;
      return slot < graph.points.size() ? &graph.points[slot] : nullptr;
    };

    for (const HostBaCameraRecord& value : mutation.camera_upserts) {
      const auto found = graph.camera_by_id.find(value.camera_id);
      if (found == graph.camera_by_id.end())
        return SetError("host BA graph camera map insertion failed", error);
      HostBaCameraSlot& slot = stage_camera(found->second);
      slot.header.alive = 1;
      slot.camera_id = value.camera_id;
      slot.model_id = value.model_id;
      slot.width = value.width;
      slot.height = value.height;
      slot.parameter_count = value.parameter_count;
    }
    for (const HostBaImageRecord& value : mutation.image_upserts) {
      const auto image = graph.image_by_id.find(value.image_id);
      const auto camera = graph.camera_by_id.find(value.camera_id);
      if (image == graph.image_by_id.end() ||
          camera == graph.camera_by_id.end()) {
        return SetError("host BA graph image references a missing camera",
                        error);
      }
      const HostBaCameraSlot* camera_slot = camera_value(camera->second);
      if (camera_slot == nullptr || !camera_slot->header.alive) {
        return SetError("host BA graph image references a dead camera", error);
      }
      HostBaImageSlot& slot = stage_image(image->second);
      slot.header.alive = 1;
      slot.image_id = value.image_id;
      slot.camera_slot = camera->second;
      slot.registered = value.registered ? 1 : 0;
    }
    for (const HostBaPointRecord& value : mutation.point_upserts) {
      const auto found = graph.point_by_id.find(value.point3D_id);
      if (found == graph.point_by_id.end())
        return SetError("host BA graph point map insertion failed", error);
      HostBaPointSlot& slot = stage_point(found->second);
      slot.header.alive = 1;
      slot.point3D_id = value.point3D_id;
    }
    for (const HostBaObservationRecord& value :
         mutation.observation_upserts) {
      const uint64_t identity =
          ObservationIdentity(value.image_id, value.point2D_idx);
      if (observation_tombstones.count(identity) != 0 &&
          graph.observation_by_identity.count(identity) == 0) {
        continue;
      }
      const auto observation = graph.observation_by_identity.find(identity);
      const auto image = graph.image_by_id.find(value.image_id);
      const auto point = graph.point_by_id.find(value.point3D_id);
      if (observation == graph.observation_by_identity.end() ||
          image == graph.image_by_id.end() ||
          point == graph.point_by_id.end() || !IsFiniteArray(value.xy)) {
        return SetError("host BA graph observation references missing data",
                        error);
      }
      const HostBaImageSlot* image_slot = image_value(image->second);
      const HostBaPointSlot* point_slot = point_value(point->second);
      if (image_slot == nullptr || point_slot == nullptr ||
          !image_slot->header.alive || !point_slot->header.alive) {
        return SetError("host BA graph observation references dead data",
                        error);
      }
      HostBaObservationSlot& slot = stage_observation(observation->second);
      slot.header.alive = 1;
      slot.image_slot = image->second;
      slot.point_slot = point->second;
      slot.point2D_idx = value.point2D_idx;
      slot.source_identity = identity;
      slot.xy = value.xy;
    }
    for (const uint64_t identity : observation_tombstones) {
      const auto found = graph.observation_by_identity.find(identity);
      if (found == graph.observation_by_identity.end()) continue;
      stage_observation(found->second).header.alive = 0;
    }
    for (const uint64_t id : mutation.tombstone_point_ids) {
      const auto found = graph.point_by_id.find(id);
      if (found != graph.point_by_id.end()) {
        tombstoned_point_slots.insert(found->second);
        stage_point(found->second).header.alive = 0;
      }
    }
    for (const uint32_t id : mutation.tombstone_image_ids) {
      const auto found = graph.image_by_id.find(id);
      if (found != graph.image_by_id.end()) {
        tombstoned_image_slots.insert(found->second);
        stage_image(found->second).header.alive = 0;
      }
    }

    const auto cascade_observations = [&](const uint32_t owner_slot,
                                          const bool image_owner) -> bool {
      const uint32_t head = image_owner
                                ? stage_image(owner_slot).adjacency_head
                                : stage_point(owner_slot).adjacency_head;
      const auto& nodes = image_owner ? graph.image_incidence_nodes
                                      : graph.point_incidence_nodes;
      uint32_t node_index = head;
      uint32_t original_live_count = 0;
      const uint32_t expected_live_count =
          image_owner ? graph.images[owner_slot].adjacency_count
                      : graph.points[owner_slot].adjacency_count;
      size_t steps = 0;
      while (node_index != kBaGraphInvalidSlot) {
        if (node_index >= nodes.size() || ++steps > nodes.size()) {
          valid_ = false;
          result->full_rebuild_required = true;
          return SetError("host BA graph incidence chain is corrupt", error);
        }
        ++result->adjacency_nodes_visited;
        const HostBaIncidenceNode& node = nodes[node_index];
        if (node.observation_slot >= graph.observations.size()) {
          valid_ = false;
          result->full_rebuild_required = true;
          return SetError("host BA graph incidence references invalid slot",
                          error);
        }
        HostBaObservationSlot& observation =
            stage_observation(node.observation_slot);
        const HostBaObservationSlot& original_observation =
            graph.observations[node.observation_slot];
        const bool original_owns =
            image_owner ? original_observation.image_slot == owner_slot
                        : original_observation.point_slot == owner_slot;
        if (original_observation.header.alive && original_owns &&
            node.association_generation ==
                original_observation.association_generation) {
          if (original_live_count == std::numeric_limits<uint32_t>::max()) {
            valid_ = false;
            result->full_rebuild_required = true;
            return SetError("host BA graph incidence live count overflow",
                            error);
          }
          ++original_live_count;
        }
        const bool owns = image_owner
                              ? observation.image_slot == owner_slot
                              : observation.point_slot == owner_slot;
        if (observation.header.alive && owns &&
            node.association_generation ==
                observation.association_generation) {
          observation.header.alive = 0;
          cascaded_observations.insert(node.observation_slot);
        }
        node_index = node.next_node;
      }
      if (original_live_count != expected_live_count) {
        valid_ = false;
        result->full_rebuild_required = true;
        return SetError("host BA graph incidence coverage mismatch", error);
      }
      return true;
    };
    for (const uint32_t slot : tombstoned_point_slots) {
      if (slot < graph.points.size() && graph.points[slot].header.alive &&
          !cascade_observations(slot, false)) {
        return false;
      }
    }
    for (const uint32_t slot : tombstoned_image_slots) {
      if (slot < graph.images.size() && graph.images[slot].header.alive &&
          !cascade_observations(slot, true)) {
        return false;
      }
    }
    for (const uint32_t slot : staged_observation_order) {
      HostBaObservationSlot& observation = staged_observations.at(slot);
      const HostBaImageSlot* image = image_value(observation.image_slot);
      const HostBaPointSlot* point = point_value(observation.point_slot);
      if (observation.header.alive &&
          (image == nullptr || point == nullptr || !image->header.alive ||
           !point->header.alive)) {
        observation.header.alive = 0;
        cascaded_observations.insert(slot);
      }
    }

    std::vector<HostBaIncidenceNode> image_nodes;
    std::vector<HostBaIncidenceNode> point_nodes;
    image_nodes.reserve(staged_observations.size());
    point_nodes.reserve(staged_observations.size());
    const auto adjust_image_count = [&](const uint32_t slot,
                                        const int delta) -> bool {
      HostBaImageSlot& image = stage_image(slot);
      if ((delta < 0 && image.adjacency_count == 0) ||
          (delta > 0 && image.adjacency_count ==
                            std::numeric_limits<uint32_t>::max())) {
        return SetError("host BA graph image adjacency count overflow", error);
      }
      image.adjacency_count = static_cast<uint32_t>(
          static_cast<int64_t>(image.adjacency_count) + delta);
      return true;
    };
    const auto adjust_point_count = [&](const uint32_t slot,
                                        const int delta) -> bool {
      HostBaPointSlot& point = stage_point(slot);
      if ((delta < 0 &&
           (point.adjacency_count == 0 || point.track_length == 0)) ||
          (delta > 0 &&
           (point.adjacency_count == std::numeric_limits<uint32_t>::max() ||
            point.track_length == std::numeric_limits<uint32_t>::max()))) {
        return SetError("host BA graph point adjacency count overflow", error);
      }
      point.adjacency_count = static_cast<uint32_t>(
          static_cast<int64_t>(point.adjacency_count) + delta);
      point.track_length = static_cast<uint32_t>(
          static_cast<int64_t>(point.track_length) + delta);
      return true;
    };
    for (const uint32_t slot : staged_observation_order) {
      HostBaObservationSlot& value = staged_observations.at(slot);
      HostBaObservationSlot original;
      original.header.slot = slot;
      if (slot < graph.observations.size()) original = graph.observations[slot];
      // A coalesced upsert followed by a tombstone must not publish irrelevant
      // owner/measurement changes into a dead stable slot.
      if (!value.header.alive && slot < graph.observations.size()) {
        const uint8_t alive = value.header.alive;
        value = original;
        value.header.alive = alive;
      }
      const bool association_changed =
          original.header.alive != value.header.alive ||
          (original.header.alive && value.header.alive &&
           (original.image_slot != value.image_slot ||
            original.point_slot != value.point_slot));
      if (!association_changed) continue;
      if (original.association_generation ==
          std::numeric_limits<uint64_t>::max()) {
        valid_ = false;
        result->full_rebuild_required = true;
        return SetError("host BA graph association generation exhausted",
                        error);
      }
      value.association_generation = original.association_generation + 1;
      if (original.header.alive &&
          (!adjust_image_count(original.image_slot, -1) ||
           !adjust_point_count(original.point_slot, -1))) {
        return false;
      }
      if (!value.header.alive) continue;
      if (!adjust_image_count(value.image_slot, 1) ||
          !adjust_point_count(value.point_slot, 1)) {
        return false;
      }
      HostBaImageSlot& image = stage_image(value.image_slot);
      HostBaPointSlot& point = stage_point(value.point_slot);
      if (image.lifetime_incidence_count ==
              std::numeric_limits<uint32_t>::max() ||
          point.lifetime_incidence_count ==
              std::numeric_limits<uint32_t>::max() ||
          graph.image_incidence_nodes.size() + image_nodes.size() >=
              std::numeric_limits<uint32_t>::max() ||
          graph.point_incidence_nodes.size() + point_nodes.size() >=
              std::numeric_limits<uint32_t>::max()) {
        valid_ = false;
        result->full_rebuild_required = true;
        return SetError("host BA graph incidence capacity exhausted", error);
      }
      HostBaIncidenceNode image_node;
      image_node.observation_slot = slot;
      image_node.next_node = image.adjacency_head;
      image_node.association_generation = value.association_generation;
      image.adjacency_head = static_cast<uint32_t>(
          graph.image_incidence_nodes.size() + image_nodes.size());
      ++image.lifetime_incidence_count;
      image_nodes.push_back(image_node);
      HostBaIncidenceNode point_node;
      point_node.observation_slot = slot;
      point_node.next_node = point.adjacency_head;
      point_node.association_generation = value.association_generation;
      point.adjacency_head = static_cast<uint32_t>(
          graph.point_incidence_nodes.size() + point_nodes.size());
      ++point.lifetime_incidence_count;
      point_nodes.push_back(point_node);
    }
    if (!CanAppendUint32(graph.image_incidence_nodes.size(),
                         image_nodes.size()) ||
        !CanAppendUint32(graph.point_incidence_nodes.size(),
                         point_nodes.size())) {
      return SetError("host BA graph incidence capacity exhausted", error);
    }
    if (!EnsureVectorCapacity(
            &graph.image_incidence_nodes,
            graph.image_incidence_nodes.size() + image_nodes.size(), result,
            error) ||
        !EnsureVectorCapacity(
            &graph.point_incidence_nodes,
            graph.point_incidence_nodes.size() + point_nodes.size(), result,
            error)) {
      return false;
    }

    for (const auto& staged : staged_images) {
      const HostBaImageSlot& image = staged.second;
      if (!image.header.alive) continue;
      const HostBaCameraSlot* camera = camera_value(image.camera_slot);
      if (camera == nullptr || !camera->header.alive) {
        return SetError("host BA graph staged image references dead camera",
                        error);
      }
    }

    std::vector<std::pair<uint32_t, HostBaCameraSlot>> camera_commits;
    std::vector<std::pair<uint32_t, HostBaImageSlot>> image_commits;
    std::vector<std::pair<uint32_t, HostBaPointSlot>> point_commits;
    std::vector<std::pair<uint32_t, HostBaObservationSlot>> observation_commits;
    camera_commits.reserve(staged_cameras.size());
    image_commits.reserve(staged_images.size());
    point_commits.reserve(staged_points.size());
    observation_commits.reserve(staged_observations.size());
    for (auto& staged : staged_cameras) {
      HostBaCameraSlot value = staged.second;
      const bool existed = staged.first < graph.cameras.size();
      const HostBaCameraSlot original = existed ? graph.cameras[staged.first]
                                                : HostBaCameraSlot();
      if (existed && SameCamera(original, value)) continue;
      value.header.generation = candidate_generation;
      camera_commits.emplace_back(staged.first, value);
      if (!existed)
        ++result->appended_slots;
      else if (original.header.alive && !value.header.alive)
        ++result->tombstoned_slots;
      else
        ++result->updated_slots;
    }
    for (auto& staged : staged_images) {
      HostBaImageSlot value = staged.second;
      const bool existed = staged.first < graph.images.size();
      const HostBaImageSlot original = existed ? graph.images[staged.first]
                                               : HostBaImageSlot();
      if (existed && SameImage(original, value)) continue;
      value.header.generation = candidate_generation;
      image_commits.emplace_back(staged.first, value);
      if (!existed)
        ++result->appended_slots;
      else if (original.header.alive && !value.header.alive)
        ++result->tombstoned_slots;
      else
        ++result->updated_slots;
    }
    for (auto& staged : staged_points) {
      HostBaPointSlot value = staged.second;
      const bool existed = staged.first < graph.points.size();
      const HostBaPointSlot original = existed ? graph.points[staged.first]
                                               : HostBaPointSlot();
      if (existed && SamePoint(original, value)) continue;
      value.header.generation = candidate_generation;
      point_commits.emplace_back(staged.first, value);
      if (!existed)
        ++result->appended_slots;
      else if (original.header.alive && !value.header.alive)
        ++result->tombstoned_slots;
      else
        ++result->updated_slots;
    }
    for (const uint32_t slot : staged_observation_order) {
      HostBaObservationSlot value = staged_observations.at(slot);
      const bool existed = slot < graph.observations.size();
      const HostBaObservationSlot original =
          existed ? graph.observations[slot] : HostBaObservationSlot();
      if (existed && SameObservation(original, value)) continue;
      value.header.generation = candidate_generation;
      observation_commits.emplace_back(slot, value);
      if (!existed)
        ++result->appended_slots;
      else if (original.header.alive && !value.header.alive)
        ++result->tombstoned_slots;
      else
        ++result->updated_slots;
      if (cascaded_observations.count(slot) != 0)
        ++result->cascaded_observation_tombstones;
    }
    result->touched_entity_records = camera_commits.size() +
                                     image_commits.size() +
                                     point_commits.size();
    result->touched_observation_records = observation_commits.size();
    result->adjacency_nodes_appended = image_nodes.size() + point_nodes.size();
    const bool semantic = !camera_commits.empty() || !image_commits.empty() ||
                          !point_commits.empty() ||
                          !observation_commits.empty() ||
                          !image_nodes.empty() || !point_nodes.empty();
    if (!semantic) {
      graph.topology_revision = mutation.revision_after;
      result->revision_after = graph.topology_revision;
      result->generation_after = graph.generation;
      result->semantic_noop = true;
      result->published = true;
      accepted_batch_failure.Disarm();
      return true;
    }

    const auto commit_records = [](const auto& commits, auto* destination) {
      for (const auto& commit : commits) {
        while (destination->size() <= commit.first)
          destination->push_back(typename std::decay<decltype(
              destination->front())>::type());
        (*destination)[commit.first] = commit.second;
      }
    };
    commit_records(camera_commits, &graph.cameras);
    commit_records(image_commits, &graph.images);
    commit_records(point_commits, &graph.points);
    commit_records(observation_commits, &graph.observations);
    for (const HostBaIncidenceNode& node : image_nodes)
      graph.image_incidence_nodes.push_back(node);
    for (const HostBaIncidenceNode& node : point_nodes)
      graph.point_incidence_nodes.push_back(node);
    graph.topology_revision = mutation.revision_after;
    graph.generation = candidate_generation;
    graph.resident_bytes = EstimateResidentBytes(graph);
    next_generation_ = candidate_generation;
    new_keys.keep = true;
    result->revision_after = graph.topology_revision;
    result->generation_after = graph.generation;
    result->published = true;
    accepted_batch_failure.Disarm();
    return true;
  } catch (const std::bad_alloc&) {
    return SetError("host BA graph mutation allocation failed", error);
  } catch (const std::length_error&) {
    return SetError("host BA graph mutation size overflow", error);
  }
}

CatalogReadLease HostBaGraphStore::AcquireReadLease() const noexcept {
  if (gate_ == nullptr) return CatalogReadLease();
  try {
    std::shared_ptr<HostBaGraphReadGuard> guard =
        std::make_shared<HostBaGraphReadGuard>(gate_);
    if (!guard->owns_lock() || !valid_ || publication_ == nullptr)
      return CatalogReadLease();
    return CatalogReadLease(publication_, std::move(guard));
  } catch (...) {
    return CatalogReadLease();
  }
}
bool HostBaGraphStore::IsCurrent(const CatalogReadLease& lease) const noexcept {
  if (!lease.valid() || gate_ == nullptr || lease.guard_ == nullptr ||
      lease.guard_->gate.get() != gate_.get()) {
    return false;
  }
  // A same-gate lease owns the shared lock, so these mutable fields are stable
  // without recursively acquiring shared_timed_mutex.
  return valid_ && publication_ != nullptr &&
         lease.publication_ == publication_ &&
         lease.generation() == publication_->generation &&
         lease.topology_revision() == publication_->topology_revision;
}

namespace {

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
  const uint8_t loss_mode = static_cast<uint8_t>(intent.config.loss_mode);

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
      loss_mode == 0 ||
      !std::isfinite(intent.config.loss_scale) ||
      (loss_mode == 2 && intent.config.loss_scale <= 0.0) ||
      (loss_mode == 1 && intent.config.loss_scale < 0.0) ||
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
  const auto image_incidence = catalog.image_incidence_nodes();
  const auto point_incidence = catalog.point_incidence_nodes();

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

  std::unordered_map<uint32_t, const TranslationSubsetPolicy*>
      translation_policy_by_slot;
  std::unordered_map<uint32_t, const CameraParameterPolicy*>
      camera_policy_by_slot;
  std::unordered_map<uint32_t, const PointFixedPolicy*> point_policy_by_slot;
  std::unordered_set<uint32_t> explicit_variable_point_slots;
  std::unordered_set<uint32_t> explicit_constant_point_slots;
  std::unordered_set<uint32_t> boundary_image_slots;
  translation_policy_by_slot.reserve(intent.translation_subsets.size());
  camera_policy_by_slot.reserve(intent.camera_policies.size());
  point_policy_by_slot.reserve(intent.point_policies.size());
  explicit_variable_point_slots.reserve(
      intent.explicit_variable_point_slots.size());
  explicit_constant_point_slots.reserve(
      intent.explicit_constant_point_slots.size());
  boundary_image_slots.reserve(intent.explicit_variable_point_slots.size() +
                               intent.explicit_constant_point_slots.size());
  for (const TranslationSubsetPolicy& policy : intent.translation_subsets) {
    if (!translation_policy_by_slot.emplace(policy.image_slot, &policy).second)
      return SetError("duplicate native translation policy", error);
  }
  for (const CameraParameterPolicy& policy : intent.camera_policies) {
    if (!camera_policy_by_slot.emplace(policy.camera_slot, &policy).second)
      return SetError("duplicate native camera policy", error);
  }
  for (const PointFixedPolicy& policy : intent.point_policies) {
    if (!point_policy_by_slot.emplace(policy.point_slot, &policy).second)
      return SetError("duplicate native point policy", error);
  }
  for (const uint32_t slot : intent.explicit_variable_point_slots) {
    if (!explicit_variable_point_slots.insert(slot).second)
      return SetError("duplicate native explicit variable point", error);
  }
  for (const uint32_t slot : intent.explicit_constant_point_slots) {
    if (explicit_variable_point_slots.count(slot) != 0 ||
        !explicit_constant_point_slots.insert(slot).second) {
      return SetError("duplicate or conflicting native explicit constant point",
                      error);
    }
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
    } else if (!boundary && boundary_image_slots.count(slot) != 0) {
      return SetError("native solve image is both active and boundary", error);
    }
    if (boundary) boundary_image_slots.insert(slot);
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
  const auto visit_incidence = [&](const uint32_t head,
                                   const uint32_t expected_live_count,
                                   const uint32_t owner_slot,
                                   const bool image_owner,
                                   const auto& visitor) -> bool {
    const auto nodes = image_owner ? image_incidence : point_incidence;
    uint32_t node_index = head;
    uint32_t live_count = 0;
    size_t steps = 0;
    while (node_index != kBaGraphInvalidSlot) {
      if (node_index >= nodes.size || ++steps > nodes.size) {
        return SetError("native incidence chain is corrupt", error);
      }
      ++runtime->catalog_observation_visits;
      const HostBaIncidenceNode& node = nodes[node_index];
      if (node.observation_slot >= observations.size) {
        return SetError("native incidence references an invalid observation",
                        error);
      }
      const HostBaObservationSlot& observation =
          observations[node.observation_slot];
      const bool owns = image_owner
                            ? observation.image_slot == owner_slot
                            : observation.point_slot == owner_slot;
      if (observation.header.alive && owns &&
          node.association_generation == observation.association_generation) {
        if (live_count == std::numeric_limits<uint32_t>::max()) {
          return SetError("native incidence live count overflow", error);
        }
        ++live_count;
        if (!visitor(node.observation_slot)) return false;
      }
      node_index = node.next_node;
    }
    if (live_count != expected_live_count) {
      return SetError("native incidence live count mismatch", error);
    }
    return true;
  };

  for (const uint32_t image_slot : intent.active_image_slots) {
    if (image_slot >= images.size || image_active_stamps_[image_slot] == stamp) {
      return SetError("native active image selection is invalid or duplicated",
                      error);
    }
    if (!activate_image(image_slot, false)) return false;
    if (!intent.active_visual_observation_slots_explicit) {
      const HostBaImageSlot& image = images[image_slot];
      if (!visit_incidence(
              image.adjacency_head, image.adjacency_count, image_slot, true,
              [&](const uint32_t observation_slot) {
                return select_observation(observation_slot, false);
              })) {
        return false;
      }
    }
  }
  if (intent.active_visual_observation_slots_explicit) {
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
    if (!visit_incidence(
            point.adjacency_head, point.adjacency_count, point_slot, false,
            [&](const uint32_t observation_slot) {
              const uint32_t image_slot =
                  observations[observation_slot].image_slot;
              const bool boundary =
                  boundary_image_slots.count(image_slot) != 0 ||
                  image_active_stamps_[image_slot] != stamp;
              return select_observation(observation_slot, boundary);
            })) {
      return false;
    }
  }
  for (const uint32_t point_slot : intent.explicit_constant_point_slots) {
    if (!activate_point(point_slot)) return false;
    const HostBaPointSlot& point = points[point_slot];
    if (!visit_incidence(
            point.adjacency_head, point.adjacency_count, point_slot, false,
            [&](const uint32_t observation_slot) {
              const uint32_t image_slot =
                  observations[observation_slot].image_slot;
              const bool boundary =
                  boundary_image_slots.count(image_slot) != 0 ||
                  image_active_stamps_[image_slot] != stamp;
              return select_observation(observation_slot, boundary);
            })) {
      return false;
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
        boundary_image_slots.count(policy.image_slot) != 0 ||
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
    const auto translation_found = translation_policy_by_slot.find(slot);
    const TranslationSubsetPolicy* translation =
        translation_found == translation_policy_by_slot.end()
            ? nullptr : translation_found->second;
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
    const auto policy_found = camera_policy_by_slot.find(slot);
    const CameraParameterPolicy* policy =
        policy_found == camera_policy_by_slot.end() ? nullptr
                                                    : policy_found->second;
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
    const auto policy_found = point_policy_by_slot.find(slot);
    const PointFixedPolicy* policy =
        policy_found == point_policy_by_slot.end() ? nullptr
                                                   : policy_found->second;
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
    if (explicit_variable_point_slots.count(slot) != 0) {
      if (policy != nullptr && policy->constant) {
        return SetError(
            "native point is both explicitly variable and constant", error);
      }
      result.constant = 0;
    }
    if (explicit_constant_point_slots.count(slot) != 0) {
      if (policy == nullptr || !policy->constant) {
        return SetError(
            "native explicit constant point lacks a constant policy", error);
      }
      result.constant = 1;
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
      if (observation.image_slot >= images.size) {
        return SetError("native residual image slot is invalid", error);
      }
      const uint32_t camera_slot = images[observation.image_slot].camera_slot;
      const CameraFixedPolicyResult* camera =
          camera_policy(camera_slot);
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
      camera_parameter.entity_slot = camera_slot;
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
    double input_norm2 = 0.0;
    for (const double value : state.quaternion) input_norm2 += value * value;
    const bool fixed_input = fixed_pose_slots.count(slot) != 0 ||
                             boundary_image_slots.count(slot) != 0;
    if (!IsFiniteArray(state.quaternion) || !IsFiniteArray(state.translation) ||
        (fixed_input &&
         (!std::isfinite(input_norm2) ||
          std::abs(std::sqrt(input_norm2) - 1.0) > 1e-12)) ||
        !NormalizeQuaternion(state.quaternion, &state.quaternion)) {
      return SetError(fixed_input
                          ? "native fixed quaternion is not normalized"
                          : "native image state normalization failed",
                      error);
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
