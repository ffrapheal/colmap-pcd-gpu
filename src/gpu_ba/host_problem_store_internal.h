#ifndef COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_INTERNAL_H_
#define COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_INTERNAL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {

class Reconstruction;

namespace gpu_ba {

struct HostCatalogCamera {
  uint32_t camera_id = 0;
  int32_t model_id = -1;
  uint64_t width = 0;
  uint64_t height = 0;
  size_t parameter_count = 0;
};

struct HostCatalogImage {
  uint32_t image_id = 0;
  uint32_t camera_id = 0;
  bool registered = false;
};

struct HostCatalogTrackElement {
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
};

struct HostCatalogPoint {
  uint64_t point3D_id = 0;
  std::vector<HostCatalogTrackElement> track;
};

struct HostCatalogObservation {
  uint64_t point3D_id = 0;
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
  std::array<double, 2> xy{{0.0, 0.0}};
};

static_assert(std::is_standard_layout<HostCatalogCamera>::value &&
                  std::is_trivially_copyable<HostCatalogCamera>::value,
              "catalog camera bridge record must be scalar POD");
static_assert(std::is_standard_layout<HostCatalogImage>::value &&
                  std::is_trivially_copyable<HostCatalogImage>::value,
              "catalog image bridge record must be scalar POD");
static_assert(std::is_standard_layout<HostCatalogTrackElement>::value &&
                  std::is_trivially_copyable<HostCatalogTrackElement>::value,
              "catalog track bridge record must be scalar POD");
static_assert(std::is_standard_layout<HostCatalogObservation>::value &&
                  std::is_trivially_copyable<HostCatalogObservation>::value,
              "catalog observation bridge record must be scalar POD");

// Immutable catalog nodes form a persistent overlay chain. They contain only
// host-owned scalar/STL records and are safe to consume from a CUDA translation
// unit without exposing Eigen-backed Reconstruction entities there.
struct StaticProblemDataCatalog {
  uint64_t owner_epoch = 0;
  uint64_t revision = 0;
  uint64_t generation = 0;
  uint32_t overlay_depth = 0;
  uint64_t resident_bytes = 0;
  std::shared_ptr<const StaticProblemDataCatalog> parent;
  std::unordered_map<uint32_t, HostCatalogCamera> cameras;
  std::unordered_map<uint32_t, HostCatalogImage> images;
  std::unordered_map<uint64_t, HostCatalogPoint> points;
  std::unordered_map<uint64_t, HostCatalogObservation> observations;
  std::unordered_set<uint32_t> deleted_cameras;
  std::unordered_set<uint32_t> deleted_images;
  std::unordered_set<uint64_t> deleted_points;
  std::unordered_set<uint64_t> deleted_observations;
};

enum class HostStructureEventKind : uint8_t {
  kImageRegistrationChanged = 0,
  kCameraAssociationChanged = 1,
  kCameraAdded = 2,
  kImageAdded = 3,
  kPointAdded = 4,
  kPointDeleted = 5,
  kPointMerged = 6,
  kObservationAdded = 7,
  kObservationDeleted = 8,
  kBulkUnknown = 9,
};

struct HostStructureEvent {
  HostStructureEventKind kind = HostStructureEventKind::kBulkUnknown;
  uint32_t image_id = std::numeric_limits<uint32_t>::max();
  uint32_t old_camera_id = std::numeric_limits<uint32_t>::max();
  uint32_t new_camera_id = std::numeric_limits<uint32_t>::max();
  uint64_t point3D_id = std::numeric_limits<uint64_t>::max();
  uint64_t old_point3D_id1 = std::numeric_limits<uint64_t>::max();
  uint64_t old_point3D_id2 = std::numeric_limits<uint64_t>::max();
  uint32_t point2D_idx = std::numeric_limits<uint32_t>::max();
  bool old_registration = false;
  bool new_registration = false;
  std::vector<HostCatalogTrackElement> old_track;
  std::string reason;
};

struct HostStructureBatch {
  uint64_t revision = 0;
  std::vector<HostStructureEvent> events;
};

struct HostStructureReadResult {
  bool complete = false;
  bool gap = false;
  uint64_t owner_epoch = 0;
  uint64_t cursor = 0;
  uint64_t current_revision = 0;
  uint64_t oldest_retained_revision = 0;
  std::vector<HostStructureBatch> batches;
};

inline uint64_t HostCatalogObservationKey(const uint32_t image_id,
                                          const uint32_t point2D_idx) {
  return (static_cast<uint64_t>(image_id) << 32) |
         static_cast<uint64_t>(point2D_idx);
}

template <typename Key, typename Value>
const Value* FindHostCatalogValue(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    const Key key,
    const std::unordered_map<Key, Value> StaticProblemDataCatalog::*values,
    const std::unordered_set<Key> StaticProblemDataCatalog::*deleted) {
  for (auto node = root; node != nullptr; node = node->parent) {
    if ((node.get()->*deleted).count(key) != 0) return nullptr;
    const auto& map = node.get()->*values;
    const auto it = map.find(key);
    if (it != map.end()) return &it->second;
  }
  return nullptr;
}

inline const HostCatalogCamera* FindCatalogCamera(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    const uint32_t id) {
  return FindHostCatalogValue(root, id, &StaticProblemDataCatalog::cameras,
                              &StaticProblemDataCatalog::deleted_cameras);
}

inline const HostCatalogImage* FindCatalogImage(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    const uint32_t id) {
  return FindHostCatalogValue(root, id, &StaticProblemDataCatalog::images,
                              &StaticProblemDataCatalog::deleted_images);
}

inline const HostCatalogPoint* FindCatalogPoint(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    const uint64_t id) {
  return FindHostCatalogValue(root, id, &StaticProblemDataCatalog::points,
                              &StaticProblemDataCatalog::deleted_points);
}

inline const HostCatalogObservation* FindCatalogObservation(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    const uint32_t image_id,
    const uint32_t point2D_idx) {
  return FindHostCatalogValue(
      root, HostCatalogObservationKey(image_id, point2D_idx),
      &StaticProblemDataCatalog::observations,
      &StaticProblemDataCatalog::deleted_observations);
}

uint64_t HostReconstructionOwnerEpoch(
    const Reconstruction* reconstruction) noexcept;
uint64_t HostReconstructionStructureRevision(
    const Reconstruction* reconstruction) noexcept;

bool ReadHostStructureEventsSince(
    const Reconstruction* reconstruction,
    uint64_t owner_epoch,
    uint64_t cursor,
    HostStructureReadResult* result,
    std::string* error);

bool BuildStaticProblemDataCatalog(
    const Reconstruction* reconstruction,
    uint64_t owner_epoch,
    uint64_t revision,
    uint64_t generation,
    std::shared_ptr<const StaticProblemDataCatalog>* output,
    std::string* error);

bool ApplyStructureJournalToCatalog(
    const Reconstruction* reconstruction,
    const HostStructureReadResult& journal,
    uint64_t generation,
    const std::shared_ptr<const StaticProblemDataCatalog>& base,
    std::shared_ptr<const StaticProblemDataCatalog>* output,
    std::string* error);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_INTERNAL_H_
