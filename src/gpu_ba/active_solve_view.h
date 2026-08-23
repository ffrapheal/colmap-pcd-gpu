#ifndef COLMAP_SRC_GPU_BA_ACTIVE_SOLVE_VIEW_H_
#define COLMAP_SRC_GPU_BA_ACTIVE_SOLVE_VIEW_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_ba/custom_cuda.h"

namespace colmap {
namespace gpu_ba {

struct HostIndexedCatalogData;

constexpr uint32_t kMapperStaticCatalogAbiVersion = 2;
constexpr uint32_t kIndexedActiveSolveAbiVersion = 1;
constexpr uint32_t kIndexedInvalidSlot = 0xffffffffu;

struct MapperCatalogSlotHeader {
  uint32_t slot = kIndexedInvalidSlot;
  uint32_t reserved = 0;
  uint64_t generation = 0;
  uint8_t alive = 0;
  uint8_t padding[7]{};
};

struct MapperCatalogCameraSlot {
  MapperCatalogSlotHeader header;
  uint32_t camera_id = 0;
  int32_t model_id = -1;
  uint64_t width = 0;
  uint64_t height = 0;
  uint32_t parameter_count = 0;
  uint32_t reserved = 0;
};

struct MapperCatalogImageSlot {
  MapperCatalogSlotHeader header;
  uint32_t image_id = 0;
  uint32_t camera_slot = kIndexedInvalidSlot;
  uint32_t observation_head = kIndexedInvalidSlot;
  uint32_t observation_tail = kIndexedInvalidSlot;
  uint32_t observation_count = 0;
  uint8_t registered = 0;
  uint8_t padding[3]{};
};

struct MapperCatalogPointSlot {
  MapperCatalogSlotHeader header;
  uint64_t point3D_id = 0;
  uint32_t observation_head = kIndexedInvalidSlot;
  uint32_t observation_tail = kIndexedInvalidSlot;
  uint32_t observation_count = 0;
  uint32_t track_length = 0;
};

struct MapperCatalogObservationSlot {
  MapperCatalogSlotHeader header;
  uint32_t image_slot = kIndexedInvalidSlot;
  uint32_t point_slot = kIndexedInvalidSlot;
  uint32_t camera_slot = kIndexedInvalidSlot;
  uint32_t point2D_idx = 0;
  uint64_t source_identity = 0;
  std::array<double, 2> xy{{0.0, 0.0}};
  uint32_t image_incidence = kIndexedInvalidSlot;
  uint32_t point_incidence = kIndexedInvalidSlot;
};

// Append-only linked incidence lets a journal patch add an observation with
// O(1) device writes. Tombstoned observations remain in the chain and are
// skipped through the observation's alive bit.
struct MapperCatalogIncidenceSlot {
  uint32_t observation_slot = kIndexedInvalidSlot;
  uint32_t next = kIndexedInvalidSlot;
  uint64_t generation = 0;
};

struct MapperStaticCatalogTombstones {
  std::vector<uint32_t> cameras;
  std::vector<uint32_t> images;
  std::vector<uint64_t> points;
  std::vector<std::array<uint32_t, 2>> observations;
};

struct MapperStaticCatalogUpdateResult {
  uint64_t generation_before = 0;
  uint64_t generation_after = 0;
  uint64_t appended_cameras = 0;
  uint64_t appended_images = 0;
  uint64_t appended_points = 0;
  uint64_t appended_observations = 0;
  uint64_t appended_incidences = 0;
  uint64_t updated_incidence_links = 0;
  uint64_t tombstoned_cameras = 0;
  uint64_t tombstoned_images = 0;
  uint64_t tombstoned_points = 0;
  uint64_t tombstoned_observations = 0;
  uint64_t full_graph_records_scanned = 0;
  uint64_t estimated_impl_copy_bytes = 0;
  bool content_changed = false;
  bool full_rebuild = false;
};

struct MapperStaticCatalogStableTables {
  uint32_t abi_version = kMapperStaticCatalogAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  std::vector<MapperCatalogCameraSlot> cameras;
  std::vector<MapperCatalogImageSlot> images;
  std::vector<MapperCatalogPointSlot> points;
  std::vector<MapperCatalogObservationSlot> observations;
  std::vector<MapperCatalogIncidenceSlot> image_incidences;
  std::vector<MapperCatalogIncidenceSlot> point_incidences;
};

// Mapper-owner serialized catalog. Slots never move and local active-set
// changes only select existing slots. Dynamic state is deliberately absent.
class MapperStaticProblemDataCatalog {
 public:
  struct Impl;

  explicit MapperStaticProblemDataCatalog(uint64_t owner_epoch,
                                          uint64_t generation_floor = 0);
  MapperStaticProblemDataCatalog(
      const MapperStaticProblemDataCatalog& other);
  ~MapperStaticProblemDataCatalog();
  MapperStaticProblemDataCatalog& operator=(
      const MapperStaticProblemDataCatalog&) = delete;

  uint64_t owner_epoch() const noexcept;
  uint64_t revision() const noexcept;
  uint64_t generation() const noexcept;
  uint64_t camera_slot_count() const noexcept;
  uint64_t image_slot_count() const noexcept;
  uint64_t point_slot_count() const noexcept;
  uint64_t observation_slot_count() const noexcept;
  uint64_t image_incidence_count() const noexcept;
  uint64_t point_incidence_count() const noexcept;
  uint64_t EstimatedResidentBytes() const noexcept;

  const MapperCatalogCameraSlot* FindCamera(uint32_t id) const noexcept;
  const MapperCatalogImageSlot* FindImage(uint32_t id) const noexcept;
  const MapperCatalogPointSlot* FindPoint(uint64_t id) const noexcept;
  const MapperCatalogObservationSlot* FindObservation(
      uint32_t image_id, uint32_t point2D_idx) const noexcept;

  bool ExportStableTables(MapperStaticCatalogStableTables* output,
                          std::string* error) const;

  // Legacy/testing helper for scalar catalog fixtures. Indexed Mapper
  // production cold-builds from HostIndexedCatalogData and never seeds the
  // catalog from an active SolveSpec subset.
  bool EnsureProblemStaticData(
      uint64_t owner_epoch,
      uint64_t revision,
      const CudaSolveProblem& problem,
      MapperStaticCatalogUpdateResult* result,
      std::string* error);

  bool ApplyDelta(uint64_t owner_epoch,
                  uint64_t revision,
                  const CudaSolveProblem& appended_problem,
                  const MapperStaticCatalogTombstones& tombstones,
                  MapperStaticCatalogUpdateResult* result,
                  std::string* error);

  // Reconciles the complete committed Reconstruction projection after the
  // existing structure journal has been applied. Stable slots are never
  // reused: removed/changed identities are tombstoned before replacements are
  // appended. The internal full Impl copy is intentionally reported as a
  // remaining performance debt until the delta patch representation lands.
  bool ReconcileFullStaticData(
      const HostIndexedCatalogData& data,
      MapperStaticCatalogUpdateResult* result,
      std::string* error);

 private:
  std::unique_ptr<Impl> impl_;
};

struct IndexedActiveSolveConfig {
  // The indexed path only accepts a controller-resolved configuration. These
  // defaults are non-operative compatibility values, not indexed defaults.
  bool effective_config_resolved = false;
  uint64_t config_generation = 1;
  CudaArithmeticPrecision arithmetic_precision =
      CudaArithmeticPrecision::kCompatibilityDefault;
  CudaHessianAssemblyBackend hessian_backend =
      CudaHessianAssemblyBackend::kCompatibilityDefault;
  CudaSchurContributionBackend schur_backend =
      CudaSchurContributionBackend::kCompatibilityDefault;
  CudaHotKernelMode hot_kernel = CudaHotKernelMode::kCompatibilityDefault;
  CudaExecutionProfile execution_profile = CudaExecutionProfile::kBaseline;
  CudaLossMode loss_mode = CudaLossMode::kFromSnapshot;
  double loss_scale = 1.0;
};

struct IndexedActiveSolveIdentity {
  uint32_t abi_version = kIndexedActiveSolveAbiVersion;
  uint32_t catalog_abi_version = kMapperStaticCatalogAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  uint64_t solve_view_generation = 0;
  uint64_t config_generation = 0;
};

struct IndexedCameraState {
  uint32_t catalog_slot = kIndexedInvalidSlot;
  uint32_t camera_id = 0;
  uint8_t constant = 1;
  uint8_t padding[3]{};
  std::vector<double> parameters;
};

struct IndexedImageState {
  uint32_t catalog_slot = kIndexedInvalidSlot;
  uint32_t image_id = 0;
  uint32_t camera_slot = kIndexedInvalidSlot;
  int32_t pose_index = -1;
  uint32_t schur_offset = 0;
  uint32_t pose_dimension = 0;
  std::array<int32_t, 3> free_translation_indices{{-1, -1, -1}};
  uint8_t selected = 0;
  uint8_t pose_constant = 1;
  uint8_t has_pose_parameter_blocks = 0;
  uint8_t constant_tvec_mask = 0;
  std::array<double, 4> quaternion{{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> translation{{0.0, 0.0, 0.0}};
};

struct IndexedPointState {
  uint32_t catalog_slot = kIndexedInvalidSlot;
  uint32_t reserved = 0;
  uint64_t point3D_id = 0;
  int32_t point_index = -1;
  uint8_t constant = 0;
  uint8_t config_role = 0;
  uint8_t has_search_range = 0;
  uint8_t padding = 0;
  double search_range = 0.0;
  std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
};

struct IndexedVisualResidual {
  uint64_t source_index = 0;
  uint32_t observation_slot = kIndexedInvalidSlot;
  uint32_t image_slot = kIndexedInvalidSlot;
  uint32_t point_slot = kIndexedInvalidSlot;
  uint32_t camera_slot = kIndexedInvalidSlot;
  int32_t pose_index = -1;
  int32_t point_index = -1;
};

struct IndexedLidarConstraint {
  uint64_t source_index = 0;
  uint64_t point3D_id = 0;
  uint32_t point_slot = kIndexedInvalidSlot;
  int32_t point_index = -1;
  uint8_t lidar_type = 0;
  uint8_t has_search_range = 0;
  uint8_t padding[2]{};
  double search_range = 0.0;
  double weight = 0.0;
  std::array<double, 3> lidar_xyz{{0.0, 0.0, 0.0}};
  std::array<double, 4> plane{{0.0, 0.0, 0.0, 0.0}};
};

struct IndexedActiveSolveRuntime {
  uint64_t descriptor_calls = 0;
  uint64_t selection_passes = 0;
  uint64_t selected_visual_residuals = 0;
  uint64_t selected_lidar_residuals = 0;
  uint64_t selected_cameras = 0;
  uint64_t selected_images = 0;
  uint64_t selected_points = 0;
  uint64_t descriptor_bytes = 0;
  double dynamic_state_wall_milliseconds = 0.0;
  double wall_milliseconds = 0.0;
};

// Solve-owned and precision-neutral. It contains FP64 dynamic entity state;
// the device context selects FP64 or fp32_mixed residual/workspace records.
// No CPU CostLayout, Hessian adjacency, Schur pair, chunk, or segment data is
// carried by this descriptor.
struct IndexedActiveSolveDescriptor {
  IndexedActiveSolveIdentity identity;
  IndexedActiveSolveConfig config;
  SnapshotMetadata metadata;
  uint32_t pose_dimension = 0;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t ambient_parameter_count = 0;
  uint64_t effective_parameter_count = 0;
  std::vector<IndexedCameraState> cameras;
  std::vector<IndexedImageState> images;
  std::vector<IndexedPointState> points;
  std::vector<IndexedVisualResidual> visual;
  std::vector<IndexedLidarConstraint> lidar;
  IndexedActiveSolveRuntime runtime;
};

bool ValidateIndexedCatalogSolveBinding(
    const MapperStaticCatalogStableTables& tables,
    const IndexedActiveSolveDescriptor& descriptor,
    const IndexedActiveSolveConfig& expected_config,
    std::string* error) noexcept;

class IndexedActiveSolveMaterializer {
 public:
  IndexedActiveSolveMaterializer() = default;
  IndexedActiveSolveMaterializer(const IndexedActiveSolveMaterializer&) =
      delete;
  IndexedActiveSolveMaterializer& operator=(
      const IndexedActiveSolveMaterializer&) = delete;

  bool Materialize(uint64_t owner_epoch,
                   const MapperStaticProblemDataCatalog& catalog,
                   const CudaSolveProblem& problem,
                   const IndexedActiveSolveConfig& config,
                   IndexedActiveSolveDescriptor* output,
                   std::string* error);

 private:
  uint64_t next_view_generation_ = 0;
  uint64_t scratch_generation_ = 0;
  std::vector<uint64_t> camera_slot_stamps_;
  std::vector<uint64_t> image_slot_stamps_;
  std::vector<uint64_t> point_slot_stamps_;
  std::vector<int32_t> camera_state_by_slot_;
  std::vector<int32_t> image_state_by_slot_;
  std::vector<int32_t> point_state_by_slot_;
};

struct DeviceCatalogPhysicalIdentity {
  int32_t device_ordinal = -1;
  uint64_t context_incarnation = 0;
  uint32_t catalog_abi_version = kMapperStaticCatalogAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  uint64_t arena_generation = 0;

  bool valid() const noexcept;
  bool operator==(const DeviceCatalogPhysicalIdentity& other) const noexcept;
};

struct DeviceCatalogSlice {
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint64_t elements = 0;
};

struct DeviceCatalogLayoutSpec {
  uint64_t arena_capacity = 0;
  uint64_t base_offset = 0;
  uint64_t alignment = 256;
};

struct DeviceCatalogLayout {
  DeviceCatalogSlice cameras;
  DeviceCatalogSlice images;
  DeviceCatalogSlice points;
  DeviceCatalogSlice observations;
  DeviceCatalogSlice image_incidences;
  DeviceCatalogSlice point_incidences;
  uint64_t alignment = 0;
  uint64_t begin = 0;
  uint64_t end = 0;
};

bool PlanPersistentDeviceCatalogLayout(
    const MapperStaticCatalogStableTables& tables,
    const DeviceCatalogLayoutSpec& spec,
    DeviceCatalogLayout* output,
    std::string* error);

// RuntimePool metadata stores arena offsets, never cross-generation pointers.
class PersistentDeviceCatalogState {
 public:
  bool Publish(const DeviceCatalogPhysicalIdentity& identity,
               const DeviceCatalogLayout& layout,
               std::string* error);
  bool CanReuse(const DeviceCatalogPhysicalIdentity& identity) const noexcept;
  bool CanReuseStorage(
      const DeviceCatalogPhysicalIdentity& identity) const noexcept;
  void Invalidate() noexcept;
  bool valid() const noexcept;
  const DeviceCatalogPhysicalIdentity& identity() const noexcept;
  const DeviceCatalogLayout& layout() const noexcept;

 private:
  bool valid_ = false;
  DeviceCatalogPhysicalIdentity identity_;
  DeviceCatalogLayout layout_;
};

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_ACTIVE_SOLVE_VIEW_H_
