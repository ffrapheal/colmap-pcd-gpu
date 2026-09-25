#ifndef COLMAP_SRC_GPU_BA_HOST_BA_GRAPH_H_
#define COLMAP_SRC_GPU_BA_HOST_BA_GRAPH_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_ba/ba_types.h"
#include "gpu_ba/linearization.h"

namespace colmap {
namespace gpu_ba {

enum class CudaArithmeticPrecision : uint8_t;
enum class CudaAuditProfile : uint8_t;
enum class CudaCurrentLinearizationCacheMode : uint8_t;
enum class CudaDeviceContextMode : uint8_t;
enum class CudaExecutionProfile : uint8_t;
enum class CudaHessianAssemblyBackend : uint8_t;
enum class CudaHotKernelMode : uint8_t;
enum class CudaLossMode : uint8_t;
enum class CudaMemoryMode : uint8_t;
enum class CudaReductionMode : uint8_t;
enum class CudaResidualOrder : uint8_t;
enum class CudaSchurContributionBackend : uint8_t;
struct CudaFullLmOptions;
class DeviceBaProblemStoreHandle;

constexpr uint32_t kHostBaGraphAbiVersion = 2;
constexpr uint32_t kNativeHostSolveViewAbiVersion = 2;
constexpr uint32_t kBaGraphInvalidSlot = 0xffffffffu;
constexpr uint64_t kBaGraphInvalidAssociationId = 0xffffffffffffffffull;

template <typename T>
struct BaArrayView {
  const T* data = nullptr;
  size_t size = 0;

  const T& operator[](const size_t index) const { return data[index]; }
  const T* begin() const noexcept { return data; }
  const T* end() const noexcept { return data == nullptr ? nullptr : data + size; }
  bool empty() const noexcept { return size == 0; }
};

struct HostBaGraphSlotHeader {
  uint32_t slot = kBaGraphInvalidSlot;
  uint32_t reserved = 0;
  uint64_t generation = 0;
  uint8_t alive = 0;
  uint8_t padding[7]{};
};

struct HostBaCameraSlot {
  HostBaGraphSlotHeader header;
  uint32_t camera_id = 0;
  int32_t model_id = -1;
  uint64_t width = 0;
  uint64_t height = 0;
  uint32_t parameter_count = 0;
  uint32_t reserved = 0;
};

struct HostBaImageSlot {
  HostBaGraphSlotHeader header;
  uint32_t image_id = 0;
  uint32_t camera_slot = kBaGraphInvalidSlot;
  uint32_t adjacency_head = kBaGraphInvalidSlot;
  uint32_t adjacency_count = 0;
  // Monotonic number of incidence nodes ever appended for this slot. Current
  // solve semantics use adjacency_count, never this lifetime diagnostic.
  uint32_t lifetime_incidence_count = 0;
  uint8_t registered = 0;
  uint8_t padding[3]{};
};

struct HostBaPointSlot {
  HostBaGraphSlotHeader header;
  uint64_t point3D_id = 0;
  uint32_t adjacency_head = kBaGraphInvalidSlot;
  uint32_t adjacency_count = 0;
  uint32_t lifetime_incidence_count = 0;
  uint32_t track_length = 0;
};

struct HostBaObservationSlot {
  HostBaGraphSlotHeader header;
  uint32_t image_slot = kBaGraphInvalidSlot;
  uint32_t point_slot = kBaGraphInvalidSlot;
  uint32_t point2D_idx = 0;
  uint64_t source_identity = 0;
  uint64_t association_generation = 0;
  std::array<double, 2> xy{{0.0, 0.0}};
};

struct HostBaIncidenceNode {
  uint32_t observation_slot = kBaGraphInvalidSlot;
  uint32_t next_node = kBaGraphInvalidSlot;
  uint64_t association_generation = 0;
};

struct HostBaCameraRecord {
  uint32_t camera_id = 0;
  int32_t model_id = -1;
  uint64_t width = 0;
  uint64_t height = 0;
  uint32_t parameter_count = 0;
};

struct HostBaImageRecord {
  uint32_t image_id = 0;
  uint32_t camera_id = 0;
  bool registered = false;
};

struct HostBaPointRecord {
  uint64_t point3D_id = 0;
};

struct HostBaObservationRecord {
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
  uint64_t point3D_id = 0;
  std::array<double, 2> xy{{0.0, 0.0}};
};

struct HostBaObservationKey {
  uint32_t image_id = 0;
  uint32_t point2D_idx = 0;
};

struct HostBaGraphColdInput {
  uint64_t owner_epoch = 0;
  uint64_t topology_revision = 0;
  std::vector<HostBaCameraRecord> cameras;
  std::vector<HostBaImageRecord> images;
  std::vector<HostBaPointRecord> points;
  std::vector<HostBaObservationRecord> observations;
};

// The Mapper-side journal coalescer publishes one complete semantic batch.
// The store never observes a partially committed mutation sequence.
struct CoalescedBaGraphMutation {
  uint64_t owner_epoch = 0;
  uint64_t revision_before = 0;
  uint64_t revision_after = 0;
  std::vector<HostBaCameraRecord> camera_upserts;
  std::vector<HostBaImageRecord> image_upserts;
  std::vector<HostBaPointRecord> point_upserts;
  std::vector<HostBaObservationRecord> observation_upserts;
  std::vector<uint32_t> tombstone_camera_ids;
  std::vector<uint32_t> tombstone_image_ids;
  std::vector<uint64_t> tombstone_point_ids;
  std::vector<HostBaObservationKey> tombstone_observations;
  bool force_full_rebuild = false;
};

struct HostBaGraphUpdateResult {
  uint64_t revision_before = 0;
  uint64_t revision_after = 0;
  uint64_t generation_before = 0;
  uint64_t generation_after = 0;
  uint64_t appended_slots = 0;
  uint64_t updated_slots = 0;
  uint64_t tombstoned_slots = 0;
  uint64_t cascaded_observation_tombstones = 0;
  uint64_t transaction_copy_bytes = 0;
  uint64_t touched_entity_records = 0;
  uint64_t touched_observation_records = 0;
  uint64_t adjacency_nodes_visited = 0;
  uint64_t adjacency_nodes_appended = 0;
  uint64_t full_catalog_scans = 0;
  uint64_t capacity_growth_events = 0;
  uint64_t capacity_growth_copy_bytes = 0;
  uint64_t hash_rehash_events = 0;
  uint64_t hash_rehash_entries = 0;
  bool published = false;
  bool semantic_noop = false;
  bool full_rebuild = false;
  bool full_rebuild_required = false;
  bool in_place_delta = false;
  bool reader_busy = false;
};

struct HostBaGraphPublication;
struct HostBaGraphLeaseGateState;
struct HostBaGraphReadGuard;

class CatalogReadLease {
 public:
  CatalogReadLease() = default;

  bool valid() const noexcept;
  uint32_t abi_version() const noexcept;
  uint64_t owner_epoch() const noexcept;
  uint64_t topology_revision() const noexcept;
  uint64_t generation() const noexcept;
  uint64_t slot_namespace_epoch() const noexcept;
  uint64_t resident_bytes() const noexcept;

  BaArrayView<HostBaCameraSlot> cameras() const noexcept;
  BaArrayView<HostBaImageSlot> images() const noexcept;
  BaArrayView<HostBaPointSlot> points() const noexcept;
  BaArrayView<HostBaObservationSlot> observations() const noexcept;
  BaArrayView<HostBaIncidenceNode> image_incidence_nodes() const noexcept;
  BaArrayView<HostBaIncidenceNode> point_incidence_nodes() const noexcept;

  const HostBaCameraSlot* FindCameraById(uint32_t id) const noexcept;
  const HostBaImageSlot* FindImageById(uint32_t id) const noexcept;
  const HostBaPointSlot* FindPointById(uint64_t id) const noexcept;
  const HostBaObservationSlot* FindObservation(
      uint32_t image_id, uint32_t point2D_idx) const noexcept;

 private:
  friend class HostBaGraphStore;
  explicit CatalogReadLease(
      std::shared_ptr<const HostBaGraphPublication> publication,
      std::shared_ptr<HostBaGraphReadGuard> guard);
  std::shared_ptr<const HostBaGraphPublication> publication_;
  std::shared_ptr<HostBaGraphReadGuard> guard_;
};

// Single-writer Mapper-lifetime structure store. Dynamic parameter values are
// intentionally absent; a lease pins one immutable structural generation.
// Normal mutations are amortized O(delta + touched lifetime adjacency). Rare
// vector/hash growth is exposed through HostBaGraphUpdateResult.
class HostBaGraphStore {
 public:
  explicit HostBaGraphStore(uint64_t owner_epoch);
  HostBaGraphStore(const HostBaGraphStore&) = delete;
  HostBaGraphStore& operator=(const HostBaGraphStore&) = delete;
  ~HostBaGraphStore();

  uint64_t owner_epoch() const noexcept;
  uint64_t topology_revision() const noexcept;
  uint64_t generation() const noexcept;
  bool empty() const noexcept;

  bool ColdBuild(const HostBaGraphColdInput& input,
                 HostBaGraphUpdateResult* result,
                 std::string* error);
  bool ApplyCoalescedMutation(const CoalescedBaGraphMutation& mutation,
                              HostBaGraphUpdateResult* result,
                              std::string* error);

  CatalogReadLease AcquireReadLease() const noexcept;
  bool IsCurrent(const CatalogReadLease& lease) const noexcept;

 private:
  uint64_t owner_epoch_ = 0;
  uint64_t next_generation_ = 0;
  uint64_t next_slot_namespace_epoch_ = 0;
  bool valid_ = false;
  std::shared_ptr<HostBaGraphLeaseGateState> gate_;
  std::shared_ptr<HostBaGraphPublication> publication_;
};

enum class CudaPreparedSelectionCacheMode : uint8_t {
  kDisabled = 0,
  kEnabled = 1,
};

enum class NativeBaVisualObservationScope : uint8_t {
  kLegacyExplicitPointTrackExpansion = 0,
  kActiveImagesOnly = 1,
};

struct NativeBaOnlineLidarIdentity {
  bool valid = false;
  uint32_t trigger_image_id = kBaGraphInvalidSlot;
  uint64_t map_version = 0;
  uint64_t max_scan_index = 0;
  std::array<uint8_t, 32> snapshot_sha256{};
  std::array<uint8_t, 32> geometry_sha256{};
  std::array<uint8_t, 32> association_sha256{};
};

bool IsValidNativeBaOnlineLidarIdentity(
    const NativeBaOnlineLidarIdentity& identity) noexcept;
bool IsCanonicalNativeBaLegacyLidarIdentity(
    const NativeBaOnlineLidarIdentity& identity) noexcept;
bool IsValidNativeBaLidarScopeIdentity(
    NativeBaVisualObservationScope scope,
    const NativeBaOnlineLidarIdentity& identity) noexcept;
bool SameNativeBaOnlineLidarIdentity(
    const NativeBaOnlineLidarIdentity& lhs,
    const NativeBaOnlineLidarIdentity& rhs) noexcept;

struct NativeCudaResolvedConfig {
  bool resolved = false;
  bool performance_mode = false;
  uint64_t config_generation = 0;
  CudaArithmeticPrecision arithmetic_precision =
      static_cast<CudaArithmeticPrecision>(0);
  CudaDeviceContextMode device_context =
      static_cast<CudaDeviceContextMode>(0);
  CudaMemoryMode memory_mode = static_cast<CudaMemoryMode>(0);
  CudaReductionMode reduction_mode = static_cast<CudaReductionMode>(0);
  CudaAuditProfile audit_profile = static_cast<CudaAuditProfile>(0);
  CudaCurrentLinearizationCacheMode linearization_cache =
      static_cast<CudaCurrentLinearizationCacheMode>(0);
  CudaHessianAssemblyBackend hessian_backend =
      static_cast<CudaHessianAssemblyBackend>(0);
  CudaSchurContributionBackend schur_backend =
      static_cast<CudaSchurContributionBackend>(0);
  CudaHotKernelMode hot_kernel = static_cast<CudaHotKernelMode>(0);
  CudaExecutionProfile execution_profile =
      static_cast<CudaExecutionProfile>(0);
  CudaResidualOrder residual_order = static_cast<CudaResidualOrder>(0);
  CudaPreparedSelectionCacheMode prepared_selection_cache =
      CudaPreparedSelectionCacheMode::kDisabled;
  CudaLossMode loss_mode = static_cast<CudaLossMode>(0);
  LidarResidualMode lidar_residual_mode = LidarResidualMode::kLegacyExact;
  double lidar_near_zero_threshold = 1e-12;
  double loss_scale = 1.0;
  int32_t device = 0;
  int32_t block_size = 128;
  int32_t cost_reduction_threads = 128;
  uint64_t pair_chunk_limit_bytes = 0;
  uint32_t hessian_segment_size = 0;
  uint32_t schur_segment_size = 0;
  int32_t max_num_iterations = 0;
  int32_t max_consecutive_invalid_steps = 10;
  double function_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  double parameter_tolerance = 0.0;
  double max_solver_time_in_seconds = 1e9;
  double initial_trust_region_radius = 1e4;
  double min_trust_region_radius = 1e-32;
  double max_trust_region_radius = 1e16;
  double min_relative_decrease = 1e-3;
  double min_lm_diagonal = 1e-6;
  double max_lm_diagonal = 1e32;
};

struct TranslationSubsetPolicy {
  uint32_t image_slot = kBaGraphInvalidSlot;
  uint8_t constant_mask = 0;
  uint8_t padding[3]{};
};

struct CameraParameterPolicy {
  uint32_t camera_slot = kBaGraphInvalidSlot;
  bool constant = true;
  std::vector<uint32_t> fixed_parameter_indices;
};

struct PointFixedPolicy {
  uint32_t point_slot = kBaGraphInvalidSlot;
  bool constant = false;
  uint8_t config_role = 0;
  bool has_search_range = false;
  double search_range = 0.0;
};

struct LidarConstraintRecord {
  uint32_t point_slot = kBaGraphInvalidSlot;
  uint32_t constraint_slot = kBaGraphInvalidSlot;
  // Stable identity within the selected LiDAR constraint generation. This is
  // not the solve-local source insertion index.
  uint64_t physical_identity = 0;
  uint64_t association_id = kBaGraphInvalidAssociationId;
  uint32_t owner_image_id = kBaGraphInvalidSlot;
  uint32_t owner_point2D_idx = kBaGraphInvalidSlot;
  uint8_t lidar_type = 0;
  uint8_t padding[3]{};
  std::array<double, 3> frozen_point3D_xyz{{0.0, 0.0, 0.0}};
  std::array<double, 4> plane{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> lidar_xyz{{0.0, 0.0, 0.0}};
  double weight = 0.0;
  double search_range = 0.0;
  uint64_t point_state_generation = 0;
};

bool IsCanonicalNativeBaLegacyLidarConstraint(
    const LidarConstraintRecord& constraint) noexcept;

struct LidarConstraintSelection {
  uint64_t lidar_map_generation = 0;
  uint64_t match_config_generation = 0;
  NativeBaOnlineLidarIdentity online_lidar_identity;
  std::vector<LidarConstraintRecord> constraints;
};

// Mapper-facing native contract. It contains only stable Reconstruction IDs,
// selection/fixed policy, LiDAR constraints, and an already-resolved CUDA
// configuration. Stable graph slots and residual ordinals are derived under a
// CatalogReadLease; callers never manufacture GPU/catalog indices.
struct NativeBaTranslationPolicy {
  uint32_t image_id = 0;
  uint8_t constant_mask = 0;
  uint8_t padding[3]{};
};

struct NativeBaCameraPolicy {
  uint32_t camera_id = 0;
  bool constant = true;
  std::vector<uint32_t> fixed_parameter_indices;
};

struct NativeBaPointPolicy {
  uint64_t point3D_id = 0;
  bool constant = false;
  uint8_t config_role = 0;
  bool has_search_range = false;
  double search_range = 0.0;
};

struct NativeBaLidarConstraint {
  uint64_t point3D_id = 0;
  uint32_t constraint_slot = kBaGraphInvalidSlot;
  uint64_t physical_identity = 0;
  uint64_t association_id = kBaGraphInvalidAssociationId;
  uint32_t owner_image_id = kBaGraphInvalidSlot;
  uint32_t owner_point2D_idx = kBaGraphInvalidSlot;
  uint8_t lidar_type = 0;
  uint8_t padding[3]{};
  std::array<double, 3> frozen_point3D_xyz{{0.0, 0.0, 0.0}};
  std::array<double, 4> plane{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> lidar_xyz{{0.0, 0.0, 0.0}};
  double weight = 0.0;
  double search_range = 0.0;
};

bool IsCanonicalNativeBaLegacyLidarConstraint(
    const NativeBaLidarConstraint& constraint) noexcept;

struct NativeBaSolveIntent {
  uint32_t abi_version = kNativeHostSolveViewAbiVersion;
  uint64_t owner_epoch = 0;
  uintptr_t reconstruction_identity = 0;
  uint64_t expected_topology_revision = 0;
  uint64_t selection_revision = 0;
  BaKind kind = BaKind::kLocal;
  NativeBaVisualObservationScope visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  NativeCudaResolvedConfig config;
  // Order is semantically significant and is the source-insertion image order.
  std::vector<uint32_t> active_image_ids;
  // Direct preparation emits external visual residuals in this category order:
  // variable points, LiDAR constraints, then constant points.
  std::vector<uint64_t> explicit_variable_point_ids;
  std::vector<uint64_t> explicit_constant_point_ids;
  std::vector<uint32_t> fixed_pose_ids;
  std::vector<NativeBaTranslationPolicy> translation_policies;
  std::vector<NativeBaCameraPolicy> camera_policies;
  std::vector<NativeBaPointPolicy> point_policies;
  uint64_t lidar_map_generation = 0;
  uint64_t lidar_match_config_generation = 0;
  NativeBaOnlineLidarIdentity online_lidar_identity;
  std::vector<NativeBaLidarConstraint> lidar_constraints;
};

struct BaSolveIntent {
  uint32_t abi_version = kNativeHostSolveViewAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  uint64_t selection_revision = 0;
  BaKind kind = BaKind::kLocal;
  NativeBaVisualObservationScope visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  NativeCudaResolvedConfig config;
  std::vector<uint32_t> active_image_slots;
  std::vector<uint32_t> active_visual_observation_slots;
  // When true, an empty vector means that no visual observations are active;
  // it must not fall back to scanning every active-image incidence.
  bool active_visual_observation_slots_explicit = false;
  // Direct ID-based preparation may resolve the complete visual residual set
  // once, including boundary observations. The materializer validates and
  // classifies these slots without traversing catalog incidence again.
  std::vector<uint32_t> resolved_visual_observation_slots;
  bool visual_observation_slots_fully_resolved = false;
  // Optional source-insertion sequence produced by the shared residual
  // enumerator. When present it must cover every selected visual and LiDAR
  // residual exactly once. It is distinct from catalog physical identity and
  // from the effective execution ordinal after canonical sorting.
  struct ResidualSelection {
    ResidualKind kind = ResidualKind::kVisual;
    uint32_t source_slot = kBaGraphInvalidSlot;
  };
  std::vector<ResidualSelection> source_insertion_order;
  std::vector<uint32_t> explicit_variable_point_slots;
  std::vector<uint32_t> explicit_constant_point_slots;
  std::vector<uint32_t> fixed_pose_slots;
  std::vector<TranslationSubsetPolicy> translation_subsets;
  std::vector<CameraParameterPolicy> camera_policies;
  std::vector<PointFixedPolicy> point_policies;
  LidarConstraintSelection lidar;
};

struct DenseCameraState {
  uint32_t camera_slot = kBaGraphInvalidSlot;
  uint64_t state_generation = 0;
  std::vector<double> parameters;
};

struct DenseImageState {
  uint32_t image_slot = kBaGraphInvalidSlot;
  uint64_t state_generation = 0;
  std::array<double, 4> quaternion{{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> translation{{0.0, 0.0, 0.0}};
};

struct DensePointState {
  uint32_t point_slot = kBaGraphInvalidSlot;
  uint64_t state_generation = 0;
  std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
};

// Solve-owned dense values gathered by the Mapper-side bridge. The graph
// store never retains this state.
struct DenseActiveState {
  uint64_t owner_epoch = 0;
  uint64_t state_generation = 0;
  std::vector<DenseCameraState> cameras;
  std::vector<DenseImageState> images;
  std::vector<DensePointState> points;
};

struct VariableImageStateDelta {
  uint32_t image_slot = kBaGraphInvalidSlot;
  uint8_t translation_subset_mask = 0;
  uint8_t padding[3]{};
  std::array<double, 4> quaternion{{1.0, 0.0, 0.0, 0.0}};
  std::array<double, 3> translation{{0.0, 0.0, 0.0}};
};

struct VariablePointStateDelta {
  uint32_t point_slot = kBaGraphInvalidSlot;
  uint32_t reserved = 0;
  std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
};

struct VariableCameraStateDelta {
  uint32_t camera_slot = kBaGraphInvalidSlot;
  uint32_t reserved = 0;
  std::vector<double> parameters;
};

// The only production state returned by a native solve. Fixed entities are
// deliberately absent. The expected slot vectors make coverage validation
// explicit before Reconstruction mutation begins.
struct VariableStateDelta {
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  uint64_t view_generation = 0;
  uint64_t solve_generation = 0;
  uint64_t state_generation = 0;
  std::vector<uint32_t> expected_image_slots;
  std::vector<uint32_t> expected_point_slots;
  std::vector<uint32_t> expected_camera_slots;
  std::vector<VariableImageStateDelta> images;
  std::vector<VariablePointStateDelta> points;
  std::vector<VariableCameraStateDelta> cameras;
};

struct ActiveStateBuffer {
  uint64_t owner_epoch = 0;
  uint64_t catalog_generation = 0;
  uint64_t state_generation = 0;
  std::vector<DenseCameraState> cameras;
  std::vector<DenseImageState> images;
  std::vector<DensePointState> points;
};

struct ResidualOrdinal {
  uint64_t execution_ordinal = 0;
  uint64_t source_insertion_index = 0;
  // Stable physical identity is audit/catalog identity only. It is never
  // written into CudaVisualInput::source_index.
  uint64_t physical_identity = 0;
  uint64_t association_id = kBaGraphInvalidAssociationId;
  uint32_t owner_image_id = kBaGraphInvalidSlot;
  uint32_t owner_point2D_idx = kBaGraphInvalidSlot;
  ResidualKind kind = ResidualKind::kVisual;
  uint32_t source_slot = kBaGraphInvalidSlot;
};

struct ParameterOrdinal {
  uint64_t ordinal = 0;
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_slot = kBaGraphInvalidSlot;
  uint32_t ambient_size = 0;
  uint32_t tangent_size = 0;
  uint8_t constant = 0;
  uint8_t translation_subset_mask = 0;
  uint8_t padding[2]{};
};

struct ImageFixedPolicyResult {
  uint32_t image_slot = kBaGraphInvalidSlot;
  uint8_t pose_constant = 1;
  uint8_t translation_subset_mask = 0;
  uint8_t boundary_pose = 0;
  uint8_t padding = 0;
  uint32_t tangent_size = 0;
};

struct CameraFixedPolicyResult {
  uint32_t camera_slot = kBaGraphInvalidSlot;
  uint8_t constant = 1;
  uint8_t padding[3]{};
  uint32_t ambient_size = 0;
  uint32_t tangent_size = 0;
  std::vector<uint32_t> fixed_parameter_indices;
};

struct PointFixedPolicyResult {
  uint32_t point_slot = kBaGraphInvalidSlot;
  uint8_t constant = 0;
  uint8_t config_role = 0;
  uint8_t has_search_range = 0;
  uint8_t padding = 0;
  double search_range = 0.0;
};

struct FixedPolicyResult {
  std::vector<ImageFixedPolicyResult> images;
  std::vector<CameraFixedPolicyResult> cameras;
  std::vector<PointFixedPolicyResult> points;
};

struct NativeHostSolveViewIdentity {
  uint32_t abi_version = kNativeHostSolveViewAbiVersion;
  uint32_t catalog_abi_version = kHostBaGraphAbiVersion;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t catalog_generation = 0;
  uint64_t selection_revision = 0;
  uint64_t config_generation = 0;
  uint64_t lidar_map_generation = 0;
  uint64_t lidar_match_config_generation = 0;
  NativeBaVisualObservationScope visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  NativeBaOnlineLidarIdentity online_lidar_identity;
};

struct PreparedSelectionPlan {
  uint64_t publication_id = 0;
  uint64_t slot_namespace_epoch = 0;
  uint64_t host_resident_bytes = 0;
  NativeBaVisualObservationScope visual_observation_scope =
      NativeBaVisualObservationScope::kLegacyExplicitPointTrackExpansion;
  NativeBaOnlineLidarIdentity online_lidar_identity;
  std::vector<uint32_t> active_camera_slots;
  std::vector<uint32_t> active_image_slots;
  std::vector<uint32_t> boundary_image_slots;
  std::vector<uint32_t> active_point_slots;
  std::vector<uint32_t> visual_observation_slots;
  FixedPolicyResult fixed;
  std::vector<LidarConstraintRecord> lidar_constraints;
  std::vector<ResidualOrdinal> residual_ordinals;
  std::vector<ParameterOrdinal> parameter_ordinals;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t ambient_parameter_count = 0;
  uint64_t effective_parameter_count = 0;
};

struct NativeHostSolveView {
  NativeHostSolveViewIdentity identity;
  BaKind kind = BaKind::kLocal;
  NativeCudaResolvedConfig config;
  CatalogReadLease catalog;
  std::vector<uint32_t> active_camera_slots;
  std::vector<uint32_t> active_image_slots;
  std::vector<uint32_t> boundary_image_slots;
  std::vector<uint32_t> active_point_slots;
  std::vector<uint32_t> visual_observation_slots;
  FixedPolicyResult fixed;
  LidarConstraintSelection lidar;
  std::vector<ResidualOrdinal> residual_ordinals;
  std::vector<ParameterOrdinal> parameter_ordinals;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t ambient_parameter_count = 0;
  uint64_t effective_parameter_count = 0;
  std::shared_ptr<const PreparedSelectionPlan> prepared_plan;

  const std::vector<uint32_t>& ActiveCameraSlots() const noexcept;
  const std::vector<uint32_t>& ActiveImageSlots() const noexcept;
  const std::vector<uint32_t>& BoundaryImageSlots() const noexcept;
  const std::vector<uint32_t>& ActivePointSlots() const noexcept;
  const std::vector<uint32_t>& VisualObservationSlots() const noexcept;
  const FixedPolicyResult& Fixed() const noexcept;
  const std::vector<LidarConstraintRecord>& LidarConstraints() const noexcept;
  const std::vector<ResidualOrdinal>& ResidualOrdinals() const noexcept;
  const std::vector<ParameterOrdinal>& ParameterOrdinals() const noexcept;
  uint64_t ResidualBlockCount() const noexcept;
  uint64_t ScalarResidualCount() const noexcept;
  uint64_t AmbientParameterCount() const noexcept;
  uint64_t EffectiveParameterCount() const noexcept;
};

struct NativeHostSolvePreparationRuntime {
  uint64_t calls = 0;
  uint64_t catalog_observation_visits = 0;
  uint64_t incidence_traversal_visits = 0;
  uint64_t active_observation_visits = 0;
  uint64_t boundary_observation_visits = 0;
  uint64_t state_values_copied = 0;
  uint64_t quaternion_normalizations = 0;
  uint64_t descriptor_bytes = 0;
  uint64_t static_materialize_calls = 0;
  uint64_t dynamic_state_gather_calls = 0;
  double wall_milliseconds = 0.0;
};

class NativeHostSolveMaterializer {
 public:
  NativeHostSolveMaterializer() = default;
  NativeHostSolveMaterializer(const NativeHostSolveMaterializer&) = delete;
  NativeHostSolveMaterializer& operator=(
      const NativeHostSolveMaterializer&) = delete;

  bool Materialize(const CatalogReadLease& catalog,
                   const BaSolveIntent& intent,
                   const DenseActiveState& gathered_state,
                   NativeHostSolveView* view,
                   ActiveStateBuffer* active_state,
                   NativeHostSolvePreparationRuntime* runtime,
                   std::string* error);

 private:
  uint64_t scratch_generation_ = 0;
  std::vector<uint64_t> camera_active_stamps_;
  std::vector<uint64_t> image_active_stamps_;
  std::vector<uint64_t> point_active_stamps_;
  std::vector<uint64_t> observation_stamps_;
  std::vector<uint64_t> camera_state_stamps_;
  std::vector<uint64_t> image_state_stamps_;
  std::vector<uint64_t> point_state_stamps_;
  std::vector<int32_t> camera_state_indices_;
  std::vector<int32_t> image_state_indices_;
  std::vector<int32_t> point_state_indices_;
};

bool ValidateNativeHostSolveView(const NativeHostSolveView& view,
                                 const ActiveStateBuffer& state,
                                 std::string* error);

struct BaSolveResult {
  struct Runtime {
    uint64_t legacy_kernel_input_bundle_calls = 0;
    uint64_t build_cuda_layer_a_inputs_calls = 0;
    uint64_t build_static_layout_calls = 0;
    uint64_t build_cost_layout_calls = 0;
    uint64_t build_layer_b_topology_calls = 0;
    uint64_t build_layer_c_topology_calls = 0;
    uint64_t temporary_visual_input_bytes = 0;
    uint64_t temporary_lidar_input_bytes = 0;
    uint64_t dense_active_state_device_download_calls = 0;
    uint64_t variable_state_delta_device_download_calls = 0;
    uint64_t variable_state_delta_device_download_bytes = 0;
    uint64_t indexed_visual_binding_bytes = 0;
    uint64_t repeated_residual_state_packing_bytes = 0;
    uint64_t device_store_lookup_calls = 0;
    uint64_t device_store_reuse_calls = 0;
    uint64_t device_store_full_upload_calls = 0;
    uint64_t device_store_full_upload_bytes = 0;
    uint64_t device_store_patch_upload_calls = 0;
    uint64_t device_store_patch_upload_bytes = 0;
    uint64_t device_store_growth_d2d_calls = 0;
    uint64_t device_store_growth_d2d_bytes = 0;
    uint64_t device_store_invalidations = 0;
    uint64_t device_store_generation = 0;
    uint64_t indexed_plan_build_calls = 0;
    uint64_t indexed_plan_upload_bytes = 0;
    uint64_t device_selection_lookup_calls = 0;
    uint64_t device_selection_hit_calls = 0;
    uint64_t device_selection_miss_calls = 0;
    uint64_t device_selection_upload_calls = 0;
    uint64_t device_selection_evictions = 0;
    uint64_t device_selection_bypasses = 0;
    uint64_t device_selection_context_invalidations = 0;
    uint64_t device_selection_poison_events = 0;
    uint64_t device_selection_static_h2d_calls = 0;
    uint64_t device_selection_static_h2d_bytes = 0;
    uint64_t device_selection_static_h2d_saved_calls = 0;
    uint64_t device_selection_static_h2d_saved_bytes = 0;
    uint64_t device_selection_resident_bytes = 0;
    uint64_t device_selection_peak_bytes = 0;
    uint64_t device_selection_cached_workspace_bytes = 0;
    double indexed_packing_milliseconds = 0.0;
    double variable_state_download_milliseconds = 0.0;
    bool temporary_legacy_kernel_abi = false;
    bool native_lm_controller_handoff_complete = false;
    uint64_t real_cost_layout_entries = 0;
    uint64_t real_pose_adjacency_entries = 0;
    uint64_t real_point_adjacency_entries = 0;
    uint64_t real_edge_adjacency_entries = 0;
    uint64_t real_schur_pair_contributions = 0;
  } runtime;
  bool success = false;
  std::string error;
  std::string termination_reason;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int32_t trial_iterations = 0;
  int32_t accepted_steps = 0;
  int32_t accepted_decisions = 0;
  int32_t accepted_commits = 0;
  int32_t rejected_steps = 0;
  int32_t invalid_steps = 0;
  uint64_t backward_error_samples = 0;
  double max_backward_error = 0.0;
  bool resource_cleanup_failed = false;
  uint64_t final_internal_state_epoch = 0;
  VariableStateDelta variable_delta;
};

struct NativeCudaSolveRequest {
  const NativeHostSolveView* view = nullptr;
  const ActiveStateBuffer* initial_state = nullptr;
  const CudaFullLmOptions* options = nullptr;
  // Mapping-session owner for static device catalog generations. The solve
  // context pins the published allocation while it is executing.
  std::shared_ptr<DeviceBaProblemStoreHandle> device_store;
};

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_HOST_BA_GRAPH_H_
