#ifndef COLMAP_SRC_GPU_BA_CUSTOM_CUDA_H_
#define COLMAP_SRC_GPU_BA_CUSTOM_CUDA_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_ba/host_ba_graph.h"
#include "gpu_ba/linearization.h"
#include "gpu_ba/snapshot.h"
#include "gpu_ba/host_problem_store.h"

namespace colmap {
namespace gpu_ba {

struct IndexedActiveSolveDescriptor;
struct MapperStaticCatalogStableTables;
struct CudaLayerBOptions;
struct NativePreparationComparisonResult;
enum class CudaHotKernelMode : uint8_t;
enum class CudaSchurContributionBackend : uint8_t;

// These flat records are the host/device ABI for the first custom_cuda layer.
// They intentionally contain only fixed-size IEEE-754 binary64 arrays so the
// kernel does not depend on Eigen, Ceres, or STL layout.
struct CudaVisualInput {
  uint64_t source_index = 0;
  uint32_t image_id = 0;
  uint64_t point3D_id = 0;
  double quaternion[4] = {1.0, 0.0, 0.0, 0.0};
  double translation[3] = {0.0, 0.0, 0.0};
  double point[3] = {0.0, 0.0, 0.0};
  double camera[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double observation[2] = {0.0, 0.0};
};

struct CudaVisualOutput {
  double residual[2] = {0.0, 0.0};
  double ambient_quaternion_jacobian[8] = {};
  double plus_jacobian[12] = {};
  double local_rotation_jacobian[6] = {};
  double translation_jacobian[6] = {};
  double point_jacobian[6] = {};
  double camera_jacobian[16] = {};
  double camera_point[3] = {};
  uint8_t finite = 0;
};

struct CudaLidarInput {
  uint64_t source_index = 0;
  uint64_t point3D_id = 0;
  double point[3] = {0.0, 0.0, 0.0};
  double plane[4] = {0.0, 0.0, 0.0, 0.0};
  double weight = 0.0;
  uint8_t mode = static_cast<uint8_t>(LidarResidualMode::kLegacyExact);
  double near_zero_threshold = 1e-12;
};

struct CudaLidarOutput {
  double signed_distance = 0.0;
  double residual = 0.0;
  double point_jacobian[3] = {};
  uint8_t exact_zero = 0;
  uint8_t near_zero = 0;
  uint8_t guarded = 0;
  uint8_t finite = 0;
};

// Temporary adapter from the VS1 native contracts to the established kernel
// ABI. It owns packed kernel inputs and solve-local topology only; it never
// owns semantic capture records or a graph-store publication.
class LegacyKernelInputBundle {
 public:
  struct Impl;

  LegacyKernelInputBundle();
  ~LegacyKernelInputBundle();
  LegacyKernelInputBundle(LegacyKernelInputBundle&& other) noexcept;
  LegacyKernelInputBundle& operator=(LegacyKernelInputBundle&& other) noexcept;
  LegacyKernelInputBundle(const LegacyKernelInputBundle&) = delete;
  LegacyKernelInputBundle& operator=(const LegacyKernelInputBundle&) = delete;

  const std::vector<CudaVisualInput>& visual() const noexcept;
  const std::vector<CudaLidarInput>& lidar() const noexcept;
  const BaSolveResult::Runtime& runtime() const noexcept;

 private:
  friend bool BuildCudaLayerAInputs(const NativeHostSolveView&,
                                    const ActiveStateBuffer&,
                                    LegacyKernelInputBundle*,
                                    std::string*);
  friend bool BuildStaticLayout(const NativeHostSolveView&,
                                const ActiveStateBuffer&,
                                LegacyKernelInputBundle*,
                                std::string*);
  friend bool BuildCostLayout(const NativeHostSolveView&,
                              LegacyKernelInputBundle*,
                              std::string*);
  friend bool BuildLayerBTopology(const NativeHostSolveView&,
                                  LegacyKernelInputBundle*,
                                  std::string*);
  friend bool BuildLayerCTopology(const NativeHostSolveView&,
                                  LegacyKernelInputBundle*,
                                  std::string*);
  friend bool BuildLegacyKernelInputBundle(const NativeHostSolveView&,
                                           const ActiveStateBuffer&,
                                           LegacyKernelInputBundle*,
                                           std::string*);
  friend bool RunCustomCudaSolve(const NativeCudaSolveRequest&,
                                 BaSolveResult*,
                                 std::string*);
  friend bool CompareNativePreparationToLegacyForTesting(
      const Snapshot&,
      const NativeHostSolveView&,
      const ActiveStateBuffer&,
      const CudaLayerBOptions&,
      uint64_t,
      CudaHotKernelMode,
      CudaSchurContributionBackend,
      uint32_t,
      NativePreparationComparisonResult*,
      std::string*);
  std::unique_ptr<Impl> impl_;
};

enum class CudaMemoryMode : uint8_t {
  kExplicitDeviceCopy = 0,
  kUnifiedManaged = 1,
};

enum class CudaResidualOrder : uint8_t {
  kCanonical = 0,
  kSourceInsertion = 1,
};

// Loss selection is intentionally explicit at the custom CUDA boundary.  The
// default reproduces the loss captured by the snapshot; no Ceres LossFunction
// is instantiated by the CUDA path.
enum class CudaLossMode : uint8_t {
  kFromSnapshot = 0,
  kTrivial = 1,
  kSoftL1 = 2,
};

// Correctness mode keeps the historical serial accumulation.  The parallel
// mode uses fixed contiguous partitions and an ordered final reduction, so it
// remains deterministic while exposing actual CUDA parallel work.
enum class CudaReductionMode : uint8_t {
  kSerialDeterministic = 0,
  kParallelDeterministic = 1,
};

enum class CudaInstrumentationMode : uint8_t {
  kDisabled = 0,
  kEnabled = 1,
};

enum class CudaCurrentLinearizationCacheMode : uint8_t {
  kDisabled = 0,
  kEnabled = 1,
};

enum class CudaDeviceContextMode : uint8_t {
  kCompatibilityDefault = 0,
  kLegacy = 1,
  kPersistent = 2,
  kDeviceState = 3,
  kDeviceControl = 4,
};

enum class CudaHotKernelMode : uint8_t {
  kCompatibilityDefault = 0,
  kReference = 1,
  kOptimized = 2,
  kTransformed = 3,
};

// Explicit Schur pose-pair contribution assembly. Compatibility/default keeps
// the Phase 7.4 direct transformed implementation. The segmented candidate is
// opt-in and never changes the Mapper production default in Phase 10.0a.
enum class CudaSchurContributionBackend : uint8_t {
  kCompatibilityDefault = 0,
  kDirectTransformed = 1,
  kSegmentedTransformed = 2,
};

// Hessian/gradient assembly is deliberately orthogonal to the Schur hot
// kernel and Schur contribution selector. Compatibility keeps the historical
// mapping from CudaHotKernelMode; the two observation-centric candidates are
// explicit opt-in Phase 10.0b implementations.
enum class CudaHessianAssemblyBackend : uint8_t {
  kCompatibilityDefault = 0,
  kPoseOwnedScalarCompatibility = 1,
  kPoseOwnedOptimizedReference = 2,
  kObservationAtomic = 3,
  kObservationSegmented = 4,
};

// Arithmetic precision is orthogonal to execution, Hessian, and Schur
// selectors. Compatibility/default deliberately resolves to the established
// FP64 path. Experimental FP32 modes never fall back to FP64 after resolution.
enum class CudaArithmeticPrecision : uint8_t {
  kCompatibilityDefault = 0,
  kFp64 = 1,
  kFp32Core = 2,
  kFp32StateQuantizedMixed = 3,
  kPureFp32Experimental = 4,
  // FP64 geometry evaluated into FP32 residual/Jacobian records and FP32
  // normal-equation products, with cancellation-sensitive reductions,
  // factorization, diagnostics, and state updates retained in FP64. Appended
  // to preserve the frozen Phase 10.1a enum values.
  kFp32MixedStable = 5,
};

// Cumulative production execution profiles. Each level includes every
// preceding level; standalone entry points keep kBaseline unless explicitly
// selected by an internal replay/test driver.
enum class CudaExecutionProfile : uint8_t {
  kBaseline = 0,
  kRuntimePool = 1,
  kArena = 2,
  kDeviceScaling = 3,
  kFastIdentity = 4,
  kUnifiedBuilder = 5,
  kCompactLayerA = 6,
  kCompactControl = 7,
};

// Optional audit work is orthogonal to the mathematical and execution
// profiles. CompatibilityDefault preserves legacy per-option behavior for
// frozen tests. Production is fail-closed: it disables optional tracing and
// instrumentation and verifies that no full-array audit mirror was executed.
enum class CudaAuditProfile : uint8_t {
  kCompatibilityDefault = 0,
  kProduction = 1,
  kCorrectness = 2,
  kForensic = 3,
};

enum class CudaResourceHealth : uint8_t {
  kClean = 0,
  kTainted = 1,
  kClearing = 2,
};

// Phase 7 instrumentation keeps host and device clocks in separate ledgers.
// The host value is an exclusive wall-time bucket; CUDA events only describe
// work recorded on the phase's existing stream and are never added to host
// wall time.
struct CudaPhaseTiming {
  double host_wall_milliseconds = 0.0;
  double cuda_event_milliseconds = 0.0;
  uint64_t calls = 0;
  uint64_t bytes = 0;
};

struct CudaStateHashCounter {
  uint64_t requests = 0;
  uint64_t computations = 0;
  uint64_t cache_hits = 0;
  uint64_t blocks_visited = 0;
  uint64_t scalars_hashed = 0;
  double host_wall_milliseconds = 0.0;
};

struct CudaStateHashAudit {
  uint64_t layout_builds = 0;
  uint64_t layout_blocks = 0;
  double layout_host_wall_milliseconds = 0.0;
  CudaStateHashCounter current;
  CudaStateHashCounter trial;
};

struct CudaSyncSiteTiming {
  uint64_t calls = 0;
  double host_wait_milliseconds = 0.0;
};

struct CudaSynchronizationAudit {
  uint64_t stream_synchronize_calls = 0;
  uint64_t event_synchronize_calls = 0;
  uint64_t device_synchronize_calls = 0;
  double stream_host_wait_milliseconds = 0.0;
  double event_host_wait_milliseconds = 0.0;
  double device_host_wait_milliseconds = 0.0;
  CudaSyncSiteTiming layer_a_final;
  CudaSyncSiteTiming layer_b_final;
  CudaSyncSiteTiming layer_c_factor;
  CudaSyncSiteTiming layer_c_solver;
  CudaSyncSiteTiming layer_c_final;
  CudaSyncSiteTiming trial_cost_final;
  CudaSyncSiteTiming resource_release;
};

struct CudaTimingLedger {
  // Inclusive call timing overlaps by design and is never included in the
  // additive exclusive host ledger.
  CudaPhaseTiming layer_a_call;
  CudaPhaseTiming layer_b_call;
  CudaPhaseTiming layer_c_call;

  CudaPhaseTiming build_cuda_layer_a_inputs;
  CudaPhaseTiming topology_lookup;
  CudaPhaseTiming topology_build;
  CudaPhaseTiming topology_refresh;
  CudaPhaseTiming topology_cache;
  CudaPhaseTiming build_cost_order;
  CudaPhaseTiming allocation;
  CudaPhaseTiming free;
  CudaPhaseTiming h2d_memcpy;
  CudaPhaseTiming d2h_memcpy;
  CudaPhaseTiming d2d_memcpy;
  CudaPhaseTiming managed_prefetch;
  CudaPhaseTiming layer_a_kernel;
  CudaPhaseTiming layer_b_pose_kernel;
  CudaPhaseTiming layer_b_point_kernel;
  CudaPhaseTiming layer_b_edge_kernel;
  CudaPhaseTiming layer_b_gradient_kernel;
  CudaPhaseTiming layer_c_point_factor;
  CudaPhaseTiming schur;
  CudaPhaseTiming rhs;
  CudaPhaseTiming cusolver;
  CudaPhaseTiming back_substitution;
  CudaPhaseTiming back_substitution_host_self;
  CudaPhaseTiming trial_cost;
  CudaPhaseTiming apply_layer_c_step;
  CudaPhaseTiming compute_layer_c_diagnostics;
  CudaPhaseTiming lm_decision;
  CudaPhaseTiming synchronization;
  CudaPhaseTiming state_hash;
  CudaPhaseTiming topology_fingerprint_audit;
  CudaPhaseTiming controller_self;
  CudaPhaseTiming resource_release;
  CudaPhaseTiming other;

  CudaSynchronizationAudit synchronization_audit;
  std::string h2d_host_memory_kind = "none";
  std::string d2h_host_memory_kind = "none";
  std::string d2d_memory_kind = "device";

  uint64_t topology_epoch = 0;
  uint64_t topology_build_count = 0;
  uint64_t topology_refresh_count = 0;
  uint64_t cache_lookup_count = 0;
  uint64_t cache_hit_count = 0;
  uint64_t cache_miss_count = 0;
  uint64_t cache_invalidation_count = 0;
  uint64_t build_cuda_layer_a_inputs_count = 0;
  uint64_t build_cost_order_count = 0;
  uint64_t layer_a_calls = 0;
  uint64_t layer_b_calls = 0;
  uint64_t layer_c_calls = 0;
  uint64_t cost_calls = 0;
  std::string topology_fingerprint;
  uint64_t topology_fingerprint_computations = 0;
  uint64_t topology_generation = 0;
  double accounted_interval_wall_milliseconds = 0.0;
  double explicitly_measured_host_wall_milliseconds = 0.0;
  double uninstrumented_remainder_milliseconds = 0.0;
  double known_host_wall_milliseconds = 0.0;
  double other_host_wall_milliseconds = 0.0;
  double host_partition_error_percent = 0.0;
  bool host_partition_pass = false;
};

// Fault injection is test-only plumbing.  Production defaults to kNone and
// no mapper path enables these flags.
enum class CudaFaultInjection : uint8_t {
  kNone = 0,
  kForceInsufficientMemory = 1,
  kForcePointFactorizationFailure = 2,
  kForceNonfiniteTrial = 3,
  kForceWorkspaceAllocationFailure = 4,
  kForceLinearizationFailure = 5,
  kForceResourceTeardownFailure = 6,
  kForcePendingPreparationFailure = 7,
  kForceTimingApiFailure = 8,
  kForceIdentityMismatch = 9,
  kForceResourceDoubleAdvance = 10,
};

enum class CudaFaultLogicalSite : uint8_t {
  kNone = 0,
  kInitialLinearization = 1,
  kCurrentTrialRecompute = 2,
  kAcceptedPendingLinearization = 3,
  kLayerCPointFactor = 4,
  kLayerCWorkspace = 5,
  kTrialCost = 6,
};

enum class CudaFaultTriggerPhase : uint8_t {
  kNone = 0,
  kAcceptedPendingPreparation = 1,
  kBeforeCacheLookup = 2,
  kAfterCacheMiss = 3,
  kAfterTopologyRefreshBeforePublish = 4,
  kLayerCOperation = 5,
};

enum class CudaTeardownType : uint8_t {
  kGetDevice = 0,
  kSetDevice = 1,
  kSolver = 2,
  kBlas = 3,
  kEvent = 4,
  kStream = 5,
  kAllocation = 6,
};

enum class CudaTimingStatus : uint8_t {
  kDisabled = 0,
  kComplete = 1,
  kIncomplete = 2,
};

enum class CudaStatusDomain : uint8_t {
  kNone = 0,
  kCudaRuntime = 1,
  kCusolver = 2,
  kCublas = 3,
  kInjected = 4,
  kCxxException = 5,
  kInternal = 6,
};

enum class CudaFailureSite : uint16_t {
  kNone = 0,
  kTimingEventAcquire = 1,
  kTimingEventRecord = 2,
  kTimingElapsedQuery = 3,
  kCleanupGetDevice = 4,
  kCleanupSetDevice = 5,
  kDestroySolver = 6,
  kDestroyBlas = 7,
  kDestroyEvent = 8,
  kDestroyStream = 9,
  kDestroyAllocation = 10,
  kPendingPreparation = 11,
  kInitialLinearization = 12,
  kCurrentLinearization = 13,
  kAcceptedPendingLinearization = 14,
  kLayerCPointFactor = 15,
  kLayerCWorkspace = 16,
  kTrialCost = 17,
  kStateHash = 18,
  kAuditPreflightCapacity = 19,
  kAuditRuntimeOverflow = 20,
  kResourceGenerationDoubleAdvance = 21,
};

struct CudaFaultTrigger {
  CudaFaultInjection fault_kind = CudaFaultInjection::kNone;
  CudaFaultLogicalSite logical_site = CudaFaultLogicalSite::kNone;
  CudaFaultTriggerPhase trigger_phase = CudaFaultTriggerPhase::kNone;
  uint16_t operation = 0;
  uint64_t target_state_epoch = 0;
  uint64_t occurrence_within_site = 1;
  bool force_resource_teardown_failure = false;
};

struct CudaLayerAOptions {
  int device = 0;
  int block_size = 128;
  CudaMemoryMode memory_mode = CudaMemoryMode::kExplicitDeviceCopy;
  CudaResidualOrder residual_order = CudaResidualOrder::kCanonical;
};

struct CudaRuntimeInfo {
  int device = -1;
  int device_major = 0;
  int device_minor = 0;
  int launch_status = 0;
  int synchronize_status = 0;
  int memory_status = 0;
  uint64_t visual_bytes = 0;
  uint64_t lidar_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t free_bytes_before = 0;
  uint64_t total_bytes = 0;
  uint64_t active_bytes = 0;
  uint64_t cached_bytes = 0;
  double kernel_milliseconds = 0.0;
  double copy_milliseconds = 0.0;
  double host_wall_milliseconds = 0.0;
  bool used_unified_memory = false;
  bool explicit_synchronization = true;
  CudaTimingLedger timing;
};

struct CudaLayerAResult {
  bool success = false;
  std::string error;
  std::vector<CudaVisualOutput> visual;
  std::vector<CudaLidarOutput> lidar;
  CudaRuntimeInfo runtime;
};

struct CudaPoseBlockOutput {
  uint32_t image_id = 0;
  uint32_t dimension = 0;
  int32_t free_translation_indices[3] = {-1, -1, -1};
  double hessian[36] = {};  // active row-major dimension x dimension
  double gradient[6] = {};
  double jacobi_scaling[6] = {};
  double damping[6] = {};
};

struct CudaPointBlockOutput {
  uint64_t point3D_id = 0;
  double hessian[9] = {};
  double gradient[3] = {};
  double jacobi_scaling[3] = {};
  double damping[3] = {};
};

struct CudaEdgeBlockOutput {
  uint32_t pose_index = 0;
  uint32_t point_index = 0;
  uint32_t pose_dimension = 0;
  double value[18] = {};  // active row-major pose_dimension x 3
};

struct CudaLayerBOptions {
  CudaLayerAOptions layer_a;
  CudaLossMode loss_mode = CudaLossMode::kFromSnapshot;
  double loss_scale = 1.0;
  int cost_reduction_threads = 1;
  CudaReductionMode reduction_mode = CudaReductionMode::kSerialDeterministic;
  CudaFaultInjection fault_injection = CudaFaultInjection::kNone;
  uint64_t memory_budget_override_bytes = 0;
  double min_lm_diagonal = 1e-6;
  double max_lm_diagonal = 1e32;
  CudaHessianAssemblyBackend hessian_assembly_backend =
      CudaHessianAssemblyBackend::kCompatibilityDefault;
  // Test/profile-only fixed segment size. Zero resolves to the frozen
  // candidate size. Production compatibility/reference paths ignore it.
  uint32_t hessian_segment_size_for_testing = 0;
  // Empty vectors request first-linearization Jacobi scaling.  Full LM stores
  // the initial values here so accepted relinearizations use the same scaling,
  // matching the validated custom_cpu/Ceres-1.14 semantics.
  std::vector<double> frozen_pose_jacobi_scaling;   // pose_count * 6
  std::vector<double> frozen_point_jacobi_scaling;  // point_count * 3
  // Internal full-solve path: the initial B slot published scaling directly
  // into solve-owned immutable device buffers. Standalone callers leave this
  // false and retain the public host-vector contract above.
  bool device_frozen_jacobi_scaling = false;
};

struct CudaLayerBRuntimeInfo {
  uint64_t metadata_bytes = 0;
  uint64_t adjacency_bytes = 0;
  uint64_t output_bytes = 0;
  int launch_status = 0;
  int synchronize_status = 0;
  double assembly_kernel_milliseconds = 0.0;
  double cost_kernel_milliseconds = 0.0;
  double gradient_kernel_milliseconds = 0.0;
  double allocation_milliseconds = 0.0;
  double synchronization_milliseconds = 0.0;
  double copy_milliseconds = 0.0;
  double host_wall_milliseconds = 0.0;
  CudaReductionMode reduction_mode = CudaReductionMode::kSerialDeterministic;
  uint64_t active_bytes = 0;
  uint64_t cached_bytes = 0;
  uint64_t resident_bytes = 0;
  uint64_t predicted_additional_bytes = 0;
  uint64_t predicted_peak_bytes = 0;
  uint64_t free_bytes_before = 0;
  uint64_t total_bytes = 0;
  bool cost_reduction_parallel = false;
  bool gradient_reduction_parallel = false;
  int configured_worker_count = 1;
  int effective_worker_count = 1;
  std::string hessian_assembly_backend_requested;
  std::string hessian_assembly_backend_effective;
  uint64_t hessian_gradient_assembly_calls = 0;
  uint64_t pose_block_assembly_calls = 0;
  uint64_t point_block_assembly_calls = 0;
  uint64_t edge_block_assembly_calls = 0;
  uint64_t jacobi_damping_finalize_calls = 0;
  uint64_t gradient_summary_calls = 0;
  uint64_t observation_atomic_kernel_launches = 0;
  uint64_t observation_segment_partial_launches = 0;
  uint64_t observation_segment_merge_launches = 0;
  uint64_t output_zero_kernel_launches = 0;
  uint64_t jacobi_finalize_kernel_launches = 0;
  uint64_t gradient_summary_launches = 0;
  uint64_t atomic_add_estimate_pose = 0;
  uint64_t atomic_add_estimate_point = 0;
  uint64_t atomic_add_estimate_edge = 0;
  uint64_t segment_count_pose = 0;
  uint64_t segment_count_point = 0;
  uint64_t segment_count_edge = 0;
  uint64_t segment_size = 0;
  uint64_t partial_workspace_bytes = 0;
  uint64_t scale_workspace_bytes = 0;
  uint64_t candidate_metadata_h2d_bytes = 0;
  uint64_t assembly_coverage_violations = 0;
  CudaTimingLedger timing;
};

struct CudaLayerBResult {
  bool success = false;
  std::string error;
  double cost = 0.0;
  CudaLayerAResult layer_a;
  std::vector<CudaPoseBlockOutput> poses;
  std::vector<CudaPointBlockOutput> points;
  std::vector<CudaEdgeBlockOutput> edges;
  double projected_gradient_max_norm = 0.0;
  double raw_tangent_gradient_max_norm = 0.0;
  double scaled_gradient_max_norm = 0.0;
  // device_control keeps the owning B blocks in its solve-owned device slot.
  // These two base values are enough for the host LM trace to reproduce the
  // radius-dependent diagonal range without materializing the blocks.
  bool compact_device_summary = false;
  double lm_diagonal_base_min = 0.0;
  double lm_diagonal_base_max = 0.0;
  CudaLayerBRuntimeInfo runtime;
};

struct CudaLinearizationIdentity {
  uint64_t solve_generation = 0;
  uint64_t state_epoch = 0;
  uint64_t topology_generation = 0;
  uint64_t config_generation = 0;
};

// Non-owning, solve-local binding for a device-resident linearization. The
// token contains no pointer and is excluded from all canonical hashes. It is
// valid only while the producing solve-owned device context is alive.
struct CudaDeviceLinearizationToken {
  bool valid = false;
  uint64_t context_identity = 0;
  uint64_t layout_id = 0;
  CudaArithmeticPrecision arithmetic_precision =
      CudaArithmeticPrecision::kFp64;
  CudaLinearizationIdentity linearization;
  uint32_t layer_b_slot = 0;
  uint64_t layer_b_slot_generation = 0;
  uint32_t state_slot = 0;
  uint64_t state_slot_generation = 0;
};

// Move-only, lambda-independent mathematical state for one current LM state.
// Layer C receives it only as a const borrow. In particular, this object never
// owns a damped point inverse, Schur complement, factorization, step, trial
// state, or trial cost.
class UndampedCurrentLinearization {
 public:
  UndampedCurrentLinearization() = default;
  UndampedCurrentLinearization(const UndampedCurrentLinearization&) = delete;
  UndampedCurrentLinearization& operator=(
      const UndampedCurrentLinearization&) = delete;
  UndampedCurrentLinearization(UndampedCurrentLinearization&&) noexcept =
      default;
  UndampedCurrentLinearization& operator=(
      UndampedCurrentLinearization&&) noexcept = default;

  static UndampedCurrentLinearization Adopt(
      const CudaLinearizationIdentity& identity,
      CudaLayerBResult&& layer_b,
      const CudaDeviceLinearizationToken& device_token = {}) noexcept;

  bool valid() const noexcept { return valid_; }
  const CudaLinearizationIdentity& identity() const noexcept {
    return identity_;
  }
  const CudaLayerBResult& layer_b() const noexcept { return layer_b_; }
  const CudaDeviceLinearizationToken& device_token() const noexcept {
    return device_token_;
  }
  const void* borrowed_layer_b_address() const noexcept { return &layer_b_; }
  void RebindIdentityForBootstrap(
      const CudaLinearizationIdentity& identity) noexcept {
    identity_ = identity;
    if (device_token_.valid) device_token_.linearization = identity;
  }
  void Swap(UndampedCurrentLinearization& other) noexcept;

 private:
  bool valid_ = false;
  CudaLinearizationIdentity identity_;
  CudaDeviceLinearizationToken device_token_;
  CudaLayerBResult layer_b_;
};

struct CudaLayerCOptions {
  CudaLayerBOptions layer_b;
  double lambda = 1e-4;
  CudaFaultInjection fault_injection = CudaFaultInjection::kNone;
  // Test-only execution-layout override. Zero selects the fixed production
  // limit; it never changes Layer C arithmetic or the public mapper options.
  uint64_t pair_chunk_limit_bytes_for_testing = 0;
  CudaSchurContributionBackend schur_contribution_backend =
      CudaSchurContributionBackend::kCompatibilityDefault;
  // Test/profile-only fixed segment override. Zero selects the frozen
  // candidate configuration. This is ignored by the direct backend.
  uint32_t schur_segment_size_for_testing = 0;
};

struct CudaLayerCRuntimeInfo {
  uint64_t predicted_peak_bytes = 0;
  uint64_t pair_adjacency_bytes = 0;
  uint64_t layer_c_pair_chunk_launch_calls = 0;
  uint64_t layer_c_max_chunks_per_step = 0;
  uint64_t layer_c_max_chunk_bytes = 0;
  uint64_t schur_contribution_calls = 0;
  uint64_t schur_rhs_calls = 0;
  uint64_t factorization_calls = 0;
  uint64_t direct_pair_count = 0;
  uint64_t segmented_pair_count = 0;
  uint64_t segment_count = 0;
  uint64_t max_segments_per_pair = 0;
  uint64_t segment_size = 0;
  uint64_t partial_workspace_bytes = 0;
  uint64_t segment_plan_builds = 0;
  uint64_t segment_metadata_h2d_bytes = 0;
  uint64_t workspace_init_kernel_launches = 0;
  uint64_t partial_kernel_launches = 0;
  uint64_t merge_kernel_launches = 0;
  uint64_t pair_contribution_chunk_launches = 0;
  uint64_t coverage_violations = 0;
  double schur_contribution_direct_milliseconds = 0.0;
  double schur_contribution_workspace_init_milliseconds = 0.0;
  double schur_contribution_partial_milliseconds = 0.0;
  double schur_contribution_merge_milliseconds = 0.0;
  double schur_contribution_inclusive_milliseconds = 0.0;
  uint64_t cusolver_workspace_bytes = 0;
  uint64_t free_bytes_before = 0;
  uint64_t total_bytes = 0;
  uint64_t active_bytes = 0;
  uint64_t cached_bytes = 0;
  uint64_t resident_bytes = 0;
  uint64_t predicted_additional_bytes = 0;
  int cusolver_create_status = 0;
  int cusolver_set_stream_status = 0;
  int cusolver_buffer_size_status = 0;
  int cusolver_potrf_status = 0;
  int cusolver_potrs_status = 0;
  int cusolver_dev_info = 0;
  int cublas_create_status = 0;
  int cublas_set_stream_status = 0;
  int cublas_copy_status = 0;
  int cuda_launch_status = 0;
  int cuda_synchronize_status = 0;
  uint64_t point_factorization_failures = 0;
  int64_t first_point_factorization_failure = -1;
  double point_kernel_milliseconds = 0.0;
  double schur_kernel_milliseconds = 0.0;
  double factorization_milliseconds = 0.0;
  double back_substitution_milliseconds = 0.0;
  double trial_cost_kernel_milliseconds = 0.0;
  double allocation_milliseconds = 0.0;
  double synchronization_milliseconds = 0.0;
  double copy_milliseconds = 0.0;
  double host_wall_milliseconds = 0.0;
  CudaTimingLedger timing;
};

// Lambda-dependent output of one borrowed-linearization Layer C invocation.
// Unlike the legacy standalone result, it never owns or copies Layer B.
struct CudaLayerCStepResult {
  bool success = false;
  std::string error;
  uint64_t pose_dimension = 0;
  uint64_t point_dimension = 0;
  std::vector<double> schur;
  std::vector<double> factor;
  std::vector<double> rhs;
  std::vector<double> camera_delta;
  std::vector<double> point_delta;
  double symmetry_error = 0.0;
  double backward_error = 0.0;
  double schur_solve_backward_error = 0.0;
  double predicted_reduction = 0.0;
  double state_norm = 0.0;
  double step_norm = 0.0;
  double lm_diagonal_min = 0.0;
  double lm_diagonal_max = 0.0;
  double trial_cost = 0.0;
  Snapshot trial_state;
  CudaLayerCRuntimeInfo runtime;
};

struct CudaLayerCResult {
  bool success = false;
  std::string error;
  CudaLayerBResult layer_b;
  uint64_t pose_dimension = 0;
  uint64_t point_dimension = 0;
  std::vector<double> schur;
  std::vector<double> factor;
  std::vector<double> rhs;
  std::vector<double> camera_delta;
  std::vector<double> point_delta;
  double symmetry_error = 0.0;
  double backward_error = 0.0;
  double predicted_reduction = 0.0;
  double trial_cost = 0.0;
  Snapshot trial_state;
  CudaLayerCRuntimeInfo runtime;
};

enum class CudaTerminationType : uint8_t {
  kConvergence = 0,
  kNoConvergence = 1,
  kFailure = 2,
};

enum class CudaSolveErrorClass : uint8_t {
  kNone = 0,
  kInvalidOptions = 1,
  kInitialLinearization = 2,
  kCurrentLinearization = 3,
  kCudaStep = 4,
  kTrialStateHash = 5,
  kNonfiniteModel = 6,
  kAcceptedPendingLinearization = 7,
  kResourceCleanup = 8,
  kFinalStateHash = 9,
  kAcceptedPendingPreparation = 10,
  kCudaTiming = 11,
  kAuditCapacity = 12,
};

constexpr size_t kCudaFaultRecordCapacity = 64;
constexpr size_t kCudaSecondaryDiagnosticCapacity = 8;
constexpr size_t kCudaTeardownRecordCapacity = 7;
constexpr size_t kCudaResourceRegistryCapacity = 65536;
constexpr size_t kCudaMaxOpenTimingIntervalCapacity = 64;

struct CudaStableDiagnosticV2 {
  CudaSolveErrorClass error_classification = CudaSolveErrorClass::kNone;
  CudaFailureSite failure_site = CudaFailureSite::kNone;
  CudaStatusDomain status_domain = CudaStatusDomain::kNone;
  int64_t raw_status_code = 0;
};

struct CudaFaultConsumptionV2 {
  CudaFaultLogicalSite logical_site = CudaFaultLogicalSite::kNone;
  CudaFaultTriggerPhase trigger_phase = CudaFaultTriggerPhase::kNone;
  uint16_t operation = 0;
  CudaFaultInjection configured_fault_kind = CudaFaultInjection::kNone;
  uint64_t epoch = 0;
  uint64_t occurrence = 0;
  bool triggered = false;
};

struct CudaTeardownRecordV2 {
  CudaTeardownType teardown_type = CudaTeardownType::kGetDevice;
  int32_t device = -1;
  uint64_t creation_sequence = 0;
  bool attempted = false;
  bool success = false;
  CudaStatusDomain status_domain = CudaStatusDomain::kNone;
  int64_t raw_status_code = 0;
  bool quarantined = false;
};

struct CudaTimingStructureV2 {
  CudaTimingStatus timing_status = CudaTimingStatus::kDisabled;
  uint64_t event_attempts = 0;
  uint64_t event_completions = 0;
  uint64_t event_failures = 0;
  uint64_t intervals_started = 0;
  uint64_t intervals_completed = 0;
  uint64_t intervals_abandoned = 0;
  uint64_t open_intervals_at_finalize = 0;
  bool has_first_incomplete_site = false;
  CudaFailureSite first_incomplete_site = CudaFailureSite::kNone;
};

struct CudaDiagnosticStructureV2 {
  bool has_primary = false;
  CudaStableDiagnosticV2 primary;
  uint64_t secondary_count = 0;
  std::array<CudaStableDiagnosticV2, kCudaSecondaryDiagnosticCapacity>
      secondary{};
  bool overflow_sentinel_set = false;
  uint64_t diagnostic_overflow_count = 0;
  CudaStableDiagnosticV2 overflow_sentinel;
};

struct CudaLmIteration {
  int32_t iteration = 0;
  double cost_before = 0.0;
  double trial_cost = 0.0;
  double cost_after = 0.0;
  double projected_gradient_max_norm = 0.0;
  double scaled_gradient_norm = 0.0;
  double radius_before = 0.0;
  double radius_after = 0.0;
  double lambda_before = 0.0;
  double lambda_after = 0.0;
  double lm_diagonal_min = 0.0;
  double lm_diagonal_max = 0.0;
  double predicted_reduction = 0.0;
  double actual_reduction = 0.0;
  double rho = 0.0;
  double function_metric = 0.0;
  double parameter_metric = 0.0;
  double step_norm = 0.0;
  double backward_error = 0.0;
  bool factorization_success = false;
  bool step_valid = false;
  bool trial_finite = false;
  bool accepted_decision = false;
  bool accepted_commit_success = false;
  bool accepted = false;
  bool invalid = false;
  std::string termination_reason;
  std::string current_state_hash;
  std::string trial_state_hash;
  uint64_t state_epoch = 0;
  uint64_t linearization_id = 0;
  std::string linearization_reason;
  uint64_t delta_layer_a_calls = 0;
  uint64_t delta_layer_b_calls = 0;
  uint64_t delta_layer_c_calls = 0;
  uint64_t delta_cost_calls = 0;
  uint64_t delta_topology_builds = 0;
  uint64_t delta_topology_refreshes = 0;
  uint64_t delta_cache_hits = 0;
  uint64_t delta_cache_lookups = 0;
  uint64_t delta_cache_misses = 0;
  uint64_t delta_cache_invalidations = 0;
  uint64_t cumulative_layer_a_calls = 0;
  uint64_t cumulative_layer_b_calls = 0;
  uint64_t cumulative_layer_c_calls = 0;
  uint64_t cumulative_cost_calls = 0;
  uint64_t cumulative_topology_builds = 0;
  uint64_t cumulative_topology_refreshes = 0;
  uint64_t cumulative_cache_hits = 0;
};

struct CudaFullLmOptions {
  CudaLayerCOptions layer_c;
  // Full-LM test-only override copied into the resolved Layer C options once
  // at solve initialization. Zero selects the 64 MiB production limit.
  uint64_t pair_chunk_limit_bytes_for_testing = 0;
  // Explicit production selections take precedence over compatibility
  // environment variables. Replay keeps the compatibility defaults.
  CudaDeviceContextMode device_context_mode =
      CudaDeviceContextMode::kCompatibilityDefault;
  CudaHotKernelMode hot_kernel_mode =
      CudaHotKernelMode::kCompatibilityDefault;
  CudaArithmeticPrecision arithmetic_precision =
      CudaArithmeticPrecision::kCompatibilityDefault;
  CudaExecutionProfile execution_profile = CudaExecutionProfile::kBaseline;
  CudaAuditProfile audit_profile = CudaAuditProfile::kCompatibilityDefault;
  int32_t max_num_iterations = -1;  // negative: use snapshot value
  int32_t max_num_consecutive_invalid_steps = -1;
  double max_solver_time_in_seconds = 1e9;
  double function_tolerance = -1.0;   // negative: use snapshot value
  double gradient_tolerance = -1.0;   // negative: use snapshot value
  double parameter_tolerance = -1.0;  // negative: use snapshot value
  double initial_trust_region_radius = 1e4;
  double min_trust_region_radius = 1e-32;
  double max_trust_region_radius = 1e16;
  double min_relative_decrease = 1e-3;
  bool capture_state_trace = false;
  // Performance mode is opt-in and never changes the correctness default.
  // It selects deterministic parallel reductions; persistent-resource reuse
  // is reported separately until the context cache is enabled.
  bool performance_mode = false;
  CudaInstrumentationMode instrumentation_mode =
      CudaInstrumentationMode::kDisabled;
  CudaCurrentLinearizationCacheMode current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kDisabled;
  CudaFaultInjection fault_injection = CudaFaultInjection::kNone;
  CudaFaultTrigger fault_trigger;
  // Zero selects the fixed production audit capacities. Non-zero values are
  // test-only preflight limits and never alter solver arithmetic.
  uint64_t audit_fault_record_capacity_for_testing = 0;
  uint64_t audit_secondary_capacity_for_testing = 0;
  uint64_t audit_resource_registry_capacity_for_testing = 0;
  uint64_t audit_timing_interval_capacity_for_testing = 0;
  // Non-owning solve-local prepared view. Standalone and replay leave this
  // null. Its host data is immutable and contains no CUDA pointer.
  const PreparedHostSolveView* prepared_host_view = nullptr;
  // Phase 11b-b internal indexed inputs. Both pointers must be present or
  // absent together. Their host lifetime covers RunCustomCudaSolve; neither
  // object contains a device pointer.
  const IndexedActiveSolveDescriptor* indexed_active_solve = nullptr;
  const MapperStaticCatalogStableTables* indexed_catalog_tables = nullptr;
};

// Phase 7.2a structural counters. These fields audit resource ownership and
// host/device boundaries only; they are deliberately excluded from every
// semantic/execution canonical hash.
struct CudaPersistentDeviceRuntimeInfo {
  bool requested = false;
  bool effective = false;
  bool close_succeeded = false;
  uint64_t context_identity = 0;
  uint64_t initialization_allocation_calls = 0;
  uint64_t initialization_allocation_bytes = 0;
  uint64_t post_initialize_allocation_calls = 0;
  uint64_t post_initialize_allocation_bytes = 0;
  uint64_t init_stream_create_count = 0;
  uint64_t init_event_create_count = 0;
  uint64_t init_solver_create_count = 0;
  uint64_t init_blas_create_count = 0;
  uint64_t steady_stream_create_count = 0;
  uint64_t steady_event_create_count = 0;
  uint64_t steady_solver_create_count = 0;
  uint64_t steady_blas_create_count = 0;
  uint64_t steady_stream_destroy_count = 0;
  uint64_t steady_event_destroy_count = 0;
  uint64_t steady_solver_destroy_count = 0;
  uint64_t steady_blas_destroy_count = 0;
  uint64_t close_stream_destroy_count = 0;
  uint64_t close_event_destroy_count = 0;
  uint64_t close_solver_destroy_count = 0;
  uint64_t close_blas_destroy_count = 0;
  uint64_t static_upload_calls = 0;
  uint64_t static_upload_bytes = 0;
  uint64_t dynamic_state_upload_calls = 0;
  uint64_t dynamic_state_upload_bytes = 0;
  uint64_t host_controller_d2h_calls = 0;
  uint64_t host_controller_d2h_bytes = 0;
  uint64_t host_diagnostics_d2h_calls = 0;
  uint64_t host_diagnostics_d2h_bytes = 0;
  uint64_t a_to_b_h2d_calls = 0;
  uint64_t a_to_b_h2d_bytes = 0;
  uint64_t a_to_b_d2h_calls = 0;
  uint64_t a_to_b_d2h_bytes = 0;
  uint64_t a_to_cost_h2d_calls = 0;
  uint64_t a_to_cost_h2d_bytes = 0;
  uint64_t a_to_cost_d2h_calls = 0;
  uint64_t a_to_cost_d2h_bytes = 0;
  uint64_t b_to_c_h2d_calls = 0;
  uint64_t b_to_c_h2d_bytes = 0;
  uint64_t b_to_c_d2h_calls = 0;
  uint64_t b_to_c_d2h_bytes = 0;
  uint64_t pending_commit_device_copy_bytes = 0;
  uint64_t large_b_copy_bytes = 0;
  uint64_t frozen_scaling_upload_calls = 0;
  uint64_t frozen_scaling_upload_bytes = 0;
  uint64_t cusolver_workspace_capacity_bytes = 0;
  uint64_t workspace_capacity_bytes = 0;
  uint64_t peak_resident_bytes = 0;
  uint64_t slot_publish_count = 0;
  uint64_t slot_swap_count = 0;
  uint64_t slot_invalidate_count = 0;
  uint64_t stale_handle_reject_count = 0;
  // Phase 7.2b device-state lifecycle evidence. These are structural audit
  // values only and remain outside Semantic V1/V2 and execution hashes.
  uint64_t static_problem_layout_builds = 0;
  uint64_t initial_entity_state_pack_calls = 0;
  uint64_t initial_entity_state_pack_bytes = 0;
  uint64_t initial_entity_state_upload_calls = 0;
  uint64_t initial_entity_state_upload_bytes = 0;
  uint64_t post_ready_dynamic_state_h2d_calls = 0;
  uint64_t post_ready_dynamic_state_h2d_bytes = 0;
  uint64_t per_observation_dynamic_pack_calls = 0;
  uint64_t per_observation_dynamic_pack_bytes = 0;
  uint64_t state_update_kernel_calls = 0;
  uint64_t trial_entity_state_d2h_calls = 0;
  uint64_t trial_entity_state_d2h_bytes = 0;
  uint64_t final_entity_state_d2h_calls = 0;
  uint64_t final_entity_state_d2h_bytes = 0;
  uint64_t state_slot_publish_count = 0;
  uint64_t state_slot_swap_count = 0;
  uint64_t state_slot_discard_count = 0;
  uint64_t state_slot_invalidate_count = 0;
  uint64_t reject_state_slot_swap_count = 0;
  uint64_t state_lineage_checks = 0;
  uint64_t state_lineage_violations = 0;
  uint64_t state_b_pair_checks = 0;
  uint64_t state_b_pair_violations = 0;
  uint64_t accepted_commit_device_copy_bytes = 0;
  uint64_t cross_solve_state_handle_hits = 0;
  // Phase 7.2c device-control boundary evidence. These counters are
  // non-canonical and do not participate in Semantic V1/V2.
  uint64_t scalar_packet_d2h_calls = 0;
  uint64_t scalar_packet_d2h_bytes = 0;
  uint64_t scalar_packet_size_bytes = 0;
  uint64_t steady_full_b_d2h_calls = 0;
  uint64_t steady_full_b_d2h_bytes = 0;
  uint64_t steady_schur_d2h_calls = 0;
  uint64_t steady_schur_d2h_bytes = 0;
  uint64_t steady_delta_d2h_calls = 0;
  uint64_t steady_delta_d2h_bytes = 0;
  uint64_t steady_trial_state_d2h_calls = 0;
  uint64_t steady_trial_state_d2h_bytes = 0;
  uint64_t final_state_materialization_operations = 0;
  uint64_t final_state_d2h_calls = 0;
  uint64_t final_state_d2h_bytes = 0;
  uint64_t bootstrap_scaling_mirror_calls = 0;
  uint64_t bootstrap_scaling_mirror_bytes = 0;
  uint64_t audit_mirror_b_calls = 0;
  uint64_t audit_mirror_b_bytes = 0;
  uint64_t audit_mirror_state_calls = 0;
  uint64_t audit_mirror_state_bytes = 0;
  uint64_t audit_mirror_schur_calls = 0;
  uint64_t audit_mirror_schur_bytes = 0;
  uint64_t audit_mirror_delta_calls = 0;
  uint64_t audit_mirror_delta_bytes = 0;
  uint64_t diagnostic_kernel_calls = 0;
  uint64_t diagnostic_reduction_failures = 0;
  double diagnostic_kernel_milliseconds = 0.0;
  double diagnostic_point_status_milliseconds = 0.0;
  double diagnostic_lm_diagonal_milliseconds = 0.0;
  double diagnostic_pose_partial_milliseconds = 0.0;
  double diagnostic_point_partial_milliseconds = 0.0;
  double diagnostic_merge_milliseconds = 0.0;
  double diagnostic_state_norm_milliseconds = 0.0;
  double diagnostic_symmetry_milliseconds = 0.0;
  double diagnostic_packet_copy_sync_milliseconds = 0.0;
  double diagnostic_total_inclusive_milliseconds = 0.0;
  double diagnostic_all_added_inclusive_milliseconds = 0.0;
  uint64_t diagnostic_workspace_bytes = 0;
  double diagnostic_last_symmetry_error = 0.0;
  double diagnostic_max_symmetry_error = 0.0;
  uint64_t commit_token_checks = 0;
  uint64_t commit_token_violations = 0;
  uint64_t b_slot_swap_count = 0;
  uint64_t reject_b_slot_swap_count = 0;
  // Phase 7.3b hot-kernel timings are non-canonical. Events are recorded on
  // the existing solve stream and read only after an existing dependency
  // synchronization point.
  bool hot_kernel_optimized = false;
  bool hot_kernel_transformed = false;
  uint64_t transform_kernel_calls = 0;
  uint64_t transform_workspace_bytes = 0;
  double transform_kernel_milliseconds = 0.0;
  double transformed_pair_kernel_milliseconds = 0.0;
  double layer_b_pose_milliseconds = 0.0;
  double layer_b_point_milliseconds = 0.0;
  double layer_b_edge_milliseconds = 0.0;
  double layer_b_gradient_milliseconds = 0.0;
  double layer_b_total_inclusive_milliseconds = 0.0;
  double schur_zero_milliseconds = 0.0;
  double schur_pose_init_milliseconds = 0.0;
  double schur_pair_milliseconds = 0.0;
  double schur_rhs_milliseconds = 0.0;
  double schur_preservation_copy_milliseconds = 0.0;
  double schur_factorization_milliseconds = 0.0;
  double schur_back_substitution_milliseconds = 0.0;
  double schur_total_inclusive_milliseconds = 0.0;
  uint64_t layer_c_pair_chunk_launch_calls = 0;
  uint64_t layer_c_max_chunks_per_step = 0;
  uint64_t layer_c_max_chunk_bytes = 0;
  std::string schur_contribution_backend_requested;
  std::string schur_contribution_backend_effective;
  uint64_t schur_contribution_calls = 0;
  uint64_t schur_rhs_calls = 0;
  uint64_t factorization_calls = 0;
  uint64_t direct_pair_count = 0;
  uint64_t segmented_pair_count = 0;
  uint64_t segment_count = 0;
  uint64_t max_segments_per_pair = 0;
  uint64_t schur_segment_size = 0;
  uint64_t schur_partial_workspace_bytes = 0;
  uint64_t schur_segment_plan_builds = 0;
  uint64_t schur_segment_metadata_h2d_bytes = 0;
  uint64_t schur_workspace_init_kernel_launches = 0;
  uint64_t schur_partial_kernel_launches = 0;
  uint64_t schur_merge_kernel_launches = 0;
  uint64_t schur_pair_contribution_chunk_launches = 0;
  uint64_t schur_contribution_coverage_violations = 0;
  double schur_contribution_direct_milliseconds = 0.0;
  double schur_contribution_workspace_init_milliseconds = 0.0;
  double schur_contribution_partial_milliseconds = 0.0;
  double schur_contribution_merge_milliseconds = 0.0;
  double schur_contribution_inclusive_milliseconds = 0.0;
  double solve_context_initialization_wall_milliseconds = 0.0;
  double problem_topology_preparation_wall_milliseconds = 0.0;
  double schur_segment_plan_host_milliseconds = 0.0;

  // Phase 10.0b observation-centric Hessian/gradient assembly. These are
  // always-on structural counters, O(1), and excluded from all semantic
  // hashes.
  std::string hessian_assembly_backend_requested;
  std::string hessian_assembly_backend_effective;
  uint64_t hessian_gradient_assembly_calls = 0;
  uint64_t pose_block_assembly_calls = 0;
  uint64_t point_block_assembly_calls = 0;
  uint64_t edge_block_assembly_calls = 0;
  uint64_t jacobi_damping_finalize_calls = 0;
  uint64_t gradient_summary_calls = 0;
  uint64_t observation_atomic_kernel_launches = 0;
  uint64_t observation_segment_partial_launches = 0;
  uint64_t observation_segment_merge_launches = 0;
  uint64_t robust_scale_kernel_launches = 0;
  uint64_t output_zero_kernel_launches = 0;
  uint64_t partial_clear_kernel_launches = 0;
  uint64_t symmetry_finalize_launches = 0;
  uint64_t jacobi_finalize_kernel_launches = 0;
  uint64_t gradient_summary_launches = 0;
  uint64_t atomic_add_estimate_pose = 0;
  uint64_t atomic_add_estimate_point = 0;
  uint64_t atomic_add_estimate_edge = 0;
  uint64_t hessian_segment_count_pose = 0;
  uint64_t hessian_segment_count_point = 0;
  uint64_t hessian_segment_count_edge = 0;
  uint64_t hessian_segment_size = 0;
  uint64_t hessian_partial_workspace_bytes = 0;
  uint64_t hessian_scale_workspace_bytes = 0;
  uint64_t hessian_candidate_metadata_h2d_bytes = 0;
  uint64_t hessian_candidate_metadata_build_calls = 0;
  uint64_t hessian_assembly_coverage_violations = 0;
  double hessian_candidate_metadata_host_milliseconds = 0.0;

  // Phase 10.1a precision experiment. These fields are structural evidence,
  // remain outside semantic hashes, and do not alter the FP64 compatibility
  // path. String fields describe the actual arithmetic/storage boundary.
  std::string arithmetic_precision_requested = "compatibility_default";
  std::string arithmetic_precision_effective = "fp64";
  std::string state_storage_precision = "fp64";
  std::string residual_jacobian_precision = "fp64";
  std::string hessian_schur_precision = "fp64";
  std::string factorization_routine = "cusolverDnDpotrf/Dpotrs";
  std::string delta_precision = "fp64";
  std::string quaternion_plus_precision = "fp64";
  std::string cost_precision = "fp64";
  std::string controller_precision = "fp64";
  uint64_t float_buffer_allocation_calls = 0;
  uint64_t float_buffer_allocation_bytes = 0;
  uint64_t float_static_upload_calls = 0;
  uint64_t float_static_upload_bytes = 0;
  uint64_t float_arena_reserved_bytes = 0;
  uint64_t float_workspace_bytes = 0;
  uint64_t double_workspace_bytes = 0;
  uint64_t spotrf_calls = 0;
  uint64_t spotrs_calls = 0;
  uint64_t dpotrf_calls = 0;
  uint64_t dpotrs_calls = 0;
  uint64_t fp64_cost_calls = 0;
  uint64_t float_state_update_calls = 0;
  uint64_t state_quantization_calls = 0;
  uint64_t precision_mirror_cross_hits = 0;
  uint64_t fixed_external_write_attempts = 0;

  // Phase 10.1b stable mixed-precision evidence. These counters are launch-
  // site or byte-accounting values and remain outside semantic hashes.
  std::string mixed_schur_math_effective = "none";
  uint64_t mixed_float_buffer_bytes = 0;
  uint64_t mixed_double_buffer_bytes = 0;
  uint64_t mixed_float_to_double_schur_calls = 0;
  uint64_t mixed_float_to_double_schur_bytes = 0;
  uint64_t mixed_fp64_gradient_calls = 0;
  uint64_t mixed_fp64_point_inverse_calls = 0;
  uint64_t mixed_fp64_rhs_calls = 0;
  uint64_t mixed_fp64_back_substitution_calls = 0;
  uint64_t mixed_scalar_packet_calls = 0;
  uint64_t mixed_scalar_packet_bytes = 0;
  uint64_t mixed_full_array_d2h_bytes = 0;
  uint64_t mixed_serial_full_scan_kernel_count = 0;
  // Phase 10.1c native-FP32 Schur / true-FP64 gradient provenance. These
  // counters are always-on launch/byte evidence and remain non-semantic.
  std::string hessian_precision = "fp64";
  std::string residual_jacobian_operand_source = "fp64_native";
  std::string gradient_operand_precision = "fp64";
  std::string gradient_accumulation_precision = "fp64";
  std::string point_inverse_precision = "fp64";
  std::string schur_contribution_precision = "fp64";
  std::string dense_factorization_precision = "fp64";
  uint64_t mixed_fp64_gradient_from_fp64_records_calls = 0;
  uint64_t mixed_fp64_gradient_from_fp32_records_calls = 0;
  uint64_t mixed_native_fp32_point_inverse_cast_calls = 0;
  uint64_t mixed_native_fp32_point_inverse_cast_bytes = 0;
  uint64_t mixed_native_fp32_transform_calls = 0;
  uint64_t mixed_native_fp32_transform_bytes = 0;
  uint64_t mixed_native_fp32_schur_calls = 0;
  uint64_t mixed_native_fp32_schur_partial_calls = 0;
  uint64_t mixed_native_fp32_schur_merge_calls = 0;
  uint64_t mixed_native_fp32_dense_conversion_calls = 0;
  uint64_t mixed_native_fp32_dense_conversion_source_bytes = 0;
  uint64_t mixed_native_fp32_dense_conversion_destination_bytes = 0;
  uint64_t mixed_layer_a_cast_calls = 0;
  uint64_t mixed_layer_a_cast_bytes = 0;
  uint64_t mixed_fp64_pose_damping_calls = 0;
  uint64_t mixed_fp64_schur_contribution_calls = 0;
  uint64_t mixed_double_edge_materialization_calls = 0;
  uint64_t mixed_double_edge_materialization_bytes = 0;
  double mixed_input_cast_milliseconds = 0.0;
  double mixed_residual_jacobian_milliseconds = 0.0;
  double mixed_hessian_milliseconds = 0.0;
  double mixed_gradient_milliseconds = 0.0;
  double mixed_point_inverse_milliseconds = 0.0;
  double mixed_schur_milliseconds = 0.0;
  double mixed_rhs_milliseconds = 0.0;
  double mixed_back_substitution_milliseconds = 0.0;
  double mixed_diagnostics_milliseconds = 0.0;
  double schur_solve_backward_error_last = 0.0;
  double schur_solve_backward_error_max = 0.0;

  // Phase 9 cumulative profile evidence. All fields are structural and are
  // excluded from semantic/execution canonical hashes.
  CudaExecutionProfile execution_profile = CudaExecutionProfile::kBaseline;
  uint64_t runtime_pool_acquire_calls = 0;
  uint64_t runtime_pool_cold_creates = 0;
  uint64_t runtime_pool_hot_leases = 0;
  uint64_t runtime_pool_returns = 0;
  uint64_t runtime_pool_retained_entries = 0;
  uint64_t runtime_pool_discards = 0;
  uint64_t runtime_pool_destroys = 0;
  uint64_t runtime_pool_poison_count = 0;
  uint64_t runtime_pool_shutdown_calls = 0;
  uint64_t runtime_pool_lease_generation = 0;
  uint64_t runtime_pool_stale_lease_rejects = 0;
  uint64_t arena_capacity_bytes = 0;
  uint64_t arena_required_bytes = 0;
  uint64_t arena_retained_bytes = 0;
  uint64_t arena_slice_count = 0;
  uint64_t arena_slice_bytes = 0;
  uint64_t arena_grow_calls = 0;
  uint64_t arena_grow_bytes = 0;
  uint64_t arena_hot_cuda_malloc_calls = 0;
  uint64_t arena_hot_cuda_free_calls = 0;
  uint64_t bootstrap_scaling_full_d2h_calls = 0;
  uint64_t bootstrap_scaling_full_d2h_bytes = 0;
  uint64_t scaling_d2d_calls = 0;
  uint64_t scaling_d2d_bytes = 0;
  uint64_t scaling_slot_publishes = 0;
  uint64_t scaling_host_mirror_materializations = 0;
  uint64_t production_identity_fingerprint_calls = 0;
  uint64_t production_identity_fingerprint_bytes = 0;
  uint64_t unified_problem_builder_calls = 0;
  uint64_t unified_problem_builder_traversals = 0;
  uint64_t independent_cost_order_rebuilds = 0;
  uint64_t host_build_cuda_layer_a_inputs_calls = 0;
  uint64_t host_build_static_layout_calls = 0;
  uint64_t host_build_cost_layout_calls = 0;
  uint64_t host_build_layer_b_topology_calls = 0;
  uint64_t host_build_layer_c_topology_calls = 0;
  uint64_t compact_visual_record_bytes = 0;
  uint64_t public_visual_record_bytes = 0;
  uint64_t compact_layer_a_slot_bytes = 0;
  uint64_t full_layer_a_slot_bytes = 0;
  uint64_t compact_layer_a_calls = 0;
  uint64_t scalar_decision_packet_calls = 0;
  uint64_t scalar_decision_packet_bytes = 0;
  uint64_t scalar_current_packet_calls = 0;
  uint64_t scalar_factor_status_packet_calls = 0;
  uint64_t scalar_trial_packet_calls = 0;
  uint64_t redundant_synchronization_calls = 0;
  // Phase 11b-b indexed catalog ownership. These are structural counters and
  // remain outside semantic/execution hashes.
  uint64_t indexed_device_catalog_lookup_calls = 0;
  uint64_t indexed_device_catalog_reuse_calls = 0;
  uint64_t indexed_device_catalog_revision_rebinds = 0;
  uint64_t indexed_device_catalog_full_upload_calls = 0;
  uint64_t indexed_device_catalog_full_upload_bytes = 0;
  uint64_t indexed_device_catalog_patch_upload_calls = 0;
  uint64_t indexed_device_catalog_patch_upload_bytes = 0;
  uint64_t indexed_device_catalog_invalidations = 0;
  uint64_t indexed_device_catalog_prefix_bytes = 0;
  uint64_t indexed_device_catalog_arena_generation = 0;
};

struct CudaFullLmRuntimeInfo {
  double layer_a_kernel_milliseconds = 0.0;
  double layer_b_kernel_milliseconds = 0.0;
  double point_kernel_milliseconds = 0.0;
  double schur_kernel_milliseconds = 0.0;
  double factorization_milliseconds = 0.0;
  double back_substitution_milliseconds = 0.0;
  double trial_cost_kernel_milliseconds = 0.0;
  double cost_reduction_kernel_milliseconds = 0.0;
  double gradient_reduction_kernel_milliseconds = 0.0;
  double allocation_milliseconds = 0.0;
  double synchronization_milliseconds = 0.0;
  double copy_milliseconds = 0.0;
  double host_wall_milliseconds = 0.0;
  uint64_t peak_predicted_bytes = 0;
  uint64_t peak_active_bytes = 0;
  uint64_t peak_cached_bytes = 0;
  uint64_t peak_resident_bytes = 0;
  uint64_t minimum_free_bytes = 0;
  bool single_stream_synchronous = true;
  bool buffers_reused_across_iterations = false;
  bool topology_reused_across_iterations = false;
  uint64_t buffer_reuse_hits = 0;
  uint64_t topology_reuse_hits = 0;
  uint64_t stream_reuse_hits = 0;
  uint64_t event_reuse_hits = 0;
  uint64_t solver_handle_reuse_hits = 0;
  uint64_t blas_handle_reuse_hits = 0;
  bool cache_access_serialized = true;
  bool cache_context_isolated = true;
  std::string audit_profile_requested = "compatibility_default";
  std::string audit_profile_effective = "compatibility_default";
  bool capture_state_trace_effective = false;
  bool production_audit_invariants_checked = false;
  bool production_audit_invariants_pass = false;
  uint64_t production_audit_violation_count = 0;
  std::string first_production_audit_violation;
  bool instrumentation_effective = false;
  bool performance_mode_requested = false;
  CudaReductionMode effective_reduction_mode =
      CudaReductionMode::kSerialDeterministic;
  CudaTimingLedger timing;
  CudaStateHashAudit state_hash_audit;
  std::string initial_state_hash;
  std::string final_state_hash;
  // Legacy Phase 7.0 instrumentation-dependent epoch. Do not use for new
  // semantic comparisons; retained to preserve historical decision artifacts.
  uint64_t final_state_epoch = 0;
  // Always-on controller epoch. Only a successful accepted-state commit
  // increments this value.
  uint64_t final_internal_state_epoch = 0;
  uint64_t final_linearization_id = 0;
  std::string final_linearization_reason;
  uint64_t layer_a_calls = 0;
  uint64_t layer_b_calls = 0;
  uint64_t layer_c_calls = 0;
  uint64_t cost_calls = 0;
  uint64_t topology_build_count = 0;
  uint64_t topology_refresh_count = 0;
  uint64_t cache_lookup_count = 0;
  uint64_t cache_hit_count = 0;
  uint64_t cache_miss_count = 0;
  uint64_t cache_invalidation_count = 0;
  uint64_t build_cuda_layer_a_inputs_count = 0;
  uint64_t build_cost_order_count = 0;
  uint64_t topology_epoch = 0;
  std::string topology_fingerprint;
  uint64_t topology_fingerprint_computations = 0;
  uint64_t topology_generation = 0;
  double cuda_solve_call_wall_milliseconds = 0.0;
  bool forced_reject_probe_requested = false;
  bool probe_valid = false;
  uint64_t actual_trials = 0;
  uint64_t accepted_trials = 0;
  uint64_t rejected_trials = 0;
  uint64_t current_linearization_logical_requests = 0;
  uint64_t current_linearization_lookup_aborts = 0;
  uint64_t current_linearization_cache_lookups = 0;
  uint64_t current_linearization_cache_hits = 0;
  uint64_t current_linearization_cache_misses = 0;
  uint64_t current_linearization_build_attempts = 0;
  uint64_t current_linearization_build_successes = 0;
  uint64_t current_linearization_build_failures = 0;
  uint64_t current_linearization_temporary_builds = 0;
  uint64_t current_linearization_publishes = 0;
  uint64_t current_linearization_replacements = 0;
  uint64_t current_linearization_invalidations = 0;
  uint64_t current_linearization_borrows = 0;
  uint64_t current_linearization_copies = 0;
  uint64_t current_linearization_teardowns = 0;
  std::vector<uint64_t> current_linearization_builds_by_state_epoch;
  uint64_t final_config_generation = 0;
  uint64_t final_topology_generation = 0;
  uint64_t final_solve_generation = 0;
  uintptr_t last_borrowed_layer_b_address = 0;
  bool current_linearization_cache_requested = false;
  bool current_linearization_cache_effective = false;
  CudaResourceHealth initial_resource_health = CudaResourceHealth::kClean;
  CudaResourceHealth final_resource_health = CudaResourceHealth::kClean;
  uint64_t resource_generation = 0;
  uint64_t resource_taint_count = 0;
  uint64_t resource_cleanup_attempts = 0;
  uint64_t resource_cleanup_successes = 0;
  uint64_t resource_cleanup_failures = 0;
  uint64_t resource_lookup_blocked_count = 0;
  uint64_t cross_solve_cache_hits = 0;
  bool audit_capacity_preflight_pass = false;
  uint64_t fault_record_capacity = kCudaFaultRecordCapacity;
  uint64_t secondary_diagnostic_capacity =
      kCudaSecondaryDiagnosticCapacity;
  uint64_t resource_registry_capacity = kCudaResourceRegistryCapacity;
  uint64_t max_open_timing_interval_capacity =
      kCudaMaxOpenTimingIntervalCapacity;
  uint64_t audit_overflow_count = 0;
  bool has_first_audit_overflow_site = false;
  CudaFailureSite first_audit_overflow_site = CudaFailureSite::kNone;
  uint64_t fault_consumption_count = 0;
  std::array<CudaFaultConsumptionV2, kCudaFaultRecordCapacity>
      fault_consumptions{};
  CudaTimingStructureV2 timing_structure_v2;
  CudaDiagnosticStructureV2 diagnostic_structure_v2;
  uint64_t resource_generation_advance_events = 0;
  uint64_t resource_generation_advance_violations = 0;
  bool resource_generation_advanced_by_this_solve = false;
  uint64_t resource_quarantine_count = 0;
  bool resource_cleanup_complete = true;
  uint64_t teardown_record_count = 0;
  std::array<CudaTeardownRecordV2, kCudaTeardownRecordCapacity>
      teardown_records{};
  bool has_first_cleanup_failure = false;
  CudaStableDiagnosticV2 first_cleanup_failure;
  uint64_t solve_generation_nonzero_checks = 0;
  uint64_t solve_generation_nonzero_violations = 0;
  bool solve_generation_nonzero = false;
  uint64_t solve_generation_consistency_checks = 0;
  uint64_t solve_generation_consistency_violations = 0;
  bool solve_generation_consistent = false;
  uint64_t linearization_identity_checks = 0;
  uint64_t linearization_identity_violations = 0;
  bool linearization_identity_consistent = false;
  uint64_t topology_context_generation_checks = 0;
  uint64_t topology_context_generation_violations = 0;
  bool topology_context_generation_consistent = false;
  // Non-canonical Phase 7.2a structural evidence.
  std::string device_context_backend = "legacy";
  CudaPersistentDeviceRuntimeInfo persistent_device;
  CudaHostProblemStoreRuntimeInfo host_problem_store;
};

struct CudaFullLmResult {
  bool success = false;
  std::string error;
  CudaTerminationType termination_type = CudaTerminationType::kFailure;
  CudaSolveErrorClass error_classification = CudaSolveErrorClass::kNone;
  std::string termination_reason;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double initial_projected_gradient_max_norm = 0.0;
  double final_projected_gradient_max_norm = 0.0;
  double initial_scaled_gradient_norm = 0.0;
  double final_scaled_gradient_norm = 0.0;
  double final_radius = 0.0;
  double final_lambda = 0.0;
  int32_t trial_iterations = 0;
  int32_t accepted_steps = 0;
  int32_t rejected_steps = 0;
  int32_t invalid_steps = 0;
  int32_t factorization_failures = 0;
  int32_t accepted_decisions = 0;
  int32_t accepted_commits = 0;
  int32_t accepted_pending_preparation_failures = 0;
  int32_t accepted_pending_linearization_failures = 0;
  std::vector<CudaLmIteration> trace;
  std::vector<Snapshot> accepted_state_trace;
  Snapshot final_state;
  CudaFullLmRuntimeInfo runtime;
};

// Build canonical, deterministic flat inputs from a Snapshot. This function
// performs only host-side lookup/packing; residual and Jacobian arithmetic is
// performed by the custom CUDA kernel in RunCudaLayerA.
bool BuildCudaLayerAInputs(const Snapshot& snapshot,
                           std::vector<CudaVisualInput>* visual,
                           std::vector<CudaLidarInput>* lidar,
                           std::string* error);

// VS1 native preparation overloads. All five consume the solve-owned view and
// pinned graph lease directly. The LegacyKernelInputBundle is explicitly
// temporary while the unchanged CUDA kernels retain their current flat ABI.
bool BuildCudaLayerAInputs(const NativeHostSolveView& view,
                           const ActiveStateBuffer& state,
                           LegacyKernelInputBundle* output,
                           std::string* error);
bool BuildStaticLayout(const NativeHostSolveView& view,
                       const ActiveStateBuffer& state,
                       LegacyKernelInputBundle* output,
                       std::string* error);
bool BuildCostLayout(const NativeHostSolveView& view,
                     LegacyKernelInputBundle* output,
                     std::string* error);
bool BuildLayerBTopology(const NativeHostSolveView& view,
                         LegacyKernelInputBundle* output,
                         std::string* error);
bool BuildLayerCTopology(const NativeHostSolveView& view,
                         LegacyKernelInputBundle* output,
                         std::string* error);
bool BuildLegacyKernelInputBundle(const NativeHostSolveView& view,
                                  const ActiveStateBuffer& state,
                                  LegacyKernelInputBundle* output,
                                  std::string* error);
// Test-only host copy. Production final state is downloaded from the current
// device slot by RunCustomCudaSolve(NativeCudaSolveRequest).
bool CopyActiveStateForTesting(const NativeHostSolveView& view,
                               const ActiveStateBuffer& state,
                               DenseActiveState* output,
                               std::string* error);

struct NativePreparationComparisonResult {
  uint64_t visual_count = 0;
  uint64_t lidar_count = 0;
  uint64_t cost_entry_count = 0;
  uint64_t pose_adjacency_count = 0;
  uint64_t point_adjacency_count = 0;
  uint64_t edge_adjacency_count = 0;
  uint64_t pair_contribution_count = 0;
  bool source_indices_dense = false;
  bool cost_entries_exact = false;
  bool layer_b_exact = false;
  bool layer_c_exact = false;
};

// Uses the production legacy host preparer and the native preparer on the
// same immutable entry values. No CUDA work or solve is performed.
bool CompareNativePreparationToLegacyForTesting(
    const Snapshot& legacy,
    const NativeHostSolveView& native_view,
    const ActiveStateBuffer& native_state,
    const CudaLayerBOptions& options,
    uint64_t pair_chunk_limit_bytes,
    CudaHotKernelMode hot_kernel_mode,
    CudaSchurContributionBackend schur_backend,
    uint32_t schur_segment_size,
    NativePreparationComparisonResult* result,
    std::string* error);

bool BuildCudaLayerAInputs(const Snapshot& snapshot,
                           CudaResidualOrder order,
                           std::vector<CudaVisualInput>* visual,
                           std::vector<CudaLidarInput>* lidar,
                           std::string* error);

// Execute Layer A on one synchronous CUDA stream. No Ceres API is called by
// this function. The explicit-copy mode is the initial correctness reference;
// managed mode is available for synchronization/page-migration diagnostics.
bool RunCudaLayerA(const std::vector<CudaVisualInput>& visual,
                   const std::vector<CudaLidarInput>& lidar,
                   const CudaLayerAOptions& options,
                   CudaLayerAResult* result);

bool RunCudaSnapshotLayerA(const Snapshot& snapshot,
                           const CudaLayerAOptions& options,
                           CudaLayerAResult* result,
                           std::string* error);

// Phase 9 component oracles. They exercise the production compact visual
// representation and device scaling publisher without exposing either
// internal ABI to public solver callers.
bool RunCudaCompactLayerAComparisonSelfTest(
    const std::vector<CudaVisualInput>& visual,
    double atol,
    double rtol,
    std::string* error);
bool RunCudaDeviceScalingComparisonSelfTest(
    const Snapshot& snapshot, double atol, double rtol, std::string* error);

// Deterministic Layer B assembly. Each pose, point, and E edge is owned by one
// CUDA thread and reduced through a canonical CSR list; no floating-point
// atomics are used. Fixed poses/points and subset translation masks are applied
// while the host builds the immutable topology.
bool RunCudaSnapshotLayerB(const Snapshot& snapshot,
                           const CudaLayerBOptions& options,
                           CudaLayerBResult* result,
                           std::string* error);

// FP64 dense-Schur CUDA single-step core. Point 3x3 blocks and deterministic
// camera-pair reductions are custom kernels; the dense SPD factorization uses
// cuSOLVER Dpotrf/Dpotrs. It does not call any Ceres controller or solver.
bool RunCudaSnapshotLayerC(const Snapshot& snapshot,
                           const CudaLayerCOptions& options,
                           CudaLayerCResult* result,
                           std::string* error);

// Independent FP64 LM orchestration.  It invokes only the custom CUDA layers,
// custom QuaternionPlus, cuSOLVER, and cuBLAS; it never calls ceres::Solve or a
// Ceres trust-region/linear-solver implementation.  Phase 6 initially keeps
// the synchronous correctness path; persistent-buffer optimization is gated
// behind full-LM fidelity.
bool RunCustomCudaSolve(const Snapshot& snapshot,
                        const CudaFullLmOptions& options,
                        CudaFullLmResult* result,
                        std::string* error);
bool RunCustomCudaSolve(const CudaSolveProblem& problem,
                        const CudaFullLmOptions& options,
                        CudaFullLmResult* result,
                        std::string* error);
bool RunCustomCudaSolve(const NativeCudaSolveRequest& request,
                        BaSolveResult* result,
                        std::string* error);

// Internal dispatch target for explicit Phase 10.1a precision experiments.
// Callers should use RunCustomCudaSolve so selector validation and telemetry
// remain centralized. It is declared here only because it lives in an
// isolated CUDA translation unit that cannot access the FP64 implementation's
// anonymous namespace.
bool RunCustomCudaFp32Experimental(const Snapshot& snapshot,
                                   const CudaFullLmOptions& options,
                                   CudaArithmeticPrecision precision,
                                   CudaFullLmResult* result,
                                   std::string* error);

// Explicit process-level lifecycle for the Phase 9 runtime pool. Production
// controllers call Shutdown while CUDA is still valid; tests may trim idle
// capacity or reset a deliberately poisoned pool between isolated cases.
struct CudaRuntimePoolShutdownInfo {
  uint64_t idle_entries_destroyed = 0;
  uint64_t streams_destroyed = 0;
  uint64_t events_destroyed = 0;
  uint64_t solvers_destroyed = 0;
  uint64_t blases_destroyed = 0;
  uint64_t arenas_destroyed = 0;
  uint64_t arena_bytes_released = 0;
};
bool ShutdownCudaRuntimePool(
    std::string* error, CudaRuntimePoolShutdownInfo* info = nullptr);
bool TrimIdleCudaRuntimePoolForTesting(std::string* error);
bool ResetCudaRuntimePoolForTesting(std::string* error);

// Test-only structural probe for stale persistent slot handles.
bool RunCudaPersistentHandleSelfTest(const Snapshot& snapshot,
                                     std::string* error);
bool RunCudaDeviceStateHandleSelfTest(const Snapshot& snapshot,
                                      std::string* error);

// Test-only direct oracle for the device QuaternionPlus used by the
// device-state update kernel.
bool RunCudaQuaternionPlusForTesting(
    const std::array<double, 4>& quaternion,
    const std::array<double, 3>& delta,
    std::array<double, 4>* result,
    uint32_t* status,
    std::string* error);

struct CudaDiagnosticReductionTestResult {
  uint64_t status_count = 0;
  int64_t first_status_index = -1;
  double first_norm = 0.0;
  double second_norm = 0.0;
  double maximum = 0.0;
  double diagonal_minimum = 0.0;
  double diagonal_maximum = 0.0;
  uint32_t finite_flags = 0;
  uint32_t reduction_status = 0;
};

// Test-only direct entry for the fixed-tree reduction primitives. Production
// solves use solve-owned preallocated workspaces; this helper owns temporary
// buffers so edge sizes and non-finite records can be tested independently.
bool RunCudaDiagnosticReductionForTesting(
    const std::vector<uint8_t>& status,
    const std::vector<std::array<double, 2>>& norm_partials,
    const std::vector<double>& maxima,
    CudaDiagnosticReductionTestResult* result,
    std::string* error);

struct CudaLayerCPairChunkingTestSummary {
  uint64_t production_limit_bytes = 0;
  uint64_t stress_pose_count = 0;
  uint64_t stress_pair_count = 0;
  uint64_t stress_contribution_count = 0;
  uint64_t stress_contribution_bytes = 0;
  uint64_t stress_chunk_count = 0;
  uint64_t stress_max_chunk_bytes = 0;
};

struct CudaSchurContributionHistogram {
  uint64_t pose_count = 0;
  uint64_t pair_count = 0;
  uint64_t contribution_count = 0;
  uint64_t whole_pair_chunk_count = 0;
  uint64_t empty_pair_count = 0;
  uint64_t single_contribution_pair_count = 0;
  uint64_t minimum = 0;
  uint64_t p50 = 0;
  uint64_t p90 = 0;
  uint64_t p95 = 0;
  uint64_t p99 = 0;
  uint64_t maximum = 0;
  std::vector<uint64_t> chunk_pair_begin;
  std::vector<uint64_t> chunk_pair_end;
  std::vector<uint64_t> chunk_contribution_begin;
  std::vector<uint64_t> chunk_contribution_end;
};

// Test/profile-only fixed-workload result. The helper constructs the current
// linearization, point inverse, transformed edges, initial Schur pose blocks,
// and immutable topology once, then alternates the direct and segmented
// contribution assemblers on independently reset Schur outputs.
struct CudaSchurContributionComponentResult {
  uint64_t pose_dimension = 0;
  uint64_t pair_count = 0;
  uint64_t contribution_count = 0;
  uint64_t direct_pair_count = 0;
  uint64_t segmented_pair_count = 0;
  uint64_t segment_count = 0;
  uint64_t max_segments_per_pair = 0;
  uint64_t segment_size = 0;
  uint64_t partial_workspace_bytes = 0;
  uint64_t coverage_violations = 0;
  uint64_t element_contract_violations = 0;
  double max_abs_error = 0.0;
  double normalized_max_error = 0.0;
  double normalized_frobenius_error = 0.0;
  double repeated_candidate_max_abs_error = 0.0;
  std::vector<double> direct_complete_milliseconds;
  std::vector<double> segmented_complete_milliseconds;
  std::vector<double> segmented_short_direct_milliseconds;
  std::vector<double> segmented_partial_milliseconds;
  std::vector<double> segmented_merge_milliseconds;
};

// Host-only topology probe. It executes no CUDA kernel and creates no CUDA
// resource; it exists to freeze the candidate workload before execution.
bool BuildCudaSchurContributionHistogramForTesting(
    const Snapshot& snapshot,
    CudaSchurContributionHistogram* histogram,
    std::string* error);

bool RunCudaSchurContributionComponentForTesting(
    const Snapshot& snapshot,
    uint32_t segment_size,
    size_t paired_samples,
    CudaSchurContributionComponentResult* result,
    std::string* error);

// Phase 10.0b fixed-workload harness. Unweighted visual/LiDAR records and all
// target metadata are built once; each backend runs its complete production
// assembly/finalize/gradient-summary kernel sequence into independent output.
struct CudaHessianAssemblyComponentResult {
  uint64_t visual_count = 0;
  uint64_t lidar_count = 0;
  uint64_t pose_count = 0;
  uint64_t point_count = 0;
  uint64_t edge_count = 0;
  uint64_t pose_segment_count = 0;
  uint64_t point_segment_count = 0;
  uint64_t point_direct_count = 0;
  uint64_t segment_size = 0;
  uint64_t partial_workspace_bytes = 0;
  uint64_t atomic_pose_add_estimate = 0;
  uint64_t atomic_point_add_estimate = 0;
  uint64_t atomic_edge_add_estimate = 0;
  uint64_t element_contract_violations_atomic = 0;
  uint64_t element_contract_violations_segmented = 0;
  double atomic_max_abs_error = 0.0;
  double atomic_normalized_max_error = 0.0;
  double atomic_normalized_frobenius_error = 0.0;
  double segmented_max_abs_error = 0.0;
  double segmented_normalized_max_error = 0.0;
  double segmented_normalized_frobenius_error = 0.0;
  double segmented_repeated_max_abs_error = 0.0;
  double atomic_repeated_max_abs_error = 0.0;
  uint64_t atomic_first_violation_index = 0;
  double atomic_first_violation_reference = 0.0;
  double atomic_first_violation_candidate = 0.0;
  uint64_t segmented_first_violation_index = 0;
  double segmented_first_violation_reference = 0.0;
  double segmented_first_violation_candidate = 0.0;
  std::vector<double> pose_owned_complete_milliseconds;
  std::vector<double> atomic_complete_milliseconds;
  std::vector<double> segmented_complete_milliseconds;
};

struct CudaPrecisionErrorStatistics {
  uint64_t count = 0;
  uint64_t finite_count = 0;
  uint64_t worst_index = 0;
  double reference_max_magnitude = 0.0;
  double candidate_max_magnitude = 0.0;
  double max_abs_error = 0.0;
  double max_relative_error = 0.0;
  double mean_abs_error = 0.0;
  double rms_abs_error = 0.0;
  double p50_abs_error = 0.0;
  double p95_abs_error = 0.0;
  double p99_abs_error = 0.0;
  double p50_relative_error = 0.0;
  double p95_relative_error = 0.0;
  double p99_relative_error = 0.0;
  double normalized_max_error = 0.0;
  double normalized_frobenius_error = 0.0;
  double worst_reference = 0.0;
  double worst_candidate = 0.0;
  std::string worst_entity_kind;
  uint64_t worst_entity_id = 0;
  uint32_t worst_row = 0;
  uint32_t worst_column = 0;
};

// Phase 10.1a test-only fixed-linearization oracle. The implementation runs
// the existing FP64 path and the explicit FP32 core on the same immutable
// snapshot/lambda, and reports only mathematically valid elements.
struct CudaPrecisionComponentResult {
  bool success = false;
  std::string error;
  std::string fp64_factorization_routine;
  std::string fp32_factorization_routine;
  int fp64_solver_info = 0;
  int fp32_solver_info = 0;
  uint64_t spotrf_calls = 0;
  uint64_t spotrs_calls = 0;
  uint64_t dpotrf_calls = 0;
  uint64_t dpotrs_calls = 0;
  CudaPrecisionErrorStatistics input_cast;
  CudaPrecisionErrorStatistics residual_jacobian;
  CudaPrecisionErrorStatistics robust_scale;
  CudaPrecisionErrorStatistics pose_hessian_gradient;
  CudaPrecisionErrorStatistics point_hessian_gradient;
  CudaPrecisionErrorStatistics edge_blocks;
  CudaPrecisionErrorStatistics jacobi_damping;
  CudaPrecisionErrorStatistics point_inverse;
  CudaPrecisionErrorStatistics transformed_edge;
  CudaPrecisionErrorStatistics schur;
  CudaPrecisionErrorStatistics rhs;
  CudaPrecisionErrorStatistics camera_delta;
  CudaPrecisionErrorStatistics point_delta;
  CudaPrecisionErrorStatistics trial_state;
  CudaPrecisionErrorStatistics fp64_trial_update_consistency;
  CudaPrecisionErrorStatistics fp32_trial_update_consistency;
  double fp64_predicted_reduction = 0.0;
  double fp32_predicted_reduction = 0.0;
  double fp64_backward_error = 0.0;
  double fp32_backward_error = 0.0;
  double fp64_trial_cost = 0.0;
  double fp32_trial_cost = 0.0;
  double pose_diagonal_min = 0.0;
  double pose_diagonal_max = 0.0;
  double point_positive_diagonal_min = 0.0;
  double point_diagonal_max = 0.0;
  double point_determinant_min_abs = 0.0;
  double point_inverse_residual_max = 0.0;
  double schur_diagonal_min = 0.0;
  double schur_diagonal_max = 0.0;
  bool schur_spd_reference = false;
  std::string schur_condition_metric = "diagonal_dynamic_range_proxy";
  double schur_min_eigenvalue = 0.0;
  double schur_max_eigenvalue = 0.0;
  double schur_condition_number = 0.0;
  uint32_t diagnostic_pose_image_id = 0;
  std::array<double, 4> diagnostic_initial_qvec{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> diagnostic_fp64_trial_qvec{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> diagnostic_fp32_trial_qvec{{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> diagnostic_fp64_camera_delta{{0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0}};
  std::array<double, 6> diagnostic_fp32_camera_delta{{0.0, 0.0, 0.0,
                                                      0.0, 0.0, 0.0}};
};

// Phase 10.1b test-only oracle for the stable mixed path.  It compares the
// main-controller FP64 and fp32_mixed device contexts at one immutable state
// and lambda; no production caller depends on this capture surface.
struct CudaMixedStableComponentResult {
  bool success = false;
  std::string error;
  CudaPrecisionErrorStatistics residual_jacobian;
  CudaPrecisionErrorStatistics robust_scale;
  CudaPrecisionErrorStatistics pose_hessian;
  CudaPrecisionErrorStatistics pose_gradient;
  CudaPrecisionErrorStatistics point_hessian;
  CudaPrecisionErrorStatistics point_gradient;
  CudaPrecisionErrorStatistics edge_blocks;
  CudaPrecisionErrorStatistics jacobi_damping;
  CudaPrecisionErrorStatistics point_inverse;
  CudaPrecisionErrorStatistics inverse_gradient;
  CudaPrecisionErrorStatistics transformed_edge;
  CudaPrecisionErrorStatistics schur;
  CudaPrecisionErrorStatistics f32_schur_then_cast;
  CudaPrecisionErrorStatistics rhs;
  CudaPrecisionErrorStatistics camera_delta;
  CudaPrecisionErrorStatistics point_delta;
  CudaPrecisionErrorStatistics trial_state;
  double fp64_predicted_reduction = 0.0;
  double mixed_predicted_reduction = 0.0;
  double fp64_backward_error = 0.0;
  double mixed_backward_error = 0.0;
  double fp64_schur_solve_backward_error = 0.0;
  double mixed_schur_solve_backward_error = 0.0;
  double fp64_trial_cost = 0.0;
  double mixed_trial_cost = 0.0;
  int fp64_solver_info = 0;
  int mixed_solver_info = 0;
  uint64_t mixed_dpotrf_calls = 0;
  uint64_t mixed_dpotrs_calls = 0;
  uint64_t mixed_spotrf_calls = 0;
  uint64_t mixed_spotrs_calls = 0;
  uint64_t mixed_serial_full_scan_kernel_count = 0;
  uint64_t mixed_full_array_d2h_bytes = 0;
  uint64_t mixed_post_initialize_allocation_calls = 0;
  uint64_t mixed_post_ready_arena_grow_calls = 0;
  uint64_t mixed_float_buffer_bytes = 0;
  uint64_t mixed_double_buffer_bytes = 0;
  std::string mixed_schur_math_effective;
  double fp64_complete_milliseconds = 0.0;
  double mixed_complete_milliseconds = 0.0;
  double mixed_residual_jacobian_milliseconds = 0.0;
  double mixed_hessian_milliseconds = 0.0;
  double mixed_gradient_milliseconds = 0.0;
  double mixed_point_inverse_milliseconds = 0.0;
  double mixed_schur_milliseconds = 0.0;
  double mixed_rhs_milliseconds = 0.0;
  double mixed_factorization_milliseconds = 0.0;
  double mixed_back_substitution_milliseconds = 0.0;
  double mixed_diagnostics_milliseconds = 0.0;
  std::vector<double> promoted_double_schur_complete_milliseconds;
  std::vector<double> f32_schur_then_cast_complete_milliseconds;
};

struct CudaMixedGradientCancellationTestResult {
  double fp32_record_gradient = 0.0;
  double fp64_record_gradient = 0.0;
  uint64_t fp32_record_path_calls = 0;
  uint64_t fp64_record_path_calls = 0;
};

struct CudaMixedPoseDampingTestResult {
  float undamped_float_diagonal = 0.0f;
  float legacy_float_damped_diagonal = 0.0f;
  double fp64_damped_diagonal = 0.0;
  double expected_fp64_damped_diagonal = 0.0;
  double lambda = 0.0;
  double damping = 0.0;
};

bool RunCudaMixedGradientCancellationForTesting(
    CudaMixedGradientCancellationTestResult* result,
    std::string* error);

bool RunCudaMixedPoseDampingForTesting(
    CudaMixedPoseDampingTestResult* result,
    std::string* error);

bool RunCudaMixedLayerACastCompatibilityForTesting(std::string* error);

// Injects one device-state status value after the production update kernels.
// This exercises the real rollback/controller path without changing any
// production solve that does not call this test-only entry point.
bool RunCustomCudaSolveWithStateUpdateStatusForTesting(
    const Snapshot& snapshot,
    const CudaFullLmOptions& options,
    uint32_t status,
    CudaFullLmResult* result,
    std::string* error);

bool RunCudaSchurSolveBackwardErrorZeroDimensionForTesting(
    double* value,
    std::string* error);

bool RunCudaPrecisionComponentForTesting(
    const Snapshot& snapshot,
    double lambda,
    size_t fp32_repeats,
    CudaPrecisionComponentResult* result,
    std::string* error);

bool RunCudaMixedStableComponentForTesting(
    const Snapshot& snapshot,
    double lambda,
    size_t repeats,
    CudaMixedStableComponentResult* result,
    std::string* error);

bool RunCudaHessianAssemblyComponentForTesting(
    const Snapshot& snapshot,
    uint32_t segment_size,
    size_t paired_samples,
    CudaHessianAssemblyComponentResult* result,
    std::string* error);

// Test-only host probe for the production pair-boundary partitioner. It
// exercises boundary/overflow cases and a call505-scale topology without
// allocating the contribution payload.
bool RunCudaLayerCPairChunkingHostSelfTest(
    CudaLayerCPairChunkingTestSummary* summary,
    std::string* error);

// Test-only state-hash helpers. The linear implementation is the production
// audit path; the legacy implementation exists only as a golden oracle.
bool ComputeCudaStateHashForTesting(const Snapshot& snapshot,
                                    bool legacy,
                                    std::string* digest,
                                    CudaStateHashAudit* audit,
                                    std::string* error);

// Compare all solver decisions and final state/topology using raw FP64 bit
// patterns. Audit-only timing, hashes, and strings are intentionally excluded.
bool CompareCudaFullLmResultsBitwise(const CudaFullLmResult& reference,
                                     const CudaFullLmResult& candidate,
                                     std::string* first_difference);

std::string CudaFullLmDecisionBitwiseSha256(
    const CudaFullLmResult& result);
std::string CudaFullLmSemanticBitwiseSha256V1(
    const CudaFullLmResult& result);
std::string CudaFullLmExecutionStructureSha256V1(
    const CudaFullLmResult& result);
std::string CudaFullLmSemanticCanonicalV2(
    const CudaFullLmResult& result);
std::string CudaFullLmExecutionCanonicalV2(
    const CudaFullLmResult& result);
std::string CudaFullLmDiagnosticCanonicalV2(
    const CudaFullLmResult& result);
std::string CudaFullLmSemanticSha256V2(
    const CudaFullLmResult& result);
std::string CudaFullLmExecutionSha256V2(
    const CudaFullLmResult& result);
std::string CudaFullLmDiagnosticSha256V2(
    const CudaFullLmResult& result);
bool CudaLegacyV1ComparisonAvailable(const CudaFullLmResult& result);
CudaSolveErrorClass CudaLegacyV1ErrorClassification(
    const CudaFullLmResult& result);
std::string CudaLegacyV1TerminationReason(
    const CudaFullLmResult& result);
std::string CudaFinalParametersBitwiseSha256(const Snapshot& snapshot);
std::string CudaFinalTopologyBitwiseSha256(const Snapshot& snapshot);

// Test-only audit for global cache serialization and device/context keying.
// It does not alter residual/Jacobian/LM arithmetic and is not used by the
// mapper production path.
bool RunCudaCacheSafetySelfTest(std::string* error);

// Test-only audit for memory headroom and resident-peak accounting. In
// particular, cudaMemGetInfo free bytes already exclude cached allocations, so
// cached bytes must not be subtracted a second time.
bool RunCudaMemoryAccountingSelfTest(std::string* error);

// Fresh-process helper used by CTest to validate one teardown failure without
// inheriting an initialized CUDA context from the parent process.
bool RunCudaCleanupFailureSelfTest(CudaTeardownType teardown_type,
                                   CudaFullLmResult* result,
                                   std::string* error);

// Test-only fixed-storage diagnostic audit. It verifies primary promotion,
// bounded secondary storage, and the dedicated overflow sentinel without
// allocating from the failure path.
bool RunCudaAuditStorageSelfTest(std::string* error);

// Test-only recovery hook used after deliberately injecting a teardown
// failure. Production code cannot mark tainted resources clean through this
// interface; the function performs a complete clear first.
bool ResetCudaResourceStateForTesting(std::string* error);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_CUSTOM_CUDA_H_
