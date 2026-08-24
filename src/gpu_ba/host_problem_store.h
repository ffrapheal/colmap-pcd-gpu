#ifndef COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_H_
#define COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_H_

#include <cstdint>
#include <memory>
#include <string>

namespace colmap {

class Reconstruction;

namespace gpu_ba {

struct Snapshot;
struct CudaSolveProblem;
struct CudaFullLmOptions;
struct IndexedActiveSolveDescriptor;
struct MapperStaticCatalogStableTables;
struct PreparedHostSolveViewData;
struct StaticProblemDataCatalog;
struct GpuBaHostProblemStoreControl;
struct NativeGraphStoreState;
struct PreparedHostStorePublication;
struct PreparedIndexedCatalogPublication;

enum class CudaHostProblemStoreMode : uint8_t {
  kDisabled = 0,
  kHostPreparedStore = 1,
};

class GpuBaHostProblemStore;
class PreparedIndexedActiveSolve;
class PreparedNativeActiveSolve;
struct NativeActiveSolveInputs;

struct CudaHostProblemStoreLeaseTestResult {
  bool shutdown_reported_store_busy = false;
  bool control_survived_owner_destruction = false;
  bool lease_completion_succeeded = false;
  bool deferred_shutdown_completed = false;
  bool mismatch_released_matching_active_lease = false;
};

struct CudaHostStoreBinding {
  GpuBaHostProblemStore* store = nullptr;
  uint64_t owner_epoch = 0;
  CudaHostProblemStoreMode mode = CudaHostProblemStoreMode::kDisabled;
};

// Always-on, non-canonical host-store evidence. Device static data is still
// uploaded by every solve in Phase 11.0a.
struct CudaHostProblemStoreRuntimeInfo {
  std::string mode_requested = "disabled";
  std::string mode_effective = "disabled";
  std::string view_action = "disabled";
  std::string rebuild_reason;
  uint64_t owner_epoch = 0;
  uint64_t catalog_generation = 0;
  uint64_t view_generation = 0;
  uint64_t config_generation = 0;
  uint64_t journal_cursor_before = 0;
  uint64_t journal_cursor_after = 0;
  uint64_t store_lookup_calls = 0;
  uint64_t store_hits = 0;
  uint64_t store_misses = 0;
  uint64_t catalog_cold_builds = 0;
  uint64_t catalog_delta_updates = 0;
  uint64_t catalog_full_rebuilds = 0;
  uint64_t journal_events = 0;
  uint64_t journal_gaps = 0;
  uint64_t journal_overflows = 0;
  uint64_t journal_unknown_events = 0;
  uint64_t exact_view_reuses = 0;
  uint64_t solve_view_patches = 0;
  uint64_t solve_view_rebuilds = 0;
  uint64_t fallback_rebuilds = 0;
  uint64_t host_builder_calls_executed = 0;
  uint64_t host_builder_calls_saved = 0;
  uint64_t host_builder_traversals_executed = 0;
  uint64_t host_builder_traversals_saved = 0;
  uint64_t estimated_host_builder_bytes_executed = 0;
  uint64_t estimated_host_builder_bytes_saved = 0;
  uint64_t static_binding_builder_calls = 0;
  uint64_t cost_layout_builder_calls = 0;
  uint64_t hessian_topology_builder_calls = 0;
  uint64_t hessian_segment_plan_builder_calls = 0;
  uint64_t schur_topology_builder_calls = 0;
  uint64_t schur_segment_plan_builder_calls = 0;
  uint64_t dynamic_state_refresh_calls = 0;
  uint64_t indexed_catalog_full_graph_build_calls = 0;
  uint64_t indexed_catalog_journal_apply_calls = 0;
  uint64_t indexed_catalog_flatten_calls = 0;
  uint64_t indexed_catalog_reconcile_calls = 0;
  uint64_t indexed_catalog_export_calls = 0;
  uint64_t indexed_active_materializer_calls = 0;
  uint64_t indexed_full_graph_records_scanned = 0;
  uint64_t indexed_estimated_impl_copy_bytes = 0;
  uint64_t indexed_estimated_export_bytes = 0;
  uint64_t device_static_h2d_saved_calls = 0;
  uint64_t device_static_h2d_saved_bytes = 0;
  uint64_t dynamic_h2d_calls = 0;
  uint64_t dynamic_h2d_bytes = 0;
  uint64_t host_resident_bytes = 0;
  uint64_t host_peak_bytes = 0;
  uint64_t evictions = 0;
  uint64_t invalidations = 0;
  uint64_t owner_identity_violations = 0;
  uint64_t coverage_violations = 0;
  uint64_t store_busy_failures = 0;
  uint64_t host_shutdown_calls = 0;
  uint64_t device_cleanup_failures_observed = 0;
  bool host_catalog_valid = false;
  bool device_poisoned = false;
  bool device_prepared_context_reuse = false;
  bool full_cross_solve_initialization_removed = false;
  double snapshot_descriptor_wall_milliseconds = 0.0;
  double store_lookup_wall_milliseconds = 0.0;
  double catalog_delta_update_wall_milliseconds = 0.0;
  double solve_view_build_or_patch_wall_milliseconds = 0.0;
  double dynamic_state_refresh_wall_milliseconds = 0.0;
  double host_preparation_total_wall_milliseconds = 0.0;
};

class PreparedHostSolveView {
 public:
  PreparedHostSolveView();
  ~PreparedHostSolveView();
  PreparedHostSolveView(const PreparedHostSolveView&) = delete;
  PreparedHostSolveView& operator=(const PreparedHostSolveView&) = delete;
  PreparedHostSolveView(PreparedHostSolveView&&) noexcept;
  PreparedHostSolveView& operator=(PreparedHostSolveView&&) noexcept;

  bool valid() const noexcept;
  uint64_t owner_epoch() const noexcept;
  uint64_t catalog_generation() const noexcept;
  uint64_t view_generation() const noexcept;
  uint64_t config_generation() const noexcept;
  const PreparedHostSolveViewData* data_for_internal_use() const noexcept;
  const CudaHostProblemStoreRuntimeInfo& runtime_info() const noexcept;

  // publish=true commits the pending catalog/view and journal cursor only
  // after the CUDA solve and Reconstruction transaction both succeed.
  bool Complete(bool publish,
                bool device_cleanup_failed,
                std::string* error) noexcept;

 private:
  friend bool PrepareCudaHostSolveView(
      const CudaSolveProblem&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedHostSolveView*,
      CudaHostProblemStoreRuntimeInfo*,
      std::string*);
  friend class GpuBaHostProblemStore;
  friend bool RunCudaHostProblemStoreLeaseLifetimeForTesting(
      CudaHostProblemStoreLeaseTestResult*, std::string*);

  void CancelIfActive() noexcept;

  std::shared_ptr<const PreparedHostSolveViewData> data_;
  std::shared_ptr<GpuBaHostProblemStoreControl> control_;
  std::shared_ptr<PreparedHostStorePublication> publication_;
  uint64_t lease_generation_ = 0;
  uint64_t owner_epoch_ = 0;
  uint64_t catalog_generation_ = 0;
  uint64_t view_generation_ = 0;
  uint64_t config_generation_ = 0;
  uint64_t source_revision_ = 0;
  bool active_ = false;
  CudaHostProblemStoreRuntimeInfo runtime_;
};

// Solve-local lease over the Mapper-owned indexed catalog. The descriptor and
// exported flat tables are immutable for the lease lifetime and contain no
// device pointers.
class PreparedIndexedActiveSolve {
 public:
  PreparedIndexedActiveSolve();
  ~PreparedIndexedActiveSolve();
  PreparedIndexedActiveSolve(const PreparedIndexedActiveSolve&) = delete;
  PreparedIndexedActiveSolve& operator=(const PreparedIndexedActiveSolve&) =
      delete;
  PreparedIndexedActiveSolve(PreparedIndexedActiveSolve&&) noexcept;
  PreparedIndexedActiveSolve& operator=(
      PreparedIndexedActiveSolve&&) noexcept;

  bool valid() const noexcept;
  const IndexedActiveSolveDescriptor* descriptor() const noexcept;
  const MapperStaticCatalogStableTables* catalog_tables() const noexcept;
  const CudaHostProblemStoreRuntimeInfo& runtime_info() const noexcept;
  // publish is true only after CUDA success and Reconstruction commit. A
  // failed solve releases the lease without advancing the indexed cursor.
  bool Complete(bool publish,
                bool device_cleanup_failed,
                std::string* error) noexcept;

 private:
  friend bool PrepareCudaIndexedActiveSolve(
      const CudaSolveProblem&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedIndexedActiveSolve*,
      std::string*);

  void CancelIfActive() noexcept;

  std::shared_ptr<const IndexedActiveSolveDescriptor> descriptor_;
  std::shared_ptr<const MapperStaticCatalogStableTables> catalog_tables_;
  std::shared_ptr<GpuBaHostProblemStoreControl> control_;
  std::shared_ptr<PreparedIndexedCatalogPublication> publication_;
  uint64_t lease_generation_ = 0;
  uint64_t owner_epoch_ = 0;
  bool active_ = false;
  CudaHostProblemStoreRuntimeInfo runtime_;
};

class GpuBaHostProblemStore {
 public:
  GpuBaHostProblemStore(const Reconstruction* reconstruction,
                        uint64_t owner_epoch);
  ~GpuBaHostProblemStore();
  GpuBaHostProblemStore(const GpuBaHostProblemStore&) = delete;
  GpuBaHostProblemStore& operator=(const GpuBaHostProblemStore&) = delete;

  uint64_t owner_epoch() const noexcept;
  bool Shutdown(std::string* error) noexcept;
  CudaHostProblemStoreRuntimeInfo LifetimeRuntimeInfo() const;

 private:
  friend class PreparedHostSolveView;
  friend bool PrepareCudaHostSolveView(
      const CudaSolveProblem&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedHostSolveView*,
      CudaHostProblemStoreRuntimeInfo*,
      std::string*);
  friend bool RunCudaHostProblemStoreLeaseLifetimeForTesting(
      CudaHostProblemStoreLeaseTestResult*, std::string*);
  friend bool PrepareCudaIndexedActiveSolve(
      const CudaSolveProblem&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedIndexedActiveSolve*,
      std::string*);
  friend bool PrepareCudaNativeActiveSolve(
      const NativeActiveSolveInputs&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedNativeActiveSolve*,
      std::string*);

  std::shared_ptr<GpuBaHostProblemStoreControl> control_;
  std::shared_ptr<NativeGraphStoreState> native_graph_state_;
};

bool ParseCudaHostProblemStoreMode(const std::string& value,
                                   CudaHostProblemStoreMode* mode);
const char* CudaHostProblemStoreModeName(CudaHostProblemStoreMode mode);

bool PrepareCudaHostSolveView(
    const CudaSolveProblem& problem,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedHostSolveView* view,
    CudaHostProblemStoreRuntimeInfo* runtime,
    std::string* error);

bool PrepareCudaIndexedActiveSolve(
    const CudaSolveProblem& problem,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedIndexedActiveSolve* prepared,
    std::string* error);

bool RunCudaHostProblemStoreLeaseLifetimeForTesting(
    CudaHostProblemStoreLeaseTestResult* result,
    std::string* error);

void SetCudaHostProblemStoreLookupHashForTesting(uint64_t value) noexcept;

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_HOST_PROBLEM_STORE_H_
