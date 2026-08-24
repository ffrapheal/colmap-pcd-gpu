#ifndef COLMAP_SRC_GPU_BA_NATIVE_GRAPH_PROBLEM_STORE_H_
#define COLMAP_SRC_GPU_BA_NATIVE_GRAPH_PROBLEM_STORE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "gpu_ba/active_ba_solve_spec.h"
#include "gpu_ba/custom_cuda.h"

namespace colmap {

class Reconstruction;

namespace gpu_ba {

struct NativeGraphStoreState;

struct NativeActiveSolveInputs {
  const CudaSolveProblem* problem = nullptr;
  BaArrayView<ActiveBaParameterBlockSpec> parameter_blocks;
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t ambient_parameter_count = 0;
  uint64_t effective_parameter_count = 0;
};

struct NativeGraphPrepareRuntime {
  uint64_t journal_cursor_before = 0;
  uint64_t journal_cursor_after = 0;
  uint64_t journal_batches = 0;
  uint64_t journal_events = 0;
  uint64_t catalog_cold_builds = 0;
  uint64_t catalog_delta_updates = 0;
  uint64_t catalog_full_rebuilds = 0;
  uint64_t reader_busy = 0;
  uint64_t prepared_views = 0;
  double catalog_sync_wall_milliseconds = 0.0;
  double materialize_wall_milliseconds = 0.0;
};

class PreparedNativeActiveSolve {
 public:
  PreparedNativeActiveSolve();
  ~PreparedNativeActiveSolve();
  PreparedNativeActiveSolve(const PreparedNativeActiveSolve&) = delete;
  PreparedNativeActiveSolve& operator=(const PreparedNativeActiveSolve&) =
      delete;
  PreparedNativeActiveSolve(PreparedNativeActiveSolve&&) noexcept;
  PreparedNativeActiveSolve& operator=(PreparedNativeActiveSolve&&) noexcept;

  bool valid() const noexcept;
  NativeCudaSolveRequest request() const noexcept;
  const NativeHostSolveView* view() const noexcept;
  const ActiveStateBuffer* initial_state() const noexcept;
  const NativeGraphPrepareRuntime& runtime() const noexcept;
  bool Complete(std::string* error) noexcept;

 private:
  friend bool PrepareCudaNativeActiveSolve(
      const NativeActiveSolveInputs&,
      const CudaFullLmOptions&,
      const CudaHostStoreBinding&,
      PreparedNativeActiveSolve*,
      std::string*);
  friend bool ValidateAndCommitNativeBaState(
      const NativeActiveSolveInputs&,
      const PreparedNativeActiveSolve&,
      const DenseActiveState&,
      Reconstruction*,
      std::string*);
  struct Data;
  std::shared_ptr<Data> data_;
};

bool MakeNativeActiveSolveInputs(const ActiveBaSolveSpec& spec,
                                 NativeActiveSolveInputs* inputs,
                                 std::string* error);

std::shared_ptr<NativeGraphStoreState> CreateNativeGraphStoreState(
    const Reconstruction* reconstruction, uint64_t owner_epoch);
bool ShutdownNativeGraphStoreState(
    const std::shared_ptr<NativeGraphStoreState>& state,
    std::string* error) noexcept;

bool PrepareCudaNativeActiveSolve(
    const NativeActiveSolveInputs& inputs,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedNativeActiveSolve* prepared,
    std::string* error);

bool ValidateAndCommitNativeBaState(
    const NativeActiveSolveInputs& inputs,
    const PreparedNativeActiveSolve& prepared,
    const DenseActiveState& candidate,
    Reconstruction* reconstruction,
    std::string* error);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_NATIVE_GRAPH_PROBLEM_STORE_H_
