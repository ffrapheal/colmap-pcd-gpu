#ifndef COLMAP_SRC_GPU_BA_ACTIVE_BA_SOLVE_SPEC_H_
#define COLMAP_SRC_GPU_BA_ACTIVE_BA_SOLVE_SPEC_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ceres/ceres.h>

#include "gpu_ba/snapshot.h"
#include "util/types.h"

namespace colmap {

class BundleAdjustmentConfig;
struct BundleAdjustmentOptions;
class Reconstruction;

namespace gpu_ba {

enum class CudaProblemSource : uint8_t {
  kLegacySnapshot = 0,
  kActiveSpec = 1,
  kIndexedCatalog = 2,
  kNativeGraph = 3,
};

bool ParseCudaProblemSource(const std::string& value,
                            CudaProblemSource* source);
const char* CudaProblemSourceName(CudaProblemSource source);

struct ActiveBaParameterBlockSpec {
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;
  uint32_t ambient_size = 0;
  uint32_t tangent_size = 0;
  bool constant = false;
  uint8_t translation_subset_mask = 0;
  std::vector<uint32_t> fixed_camera_parameter_indices;
};

struct ActiveBaSolveSpec {
  uint64_t owner_epoch = 0;
  uint64_t catalog_revision = 0;
  CudaSolveProblem problem;
  std::vector<ActiveBaParameterBlockSpec> parameter_blocks;
  uint64_t residual_block_count = 0;
  uint64_t scalar_residual_count = 0;
  uint64_t ambient_parameter_count = 0;
  uint64_t effective_parameter_count = 0;
  uint64_t residual_enumerator_passes = 0;
  uint64_t residual_enumerator_items = 0;
};

class ActiveBaSolveSpecBuilder {
 public:
  ActiveBaSolveSpecBuilder();
  ~ActiveBaSolveSpecBuilder();
  ActiveBaSolveSpecBuilder(const ActiveBaSolveSpecBuilder&) = delete;
  ActiveBaSolveSpecBuilder& operator=(const ActiveBaSolveSpecBuilder&) = delete;

  void RecordVisualResidual(uint32_t image_id,
                            uint32_t point2D_idx,
                            uint64_t point3D_id,
                            const std::array<double, 2>& xy,
                            bool pose_constant);
  void RecordLidarResidual(uint64_t point3D_id,
                           uint8_t lidar_type,
                           const std::array<double, 3>& lidar_xyz,
                           const std::array<double, 4>& plane,
                           double weight,
                           bool has_search_range,
                           double search_range);
  void RecordParameterBlock(ParameterKind kind, uint64_t entity_id);
  uint64_t ResidualBlockCount() const noexcept;
  uint64_t ScalarResidualCount() const noexcept;

  bool Finalize(
      const BundleAdjustmentOptions& options,
      const ceres::Solver::Options& effective_solver_options,
      const BundleAdjustmentConfig& config,
      const Reconstruction& reconstruction,
      const std::unordered_set<camera_t>& active_camera_ids,
      const std::unordered_map<point3D_t, size_t>& point_observation_counts,
      BaKind ba_kind,
      uint64_t ba_call_index,
      uint64_t owner_epoch,
      uint64_t catalog_revision,
      ActiveBaSolveSpec* output,
      std::string* error);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool CompareActiveBaSolveSpecToSnapshot(const ActiveBaSolveSpec& active,
                                        const Snapshot& legacy,
                                        std::string* error);

bool ValidateAndCommitActiveBaState(const ActiveBaSolveSpec& initial,
                                    const CudaSolveProblem& candidate,
                                    Reconstruction* reconstruction,
                                    std::string* error);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_ACTIVE_BA_SOLVE_SPEC_H_
