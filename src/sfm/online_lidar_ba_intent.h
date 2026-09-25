#ifndef COLMAP_SRC_SFM_ONLINE_LIDAR_BA_INTENT_H_
#define COLMAP_SRC_SFM_ONLINE_LIDAR_BA_INTENT_H_

#ifdef GPU_BA_CUDA_ENABLED

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gpu_ba/custom_cuda.h"
#include "gpu_ba/host_ba_graph.h"
#include "sfm/online_lidar_association.h"

namespace colmap {

struct BundleAdjustmentOptions;

class OnlineLidarNativeBaSolveIntentBuildOutput {
 public:
  OnlineLidarNativeBaSolveIntentBuildOutput() noexcept = default;
  OnlineLidarNativeBaSolveIntentBuildOutput(
      const OnlineLidarNativeBaSolveIntentBuildOutput&) = default;
  OnlineLidarNativeBaSolveIntentBuildOutput& operator=(
      const OnlineLidarNativeBaSolveIntentBuildOutput&) = default;
  OnlineLidarNativeBaSolveIntentBuildOutput(
      OnlineLidarNativeBaSolveIntentBuildOutput&& other) noexcept;
  OnlineLidarNativeBaSolveIntentBuildOutput& operator=(
      OnlineLidarNativeBaSolveIntentBuildOutput&& other) noexcept;

  bool built() const noexcept;
  const OnlineLidarAssociationOutput& association_output() const noexcept;
  const OnlineLidarAssociationAudit& association_audit() const noexcept;
  const gpu_ba::NativeBaSolveIntent& intent() const noexcept;
  const gpu_ba::CudaFullLmOptions& resolved_cuda_options() const noexcept;

 private:
  void Reset() noexcept;
  void Seal(OnlineLidarAssociationOutput association_output,
            OnlineLidarAssociationAudit association_audit,
            gpu_ba::NativeBaSolveIntent intent,
            gpu_ba::CudaFullLmOptions resolved_cuda_options);

  bool built_ = false;
  OnlineLidarAssociationOutput association_output_;
  OnlineLidarAssociationAudit association_audit_;
  gpu_ba::NativeBaSolveIntent intent_;
  gpu_ba::CudaFullLmOptions resolved_cuda_options_;

  friend bool BuildOnlineLidarNativeBaSolveIntent(
      const Reconstruction& reconstruction,
      const BundleAdjustmentOptions& options,
      const OnlineLidarAssociationRequest& association_request,
      uint64_t owner_epoch,
      uint64_t topology_revision,
      uint64_t selection_revision,
      gpu_ba::BaKind kind,
      OnlineLidarNativeBaSolveIntentBuildOutput* output,
      std::string* error,
      OnlineLidarAssociationTriggerGate* trigger_gate) noexcept;
};

struct OnlineLidarMaterializationSummary {
  std::map<image_t, uint64_t> per_image_materialized_count;
  uint64_t trigger_materialized_count = 0;
  uint64_t materialized_association_count = 0;
  uint64_t materialized_visual_residual_count = 0;
  uint64_t materialized_lidar_residual_count = 0;
  std::vector<uint64_t> materialized_association_ids;
  gpu_ba::NativeBaOnlineLidarIdentity materialized_online_lidar_identity;
  bool solver_evaluation_pending = false;
};

bool BuildOnlineLidarNativeBaSolveIntent(
    const Reconstruction& reconstruction,
    const BundleAdjustmentOptions& options,
    const OnlineLidarAssociationRequest& association_request,
    uint64_t owner_epoch,
    uint64_t topology_revision,
    uint64_t selection_revision,
    gpu_ba::BaKind kind,
    OnlineLidarNativeBaSolveIntentBuildOutput* output,
    std::string* error,
    OnlineLidarAssociationTriggerGate* trigger_gate = nullptr) noexcept;

bool SummarizeOnlineLidarMaterialization(
    const gpu_ba::NativeHostSolveView& view,
    const OnlineLidarNativeBaSolveIntentBuildOutput& build_output,
    OnlineLidarMaterializationSummary* summary,
    std::string* error) noexcept;

}  // namespace colmap

#endif  // GPU_BA_CUDA_ENABLED

#endif  // COLMAP_SRC_SFM_ONLINE_LIDAR_BA_INTENT_H_
