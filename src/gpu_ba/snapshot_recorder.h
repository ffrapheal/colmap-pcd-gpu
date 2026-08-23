#ifndef COLMAP_SRC_GPU_BA_SNAPSHOT_RECORDER_H_
#define COLMAP_SRC_GPU_BA_SNAPSHOT_RECORDER_H_

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>

#include "gpu_ba/snapshot.h"

namespace colmap {

class BundleAdjustmentConfig;
struct BundleAdjustmentOptions;
class Reconstruction;

namespace gpu_ba {

bool ShouldCaptureSnapshot(const std::string& capture_mode,
                           const std::string& registered_images,
                           BaKind ba_kind,
                           uint64_t registered_image_count,
                           std::string* error);

class SnapshotRecorder {
 public:
  explicit SnapshotRecorder(bool compute_prepared_host_descriptor = false);

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

  void RecordParameterBlock(ParameterKind kind,
                            uint64_t entity_id,
                            double* values);

  bool Finalize(const BundleAdjustmentOptions& options,
                const ceres::Solver::Options& effective_solver_options,
                const BundleAdjustmentConfig& config,
                const Reconstruction& reconstruction,
                const ceres::Problem& problem,
                BaKind ba_kind,
                uint64_t ba_call_index,
                Snapshot* snapshot,
                std::string* error) const;

  bool FinalizeAndWrite(const BundleAdjustmentOptions& options,
                        const ceres::Solver::Options& effective_solver_options,
                        const BundleAdjustmentConfig& config,
                        const Reconstruction& reconstruction,
                        const ceres::Problem& problem,
                        BaKind ba_kind,
                        uint64_t ba_call_index,
                        Snapshot* snapshot,
                        SnapshotWriteResult* result,
                        std::string* error) const;

 private:
  struct RecordedParameter {
    ParameterKind kind = ParameterKind::kPoint3D;
    uint64_t entity_id = 0;
    double* values = nullptr;
  };

  std::vector<ObservationSnapshot> observations_;
  std::vector<LidarSnapshot> lidar_;
  std::vector<OrderEntrySnapshot> source_order_;
  std::vector<RecordedParameter> parameters_;
  std::unordered_map<double*, size_t> parameter_indices_;
  bool compute_prepared_host_descriptor_ = false;
  uint64_t residual_descriptor_identity_ = 1469598103934665603ull;
  uint64_t residual_descriptor_items_ = 0;
  uint64_t residual_descriptor_hash_updates_ = 0;
};

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_SNAPSHOT_RECORDER_H_
