#ifndef COLMAP_SRC_GPU_BA_VALIDATION_H_
#define COLMAP_SRC_GPU_BA_VALIDATION_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {

struct ErrorSummary {
  std::string name;
  double atol = 0.0;
  double rtol = 0.0;
  uint64_t count = 0;
  uint64_t failures = 0;
  uint64_t nonfinite = 0;
  double max_absolute = 0.0;
  double mean_absolute = 0.0;
  double rms_absolute = 0.0;
  double median_absolute = 0.0;
  double p95_absolute = 0.0;
  double max_relative = 0.0;
  double mean_relative = 0.0;
  double rms_relative = 0.0;
  double median_relative = 0.0;
  double p95_relative = 0.0;
  double worst_gate_ratio = 0.0;
  std::string worst_element_id;
  double worst_reference = 0.0;
  double worst_candidate = 0.0;
  bool pass = true;
};

struct LinearizationValidationOptions {
  size_t max_finite_difference_observations = 4096;
  double finite_difference_relative_step = 1e-4;
  double lidar_near_zero_threshold = 1e-12;
};

struct LinearizationValidationResult {
  bool pass = false;
  std::string lidar_mode;
  uint64_t visual_observations = 0;
  uint64_t lidar_residuals = 0;
  uint64_t finite_difference_observations = 0;
  uint64_t visual_reference_failures = 0;
  uint64_t visual_candidate_failures = 0;
  uint64_t lidar_reference_failures = 0;
  uint64_t lidar_classification_mismatches = 0;
  uint64_t lidar_guarded_count = 0;
  std::vector<uint64_t> lidar_exact_zero_point_ids;
  std::vector<uint64_t> lidar_near_zero_point_ids;
  std::string finite_difference_source_indices_sha256;
  std::vector<ErrorSummary> metrics;
};

bool ValidateResidualsAndJacobians(
    const Snapshot& snapshot,
    const LinearizationValidationOptions& options,
    LinearizationValidationResult* result,
    std::string* error);

std::string LinearizationValidationJson(
    const LinearizationValidationResult& result,
    const LinearizationValidationOptions& options,
    size_t indent_spaces = 2);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_VALIDATION_H_
