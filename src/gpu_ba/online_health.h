#ifndef COLMAP_SRC_GPU_BA_ONLINE_HEALTH_H_
#define COLMAP_SRC_GPU_BA_ONLINE_HEALTH_H_

#include <cstddef>
#include <string>
#include <vector>

namespace colmap {
namespace gpu_ba {

struct CudaOnlineHealthOptions {
  bool enabled = false;
  double relative_rank_tolerance = 1e-12;
  double max_condition_number = 1e12;
  size_t max_pose_dimension = 120;
};

struct CudaOnlineHealthCapture {
  // Flat row-major 3x3 undamped Hessians in variable-point order.
  std::vector<double> point_hessians;
  size_t pose_dimension = 0;
  // Flat row-major undamped reduced pose Schur matrix.
  std::vector<double> reduced_pose_schur;
  std::vector<double> pose_jacobi_scaling;
};

struct CudaOnlineHealthMatrixResult {
  bool requested = false;
  bool completed = false;
  bool passed = false;
  // Describes the transformed matrix, spectrum, and rank threshold. A
  // rank-deficient matrix may still be finite while its condition is infinity.
  bool finite = false;
  bool full_rank = false;
  bool condition_within_limit = false;
  bool has_significant_negative_eigenvalue = false;
  size_t dimension = 0;
  size_t rank = 0;
  double spectral_radius = 0.0;
  double rank_threshold = 0.0;
  double condition_number = 0.0;
  double min_eigenvalue = 0.0;
  double max_eigenvalue = 0.0;
  double relative_rank_tolerance = 0.0;
  double max_condition_number = 0.0;
  std::string failure_reason;
};

struct CudaOnlineHealthResult {
  bool requested = false;
  bool completed = false;
  bool passed = false;
  // True when every requested matrix has a finite transformed spectrum. An
  // expected infinity condition sentinel for degeneracy does not clear it.
  bool finite = false;
  bool point_matrices_finite = false;
  bool schur_finite = false;
  bool has_significant_negative_eigenvalue = false;
  bool point_has_significant_negative_eigenvalue = false;
  bool schur_has_significant_negative_eigenvalue = false;

  size_t point_count = 0;
  size_t full_rank_count = 0;
  size_t min_rank = 0;
  double worst_condition = 0.0;

  size_t pose_dimension = 0;
  size_t schur_rank = 0;
  double schur_condition = 0.0;
  double schur_min_eigenvalue = 0.0;
  double schur_max_eigenvalue = 0.0;

  double relative_rank_tolerance = 0.0;
  double max_condition_number = 0.0;
  size_t max_pose_dimension = 0;

  std::vector<CudaOnlineHealthMatrixResult> point_results;
  CudaOnlineHealthMatrixResult schur_result;
  std::string failure_reason;
};

// Fault injection is test-only plumbing. Production defaults to kNone, and
// finite-input eigensolver failures are not synthesized by this hook.
enum class CudaOnlineHealthFailureInjectionForTesting {
  kNone = 0,
  kThrowAfterFirstPoint = 1,
};

void SetCudaOnlineHealthFailureInjectionForTesting(
    CudaOnlineHealthFailureInjectionForTesting failure) noexcept;

// Disabled evaluation returns true with a default, unrequested result. Enabled
// evaluation returns true when diagnostics complete, including finite
// degeneracy, and false for invalid input, numerical failure, or exceptions.
// Non-null outputs are replaced only at return and never expose partial work.
bool EvaluateCudaOnlineHealth(const CudaOnlineHealthOptions& options,
                              const CudaOnlineHealthCapture& capture,
                              CudaOnlineHealthResult* result,
                              std::string* error) noexcept;

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_ONLINE_HEALTH_H_
