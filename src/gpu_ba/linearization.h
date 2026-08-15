#ifndef COLMAP_SRC_GPU_BA_LINEARIZATION_H_
#define COLMAP_SRC_GPU_BA_LINEARIZATION_H_

#include <array>
#include <cstdint>
#include <string>

namespace colmap {
namespace gpu_ba {

enum class LidarResidualMode : uint8_t {
  kLegacyExact = 0,
  kLegacyGuarded = 1,
  kSigned = 2,
};

bool ParseLidarResidualMode(const std::string& value,
                            LidarResidualMode* mode);
std::string LidarResidualModeName(LidarResidualMode mode);

struct VisualEvaluation {
  std::array<double, 2> residual{{0.0, 0.0}};
  // Row-major matrices.
  std::array<double, 8> ambient_quaternion_jacobian{{}};  // 2x4
  std::array<double, 12> plus_jacobian{{}};               // 4x3
  std::array<double, 6> local_rotation_jacobian{{}};      // 2x3
  std::array<double, 6> translation_jacobian{{}};         // 2x3
  std::array<double, 6> point_jacobian{{}};               // 2x3
  std::array<double, 16> camera_jacobian{{}};              // 2x8
  std::array<double, 3> camera_point{{}};
  bool finite = false;
};

struct LidarEvaluation {
  double signed_distance = 0.0;
  double residual = 0.0;
  std::array<double, 3> point_jacobian{{0.0, 0.0, 0.0}};
  bool exact_zero = false;
  bool near_zero = false;
  bool guarded = false;
  bool finite = false;
};

bool NormalizeQuaternion(const std::array<double, 4>& input,
                         std::array<double, 4>* output);

void QuaternionPlusJacobian(const std::array<double, 4>& quaternion,
                            std::array<double, 12>* jacobian);

bool QuaternionPlus(const std::array<double, 4>& quaternion,
                    const std::array<double, 3>& delta,
                    std::array<double, 4>* result);

bool EvaluateOpenCVVisual(const std::array<double, 4>& quaternion,
                          const std::array<double, 3>& translation,
                          const std::array<double, 3>& point,
                          const std::array<double, 8>& camera,
                          const std::array<double, 2>& observation,
                          VisualEvaluation* evaluation);

LidarEvaluation EvaluateLidar(const std::array<double, 3>& point,
                              const std::array<double, 4>& plane,
                              double weight,
                              LidarResidualMode mode,
                              double near_zero_threshold = 1e-12);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_LINEARIZATION_H_
