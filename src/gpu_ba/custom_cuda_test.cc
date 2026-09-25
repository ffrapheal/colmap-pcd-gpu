#define TEST_NAME "gpu_ba/custom_cuda"
#include "util/testing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_ba/custom_cuda.h"
#include "gpu_ba/fixed_linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

static_assert(sizeof(CudaVisualInput) == 184,
              "CudaVisualInput ABI size changed");
static_assert(offsetof(CudaVisualInput, point3D_id) == 16,
              "CudaVisualInput ABI alignment changed");
static_assert(offsetof(CudaVisualInput, observation) == 168,
              "CudaVisualInput ABI field layout changed");
static_assert(sizeof(CudaLidarInput) == 96,
              "CudaLidarInput ABI size changed");
static_assert(offsetof(CudaLidarInput, near_zero_threshold) == 88,
              "CudaLidarInput ABI field layout changed");
static_assert(sizeof(CudaVisualOutput) == 480,
              "CudaVisualOutput ABI size changed");
static_assert(sizeof(CudaLidarOutput) == 48,
              "CudaLidarOutput ABI size changed");

constexpr double kDeviceControlScalarAtol = 1e-10;
constexpr double kDeviceControlScalarRtol = 1e-9;
constexpr double kDeviceControlFinalCostAtol = 1e-8;
constexpr double kDeviceControlFinalCostRtol = 1e-6;
// Phase 7.4 transformed-edge contracts are frozen before running any
// transformed component or end-to-end comparison.
constexpr double kTransformedSchurAtol = 1e-10;
constexpr double kTransformedSchurRtol = 5e-11;
constexpr double kTransformedSchurFrobeniusRtol = 5e-11;
constexpr double kTransformedDeltaAtol = 1e-10;
constexpr double kTransformedDeltaRtol = 1e-8;
constexpr double kTransformedPredictedAtol = 1e-10;
constexpr double kTransformedPredictedRtol = 1e-8;
constexpr double kTransformedFinalCostAtol = 1e-8;
constexpr double kTransformedFinalCostRtol = 1e-10;

bool TransformedAbsRelPass(const double candidate,
                           const double reference,
                           const double atol,
                           const double rtol) {
  return std::isfinite(candidate) && std::isfinite(reference) &&
         std::abs(candidate - reference) <=
             atol + rtol * std::max(1.0, std::abs(reference));
}

void TransformEdgeHost(const double* edge,
                       const double* inverse,
                       const size_t dimension,
                       double* transformed) {
  for (size_t row = 0; row < 6; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      transformed[row * 3 + col] =
          row < dimension
              ? edge[row * 3 + 0] * inverse[0 * 3 + col] +
                    edge[row * 3 + 1] * inverse[1 * 3 + col] +
                    edge[row * 3 + 2] * inverse[2 * 3 + col]
              : 0.0;
    }
  }
}

double OriginalEdgeContribution(const double* lhs,
                                const double* inverse,
                                const double* rhs,
                                const size_t row,
                                const size_t col) {
  double value = 0.0;
  for (size_t a = 0; a < 3; ++a) {
    for (size_t b = 0; b < 3; ++b) {
      value += lhs[row * 3 + a] * inverse[a * 3 + b] *
               rhs[col * 3 + b];
    }
  }
  return value;
}

double TransformedEdgeContribution(const double* transformed,
                                   const double* rhs,
                                   const size_t row,
                                   const size_t col) {
  return transformed[row * 3 + 0] * rhs[col * 3 + 0] +
         transformed[row * 3 + 1] * rhs[col * 3 + 1] +
         transformed[row * 3 + 2] * rhs[col * 3 + 2];
}

void CheckDeviceControlScalar(const double reference,
                              const double candidate) {
  const double bound = kDeviceControlScalarAtol +
      kDeviceControlScalarRtol * std::max(1.0, std::abs(reference));
  BOOST_CHECK_SMALL(reference - candidate, bound);
}

class ScopedEnvironmentValue {
 public:
  ScopedEnvironmentValue(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    if (previous != nullptr) {
      had_previous_ = true;
      previous_ = previous;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnvironmentValue() {
    if (had_previous_) {
      setenv(name_.c_str(), previous_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string previous_;
  bool had_previous_ = false;
};

class ScopedEnvironmentUnset {
 public:
  explicit ScopedEnvironmentUnset(const char* name) : name_(name) {
    const char* previous = std::getenv(name);
    if (previous != nullptr) {
      had_previous_ = true;
      previous_ = previous;
    }
    unsetenv(name);
  }
  ~ScopedEnvironmentUnset() {
    if (had_previous_) setenv(name_.c_str(), previous_.c_str(), 1);
  }

 private:
  std::string name_;
  std::string previous_;
  bool had_previous_ = false;
};

void CheckScalar(const double reference,
                 const double candidate,
                 const double atol = 1e-10,
                 const double rtol = 1e-8) {
  if (!std::isfinite(reference) || !std::isfinite(candidate)) {
    BOOST_CHECK(!std::isfinite(reference) && !std::isfinite(candidate));
    return;
  }
  const double bound = atol + rtol * std::max(std::abs(reference),
                                               std::abs(candidate));
  BOOST_CHECK_SMALL(reference - candidate, bound);
}

template <size_t N>
void CheckArray(const std::array<double, N>& reference,
                const double* candidate,
                const double atol = 1e-10,
                const double rtol = 1e-8) {
  for (size_t i = 0; i < N; ++i) {
    CheckScalar(reference[i], candidate[i], atol, rtol);
  }
}

CudaVisualInput VisualInput() {
  CudaVisualInput input;
  input.source_index = 4;
  input.image_id = 35;
  input.point3D_id = 17;
  input.quaternion[0] = 0.91;
  input.quaternion[1] = 0.13;
  input.quaternion[2] = -0.21;
  input.quaternion[3] = 0.31;
  std::array<double, 4> raw{{input.quaternion[0], input.quaternion[1],
                              input.quaternion[2], input.quaternion[3]}};
  std::array<double, 4> normalized;
  const bool normalized_ok = NormalizeQuaternion(raw, &normalized);
  BOOST_CHECK(normalized_ok);
  if (!normalized_ok) return input;
  for (size_t i = 0; i < 4; ++i) input.quaternion[i] = normalized[i];
  input.translation[0] = 0.2;
  input.translation[1] = -0.1;
  input.translation[2] = 0.4;
  input.point[0] = 1.2;
  input.point[1] = -0.4;
  input.point[2] = 4.5;
  const double camera[8] = {610.0, 605.0, 320.0, 240.0,
                            -0.03, 0.004, 0.001, -0.0007};
  const double observation[2] = {421.3, 187.4};
  for (size_t i = 0; i < 8; ++i) input.camera[i] = camera[i];
  input.observation[0] = observation[0];
  input.observation[1] = observation[1];
  return input;
}

Snapshot LayerBSnapshot() {
  Snapshot snapshot;
  snapshot.metadata.snapshot_id = "cuda-layer-b-synthetic";
  snapshot.metadata.loss_function = "TRIVIAL";
  snapshot.metadata.lidar_residual_mode = "legacy_exact";

  CameraSnapshot camera;
  camera.camera_id = 7;
  camera.model_id = 4;
  camera.constant = true;
  camera.params = {610.0, 605.0, 320.0, 240.0,
                   -0.03, 0.004, 0.001, -0.0007};
  snapshot.cameras.push_back(camera);

  ImageSnapshot variable_image;
  variable_image.image_id = 11;
  variable_image.camera_id = camera.camera_id;
  variable_image.selected = true;
  variable_image.pose_constant = false;
  variable_image.has_pose_parameter_blocks = true;
  variable_image.constant_tvec_mask = 1u << 1;  // y translation fixed.
  variable_image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  variable_image.tvec = {{0.1, -0.2, 0.3}};
  snapshot.images.push_back(variable_image);
  ImageSnapshot constant_image = variable_image;
  constant_image.image_id = 12;
  constant_image.pose_constant = true;
  constant_image.constant_tvec_mask = 0;
  constant_image.tvec = {{-0.2, 0.1, 0.4}};
  snapshot.images.push_back(constant_image);

  PointSnapshot variable_point;
  variable_point.point3D_id = 101;
  variable_point.xyz = {{0.2, -0.1, 3.0}};
  snapshot.points.push_back(variable_point);
  PointSnapshot constant_point;
  constant_point.point3D_id = 102;
  constant_point.constant = true;
  constant_point.xyz = {{-0.4, 0.3, 4.0}};
  snapshot.points.push_back(constant_point);

  auto add_observation = [&](uint64_t source, uint32_t image, uint32_t index,
                             uint64_t point, double x, double y) {
    ObservationSnapshot observation;
    observation.source_index = source;
    observation.image_id = image;
    observation.point2D_idx = index;
    observation.point3D_id = point;
    observation.xy = {{x, y}};
    snapshot.observations.push_back(observation);
    OrderEntrySnapshot entry;
    entry.source_index = source;
    entry.residual_kind = ResidualKind::kVisual;
    entry.image_id = image;
    entry.point2D_idx = index;
    entry.point3D_id = point;
    snapshot.source_insertion_order.push_back(entry);
  };
  add_observation(0, 11, 1, 101, 360.0, 190.0);
  add_observation(1, 12, 1, 101, 350.0, 200.0);
  add_observation(2, 11, 2, 102, 300.0, 260.0);

  LidarSnapshot lidar;
  lidar.source_index = 3;
  lidar.point3D_id = 101;
  lidar.weight = 4.0;
  lidar.plane = {{0.2, -0.3, 0.7, -1.8}};
  snapshot.lidar.push_back(lidar);
  OrderEntrySnapshot lidar_entry;
  lidar_entry.source_index = 3;
  lidar_entry.residual_kind = ResidualKind::kLidar;
  lidar_entry.point3D_id = 101;
  snapshot.source_insertion_order.push_back(lidar_entry);
  snapshot.canonical_order = snapshot.source_insertion_order;
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs, const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id,
                              lhs.point2D_idx, lhs.point3D_id,
                              lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id,
                              rhs.point2D_idx, rhs.point3D_id,
                              rhs.source_index);
            });

  auto add_parameter = [&](ParameterKind kind, uint64_t entity,
                           uint32_t ambient, uint32_t tangent) {
    ParameterBlockSnapshot block;
    block.source_index = snapshot.parameter_blocks_source_order.size();
    block.kind = kind;
    block.entity_id = entity;
    block.ambient_size = ambient;
    block.tangent_size = tangent;
    block.constant = false;
    snapshot.parameter_blocks_source_order.push_back(block);
  };
  add_parameter(ParameterKind::kQuaternion, 11, 4, 3);
  add_parameter(ParameterKind::kTranslation, 11, 3, 2);
  add_parameter(ParameterKind::kPoint3D, 101, 3, 3);
  return snapshot;
}

Snapshot PrecisionExperimentSnapshot() {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.snapshot_id = "cuda-precision-experiment-synthetic";
  for (ObservationSnapshot& observation : snapshot.observations) {
    const auto image = std::find_if(
        snapshot.images.begin(), snapshot.images.end(),
        [&observation](const ImageSnapshot& value) {
          return value.image_id == observation.image_id;
        });
    observation.pose_constant = image->pose_constant;
    snapshot.tracks.push_back(
        {observation.point3D_id, observation.image_id,
         observation.point2D_idx});
  }
  snapshot.parameter_blocks_canonical_order.resize(
      snapshot.parameter_blocks_source_order.size());
  std::iota(snapshot.parameter_blocks_canonical_order.begin(),
            snapshot.parameter_blocks_canonical_order.end(), 0);
  std::sort(snapshot.parameter_blocks_canonical_order.begin(),
            snapshot.parameter_blocks_canonical_order.end(),
            [&snapshot](const uint64_t lhs, const uint64_t rhs) {
              const auto& a = snapshot.parameter_blocks_source_order[lhs];
              const auto& b = snapshot.parameter_blocks_source_order[rhs];
              return std::tie(a.kind, a.entity_id, a.source_index) <
                     std::tie(b.kind, b.entity_id, b.source_index);
            });
  return snapshot;
}

Snapshot PairChunkSnapshot() {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.snapshot_id = "cuda-layer-c-pair-chunk-synthetic";
  snapshot.images[1].pose_constant = false;
  snapshot.images[1].has_pose_parameter_blocks = true;
  ParameterBlockSnapshot quaternion;
  quaternion.source_index = snapshot.parameter_blocks_source_order.size();
  quaternion.kind = ParameterKind::kQuaternion;
  quaternion.entity_id = snapshot.images[1].image_id;
  quaternion.ambient_size = 4;
  quaternion.tangent_size = 3;
  snapshot.parameter_blocks_source_order.push_back(quaternion);
  ParameterBlockSnapshot translation;
  translation.source_index = snapshot.parameter_blocks_source_order.size();
  translation.kind = ParameterKind::kTranslation;
  translation.entity_id = snapshot.images[1].image_id;
  translation.ambient_size = 3;
  translation.tangent_size = 3;
  snapshot.parameter_blocks_source_order.push_back(translation);
  return snapshot;
}

Snapshot SegmentedPairSnapshot() {
  Snapshot snapshot = PairChunkSnapshot();
  snapshot.metadata.snapshot_id = "cuda-schur-segmented-pair-synthetic";
  uint64_t source_index = 4;
  for (size_t i = 0; i < 40; ++i) {
    PointSnapshot point = snapshot.points.front();
    point.point3D_id = 1000 + i;
    point.xyz[0] += 1e-3 * static_cast<double>(i + 1);
    point.xyz[1] -= 5e-4 * static_cast<double>(i + 1);
    snapshot.points.push_back(point);
    for (size_t image = 0; image < 2; ++image) {
      ObservationSnapshot observation;
      observation.source_index = source_index++;
      observation.image_id = snapshot.images[image].image_id;
      observation.point2D_idx = static_cast<uint32_t>(100 + i);
      observation.point3D_id = point.point3D_id;
      observation.xy = {{340.0 + static_cast<double>(i) * 0.1,
                         210.0 - static_cast<double>(image) * 2.0}};
      snapshot.observations.push_back(observation);
      OrderEntrySnapshot entry;
      entry.source_index = observation.source_index;
      entry.residual_kind = ResidualKind::kVisual;
      entry.image_id = observation.image_id;
      entry.point2D_idx = observation.point2D_idx;
      entry.point3D_id = observation.point3D_id;
      snapshot.source_insertion_order.push_back(entry);
    }
    ParameterBlockSnapshot block;
    block.source_index = snapshot.parameter_blocks_source_order.size();
    block.kind = ParameterKind::kPoint3D;
    block.entity_id = point.point3D_id;
    block.ambient_size = 3;
    block.tangent_size = 3;
    snapshot.parameter_blocks_source_order.push_back(block);
  }
  snapshot.canonical_order = snapshot.source_insertion_order;
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id,
                              lhs.point2D_idx, lhs.point3D_id,
                              lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id,
                              rhs.point2D_idx, rhs.point3D_id,
                              rhs.source_index);
            });
  return snapshot;
}

}  // namespace

BOOST_AUTO_TEST_CASE(LayerAVisualAndLidarExplicitCopy) {
  int device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&device_count), cudaSuccess);
  BOOST_REQUIRE_GT(device_count, 0);

  const CudaVisualInput visual = VisualInput();
  std::vector<CudaVisualInput> visual_inputs{visual};
  CudaLidarInput lidar;
  lidar.source_index = 9;
  lidar.point3D_id = visual.point3D_id;
  lidar.point[0] = 1.2;
  lidar.point[1] = -0.4;
  lidar.point[2] = 4.5;
  lidar.plane[0] = 0.2;
  lidar.plane[1] = -0.3;
  lidar.plane[2] = 0.7;
  lidar.plane[3] = -3.0;
  lidar.weight = 2.5;
  lidar.mode = static_cast<uint8_t>(LidarResidualMode::kLegacyExact);
  std::vector<CudaLidarInput> lidar_inputs{lidar};

  CudaLayerAResult result;
  CudaLayerAOptions options;
  options.device = 0;
  BOOST_REQUIRE(RunCudaLayerA(visual_inputs, lidar_inputs, options, &result));
  BOOST_REQUIRE(result.success);
  BOOST_REQUIRE_EQUAL(result.visual.size(), 1);
  BOOST_REQUIRE_EQUAL(result.lidar.size(), 1);
  BOOST_CHECK_EQUAL(result.runtime.launch_status, 0);
  BOOST_CHECK_EQUAL(result.runtime.synchronize_status, 0);
  BOOST_CHECK_GT(result.runtime.kernel_milliseconds, 0.0);

  const std::array<double, 4> q{{visual.quaternion[0], visual.quaternion[1],
                                  visual.quaternion[2], visual.quaternion[3]}};
  const std::array<double, 3> t{{visual.translation[0], visual.translation[1],
                                  visual.translation[2]}};
  const std::array<double, 3> p{{visual.point[0], visual.point[1],
                                  visual.point[2]}};
  const std::array<double, 8> c{{visual.camera[0], visual.camera[1],
                                  visual.camera[2], visual.camera[3],
                                  visual.camera[4], visual.camera[5],
                                  visual.camera[6], visual.camera[7]}};
  const std::array<double, 2> observation{{visual.observation[0],
                                            visual.observation[1]}};
  VisualEvaluation reference;
  BOOST_REQUIRE(EvaluateOpenCVVisual(q, t, p, c, observation, &reference));
  const CudaVisualOutput& candidate = result.visual.front();
  BOOST_CHECK_EQUAL(candidate.finite, 1);
  CheckArray(reference.residual, candidate.residual);
  CheckArray(reference.ambient_quaternion_jacobian,
             candidate.ambient_quaternion_jacobian);
  CheckArray(reference.plus_jacobian, candidate.plus_jacobian);
  CheckArray(reference.local_rotation_jacobian,
             candidate.local_rotation_jacobian);
  CheckArray(reference.translation_jacobian, candidate.translation_jacobian);
  CheckArray(reference.point_jacobian, candidate.point_jacobian);
  CheckArray(reference.camera_jacobian, candidate.camera_jacobian);
  CheckArray(reference.camera_point, candidate.camera_point);

  const std::array<double, 3> lidar_point{{lidar.point[0], lidar.point[1],
                                            lidar.point[2]}};
  const std::array<double, 4> plane{{lidar.plane[0], lidar.plane[1],
                                     lidar.plane[2], lidar.plane[3]}};
  const LidarEvaluation lidar_reference = EvaluateLidar(
      lidar_point, plane, lidar.weight, LidarResidualMode::kLegacyExact);
  const CudaLidarOutput& lidar_candidate = result.lidar.front();
  BOOST_CHECK_EQUAL(lidar_candidate.finite, lidar_reference.finite ? 1 : 0);
  CheckScalar(lidar_reference.signed_distance,
              lidar_candidate.signed_distance);
  CheckScalar(lidar_reference.residual, lidar_candidate.residual);
  CheckArray(lidar_reference.point_jacobian, lidar_candidate.point_jacobian);
}

BOOST_AUTO_TEST_CASE(LayerAUnifiedMemoryAndGuardedNearZero) {
  int device_count = 0;
  BOOST_REQUIRE_EQUAL(cudaGetDeviceCount(&device_count), cudaSuccess);
  BOOST_REQUIRE_GT(device_count, 0);

  CudaLidarInput lidar;
  lidar.point[0] = 1.0;
  lidar.point[1] = 2.0;
  lidar.point[2] = 3.0;
  lidar.plane[0] = 1.0;
  lidar.plane[1] = 0.0;
  lidar.plane[2] = 0.0;
  lidar.plane[3] = -1.0;
  lidar.weight = 10.0;
  lidar.mode = static_cast<uint8_t>(LidarResidualMode::kLegacyGuarded);
  lidar.near_zero_threshold = 1e-12;
  std::vector<CudaLidarInput> inputs{lidar};
  CudaLayerAResult result;
  CudaLayerAOptions options;
  options.memory_mode = CudaMemoryMode::kUnifiedManaged;
  const bool ran = RunCudaLayerA({}, inputs, options, &result);
  BOOST_REQUIRE_MESSAGE(ran, result.error);
  BOOST_REQUIRE_EQUAL(result.lidar.size(), 1);
  const LidarEvaluation reference = EvaluateLidar(
      {{1.0, 2.0, 3.0}}, {{1.0, 0.0, 0.0, -1.0}}, 10.0,
      LidarResidualMode::kLegacyGuarded);
  BOOST_CHECK_EQUAL(result.lidar[0].near_zero, 1);
  BOOST_CHECK_EQUAL(result.lidar[0].guarded, 1);
  BOOST_CHECK_EQUAL(result.lidar[0].finite, reference.finite ? 1 : 0);
  CheckArray(reference.point_jacobian, result.lidar[0].point_jacobian);
  BOOST_CHECK(result.runtime.used_unified_memory);
}

BOOST_AUTO_TEST_CASE(LayerBMatchesCanonicalCpuWithConstantAndSubsetBlocks) {
  const Snapshot snapshot = LayerBSnapshot();
  CustomCpuLinearizationExport reference;
  std::string error;
  BOOST_REQUIRE_MESSAGE(ExportCustomCpuCanonicalLinearization(
                            snapshot, 1e-6, 1e32, &reference, &error),
                        error);
  CudaLayerBResult candidate;
  CudaLayerBOptions options;
  BOOST_REQUIRE_MESSAGE(
      RunCudaSnapshotLayerB(snapshot, options, &candidate, &error), error);
  BOOST_REQUIRE_EQUAL(reference.poses.size(), candidate.poses.size());
  BOOST_REQUIRE_EQUAL(reference.points.size(), candidate.points.size());
  BOOST_REQUIRE_EQUAL(reference.edges.size(), candidate.edges.size());
  for (size_t block = 0; block < reference.poses.size(); ++block) {
    BOOST_CHECK_EQUAL(reference.poses[block].image_id,
                      candidate.poses[block].image_id);
    BOOST_CHECK_EQUAL(reference.poses[block].dimension,
                      candidate.poses[block].dimension);
    const size_t dimension = reference.poses[block].dimension;
    for (size_t i = 0; i < dimension * dimension; ++i)
      CheckScalar(reference.poses[block].hessian[i],
                  candidate.poses[block].hessian[i]);
    for (size_t i = 0; i < dimension; ++i) {
      CheckScalar(reference.poses[block].gradient[i],
                  candidate.poses[block].gradient[i]);
      CheckScalar(reference.poses[block].jacobi_scaling[i],
                  candidate.poses[block].jacobi_scaling[i]);
      CheckScalar(reference.poses[block].damping[i],
                  candidate.poses[block].damping[i]);
    }
  }
  for (size_t block = 0; block < reference.points.size(); ++block) {
    BOOST_CHECK_EQUAL(reference.points[block].point3D_id,
                      candidate.points[block].point3D_id);
    for (size_t i = 0; i < 9; ++i)
      CheckScalar(reference.points[block].hessian[i],
                  candidate.points[block].hessian[i]);
    for (size_t i = 0; i < 3; ++i) {
      CheckScalar(reference.points[block].gradient[i],
                  candidate.points[block].gradient[i]);
      CheckScalar(reference.points[block].jacobi_scaling[i],
                  candidate.points[block].jacobi_scaling[i]);
      CheckScalar(reference.points[block].damping[i],
                  candidate.points[block].damping[i]);
    }
  }
  for (size_t block = 0; block < reference.edges.size(); ++block) {
    BOOST_CHECK_EQUAL(reference.edges[block].pose_index,
                      candidate.edges[block].pose_index);
    BOOST_CHECK_EQUAL(reference.edges[block].point_index,
                      candidate.edges[block].point_index);
    BOOST_CHECK_EQUAL(reference.edges[block].pose_dimension,
                      candidate.edges[block].pose_dimension);
    for (size_t i = 0; i < reference.edges[block].pose_dimension * 3; ++i)
      CheckScalar(reference.edges[block].value[i],
                  candidate.edges[block].value[i]);
  }
  CheckScalar(reference.gradient_norms.projected_max_norm,
              candidate.projected_gradient_max_norm);
  CheckScalar(reference.gradient_norms.raw_tangent_max_norm,
              candidate.raw_tangent_gradient_max_norm);
  CheckScalar(reference.gradient_norms.scaled_max_norm,
              candidate.scaled_gradient_max_norm);
}

BOOST_AUTO_TEST_CASE(LayerCSingleStepCoreMatchesCanonicalCpu) {
  const Snapshot snapshot = LayerBSnapshot();
  CustomCpuCanonicalStepExport reference;
  std::string error;
  BOOST_REQUIRE_MESSAGE(ExportCustomCpuCanonicalStep(
                            snapshot, 1e-4, 1e-6, 1e32,
                            &reference, &error),
                        error);
  CudaLayerCResult candidate;
  CudaLayerCOptions options;
  options.lambda = 1e-4;
  BOOST_REQUIRE_MESSAGE(
      RunCudaSnapshotLayerC(snapshot, options, &candidate, &error), error);
  BOOST_REQUIRE_EQUAL(reference.pose_dimension, candidate.pose_dimension);
  BOOST_REQUIRE_EQUAL(reference.point_dimension, candidate.point_dimension);
  BOOST_REQUIRE_EQUAL(reference.schur.size(), candidate.schur.size());
  BOOST_REQUIRE_EQUAL(reference.rhs.size(), candidate.rhs.size());
  BOOST_REQUIRE_EQUAL(reference.camera_delta.size(),
                      candidate.camera_delta.size());
  BOOST_REQUIRE_EQUAL(reference.point_delta.size(), candidate.point_delta.size());
  for (size_t i = 0; i < reference.schur.size(); ++i)
    CheckScalar(reference.schur[i], candidate.schur[i], 1e-10, 1e-8);
  for (size_t i = 0; i < reference.rhs.size(); ++i)
    CheckScalar(reference.rhs[i], candidate.rhs[i], 1e-10, 1e-8);
  for (size_t i = 0; i < reference.camera_delta.size(); ++i)
    CheckScalar(reference.camera_delta[i], candidate.camera_delta[i],
                1e-10, 1e-8);
  for (size_t i = 0; i < reference.point_delta.size(); ++i)
    CheckScalar(reference.point_delta[i], candidate.point_delta[i],
                1e-10, 1e-8);
  CheckScalar(reference.predicted_reduction,
              candidate.predicted_reduction, 1e-9, 1e-8);
  CheckScalar(reference.trial_cost, candidate.trial_cost, 1e-9, 1e-8);
  BOOST_CHECK_LE(candidate.backward_error, 1e-8);
  BOOST_CHECK_EQUAL(candidate.runtime.point_factorization_failures, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cusolver_create_status, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cusolver_potrf_status, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cusolver_potrs_status, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cusolver_dev_info, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cublas_create_status, 0);
  BOOST_CHECK_EQUAL(candidate.runtime.cublas_copy_status, 0);
}

BOOST_AUTO_TEST_CASE(LayerBSoftL1MatchesCanonicalCpu) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.loss_function = "SOFT_L1";
  CustomCpuLinearizationExport reference;
  std::string error;
  BOOST_REQUIRE_MESSAGE(ExportCustomCpuCanonicalLinearization(
                            snapshot, 1e-6, 1e32, &reference, &error),
                        error);
  CudaLayerBOptions options;
  options.loss_scale = 1.0;
  CudaLayerBResult candidate;
  BOOST_REQUIRE_MESSAGE(
      RunCudaSnapshotLayerB(snapshot, options, &candidate, &error), error);
  CheckScalar(reference.cost, candidate.cost, 1e-10, 1e-8);
  BOOST_REQUIRE_EQUAL(reference.poses.size(), candidate.poses.size());
  BOOST_REQUIRE_EQUAL(reference.points.size(), candidate.points.size());
  BOOST_REQUIRE_EQUAL(reference.edges.size(), candidate.edges.size());
  for (size_t block = 0; block < reference.poses.size(); ++block) {
    BOOST_REQUIRE_EQUAL(reference.poses[block].image_id,
                        candidate.poses[block].image_id);
    const size_t dimension = reference.poses[block].dimension;
    for (size_t i = 0; i < dimension * dimension; ++i) {
      CheckScalar(reference.poses[block].hessian[i],
                  candidate.poses[block].hessian[i]);
    }
    for (size_t i = 0; i < dimension; ++i) {
      CheckScalar(reference.poses[block].gradient[i],
                  candidate.poses[block].gradient[i]);
      CheckScalar(reference.poses[block].damping[i],
                  candidate.poses[block].damping[i]);
    }
  }
  for (size_t block = 0; block < reference.points.size(); ++block) {
    BOOST_REQUIRE_EQUAL(reference.points[block].point3D_id,
                        candidate.points[block].point3D_id);
    for (size_t i = 0; i < 9; ++i) {
      CheckScalar(reference.points[block].hessian[i],
                  candidate.points[block].hessian[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
      CheckScalar(reference.points[block].gradient[i],
                  candidate.points[block].gradient[i]);
      CheckScalar(reference.points[block].damping[i],
                  candidate.points[block].damping[i]);
    }
  }
  for (size_t block = 0; block < reference.edges.size(); ++block) {
    BOOST_REQUIRE_EQUAL(reference.edges[block].pose_index,
                        candidate.edges[block].pose_index);
    BOOST_REQUIRE_EQUAL(reference.edges[block].point_index,
                        candidate.edges[block].point_index);
    for (size_t i = 0; i < reference.edges[block].pose_dimension * 3; ++i) {
      CheckScalar(reference.edges[block].value[i],
                  candidate.edges[block].value[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(FullLmMeetsFrozenCustomCpuQualityEnvelope) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.max_linear_solver_iterations = 100;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;

  CustomCpuSolverOptions cpu_options;
  cpu_options.loss_type = "TRIVIAL";
  cpu_options.max_num_iterations = 4;
  cpu_options.max_linear_solver_iterations = 100;
  cpu_options.function_tolerance = 0.0;
  cpu_options.gradient_tolerance = 0.0;
  cpu_options.parameter_tolerance = 0.0;
  CustomCpuSolveResult cpu;
  Snapshot cpu_state;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCpuSolve(
                            snapshot, cpu_options, CustomCpuSolveMode::kFull,
                            &cpu, &cpu_state, &error),
                        error);

  CudaFullLmOptions cuda_options;
  cuda_options.max_num_iterations = 4;
  cuda_options.function_tolerance = 0.0;
  cuda_options.gradient_tolerance = 0.0;
  cuda_options.parameter_tolerance = 0.0;
  cuda_options.layer_c.layer_b.cost_reduction_threads = 1;
  CudaFullLmResult cuda;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, cuda_options, &cuda, &error), error);
  BOOST_REQUIRE(cuda.success);
  BOOST_CHECK(std::isfinite(cpu.final_cost));
  BOOST_CHECK(std::isfinite(cuda.final_cost));
  const double final_cost_bound =
      1e-8 + 1e-6 * std::max(std::abs(cpu.final_cost),
                              std::abs(cuda.final_cost));
  BOOST_CHECK_SMALL(cpu.final_cost - cuda.final_cost, final_cost_bound);

  BOOST_REQUIRE_EQUAL(cpu_state.images.size(), cuda.final_state.images.size());
  BOOST_REQUIRE_EQUAL(cpu_state.points.size(), cuda.final_state.points.size());
  for (size_t i = 0; i < cpu_state.images.size(); ++i) {
    BOOST_CHECK_EQUAL(cpu_state.images[i].image_id,
                      cuda.final_state.images[i].image_id);
    double dot_value = 0.0;
    double cpu_norm_squared = 0.0;
    double cuda_norm_squared = 0.0;
    for (size_t k = 0; k < 4; ++k) {
      dot_value += cpu_state.images[i].qvec[k] *
                   cuda.final_state.images[i].qvec[k];
      cpu_norm_squared += cpu_state.images[i].qvec[k] *
                          cpu_state.images[i].qvec[k];
      cuda_norm_squared += cuda.final_state.images[i].qvec[k] *
                           cuda.final_state.images[i].qvec[k];
    }
    const double dot = std::min(
        1.0, std::abs(dot_value) /
                 std::sqrt(cpu_norm_squared * cuda_norm_squared));
    const double rotation_degrees = 2.0 * std::acos(dot) * 180.0 / M_PI;
    BOOST_CHECK_LT(rotation_degrees, 1e-4);
    for (size_t axis = 0; axis < 3; ++axis) {
      BOOST_CHECK_SMALL(cpu_state.images[i].tvec[axis] -
                            cuda.final_state.images[i].tvec[axis],
                        1e-6);
    }
  }
  for (size_t i = 0; i < cpu_state.points.size(); ++i) {
    BOOST_CHECK_EQUAL(cpu_state.points[i].point3D_id,
                      cuda.final_state.points[i].point3D_id);
    for (size_t axis = 0; axis < 3; ++axis) {
      BOOST_CHECK_SMALL(cpu_state.points[i].xyz[axis] -
                            cuda.final_state.points[i].xyz[axis],
                        1e-5);
    }
  }

  CudaFullLmOptions performance_options = cuda_options;
  performance_options.performance_mode = true;
  CudaFullLmResult performance;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            snapshot, performance_options, &performance,
                            &error),
                        error);
  BOOST_CHECK(performance.runtime.buffers_reused_across_iterations);
  BOOST_CHECK(performance.runtime.topology_reused_across_iterations);
  BOOST_CHECK_GT(performance.runtime.topology_reuse_hits, 0);
  CheckScalar(cuda.final_cost, performance.final_cost, 1e-8, 1e-7);
}

BOOST_AUTO_TEST_CASE(ParallelDeterministicReductionMatchesSerial) {
  const Snapshot snapshot = LayerBSnapshot();
  std::string error;
  CudaLayerBOptions serial_options;
  serial_options.cost_reduction_threads = 8;
  CudaLayerBResult serial;
  BOOST_REQUIRE_MESSAGE(
      RunCudaSnapshotLayerB(snapshot, serial_options, &serial, &error), error);
  CudaLayerBOptions parallel_options = serial_options;
  parallel_options.reduction_mode = CudaReductionMode::kParallelDeterministic;
  CudaLayerBResult parallel;
  BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerB(
                            snapshot, parallel_options, &parallel, &error),
                        error);
  BOOST_CHECK(parallel.runtime.cost_reduction_parallel);
  BOOST_CHECK(parallel.runtime.gradient_reduction_parallel);
  BOOST_CHECK(!serial.runtime.cost_reduction_parallel);
  BOOST_CHECK(!serial.runtime.gradient_reduction_parallel);
  BOOST_CHECK_EQUAL(static_cast<int>(serial.runtime.reduction_mode),
                    static_cast<int>(CudaReductionMode::kSerialDeterministic));
  BOOST_CHECK_EQUAL(static_cast<int>(parallel.runtime.reduction_mode),
                    static_cast<int>(CudaReductionMode::kParallelDeterministic));
  CheckScalar(serial.cost, parallel.cost, 1e-12, 1e-12);
  CheckScalar(serial.projected_gradient_max_norm,
              parallel.projected_gradient_max_norm, 1e-12, 1e-12);
  CheckScalar(serial.scaled_gradient_max_norm,
              parallel.scaled_gradient_max_norm, 1e-12, 1e-12);
}

BOOST_AUTO_TEST_CASE(FaultInjectionAndRollbackGuards) {
  const Snapshot snapshot = LayerBSnapshot();
  std::string error;

  CudaLayerBOptions oom_options;
  oom_options.fault_injection = CudaFaultInjection::kForceInsufficientMemory;
  CudaLayerBResult oom;
  BOOST_CHECK(!RunCudaSnapshotLayerB(snapshot, oom_options, &oom, &error));
  BOOST_CHECK(error.find("INSUFFICIENT_GPU_MEMORY") != std::string::npos);

  CudaLayerBOptions near_budget_options;
  near_budget_options.memory_budget_override_bytes = 1;
  CudaLayerBResult near_budget;
  error.clear();
  BOOST_CHECK(!RunCudaSnapshotLayerB(snapshot, near_budget_options,
                                     &near_budget, &error));
  BOOST_CHECK(error.find("INSUFFICIENT_GPU_MEMORY") != std::string::npos);

  CudaLayerCOptions factor_options;
  factor_options.fault_injection =
      CudaFaultInjection::kForcePointFactorizationFailure;
  CudaLayerCResult factor;
  error.clear();
  BOOST_CHECK(!RunCudaSnapshotLayerC(snapshot, factor_options, &factor,
                                     &error));
  BOOST_CHECK(error.find("point block Cholesky failed") != std::string::npos);

  CudaLayerCOptions workspace_options;
  workspace_options.fault_injection =
      CudaFaultInjection::kForceWorkspaceAllocationFailure;
  CudaLayerCResult workspace;
  error.clear();
  BOOST_CHECK(!RunCudaSnapshotLayerC(snapshot, workspace_options, &workspace,
                                     &error));
  BOOST_CHECK(error.find("cuSOLVER workspace") != std::string::npos);

  Snapshot lm_snapshot = snapshot;
  lm_snapshot.metadata.max_num_iterations = 1;
  lm_snapshot.metadata.max_consecutive_invalid_steps = 10;
  lm_snapshot.metadata.function_tolerance = 0.0;
  lm_snapshot.metadata.gradient_tolerance = 0.0;
  lm_snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions reject_options;
  reject_options.max_num_iterations = 1;
  reject_options.function_tolerance = 0.0;
  reject_options.gradient_tolerance = 0.0;
  reject_options.parameter_tolerance = 0.0;
  reject_options.min_relative_decrease = 2.0;
  CudaFullLmResult rejected;
  error.clear();
  BOOST_CHECK(RunCustomCudaSolve(lm_snapshot, reject_options, &rejected,
                                 &error));
  BOOST_CHECK_GE(rejected.rejected_steps, 1);
  BOOST_REQUIRE_EQUAL(rejected.final_state.points.size(),
                      lm_snapshot.points.size());
  for (size_t i = 0; i < lm_snapshot.points.size(); ++i)
    for (size_t j = 0; j < 3; ++j)
      BOOST_CHECK_EQUAL(rejected.final_state.points[i].xyz[j],
                        lm_snapshot.points[i].xyz[j]);

  CudaFullLmOptions nonfinite_options = reject_options;
  nonfinite_options.min_relative_decrease = 1e-3;
  nonfinite_options.fault_injection = CudaFaultInjection::kForceNonfiniteTrial;
  CudaFullLmResult nonfinite;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(lm_snapshot, nonfinite_options, &nonfinite,
                                  &error));
  BOOST_CHECK(!error.empty());
  BOOST_REQUIRE_EQUAL(nonfinite.final_state.points.size(),
                      lm_snapshot.points.size());
  BOOST_CHECK_EQUAL(nonfinite.final_state.points[0].xyz[0],
                    lm_snapshot.points[0].xyz[0]);

  CudaFullLmOptions consecutive = reject_options;
  consecutive.max_num_iterations = 4;
  consecutive.max_num_consecutive_invalid_steps = 2;
  consecutive.layer_c.fault_injection =
      CudaFaultInjection::kForcePointFactorizationFailure;
  CudaFullLmResult invalid;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(lm_snapshot, consecutive, &invalid, &error));
  BOOST_CHECK(error.find("maximum_consecutive_invalid_steps") !=
              std::string::npos);
  BOOST_CHECK_EQUAL(invalid.factorization_failures, 2);
  BOOST_CHECK_EQUAL(invalid.invalid_steps, 2);
  BOOST_CHECK_EQUAL(invalid.rejected_steps, 0);
}

BOOST_AUTO_TEST_CASE(CacheSafetySelfTest) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaCacheSafetySelfTest(&error), error);
}

BOOST_AUTO_TEST_CASE(MemoryAccountingSelfTest) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaMemoryAccountingSelfTest(&error), error);
}

BOOST_AUTO_TEST_CASE(AuditStorageIsBoundedAndPromotesCleanup) {
  std::string error;
  BOOST_CHECK_MESSAGE(RunCudaAuditStorageSelfTest(&error), error);
}

BOOST_AUTO_TEST_CASE(StateHashLayoutMatchesLegacyAndIsLinear) {
  Snapshot snapshot = LayerBSnapshot();
  CameraSnapshot second_camera = snapshot.cameras.front();
  second_camera.camera_id = 8;
  second_camera.params[0] += 1.0;
  snapshot.cameras.push_back(second_camera);
  auto add_camera_parameter = [&](const uint32_t camera_id) {
    ParameterBlockSnapshot block;
    block.source_index = snapshot.parameter_blocks_source_order.size();
    block.kind = ParameterKind::kCamera;
    block.entity_id = camera_id;
    block.ambient_size = 8;
    block.tangent_size = 0;
    block.constant = true;
    snapshot.parameter_blocks_source_order.push_back(block);
  };
  add_camera_parameter(7);
  add_camera_parameter(8);
  std::string legacy_digest;
  std::string linear_digest;
  std::string error;
  CudaStateHashAudit legacy_audit;
  CudaStateHashAudit linear_audit;
  BOOST_REQUIRE(ComputeCudaStateHashForTesting(
      snapshot, true, &legacy_digest, &legacy_audit, &error));
  BOOST_REQUIRE_MESSAGE(ComputeCudaStateHashForTesting(
                            snapshot, false, &linear_digest, &linear_audit,
                            &error),
                        error);
  BOOST_CHECK_EQUAL(legacy_digest, linear_digest);
  BOOST_CHECK_EQUAL(linear_audit.layout_builds, 1);
  BOOST_CHECK_EQUAL(linear_audit.layout_blocks,
                    snapshot.parameter_blocks_source_order.size());
  BOOST_CHECK_EQUAL(linear_audit.current.computations, 1);
  BOOST_CHECK_EQUAL(linear_audit.current.blocks_visited,
                    snapshot.parameter_blocks_source_order.size());
  BOOST_CHECK_EQUAL(linear_audit.current.scalars_hashed, 26);

  Snapshot larger = snapshot;
  constexpr size_t kAdditionalPoints = 63;
  for (size_t i = 0; i < kAdditionalPoints; ++i) {
    PointSnapshot point;
    point.point3D_id = 1000 + i;
    point.xyz = {{static_cast<double>(i), 2.0, 3.0}};
    larger.points.push_back(point);
    ParameterBlockSnapshot block;
    block.source_index = larger.parameter_blocks_source_order.size();
    block.kind = ParameterKind::kPoint3D;
    block.entity_id = point.point3D_id;
    block.ambient_size = 3;
    block.tangent_size = 3;
    larger.parameter_blocks_source_order.push_back(block);
  }
  CudaStateHashAudit larger_linear_audit;
  CudaStateHashAudit larger_legacy_audit;
  std::string larger_linear_digest;
  std::string larger_legacy_digest;
  BOOST_REQUIRE_MESSAGE(ComputeCudaStateHashForTesting(
                            larger, false, &larger_linear_digest,
                            &larger_linear_audit, &error),
                        error);
  BOOST_REQUIRE_MESSAGE(ComputeCudaStateHashForTesting(
                            larger, true, &larger_legacy_digest,
                            &larger_legacy_audit, &error),
                        error);
  BOOST_CHECK_EQUAL(larger_linear_digest, larger_legacy_digest);
  BOOST_CHECK_EQUAL(larger_linear_audit.current.blocks_visited,
                    linear_audit.current.blocks_visited + kAdditionalPoints);
  BOOST_CHECK_EQUAL(larger_linear_audit.current.scalars_hashed,
                    linear_audit.current.scalars_hashed +
                        3 * kAdditionalPoints);

  Snapshot reordered = snapshot;
  std::reverse(reordered.images.begin(), reordered.images.end());
  std::reverse(reordered.points.begin(), reordered.points.end());
  std::reverse(reordered.cameras.begin(), reordered.cameras.end());
  std::string reordered_digest;
  CudaStateHashAudit reordered_audit;
  BOOST_REQUIRE_MESSAGE(ComputeCudaStateHashForTesting(
                            reordered, false, &reordered_digest,
                            &reordered_audit, &error),
                        error);
  BOOST_CHECK_EQUAL(linear_digest, reordered_digest);

  Snapshot duplicate = snapshot;
  duplicate.images.push_back(snapshot.images.front());
  CudaStateHashAudit malformed_audit;
  std::string malformed_digest;
  error.clear();
  BOOST_CHECK(!ComputeCudaStateHashForTesting(
      duplicate, false, &malformed_digest, &malformed_audit, &error));
  BOOST_CHECK(error.find("duplicate image") != std::string::npos);

  Snapshot missing = snapshot;
  missing.points.erase(missing.points.begin());
  error.clear();
  BOOST_CHECK(!ComputeCudaStateHashForTesting(
      missing, false, &malformed_digest, &malformed_audit, &error));
  BOOST_CHECK(error.find("missing point") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(InstrumentationOnOffIsBitwiseTransparent) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 3;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions disabled_options;
  disabled_options.max_num_iterations = 3;
  disabled_options.function_tolerance = 0.0;
  disabled_options.gradient_tolerance = 0.0;
  disabled_options.parameter_tolerance = 0.0;
  disabled_options.capture_state_trace = false;
  disabled_options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
  CudaFullLmOptions enabled_options = disabled_options;
  enabled_options.instrumentation_mode = CudaInstrumentationMode::kEnabled;

  std::string error;
  CudaFullLmResult disabled;
  CudaFullLmResult enabled;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, disabled_options, &disabled, &error), error);
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, enabled_options, &enabled, &error), error);
  std::string first_difference;
  BOOST_CHECK_MESSAGE(
      CompareCudaFullLmResultsBitwise(disabled, enabled, &first_difference),
      first_difference);
  BOOST_CHECK_EQUAL(CudaFullLmDecisionBitwiseSha256(disabled),
                    CudaFullLmDecisionBitwiseSha256(enabled));
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(disabled.final_state),
                    CudaFinalParametersBitwiseSha256(enabled.final_state));
  BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(disabled.final_state),
                    CudaFinalTopologyBitwiseSha256(enabled.final_state));

  BOOST_CHECK(!disabled.runtime.instrumentation_effective);
  BOOST_CHECK(disabled.runtime.initial_state_hash.empty());
  BOOST_CHECK(disabled.runtime.final_state_hash.empty());
  BOOST_CHECK(disabled.runtime.topology_fingerprint.empty());
  BOOST_CHECK_EQUAL(disabled.runtime.state_hash_audit.layout_builds, 0);
  BOOST_CHECK_EQUAL(disabled.runtime.timing.other.calls, 0);
  BOOST_CHECK_EQUAL(disabled.runtime.timing.accounted_interval_wall_milliseconds,
                    0.0);
  BOOST_CHECK_EQUAL(disabled.runtime.timing.synchronization_audit
                        .stream_synchronize_calls,
                    0);
  BOOST_CHECK(enabled.runtime.instrumentation_effective);
  BOOST_CHECK(!enabled.runtime.initial_state_hash.empty());
  BOOST_CHECK_EQUAL(enabled.runtime.state_hash_audit.layout_builds, 1);
  BOOST_CHECK_EQUAL(enabled.runtime.state_hash_audit.current.computations, 1);
  BOOST_CHECK_GT(enabled.runtime.accepted_trials, 0);
  BOOST_CHECK_EQUAL(enabled.runtime.topology_fingerprint_computations, 1);
  BOOST_CHECK(!enabled.runtime.topology_fingerprint.empty());
  BOOST_CHECK_GT(enabled.runtime.topology_build_count, 1);
  BOOST_CHECK_GT(enabled.runtime.timing.h2d_memcpy.calls, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.h2d_memcpy.bytes, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.h2d_memcpy.cuda_event_milliseconds, 0.0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2h_memcpy.calls, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2h_memcpy.bytes, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2h_memcpy.cuda_event_milliseconds, 0.0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2d_memcpy.calls, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2d_memcpy.bytes, 0);
  BOOST_CHECK_GT(enabled.runtime.timing.d2d_memcpy.cuda_event_milliseconds, 0.0);
  BOOST_CHECK_GT(enabled.runtime.timing.synchronization_audit
                     .stream_synchronize_calls,
                 0);
  const CudaSynchronizationAudit& synchronization =
      enabled.runtime.timing.synchronization_audit;
  BOOST_CHECK_EQUAL(
      synchronization.stream_synchronize_calls,
      synchronization.layer_a_final.calls +
          synchronization.layer_b_final.calls +
          synchronization.layer_c_factor.calls +
          synchronization.layer_c_solver.calls +
          synchronization.layer_c_final.calls +
          synchronization.trial_cost_final.calls +
          synchronization.resource_release.calls);
  BOOST_CHECK(enabled.runtime.timing.host_partition_pass);
  BOOST_CHECK_GE(enabled.runtime.timing.accounted_interval_wall_milliseconds,
                 enabled.runtime.host_wall_milliseconds);
  CheckScalar(
      enabled.runtime.timing.accounted_interval_wall_milliseconds,
      enabled.runtime.timing.explicitly_measured_host_wall_milliseconds +
          enabled.runtime.timing.uninstrumented_remainder_milliseconds,
      1e-10, 1e-12);
}

BOOST_AUTO_TEST_CASE(ForcedRejectReusesCurrentHashAndTopologyFingerprint) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.min_relative_decrease = 2.0;
  options.performance_mode = true;
  options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
  options.capture_state_trace = false;
  CudaFullLmResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &result, &error),
                        error);
  BOOST_CHECK_EQUAL(result.trial_iterations, 4);
  BOOST_CHECK_EQUAL(result.runtime.accepted_trials, 0);
  BOOST_CHECK_EQUAL(result.runtime.rejected_trials,
                    result.runtime.actual_trials);
  BOOST_CHECK_EQUAL(result.runtime.state_hash_audit.current.computations, 1);
  BOOST_CHECK_GT(result.runtime.state_hash_audit.current.cache_hits, 0);
  BOOST_CHECK_EQUAL(result.runtime.topology_fingerprint_computations, 1);
  BOOST_CHECK_GT(result.runtime.topology_reuse_hits, 0);
  BOOST_REQUIRE_EQUAL(result.trace.size(), 5);
  for (size_t i = 1; i < result.trace.size(); ++i) {
    BOOST_CHECK(!result.trace[i].accepted);
    BOOST_CHECK_EQUAL(result.trace[i].state_epoch, 0);
    BOOST_CHECK_EQUAL(result.trace[i].current_state_hash,
                      result.runtime.initial_state_hash);
    BOOST_CHECK_EQUAL(result.trace[i].linearization_id, 1);
    BOOST_CHECK_EQUAL(result.trace[i].linearization_reason,
                      "same_state_recompute");
  }
}

BOOST_AUTO_TEST_CASE(CurrentLinearizationCacheSemanticMatrix) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 3;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  std::array<CudaFullLmResult, 4> results;
  std::string error;
  for (size_t cache = 0; cache < 2; ++cache) {
    for (size_t instrumentation = 0; instrumentation < 2;
         ++instrumentation) {
      CudaFullLmOptions options;
      options.max_num_iterations = 3;
      options.function_tolerance = 0.0;
      options.gradient_tolerance = 0.0;
      options.parameter_tolerance = 0.0;
      options.current_linearization_cache_mode =
          cache == 0 ? CudaCurrentLinearizationCacheMode::kDisabled
                     : CudaCurrentLinearizationCacheMode::kEnabled;
      options.instrumentation_mode =
          instrumentation == 0 ? CudaInstrumentationMode::kDisabled
                               : CudaInstrumentationMode::kEnabled;
      CudaFullLmResult& result = results[cache * 2 + instrumentation];
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &result, &error), error);
      BOOST_CHECK_EQUAL(result.runtime.final_internal_state_epoch,
                        result.accepted_commits);
      BOOST_CHECK_EQUAL(result.runtime.current_linearization_copies, 0);
      BOOST_CHECK_EQUAL(
          result.runtime.current_linearization_logical_requests,
          result.runtime.current_linearization_cache_hits +
              result.runtime.current_linearization_cache_misses);
      BOOST_CHECK_EQUAL(
          result.runtime.current_linearization_build_attempts,
          result.runtime.current_linearization_build_successes +
              result.runtime.current_linearization_build_failures);
      BOOST_CHECK_EQUAL(
          result.runtime.current_linearization_build_successes,
          result.runtime.current_linearization_publishes +
              result.runtime.current_linearization_temporary_builds);
      BOOST_CHECK_LE(result.runtime.current_linearization_replacements,
                     result.runtime.current_linearization_publishes);
      BOOST_CHECK_EQUAL(result.runtime.resource_generation_advance_events, 1);
      BOOST_CHECK_EQUAL(result.runtime.resource_generation_advance_violations,
                        0);
      BOOST_CHECK(result.runtime.resource_generation_advanced_by_this_solve);
      BOOST_CHECK(result.runtime.solve_generation_nonzero);
      BOOST_CHECK(result.runtime.solve_generation_consistent);
      BOOST_CHECK(result.runtime.linearization_identity_consistent);
      BOOST_CHECK(result.runtime.topology_context_generation_consistent);
      BOOST_CHECK(!result.runtime.diagnostic_structure_v2.has_primary);
      BOOST_CHECK_EQUAL(result.runtime.build_cuda_layer_a_inputs_count,
                        result.runtime.layer_a_calls);
      BOOST_CHECK_EQUAL(result.runtime.build_cost_order_count, 1);
      BOOST_CHECK_EQUAL(result.runtime.cross_solve_cache_hits, 0);
      if (instrumentation == 0) {
        BOOST_CHECK_EQUAL(result.runtime.final_state_epoch, 0);
        BOOST_CHECK_EQUAL(
            static_cast<int>(result.runtime.timing_structure_v2.timing_status),
            static_cast<int>(CudaTimingStatus::kDisabled));
        BOOST_CHECK_EQUAL(result.runtime.timing_structure_v2.event_attempts, 0);
        BOOST_CHECK_EQUAL(result.runtime.timing_structure_v2.intervals_started,
                          0);
      } else {
        BOOST_CHECK_EQUAL(result.runtime.final_state_epoch,
                          result.runtime.final_internal_state_epoch);
        BOOST_CHECK_EQUAL(
            static_cast<int>(result.runtime.timing_structure_v2.timing_status),
            static_cast<int>(CudaTimingStatus::kComplete));
      }
    }
  }
  for (size_t i = 1; i < results.size(); ++i) {
    BOOST_CHECK_EQUAL(CudaFullLmSemanticBitwiseSha256V1(results[0]),
                      CudaFullLmSemanticBitwiseSha256V1(results[i]));
    BOOST_CHECK_EQUAL(CudaFullLmSemanticSha256V2(results[0]),
                      CudaFullLmSemanticSha256V2(results[i]));
    BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(results[0].final_state),
                      CudaFinalParametersBitwiseSha256(results[i].final_state));
    BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(results[0].final_state),
                      CudaFinalTopologyBitwiseSha256(results[i].final_state));
  }
  const CudaFullLmResult& cache_off = results[0];
  const CudaFullLmResult& cache_on = results[2];
  BOOST_CHECK_EQUAL(cache_off.runtime.current_linearization_cache_hits, 0);
  BOOST_CHECK_EQUAL(cache_off.runtime.current_linearization_build_attempts,
                    1 + cache_off.trial_iterations +
                        cache_off.accepted_commits);
  BOOST_CHECK_EQUAL(cache_off.runtime.current_linearization_temporary_builds,
                    cache_off.trial_iterations);
  BOOST_CHECK_EQUAL(cache_on.runtime.current_linearization_cache_hits,
                    cache_on.trial_iterations);
  BOOST_CHECK_EQUAL(cache_on.runtime.current_linearization_build_attempts,
                    1 + cache_on.accepted_commits);
  BOOST_CHECK_EQUAL(cache_on.runtime.current_linearization_temporary_builds,
                    0);
  BOOST_CHECK_NE(CudaFullLmExecutionStructureSha256V1(cache_off),
                 CudaFullLmExecutionStructureSha256V1(cache_on));
  BOOST_CHECK_NE(CudaFullLmDecisionBitwiseSha256(cache_off),
                 CudaFullLmDecisionBitwiseSha256(cache_on));
}

BOOST_AUTO_TEST_CASE(FrozenScalingBootstrapIsBitwiseStable) {
  const Snapshot snapshot = LayerBSnapshot();
  CudaLayerBOptions bootstrap_options;
  CudaLayerBResult bootstrap;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerB(
                            snapshot, bootstrap_options, &bootstrap, &error),
                        error);
  CudaLayerBOptions frozen_options = bootstrap_options;
  frozen_options.frozen_pose_jacobi_scaling.assign(
      bootstrap.poses.size() * 6, 0.0);
  frozen_options.frozen_point_jacobi_scaling.assign(
      bootstrap.points.size() * 3, 0.0);
  for (size_t pose = 0; pose < bootstrap.poses.size(); ++pose) {
    for (uint32_t i = 0; i < bootstrap.poses[pose].dimension; ++i) {
      frozen_options.frozen_pose_jacobi_scaling[pose * 6 + i] =
          bootstrap.poses[pose].jacobi_scaling[i];
    }
  }
  for (size_t point = 0; point < bootstrap.points.size(); ++point) {
    for (size_t i = 0; i < 3; ++i) {
      frozen_options.frozen_point_jacobi_scaling[point * 3 + i] =
          bootstrap.points[point].jacobi_scaling[i];
    }
  }
  CudaLayerBResult frozen;
  BOOST_REQUIRE_MESSAGE(
      RunCudaSnapshotLayerB(snapshot, frozen_options, &frozen, &error), error);
  BOOST_CHECK_EQUAL(bootstrap.cost, frozen.cost);
  BOOST_CHECK_EQUAL(bootstrap.projected_gradient_max_norm,
                    frozen.projected_gradient_max_norm);
  BOOST_CHECK_EQUAL(bootstrap.raw_tangent_gradient_max_norm,
                    frozen.raw_tangent_gradient_max_norm);
  BOOST_CHECK_EQUAL(bootstrap.scaled_gradient_max_norm,
                    frozen.scaled_gradient_max_norm);
  BOOST_REQUIRE_EQUAL(bootstrap.poses.size(), frozen.poses.size());
  BOOST_REQUIRE_EQUAL(bootstrap.points.size(), frozen.points.size());
  BOOST_REQUIRE_EQUAL(bootstrap.edges.size(), frozen.edges.size());
  for (size_t block = 0; block < bootstrap.poses.size(); ++block) {
    BOOST_CHECK_EQUAL(bootstrap.poses[block].image_id,
                      frozen.poses[block].image_id);
    BOOST_CHECK_EQUAL(bootstrap.poses[block].dimension,
                      frozen.poses[block].dimension);
    for (size_t i = 0; i < 36; ++i)
      BOOST_CHECK_EQUAL(bootstrap.poses[block].hessian[i],
                        frozen.poses[block].hessian[i]);
    for (size_t i = 0; i < 6; ++i) {
      BOOST_CHECK_EQUAL(bootstrap.poses[block].gradient[i],
                        frozen.poses[block].gradient[i]);
      BOOST_CHECK_EQUAL(bootstrap.poses[block].jacobi_scaling[i],
                        frozen.poses[block].jacobi_scaling[i]);
      BOOST_CHECK_EQUAL(bootstrap.poses[block].damping[i],
                        frozen.poses[block].damping[i]);
    }
  }
  for (size_t block = 0; block < bootstrap.points.size(); ++block) {
    BOOST_CHECK_EQUAL(bootstrap.points[block].point3D_id,
                      frozen.points[block].point3D_id);
    for (size_t i = 0; i < 9; ++i)
      BOOST_CHECK_EQUAL(bootstrap.points[block].hessian[i],
                        frozen.points[block].hessian[i]);
    for (size_t i = 0; i < 3; ++i) {
      BOOST_CHECK_EQUAL(bootstrap.points[block].gradient[i],
                        frozen.points[block].gradient[i]);
      BOOST_CHECK_EQUAL(bootstrap.points[block].jacobi_scaling[i],
                        frozen.points[block].jacobi_scaling[i]);
      BOOST_CHECK_EQUAL(bootstrap.points[block].damping[i],
                        frozen.points[block].damping[i]);
    }
  }
  for (size_t block = 0; block < bootstrap.edges.size(); ++block) {
    BOOST_CHECK_EQUAL(bootstrap.edges[block].pose_index,
                      frozen.edges[block].pose_index);
    BOOST_CHECK_EQUAL(bootstrap.edges[block].point_index,
                      frozen.edges[block].point_index);
    for (size_t i = 0; i < 18; ++i)
      BOOST_CHECK_EQUAL(bootstrap.edges[block].value[i],
                        frozen.edges[block].value[i]);
  }
}

BOOST_AUTO_TEST_CASE(MathAndResourceGenerationsAreSeparated) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 0;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  CudaFullLmResult first;
  CudaFullLmResult second;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &first, &error),
                        error);
  options.layer_c.layer_b.memory_budget_override_bytes =
      2ull * 1024ull * 1024ull * 1024ull;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &second, &error),
                        error);
  BOOST_CHECK_EQUAL(first.runtime.final_config_generation,
                    second.runtime.final_config_generation);
  BOOST_CHECK_EQUAL(first.runtime.final_topology_generation,
                    second.runtime.final_topology_generation);
  BOOST_CHECK_NE(first.runtime.resource_generation,
                 second.runtime.resource_generation);
  BOOST_CHECK_NE(first.runtime.final_solve_generation,
                 second.runtime.final_solve_generation);

  CudaFullLmOptions changed_config = options;
  changed_config.layer_c.layer_b.min_lm_diagonal *= 2.0;
  CudaFullLmResult config;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, changed_config, &config,
                                           &error),
                        error);
  BOOST_CHECK_NE(first.runtime.final_config_generation,
                 config.runtime.final_config_generation);

  Snapshot changed_topology = snapshot;
  changed_topology.images.front().constant_tvec_mask ^= 1u << 2;
  CudaFullLmResult topology;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(changed_topology, options,
                                           &topology, &error),
                        error);
  BOOST_CHECK_NE(first.runtime.final_topology_generation,
                 topology.runtime.final_topology_generation);
}

BOOST_AUTO_TEST_CASE(SolveContextCostLayoutAndPackingContracts) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 0;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  CudaFullLmResult valid;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &valid, &error),
                        error);
  BOOST_CHECK_EQUAL(valid.runtime.build_cuda_layer_a_inputs_count,
                    valid.runtime.layer_a_calls);
  BOOST_CHECK_EQUAL(valid.runtime.build_cost_order_count, 1);
  BOOST_CHECK_EQUAL(valid.runtime.cross_solve_cache_hits, 0);

  Snapshot duplicate = snapshot;
  BOOST_REQUIRE_GE(duplicate.source_insertion_order.size(), 2);
  duplicate.source_insertion_order[1] = duplicate.source_insertion_order[0];
  CudaFullLmResult duplicate_result;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(duplicate, options, &duplicate_result,
                                  &error));
  BOOST_CHECK(error.find("duplicate") != std::string::npos);

  Snapshot incomplete = snapshot;
  BOOST_REQUIRE(!incomplete.source_insertion_order.empty());
  incomplete.source_insertion_order.pop_back();
  CudaFullLmResult incomplete_result;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(incomplete, options, &incomplete_result,
                                  &error));
  BOOST_CHECK(error.find("incomplete") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(PersistentDeviceContextStructureAndLegacyParity) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.performance_mode = true;
  CudaFullLmResult legacy;
  CudaFullLmResult persistent;
  std::string error;
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "legacy");
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &legacy,
                                             &error), error);
  }
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "persistent");
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &persistent,
                                             &error), error);
  }
  BOOST_CHECK_EQUAL(CudaFullLmSemanticBitwiseSha256V1(legacy),
                    CudaFullLmSemanticBitwiseSha256V1(persistent));
  BOOST_CHECK_EQUAL(CudaFullLmSemanticSha256V2(legacy),
                    CudaFullLmSemanticSha256V2(persistent));
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(legacy.final_state),
                    CudaFinalParametersBitwiseSha256(persistent.final_state));
  BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(legacy.final_state),
                    CudaFinalTopologyBitwiseSha256(persistent.final_state));
  BOOST_CHECK_EQUAL(legacy.termination_reason, persistent.termination_reason);
  BOOST_CHECK_EQUAL(legacy.accepted_commits, persistent.accepted_commits);
  BOOST_CHECK_EQUAL(legacy.rejected_steps, persistent.rejected_steps);

  const CudaPersistentDeviceRuntimeInfo& audit =
      persistent.runtime.persistent_device;
  BOOST_CHECK_EQUAL(persistent.runtime.device_context_backend, "persistent");
  BOOST_CHECK(audit.requested);
  BOOST_CHECK(audit.effective);
  BOOST_CHECK(audit.close_succeeded);
  BOOST_CHECK_GT(audit.initialization_allocation_calls, 0);
  BOOST_CHECK_GT(audit.initialization_allocation_bytes, 0);
  BOOST_CHECK_EQUAL(audit.post_initialize_allocation_calls, 0);
  BOOST_CHECK_EQUAL(audit.post_initialize_allocation_bytes, 0);
  BOOST_CHECK_EQUAL(audit.init_stream_create_count, 1);
  BOOST_CHECK_EQUAL(audit.init_event_create_count, 20);
  BOOST_CHECK_EQUAL(audit.init_solver_create_count, 1);
  BOOST_CHECK_EQUAL(audit.init_blas_create_count, 1);
  BOOST_CHECK_EQUAL(audit.steady_stream_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_event_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_solver_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_blas_create_count, 0);
  BOOST_CHECK_EQUAL(audit.a_to_b_h2d_calls + audit.a_to_b_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.a_to_cost_h2d_calls + audit.a_to_cost_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.b_to_c_h2d_calls + audit.b_to_c_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.pending_commit_device_copy_bytes, 0);
  BOOST_CHECK_EQUAL(audit.large_b_copy_bytes, 0);
  BOOST_CHECK_EQUAL(audit.frozen_scaling_upload_calls, 2);
  BOOST_CHECK_GT(audit.workspace_capacity_bytes,
                 audit.cusolver_workspace_capacity_bytes);
  BOOST_CHECK_EQUAL(persistent.runtime.build_cuda_layer_a_inputs_count,
                    persistent.runtime.layer_a_calls);
  BOOST_CHECK_EQUAL(persistent.runtime.build_cost_order_count, 1);
  BOOST_CHECK_EQUAL(persistent.runtime.current_linearization_copies, 0);
  BOOST_CHECK_EQUAL(persistent.runtime.cross_solve_cache_hits, 0);
}

BOOST_AUTO_TEST_CASE(PersistentDeviceContextInvalidSelectorFailsStably) {
  Snapshot snapshot = LayerBSnapshot();
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  CudaFullLmResult result;
  std::string error;
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT", "bad");
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &result, &error));
  BOOST_CHECK_EQUAL(static_cast<int>(result.error_classification),
                    static_cast<int>(CudaSolveErrorClass::kInvalidOptions));
  BOOST_CHECK_EQUAL(error,
                    "COLMAP_PCD_GPU_BA_DEVICE_CONTEXT must be legacy, persistent, device_state, or device_control");
}

BOOST_AUTO_TEST_CASE(TypedProductionSelectorsOverrideCompatibilityEnvironment) {
  Snapshot snapshot = LayerBSnapshot();
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  options.performance_mode = true;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult result;
  std::string error;
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT", "bad");
  ScopedEnvironmentValue hot(
      "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "bad");
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, options, &result, &error), error);
  BOOST_CHECK_EQUAL(result.runtime.device_context_backend, "device_control");
  BOOST_CHECK(result.runtime.persistent_device.hot_kernel_transformed);
  BOOST_CHECK(!result.runtime.persistent_device.hot_kernel_optimized);
}

BOOST_AUTO_TEST_CASE(RuntimePoolArenaAndCompactPipelineLifecycle) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  BOOST_REQUIRE_MESSAGE(RunCudaCompactLayerAComparisonSelfTest(
                            {VisualInput()}, 1e-12, 1e-10, &error),
                        error);
  BOOST_REQUIRE_MESSAGE(RunCudaDeviceScalingComparisonSelfTest(
                            snapshot, 1e-12, 1e-10, &error),
                        error);
  snapshot.metadata.max_num_iterations = 2;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;

  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  options.execution_profile = CudaExecutionProfile::kCompactControl;
  options.performance_mode = true;
  options.capture_state_trace = false;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;

  CudaFullLmResult cold;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &cold, &error),
                        error);
  const CudaPersistentDeviceRuntimeInfo& cold_runtime =
      cold.runtime.persistent_device;
  BOOST_CHECK_EQUAL(cold_runtime.runtime_pool_cold_creates, 1);
  BOOST_CHECK_EQUAL(cold_runtime.runtime_pool_hot_leases, 0);
  BOOST_CHECK_EQUAL(cold_runtime.runtime_pool_returns, 1);
  BOOST_CHECK_EQUAL(cold_runtime.runtime_pool_retained_entries, 1);
  BOOST_CHECK_EQUAL(cold_runtime.arena_grow_calls, 1);
  BOOST_CHECK_EQUAL(cold_runtime.initialization_allocation_calls, 1);
  BOOST_CHECK_EQUAL(cold_runtime.post_initialize_allocation_calls, 0);
  BOOST_CHECK_EQUAL(cold_runtime.bootstrap_scaling_full_d2h_calls, 0);
  BOOST_CHECK_EQUAL(cold_runtime.bootstrap_scaling_full_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.scaling_slot_publishes, 1);
  BOOST_CHECK_EQUAL(cold_runtime.production_identity_fingerprint_calls, 0);
  BOOST_CHECK_EQUAL(cold_runtime.production_identity_fingerprint_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.unified_problem_builder_calls, 1);
  BOOST_CHECK_EQUAL(cold_runtime.independent_cost_order_rebuilds, 0);
  BOOST_CHECK_EQUAL(cold_runtime.public_visual_record_bytes, 480);
  BOOST_CHECK_LE(cold_runtime.compact_visual_record_bytes, 192);
  BOOST_CHECK_GT(cold_runtime.compact_layer_a_slot_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.full_layer_a_slot_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.steady_full_b_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.steady_schur_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.steady_delta_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.steady_trial_state_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(cold_runtime.final_state_materialization_operations, 1);

  CudaFullLmResult hot;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &hot, &error),
                        error);
  const CudaPersistentDeviceRuntimeInfo& hot_runtime =
      hot.runtime.persistent_device;
  BOOST_CHECK_EQUAL(hot_runtime.runtime_pool_cold_creates, 0);
  BOOST_CHECK_EQUAL(hot_runtime.runtime_pool_hot_leases, 1);
  BOOST_CHECK_EQUAL(hot_runtime.arena_grow_calls, 0);
  BOOST_CHECK_EQUAL(hot_runtime.initialization_allocation_calls, 0);
  BOOST_CHECK_EQUAL(hot_runtime.init_stream_create_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.init_event_create_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.init_solver_create_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.init_blas_create_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.close_stream_destroy_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.close_event_destroy_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.close_solver_destroy_count, 0);
  BOOST_CHECK_EQUAL(hot_runtime.close_blas_destroy_count, 0);
  BOOST_CHECK_EQUAL(hot.runtime.cross_solve_cache_hits, 0);
  BOOST_CHECK_EQUAL(hot_runtime.cross_solve_state_handle_hits, 0);
  BOOST_CHECK_EQUAL(hot_runtime.state_lineage_violations, 0);
  BOOST_CHECK_EQUAL(hot_runtime.state_b_pair_violations, 0);
  CheckScalar(cold.final_cost, hot.final_cost, 1e-8, 1e-10);
  BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(cold.final_state),
                    CudaFinalTopologyBitwiseSha256(hot.final_state));

  Snapshot large = snapshot;
  large.metadata.snapshot_id = "runtime-pool-large-arena";
  const ObservationSnapshot template_observation = large.observations.front();
  constexpr size_t kAdditionalObservations = 30000;
  large.observations.reserve(large.observations.size() +
                             kAdditionalObservations);
  large.source_insertion_order.reserve(large.source_insertion_order.size() +
                                       kAdditionalObservations);
  for (size_t i = 0; i < kAdditionalObservations; ++i) {
    ObservationSnapshot observation = template_observation;
    observation.source_index = 1000 + i;
    observation.point2D_idx = static_cast<uint32_t>(1000 + i);
    large.observations.push_back(observation);
    OrderEntrySnapshot order;
    order.source_index = observation.source_index;
    order.residual_kind = ResidualKind::kVisual;
    order.image_id = observation.image_id;
    order.point2D_idx = observation.point2D_idx;
    order.point3D_id = observation.point3D_id;
    large.source_insertion_order.push_back(order);
  }
  large.canonical_order = large.source_insertion_order;
  std::sort(large.canonical_order.begin(), large.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id,
                              lhs.point2D_idx, lhs.point3D_id,
                              lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id,
                              rhs.point2D_idx, rhs.point3D_id,
                              rhs.source_index);
            });
  CudaFullLmOptions capacity_options = options;
  capacity_options.max_num_iterations = 0;
  CudaFullLmResult large_result;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            large, capacity_options, &large_result, &error),
                        error);
  BOOST_CHECK_EQUAL(
      large_result.runtime.persistent_device.runtime_pool_hot_leases, 1);
  BOOST_CHECK_EQUAL(large_result.runtime.persistent_device.arena_grow_calls, 1);
  const uint64_t large_capacity =
      large_result.runtime.persistent_device.arena_capacity_bytes;
  BOOST_CHECK_GT(large_capacity, cold_runtime.arena_capacity_bytes);
  CudaFullLmResult small_after_large;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            snapshot, capacity_options, &small_after_large,
                            &error), error);
  BOOST_CHECK_EQUAL(
      small_after_large.runtime.persistent_device.runtime_pool_hot_leases, 1);
  BOOST_CHECK_EQUAL(
      small_after_large.runtime.persistent_device.arena_grow_calls, 0);
  BOOST_CHECK_EQUAL(
      small_after_large.runtime.persistent_device.arena_capacity_bytes,
      large_capacity);
  BOOST_CHECK_EQUAL(small_after_large.runtime.cross_solve_cache_hits, 0);
  CudaRuntimePoolShutdownInfo shutdown;
  BOOST_REQUIRE_MESSAGE(ShutdownCudaRuntimePool(&error, &shutdown), error);
  BOOST_CHECK_EQUAL(shutdown.idle_entries_destroyed, 1);
  BOOST_CHECK_EQUAL(shutdown.streams_destroyed, 1);
  BOOST_CHECK_EQUAL(shutdown.events_destroyed, 20);
  BOOST_CHECK_EQUAL(shutdown.solvers_destroyed, 1);
  BOOST_CHECK_EQUAL(shutdown.blases_destroyed, 1);
  BOOST_CHECK_EQUAL(shutdown.arenas_destroyed, 1);
  BOOST_CHECK_GT(shutdown.arena_bytes_released, 0);

  // A late accepted-pending failure taints the current entry but successful
  // teardown permits a clean cold rebuild; no tainted entry may be reused.
  CudaFullLmOptions fault = options;
  fault.max_num_iterations = 1;
  fault.fault_trigger.fault_kind =
      CudaFaultInjection::kForceLinearizationFailure;
  fault.fault_trigger.logical_site =
      CudaFaultLogicalSite::kAcceptedPendingLinearization;
  fault.fault_trigger.target_state_epoch = 1;
  fault.fault_trigger.occurrence_within_site = 1;
  CudaFullLmResult failed;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, fault, &failed, &error));
  BOOST_CHECK_EQUAL(failed.runtime.resource_cleanup_successes, 1);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.slot_swap_count, 0);
  CudaFullLmOptions retry = fault;
  retry.fault_trigger = CudaFaultTrigger();
  CudaFullLmResult recovered;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, retry, &recovered,
                                           &error), error);
  BOOST_CHECK_EQUAL(
      recovered.runtime.persistent_device.runtime_pool_cold_creates, 1);
  BOOST_CHECK_EQUAL(recovered.runtime.persistent_device.runtime_pool_hot_leases,
                    0);
  BOOST_CHECK_EQUAL(recovered.runtime.cross_solve_cache_hits, 0);
  BOOST_REQUIRE_MESSAGE(ShutdownCudaRuntimePool(&error), error);

  CudaFullLmOptions cleanup_failure = fault;
  cleanup_failure.fault_trigger.fault_kind =
      CudaFaultInjection::kForceResourceTeardownFailure;
  CudaFullLmResult poisoned;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, cleanup_failure, &poisoned,
                                  &error));
  BOOST_CHECK_EQUAL(poisoned.runtime.resource_cleanup_failures, 1);
  CudaFullLmResult blocked;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, retry, &blocked, &error));
  BOOST_CHECK(error.find("ResourceCleanup:") == 0);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(DeviceControlScalarBoundaryAndDeviceStateParity) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.performance_mode = true;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult device_state;
  CudaFullLmResult device_control;
  std::string error;
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "device_state");
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &device_state, &error), error);
  }
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "device_control");
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &device_control, &error), error);
  }
  BOOST_CHECK_EQUAL(device_control.runtime.device_context_backend,
                    "device_control");
  BOOST_CHECK_EQUAL(static_cast<int>(device_control.termination_type),
                    static_cast<int>(device_state.termination_type));
  BOOST_CHECK_EQUAL(device_control.termination_reason,
                    device_state.termination_reason);
  BOOST_CHECK_EQUAL(device_control.trial_iterations,
                    device_state.trial_iterations);
  BOOST_CHECK_EQUAL(device_control.accepted_commits,
                    device_state.accepted_commits);
  BOOST_CHECK_EQUAL(device_control.rejected_steps, device_state.rejected_steps);
  BOOST_REQUIRE_EQUAL(device_control.trace.size(), device_state.trace.size());
  for (size_t i = 0; i < device_control.trace.size(); ++i) {
    BOOST_CHECK_EQUAL(device_control.trace[i].accepted_decision,
                      device_state.trace[i].accepted_decision);
    BOOST_CHECK_EQUAL(device_control.trace[i].invalid,
                      device_state.trace[i].invalid);
    CheckDeviceControlScalar(device_state.trace[i].predicted_reduction,
                             device_control.trace[i].predicted_reduction);
    CheckDeviceControlScalar(device_state.trace[i].backward_error,
                             device_control.trace[i].backward_error);
    CheckDeviceControlScalar(device_state.trace[i].step_norm,
                             device_control.trace[i].step_norm);
  }
  const double cost_bound = kDeviceControlFinalCostAtol +
      kDeviceControlFinalCostRtol *
          std::max(std::abs(device_state.final_cost),
                   std::abs(device_control.final_cost));
  BOOST_CHECK_SMALL(device_state.final_cost - device_control.final_cost,
                    cost_bound);
  BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(device_control.final_state),
                    CudaFinalTopologyBitwiseSha256(device_state.final_state));

  const CudaPersistentDeviceRuntimeInfo& audit =
      device_control.runtime.persistent_device;
  BOOST_CHECK_EQUAL(audit.steady_full_b_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.steady_schur_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.steady_delta_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.steady_trial_state_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_b_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_state_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_schur_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_delta_bytes, 0);
  BOOST_CHECK_EQUAL(audit.final_state_materialization_operations, 1);
  BOOST_CHECK_EQUAL(audit.post_ready_dynamic_state_h2d_bytes, 0);
  BOOST_CHECK_EQUAL(audit.post_initialize_allocation_bytes, 0);
  BOOST_CHECK_EQUAL(audit.commit_token_violations, 0);
  BOOST_CHECK_EQUAL(audit.state_lineage_violations, 0);
  BOOST_CHECK_EQUAL(audit.state_b_pair_violations, 0);
  BOOST_CHECK_EQUAL(audit.diagnostic_reduction_failures, 0);
  CheckDeviceControlScalar(
      device_state.runtime.persistent_device.diagnostic_max_symmetry_error,
      audit.diagnostic_max_symmetry_error);
  BOOST_CHECK_EQUAL(audit.state_slot_swap_count,
                    static_cast<uint64_t>(device_control.accepted_commits));
  BOOST_CHECK_EQUAL(audit.b_slot_swap_count,
                    static_cast<uint64_t>(device_control.accepted_commits));
  BOOST_CHECK_EQUAL(audit.accepted_commit_device_copy_bytes, 0);
}

BOOST_AUTO_TEST_CASE(TransformedEdgeHostMathRejectsTransposeAndIndexErrors) {
  const std::array<double, 18> lhs{{
      0.2, -0.4, 0.7, 1.1, 0.3, -0.6, -0.8, 0.5, 1.4,
      0.9, -1.2, 0.1, 0.6, 0.2, -0.3, -0.7, 1.3, 0.4}};
  const std::array<double, 18> rhs{{
      -0.5, 0.8, 0.1, 0.4, -1.1, 0.6, 1.2, 0.7, -0.2,
      -0.9, 0.3, 1.5, 0.2, -0.6, 0.9, 1.0, 0.4, -0.8}};
  // Deliberately non-symmetric. This helper test does not enter the production
  // point-factor path; it isolates E*C_inverse orientation and diagonal terms.
  const std::array<double, 9> inverse{{
      1.2, -0.3, 0.7, 0.4, 0.9, -0.8, -0.2, 0.6, 1.4}};
  std::array<double, 18> transformed{};
  TransformEdgeHost(lhs.data(), inverse.data(), 6, transformed.data());
  for (const size_t dimension : {size_t{3}, size_t{5}, size_t{6}}) {
    TransformEdgeHost(lhs.data(), inverse.data(), dimension,
                      transformed.data());
    for (size_t row = 0; row < dimension; ++row) {
      for (size_t col = 0; col < dimension; ++col) {
        const double reference = OriginalEdgeContribution(
            lhs.data(), inverse.data(), rhs.data(), row, col);
        const double candidate = TransformedEdgeContribution(
            transformed.data(), rhs.data(), row, col);
        BOOST_CHECK(TransformedAbsRelPass(candidate, reference, 1e-14, 1e-14));
      }
    }
    for (size_t row = dimension; row < 6; ++row) {
      for (size_t col = 0; col < 3; ++col)
        BOOST_CHECK_EQUAL(transformed[row * 3 + col], 0.0);
    }
  }
  TransformEdgeHost(lhs.data(), inverse.data(), 6, transformed.data());
  const size_t row = 1;
  const size_t col = 4;
  const double value = TransformedEdgeContribution(
      transformed.data(), rhs.data(), row, col);
  const double transpose = TransformedEdgeContribution(
      transformed.data(), rhs.data(), col, row);
  const double diagonal = 0.5 * (value + transpose);
  const double reference_diagonal = 0.5 * (
      OriginalEdgeContribution(lhs.data(), inverse.data(), rhs.data(), row,
                               col) +
      OriginalEdgeContribution(lhs.data(), inverse.data(), rhs.data(), col,
                               row));
  BOOST_CHECK(TransformedAbsRelPass(diagonal, reference_diagonal, 1e-14,
                                   1e-14));

  std::array<double, 18> rhs_transformed{};
  TransformEdgeHost(rhs.data(), inverse.data(), 6, rhs_transformed.data());
  const double wrong_transpose = TransformedEdgeContribution(
      rhs_transformed.data(), lhs.data(), col, row);
  BOOST_CHECK_GT(std::abs(value - wrong_transpose), 1e-3);
  BOOST_CHECK_GT(std::abs(value - transpose), 1e-3);
  BOOST_CHECK_GT(std::abs(diagonal - value), 1e-3);
}

BOOST_AUTO_TEST_CASE(TransformedSchurComponentMeetsFrozenContract) {
  const std::array<uint8_t, 4> masks{{7, 2, 0, 2}};
  for (size_t variant = 0; variant < masks.size(); ++variant) {
    Snapshot snapshot = LayerBSnapshot();
    snapshot.images[0].constant_tvec_mask = masks[variant];
    snapshot.metadata.loss_function = variant == 3 ? "SOFT_L1" : "TRIVIAL";
    CudaLayerCOptions options;
    options.lambda = variant == 1 ? 1e-2 : 1e-4;
    CudaLayerCResult reference;
    CudaLayerCResult optimized;
    CudaLayerCResult transformed;
    CudaLayerCResult repeated;
    std::string error;
    {
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "reference");
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, options, &reference, &error), error);
    }
    {
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "optimized");
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, options, &optimized, &error), error);
    }
    {
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed");
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, options, &transformed, &error), error);
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, options, &repeated, &error), error);
    }
    BOOST_REQUIRE_EQUAL(reference.schur.size(), transformed.schur.size());
    BOOST_REQUIRE_EQUAL(reference.rhs.size(), transformed.rhs.size());
    BOOST_REQUIRE_EQUAL(reference.camera_delta.size(),
                        transformed.camera_delta.size());
    BOOST_REQUIRE_EQUAL(reference.point_delta.size(),
                        transformed.point_delta.size());
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;
    for (size_t i = 0; i < reference.schur.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          transformed.schur[i], reference.schur[i], kTransformedSchurAtol,
          kTransformedSchurRtol));
      const long double difference = static_cast<long double>(
          transformed.schur[i] - reference.schur[i]);
      difference_squared += difference * difference;
      const long double value = reference.schur[i];
      reference_squared += value * value;
    }
    const double relative_frobenius = static_cast<double>(
        std::sqrt(difference_squared) /
        std::max<long double>(1.0L, std::sqrt(reference_squared)));
    BOOST_CHECK_LE(relative_frobenius, kTransformedSchurFrobeniusRtol);
    BOOST_CHECK_EQUAL(reference.rhs.size() * sizeof(double),
                      transformed.rhs.size() * sizeof(double));
    BOOST_CHECK_EQUAL(
        std::memcmp(reference.rhs.data(), transformed.rhs.data(),
                    reference.rhs.size() * sizeof(double)),
        0);
    for (size_t i = 0; i < reference.camera_delta.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          transformed.camera_delta[i], reference.camera_delta[i],
          kTransformedDeltaAtol, kTransformedDeltaRtol));
    }
    for (size_t i = 0; i < reference.point_delta.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          transformed.point_delta[i], reference.point_delta[i],
          kTransformedDeltaAtol, kTransformedDeltaRtol));
    }
    BOOST_CHECK(TransformedAbsRelPass(
        transformed.predicted_reduction, reference.predicted_reduction,
        kTransformedPredictedAtol, kTransformedPredictedRtol));
    BOOST_CHECK(std::isfinite(transformed.backward_error));
    BOOST_CHECK_GE(transformed.backward_error, 0.0);
    BOOST_CHECK_LE(transformed.backward_error,
                   std::max(1e-12, 10.0 * reference.backward_error));
    BOOST_CHECK(TransformedAbsRelPass(
        transformed.trial_cost, reference.trial_cost,
        kTransformedFinalCostAtol, kTransformedFinalCostRtol));
    BOOST_CHECK_EQUAL(
        std::memcmp(transformed.schur.data(), repeated.schur.data(),
                    transformed.schur.size() * sizeof(double)),
        0);
    BOOST_CHECK_EQUAL(
        std::memcmp(transformed.camera_delta.data(),
                    repeated.camera_delta.data(),
                    transformed.camera_delta.size() * sizeof(double)),
        0);
    BOOST_CHECK_EQUAL(reference.rhs.size(), optimized.rhs.size());
    BOOST_CHECK_EQUAL(
        std::memcmp(reference.rhs.data(), optimized.rhs.data(),
                    reference.rhs.size() * sizeof(double)),
        0);
  }

  Snapshot no_edge_snapshot = LayerBSnapshot();
  for (PointSnapshot& point : no_edge_snapshot.points) point.constant = true;
  CudaLayerCOptions no_edge_options;
  CudaLayerCResult no_edge;
  std::string error;
  {
    ScopedEnvironmentValue hot(
        "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed");
    BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                              no_edge_snapshot, no_edge_options, &no_edge,
                              &error),
                          error);
  }
  BOOST_CHECK(no_edge.success);
  BOOST_CHECK_EQUAL(no_edge.runtime.point_factorization_failures, 0);
  for (const double value : no_edge.schur) BOOST_CHECK(std::isfinite(value));
}

BOOST_AUTO_TEST_CASE(SegmentedSchurContributionSelectorCoverageAndNumerics) {
  const Snapshot snapshot = SegmentedPairSnapshot();
  std::string error;
  CudaSchurContributionComponentResult component;
  BOOST_REQUIRE_MESSAGE(RunCudaSchurContributionComponentForTesting(
                            snapshot, 16, 3, &component, &error),
                        error);
  BOOST_CHECK_GT(component.segment_count, 1);
  BOOST_CHECK_GT(component.segmented_pair_count, 0);
  BOOST_CHECK_EQUAL(component.coverage_violations, 0);
  BOOST_CHECK_EQUAL(component.element_contract_violations, 0);
  BOOST_CHECK_LE(component.normalized_max_error, 5e-11);
  BOOST_CHECK_LE(component.normalized_frobenius_error, 5e-11);
  BOOST_CHECK_EQUAL(component.repeated_candidate_max_abs_error, 0.0);
  BOOST_REQUIRE_EQUAL(component.direct_complete_milliseconds.size(), 3);
  BOOST_REQUIRE_EQUAL(component.segmented_complete_milliseconds.size(), 3);

  CudaLayerCOptions direct_options;
  direct_options.schur_contribution_backend =
      CudaSchurContributionBackend::kDirectTransformed;
  CudaLayerCOptions segmented_options = direct_options;
  segmented_options.schur_contribution_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  segmented_options.schur_segment_size_for_testing = 16;
  CudaLayerCResult direct_before;
  CudaLayerCResult segmented;
  CudaLayerCResult repeated;
  CudaLayerCResult direct_after;
  CudaLayerCResult compatibility_default;
  ScopedEnvironmentValue hot(
      "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed");
  {
    ScopedEnvironmentValue compatibility(
        "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND", "segmented");
    BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                              snapshot, direct_options, &direct_before,
                              &error),
                          error);
  }
  BOOST_CHECK_EQUAL(direct_before.runtime.segment_plan_builds, 0);
  BOOST_CHECK_EQUAL(direct_before.runtime.partial_workspace_bytes, 0);
  BOOST_CHECK_EQUAL(direct_before.runtime.segment_metadata_h2d_bytes, 0);
  {
    ScopedEnvironmentValue compatibility(
        "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND", "direct");
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                              snapshot, segmented_options, &segmented,
                              &error),
                          error);
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                              snapshot, segmented_options, &repeated,
                              &error),
                          error);
  }
  BOOST_CHECK_EQUAL(segmented.runtime.segment_plan_builds, 1);
  BOOST_CHECK_GT(segmented.runtime.segment_count, 1);
  BOOST_CHECK_GT(segmented.runtime.partial_workspace_bytes, 0);
  BOOST_CHECK_GT(segmented.runtime.segment_metadata_h2d_bytes, 0);
  BOOST_CHECK_EQUAL(segmented.runtime.coverage_violations, 0);
  BOOST_CHECK_EQUAL(segmented.runtime.schur_contribution_calls, 1);
  BOOST_CHECK_EQUAL(segmented.runtime.schur_rhs_calls, 1);
  BOOST_CHECK_EQUAL(segmented.runtime.factorization_calls, 1);
  BOOST_CHECK_EQUAL(segmented.runtime.pair_contribution_chunk_launches, 1);
  BOOST_CHECK_NE(segmented.runtime.schur_contribution_calls,
                 segmented.runtime.partial_kernel_launches +
                     segmented.runtime.merge_kernel_launches);

  BOOST_REQUIRE_EQUAL(direct_before.schur.size(), segmented.schur.size());
  long double reference_squared = 0.0L;
  long double difference_squared = 0.0L;
  double maximum = 0.0;
  double reference_maximum = 0.0;
  for (size_t i = 0; i < direct_before.schur.size(); ++i) {
    const double difference =
        std::abs(segmented.schur[i] - direct_before.schur[i]);
    const double limit = 1e-10 + 5e-11 * std::max(
        std::abs(direct_before.schur[i]), std::abs(segmented.schur[i]));
    BOOST_CHECK_LE(difference, limit);
    maximum = std::max(maximum, difference);
    reference_maximum =
        std::max(reference_maximum, std::abs(direct_before.schur[i]));
    reference_squared += static_cast<long double>(direct_before.schur[i]) *
                         direct_before.schur[i];
    difference_squared += static_cast<long double>(difference) * difference;
  }
  BOOST_CHECK_LE(maximum / std::max(1.0, reference_maximum), 5e-11);
  BOOST_CHECK_LE(static_cast<double>(std::sqrt(difference_squared) /
                                    std::max<long double>(
                                        1.0L, std::sqrt(reference_squared))),
                 5e-11);
  BOOST_REQUIRE_EQUAL(direct_before.rhs.size(), segmented.rhs.size());
  BOOST_CHECK_EQUAL(std::memcmp(direct_before.rhs.data(), segmented.rhs.data(),
                                direct_before.rhs.size() * sizeof(double)),
                    0);
  for (size_t i = 0; i < direct_before.camera_delta.size(); ++i) {
    BOOST_CHECK(TransformedAbsRelPass(
        segmented.camera_delta[i], direct_before.camera_delta[i],
        kTransformedDeltaAtol, kTransformedDeltaRtol));
  }
  for (size_t i = 0; i < direct_before.point_delta.size(); ++i) {
    BOOST_CHECK(TransformedAbsRelPass(
        segmented.point_delta[i], direct_before.point_delta[i],
        kTransformedDeltaAtol, kTransformedDeltaRtol));
  }
  BOOST_CHECK_EQUAL(std::memcmp(segmented.schur.data(), repeated.schur.data(),
                                segmented.schur.size() * sizeof(double)),
                    0);

  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                            snapshot, direct_options, &direct_after, &error),
                        error);
  BOOST_CHECK_EQUAL(
      std::memcmp(direct_before.schur.data(), direct_after.schur.data(),
                  direct_before.schur.size() * sizeof(double)),
      0);
  {
    ScopedEnvironmentUnset compatibility(
        "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND");
    CudaLayerCOptions default_options;
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCudaSnapshotLayerC(
                              snapshot, default_options,
                              &compatibility_default, &error),
                          error);
  }
  BOOST_CHECK_EQUAL(compatibility_default.runtime.segment_plan_builds, 0);
  BOOST_CHECK_EQUAL(compatibility_default.runtime.partial_workspace_bytes, 0);
}

BOOST_AUTO_TEST_CASE(SegmentedSchurContributionFullLmTypedAndCounters) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
  Snapshot snapshot = SegmentedPairSnapshot();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions direct_options;
  direct_options.max_num_iterations = 1;
  direct_options.function_tolerance = 0.0;
  direct_options.gradient_tolerance = 0.0;
  direct_options.parameter_tolerance = 0.0;
  direct_options.performance_mode = true;
  direct_options.capture_state_trace = false;
  direct_options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
  direct_options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  direct_options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  direct_options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  direct_options.execution_profile = CudaExecutionProfile::kCompactControl;
  direct_options.layer_c.schur_contribution_backend =
      CudaSchurContributionBackend::kDirectTransformed;
  CudaFullLmOptions segmented_options = direct_options;
  segmented_options.layer_c.schur_contribution_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  segmented_options.layer_c.schur_segment_size_for_testing = 16;

  CudaFullLmResult direct_before;
  CudaFullLmResult segmented;
  CudaFullLmResult segmented_instrumented;
  CudaFullLmResult direct_after;
  {
    ScopedEnvironmentValue compatibility(
        "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND", "segmented");
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                              snapshot, direct_options, &direct_before,
                              &error),
                          error);
  }
  {
    ScopedEnvironmentValue compatibility(
        "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND", "direct");
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                              snapshot, segmented_options, &segmented,
                              &error),
                          error);
    CudaFullLmOptions instrumented_options = segmented_options;
    instrumented_options.instrumentation_mode =
        CudaInstrumentationMode::kEnabled;
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                              snapshot, instrumented_options,
                              &segmented_instrumented, &error),
                          error);
  }
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            snapshot, direct_options, &direct_after, &error),
                        error);
  const auto& direct_audit = direct_before.runtime.persistent_device;
  const auto& segmented_audit = segmented.runtime.persistent_device;
  const auto& instrumented_audit =
      segmented_instrumented.runtime.persistent_device;
  const auto& direct_after_audit = direct_after.runtime.persistent_device;
  BOOST_CHECK_EQUAL(direct_audit.schur_contribution_backend_effective,
                    "direct");
  BOOST_CHECK_EQUAL(direct_audit.schur_segment_plan_builds, 0);
  BOOST_CHECK_EQUAL(direct_audit.schur_segment_metadata_h2d_bytes, 0);
  BOOST_CHECK_EQUAL(direct_audit.schur_partial_workspace_bytes, 0);
  BOOST_CHECK_EQUAL(direct_audit.schur_segment_plan_host_milliseconds, 0.0);
  BOOST_CHECK_GT(direct_audit.solve_context_initialization_wall_milliseconds,
                 0.0);
  BOOST_CHECK_GT(direct_audit.problem_topology_preparation_wall_milliseconds,
                 0.0);
  BOOST_CHECK_EQUAL(segmented_audit.schur_contribution_backend_effective,
                    "segmented");
  BOOST_CHECK_EQUAL(segmented_audit.schur_segment_plan_builds, 1);
  BOOST_CHECK_GT(segmented_audit.segment_count, 1);
  BOOST_CHECK_GT(segmented_audit.schur_partial_workspace_bytes, 0);
  BOOST_CHECK_GT(segmented_audit.schur_segment_plan_host_milliseconds, 0.0);
  BOOST_CHECK_GT(
      segmented_audit.solve_context_initialization_wall_milliseconds, 0.0);
  BOOST_CHECK_GT(
      segmented_audit.problem_topology_preparation_wall_milliseconds, 0.0);
  BOOST_CHECK_EQUAL(segmented_audit.schur_contribution_coverage_violations, 0);
  BOOST_CHECK_EQUAL(segmented_audit.post_initialize_allocation_calls, 0);
  BOOST_CHECK_EQUAL(segmented_audit.schur_contribution_calls,
                    segmented.runtime.layer_c_calls);
  BOOST_CHECK_EQUAL(segmented_audit.schur_rhs_calls,
                    segmented.runtime.layer_c_calls);
  BOOST_CHECK_EQUAL(segmented_audit.factorization_calls,
                    segmented.runtime.layer_c_calls);
  BOOST_CHECK_EQUAL(segmented_audit.schur_pair_contribution_chunk_launches,
                    segmented.runtime.layer_c_calls);
  BOOST_CHECK_EQUAL(segmented_audit.layer_c_max_chunks_per_step, 1);
  BOOST_CHECK_EQUAL(instrumented_audit.schur_contribution_calls,
                    segmented_audit.schur_contribution_calls);
  BOOST_CHECK_EQUAL(instrumented_audit.schur_rhs_calls,
                    segmented_audit.schur_rhs_calls);
  BOOST_CHECK_EQUAL(instrumented_audit.factorization_calls,
                    segmented_audit.factorization_calls);
  BOOST_CHECK_EQUAL(direct_after_audit.schur_segment_plan_builds, 0);
  BOOST_CHECK_EQUAL(direct_after_audit.schur_partial_workspace_bytes, 0);
  BOOST_CHECK_EQUAL(direct_after_audit.arena_grow_calls, 0);
  BOOST_CHECK_EQUAL(direct_after_audit.schur_contribution_backend_effective,
                    "direct");
  const double cost_limit = 1e-8 + 1e-10 *
      std::max(1.0, std::abs(direct_before.final_cost));
  BOOST_CHECK_LE(std::abs(segmented.final_cost - direct_before.final_cost),
                 cost_limit);
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(ObservationHessianAssemblySelectorMatrixAndComponent) {
  std::string error;
  Snapshot snapshot = SegmentedPairSnapshot();
  CudaHessianAssemblyComponentResult component;
  BOOST_REQUIRE_MESSAGE(RunCudaHessianAssemblyComponentForTesting(
                            snapshot, 16, 2, &component, &error),
                        error);
  BOOST_CHECK_GT(component.pose_segment_count, 1);
  BOOST_CHECK_EQUAL(component.element_contract_violations_segmented, 0);
  BOOST_CHECK_EQUAL(component.segmented_repeated_max_abs_error, 0.0);
  BOOST_REQUIRE_EQUAL(component.pose_owned_complete_milliseconds.size(), 2);
  BOOST_REQUIRE_EQUAL(component.atomic_complete_milliseconds.size(), 2);
  BOOST_REQUIRE_EQUAL(component.segmented_complete_milliseconds.size(), 2);

  snapshot.metadata.max_num_iterations = 0;
  const std::array<std::pair<CudaHessianAssemblyBackend, const char*>, 3>
      hessian_backends{{
          {CudaHessianAssemblyBackend::kPoseOwnedOptimizedReference,
           "pose_owned"},
          {CudaHessianAssemblyBackend::kObservationAtomic,
           "observation_atomic"},
          {CudaHessianAssemblyBackend::kObservationSegmented,
           "observation_segmented"}}};
  const std::array<std::pair<CudaSchurContributionBackend, const char*>, 2>
      schur_backends{{
          {CudaSchurContributionBackend::kDirectTransformed, "direct"},
          {CudaSchurContributionBackend::kSegmentedTransformed,
           "segmented"}}};
  ScopedEnvironmentValue hessian_environment(
      "COLMAP_PCD_GPU_BA_HESSIAN_ASSEMBLY_BACKEND", "pose_owned_scalar");
  ScopedEnvironmentValue schur_environment(
      "COLMAP_PCD_GPU_BA_SCHUR_CONTRIBUTION_BACKEND", "direct");
  for (const auto& hessian : hessian_backends) {
    for (const auto& schur : schur_backends) {
      CudaFullLmOptions options;
      options.max_num_iterations = 0;
      options.performance_mode = true;
      options.capture_state_trace = false;
      options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
      options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
      options.execution_profile = CudaExecutionProfile::kCompactControl;
      options.current_linearization_cache_mode =
          CudaCurrentLinearizationCacheMode::kEnabled;
      options.layer_c.layer_b.hessian_assembly_backend = hessian.first;
      options.layer_c.layer_b.hessian_segment_size_for_testing = 16;
      options.layer_c.schur_contribution_backend = schur.first;
      options.layer_c.schur_segment_size_for_testing = 16;
      CudaFullLmResult solve;
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &solve, &error), error);
      const auto& audit = solve.runtime.persistent_device;
      BOOST_CHECK_EQUAL(audit.hessian_assembly_backend_effective,
                        hessian.second);
      BOOST_CHECK_EQUAL(audit.schur_contribution_backend_effective,
                        schur.second);
      BOOST_CHECK_EQUAL(audit.hessian_gradient_assembly_calls, 1);
      BOOST_CHECK_EQUAL(audit.pose_block_assembly_calls, 1);
      BOOST_CHECK_EQUAL(audit.point_block_assembly_calls, 1);
      BOOST_CHECK_EQUAL(audit.edge_block_assembly_calls, 1);
      BOOST_CHECK_EQUAL(audit.jacobi_damping_finalize_calls, 1);
      BOOST_CHECK_EQUAL(audit.gradient_summary_calls, 1);
      BOOST_CHECK_EQUAL(audit.hessian_assembly_coverage_violations, 0);
      if (hessian.first ==
          CudaHessianAssemblyBackend::kPoseOwnedOptimizedReference) {
        BOOST_CHECK_EQUAL(audit.hessian_candidate_metadata_build_calls, 0);
        BOOST_CHECK_EQUAL(audit.hessian_candidate_metadata_h2d_bytes, 0);
        BOOST_CHECK_EQUAL(audit.hessian_partial_workspace_bytes, 0);
      } else {
        BOOST_CHECK_EQUAL(audit.hessian_candidate_metadata_build_calls, 1);
        BOOST_CHECK_GT(audit.hessian_candidate_metadata_h2d_bytes, 0);
      }
      BOOST_CHECK_EQUAL(audit.post_initialize_allocation_calls, 0);
    }
  }
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(LayerCPairChunkHostPartitionAndCall505Scale) {
  CudaLayerCPairChunkingTestSummary summary;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaLayerCPairChunkingHostSelfTest(&summary, &error), error);
  BOOST_CHECK_EQUAL(summary.production_limit_bytes, 64ull * 1024ull * 1024ull);
  BOOST_CHECK_EQUAL(summary.stress_pose_count, 229);
  BOOST_CHECK_GT(summary.stress_pair_count, 0);
  BOOST_CHECK_GT(summary.stress_contribution_bytes,
                 summary.production_limit_bytes);
  BOOST_CHECK_GT(summary.stress_chunk_count, 1);
  BOOST_CHECK_LE(summary.stress_max_chunk_bytes,
                 summary.production_limit_bytes);
  BOOST_TEST_MESSAGE(
      "call505_scale poses=" << summary.stress_pose_count
      << " pairs=" << summary.stress_pair_count
      << " contributions=" << summary.stress_contribution_count
      << " bytes=" << summary.stress_contribution_bytes
      << " chunks=" << summary.stress_chunk_count
      << " max_chunk_bytes=" << summary.stress_max_chunk_bytes);
}

BOOST_AUTO_TEST_CASE(LayerCPairChunkLaunchesMeetFrozenCudaContract) {
  struct Selector {
    const char* name;
    CudaHotKernelMode typed;
  };
  const std::array<Selector, 3> selectors{{
      {"reference", CudaHotKernelMode::kReference},
      {"optimized", CudaHotKernelMode::kOptimized},
      {"transformed", CudaHotKernelMode::kTransformed},
  }};
  const uint64_t forced_limit = 12;
  const auto compare_layer_c = [](const CudaLayerCResult& reference,
                                  const CudaLayerCResult& candidate) {
    BOOST_REQUIRE_EQUAL(reference.schur.size(), candidate.schur.size());
    BOOST_REQUIRE_EQUAL(reference.rhs.size(), candidate.rhs.size());
    BOOST_REQUIRE_EQUAL(reference.camera_delta.size(),
                        candidate.camera_delta.size());
    BOOST_REQUIRE_EQUAL(reference.point_delta.size(),
                        candidate.point_delta.size());
    long double difference_squared = 0.0L;
    long double reference_squared = 0.0L;
    for (size_t i = 0; i < reference.schur.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          candidate.schur[i], reference.schur[i], kTransformedSchurAtol,
          kTransformedSchurRtol));
      const long double difference = static_cast<long double>(
          candidate.schur[i] - reference.schur[i]);
      difference_squared += difference * difference;
      const long double value = reference.schur[i];
      reference_squared += value * value;
    }
    const double relative_frobenius = static_cast<double>(
        std::sqrt(difference_squared) /
        std::max<long double>(1.0L, std::sqrt(reference_squared)));
    BOOST_CHECK_LE(relative_frobenius, kTransformedSchurFrobeniusRtol);
    BOOST_CHECK_EQUAL(
        std::memcmp(reference.rhs.data(), candidate.rhs.data(),
                    reference.rhs.size() * sizeof(double)),
        0);
    for (size_t i = 0; i < reference.camera_delta.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          candidate.camera_delta[i], reference.camera_delta[i],
          kTransformedDeltaAtol, kTransformedDeltaRtol));
    }
    for (size_t i = 0; i < reference.point_delta.size(); ++i) {
      BOOST_CHECK(TransformedAbsRelPass(
          candidate.point_delta[i], reference.point_delta[i],
          kTransformedDeltaAtol, kTransformedDeltaRtol));
    }
    BOOST_CHECK(TransformedAbsRelPass(
        candidate.predicted_reduction, reference.predicted_reduction,
        kTransformedPredictedAtol, kTransformedPredictedRtol));
    BOOST_CHECK(TransformedAbsRelPass(candidate.trial_cost,
                                     reference.trial_cost,
                                     kTransformedFinalCostAtol,
                                     kTransformedFinalCostRtol));
    BOOST_CHECK(std::isfinite(candidate.backward_error));
    BOOST_CHECK_GE(candidate.backward_error, 0.0);
    BOOST_CHECK_LE(candidate.backward_error,
                   std::max(1e-12, 10.0 * reference.backward_error));
  };

  for (const Selector& selector : selectors) {
    const Snapshot snapshot = PairChunkSnapshot();
    CudaLayerCOptions one_options;
    CudaLayerCOptions multi_options;
    multi_options.pair_chunk_limit_bytes_for_testing = forced_limit;
    CudaLayerCResult one;
    CudaLayerCResult multi;
    CudaLayerCResult repeated;
    std::string error;
    {
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", selector.name);
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, one_options, &one, &error), error);
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, multi_options, &multi, &error), error);
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCudaSnapshotLayerC(snapshot, multi_options, &repeated, &error),
          error);
    }
    BOOST_CHECK_EQUAL(one.runtime.layer_c_max_chunks_per_step, 1);
    BOOST_CHECK_EQUAL(one.runtime.layer_c_pair_chunk_launch_calls, 1);
    BOOST_CHECK_GT(multi.runtime.layer_c_max_chunks_per_step, 1);
    BOOST_CHECK_EQUAL(multi.runtime.layer_c_pair_chunk_launch_calls,
                      multi.runtime.layer_c_max_chunks_per_step);
    BOOST_CHECK_LE(multi.runtime.layer_c_max_chunk_bytes, forced_limit);
    compare_layer_c(one, multi);
    double schur_max_abs = 0.0;
    double camera_delta_max_abs = 0.0;
    double point_delta_max_abs = 0.0;
    for (size_t i = 0; i < one.schur.size(); ++i) {
      schur_max_abs =
          std::max(schur_max_abs, std::abs(one.schur[i] - multi.schur[i]));
    }
    for (size_t i = 0; i < one.camera_delta.size(); ++i) {
      camera_delta_max_abs = std::max(
          camera_delta_max_abs,
          std::abs(one.camera_delta[i] - multi.camera_delta[i]));
    }
    for (size_t i = 0; i < one.point_delta.size(); ++i) {
      point_delta_max_abs = std::max(
          point_delta_max_abs,
          std::abs(one.point_delta[i] - multi.point_delta[i]));
    }
    BOOST_TEST_MESSAGE(
        "pair_chunk selector=" << selector.name
        << " chunks=" << multi.runtime.layer_c_max_chunks_per_step
        << " launches=" << multi.runtime.layer_c_pair_chunk_launch_calls
        << " max_chunk_bytes=" << multi.runtime.layer_c_max_chunk_bytes
        << " schur_max_abs=" << schur_max_abs
        << " rhs_exact=true camera_delta_max_abs=" << camera_delta_max_abs
        << " point_delta_max_abs=" << point_delta_max_abs
        << " predicted_abs="
        << std::abs(one.predicted_reduction - multi.predicted_reduction)
        << " trial_cost_abs=" << std::abs(one.trial_cost - multi.trial_cost));
    BOOST_CHECK_EQUAL(
        std::memcmp(multi.schur.data(), repeated.schur.data(),
                    multi.schur.size() * sizeof(double)),
        0);
    BOOST_CHECK_EQUAL(
        std::memcmp(multi.camera_delta.data(), repeated.camera_delta.data(),
                    multi.camera_delta.size() * sizeof(double)),
        0);

    CudaFullLmOptions full_one_options;
    full_one_options.max_num_iterations = 1;
    full_one_options.function_tolerance = 0.0;
    full_one_options.gradient_tolerance = 0.0;
    full_one_options.parameter_tolerance = 0.0;
    full_one_options.performance_mode = true;
    full_one_options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
    full_one_options.hot_kernel_mode = selector.typed;
    CudaFullLmOptions full_multi_options = full_one_options;
    full_multi_options.pair_chunk_limit_bytes_for_testing = forced_limit;
    CudaFullLmResult full_one;
    CudaFullLmResult full_multi;
    CudaFullLmResult full_repeated;
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, full_one_options, &full_one, &error), error);
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                              snapshot, full_multi_options, &full_multi, &error),
                          error);
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, full_multi_options, &full_repeated, &error),
        error);
    const CudaPersistentDeviceRuntimeInfo& one_audit =
        full_one.runtime.persistent_device;
    const CudaPersistentDeviceRuntimeInfo& multi_audit =
        full_multi.runtime.persistent_device;
    BOOST_CHECK_EQUAL(one_audit.layer_c_max_chunks_per_step, 1);
    BOOST_CHECK_GT(multi_audit.layer_c_max_chunks_per_step, 1);
    BOOST_CHECK_EQUAL(one_audit.layer_c_pair_chunk_launch_calls,
                      full_one.runtime.layer_c_calls);
    BOOST_CHECK_EQUAL(
        multi_audit.layer_c_pair_chunk_launch_calls,
        full_multi.runtime.layer_c_calls *
            multi_audit.layer_c_max_chunks_per_step);
    BOOST_CHECK_EQUAL(one_audit.initialization_allocation_calls,
                      multi_audit.initialization_allocation_calls);
    BOOST_CHECK_EQUAL(one_audit.initialization_allocation_bytes,
                      multi_audit.initialization_allocation_bytes);
    BOOST_CHECK_EQUAL(one_audit.static_upload_calls,
                      multi_audit.static_upload_calls);
    BOOST_CHECK_EQUAL(one_audit.static_upload_bytes,
                      multi_audit.static_upload_bytes);
    BOOST_CHECK_EQUAL(multi_audit.post_initialize_allocation_bytes, 0);
    BOOST_CHECK_EQUAL(multi_audit.post_ready_dynamic_state_h2d_bytes, 0);
    if (selector.typed == CudaHotKernelMode::kTransformed) {
      BOOST_CHECK_EQUAL(multi_audit.transform_kernel_calls,
                        full_multi.runtime.layer_c_calls);
    }
    BOOST_CHECK(TransformedAbsRelPass(
        full_multi.final_cost, full_one.final_cost,
        kTransformedFinalCostAtol, kTransformedFinalCostRtol));
    BOOST_TEST_MESSAGE(
        "pair_chunk persistent selector=" << selector.name
        << " chunks=" << multi_audit.layer_c_max_chunks_per_step
        << " launches=" << multi_audit.layer_c_pair_chunk_launch_calls
        << " layer_c_calls=" << full_multi.runtime.layer_c_calls
        << " final_cost_abs="
        << std::abs(full_one.final_cost - full_multi.final_cost));
    BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(full_multi.final_state),
                      CudaFinalParametersBitwiseSha256(
                          full_repeated.final_state));
  }

  CudaLayerCOptions too_small;
  too_small.pair_chunk_limit_bytes_for_testing = 1;
  CudaLayerCResult rejected;
  std::string error;
  BOOST_CHECK(!RunCudaSnapshotLayerC(
      LayerBSnapshot(), too_small, &rejected, &error));
  BOOST_CHECK_EQUAL(
      error.find("Layer C single pair adjacency exceeds chunk limit"), 0);
}

BOOST_AUTO_TEST_CASE(HotKernelReferenceOptimizedAndTransformedContracts) {
  const std::array<uint8_t, 4> masks{{7, 2, 0, 2}};
  for (size_t variant = 0; variant < masks.size(); ++variant) {
    Snapshot snapshot = LayerBSnapshot();
    snapshot.images[0].constant_tvec_mask = masks[variant];
    snapshot.metadata.loss_function =
        variant == 3 ? "SOFT_L1" : "TRIVIAL";
    snapshot.metadata.max_num_iterations = 3;
    snapshot.metadata.function_tolerance = 0.0;
    snapshot.metadata.gradient_tolerance = 0.0;
    snapshot.metadata.parameter_tolerance = 0.0;
    CudaFullLmOptions options;
    options.max_num_iterations = 3;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 0.0;
    options.parameter_tolerance = 0.0;
    options.performance_mode = true;
    options.capture_state_trace = true;
    options.current_linearization_cache_mode =
        CudaCurrentLinearizationCacheMode::kEnabled;
    CudaFullLmResult reference;
    CudaFullLmResult optimized;
    CudaFullLmResult transformed;
    std::string error;
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "reference");
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &reference, &error), error);
    }
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "optimized");
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &optimized, &error), error);
    }
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed");
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &transformed, &error), error);
    }
    BOOST_CHECK(!reference.runtime.persistent_device.hot_kernel_optimized);
    BOOST_CHECK(optimized.runtime.persistent_device.hot_kernel_optimized);
    BOOST_CHECK(transformed.runtime.persistent_device.hot_kernel_transformed);
    BOOST_CHECK_GT(transformed.runtime.persistent_device.transform_kernel_calls,
                   0);
    BOOST_CHECK_EQUAL(
        transformed.runtime.persistent_device.transform_kernel_calls,
        static_cast<uint64_t>(transformed.trial_iterations));
    BOOST_CHECK_GT(
        transformed.runtime.persistent_device.transform_workspace_bytes, 0);
    BOOST_CHECK_EQUAL(CudaFullLmSemanticBitwiseSha256V1(reference),
                      CudaFullLmSemanticBitwiseSha256V1(optimized));
    BOOST_CHECK_EQUAL(CudaFullLmSemanticSha256V2(reference),
                      CudaFullLmSemanticSha256V2(optimized));
    BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(reference.final_state),
                      CudaFinalParametersBitwiseSha256(optimized.final_state));
    BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(reference.final_state),
                      CudaFinalTopologyBitwiseSha256(optimized.final_state));
    BOOST_CHECK_EQUAL(reference.termination_reason,
                      optimized.termination_reason);
    BOOST_CHECK_EQUAL(reference.accepted_commits,
                      optimized.accepted_commits);
    BOOST_CHECK_EQUAL(reference.rejected_steps, optimized.rejected_steps);
    BOOST_CHECK_EQUAL(reference.termination_reason,
                      transformed.termination_reason);
    BOOST_CHECK_EQUAL(reference.accepted_commits,
                      transformed.accepted_commits);
    BOOST_CHECK_EQUAL(reference.rejected_steps, transformed.rejected_steps);
    BOOST_REQUIRE_EQUAL(reference.trace.size(), transformed.trace.size());
    for (size_t i = 0; i < reference.trace.size(); ++i) {
      BOOST_CHECK_EQUAL(reference.trace[i].accepted,
                        transformed.trace[i].accepted);
      BOOST_CHECK_EQUAL(reference.trace[i].invalid,
                        transformed.trace[i].invalid);
    }
    BOOST_CHECK(TransformedAbsRelPass(
        transformed.final_cost, reference.final_cost,
        kTransformedFinalCostAtol, kTransformedFinalCostRtol));
    BOOST_REQUIRE_EQUAL(reference.final_state.images.size(),
                        transformed.final_state.images.size());
    BOOST_REQUIRE_EQUAL(reference.final_state.points.size(),
                        transformed.final_state.points.size());
    for (size_t i = 0; i < reference.final_state.images.size(); ++i) {
      for (size_t j = 0; j < 4; ++j) {
        BOOST_CHECK_SMALL(reference.final_state.images[i].qvec[j] -
                              transformed.final_state.images[i].qvec[j],
                          1e-8);
      }
      for (size_t j = 0; j < 3; ++j) {
        BOOST_CHECK_SMALL(reference.final_state.images[i].tvec[j] -
                              transformed.final_state.images[i].tvec[j],
                          1e-6);
      }
    }
    for (size_t i = 0; i < reference.final_state.points.size(); ++i) {
      for (size_t j = 0; j < 3; ++j) {
        BOOST_CHECK_SMALL(reference.final_state.points[i].xyz[j] -
                              transformed.final_state.points[i].xyz[j],
                          1e-5);
      }
    }
    BOOST_CHECK_EQUAL(reference.runtime.persistent_device.init_event_create_count,
                      optimized.runtime.persistent_device.init_event_create_count);
    BOOST_CHECK_EQUAL(
        reference.runtime.persistent_device.initialization_allocation_calls,
        optimized.runtime.persistent_device.initialization_allocation_calls);
    BOOST_CHECK_EQUAL(
        transformed.runtime.persistent_device.initialization_allocation_calls,
        optimized.runtime.persistent_device.initialization_allocation_calls +
            1);
    BOOST_CHECK_EQUAL(
        transformed.runtime.persistent_device.initialization_allocation_bytes,
        optimized.runtime.persistent_device.initialization_allocation_bytes +
            transformed.runtime.persistent_device.transform_workspace_bytes);
  }

  {
    Snapshot snapshot = LayerBSnapshot();
    snapshot.metadata.max_num_iterations = 1;
    snapshot.metadata.function_tolerance = 0.0;
    snapshot.metadata.gradient_tolerance = 0.0;
    snapshot.metadata.parameter_tolerance = 0.0;
    CudaFullLmOptions options;
    options.max_num_iterations = 1;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 0.0;
    options.parameter_tolerance = 0.0;
    options.performance_mode = true;
    CudaFullLmResult default_result;
    CudaFullLmResult optimized_result;
    std::string error;
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentUnset hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION");
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &default_result, &error), error);
    }
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "optimized");
      error.clear();
      BOOST_REQUIRE_MESSAGE(
          RunCustomCudaSolve(snapshot, options, &optimized_result, &error),
          error);
    }
    BOOST_CHECK(default_result.runtime.persistent_device.hot_kernel_optimized);
    BOOST_CHECK(!default_result.runtime.persistent_device.hot_kernel_transformed);
    BOOST_CHECK_EQUAL(
        default_result.runtime.persistent_device.transform_workspace_bytes, 0);
    BOOST_CHECK_EQUAL(CudaFullLmSemanticBitwiseSha256V1(default_result),
                      CudaFullLmSemanticBitwiseSha256V1(optimized_result));
    BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(default_result.final_state),
                      CudaFinalParametersBitwiseSha256(
                          optimized_result.final_state));
    BOOST_CHECK_EQUAL(
        default_result.runtime.persistent_device.initialization_allocation_calls,
        optimized_result.runtime.persistent_device.initialization_allocation_calls);
    BOOST_CHECK_EQUAL(
        default_result.runtime.persistent_device.initialization_allocation_bytes,
        optimized_result.runtime.persistent_device.initialization_allocation_bytes);

    CudaFullLmOptions budget_options = options;
    budget_options.layer_c.layer_b.memory_budget_override_bytes = 1;
    CudaFullLmResult optimized_budget;
    CudaFullLmResult transformed_budget;
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "optimized");
      error.clear();
      BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                                snapshot, budget_options, &optimized_budget,
                                &error),
                            error);
    }
    {
      ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                     "device_control");
      ScopedEnvironmentValue hot(
          "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "transformed");
      error.clear();
      BOOST_CHECK(!RunCustomCudaSolve(snapshot, budget_options,
                                     &transformed_budget, &error));
      BOOST_CHECK(error.find("INSUFFICIENT_GPU_MEMORY for transformed edge workspace") !=
                  std::string::npos);
    }
  }

  Snapshot snapshot = LayerBSnapshot();
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  CudaFullLmResult result;
  std::string error;
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "device_control");
  ScopedEnvironmentValue hot(
      "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION", "invalid");
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &result, &error));
  BOOST_CHECK_EQUAL(
      error,
      "COLMAP_PCD_GPU_BA_HOT_KERNEL_IMPLEMENTATION must be reference, optimized, or transformed");
}

BOOST_AUTO_TEST_CASE(DeterministicDiagnosticReductionEdgeCases) {
  const std::array<size_t, 7> sizes{{0, 1, 128, 512, 513, 1500, 300000}};
  for (const size_t size : sizes) {
    std::vector<uint8_t> status(size, 0);
    std::vector<std::array<double, 2>> norm(size);
    std::vector<double> maxima(size, -0.0);
    long double first_sum = 0.0;
    long double second_sum = 0.0;
    double expected_maximum = 0.0;
    double expected_diagonal_minimum =
        std::numeric_limits<double>::max();
    double expected_diagonal_maximum = 0.0;
    for (size_t i = 0; i < size; ++i) {
      norm[i][0] = static_cast<double>(i % 17) * 1e-6;
      norm[i][1] = static_cast<double>(i % 11) * 2e-6;
      maxima[i] = i % 2 == 0 ? -0.0 : static_cast<double>(i % 97) * 1e-5;
      first_sum += norm[i][0];
      second_sum += norm[i][1];
      expected_maximum = std::max(expected_maximum, maxima[i]);
      expected_diagonal_minimum = std::min(
          expected_diagonal_minimum, std::abs(maxima[i]));
      expected_diagonal_maximum = std::max(
          expected_diagonal_maximum, std::abs(maxima[i]));
    }
    if (size > 7) status[7] = 1;
    if (size > 777) status[777] = 1;
    CudaDiagnosticReductionTestResult result;
    std::string error;
    BOOST_REQUIRE_MESSAGE(RunCudaDiagnosticReductionForTesting(
                              status, norm, maxima, &result, &error), error);
    BOOST_CHECK_EQUAL(result.status_count,
                      (size > 7 ? 1u : 0u) + (size > 777 ? 1u : 0u));
    BOOST_CHECK_EQUAL(result.first_status_index, size > 7 ? 7 : -1);
    CheckDeviceControlScalar(std::sqrt(static_cast<double>(first_sum)),
                             result.first_norm);
    CheckDeviceControlScalar(std::sqrt(static_cast<double>(second_sum)),
                             result.second_norm);
    BOOST_CHECK_EQUAL(result.maximum, expected_maximum);
    BOOST_CHECK_EQUAL(result.diagonal_minimum,
                      size == 0 ? 0.0 : expected_diagonal_minimum);
    BOOST_CHECK_EQUAL(result.diagonal_maximum, expected_diagonal_maximum);
    BOOST_CHECK_EQUAL(result.finite_flags, 0);
    BOOST_CHECK_EQUAL(result.reduction_status, 0);
  }

  std::vector<uint8_t> status(1500, 0);
  status[13] = 1;
  status[1400] = 1;
  std::vector<std::array<double, 2>> norm(1500);
  std::vector<double> maxima(1500);
  for (size_t i = 0; i < norm.size(); ++i) {
    norm[i] = {{static_cast<double>(i % 13) * 1e-5,
                static_cast<double>(i % 19) * 1e-6}};
    maxima[i] = static_cast<double>(i % 101) * 1e-4;
  }
  CudaDiagnosticReductionTestResult reference;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaDiagnosticReductionForTesting(
                            status, norm, maxima, &reference, &error), error);
  for (int repeat = 0; repeat < 20; ++repeat) {
    CudaDiagnosticReductionTestResult candidate;
    error.clear();
    BOOST_REQUIRE_MESSAGE(RunCudaDiagnosticReductionForTesting(
                              status, norm, maxima, &candidate, &error), error);
    BOOST_CHECK_EQUAL(candidate.status_count, reference.status_count);
    BOOST_CHECK_EQUAL(candidate.first_status_index,
                      reference.first_status_index);
    BOOST_CHECK_EQUAL(candidate.first_norm, reference.first_norm);
    BOOST_CHECK_EQUAL(candidate.second_norm, reference.second_norm);
    BOOST_CHECK_EQUAL(candidate.maximum, reference.maximum);
    BOOST_CHECK_EQUAL(candidate.diagonal_minimum,
                      reference.diagonal_minimum);
    BOOST_CHECK_EQUAL(candidate.diagonal_maximum,
                      reference.diagonal_maximum);
    BOOST_CHECK_EQUAL(candidate.finite_flags, reference.finite_flags);
    BOOST_CHECK_EQUAL(candidate.reduction_status, reference.reduction_status);
  }

  status[42] = 2;
  norm[17][0] = std::numeric_limits<double>::infinity();
  maxima[23] = std::numeric_limits<double>::quiet_NaN();
  CudaDiagnosticReductionTestResult nonfinite;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCudaDiagnosticReductionForTesting(
                            status, norm, maxima, &nonfinite, &error), error);
  BOOST_CHECK_NE(nonfinite.finite_flags, 0);
  BOOST_CHECK_NE(nonfinite.reduction_status, 0);
  BOOST_CHECK_EQUAL(nonfinite.first_status_index, 13);
}

BOOST_AUTO_TEST_CASE(DeviceControlRejectAndToleranceKeepCurrentBinding) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.min_relative_decrease = 2.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "device_control");
  CudaFullLmResult rejected;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, options, &rejected, &error), error);
  BOOST_CHECK_EQUAL(rejected.accepted_commits, 0);
  BOOST_CHECK_EQUAL(rejected.rejected_steps, 4);
  BOOST_CHECK_EQUAL(rejected.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(rejected.runtime.persistent_device.state_slot_swap_count,
                    0);
  BOOST_CHECK_EQUAL(rejected.runtime.persistent_device.b_slot_swap_count, 0);
  BOOST_CHECK_EQUAL(
      rejected.runtime.persistent_device.state_slot_discard_count, 4);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(rejected.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));
  BOOST_CHECK_EQUAL(
      rejected.runtime.persistent_device.final_state_materialization_operations,
      1);

  options.min_relative_decrease = 1e-3;
  options.max_num_iterations = 2;
  options.parameter_tolerance = std::numeric_limits<double>::max();
  CudaFullLmResult tolerance;
  error.clear();
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, options, &tolerance, &error), error);
  BOOST_CHECK_EQUAL(tolerance.termination_reason, "parameter_tolerance");
  BOOST_CHECK_EQUAL(tolerance.accepted_commits, 0);
  BOOST_CHECK_EQUAL(tolerance.runtime.persistent_device.state_slot_swap_count,
                    0);
  BOOST_CHECK_EQUAL(tolerance.runtime.persistent_device.b_slot_swap_count, 0);
  BOOST_CHECK_EQUAL(
      tolerance.runtime.persistent_device.final_state_materialization_operations,
      1);
  BOOST_CHECK_EQUAL(
      tolerance.runtime.persistent_device.steady_trial_state_d2h_bytes, 0);
}

BOOST_AUTO_TEST_CASE(DeviceStateUpdateStatusRecoveryAndFailClosed) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 3;
  snapshot.metadata.max_consecutive_invalid_steps = 3;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;

  CudaFullLmOptions options;
  options.max_num_iterations = 3;
  options.max_num_consecutive_invalid_steps = 3;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  options.execution_profile = CudaExecutionProfile::kCompactControl;
  options.performance_mode = true;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;

  const std::string entry_hash = CudaFinalParametersBitwiseSha256(snapshot);
  for (const uint32_t status : {4u, 8u}) {
    CudaFullLmResult recovered;
    std::string error;
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolveWithStateUpdateStatusForTesting(
            snapshot, options, status, &recovered, &error),
        error);
    BOOST_CHECK_EQUAL(recovered.invalid_steps, 1);
    BOOST_CHECK_EQUAL(recovered.factorization_failures, 0);
    BOOST_REQUIRE_GE(recovered.trace.size(), 3);
    const CudaLmIteration& invalid = recovered.trace[1];
    BOOST_CHECK(invalid.invalid);
    BOOST_CHECK(invalid.factorization_success);
    BOOST_CHECK(!invalid.accepted_decision);
    BOOST_CHECK_LT(invalid.radius_after, invalid.radius_before);
    BOOST_CHECK_EQUAL(invalid.trial_state_hash, invalid.current_state_hash);
    BOOST_CHECK(!recovered.trace[2].invalid);
    BOOST_CHECK_GE(
        recovered.runtime.persistent_device.state_slot_invalidate_count, 1);
  }

  for (const uint32_t status : {1u, 2u}) {
    CudaFullLmResult failed;
    std::string error;
    BOOST_CHECK(!RunCustomCudaSolveWithStateUpdateStatusForTesting(
        snapshot, options, status, &failed, &error));
    BOOST_CHECK(error.find("invalid mapping") != std::string::npos);
    BOOST_CHECK_EQUAL(failed.invalid_steps, 0);
    BOOST_CHECK_EQUAL(failed.factorization_failures, 0);
    BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(failed.final_state),
                      entry_hash);
  }
}

BOOST_AUTO_TEST_CASE(DeviceControlAuditMirrorMatchesDeviceStateOracle) {
  Snapshot snapshot = LayerBSnapshot();
  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.capture_state_trace = true;
  options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult device_state;
  CudaFullLmResult device_control;
  std::string error;
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "device_state");
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &device_state, &error), error);
  }
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "device_control");
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &device_control, &error), error);
  }
  BOOST_CHECK_EQUAL(CudaFullLmSemanticBitwiseSha256V1(device_control),
                    CudaFullLmSemanticBitwiseSha256V1(device_state));
  BOOST_CHECK_EQUAL(CudaFullLmSemanticSha256V2(device_control),
                    CudaFullLmSemanticSha256V2(device_state));
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(device_control.final_state),
                    CudaFinalParametersBitwiseSha256(device_state.final_state));
  BOOST_CHECK_GT(device_control.runtime.persistent_device.audit_mirror_b_bytes,
                 0);
  BOOST_CHECK_GT(
      device_control.runtime.persistent_device.audit_mirror_state_bytes, 0);
  BOOST_CHECK_GT(
      device_control.runtime.persistent_device.audit_mirror_schur_bytes, 0);
  BOOST_CHECK_GT(
      device_control.runtime.persistent_device.audit_mirror_delta_bytes, 0);
  BOOST_CHECK_EQUAL(
      device_control.runtime.persistent_device.commit_token_violations, 0);
}

BOOST_AUTO_TEST_CASE(DeviceStateLifecycleAndPersistentParity) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.performance_mode = true;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult persistent;
  CudaFullLmResult device_state;
  std::string error;
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "persistent");
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &persistent, &error), error);
  }
  {
    ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                   "device_state");
    error.clear();
    BOOST_REQUIRE_MESSAGE(
        RunCustomCudaSolve(snapshot, options, &device_state, &error), error);
  }
  BOOST_CHECK_EQUAL(device_state.runtime.device_context_backend,
                    "device_state");
  BOOST_CHECK_EQUAL(static_cast<int>(device_state.termination_type),
                    static_cast<int>(persistent.termination_type));
  BOOST_CHECK_EQUAL(device_state.termination_reason,
                    persistent.termination_reason);
  BOOST_CHECK_EQUAL(device_state.trial_iterations,
                    persistent.trial_iterations);
  BOOST_CHECK_EQUAL(device_state.accepted_commits,
                    persistent.accepted_commits);
  BOOST_CHECK_EQUAL(device_state.rejected_steps, persistent.rejected_steps);
  BOOST_REQUIRE_EQUAL(device_state.trace.size(), persistent.trace.size());
  for (size_t i = 0; i < device_state.trace.size(); ++i) {
    BOOST_CHECK_EQUAL(device_state.trace[i].accepted_decision,
                      persistent.trace[i].accepted_decision);
    BOOST_CHECK_EQUAL(device_state.trace[i].invalid,
                      persistent.trace[i].invalid);
  }
  BOOST_CHECK_CLOSE_FRACTION(device_state.final_cost, persistent.final_cost,
                             1e-9);
  BOOST_CHECK_EQUAL(CudaFinalTopologyBitwiseSha256(device_state.final_state),
                    CudaFinalTopologyBitwiseSha256(persistent.final_state));
  BOOST_REQUIRE_EQUAL(device_state.final_state.images.size(), 2);
  BOOST_REQUIRE_EQUAL(device_state.final_state.points.size(), 2);
  BOOST_CHECK_EQUAL(device_state.final_state.images[0].tvec[1],
                    snapshot.images[0].tvec[1]);
  BOOST_CHECK(device_state.final_state.images[1].qvec ==
              snapshot.images[1].qvec);
  BOOST_CHECK(device_state.final_state.images[1].tvec ==
              snapshot.images[1].tvec);
  BOOST_CHECK(device_state.final_state.points[1].xyz ==
              snapshot.points[1].xyz);
  BOOST_CHECK(device_state.final_state.cameras[0].params ==
              snapshot.cameras[0].params);

  const CudaPersistentDeviceRuntimeInfo& audit =
      device_state.runtime.persistent_device;
  BOOST_CHECK_EQUAL(device_state.runtime.build_cuda_layer_a_inputs_count, 1);
  BOOST_CHECK_EQUAL(device_state.runtime.build_cost_order_count, 1);
  BOOST_CHECK_EQUAL(audit.static_problem_layout_builds, 1);
  BOOST_CHECK_EQUAL(audit.initial_entity_state_pack_calls, 1);
  BOOST_CHECK_EQUAL(audit.initial_entity_state_upload_calls, 2);
  BOOST_CHECK_EQUAL(audit.post_ready_dynamic_state_h2d_calls, 0);
  BOOST_CHECK_EQUAL(audit.post_ready_dynamic_state_h2d_bytes, 0);
  BOOST_CHECK_EQUAL(audit.per_observation_dynamic_pack_calls, 0);
  BOOST_CHECK_EQUAL(audit.per_observation_dynamic_pack_bytes, 0);
  BOOST_CHECK_EQUAL(audit.dynamic_state_upload_calls, 0);
  BOOST_CHECK_EQUAL(audit.dynamic_state_upload_bytes, 0);
  BOOST_CHECK_EQUAL(audit.state_update_kernel_calls,
                    static_cast<uint64_t>(device_state.trial_iterations));
  BOOST_CHECK_EQUAL(audit.state_slot_swap_count,
                    static_cast<uint64_t>(device_state.accepted_commits));
  BOOST_CHECK_EQUAL(audit.reject_state_slot_swap_count, 0);
  BOOST_CHECK_EQUAL(audit.accepted_commit_device_copy_bytes, 0);
  BOOST_CHECK_EQUAL(audit.state_lineage_violations, 0);
  BOOST_CHECK_EQUAL(audit.state_b_pair_violations, 0);
  BOOST_CHECK_EQUAL(audit.cross_solve_state_handle_hits, 0);
  BOOST_CHECK_EQUAL(audit.a_to_b_h2d_calls + audit.a_to_b_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.a_to_cost_h2d_calls + audit.a_to_cost_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.b_to_c_h2d_calls + audit.b_to_c_d2h_calls, 0);
  BOOST_CHECK_EQUAL(audit.post_initialize_allocation_calls, 0);
  BOOST_CHECK_EQUAL(audit.steady_stream_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_event_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_solver_create_count, 0);
  BOOST_CHECK_EQUAL(audit.steady_blas_create_count, 0);
}

BOOST_AUTO_TEST_CASE(DeviceStateForcedRejectPreservesCurrentBinding) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.min_relative_decrease = 2.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult result;
  std::string error;
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "device_state");
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &result, &error),
                        error);
  BOOST_CHECK_EQUAL(result.accepted_commits, 0);
  BOOST_CHECK_EQUAL(result.rejected_steps, 4);
  BOOST_CHECK_EQUAL(result.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(result.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.state_slot_swap_count, 0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.reject_state_slot_swap_count,
                    0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.state_slot_discard_count,
                    4);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.state_lineage_violations,
                    0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.state_b_pair_violations,
                    0);
}

BOOST_AUTO_TEST_CASE(DeviceStateQuaternionPlusMatchesHostOracle) {
  const std::array<double, 4> quaternion{{0.91, 0.13, -0.21, 0.31}};
  const std::array<std::array<double, 3>, 3> deltas{{
      {{0.0, 0.0, 0.0}},
      {{1e-14, -2e-14, 3e-14}},
      {{0.03, -0.02, 0.015}},
  }};
  for (const auto& delta : deltas) {
    std::array<double, 4> expected;
    std::array<double, 4> actual;
    uint32_t status = 99;
    std::string error;
    BOOST_REQUIRE(QuaternionPlusCeres14(quaternion, delta, &expected));
    BOOST_REQUIRE_MESSAGE(RunCudaQuaternionPlusForTesting(
                              quaternion, delta, &actual, &status, &error),
                          error);
    BOOST_CHECK_EQUAL(status, 0);
    for (size_t i = 0; i < actual.size(); ++i) {
      if (delta == deltas.front()) {
        BOOST_CHECK_EQUAL(actual[i], expected[i]);
      } else {
        BOOST_CHECK_SMALL(actual[i] - expected[i], 2e-15);
      }
    }
  }
  std::array<double, 3> nonfinite{{
      std::numeric_limits<double>::infinity(), 0.0, 0.0}};
  std::array<double, 4> ignored;
  uint32_t status = 0;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaQuaternionPlusForTesting(
                            quaternion, nonfinite, &ignored, &status, &error),
                        error);
  BOOST_CHECK_NE(status, 0);
}

BOOST_AUTO_TEST_CASE(DeviceStateLatePendingFailureDoesNotPublishTrial) {
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "device_state");
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 2;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  const std::string initial = CudaFinalParametersBitwiseSha256(snapshot);
  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.fault_trigger.fault_kind =
      CudaFaultInjection::kForceLinearizationFailure;
  options.fault_trigger.logical_site =
      CudaFaultLogicalSite::kAcceptedPendingLinearization;
  options.fault_trigger.trigger_phase =
      CudaFaultTriggerPhase::kAfterTopologyRefreshBeforePublish;
  options.fault_trigger.target_state_epoch = 1;
  CudaFullLmResult failed;
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &failed, &error));
  BOOST_CHECK_EQUAL(failed.accepted_commits, 0);
  BOOST_CHECK_EQUAL(failed.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(failed.final_state),
                    initial);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.state_slot_swap_count, 0);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.accepted_commit_device_copy_bytes,
                    0);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.state_b_pair_violations, 0);
  BOOST_CHECK_EQUAL(static_cast<int>(failed.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kTainted));
  BOOST_CHECK(failed.runtime.resource_cleanup_complete);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(DeviceStatePrecommitToleranceDiscardsTrial) {
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "device_state");
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 2;
  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.gradient_tolerance = 0.0;
  options.function_tolerance = 0.0;
  options.parameter_tolerance = std::numeric_limits<double>::max();
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult parameter;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, options, &parameter, &error), error);
  BOOST_CHECK_EQUAL(parameter.termination_reason, "parameter_tolerance");
  BOOST_CHECK_EQUAL(parameter.accepted_commits, 0);
  BOOST_CHECK_EQUAL(parameter.runtime.persistent_device.state_slot_swap_count,
                    0);
  BOOST_CHECK_EQUAL(parameter.runtime.persistent_device.state_slot_discard_count,
                    1);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(parameter.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));

  options.parameter_tolerance = 0.0;
  options.function_tolerance = std::numeric_limits<double>::max();
  CudaFullLmResult function;
  error.clear();
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, options, &function, &error), error);
  BOOST_CHECK_EQUAL(function.termination_reason, "function_tolerance");
  BOOST_CHECK_EQUAL(function.accepted_commits, 0);
  BOOST_CHECK_EQUAL(function.runtime.persistent_device.state_slot_swap_count,
                    0);
  BOOST_CHECK_EQUAL(function.runtime.persistent_device.state_slot_discard_count,
                    1);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(function.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));
}

BOOST_AUTO_TEST_CASE(PersistentDeviceHandleRejectsStaleSlotGeneration) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaPersistentHandleSelfTest(LayerBSnapshot(), &error), error);
}

BOOST_AUTO_TEST_CASE(DeviceStateHandlesRejectStaleAndCrossSolveIdentity) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaDeviceStateHandleSelfTest(LayerBSnapshot(), &error), error);
}

BOOST_AUTO_TEST_CASE(PersistentPendingFailureKeepsStateAndRebuildsResources) {
  ScopedEnvironmentValue backend("COLMAP_PCD_GPU_BA_DEVICE_CONTEXT",
                                 "persistent");
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 2;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  const std::string initial_parameters =
      CudaFinalParametersBitwiseSha256(snapshot);
  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.fault_trigger.fault_kind =
      CudaFaultInjection::kForceLinearizationFailure;
  options.fault_trigger.logical_site =
      CudaFaultLogicalSite::kAcceptedPendingLinearization;
  options.fault_trigger.trigger_phase =
      CudaFaultTriggerPhase::kAfterTopologyRefreshBeforePublish;
  options.fault_trigger.target_state_epoch = 1;
  CudaFullLmResult failed;
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &failed, &error));
  BOOST_CHECK_EQUAL(static_cast<int>(failed.error_classification),
                    static_cast<int>(
                        CudaSolveErrorClass::kAcceptedPendingLinearization));
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(failed.final_state),
                    initial_parameters);
  BOOST_CHECK_EQUAL(failed.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.slot_swap_count, 0);
  BOOST_CHECK_EQUAL(failed.runtime.persistent_device.pending_commit_device_copy_bytes,
                    0);
  BOOST_CHECK_EQUAL(static_cast<int>(failed.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kTainted));
  BOOST_CHECK(failed.runtime.resource_cleanup_complete);

  CudaFullLmOptions retry = options;
  retry.fault_trigger = CudaFaultTrigger();
  CudaFullLmResult recovered;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, retry, &recovered,
                                           &error), error);
  BOOST_CHECK_EQUAL(recovered.runtime.cross_solve_cache_hits, 0);
  BOOST_CHECK_EQUAL(recovered.runtime.persistent_device.post_initialize_allocation_calls,
                    0);
  BOOST_CHECK_GT(recovered.runtime.persistent_device.initialization_allocation_calls,
                 0);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(CurrentLinearizationForcedRejectHardCounts) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 4;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 4;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.min_relative_decrease = 2.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
  CudaFullLmResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &result, &error),
                        error);
  BOOST_CHECK_EQUAL(result.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(result.runtime.current_linearization_build_attempts, 1);
  BOOST_CHECK_EQUAL(result.runtime.current_linearization_publishes, 1);
  BOOST_CHECK_EQUAL(result.runtime.current_linearization_cache_hits, 4);
  BOOST_CHECK_EQUAL(result.runtime.current_linearization_replacements, 0);
  BOOST_CHECK_EQUAL(result.runtime.layer_a_calls, 5);
  BOOST_CHECK_EQUAL(result.runtime.layer_b_calls, 1);
  BOOST_CHECK_EQUAL(result.runtime.layer_c_calls, 4);
  BOOST_CHECK_EQUAL(result.runtime.cost_calls, 5);
  BOOST_CHECK_EQUAL(result.runtime.build_cuda_layer_a_inputs_count, 5);
  BOOST_CHECK_EQUAL(result.runtime.build_cuda_layer_a_inputs_count,
                    result.runtime.layer_a_calls);
  BOOST_CHECK_EQUAL(result.runtime.build_cost_order_count, 1);
  BOOST_CHECK_EQUAL(result.runtime.cross_solve_cache_hits, 0);
  BOOST_CHECK_EQUAL(result.runtime.rejected_trials, 4);
  BOOST_CHECK_EQUAL(result.runtime.accepted_trials, 0);
}

BOOST_AUTO_TEST_CASE(InternalEpochTerminationBoundaries) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 0;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult maximum_iterations;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            snapshot, options, &maximum_iterations, &error),
                        error);
  BOOST_CHECK_EQUAL(maximum_iterations.runtime.final_internal_state_epoch, 0);

  options.max_num_iterations = 4;
  options.gradient_tolerance = std::numeric_limits<double>::max();
  CudaFullLmResult gradient;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &gradient,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(gradient.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(gradient.trial_iterations, 0);

  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = std::numeric_limits<double>::max();
  CudaFullLmResult parameter;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &parameter,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(parameter.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(parameter.termination_reason, "parameter_tolerance");

  options.parameter_tolerance = 0.0;
  options.function_tolerance = std::numeric_limits<double>::max();
  CudaFullLmResult function;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &function,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(function.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(function.termination_reason, "function_tolerance");
}

BOOST_AUTO_TEST_CASE(AcceptedPendingFailureIsTransactionalAndTaintsResources) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 1;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.performance_mode = true;
  options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  options.fault_trigger.fault_kind =
      CudaFaultInjection::kForceLinearizationFailure;
  options.fault_trigger.logical_site =
      CudaFaultLogicalSite::kAcceptedPendingLinearization;
  options.fault_trigger.target_state_epoch = 1;
  options.fault_trigger.occurrence_within_site = 1;
  CudaFullLmResult failed;
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &failed, &error));
  BOOST_CHECK_EQUAL(failed.accepted_decisions, 1);
  BOOST_CHECK_EQUAL(failed.accepted_commits, 0);
  BOOST_CHECK_EQUAL(failed.accepted_pending_preparation_failures, 0);
  BOOST_CHECK_EQUAL(failed.accepted_pending_linearization_failures, 1);
  BOOST_CHECK_EQUAL(failed.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(
      CudaFinalParametersBitwiseSha256(failed.final_state),
      CudaFinalParametersBitwiseSha256(snapshot));
  BOOST_CHECK_EQUAL(static_cast<int>(failed.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kTainted));
  BOOST_CHECK_EQUAL(failed.runtime.resource_cleanup_successes, 1);

  CudaFullLmOptions retry = options;
  retry.fault_trigger = CudaFaultTrigger();
  CudaFullLmResult recovered;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, retry, &recovered,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(recovered.runtime.cross_solve_cache_hits, 0);
  BOOST_CHECK_EQUAL(recovered.runtime.current_linearization_builds_by_state_epoch[0],
                    1);

  options.fault_trigger.fault_kind =
      CudaFaultInjection::kForceResourceTeardownFailure;
  CudaFullLmResult teardown_failed;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, options, &teardown_failed, &error));
  BOOST_CHECK_EQUAL(static_cast<int>(teardown_failed.error_classification),
                    static_cast<int>(CudaSolveErrorClass::kResourceCleanup));
  BOOST_CHECK_EQUAL(teardown_failed.runtime.resource_cleanup_failures, 1);
  CudaFullLmResult blocked;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, retry, &blocked, &error));
  BOOST_CHECK(error.find("ResourceCleanup:") == 0);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(AcceptedPendingFaultPhasesHaveExactCountersAndTaint) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  auto options_for = [](const CudaFaultTriggerPhase phase,
                        const CudaFaultInjection kind) {
    CudaFullLmOptions options;
    options.max_num_iterations = 1;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 0.0;
    options.parameter_tolerance = 0.0;
    options.performance_mode = true;
    options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
    options.current_linearization_cache_mode =
        CudaCurrentLinearizationCacheMode::kEnabled;
    options.fault_trigger.fault_kind = kind;
    options.fault_trigger.logical_site =
        CudaFaultLogicalSite::kAcceptedPendingLinearization;
    options.fault_trigger.trigger_phase = phase;
    options.fault_trigger.target_state_epoch = 1;
    options.fault_trigger.occurrence_within_site = 1;
    return options;
  };
  const std::string initial_parameters =
      CudaFinalParametersBitwiseSha256(snapshot);

  CudaFullLmResult preparation;
  CudaFullLmOptions preparation_options = options_for(
      CudaFaultTriggerPhase::kAcceptedPendingPreparation,
      CudaFaultInjection::kForcePendingPreparationFailure);
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, preparation_options, &preparation,
                                  &error));
  BOOST_CHECK_EQUAL(preparation.accepted_decisions, 1);
  BOOST_CHECK_EQUAL(preparation.accepted_commits, 0);
  BOOST_CHECK_EQUAL(preparation.accepted_pending_preparation_failures, 1);
  BOOST_CHECK_EQUAL(preparation.accepted_pending_linearization_failures, 0);
  BOOST_CHECK_EQUAL(preparation.runtime.current_linearization_logical_requests,
                    2);
  BOOST_CHECK_EQUAL(preparation.runtime.current_linearization_cache_lookups, 2);
  BOOST_CHECK_EQUAL(preparation.runtime.current_linearization_cache_hits, 1);
  BOOST_CHECK_EQUAL(preparation.runtime.current_linearization_cache_misses, 1);
  BOOST_CHECK_EQUAL(preparation.runtime.current_linearization_build_attempts,
                    1);
  BOOST_CHECK_EQUAL(preparation.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(static_cast<int>(preparation.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kClean));
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(preparation.final_state),
                    initial_parameters);
  BOOST_CHECK(CudaLegacyV1ComparisonAvailable(preparation));
  BOOST_CHECK_EQUAL(
      static_cast<int>(CudaLegacyV1ErrorClassification(preparation)),
      static_cast<int>(CudaSolveErrorClass::kAcceptedPendingLinearization));
  BOOST_CHECK_EQUAL(CudaLegacyV1TerminationReason(preparation),
                    "accepted_state_pending_preparation_failed");

  CudaFullLmResult before_lookup;
  CudaFullLmOptions before_options = options_for(
      CudaFaultTriggerPhase::kBeforeCacheLookup,
      CudaFaultInjection::kForceLinearizationFailure);
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, before_options, &before_lookup,
                                  &error));
  BOOST_CHECK_EQUAL(before_lookup.accepted_decisions, 1);
  BOOST_CHECK_EQUAL(before_lookup.accepted_pending_linearization_failures, 1);
  BOOST_CHECK_EQUAL(before_lookup.runtime.current_linearization_logical_requests,
                    3);
  BOOST_CHECK_EQUAL(before_lookup.runtime.current_linearization_lookup_aborts,
                    1);
  BOOST_CHECK_EQUAL(before_lookup.runtime.current_linearization_cache_lookups,
                    2);
  BOOST_CHECK_EQUAL(before_lookup.runtime.current_linearization_build_attempts,
                    1);
  BOOST_CHECK_EQUAL(static_cast<int>(before_lookup.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kClean));

  CudaFullLmResult after_miss;
  CudaFullLmOptions miss_options = options_for(
      CudaFaultTriggerPhase::kAfterCacheMiss,
      CudaFaultInjection::kForceLinearizationFailure);
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, miss_options, &after_miss, &error));
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_logical_requests,
                    3);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_cache_lookups, 3);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_cache_hits, 1);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_cache_misses, 2);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_build_attempts, 2);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_build_successes,
                    1);
  BOOST_CHECK_EQUAL(after_miss.runtime.current_linearization_build_failures, 1);
  BOOST_CHECK_EQUAL(static_cast<int>(after_miss.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kClean));

  CudaFullLmResult after_topology;
  CudaFullLmOptions topology_options = options_for(
      CudaFaultTriggerPhase::kAfterTopologyRefreshBeforePublish,
      CudaFaultInjection::kForceLinearizationFailure);
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, topology_options, &after_topology,
                                  &error));
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_logical_requests,
                    3);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_cache_lookups,
                    3);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_cache_hits, 1);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_cache_misses,
                    2);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_build_attempts,
                    2);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_build_successes,
                    1);
  BOOST_CHECK_EQUAL(after_topology.runtime.current_linearization_build_failures,
                    1);
  BOOST_CHECK_GT(after_topology.runtime.topology_refresh_count,
                 after_miss.runtime.topology_refresh_count);
  BOOST_CHECK_EQUAL(after_topology.runtime.final_internal_state_epoch, 0);
  BOOST_CHECK_EQUAL(static_cast<int>(after_topology.runtime.final_resource_health),
                    static_cast<int>(CudaResourceHealth::kTainted));
  BOOST_CHECK(after_topology.runtime.resource_cleanup_complete);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(after_topology.final_state),
                    initial_parameters);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(TimingAuditAndGenerationStateMachine) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 0;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions normal_options;
  normal_options.max_num_iterations = 0;
  normal_options.function_tolerance = 0.0;
  normal_options.gradient_tolerance = 0.0;
  normal_options.parameter_tolerance = 0.0;
  normal_options.instrumentation_mode = CudaInstrumentationMode::kEnabled;
  CudaFullLmResult normal;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, normal_options, &normal,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(normal.runtime.resource_generation_advance_events, 1);
  BOOST_CHECK_EQUAL(normal.runtime.resource_generation_advance_violations, 0);
  BOOST_CHECK(normal.runtime.resource_generation_advanced_by_this_solve);
  BOOST_CHECK(normal.runtime.solve_generation_nonzero);
  BOOST_CHECK(normal.runtime.solve_generation_consistent);
  BOOST_CHECK(normal.runtime.linearization_identity_consistent);
  BOOST_CHECK(normal.runtime.topology_context_generation_consistent);
  BOOST_CHECK_EQUAL(static_cast<int>(normal.runtime.timing_structure_v2.timing_status),
                    static_cast<int>(CudaTimingStatus::kComplete));
  BOOST_CHECK_EQUAL(
      normal.runtime.timing_structure_v2.event_attempts,
      normal.runtime.timing_structure_v2.event_completions +
          normal.runtime.timing_structure_v2.event_failures);
  BOOST_CHECK_EQUAL(
      normal.runtime.timing_structure_v2.intervals_started,
      normal.runtime.timing_structure_v2.intervals_completed +
          normal.runtime.timing_structure_v2.intervals_abandoned);
  BOOST_CHECK(!normal.runtime.diagnostic_structure_v2.has_primary);

  for (const CudaFailureSite site :
       {CudaFailureSite::kTimingEventAcquire,
        CudaFailureSite::kTimingEventRecord,
        CudaFailureSite::kTimingElapsedQuery}) {
    CudaFullLmOptions fault_options = normal_options;
    fault_options.max_num_iterations = 1;
    fault_options.fault_trigger.fault_kind =
        CudaFaultInjection::kForceTimingApiFailure;
    fault_options.fault_trigger.operation = static_cast<uint16_t>(site);
    CudaFullLmResult failed;
    error.clear();
    BOOST_CHECK(!RunCustomCudaSolve(snapshot, fault_options, &failed, &error));
    BOOST_CHECK_EQUAL(static_cast<int>(failed.error_classification),
                      static_cast<int>(CudaSolveErrorClass::kCudaTiming));
    BOOST_REQUIRE(failed.runtime.diagnostic_structure_v2.has_primary);
    BOOST_CHECK_EQUAL(
        static_cast<int>(failed.runtime.diagnostic_structure_v2.primary
                             .error_classification),
        static_cast<int>(CudaSolveErrorClass::kCudaTiming));
    BOOST_CHECK_EQUAL(static_cast<int>(failed.runtime.timing_structure_v2
                                           .timing_status),
                      static_cast<int>(CudaTimingStatus::kIncomplete));
    BOOST_CHECK_EQUAL(
        failed.runtime.timing_structure_v2.event_attempts,
        failed.runtime.timing_structure_v2.event_completions +
            failed.runtime.timing_structure_v2.event_failures);
    BOOST_CHECK_EQUAL(
        failed.runtime.timing_structure_v2.intervals_started,
        failed.runtime.timing_structure_v2.intervals_completed +
            failed.runtime.timing_structure_v2.intervals_abandoned);
    BOOST_CHECK(!CudaLegacyV1ComparisonAvailable(failed));
  }

  CudaFullLmOptions capacity_options = normal_options;
  capacity_options.audit_resource_registry_capacity_for_testing = 1;
  CudaFullLmResult capacity;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, capacity_options, &capacity,
                                  &error));
  BOOST_CHECK_EQUAL(static_cast<int>(capacity.error_classification),
                    static_cast<int>(CudaSolveErrorClass::kAuditCapacity));
  BOOST_CHECK_EQUAL(capacity.runtime.resource_generation_advance_events, 0);
  BOOST_CHECK(!capacity.runtime.audit_capacity_preflight_pass);
  BOOST_REQUIRE(capacity.runtime.diagnostic_structure_v2.has_primary);
  BOOST_CHECK_EQUAL(static_cast<int>(capacity.runtime.diagnostic_structure_v2
                                         .primary.failure_site),
                    static_cast<int>(CudaFailureSite::kAuditPreflightCapacity));
  BOOST_CHECK(!CudaLegacyV1ComparisonAvailable(capacity));

  CudaFullLmOptions double_options = normal_options;
  double_options.fault_trigger.fault_kind =
      CudaFaultInjection::kForceResourceDoubleAdvance;
  CudaFullLmResult double_advance;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, double_options, &double_advance,
                                  &error));
  BOOST_CHECK_EQUAL(double_advance.runtime.resource_generation_advance_events,
                    2);
  BOOST_CHECK_EQUAL(
      double_advance.runtime.resource_generation_advance_violations, 1);
  BOOST_CHECK(!double_advance.runtime.resource_generation_advanced_by_this_solve);
  BOOST_CHECK_EQUAL(static_cast<int>(double_advance.error_classification),
                    static_cast<int>(CudaSolveErrorClass::kResourceCleanup));
  BOOST_REQUIRE(double_advance.runtime.diagnostic_structure_v2.has_primary);
  BOOST_CHECK_EQUAL(
      static_cast<int>(double_advance.runtime.diagnostic_structure_v2.primary
                           .failure_site),
      static_cast<int>(CudaFailureSite::kResourceGenerationDoubleAdvance));
  BOOST_CHECK(!CudaLegacyV1ComparisonAvailable(double_advance));
  const uint64_t failed_generation = double_advance.runtime.resource_generation;
  CudaFullLmResult recovered;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, normal_options, &recovered,
                                           &error),
                        error);
  BOOST_CHECK_EQUAL(recovered.runtime.resource_generation,
                    failed_generation + 1);
  BOOST_CHECK_EQUAL(recovered.runtime.cross_solve_cache_hits, 0);
  BOOST_REQUIRE_MESSAGE(ResetCudaResourceStateForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(OneFactorizationFailureCanRecoverWithoutRejectedStep) {
  Snapshot snapshot = LayerBSnapshot();
  snapshot.metadata.max_num_iterations = 2;
  snapshot.metadata.max_consecutive_invalid_steps = 10;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  CudaFullLmOptions options;
  options.max_num_iterations = 2;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  options.fault_trigger.fault_kind =
      CudaFaultInjection::kForcePointFactorizationFailure;
  options.fault_trigger.logical_site =
      CudaFaultLogicalSite::kLayerCPointFactor;
  options.fault_trigger.trigger_phase = CudaFaultTriggerPhase::kLayerCOperation;
  options.fault_trigger.operation = 1;
  options.fault_trigger.target_state_epoch = 0;
  options.fault_trigger.occurrence_within_site = 1;
  CudaFullLmResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(snapshot, options, &result, &error),
                        error);
  BOOST_CHECK_EQUAL(result.factorization_failures, 1);
  BOOST_CHECK_EQUAL(result.invalid_steps, 1);
  BOOST_CHECK_EQUAL(result.rejected_steps, 0);
  BOOST_CHECK_GE(result.accepted_commits, 1);
  BOOST_REQUIRE(result.trace.size() >= 3);
  BOOST_CHECK(result.trace[1].invalid);
  BOOST_CHECK(!result.trace[1].accepted_decision);
  BOOST_CHECK(result.trace[2].accepted_commit_success);
}

CudaFullLmOptions PrecisionExperimentOptions(
    const CudaArithmeticPrecision precision) {
  CudaFullLmOptions options;
  options.arithmetic_precision = precision;
  options.device_context_mode = CudaDeviceContextMode::kDeviceControl;
  options.execution_profile = CudaExecutionProfile::kCompactControl;
  options.hot_kernel_mode = CudaHotKernelMode::kTransformed;
  options.layer_c.layer_b.hessian_assembly_backend =
      CudaHessianAssemblyBackend::kObservationSegmented;
  options.layer_c.layer_b.hessian_segment_size_for_testing = 64;
  options.layer_c.schur_contribution_backend =
      CudaSchurContributionBackend::kSegmentedTransformed;
  options.layer_c.schur_segment_size_for_testing = 64;
  options.layer_c.layer_b.cost_reduction_threads = 128;
  options.max_num_iterations = 1;
  options.function_tolerance = 0.0;
  options.gradient_tolerance = 0.0;
  options.parameter_tolerance = 0.0;
  return options;
}

BOOST_AUTO_TEST_CASE(ArithmeticPrecisionSelectorAndBufferIsolation) {
  Snapshot snapshot = PrecisionExperimentSnapshot();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  std::string error;

  CudaFullLmResult fp64_first;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(
          snapshot, PrecisionExperimentOptions(CudaArithmeticPrecision::kFp64),
          &fp64_first, &error), error);
  const auto& fp64_runtime = fp64_first.runtime.persistent_device;
  BOOST_CHECK_EQUAL(fp64_runtime.arithmetic_precision_effective, "fp64");
  BOOST_CHECK_EQUAL(fp64_runtime.float_buffer_allocation_calls, 0);
  BOOST_CHECK_EQUAL(fp64_runtime.float_static_upload_calls, 0);
  BOOST_CHECK_EQUAL(fp64_runtime.float_arena_reserved_bytes, 0);
  BOOST_CHECK_EQUAL(fp64_runtime.spotrf_calls, 0);
  BOOST_CHECK_EQUAL(fp64_runtime.spotrs_calls, 0);
  BOOST_CHECK_GT(fp64_runtime.dpotrf_calls, 0);
  BOOST_CHECK_GT(fp64_runtime.dpotrs_calls, 0);

  CudaFullLmResult fp32;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
      snapshot, PrecisionExperimentOptions(CudaArithmeticPrecision::kFp32Core),
      &fp32, &error), error);
  const auto& fp32_runtime = fp32.runtime.persistent_device;
  BOOST_CHECK_EQUAL(fp32_runtime.arithmetic_precision_effective, "fp32_core");
  BOOST_CHECK_EQUAL(fp32_runtime.state_storage_precision, "fp64_device_state");
  BOOST_CHECK_EQUAL(fp32_runtime.residual_jacobian_precision, "fp32");
  BOOST_CHECK_EQUAL(fp32_runtime.factorization_routine,
                    "cusolverDnSpotrf/Spotrs");
  BOOST_CHECK_GT(fp32_runtime.float_buffer_allocation_calls, 0);
  BOOST_CHECK_GT(fp32_runtime.float_static_upload_calls, 0);
  BOOST_CHECK_GT(fp32_runtime.spotrf_calls, 0);
  BOOST_CHECK_GT(fp32_runtime.spotrs_calls, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.dpotrf_calls, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.dpotrs_calls, 0);
  BOOST_CHECK_GT(fp32_runtime.fp64_cost_calls, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.float_state_update_calls, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.state_quantization_calls, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.precision_mirror_cross_hits, 0);
  BOOST_CHECK_EQUAL(fp32_runtime.fixed_external_write_attempts, 0);

  CudaFullLmResult quantized;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
      snapshot,
      PrecisionExperimentOptions(
          CudaArithmeticPrecision::kFp32StateQuantizedMixed),
      &quantized, &error), error);
  BOOST_CHECK_EQUAL(quantized.runtime.persistent_device
                        .arithmetic_precision_effective,
                    "fp32_state_quantized");
  BOOST_CHECK_GT(
      quantized.runtime.persistent_device.state_quantization_calls, 0);
  BOOST_CHECK_EQUAL(
      quantized.runtime.persistent_device.precision_mirror_cross_hits, 0);
  for (size_t i = 0; i < snapshot.images.size(); ++i) {
    if (snapshot.images[i].pose_constant) {
      BOOST_CHECK_EQUAL_COLLECTIONS(
          snapshot.images[i].qvec.begin(), snapshot.images[i].qvec.end(),
          quantized.final_state.images[i].qvec.begin(),
          quantized.final_state.images[i].qvec.end());
      BOOST_CHECK_EQUAL_COLLECTIONS(
          snapshot.images[i].tvec.begin(), snapshot.images[i].tvec.end(),
          quantized.final_state.images[i].tvec.begin(),
          quantized.final_state.images[i].tvec.end());
    }
    for (size_t component = 0; component < 3; ++component) {
      if ((snapshot.images[i].constant_tvec_mask & (1u << component)) != 0) {
        BOOST_CHECK_EQUAL(snapshot.images[i].tvec[component],
                          quantized.final_state.images[i].tvec[component]);
      }
    }
  }
  for (size_t i = 0; i < snapshot.points.size(); ++i) {
    if (snapshot.points[i].constant)
      BOOST_CHECK_EQUAL_COLLECTIONS(
          snapshot.points[i].xyz.begin(), snapshot.points[i].xyz.end(),
          quantized.final_state.points[i].xyz.begin(),
          quantized.final_state.points[i].xyz.end());
  }
  BOOST_CHECK_EQUAL_COLLECTIONS(
      snapshot.cameras.front().params.begin(),
      snapshot.cameras.front().params.end(),
      quantized.final_state.cameras.front().params.begin(),
      quantized.final_state.cameras.front().params.end());

  const std::string entry_parameters =
      CudaFinalParametersBitwiseSha256(snapshot);
  CudaFullLmOptions rollback_options = PrecisionExperimentOptions(
      CudaArithmeticPrecision::kFp32StateQuantizedMixed);
  rollback_options.layer_c.fault_injection =
      CudaFaultInjection::kForceNonfiniteTrial;
  CudaFullLmResult rollback;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, rollback_options, &rollback,
                                  &error));
  BOOST_CHECK_EQUAL(error, "FP32 injected post-trial failure");
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(snapshot),
                    entry_parameters);
  BOOST_CHECK_EQUAL(
      rollback.runtime.persistent_device.precision_mirror_cross_hits, 0);
  for (size_t i = 0; i < snapshot.images.size(); ++i) {
    if (!snapshot.images[i].pose_constant) continue;
    BOOST_CHECK_EQUAL_COLLECTIONS(
        snapshot.images[i].qvec.begin(), snapshot.images[i].qvec.end(),
        rollback.final_state.images[i].qvec.begin(),
        rollback.final_state.images[i].qvec.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(
        snapshot.images[i].tvec.begin(), snapshot.images[i].tvec.end(),
        rollback.final_state.images[i].tvec.begin(),
        rollback.final_state.images[i].tvec.end());
  }

  CudaFullLmResult fp64_second;
  error.clear();
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(
          snapshot, PrecisionExperimentOptions(CudaArithmeticPrecision::kFp64),
          &fp64_second, &error), error);
  BOOST_CHECK_EQUAL(fp64_second.runtime.persistent_device
                        .arithmetic_precision_effective,
                    "fp64");
  BOOST_CHECK_EQUAL(
      fp64_second.runtime.persistent_device.precision_mirror_cross_hits, 0);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(fp64_first.final_state),
                    CudaFinalParametersBitwiseSha256(fp64_second.final_state));
}

BOOST_AUTO_TEST_CASE(Fp32PrecisionComponentUsesTrueFloatSolver) {
  CudaPrecisionComponentResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(RunCudaPrecisionComponentForTesting(
      PrecisionExperimentSnapshot(), 1.0, 3, &result, &error), error);
  BOOST_CHECK(result.success);
  BOOST_CHECK_EQUAL(result.fp32_factorization_routine,
                    "cusolverDnSpotrf/Spotrs");
  BOOST_CHECK_GT(result.spotrf_calls, 0);
  BOOST_CHECK_GT(result.spotrs_calls, 0);
  BOOST_CHECK_GT(result.dpotrf_calls, 0);
  BOOST_CHECK_GT(result.dpotrs_calls, 0);
  BOOST_CHECK_EQUAL(result.residual_jacobian.finite_count,
                    result.residual_jacobian.count);
  BOOST_CHECK_EQUAL(result.schur.finite_count, result.schur.count);
  BOOST_CHECK_EQUAL(result.trial_state.finite_count, result.trial_state.count);
  BOOST_CHECK_SMALL(result.fp32_trial_update_consistency.max_abs_error, 1e-15);

  Snapshot soft_l1 = PrecisionExperimentSnapshot();
  soft_l1.metadata.loss_function = "SOFT_L1";
  CudaPrecisionComponentResult soft_l1_result;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCudaPrecisionComponentForTesting(
      soft_l1, 1.0, 1, &soft_l1_result, &error), error);
  BOOST_CHECK_EQUAL(soft_l1_result.robust_scale.finite_count,
                    soft_l1_result.robust_scale.count);
  BOOST_CHECK_GT(soft_l1_result.robust_scale.count, 0);
}

BOOST_AUTO_TEST_CASE(Fp32MixedUsesMainControllerAndTypedComponent) {
  Snapshot snapshot = PrecisionExperimentSnapshot();
  snapshot.metadata.max_num_iterations = 1;
  snapshot.metadata.function_tolerance = 0.0;
  snapshot.metadata.gradient_tolerance = 0.0;
  snapshot.metadata.parameter_tolerance = 0.0;
  std::string error;
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);

  CudaFullLmResult fp64_before;
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
                            snapshot,
                            PrecisionExperimentOptions(
                                CudaArithmeticPrecision::kFp64),
                            &fp64_before, &error),
                        error);
  const std::string fp64_before_hash =
      CudaFinalParametersBitwiseSha256(fp64_before.final_state);

  CudaFullLmOptions mixed_options = PrecisionExperimentOptions(
      CudaArithmeticPrecision::kFp32MixedStable);
  mixed_options.audit_profile = CudaAuditProfile::kProduction;
  mixed_options.performance_mode = true;
  mixed_options.capture_state_trace = false;
  mixed_options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
  mixed_options.current_linearization_cache_mode =
      CudaCurrentLinearizationCacheMode::kEnabled;
  CudaFullLmResult mixed;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, mixed_options, &mixed, &error), error);
  const auto& audit = mixed.runtime.persistent_device;
  BOOST_CHECK_EQUAL(mixed.runtime.audit_profile_requested, "production");
  BOOST_CHECK_EQUAL(mixed.runtime.audit_profile_effective, "production");
  BOOST_CHECK(!mixed.runtime.capture_state_trace_effective);
  BOOST_CHECK(mixed.runtime.production_audit_invariants_checked);
  BOOST_CHECK(mixed.runtime.production_audit_invariants_pass);
  BOOST_CHECK_EQUAL(mixed.runtime.production_audit_violation_count, 0);
  BOOST_CHECK(mixed.runtime.first_production_audit_violation.empty());
  BOOST_CHECK_EQUAL(audit.arithmetic_precision_effective, "fp32_mixed");
  BOOST_CHECK_EQUAL(audit.state_storage_precision, "fp64");
  BOOST_CHECK_EQUAL(audit.residual_jacobian_precision,
                    "fp64_geometry_fp32_records");
  BOOST_CHECK_EQUAL(audit.residual_jacobian_operand_source,
                    "fp64_geometry_cast");
  BOOST_CHECK_EQUAL(audit.hessian_schur_precision,
                    "fp32_hessian_fp32_products_fp64_dense_schur");
  BOOST_CHECK_EQUAL(audit.hessian_precision, "fp32");
  BOOST_CHECK_EQUAL(audit.gradient_operand_precision, "fp64");
  BOOST_CHECK_EQUAL(audit.gradient_accumulation_precision, "fp64");
  BOOST_CHECK_EQUAL(audit.point_inverse_precision, "fp64");
  BOOST_CHECK_EQUAL(audit.schur_contribution_precision,
                    "fp32_products_fp32_partials_fp64_merge");
  BOOST_CHECK_EQUAL(audit.dense_factorization_precision, "fp64");
  BOOST_CHECK_EQUAL(audit.factorization_routine,
                    "cusolverDnDpotrf/Dpotrs");
  BOOST_CHECK_GT(audit.runtime_pool_acquire_calls, 0);
  BOOST_CHECK_GT(audit.dpotrf_calls, 0);
  BOOST_CHECK_GT(audit.dpotrs_calls, 0);
  BOOST_CHECK_EQUAL(audit.spotrf_calls, 0);
  BOOST_CHECK_EQUAL(audit.spotrs_calls, 0);
  BOOST_CHECK_GT(audit.mixed_fp64_gradient_calls, 0);
  BOOST_CHECK_GT(audit.mixed_fp64_gradient_from_fp64_records_calls, 0);
  BOOST_CHECK_EQUAL(audit.mixed_fp64_gradient_from_fp32_records_calls, 0);
  BOOST_CHECK_GT(audit.mixed_fp64_point_inverse_calls, 0);
  BOOST_CHECK_GT(audit.mixed_fp64_rhs_calls, 0);
  BOOST_CHECK_GT(audit.mixed_fp64_back_substitution_calls, 0);
  BOOST_CHECK_EQUAL(audit.mixed_serial_full_scan_kernel_count, 0);
  BOOST_CHECK_EQUAL(audit.post_initialize_allocation_calls, 0);
  BOOST_CHECK_GT(audit.mixed_native_fp32_point_inverse_cast_calls, 0);
  BOOST_CHECK_GT(audit.mixed_native_fp32_transform_calls, 0);
  BOOST_CHECK_GT(audit.mixed_native_fp32_schur_calls, 0);
  BOOST_CHECK_EQUAL(audit.mixed_native_fp32_schur_merge_calls, 0);
  BOOST_CHECK_GT(audit.direct_pair_count + audit.segmented_pair_count, 0);
  if (audit.segmented_pair_count != 0) {
    BOOST_CHECK_GT(audit.mixed_native_fp32_schur_partial_calls, 0);
    BOOST_CHECK_GT(audit.schur_merge_kernel_launches, 0);
  } else {
    BOOST_CHECK_EQUAL(audit.mixed_native_fp32_schur_partial_calls, 0);
    BOOST_CHECK_EQUAL(audit.schur_merge_kernel_launches, 0);
  }
  BOOST_CHECK_EQUAL(audit.mixed_native_fp32_dense_conversion_calls, 0);
  BOOST_CHECK_EQUAL(audit.mixed_native_fp32_dense_conversion_source_bytes, 0);
  BOOST_CHECK_EQUAL(
      audit.mixed_native_fp32_dense_conversion_destination_bytes, 0);
  BOOST_CHECK_EQUAL(audit.mixed_float_to_double_schur_calls, 0);
  BOOST_CHECK_EQUAL(audit.mixed_float_to_double_schur_bytes, 0);
  BOOST_CHECK_EQUAL(audit.float_buffer_allocation_calls, 11);
  BOOST_CHECK_GT(audit.mixed_layer_a_cast_calls, 0);
  BOOST_CHECK_GT(audit.mixed_layer_a_cast_bytes, 0);
  BOOST_CHECK_EQUAL(audit.mixed_fp64_pose_damping_calls,
                    audit.schur_contribution_calls);
  BOOST_CHECK_EQUAL(audit.mixed_fp64_schur_contribution_calls,
                    audit.schur_contribution_calls);
  BOOST_CHECK_EQUAL(audit.mixed_double_edge_materialization_calls, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_b_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_state_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_schur_bytes, 0);
  BOOST_CHECK_EQUAL(audit.audit_mirror_delta_bytes, 0);
  BOOST_CHECK_EQUAL(audit.host_diagnostics_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.mixed_full_array_d2h_bytes, 0);
  BOOST_CHECK_EQUAL(audit.mixed_schur_math_effective,
                    "fp32_products_fp64_dense_accum_f64_solve");

  CudaMixedStableComponentResult component;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCudaMixedStableComponentForTesting(
                            snapshot, 1.0, 2, &component, &error),
                        error);
  BOOST_CHECK(component.success);
  BOOST_CHECK_EQUAL(component.residual_jacobian.finite_count,
                    component.residual_jacobian.count);
  BOOST_CHECK_EQUAL(component.pose_hessian.finite_count,
                    component.pose_hessian.count);
  BOOST_CHECK_EQUAL(component.pose_gradient.finite_count,
                    component.pose_gradient.count);
  BOOST_CHECK_EQUAL(component.schur.finite_count, component.schur.count);
  BOOST_CHECK_EQUAL(component.rhs.finite_count, component.rhs.count);
  BOOST_CHECK_GT(component.mixed_dpotrf_calls, 0);
  BOOST_CHECK_GT(component.mixed_dpotrs_calls, 0);
  BOOST_CHECK_EQUAL(component.mixed_spotrf_calls, 0);
  BOOST_CHECK_EQUAL(component.mixed_spotrs_calls, 0);
  BOOST_CHECK_EQUAL(component.mixed_serial_full_scan_kernel_count, 0);
  BOOST_CHECK_EQUAL(component.mixed_post_initialize_allocation_calls, 0);
  BOOST_CHECK_EQUAL(component.mixed_schur_math_effective,
                    "fp32_products_fp64_dense_accum_f64_solve");
  BOOST_CHECK_LE(component.pose_gradient.normalized_max_error, 1e-8);
  BOOST_CHECK_LE(component.pose_gradient.normalized_frobenius_error, 1e-8);
  BOOST_CHECK_LE(component.point_gradient.normalized_max_error, 1e-8);
  BOOST_CHECK_LE(component.point_gradient.normalized_frobenius_error, 1e-8);
  BOOST_CHECK_LE(component.mixed_schur_solve_backward_error, 1e-12);

  CudaFullLmOptions reject_options = mixed_options;
  reject_options.min_relative_decrease = 2.0;
  CudaFullLmResult rejected;
  error.clear();
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, reject_options, &rejected, &error), error);
  BOOST_CHECK_GE(rejected.rejected_steps, 1);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(rejected.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));

  CudaFullLmOptions rollback_options = mixed_options;
  rollback_options.audit_profile = CudaAuditProfile::kCompatibilityDefault;
  rollback_options.fault_injection =
      CudaFaultInjection::kForceNonfiniteTrial;
  CudaFullLmResult rollback;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, rollback_options, &rollback,
                                  &error));
  BOOST_CHECK(!error.empty());
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(rollback.final_state),
                    CudaFinalParametersBitwiseSha256(snapshot));

  CudaFullLmResult fp64_after;
  error.clear();
  BOOST_REQUIRE_MESSAGE(RunCustomCudaSolve(
      snapshot, PrecisionExperimentOptions(CudaArithmeticPrecision::kFp64),
      &fp64_after, &error), error);
  BOOST_CHECK_EQUAL(fp64_after.runtime.persistent_device
                        .arithmetic_precision_effective,
                    "fp64");
  BOOST_CHECK_EQUAL(
      fp64_after.runtime.persistent_device.float_buffer_allocation_calls, 0);
  BOOST_CHECK_EQUAL(
      fp64_after.runtime.persistent_device.precision_mirror_cross_hits, 0);
  BOOST_CHECK_EQUAL(
      fp64_after.runtime.persistent_device.mixed_native_fp32_schur_calls, 0);
  BOOST_CHECK_EQUAL(
      fp64_after.runtime.persistent_device.mixed_fp64_schur_contribution_calls,
      0);
  BOOST_CHECK_EQUAL(CudaFinalParametersBitwiseSha256(fp64_after.final_state),
                    fp64_before_hash);
  BOOST_REQUIRE_MESSAGE(ResetCudaRuntimePoolForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(TypedAuditProfilesAreExplicitAndFailClosed) {
  Snapshot snapshot = PrecisionExperimentSnapshot();
  snapshot.metadata.max_num_iterations = 0;
  std::string error;

  CudaFullLmOptions forensic_options =
      PrecisionExperimentOptions(CudaArithmeticPrecision::kFp64);
  forensic_options.max_num_iterations = 0;
  forensic_options.audit_profile = CudaAuditProfile::kForensic;
  forensic_options.capture_state_trace = false;
  forensic_options.instrumentation_mode = CudaInstrumentationMode::kDisabled;
  CudaFullLmResult forensic;
  BOOST_REQUIRE_MESSAGE(
      RunCustomCudaSolve(snapshot, forensic_options, &forensic, &error), error);
  BOOST_CHECK_EQUAL(forensic.runtime.audit_profile_requested, "forensic");
  BOOST_CHECK_EQUAL(forensic.runtime.audit_profile_effective, "forensic");
  BOOST_CHECK(forensic.runtime.capture_state_trace_effective);
  BOOST_CHECK(forensic.runtime.instrumentation_effective);
  BOOST_CHECK(!forensic.runtime.initial_state_hash.empty());
  BOOST_CHECK(!forensic.runtime.production_audit_invariants_checked);
  BOOST_CHECK(!forensic.accepted_state_trace.empty());

  CudaFullLmOptions legacy_options =
      PrecisionExperimentOptions(CudaArithmeticPrecision::kFp32Core);
  legacy_options.audit_profile = CudaAuditProfile::kProduction;
  CudaFullLmResult legacy;
  error.clear();
  BOOST_CHECK(!RunCustomCudaSolve(snapshot, legacy_options, &legacy, &error));
  BOOST_CHECK_EQUAL(legacy.runtime.audit_profile_requested, "production");
  BOOST_CHECK_EQUAL(legacy.runtime.audit_profile_effective, "unsupported");
  BOOST_CHECK_EQUAL(legacy.termination_reason,
                    "audit_unmanaged_precision_path");
  BOOST_CHECK_NE(error.find("legacy FP32 experiment paths are audit-unmanaged"),
                 std::string::npos);
}

BOOST_AUTO_TEST_CASE(Fp32MixedRhsCancellationStartsInFp64) {
  CudaMixedGradientCancellationTestResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaMixedGradientCancellationForTesting(&result, &error), error);
  BOOST_CHECK_EQUAL(result.fp32_record_path_calls, 1);
  BOOST_CHECK_EQUAL(result.fp64_record_path_calls, 1);
  BOOST_CHECK_EQUAL(result.fp32_record_gradient, 0.0);
  BOOST_CHECK_EQUAL(result.fp64_record_gradient, 1.0);

  double empty_backward_error = std::numeric_limits<double>::quiet_NaN();
  error.clear();
  BOOST_REQUIRE_MESSAGE(
      RunCudaSchurSolveBackwardErrorZeroDimensionForTesting(
          &empty_backward_error, &error),
      error);
  BOOST_CHECK(std::isfinite(empty_backward_error));
  BOOST_CHECK_EQUAL(empty_backward_error, 0.0);
}

BOOST_AUTO_TEST_CASE(Fp32MixedPoseDampingIsAddedAfterDenseConversion) {
  CudaMixedPoseDampingTestResult result;
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaMixedPoseDampingForTesting(&result, &error), error);
  BOOST_CHECK_EQUAL(result.legacy_float_damped_diagonal,
                    result.undamped_float_diagonal);
  BOOST_CHECK_GT(result.fp64_damped_diagonal,
                 static_cast<double>(result.undamped_float_diagonal));
  BOOST_CHECK_CLOSE_FRACTION(result.fp64_damped_diagonal,
                             result.expected_fp64_damped_diagonal, 1e-15);
  BOOST_CHECK_GT(result.lambda * result.damping, 0.0);
}

BOOST_AUTO_TEST_CASE(Fp32MixedLayerACastSupportsCompactAndFullRecords) {
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      RunCudaMixedLayerACastCompatibilityForTesting(&error), error);
}

BOOST_AUTO_TEST_CASE(PureFp32SelectorIsExplicitlyNotImplemented) {
  CudaFullLmOptions options = PrecisionExperimentOptions(
      CudaArithmeticPrecision::kPureFp32Experimental);
  CudaFullLmResult result;
  std::string error;
  BOOST_CHECK(!RunCustomCudaSolve(PrecisionExperimentSnapshot(), options,
                                  &result, &error));
  BOOST_CHECK(error.find("pure FP32 arithmetic is not implemented") !=
              std::string::npos);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device
                        .arithmetic_precision_effective,
                    "invalid");
}

BOOST_AUTO_TEST_CASE(Fp32LegacyExactQuantizedZeroFailsBeforeFactorization) {
  Snapshot snapshot = PrecisionExperimentSnapshot();
  BOOST_REQUIRE(!snapshot.lidar.empty());
  const auto point = std::find_if(
      snapshot.points.begin(), snapshot.points.end(),
      [&snapshot](const PointSnapshot& value) {
        return value.point3D_id == snapshot.lidar.front().point3D_id;
      });
  BOOST_REQUIRE(point != snapshot.points.end());
  snapshot.metadata.lidar_residual_mode = "legacy_exact";
  snapshot.lidar.front().plane =
      {{1.0, 0.0, 0.0, -point->xyz[0]}};
  CudaFullLmResult result;
  std::string error;
  BOOST_CHECK(!RunCustomCudaSolve(
      snapshot, PrecisionExperimentOptions(CudaArithmeticPrecision::kFp32Core),
      &result, &error));
  BOOST_CHECK(error.find("FP32 non-finite residual/Jacobian") !=
              std::string::npos);
  BOOST_CHECK(error.find("lidar_failures=1") != std::string::npos);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.spotrf_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.spotrs_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.dpotrf_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.dpotrs_calls, 0);
  BOOST_CHECK_EQUAL(result.runtime.persistent_device.precision_mirror_cross_hits,
                    0);
}

}  // namespace gpu_ba
}  // namespace colmap
