#define TEST_NAME "gpu_ba/fixed_linearization"
#include "util/testing.h"

#include <algorithm>
#include <limits>
#include <tuple>

#include "base/camera_models.h"
#include "gpu_ba/fixed_linearization.h"

namespace colmap {
namespace gpu_ba {
namespace {

Snapshot SyntheticFixedProblem() {
  Snapshot snapshot;
  snapshot.metadata.loss_function = "trivial";
  snapshot.metadata.lidar_residual_mode = "legacy_exact";

  CameraSnapshot camera;
  camera.camera_id = 7;
  camera.model_id = OpenCVCameraModel::kModelId;
  camera.constant = true;
  camera.params = {500.0, 505.0, 320.0, 240.0,
                   -0.02, 0.003, 0.001, -0.0004};
  snapshot.cameras.push_back(camera);

  ImageSnapshot variable_image;
  variable_image.image_id = 11;
  variable_image.camera_id = camera.camera_id;
  variable_image.selected = true;
  variable_image.pose_constant = false;
  variable_image.has_pose_parameter_blocks = true;
  variable_image.constant_tvec_mask = 1u << 0;
  variable_image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  variable_image.tvec = {{0.1, -0.2, 0.3}};
  snapshot.images.push_back(variable_image);

  ImageSnapshot constant_image;
  constant_image.image_id = 12;
  constant_image.camera_id = camera.camera_id;
  constant_image.selected = true;
  constant_image.pose_constant = true;
  constant_image.has_pose_parameter_blocks = false;
  constant_image.qvec = {{0.9999500004166653, 0.0, 0.009999833334166664,
                          0.0}};
  constant_image.tvec = {{0.4, 0.0, 0.2}};
  snapshot.images.push_back(constant_image);

  PointSnapshot variable_point;
  variable_point.point3D_id = 101;
  variable_point.constant = false;
  variable_point.xyz = {{0.2, -0.1, 3.0}};
  snapshot.points.push_back(variable_point);
  PointSnapshot constant_point;
  constant_point.point3D_id = 102;
  constant_point.constant = true;
  constant_point.xyz = {{-0.3, 0.4, 4.0}};
  snapshot.points.push_back(constant_point);

  ObservationSnapshot observation0;
  observation0.source_index = 0;
  observation0.image_id = variable_image.image_id;
  observation0.point2D_idx = 4;
  observation0.point3D_id = variable_point.point3D_id;
  observation0.xy = {{370.0, 180.0}};
  snapshot.observations.push_back(observation0);
  ObservationSnapshot observation1 = observation0;
  observation1.source_index = 1;
  observation1.point2D_idx = 5;
  observation1.point3D_id = constant_point.point3D_id;
  observation1.xy = {{290.0, 270.0}};
  snapshot.observations.push_back(observation1);
  ObservationSnapshot observation2 = observation0;
  observation2.source_index = 2;
  observation2.image_id = constant_image.image_id;
  observation2.point2D_idx = 2;
  observation2.xy = {{410.0, 220.0}};
  observation2.pose_constant = true;
  snapshot.observations.push_back(observation2);

  LidarSnapshot lidar;
  lidar.source_index = 3;
  lidar.point3D_id = variable_point.point3D_id;
  lidar.weight = 10.0;
  lidar.plane = {{1.0, 0.0, 0.0, -0.25}};
  snapshot.lidar.push_back(lidar);

  for (const ObservationSnapshot& observation : snapshot.observations) {
    OrderEntrySnapshot entry;
    entry.source_index = observation.source_index;
    entry.residual_kind = ResidualKind::kVisual;
    entry.image_id = observation.image_id;
    entry.point2D_idx = observation.point2D_idx;
    entry.point3D_id = observation.point3D_id;
    snapshot.source_insertion_order.push_back(entry);
    snapshot.canonical_order.push_back(entry);
  }
  OrderEntrySnapshot lidar_entry;
  lidar_entry.source_index = lidar.source_index;
  lidar_entry.residual_kind = ResidualKind::kLidar;
  lidar_entry.image_id = std::numeric_limits<uint32_t>::max();
  lidar_entry.point2D_idx = std::numeric_limits<uint32_t>::max();
  lidar_entry.point3D_id = lidar.point3D_id;
  snapshot.source_insertion_order.push_back(lidar_entry);
  snapshot.canonical_order.push_back(lidar_entry);
  std::sort(snapshot.canonical_order.begin(), snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                              lhs.point3D_id, lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                              rhs.point3D_id, rhs.source_index);
            });

  ParameterBlockSnapshot quaternion;
  quaternion.source_index = 0;
  quaternion.kind = ParameterKind::kQuaternion;
  quaternion.entity_id = variable_image.image_id;
  quaternion.ambient_size = 4;
  quaternion.tangent_size = 3;
  quaternion.constant = false;
  snapshot.parameter_blocks_source_order.push_back(quaternion);
  ParameterBlockSnapshot translation;
  translation.source_index = 1;
  translation.kind = ParameterKind::kTranslation;
  translation.entity_id = variable_image.image_id;
  translation.ambient_size = 3;
  translation.tangent_size = 2;
  translation.constant = false;
  snapshot.parameter_blocks_source_order.push_back(translation);
  ParameterBlockSnapshot point;
  point.source_index = 2;
  point.kind = ParameterKind::kPoint3D;
  point.entity_id = variable_point.point3D_id;
  point.ambient_size = 3;
  point.tangent_size = 3;
  point.constant = false;
  snapshot.parameter_blocks_source_order.push_back(point);
  return snapshot;
}

}  // namespace

BOOST_AUTO_TEST_CASE(CanonicalSubsetAndConstantBlocks) {
  const Snapshot snapshot = SyntheticFixedProblem();
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_REQUIRE(RunFixedLinearizationComparison(snapshot, options, &result,
                                                &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.variable_pose_blocks, 1);
  BOOST_CHECK_EQUAL(result.pose_tangent_dimension, 5);
  BOOST_CHECK_EQUAL(result.variable_point_blocks, 1);
  BOOST_CHECK_EQUAL(result.point_tangent_dimension, 3);
  BOOST_CHECK_EQUAL(result.pose_point_edges, 1);
  BOOST_CHECK_EQUAL(result.constant_pose_residuals, 1);
  BOOST_CHECK_EQUAL(result.constant_point_residuals, 1);
  BOOST_CHECK_EQUAL(result.fully_constant_residuals, 0);
  BOOST_CHECK(result.reference_backward_error <= 1e-10);
  BOOST_CHECK(result.candidate_backward_error <= 1e-10);
  BOOST_CHECK_EQUAL(result.reference_accept, result.candidate_accept);
  for (const ErrorSummary& metric : result.metrics) {
    BOOST_CHECK_MESSAGE(metric.pass, metric.name);
  }
  const std::string json = FixedLinearizationJson(result, options);
  BOOST_CHECK(json.find("shared_frozen_reference_residual\": true") !=
              std::string::npos);
  BOOST_CHECK(
      json.find("shared_frozen_reference_gradient_for_schur\": true") !=
      std::string::npos);
}

BOOST_AUTO_TEST_CASE(VariableIntrinsicsAreExplicitlyUnsupported) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.cameras.front().constant = false;
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_CHECK(!RunFixedLinearizationComparison(snapshot, options, &result,
                                               &error));
  BOOST_CHECK(error.find("variable intrinsics") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(NonLegacyLidarIsNotStrictParity) {
  Snapshot snapshot = SyntheticFixedProblem();
  snapshot.metadata.lidar_residual_mode = "legacy_guarded";
  FixedLinearizationOptions options;
  FixedLinearizationResult result;
  std::string error;
  BOOST_CHECK(!RunFixedLinearizationComparison(snapshot, options, &result,
                                               &error));
  BOOST_CHECK(error.find("legacy_exact") != std::string::npos);
}

}  // namespace gpu_ba
}  // namespace colmap
