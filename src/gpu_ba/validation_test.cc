#define TEST_NAME "gpu_ba/validation"
#include "util/testing.h"

#include "base/camera_models.h"
#include "gpu_ba/validation.h"

namespace colmap {
namespace gpu_ba {
namespace {

Snapshot SyntheticSnapshot(const std::string& lidar_mode) {
  Snapshot snapshot;
  snapshot.metadata.loss_function = "trivial";
  snapshot.metadata.lidar_residual_mode = lidar_mode;

  CameraSnapshot camera;
  camera.camera_id = 7;
  camera.model_id = OpenCVCameraModel::kModelId;
  camera.params = {500.0, 505.0, 320.0, 240.0,
                   -0.02, 0.003, 0.001, -0.0004};
  snapshot.cameras.push_back(camera);

  ImageSnapshot image;
  image.image_id = 11;
  image.camera_id = camera.camera_id;
  image.selected = true;
  image.pose_constant = false;
  image.has_pose_parameter_blocks = true;
  image.qvec = {{1.0, 0.0, 0.0, 0.0}};
  image.tvec = {{0.1, -0.2, 0.3}};
  snapshot.images.push_back(image);

  PointSnapshot visual_point;
  visual_point.point3D_id = 101;
  visual_point.xyz = {{0.2, -0.1, 3.0}};
  snapshot.points.push_back(visual_point);
  PointSnapshot zero_lidar_point;
  zero_lidar_point.point3D_id = 102;
  zero_lidar_point.xyz = {{1.0, 2.0, 3.0}};
  snapshot.points.push_back(zero_lidar_point);
  PointSnapshot near_zero_lidar_point;
  near_zero_lidar_point.point3D_id = 103;
  near_zero_lidar_point.xyz = {{1.0 + 5e-13, 2.0, 3.0}};
  snapshot.points.push_back(near_zero_lidar_point);

  ObservationSnapshot observation;
  observation.source_index = 0;
  observation.image_id = image.image_id;
  observation.point2D_idx = 9;
  observation.point3D_id = visual_point.point3D_id;
  observation.xy = {{360.0, 190.0}};
  snapshot.observations.push_back(observation);

  LidarSnapshot lidar;
  lidar.source_index = 1;
  lidar.point3D_id = zero_lidar_point.point3D_id;
  lidar.weight = 10.0;
  lidar.plane = {{1.0, 0.0, 0.0, -1.0}};
  snapshot.lidar.push_back(lidar);
  LidarSnapshot near_zero_lidar = lidar;
  near_zero_lidar.source_index = 2;
  near_zero_lidar.point3D_id = near_zero_lidar_point.point3D_id;
  snapshot.lidar.push_back(near_zero_lidar);

  OrderEntrySnapshot visual_order;
  visual_order.source_index = observation.source_index;
  visual_order.residual_kind = ResidualKind::kVisual;
  visual_order.image_id = observation.image_id;
  visual_order.point2D_idx = observation.point2D_idx;
  visual_order.point3D_id = observation.point3D_id;
  snapshot.canonical_order.push_back(visual_order);
  OrderEntrySnapshot lidar_order;
  lidar_order.source_index = lidar.source_index;
  lidar_order.residual_kind = ResidualKind::kLidar;
  lidar_order.point3D_id = lidar.point3D_id;
  snapshot.canonical_order.push_back(lidar_order);
  OrderEntrySnapshot near_zero_lidar_order = lidar_order;
  near_zero_lidar_order.source_index = near_zero_lidar.source_index;
  near_zero_lidar_order.point3D_id = near_zero_lidar.point3D_id;
  snapshot.canonical_order.push_back(near_zero_lidar_order);
  return snapshot;
}

}  // namespace

BOOST_AUTO_TEST_CASE(LegacyExactReportsUndefinedZeroDerivative) {
  const Snapshot snapshot = SyntheticSnapshot("legacy_exact");
  LinearizationValidationOptions options;
  LinearizationValidationResult result;
  std::string error;
  BOOST_REQUIRE(ValidateResidualsAndJacobians(snapshot, options, &result,
                                              &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.lidar_exact_zero_point_ids.size(), 1);
  BOOST_CHECK_EQUAL(result.lidar_near_zero_point_ids.size(), 2);
  BOOST_CHECK_EQUAL(result.lidar_exact_zero_point_ids.front(), 102);
  BOOST_CHECK_EQUAL(result.lidar_classification_mismatches, 0);
  BOOST_CHECK_EQUAL(result.lidar_guarded_count, 0);
}

BOOST_AUTO_TEST_CASE(GuardedDoesNotClaimSourceJacobianParityAtZero) {
  const Snapshot snapshot = SyntheticSnapshot("legacy_guarded");
  LinearizationValidationOptions options;
  LinearizationValidationResult result;
  std::string error;
  BOOST_REQUIRE(ValidateResidualsAndJacobians(snapshot, options, &result,
                                              &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.lidar_guarded_count, 2);
  BOOST_CHECK_EQUAL(result.lidar_classification_mismatches, 0);
  const std::string json = LinearizationValidationJson(result, options);
  BOOST_CHECK(json.find("legacy_guarded") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(SignedIsAnExplicitExperimentMode) {
  const Snapshot snapshot = SyntheticSnapshot("signed");
  LinearizationValidationOptions options;
  LinearizationValidationResult result;
  std::string error;
  BOOST_REQUIRE(ValidateResidualsAndJacobians(snapshot, options, &result,
                                              &error));
  BOOST_CHECK_MESSAGE(result.pass, error);
  BOOST_CHECK_EQUAL(result.lidar_mode, "signed");
  BOOST_CHECK_EQUAL(result.lidar_guarded_count, 0);
}

}  // namespace gpu_ba
}  // namespace colmap
