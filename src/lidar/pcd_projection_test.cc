#define TEST_NAME "lidar/pcd_projection"
#include "util/testing.h"

#include <limits>

#include "base/camera_models.h"
#include "base/pose.h"
#include "lidar/pcd_projection.h"

#ifdef GPU_BA_CUDA_ENABLED
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#endif

using namespace colmap;

namespace {

Camera CreatePinholeCamera() {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 800.0, 640, 480);
  camera.SetParams({800.0, 760.0, 320.0, 240.0});
  return camera;
}

}  // namespace

BOOST_AUTO_TEST_CASE(TestWorldFrameIntersectionAndReprojection) {
  const Camera camera = CreatePinholeCamera();
  const Eigen::Matrix3d rotation_cw =
      EulerAnglesToRotationMatrix(0.31, -0.27, 0.42);
  const Eigen::Vector3d camera_center_world(1.5, -2.0, 0.75);
  const Eigen::Vector3d translation_cw =
      -rotation_cw * camera_center_world;
  const Eigen::Vector3d point_camera(0.8, -0.45, 4.2);
  const Eigen::Vector3d expected_world =
      camera_center_world + rotation_cw.transpose() * point_camera;
  const Eigen::Vector2d image_point =
      camera.WorldToImage(point_camera.hnormalized());
  const Eigen::Vector3d plane_normal_world =
      rotation_cw.transpose() * Eigen::Vector3d(0.2, -0.1, 1.0);

  Eigen::Vector3d intersection_world;
  BOOST_REQUIRE(lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      rotation_cw,
      translation_cw,
      image_point,
      expected_world,
      plane_normal_world,
      &intersection_world));
  BOOST_CHECK(intersection_world.isApprox(expected_world, 1e-10));

  const Eigen::Vector3d reprojected_camera =
      rotation_cw * intersection_world + translation_cw;
  BOOST_REQUIRE_GT(reprojected_camera.z(), 0.0);
  BOOST_CHECK(camera.WorldToImage(reprojected_camera.hnormalized())
                  .isApprox(image_point, 1e-10));
}

BOOST_AUTO_TEST_CASE(TestOpenCVDistortionIntersection) {
  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 800.0, 640, 480);
  camera.SetParams(
      {800.0, 780.0, 320.0, 240.0, 0.12, -0.04, 0.003, -0.002});
  const Eigen::Vector2d normalized_point(0.36, -0.24);
  const Eigen::Vector2d image_point = camera.WorldToImage(normalized_point);
  const Eigen::Vector2d pinhole_image_point(
      800.0 * normalized_point.x() + 320.0,
      780.0 * normalized_point.y() + 240.0);
  BOOST_REQUIRE_GT((image_point - pinhole_image_point).norm(), 1.0);

  const Eigen::Vector3d expected_world(
      3.25 * normalized_point.x(), 3.25 * normalized_point.y(), 3.25);
  Eigen::Vector3d intersection_world;
  BOOST_REQUIRE(lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      Eigen::Matrix3d::Identity(),
      Eigen::Vector3d::Zero(),
      image_point,
      expected_world,
      Eigen::Vector3d::UnitZ(),
      &intersection_world));
  BOOST_CHECK(intersection_world.isApprox(expected_world, 1e-9));
  BOOST_CHECK(camera.WorldToImage(intersection_world.hnormalized())
                  .isApprox(image_point, 1e-9));
}

BOOST_AUTO_TEST_CASE(TestInvalidIntersections) {
  const Camera camera = CreatePinholeCamera();
  const Eigen::Vector2d image_point(320.0, 240.0);
  const Eigen::Matrix3d rotation_cw = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d translation_cw = Eigen::Vector3d::Zero();
  Eigen::Vector3d intersection_world;

  BOOST_CHECK(!lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      rotation_cw,
      translation_cw,
      image_point,
      Eigen::Vector3d(1.0, 0.0, 2.0),
      Eigen::Vector3d::UnitX(),
      &intersection_world));
  BOOST_CHECK(intersection_world.isZero());

  BOOST_CHECK(!lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      rotation_cw,
      translation_cw,
      image_point,
      Eigen::Vector3d(0.0, 0.0, 2.0),
      Eigen::Vector3d::Zero(),
      &intersection_world));
  BOOST_CHECK(intersection_world.isZero());

  Eigen::Vector3d nan_normal = Eigen::Vector3d::UnitZ();
  nan_normal.x() = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK(!lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      rotation_cw,
      translation_cw,
      image_point,
      Eigen::Vector3d(0.0, 0.0, 2.0),
      nan_normal,
      &intersection_world));
  BOOST_CHECK(intersection_world.isZero());

  BOOST_CHECK(!lidar::internal::IntersectCameraRayWithWorldPlane(
      camera,
      rotation_cw,
      translation_cw,
      image_point,
      Eigen::Vector3d(0.0, 0.0, -2.0),
      Eigen::Vector3d::UnitZ(),
      &intersection_world));
  BOOST_CHECK(intersection_world.isZero());
}

BOOST_AUTO_TEST_CASE(TestPcdProjectionOptionsHaveConservativeDefaultDistance) {
  const lidar::PcdProjectionOptions options;
  BOOST_CHECK_EQUAL(options.min_lidar_proj_dist, 0.0);
}

#ifdef GPU_BA_CUDA_ENABLED
namespace {

using WorldPoint = std::array<float, 3>;
using ProjectionMatches =
    std::map<point3D_t, lidar::SnapshotProjectionMatch>;

class ProjectionTempPcd {
 public:
  ProjectionTempPcd() {
    static std::atomic<uint64_t> next_id{0};
    path_ = "/tmp/pcd_projection_snapshot_test_" +
            std::to_string(static_cast<uint64_t>(getpid())) + "_" +
            std::to_string(next_id.fetch_add(1)) + ".pcd";
  }

  ~ProjectionTempPcd() { std::remove(path_.c_str()); }

  ProjectionTempPcd(const ProjectionTempPcd&) = delete;
  ProjectionTempPcd& operator=(const ProjectionTempPcd&) = delete;

  const std::string& path() const { return path_; }

  void Write(const std::vector<WorldPoint>& points) const {
    std::ofstream file(path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
      throw std::runtime_error("cannot create projection test PCD");
    }
    file << "VERSION 0.7\n"
         << "FIELDS x y z intensity normal_x normal_y normal_z curvature\n"
         << "SIZE 4 4 4 4 4 4 4 4\n"
         << "TYPE F F F F F F F F\n"
         << "COUNT 1 1 1 1 1 1 1 1\n"
         << "WIDTH " << points.size() << "\n"
         << "HEIGHT 1\n"
         << "POINTS " << points.size() << "\n"
         << "DATA ascii\n"
         << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const WorldPoint& point : points) {
      file << point[2] << ' ' << -point[0] << ' ' << -point[1]
           << " 1 0 0 0 0\n";
    }
    if (!file) {
      throw std::runtime_error("cannot write projection test PCD");
    }
  }

 private:
  std::string path_;
};

lidar::IncrementalCausalLidarMapDependencies ProjectionDependencies() {
  lidar::IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [](const std::vector<float>& support_xyz,
         const std::vector<float>& query_xyz,
         const float outer_radius,
         const float inner_radius,
         std::vector<float>* outer_normals,
         std::vector<float>* inner_normals,
         lidar::CudaNormalEstimationTiming* timing,
         std::string* error) {
        (void)support_xyz;
        (void)outer_radius;
        (void)inner_radius;
        if (outer_normals == nullptr || inner_normals == nullptr ||
            timing == nullptr || error == nullptr) {
          return false;
        }
        outer_normals->clear();
        inner_normals->clear();
        outer_normals->reserve(query_xyz.size() / 3 * 4);
        inner_normals->reserve(query_xyz.size() / 3 * 4);
        for (size_t index = 0; index < query_xyz.size() / 3; ++index) {
          outer_normals->insert(
              outer_normals->end(), {1.0f, 2.0f, 3.0f, 0.125f});
          inner_normals->insert(
              inner_normals->end(), {0.0f, 1.0f, 0.0f, 0.25f});
        }
        *timing = lidar::CudaNormalEstimationTiming();
        error->clear();
        return true;
      };
  return dependencies;
}

lidar::ScanSource ProjectionSource(const uint64_t scan_index,
                                   const std::string& path) {
  lidar::ScanSource source;
  source.scan_index = scan_index;
  source.pcd_path = path;
  source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
  return source;
}

std::shared_ptr<const lidar::LidarMapSnapshot> SnapshotFromPoints(
    const std::vector<WorldPoint>& points) {
  ProjectionTempPcd scan;
  scan.Write(points);
  lidar::IncrementalCausalLidarMap map(ProjectionDependencies());
  std::string error;
  if (!map.AppendScan(ProjectionSource(1, scan.path()), nullptr, &error)) {
    throw std::runtime_error(error);
  }
  return map.GetSnapshot();
}

Camera CreateProjectionCamera() {
  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 100.0, 100, 100);
  camera.SetParams({100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0});
  return camera;
}

lidar::PcdProjectionOptions CreateProjectionOptions() {
  lidar::PcdProjectionOptions options;
  options.depth_image_scale = 1.0;
  options.max_proj_scale = 0;
  options.min_proj_scale = 0;
  options.min_proj_dist = 1.0;
  options.choose_meter = 10.0f;
  options.min_lidar_proj_dist = 0.1;
  return options;
}

WorldPoint CameraPointToWorld(const Eigen::Matrix3d& rotation_cw,
                              const Eigen::Vector3d& point_camera) {
  const Eigen::Vector3d point_world = rotation_cw.transpose() * point_camera;
  return WorldPoint{{static_cast<float>(point_world.x()),
                     static_cast<float>(point_world.y()),
                     static_cast<float>(point_world.z())}};
}

Image CreateProjectionImage(const std::vector<Eigen::Vector2d>& points,
                            const std::vector<point3D_t>& point3D_ids,
                            const Eigen::Matrix3d& rotation_cw =
                                Eigen::Matrix3d::Identity(),
                            const Eigen::Vector3d& translation_cw =
                                Eigen::Vector3d::Zero()) {
  if (points.size() != point3D_ids.size()) {
    throw std::runtime_error("projection test image inputs differ in size");
  }
  Image image;
  image.SetQvec(RotationMatrixToQuaternion(rotation_cw));
  image.SetTvec(translation_cw);
  image.SetPoints2D(points);
  for (point2D_t index = 0; index < point3D_ids.size(); ++index) {
    if (point3D_ids[index] != kInvalidPoint3DId) {
      image.SetPoint3DForPoint2D(index, point3D_ids[index]);
    }
  }
  return image;
}

lidar::VoxelKey WorldVoxelKey(const WorldPoint& point) {
  return lidar::VoxelKey{
      static_cast<int64_t>(std::floor(static_cast<double>(point[0]) * 100.0)),
      static_cast<int64_t>(std::floor(static_cast<double>(point[1]) * 100.0)),
      static_cast<int64_t>(std::floor(static_cast<double>(point[2]) * 100.0))};
}

void PrimeProjectionOutputs(ProjectionMatches* matches,
                            lidar::SnapshotProjectionAudit* audit,
                            std::string* error) {
  lidar::SnapshotProjectionMatch stale_match;
  stale_match.point3D_id = 999;
  stale_match.map_version = 999;
  matches->emplace(stale_match.point3D_id, stale_match);
  audit->map_version = 999;
  audit->snapshot_sha256 = "stale";
  audit->full_snapshot_due_to_distortion = true;
  audit->aabb_visited_block_count = 999;
  audit->candidate_plane_count = 999;
  audit->positive_depth_hit_count = 999;
  audit->axial_depth_hit_count = 999;
  audit->coverage_pixel_visit_count = 999;
  *error = "stale";
}

void CheckProjectionFailureCleared(
    const ProjectionMatches& matches,
    const lidar::SnapshotProjectionAudit& audit,
    const std::string& error) {
  BOOST_CHECK(matches.empty());
  BOOST_CHECK_EQUAL(audit.map_version, 0);
  BOOST_CHECK(audit.snapshot_sha256.empty());
  BOOST_CHECK(!audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, 0);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 0);
  BOOST_CHECK_EQUAL(audit.positive_depth_hit_count, 0);
  BOOST_CHECK_EQUAL(audit.axial_depth_hit_count, 0);
  BOOST_CHECK_EQUAL(audit.coverage_pixel_visit_count, 0);
  BOOST_CHECK(!error.empty());
}

void CheckMatchesEqual(const ProjectionMatches& lhs,
                       const ProjectionMatches& rhs) {
  BOOST_REQUIRE_EQUAL(lhs.size(), rhs.size());
  auto lhs_it = lhs.begin();
  auto rhs_it = rhs.begin();
  while (lhs_it != lhs.end()) {
    BOOST_CHECK_EQUAL(lhs_it->first, rhs_it->first);
    const lidar::SnapshotProjectionMatch& lhs_match = lhs_it->second;
    const lidar::SnapshotProjectionMatch& rhs_match = rhs_it->second;
    BOOST_CHECK_EQUAL(lhs_match.point3D_id, rhs_match.point3D_id);
    BOOST_CHECK_EQUAL(lhs_match.point2D_idx, rhs_match.point2D_idx);
    BOOST_CHECK_EQUAL(lhs_match.scaled_pixel[0], rhs_match.scaled_pixel[0]);
    BOOST_CHECK_EQUAL(lhs_match.scaled_pixel[1], rhs_match.scaled_pixel[1]);
    BOOST_CHECK(lhs_match.plane_key == rhs_match.plane_key);
    BOOST_CHECK(lhs_match.plane.key == rhs_match.plane.key);
    BOOST_CHECK_EQUAL(lhs_match.map_version, rhs_match.map_version);
    BOOST_CHECK_EQUAL(lhs_match.camera_distance,
                      rhs_match.camera_distance);
    for (size_t axis = 0; axis < 3; ++axis) {
      BOOST_CHECK_EQUAL(lhs_match.plane.point[axis],
                        rhs_match.plane.point[axis]);
      BOOST_CHECK_EQUAL(lhs_match.plane.normal[axis],
                        rhs_match.plane.normal[axis]);
    }
    ++lhs_it;
    ++rhs_it;
  }
}

}  // namespace

BOOST_AUTO_TEST_CASE(SnapshotProjectionIdentityFrustumDepthAndNearest) {
  const WorldPoint near_center{{0.0f, 0.0f, 2.0f}};
  const WorldPoint far_center{{0.0f, 0.0f, 4.0f}};
  const WorldPoint outside_frustum{{2.0f, 0.0f, 2.0f}};
  const WorldPoint zero_depth{{0.0f, 0.0f, 0.0f}};
  const WorldPoint behind_camera{{0.0f, 0.0f, -1.0f}};
  const WorldPoint on_far_plane{{2.0f, 0.0f, 10.0f}};
  const WorldPoint beyond_far_plane{{2.2f, 0.0f, 11.0f}};
  const WorldPoint remote_block{{100.0f, 100.0f, 100.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({near_center,
                          far_center,
                          outside_frustum,
                          zero_depth,
                          behind_camera,
                          on_far_plane,
                          beyond_far_plane,
                          remote_block});
  const Camera camera = CreateProjectionCamera();
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(50.0, 50.0),
       Eigen::Vector2d(70.0, 50.0),
       Eigen::Vector2d(50.0, 50.0)},
      {20, 10, kInvalidPoint3DId});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
  BOOST_CHECK_EQUAL(matches.begin()->first, 10);
  BOOST_REQUIRE_EQUAL(matches.count(20), 1);
  BOOST_REQUIRE_EQUAL(matches.count(10), 1);
  BOOST_CHECK(matches.at(20).plane_key == WorldVoxelKey(near_center));
  BOOST_CHECK(matches.at(10).plane_key == WorldVoxelKey(on_far_plane));
  BOOST_CHECK_EQUAL(matches.at(20).camera_distance, 2.0f);
  BOOST_CHECK_EQUAL(matches.at(20).scaled_pixel[0], 50);
  BOOST_CHECK_EQUAL(matches.at(20).scaled_pixel[1], 50);
  BOOST_CHECK_EQUAL(matches.at(10).scaled_pixel[0], 70);
  BOOST_CHECK_EQUAL(matches.at(10).scaled_pixel[1], 50);
  BOOST_CHECK_EQUAL(matches.at(20).map_version, snapshot->Version());
  BOOST_CHECK_EQUAL(matches.at(10).map_version, snapshot->Version());

  BOOST_CHECK_EQUAL(audit.map_version, snapshot->Version());
  BOOST_CHECK_EQUAL(audit.snapshot_sha256, snapshot->SnapshotSha256());
  BOOST_CHECK_EQUAL(audit.geometry_sha256, snapshot->GeometrySha256());
  BOOST_CHECK_EQUAL(audit.snapshot_voxel_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.snapshot_block_count, snapshot->BlockCount());
  BOOST_CHECK(!audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.input_feature_count, 2);
  BOOST_CHECK_EQUAL(audit.in_image_feature_count, 2);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 5);
  BOOST_CHECK_EQUAL(audit.positive_depth_hit_count, 4);
  BOOST_CHECK_EQUAL(audit.axial_depth_hit_count, 4);
  BOOST_CHECK_EQUAL(audit.pixel_hit_count, 3);
  BOOST_CHECK_EQUAL(audit.coverage_pixel_visit_count, 3);
  BOOST_CHECK_EQUAL(audit.feature_coverage_hit_count, 3);
  BOOST_CHECK_EQUAL(audit.feature_pixel_hit_count, 2);
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 2);
  BOOST_CHECK_LT(audit.aabb_visited_block_count, snapshot->BlockCount());

  const lidar::PlaneCollectionResult collection =
      snapshot->CollectPlanesInAabb(audit.projection_aabb,
                                    lidar::LidarNormalScale::PROJECTION);
  BOOST_REQUIRE(collection.ok);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count,
                    collection.visited_block_count);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, collection.planes.size());
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionTieSharesPixelAndReturnsOuterNormal) {
  const WorldPoint lower_key{{-0.004f, 0.0f, 2.0f}};
  const WorldPoint higher_key{{0.004f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({higher_key, lower_key});
  const Camera camera = CreateProjectionCamera();
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(50.0, 50.0), Eigen::Vector2d(50.0, 50.0)},
      {8, 3});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
  const lidar::VoxelKey expected_key = WorldVoxelKey(lower_key);
  BOOST_CHECK(matches.at(8).plane_key == expected_key);
  BOOST_CHECK(matches.at(3).plane_key == expected_key);
  BOOST_CHECK_EQUAL(matches.at(8).camera_distance,
                    matches.at(3).camera_distance);
  BOOST_CHECK_EQUAL(matches.at(8).point2D_idx, 0);
  BOOST_CHECK_EQUAL(matches.at(3).point2D_idx, 1);
  BOOST_CHECK(matches.at(8).plane.key == matches.at(8).plane_key);
  BOOST_CHECK(matches.at(3).plane.key == matches.at(3).plane_key);
  BOOST_CHECK_EQUAL(audit.feature_pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 2);

  lidar::LidarVoxelRecord record;
  BOOST_REQUIRE(snapshot->FindVoxel(expected_key, &record));
  BOOST_REQUIRE(record.outer_normal.valid);
  BOOST_REQUIRE(record.inner_normal.valid);
  BOOST_CHECK(matches.at(8).plane.scale ==
              lidar::LidarNormalScale::PROJECTION);
  BOOST_CHECK_EQUAL(matches.at(8).plane.curvature,
                    record.outer_normal.curvature);
  for (size_t axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_EQUAL(matches.at(8).plane.point[axis], record.centroid[axis]);
    BOOST_CHECK_EQUAL(matches.at(8).plane.normal[axis],
                      record.outer_normal.normal[axis]);
  }
  BOOST_CHECK(matches.at(8).plane.normal != record.inner_normal.normal);
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionUsesNonTrivialWorldPoseAndDistortion) {
  Camera camera = CreateProjectionCamera();
  camera.SetParams(
      {100.0, 95.0, 50.0, 50.0, 0.05, -0.01, 0.002, -0.003});
  const Eigen::Matrix3d rotation_cw =
      EulerAnglesToRotationMatrix(0.31, -0.27, 0.42);
  const Eigen::Vector3d camera_center_world(1.5, -2.0, 0.75);
  const Eigen::Vector3d translation_cw =
      -rotation_cw * camera_center_world;
  const Eigen::Vector3d point_camera(0.6, -0.3, 3.0);
  const Eigen::Vector3d point_world =
      camera_center_world + rotation_cw.transpose() * point_camera;
  const WorldPoint stored_world{{static_cast<float>(point_world.x()),
                                 static_cast<float>(point_world.y()),
                                 static_cast<float>(point_world.z())}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({stored_world});
  const Eigen::Vector2d distorted =
      camera.WorldToImage(point_camera.hnormalized());
  const Eigen::Vector2d feature(std::round(distorted.x()),
                                std::round(distorted.y()));
  const Image image = CreateProjectionImage(
      {feature}, {42}, rotation_cw, translation_cw);
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  const lidar::SnapshotProjectionMatch& match = matches.at(42);
  BOOST_CHECK(match.plane_key == WorldVoxelKey(stored_world));
  const Eigen::Vector3d matched_world(
      match.plane.point[0], match.plane.point[1], match.plane.point[2]);
  const Eigen::Vector3d matched_camera =
      rotation_cw * matched_world + translation_cw;
  BOOST_CHECK(matched_camera.isApprox(point_camera, 1e-5));
  BOOST_CHECK_CLOSE(match.camera_distance,
                    static_cast<float>(matched_camera.cast<float>().norm()),
                    1e-5);
  const Eigen::Vector2d matched_pixel =
      camera.WorldToImage(matched_camera.hnormalized());
  BOOST_CHECK_EQUAL(match.scaled_pixel[0],
                    static_cast<int>(std::round(matched_pixel.x())));
  BOOST_CHECK_EQUAL(match.scaled_pixel[1],
                    static_cast<int>(std::round(matched_pixel.y())));
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionIncludesBarrelDistortedBoundaryPoint) {
  Camera camera = CreateProjectionCamera();
  camera.SetParams(
      {100.0, 100.0, 50.0, 50.0, -0.5, 0.0, 0.0, 0.0});
  const WorldPoint distorted_edge_point{{6.0f, 0.0f, 10.0f}};
  const Eigen::Vector2d distorted_pixel =
      camera.WorldToImage(Eigen::Vector2d(0.6, 0.0));
  BOOST_CHECK_CLOSE(distorted_pixel.x(), 99.2, 1e-10);
  BOOST_CHECK_CLOSE(distorted_pixel.y(), 50.0, 1e-10);
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({distorted_edge_point});
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(99.2, 50.0)}, {81});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(81).plane_key ==
              WorldVoxelKey(distorted_edge_point));
  BOOST_CHECK_EQUAL(matches.at(81).scaled_pixel[0], 99);
  BOOST_CHECK_EQUAL(matches.at(81).scaled_pixel[1], 50);
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 1);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.axial_depth_hit_count, 1);
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 1);
}

BOOST_AUTO_TEST_CASE(
    SnapshotProjectionIncludesTangentialDistortionBoundaryCounterexample) {
  Camera camera = CreateProjectionCamera();
  camera.SetParams(
      {100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.1});
  const WorldPoint distorted_edge_point{
      {4.375617980957031f, 0.0f, 10.0f}};
  const Eigen::Vector2d distorted_pixel = camera.WorldToImage(
      Eigen::Vector2d(static_cast<double>(distorted_edge_point[0]) /
                          static_cast<double>(distorted_edge_point[2]),
                      0.0));
  BOOST_CHECK_EQUAL(static_cast<int>(std::round(distorted_pixel.x())), 99);
  BOOST_CHECK_EQUAL(static_cast<int>(std::round(distorted_pixel.y())), 50);
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({distorted_edge_point});
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(99.2, 50.0)}, {89});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(89).plane_key ==
              WorldVoxelKey(distorted_edge_point));
  BOOST_CHECK_EQUAL(matches.at(89).scaled_pixel[0], 99);
  BOOST_CHECK_EQUAL(matches.at(89).scaled_pixel[1], 50);
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.projection_aabb.min[0],
                    std::numeric_limits<double>::lowest());
  BOOST_CHECK_EQUAL(audit.projection_aabb.max[0],
                    std::numeric_limits<double>::max());
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 1);
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionUsesFullSnapshotForAnyNonzeroDistortion) {
  Camera camera = CreateProjectionCamera();
  camera.SetParams(
      {100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 1e-12});
  BOOST_REQUIRE(camera.IsUndistorted());
  const WorldPoint visible{{0.0f, 0.0f, 2.0f}};
  const WorldPoint remote{{100.0f, 100.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({visible, remote});
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(50.0, 50.0)}, {90});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(90).plane_key == WorldVoxelKey(visible));
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, snapshot->VoxelCount());
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionSkipsFiniteCentersOutsideIntegerRange) {
  Camera camera = CreateProjectionCamera();
  camera.Params(5) = 0.1;
  const WorldPoint visible{{0.0f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({visible,
                          WorldPoint{{10000.0f, 0.0f, 0.125f}},
                          WorldPoint{{-10000.0f, 0.0f, 0.125f}},
                          WorldPoint{{0.0f, 10000.0f, 0.125f}},
                          WorldPoint{{0.0f, -10000.0f, 0.125f}}});
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(50.0, 50.0)}, {90});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(90).plane_key == WorldVoxelKey(visible));
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 5);
  BOOST_CHECK_EQUAL(audit.axial_depth_hit_count, 5);
  BOOST_CHECK_EQUAL(audit.pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 1);
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionClipsCoverageWithCenterOutsideImage) {
  lidar::PcdProjectionOptions options = CreateProjectionOptions();
  options.max_proj_scale = 13;
  options.min_proj_scale = 13;
  const WorldPoint outside_center{{-1.02f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({outside_center});
  const Camera camera = CreateProjectionCamera();
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(0.2, 50.0)}, {82});
  lidar::PcdProj projector(options);
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(82).plane_key == WorldVoxelKey(outside_center));
  BOOST_CHECK_EQUAL(matches.at(82).scaled_pixel[0], 0);
  BOOST_CHECK_EQUAL(matches.at(82).scaled_pixel[1], 50);
  BOOST_CHECK(!audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 1);
  BOOST_CHECK_EQUAL(audit.pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(audit.feature_hit_count, 1);
}

BOOST_AUTO_TEST_CASE(
    SnapshotProjectionRejectsNegativeFractionalFeatureCoordinates) {
  const WorldPoint image_edge_point{{-1.0f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({image_edge_point});
  const Camera camera = CreateProjectionCamera();
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(-0.2, 50.0), Eigen::Vector2d(0.2, 50.0)},
      {83, 84});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_CHECK_EQUAL(audit.input_feature_count, 2);
  BOOST_CHECK_EQUAL(audit.in_image_feature_count, 1);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK_EQUAL(matches.count(83), 0);
  BOOST_CHECK_EQUAL(matches.count(84), 1);
  BOOST_CHECK_EQUAL(matches.at(84).scaled_pixel[0], 0);
}

BOOST_AUTO_TEST_CASE(
    SnapshotProjectionExplicitAxialBoundsRejectBehindAndBeyondFar) {
  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 50.0, 100, 100);
  camera.SetParams({50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.1});
  const Eigen::Vector3d camera_x =
      Eigen::Vector3d(1.0, -1.0, 0.0).normalized();
  const Eigen::Vector3d camera_z = Eigen::Vector3d::Ones().normalized();
  const Eigen::Vector3d camera_y = camera_z.cross(camera_x).normalized();
  Eigen::Matrix3d rotation_wc;
  rotation_wc.col(0) = camera_x;
  rotation_wc.col(1) = camera_y;
  rotation_wc.col(2) = camera_z;
  const Eigen::Matrix3d rotation_cw = rotation_wc.transpose();
  const WorldPoint visible =
      CameraPointToWorld(rotation_cw, Eigen::Vector3d(0.0, 0.0, 2.0));
  const WorldPoint behind =
      CameraPointToWorld(rotation_cw, Eigen::Vector3d(0.0, 0.0, -0.25));
  const WorldPoint beyond_far =
      CameraPointToWorld(rotation_cw, Eigen::Vector3d(0.0, 0.0, 10.5));
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({visible, behind, beyond_far});
  const Image image = CreateProjectionImage(
      {Eigen::Vector2d(50.0, 50.0)},
      {85},
      rotation_cw,
      Eigen::Vector3d::Zero());
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, 3);
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.positive_depth_hit_count, 2);
  BOOST_CHECK_EQUAL(audit.axial_depth_hit_count, 1);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(85).plane_key == WorldVoxelKey(visible));
  BOOST_CHECK(matches.at(85).plane_key != WorldVoxelKey(behind));
  BOOST_CHECK(matches.at(85).plane_key != WorldVoxelKey(beyond_far));
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionDefaultOptionsAreUsable) {
  const WorldPoint point{{0.0f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({point});
  const Camera camera = CreateProjectionCamera();
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(50.0, 50.0)}, {86});
  lidar::PcdProj projector{lidar::PcdProjectionOptions()};
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(86).plane_key == WorldVoxelKey(point));
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionDoesNotScanTheImagePerimeter) {
  Camera camera;
  camera.InitializeWithId(
      OpenCVCameraModel::model_id, 100.0, 600000, 1);
  camera.SetParams(
      {100.0, 100.0, 300000.0, 0.0, 0.0, 0.0, 0.0, 0.1});
  const WorldPoint point{{0.0f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({point});
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(300000.0, 0.0)}, {87});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image, camera, *snapshot, &matches, &audit, &error),
                        error);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
  BOOST_CHECK(matches.at(87).plane_key == WorldVoxelKey(point));
  BOOST_CHECK(audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(audit.aabb_visited_block_count, snapshot->BlockCount());
  BOOST_CHECK_EQUAL(audit.candidate_plane_count, snapshot->VoxelCount());
  BOOST_CHECK_EQUAL(audit.coverage_pixel_visit_count, 1);
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionUsesOnlyThePassedSnapshot) {
  const WorldPoint version1_point{{0.0f, 0.0f, 4.0f}};
  const WorldPoint version2_point{{0.0f, 0.0f, 2.0f}};
  ProjectionTempPcd scan1;
  ProjectionTempPcd scan2;
  scan1.Write({version1_point});
  scan2.Write({version2_point});
  lidar::IncrementalCausalLidarMap map(ProjectionDependencies());
  std::string append_error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(ProjectionSource(1, scan1.path()), nullptr, &append_error),
      append_error);
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot1 =
      map.GetSnapshot();
  const Camera camera = CreateProjectionCamera();
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(50.0, 50.0)}, {7});
  lidar::PcdProj projector(CreateProjectionOptions());
  ProjectionMatches before_append;
  lidar::SnapshotProjectionAudit before_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image,
                            camera,
                            *snapshot1,
                            &before_append,
                            &before_audit,
                            &error),
                        error);
  BOOST_CHECK(before_append.at(7).plane_key == WorldVoxelKey(version1_point));

  append_error.clear();
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(ProjectionSource(2, scan2.path()), nullptr, &append_error),
      append_error);
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot2 =
      map.GetSnapshot();
  BOOST_REQUIRE_NE(snapshot1->Version(), snapshot2->Version());
  BOOST_REQUIRE_NE(snapshot1->SnapshotSha256(), snapshot2->SnapshotSha256());

  ProjectionMatches old_snapshot_matches;
  ProjectionMatches new_snapshot_matches;
  ProjectionMatches repeated_matches;
  lidar::SnapshotProjectionAudit old_audit;
  lidar::SnapshotProjectionAudit new_audit;
  lidar::SnapshotProjectionAudit repeated_audit;
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image,
                            camera,
                            *snapshot1,
                            &old_snapshot_matches,
                            &old_audit,
                            &error),
                        error);
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image,
                            camera,
                            *snapshot2,
                            &new_snapshot_matches,
                            &new_audit,
                            &error),
                        error);
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            image,
                            camera,
                            *snapshot2,
                            &repeated_matches,
                            &repeated_audit,
                            &error),
                        error);

  CheckMatchesEqual(before_append, old_snapshot_matches);
  CheckMatchesEqual(new_snapshot_matches, repeated_matches);
  BOOST_CHECK(old_snapshot_matches.at(7).plane_key ==
              WorldVoxelKey(version1_point));
  BOOST_CHECK(new_snapshot_matches.at(7).plane_key ==
              WorldVoxelKey(version2_point));
  BOOST_CHECK_EQUAL(old_snapshot_matches.at(7).map_version,
                    snapshot1->Version());
  BOOST_CHECK_EQUAL(new_snapshot_matches.at(7).map_version,
                    snapshot2->Version());
  BOOST_CHECK_EQUAL(old_audit.map_version, snapshot1->Version());
  BOOST_CHECK_EQUAL(new_audit.map_version, snapshot2->Version());
  BOOST_CHECK_EQUAL(old_audit.snapshot_sha256, snapshot1->SnapshotSha256());
  BOOST_CHECK_EQUAL(new_audit.snapshot_sha256, snapshot2->SnapshotSha256());
  BOOST_CHECK_EQUAL(new_audit.candidate_plane_count,
                    repeated_audit.candidate_plane_count);
  BOOST_CHECK_EQUAL(new_audit.axial_depth_hit_count,
                    repeated_audit.axial_depth_hit_count);
  BOOST_CHECK_EQUAL(new_audit.feature_coverage_hit_count,
                    repeated_audit.feature_coverage_hit_count);
  BOOST_CHECK_EQUAL(new_audit.full_snapshot_due_to_distortion,
                    repeated_audit.full_snapshot_due_to_distortion);
  BOOST_CHECK_EQUAL(new_audit.coverage_pixel_visit_count,
                    repeated_audit.coverage_pixel_visit_count);
}

BOOST_AUTO_TEST_CASE(SnapshotProjectionRejectsInputsAndClearsOutputs) {
  const WorldPoint valid_point{{0.0f, 0.0f, 2.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({valid_point});
  const Camera camera = CreateProjectionCamera();
  const Image image =
      CreateProjectionImage({Eigen::Vector2d(50.0, 50.0)}, {1});
  const lidar::PcdProjectionOptions valid_options = CreateProjectionOptions();
  lidar::PcdProj projector(valid_options);
  ProjectionMatches matches;
  lidar::SnapshotProjectionAudit audit;
  std::string error;

  lidar::IncrementalCausalLidarMap empty_map(ProjectionDependencies());
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!projector.ProjectImageToSnapshot(
      image, camera, *empty_map.GetSnapshot(), &matches, &audit, &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("version") != std::string::npos);

  lidar::PcdProjectionOptions invalid_options = valid_options;
  invalid_options.max_proj_scale = -1;
  lidar::PcdProj invalid_projector(invalid_options);
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!invalid_projector.ProjectImageToSnapshot(
      image, camera, *snapshot, &matches, &audit, &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("options") != std::string::npos);

  Camera invalid_camera = camera;
  invalid_camera.Params(0) = std::numeric_limits<double>::quiet_NaN();
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!projector.ProjectImageToSnapshot(
      image, invalid_camera, *snapshot, &matches, &audit, &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("camera") != std::string::npos);

  Camera unreliable_boundary_camera = camera;
  unreliable_boundary_camera.Params(0) =
      std::numeric_limits<double>::min();
  unreliable_boundary_camera.Params(1) =
      std::numeric_limits<double>::min();
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!projector.ProjectImageToSnapshot(image,
                                                 unreliable_boundary_camera,
                                                 *snapshot,
                                                 &matches,
                                                 &audit,
                                                 &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("boundary") != std::string::npos);

  const Image duplicate_image = CreateProjectionImage(
      {Eigen::Vector2d(50.0, 50.0), Eigen::Vector2d(51.0, 50.0)},
      {5, 5});
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!projector.ProjectImageToSnapshot(
      duplicate_image, camera, *snapshot, &matches, &audit, &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("duplicate point3D_id") != std::string::npos);

  Image invalid_image = image;
  invalid_image.SetQvec(Eigen::Vector4d::Zero());
  PrimeProjectionOutputs(&matches, &audit, &error);
  BOOST_CHECK(!projector.ProjectImageToSnapshot(
      invalid_image, camera, *snapshot, &matches, &audit, &error));
  CheckProjectionFailureCleared(matches, audit, error);
  BOOST_CHECK(error.find("pose") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(
    SnapshotProjectionUsesAsymmetricNearFarRadiiWithManyIrrelevantFeatures) {
  Camera camera;
  camera.InitializeWithId(OpenCVCameraModel::model_id, 100.0, 1000, 1000);
  camera.SetParams(
      {100.0, 200.0, 500.0, 500.0, 0.0, 0.0, 0.0, 0.0});
  lidar::PcdProjectionOptions options = CreateProjectionOptions();
  options.max_proj_scale = 60;
  options.min_proj_scale = 12;
  options.min_proj_dist = 2.0;
  const WorldPoint near_point{{-2.0f, -1.0f, 1.0f}};
  const WorldPoint far_point{{20.0f, 10.0f, 10.0f}};
  const std::shared_ptr<const lidar::LidarMapSnapshot> snapshot =
      SnapshotFromPoints({near_point, far_point});

  std::vector<Eigen::Vector2d> baseline_points = {
      Eigen::Vector2d(291.0, 300.0),
      Eigen::Vector2d(309.0, 300.0),
      Eigen::Vector2d(300.0, 281.0),
      Eigen::Vector2d(300.0, 319.0),
      Eigen::Vector2d(290.0, 300.0),
      Eigen::Vector2d(310.0, 300.0),
      Eigen::Vector2d(300.0, 280.0),
      Eigen::Vector2d(300.0, 320.0),
      Eigen::Vector2d(699.0, 700.0),
      Eigen::Vector2d(701.0, 700.0),
      Eigen::Vector2d(700.0, 697.0),
      Eigen::Vector2d(700.0, 703.0),
      Eigen::Vector2d(698.0, 700.0),
      Eigen::Vector2d(702.0, 700.0),
      Eigen::Vector2d(700.0, 696.0),
      Eigen::Vector2d(700.0, 704.0),
  };
  std::vector<point3D_t> baseline_ids;
  for (point3D_t id = 1; id <= baseline_points.size(); ++id) {
    baseline_ids.push_back(id);
  }
  const Image baseline_image =
      CreateProjectionImage(baseline_points, baseline_ids);

  std::vector<Eigen::Vector2d> dense_points = baseline_points;
  std::vector<point3D_t> dense_ids = baseline_ids;
  point3D_t irrelevant_id = 1000;
  for (int y = 800; y < 850; ++y) {
    for (int x = 0; x < 100; ++x) {
      dense_points.emplace_back(static_cast<double>(x),
                                static_cast<double>(y));
      dense_ids.push_back(irrelevant_id++);
    }
  }
  const Image dense_image = CreateProjectionImage(dense_points, dense_ids);

  lidar::PcdProj projector(options);
  ProjectionMatches baseline_matches;
  ProjectionMatches dense_matches;
  lidar::SnapshotProjectionAudit baseline_audit;
  lidar::SnapshotProjectionAudit dense_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            baseline_image,
                            camera,
                            *snapshot,
                            &baseline_matches,
                            &baseline_audit,
                            &error),
                        error);
  BOOST_REQUIRE_MESSAGE(projector.ProjectImageToSnapshot(
                            dense_image,
                            camera,
                            *snapshot,
                            &dense_matches,
                            &dense_audit,
                            &error),
                        error);

  BOOST_REQUIRE_EQUAL(baseline_matches.size(), 8);
  for (const point3D_t id : {point3D_t{1}, point3D_t{2}, point3D_t{3},
                             point3D_t{4}}) {
    BOOST_REQUIRE_EQUAL(baseline_matches.count(id), 1);
    BOOST_CHECK(baseline_matches.at(id).plane_key == WorldVoxelKey(near_point));
  }
  for (const point3D_t id : {point3D_t{9}, point3D_t{10}, point3D_t{11},
                             point3D_t{12}}) {
    BOOST_REQUIRE_EQUAL(baseline_matches.count(id), 1);
    BOOST_CHECK(baseline_matches.at(id).plane_key == WorldVoxelKey(far_point));
  }
  for (const point3D_t id : {point3D_t{5}, point3D_t{6}, point3D_t{7},
                             point3D_t{8}, point3D_t{13}, point3D_t{14},
                             point3D_t{15}, point3D_t{16}}) {
    BOOST_CHECK_EQUAL(baseline_matches.count(id), 0);
  }
  CheckMatchesEqual(baseline_matches, dense_matches);
  BOOST_CHECK_EQUAL(baseline_audit.feature_coverage_hit_count, 8);
  BOOST_CHECK_EQUAL(baseline_audit.coverage_pixel_visit_count, 762);
  BOOST_CHECK_EQUAL(baseline_audit.feature_pixel_hit_count, 8);
  BOOST_CHECK_EQUAL(baseline_audit.feature_hit_count, 8);
  BOOST_CHECK_EQUAL(dense_audit.input_feature_count,
                    baseline_audit.input_feature_count + 5000);
  BOOST_CHECK_EQUAL(dense_audit.in_image_feature_count,
                    baseline_audit.in_image_feature_count + 5000);
  BOOST_CHECK_EQUAL(dense_audit.candidate_plane_count,
                    baseline_audit.candidate_plane_count);
  BOOST_CHECK_EQUAL(dense_audit.axial_depth_hit_count,
                    baseline_audit.axial_depth_hit_count);
  BOOST_CHECK_EQUAL(dense_audit.positive_depth_hit_count,
                    baseline_audit.positive_depth_hit_count);
  BOOST_CHECK_EQUAL(dense_audit.pixel_hit_count,
                    baseline_audit.pixel_hit_count);
  BOOST_CHECK_EQUAL(dense_audit.feature_coverage_hit_count,
                    baseline_audit.feature_coverage_hit_count);
  BOOST_CHECK_EQUAL(dense_audit.coverage_pixel_visit_count,
                    baseline_audit.coverage_pixel_visit_count);
  const uint64_t all_candidate_feature_pairs =
      dense_audit.candidate_plane_count * dense_audit.in_image_feature_count;
  BOOST_REQUIRE_GT(all_candidate_feature_pairs, 0);
  BOOST_CHECK_LT(dense_audit.coverage_pixel_visit_count,
                 all_candidate_feature_pairs / 10);
  BOOST_CHECK_EQUAL(dense_audit.feature_pixel_hit_count,
                    baseline_audit.feature_pixel_hit_count);
  BOOST_CHECK_EQUAL(dense_audit.feature_hit_count,
                    baseline_audit.feature_hit_count);
}
#endif
