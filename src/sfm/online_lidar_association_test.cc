#define TEST_NAME "sfm/online_lidar_association"
#include "util/testing.h"

#ifdef GPU_BA_CUDA_ENABLED

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "sfm/online_lidar_association.h"

using namespace colmap;

namespace {

using WorldPoint = std::array<float, 3>;

constexpr size_t kImagePointCount = 128;

class AssociationTempPcd {
 public:
  AssociationTempPcd() {
    static std::atomic<uint64_t> next_id{0};
    path_ = "/tmp/online_lidar_association_test_" +
            std::to_string(static_cast<uint64_t>(getpid())) + "_" +
            std::to_string(next_id.fetch_add(1)) + ".pcd";
  }

  ~AssociationTempPcd() { std::remove(path_.c_str()); }

  AssociationTempPcd(const AssociationTempPcd&) = delete;
  AssociationTempPcd& operator=(const AssociationTempPcd&) = delete;

  const std::string& Path() const { return path_; }

  void Write(const std::vector<WorldPoint>& points) const {
    std::ofstream file(path_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
      throw std::runtime_error("cannot create association test PCD");
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
      throw std::runtime_error("cannot write association test PCD");
    }
  }

 private:
  std::string path_;
};

lidar::IncrementalCausalLidarMapDependencies AssociationDependencies(
    const std::array<float, 4>& outer_normal,
    const std::array<float, 4>& inner_normal) {
  lidar::IncrementalCausalLidarMapDependencies dependencies;
  dependencies.estimate_normals =
      [outer_normal, inner_normal](
          const std::vector<float>& support_xyz,
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
            timing == nullptr || error == nullptr ||
            query_xyz.size() % 3 != 0) {
          return false;
        }
        outer_normals->clear();
        inner_normals->clear();
        outer_normals->reserve(query_xyz.size() / 3 * 4);
        inner_normals->reserve(query_xyz.size() / 3 * 4);
        for (size_t index = 0; index < query_xyz.size() / 3; ++index) {
          outer_normals->insert(
              outer_normals->end(), outer_normal.begin(), outer_normal.end());
          inner_normals->insert(
              inner_normals->end(), inner_normal.begin(), inner_normal.end());
        }
        *timing = lidar::CudaNormalEstimationTiming();
        error->clear();
        return true;
      };
  return dependencies;
}

std::shared_ptr<const lidar::LidarMapSnapshot> SnapshotFromPoints(
    const std::vector<WorldPoint>& points,
    const std::array<float, 4>& outer_normal =
        std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.125f}},
    const std::array<float, 4>& inner_normal =
        std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.25f}}) {
  AssociationTempPcd scan;
  scan.Write(points);
  lidar::IncrementalCausalLidarMap map(
      AssociationDependencies(outer_normal, inner_normal));
  lidar::ScanSource source;
  source.scan_index = 1;
  source.pcd_path = scan.Path();
  source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
  std::string error;
  if (!map.AppendScan(source, nullptr, &error)) {
    throw std::runtime_error(error);
  }
  return map.GetSnapshot();
}

lidar::ScanSource ScanSource(const uint64_t scan_index,
                             const std::string& path) {
  lidar::ScanSource source;
  source.scan_index = scan_index;
  source.pcd_path = path;
  source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
  return source;
}

lidar::PcdProjectionOptions ProjectionOptions() {
  lidar::PcdProjectionOptions options;
  options.depth_image_scale = 1.0;
  options.max_proj_scale = 0;
  options.min_proj_scale = 0;
  options.min_proj_dist = 1.0;
  options.choose_meter = 10.0f;
  options.min_lidar_proj_dist = 0.0;
  return options;
}

lidar::VoxelKey WorldVoxelKey(const WorldPoint& point) {
  return lidar::VoxelKey{
      static_cast<int64_t>(std::floor(static_cast<double>(point[0]) * 100.0)),
      static_cast<int64_t>(std::floor(static_cast<double>(point[1]) * 100.0)),
      static_cast<int64_t>(std::floor(static_cast<double>(point[2]) * 100.0))};
}

Reconstruction CreateReconstruction() {
  Reconstruction reconstruction;
  reconstruction.BeginStructureJournal(71);

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(OpenCVCameraModel::model_id, 100.0, 100, 80);
  camera.SetParams({100.0, 100.0, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0});
  reconstruction.AddCamera(camera);

  for (image_t image_id = 1; image_id <= 4; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera.CameraId());
    image.SetName("association_image_" + std::to_string(image_id));
    const double half_angle = 0.01 * static_cast<double>(image_id);
    image.SetQvec(Eigen::Vector4d(
        std::cos(half_angle), 0.0, std::sin(half_angle), 0.0));
    image.SetTvec(Eigen::Vector3d(0.1 * image_id,
                                  -0.2 * image_id,
                                  0.3 * image_id));
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(kImagePointCount);
    for (size_t index = 0; index < kImagePointCount; ++index) {
      points2D.emplace_back(10.0 + index, 20.0 + image_id);
    }
    image.SetPoints2D(points2D);
    reconstruction.AddImage(std::move(image));
    reconstruction.RegisterImage(image_id);
  }

  Eigen::Vector3d lidar_xyz(9.0, 8.0, 7.0);
  Eigen::Vector4d lidar_abcd(0.0, 1.0, 0.0, -8.0);
  LidarPoint lidar_point(LidarPointType::IcpGround, lidar_xyz, lidar_abcd);
  Eigen::Vector3ub lidar_color(11, 22, 33);
  lidar_point.SetColor(lidar_color);
  lidar_point.SetDist(0.125);
  lidar_point.SetAngle(0.75);
  reconstruction.AddLidarPoint(101, lidar_point);

  Eigen::Vector3d global_xyz(6.0, 5.0, 4.0);
  Eigen::Vector4d global_abcd(1.0, 0.0, 0.0, -6.0);
  LidarPoint global_point(LidarPointType::Icp, global_xyz, global_abcd);
  Eigen::Vector3ub global_color(44, 55, 66);
  global_point.SetColor(global_color);
  global_point.SetDist(0.25);
  global_point.SetAngle(0.5);
  reconstruction.AddLidarPointInGlobal(202, global_point);
  return reconstruction;
}

void AddRegisteredImage(Reconstruction* reconstruction,
                        const image_t image_id) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(1);
  image.SetName("association_image_" + std::to_string(image_id));
  image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
  image.SetTvec(Eigen::Vector3d::Zero());
  image.SetPoints2D(std::vector<Eigen::Vector2d>(
      kImagePointCount, Eigen::Vector2d(50.0, 40.0)));
  reconstruction->AddImage(std::move(image));
  reconstruction->RegisterImage(image_id);
}

void SetIdentityProjectionPose(Reconstruction* reconstruction,
                               const std::vector<image_t>& image_ids) {
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction->Image(image_id);
    image.SetQvec(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    image.SetTvec(Eigen::Vector3d::Zero());
  }
}

void SetFeature(Reconstruction* reconstruction,
                const image_t image_id,
                const point2D_t point2D_idx,
                const double x,
                const double y) {
  reconstruction->Image(image_id)
      .Point2D(point2D_idx)
      .SetXY(Eigen::Vector2d(x, y));
}

void SetSnapshotIdentity(
    const std::shared_ptr<const lidar::LidarMapSnapshot>& snapshot,
    OnlineLidarAssociationRequest* request) {
  request->expected_map_version = snapshot->Version();
  request->expected_max_scan_index = snapshot->MaxScanIndex();
  request->expected_snapshot_sha256 = snapshot->SnapshotSha256();
  request->expected_geometry_sha256 = snapshot->GeometrySha256();
  request->snapshot = snapshot;
}

point3D_t AddPoint(Reconstruction* reconstruction,
                   const Eigen::Vector3d& xyz,
                   std::vector<TrackElement> elements,
                   const int global_opt_num = 0) {
  Track track;
  track.SetElements(std::move(elements));
  const point3D_t point3D_id = reconstruction->AddPoint3D(
      xyz, std::move(track), Eigen::Vector3ub(7, 8, 9));
  Point3D& point3D = reconstruction->Point3D(point3D_id);
  point3D.SetLidarXYZ(xyz + Eigen::Vector3d(0.5, 0.25, 0.125));
  point3D.SetError(0.01 * static_cast<double>(point3D_id));
  point3D.IfInSphere() = point3D_id % 2 == 0;
  for (int index = 0; index < global_opt_num; ++index) {
    point3D.AddGlobalOptNum();
  }
  return point3D_id;
}

bool IsLowercaseSha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void PrimeOutput(OnlineLidarAssociationOutput* output) {
  OnlineLidarAssociation stale;
  stale.association_id = 99;
  stale.point3D_id = 99;
  output->associations.push_back(stale);
}

void PrimeAudit(OnlineLidarAssociationAudit* audit) {
  audit->attempt_id = 99;
  audit->pass_index = 99;
  audit->trigger_image_id = 99;
  audit->map_version = 99;
  audit->max_scan_index = 99;
  audit->snapshot_sha256 = "stale";
  audit->geometry_sha256 = "stale";
  audit->input_point_count = 99;
  audit->selected_association_count = 99;
  audit->skipped_point_count = 99;
  audit->skipped_no_window_observation_count = 99;
  audit->skipped_pair_threshold_point_count = 99;
  audit->skipped_no_plane_count = 99;
  audit->pair_threshold_skipped_observation_count = 99;
  audit->projection_route_point_count = 99;
  audit->kdtree_route_point_count = 99;
  audit->projection_selected_association_count = 99;
  audit->kdtree_selected_association_count = 99;
  audit->projection_call_count = 99;
  audit->projection_calls.resize(1);
  audit->projection_calls.front().image_id = 99;
  audit->preliminary_selected_count_by_image.emplace(99, 99);
  audit->trigger_preliminary_selected_count = 99;
  audit->association_sha256 = "stale";
}

void CheckOutputCleared(const OnlineLidarAssociationOutput& output) {
  BOOST_CHECK(output.associations.empty());
}

void CheckAuditCleared(const OnlineLidarAssociationAudit& audit) {
  BOOST_CHECK_EQUAL(audit.attempt_id, 0);
  BOOST_CHECK_EQUAL(audit.pass_index, 0);
  BOOST_CHECK_EQUAL(audit.trigger_image_id, kInvalidImageId);
  BOOST_CHECK_EQUAL(audit.map_version, 0);
  BOOST_CHECK_EQUAL(audit.max_scan_index, 0);
  BOOST_CHECK(audit.snapshot_sha256.empty());
  BOOST_CHECK(audit.geometry_sha256.empty());
  BOOST_CHECK_EQUAL(audit.input_point_count, 0);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_window_observation_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_pair_threshold_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 0);
  BOOST_CHECK_EQUAL(audit.pair_threshold_skipped_observation_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 0);
  BOOST_CHECK(audit.projection_calls.empty());
  BOOST_CHECK(audit.preliminary_selected_count_by_image.empty());
  BOOST_CHECK_EQUAL(audit.trigger_preliminary_selected_count, 0);
  BOOST_CHECK(audit.association_sha256.empty());
}

void CheckBuildFailure(const Reconstruction& reconstruction,
                       const OnlineLidarAssociationRequest& request,
                       const std::string& expected_error) {
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error = "stale";
  PrimeOutput(&output);
  PrimeAudit(&audit);
  BOOST_CHECK(!BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error));
  CheckOutputCleared(output);
  CheckAuditCleared(audit);
  BOOST_CHECK(!error.empty());
  BOOST_CHECK_MESSAGE(error.find(expected_error) != std::string::npos, error);
}

void CheckLidarMapsEqual(const Reconstruction& before,
                         const Reconstruction& after,
                         const bool global) {
  const auto& before_map =
      global ? before.LidarPointsInGlobal() : before.LidarPoints();
  const auto& after_map =
      global ? after.LidarPointsInGlobal() : after.LidarPoints();
  BOOST_REQUIRE_EQUAL(before_map.size(), after_map.size());
  for (const auto& item : before_map) {
    const auto found = after_map.find(item.first);
    BOOST_REQUIRE(found != after_map.end());
    LidarPoint before_point = item.second;
    LidarPoint after_point = found->second;
    BOOST_CHECK_EQUAL(static_cast<int>(before_point.Type()),
                      static_cast<int>(after_point.Type()));
    const Eigen::Vector3d before_xyz = before_point.LidarXYZ();
    const Eigen::Vector3d after_xyz = after_point.LidarXYZ();
    const Eigen::Vector4d before_abcd = before_point.LidarABCD();
    const Eigen::Vector4d after_abcd = after_point.LidarABCD();
    for (size_t axis = 0; axis < 3; ++axis) {
      BOOST_CHECK_EQUAL(before_xyz(axis), after_xyz(axis));
      BOOST_CHECK_EQUAL(static_cast<int>(before_point.Color()(axis)),
                        static_cast<int>(after_point.Color()(axis)));
    }
    for (size_t axis = 0; axis < 4; ++axis) {
      BOOST_CHECK_EQUAL(before_abcd(axis), after_abcd(axis));
    }
    BOOST_CHECK_EQUAL(before_point.Dist(), after_point.Dist());
    BOOST_CHECK_EQUAL(before_point.Angle(), after_point.Angle());
  }
}

void CheckReconstructionStateEqual(const Reconstruction& before,
                                   const Reconstruction& after) {
  BOOST_CHECK_EQUAL(before.StructureOwnerEpoch(), after.StructureOwnerEpoch());
  BOOST_CHECK_EQUAL(before.StructureRevision(), after.StructureRevision());
  BOOST_CHECK_EQUAL(before.OldestRetainedStructureRevision(),
                    after.OldestRetainedStructureRevision());
  BOOST_CHECK_EQUAL(before.StructureJournalEnabled(),
                    after.StructureJournalEnabled());
  BOOST_CHECK(before.RegImageIds() == after.RegImageIds());

  BOOST_REQUIRE_EQUAL(before.Cameras().size(), after.Cameras().size());
  for (const auto& item : before.Cameras()) {
    const auto found = after.Cameras().find(item.first);
    BOOST_REQUIRE(found != after.Cameras().end());
    BOOST_CHECK_EQUAL(item.second.CameraId(), found->second.CameraId());
    BOOST_CHECK_EQUAL(item.second.ModelId(), found->second.ModelId());
    BOOST_CHECK_EQUAL(item.second.Width(), found->second.Width());
    BOOST_CHECK_EQUAL(item.second.Height(), found->second.Height());
    BOOST_CHECK(item.second.Params() == found->second.Params());
  }

  BOOST_REQUIRE_EQUAL(before.Images().size(), after.Images().size());
  for (const auto& item : before.Images()) {
    const auto found = after.Images().find(item.first);
    BOOST_REQUIRE(found != after.Images().end());
    BOOST_CHECK_EQUAL(item.second.ImageId(), found->second.ImageId());
    BOOST_CHECK_EQUAL(item.second.CameraId(), found->second.CameraId());
    BOOST_CHECK_EQUAL(item.second.Name(), found->second.Name());
    BOOST_CHECK_EQUAL(item.second.IsRegistered(), found->second.IsRegistered());
    BOOST_CHECK_EQUAL(item.second.NumPoints2D(),
                      found->second.NumPoints2D());
    BOOST_CHECK_EQUAL(item.second.NumPoints3D(),
                      found->second.NumPoints3D());
    for (size_t index = 0; index < 4; ++index) {
      BOOST_CHECK_EQUAL(item.second.Qvec()(index), found->second.Qvec()(index));
    }
    for (size_t index = 0; index < 3; ++index) {
      BOOST_CHECK_EQUAL(item.second.Tvec()(index), found->second.Tvec()(index));
    }
    for (point2D_t point2D_idx = 0;
         point2D_idx < item.second.NumPoints2D();
         ++point2D_idx) {
      const class Point2D& before_point2D = item.second.Point2D(point2D_idx);
      const class Point2D& after_point2D =
          found->second.Point2D(point2D_idx);
      BOOST_CHECK(before_point2D.XY() == after_point2D.XY());
      BOOST_CHECK_EQUAL(before_point2D.HasPoint3D(),
                        after_point2D.HasPoint3D());
      BOOST_CHECK_EQUAL(before_point2D.Point3DId(),
                        after_point2D.Point3DId());
    }
  }

  BOOST_REQUIRE_EQUAL(before.Points3D().size(), after.Points3D().size());
  for (const auto& item : before.Points3D()) {
    const auto found = after.Points3D().find(item.first);
    BOOST_REQUIRE(found != after.Points3D().end());
    const Point3D& before_point = item.second;
    const Point3D& after_point = found->second;
    for (size_t axis = 0; axis < 3; ++axis) {
      BOOST_CHECK_EQUAL(before_point.XYZ()(axis), after_point.XYZ()(axis));
      BOOST_CHECK_EQUAL(before_point.LidarXYZ()(axis),
                        after_point.LidarXYZ()(axis));
      BOOST_CHECK_EQUAL(static_cast<int>(before_point.Color()(axis)),
                        static_cast<int>(after_point.Color()(axis)));
    }
    BOOST_CHECK_EQUAL(before_point.Error(), after_point.Error());
    BOOST_CHECK_EQUAL(before_point.GlobalOptNum(), after_point.GlobalOptNum());
    BOOST_CHECK_EQUAL(before_point.IfInSphere(), after_point.IfInSphere());
    BOOST_REQUIRE_EQUAL(before_point.Track().Length(),
                        after_point.Track().Length());
    for (size_t index = 0; index < before_point.Track().Length(); ++index) {
      BOOST_CHECK_EQUAL(before_point.Track().Element(index).image_id,
                        after_point.Track().Element(index).image_id);
      BOOST_CHECK_EQUAL(before_point.Track().Element(index).point2D_idx,
                        after_point.Track().Element(index).point2D_idx);
    }
  }
  CheckLidarMapsEqual(before, after, false);
  CheckLidarMapsEqual(before, after, true);
}

void CheckProjectionAuditEqual(const lidar::SnapshotProjectionAudit& first,
                               const lidar::SnapshotProjectionAudit& second) {
  BOOST_CHECK_EQUAL(first.map_version, second.map_version);
  BOOST_CHECK_EQUAL(first.snapshot_sha256, second.snapshot_sha256);
  BOOST_CHECK_EQUAL(first.geometry_sha256, second.geometry_sha256);
  BOOST_CHECK_EQUAL(first.snapshot_voxel_count, second.snapshot_voxel_count);
  BOOST_CHECK_EQUAL(first.snapshot_block_count, second.snapshot_block_count);
  BOOST_CHECK_EQUAL(first.full_snapshot_due_to_distortion,
                    second.full_snapshot_due_to_distortion);
  for (size_t axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_EQUAL(first.projection_aabb.min[axis],
                      second.projection_aabb.min[axis]);
    BOOST_CHECK_EQUAL(first.projection_aabb.max[axis],
                      second.projection_aabb.max[axis]);
  }
  BOOST_CHECK_EQUAL(static_cast<int>(first.projection_aabb.frame),
                    static_cast<int>(second.projection_aabb.frame));
  BOOST_CHECK_EQUAL(first.input_feature_count, second.input_feature_count);
  BOOST_CHECK_EQUAL(first.in_image_feature_count,
                    second.in_image_feature_count);
  BOOST_CHECK_EQUAL(first.aabb_visited_block_count,
                    second.aabb_visited_block_count);
  BOOST_CHECK_EQUAL(first.candidate_plane_count,
                    second.candidate_plane_count);
  BOOST_CHECK_EQUAL(first.positive_depth_hit_count,
                    second.positive_depth_hit_count);
  BOOST_CHECK_EQUAL(first.axial_depth_hit_count,
                    second.axial_depth_hit_count);
  BOOST_CHECK_EQUAL(first.pixel_hit_count, second.pixel_hit_count);
  BOOST_CHECK_EQUAL(first.coverage_pixel_visit_count,
                    second.coverage_pixel_visit_count);
  BOOST_CHECK_EQUAL(first.feature_coverage_hit_count,
                    second.feature_coverage_hit_count);
  BOOST_CHECK_EQUAL(first.feature_pixel_hit_count,
                    second.feature_pixel_hit_count);
  BOOST_CHECK_EQUAL(first.feature_hit_count, second.feature_hit_count);
}

void CheckPlaneEqual(const lidar::PlaneSample& first,
                     const lidar::PlaneSample& second) {
  BOOST_CHECK(first.key == second.key);
  BOOST_CHECK_EQUAL(static_cast<int>(first.frame),
                    static_cast<int>(second.frame));
  BOOST_CHECK_EQUAL(static_cast<int>(first.scale),
                    static_cast<int>(second.scale));
  for (size_t axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_EQUAL(first.point[axis], second.point[axis]);
    BOOST_CHECK_EQUAL(first.normal[axis], second.normal[axis]);
  }
  BOOST_CHECK_EQUAL(first.curvature, second.curvature);
  BOOST_CHECK_EQUAL(first.normal_revision, second.normal_revision);
  BOOST_CHECK_EQUAL(first.voxel_count, second.voxel_count);
}

void CheckAssociationEqual(const OnlineLidarAssociation& first,
                           const OnlineLidarAssociation& second) {
  BOOST_CHECK_EQUAL(first.association_id, second.association_id);
  BOOST_CHECK_EQUAL(first.attempt_id, second.attempt_id);
  BOOST_CHECK_EQUAL(first.pass_index, second.pass_index);
  BOOST_CHECK_EQUAL(first.point3D_id, second.point3D_id);
  for (size_t axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_EQUAL(first.point3D_xyz[axis], second.point3D_xyz[axis]);
  }
  BOOST_CHECK_EQUAL(first.owner_image_id, second.owner_image_id);
  BOOST_CHECK_EQUAL(first.owner_point2D_idx, second.owner_point2D_idx);
  BOOST_CHECK_EQUAL(first.map_version, second.map_version);
  BOOST_CHECK_EQUAL(first.max_scan_index, second.max_scan_index);
  BOOST_CHECK_EQUAL(first.snapshot_sha256, second.snapshot_sha256);
  BOOST_CHECK_EQUAL(first.geometry_sha256, second.geometry_sha256);
  BOOST_CHECK_EQUAL(static_cast<int>(first.route),
                    static_cast<int>(second.route));
  CheckPlaneEqual(first.plane, second.plane);
  BOOST_CHECK(first.plane_key == second.plane_key);
  for (size_t index = 0; index < 4; ++index) {
    BOOST_CHECK_EQUAL(first.plane_abcd[index], second.plane_abcd[index]);
  }
  BOOST_CHECK_EQUAL(first.has_search_range, second.has_search_range);
  BOOST_CHECK_EQUAL(first.search_range, second.search_range);
  BOOST_CHECK_EQUAL(static_cast<int>(first.lidar_point_type),
                    static_cast<int>(second.lidar_point_type));
  BOOST_CHECK_EQUAL(first.projection_camera_distance,
                    second.projection_camera_distance);
  BOOST_CHECK_EQUAL(first.projection_angle_score,
                    second.projection_angle_score);
}

void CheckOutputEqual(const OnlineLidarAssociationOutput& first,
                      const OnlineLidarAssociationOutput& second) {
  BOOST_REQUIRE_EQUAL(first.associations.size(), second.associations.size());
  for (size_t index = 0; index < first.associations.size(); ++index) {
    CheckAssociationEqual(first.associations[index], second.associations[index]);
  }
}

void CheckAuditEqual(const OnlineLidarAssociationAudit& first,
                     const OnlineLidarAssociationAudit& second) {
  BOOST_CHECK_EQUAL(first.attempt_id, second.attempt_id);
  BOOST_CHECK_EQUAL(first.pass_index, second.pass_index);
  BOOST_CHECK_EQUAL(first.trigger_image_id, second.trigger_image_id);
  BOOST_CHECK_EQUAL(first.map_version, second.map_version);
  BOOST_CHECK_EQUAL(first.max_scan_index, second.max_scan_index);
  BOOST_CHECK_EQUAL(first.snapshot_sha256, second.snapshot_sha256);
  BOOST_CHECK_EQUAL(first.geometry_sha256, second.geometry_sha256);
  BOOST_CHECK_EQUAL(first.input_point_count, second.input_point_count);
  BOOST_CHECK_EQUAL(first.selected_association_count,
                    second.selected_association_count);
  BOOST_CHECK_EQUAL(first.skipped_point_count, second.skipped_point_count);
  BOOST_CHECK_EQUAL(first.skipped_no_window_observation_count,
                    second.skipped_no_window_observation_count);
  BOOST_CHECK_EQUAL(first.skipped_pair_threshold_point_count,
                    second.skipped_pair_threshold_point_count);
  BOOST_CHECK_EQUAL(first.skipped_no_plane_count,
                    second.skipped_no_plane_count);
  BOOST_CHECK_EQUAL(first.pair_threshold_skipped_observation_count,
                    second.pair_threshold_skipped_observation_count);
  BOOST_CHECK_EQUAL(first.projection_route_point_count,
                    second.projection_route_point_count);
  BOOST_CHECK_EQUAL(first.kdtree_route_point_count,
                    second.kdtree_route_point_count);
  BOOST_CHECK_EQUAL(first.projection_selected_association_count,
                    second.projection_selected_association_count);
  BOOST_CHECK_EQUAL(first.kdtree_selected_association_count,
                    second.kdtree_selected_association_count);
  BOOST_CHECK_EQUAL(first.projection_call_count, second.projection_call_count);
  BOOST_REQUIRE_EQUAL(first.projection_calls.size(),
                      second.projection_calls.size());
  for (size_t index = 0; index < first.projection_calls.size(); ++index) {
    BOOST_CHECK_EQUAL(first.projection_calls[index].image_id,
                      second.projection_calls[index].image_id);
    CheckProjectionAuditEqual(first.projection_calls[index].projection,
                              second.projection_calls[index].projection);
  }
  BOOST_CHECK(first.preliminary_selected_count_by_image ==
              second.preliminary_selected_count_by_image);
  BOOST_CHECK_EQUAL(first.trigger_preliminary_selected_count,
                    second.trigger_preliminary_selected_count);
  BOOST_CHECK_EQUAL(first.association_sha256, second.association_sha256);
}

std::vector<point3D_t> AddDeterministicPoints(
    Reconstruction* reconstruction, const bool alternate_track_order) {
  const std::array<Eigen::Vector3d, 3> xyzs{{
      Eigen::Vector3d(1.0, 2.0, 3.0),
      Eigen::Vector3d(4.0, 5.0, 6.0),
      Eigen::Vector3d(7.0, 8.0, 9.0),
  }};
  std::vector<point3D_t> point3D_ids;
  for (size_t index = 0; index < xyzs.size(); ++index) {
    const point2D_t point2D_idx = static_cast<point2D_t>(10 + index);
    std::vector<TrackElement> elements;
    if (alternate_track_order) {
      elements = {{2, point2D_idx}, {3, point2D_idx}, {1, point2D_idx}};
    } else {
      elements = {{3, point2D_idx}, {1, point2D_idx}, {2, point2D_idx}};
    }
    point3D_ids.push_back(AddPoint(reconstruction,
                                   xyzs[index],
                                   std::move(elements),
                                   static_cast<int>(index)));
  }
  return point3D_ids;
}

struct OnlineLidarAssociationFixture {
  OnlineLidarAssociationFixture()
      : reconstruction(CreateReconstruction()),
        snapshot(SnapshotFromPoints({
            WorldPoint{{1.0f, 2.0f, 3.0f}},
            WorldPoint{{4.0f, 5.0f, 6.0f}},
            WorldPoint{{7.0f, 8.0f, 9.0f}},
        })) {}

  OnlineLidarAssociationRequest Request(
      const std::vector<point3D_t>& point3D_ids) const {
    OnlineLidarAssociationRequest request;
    request.attempt_id = 17;
    request.pass_index = 3;
    request.trigger_image_id = 2;
    request.ordered_frozen_image_ids = {1, 3, 2};
    request.point3D_ids = point3D_ids;
    request.options.local_lidar_kdtree_only = true;
    request.options.kdtree_max_search_range = 0.6;
    request.options.kdtree_min_search_range = 0.25;
    request.options.search_range_drop_speed = 0.1;
    SetSnapshotIdentity(snapshot, &request);
    return request;
  }

  OnlineLidarAssociationRequest ProjectionRequest(
      const std::vector<point3D_t>& point3D_ids) const {
    OnlineLidarAssociationRequest request = Request(point3D_ids);
    request.options.local_lidar_kdtree_only = false;
    request.options.min_proj_num = 100;
    request.projection_options = ProjectionOptions();
    return request;
  }

  void EnablePairGraph() {
    for (const auto& image : reconstruction.Images()) {
      const auto result = correspondence_graph.TryAddImage(
          image.first, image.second.NumPoints2D());
      if (!result.IsSuccess()) {
        throw std::runtime_error("cannot add association graph image");
      }
    }
    reconstruction.SetUp(&correspondence_graph);
  }

  void AddPair(const image_t first_image_id,
               const image_t second_image_id,
               const size_t count) {
    FeatureMatches matches;
    matches.reserve(count);
    for (point2D_t index = 0; index < count; ++index) {
      matches.emplace_back(static_cast<point2D_t>(100 + index),
                           static_cast<point2D_t>(100 + index));
    }
    const auto result = correspondence_graph.TryAddCorrespondences(
        first_image_id, second_image_id, matches);
    if (!result.IsSuccess() || result.num_added_matches != count ||
        !reconstruction.AddImagePairFromCorrespondenceGraph(first_image_id,
                                                             second_image_id)) {
      throw std::runtime_error("cannot add association image pair");
    }
  }

  CorrespondenceGraph correspondence_graph;
  Reconstruction reconstruction;
  std::shared_ptr<const lidar::LidarMapSnapshot> snapshot;
};

template <typename...>
using VoidT = void;

template <typename T, typename = void>
struct HasTriggerActualSelectedCount : std::false_type {};

template <typename T>
struct HasTriggerActualSelectedCount<
    T,
    VoidT<decltype(std::declval<T&>().trigger_actual_selected_count)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasActualSelectedCountByImage : std::false_type {};

template <typename T>
struct HasActualSelectedCountByImage<
    T,
    VoidT<decltype(std::declval<T&>().actual_selected_count_by_image)>>
    : std::true_type {};

static_assert(
    !HasTriggerActualSelectedCount<OnlineLidarAssociationAudit>::value,
    "association audit must expose preliminary, not actual, trigger count");
static_assert(
    !HasActualSelectedCountByImage<OnlineLidarAssociationAudit>::value,
    "association audit must expose preliminary, not actual, per-image counts");

}  // namespace

BOOST_AUTO_TEST_CASE(RouteBoundaryAndInvalidArguments) {
  OnlineLidarAssociationOptions options;
  options.min_proj_num = 4;
  OnlineLidarAssociationRoute route = OnlineLidarAssociationRoute::KDTREE;
  std::string error = "stale";

  BOOST_REQUIRE(SelectOnlineLidarAssociationRoute(
      options, options.min_proj_num + 2, &route, &error));
  BOOST_CHECK(route == OnlineLidarAssociationRoute::PROJECTION);
  BOOST_CHECK(error.empty());

  BOOST_REQUIRE(SelectOnlineLidarAssociationRoute(
      options, options.min_proj_num + 3, &route, &error));
  BOOST_CHECK(route == OnlineLidarAssociationRoute::KDTREE);
  BOOST_CHECK(error.empty());

  options.local_lidar_kdtree_only = true;
  BOOST_REQUIRE(SelectOnlineLidarAssociationRoute(options, 0, &route, &error));
  BOOST_CHECK(route == OnlineLidarAssociationRoute::KDTREE);

  options.local_lidar_kdtree_only = false;
  options.min_proj_num = -1;
  route = OnlineLidarAssociationRoute::PROJECTION;
  BOOST_CHECK(!SelectOnlineLidarAssociationRoute(options, 0, &route, &error));
  BOOST_CHECK(!error.empty());
  BOOST_CHECK(route == OnlineLidarAssociationRoute::PROJECTION);

  options.min_proj_num = 1;
  error = "stale";
  BOOST_CHECK(!SelectOnlineLidarAssociationRoute(options, 0, nullptr, &error));
  BOOST_CHECK(!error.empty());
  route = OnlineLidarAssociationRoute::KDTREE;
  BOOST_CHECK(!SelectOnlineLidarAssociationRoute(options, 0, &route, nullptr));
  BOOST_CHECK(route == OnlineLidarAssociationRoute::KDTREE);
}

BOOST_FIXTURE_TEST_CASE(TriggerGateUsesValidPlanesAndPreservesFullAssociations,
                        OnlineLidarAssociationFixture) {
  const point3D_t trigger_hit = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}, {1, 0}});
  const point3D_t historical_hit = AddPoint(
      &reconstruction, Eigen::Vector3d(4.0, 5.0, 6.0), {{1, 1}, {3, 1}});
  const point3D_t trigger_miss = AddPoint(
      &reconstruction, Eigen::Vector3d(100.0, 100.0, 100.0), {{2, 2}, {3, 2}});
  const auto request = Request({trigger_hit, historical_hit, trigger_miss});
  OnlineLidarAssociationOutput original;
  OnlineLidarAssociationAudit original_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &original, &original_audit, &error), error);

  OnlineLidarAssociationTriggerGate gate;
  gate.minimum_count = 2;
  OnlineLidarAssociationOutput output = original;
  OnlineLidarAssociationAudit audit = original_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate), error);
  BOOST_CHECK(gate.checked);
  BOOST_CHECK(!gate.passed);
  BOOST_CHECK_EQUAL(gate.actual_count, 1);
  BOOST_CHECK_EQUAL(gate.nearest_query_count, 2);
  BOOST_CHECK(output.associations.empty());
  BOOST_CHECK(audit.association_sha256.empty());

  gate.minimum_count = 1;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate), error);
  BOOST_CHECK(gate.passed);
  BOOST_CHECK_EQUAL(output.associations.size(), original.associations.size());
  BOOST_CHECK_EQUAL(audit.association_sha256, original_audit.association_sha256);

  gate.previous_count = 1;
  gate.minimum_increase = 1;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate), error);
  BOOST_CHECK(!gate.passed);
  gate.previous_count = 0;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate), error);
  BOOST_CHECK(gate.passed);
  BOOST_CHECK_EQUAL(audit.association_sha256, original_audit.association_sha256);
  gate.previous_count = std::numeric_limits<uint64_t>::max();
  BOOST_CHECK(!gate.Accepts(1));
}

BOOST_FIXTURE_TEST_CASE(RejectedTriggerDefersUnrelatedPointPreparation,
                        OnlineLidarAssociationFixture) {
  const point3D_t trigger_hit = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}, {1, 0}});
  const point3D_t historical_hit = AddPoint(
      &reconstruction, Eigen::Vector3d(4.0, 5.0, 6.0), {{1, 1}, {3, 1}});
  reconstruction.Point3D(historical_hit).Track().AddElement(1, 1);
  const auto request = Request({historical_hit, trigger_hit});
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  OnlineLidarAssociationTriggerGate gate;
  gate.minimum_count = 2;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate), error);
  BOOST_CHECK(gate.checked);
  BOOST_CHECK(!gate.passed);
  BOOST_CHECK_EQUAL(gate.actual_count, 1);
  BOOST_CHECK_EQUAL(gate.nearest_query_count, 1);
  CheckOutputCleared(output);
  CheckAuditCleared(audit);
  gate.minimum_count = 1;
  BOOST_CHECK(!BuildOnlineLidarAssociations(
      reconstruction, request, &output, &audit, &error, &gate));
  BOOST_CHECK(gate.passed);
  BOOST_CHECK(error.find("same window observation twice") != std::string::npos);
  CheckOutputCleared(output);
}

BOOST_FIXTURE_TEST_CASE(KdFieldsOwnerRangeAndInputState,
                        OnlineLidarAssociationFixture) {
  const point3D_t first_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(1.0, 2.0, 3.0),
      {{3, 4}, {1, 5}, {2, 9}},
      2);
  const point3D_t second_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(4.0, 5.0, 6.0),
      {{1, 6}, {2, 8}, {3, 7}},
      10);
  const Point3D& const_point = reconstruction.Point3D(first_id);
  BOOST_CHECK_EQUAL(const_point.GlobalOptNum(), 2);

  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  const OnlineLidarAssociationRequest request = Request({second_id, first_id});
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  BOOST_CHECK(error.empty());
  CheckReconstructionStateEqual(before, reconstruction);

  BOOST_REQUIRE_EQUAL(output.associations.size(), 2);
  const OnlineLidarAssociation& association = output.associations[0];
  BOOST_CHECK_EQUAL(association.association_id, 0);
  BOOST_CHECK_EQUAL(association.point3D_id, first_id);
  BOOST_CHECK_EQUAL(association.attempt_id, request.attempt_id);
  BOOST_CHECK_EQUAL(association.pass_index, request.pass_index);
  BOOST_CHECK_EQUAL(association.point3D_xyz[0], 1.0);
  BOOST_CHECK_EQUAL(association.point3D_xyz[1], 2.0);
  BOOST_CHECK_EQUAL(association.point3D_xyz[2], 3.0);
  BOOST_CHECK_EQUAL(association.owner_image_id, request.trigger_image_id);
  BOOST_CHECK_EQUAL(association.owner_point2D_idx, 9);
  BOOST_CHECK(association.route == OnlineLidarAssociationRoute::KDTREE);
  const lidar::VoxelKey expected_plane_key{100, 200, 300};
  BOOST_CHECK(association.plane_key == expected_plane_key);
  BOOST_CHECK(association.plane.key == association.plane_key);
  BOOST_CHECK(association.plane.frame ==
              lidar::LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK(association.plane.scale == lidar::LidarNormalScale::BA);
  BOOST_CHECK_EQUAL(association.plane.point[0], 1.0f);
  BOOST_CHECK_EQUAL(association.plane.point[1], 2.0f);
  BOOST_CHECK_EQUAL(association.plane.point[2], 3.0f);
  BOOST_CHECK_EQUAL(association.plane.normal[0], 0.0f);
  BOOST_CHECK_EQUAL(association.plane.normal[1], 1.0f);
  BOOST_CHECK_EQUAL(association.plane.normal[2], 0.0f);
  BOOST_CHECK_EQUAL(association.plane.curvature, 0.25f);
  BOOST_CHECK_EQUAL(association.plane.normal_revision, snapshot->Version());
  BOOST_CHECK_EQUAL(association.plane.voxel_count, 1);
  BOOST_CHECK_EQUAL(association.plane_abcd[0], 0.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[1], 1.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[2], 0.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[3], -2.0);
  BOOST_CHECK(association.has_search_range);
  BOOST_CHECK_SMALL(std::abs(association.search_range - 0.4), 1e-12);
  BOOST_CHECK(association.lidar_point_type == LidarPointType::IcpGround);
  BOOST_CHECK_EQUAL(association.projection_camera_distance, 0.0);
  BOOST_CHECK_EQUAL(association.projection_angle_score, 0.0);
  BOOST_CHECK_EQUAL(association.map_version, snapshot->Version());
  BOOST_CHECK_EQUAL(association.max_scan_index, snapshot->MaxScanIndex());
  BOOST_CHECK_EQUAL(association.snapshot_sha256, snapshot->SnapshotSha256());
  BOOST_CHECK_EQUAL(association.geometry_sha256, snapshot->GeometrySha256());

  BOOST_CHECK_EQUAL(output.associations[1].association_id, 1);
  BOOST_CHECK_EQUAL(output.associations[1].point3D_id, second_id);
  BOOST_CHECK_SMALL(
      std::abs(output.associations[1].search_range - 0.25), 1e-12);
  BOOST_CHECK(output.associations[1].lidar_point_type ==
              LidarPointType::IcpGround);

  lidar::LidarVoxelRecord record;
  BOOST_REQUIRE(snapshot->FindVoxel(association.plane_key, &record));
  BOOST_CHECK_EQUAL(record.count, association.plane.voxel_count);
  BOOST_CHECK(record.centroid == association.plane.point);
  BOOST_CHECK(record.inner_normal.normal == association.plane.normal);
  BOOST_CHECK_EQUAL(record.inner_normal.curvature,
                    association.plane.curvature);
  BOOST_CHECK_EQUAL(record.inner_normal.revision,
                    association.plane.normal_revision);
  BOOST_CHECK(record.outer_normal.normal != record.inner_normal.normal);

  BOOST_CHECK_EQUAL(audit.input_point_count, 2);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 2);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 2);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_selected_association_count, 2);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 0);
  BOOST_CHECK(audit.projection_calls.empty());
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(1), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(3), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(2), 2);
  BOOST_CHECK_EQUAL(audit.trigger_preliminary_selected_count, 2);
  BOOST_CHECK(IsLowercaseSha256(audit.association_sha256));
}

BOOST_FIXTURE_TEST_CASE(DistinctSameImageFeaturesUseStableAssociationOwner,
                        OnlineLidarAssociationFixture) {
  const auto plane_snapshot = SnapshotFromPoints({WorldPoint{{0.0f, 0.0f, 2.0f}}});
  SetIdentityProjectionPose(&reconstruction, {2});
  SetFeature(&reconstruction, 2, 0, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 1, 50.0, 40.0);
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 1}, {2, 0}});
  for (const bool kdtree_only : {false, true}) {
    OnlineLidarAssociationRequest request = Request({point3D_id});
    request.options.local_lidar_kdtree_only = kdtree_only;
    SetSnapshotIdentity(plane_snapshot, &request);
    OnlineLidarAssociationOutput first;
    OnlineLidarAssociationAudit first_audit;
    std::string error;
    BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                              reconstruction, request, &first, &first_audit,
                              &error), error);
    BOOST_REQUIRE_EQUAL(first.associations.size(), 1);
    BOOST_CHECK_EQUAL(first.associations.front().owner_image_id, 2);
    BOOST_CHECK_EQUAL(first.associations.front().owner_point2D_idx, 0);
    BOOST_CHECK_EQUAL(first_audit.trigger_preliminary_selected_count, 1);
    reconstruction.Point3D(point3D_id).Track().SetElements({{2, 0}, {2, 1}});
    OnlineLidarAssociationOutput reordered;
    OnlineLidarAssociationAudit reordered_audit;
    BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                              reconstruction, request, &reordered,
                              &reordered_audit, &error), error);
    BOOST_CHECK_EQUAL(first_audit.association_sha256,
                      reordered_audit.association_sha256);
    BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 2);
  }
}

BOOST_FIXTURE_TEST_CASE(KdSkipsAreSuccessfulAndEmptyOutputIsHashed,
                        OnlineLidarAssociationFixture) {
  const point3D_t no_window_id = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{4, 0}});
  const point3D_t no_hit_id = AddPoint(
      &reconstruction, Eigen::Vector3d(20.0, 20.0, 20.0), {{2, 0}});
  OnlineLidarAssociationRequest request = Request({no_hit_id, no_window_id});
  request.options.kdtree_max_search_range = 0.3;
  request.options.kdtree_min_search_range = 0.1;

  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  BOOST_CHECK(error.empty());
  BOOST_CHECK(output.associations.empty());
  BOOST_CHECK_EQUAL(audit.input_point_count, 2);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 2);
  BOOST_CHECK_EQUAL(audit.skipped_no_window_observation_count, 1);
  BOOST_CHECK_EQUAL(audit.skipped_pair_threshold_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 1);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 2);
  BOOST_CHECK_EQUAL(audit.kdtree_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(1), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(3), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(2), 0);
  BOOST_CHECK_EQUAL(audit.trigger_preliminary_selected_count, 0);
  BOOST_CHECK(IsLowercaseSha256(audit.association_sha256));
}

BOOST_FIXTURE_TEST_CASE(
    KdNoPlaneHashTracksFrozenPointInputsAndIgnoresStorageOrder,
    OnlineLidarAssociationFixture) {
  const point3D_t first_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(20.0, -0.0, 20.0),
      {{3, 10}, {2, 10}, {1, 10}},
      2);
  const point3D_t second_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(30.0, 30.0, 30.0),
      {{2, 11}, {1, 11}},
      1);
  OnlineLidarAssociationRequest request = Request({second_id, first_id});
  request.options.kdtree_max_search_range = 0.3;
  request.options.kdtree_min_search_range = 0.1;

  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  BOOST_REQUIRE(output.associations.empty());
  BOOST_CHECK_EQUAL(audit.input_point_count, 2);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 2);

  Reconstruction reordered = reconstruction;
  reordered.Point3D(first_id).SetXYZ(Eigen::Vector3d(20.0, 0.0, 20.0));
  reordered.Point3D(first_id)
      .Track()
      .SetElements({{1, 10}, {3, 10}, {2, 10}});
  reordered.Point3D(second_id)
      .Track()
      .SetElements({{1, 11}, {2, 11}});
  OnlineLidarAssociationRequest reordered_request = request;
  reordered_request.point3D_ids = {first_id, second_id};
  OnlineLidarAssociationOutput reordered_output;
  OnlineLidarAssociationAudit reordered_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reordered,
                                                     reordered_request,
                                                     &reordered_output,
                                                     &reordered_audit,
                                                     &error),
                        error);
  CheckOutputEqual(output, reordered_output);
  CheckAuditEqual(audit, reordered_audit);

  Reconstruction moved = reconstruction;
  moved.Point3D(first_id).SetXYZ(Eigen::Vector3d(20.5, -0.0, 20.0));
  OnlineLidarAssociationOutput moved_output;
  OnlineLidarAssociationAudit moved_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            moved, request, &moved_output, &moved_audit, &error),
                        error);
  CheckOutputEqual(output, moved_output);
  BOOST_CHECK_EQUAL(audit.input_point_count, moved_audit.input_point_count);
  BOOST_CHECK_EQUAL(audit.selected_association_count,
                    moved_audit.selected_association_count);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count,
                    moved_audit.skipped_no_plane_count);
  const std::string moved_sha256 = moved_audit.association_sha256;
  moved_audit.association_sha256 = audit.association_sha256;
  CheckAuditEqual(audit, moved_audit);
  BOOST_CHECK_NE(audit.association_sha256, moved_sha256);
}

BOOST_FIXTURE_TEST_CASE(CanonicalAssociationHashMatchesGoldenDigest,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(40.0, -0.0, 40.0),
      {{2, 7}},
      2);
  OnlineLidarAssociationRequest request = Request({point3D_id});
  request.options.kdtree_max_search_range = 0.5;
  request.options.kdtree_min_search_range = 0.25;
  request.options.search_range_drop_speed = 0.0;

  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  BOOST_REQUIRE(output.associations.empty());
  BOOST_REQUIRE_EQUAL(audit.skipped_no_plane_count, 1);
  BOOST_CHECK_EQUAL(
      audit.association_sha256,
      "5fc06a8137271f574b67628ee0392c4b08aff0a1df22161e75406ea51ec2f5fb");
}

BOOST_FIXTURE_TEST_CASE(ProjectionFieldsZeroAngleAuditAndInputState,
                        OnlineLidarAssociationFixture) {
  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto projection_snapshot = SnapshotFromPoints(
      {plane_point},
      std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.125f}},
      std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.25f}});
  SetIdentityProjectionPose(&reconstruction, {2});
  SetFeature(&reconstruction, 2, 7, 50.0, 40.0);
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 7}}, 4);
  OnlineLidarAssociationRequest request = ProjectionRequest({point3D_id});
  SetSnapshotIdentity(projection_snapshot, &request);

  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  BOOST_CHECK(error.empty());
  CheckReconstructionStateEqual(before, reconstruction);

  BOOST_REQUIRE_EQUAL(output.associations.size(), 1);
  const OnlineLidarAssociation& association = output.associations.front();
  BOOST_CHECK_EQUAL(association.association_id, 0);
  BOOST_CHECK_EQUAL(association.attempt_id, request.attempt_id);
  BOOST_CHECK_EQUAL(association.pass_index, request.pass_index);
  BOOST_CHECK_EQUAL(association.point3D_id, point3D_id);
  BOOST_CHECK_EQUAL(association.point3D_xyz[0], 0.0);
  BOOST_CHECK_EQUAL(association.point3D_xyz[1], 0.0);
  BOOST_CHECK_EQUAL(association.point3D_xyz[2], 2.0);
  BOOST_CHECK_EQUAL(association.owner_image_id, 2);
  BOOST_CHECK_EQUAL(association.owner_point2D_idx, 7);
  BOOST_CHECK(association.route == OnlineLidarAssociationRoute::PROJECTION);
  BOOST_CHECK(association.plane_key == WorldVoxelKey(plane_point));
  BOOST_CHECK(association.plane.key == association.plane_key);
  BOOST_CHECK(association.plane.frame ==
              lidar::LidarCoordinateFrame::COLMAP_WORLD);
  BOOST_CHECK(association.plane.scale ==
              lidar::LidarNormalScale::OUTER_0_15_M);
  BOOST_CHECK(association.plane.scale ==
              lidar::LidarNormalScale::PROJECTION);
  BOOST_CHECK_EQUAL(association.plane.point[0], plane_point[0]);
  BOOST_CHECK_EQUAL(association.plane.point[1], plane_point[1]);
  BOOST_CHECK_EQUAL(association.plane.point[2], plane_point[2]);
  BOOST_CHECK_EQUAL(association.plane.normal[0], 1.0f);
  BOOST_CHECK_EQUAL(association.plane.normal[1], 0.0f);
  BOOST_CHECK_EQUAL(association.plane.normal[2], 0.0f);
  BOOST_CHECK_EQUAL(association.plane.curvature, 0.125f);
  BOOST_CHECK_EQUAL(association.plane.normal_revision,
                    projection_snapshot->Version());
  BOOST_CHECK_EQUAL(association.plane.voxel_count, 1);
  BOOST_CHECK_EQUAL(association.plane_abcd[0], 1.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[1], 0.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[2], 0.0);
  BOOST_CHECK_EQUAL(association.plane_abcd[3], 0.0);
  BOOST_CHECK(!association.has_search_range);
  BOOST_CHECK_EQUAL(association.search_range, 0.0);
  BOOST_CHECK(association.lidar_point_type == LidarPointType::Proj);
  BOOST_CHECK_GT(association.projection_camera_distance, 0.0);
  BOOST_CHECK_EQUAL(association.projection_camera_distance, 2.0);
  BOOST_CHECK_EQUAL(association.projection_angle_score, 0.0);
  BOOST_CHECK_EQUAL(association.map_version, projection_snapshot->Version());
  BOOST_CHECK_EQUAL(association.max_scan_index,
                    projection_snapshot->MaxScanIndex());
  BOOST_CHECK_EQUAL(association.snapshot_sha256,
                    projection_snapshot->SnapshotSha256());
  BOOST_CHECK_EQUAL(association.geometry_sha256,
                    projection_snapshot->GeometrySha256());

  BOOST_CHECK_EQUAL(audit.attempt_id, request.attempt_id);
  BOOST_CHECK_EQUAL(audit.pass_index, request.pass_index);
  BOOST_CHECK_EQUAL(audit.trigger_image_id, request.trigger_image_id);
  BOOST_CHECK_EQUAL(audit.map_version, projection_snapshot->Version());
  BOOST_CHECK_EQUAL(audit.max_scan_index,
                    projection_snapshot->MaxScanIndex());
  BOOST_CHECK_EQUAL(audit.snapshot_sha256,
                    projection_snapshot->SnapshotSha256());
  BOOST_CHECK_EQUAL(audit.geometry_sha256,
                    projection_snapshot->GeometrySha256());
  BOOST_CHECK_EQUAL(audit.input_point_count, 1);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 1);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_window_observation_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_pair_threshold_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 0);
  BOOST_CHECK_EQUAL(audit.pair_threshold_skipped_observation_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 1);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 1);
  BOOST_CHECK_EQUAL(audit.kdtree_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 1);
  BOOST_REQUIRE_EQUAL(audit.projection_calls.size(), 1);
  BOOST_CHECK_EQUAL(audit.projection_calls.front().image_id, 2);
  const lidar::SnapshotProjectionAudit& projection_audit =
      audit.projection_calls.front().projection;
  BOOST_CHECK_EQUAL(projection_audit.map_version,
                    projection_snapshot->Version());
  BOOST_CHECK_EQUAL(projection_audit.snapshot_sha256,
                    projection_snapshot->SnapshotSha256());
  BOOST_CHECK_EQUAL(projection_audit.geometry_sha256,
                    projection_snapshot->GeometrySha256());
  BOOST_CHECK_EQUAL(projection_audit.snapshot_voxel_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.snapshot_block_count, 1);
  BOOST_CHECK(!projection_audit.full_snapshot_due_to_distortion);
  BOOST_CHECK(projection_audit.projection_aabb.frame ==
              lidar::LidarCoordinateFrame::COLMAP_WORLD);
  for (size_t axis = 0; axis < 3; ++axis) {
    BOOST_CHECK_LE(projection_audit.projection_aabb.min[axis],
                   static_cast<double>(plane_point[axis]));
    BOOST_CHECK_GE(projection_audit.projection_aabb.max[axis],
                   static_cast<double>(plane_point[axis]));
  }
  BOOST_CHECK_EQUAL(projection_audit.input_feature_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.in_image_feature_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.aabb_visited_block_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.candidate_plane_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.positive_depth_hit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.axial_depth_hit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.coverage_pixel_visit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.feature_coverage_hit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.feature_pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(projection_audit.feature_hit_count, 1);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(1), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(3), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(2), 1);
  BOOST_CHECK_EQUAL(audit.trigger_preliminary_selected_count, 1);
  BOOST_CHECK(IsLowercaseSha256(audit.association_sha256));
}

BOOST_FIXTURE_TEST_CASE(ProjectionCompleteTiesUseCanonicalOwnerAndPlaneKey,
                        OnlineLidarAssociationFixture) {
  const WorldPoint left{{-0.4f, 0.0f, 2.0f}};
  const WorldPoint right{{0.4f, 0.0f, 2.0f}};
  const WorldPoint center_left{{-0.004f, 0.0f, 2.0f}};
  const WorldPoint center_right{{0.004f, 0.0f, 2.0f}};
  const auto tie_snapshot = SnapshotFromPoints(
      {right, center_right, left, center_left},
      std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.375f}},
      std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.5f}});
  SetIdentityProjectionPose(&reconstruction, {1, 2, 3});
  SetFeature(&reconstruction, 1, 0, 30.0, 40.0);
  SetFeature(&reconstruction, 3, 0, 70.0, 40.0);
  SetFeature(&reconstruction, 2, 1, 50.0, 40.0);
  const point3D_t owner_tie_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(0.0, 0.0, 2.0),
      {{3, 0}, {1, 0}});
  const point3D_t plane_tie_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 1}});

  OnlineLidarAssociationRequest request =
      ProjectionRequest({plane_tie_id, owner_tie_id});
  request.ordered_frozen_image_ids = {3, 2, 1};
  SetSnapshotIdentity(tie_snapshot, &request);
  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  CheckReconstructionStateEqual(before, reconstruction);

  BOOST_REQUIRE_EQUAL(output.associations.size(), 2);
  BOOST_CHECK_EQUAL(output.associations[0].point3D_id, owner_tie_id);
  BOOST_CHECK_EQUAL(output.associations[0].owner_image_id, 1);
  BOOST_CHECK_EQUAL(output.associations[0].owner_point2D_idx, 0);
  BOOST_CHECK(output.associations[0].plane_key == WorldVoxelKey(left));
  BOOST_CHECK_EQUAL(output.associations[0].projection_angle_score, 0.0);
  BOOST_CHECK_EQUAL(output.associations[1].point3D_id, plane_tie_id);
  BOOST_CHECK_EQUAL(output.associations[1].owner_image_id, 2);
  BOOST_CHECK_EQUAL(output.associations[1].owner_point2D_idx, 1);
  BOOST_CHECK(output.associations[1].plane_key ==
              WorldVoxelKey(center_left));
  BOOST_CHECK(output.associations[1].plane_key <
              WorldVoxelKey(center_right));
  BOOST_CHECK_EQUAL(output.associations[1].projection_angle_score, 0.0);
  BOOST_REQUIRE_EQUAL(audit.projection_calls.size(), 3);
  BOOST_CHECK_EQUAL(audit.projection_calls[0].image_id, 3);
  BOOST_CHECK_EQUAL(audit.projection_calls[1].image_id, 2);
  BOOST_CHECK_EQUAL(audit.projection_calls[2].image_id, 1);

  Reconstruction reordered = reconstruction;
  reordered.Point3D(owner_tie_id)
      .Track()
      .SetElements({{1, 0}, {3, 0}});
  OnlineLidarAssociationOutput reordered_output;
  OnlineLidarAssociationAudit reordered_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reordered,
                                                     request,
                                                     &reordered_output,
                                                     &reordered_audit,
                                                     &error),
                        error);
  CheckOutputEqual(output, reordered_output);
  CheckAuditEqual(audit, reordered_audit);
}

BOOST_FIXTURE_TEST_CASE(ProjectionAngleScoreChoosesSmallerLegacyFormulaValue,
                        OnlineLidarAssociationFixture) {
  const WorldPoint aligned{{0.4f, 0.0f, 2.0f}};
  const WorldPoint orthogonal{{0.0f, 0.4f, 2.0f}};
  const auto score_snapshot = SnapshotFromPoints(
      {orthogonal, aligned},
      std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.125f}},
      std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.25f}});
  SetIdentityProjectionPose(&reconstruction, {1, 2, 3});
  SetFeature(&reconstruction, 1, 0, 70.0, 40.0);
  SetFeature(&reconstruction, 3, 0, 50.0, 60.0);
  const point3D_t point3D_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(0.0, 0.0, 2.0),
      {{1, 0}, {3, 0}});

  OnlineLidarAssociationRequest aligned_only =
      ProjectionRequest({point3D_id});
  aligned_only.ordered_frozen_image_ids = {2, 1};
  SetSnapshotIdentity(score_snapshot, &aligned_only);
  OnlineLidarAssociationOutput aligned_output;
  OnlineLidarAssociationAudit aligned_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     aligned_only,
                                                     &aligned_output,
                                                     &aligned_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(aligned_output.associations.size(), 1);
  BOOST_CHECK_EQUAL(aligned_output.associations.front().owner_image_id, 1);
  BOOST_CHECK_CLOSE(aligned_output.associations.front().projection_angle_score,
                    1.0,
                    1e-10);

  OnlineLidarAssociationRequest both = ProjectionRequest({point3D_id});
  both.ordered_frozen_image_ids = {2, 1, 3};
  SetSnapshotIdentity(score_snapshot, &both);
  OnlineLidarAssociationOutput both_output;
  OnlineLidarAssociationAudit both_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, both, &both_output, &both_audit,
                            &error),
                        error);
  BOOST_REQUIRE_EQUAL(both_output.associations.size(), 1);
  BOOST_CHECK_EQUAL(both_output.associations.front().owner_image_id, 3);
  BOOST_CHECK_EQUAL(both_output.associations.front().owner_point2D_idx, 0);
  BOOST_CHECK(both_output.associations.front().plane_key ==
              WorldVoxelKey(orthogonal));
  BOOST_CHECK_EQUAL(both_output.associations.front().projection_angle_score,
                    0.0);
  BOOST_CHECK_LT(both_output.associations.front().projection_angle_score,
                 aligned_output.associations.front().projection_angle_score);
}

BOOST_FIXTURE_TEST_CASE(ProjectionPairGateCoversBypassAndStrictThreshold,
                        OnlineLidarAssociationFixture) {
  AddRegisteredImage(&reconstruction, 5);
  SetIdentityProjectionPose(&reconstruction, {1, 2, 3, 4, 5});
  for (const image_t image_id : {1, 2, 3, 4, 5}) {
    SetFeature(&reconstruction, image_id, 0, 50.0, 40.0);
  }
  SetFeature(&reconstruction, 1, 1, 50.0, 40.0);

  const point3D_t trigger_bypass_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(0.0, 0.0, 2.0),
      {{1, 1}, {2, 0}});
  const point3D_t below_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{1, 0}});
  const point3D_t equal_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{3, 0}});
  const point3D_t above_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{4, 0}});
  const point3D_t absent_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{5, 0}});

  EnablePairGraph();
  constexpr size_t kThreshold = 2;
  AddPair(2, 1, kThreshold - 1);
  AddPair(2, 3, kThreshold);
  AddPair(2, 4, kThreshold + 1);
  BOOST_CHECK(!reconstruction.ExistsImagePair(2, 5));

  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto gate_snapshot = SnapshotFromPoints({plane_point});
  OnlineLidarAssociationRequest request = ProjectionRequest(
      {absent_id, equal_id, above_id, below_id, trigger_bypass_id});
  request.ordered_frozen_image_ids = {1, 2, 3, 4, 5};
  request.options.ba_match_features_threshold = kThreshold;
  SetSnapshotIdentity(gate_snapshot, &request);

  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  CheckReconstructionStateEqual(before, reconstruction);

  BOOST_REQUIRE_EQUAL(output.associations.size(), 3);
  std::map<point3D_t, image_t> owners;
  for (const OnlineLidarAssociation& association : output.associations) {
    owners.emplace(association.point3D_id, association.owner_image_id);
  }
  BOOST_CHECK_EQUAL(owners.at(trigger_bypass_id), 2);
  BOOST_CHECK_EQUAL(owners.at(above_id), 4);
  BOOST_CHECK_EQUAL(owners.at(absent_id), 5);
  BOOST_CHECK_EQUAL(owners.count(below_id), 0);
  BOOST_CHECK_EQUAL(owners.count(equal_id), 0);
  BOOST_CHECK_EQUAL(audit.input_point_count, 5);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 3);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 2);
  BOOST_CHECK_EQUAL(audit.skipped_no_window_observation_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_pair_threshold_point_count, 2);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 0);
  BOOST_CHECK_EQUAL(audit.pair_threshold_skipped_observation_count, 3);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 5);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 3);
  BOOST_CHECK_EQUAL(audit.kdtree_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 3);
  BOOST_REQUIRE_EQUAL(audit.projection_calls.size(), 3);
  BOOST_CHECK_EQUAL(audit.projection_calls[0].image_id, 2);
  BOOST_CHECK_EQUAL(audit.projection_calls[1].image_id, 4);
  BOOST_CHECK_EQUAL(audit.projection_calls[2].image_id, 5);
  for (const OnlineLidarProjectionCallAudit& call : audit.projection_calls) {
    BOOST_CHECK_EQUAL(call.projection.input_feature_count, 1);
    BOOST_CHECK_EQUAL(call.projection.feature_hit_count, 1);
  }
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(1), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(2), 1);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(3), 0);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(4), 1);
  BOOST_CHECK_EQUAL(audit.preliminary_selected_count_by_image.at(5), 1);
  BOOST_CHECK_EQUAL(audit.trigger_preliminary_selected_count, 1);

  Reconstruction changed_pair_count = reconstruction;
  ++changed_pair_count.ImagePair(2, 4).num_total_corrs;
  OnlineLidarAssociationOutput changed_output;
  OnlineLidarAssociationAudit changed_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(changed_pair_count,
                                                     request,
                                                     &changed_output,
                                                     &changed_audit,
                                                     &error),
                        error);
  CheckOutputEqual(output, changed_output);
  const std::string changed_sha256 = changed_audit.association_sha256;
  changed_audit.association_sha256 = audit.association_sha256;
  CheckAuditEqual(audit, changed_audit);
  BOOST_CHECK_NE(audit.association_sha256, changed_sha256);
}

BOOST_FIXTURE_TEST_CASE(
    ProjectionBatchesEligiblePointsAndIsolatesFilteredImageCopy,
    OnlineLidarAssociationFixture) {
  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto filtered_snapshot = SnapshotFromPoints({plane_point});
  SetIdentityProjectionPose(&reconstruction, {1, 2, 3});
  SetFeature(&reconstruction, 2, 0, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 1, 50.0, 40.0);
  const point3D_t first_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 0}});
  const point3D_t second_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 1}});
  const point3D_t outside_window_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{4, 2}});
  const point3D_t non_request_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{4, 3}});

  reconstruction.Image(2).SetPoint3DForPoint2D(30, outside_window_id);
  reconstruction.Image(2).SetPoint3DForPoint2D(31, non_request_id);
  reconstruction.Image(2).SetPoint3DForPoint2D(32, non_request_id);
  reconstruction.Image(2).SetPoint3DForPoint2D(33, first_id);
  SetFeature(&reconstruction, 2, 30, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 31, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 32, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 33, 50.0, 40.0);

  OnlineLidarAssociationRequest request =
      ProjectionRequest({second_id, outside_window_id, first_id});
  request.ordered_frozen_image_ids = {1, 2, 3};
  SetSnapshotIdentity(filtered_snapshot, &request);
  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  CheckReconstructionStateEqual(before, reconstruction);

  BOOST_REQUIRE_EQUAL(output.associations.size(), 2);
  BOOST_CHECK_EQUAL(output.associations[0].point3D_id, first_id);
  BOOST_CHECK_EQUAL(output.associations[1].point3D_id, second_id);
  BOOST_CHECK_EQUAL(audit.input_point_count, 3);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 2);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 1);
  BOOST_CHECK_EQUAL(audit.skipped_no_window_observation_count, 1);
  BOOST_CHECK_EQUAL(audit.skipped_pair_threshold_point_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 3);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 2);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 1);
  BOOST_REQUIRE_EQUAL(audit.projection_calls.size(), 1);
  BOOST_CHECK_EQUAL(audit.projection_calls.front().image_id, 2);
  BOOST_CHECK_EQUAL(
      audit.projection_calls.front().projection.input_feature_count, 2);
  BOOST_CHECK_EQUAL(
      audit.projection_calls.front().projection.in_image_feature_count, 2);
  BOOST_CHECK_EQUAL(
      audit.projection_calls.front().projection.feature_pixel_hit_count, 1);
  BOOST_CHECK_EQUAL(
      audit.projection_calls.front().projection.feature_hit_count, 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(30).Point3DId(),
                    outside_window_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(31).Point3DId(),
                    non_request_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(32).Point3DId(),
                    non_request_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(33).Point3DId(), first_id);
}

BOOST_FIXTURE_TEST_CASE(BuildRejectsInvalidOptionsAndOutputPointers,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}});
  const OnlineLidarAssociationRequest valid_request = Request({point3D_id});

  OnlineLidarAssociationRequest request = valid_request;
  request.options.kdtree_min_search_range = 0.7;
  CheckBuildFailure(reconstruction, request, "exceeds");

  request = valid_request;
  request.projection_options.depth_image_scale =
      std::numeric_limits<double>::quiet_NaN();
  CheckBuildFailure(reconstruction, request, "non-finite");

  request = valid_request;
  request.snapshot.reset();
  CheckBuildFailure(reconstruction, request, "snapshot is null");

  OnlineLidarAssociationAudit audit;
  std::string error = "stale";
  PrimeAudit(&audit);
  BOOST_CHECK(!BuildOnlineLidarAssociations(
      reconstruction, valid_request, nullptr, &audit, &error));
  CheckAuditCleared(audit);
  BOOST_CHECK(!error.empty());

  OnlineLidarAssociationOutput output;
  PrimeOutput(&output);
  error = "stale";
  BOOST_CHECK(!BuildOnlineLidarAssociations(
      reconstruction, valid_request, &output, nullptr, &error));
  CheckOutputCleared(output);
  BOOST_CHECK(!error.empty());

  PrimeOutput(&output);
  PrimeAudit(&audit);
  BOOST_CHECK(!BuildOnlineLidarAssociations(
      reconstruction, valid_request, &output, &audit, nullptr));
  CheckOutputCleared(output);
  CheckAuditCleared(audit);
}

BOOST_FIXTURE_TEST_CASE(BuildRejectsInvalidWindowAndCameraInputs,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}});
  const OnlineLidarAssociationRequest valid_request = Request({point3D_id});

  OnlineLidarAssociationRequest request = valid_request;
  request.ordered_frozen_image_ids.clear();
  CheckBuildFailure(reconstruction, request, "window size");

  request = valid_request;
  request.ordered_frozen_image_ids = {1, 1};
  request.trigger_image_id = 1;
  CheckBuildFailure(reconstruction, request, "duplicate images");

  request = valid_request;
  request.ordered_frozen_image_ids.clear();
  for (image_t image_id = 1; image_id <= 21; ++image_id) {
    request.ordered_frozen_image_ids.push_back(image_id);
  }
  CheckBuildFailure(reconstruction, request, "window size");

  request = valid_request;
  request.trigger_image_id = 4;
  CheckBuildFailure(reconstruction, request, "exactly once");

  request = valid_request;
  request.ordered_frozen_image_ids = {1, 2, 99};
  CheckBuildFailure(reconstruction, request, "missing image");

  Reconstruction unregistered = reconstruction;
  unregistered.Image(1).SetRegistered(false);
  CheckBuildFailure(unregistered, valid_request, "unregistered image");

  Reconstruction no_camera = reconstruction;
  Image image_without_camera;
  image_without_camera.SetImageId(5);
  image_without_camera.SetName("image_without_camera");
  no_camera.AddImage(std::move(image_without_camera));
  no_camera.RegisterImage(5);
  request = valid_request;
  request.ordered_frozen_image_ids = {1, 2, 5};
  CheckBuildFailure(no_camera, request, "no existing camera");

  Reconstruction missing_camera = reconstruction;
  missing_camera.Image(1).SetCameraId(999);
  CheckBuildFailure(missing_camera, valid_request, "no existing camera");
}

BOOST_FIXTURE_TEST_CASE(BuildRejectsSnapshotIdentityMismatches,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}});
  const OnlineLidarAssociationRequest valid_request = Request({point3D_id});

  OnlineLidarAssociationRequest request = valid_request;
  ++request.expected_map_version;
  CheckBuildFailure(reconstruction, request, "version does not match");

  request = valid_request;
  ++request.expected_max_scan_index;
  CheckBuildFailure(reconstruction, request, "max scan index does not match");

  request = valid_request;
  request.expected_snapshot_sha256 = std::string(64, 'a');
  CheckBuildFailure(reconstruction, request, "snapshot SHA256");

  request = valid_request;
  request.expected_geometry_sha256 = std::string(64, 'b');
  CheckBuildFailure(reconstruction, request, "geometry SHA256");
}

BOOST_FIXTURE_TEST_CASE(BuildRejectsPointAndTrackIntegrityErrors,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(1.0, 2.0, 3.0),
      {{1, 0}, {2, 0}, {3, 0}});
  const OnlineLidarAssociationRequest valid_request = Request({point3D_id});

  OnlineLidarAssociationRequest request = valid_request;
  request.point3D_ids = {point3D_id, point3D_id};
  CheckBuildFailure(reconstruction, request, "duplicate point3D IDs");

  request = valid_request;
  request.point3D_ids = {kInvalidPoint3DId};
  CheckBuildFailure(reconstruction, request, "invalid point3D ID");

  request = valid_request;
  request.point3D_ids = {999};
  CheckBuildFailure(reconstruction, request, "missing Point3D");

  Reconstruction invalid_index = reconstruction;
  invalid_index.Point3D(point3D_id).Track().Element(0).point2D_idx =
      invalid_index.Image(1).NumPoints2D();
  CheckBuildFailure(invalid_index, valid_request, "invalid point2D index");

  Reconstruction wrong_reverse = reconstruction;
  wrong_reverse.Image(1).ResetPoint3DForPoint2D(0);
  CheckBuildFailure(wrong_reverse, valid_request, "does not point back");

  Reconstruction duplicate_image = reconstruction;
  duplicate_image.Point3D(point3D_id).Track().AddElement(1, 0);
  CheckBuildFailure(duplicate_image, valid_request,
                    "same window observation twice");
}

BOOST_FIXTURE_TEST_CASE(CanonicalOrderAndHashIgnoreInputAndTrackOrder,
                        OnlineLidarAssociationFixture) {
  const std::vector<point3D_t> point3D_ids =
      AddDeterministicPoints(&reconstruction, false);
  OnlineLidarAssociationRequest unsorted_request =
      Request({point3D_ids[2], point3D_ids[0], point3D_ids[1]});
  OnlineLidarAssociationRequest sorted_request =
      Request({point3D_ids[0], point3D_ids[1], point3D_ids[2]});

  OnlineLidarAssociationOutput unsorted_output;
  OnlineLidarAssociationOutput sorted_output;
  OnlineLidarAssociationAudit unsorted_audit;
  OnlineLidarAssociationAudit sorted_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     unsorted_request,
                                                     &unsorted_output,
                                                     &unsorted_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     sorted_request,
                                                     &sorted_output,
                                                     &sorted_audit,
                                                     &error),
                        error);
  CheckOutputEqual(unsorted_output, sorted_output);
  CheckAuditEqual(unsorted_audit, sorted_audit);
  BOOST_CHECK_EQUAL(unsorted_audit.association_sha256,
                    sorted_audit.association_sha256);

  BOOST_REQUIRE_EQUAL(sorted_output.associations.size(), point3D_ids.size());
  for (size_t index = 0; index < sorted_output.associations.size(); ++index) {
    BOOST_CHECK_EQUAL(sorted_output.associations[index].association_id, index);
    BOOST_CHECK_EQUAL(sorted_output.associations[index].point3D_id,
                      point3D_ids[index]);
  }

  Reconstruction equivalent = CreateReconstruction();
  const std::vector<point3D_t> equivalent_ids =
      AddDeterministicPoints(&equivalent, true);
  BOOST_REQUIRE(point3D_ids == equivalent_ids);
  OnlineLidarAssociationRequest equivalent_request =
      Request({equivalent_ids[1], equivalent_ids[2], equivalent_ids[0]});
  OnlineLidarAssociationOutput equivalent_output;
  OnlineLidarAssociationAudit equivalent_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(equivalent,
                                                     equivalent_request,
                                                     &equivalent_output,
                                                     &equivalent_audit,
                                                     &error),
                        error);
  CheckOutputEqual(sorted_output, equivalent_output);
  CheckAuditEqual(sorted_audit, equivalent_audit);
  BOOST_CHECK_EQUAL(sorted_audit.association_sha256,
                    equivalent_audit.association_sha256);

  Reconstruction semantically_changed = reconstruction;
  semantically_changed.Point3D(point3D_ids[0])
      .SetXYZ(Eigen::Vector3d(1.01, 2.0, 3.0));
  OnlineLidarAssociationOutput changed_output;
  OnlineLidarAssociationAudit changed_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(semantically_changed,
                                                     sorted_request,
                                                     &changed_output,
                                                     &changed_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(changed_output.associations.size(),
                      sorted_output.associations.size());
  BOOST_CHECK_NE(changed_output.associations[0].point3D_xyz[0],
                 sorted_output.associations[0].point3D_xyz[0]);
  BOOST_CHECK_NE(changed_audit.association_sha256,
                 sorted_audit.association_sha256);
  BOOST_CHECK(IsLowercaseSha256(changed_audit.association_sha256));
}

BOOST_FIXTURE_TEST_CASE(SnapshotRequestsRemainPinnedAcrossLaterAppend,
                        OnlineLidarAssociationFixture) {
  const WorldPoint first_plane{{0.0f, 0.0f, 2.0f}};
  const WorldPoint second_plane{{0.4f, 0.0f, 2.0f}};
  AssociationTempPcd first_scan;
  AssociationTempPcd second_scan;
  first_scan.Write({first_plane});
  second_scan.Write({second_plane});
  lidar::IncrementalCausalLidarMap map(AssociationDependencies(
      std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.125f}},
      std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.25f}}));
  std::string error;
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(ScanSource(1, first_scan.Path()), nullptr, &error), error);
  const std::shared_ptr<const lidar::LidarMapSnapshot> first_snapshot =
      map.GetSnapshot();
  BOOST_REQUIRE_MESSAGE(
      map.AppendScan(ScanSource(2, second_scan.Path()), nullptr, &error), error);
  const std::shared_ptr<const lidar::LidarMapSnapshot> second_snapshot =
      map.GetSnapshot();
  BOOST_REQUIRE(first_snapshot != nullptr);
  BOOST_REQUIRE(second_snapshot != nullptr);
  BOOST_CHECK_EQUAL(first_snapshot->Version(), 1);
  BOOST_CHECK_EQUAL(first_snapshot->MaxScanIndex(), 1);
  BOOST_CHECK_EQUAL(second_snapshot->Version(), 2);
  BOOST_CHECK_EQUAL(second_snapshot->MaxScanIndex(), 2);
  BOOST_CHECK_NE(first_snapshot->SnapshotSha256(),
                 second_snapshot->SnapshotSha256());
  BOOST_CHECK_NE(first_snapshot->GeometrySha256(),
                 second_snapshot->GeometrySha256());

  SetIdentityProjectionPose(&reconstruction, {2});
  SetFeature(&reconstruction, 2, 0, 50.0, 40.0);
  SetFeature(&reconstruction, 2, 1, 70.0, 40.0);
  const point3D_t first_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 0}});
  const point3D_t second_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.4, 0.0, 2.0), {{2, 1}});

  OnlineLidarAssociationRequest first_request = ProjectionRequest({first_id});
  first_request.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(first_snapshot, &first_request);
  OnlineLidarAssociationRequest second_request =
      ProjectionRequest({second_id});
  second_request.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(second_snapshot, &second_request);

  OnlineLidarAssociationOutput first_output;
  OnlineLidarAssociationAudit first_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     first_request,
                                                     &first_output,
                                                     &first_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(first_output.associations.size(), 1);
  BOOST_CHECK(first_output.associations.front().plane_key ==
              WorldVoxelKey(first_plane));
  BOOST_CHECK_EQUAL(first_output.associations.front().map_version, 1);
  BOOST_CHECK_EQUAL(first_audit.map_version, 1);

  OnlineLidarAssociationOutput second_output;
  OnlineLidarAssociationAudit second_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     second_request,
                                                     &second_output,
                                                     &second_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(second_output.associations.size(), 1);
  BOOST_CHECK(second_output.associations.front().plane_key ==
              WorldVoxelKey(second_plane));
  BOOST_CHECK(second_output.associations.front().plane_key !=
              WorldVoxelKey(first_plane));
  BOOST_CHECK_EQUAL(second_output.associations.front().map_version, 2);
  BOOST_CHECK_EQUAL(second_audit.map_version, 2);
  BOOST_CHECK_NE(first_audit.association_sha256,
                 second_audit.association_sha256);

  OnlineLidarAssociationOutput repeated_output;
  OnlineLidarAssociationAudit repeated_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     second_request,
                                                     &repeated_output,
                                                     &repeated_audit,
                                                     &error),
                        error);
  CheckOutputEqual(second_output, repeated_output);
  CheckAuditEqual(second_audit, repeated_audit);
  BOOST_CHECK_EQUAL(second_audit.association_sha256,
                    repeated_audit.association_sha256);
}

BOOST_FIXTURE_TEST_CASE(TriggerPreliminaryAuditCountsActual49And50Associations,
                        OnlineLidarAssociationFixture) {
  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto count_snapshot = SnapshotFromPoints({plane_point});
  SetIdentityProjectionPose(&reconstruction, {2});
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(50);
  for (point2D_t point2D_idx = 0; point2D_idx < 50; ++point2D_idx) {
    SetFeature(&reconstruction, 2, point2D_idx, 50.0, 40.0);
    point3D_ids.push_back(AddPoint(&reconstruction,
                                   Eigen::Vector3d(0.0, 0.0, 2.0),
                                   {{2, point2D_idx}}));
  }

  std::vector<point3D_t> first_49(point3D_ids.begin(),
                                  point3D_ids.begin() + 49);
  OnlineLidarAssociationRequest request_49 = ProjectionRequest(first_49);
  request_49.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(count_snapshot, &request_49);
  OnlineLidarAssociationOutput output_49;
  OnlineLidarAssociationAudit audit_49;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     request_49,
                                                     &output_49,
                                                     &audit_49,
                                                     &error),
                        error);
  BOOST_CHECK_EQUAL(output_49.associations.size(), 49);
  BOOST_CHECK_EQUAL(audit_49.input_point_count, 49);
  BOOST_CHECK_EQUAL(audit_49.selected_association_count, 49);
  BOOST_CHECK_EQUAL(audit_49.projection_selected_association_count, 49);
  BOOST_CHECK_EQUAL(audit_49.preliminary_selected_count_by_image.size(), 1);
  BOOST_CHECK_EQUAL(audit_49.preliminary_selected_count_by_image.at(2), 49);
  BOOST_CHECK_EQUAL(audit_49.trigger_preliminary_selected_count, 49);
  BOOST_CHECK_EQUAL(audit_49.projection_call_count, 1);
  BOOST_CHECK_EQUAL(
      audit_49.projection_calls.front().projection.input_feature_count, 49);

  OnlineLidarAssociationRequest request_50 = ProjectionRequest(point3D_ids);
  request_50.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(count_snapshot, &request_50);
  OnlineLidarAssociationOutput output_50;
  OnlineLidarAssociationAudit audit_50;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     request_50,
                                                     &output_50,
                                                     &audit_50,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(output_50.associations.size(), 50);
  BOOST_CHECK_EQUAL(audit_50.input_point_count, 50);
  BOOST_CHECK_EQUAL(audit_50.selected_association_count, 50);
  BOOST_CHECK_EQUAL(audit_50.projection_selected_association_count, 50);
  BOOST_CHECK_EQUAL(audit_50.preliminary_selected_count_by_image.size(), 1);
  BOOST_CHECK_EQUAL(audit_50.preliminary_selected_count_by_image.at(2), 50);
  BOOST_CHECK_EQUAL(audit_50.trigger_preliminary_selected_count, 50);
  BOOST_CHECK_EQUAL(audit_50.projection_call_count, 1);
  BOOST_CHECK_EQUAL(
      audit_50.projection_calls.front().projection.input_feature_count, 50);
  uint64_t per_image_sum = 0;
  for (const auto& image_count :
       audit_50.preliminary_selected_count_by_image) {
    per_image_sum += image_count.second;
  }
  BOOST_CHECK_EQUAL(per_image_sum, audit_50.selected_association_count);
  for (size_t index = 0; index < output_50.associations.size(); ++index) {
    BOOST_CHECK_EQUAL(output_50.associations[index].association_id, index);
    BOOST_CHECK_EQUAL(output_50.associations[index].owner_image_id, 2);
    BOOST_CHECK(output_50.associations[index].route ==
                OnlineLidarAssociationRoute::PROJECTION);
  }
}

BOOST_FIXTURE_TEST_CASE(ProjectionFailuresClearStaleResultsAndNoHitSucceeds,
                        OnlineLidarAssociationFixture) {
  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto error_snapshot = SnapshotFromPoints({plane_point});
  SetIdentityProjectionPose(&reconstruction, {2});
  SetFeature(&reconstruction, 2, 0, 10.0, 10.0);
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 0}});
  OnlineLidarAssociationRequest valid_request =
      ProjectionRequest({point3D_id});
  valid_request.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(error_snapshot, &valid_request);

  Reconstruction wrong_camera = reconstruction;
  Camera simple_camera;
  simple_camera.SetCameraId(1);
  simple_camera.InitializeWithId(
      SimplePinholeCameraModel::model_id, 100.0, 100, 80);
  wrong_camera.Camera(1) = simple_camera;
  CheckBuildFailure(wrong_camera, valid_request, "OPENCV camera model");

  OnlineLidarAssociationRequest invalid_options = valid_request;
  invalid_options.projection_options.min_proj_dist = 10.0;
  invalid_options.projection_options.choose_meter = 10.0f;
  CheckBuildFailure(reconstruction, invalid_options, "options are invalid");

  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error = "stale";
  PrimeOutput(&output);
  PrimeAudit(&audit);
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     valid_request,
                                                     &output,
                                                     &audit,
                                                     &error),
                        error);
  BOOST_CHECK(error.empty());
  CheckReconstructionStateEqual(before, reconstruction);
  BOOST_CHECK(output.associations.empty());
  BOOST_CHECK_EQUAL(audit.input_point_count, 1);
  BOOST_CHECK_EQUAL(audit.selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.skipped_point_count, 1);
  BOOST_CHECK_EQUAL(audit.skipped_no_plane_count, 1);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 1);
  BOOST_CHECK_EQUAL(audit.projection_selected_association_count, 0);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 1);
  BOOST_REQUIRE_EQUAL(audit.projection_calls.size(), 1);
  BOOST_CHECK_EQUAL(audit.projection_calls.front().projection.input_feature_count,
                    1);
  BOOST_CHECK_EQUAL(audit.projection_calls.front().projection.feature_hit_count,
                    0);
  BOOST_CHECK(IsLowercaseSha256(audit.association_sha256));
}

BOOST_FIXTURE_TEST_CASE(ProjectionNoHitHashTracksFrozenProjectionInputs,
                        OnlineLidarAssociationFixture) {
  const WorldPoint plane_point{{0.0f, 0.0f, 2.0f}};
  const auto no_hit_snapshot = SnapshotFromPoints({plane_point});
  SetIdentityProjectionPose(&reconstruction, {2});
  SetFeature(&reconstruction, 2, 0, 10.0, 0.0);
  SetFeature(&reconstruction, 2, 1, 20.0, 0.0);
  std::vector<double> camera_params = reconstruction.Camera(1).Params();
  camera_params[4] = 0.01;
  reconstruction.Camera(1).SetParams(camera_params);
  AddPoint(&reconstruction,
           Eigen::Vector3d(1.0, 0.0, 2.0),
           {{2, 1}});
  const point3D_t retained_point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(0.0, 0.0, 2.0), {{2, 0}});

  OnlineLidarAssociationRequest request =
      ProjectionRequest({retained_point3D_id});
  request.ordered_frozen_image_ids = {2};
  SetSnapshotIdentity(no_hit_snapshot, &request);

  std::string error;
  const auto build = [&](const Reconstruction& input,
                         OnlineLidarAssociationOutput* output,
                         OnlineLidarAssociationAudit* audit) {
    error = "stale";
    BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                              input, request, output, audit, &error),
                          error);
    BOOST_CHECK(error.empty());
  };

  OnlineLidarAssociationOutput base_output;
  OnlineLidarAssociationAudit base_audit;
  build(reconstruction, &base_output, &base_audit);
  BOOST_REQUIRE(base_output.associations.empty());
  BOOST_REQUIRE_EQUAL(base_audit.projection_call_count, 1);
  BOOST_REQUIRE_EQUAL(base_audit.projection_calls.size(), 1);
  BOOST_CHECK_EQUAL(
      base_audit.projection_calls.front().projection.input_feature_count, 1);
  BOOST_CHECK_EQUAL(
      base_audit.projection_calls.front().projection.feature_hit_count, 0);

  const auto check_only_hash_differs =
      [&](const OnlineLidarAssociationOutput& changed_output,
          const OnlineLidarAssociationAudit& changed_audit) {
        CheckOutputEqual(base_output, changed_output);
        const std::string changed_sha256 = changed_audit.association_sha256;
        OnlineLidarAssociationAudit normalized_audit = changed_audit;
        normalized_audit.association_sha256 = base_audit.association_sha256;
        CheckAuditEqual(base_audit, normalized_audit);
        BOOST_CHECK_NE(base_audit.association_sha256, changed_sha256);
      };

  OnlineLidarAssociationOutput repeated_output;
  OnlineLidarAssociationAudit repeated_audit;
  build(reconstruction, &repeated_output, &repeated_audit);
  CheckOutputEqual(base_output, repeated_output);
  CheckAuditEqual(base_audit, repeated_audit);

  Reconstruction equivalent = reconstruction;
  SetFeature(&equivalent, 2, 0, 10.0, -0.0);
  equivalent.Image(2).SetTvec(Eigen::Vector3d(-0.0, 0.0, -0.0));
  equivalent.Camera(1).Params(5) = -0.0;
  OnlineLidarAssociationOutput equivalent_output;
  OnlineLidarAssociationAudit equivalent_audit;
  build(equivalent, &equivalent_output, &equivalent_audit);
  CheckOutputEqual(base_output, equivalent_output);
  CheckAuditEqual(base_audit, equivalent_audit);

  Reconstruction changed_xy = reconstruction;
  SetFeature(&changed_xy, 2, 0, 11.0, 0.0);
  OnlineLidarAssociationOutput changed_xy_output;
  OnlineLidarAssociationAudit changed_xy_audit;
  build(changed_xy, &changed_xy_output, &changed_xy_audit);
  check_only_hash_differs(changed_xy_output, changed_xy_audit);

  Reconstruction changed_pose = reconstruction;
  changed_pose.Image(2).SetTvec(Eigen::Vector3d(0.1, 0.0, 0.0));
  OnlineLidarAssociationOutput changed_pose_output;
  OnlineLidarAssociationAudit changed_pose_audit;
  build(changed_pose, &changed_pose_output, &changed_pose_audit);
  check_only_hash_differs(changed_pose_output, changed_pose_audit);

  Reconstruction changed_camera = reconstruction;
  changed_camera.Camera(1).Params(0) = 101.0;
  OnlineLidarAssociationOutput changed_camera_output;
  OnlineLidarAssociationAudit changed_camera_audit;
  build(changed_camera, &changed_camera_output, &changed_camera_audit);
  check_only_hash_differs(changed_camera_output, changed_camera_audit);

  Reconstruction changed_filtered = reconstruction;
  SetFeature(&changed_filtered, 2, 1, 21.0, 0.0);
  OnlineLidarAssociationOutput changed_filtered_output;
  OnlineLidarAssociationAudit changed_filtered_audit;
  build(changed_filtered,
        &changed_filtered_output,
        &changed_filtered_audit);
  CheckOutputEqual(base_output, changed_filtered_output);
  CheckAuditEqual(base_audit, changed_filtered_audit);

  Reconstruction changed_slot_count = reconstruction;
  const Image source_image = changed_slot_count.Image(2);
  std::vector<Eigen::Vector2d> expanded_points2D;
  expanded_points2D.reserve(source_image.Points2D().size() + 1);
  for (const class Point2D& point2D : source_image.Points2D()) {
    expanded_points2D.push_back(point2D.XY());
  }
  expanded_points2D.emplace_back(30.0, 0.0);
  Image expanded_image;
  expanded_image.SetImageId(source_image.ImageId());
  expanded_image.SetCameraId(source_image.CameraId());
  expanded_image.SetName(source_image.Name());
  expanded_image.SetQvec(source_image.Qvec());
  expanded_image.SetTvec(source_image.Tvec());
  expanded_image.SetPoints2D(expanded_points2D);
  for (point2D_t point2D_idx = 0;
       point2D_idx < source_image.NumPoints2D();
       ++point2D_idx) {
    const class Point2D& point2D = source_image.Point2D(point2D_idx);
    if (point2D.HasPoint3D()) {
      expanded_image.SetPoint3DForPoint2D(point2D_idx,
                                          point2D.Point3DId());
    }
  }
  expanded_image.SetRegistered(source_image.IsRegistered());
  expanded_image.SetNumObservations(source_image.NumObservations());
  expanded_image.SetNumCorrespondences(source_image.NumCorrespondences());
  changed_slot_count.Image(2) = std::move(expanded_image);
  OnlineLidarAssociationOutput changed_slot_count_output;
  OnlineLidarAssociationAudit changed_slot_count_audit;
  build(changed_slot_count,
        &changed_slot_count_output,
        &changed_slot_count_audit);
  check_only_hash_differs(changed_slot_count_output,
                          changed_slot_count_audit);
}

BOOST_FIXTURE_TEST_CASE(
    ProjectionHashCanonicalizesNegativeZeroAndTracksSemanticChanges,
    OnlineLidarAssociationFixture) {
  const WorldPoint left{{-0.4f, 0.0f, 2.0f}};
  const WorldPoint right{{0.4f, 0.0f, 2.0f}};
  const auto hash_snapshot = SnapshotFromPoints(
      {right, left},
      std::array<float, 4>{{0.0f, 1.0f, 0.0f, 0.125f}},
      std::array<float, 4>{{1.0f, 0.0f, 0.0f, 0.25f}});
  SetIdentityProjectionPose(&reconstruction, {1, 2, 3});
  SetFeature(&reconstruction, 1, 0, 30.0, 40.0);
  SetFeature(&reconstruction, 3, 0, 70.0, 40.0);
  const point3D_t point3D_id = AddPoint(
      &reconstruction,
      Eigen::Vector3d(0.0, -0.0, 2.0),
      {{3, 0}, {1, 0}});
  OnlineLidarAssociationRequest base_request =
      ProjectionRequest({point3D_id});
  base_request.ordered_frozen_image_ids = {3, 2, 1};
  SetSnapshotIdentity(hash_snapshot, &base_request);

  OnlineLidarAssociationOutput base_output;
  OnlineLidarAssociationAudit base_audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     base_request,
                                                     &base_output,
                                                     &base_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(base_output.associations.size(), 1);
  BOOST_CHECK_EQUAL(base_output.associations.front().owner_image_id, 1);

  Reconstruction negative_zero = reconstruction;
  negative_zero.Point3D(point3D_id)
      .SetXYZ(Eigen::Vector3d(-0.0, 0.0, 2.0));
  negative_zero.Point3D(point3D_id)
      .Track()
      .SetElements({{1, 0}, {3, 0}});
  OnlineLidarAssociationOutput negative_zero_output;
  OnlineLidarAssociationAudit negative_zero_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(negative_zero,
                                                     base_request,
                                                     &negative_zero_output,
                                                     &negative_zero_audit,
                                                     &error),
                        error);
  CheckOutputEqual(base_output, negative_zero_output);
  CheckAuditEqual(base_audit, negative_zero_audit);
  BOOST_CHECK_EQUAL(base_audit.association_sha256,
                    negative_zero_audit.association_sha256);

  OnlineLidarAssociationRequest changed_projection = base_request;
  changed_projection.projection_options.min_proj_scale = 1;
  changed_projection.projection_options.max_proj_scale = 1;
  OnlineLidarAssociationOutput changed_projection_output;
  OnlineLidarAssociationAudit changed_projection_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     changed_projection,
                                                     &changed_projection_output,
                                                     &changed_projection_audit,
                                                     &error),
                        error);
  BOOST_CHECK_NE(base_audit.association_sha256,
                 changed_projection_audit.association_sha256);

  OnlineLidarAssociationRequest changed_order = base_request;
  changed_order.ordered_frozen_image_ids = {1, 2, 3};
  OnlineLidarAssociationOutput changed_order_output;
  OnlineLidarAssociationAudit changed_order_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     changed_order,
                                                     &changed_order_output,
                                                     &changed_order_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(changed_order_output.associations.size(), 1);
  BOOST_CHECK_EQUAL(changed_order_output.associations.front().owner_image_id,
                    1);
  BOOST_CHECK_NE(base_audit.association_sha256,
                 changed_order_audit.association_sha256);

  OnlineLidarAssociationRequest changed_owner = base_request;
  changed_owner.ordered_frozen_image_ids = {2, 3};
  OnlineLidarAssociationOutput changed_owner_output;
  OnlineLidarAssociationAudit changed_owner_audit;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(reconstruction,
                                                     changed_owner,
                                                     &changed_owner_output,
                                                     &changed_owner_audit,
                                                     &error),
                        error);
  BOOST_REQUIRE_EQUAL(changed_owner_output.associations.size(), 1);
  BOOST_CHECK_EQUAL(changed_owner_output.associations.front().owner_image_id,
                    3);
  BOOST_CHECK_NE(base_audit.association_sha256,
                 changed_owner_audit.association_sha256);
}

BOOST_FIXTURE_TEST_CASE(RouteUsesFullTrackWithoutDereferencingOutsideWindow,
                        OnlineLidarAssociationFixture) {
  const point3D_t point3D_id = AddPoint(
      &reconstruction, Eigen::Vector3d(1.0, 2.0, 3.0), {{2, 0}});
  reconstruction.Point3D(point3D_id).Track().AddElement(
      999, kInvalidPoint2DIdx);
  reconstruction.Point3D(point3D_id).Track().AddElement(1000, 9999);
  reconstruction.Point3D(point3D_id).Track().AddElement(1001, 0);
  BOOST_REQUIRE_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 4);

  OnlineLidarAssociationRequest request = Request({point3D_id});
  request.options.local_lidar_kdtree_only = false;
  request.options.min_proj_num = 1;
  const Reconstruction before = reconstruction;
  OnlineLidarAssociationOutput output;
  OnlineLidarAssociationAudit audit;
  std::string error;
  BOOST_REQUIRE_MESSAGE(BuildOnlineLidarAssociations(
                            reconstruction, request, &output, &audit, &error),
                        error);
  CheckReconstructionStateEqual(before, reconstruction);
  BOOST_REQUIRE_EQUAL(output.associations.size(), 1);
  BOOST_CHECK(output.associations.front().route ==
              OnlineLidarAssociationRoute::KDTREE);
  BOOST_CHECK_EQUAL(output.associations.front().owner_image_id, 2);
  BOOST_CHECK_EQUAL(output.associations.front().owner_point2D_idx, 0);
  BOOST_CHECK_EQUAL(audit.projection_route_point_count, 0);
  BOOST_CHECK_EQUAL(audit.kdtree_route_point_count, 1);
  BOOST_CHECK_EQUAL(audit.projection_call_count, 0);
}

#endif  // GPU_BA_CUDA_ENABLED
