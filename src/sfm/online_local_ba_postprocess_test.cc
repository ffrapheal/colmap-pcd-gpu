#define TEST_NAME "sfm/online_local_ba_postprocess"
#include "util/testing.h"

#include "sfm/online_local_ba_postprocess.h"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/camera_models.h"
#include "base/pose.h"
#include "sfm/incremental_triangulator.h"

using namespace colmap;

namespace {

using Observation = std::pair<image_t, point2D_t>;

class FilterScene {
 public:
  FilterScene() {
    Camera camera;
    camera.SetCameraId(1);
    camera.InitializeWithId(SimplePinholeCameraModel::model_id, 100.0, 200,
                            200);
    reconstruction.AddCamera(camera);
    for (image_t image_id = 1; image_id <= 5; ++image_id) {
      Image image;
      image.SetImageId(image_id);
      image.SetCameraId(1);
      image.SetName("image" + std::to_string(image_id));
      image.SetQvec(ComposeIdentityQuaternion());
      image.SetTvec(Eigen::Vector3d(1.0 - 0.5 * image_id,
                                   image_id == 5 ? -1.0 : 0.0, 0.0));
      image.SetPoints2D(
          std::vector<Eigen::Vector2d>(6, Eigen::Vector2d::Zero()));
      reconstruction.AddImage(std::move(image));
      correspondence_graph.AddImage(image_id, 6);
    }
  }

  void AddCorrespondence(const image_t image_id1,
                         const point2D_t point2D_idx1,
                         const image_t image_id2,
                         const point2D_t point2D_idx2) {
    if (image_id1 < image_id2) {
      matches[std::make_pair(image_id1, image_id2)].emplace_back(
          point2D_idx1, point2D_idx2);
    } else {
      matches[std::make_pair(image_id2, image_id1)].emplace_back(
          point2D_idx2, point2D_idx1);
    }
  }

  void FinalizeGraph() {
    for (const auto& pair_matches : matches) {
      BOOST_REQUIRE(correspondence_graph
                        .TryAddCorrespondences(pair_matches.first.first,
                                               pair_matches.first.second,
                                               pair_matches.second)
                        .IsSuccess());
    }
    reconstruction.SetUp(&correspondence_graph);
    for (image_t image_id = 1; image_id <= 5; ++image_id) {
      reconstruction.RegisterImage(image_id);
    }
    for (const auto& pair_matches : matches) {
      BOOST_REQUIRE(reconstruction.AddImagePairFromCorrespondenceGraph(
          pair_matches.first.first, pair_matches.first.second));
    }
  }

  point3D_t AddPoint(const Eigen::Vector3d& stored_xyz,
                     const Eigen::Vector3d& measured_xyz,
                     const std::initializer_list<Observation>& observations) {
    Track track;
    for (const Observation& observation : observations) {
      const Image& image = reconstruction.Image(observation.first);
      const Eigen::Vector3d camera_xyz =
          image.RotationMatrix() * measured_xyz + image.Tvec();
      const Eigen::Vector2d normalized(camera_xyz.x() / camera_xyz.z(),
                                       camera_xyz.y() / camera_xyz.z());
      reconstruction.Image(observation.first)
          .Point2D(observation.second)
          .SetXY(reconstruction.Camera(image.CameraId()).WorldToImage(
              normalized));
      track.AddElement(observation.first, observation.second);
    }
    return reconstruction.AddPoint3D(stored_xyz, std::move(track));
  }

  std::unique_ptr<Reconstruction> Clone() const {
    return std::unique_ptr<Reconstruction>(new Reconstruction(reconstruction));
  }

  CorrespondenceGraph correspondence_graph;
  Reconstruction reconstruction;

 private:
  std::map<std::pair<image_t, image_t>, FeatureMatches> matches;
};

struct Scenario {
  FilterScene scene;
  point3D_t merge_left = kInvalidPoint3DId;
  point3D_t merge_right = kInvalidPoint3DId;
  point3D_t restored_invalid = kInvalidPoint3DId;
  point3D_t negative_depth = kInvalidPoint3DId;
  point3D_t modified_before_ba = kInvalidPoint3DId;
  point3D_t clean_background = kInvalidPoint3DId;
};

std::unique_ptr<Scenario> BuildScenario() {
  std::unique_ptr<Scenario> scenario(new Scenario());
  FilterScene& scene = scenario->scene;
  scene.AddCorrespondence(1, 0, 2, 0);
  scene.AddCorrespondence(2, 0, 3, 0);
  scene.AddCorrespondence(3, 0, 4, 0);
  scene.AddCorrespondence(4, 1, 5, 1);
  scene.AddCorrespondence(1, 2, 2, 2);
  scene.AddCorrespondence(2, 2, 3, 2);
  scene.AddCorrespondence(3, 3, 5, 3);
  scene.AddCorrespondence(1, 4, 4, 4);
  scene.FinalizeGraph();

  const Eigen::Vector3d merge_measurement(-0.6, -0.2, 5.0);
  const Eigen::Vector3d merge_geometry =
      merge_measurement + Eigen::Vector3d(0.3, 0.0, 0.0);
  scenario->merge_left = scene.AddPoint(
      merge_geometry, merge_measurement, {{1, 0}, {2, 0}});
  scenario->merge_right = scene.AddPoint(
      merge_geometry, merge_measurement, {{3, 0}, {4, 0}});

  const Eigen::Vector3d restored_measurement(0.4, 0.3, 5.5);
  scenario->restored_invalid = scene.AddPoint(
      restored_measurement + Eigen::Vector3d(0.6, 0.0, 0.0),
      restored_measurement, {{4, 1}, {5, 1}});

  const Eigen::Vector3d negative_xyz(0.2, 0.1, -5.0);
  scenario->negative_depth = scene.AddPoint(
      negative_xyz, negative_xyz, {{1, 2}, {2, 2}, {3, 2}});

  const Eigen::Vector3d modified_xyz(0.7, 0.4, 5.8);
  scenario->modified_before_ba = scene.AddPoint(
      modified_xyz, modified_xyz, {{3, 3}, {5, 3}});

  const Eigen::Vector3d background_xyz(-0.7, 0.4, 6.2);
  scenario->clean_background = scene.AddPoint(
      background_xyz, background_xyz, {{1, 4}, {4, 4}});
  return scenario;
}

void AddImagePoints(const Reconstruction& reconstruction,
                    const image_t image_id,
                    std::unordered_set<point3D_t>* point3D_ids) {
  for (const Point2D& point2D : reconstruction.Image(image_id).Points2D()) {
    if (point2D.HasPoint3D()) {
      point3D_ids->insert(point2D.Point3DId());
    }
  }
}

std::unordered_set<point3D_t> PrepareTouchedPoints(
    const Scenario& scenario,
    Reconstruction* reconstruction,
    IncrementalTriangulator* triangulator) {
  std::unordered_set<point3D_t> touched = {scenario.modified_before_ba};
  AddImagePoints(*reconstruction, 2, &touched);
  AddImagePoints(*reconstruction, 5, &touched);
  IncrementalTriangulator::Options options;
  options.ignore_two_view_tracks = false;
  triangulator->MergeAllTracks(options);
  triangulator->CompleteAllTracks(options);
  const auto& modified = triangulator->GetModifiedPoints3D();
  touched.insert(modified.begin(), modified.end());
  return touched;
}

struct GeometricPoint {
  std::vector<Observation> track;
  Eigen::Vector3d xyz;
};

std::vector<GeometricPoint> CaptureGeometry(
    const Reconstruction& reconstruction) {
  std::vector<GeometricPoint> geometry;
  for (const auto& point_entry : reconstruction.Points3D()) {
    GeometricPoint point;
    point.xyz = point_entry.second.XYZ();
    for (const TrackElement& track_element :
         point_entry.second.Track().Elements()) {
      point.track.emplace_back(track_element.image_id,
                               track_element.point2D_idx);
    }
    std::sort(point.track.begin(), point.track.end());
    geometry.push_back(std::move(point));
  }
  std::sort(geometry.begin(), geometry.end(),
            [](const GeometricPoint& left, const GeometricPoint& right) {
              return left.track < right.track;
            });
  return geometry;
}

void CheckGeometricallyEquivalent(const Reconstruction& expected,
                                  const Reconstruction& actual) {
  const std::vector<GeometricPoint> expected_geometry =
      CaptureGeometry(expected);
  const std::vector<GeometricPoint> actual_geometry = CaptureGeometry(actual);
  BOOST_REQUIRE_EQUAL(actual_geometry.size(), expected_geometry.size());
  for (size_t point_index = 0; point_index < expected_geometry.size();
       ++point_index) {
    BOOST_REQUIRE(actual_geometry[point_index].track ==
                  expected_geometry[point_index].track);
    BOOST_CHECK_SMALL((actual_geometry[point_index].xyz -
                       expected_geometry[point_index].xyz)
                          .norm(),
                      1e-12);
  }
}

BOOST_AUTO_TEST_CASE(ScopedFilterMatchesFullGlobalOnCleanBackground) {
  std::unique_ptr<Scenario> scenario = BuildScenario();
  std::unique_ptr<Reconstruction> full = scenario->scene.Clone();
  std::unique_ptr<Reconstruction> scoped = scenario->scene.Clone();
  IncrementalTriangulator full_triangulator(
      &scenario->scene.correspondence_graph, full.get());
  IncrementalTriangulator scoped_triangulator(
      &scenario->scene.correspondence_graph, scoped.get());

  PrepareTouchedPoints(*scenario, full.get(), &full_triangulator);
  const std::unordered_set<point3D_t> touched = PrepareTouchedPoints(
      *scenario, scoped.get(), &scoped_triangulator);
  full->FilterAllPoints3D(4.0, 1.5);
  const OnlineLocalBaTouchedFilterResult result =
      FilterOnlineLocalBaTouchedPoints(scoped.get(), touched, 4.0, 1.5);

  BOOST_REQUIRE_MESSAGE(result.success, result.detail);
  BOOST_CHECK_GT(result.missing_input_point_count, 0);
  BOOST_CHECK_EQUAL(touched.count(scenario->clean_background), 0);
  CheckGeometricallyEquivalent(*full, *scoped);
  BOOST_CHECK(scoped->ExistsPoint3D(scenario->clean_background));
}

BOOST_AUTO_TEST_CASE(CallerPassesMergeCreatedPointId) {
  std::unique_ptr<Scenario> scenario = BuildScenario();
  std::unique_ptr<Reconstruction> reconstruction = scenario->scene.Clone();
  IncrementalTriangulator triangulator(
      &scenario->scene.correspondence_graph, reconstruction.get());
  const std::unordered_set<point3D_t> touched = PrepareTouchedPoints(
      *scenario, reconstruction.get(), &triangulator);
  const point3D_t merged_id =
      reconstruction->Image(2).Point2D(0).Point3DId();

  BOOST_CHECK_NE(merged_id, scenario->merge_left);
  BOOST_CHECK_NE(merged_id, scenario->merge_right);
  BOOST_CHECK(touched.count(merged_id) > 0);
  const OnlineLocalBaTouchedFilterResult result =
      FilterOnlineLocalBaTouchedPoints(reconstruction.get(), touched, 4.0,
                                       1.5);
  BOOST_REQUIRE_MESSAGE(result.success, result.detail);
  BOOST_CHECK(!reconstruction->ExistsPoint3D(merged_id));
}

BOOST_AUTO_TEST_CASE(RestoredPoseImagePointsAreCallerVisibleInputs) {
  std::unique_ptr<Scenario> scenario = BuildScenario();
  std::unique_ptr<Reconstruction> reconstruction = scenario->scene.Clone();
  std::unordered_set<point3D_t> touched;
  AddImagePoints(*reconstruction, 5, &touched);

  BOOST_CHECK(touched.count(scenario->restored_invalid) > 0);
  const OnlineLocalBaTouchedFilterResult result =
      FilterOnlineLocalBaTouchedPoints(reconstruction.get(), touched, 4.0,
                                       1.5);
  BOOST_REQUIRE_MESSAGE(result.success, result.detail);
  BOOST_CHECK(!reconstruction->ExistsPoint3D(scenario->restored_invalid));
}

BOOST_AUTO_TEST_CASE(NegativeDepthFallbackChecksAllTouchedTrackObservations) {
  std::unique_ptr<Scenario> scenario = BuildScenario();
  std::unique_ptr<Reconstruction> reconstruction = scenario->scene.Clone();
  const OnlineLocalBaTouchedFilterResult result =
      FilterOnlineLocalBaTouchedPoints(
          reconstruction.get(), {scenario->negative_depth}, 1e200, 0.0);

  BOOST_REQUIRE_MESSAGE(result.success, result.detail);
  BOOST_CHECK(result.negative_depth_filtered_observation_count > 0);
  BOOST_CHECK(!reconstruction->ExistsPoint3D(scenario->negative_depth));
}

BOOST_AUTO_TEST_CASE(DisconnectedCleanBackgroundIsNotVisited) {
  std::unique_ptr<Scenario> scenario = BuildScenario();
  std::unique_ptr<Reconstruction> reconstruction = scenario->scene.Clone();
  const Eigen::Vector3d before =
      reconstruction->Point3D(scenario->clean_background).XYZ();
  const OnlineLocalBaTouchedFilterResult result =
      FilterOnlineLocalBaTouchedPoints(
          reconstruction.get(), {scenario->restored_invalid}, 4.0, 1.5);

  BOOST_REQUIRE_MESSAGE(result.success, result.detail);
  BOOST_CHECK(reconstruction->ExistsPoint3D(scenario->clean_background));
  BOOST_CHECK(reconstruction->Point3D(scenario->clean_background)
                  .XYZ()
                  .isApprox(before, 0.0));
  BOOST_CHECK_EQUAL(result.input_touched_point_count, 1);
}

}  // namespace
