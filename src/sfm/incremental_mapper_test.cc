#define TEST_NAME "sfm/incremental_mapper"
#include "util/testing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/camera_models.h"
#include "base/database_cache.h"
#include "base/pose.h"
#include "base/projection.h"
#include "sfm/incremental_mapper.h"
#include "sfm/incremental_triangulator.h"
#include "util/random.h"

using namespace colmap;

namespace {

constexpr camera_t kCameraId = 1;
constexpr camera_t kOtherCameraId = 2;
constexpr image_t kReference1 = 1;
constexpr image_t kReference2 = 2;
constexpr image_t kReference3 = 3;
constexpr image_t kReference4 = 4;
constexpr image_t kTargetImageId = 5;
constexpr image_t kUnregisteredImageId = 6;
constexpr image_t kExtraImageId = 7;
constexpr image_t kContinuedOnlyImageId = 8;
constexpr size_t kNumPointsPerReference = 8;
constexpr size_t kNumPoints = 4 * kNumPointsPerReference;

struct SceneState {
  uint64_t structure_revision = 0;
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  std::vector<double> camera_params;
  bool target_is_registered = false;
  point2D_t target_num_points3D = 0;
  size_t num_reg_images = 0;
  size_t num_points3D = 0;
  size_t num_total_reg_images = 0;
  size_t num_shared_reg_images = 0;
  std::vector<Eigen::Vector2d> target_points2D;
  std::vector<size_t> track_lengths;
  std::vector<Eigen::Vector3d> point3D_xyz;
  std::vector<point3D_t> target_point3D_ids;
};

class SyntheticRegistrationFixture {
 public:
  SyntheticRegistrationFixture() : mapper(&database_cache) {
    SetPRNGSeed(0);

    Camera camera;
    camera.SetCameraId(kCameraId);
    camera.InitializeWithId(
        SimpleRadialCameraModel::model_id, 800.0, 1000, 800);
    camera.SetPriorFocalLength(true);
    database_cache.AddCamera(camera);

    Camera other_camera = camera;
    other_camera.SetCameraId(kOtherCameraId);
    database_cache.AddCamera(other_camera);

    world_points.reserve(kNumPoints);
    for (size_t group = 0; group < 2; ++group) {
      for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
          world_points.emplace_back(
              -1.0 + 0.6 * col + 0.08 * row + 0.12 * group,
              -0.8 + 0.48 * row + 0.04 * col - 0.08 * group,
              4.0 + 0.17 * ((row + 2 * col) % 4) + 0.09 * row * col +
                  0.15 * group);
        }
      }
    }

    const Eigen::Vector4d identity = ComposeIdentityQuaternion();
    AddImage(kReference1, identity, Eigen::Vector3d(-0.3, 0.0, 0.2),
             identity, Eigen::Vector3d(-0.3, 0.0, 0.2));
    AddImage(kReference2, identity, Eigen::Vector3d(0.25, -0.1, 0.3),
             identity, Eigen::Vector3d(0.25, -0.1, 0.3));
    AddImage(kReference3, identity, Eigen::Vector3d(0.05, 0.25, 0.15),
             identity, Eigen::Vector3d(0.05, 0.25, 0.15));
    AddImage(kReference4, identity, Eigen::Vector3d(-0.15, -0.2, 0.4),
             identity, Eigen::Vector3d(-0.15, -0.2, 0.4));

    const Eigen::Vector4d target_initial_qvec(
        0.9950041652780258, 0.0, 0.0998334166468282, 0.0);
    const Eigen::Vector3d target_initial_tvec(2.0, -1.0, 0.5);
    const Eigen::Vector3d target_true_tvec(0.2, -0.15, 0.35);
    AddImage(kTargetImageId, target_initial_qvec, target_initial_tvec,
             identity, target_true_tvec);
    AddImage(kUnregisteredImageId, identity, Eigen::Vector3d::Zero(), identity,
             target_true_tvec);

    const std::array<image_t, 4> reference_ids = {
        kReference1, kReference2, kReference3, kReference4};
    for (size_t reference_idx = 0; reference_idx < reference_ids.size();
         ++reference_idx) {
      FeatureMatches matches;
      matches.reserve(kNumPointsPerReference);
      const point2D_t begin =
          static_cast<point2D_t>(reference_idx * kNumPointsPerReference);
      const point2D_t end = begin + kNumPointsPerReference;
      for (point2D_t point2D_idx = begin; point2D_idx < end;
           ++point2D_idx) {
        matches.emplace_back(point2D_idx, point2D_idx);
      }
      const auto result = database_cache.AddVerifiedCorrespondences(
          kTargetImageId, reference_ids[reference_idx], matches);
      BOOST_REQUIRE(result.IsSuccess());
      BOOST_REQUIRE_EQUAL(result.num_added_matches, kNumPointsPerReference);
    }

    reconstruction.Load(database_cache);
    for (const image_t reference_id : reference_ids) {
      reconstruction.RegisterImage(reference_id);
    }

    point3D_ids.reserve(kNumPoints);
    for (point2D_t point2D_idx = 0; point2D_idx < kNumPoints;
         ++point2D_idx) {
      const point3D_t point3D_id = reconstruction.AddPoint3D(
          world_points[point2D_idx], Track());
      const image_t reference_id =
          reference_ids[point2D_idx / kNumPointsPerReference];
      reconstruction.Image(reference_id)
          .SetPoint3DForPoint2D(point2D_idx, point3D_id);
      reconstruction.Point3D(point3D_id)
          .Track()
          .AddElement(reference_id, point2D_idx);
      point3D_ids.push_back(point3D_id);
    }

    mapper.BeginReconstruction(&reconstruction);

    options.abs_pose_min_num_inliers = 10;
    options.abs_pose_min_inlier_ratio = 0.9;
    options.abs_pose_max_error = 0.5;
    options.abs_pose_refine_focal_length = false;
    options.abs_pose_refine_extra_params = false;
    options.num_threads = 1;
  }

  ~SyntheticRegistrationFixture() { mapper.EndReconstruction(false); }

  std::vector<image_t> MainReferences() const {
    return {kReference1, kReference2};
  }

  std::vector<image_t> LoopReferences() const {
    return {kReference3, kReference4};
  }

  bool Estimate(
      const std::vector<image_t>& reference_ids,
      IncrementalMapper::ImagePoseEstimation* result) {
    SetPRNGSeed(0);
    return mapper.EstimateImagePoseWithReferences(
        options, kTargetImageId, reference_ids, result);
  }

  void AddExtraImageToCache() {
    const Eigen::Vector4d identity = ComposeIdentityQuaternion();
    AddImage(kExtraImageId, identity, Eigen::Vector3d::Zero(), identity,
             Eigen::Vector3d(0.1, -0.05, 0.25));
  }

  void AddExtraCorrespondence(const point2D_t target_point2D_idx) {
    const auto result = database_cache.AddVerifiedCorrespondences(
        kTargetImageId, kExtraImageId,
        {FeatureMatch(target_point2D_idx, target_point2D_idx)});
    BOOST_REQUIRE(result.IsSuccess());
    BOOST_REQUIRE_EQUAL(result.num_added_matches, 1);
  }

  void RestartWithUncachedRegisteredImage() {
    mapper.EndReconstruction(false);

    Image continued_image;
    continued_image.SetImageId(kContinuedOnlyImageId);
    continued_image.SetCameraId(kCameraId);
    continued_image.SetName("continued_only");
    reconstruction.AddImage(std::move(continued_image));
    reconstruction.RegisterImage(kContinuedOnlyImageId);

    mapper.BeginReconstruction(&reconstruction);
  }

  SceneState CaptureState() const {
    SceneState state;
    const Image& target = reconstruction.Image(kTargetImageId);
    state.structure_revision = reconstruction.StructureRevision();
    state.qvec = target.Qvec();
    state.tvec = target.Tvec();
    state.camera_params = reconstruction.Camera(kCameraId).Params();
    state.target_is_registered = target.IsRegistered();
    state.target_num_points3D = target.NumPoints3D();
    state.num_reg_images = reconstruction.NumRegImages();
    state.num_points3D = reconstruction.NumPoints3D();
    state.num_total_reg_images = mapper.NumTotalRegImages();
    state.num_shared_reg_images = mapper.NumSharedRegImages();
    state.target_points2D.reserve(target.NumPoints2D());
    for (point2D_t point2D_idx = 0; point2D_idx < target.NumPoints2D();
         ++point2D_idx) {
      state.target_points2D.push_back(target.Point2D(point2D_idx).XY());
    }
    state.track_lengths.reserve(point3D_ids.size());
    state.point3D_xyz.reserve(point3D_ids.size());
    for (const point3D_t point3D_id : point3D_ids) {
      state.track_lengths.push_back(
          reconstruction.Point3D(point3D_id).Track().Length());
      state.point3D_xyz.push_back(reconstruction.Point3D(point3D_id).XYZ());
    }
    state.target_point3D_ids.reserve(target.NumPoints2D());
    for (point2D_t point2D_idx = 0; point2D_idx < target.NumPoints2D();
         ++point2D_idx) {
      state.target_point3D_ids.push_back(
          target.Point2D(point2D_idx).Point3DId());
    }
    return state;
  }

  void CheckState(const SceneState& expected) const {
    const Image& target = reconstruction.Image(kTargetImageId);
    BOOST_CHECK_EQUAL(reconstruction.StructureRevision(),
                      expected.structure_revision);
    BOOST_CHECK(target.Qvec().isApprox(expected.qvec, 0.0));
    BOOST_CHECK(target.Tvec().isApprox(expected.tvec, 0.0));
    BOOST_CHECK(reconstruction.Camera(kCameraId).Params() ==
                expected.camera_params);
    BOOST_CHECK_EQUAL(target.IsRegistered(), expected.target_is_registered);
    BOOST_CHECK_EQUAL(target.NumPoints3D(), expected.target_num_points3D);
    BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), expected.num_reg_images);
    BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), expected.num_points3D);
    BOOST_CHECK_EQUAL(mapper.NumTotalRegImages(),
                      expected.num_total_reg_images);
    BOOST_CHECK_EQUAL(mapper.NumSharedRegImages(),
                      expected.num_shared_reg_images);
    BOOST_REQUIRE_EQUAL(target.NumPoints2D(), expected.target_points2D.size());
    for (point2D_t point2D_idx = 0; point2D_idx < target.NumPoints2D();
         ++point2D_idx) {
      BOOST_CHECK(target.Point2D(point2D_idx).XY().isApprox(
          expected.target_points2D[point2D_idx], 0.0));
    }
    BOOST_REQUIRE_EQUAL(point3D_ids.size(), expected.track_lengths.size());
    BOOST_REQUIRE_EQUAL(point3D_ids.size(), expected.point3D_xyz.size());
    for (size_t idx = 0; idx < point3D_ids.size(); ++idx) {
      BOOST_CHECK_EQUAL(
          reconstruction.Point3D(point3D_ids[idx]).Track().Length(),
          expected.track_lengths[idx]);
      BOOST_CHECK(reconstruction.Point3D(point3D_ids[idx]).XYZ().isApprox(
          expected.point3D_xyz[idx], 0.0));
    }
    BOOST_REQUIRE_EQUAL(target.NumPoints2D(),
                        expected.target_point3D_ids.size());
    for (point2D_t point2D_idx = 0; point2D_idx < target.NumPoints2D();
         ++point2D_idx) {
      BOOST_CHECK_EQUAL(target.Point2D(point2D_idx).Point3DId(),
                        expected.target_point3D_ids[point2D_idx]);
    }
  }

  void CheckCommitRejected(
      const IncrementalMapper::ImagePoseEstimation& result) {
    const SceneState at_call_entry = CaptureState();
    bool committed = true;
    BOOST_CHECK_NO_THROW(committed = mapper.CommitImageRegistration(result));
    BOOST_CHECK(!committed);
    CheckState(at_call_entry);
  }

  void CheckInvalidResult(
      const IncrementalMapper::ImagePoseEstimation& result) const {
    BOOST_CHECK(!result.is_valid);
    BOOST_CHECK_EQUAL(result.mapper_owner_id, 0);
    BOOST_CHECK_EQUAL(result.reconstruction_generation, 0);
    BOOST_CHECK(result.reconstruction == nullptr);
    BOOST_CHECK_EQUAL(result.image_id, kInvalidImageId);
    BOOST_CHECK_EQUAL(result.camera_id, kInvalidCameraId);
    BOOST_CHECK(result.qvec.isZero());
    BOOST_CHECK(result.tvec.isZero());
    BOOST_CHECK(result.correspondences.empty());
    BOOST_CHECK(result.inlier_mask.empty());
    BOOST_CHECK_EQUAL(result.num_inliers, 0);
  }

  DatabaseCache database_cache;
  Reconstruction reconstruction;
  IncrementalMapper mapper;
  IncrementalMapper::Options options;
  std::vector<Eigen::Vector3d> world_points;
  std::vector<point3D_t> point3D_ids;

 private:
  void AddImage(const image_t image_id,
                const Eigen::Vector4d& stored_qvec,
                const Eigen::Vector3d& stored_tvec,
                const Eigen::Vector4d& projection_qvec,
                const Eigen::Vector3d& projection_tvec) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(kCameraId);
    image.SetName("image" + std::to_string(image_id));
    image.SetQvec(stored_qvec);
    image.SetTvec(stored_tvec);
    const Eigen::Matrix3x4d projection =
        ComposeProjectionMatrix(projection_qvec, projection_tvec);
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(world_points.size());
    for (const Eigen::Vector3d& point3D : world_points) {
      points2D.push_back(ProjectPointToImage(
          point3D, projection, database_cache.Camera(kCameraId)));
    }
    image.SetPoints2D(points2D);
    database_cache.AddImage(std::move(image));
  }
};

void CheckReferenceSubset(
    const IncrementalMapper::ImagePoseEstimation& result,
    const std::vector<image_t>& allowed_reference_ids) {
  for (const auto& correspondence : result.correspondences) {
    BOOST_CHECK(std::find(allowed_reference_ids.begin(),
                          allowed_reference_ids.end(),
                          correspondence.reference_image_id) !=
                allowed_reference_ids.end());
  }
}

std::vector<size_t> FindInlierIndices(
    const IncrementalMapper::ImagePoseEstimation& result) {
  std::vector<size_t> indices;
  indices.reserve(result.num_inliers);
  for (size_t idx = 0; idx < result.inlier_mask.size(); ++idx) {
    if (result.inlier_mask[idx] == 1) {
      indices.push_back(idx);
    }
  }
  return indices;
}

}  // namespace

BOOST_FIXTURE_TEST_CASE(AllowlistEstimateIsIsolatedAndNonMutating,
                        SyntheticRegistrationFixture) {
  const SceneState before = CaptureState();
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));

  BOOST_CHECK(result.is_valid);
  BOOST_CHECK_NE(result.mapper_owner_id, 0);
  BOOST_CHECK_NE(result.reconstruction_generation, 0);
  BOOST_CHECK(result.reconstruction == &reconstruction);
  BOOST_CHECK_EQUAL(result.image_id, kTargetImageId);
  BOOST_CHECK_EQUAL(result.camera_id, kCameraId);
  BOOST_REQUIRE_EQUAL(result.correspondences.size(),
                      2 * kNumPointsPerReference);
  BOOST_REQUIRE_EQUAL(result.correspondences.size(), result.inlier_mask.size());
  BOOST_CHECK_EQUAL(result.num_inliers, result.correspondences.size());
  BOOST_CHECK_SMALL(std::abs(result.qvec.norm() - 1.0), 1e-12);
  BOOST_CHECK_EQUAL(result.min_focal_length_ratio,
                    options.min_focal_length_ratio);
  BOOST_CHECK_EQUAL(result.max_focal_length_ratio,
                    options.max_focal_length_ratio);
  BOOST_CHECK_EQUAL(result.max_extra_param, options.max_extra_param);
  CheckReferenceSubset(result, MainReferences());
  for (const auto& correspondence : result.correspondences) {
    BOOST_CHECK_EQUAL(correspondence.reference_point2D_idx,
                      correspondence.point2D_idx);
  }

  const CorrespondenceGraph& graph = database_cache.CorrespondenceGraph();
  BOOST_CHECK_EQUAL(graph.NumCorrespondencesBetweenImages(
                        kTargetImageId, kReference1),
                    kNumPointsPerReference);
  BOOST_CHECK_EQUAL(graph.NumCorrespondencesBetweenImages(
                        kTargetImageId, kReference2),
                    kNumPointsPerReference);
  BOOST_CHECK_EQUAL(graph.NumCorrespondencesBetweenImages(
                        kTargetImageId, kReference3),
                    kNumPointsPerReference);
  BOOST_CHECK_EQUAL(graph.NumCorrespondencesBetweenImages(
                        kTargetImageId, kReference4),
                    kNumPointsPerReference);
  CheckState(before);
}

BOOST_FIXTURE_TEST_CASE(InsufficientAllowlistResetsResultWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const SceneState before = CaptureState();

  BOOST_CHECK(!Estimate({kReference1}, &result));
  CheckInvalidResult(result);
  CheckState(before);
}

BOOST_FIXTURE_TEST_CASE(DisjointEstimatesCommitOnlyChosenInliersOnce,
                        SyntheticRegistrationFixture) {
  const SceneState before_estimates = CaptureState();
  IncrementalMapper::ImagePoseEstimation main_result;
  IncrementalMapper::ImagePoseEstimation loop_result;
  BOOST_REQUIRE(Estimate(MainReferences(), &main_result));
  BOOST_REQUIRE(Estimate(LoopReferences(), &loop_result));
  CheckReferenceSubset(main_result, MainReferences());
  CheckReferenceSubset(loop_result, LoopReferences());
  CheckState(before_estimates);

  std::unordered_map<point2D_t, point3D_t> expected_observations;
  std::unordered_set<point3D_t> expected_point3D_ids;
  for (size_t idx = 0; idx < main_result.inlier_mask.size(); ++idx) {
    if (main_result.inlier_mask[idx]) {
      const auto& correspondence = main_result.correspondences[idx];
      expected_observations.emplace(correspondence.point2D_idx,
                                    correspondence.point3D_id);
      expected_point3D_ids.insert(correspondence.point3D_id);
    }
  }
  BOOST_REQUIRE_EQUAL(expected_observations.size(), main_result.num_inliers);

  BOOST_REQUIRE(mapper.CommitImageRegistration(main_result));
  const Image& target = reconstruction.Image(kTargetImageId);
  BOOST_CHECK(target.IsRegistered());
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(),
                    before_estimates.num_reg_images + 1);
  BOOST_CHECK_EQUAL(std::count(reconstruction.RegImageIds().begin(),
                               reconstruction.RegImageIds().end(),
                               kTargetImageId),
                    1);
  BOOST_CHECK_EQUAL(target.NumPoints3D(), expected_observations.size());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(),
                    before_estimates.num_points3D);
  BOOST_CHECK(target.Qvec().isApprox(main_result.qvec));
  BOOST_CHECK(target.Tvec().isApprox(main_result.tvec));
  BOOST_CHECK(reconstruction.Camera(kCameraId).Params() ==
              main_result.camera.Params());

  for (point2D_t point2D_idx = 0; point2D_idx < target.NumPoints2D();
       ++point2D_idx) {
    const auto expected_it = expected_observations.find(point2D_idx);
    if (expected_it == expected_observations.end()) {
      BOOST_CHECK(!target.Point2D(point2D_idx).HasPoint3D());
    } else {
      BOOST_CHECK_EQUAL(target.Point2D(point2D_idx).Point3DId(),
                        expected_it->second);
    }
  }
  for (size_t idx = 0; idx < point3D_ids.size(); ++idx) {
    const size_t expected_length =
        before_estimates.track_lengths[idx] +
        expected_point3D_ids.count(point3D_ids[idx]);
    BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_ids[idx]).Track().Length(),
                      expected_length);
  }

  const SceneState after_commit = CaptureState();
  BOOST_CHECK(!mapper.CommitImageRegistration(main_result));
  CheckState(after_commit);
  BOOST_CHECK(!mapper.CommitImageRegistration(loop_result));
  CheckState(after_commit);
}

BOOST_FIXTURE_TEST_CASE(EstimateCannotCommitAcrossMapperInstances,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const SceneState source_before = CaptureState();

  SyntheticRegistrationFixture other_fixture;
  const SceneState other_before = other_fixture.CaptureState();
  BOOST_CHECK(!other_fixture.mapper.CommitImageRegistration(result));
  other_fixture.CheckState(other_before);
  CheckState(source_before);
}

BOOST_FIXTURE_TEST_CASE(EstimateCannotCommitAfterNewSessionWithoutJournal,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  BOOST_REQUIRE(!result.structure_journal_enabled);

  mapper.EndReconstruction(false);
  mapper.BeginReconstruction(&reconstruction);

  const SceneState before = CaptureState();
  BOOST_CHECK(!mapper.CommitImageRegistration(result));
  CheckState(before);
}

BOOST_FIXTURE_TEST_CASE(EstimateCannotCommitAfterNewSessionWithSameJournal,
                        SyntheticRegistrationFixture) {
  constexpr uint64_t kOwnerEpoch = 29;
  reconstruction.BeginStructureJournal(kOwnerEpoch);

  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  BOOST_REQUIRE(result.structure_journal_enabled);
  BOOST_REQUIRE_EQUAL(result.structure_owner_epoch, kOwnerEpoch);
  reconstruction.EndStructureJournal();

  mapper.EndReconstruction(false);
  mapper.BeginReconstruction(&reconstruction);
  reconstruction.BeginStructureJournal(kOwnerEpoch);
  BOOST_REQUIRE_EQUAL(reconstruction.StructureRevision(),
                      result.structure_revision);

  const SceneState before = CaptureState();
  BOOST_CHECK(!mapper.CommitImageRegistration(result));
  CheckState(before);
  reconstruction.EndStructureJournal();
}

BOOST_FIXTURE_TEST_CASE(InvalidAllowlistsRejectWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation valid_result;
  BOOST_REQUIRE(Estimate(MainReferences(), &valid_result));
  const SceneState before = CaptureState();
  const std::vector<std::vector<image_t>> invalid_reference_lists = {
      {},
      {kReference1, kTargetImageId},
      {kReference1, 999},
      {kReference1, kUnregisteredImageId},
  };

  for (const auto& reference_ids : invalid_reference_lists) {
    IncrementalMapper::ImagePoseEstimation result = valid_result;
    BOOST_CHECK(!Estimate(reference_ids, &result));
    CheckInvalidResult(result);
    CheckState(before);
  }
}

BOOST_FIXTURE_TEST_CASE(LegacyRegisterNextImageUsesAllReferences,
                        SyntheticRegistrationFixture) {
  const size_t num_reg_images = reconstruction.NumRegImages();
  SetPRNGSeed(0);
  BOOST_REQUIRE(mapper.RegisterNextImage(options, kTargetImageId));
  BOOST_CHECK(reconstruction.Image(kTargetImageId).IsRegistered());
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), num_reg_images + 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(kTargetImageId).NumPoints3D(),
                    kNumPoints);
}

BOOST_FIXTURE_TEST_CASE(TriangulatorGeometryCachesClearBetweenOperations,
                        SyntheticRegistrationFixture) {
  SyntheticRegistrationFixture fresh_fixture;
  const Eigen::Vector4d identity = ComposeIdentityQuaternion();
  const Eigen::Vector3d true_tvec(0.2, -0.15, 0.35);
  const std::array<SyntheticRegistrationFixture*, 2> fixtures = {
      {this, &fresh_fixture}};
  for (SyntheticRegistrationFixture* fixture : fixtures) {
    Image& target = fixture->reconstruction.Image(kTargetImageId);
    target.SetQvec(identity);
    target.SetTvec(true_tvec);
    fixture->reconstruction.RegisterImage(kTargetImageId);
    fixture->reconstruction.DeletePoint3D(fixture->point3D_ids.front());
  }

  IncrementalTriangulator::Options triangulation_options;
  triangulation_options.ignore_two_view_tracks = false;
  IncrementalTriangulator reused_triangulator(
      &database_cache.CorrespondenceGraph(), &reconstruction);
  {
    IncrementalTriangulator fresh_triangulator(
        &fresh_fixture.database_cache.CorrespondenceGraph(),
        &fresh_fixture.reconstruction);
    SetPRNGSeed(0);
    const size_t reused_count = reused_triangulator.TriangulateImage(
        triangulation_options, kTargetImageId);
    SetPRNGSeed(0);
    const size_t fresh_count = fresh_triangulator.TriangulateImage(
        triangulation_options, kTargetImageId);
    BOOST_REQUIRE_EQUAL(reused_count, fresh_count);
  }

  const auto clear_created_point = [](SyntheticRegistrationFixture* fixture) {
    const Point2D& point2D = fixture->reconstruction.Image(kTargetImageId)
                                 .Point2D(0);
    BOOST_REQUIRE(point2D.HasPoint3D());
    fixture->reconstruction.DeletePoint3D(point2D.Point3DId());
  };
  clear_created_point(this);
  clear_created_point(&fresh_fixture);

  const Eigen::Vector4d changed_qvec(
      0.9999500004166653, 0.0, 0.0099998333341667, 0.0);
  const Eigen::Vector3d changed_tvec(0.23, -0.17, 0.36);
  const Eigen::Matrix3x4d original_projection =
      reconstruction.Image(kTargetImageId).ProjectionMatrix();
  const Eigen::Vector3d original_center =
      reconstruction.Image(kTargetImageId).ProjectionCenter();
  for (SyntheticRegistrationFixture* fixture : fixtures) {
    Image& target = fixture->reconstruction.Image(kTargetImageId);
    target.SetQvec(changed_qvec);
    target.SetTvec(changed_tvec);
  }
  BOOST_REQUIRE(!reconstruction.Image(kTargetImageId)
                     .ProjectionMatrix()
                     .isApprox(original_projection, 0.0));
  BOOST_REQUIRE(!reconstruction.Image(kTargetImageId)
                     .ProjectionCenter()
                     .isApprox(original_center, 0.0));

  size_t reused_count = 0;
  size_t fresh_count = 0;
  SetPRNGSeed(0);
  reused_count = reused_triangulator.TriangulateImage(
      triangulation_options, kTargetImageId);
  {
    IncrementalTriangulator fresh_triangulator(
        &fresh_fixture.database_cache.CorrespondenceGraph(),
        &fresh_fixture.reconstruction);
    SetPRNGSeed(0);
    fresh_count = fresh_triangulator.TriangulateImage(
        triangulation_options, kTargetImageId);
  }
  BOOST_REQUIRE_EQUAL(reused_count, fresh_count);
  const auto check_created_points_equal = [&]() {
    const Point2D& reused_point2D =
        reconstruction.Image(kTargetImageId).Point2D(0);
    const Point2D& fresh_point2D =
        fresh_fixture.reconstruction.Image(kTargetImageId).Point2D(0);
    BOOST_REQUIRE(reused_point2D.HasPoint3D());
    BOOST_REQUIRE(fresh_point2D.HasPoint3D());
    BOOST_CHECK(reconstruction.Point3D(reused_point2D.Point3DId())
                    .XYZ()
                    .isApprox(fresh_fixture.reconstruction
                                  .Point3D(fresh_point2D.Point3DId())
                                  .XYZ(),
                              0.0));
  };
  check_created_points_equal();

  clear_created_point(this);
  clear_created_point(&fresh_fixture);
  const Eigen::Vector2d original_normalized =
      reconstruction.Camera(kCameraId)
          .ImageToWorld(
              reconstruction.Image(kTargetImageId).Point2D(0).XY());
  for (SyntheticRegistrationFixture* fixture : fixtures) {
    Camera& camera = fixture->reconstruction.Camera(kCameraId);
    camera.SetFocalLength(camera.FocalLength() * 1.01);
  }
  BOOST_REQUIRE(!reconstruction.Camera(kCameraId)
                     .ImageToWorld(
                         reconstruction.Image(kTargetImageId).Point2D(0).XY())
                     .isApprox(original_normalized, 0.0));

  SetPRNGSeed(0);
  reused_count = reused_triangulator.TriangulateImage(
      triangulation_options, kTargetImageId);
  {
    IncrementalTriangulator fresh_triangulator(
        &fresh_fixture.database_cache.CorrespondenceGraph(),
        &fresh_fixture.reconstruction);
    SetPRNGSeed(0);
    fresh_count = fresh_triangulator.TriangulateImage(
        triangulation_options, kTargetImageId);
  }
  BOOST_REQUIRE_EQUAL(reused_count, fresh_count);
  check_created_points_equal();
}

BOOST_FIXTURE_TEST_CASE(
    LegacyRegisterNextImageSkipsUncachedUnrelatedContinuedImage,
    SyntheticRegistrationFixture) {
  RestartWithUncachedRegisteredImage();
  BOOST_REQUIRE(reconstruction.ExistsImage(kContinuedOnlyImageId));
  BOOST_REQUIRE(reconstruction.Image(kContinuedOnlyImageId).IsRegistered());
  BOOST_REQUIRE(!database_cache.ExistsImage(kContinuedOnlyImageId));
  BOOST_REQUIRE(!database_cache.CorrespondenceGraph().ExistsImage(
      kContinuedOnlyImageId));

  std::vector<image_t> strict_references = MainReferences();
  strict_references.push_back(kContinuedOnlyImageId);
  IncrementalMapper::ImagePoseEstimation rejected_result;
  const SceneState before_strict_estimate = CaptureState();
  BOOST_CHECK(!Estimate(strict_references, &rejected_result));
  CheckInvalidResult(rejected_result);
  CheckState(before_strict_estimate);

  SetPRNGSeed(0);
  BOOST_REQUIRE(mapper.RegisterNextImage(options, kTargetImageId));
  BOOST_CHECK(reconstruction.Image(kTargetImageId).IsRegistered());
  BOOST_CHECK_EQUAL(reconstruction.Image(kTargetImageId).NumPoints3D(),
                    kNumPoints);
}

BOOST_FIXTURE_TEST_CASE(
    KnownPoseWrongThreadDoesNotConsumeTrialOrPublishEvent,
    SyntheticRegistrationFixture) {
  options.max_reg_trials = 1;
  const auto candidates_before = mapper.FindNextImages(options);
  BOOST_REQUIRE(std::find(candidates_before.begin(), candidates_before.end(),
                          kTargetImageId) != candidates_before.end());
  const SceneState before = CaptureState();

  constexpr uint64_t kOwnerEpoch = 37;
  reconstruction.BeginStructureJournal(kOwnerEpoch);
  const uint64_t cursor = reconstruction.StructureRevision();
  IncrementalMapper::KnownPoseRegistrationResult result;
  std::thread worker([&]() {
    result = mapper.RegisterImageFromKnownPose(
        kTargetImageId,
        ComposeIdentityQuaternion(),
        Eigen::Vector3d::Zero());
  });
  worker.join();
  const ReconstructionStructureReadResult journal =
      reconstruction.ReadStructureEventsSince(kOwnerEpoch, cursor);
  reconstruction.EndStructureJournal();

  BOOST_CHECK(result.status ==
              IncrementalMapper::KnownPoseRegistrationStatus::REJECTED);
  BOOST_CHECK(result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::WRONG_THREAD);
  BOOST_CHECK(journal.complete);
  BOOST_CHECK(!journal.gap);
  BOOST_CHECK_EQUAL(journal.current_revision, cursor);
  BOOST_CHECK(journal.batches.empty());
  CheckState(before);

  const auto candidates_after = mapper.FindNextImages(options);
  BOOST_CHECK(std::find(candidates_after.begin(), candidates_after.end(),
                        kTargetImageId) != candidates_after.end());
}

BOOST_FIXTURE_TEST_CASE(MalformedCommitPayloadsRejectWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation valid_result;
  BOOST_REQUIRE(Estimate(MainReferences(), &valid_result));

  const auto inlier_it =
      std::find(valid_result.inlier_mask.begin(),
                valid_result.inlier_mask.end(), true);
  BOOST_REQUIRE(inlier_it != valid_result.inlier_mask.end());
  const size_t inlier_idx =
      std::distance(valid_result.inlier_mask.begin(), inlier_it);

  const auto reference1_it = std::find_if(
      valid_result.correspondences.begin(), valid_result.correspondences.end(),
      [](const IncrementalMapper::ImagePoseCorrespondence& correspondence) {
        return correspondence.reference_image_id == kReference1;
      });
  BOOST_REQUIRE(reference1_it != valid_result.correspondences.end());
  const size_t reference1_idx =
      std::distance(valid_result.correspondences.begin(), reference1_it);
  BOOST_REQUIRE(valid_result.inlier_mask[reference1_idx]);

  const point3D_t substituted_point3D_id =
      valid_result.correspondences[reference1_idx].point3D_id;
  constexpr point2D_t kSubstitutedReferencePoint2DIdx = kNumPoints - 1;
  BOOST_REQUIRE(!reconstruction.Image(kReference2)
                     .Point2D(kSubstitutedReferencePoint2DIdx)
                     .HasPoint3D());
  reconstruction.AddObservation(
      substituted_point3D_id,
      TrackElement(kReference2, kSubstitutedReferencePoint2DIdx));
  BOOST_REQUIRE(reconstruction.Image(kReference2)
                    .HasPoint3D(substituted_point3D_id));

  const SceneState before = CaptureState();

  const auto expect_rejected = [&](const auto& mutate) {
    IncrementalMapper::ImagePoseEstimation result = valid_result;
    mutate(&result);
    BOOST_CHECK(!mapper.CommitImageRegistration(result));
    CheckState(before);
  };

  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->camera = Camera();
    result->camera.SetCameraId(kCameraId);
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->camera.SetParams({});
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->camera_id = kOtherCameraId;
    result->camera.SetCameraId(kOtherCameraId);
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->qvec *= 2.0;
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->camera.SetFocalLength(1.0);
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->camera.Params().back() = 2.0;
  });
  expect_rejected([](IncrementalMapper::ImagePoseEstimation* result) {
    result->image_id = kUnregisteredImageId;
  });

  expect_rejected([&](IncrementalMapper::ImagePoseEstimation* result) {
    result->correspondences[inlier_idx].point2D_idx = kNumPoints;
  });
  expect_rejected([&](IncrementalMapper::ImagePoseEstimation* result) {
    result->correspondences[inlier_idx].point3D_id =
        point3D_ids.back() + 1000;
  });
  expect_rejected([&](IncrementalMapper::ImagePoseEstimation* result) {
    result->correspondences[reference1_idx].reference_point2D_idx = kNumPoints;
  });
  expect_rejected([&](IncrementalMapper::ImagePoseEstimation* result) {
    result->correspondences[reference1_idx].reference_image_id = kReference2;
    result->correspondences[reference1_idx].reference_point2D_idx =
        kSubstitutedReferencePoint2DIdx;
  });
}

BOOST_FIXTURE_TEST_CASE(InvalidInputCameraModelRejectsWithoutThrowOrMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));

  result.input_camera = Camera();
  result.input_camera.SetCameraId(result.camera_id);

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(NonBinaryInlierMaskRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());

  result.inlier_mask[inlier_indices.front()] = 2;

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(InlierCountBelowStoredMinimumRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(result.abs_pose_min_num_inliers > 0);
  BOOST_REQUIRE(inlier_indices.size() >= result.abs_pose_min_num_inliers);

  const size_t forged_num_inliers = result.abs_pose_min_num_inliers - 1;
  for (size_t idx = forged_num_inliers; idx < inlier_indices.size(); ++idx) {
    result.inlier_mask[inlier_indices[idx]] = 0;
  }
  result.num_inliers = forged_num_inliers;

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(DuplicateInlierTargetPointRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(inlier_indices.size() >= 2);
  const size_t first_idx = inlier_indices[0];
  const size_t second_idx = inlier_indices[1];
  BOOST_REQUIRE(result.correspondences[first_idx].point2D_idx !=
                result.correspondences[second_idx].point2D_idx);

  result.correspondences[second_idx].point2D_idx =
      result.correspondences[first_idx].point2D_idx;
  result.input_points2D[second_idx] = result.input_points2D[first_idx];

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(DuplicateInlierPoint3DRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(inlier_indices.size() >= 2);
  const size_t first_idx = inlier_indices[0];
  const size_t second_idx = inlier_indices[1];
  BOOST_REQUIRE(result.correspondences[first_idx].point3D_id !=
                result.correspondences[second_idx].point3D_id);

  result.correspondences[second_idx].point3D_id =
      result.correspondences[first_idx].point3D_id;
  result.input_points3D[second_idx] = result.input_points3D[first_idx];

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(TargetInlierOccupiedAfterEstimateRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());
  const auto& correspondence = result.correspondences[inlier_indices.front()];
  const auto occupant_it = std::find_if(
      point3D_ids.begin(), point3D_ids.end(),
      [&correspondence](const point3D_t point3D_id) {
        return point3D_id != correspondence.point3D_id;
      });
  BOOST_REQUIRE(occupant_it != point3D_ids.end());

  reconstruction.Image(kTargetImageId)
      .SetPoint3DForPoint2D(correspondence.point2D_idx, *occupant_it);
  reconstruction.Point3D(*occupant_it)
      .Track()
      .AddElement(kTargetImageId, correspondence.point2D_idx);

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(SharedCameraParamsChangedAfterEstimateRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  std::vector<double>& live_params =
      reconstruction.Camera(result.camera_id).Params();
  BOOST_REQUIRE(!live_params.empty());

  live_params.front() += 1.0;

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(TargetPointChangedAfterEstimateRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());
  const point2D_t point2D_idx =
      result.correspondences[inlier_indices.front()].point2D_idx;
  Point2D& point2D = reconstruction.Image(kTargetImageId).Point2D(point2D_idx);

  point2D.SetXY(point2D.XY() + Eigen::Vector2d(0.25, -0.5));

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(InlierPoint3DChangedAfterEstimateRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());
  const point3D_t point3D_id =
      result.correspondences[inlier_indices.front()].point3D_id;
  Point3D& point3D = reconstruction.Point3D(point3D_id);

  point3D.SetXYZ(point3D.XYZ() + Eigen::Vector3d(0.1, -0.2, 0.3));

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(UnrelatedPoint3DAddedAfterEstimateRejectsWithoutMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  BOOST_REQUIRE(!result.structure_journal_enabled);
  const uint64_t structure_revision = reconstruction.StructureRevision();
  const size_t num_points3D = reconstruction.NumPoints3D();
  BOOST_REQUIRE_EQUAL(result.num_points3D, num_points3D);

  reconstruction.AddPoint3D(Eigen::Vector3d(10.0, 20.0, 30.0), Track());
  BOOST_CHECK_EQUAL(reconstruction.StructureRevision(), structure_revision);
  BOOST_REQUIRE_EQUAL(reconstruction.NumPoints3D(), num_points3D + 1);

  CheckCommitRejected(result);
}

BOOST_FIXTURE_TEST_CASE(LiveEdgeMissingReconstructionImageRejectsBeforeMutation,
                        SyntheticRegistrationFixture) {
  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());
  const point2D_t target_point2D_idx =
      result.correspondences[inlier_indices.front()].point2D_idx;

  AddExtraImageToCache();
  AddExtraCorrespondence(target_point2D_idx);
  BOOST_REQUIRE(!reconstruction.ExistsImage(kExtraImageId));
  const size_t num_image_pairs = reconstruction.NumImagePairs();

  CheckCommitRejected(result);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), num_image_pairs);
  BOOST_CHECK(!reconstruction.ExistsImage(kExtraImageId));
}

BOOST_FIXTURE_TEST_CASE(LiveEdgeMissingReconstructionPairRejectsBeforeMutation,
                        SyntheticRegistrationFixture) {
  AddExtraImageToCache();
  BOOST_REQUIRE(
      reconstruction.AddImageFromDatabaseCache(database_cache, kExtraImageId));

  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const std::vector<size_t> inlier_indices = FindInlierIndices(result);
  BOOST_REQUIRE(!inlier_indices.empty());
  const point2D_t target_point2D_idx =
      result.correspondences[inlier_indices.front()].point2D_idx;

  AddExtraCorrespondence(target_point2D_idx);
  BOOST_REQUIRE(!reconstruction.ExistsImagePair(kTargetImageId, kExtraImageId));
  const CorrespondenceGraph& correspondence_graph =
      database_cache.CorrespondenceGraph();
  for (const image_t image_id : {kTargetImageId, kExtraImageId}) {
    reconstruction.Image(image_id).SetNumObservations(
        correspondence_graph.NumObservationsForImage(image_id));
    reconstruction.Image(image_id).SetNumCorrespondences(
        correspondence_graph.NumCorrespondencesForImage(image_id));
  }
  const point2D_t extra_num_visible_points3D =
      reconstruction.Image(kExtraImageId).NumVisiblePoints3D();
  const size_t num_image_pairs = reconstruction.NumImagePairs();

  CheckCommitRejected(result);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), num_image_pairs);
  BOOST_CHECK_EQUAL(reconstruction.Image(kExtraImageId).NumVisiblePoints3D(),
                    extra_num_visible_points3D);
  BOOST_CHECK(!reconstruction.ExistsImagePair(kTargetImageId, kExtraImageId));
}

BOOST_FIXTURE_TEST_CASE(SynchronizedExtraLiveEdgeAllowsFreshEstimateCommit,
                        SyntheticRegistrationFixture) {
  AddExtraImageToCache();
  BOOST_REQUIRE(
      reconstruction.AddImageFromDatabaseCache(database_cache, kExtraImageId));
  constexpr point2D_t kTargetPoint2DIdx = 0;
  const point3D_t point3D_id = point3D_ids[kTargetPoint2DIdx];
  reconstruction.Image(kExtraImageId)
      .SetPoint3DForPoint2D(kTargetPoint2DIdx, point3D_id);
  reconstruction.Point3D(point3D_id)
      .Track()
      .AddElement(kExtraImageId, kTargetPoint2DIdx);
  AddExtraCorrespondence(kTargetPoint2DIdx);
  BOOST_REQUIRE(reconstruction.AddImagePairFromCorrespondenceGraph(
      kTargetImageId, kExtraImageId));

  IncrementalMapper::ImagePoseEstimation result;
  BOOST_REQUIRE(Estimate(MainReferences(), &result));
  const auto target_corr_it = std::find_if(
      result.correspondences.begin(), result.correspondences.end(),
      [](const IncrementalMapper::ImagePoseCorrespondence& correspondence) {
        return correspondence.point2D_idx == kTargetPoint2DIdx;
      });
  BOOST_REQUIRE(target_corr_it != result.correspondences.end());
  const size_t target_corr_idx =
      std::distance(result.correspondences.begin(), target_corr_it);
  BOOST_REQUIRE(result.inlier_mask[target_corr_idx]);

  BOOST_REQUIRE(mapper.CommitImageRegistration(result));
  BOOST_CHECK(reconstruction.Image(kTargetImageId).IsRegistered());
  BOOST_CHECK_EQUAL(reconstruction.Image(kTargetImageId)
                        .Point2D(kTargetPoint2DIdx)
                        .Point3DId(),
                    point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(kTargetImageId, kExtraImageId)
                        .num_tri_corrs,
                    1);
}

BOOST_FIXTURE_TEST_CASE(JournalEnabledCommitPublishesSingleTransaction,
                        SyntheticRegistrationFixture) {
  constexpr uint64_t kOwnerEpoch = 17;
  reconstruction.BeginStructureJournal(kOwnerEpoch);

  IncrementalMapper::ImagePoseEstimation result;
  bool estimated = false;
  BOOST_CHECK_NO_THROW(estimated = Estimate(MainReferences(), &result));
  if (!estimated) {
    reconstruction.EndStructureJournal();
    BOOST_FAIL("Pose estimation failed with the structure journal enabled");
    return;
  }

  const uint64_t initial_revision = reconstruction.StructureRevision();
  bool committed = false;
  BOOST_CHECK_NO_THROW(committed = mapper.CommitImageRegistration(result));
  const ReconstructionStructureReadResult journal =
      reconstruction.ReadStructureEventsSince(kOwnerEpoch, initial_revision);
  reconstruction.EndStructureJournal();

  BOOST_REQUIRE(committed);
  BOOST_CHECK(journal.complete);
  BOOST_CHECK(!journal.gap);
  BOOST_CHECK_EQUAL(journal.current_revision, initial_revision + 1);
  BOOST_REQUIRE_EQUAL(journal.batches.size(), 1);
  BOOST_CHECK_EQUAL(journal.batches.front().events.size(),
                    result.num_inliers + 1);
}

namespace {

constexpr camera_t kKnownPoseCameraId = 101;
constexpr camera_t kKnownPoseOtherCameraId = 102;
constexpr image_t kKnownPoseFirstImageId = 101;
constexpr image_t kKnownPoseSecondImageId = 102;
constexpr image_t kKnownPoseThirdImageId = 103;
constexpr image_t kKnownPoseMissingImageId = 104;

Camera MakeKnownPoseCamera(const camera_t camera_id) {
  Camera camera;
  camera.SetCameraId(camera_id);
  camera.InitializeWithId(
      SimpleRadialCameraModel::model_id, 800.0, 1000, 800);
  camera.SetPriorFocalLength(true);
  return camera;
}

Image MakeKnownPoseImage(const image_t image_id,
                         const camera_t camera_id,
                         const size_t num_points2D = 0) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(camera_id);
  image.SetName("known_pose_" + std::to_string(image_id));
  if (num_points2D > 0) {
    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(num_points2D);
    for (size_t point2D_idx = 0; point2D_idx < num_points2D;
         ++point2D_idx) {
      points2D.emplace_back(static_cast<double>(point2D_idx),
                            static_cast<double>(point2D_idx));
    }
    image.SetPoints2D(points2D);
  }
  return image;
}

struct KnownPoseState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  uint64_t structure_revision = 0;
  size_t num_reg_images = 0;
  size_t num_total_reg_images = 0;
  size_t num_shared_reg_images = 0;
  std::vector<image_t> reg_image_ids;
  std::array<bool, 3> registered = {{false, false, false}};
  Eigen::Matrix<double, 7, 3> poses = Eigen::Matrix<double, 7, 3>::Zero();
  int camera_model_id = kInvalidCameraModelId;
  size_t camera_width = 0;
  size_t camera_height = 0;
  bool camera_has_prior_focal_length = false;
  std::vector<double> camera_params;
};

class KnownPoseRegistrationFixture {
 public:
  KnownPoseRegistrationFixture() : mapper(&database_cache) {
    database_cache.AddCamera(MakeKnownPoseCamera(kKnownPoseCameraId));
    database_cache.AddCamera(MakeKnownPoseCamera(kKnownPoseOtherCameraId));
    database_cache.AddImage(
        MakeKnownPoseImage(kKnownPoseFirstImageId, kKnownPoseCameraId));
    database_cache.AddImage(
        MakeKnownPoseImage(kKnownPoseSecondImageId, kKnownPoseCameraId));
    database_cache.AddImage(
        MakeKnownPoseImage(kKnownPoseThirdImageId, kKnownPoseCameraId));
    mapper.BeginReconstruction(&reconstruction);
  }

  ~KnownPoseRegistrationFixture() { mapper.EndReconstruction(false); }

  KnownPoseState CaptureState() const {
    KnownPoseState state;
    state.structure_revision = reconstruction.StructureRevision();
    state.num_reg_images = reconstruction.NumRegImages();
    state.num_total_reg_images = mapper.NumTotalRegImages();
    state.num_shared_reg_images = mapper.NumSharedRegImages();
    state.reg_image_ids = reconstruction.RegImageIds();
    const Camera& camera = reconstruction.Camera(kKnownPoseCameraId);
    state.camera_model_id = camera.ModelId();
    state.camera_width = camera.Width();
    state.camera_height = camera.Height();
    state.camera_has_prior_focal_length = camera.HasPriorFocalLength();
    state.camera_params = camera.Params();
    for (size_t idx = 0; idx < image_ids.size(); ++idx) {
      const Image& image = reconstruction.Image(image_ids[idx]);
      state.registered[idx] = image.IsRegistered();
      state.poses.col(idx).head<4>() = image.Qvec();
      state.poses.col(idx).tail<3>() = image.Tvec();
    }
    return state;
  }

  void CheckState(const KnownPoseState& expected) const {
    const KnownPoseState actual = CaptureState();
    BOOST_CHECK_EQUAL(actual.structure_revision,
                      expected.structure_revision);
    BOOST_CHECK_EQUAL(actual.num_reg_images, expected.num_reg_images);
    BOOST_CHECK_EQUAL(actual.num_total_reg_images,
                      expected.num_total_reg_images);
    BOOST_CHECK_EQUAL(actual.num_shared_reg_images,
                      expected.num_shared_reg_images);
    BOOST_CHECK(actual.reg_image_ids == expected.reg_image_ids);
    BOOST_CHECK(actual.registered == expected.registered);
    BOOST_CHECK((actual.poses.array() == expected.poses.array()).all());
    BOOST_CHECK_EQUAL(actual.camera_model_id, expected.camera_model_id);
    BOOST_CHECK_EQUAL(actual.camera_width, expected.camera_width);
    BOOST_CHECK_EQUAL(actual.camera_height, expected.camera_height);
    BOOST_CHECK_EQUAL(actual.camera_has_prior_focal_length,
                      expected.camera_has_prior_focal_length);
    BOOST_REQUIRE_EQUAL(actual.camera_params.size(),
                        expected.camera_params.size());
    for (size_t idx = 0; idx < actual.camera_params.size(); ++idx) {
      BOOST_CHECK(actual.camera_params[idx] == expected.camera_params[idx] ||
                  (std::isnan(actual.camera_params[idx]) &&
                   std::isnan(expected.camera_params[idx])));
    }
  }

  void ExpectRejected(
      const image_t image_id,
      const Eigen::Vector4d& qvec,
      const Eigen::Vector3d& tvec,
      const IncrementalMapper::KnownPoseRegistrationReason reason) {
    const KnownPoseState before = CaptureState();
    IncrementalMapper::KnownPoseRegistrationResult result;
    BOOST_CHECK_NO_THROW(
        result = mapper.RegisterImageFromKnownPose(image_id, qvec, tvec));
    BOOST_CHECK(result.status ==
                IncrementalMapper::KnownPoseRegistrationStatus::REJECTED);
    BOOST_CHECK(result.reason == reason);
    BOOST_CHECK(!result.IsSuccess());
    CheckState(before);
  }

  DatabaseCache database_cache;
  Reconstruction reconstruction;
  IncrementalMapper mapper;

 private:
  const std::array<image_t, 3> image_ids = {
      kKnownPoseFirstImageId,
      kKnownPoseSecondImageId,
      kKnownPoseThirdImageId,
  };
};

class LegacyPosePriorRegistrationFixture {
 public:
  LegacyPosePriorRegistrationFixture() : mapper(&database_cache) {
    database_cache.AddCamera(MakeKnownPoseCamera(kKnownPoseCameraId));
    database_cache.AddImage(MakeKnownPoseImage(
        kKnownPoseFirstImageId, kKnownPoseCameraId, 1));
    database_cache.AddImage(MakeKnownPoseImage(
        kKnownPoseSecondImageId, kKnownPoseCameraId, 1));
    database_cache.AddImage(MakeKnownPoseImage(
        kKnownPoseThirdImageId, kKnownPoseCameraId, 1));
    const auto correspondences = database_cache.AddVerifiedCorrespondences(
        kKnownPoseFirstImageId,
        kKnownPoseSecondImageId,
        {FeatureMatch(0, 0)});
    BOOST_REQUIRE(correspondences.IsSuccess());

    poses.emplace(kKnownPoseFirstImageId,
                  std::vector<double>{1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0});
    poses.emplace(kKnownPoseSecondImageId,
                  std::vector<double>{-1.0, 0.5, 4.0, 0.0, 0.0, 0.0, 3.0});
    poses.emplace(kKnownPoseThirdImageId,
                  std::vector<double>{0.25, -0.5, 0.75, 1.0, 2.0, 3.0, 4.0});
    mapper.LoadExistedImagePoses(poses);
    mapper.BeginReconstruction(&reconstruction);
  }

  ~LegacyPosePriorRegistrationFixture() {
    mapper.EndReconstruction(false);
  }

  std::map<uint32_t, std::vector<double>> poses;
  DatabaseCache database_cache;
  Reconstruction reconstruction;
  IncrementalMapper mapper;
  IncrementalMapper::Options options;
};

void CheckRegistrationEvent(
    const ReconstructionStructureReadResult& journal,
    const uint64_t expected_revision,
    const image_t expected_image_id) {
  BOOST_CHECK(journal.complete);
  BOOST_CHECK(!journal.gap);
  BOOST_CHECK_EQUAL(journal.current_revision, expected_revision);
  BOOST_REQUIRE_EQUAL(journal.batches.size(), 1);
  BOOST_REQUIRE_EQUAL(journal.batches.front().events.size(), 1);
  const ReconstructionStructureEvent& event =
      journal.batches.front().events.front();
  BOOST_CHECK(event.kind ==
              ReconstructionStructureEventKind::kImageRegistrationChanged);
  BOOST_CHECK_EQUAL(event.image_id, expected_image_id);
  BOOST_CHECK(!event.old_registration);
  BOOST_CHECK(event.new_registration);
}

}  // namespace

BOOST_FIXTURE_TEST_CASE(KnownPoseRegistersFirstAndSecondWithoutVisualSupport,
                        KnownPoseRegistrationFixture) {
  const CorrespondenceGraph& graph = database_cache.CorrespondenceGraph();
  BOOST_REQUIRE_EQUAL(
      graph.NumCorrespondencesForImage(kKnownPoseFirstImageId), 0);
  BOOST_REQUIRE_EQUAL(
      graph.NumCorrespondencesForImage(kKnownPoseSecondImageId), 0);
  BOOST_REQUIRE_EQUAL(
      reconstruction.Image(kKnownPoseFirstImageId).NumVisiblePoints3D(), 0);
  BOOST_REQUIRE_EQUAL(reconstruction.NumRegImages(), 0);

  constexpr uint64_t kOwnerEpoch = 31;
  reconstruction.BeginStructureJournal(kOwnerEpoch);

  const Eigen::Vector4d first_qvec(2.0, -2.0, 1.0, -1.0);
  const Eigen::Vector3d first_tvec(1.25, -2.5, 3.75);
  const Eigen::Vector4d expected_first_qvec = first_qvec / first_qvec.norm();
  const uint64_t first_cursor = reconstruction.StructureRevision();
  const auto first_result = mapper.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId, first_qvec, first_tvec);
  const auto first_journal =
      reconstruction.ReadStructureEventsSince(kOwnerEpoch, first_cursor);

  const Eigen::Vector4d second_qvec(1.0, 2.0, -3.0, 4.0);
  const Eigen::Vector3d second_tvec(-4.5, 5.25, -6.75);
  const Eigen::Vector4d expected_second_qvec =
      second_qvec / second_qvec.norm();
  const uint64_t second_cursor = reconstruction.StructureRevision();
  const auto second_result = mapper.RegisterImageFromKnownPose(
      kKnownPoseSecondImageId, second_qvec, second_tvec);
  const auto second_journal =
      reconstruction.ReadStructureEventsSince(kOwnerEpoch, second_cursor);
  reconstruction.EndStructureJournal();

  BOOST_REQUIRE(first_result.IsSuccess());
  BOOST_CHECK(first_result.status ==
              IncrementalMapper::KnownPoseRegistrationStatus::SUCCESS);
  BOOST_CHECK(first_result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::NONE);
  BOOST_REQUIRE(second_result.IsSuccess());
  BOOST_CHECK(second_result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::NONE);

  const Image& first_image =
      reconstruction.Image(kKnownPoseFirstImageId);
  const Image& second_image =
      reconstruction.Image(kKnownPoseSecondImageId);
  BOOST_CHECK(first_image.IsRegistered());
  BOOST_CHECK(second_image.IsRegistered());
  BOOST_CHECK((first_image.Qvec().array() ==
               expected_first_qvec.array())
                  .all());
  BOOST_CHECK((first_image.Tvec().array() == first_tvec.array()).all());
  BOOST_CHECK((second_image.Qvec().array() ==
               expected_second_qvec.array())
                  .all());
  BOOST_CHECK((second_image.Tvec().array() == second_tvec.array()).all());
  BOOST_CHECK_EQUAL(first_image.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(second_image.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 2);
  BOOST_CHECK_EQUAL(mapper.NumTotalRegImages(), 2);
  BOOST_CHECK_EQUAL(mapper.NumSharedRegImages(), 0);
  BOOST_CHECK_EQUAL(
      std::count(reconstruction.RegImageIds().begin(),
                 reconstruction.RegImageIds().end(),
                 kKnownPoseFirstImageId),
      1);
  BOOST_CHECK_EQUAL(
      std::count(reconstruction.RegImageIds().begin(),
                 reconstruction.RegImageIds().end(),
                 kKnownPoseSecondImageId),
      1);
  CheckRegistrationEvent(
      first_journal, first_cursor + 1, kKnownPoseFirstImageId);
  CheckRegistrationEvent(
      second_journal, second_cursor + 1, kKnownPoseSecondImageId);
}

BOOST_AUTO_TEST_CASE(KnownPoseRejectsMapperWithoutActiveReconstruction) {
  DatabaseCache database_cache;
  IncrementalMapper mapper(&database_cache);
  const auto result = mapper.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId,
      ComposeIdentityQuaternion(),
      Eigen::Vector3d::Zero());
  BOOST_CHECK(result.status ==
              IncrementalMapper::KnownPoseRegistrationStatus::REJECTED);
  BOOST_CHECK(result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::
                  SESSION_NOT_ACTIVE);

  IncrementalMapper mapper_without_cache(nullptr);
  const auto no_cache_result = mapper_without_cache.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId,
      ComposeIdentityQuaternion(),
      Eigen::Vector3d::Zero());
  BOOST_CHECK(no_cache_result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::
                  SESSION_NOT_ACTIVE);
}

BOOST_AUTO_TEST_CASE(KnownPoseRejectsEndedSessionWithoutMutation) {
  DatabaseCache database_cache;
  database_cache.AddCamera(MakeKnownPoseCamera(kKnownPoseCameraId));
  database_cache.AddImage(
      MakeKnownPoseImage(kKnownPoseFirstImageId, kKnownPoseCameraId));
  Reconstruction reconstruction;
  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);
  BOOST_REQUIRE(mapper
                    .RegisterImageFromKnownPose(
                        kKnownPoseFirstImageId,
                        ComposeIdentityQuaternion(),
                        Eigen::Vector3d(1.0, 2.0, 3.0))
                    .IsSuccess());
  mapper.EndReconstruction(false);

  const Eigen::Vector4d qvec =
      reconstruction.Image(kKnownPoseFirstImageId).Qvec();
  const Eigen::Vector3d tvec =
      reconstruction.Image(kKnownPoseFirstImageId).Tvec();
  const std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
  const size_t num_total_reg_images = mapper.NumTotalRegImages();
  const size_t num_shared_reg_images = mapper.NumSharedRegImages();

  const auto result = mapper.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId,
      Eigen::Vector4d(1.0, 2.0, 3.0, 4.0),
      Eigen::Vector3d(-1.0, -2.0, -3.0));

  BOOST_CHECK(result.status ==
              IncrementalMapper::KnownPoseRegistrationStatus::REJECTED);
  BOOST_CHECK(result.reason ==
              IncrementalMapper::KnownPoseRegistrationReason::
                  SESSION_NOT_ACTIVE);
  BOOST_CHECK(reconstruction.Image(kKnownPoseFirstImageId)
                  .Qvec()
                  .isApprox(qvec, 0.0));
  BOOST_CHECK(reconstruction.Image(kKnownPoseFirstImageId)
                  .Tvec()
                  .isApprox(tvec, 0.0));
  BOOST_CHECK(reconstruction.RegImageIds() == reg_image_ids);
  BOOST_CHECK_EQUAL(mapper.NumTotalRegImages(), num_total_reg_images);
  BOOST_CHECK_EQUAL(mapper.NumSharedRegImages(), num_shared_reg_images);
}

BOOST_FIXTURE_TEST_CASE(KnownPoseInvalidInputsDoNotMutateState,
                        KnownPoseRegistrationFixture) {
  const Eigen::Vector4d identity = ComposeIdentityQuaternion();
  const Eigen::Vector3d zero_translation = Eigen::Vector3d::Zero();

  ExpectRejected(kInvalidImageId,
                 identity,
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     INVALID_IMAGE_ID);
  ExpectRejected(kKnownPoseMissingImageId,
                 identity,
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     IMAGE_NOT_FOUND);

  Eigen::Vector4d nan_qvec = identity;
  nan_qvec(2) = std::numeric_limits<double>::quiet_NaN();
  ExpectRejected(kKnownPoseFirstImageId,
                 nan_qvec,
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     POSE_NOT_FINITE);

  Eigen::Vector3d nan_tvec = zero_translation;
  nan_tvec(1) = std::numeric_limits<double>::quiet_NaN();
  ExpectRejected(kKnownPoseFirstImageId,
                 identity,
                 nan_tvec,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     POSE_NOT_FINITE);

  Eigen::Vector4d infinite_qvec = identity;
  infinite_qvec(0) = std::numeric_limits<double>::infinity();
  ExpectRejected(kKnownPoseFirstImageId,
                 infinite_qvec,
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     POSE_NOT_FINITE);

  ExpectRejected(kKnownPoseFirstImageId,
                 Eigen::Vector4d::Zero(),
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     INVALID_QUATERNION);

  ExpectRejected(kKnownPoseFirstImageId,
                 Eigen::Vector4d::Constant(
                     std::numeric_limits<double>::denorm_min()),
                 zero_translation,
                 IncrementalMapper::KnownPoseRegistrationReason::
                     INVALID_QUATERNION);
}

BOOST_FIXTURE_TEST_CASE(KnownPoseNormalizesHugeFiniteQuaternion,
                        KnownPoseRegistrationFixture) {
  const Eigen::Vector4d qvec(1e308, -1e308, 5e307, -5e307);
  const double scale = qvec.cwiseAbs().maxCoeff();
  const Eigen::Vector4d scaled_qvec = qvec / scale;
  const Eigen::Vector4d expected_qvec = scaled_qvec / scaled_qvec.norm();

  const auto result = mapper.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId, qvec, Eigen::Vector3d(1.0, -2.0, 3.0));

  BOOST_REQUIRE(result.IsSuccess());
  const Eigen::Vector4d actual_qvec =
      reconstruction.Image(kKnownPoseFirstImageId).Qvec();
  BOOST_CHECK(actual_qvec.allFinite());
  BOOST_CHECK(actual_qvec.isApprox(expected_qvec, 1e-15));
  BOOST_CHECK_SMALL(std::abs(actual_qvec.norm() - 1.0), 1e-15);
}

BOOST_FIXTURE_TEST_CASE(KnownPoseInvalidCameraDefinitionsDoNotMutateState,
                        KnownPoseRegistrationFixture) {
  const Camera valid_camera = reconstruction.Camera(kKnownPoseCameraId);
  std::vector<Camera> invalid_cameras;

  Camera zero_width = valid_camera;
  zero_width.SetWidth(0);
  invalid_cameras.push_back(zero_width);

  Camera zero_height = valid_camera;
  zero_height.SetHeight(0);
  invalid_cameras.push_back(zero_height);

  Camera invalid_model;
  invalid_model.SetCameraId(kKnownPoseCameraId);
  invalid_model.SetWidth(valid_camera.Width());
  invalid_model.SetHeight(valid_camera.Height());
  invalid_model.SetParams(valid_camera.Params());
  invalid_model.SetPriorFocalLength(valid_camera.HasPriorFocalLength());
  invalid_cameras.push_back(invalid_model);

  Camera invalid_params = valid_camera;
  invalid_params.SetParams({});
  invalid_cameras.push_back(invalid_params);

  Camera nan_params = valid_camera;
  nan_params.Params().front() = std::numeric_limits<double>::quiet_NaN();
  invalid_cameras.push_back(nan_params);

  Camera infinite_params = valid_camera;
  infinite_params.Params().back() = std::numeric_limits<double>::infinity();
  invalid_cameras.push_back(infinite_params);

  for (const Camera& invalid_camera : invalid_cameras) {
    reconstruction.Camera(kKnownPoseCameraId) = invalid_camera;
    ExpectRejected(kKnownPoseFirstImageId,
                   ComposeIdentityQuaternion(),
                   Eigen::Vector3d::Zero(),
                   IncrementalMapper::KnownPoseRegistrationReason::
                       INVALID_CAMERA);
    reconstruction.Camera(kKnownPoseCameraId) = valid_camera;
  }

  const Camera valid_cached_camera = database_cache.Camera(kKnownPoseCameraId);
  database_cache.Camera(kKnownPoseCameraId).Params().front() =
      std::numeric_limits<double>::quiet_NaN();
  ExpectRejected(kKnownPoseFirstImageId,
                 ComposeIdentityQuaternion(),
                 Eigen::Vector3d::Zero(),
                 IncrementalMapper::KnownPoseRegistrationReason::
                     INVALID_CAMERA);
  database_cache.Camera(kKnownPoseCameraId) = valid_cached_camera;
}

BOOST_FIXTURE_TEST_CASE(KnownPoseDuplicateDoesNotMutateState,
                        KnownPoseRegistrationFixture) {
  const auto first_result = mapper.RegisterImageFromKnownPose(
      kKnownPoseFirstImageId,
      ComposeIdentityQuaternion(),
      Eigen::Vector3d(1.0, 2.0, 3.0));
  BOOST_REQUIRE(first_result.IsSuccess());

  ExpectRejected(kKnownPoseFirstImageId,
                 Eigen::Vector4d(1.0, 2.0, 3.0, 4.0),
                 Eigen::Vector3d(-1.0, -2.0, -3.0),
                 IncrementalMapper::KnownPoseRegistrationReason::
                     IMAGE_ALREADY_REGISTERED);
}

BOOST_FIXTURE_TEST_CASE(KnownPoseCacheMismatchDoesNotMutateState,
                        KnownPoseRegistrationFixture) {
  database_cache.Image(kKnownPoseThirdImageId)
      .SetCameraId(kKnownPoseOtherCameraId);
  ExpectRejected(kKnownPoseThirdImageId,
                 ComposeIdentityQuaternion(),
                 Eigen::Vector3d::Zero(),
                 IncrementalMapper::KnownPoseRegistrationReason::
                     CACHE_MISMATCH);
}

BOOST_FIXTURE_TEST_CASE(LegacyPosePriorRegistrationStillWorks,
                        LegacyPosePriorRegistrationFixture) {
  BOOST_REQUIRE(mapper.RegisterInitialImagePairFromPosePrior(
      options, kKnownPoseFirstImageId, kKnownPoseSecondImageId));
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 2);

  const Eigen::Vector4d first_input(
      poses.at(kKnownPoseFirstImageId)[3],
      poses.at(kKnownPoseFirstImageId)[4],
      poses.at(kKnownPoseFirstImageId)[5],
      poses.at(kKnownPoseFirstImageId)[6]);
  const Eigen::Vector3d first_translation(
      poses.at(kKnownPoseFirstImageId)[0],
      poses.at(kKnownPoseFirstImageId)[1],
      poses.at(kKnownPoseFirstImageId)[2]);
  BOOST_CHECK(reconstruction.Image(kKnownPoseFirstImageId)
                  .Qvec()
                  .isApprox(first_input / first_input.norm(), 0.0));
  BOOST_CHECK(reconstruction.Image(kKnownPoseFirstImageId)
                  .Tvec()
                  .isApprox(first_translation, 0.0));

  BOOST_REQUIRE(mapper.RegisterNextImageFromPosePrior(
      options, kKnownPoseThirdImageId));
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 3);
  BOOST_CHECK_EQUAL(mapper.NumTotalRegImages(), 3);
  BOOST_CHECK_EQUAL(mapper.NumSharedRegImages(), 0);

  const Eigen::Vector4d third_input(
      poses.at(kKnownPoseThirdImageId)[3],
      poses.at(kKnownPoseThirdImageId)[4],
      poses.at(kKnownPoseThirdImageId)[5],
      poses.at(kKnownPoseThirdImageId)[6]);
  const Eigen::Vector3d third_translation(
      poses.at(kKnownPoseThirdImageId)[0],
      poses.at(kKnownPoseThirdImageId)[1],
      poses.at(kKnownPoseThirdImageId)[2]);
  BOOST_CHECK(reconstruction.Image(kKnownPoseThirdImageId)
                  .Qvec()
                  .isApprox(third_input / third_input.norm(), 0.0));
  BOOST_CHECK(reconstruction.Image(kKnownPoseThirdImageId)
                  .Tvec()
                  .isApprox(third_translation, 0.0));
}
