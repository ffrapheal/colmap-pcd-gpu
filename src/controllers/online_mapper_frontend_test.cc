// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#define TEST_NAME "controllers/online_mapper_frontend_test"
#include "util/testing.h"

#include "controllers/online_mapper_frontend.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "base/database.h"
#include "base/database_cache.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "feature/extraction.h"
#include "util/bitmap.h"

using namespace colmap;

namespace {

Camera CreateCamera() {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 500.0, 640, 480);
  camera.SetCameraId(1);
  camera.SetPriorFocalLength(true);
  return camera;
}

Image CreateImage(const image_t image_id) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(1);
  image.SetName("frame_" + std::to_string(image_id) + ".jpg");
  return image;
}

KnownPoseSE3 PoseFromCameraCenter(const Eigen::Vector3d& camera_center,
                                 const double yaw_degrees = 0.0) {
  const double yaw_radians =
      yaw_degrees * 3.14159265358979323846 / 180.0;
  const Eigen::Quaterniond quaternion(
      Eigen::AngleAxisd(yaw_radians, Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d tvec =
      -(quaternion.toRotationMatrix() * camera_center);
  return KnownPoseSE3(
      Eigen::Vector4d(quaternion.w(), quaternion.x(), quaternion.y(),
                      quaternion.z()),
      tvec);
}

FeatureKeypoints CreateKeypoints(const float offset,
                                 const size_t count = 8) {
  const std::vector<std::pair<float, float>> points = {
      {100.0f, 100.0f}, {200.0f, 100.0f}, {300.0f, 120.0f},
      {120.0f, 200.0f}, {220.0f, 220.0f}, {340.0f, 200.0f},
      {160.0f, 320.0f}, {300.0f, 330.0f}};
  CHECK_LE(count, points.size());
  FeatureKeypoints keypoints;
  keypoints.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto& point = points[i];
    keypoints.emplace_back(point.first + offset, point.second + offset);
  }
  return keypoints;
}

FeatureDescriptors CreateDescriptors(const size_t count = 8) {
  FeatureDescriptors descriptors(static_cast<int>(count), 128);
  descriptors.setZero();
  for (int row = 0; row < descriptors.rows(); ++row) {
    for (int offset = 0; offset < 4; ++offset) {
      descriptors(row, 4 * row + offset) = 255;
    }
  }
  return descriptors;
}

FeatureMatches CreateIdentityMatches(const size_t count) {
  FeatureMatches matches;
  matches.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    const point2D_t point2D_idx = static_cast<point2D_t>(index);
    matches.emplace_back(point2D_idx, point2D_idx);
  }
  return matches;
}

struct ExtractionOutput {
  bool success = true;
  std::string rejection_reason;
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

class FakeSingleImageFeatureExtractor : public SingleImageFeatureExtractor {
 public:
  explicit FakeSingleImageFeatureExtractor(
      std::vector<ExtractionOutput> outputs,
      std::function<void()> after_successful_extract = {})
      : outputs_(std::move(outputs)),
        after_successful_extract_(std::move(after_successful_extract)) {}

  SingleImageFeatureExtractionResult Setup() override {
    ++setup_count_;
    is_setup_ = true;
    return {true, ""};
  }

  SingleImageFeatureExtractionResult Extract(
      const Camera&,
      Bitmap*,
      const Bitmap*,
      FeatureKeypoints* keypoints,
      FeatureDescriptors* descriptors) override {
    ++extract_count_;
    if (next_output_ >= outputs_.size()) {
      return {false, "no fake extraction output"};
    }
    const ExtractionOutput& output = outputs_[next_output_++];
    if (!output.success) {
      return {false, output.rejection_reason};
    }
    *keypoints = output.keypoints;
    *descriptors = output.descriptors;
    if (after_successful_extract_) {
      after_successful_extract_();
    }
    return {true, ""};
  }

  bool IsSetup() const override { return is_setup_; }
  size_t SetupCount() const override { return setup_count_; }
  size_t ExtractCount() const override { return extract_count_; }
  bool UsesCuda() const override { return false; }

 private:
  std::vector<ExtractionOutput> outputs_;
  std::function<void()> after_successful_extract_;
  size_t next_output_ = 0;
  bool is_setup_ = false;
  size_t setup_count_ = 0;
  size_t extract_count_ = 0;
};

ExtractionOutput SuccessfulExtraction(const float offset) {
  ExtractionOutput output;
  output.keypoints = CreateKeypoints(offset);
  output.descriptors = CreateDescriptors();
  return output;
}

ExtractionOutput SuccessfulExtraction(const float offset,
                                      const size_t count) {
  ExtractionOutput output;
  output.keypoints = CreateKeypoints(offset, count);
  output.descriptors = CreateDescriptors(count);
  return output;
}

SiftMatchingOptions CreateMatchingOptions() {
  SiftMatchingOptions options;
  options.use_gpu = false;
  options.num_threads = 1;
  options.max_num_matches = 32;
  options.min_num_inliers = 1;
  options.max_error = 0.5;
  options.planar_scene = true;
  return options;
}

struct FrontendState {
  FrontendState()
      : database(":memory:"), matcher_cache(32, &database) {
    const Camera camera = CreateCamera();
    BOOST_REQUIRE_EQUAL(database.WriteCamera(camera, true), camera.CameraId());
    matcher_cache.Setup();
    database_cache.Load(database, 1, false, {});
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());
  }

  Database database;
  FeatureMatcherCache matcher_cache;
  DatabaseCache database_cache;
  Reconstruction reconstruction;
};

void AddReadyImage(FrontendState* state,
                   const image_t image_id,
                   const float keypoint_offset,
                   const size_t graph_point_count) {
  const Image image = CreateImage(image_id);
  const FeatureKeypoints keypoints = CreateKeypoints(keypoint_offset);
  const FeatureDescriptors descriptors = CreateDescriptors();
  BOOST_REQUIRE(graph_point_count <= keypoints.size());
  BOOST_REQUIRE_EQUAL(state->database.WriteImage(image, true), image_id);
  state->database.WriteKeypoints(image_id, keypoints);
  state->database.WriteDescriptors(image_id, descriptors);
  BOOST_REQUIRE(state->matcher_cache.AddImage(image, keypoints, descriptors));

  const FeatureKeypoints graph_keypoints(
      keypoints.begin(), keypoints.begin() + graph_point_count);
  BOOST_REQUIRE(state->database_cache
                    .AddImageWithKeypoints(image, graph_keypoints)
                    .IsSuccess());
  BOOST_REQUIRE_EQUAL(state->database_cache.Image(image_id).NumPoints2D(),
                      graph_point_count);
  BOOST_REQUIRE(state->reconstruction.AddImageFromDatabaseCache(
      state->database_cache, image_id));
  BOOST_REQUIRE_EQUAL(state->reconstruction.Image(image_id).NumPoints2D(),
                      graph_point_count);
}

Bitmap CreateBitmap() {
  Bitmap bitmap;
  BOOST_REQUIRE(bitmap.Allocate(640, 480, false));
  return bitmap;
}

void CheckExactIdentityMatches(const FeatureMatches& matches,
                               const size_t expected_count) {
  BOOST_REQUIRE_EQUAL(matches.size(), expected_count);
  for (size_t i = 0; i < matches.size(); ++i) {
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, i);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, i);
  }
}

}  // namespace

BOOST_AUTO_TEST_CASE(
    ConfirmedNormalizedSE3SelectorUsesRotationAndBothVisualStates) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(
                            30, 1,
                            PoseFromCameraCenter(Eigen::Vector3d(0.01, 0.0, 0.0),
                                                 180.0))
                    .IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(
                            20, 2,
                            PoseFromCameraCenter(Eigen::Vector3d(0.1, 0.0, 0.0)))
                    .IsSuccess());
  BOOST_REQUIRE(registry.PromoteToVisualActive(
                            20,
                            PoseFromCameraCenter(Eigen::Vector3d(0.1, 0.0, 0.0)))
                    .IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(
                            99, 3, PoseFromCameraCenter(Eigen::Vector3d::Zero()))
                    .IsSuccess());
  BOOST_REQUIRE(registry.PromoteToVisualActive(
                            99,
                            PoseFromCameraCenter(Eigen::Vector3d(100.0, 0.0, 0.0),
                                                 90.0))
                    .IsSuccess());

  ConfirmedNormalizedSE3SelectorOptions options;
  options.translation_scale_meters = 0.1;
  options.rotation_scale_degrees = 20.0;
  ConfirmedNormalizedSE3ReferenceSelector selector(options);
  const auto selection = selector.Select(registry, 99);

  BOOST_REQUIRE(selection.IsSuccess());
  BOOST_REQUIRE_EQUAL(selection.references.size(), 2);
  BOOST_CHECK_EQUAL(selection.references[0].image_id, 20);
  BOOST_CHECK_EQUAL(selection.references[1].image_id, 30);
  BOOST_CHECK_CLOSE(selection.references[0].score, 1.0, 1e-9);
  BOOST_CHECK_CLOSE(selection.references[1].translation_distance_meters, 0.01,
                    1e-9);
  BOOST_CHECK_CLOSE(selection.references[1].rotation_distance_degrees, 180.0,
                    1e-9);
  BOOST_CHECK(selection.references[0].visual_state ==
              KnownPoseVisualState::VISUAL_ACTIVE);
  BOOST_CHECK(selection.references[1].visual_state ==
              KnownPoseVisualState::POSE_ONLY);
  BOOST_CHECK(std::none_of(
      selection.references.begin(), selection.references.end(),
      [](const ConfirmedNormalizedSE3Reference& reference) {
        return reference.image_id == 99;
      }));
}

BOOST_AUTO_TEST_CASE(ConfirmedNormalizedSE3SelectorUsesLatestHistoricalPose) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(
                            9, 1,
                            PoseFromCameraCenter(Eigen::Vector3d(0.02, 0.0, 0.0)))
                    .IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(
                            7, 2,
                            PoseFromCameraCenter(Eigen::Vector3d(0.1, 0.0, 0.0)))
                    .IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(
                            6, 3,
                            PoseFromCameraCenter(
                                Eigen::Vector3d(1000000.0, 0.0, 0.0)))
                    .IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(
                            100, 4,
                            PoseFromCameraCenter(Eigen::Vector3d::Zero()))
                    .IsSuccess());

  ConfirmedNormalizedSE3SelectorOptions default_options;
  BOOST_CHECK_EQUAL(default_options.translation_scale_meters, 0.05);
  BOOST_CHECK_EQUAL(default_options.rotation_scale_degrees, 10.0);
  ConfirmedNormalizedSE3ReferenceSelector selector;
  auto selection = selector.Select(registry, 100);
  BOOST_REQUIRE(selection.IsSuccess());
  BOOST_REQUIRE_EQUAL(selection.references.size(), 3);
  BOOST_CHECK_EQUAL(selection.references[0].image_id, 9);
  BOOST_CHECK_CLOSE(selection.references[0].score, 0.4, 1e-9);
  BOOST_CHECK_EQUAL(selection.references[2].image_id, 6);

  BOOST_REQUIRE(registry.PromoteToVisualActive(
                            9,
                            PoseFromCameraCenter(Eigen::Vector3d(1.0, 0.0, 0.0)))
                    .IsSuccess());
  selection = selector.Select(registry, 100);
  BOOST_REQUIRE(selection.IsSuccess());
  BOOST_CHECK_EQUAL(selection.references[0].image_id, 7);

  BOOST_REQUIRE(registry.UpdateLatestPose(
                            9,
                            PoseFromCameraCenter(Eigen::Vector3d(0.01, 0.0, 0.0)))
                    .IsSuccess());
  selection = selector.Select(registry, 100);
  BOOST_REQUIRE(selection.IsSuccess());
  BOOST_CHECK_EQUAL(selection.references[0].image_id, 9);
}

BOOST_AUTO_TEST_CASE(ConfirmedNormalizedSE3SelectorCapsAtTwentyAndBreaksTies) {
  KnownPoseRegistry registry;
  for (size_t frame_index = 1; frame_index <= 22; ++frame_index) {
    const image_t image_id = static_cast<image_t>(101 - frame_index);
    BOOST_REQUIRE(registry.AddKnownPose(
                              image_id, frame_index,
                              PoseFromCameraCenter(Eigen::Vector3d::Zero()))
                      .IsSuccess());
  }
  BOOST_REQUIRE(registry.AddKnownPose(
                            200, 23,
                            PoseFromCameraCenter(Eigen::Vector3d::Zero()))
                    .IsSuccess());

  ConfirmedNormalizedSE3ReferenceSelector selector;
  const auto selection = selector.Select(registry, 200);
  BOOST_REQUIRE(selection.IsSuccess());
  BOOST_REQUIRE_EQUAL(
      selection.references.size(),
      ConfirmedNormalizedSE3ReferenceSelector::kMaxCandidateCount);
  for (size_t i = 0; i < selection.references.size(); ++i) {
    BOOST_CHECK_EQUAL(selection.references[i].image_id,
                      static_cast<image_t>(100 - i));
    BOOST_CHECK_SMALL(selection.references[i].score, 1e-12);
    BOOST_CHECK_EQUAL(selection.references[i].registration_sequence, i + 1);
  }
}

BOOST_AUTO_TEST_CASE(RejectsInvalidOptionsBeforeExtractionOrMutation) {
  OnlineMapperFrontendOptions options;
  BOOST_CHECK(options.Check());
  options.max_references_per_batch = 0;
  BOOST_CHECK(!options.Check());
  options.max_references_per_batch =
      OnlineMapperFrontendOptions::kDefaultMaxReferencesPerBatch + 1;
  BOOST_CHECK(!options.Check());
  options.max_references_per_batch =
      OnlineMapperFrontendOptions::kDefaultMaxReferencesPerBatch;
  options.max_references_per_current =
      OnlineMapperFrontendOptions::kDefaultMaxReferencesPerCurrent + 1;
  BOOST_CHECK(!options.Check());

  FrontendState state;
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(0.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  options.max_references_per_batch = 0;
  OnlineMapperFrontend frontend(options, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);
  Bitmap bitmap = CreateBitmap();

  const auto result =
      frontend.ProcessFrame(CreateImage(1), &bitmap, nullptr, {});
  BOOST_CHECK(result.status == OnlineMapperFrameStatus::REJECTED);
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::INVALID_OPTIONS);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 0);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 0);
  BOOST_CHECK(state.matcher_cache.GetImageIds().empty());
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImages(), 0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(RejectsNonUnitMatcherMinimumBeforeIngestMutation) {
  FrontendState state;
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(0.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftMatchingOptions matching_options = CreateMatchingOptions();
  matching_options.min_num_inliers = 4;
  SiftFeatureMatcher matcher(matching_options, &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);
  Bitmap bitmap = CreateBitmap();

  const auto result = frontend.IngestFrame(CreateImage(1), &bitmap, nullptr);
  BOOST_CHECK(result.status == OnlineMapperFrameStatus::REJECTED);
  BOOST_CHECK(result.reason ==
              OnlineMapperFrameReason::INVALID_MATCHER_CONFIGURATION);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 0);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 0);
  BOOST_CHECK(state.matcher_cache.GetImageIds().empty());
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(RejectsMatcherBeforeSetupWithoutMutation) {
  FrontendState state;
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(0.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_CHECK(!matcher.IsSetup());
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);
  Bitmap bitmap = CreateBitmap();

  const auto result =
      frontend.ProcessFrame(CreateImage(1), &bitmap, nullptr, {});
  BOOST_CHECK(result.status == OnlineMapperFrameStatus::REJECTED);
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::MATCHER_NOT_READY);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 0);
  BOOST_CHECK(!matcher.IsSetup());
  BOOST_CHECK_EQUAL(state.database.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 0);
  BOOST_CHECK(state.matcher_cache.GetImageIds().empty());
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImages(), 0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(ExtractionFailureDoesNotMutatePersistentOrLiveState) {
  FrontendState state;
  ExtractionOutput failure;
  failure.success = false;
  failure.rejection_reason = "synthetic extraction failure";
  FakeSingleImageFeatureExtractor extractor({failure});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);
  Bitmap bitmap = CreateBitmap();

  const auto result =
      frontend.ProcessFrame(CreateImage(1), &bitmap, nullptr, {});
  BOOST_CHECK(result.status == OnlineMapperFrameStatus::EXTRACTION_FAILED);
  BOOST_CHECK(result.reason ==
              OnlineMapperFrameReason::FEATURE_EXTRACTION_FAILED);
  BOOST_CHECK_EQUAL(result.detail, "synthetic extraction failure");
  BOOST_CHECK_EQUAL(state.database.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 0);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 0);
  BOOST_CHECK(state.matcher_cache.GetImageIds().empty());
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImages(), 0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(ProcessesOrderedPairsAndReusesPersistentFrontendState) {
  FrontendState state;
  FakeSingleImageFeatureExtractor extractor(
      {SuccessfulExtraction(0.0f), SuccessfulExtraction(5.0f),
       SuccessfulExtraction(10.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap1 = CreateBitmap();
  const auto first =
      frontend.ProcessFrame(CreateImage(1), &bitmap1, nullptr, {});
  BOOST_REQUIRE(first.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_CHECK(first.pairs.empty());

  Bitmap bitmap2 = CreateBitmap();
  const auto second =
      frontend.ProcessFrame(CreateImage(2), &bitmap2, nullptr, {1});
  BOOST_REQUIRE(second.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(second.pairs.size(), 1);
  BOOST_CHECK_EQUAL(second.pairs[0].image_id, 2);
  BOOST_CHECK_EQUAL(second.pairs[0].reference_image_id, 1);
  BOOST_CHECK(second.pairs[0].matcher_status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(second.pairs[0].raw_match_count, 8);
  BOOST_CHECK_EQUAL(second.pairs[0].verified_inlier_count, 8);
  BOOST_CHECK_EQUAL(second.keypoint_count, 8);
  BOOST_CHECK_EQUAL(second.descriptor_count, 8);
  BOOST_CHECK(second.pairs[0].geometry_configuration ==
              TwoViewGeometry::PLANAR_OR_PANORAMIC);
  BOOST_CHECK(second.pairs[0].graph_insertion_attempted);
  BOOST_CHECK(second.pairs[0].graph_insertion.IsSuccess());
  BOOST_CHECK_EQUAL(second.pairs[0].graph_insertion.num_added_matches, 8);
  BOOST_CHECK(second.pairs[0].reconstruction_pair_sync_success);

  Bitmap bitmap3 = CreateBitmap();
  const auto third =
      frontend.ProcessFrame(CreateImage(3), &bitmap3, nullptr, {2, 1});
  BOOST_REQUIRE(third.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(third.pairs.size(), 2);
  BOOST_CHECK_EQUAL(third.pairs[0].reference_image_id, 2);
  BOOST_CHECK_EQUAL(third.pairs[1].reference_image_id, 1);
  BOOST_CHECK_EQUAL(third.pairs[0].raw_match_count, 8);
  BOOST_CHECK_EQUAL(third.pairs[0].verified_inlier_count, 8);
  BOOST_CHECK_EQUAL(third.pairs[1].raw_match_count, 8);
  BOOST_CHECK_EQUAL(third.pairs[1].verified_inlier_count, 8);
  BOOST_CHECK(third.pairs[0].reconstruction_pair_sync_success);
  BOOST_CHECK(third.pairs[1].reconstruction_pair_sync_success);

  BOOST_CHECK_EQUAL(extractor.SetupCount(), 1);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 3);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 24);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 24);
  BOOST_CHECK_EQUAL(state.matcher_cache.GetImageIds().size(), 3);
  BOOST_CHECK(state.matcher_cache.ExistsKeypoints(3));
  BOOST_CHECK(state.matcher_cache.ExistsDescriptors(3));
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    3);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 3);
  CheckExactIdentityMatches(
      state.database_cache.CorrespondenceGraph()
          .FindCorrespondencesBetweenImages(2, 1),
      8);
  CheckExactIdentityMatches(
      state.database_cache.CorrespondenceGraph()
          .FindCorrespondencesBetweenImages(3, 2),
      8);
  BOOST_CHECK_EQUAL(state.reconstruction.ImagePair(2, 1).num_total_corrs, 8);
  BOOST_CHECK_EQUAL(state.reconstruction.ImagePair(3, 2).num_total_corrs, 8);
  BOOST_CHECK(!state.reconstruction.Image(1).IsRegistered());
  BOOST_CHECK(!state.reconstruction.Image(2).IsRegistered());
  BOOST_CHECK(!state.reconstruction.Image(3).IsRegistered());
}

BOOST_AUTO_TEST_CASE(IngestsOnceThenMatchesTwoOrderedBatchesAndCachesPairs) {
  FrontendState state;
  for (image_t image_id = 1; image_id <= 11; ++image_id) {
    AddReadyImage(&state, image_id, static_cast<float>(image_id), 8);
  }
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(20.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap = CreateBitmap();
  const auto ingest = frontend.IngestFrame(CreateImage(12), &bitmap, nullptr);
  BOOST_REQUIRE(ingest.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_CHECK_EQUAL(ingest.keypoint_count, 8);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);

  const std::vector<image_t> first_reference_ids = {1, 2, 3, 4, 5};
  const auto first_batch =
      frontend.MatchExplicitReferences(12, first_reference_ids);
  BOOST_REQUIRE(first_batch.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(first_batch.pairs.size(), first_reference_ids.size());
  for (size_t i = 0; i < first_batch.pairs.size(); ++i) {
    BOOST_CHECK_EQUAL(first_batch.pairs[i].reference_image_id,
                      first_reference_ids[i]);
    BOOST_CHECK(first_batch.pairs[i].submission_status ==
                OnlineMapperPairSubmissionStatus::PROCESSED);
  }

  const std::vector<image_t> second_reference_ids = {6, 7, 8, 9, 10};
  const auto second_batch =
      frontend.MatchExplicitReferences(12, second_reference_ids);
  BOOST_REQUIRE(second_batch.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(second_batch.pairs.size(), second_reference_ids.size());
  for (size_t i = 0; i < second_batch.pairs.size(); ++i) {
    BOOST_CHECK_EQUAL(second_batch.pairs[i].reference_image_id,
                      second_reference_ids[i]);
    BOOST_CHECK(second_batch.pairs[i].submission_status ==
                OnlineMapperPairSubmissionStatus::PROCESSED);
  }

  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 12);
  BOOST_CHECK_EQUAL(state.database.NumKeypoints(), 96);
  BOOST_CHECK_EQUAL(state.database.NumDescriptors(), 96);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    10);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 10);

  const size_t database_match_count = state.database.NumMatches();
  const size_t graph_pair_count =
      state.database_cache.CorrespondenceGraph().NumImagePairs();
  const size_t reconstruction_pair_count = state.reconstruction.NumImagePairs();
  const auto over_current_limit = frontend.MatchExplicitReferences(12, {11});
  BOOST_CHECK(over_current_limit.reason ==
              OnlineMapperFrameReason::TOO_MANY_REFERENCES_FOR_CURRENT);
  BOOST_CHECK_EQUAL(state.database.NumMatches(), database_match_count);
  BOOST_CHECK_EQUAL(
      state.database_cache.CorrespondenceGraph().NumImagePairs(),
      graph_pair_count);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(),
                    reconstruction_pair_count);

  const auto duplicate = frontend.MatchExplicitReferences(12, {3});
  BOOST_REQUIRE(duplicate.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(duplicate.pairs.size(), 1);
  BOOST_CHECK(duplicate.pairs[0].submission_status ==
              OnlineMapperPairSubmissionStatus::CACHED);
  BOOST_CHECK_EQUAL(duplicate.pairs[0].raw_match_count, 8);
  BOOST_CHECK_EQUAL(duplicate.pairs[0].verified_inlier_count, 8);
  BOOST_CHECK(!duplicate.pairs[0].graph_insertion_attempted);
  BOOST_CHECK(!duplicate.pairs[0].reconstruction_pair_sync_attempted);
  BOOST_CHECK_EQUAL(state.database.NumMatches(), database_match_count);
  BOOST_CHECK_EQUAL(
      state.database_cache.CorrespondenceGraph().NumImagePairs(),
      graph_pair_count);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(),
                    reconstruction_pair_count);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);
}

BOOST_AUTO_TEST_CASE(MatchStageRejectsInvalidStateWithoutMutation) {
  FrontendState state;
  for (image_t image_id = 1; image_id <= 6; ++image_id) {
    AddReadyImage(&state, image_id, static_cast<float>(image_id), 8);
  }
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(10.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  const size_t initial_database_match_count = state.database.NumMatches();
  const size_t initial_graph_pair_count =
      state.database_cache.CorrespondenceGraph().NumImagePairs();
  const size_t initial_reconstruction_pair_count =
      state.reconstruction.NumImagePairs();
  auto result = frontend.MatchExplicitReferences(7, {1});
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::NO_INGESTED_CURRENT);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 0);
  BOOST_CHECK_EQUAL(state.database.NumMatches(), initial_database_match_count);

  Bitmap bitmap = CreateBitmap();
  BOOST_REQUIRE(frontend.IngestFrame(CreateImage(7), &bitmap, nullptr).status ==
                OnlineMapperFrameStatus::SUCCESS);
  const size_t image_count_after_ingest = state.database.NumImages();
  const size_t match_count_after_ingest = state.database.NumMatches();
  result = frontend.MatchExplicitReferences(8, {1});
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::WRONG_CURRENT_IMAGE);
  result = frontend.MatchExplicitReferences(7, {1, 2, 3, 4, 5, 6});
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::TOO_MANY_REFERENCES);

  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);
  BOOST_CHECK_EQUAL(state.database.NumImages(), image_count_after_ingest);
  BOOST_CHECK_EQUAL(state.database.NumMatches(), match_count_after_ingest);
  BOOST_CHECK_EQUAL(
      state.database_cache.CorrespondenceGraph().NumImagePairs(),
      initial_graph_pair_count);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(),
                    initial_reconstruction_pair_count);
}

BOOST_AUTO_TEST_CASE(RejectsInvalidReferencesWithoutMatchingMutation) {
  FrontendState state;
  FakeSingleImageFeatureExtractor extractor(
      {SuccessfulExtraction(0.0f), SuccessfulExtraction(5.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);
  Bitmap bitmap = CreateBitmap();
  BOOST_REQUIRE(frontend.ProcessFrame(CreateImage(1), &bitmap, nullptr, {})
                    .status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(extractor.ExtractCount(), 1);
  BOOST_REQUIRE(frontend.IngestFrame(CreateImage(2), &bitmap, nullptr).status ==
                OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(extractor.ExtractCount(), 2);

  auto result = frontend.MatchExplicitReferences(2, {1, 1});
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::DUPLICATE_REFERENCE);
  result = frontend.MatchExplicitReferences(2, {2});
  BOOST_CHECK(result.reason ==
              OnlineMapperFrameReason::CURRENT_IMAGE_IS_REFERENCE);
  result = frontend.MatchExplicitReferences(2, {99});
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::REFERENCE_NOT_READY);
  result = frontend.IngestFrame(CreateImage(2), &bitmap, nullptr);
  BOOST_CHECK(result.reason == OnlineMapperFrameReason::DUPLICATE_IMAGE);

  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 2);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 2);
  BOOST_CHECK_EQUAL(state.matcher_cache.GetImageIds().size(), 2);
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 2);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 2);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    0);
}

BOOST_AUTO_TEST_CASE(PositiveGeometryBelowRobustFifteenEntersCausalGraph) {
  FrontendState state;
  FakeSingleImageFeatureExtractor extractor(
      {SuccessfulExtraction(0.0f, 4), SuccessfulExtraction(5.0f, 4)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap1 = CreateBitmap();
  BOOST_REQUIRE(frontend.ProcessFrame(CreateImage(1), &bitmap1, nullptr, {})
                    .status == OnlineMapperFrameStatus::SUCCESS);
  Bitmap bitmap2 = CreateBitmap();
  const auto result =
      frontend.ProcessFrame(CreateImage(2), &bitmap2, nullptr, {1});
  BOOST_REQUIRE(result.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(result.pairs.size(), 1);
  BOOST_CHECK_EQUAL(result.pairs[0].raw_match_count, 4);
  BOOST_CHECK_EQUAL(result.pairs[0].verified_inlier_count, 4);
  BOOST_CHECK(result.pairs[0].geometry_configuration ==
              TwoViewGeometry::PLANAR_OR_PANORAMIC);
  BOOST_CHECK(result.pairs[0].rejection_reason ==
              OnlineMapperPairRejectionReason::NONE);
  BOOST_CHECK(result.pairs[0].graph_insertion_attempted);
  BOOST_CHECK_EQUAL(result.pairs[0].graph_insertion.num_added_matches, 4);
  BOOST_CHECK(result.pairs[0].reconstruction_pair_sync_success);
  BOOST_CHECK(state.database.ExistsMatches(2, 1));
  BOOST_CHECK(state.database.ExistsInlierMatches(2, 1));
  BOOST_CHECK_EQUAL(state.database.ReadMatches(2, 1).size(), 4);
  BOOST_CHECK_EQUAL(
      state.database.ReadTwoViewGeometry(2, 1).inlier_matches.size(), 4);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    1);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 1);
}

BOOST_AUTO_TEST_CASE(PrecachedPairsMaterializeAndSynchronizePositiveOnly) {
  FrontendState state;
  AddReadyImage(&state, 2, 5.0f, 8);
  AddReadyImage(&state, 3, 10.0f, 8);
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(0.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap = CreateBitmap();
  BOOST_REQUIRE(frontend.IngestFrame(CreateImage(1), &bitmap, nullptr).status ==
                OnlineMapperFrameStatus::SUCCESS);

  const FeatureMatches reverse_orientation_matches = {
      FeatureMatch(0, 4), FeatureMatch(1, 5), FeatureMatch(2, 6),
      FeatureMatch(3, 7)};
  TwoViewGeometry positive_geometry;
  positive_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  positive_geometry.H = Eigen::Matrix3d::Identity();
  positive_geometry.qvec = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  positive_geometry.inlier_matches = reverse_orientation_matches;
  state.database.WriteMatches(2, 1, reverse_orientation_matches);
  state.database.WriteTwoViewGeometry(2, 1, positive_geometry);

  state.database.WriteMatches(1, 3, CreateIdentityMatches(4));
  state.database.WriteTwoViewGeometry(1, 3, TwoViewGeometry());

  const auto result = frontend.MatchExplicitReferences(1, {2, 3});
  BOOST_REQUIRE(result.status == OnlineMapperFrameStatus::SUCCESS);
  BOOST_REQUIRE_EQUAL(result.pairs.size(), 2);
  BOOST_CHECK(result.pairs[0].submission_status ==
              OnlineMapperPairSubmissionStatus::CACHED);
  BOOST_CHECK(result.pairs[0].matcher_status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_EXISTING_PAIR);
  BOOST_CHECK_EQUAL(result.pairs[0].raw_match_count, 4);
  BOOST_CHECK_EQUAL(result.pairs[0].verified_inlier_count, 4);
  BOOST_CHECK(result.pairs[0].geometry_configuration ==
              TwoViewGeometry::PLANAR_OR_PANORAMIC);
  BOOST_CHECK(result.pairs[0].graph_insertion_attempted);
  BOOST_CHECK_EQUAL(result.pairs[0].graph_insertion.num_added_matches, 4);
  BOOST_CHECK(result.pairs[0].reconstruction_pair_sync_success);

  BOOST_CHECK(result.pairs[1].submission_status ==
              OnlineMapperPairSubmissionStatus::CACHED);
  BOOST_CHECK(result.pairs[1].matcher_status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_EXISTING_PAIR);
  BOOST_CHECK_EQUAL(result.pairs[1].raw_match_count, 4);
  BOOST_CHECK_EQUAL(result.pairs[1].verified_inlier_count, 0);
  BOOST_CHECK(result.pairs[1].geometry_configuration ==
              TwoViewGeometry::UNDEFINED);
  BOOST_CHECK(result.pairs[1].rejection_reason ==
              OnlineMapperPairRejectionReason::NO_VERIFIED_GEOMETRY);
  BOOST_CHECK(!result.pairs[1].graph_insertion_attempted);
  BOOST_CHECK(!result.pairs[1].reconstruction_pair_sync_attempted);

  const FeatureMatches synchronized_matches =
      state.database_cache.CorrespondenceGraph()
          .FindCorrespondencesBetweenImages(1, 2);
  BOOST_REQUIRE_EQUAL(synchronized_matches.size(), 4);
  for (size_t i = 0; i < synchronized_matches.size(); ++i) {
    BOOST_CHECK_EQUAL(synchronized_matches[i].point2D_idx1, i + 4);
    BOOST_CHECK_EQUAL(synchronized_matches[i].point2D_idx2, i);
  }
  BOOST_CHECK_EQUAL(state.reconstruction.ImagePair(1, 2).num_total_corrs, 4);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    1);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 1);
}

BOOST_AUTO_TEST_CASE(InconsistentPrecachedGeometryLatchesBeforeGraphMutation) {
  FrontendState state;
  AddReadyImage(&state, 2, 5.0f, 8);
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(0.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap = CreateBitmap();
  BOOST_REQUIRE(frontend.IngestFrame(CreateImage(1), &bitmap, nullptr).status ==
                OnlineMapperFrameStatus::SUCCESS);
  TwoViewGeometry invalid_geometry;
  invalid_geometry.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  invalid_geometry.H = Eigen::Matrix3d::Identity();
  invalid_geometry.qvec = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  invalid_geometry.inlier_matches = {FeatureMatch(8, 0)};
  state.database.WriteMatches(1, 2, {FeatureMatch(0, 0)});
  state.database.WriteTwoViewGeometry(1, 2, invalid_geometry);

  const auto fatal = frontend.MatchExplicitReferences(1, {2});
  BOOST_REQUIRE(fatal.status == OnlineMapperFrameStatus::FATAL_INCONSISTENCY);
  BOOST_CHECK(fatal.reason ==
              OnlineMapperFrameReason::CACHED_PAIR_DATABASE_INCONSISTENCY);
  BOOST_REQUIRE_EQUAL(fatal.pairs.size(), 1);
  BOOST_CHECK(fatal.pairs[0].submission_status ==
              OnlineMapperPairSubmissionStatus::CACHED);
  BOOST_CHECK(fatal.pairs[0].matcher_status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_EXISTING_PAIR);
  BOOST_CHECK(fatal.pairs[0].rejection_reason ==
              OnlineMapperPairRejectionReason::
                  CACHED_PAIR_DATABASE_INCONSISTENCY);
  BOOST_CHECK(!fatal.pairs[0].graph_insertion_attempted);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    0);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(FatalPartialPairSyncRetainsMaterializedLaterAudits) {
  FrontendState state;
  AddReadyImage(&state, 1, 0.0f, 8);
  AddReadyImage(&state, 2, 5.0f, 8);
  FakeSingleImageFeatureExtractor extractor({SuccessfulExtraction(10.0f)});
  BOOST_REQUIRE(extractor.Setup().success);
  SiftFeatureMatcher matcher(CreateMatchingOptions(), &state.database,
                             &state.matcher_cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(32));
  OnlineMapperFrontend frontend({}, &state.database, &state.matcher_cache,
                                &state.database_cache, &state.reconstruction,
                                &extractor, &matcher);

  Bitmap bitmap3 = CreateBitmap();
  BOOST_REQUIRE(frontend.IngestFrame(CreateImage(3), &bitmap3, nullptr).status ==
                OnlineMapperFrameStatus::SUCCESS);

  BOOST_REQUIRE(state.database_cache
                    .AddVerifiedCorrespondences(3, 1,
                                                CreateIdentityMatches(8))
                    .IsSuccess());
  BOOST_REQUIRE(
      state.reconstruction.AddImagePairFromCorrespondenceGraph(3, 1));

  DatabaseCache replacement_database_cache;
  replacement_database_cache.AddCamera(CreateCamera());
  BOOST_REQUIRE(replacement_database_cache
                    .AddImageWithKeypoints(CreateImage(1), CreateKeypoints(0.0f))
                    .IsSuccess());
  BOOST_REQUIRE(replacement_database_cache
                    .AddImageWithKeypoints(CreateImage(2), CreateKeypoints(5.0f))
                    .IsSuccess());
  BOOST_REQUIRE(replacement_database_cache
                    .AddImageWithKeypoints(CreateImage(3),
                                           CreateKeypoints(10.0f))
                    .IsSuccess());
  state.database_cache = std::move(replacement_database_cache);
  BOOST_REQUIRE_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                      0);
  BOOST_REQUIRE_EQUAL(state.reconstruction.NumImagePairs(), 1);

  const auto fatal = frontend.MatchExplicitReferences(3, {1, 2});
  BOOST_REQUIRE(fatal.status == OnlineMapperFrameStatus::FATAL_INCONSISTENCY);
  BOOST_CHECK(fatal.reason ==
              OnlineMapperFrameReason::RECONSTRUCTION_PAIR_SYNC_FAILED);
  BOOST_REQUIRE_EQUAL(fatal.pairs.size(), 2);
  BOOST_CHECK(fatal.pairs[0].matcher_status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(fatal.pairs[0].verified_inlier_count, 8);
  BOOST_CHECK(fatal.pairs[0].graph_insertion_attempted);
  BOOST_CHECK(fatal.pairs[0].graph_insertion.IsSuccess());
  BOOST_CHECK_EQUAL(fatal.pairs[0].graph_insertion.num_added_matches, 8);
  BOOST_CHECK(fatal.pairs[0].rejection_reason ==
              OnlineMapperPairRejectionReason::
                  RECONSTRUCTION_PAIR_SYNC_FAILED);
  BOOST_CHECK(fatal.pairs[0].reconstruction_pair_sync_attempted);
  BOOST_CHECK(!fatal.pairs[0].reconstruction_pair_sync_success);
  BOOST_CHECK(fatal.pairs[1].submission_status ==
              OnlineMapperPairSubmissionStatus::PROCESSED);
  BOOST_CHECK(fatal.pairs[1].matcher_status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(fatal.pairs[1].raw_match_count, 8);
  BOOST_CHECK_EQUAL(fatal.pairs[1].verified_inlier_count, 8);
  BOOST_CHECK(fatal.pairs[1].geometry_configuration ==
              TwoViewGeometry::PLANAR_OR_PANORAMIC);
  BOOST_CHECK(fatal.pairs[1].rejection_reason ==
              OnlineMapperPairRejectionReason::
                  NOT_PROCESSED_AFTER_FATAL_ERROR);
  BOOST_CHECK(!fatal.pairs[1].graph_insertion_attempted);
  BOOST_CHECK(!fatal.pairs[1].reconstruction_pair_sync_attempted);
  BOOST_CHECK(state.database.ExistsMatches(3, 2));
  BOOST_CHECK(state.database.ExistsInlierMatches(3, 2));
  BOOST_CHECK_EQUAL(state.database.ReadMatches(3, 2).size(), 8);
  BOOST_CHECK_EQUAL(
      state.database.ReadTwoViewGeometry(3, 2).inlier_matches.size(), 8);
  BOOST_CHECK_EQUAL(state.database.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.matcher_cache.GetImageIds().size(), 3);
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.database_cache.CorrespondenceGraph().NumImagePairs(),
                    1);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImagePairs(), 1);

  const size_t extract_count_before_retry = extractor.ExtractCount();
  Bitmap bitmap4 = CreateBitmap();
  const auto retry = frontend.IngestFrame(CreateImage(4), &bitmap4, nullptr);
  BOOST_CHECK(retry.status == OnlineMapperFrameStatus::FATAL_INCONSISTENCY);
  BOOST_CHECK(retry.reason == fatal.reason);
  BOOST_CHECK(retry.detail.find("latched") != std::string::npos);
  BOOST_CHECK(retry.pairs.empty());
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), extract_count_before_retry);
  BOOST_CHECK(!state.database.ExistsImage(4));
  BOOST_CHECK_EQUAL(state.database.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.matcher_cache.GetImageIds().size(), 3);
  BOOST_CHECK_EQUAL(state.database_cache.NumImages(), 3);
  BOOST_CHECK_EQUAL(state.reconstruction.NumImages(), 3);
}
