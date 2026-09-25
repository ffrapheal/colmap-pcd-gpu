// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#define TEST_NAME "feature/incremental_matching_test"
#include "util/testing.h"

#include "feature/matching.h"

using namespace colmap;

namespace {

Camera CreateCamera(const camera_t camera_id) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 500.0, 640, 480);
  camera.SetCameraId(camera_id);
  return camera;
}

Image CreateImage(const image_t image_id, const camera_t camera_id,
                  const std::string& name) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(camera_id);
  image.SetName(name);
  return image;
}

FeatureKeypoints CreateKeypoints(const float offset) {
  return {FeatureKeypoint(offset, offset),
          FeatureKeypoint(offset + 10.0f, offset + 5.0f)};
}

FeatureDescriptors CreateMatchingDescriptors() {
  FeatureDescriptors descriptors(2, 128);
  descriptors.setZero();
  for (int col = 0; col < 4; ++col) {
    descriptors(0, col) = 255;
    descriptors(1, col + 4) = 255;
  }
  return descriptors;
}

FeatureKeypoints CreatePlanarKeypoints(const float offset) {
  const std::vector<std::pair<float, float>> points = {
      {100.0f, 100.0f}, {200.0f, 100.0f}, {300.0f, 120.0f},
      {120.0f, 200.0f}, {220.0f, 220.0f}, {340.0f, 200.0f},
      {160.0f, 320.0f}, {300.0f, 330.0f}};
  FeatureKeypoints keypoints;
  keypoints.reserve(points.size());
  for (const auto& point : points) {
    keypoints.emplace_back(point.first + offset, point.second + offset);
  }
  return keypoints;
}

FeatureDescriptors CreateDistinctDescriptors() {
  FeatureDescriptors descriptors(8, 128);
  descriptors.setZero();
  for (int row = 0; row < descriptors.rows(); ++row) {
    for (int offset = 0; offset < 4; ++offset) {
      descriptors(row, 4 * row + offset) = 255;
    }
  }
  return descriptors;
}

FeatureDescriptors CreateSingleMatchDescriptors() {
  FeatureDescriptors descriptors(8, 128);
  descriptors.setZero();
  for (int offset = 0; offset < 4; ++offset) {
    descriptors(0, offset) = 255;
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

void WriteAndCacheCamera(Database* database, FeatureMatcherCache* cache,
                         const Camera& camera) {
  CHECK_EQ(database->WriteCamera(camera, true), camera.CameraId());
  CHECK(cache->AddCamera(camera));
}

void WriteAndCacheImage(Database* database, FeatureMatcherCache* cache,
                        const Image& image,
                        const FeatureKeypoints& keypoints,
                        const FeatureDescriptors& descriptors) {
  CHECK_EQ(database->WriteImage(image, true), image.ImageId());
  database->WriteKeypoints(image.ImageId(), keypoints);
  database->WriteDescriptors(image.ImageId(), descriptors);
  CHECK(cache->AddImage(image, keypoints, descriptors));
}

SiftMatchingOptions CreateCpuMatchingOptions() {
  SiftMatchingOptions options;
  options.use_gpu = false;
  options.num_threads = 1;
  options.max_num_matches = 32;
  return options;
}

#ifdef CUDA_ENABLED
FeatureKeypoints CreateGpuKeypoints(const float offset) {
  return {FeatureKeypoint(offset, offset),
          FeatureKeypoint(offset + 10.0f, offset + 5.0f),
          FeatureKeypoint(offset + 20.0f, offset + 10.0f)};
}

FeatureDescriptors CreatePermutedDescriptors(
    const std::vector<int>& descriptor_patterns) {
  FeatureDescriptors descriptors(descriptor_patterns.size(), 128);
  descriptors.setZero();
  for (size_t row = 0; row < descriptor_patterns.size(); ++row) {
    for (int offset = 0; offset < 4; ++offset) {
      descriptors(row, 4 * descriptor_patterns[row] + offset) = 255;
    }
  }
  return descriptors;
}
#endif

}  // namespace

BOOST_AUTO_TEST_CASE(SetupEmptyDatabase) {
  Database database(":memory:");
  FeatureMatcherCache cache(0, &database);

  cache.Setup();

  BOOST_CHECK(cache.GetImageIds().empty());
  BOOST_CHECK(!cache.ExistsKeypoints(1));
  BOOST_CHECK(!cache.ExistsDescriptors(1));
}

BOOST_AUTO_TEST_CASE(IncrementalInsertionReplacesCachedMisses) {
  Database database(":memory:");
  FeatureMatcherCache cache(2, &database);
  cache.Setup();

  const image_t image_id = 7;
  BOOST_CHECK(!cache.ExistsKeypoints(image_id));
  BOOST_CHECK(!cache.ExistsDescriptors(image_id));

  const Camera camera = CreateCamera(3);
  CHECK_EQ(database.WriteCamera(camera, true), camera.CameraId());
  BOOST_REQUIRE(cache.AddCamera(camera));

  const Image image = CreateImage(image_id, camera.CameraId(), "online.jpg");
  const FeatureKeypoints keypoints = CreateKeypoints(1.0f);
  const FeatureDescriptors descriptors = CreateMatchingDescriptors();
  CHECK_EQ(database.WriteImage(image, true), image.ImageId());
  database.WriteKeypoints(image.ImageId(), keypoints);
  database.WriteDescriptors(image.ImageId(), descriptors);
  BOOST_REQUIRE(cache.AddImage(image, keypoints, descriptors));

  BOOST_CHECK_EQUAL(cache.GetCamera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  BOOST_CHECK_EQUAL(cache.GetImage(image.ImageId()).ImageId(), image.ImageId());
  BOOST_CHECK(cache.ExistsKeypoints(image.ImageId()));
  BOOST_CHECK(cache.ExistsDescriptors(image.ImageId()));

  const auto cached_keypoints = cache.GetKeypoints(image.ImageId());
  const auto cached_descriptors = cache.GetDescriptors(image.ImageId());
  BOOST_REQUIRE_EQUAL(cached_keypoints->size(), keypoints.size());
  BOOST_REQUIRE_EQUAL(cached_descriptors->rows(), descriptors.rows());
  BOOST_CHECK_EQUAL((*cached_keypoints)[1].x, keypoints[1].x);
  BOOST_CHECK_EQUAL((*cached_descriptors)(1, 4), descriptors(1, 4));
}

BOOST_AUTO_TEST_CASE(RejectsInconsistentFeatureRows) {
  Database database(":memory:");
  FeatureMatcherCache cache(2, &database);
  cache.Setup();

  const Camera camera = CreateCamera(1);
  WriteAndCacheCamera(&database, &cache, camera);

  const Image image = CreateImage(2, camera.CameraId(), "invalid.jpg");
  const FeatureKeypoints keypoints = CreateKeypoints(0.0f);
  FeatureDescriptors descriptors(1, 128);
  descriptors.setZero();
  CHECK_EQ(database.WriteImage(image, true), image.ImageId());
  database.WriteKeypoints(image.ImageId(), keypoints);
  database.WriteDescriptors(image.ImageId(), descriptors);

  BOOST_CHECK(!cache.AddImage(image, keypoints, descriptors));
  BOOST_CHECK(cache.GetImageIds().empty());
}

BOOST_AUTO_TEST_CASE(RejectsMismatchedPersistedMetadata) {
  Database database(":memory:");
  FeatureMatcherCache cache(2, &database);
  cache.Setup();

  const Camera camera = CreateCamera(1);
  CHECK_EQ(database.WriteCamera(camera, true), camera.CameraId());
  Camera mismatched_camera = camera;
  mismatched_camera.SetWidth(camera.Width() + 1);
  BOOST_CHECK(!cache.AddCamera(mismatched_camera));
  mismatched_camera = camera;
  mismatched_camera.SetPriorFocalLength(!camera.HasPriorFocalLength());
  BOOST_CHECK(!cache.AddCamera(mismatched_camera));
  BOOST_REQUIRE(cache.AddCamera(camera));

  Image image = CreateImage(2, camera.CameraId(), "persisted.jpg");
  image.SetQvecPrior(Eigen::Vector4d(1.0, 0.1, 0.2, 0.3));
  image.SetTvecPrior(Eigen::Vector3d(4.0, 5.0, 6.0));
  const FeatureKeypoints keypoints = CreateKeypoints(0.0f);
  const FeatureDescriptors descriptors = CreateMatchingDescriptors();
  CHECK_EQ(database.WriteImage(image, true), image.ImageId());
  database.WriteKeypoints(image.ImageId(), keypoints);
  database.WriteDescriptors(image.ImageId(), descriptors);

  Image mismatched_image = image;
  mismatched_image.SetName("different.jpg");
  BOOST_CHECK(!cache.AddImage(mismatched_image, keypoints, descriptors));
  mismatched_image = image;
  mismatched_image.TvecPrior(1) += 1.0;
  BOOST_CHECK(!cache.AddImage(mismatched_image, keypoints, descriptors));
  BOOST_REQUIRE(cache.AddImage(image, keypoints, descriptors));
  BOOST_CHECK_EQUAL(cache.GetImage(image.ImageId()).Name(), image.Name());
}

BOOST_AUTO_TEST_CASE(OrderedStatusesAndIdempotentCpuSetup) {
  Database database(":memory:");
  FeatureMatcherCache cache(8, &database);
  cache.Setup();

  database.WriteMatches(4, 5, {FeatureMatch(0, 1)});
  TwoViewGeometry existing_geometry;
  existing_geometry.inlier_matches = {FeatureMatch(0, 1)};
  database.WriteTwoViewGeometry(4, 5, existing_geometry);

  SiftFeatureMatcher matcher(CreateCpuMatchingOptions(), &database, &cache);
  BOOST_CHECK(!matcher.IsSetup());
  BOOST_CHECK(!matcher.IsSetup());
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(8));
  BOOST_CHECK(matcher.IsSetup());
  BOOST_CHECK(matcher.SetupForMaxNumFeatures(8));
  BOOST_CHECK(!matcher.SetupForMaxNumFeatures(9));
  BOOST_CHECK(matcher.Setup());
  BOOST_CHECK(matcher.SetupForMaxNumFeatures(8));

  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {1, 1}, {2, 3}, {3, 2}, {4, 5}, {5, 4}};
  std::vector<SiftFeatureMatcher::MatchResult> results;
  {
    DatabaseTransaction transaction(&database);
    results = matcher.MatchWithResults(image_pairs);
  }

  BOOST_REQUIRE_EQUAL(results.size(), image_pairs.size());
  for (size_t i = 0; i < results.size(); ++i) {
    BOOST_CHECK_EQUAL(results[i].image_id1, image_pairs[i].first);
    BOOST_CHECK_EQUAL(results[i].image_id2, image_pairs[i].second);
  }
  BOOST_CHECK(results[0].status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_SELF_MATCH);
  BOOST_CHECK(results[1].status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK(results[2].status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_DUPLICATE_PAIR);
  BOOST_CHECK(results[3].status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_EXISTING_PAIR);
  BOOST_CHECK(results[4].status ==
              SiftFeatureMatcher::MatchStatus::SKIPPED_DUPLICATE_PAIR);
  BOOST_CHECK(database.ExistsMatches(2, 3));
  BOOST_CHECK(database.ExistsInlierMatches(2, 3));
}

BOOST_AUTO_TEST_CASE(ReturnsRawMatchesBeforeLegacyDatabaseFiltering) {
  Database database(":memory:");
  FeatureMatcherCache cache(8, &database);
  cache.Setup();

  const Camera camera = CreateCamera(1);
  WriteAndCacheCamera(&database, &cache, camera);
  const FeatureDescriptors descriptors = CreateMatchingDescriptors();
  WriteAndCacheImage(&database, &cache,
                     CreateImage(10, camera.CameraId(), "current.jpg"),
                     CreateKeypoints(0.0f), descriptors);
  WriteAndCacheImage(&database, &cache,
                     CreateImage(11, camera.CameraId(), "reference.jpg"),
                     CreateKeypoints(2.0f), descriptors);
  WriteAndCacheImage(&database, &cache,
                     CreateImage(12, camera.CameraId(), "legacy-current.jpg"),
                     CreateKeypoints(4.0f), descriptors);
  WriteAndCacheImage(&database, &cache,
                     CreateImage(13, camera.CameraId(), "legacy-reference.jpg"),
                     CreateKeypoints(6.0f), descriptors);

  SiftMatchingOptions options = CreateCpuMatchingOptions();
  options.min_num_inliers = 3;
  SiftFeatureMatcher matcher(options, &database, &cache);
  BOOST_CHECK_EQUAL(matcher.MinNumInliers(), 3);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(2));

  std::vector<SiftFeatureMatcher::MatchResult> results;
  {
    DatabaseTransaction transaction(&database);
    results = matcher.MatchWithResults({{11, 10}});
  }

  BOOST_REQUIRE_EQUAL(results.size(), 1);
  BOOST_CHECK(results[0].status == SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(results[0].image_id1, 11);
  BOOST_CHECK_EQUAL(results[0].image_id2, 10);
  BOOST_REQUIRE_EQUAL(results[0].matches.size(), 2);
  BOOST_CHECK_EQUAL(results[0].matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(results[0].matches[0].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(results[0].matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(results[0].matches[1].point2D_idx2, 1);
  BOOST_CHECK(results[0].two_view_geometry.inlier_matches.empty());
  BOOST_CHECK(database.ExistsMatches(11, 10));
  BOOST_CHECK(database.ReadMatches(11, 10).empty());
  BOOST_CHECK(database.ExistsInlierMatches(11, 10));

  {
    DatabaseTransaction transaction(&database);
    matcher.Match({{13, 12}});
  }
  BOOST_CHECK(database.ExistsMatches(13, 12));
  BOOST_CHECK(database.ReadMatches(13, 12).empty());
  BOOST_CHECK(database.ExistsInlierMatches(13, 12));
}

BOOST_AUTO_TEST_CASE(ReturnedGeometryMatchesGuidedFilteredDatabaseValue) {
  Database database(":memory:");
  FeatureMatcherCache cache(8, &database);
  cache.Setup();

  const Camera camera = CreateCamera(1);
  WriteAndCacheCamera(&database, &cache, camera);
  WriteAndCacheImage(&database, &cache,
                     CreateImage(14, camera.CameraId(), "guided-first.jpg"),
                     CreatePlanarKeypoints(0.0f),
                     CreateDistinctDescriptors());
  WriteAndCacheImage(&database, &cache,
                     CreateImage(15, camera.CameraId(), "guided-second.jpg"),
                     CreatePlanarKeypoints(5.0f),
                     CreateSingleMatchDescriptors());
  database.WriteMatches(15, 14, CreateIdentityMatches(8));

  SiftMatchingOptions options = CreateCpuMatchingOptions();
  options.guided_matching = true;
  options.min_num_inliers = 4;
  options.max_error = 0.5;
  options.planar_scene = true;
  SiftFeatureMatcher matcher(options, &database, &cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(8));

  std::vector<SiftFeatureMatcher::MatchResult> results;
  {
    DatabaseTransaction transaction(&database);
    results = matcher.MatchWithResults({{15, 14}});
  }

  BOOST_REQUIRE_EQUAL(results.size(), 1);
  BOOST_CHECK(results[0].status == SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(results[0].matches.size(), 8);
  const TwoViewGeometry persisted_geometry =
      database.ReadTwoViewGeometry(15, 14);
  BOOST_CHECK(results[0].two_view_geometry.config ==
              TwoViewGeometry::UNDEFINED);
  BOOST_CHECK(results[0].two_view_geometry.inlier_matches.empty());
  BOOST_CHECK(results[0].two_view_geometry.config == persisted_geometry.config);
  BOOST_CHECK_EQUAL(results[0].two_view_geometry.inlier_matches.size(),
                    persisted_geometry.inlier_matches.size());
}

#ifdef CUDA_ENABLED
BOOST_AUTO_TEST_CASE(ReusesGpuMatcherAcrossIncrementalBatches) {
  Database database(":memory:");
  FeatureMatcherCache cache(8, &database);
  cache.Setup();

  const Camera camera = CreateCamera(1);
  WriteAndCacheCamera(&database, &cache, camera);
  const FeatureDescriptors descriptors1 =
      CreatePermutedDescriptors({0, 1, 2});
  const FeatureDescriptors descriptors2 =
      CreatePermutedDescriptors({2, 0, 1});
  WriteAndCacheImage(&database, &cache,
                     CreateImage(20, camera.CameraId(), "first.jpg"),
                     CreateGpuKeypoints(0.0f), descriptors1);
  WriteAndCacheImage(&database, &cache,
                     CreateImage(21, camera.CameraId(), "second.jpg"),
                     CreateGpuKeypoints(1.0f), descriptors2);

  SiftMatchingOptions options;
  options.use_gpu = true;
  options.gpu_index = "0";
  options.num_threads = 1;
  options.max_num_matches = 16;
  options.min_num_inliers = 4;
  SiftFeatureMatcher matcher(options, &database, &cache);
  BOOST_REQUIRE(matcher.SetupForMaxNumFeatures(4));

  std::vector<SiftFeatureMatcher::MatchResult> first_results;
  {
    DatabaseTransaction transaction(&database);
    first_results = matcher.MatchWithResults({{21, 20}});
  }
  BOOST_REQUIRE_EQUAL(first_results.size(), 1);
  BOOST_CHECK(first_results[0].status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(first_results[0].image_id1, 21);
  BOOST_CHECK_EQUAL(first_results[0].image_id2, 20);
  BOOST_REQUIRE_EQUAL(first_results[0].matches.size(), 3);
  BOOST_CHECK_EQUAL(first_results[0].matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(first_results[0].matches[0].point2D_idx2, 2);
  BOOST_CHECK_EQUAL(first_results[0].matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(first_results[0].matches[1].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(first_results[0].matches[2].point2D_idx1, 2);
  BOOST_CHECK_EQUAL(first_results[0].matches[2].point2D_idx2, 1);

  const FeatureDescriptors descriptors3 =
      CreatePermutedDescriptors({1, 0, 2});
  WriteAndCacheImage(&database, &cache,
                     CreateImage(22, camera.CameraId(), "third.jpg"),
                     CreateGpuKeypoints(2.0f), descriptors3);

  std::vector<SiftFeatureMatcher::MatchResult> second_results;
  {
    DatabaseTransaction transaction(&database);
    second_results = matcher.MatchWithResults({{20, 22}});
  }
  BOOST_REQUIRE_EQUAL(second_results.size(), 1);
  BOOST_CHECK(second_results[0].status ==
              SiftFeatureMatcher::MatchStatus::COMPUTED);
  BOOST_CHECK_EQUAL(second_results[0].image_id1, 20);
  BOOST_CHECK_EQUAL(second_results[0].image_id2, 22);
  BOOST_REQUIRE_EQUAL(second_results[0].matches.size(), 3);
  BOOST_CHECK_EQUAL(second_results[0].matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(second_results[0].matches[0].point2D_idx2, 1);
  BOOST_CHECK_EQUAL(second_results[0].matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(second_results[0].matches[1].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(second_results[0].matches[2].point2D_idx1, 2);
  BOOST_CHECK_EQUAL(second_results[0].matches[2].point2D_idx2, 2);

  BOOST_CHECK_EQUAL(first_results[0].image_id1, 21);
  BOOST_CHECK_EQUAL(first_results[0].image_id2, 20);
  BOOST_CHECK(!first_results[0].matches.empty());
}
#endif
