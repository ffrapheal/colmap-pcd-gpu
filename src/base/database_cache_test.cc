// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/database_cache"
#include "util/testing.h"

#include "base/database_cache.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  DatabaseCache cache;
  BOOST_CHECK_EQUAL(cache.NumCameras(), 0);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 1);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
  BOOST_CHECK(cache.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(cache.Camera(camera.CameraId()).ModelId(),
                    camera.ModelId());
}

BOOST_AUTO_TEST_CASE(TestDegenerateCamera) {
  DatabaseCache cache;
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  cache.AddCamera(camera);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 1);
  BOOST_CHECK_EQUAL(cache.NumImages(), 0);
  BOOST_CHECK(cache.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(cache.Camera(camera.CameraId()).MeanFocalLength(), 1);
}

BOOST_AUTO_TEST_CASE(TestAddImage) {
  DatabaseCache cache;
  Image image;
  image.SetImageId(1);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  cache.AddImage(image);
  BOOST_CHECK_EQUAL(cache.NumCameras(), 0);
  BOOST_CHECK_EQUAL(cache.NumImages(), 1);
  BOOST_CHECK(cache.ExistsImage(image.ImageId()));
  BOOST_CHECK_EQUAL(cache.Image(image.ImageId()).NumPoints2D(),
                    image.NumPoints2D());
  BOOST_CHECK(cache.CorrespondenceGraph().ExistsImage(image.ImageId()));
  BOOST_CHECK_EQUAL(
      cache.CorrespondenceGraph().NumCorrespondencesForImage(image.ImageId()),
      0);
  BOOST_CHECK_EQUAL(
      cache.CorrespondenceGraph().NumObservationsForImage(image.ImageId()), 0);
}

BOOST_AUTO_TEST_CASE(TestAddImageWithKeypointsAndDuplicate) {
  DatabaseCache cache;
  Camera camera;
  camera.SetCameraId(7);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 10, 8, 5);
  cache.AddCamera(camera);

  Image image;
  image.SetImageId(1);
  image.SetCameraId(camera.CameraId());
  image.SetNumObservations(11);
  image.SetNumCorrespondences(13);
  const FeatureKeypoints keypoints = {FeatureKeypoint(1.0f, 2.0f),
                                      FeatureKeypoint(3.0f, 4.0f)};

  const auto added = cache.AddImageWithKeypoints(image, keypoints);
  BOOST_CHECK(added.IsSuccess());
  BOOST_CHECK_EQUAL(cache.NumImages(), 1);
  BOOST_CHECK_EQUAL(cache.Image(1).NumPoints2D(), 2);
  BOOST_CHECK_EQUAL(cache.Image(1).Point2D(0).X(), 1.0);
  BOOST_CHECK_EQUAL(cache.Image(1).Point2D(0).Y(), 2.0);
  BOOST_CHECK_EQUAL(cache.Image(1).Point2D(1).X(), 3.0);
  BOOST_CHECK_EQUAL(cache.Image(1).Point2D(1).Y(), 4.0);
  BOOST_CHECK_EQUAL(cache.Image(1).NumObservations(), 0);
  BOOST_CHECK_EQUAL(cache.Image(1).NumCorrespondences(), 0);
  BOOST_CHECK_EQUAL(cache.Image(1).CameraId(), camera.CameraId());
  BOOST_CHECK(cache.ExistsCamera(cache.Image(1).CameraId()));

  Image duplicate_image;
  duplicate_image.SetImageId(1);
  duplicate_image.SetCameraId(99);
  duplicate_image.SetPoints2D(std::vector<Eigen::Vector2d>(1));
  const auto duplicate =
      cache.AddImageWithKeypoints(duplicate_image, keypoints);
  BOOST_CHECK(duplicate.status ==
              CorrespondenceGraph::AddImageStatus::DUPLICATE_IMAGE);
  BOOST_CHECK_EQUAL(cache.NumImages(), 1);
  BOOST_CHECK_EQUAL(cache.Image(1).NumPoints2D(), 2);
  BOOST_CHECK_EQUAL(cache.Image(1).CameraId(), camera.CameraId());

  Image invalid_image;
  invalid_image.SetImageId(kInvalidImageId);
  const auto invalid = cache.AddImageWithKeypoints(invalid_image, keypoints);
  BOOST_CHECK(invalid.status ==
              CorrespondenceGraph::AddImageStatus::INVALID_IMAGE_ID);
  BOOST_CHECK_EQUAL(cache.NumImages(), 1);
  BOOST_CHECK_EQUAL(cache.CorrespondenceGraph().NumImages(), 1);
}

BOOST_AUTO_TEST_CASE(TestAddVerifiedCorrespondences) {
  DatabaseCache cache;
  Image image1;
  image1.SetImageId(1);
  Image image2;
  image2.SetImageId(2);
  Image image3;
  image3.SetImageId(3);
  const FeatureKeypoints keypoints = {FeatureKeypoint(0.0f, 0.0f),
                                      FeatureKeypoint(1.0f, 1.0f),
                                      FeatureKeypoint(2.0f, 2.0f)};
  BOOST_CHECK(cache.AddImageWithKeypoints(image1, keypoints).IsSuccess());
  BOOST_CHECK(cache.AddImageWithKeypoints(image2, keypoints).IsSuccess());
  BOOST_CHECK(cache.AddImageWithKeypoints(image3, keypoints).IsSuccess());

  const auto first_pair = cache.AddVerifiedCorrespondences(
      1, 2, {FeatureMatch(0, 0), FeatureMatch(1, 1)});
  BOOST_CHECK(first_pair.IsSuccess());
  BOOST_CHECK_EQUAL(first_pair.num_added_matches, 2);
  BOOST_CHECK_EQUAL(first_pair.num_rejected_matches, 0);
  BOOST_CHECK_EQUAL(cache.Image(1).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(1).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(cache.Image(2).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(2).NumCorrespondences(), 2);

  const auto appended_observation = cache.AddVerifiedCorrespondences(
      1, 3, {FeatureMatch(0, 0), FeatureMatch(2, 1)});
  BOOST_CHECK(appended_observation.IsSuccess());
  BOOST_CHECK_EQUAL(cache.Image(1).NumObservations(), 3);
  BOOST_CHECK_EQUAL(cache.Image(1).NumCorrespondences(), 4);
  BOOST_CHECK_EQUAL(cache.Image(3).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(3).NumCorrespondences(), 2);

  const auto duplicate_pair =
      cache.AddVerifiedCorrespondences(3, 1, {FeatureMatch(2, 2)});
  BOOST_CHECK(duplicate_pair.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  DUPLICATE_IMAGE_PAIR);
  BOOST_CHECK_EQUAL(duplicate_pair.num_added_matches, 0);
  BOOST_CHECK_EQUAL(duplicate_pair.num_rejected_matches, 1);
  BOOST_CHECK_EQUAL(cache.Image(1).NumObservations(), 3);
  BOOST_CHECK_EQUAL(cache.Image(1).NumCorrespondences(), 4);
  BOOST_CHECK_EQUAL(cache.Image(3).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(3).NumCorrespondences(), 2);

  const auto invalid =
      cache.AddVerifiedCorrespondences(2, 3, {FeatureMatch(3, 0)});
  BOOST_CHECK(invalid.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  NO_VALID_CORRESPONDENCES);
  BOOST_CHECK_EQUAL(invalid.num_added_matches, 0);
  BOOST_CHECK_EQUAL(invalid.num_rejected_matches, 1);
  BOOST_CHECK_EQUAL(cache.CorrespondenceGraph()
                        .NumCorrespondencesBetweenImages(2, 3),
                    0);
  BOOST_CHECK_EQUAL(cache.Image(2).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(cache.Image(3).NumCorrespondences(), 2);

  const auto retried =
      cache.AddVerifiedCorrespondences(3, 2, {FeatureMatch(2, 2)});
  BOOST_CHECK(retried.IsSuccess());
  BOOST_CHECK_EQUAL(retried.num_added_matches, 1);
  BOOST_CHECK_EQUAL(cache.Image(2).NumObservations(), 3);
  BOOST_CHECK_EQUAL(cache.Image(2).NumCorrespondences(), 3);
  BOOST_CHECK_EQUAL(cache.Image(3).NumObservations(), 3);
  BOOST_CHECK_EQUAL(cache.Image(3).NumCorrespondences(), 3);
}

BOOST_AUTO_TEST_CASE(TestOfflineLoadThenIncrementalAdd) {
  Database database(":memory:");
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database.WriteCamera(camera));

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera.CameraId());
  image1.SetImageId(database.WriteImage(image1));
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera.CameraId());
  image2.SetImageId(database.WriteImage(image2));
  const FeatureKeypoints keypoints = {
      FeatureKeypoint(0.0f, 0.0f), FeatureKeypoint(1.0f, 1.0f),
      FeatureKeypoint(2.0f, 2.0f)};
  database.WriteKeypoints(image1.ImageId(), keypoints);
  database.WriteKeypoints(image2.ImageId(), keypoints);
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = {FeatureMatch(0, 0), FeatureMatch(1, 1)};
  database.WriteTwoViewGeometry(image1.ImageId(), image2.ImageId(),
                                two_view_geometry);

  DatabaseCache cache;
  cache.Load(database, 1, false, std::unordered_set<std::string>());
  BOOST_CHECK_EQUAL(cache.NumImages(), 2);
  BOOST_CHECK_EQUAL(cache.Image(image1.ImageId()).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(image1.ImageId()).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(cache.Image(image2.ImageId()).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(image2.ImageId()).NumCorrespondences(), 2);

  Image image3;
  image3.SetImageId(3);
  image3.SetCameraId(camera.CameraId());
  BOOST_CHECK(cache.AddImageWithKeypoints(image3, keypoints).IsSuccess());
  BOOST_CHECK_EQUAL(cache.Image(image3.ImageId()).CameraId(),
                    camera.CameraId());
  const auto result = cache.AddVerifiedCorrespondences(
      image1.ImageId(), image3.ImageId(),
      {FeatureMatch(0, 0), FeatureMatch(2, 2)});
  BOOST_CHECK(result.IsSuccess());
  BOOST_CHECK_EQUAL(cache.Image(image1.ImageId()).NumObservations(), 3);
  BOOST_CHECK_EQUAL(cache.Image(image1.ImageId()).NumCorrespondences(), 4);
  BOOST_CHECK_EQUAL(cache.Image(image3.ImageId()).NumObservations(), 2);
  BOOST_CHECK_EQUAL(cache.Image(image3.ImageId()).NumCorrespondences(), 2);
}
