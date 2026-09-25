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

#define TEST_NAME "base/reconstruction"
#include "util/testing.h"

#include "base/camera_models.h"
#include "base/correspondence_graph.h"
#include "base/database_cache.h"
#include "base/pose.h"
#include "base/reconstruction.h"
#include "base/similarity_transform.h"

#include <thread>
#include <utility>

using namespace colmap;

void GenerateReconstruction(const image_t num_images,
                            Reconstruction* reconstruction,
                            CorrespondenceGraph* correspondence_graph) {
  const size_t kNumPoints2D = 10;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction->AddCamera(camera);

  for (image_t image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Zero()));
    reconstruction->AddImage(image);
    reconstruction->RegisterImage(image_id);
    correspondence_graph->AddImage(image_id, kNumPoints2D);
  }

  reconstruction->SetUp(correspondence_graph);
}

Camera MakeIncrementalCamera(const camera_t camera_id,
                             const size_t width = 10,
                             const size_t height = 10) {
  Camera camera;
  camera.SetCameraId(camera_id);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 10, width,
                          height);
  return camera;
}

Image MakeIncrementalImage(const image_t image_id, const camera_t camera_id,
                           const size_t num_points2D = 3) {
  Image image;
  image.SetImageId(image_id);
  image.SetCameraId(camera_id);
  image.SetName("image" + std::to_string(image_id));
  std::vector<Eigen::Vector2d> points2D;
  points2D.reserve(num_points2D);
  for (size_t idx = 0; idx < num_points2D; ++idx) {
    points2D.emplace_back(idx + 1, idx + 1);
  }
  image.SetPoints2D(points2D);
  return image;
}

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Reconstruction reconstruction;
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddCamera) {
  Reconstruction reconstruction;
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 1, 1);
  reconstruction.AddCamera(camera);
  BOOST_CHECK(reconstruction.ExistsCamera(camera.CameraId()));
  BOOST_CHECK_EQUAL(reconstruction.Camera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  BOOST_CHECK_EQUAL(reconstruction.Cameras().count(camera.CameraId()), 1);
  BOOST_CHECK_EQUAL(reconstruction.Cameras().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddImage) {
  Reconstruction reconstruction;
  Image image;
  image.SetImageId(1);
  reconstruction.AddImage(image);
  BOOST_CHECK(reconstruction.ExistsImage(1));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).ImageId(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), false);
  BOOST_CHECK_EQUAL(reconstruction.Images().count(1), 1);
  BOOST_CHECK_EQUAL(reconstruction.Images().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
}

static_assert(
    noexcept(std::declval<Reconstruction&>().CommitPreparedTransaction(
        std::declval<ReconstructionTransaction*>())),
    "prepared Reconstruction publication must be noexcept");

BOOST_AUTO_TEST_CASE(TestTransactionPublishesCandidateAndJournalAtomically) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  reconstruction.BeginStructureJournal(701, 16);
  const ReconstructionCanonicalVersion before =
      reconstruction.CanonicalVersion();

  ReconstructionTransaction transaction;
  BOOST_REQUIRE(
      reconstruction.CreateTransactionSnapshot(&transaction).IsSuccess());
  Reconstruction* candidate = transaction.MutableCandidate();
  BOOST_REQUIRE(candidate != nullptr);
  candidate->Image(1).SetTvec(Eigen::Vector3d(1.0, 2.0, 3.0));
  Track track;
  track.AddElement(1, 0);
  const point3D_t point3D_id =
      candidate->AddPoint3D(Eigen::Vector3d(4.0, 5.0, 6.0), track);

  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  BOOST_CHECK(reconstruction.Image(1).Tvec() == Eigen::Vector3d::Zero());
  BOOST_REQUIRE(
      reconstruction.PrepareTransactionCommit(&transaction).IsSuccess());
  BOOST_REQUIRE(
      reconstruction.ValidatePreparedTransaction(transaction).IsSuccess());
  reconstruction.CommitPreparedTransaction(&transaction);

  BOOST_CHECK(!transaction.HasCandidate());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  BOOST_CHECK(reconstruction.Point3D(point3D_id).XYZ() ==
              Eigen::Vector3d(4.0, 5.0, 6.0));
  BOOST_CHECK(reconstruction.Image(1).Tvec() ==
              Eigen::Vector3d(1.0, 2.0, 3.0));
  BOOST_CHECK_EQUAL(reconstruction.StructureOwnerEpoch(),
                    before.structure_owner_epoch);
  BOOST_CHECK_EQUAL(reconstruction.StructureRevision(),
                    before.structure_revision + 1);
  BOOST_CHECK_EQUAL(reconstruction.CanonicalVersion().publish_version,
                    before.publish_version + 1);
  BOOST_CHECK_EQUAL(
      reconstruction.CanonicalVersion().correspondence_graph_identity,
      before.correspondence_graph_identity);

  const ReconstructionStructureReadResult journal =
      reconstruction.ReadStructureEventsSince(before.structure_owner_epoch,
                                               before.structure_revision);
  BOOST_REQUIRE(journal.complete);
  BOOST_CHECK(!journal.gap);
  BOOST_REQUIRE_EQUAL(journal.batches.size(), 1);
  BOOST_REQUIRE_EQUAL(journal.batches.front().events.size(), 1);
  BOOST_CHECK(journal.batches.front().events.front().kind ==
              ReconstructionStructureEventKind::kPointAdded);
  BOOST_CHECK_EQUAL(journal.batches.front().events.front().point3D_id,
                    point3D_id);

  reconstruction.DeleteObservation(1, 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(TestTransactionFailureLeavesCanonicalUntouched) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(1, &reconstruction, &correspondence_graph);
  reconstruction.BeginStructureJournal(702, 8);
  const ReconstructionCanonicalVersion before =
      reconstruction.CanonicalVersion();

  ReconstructionTransaction wrong_thread_transaction;
  ReconstructionTransactionResult wrong_thread_result;
  std::thread worker([&]() {
    wrong_thread_result = reconstruction.CreateTransactionSnapshot(
        &wrong_thread_transaction);
  });
  worker.join();
  BOOST_CHECK(wrong_thread_result.status ==
              ReconstructionTransactionStatus::WRONG_THREAD);
  BOOST_CHECK(!wrong_thread_transaction.HasCandidate());

  ReconstructionTransaction transaction;
  BOOST_REQUIRE(
      reconstruction.CreateTransactionSnapshot(&transaction).IsSuccess());
  Reconstruction* candidate = transaction.MutableCandidate();
  BOOST_REQUIRE(candidate != nullptr);
  candidate->Image(1).SetTvec(Eigen::Vector3d(9.0, 8.0, 7.0));
  candidate->EndStructureJournal();

  const ReconstructionTransactionResult result =
      reconstruction.PrepareTransactionCommit(&transaction);
  BOOST_CHECK(result.status ==
              ReconstructionTransactionStatus::STRUCTURE_JOURNAL_MISMATCH);
  BOOST_CHECK(reconstruction.CanonicalVersion() == before);
  BOOST_CHECK(reconstruction.Image(1).Tvec() == Eigen::Vector3d::Zero());
  const ReconstructionStructureReadResult journal =
      reconstruction.ReadStructureEventsSince(before.structure_owner_epoch,
                                               before.structure_revision);
  BOOST_CHECK(journal.complete);
  BOOST_CHECK(journal.batches.empty());
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(TestTransactionDetectsCanonicalVersionDrift) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(1, &reconstruction, &correspondence_graph);
  reconstruction.BeginStructureJournal(703, 8);
  const ReconstructionCanonicalVersion before =
      reconstruction.CanonicalVersion();

  ReconstructionTransaction transaction;
  BOOST_REQUIRE(
      reconstruction.CreateTransactionSnapshot(&transaction).IsSuccess());
  Reconstruction* candidate = transaction.MutableCandidate();
  BOOST_REQUIRE(candidate != nullptr);
  candidate->Image(1).SetTvec(Eigen::Vector3d(3.0, 2.0, 1.0));
  reconstruction.MarkStructureUnknown("canonical drift");

  const ReconstructionTransactionResult result =
      reconstruction.PrepareTransactionCommit(&transaction);
  BOOST_CHECK(result.status ==
              ReconstructionTransactionStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(reconstruction.Image(1).Tvec() == Eigen::Vector3d::Zero());
  BOOST_CHECK_EQUAL(reconstruction.CanonicalVersion().publish_version,
                    before.publish_version);
  BOOST_CHECK_EQUAL(reconstruction.StructureRevision(),
                    before.structure_revision + 1);
  const ReconstructionStructureReadResult journal =
      reconstruction.ReadStructureEventsSince(before.structure_owner_epoch,
                                               before.structure_revision);
  BOOST_REQUIRE_EQUAL(journal.batches.size(), 1);
  BOOST_CHECK(journal.batches.front().events.front().reason ==
              "canonical drift");
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(TestIncrementalImageAndPairSynchronization) {
  DatabaseCache database_cache;
  Reconstruction reconstruction;
  reconstruction.Load(database_cache);
  reconstruction.SetUp(&database_cache.CorrespondenceGraph());

  const Camera camera = MakeIncrementalCamera(1);
  database_cache.AddCamera(camera);
  database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));

  BOOST_CHECK(reconstruction.AddImageFromDatabaseCache(database_cache, 1));
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints2D(), 3);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(1).X(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(1).Y(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumCorrespondences(), 0);
  BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 1);

  reconstruction.RegisterImage(1);
  Track track;
  track.AddElement(1, 0);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), track);

  database_cache.AddImage(MakeIncrementalImage(2, camera.CameraId()));
  BOOST_CHECK(reconstruction.AddImageFromDatabaseCache(database_cache, 2));

  const FeatureMatches raw_matches = {
      FeatureMatch(0, 0), FeatureMatch(0, 1), FeatureMatch(1, 1),
      FeatureMatch(1, 1), FeatureMatch(3, 2)};
  const auto add_pair_result =
      database_cache.AddVerifiedCorrespondences(1, 2, raw_matches);
  BOOST_REQUIRE(add_pair_result.IsSuccess());
  BOOST_CHECK_EQUAL(add_pair_result.num_added_matches, 2);
  BOOST_CHECK_EQUAL(add_pair_result.num_rejected_matches, 3);

  const FeatureMatches accepted_matches =
      database_cache.CorrespondenceGraph()
          .FindCorrespondencesBetweenImages(1, 2);
  BOOST_REQUIRE_EQUAL(accepted_matches.size(), 2);
  BOOST_CHECK_EQUAL(accepted_matches[0].point2D_idx1, 0);
  BOOST_CHECK_EQUAL(accepted_matches[0].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(accepted_matches[1].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(accepted_matches[1].point2D_idx2, 1);

  BOOST_CHECK(reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 1);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_total_corrs, 2);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_tri_corrs, 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumObservations(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 1);
  BOOST_CHECK_EQUAL(
      database_cache.CorrespondenceGraph().NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(
      database_cache.CorrespondenceGraph().NumCorrespondencesForImage(1), 2);
  BOOST_CHECK_EQUAL(
      database_cache.CorrespondenceGraph().NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(
      database_cache.CorrespondenceGraph().NumCorrespondencesForImage(2), 2);

  database_cache.AddImage(MakeIncrementalImage(3, camera.CameraId()));
  BOOST_CHECK(reconstruction.AddImageFromDatabaseCache(database_cache, 3));
  BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(2, 1));
  BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 1));
  BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 99));
  BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(2, 3));
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 1);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_total_corrs, 2);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_tri_corrs, 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumObservations(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumCorrespondences(), 2);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 1);

  reconstruction.RegisterImage(2);
  reconstruction.AddObservation(point3D_id, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_tri_corrs, 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 1);
}

BOOST_AUTO_TEST_CASE(TestIncrementalPairStartsWithTriangulatedMatch) {
  DatabaseCache database_cache;
  Reconstruction reconstruction;
  reconstruction.Load(database_cache);
  reconstruction.SetUp(&database_cache.CorrespondenceGraph());

  const Camera camera = MakeIncrementalCamera(1);
  database_cache.AddCamera(camera);
  database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));
  database_cache.AddImage(MakeIncrementalImage(2, camera.CameraId()));
  BOOST_REQUIRE(reconstruction.AddImageFromDatabaseCache(database_cache, 1));
  BOOST_REQUIRE(reconstruction.AddImageFromDatabaseCache(database_cache, 2));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);

  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), track);

  const auto add_pair_result = database_cache.AddVerifiedCorrespondences(
      1, 2, {FeatureMatch(0, 0)});
  BOOST_REQUIRE(add_pair_result.IsSuccess());
  BOOST_CHECK(reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_total_corrs, 1);
  BOOST_CHECK_EQUAL(reconstruction.ImagePair(1, 2).num_tri_corrs, 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 1);
}

BOOST_AUTO_TEST_CASE(TestIncrementalImageSynchronizationRejectsInvalidInput) {
  {
    DatabaseCache database_cache;
    const Camera camera = MakeIncrementalCamera(1);
    database_cache.AddCamera(camera);
    database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));

    Reconstruction reconstruction;
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);

    CorrespondenceGraph other_correspondence_graph;
    reconstruction.SetUp(&other_correspondence_graph);
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }

  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);

    Image image;
    image.SetImageId(1);
    image.SetPoints2D(std::vector<Eigen::Vector2d>(1));
    database_cache.AddImage(image);
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }

  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());

    Camera invalid_camera;
    invalid_camera.SetCameraId(1);
    database_cache.AddCamera(invalid_camera);
    database_cache.AddImage(MakeIncrementalImage(1, 1));
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }

  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());

    const Camera camera = MakeIncrementalCamera(1);
    database_cache.AddCamera(camera);
    database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));
    database_cache.Image(1).SetRegistered(true);
    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }

  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());

    const Camera camera = MakeIncrementalCamera(1);
    database_cache.AddCamera(camera);
    database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));
    Camera mismatched_camera = camera;
    mismatched_camera.SetWidth(camera.Width() + 1);
    reconstruction.AddCamera(mismatched_camera);

    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Camera(1).Width(), camera.Width() + 1);
  }
}

BOOST_AUTO_TEST_CASE(
    TestIncrementalImageSynchronizationRejectsStalePoint3DState) {
  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());

    const Camera camera = MakeIncrementalCamera(1);
    database_cache.AddCamera(camera);
    Image image = MakeIncrementalImage(1, camera.CameraId());
    image.SetPoint3DForPoint2D(0, 7);
    image.Point2D(0).SetPoint3DId(kInvalidPoint3DId);
    BOOST_REQUIRE_EQUAL(image.NumPoints3D(), 1);
    BOOST_REQUIRE(!image.Point2D(0).HasPoint3D());
    database_cache.AddImage(std::move(image));

    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }

  {
    DatabaseCache database_cache;
    Reconstruction reconstruction;
    reconstruction.Load(database_cache);
    reconstruction.SetUp(&database_cache.CorrespondenceGraph());

    const Camera camera = MakeIncrementalCamera(1);
    database_cache.AddCamera(camera);
    Image image = MakeIncrementalImage(1, camera.CameraId());
    image.Point2D(0).SetPoint3DId(7);
    BOOST_REQUIRE_EQUAL(image.NumPoints3D(), 0);
    BOOST_REQUIRE(image.Point2D(0).HasPoint3D());
    database_cache.AddImage(std::move(image));

    BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
    BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
    BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  }
}

BOOST_AUTO_TEST_CASE(
    TestIncrementalImageSynchronizationRejectsStaleVisiblePointState) {
  DatabaseCache database_cache;
  Reconstruction reconstruction;
  reconstruction.Load(database_cache);
  reconstruction.SetUp(&database_cache.CorrespondenceGraph());

  const Camera camera = MakeIncrementalCamera(1);
  database_cache.AddCamera(camera);
  Image image = MakeIncrementalImage(1, camera.CameraId());
  image.SetNumObservations(1);
  image.SetUp(camera);
  image.IncrementCorrespondenceHasPoint3D(0);
  BOOST_REQUIRE_EQUAL(image.NumPoints3D(), 0);
  BOOST_REQUIRE_EQUAL(image.NumVisiblePoints3D(), 1);
  database_cache.AddImage(std::move(image));

  BOOST_CHECK(!reconstruction.AddImageFromDatabaseCache(database_cache, 1));
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
}

BOOST_AUTO_TEST_CASE(TestIncrementalPairSynchronizationRejectsMissingGraph) {
  {
    const Camera camera = MakeIncrementalCamera(1);
    Reconstruction reconstruction;
    reconstruction.AddCamera(camera);
    reconstruction.AddImage(MakeIncrementalImage(1, camera.CameraId()));
    reconstruction.AddImage(MakeIncrementalImage(2, camera.CameraId()));

    BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
    BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 0);

    CorrespondenceGraph correspondence_graph;
    correspondence_graph.AddImage(1, 3);
    reconstruction.SetUp(&correspondence_graph);
    BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
    BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(2).NumObservations(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 0);
  }

  {
    const Camera camera = MakeIncrementalCamera(1);
    Reconstruction reconstruction;
    reconstruction.AddCamera(camera);
    reconstruction.AddImage(MakeIncrementalImage(1, camera.CameraId(), 1));
    reconstruction.AddImage(MakeIncrementalImage(2, camera.CameraId(), 1));

    CorrespondenceGraph correspondence_graph;
    correspondence_graph.AddImage(1, 2);
    correspondence_graph.AddImage(2, 2);
    BOOST_REQUIRE(correspondence_graph
                      .TryAddCorrespondences(1, 2, {FeatureMatch(1, 1)})
                      .IsSuccess());
    reconstruction.SetUp(&correspondence_graph);

    BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
    BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(2).NumObservations(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
    BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 0);
  }
}

BOOST_AUTO_TEST_CASE(
    TestIncrementalPairSynchronizationRejectsDanglingPoint3D) {
  DatabaseCache database_cache;
  Reconstruction reconstruction;
  reconstruction.Load(database_cache);
  reconstruction.SetUp(&database_cache.CorrespondenceGraph());

  const Camera camera = MakeIncrementalCamera(1);
  database_cache.AddCamera(camera);
  database_cache.AddImage(MakeIncrementalImage(1, camera.CameraId()));
  database_cache.AddImage(MakeIncrementalImage(2, camera.CameraId()));
  BOOST_REQUIRE(reconstruction.AddImageFromDatabaseCache(database_cache, 1));
  BOOST_REQUIRE(reconstruction.AddImageFromDatabaseCache(database_cache, 2));

  const auto add_pair_result = database_cache.AddVerifiedCorrespondences(
      1, 2, {FeatureMatch(0, 0)});
  BOOST_REQUIRE(add_pair_result.IsSuccess());
  reconstruction.Image(1).SetPoint3DForPoint2D(0, 7);
  BOOST_REQUIRE(!reconstruction.ExistsPoint3D(7));

  const point2D_t image1_num_observations =
      reconstruction.Image(1).NumObservations();
  const point2D_t image1_num_correspondences =
      reconstruction.Image(1).NumCorrespondences();
  const point2D_t image2_num_observations =
      reconstruction.Image(2).NumObservations();
  const point2D_t image2_num_correspondences =
      reconstruction.Image(2).NumCorrespondences();

  BOOST_CHECK(!reconstruction.AddImagePairFromCorrespondenceGraph(1, 2));
  BOOST_CHECK(!reconstruction.ExistsImagePair(1, 2));
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumObservations(),
                    image1_num_observations);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumCorrespondences(),
                    image1_num_correspondences);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumObservations(),
                    image2_num_observations);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumCorrespondences(),
                    image2_num_correspondences);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumVisiblePoints3D(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).NumVisiblePoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestAddPoint3D) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK(reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Points3D().count(point3D_id), 1);
  BOOST_CHECK_EQUAL(reconstruction.Points3D().size(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImagePairs(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Point3DIds().count(point3D_id), 1);
}

BOOST_AUTO_TEST_CASE(TestAddObservation) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(1, &reconstruction, &correspondence_graph);
  Track track;
  track.AddElement(1, 0);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), track);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 1);
  BOOST_CHECK(reconstruction.Image(1).Point2D(0).HasPoint3D());
  BOOST_CHECK(!reconstruction.Image(1).Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 1);
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 2);
  BOOST_CHECK(reconstruction.Image(1).Point2D(0).HasPoint3D());
  BOOST_CHECK(reconstruction.Image(1).Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 2);
}

BOOST_AUTO_TEST_CASE(TestMergePoints3D) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.Point3D(point3D_id1).Color() =
      Eigen::Matrix<uint8_t, 3, 1>(0, 0, 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id2, TrackElement(2, 1));
  reconstruction.Point3D(point3D_id2).Color() =
      Eigen::Matrix<uint8_t, 3, 1>(20, 20, 20);
  const point3D_t merged_point3D_id =
      reconstruction.MergePoints3D(point3D_id1, point3D_id2);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id1));
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id2));
  BOOST_CHECK(reconstruction.ExistsPoint3D(merged_point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(0).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).Point2D(1).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(0).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK_EQUAL(reconstruction.Image(2).Point2D(1).Point3DId(),
                    merged_point3D_id);
  BOOST_CHECK(reconstruction.Point3D(merged_point3D_id)
                  .XYZ()
                  .isApprox(Eigen::Vector3d(0.5, 0.5, 0.5)));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(merged_point3D_id).Color(),
                    Eigen::Vector3ub(10, 10, 10));
}

BOOST_AUTO_TEST_CASE(TestDeletePoint3D) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(1, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.DeletePoint3D(point3D_id);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestDeleteObservation) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 0), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(1, 2));
  reconstruction.DeleteObservation(1, 0);
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).Track().Length(), 2);
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(0).HasPoint3D());
  reconstruction.DeleteObservation(1, 1);
  BOOST_CHECK(!reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(1).HasPoint3D());
  BOOST_CHECK(!reconstruction.Image(point3D_id).Point2D(2).HasPoint3D());
}

BOOST_AUTO_TEST_CASE(TestRegisterImage) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(1, &reconstruction, &correspondence_graph);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), true);
  BOOST_CHECK(reconstruction.IsImageRegistered(1));
  reconstruction.RegisterImage(1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 1);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), true);
  BOOST_CHECK(reconstruction.IsImageRegistered(1));
  reconstruction.DeRegisterImage(1);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(reconstruction.Image(1).IsRegistered(), false);
  BOOST_CHECK(!reconstruction.IsImageRegistered(1));
}

BOOST_AUTO_TEST_CASE(TestNormalize) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, &reconstruction, &correspondence_graph);
  reconstruction.Image(1).Tvec(2) = -10.0;
  reconstruction.Image(2).Tvec(2) = 0.0;
  reconstruction.Image(3).Tvec(2) = 10.0;
  reconstruction.DeRegisterImage(1);
  reconstruction.DeRegisterImage(2);
  reconstruction.DeRegisterImage(3);
  reconstruction.Normalize();
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 10), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 10), 1e-6);
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);
  reconstruction.Normalize();
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  reconstruction.Normalize(5);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 2.5), 1e-6);
  reconstruction.Normalize(10, 0.0, 1.0);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  reconstruction.Normalize(20);
  Image image;
  image.SetImageId(4);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(4);
  image.SetImageId(5);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(5);
  image.SetImageId(6);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(6);
  image.SetImageId(7);
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(7);
  reconstruction.Image(4).Tvec(2) = -7.5;
  reconstruction.Image(5).Tvec(2) = -5.0;
  reconstruction.Image(6).Tvec(2) = 5.0;
  reconstruction.Image(7).Tvec(2) = 7.5;
  reconstruction.RegisterImage(7);
  reconstruction.Normalize(10, 0.0, 1.0);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(1).Tvec(2) + 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(2).Tvec(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(3).Tvec(2) - 5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(4).Tvec(2) + 3.75), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(5).Tvec(2) + 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(6).Tvec(2) - 2.5), 1e-6);
  BOOST_CHECK_LT(std::abs(reconstruction.Image(7).Tvec(2) - 3.75), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestComputeBoundsAndCentroid) {
  Reconstruction reconstruction;

  // Test emtpy reconstruction first
  auto centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  auto bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  BOOST_CHECK_LT(std::abs(centroid(0)), 1e-6);
  BOOST_CHECK_LT(std::abs(centroid(1)), 1e-6);
  BOOST_CHECK_LT(std::abs(centroid(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(0)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(1)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(0)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(1)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(2)), 1e-6);

  // Test reconstruction with 3D points
  reconstruction.AddPoint3D(Eigen::Vector3d(3.0, 0.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 3.0, 0.0), Track());
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 3.0), Track());
  centroid = reconstruction.ComputeCentroid(0.0, 1.0);
  bbox = reconstruction.ComputeBoundingBox(0.0, 1.0);
  BOOST_CHECK_LT(std::abs(centroid(0) - 1.0), 1e-6);
  BOOST_CHECK_LT(std::abs(centroid(1) - 1.0), 1e-6);
  BOOST_CHECK_LT(std::abs(centroid(2) - 1.0), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(0)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(1)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.first(2)), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(0) - 3.0), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(1) - 3.0), 1e-6);
  BOOST_CHECK_LT(std::abs(bbox.second(2) - 3.0), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestCrop) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, &reconstruction, &correspondence_graph);
  point3D_t point_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 1));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(1, 2));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 1.0, 0.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 3));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 0.5), Track());
  reconstruction.AddObservation(point_id, TrackElement(2, 4));
  point_id = reconstruction.AddPoint3D(Eigen::Vector3d(0.5, 0.5, 1.0), Track());
  reconstruction.AddObservation(point_id, TrackElement(3, 5));

  // Check correct reconstruction setup
  BOOST_CHECK_EQUAL(reconstruction.NumCameras(), 1);
  BOOST_CHECK_EQUAL(reconstruction.NumImages(), 3);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 3);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 5);

  std::pair<Eigen::Vector3d, Eigen::Vector3d> bbox;

  // Test emtpy reconstruction after cropping
  bbox.first = Eigen::Vector3d(-1, -1, -1);
  bbox.second = Eigen::Vector3d(-0.5, -0.5, -0.5);
  Reconstruction recon1 = reconstruction.Crop(bbox);
  BOOST_CHECK_EQUAL(recon1.NumCameras(), 1);
  BOOST_CHECK_EQUAL(recon1.NumImages(), 3);
  BOOST_CHECK_EQUAL(recon1.NumRegImages(), 0);
  BOOST_CHECK_EQUAL(recon1.NumPoints3D(), 0);

  // Test reconstruction with contents after cropping
  bbox.first = Eigen::Vector3d(0.0, 0.0, 0.0);
  bbox.second = Eigen::Vector3d(0.75, 0.75, 0.75);
  Reconstruction recon2 = reconstruction.Crop(bbox);
  BOOST_CHECK_EQUAL(recon2.NumCameras(), 1);
  BOOST_CHECK_EQUAL(recon2.NumImages(), 3);
  BOOST_CHECK_EQUAL(recon2.NumRegImages(), 2);
  BOOST_CHECK_EQUAL(recon2.NumPoints3D(), 3);
  BOOST_CHECK(recon2.IsImageRegistered(1));
  BOOST_CHECK(recon2.IsImageRegistered(2));
  BOOST_CHECK(!recon2.IsImageRegistered(3));
}

BOOST_AUTO_TEST_CASE(TestTransform) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 1, 1), Track());
  reconstruction.AddObservation(point3D_id, TrackElement(1, 1));
  reconstruction.AddObservation(point3D_id, TrackElement(2, 1));
  reconstruction.Transform(SimilarityTransform3(2, ComposeIdentityQuaternion(),
                                                Eigen::Vector3d(0, 1, 2)));
  BOOST_CHECK_EQUAL(reconstruction.Image(1).ProjectionCenter(),
                    Eigen::Vector3d(0, 1, 2));
  BOOST_CHECK_EQUAL(reconstruction.Point3D(point3D_id).XYZ(),
                    Eigen::Vector3d(2, 3, 4));
}

BOOST_AUTO_TEST_CASE(TestFindImageWithName) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  BOOST_CHECK_EQUAL(reconstruction.FindImageWithName("image1"),
                    &reconstruction.Image(1));
  BOOST_CHECK_EQUAL(reconstruction.FindImageWithName("image2"),
                    &reconstruction.Image(2));
  BOOST_CHECK(reconstruction.FindImageWithName("image3") == nullptr);
}

BOOST_AUTO_TEST_CASE(TestFilterPoints3D) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0, std::unordered_set<point3D_t>{});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id1 + 1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id2});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3D(0.0, 0.0,
                                std::unordered_set<point3D_t>{point3D_id3});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.0, 1e-3,
                                std::unordered_set<point3D_t>{point3D_id3});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3D(0.1, 0.0,
                                std::unordered_set<point3D_t>{point3D_id4});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3D(0.09, 0.0,
                                std::unordered_set<point3D_t>{point3D_id4});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterPoints3DInImages) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{2});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(0.0, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.0, 1e-3,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterPoints3DInImages(0.1, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterPoints3DInImages(0.09, 0.0,
                                        std::unordered_set<image_t>{1});
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterAllPoints) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id2, TrackElement(1, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.5, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id3, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id3, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.0, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.0, 1e-3);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
  const point3D_t point3D_id4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.6, -0.5, 1), Track());
  reconstruction.AddObservation(point3D_id4, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id4, TrackElement(2, 0));
  reconstruction.FilterAllPoints3D(0.1, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterAllPoints3D(0.09, 0.0);
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterObservationsWithNegativeDepth) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), Track());
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.001;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 1);
  reconstruction.Point3D(point3D_id1).XYZ(2) = 0.0;
  reconstruction.FilterObservationsWithNegativeDepth();
  BOOST_CHECK_EQUAL(reconstruction.NumPoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestFilterImages) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(4, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  reconstruction.AddObservation(point3D_id1, TrackElement(3, 0));
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 3);
  reconstruction.DeleteObservation(3, 0);
  reconstruction.FilterImages(0.0, 10.0, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 2);
  reconstruction.FilterImages(0.0, 0.9, 1.0);
  BOOST_CHECK_EQUAL(reconstruction.NumRegImages(), 0);
}

BOOST_AUTO_TEST_CASE(TestComputeNumObservations) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeNumObservations(), 3);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanTrackLength) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 1);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 2);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanTrackLength(), 3);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanObservationsPerRegImage) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 0.5);
  reconstruction.AddObservation(point3D_id1, TrackElement(1, 1));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 1.0);
  reconstruction.AddObservation(point3D_id1, TrackElement(2, 0));
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanObservationsPerRegImage(), 1.5);
}

BOOST_AUTO_TEST_CASE(TestComputeMeanReprojectionError) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, &reconstruction, &correspondence_graph);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  const point3D_t point3D_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d::Random(), Track());
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).SetError(0.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 0);
  reconstruction.Point3D(point3D_id1).SetError(1.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 1);
  reconstruction.Point3D(point3D_id1).SetError(2.0);
  BOOST_CHECK_EQUAL(reconstruction.ComputeMeanReprojectionError(), 2.0);
}
