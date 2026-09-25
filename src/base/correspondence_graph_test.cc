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

#define TEST_NAME "base/correspondence_graph"
#include "util/testing.h"

#include <stdexcept>

#include "base/correspondence_graph.h"

using namespace colmap;

int FindNumTransitiveCorrespondences(const CorrespondenceGraph& graph,
                                     const image_t image_id,
                                     const point2D_t point2D_idx,
                                     const size_t transitivity) {
  std::vector<CorrespondenceGraph::Correspondence> corrs;
  graph.FindTransitiveCorrespondences(image_id, point2D_idx, transitivity,
                                      &corrs);
  return corrs.size();
}

BOOST_AUTO_TEST_CASE(TestDefault) {
  CorrespondenceGraph correspondence_graph;
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
}

BOOST_AUTO_TEST_CASE(TestTwoView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesBetweenImages(0, 1),
                    0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(1, i));
  }
  FeatureMatches matches(4);
  matches[0].point2D_idx1 = 0;
  matches[0].point2D_idx2 = 0;
  matches[1].point2D_idx1 = 1;
  matches[1].point2D_idx2 = 2;
  matches[2].point2D_idx1 = 3;
  matches[2].point2D_idx2 = 7;
  matches[3].point2D_idx1 = 4;
  matches[3].point2D_idx2 = 8;
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 4);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 0).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 0));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 0));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 0).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 0));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 0));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 1).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 1));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 1));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 1).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 1).at(0).point2D_idx, 2);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 2).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 2));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 2));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 2).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 2).at(0).point2D_idx, 1);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 4).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(0, 3));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(0, 4));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 3).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 3).at(0).point2D_idx, 7);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 4).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 4).at(0).point2D_idx, 8);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 7).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 7));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 7));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 7).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 7).at(0).point2D_idx, 3);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 8).size(), 1);
  BOOST_CHECK(correspondence_graph.HasCorrespondences(1, 8));
  BOOST_CHECK(correspondence_graph.IsTwoViewObservation(1, 8));
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 8).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 8).at(0).point2D_idx, 4);
  std::vector<CorrespondenceGraph::Correspondence> corrs;
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(
        FindNumTransitiveCorrespondences(correspondence_graph, 0, i, 0), 0);
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(0, i).size(),
        FindNumTransitiveCorrespondences(correspondence_graph, 0, i, 2));
    BOOST_CHECK_EQUAL(
        FindNumTransitiveCorrespondences(correspondence_graph, 1, i, 0), 0);
    BOOST_CHECK_EQUAL(
        correspondence_graph.FindCorrespondences(1, i).size(),
        FindNumTransitiveCorrespondences(correspondence_graph, 1, i, 2));
  }
  const auto corrs01 =
      correspondence_graph.FindCorrespondencesBetweenImages(0, 1);
  const auto corrs10 =
      correspondence_graph.FindCorrespondencesBetweenImages(1, 0);
  BOOST_CHECK_EQUAL(corrs01.size(), matches.size());
  BOOST_CHECK_EQUAL(corrs10.size(), matches.size());
  for (size_t i = 0; i < corrs01.size(); ++i) {
    BOOST_CHECK_EQUAL(corrs01[i].point2D_idx1, corrs10[i].point2D_idx2);
    BOOST_CHECK_EQUAL(corrs01[i].point2D_idx2, corrs10[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx1, corrs01[i].point2D_idx1);
    BOOST_CHECK_EQUAL(matches[i].point2D_idx2, corrs01[i].point2D_idx2);
  }
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 4);
}

BOOST_AUTO_TEST_CASE(TestThreeView) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  correspondence_graph.AddImage(2, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 0);
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(0, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(1, i));
    BOOST_CHECK(!correspondence_graph.HasCorrespondences(2, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(0, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(1, i));
    BOOST_CHECK(!correspondence_graph.IsTwoViewObservation(2, i));
  }
  FeatureMatches matches01(1);
  matches01[0].point2D_idx1 = 0;
  matches01[0].point2D_idx2 = 0;
  correspondence_graph.AddCorrespondences(0, 1, matches01);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 0);
  FeatureMatches matches02(1);
  matches02[0].point2D_idx1 = 0;
  matches02[0].point2D_idx2 = 0;
  correspondence_graph.AddCorrespondences(0, 2, matches02);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 1);
  FeatureMatches matches12(2);
  matches12[0].point2D_idx1 = 0;
  matches12[0].point2D_idx2 = 0;
  matches12[1].point2D_idx1 = 5;
  matches12[1].point2D_idx2 = 5;
  correspondence_graph.AddCorrespondences(1, 2, matches12);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
  const image_pair_t pair_id01 = Database::ImagePairToPairId(0, 1);
  const image_pair_t pair_id02 = Database::ImagePairToPairId(0, 2);
  const image_pair_t pair_id12 = Database::ImagePairToPairId(1, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().size(), 3);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id01), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id02), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id12), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(0, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(0, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(1).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(2, 0).size(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(0).image_id, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(0).point2D_idx, 0);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(1).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 0).at(1).point2D_idx, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(1, 5).size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 5).at(0).image_id, 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 5).at(0).point2D_idx, 5);
  BOOST_CHECK_EQUAL(correspondence_graph.FindCorrespondences(2, 5).size(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 5).at(0).image_id, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(2, 5).at(0).point2D_idx, 5);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 0, 0, 2), 2);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 1, 0, 2), 2);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 2, 0, 2), 2);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 0, 0, 3), 2);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 1, 0, 3), 2);
  BOOST_CHECK_EQUAL(
      FindNumTransitiveCorrespondences(correspondence_graph, 2, 0, 3), 2);
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
  correspondence_graph.AddImage(3, 10);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), true);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 4);
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(0), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(1), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(2), true);
  BOOST_CHECK_EQUAL(correspondence_graph.ExistsImage(3), false);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 3);
}

BOOST_AUTO_TEST_CASE(TestOutOfBounds) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 4);
  FeatureMatches matches(3);
  matches[0].point2D_idx1 = 9;
  matches[0].point2D_idx2 = 3;
  matches[1].point2D_idx1 = 10;
  matches[1].point2D_idx2 = 3;
  matches[2].point2D_idx1 = 9;
  matches[2].point2D_idx2 = 4;
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 1);
}

BOOST_AUTO_TEST_CASE(TestDuplicate) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 10);
  correspondence_graph.AddImage(1, 10);
  FeatureMatches matches(5);
  matches[0].point2D_idx1 = 0;
  matches[0].point2D_idx2 = 0;
  matches[1].point2D_idx1 = 1;
  matches[1].point2D_idx2 = 1;
  matches[2].point2D_idx1 = 1;
  matches[2].point2D_idx2 = 1;
  matches[3].point2D_idx1 = 3;
  matches[3].point2D_idx2 = 3;
  matches[4].point2D_idx1 = 3;
  matches[4].point2D_idx2 = 4;
  correspondence_graph.AddCorrespondences(0, 1, matches);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 3);
  const image_pair_t pair_id = Database::ImagePairToPairId(0, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id), 3);
}

BOOST_AUTO_TEST_CASE(TestLegacyInvalidPairCompatibility) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 1);
  correspondence_graph.AddImage(1, 1);
  correspondence_graph.AddImage(2, 1);

  const image_pair_t pair_id01 = Database::ImagePairToPairId(0, 1);
  correspondence_graph.AddCorrespondences(0, 1, {FeatureMatch(1, 0)});
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id01), 0);

  const image_pair_t pair_id02 = Database::ImagePairToPairId(0, 2);
  correspondence_graph.AddCorrespondences(0, 2, FeatureMatches());
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages().at(pair_id02), 0);

  const auto duplicate_pair = correspondence_graph.TryAddCorrespondences(
      1, 0, {FeatureMatch(0, 0)});
  BOOST_CHECK(duplicate_pair.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  DUPLICATE_IMAGE_PAIR);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);
}

BOOST_AUTO_TEST_CASE(TestLegacyMissingImageCompatibility) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 1);

  BOOST_CHECK_THROW(correspondence_graph.AddCorrespondences(
                        0, 1, {FeatureMatch(0, 0)}),
                    std::out_of_range);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
}

BOOST_AUTO_TEST_CASE(TestTryAddImage) {
  CorrespondenceGraph correspondence_graph;
  const auto added = correspondence_graph.TryAddImage(1, 3);
  BOOST_CHECK(added.IsSuccess());
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 1);

  const auto duplicate = correspondence_graph.TryAddImage(1, 4);
  BOOST_CHECK(duplicate.status ==
              CorrespondenceGraph::AddImageStatus::DUPLICATE_IMAGE);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.FindCorrespondences(1, 2).size(), 0);

  const auto invalid =
      correspondence_graph.TryAddImage(kInvalidImageId, 3);
  BOOST_CHECK(invalid.status ==
              CorrespondenceGraph::AddImageStatus::INVALID_IMAGE_ID);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 1);

  const auto too_large = correspondence_graph.TryAddImage(
      static_cast<image_t>(Database::kMaxNumImages), 3);
  BOOST_CHECK(too_large.status ==
              CorrespondenceGraph::AddImageStatus::INVALID_IMAGE_ID);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 1);
}

BOOST_AUTO_TEST_CASE(TestIncrementalAfterFinalize) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 3);
  correspondence_graph.AddImage(1, 3);
  correspondence_graph.AddCorrespondences(0, 1, {FeatureMatch(0, 0)});
  correspondence_graph.Finalize();

  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 1);

  BOOST_CHECK(correspondence_graph.TryAddImage(2, 3).IsSuccess());
  const auto result = correspondence_graph.TryAddCorrespondences(
      0, 2, {FeatureMatch(0, 0), FeatureMatch(1, 1)});
  BOOST_CHECK(result.IsSuccess());
  BOOST_CHECK_EQUAL(result.num_added_matches, 2);
  BOOST_CHECK_EQUAL(result.num_rejected_matches, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 2), 2);

  BOOST_CHECK(correspondence_graph.TryAddImage(3, 3).IsSuccess());
  correspondence_graph.AddCorrespondences(0, 3, {FeatureMatch(2, 2)});
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(3), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(3), 1);

  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 2);
}

BOOST_AUTO_TEST_CASE(TestIncrementalInitializesLegacyObservationCounts) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 2);
  correspondence_graph.AddImage(1, 2);
  correspondence_graph.AddImage(2, 2);
  correspondence_graph.AddCorrespondences(0, 1, {FeatureMatch(0, 0)});

  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  const auto result = correspondence_graph.TryAddCorrespondences(
      0, 2, {FeatureMatch(0, 0), FeatureMatch(1, 1)});
  BOOST_CHECK(result.IsSuccess());
  BOOST_CHECK(result.observation_counts_recomputed);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(2), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 3);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(2), 2);
}

BOOST_AUTO_TEST_CASE(TestIncrementalDuplicateMatchesAndPair) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 3);
  correspondence_graph.AddImage(1, 3);
  const FeatureMatches matches = {
      FeatureMatch(0, 0), FeatureMatch(0, 0), FeatureMatch(0, 1),
      FeatureMatch(1, 0), FeatureMatch(1, 1), FeatureMatch(3, 2)};

  const auto result =
      correspondence_graph.TryAddCorrespondences(0, 1, matches);
  BOOST_CHECK(result.IsSuccess());
  BOOST_CHECK_EQUAL(result.num_input_matches, 6);
  BOOST_CHECK_EQUAL(result.num_added_matches, 2);
  BOOST_CHECK_EQUAL(result.num_rejected_matches, 4);
  BOOST_CHECK_EQUAL(result.num_duplicate_matches, 3);
  BOOST_CHECK_EQUAL(result.num_invalid_matches, 1);
  BOOST_CHECK_EQUAL(result.num_input_matches,
                    result.num_added_matches + result.num_rejected_matches);
  BOOST_CHECK(result.observation_counts_recomputed);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 1), 2);

  const auto duplicate_pair = correspondence_graph.TryAddCorrespondences(
      1, 0, {FeatureMatch(2, 2)});
  BOOST_CHECK(duplicate_pair.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  DUPLICATE_IMAGE_PAIR);
  BOOST_CHECK_EQUAL(duplicate_pair.num_added_matches, 0);
  BOOST_CHECK_EQUAL(duplicate_pair.num_rejected_matches, 1);
  BOOST_CHECK_EQUAL(duplicate_pair.num_duplicate_matches, 0);
  BOOST_CHECK_EQUAL(duplicate_pair.num_invalid_matches, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 2);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 1), 2);

  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 2);
}

BOOST_AUTO_TEST_CASE(TestIncrementalInvalidInputDoesNotReservePair) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 2);
  correspondence_graph.AddImage(1, 2);

  const auto invalid = correspondence_graph.TryAddCorrespondences(
      0, 1, {FeatureMatch(2, 0), FeatureMatch(0, 2)});
  BOOST_CHECK(invalid.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  NO_VALID_CORRESPONDENCES);
  BOOST_CHECK_EQUAL(invalid.num_invalid_matches, 2);
  BOOST_CHECK_EQUAL(invalid.num_rejected_matches, 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 0);

  const auto missing = correspondence_graph.TryAddCorrespondences(
      0, 2, {FeatureMatch(0, 0)});
  BOOST_CHECK(missing.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::IMAGE_NOT_FOUND);
  BOOST_CHECK_EQUAL(missing.num_rejected_matches, 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 0);

  const auto self = correspondence_graph.TryAddCorrespondences(
      0, 0, {FeatureMatch(0, 0), FeatureMatch(1, 1)});
  BOOST_CHECK(self.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::SELF_MATCH);
  BOOST_CHECK_EQUAL(self.num_rejected_matches, 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 0);

  const auto empty =
      correspondence_graph.TryAddCorrespondences(0, 1, FeatureMatches());
  BOOST_CHECK(empty.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  NO_VALID_CORRESPONDENCES);
  BOOST_CHECK_EQUAL(empty.num_input_matches, 0);
  BOOST_CHECK_EQUAL(empty.num_added_matches, 0);
  BOOST_CHECK_EQUAL(empty.num_rejected_matches, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 0);

  const auto valid = correspondence_graph.TryAddCorrespondences(
      0, 1, {FeatureMatch(0, 0)});
  BOOST_CHECK(valid.IsSuccess());
  BOOST_CHECK_EQUAL(correspondence_graph.NumImagePairs(), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 1);
}

BOOST_AUTO_TEST_CASE(TestLegacyPairCanBeAppendedInEitherOrder) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 5);
  correspondence_graph.AddImage(1, 5);

  correspondence_graph.AddCorrespondences(
      0, 1, {FeatureMatch(0, 0), FeatureMatch(1, 1)});
  correspondence_graph.AddCorrespondences(
      1, 0,
      {FeatureMatch(2, 2), FeatureMatch(1, 1), FeatureMatch(3, 2)});
  correspondence_graph.AddCorrespondences(0, 1, {FeatureMatch(3, 3)});

  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 4);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 1), 4);

  const auto online_duplicate = correspondence_graph.TryAddCorrespondences(
      1, 0, {FeatureMatch(4, 4)});
  BOOST_CHECK(online_duplicate.status ==
              CorrespondenceGraph::AddCorrespondencesStatus::
                  DUPLICATE_IMAGE_PAIR);
  BOOST_CHECK_EQUAL(online_duplicate.num_added_matches, 0);
  BOOST_CHECK_EQUAL(online_duplicate.num_rejected_matches, 1);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 1), 4);

  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 4);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 4);

  correspondence_graph.AddCorrespondences(
      1, 0, {FeatureMatch(4, 4), FeatureMatch(0, 0)});
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 5);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 5);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 5);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 5);
  BOOST_CHECK_EQUAL(
      correspondence_graph.NumCorrespondencesBetweenImages(0, 1), 5);
}

BOOST_AUTO_TEST_CASE(TestIncrementalReversePairOrientation) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 2);
  correspondence_graph.AddImage(1, 2);

  const auto result = correspondence_graph.TryAddCorrespondences(
      1, 0, {FeatureMatch(0, 1)});
  BOOST_CHECK(result.IsSuccess());
  BOOST_CHECK_EQUAL(result.num_added_matches, 1);
  BOOST_CHECK_EQUAL(result.num_rejected_matches, 0);

  const FeatureMatches forward =
      correspondence_graph.FindCorrespondencesBetweenImages(0, 1);
  BOOST_REQUIRE_EQUAL(forward.size(), 1);
  BOOST_CHECK_EQUAL(forward[0].point2D_idx1, 1);
  BOOST_CHECK_EQUAL(forward[0].point2D_idx2, 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 1);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 1);
}

BOOST_AUTO_TEST_CASE(TestOfflineFinalizeRegression) {
  CorrespondenceGraph correspondence_graph;
  correspondence_graph.AddImage(0, 2);
  correspondence_graph.AddImage(1, 2);
  correspondence_graph.AddImage(2, 2);
  correspondence_graph.AddCorrespondences(
      0, 1, {FeatureMatch(0, 0), FeatureMatch(1, 1)});

  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 0);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 0);
  correspondence_graph.Finalize();
  BOOST_CHECK_EQUAL(correspondence_graph.NumImages(), 2);
  BOOST_CHECK(!correspondence_graph.ExistsImage(2));
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumObservationsForImage(1), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(0), 2);
  BOOST_CHECK_EQUAL(correspondence_graph.NumCorrespondencesForImage(1), 2);
}
