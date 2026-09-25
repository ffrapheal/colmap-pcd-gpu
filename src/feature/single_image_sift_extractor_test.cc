// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#define TEST_NAME "feature/single_image_sift_extractor_test"
#include "util/testing.h"

#include <memory>
#include <thread>

#include "feature/extraction.h"

using namespace colmap;

namespace {

constexpr int kOriginalImageSize = 512;
constexpr int kScaledImageSize = 256;

SiftExtractionOptions MakeCudaOptions() {
  SiftExtractionOptions options;
  options.use_gpu = true;
  options.gpu_index = "0";
  options.max_image_size = kScaledImageSize;
  options.max_num_features = 2048;
  options.estimate_affine_shape = false;
  options.domain_size_pooling = false;
  options.darkness_adaptivity = false;
  return options;
}

Camera MakeCamera(const int size) {
  Camera camera;
  camera.SetWidth(size);
  camera.SetHeight(size);
  return camera;
}

#ifdef CUDA_ENABLED
void CreateFeatureRichImage(const int size, Bitmap* bitmap) {
  BOOST_REQUIRE(bitmap->Allocate(size, size, false));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const int checker = ((x / 32 + y / 32) % 2 == 0) ? 32 : 208;
      const int value =
          (x % 71 < 4 || y % 67 < 4) ? 255 - checker : checker;
      bitmap->SetPixel(x, y, BitmapColor<uint8_t>(value));
    }
  }
}

std::shared_ptr<Bitmap> CreateLeftHalfMask(const int size) {
  auto mask = std::make_shared<Bitmap>();
  BOOST_REQUIRE(mask->Allocate(size, size, false));
  mask->Fill(BitmapColor<uint8_t>(0));
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size / 2; ++x) {
      mask->SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return mask;
}

Bitmap CreateTopHalfMask(const int size) {
  Bitmap mask;
  BOOST_REQUIRE(mask.Allocate(size, size, false));
  mask.Fill(BitmapColor<uint8_t>(0));
  for (int y = 0; y < size / 2; ++y) {
    for (int x = 0; x < size; ++x) {
      mask.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return mask;
}
#endif

void SeedOutputs(FeatureKeypoints* keypoints,
                 FeatureDescriptors* descriptors) {
  keypoints->emplace_back(1.0f, 2.0f);
  descriptors->resize(1, 128);
  descriptors->setConstant(7);
}

void CheckSetupRejected(SiftExtractionOptions options,
                        const std::string& expected_reason) {
  PersistentCudaSiftFeatureExtractor extractor(options);
  const auto result = extractor.Setup();
  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.rejection_reason.find(expected_reason) !=
              std::string::npos);
  BOOST_CHECK(!extractor.IsSetup());
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 0);
  BOOST_CHECK(!extractor.UsesCuda());
}

}  // namespace

BOOST_AUTO_TEST_CASE(RejectsCpuExtraction) {
  auto options = MakeCudaOptions();
  options.use_gpu = false;
  CheckSetupRejected(options, "use_gpu=true");
}

BOOST_AUTO_TEST_CASE(RejectsNonCudaSiftModes) {
  auto options = MakeCudaOptions();
  options.gpu_index = "-1";
  CheckSetupRejected(options, "exactly gpu_index=0");

  options = MakeCudaOptions();
  options.gpu_index = "0,1";
  CheckSetupRejected(options, "exactly gpu_index=0");

  options = MakeCudaOptions();
  options.estimate_affine_shape = true;
  CheckSetupRejected(options, "affine shape");

  options = MakeCudaOptions();
  options.domain_size_pooling = true;
  CheckSetupRejected(options, "domain size pooling");

  options = MakeCudaOptions();
  options.darkness_adaptivity = true;
  CheckSetupRejected(options, "darkness adaptivity");
}

BOOST_AUTO_TEST_CASE(RejectsSetupFromAnotherThread) {
  PersistentCudaSiftFeatureExtractor extractor(MakeCudaOptions());
  SingleImageFeatureExtractionResult result;
  std::thread setup_thread([&extractor, &result]() { result = extractor.Setup(); });
  setup_thread.join();

  BOOST_CHECK(!result.success);
  BOOST_CHECK(result.rejection_reason.find("construction thread") !=
              std::string::npos);
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 0);
}

#ifndef CUDA_ENABLED
BOOST_AUTO_TEST_CASE(RejectsCudaInCudaDisabledBuildAndClearsOutputs) {
  PersistentCudaSiftFeatureExtractor extractor(MakeCudaOptions());
  const auto setup_result = extractor.Setup();
  BOOST_REQUIRE(!setup_result.success);
  BOOST_CHECK(setup_result.rejection_reason.find("CUDA_ENABLED") !=
              std::string::npos);
  BOOST_CHECK(!extractor.IsSetup());
  BOOST_CHECK(!extractor.UsesCuda());
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 0);

  Bitmap bitmap;
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  SeedOutputs(&keypoints, &descriptors);
  const auto extract_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &bitmap, nullptr, &keypoints, &descriptors);
  BOOST_CHECK(!extract_result.success);
  BOOST_CHECK(keypoints.empty());
  BOOST_CHECK_EQUAL(descriptors.size(), 0);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 0);
}
#else
BOOST_AUTO_TEST_CASE(ExtractsTwoFramesAndRestoresCameraCoordinates) {
  PersistentCudaSiftFeatureExtractor extractor(MakeCudaOptions());
  const auto setup_result = extractor.Setup();
  BOOST_REQUIRE_MESSAGE(setup_result.success, setup_result.rejection_reason);
  const auto repeated_setup_result = extractor.Setup();
  BOOST_REQUIRE_MESSAGE(repeated_setup_result.success,
                        repeated_setup_result.rejection_reason);
  BOOST_CHECK(extractor.IsSetup());
  BOOST_CHECK(extractor.UsesCuda());
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 1);

  Bitmap original_bitmap;
  CreateFeatureRichImage(kOriginalImageSize, &original_bitmap);
  FeatureKeypoints original_keypoints;
  FeatureDescriptors original_descriptors;
  const auto first_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &original_bitmap, nullptr,
      &original_keypoints, &original_descriptors);
  BOOST_REQUIRE_MESSAGE(first_result.success, first_result.rejection_reason);
  BOOST_REQUIRE(!original_keypoints.empty());
  BOOST_CHECK_EQUAL(static_cast<size_t>(original_descriptors.rows()),
                    original_keypoints.size());
  BOOST_CHECK_EQUAL(original_bitmap.Width(), kScaledImageSize);
  BOOST_CHECK_EQUAL(original_bitmap.Height(), kScaledImageSize);
  bool has_restored_x = false;
  bool has_restored_y = false;
  for (const auto& keypoint : original_keypoints) {
    BOOST_CHECK_GE(keypoint.x, 0.0f);
    BOOST_CHECK_GE(keypoint.y, 0.0f);
    BOOST_CHECK_LT(keypoint.x, static_cast<float>(kOriginalImageSize));
    BOOST_CHECK_LT(keypoint.y, static_cast<float>(kOriginalImageSize));
    has_restored_x = has_restored_x || keypoint.x >= kScaledImageSize;
    has_restored_y = has_restored_y || keypoint.y >= kScaledImageSize;
  }
  BOOST_CHECK(has_restored_x);
  BOOST_CHECK(has_restored_y);

  Bitmap second_bitmap;
  CreateFeatureRichImage(kScaledImageSize, &second_bitmap);
  FeatureKeypoints second_keypoints;
  FeatureDescriptors second_descriptors;
  const auto second_result = extractor.Extract(
      MakeCamera(kScaledImageSize), &second_bitmap, nullptr, &second_keypoints,
      &second_descriptors);
  BOOST_REQUIRE_MESSAGE(second_result.success, second_result.rejection_reason);
  BOOST_CHECK(!second_keypoints.empty());
  BOOST_CHECK_EQUAL(static_cast<size_t>(second_descriptors.rows()),
                    second_keypoints.size());
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 1);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 2);

  Bitmap empty_bitmap;
  SeedOutputs(&second_keypoints, &second_descriptors);
  const auto empty_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &empty_bitmap, nullptr, &second_keypoints,
      &second_descriptors);
  BOOST_CHECK(!empty_result.success);
  BOOST_CHECK(second_keypoints.empty());
  BOOST_CHECK_EQUAL(second_descriptors.size(), 0);

  Bitmap wrong_size_bitmap;
  CreateFeatureRichImage(kScaledImageSize, &wrong_size_bitmap);
  SeedOutputs(&second_keypoints, &second_descriptors);
  const auto wrong_size_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &wrong_size_bitmap, nullptr,
      &second_keypoints, &second_descriptors);
  BOOST_CHECK(!wrong_size_result.success);
  BOOST_CHECK(wrong_size_result.rejection_reason.find("match the camera") !=
              std::string::npos);
  BOOST_CHECK(second_keypoints.empty());
  BOOST_CHECK_EQUAL(second_descriptors.size(), 0);

  Bitmap valid_bitmap;
  CreateFeatureRichImage(kOriginalImageSize, &valid_bitmap);
  Bitmap wrong_frame_mask;
  BOOST_REQUIRE(
      wrong_frame_mask.Allocate(kScaledImageSize, kScaledImageSize, false));
  SeedOutputs(&second_keypoints, &second_descriptors);
  const auto mask_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &valid_bitmap, &wrong_frame_mask,
      &second_keypoints, &second_descriptors);
  BOOST_CHECK(!mask_result.success);
  BOOST_CHECK(second_keypoints.empty());
  BOOST_CHECK_EQUAL(second_descriptors.size(), 0);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 2);
}

BOOST_AUTO_TEST_CASE(AppliesCameraAndFrameMasksAfterScaling) {
  const auto camera_mask = CreateLeftHalfMask(kOriginalImageSize);
  PersistentCudaSiftFeatureExtractor extractor(MakeCudaOptions(), camera_mask);
  const auto setup_result = extractor.Setup();
  BOOST_REQUIRE_MESSAGE(setup_result.success, setup_result.rejection_reason);

  Bitmap bitmap;
  CreateFeatureRichImage(kOriginalImageSize, &bitmap);
  const Bitmap frame_mask = CreateTopHalfMask(kOriginalImageSize);
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  const auto extract_result = extractor.Extract(
      MakeCamera(kOriginalImageSize), &bitmap, &frame_mask, &keypoints,
      &descriptors);
  BOOST_REQUIRE_MESSAGE(extract_result.success, extract_result.rejection_reason);
  BOOST_REQUIRE(!keypoints.empty());
  BOOST_CHECK_EQUAL(static_cast<size_t>(descriptors.rows()), keypoints.size());
  for (const auto& keypoint : keypoints) {
    BOOST_CHECK_LT(keypoint.x, kOriginalImageSize / 2.0f);
    BOOST_CHECK_LT(keypoint.y, kOriginalImageSize / 2.0f);
  }
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);

  Bitmap wrong_camera_bitmap;
  CreateFeatureRichImage(kScaledImageSize, &wrong_camera_bitmap);
  SeedOutputs(&keypoints, &descriptors);
  const auto wrong_camera_result = extractor.Extract(
      MakeCamera(kScaledImageSize), &wrong_camera_bitmap, nullptr, &keypoints,
      &descriptors);
  BOOST_CHECK(!wrong_camera_result.success);
  BOOST_CHECK(keypoints.empty());
  BOOST_CHECK_EQUAL(descriptors.size(), 0);
  BOOST_CHECK_EQUAL(extractor.ExtractCount(), 1);
}

BOOST_AUTO_TEST_CASE(RejectsEmptyCameraMaskBeforeCudaSetup) {
  const auto empty_mask = std::make_shared<Bitmap>();
  PersistentCudaSiftFeatureExtractor extractor(MakeCudaOptions(), empty_mask);
  const auto setup_result = extractor.Setup();
  BOOST_CHECK(!setup_result.success);
  BOOST_CHECK(setup_result.rejection_reason.find("camera mask is empty") !=
              std::string::npos);
  BOOST_CHECK_EQUAL(extractor.SetupCount(), 0);
}
#endif
