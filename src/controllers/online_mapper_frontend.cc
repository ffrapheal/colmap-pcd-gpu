// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "controllers/online_mapper_frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <Eigen/Geometry>

#include "base/database.h"
#include "base/database_cache.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "feature/extraction.h"
#include "util/bitmap.h"

namespace colmap {
namespace {

constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

bool SameCamera(const Camera& camera1, const Camera& camera2) {
  return camera1.CameraId() == camera2.CameraId() &&
         camera1.ModelId() == camera2.ModelId() &&
         camera1.Width() == camera2.Width() &&
         camera1.Height() == camera2.Height() &&
         camera1.Params() == camera2.Params() &&
         camera1.HasPriorFocalLength() == camera2.HasPriorFocalLength();
}

bool SamePersistedValue(const double value1, const double value2) {
  return value1 == value2 || (std::isnan(value1) && std::isnan(value2));
}

bool SamePersistedImage(const Image& image1, const Image& image2) {
  if (image1.ImageId() != image2.ImageId() ||
      image1.CameraId() != image2.CameraId() || image1.Name() != image2.Name()) {
    return false;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (!SamePersistedValue(image1.QvecPrior(i), image2.QvecPrior(i))) {
      return false;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    if (!SamePersistedValue(image1.TvecPrior(i), image2.TvecPrior(i))) {
      return false;
    }
  }
  return true;
}

bool IsValidImageId(const image_t image_id) {
  return image_id != kInvalidImageId && image_id < Database::kMaxNumImages;
}

bool HaveValidMatchIndices(const FeatureMatches& matches,
                           const size_t num_points2D1,
                           const size_t num_points2D2,
                           const bool require_unique_indices) {
  std::unordered_set<point2D_t> indices1;
  std::unordered_set<point2D_t> indices2;
  if (require_unique_indices) {
    indices1.reserve(matches.size());
    indices2.reserve(matches.size());
  }
  for (const FeatureMatch& match : matches) {
    if (match.point2D_idx1 >= num_points2D1 ||
        match.point2D_idx2 >= num_points2D2) {
      return false;
    }
    if (require_unique_indices &&
        (!indices1.insert(match.point2D_idx1).second ||
         !indices2.insert(match.point2D_idx2).second)) {
      return false;
    }
  }
  return true;
}

bool IsConsistentCachedPair(const FeatureMatches& matches,
                            const TwoViewGeometry& two_view_geometry,
                            const size_t num_points2D1,
                            const size_t num_points2D2) {
  if (!HaveValidMatchIndices(matches, num_points2D1, num_points2D2, false) ||
      !HaveValidMatchIndices(two_view_geometry.inlier_matches, num_points2D1,
                             num_points2D2, true) ||
      two_view_geometry.config < TwoViewGeometry::UNDEFINED ||
      two_view_geometry.config > TwoViewGeometry::MULTIPLE) {
    return false;
  }
  if (two_view_geometry.inlier_matches.empty()) {
    return two_view_geometry.config == TwoViewGeometry::UNDEFINED;
  }
  return !matches.empty() &&
         two_view_geometry.config != TwoViewGeometry::UNDEFINED;
}

bool CameraCenterFromWorldToCameraPose(const KnownPoseSE3& pose,
                                       Eigen::Vector3d* camera_center) {
  if (!pose.qvec.allFinite() || !pose.tvec.allFinite()) {
    return false;
  }
  const double quaternion_norm = pose.qvec.norm();
  if (!std::isfinite(quaternion_norm) ||
      !(quaternion_norm > std::numeric_limits<double>::epsilon())) {
    return false;
  }
  const Eigen::Vector4d normalized_qvec = pose.qvec / quaternion_norm;
  const Eigen::Quaterniond quaternion(normalized_qvec(0), normalized_qvec(1),
                                      normalized_qvec(2), normalized_qvec(3));
  *camera_center = -(quaternion.toRotationMatrix().transpose() * pose.tvec);
  return camera_center->allFinite();
}

bool RotationDistanceDegrees(const KnownPoseSE3& pose1,
                             const KnownPoseSE3& pose2,
                             double* rotation_distance_degrees) {
  if (!pose1.qvec.allFinite() || !pose2.qvec.allFinite()) {
    return false;
  }
  const double norm1 = pose1.qvec.norm();
  const double norm2 = pose2.qvec.norm();
  if (!std::isfinite(norm1) || !std::isfinite(norm2) ||
      !(norm1 > std::numeric_limits<double>::epsilon()) ||
      !(norm2 > std::numeric_limits<double>::epsilon())) {
    return false;
  }
  const double absolute_dot =
      std::abs(pose1.qvec.dot(pose2.qvec) / (norm1 * norm2));
  const double clamped_dot = std::max(0.0, std::min(1.0, absolute_dot));
  *rotation_distance_degrees =
      2.0 * std::acos(clamped_dot) * kRadiansToDegrees;
  return std::isfinite(*rotation_distance_degrees);
}

ConfirmedNormalizedSE3SelectionResult SelectionFailure(
    const ConfirmedNormalizedSE3SelectionStatus status,
    const std::string& detail) {
  ConfirmedNormalizedSE3SelectionResult result;
  result.status = status;
  result.detail = detail;
  return result;
}

bool IsSynchronizedFeatureImage(const image_t image_id,
                                Database* database,
                                FeatureMatcherCache* matcher_cache,
                                DatabaseCache* database_cache,
                                Reconstruction* reconstruction,
                                const std::unordered_set<image_t>&
                                    matcher_image_ids,
                                std::string* detail) {
  if (!database->ExistsImage(image_id) ||
      !database->ExistsKeypoints(image_id) ||
      !database->ExistsDescriptors(image_id) ||
      matcher_image_ids.count(image_id) == 0 ||
      !database_cache->ExistsImage(image_id) ||
      !database_cache->CorrespondenceGraph().ExistsImage(image_id) ||
      !reconstruction->ExistsImage(image_id) ||
      !matcher_cache->ExistsKeypoints(image_id) ||
      !matcher_cache->ExistsDescriptors(image_id)) {
    *detail = "image is absent from persistent or live feature state";
    return false;
  }

  try {
    const Image persisted_image = database->ReadImage(image_id);
    const camera_t camera_id = persisted_image.CameraId();
    if (camera_id == kInvalidCameraId || !database->ExistsCamera(camera_id) ||
        !database_cache->ExistsCamera(camera_id) ||
        !reconstruction->ExistsCamera(camera_id)) {
      *detail = "image camera is absent from one or more owners";
      return false;
    }
    const Camera database_camera = database->ReadCamera(camera_id);
    if (!SamePersistedImage(persisted_image,
                            matcher_cache->GetImage(image_id)) ||
        !SamePersistedImage(persisted_image,
                            database_cache->Image(image_id)) ||
        !SamePersistedImage(persisted_image,
                            reconstruction->Image(image_id)) ||
        !SameCamera(database_camera, matcher_cache->GetCamera(camera_id)) ||
        !SameCamera(database_camera, database_cache->Camera(camera_id)) ||
        !SameCamera(database_camera, reconstruction->Camera(camera_id))) {
      *detail = "image or camera differs between owners";
      return false;
    }

    const auto cached_keypoints = matcher_cache->GetKeypoints(image_id);
    const auto cached_descriptors = matcher_cache->GetDescriptors(image_id);
    const size_t num_keypoints = database->NumKeypointsForImage(image_id);
    const size_t num_descriptors = database->NumDescriptorsForImage(image_id);
    if (num_keypoints != num_descriptors ||
        cached_keypoints->size() != num_keypoints ||
        static_cast<size_t>(cached_descriptors->rows()) != num_descriptors ||
        database_cache->Image(image_id).NumPoints2D() != num_keypoints ||
        reconstruction->Image(image_id).NumPoints2D() != num_keypoints) {
      *detail = "image feature counts differ between owners";
      return false;
    }
  } catch (const std::out_of_range&) {
    *detail = "matcher cache does not contain the image or camera";
    return false;
  }
  return true;
}

}  // namespace

constexpr double
    ConfirmedNormalizedSE3SelectorOptions::kDefaultTranslationScaleMeters;
constexpr double
    ConfirmedNormalizedSE3SelectorOptions::kDefaultRotationScaleDegrees;
constexpr size_t ConfirmedNormalizedSE3ReferenceSelector::kMaxCandidateCount;
constexpr size_t OnlineMapperFrontendOptions::kDefaultMaxReferencesPerBatch;
constexpr size_t OnlineMapperFrontendOptions::kDefaultMaxReferencesPerCurrent;

bool ConfirmedNormalizedSE3SelectorOptions::Check() const {
  return std::isfinite(translation_scale_meters) &&
         translation_scale_meters > 0.0 &&
         std::isfinite(rotation_scale_degrees) &&
         rotation_scale_degrees > 0.0;
}

ConfirmedNormalizedSE3ReferenceSelector::
    ConfirmedNormalizedSE3ReferenceSelector(
        const ConfirmedNormalizedSE3SelectorOptions& options)
    : options_(options) {}

ConfirmedNormalizedSE3SelectionResult
ConfirmedNormalizedSE3ReferenceSelector::Select(
    const KnownPoseRegistry& registry, const image_t current_image_id) const {
  if (!options_.Check()) {
    return SelectionFailure(
        ConfirmedNormalizedSE3SelectionStatus::INVALID_OPTIONS,
        "normalized SE3 scales must be finite and positive");
  }
  if (!IsValidImageId(current_image_id)) {
    return SelectionFailure(
        ConfirmedNormalizedSE3SelectionStatus::INVALID_CURRENT_IMAGE_ID,
        "current image identifier is invalid");
  }

  const KnownPoseRecordQueryResult current_query =
      registry.GetByImageId(current_image_id);
  if (!current_query.IsSuccess()) {
    if (current_query.status == KnownPoseRegistryStatus::IMAGE_NOT_FOUND) {
      return SelectionFailure(
          ConfirmedNormalizedSE3SelectionStatus::CURRENT_IMAGE_NOT_FOUND,
          current_query.detail);
    }
    return SelectionFailure(
        ConfirmedNormalizedSE3SelectionStatus::REGISTRY_QUERY_FAILED,
        current_query.detail);
  }
  const KnownPoseRecord& current_record = current_query.record;
  Eigen::Vector3d current_center;
  if (!CameraCenterFromWorldToCameraPose(current_record.fastlio_T_cw,
                                         &current_center)) {
    return SelectionFailure(
        ConfirmedNormalizedSE3SelectionStatus::INVALID_REGISTRY_RECORD,
        "current FAST-LIO pose is not a valid finite SE3");
  }

  const KnownPoseImageIdsResult image_ids = registry.GetRegisteredImageIds();
  if (!image_ids.IsSuccess()) {
    return SelectionFailure(
        ConfirmedNormalizedSE3SelectionStatus::REGISTRY_QUERY_FAILED,
        image_ids.detail);
  }

  ConfirmedNormalizedSE3SelectionResult result;
  result.status = ConfirmedNormalizedSE3SelectionStatus::SUCCESS;
  result.references.reserve(
      std::min(image_ids.image_ids.size(), kMaxCandidateCount));
  for (const image_t image_id : image_ids.image_ids) {
    if (image_id == current_image_id) {
      continue;
    }
    const KnownPoseRecordQueryResult historical_query =
        registry.GetByImageId(image_id);
    if (!historical_query.IsSuccess()) {
      return SelectionFailure(
          ConfirmedNormalizedSE3SelectionStatus::REGISTRY_QUERY_FAILED,
          historical_query.detail);
    }
    const KnownPoseRecord& historical_record = historical_query.record;
    if (historical_record.registration_sequence >=
        current_record.registration_sequence) {
      continue;
    }

    Eigen::Vector3d historical_center;
    double rotation_distance_degrees = 0.0;
    if (!CameraCenterFromWorldToCameraPose(historical_record.latest_T_cw,
                                           &historical_center) ||
        !RotationDistanceDegrees(current_record.fastlio_T_cw,
                                 historical_record.latest_T_cw,
                                 &rotation_distance_degrees)) {
      return SelectionFailure(
          ConfirmedNormalizedSE3SelectionStatus::INVALID_REGISTRY_RECORD,
          "historical latest pose is not a valid finite SE3");
    }

    const double translation_distance_meters =
        (current_center - historical_center).norm();
    const double score = std::hypot(
        translation_distance_meters / options_.translation_scale_meters,
        rotation_distance_degrees / options_.rotation_scale_degrees);
    if (std::isnan(translation_distance_meters) || std::isnan(score)) {
      return SelectionFailure(
          ConfirmedNormalizedSE3SelectionStatus::INVALID_REGISTRY_RECORD,
          "normalized SE3 distance produced NaN");
    }

    ConfirmedNormalizedSE3Reference reference;
    reference.image_id = historical_record.image_id;
    reference.score = score;
    reference.translation_distance_meters = translation_distance_meters;
    reference.rotation_distance_degrees = rotation_distance_degrees;
    reference.registration_sequence =
        historical_record.registration_sequence;
    reference.visual_state = historical_record.visual_state;
    result.references.push_back(reference);
  }

  std::sort(
      result.references.begin(), result.references.end(),
      [](const ConfirmedNormalizedSE3Reference& lhs,
         const ConfirmedNormalizedSE3Reference& rhs) {
        return std::tie(lhs.score, lhs.translation_distance_meters,
                        lhs.rotation_distance_degrees,
                        lhs.registration_sequence, lhs.image_id) <
               std::tie(rhs.score, rhs.translation_distance_meters,
                        rhs.rotation_distance_degrees,
                        rhs.registration_sequence, rhs.image_id);
      });
  if (result.references.size() > kMaxCandidateCount) {
    result.references.resize(kMaxCandidateCount);
  }
  return result;
}

bool OnlineMapperFrontendOptions::Check() const {
  return max_references_per_batch > 0 &&
         max_references_per_batch <= kDefaultMaxReferencesPerBatch &&
         max_references_per_current >= max_references_per_batch &&
         max_references_per_current <= kDefaultMaxReferencesPerCurrent;
}

OnlineMapperFrontend::OnlineMapperFrontend(
    const OnlineMapperFrontendOptions& options,
    Database* database,
    FeatureMatcherCache* matcher_cache,
    DatabaseCache* database_cache,
    Reconstruction* reconstruction,
    SingleImageFeatureExtractor* extractor,
    SiftFeatureMatcher* matcher)
    : options_(options),
      database_(database),
      matcher_cache_(matcher_cache),
      database_cache_(database_cache),
      reconstruction_(reconstruction),
      extractor_(extractor),
      matcher_(matcher),
      control_thread_id_(std::this_thread::get_id()) {}

OnlineMapperFrameResult OnlineMapperFrontend::Reject(
    const OnlineMapperFrameReason reason, const std::string& detail) const {
  OnlineMapperFrameResult result;
  result.status = OnlineMapperFrameStatus::REJECTED;
  result.reason = reason;
  result.detail = detail;
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::RejectForCurrent(
    const OnlineMapperFrameReason reason, const std::string& detail) const {
  OnlineMapperFrameResult result = Reject(reason, detail);
  result.keypoint_count = current_keypoint_count_;
  result.descriptor_count = current_descriptor_count_;
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::Fatal(
    const OnlineMapperFrameReason reason,
    const std::string& detail,
    const size_t keypoint_count,
    const size_t descriptor_count,
    std::vector<OnlineMapperPairAudit> pairs) {
  fatal_inconsistency_ = true;
  fatal_reason_ = reason;
  fatal_detail_ = detail;

  OnlineMapperFrameResult result;
  result.status = OnlineMapperFrameStatus::FATAL_INCONSISTENCY;
  result.reason = reason;
  result.detail = detail;
  result.keypoint_count = keypoint_count;
  result.descriptor_count = descriptor_count;
  result.pairs = std::move(pairs);
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::LatchedFatal() const {
  OnlineMapperFrameResult result;
  result.status = OnlineMapperFrameStatus::FATAL_INCONSISTENCY;
  result.reason = fatal_reason_;
  result.detail =
      "frontend is latched after fatal inconsistency: " + fatal_detail_;
  result.keypoint_count = current_keypoint_count_;
  result.descriptor_count = current_descriptor_count_;
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::IngestFrame(
    Image image, Bitmap* bitmap, const Bitmap* frame_mask) {
  if (fatal_inconsistency_) {
    return LatchedFatal();
  }
  if (!options_.Check()) {
    return Reject(OnlineMapperFrameReason::INVALID_OPTIONS,
                  "invalid online frontend options");
  }
  if (database_ == nullptr || matcher_cache_ == nullptr ||
      database_cache_ == nullptr || reconstruction_ == nullptr ||
      extractor_ == nullptr || matcher_ == nullptr) {
    return Reject(OnlineMapperFrameReason::INVALID_DEPENDENCY,
                  "all frontend dependencies must remain valid");
  }
  if (std::this_thread::get_id() != control_thread_id_) {
    return Reject(OnlineMapperFrameReason::WRONG_THREAD,
                  "IngestFrame must run on the controller thread");
  }
  if (matcher_->MinNumInliers() != 1) {
    return Reject(OnlineMapperFrameReason::INVALID_MATCHER_CONFIGURATION,
                  "online matching requires min_num_inliers to equal 1");
  }
  if (!extractor_->IsSetup()) {
    return Reject(OnlineMapperFrameReason::EXTRACTOR_NOT_READY,
                  "single-image extractor is not set up");
  }
  if (!matcher_->IsSetup()) {
    return Reject(OnlineMapperFrameReason::MATCHER_NOT_READY,
                  "SIFT feature matcher is not set up");
  }

  const image_t image_id = image.ImageId();
  if (!IsValidImageId(image_id) || image.Name().empty() || !image.HasCamera() ||
      image.IsRegistered() || image.NumPoints2D() != 0 ||
      image.NumPoints3D() != 0 || image.NumObservations() != 0 ||
      image.NumCorrespondences() != 0) {
    return Reject(OnlineMapperFrameReason::INVALID_IMAGE,
                  "current image must be fresh and fully identified");
  }

  const std::vector<image_t> matcher_image_ids = matcher_cache_->GetImageIds();
  const std::unordered_set<image_t> matcher_image_id_set(
      matcher_image_ids.begin(), matcher_image_ids.end());
  if (database_->ExistsImage(image_id) ||
      database_->ExistsImageWithName(image.Name()) ||
      database_->ExistsKeypoints(image_id) ||
      database_->ExistsDescriptors(image_id) ||
      matcher_image_id_set.count(image_id) != 0 ||
      database_cache_->ExistsImage(image_id) ||
      database_cache_->CorrespondenceGraph().ExistsImage(image_id) ||
      reconstruction_->ExistsImage(image_id)) {
    return Reject(OnlineMapperFrameReason::DUPLICATE_IMAGE,
                  "current image already exists in persistent or live state");
  }

  const camera_t camera_id = image.CameraId();
  if (camera_id == kInvalidCameraId || !database_->ExistsCamera(camera_id) ||
      !database_cache_->ExistsCamera(camera_id) ||
      !reconstruction_->ExistsCamera(camera_id)) {
    return Reject(OnlineMapperFrameReason::CAMERA_MISSING_OR_MISMATCHED,
                  "current camera is not loaded in every owner");
  }
  const Camera database_camera = database_->ReadCamera(camera_id);
  if (database_camera.CameraId() != camera_id ||
      !database_camera.VerifyParams() || database_camera.Width() == 0 ||
      database_camera.Height() == 0) {
    return Reject(OnlineMapperFrameReason::CAMERA_MISSING_OR_MISMATCHED,
                  "database camera metadata is invalid");
  }
  try {
    if (!SameCamera(database_camera, matcher_cache_->GetCamera(camera_id)) ||
        !SameCamera(database_camera, database_cache_->Camera(camera_id)) ||
        !SameCamera(database_camera, reconstruction_->Camera(camera_id))) {
      return Reject(OnlineMapperFrameReason::CAMERA_MISSING_OR_MISMATCHED,
                    "camera metadata differs between owners");
    }
  } catch (const std::out_of_range&) {
    return Reject(OnlineMapperFrameReason::CAMERA_MISSING_OR_MISMATCHED,
                  "matcher cache does not contain the current camera");
  }

  if (bitmap == nullptr || bitmap->Data() == nullptr || !bitmap->IsGrey() ||
      bitmap->Width() != static_cast<int>(database_camera.Width()) ||
      bitmap->Height() != static_cast<int>(database_camera.Height())) {
    return Reject(OnlineMapperFrameReason::INVALID_BITMAP,
                  "bitmap must be allocated grayscale camera-sized data");
  }
  if (frame_mask != nullptr &&
      (frame_mask->Data() == nullptr || !frame_mask->IsGrey() ||
       frame_mask->Width() != bitmap->Width() ||
       frame_mask->Height() != bitmap->Height())) {
    return Reject(OnlineMapperFrameReason::INVALID_FRAME_MASK,
                  "frame mask must be allocated grayscale bitmap-sized data");
  }

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  const SingleImageFeatureExtractionResult extraction = extractor_->Extract(
      database_camera, bitmap, frame_mask, &keypoints, &descriptors);
  if (!extraction.success) {
    OnlineMapperFrameResult result;
    result.status = OnlineMapperFrameStatus::EXTRACTION_FAILED;
    result.reason = OnlineMapperFrameReason::FEATURE_EXTRACTION_FAILED;
    result.detail = extraction.rejection_reason;
    return result;
  }
  const size_t keypoint_count = keypoints.size();
  const size_t descriptor_count = static_cast<size_t>(descriptors.rows());
  if (keypoint_count != descriptor_count || descriptors.cols() != 128) {
    OnlineMapperFrameResult result;
    result.status = OnlineMapperFrameStatus::EXTRACTION_FAILED;
    result.reason = OnlineMapperFrameReason::INVALID_FEATURE_OUTPUT;
    result.detail = "extractor returned inconsistent SIFT features";
    result.keypoint_count = keypoint_count;
    result.descriptor_count = descriptor_count;
    return result;
  }

  image_t written_image_id = kInvalidImageId;
  {
    DatabaseTransaction transaction(database_);
    written_image_id = database_->WriteImage(image, true);
    database_->WriteKeypoints(image_id, keypoints);
    database_->WriteDescriptors(image_id, descriptors);
  }
  if (written_image_id != image_id || !database_->ExistsImage(image_id) ||
      !database_->ExistsKeypoints(image_id) ||
      !database_->ExistsDescriptors(image_id) ||
      database_->NumKeypointsForImage(image_id) != keypoint_count ||
      database_->NumDescriptorsForImage(image_id) != descriptor_count) {
    return Fatal(OnlineMapperFrameReason::DATABASE_WRITE_FAILED,
                 "database feature transaction did not persist exact rows",
                 keypoint_count, descriptor_count);
  }

  if (!matcher_cache_->AddImage(image, keypoints, descriptors)) {
    return Fatal(OnlineMapperFrameReason::MATCHER_CACHE_SYNC_FAILED,
                 "database committed but matcher cache rejected the image",
                 keypoint_count, descriptor_count);
  }
  const auto add_image_result =
      database_cache_->AddImageWithKeypoints(image, keypoints);
  if (!add_image_result.IsSuccess()) {
    return Fatal(OnlineMapperFrameReason::DATABASE_CACHE_SYNC_FAILED,
                 "database committed but database cache rejected the image",
                 keypoint_count, descriptor_count);
  }
  if (!reconstruction_->AddImageFromDatabaseCache(*database_cache_, image_id)) {
    return Fatal(
        OnlineMapperFrameReason::RECONSTRUCTION_IMAGE_SYNC_FAILED,
        "database committed but reconstruction rejected the cached image",
        keypoint_count, descriptor_count);
  }

  current_image_id_ = image_id;
  current_keypoint_count_ = keypoint_count;
  current_descriptor_count_ = descriptor_count;
  current_submitted_reference_count_ = 0;

  OnlineMapperFrameResult result;
  result.status = OnlineMapperFrameStatus::SUCCESS;
  result.reason = OnlineMapperFrameReason::NONE;
  result.keypoint_count = keypoint_count;
  result.descriptor_count = descriptor_count;
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::MatchExplicitReferences(
    const image_t current_image_id,
    const std::vector<image_t>& ordered_reference_image_ids) {
  if (fatal_inconsistency_) {
    return LatchedFatal();
  }
  if (!options_.Check()) {
    return Reject(OnlineMapperFrameReason::INVALID_OPTIONS,
                  "invalid online frontend options");
  }
  if (database_ == nullptr || matcher_cache_ == nullptr ||
      database_cache_ == nullptr || reconstruction_ == nullptr ||
      extractor_ == nullptr || matcher_ == nullptr) {
    return Reject(OnlineMapperFrameReason::INVALID_DEPENDENCY,
                  "all frontend dependencies must remain valid");
  }
  if (std::this_thread::get_id() != control_thread_id_) {
    return Reject(OnlineMapperFrameReason::WRONG_THREAD,
                  "MatchExplicitReferences must run on the controller thread");
  }
  if (matcher_->MinNumInliers() != 1) {
    return RejectForCurrent(
        OnlineMapperFrameReason::INVALID_MATCHER_CONFIGURATION,
        "online matching requires min_num_inliers to equal 1");
  }
  if (!matcher_->IsSetup()) {
    return RejectForCurrent(OnlineMapperFrameReason::MATCHER_NOT_READY,
                            "SIFT feature matcher is not set up");
  }
  if (current_image_id_ == kInvalidImageId) {
    return Reject(OnlineMapperFrameReason::NO_INGESTED_CURRENT,
                  "no image has completed IngestFrame");
  }
  if (!IsValidImageId(current_image_id) ||
      current_image_id != current_image_id_) {
    return RejectForCurrent(
        OnlineMapperFrameReason::WRONG_CURRENT_IMAGE,
        "current image identifier does not match the latest ingested image");
  }
  if (ordered_reference_image_ids.size() >
      options_.max_references_per_batch) {
    return RejectForCurrent(
        OnlineMapperFrameReason::TOO_MANY_REFERENCES,
        "reference count exceeds the configured explicit matching batch");
  }

  const std::vector<image_t> matcher_image_ids = matcher_cache_->GetImageIds();
  const std::unordered_set<image_t> matcher_image_id_set(
      matcher_image_ids.begin(), matcher_image_ids.end());
  std::string readiness_detail;
  if (!IsSynchronizedFeatureImage(
          current_image_id, database_, matcher_cache_, database_cache_,
          reconstruction_, matcher_image_id_set, &readiness_detail)) {
    return Fatal(OnlineMapperFrameReason::RECONSTRUCTION_IMAGE_SYNC_FAILED,
                 "ingested current image lost synchronization: " +
                     readiness_detail,
                 current_keypoint_count_, current_descriptor_count_);
  }

  std::unordered_set<image_t> unique_references;
  unique_references.reserve(ordered_reference_image_ids.size());
  size_t uncached_reference_count = 0;
  for (const image_t reference_image_id : ordered_reference_image_ids) {
    if (!IsValidImageId(reference_image_id)) {
      return RejectForCurrent(OnlineMapperFrameReason::INVALID_REFERENCE,
                              "reference image identifier is invalid");
    }
    if (reference_image_id == current_image_id) {
      return RejectForCurrent(
          OnlineMapperFrameReason::CURRENT_IMAGE_IS_REFERENCE,
          "current image cannot explicitly reference itself");
    }
    if (!unique_references.insert(reference_image_id).second) {
      return RejectForCurrent(
          OnlineMapperFrameReason::DUPLICATE_REFERENCE,
          "reference identifiers must be unique within a batch");
    }
    if (!IsSynchronizedFeatureImage(
            reference_image_id, database_, matcher_cache_, database_cache_,
            reconstruction_, matcher_image_id_set, &readiness_detail)) {
      return RejectForCurrent(OnlineMapperFrameReason::REFERENCE_NOT_READY,
                              readiness_detail);
    }

    const image_pair_t pair_id =
        Database::ImagePairToPairId(current_image_id, reference_image_id);
    if (processed_pair_audits_.count(pair_id) == 0) {
      ++uncached_reference_count;
    }
  }
  if (uncached_reference_count >
      options_.max_references_per_current -
          current_submitted_reference_count_) {
    return RejectForCurrent(
        OnlineMapperFrameReason::TOO_MANY_REFERENCES_FOR_CURRENT,
        "new pairs would exceed the configured per-current matching limit");
  }

  std::vector<OnlineMapperPairAudit> audits(
      ordered_reference_image_ids.size());
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<size_t> uncached_audit_indices;
  std::vector<FeatureMatches> verified_matches(audits.size());
  std::vector<bool> matcher_cached_pairs(audits.size(), false);
  image_pairs.reserve(uncached_reference_count);
  uncached_audit_indices.reserve(uncached_reference_count);
  for (size_t i = 0; i < ordered_reference_image_ids.size(); ++i) {
    const image_t reference_image_id = ordered_reference_image_ids[i];
    const image_pair_t pair_id =
        Database::ImagePairToPairId(current_image_id, reference_image_id);
    const auto processed_it = processed_pair_audits_.find(pair_id);
    if (processed_it != processed_pair_audits_.end()) {
      audits[i] = processed_it->second;
      audits[i].submission_status = OnlineMapperPairSubmissionStatus::CACHED;
      audits[i].graph_insertion_attempted = false;
      audits[i].reconstruction_pair_sync_attempted = false;
      continue;
    }

    audits[i].image_id = current_image_id;
    audits[i].reference_image_id = reference_image_id;
    audits[i].rejection_reason =
        OnlineMapperPairRejectionReason::NOT_PROCESSED_AFTER_FATAL_ERROR;
    image_pairs.emplace_back(current_image_id, reference_image_id);
    uncached_audit_indices.push_back(i);
  }

  std::vector<SiftFeatureMatcher::MatchResult> match_results;
  if (!image_pairs.empty()) {
    DatabaseTransaction transaction(database_);
    match_results = matcher_->MatchWithResults(image_pairs);
  }
  if (match_results.size() != image_pairs.size()) {
    return Fatal(OnlineMapperFrameReason::MATCH_RESULT_MISMATCH,
                 "matcher returned the wrong number of ordered results",
                 current_keypoint_count_, current_descriptor_count_,
                 std::move(audits));
  }
  for (size_t i = 0; i < match_results.size(); ++i) {
    if (match_results[i].image_id1 != image_pairs[i].first ||
        match_results[i].image_id2 != image_pairs[i].second) {
      return Fatal(OnlineMapperFrameReason::MATCH_RESULT_MISMATCH,
                   "matcher changed pair orientation or order",
                   current_keypoint_count_, current_descriptor_count_,
                   std::move(audits));
    }
  }
  current_submitted_reference_count_ += uncached_reference_count;

  size_t inconsistent_cached_audit_index = audits.size();
  for (size_t i = 0; i < match_results.size(); ++i) {
    const SiftFeatureMatcher::MatchResult& match_result = match_results[i];
    const size_t audit_index = uncached_audit_indices[i];
    OnlineMapperPairAudit& audit = audits[audit_index];
    audit.matcher_status = match_result.status;
    if (match_result.status == SiftFeatureMatcher::MatchStatus::COMPUTED) {
      audit.submission_status = OnlineMapperPairSubmissionStatus::PROCESSED;
      audit.raw_match_count = match_result.matches.size();
      audit.verified_inlier_count =
          match_result.two_view_geometry.inlier_matches.size();
      audit.geometry_configuration =
          static_cast<TwoViewGeometry::ConfigurationType>(
              match_result.two_view_geometry.config);
      verified_matches[audit_index] =
          match_result.two_view_geometry.inlier_matches;
      continue;
    }

    if (match_result.status ==
        SiftFeatureMatcher::MatchStatus::SKIPPED_EXISTING_PAIR) {
      audit.submission_status = OnlineMapperPairSubmissionStatus::CACHED;
      matcher_cached_pairs[audit_index] = true;
      if (!database_->ExistsMatches(audit.image_id,
                                    audit.reference_image_id) ||
          !database_->ExistsInlierMatches(audit.image_id,
                                          audit.reference_image_id)) {
        audit.rejection_reason = OnlineMapperPairRejectionReason::
            CACHED_PAIR_DATABASE_INCONSISTENCY;
        if (inconsistent_cached_audit_index == audits.size()) {
          inconsistent_cached_audit_index = audit_index;
        }
        continue;
      }

      const FeatureMatches persisted_matches = database_->ReadMatches(
          audit.image_id, audit.reference_image_id);
      const TwoViewGeometry persisted_geometry =
          database_->ReadTwoViewGeometry(audit.image_id,
                                         audit.reference_image_id);
      audit.raw_match_count = persisted_matches.size();
      audit.verified_inlier_count =
          persisted_geometry.inlier_matches.size();
      audit.geometry_configuration =
          static_cast<TwoViewGeometry::ConfigurationType>(
              persisted_geometry.config);
      if (!IsConsistentCachedPair(
              persisted_matches, persisted_geometry,
              database_cache_->Image(audit.image_id).NumPoints2D(),
              database_cache_->Image(audit.reference_image_id)
                  .NumPoints2D())) {
        audit.rejection_reason = OnlineMapperPairRejectionReason::
            CACHED_PAIR_DATABASE_INCONSISTENCY;
        if (inconsistent_cached_audit_index == audits.size()) {
          inconsistent_cached_audit_index = audit_index;
        }
        continue;
      }
      verified_matches[audit_index] = persisted_geometry.inlier_matches;
      continue;
    }

    audit.submission_status = OnlineMapperPairSubmissionStatus::PROCESSED;
    audit.rejection_reason =
        OnlineMapperPairRejectionReason::MATCHER_DID_NOT_COMPUTE;
  }

  if (inconsistent_cached_audit_index != audits.size()) {
    return Fatal(
        OnlineMapperFrameReason::CACHED_PAIR_DATABASE_INCONSISTENCY,
        "matcher skipped an existing pair whose database rows are missing or "
        "inconsistent",
        current_keypoint_count_, current_descriptor_count_, std::move(audits));
  }

  for (size_t i = 0; i < match_results.size(); ++i) {
    const size_t audit_index = uncached_audit_indices[i];
    OnlineMapperPairAudit& audit = audits[audit_index];
    const image_pair_t pair_id = Database::ImagePairToPairId(
        audit.image_id, audit.reference_image_id);
    if (audit.rejection_reason ==
        OnlineMapperPairRejectionReason::MATCHER_DID_NOT_COMPUTE) {
      audit.rejection_reason =
          OnlineMapperPairRejectionReason::MATCHER_DID_NOT_COMPUTE;
      processed_pair_audits_.emplace(pair_id, audit);
      continue;
    }
    if (audit.verified_inlier_count == 0) {
      audit.rejection_reason =
          OnlineMapperPairRejectionReason::NO_VERIFIED_GEOMETRY;
      processed_pair_audits_.emplace(pair_id, audit);
      continue;
    }

    if (matcher_cached_pairs[audit_index] &&
        database_cache_->CorrespondenceGraph()
                .NumCorrespondencesBetweenImages(
                    audit.image_id, audit.reference_image_id) > 0) {
      audit.rejection_reason = OnlineMapperPairRejectionReason::NONE;
      processed_pair_audits_.emplace(pair_id, audit);
      continue;
    }

    audit.graph_insertion_attempted = true;
    audit.graph_insertion = database_cache_->AddVerifiedCorrespondences(
        audit.image_id, audit.reference_image_id,
        verified_matches[audit_index]);
    if (!audit.graph_insertion.IsSuccess()) {
      audit.rejection_reason =
          OnlineMapperPairRejectionReason::GRAPH_INSERTION_FAILED;
      return Fatal(OnlineMapperFrameReason::GRAPH_INSERTION_FAILED,
                   "positive verified pair was rejected by the causal graph",
                   current_keypoint_count_, current_descriptor_count_,
                   std::move(audits));
    }
    if (audit.graph_insertion.num_added_matches !=
        audit.verified_inlier_count) {
      audit.rejection_reason = OnlineMapperPairRejectionReason::
          GRAPH_ACCEPTED_INCOMPLETE_GEOMETRY;
      return Fatal(
          OnlineMapperFrameReason::GRAPH_ACCEPTED_INCOMPLETE_GEOMETRY,
          "causal graph accepted only part of the verified geometry",
          current_keypoint_count_, current_descriptor_count_,
          std::move(audits));
    }

    audit.reconstruction_pair_sync_attempted = true;
    audit.reconstruction_pair_sync_success =
        reconstruction_->AddImagePairFromCorrespondenceGraph(
            audit.image_id, audit.reference_image_id);
    if (!audit.reconstruction_pair_sync_success) {
      audit.rejection_reason =
          OnlineMapperPairRejectionReason::RECONSTRUCTION_PAIR_SYNC_FAILED;
      return Fatal(OnlineMapperFrameReason::RECONSTRUCTION_PAIR_SYNC_FAILED,
                   "reconstruction rejected an accepted causal graph pair",
                   current_keypoint_count_, current_descriptor_count_,
                   std::move(audits));
    }
    audit.rejection_reason = OnlineMapperPairRejectionReason::NONE;
    processed_pair_audits_.emplace(pair_id, audit);
  }

  OnlineMapperFrameResult result;
  result.status = OnlineMapperFrameStatus::SUCCESS;
  result.reason = OnlineMapperFrameReason::NONE;
  result.keypoint_count = current_keypoint_count_;
  result.descriptor_count = current_descriptor_count_;
  result.pairs = std::move(audits);
  return result;
}

OnlineMapperFrameResult OnlineMapperFrontend::ProcessFrame(
    Image image,
    Bitmap* bitmap,
    const Bitmap* frame_mask,
    const std::vector<image_t>& reference_image_ids) {
  const image_t image_id = image.ImageId();
  OnlineMapperFrameResult ingest_result =
      IngestFrame(std::move(image), bitmap, frame_mask);
  if (ingest_result.status != OnlineMapperFrameStatus::SUCCESS) {
    return ingest_result;
  }
  return MatchExplicitReferences(image_id, reference_image_ids);
}

}  // namespace colmap
