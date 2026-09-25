// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_FRONTEND_H_
#define COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_FRONTEND_H_

#include <cstddef>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "base/correspondence_graph.h"
#include "controllers/online_mapper_state.h"
#include "feature/matching.h"
#include "util/types.h"

namespace colmap {

class Bitmap;
class Database;
class DatabaseCache;
class Image;
class Reconstruction;
class SingleImageFeatureExtractor;

struct ConfirmedNormalizedSE3SelectorOptions {
  static constexpr double kDefaultTranslationScaleMeters = 0.05;
  static constexpr double kDefaultRotationScaleDegrees = 10.0;

  double translation_scale_meters = kDefaultTranslationScaleMeters;
  double rotation_scale_degrees = kDefaultRotationScaleDegrees;

  bool Check() const;
};

enum class ConfirmedNormalizedSE3SelectionStatus {
  SUCCESS,
  INVALID_OPTIONS,
  INVALID_CURRENT_IMAGE_ID,
  CURRENT_IMAGE_NOT_FOUND,
  REGISTRY_QUERY_FAILED,
  INVALID_REGISTRY_RECORD,
};

struct ConfirmedNormalizedSE3Reference {
  image_t image_id = kInvalidImageId;
  double score = std::numeric_limits<double>::quiet_NaN();
  double translation_distance_meters =
      std::numeric_limits<double>::quiet_NaN();
  double rotation_distance_degrees =
      std::numeric_limits<double>::quiet_NaN();
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  KnownPoseVisualState visual_state = KnownPoseVisualState::POSE_ONLY;
};

struct ConfirmedNormalizedSE3SelectionResult {
  ConfirmedNormalizedSE3SelectionStatus status =
      ConfirmedNormalizedSE3SelectionStatus::INVALID_OPTIONS;
  std::string detail;
  std::vector<ConfirmedNormalizedSE3Reference> references;

  bool IsSuccess() const {
    return status == ConfirmedNormalizedSE3SelectionStatus::SUCCESS;
  }
};

// Selects only records causally older than the explicit current image. The
// current image always uses fastlio_T_cw, while historical records use their
// latest_T_cw regardless of POSE_ONLY or VISUAL_ACTIVE state.
class ConfirmedNormalizedSE3ReferenceSelector {
 public:
  static constexpr size_t kMaxCandidateCount = 20;

  explicit ConfirmedNormalizedSE3ReferenceSelector(
      const ConfirmedNormalizedSE3SelectorOptions& options =
          ConfirmedNormalizedSE3SelectorOptions());

  ConfirmedNormalizedSE3SelectionResult Select(
      const KnownPoseRegistry& registry, image_t current_image_id) const;

 private:
  const ConfirmedNormalizedSE3SelectorOptions options_;
};

struct OnlineMapperFrontendOptions {
  static constexpr size_t kDefaultMaxReferencesPerBatch = 5;
  static constexpr size_t kDefaultMaxReferencesPerCurrent = 10;

  size_t max_references_per_batch = kDefaultMaxReferencesPerBatch;
  size_t max_references_per_current = kDefaultMaxReferencesPerCurrent;

  bool Check() const;
};

enum class OnlineMapperFrameStatus {
  SUCCESS,
  REJECTED,
  EXTRACTION_FAILED,
  FATAL_INCONSISTENCY,
};

enum class OnlineMapperFrameReason {
  NONE,
  INVALID_OPTIONS,
  INVALID_DEPENDENCY,
  WRONG_THREAD,
  EXTRACTOR_NOT_READY,
  MATCHER_NOT_READY,
  INVALID_MATCHER_CONFIGURATION,
  INVALID_IMAGE,
  DUPLICATE_IMAGE,
  CAMERA_MISSING_OR_MISMATCHED,
  INVALID_BITMAP,
  INVALID_FRAME_MASK,
  NO_INGESTED_CURRENT,
  WRONG_CURRENT_IMAGE,
  TOO_MANY_REFERENCES,
  TOO_MANY_REFERENCES_FOR_CURRENT,
  INVALID_REFERENCE,
  DUPLICATE_REFERENCE,
  CURRENT_IMAGE_IS_REFERENCE,
  REFERENCE_NOT_READY,
  FEATURE_EXTRACTION_FAILED,
  INVALID_FEATURE_OUTPUT,
  DATABASE_WRITE_FAILED,
  MATCHER_CACHE_SYNC_FAILED,
  DATABASE_CACHE_SYNC_FAILED,
  RECONSTRUCTION_IMAGE_SYNC_FAILED,
  MATCH_RESULT_MISMATCH,
  CACHED_PAIR_DATABASE_INCONSISTENCY,
  GRAPH_INSERTION_FAILED,
  GRAPH_ACCEPTED_INCOMPLETE_GEOMETRY,
  RECONSTRUCTION_PAIR_SYNC_FAILED,
};

enum class OnlineMapperPairSubmissionStatus {
  NOT_PROCESSED,
  PROCESSED,
  CACHED,
};

enum class OnlineMapperPairRejectionReason {
  NONE,
  MATCHER_DID_NOT_COMPUTE,
  CACHED_PAIR_DATABASE_INCONSISTENCY,
  NO_VERIFIED_GEOMETRY,
  GRAPH_INSERTION_FAILED,
  GRAPH_ACCEPTED_INCOMPLETE_GEOMETRY,
  RECONSTRUCTION_PAIR_SYNC_FAILED,
  NOT_PROCESSED_AFTER_FATAL_ERROR,
};

struct OnlineMapperPairAudit {
  image_t image_id = kInvalidImageId;
  image_t reference_image_id = kInvalidImageId;
  OnlineMapperPairSubmissionStatus submission_status =
      OnlineMapperPairSubmissionStatus::NOT_PROCESSED;
  // CACHED submissions retain the original matcher and geometry audit below.
  SiftFeatureMatcher::MatchStatus matcher_status =
      SiftFeatureMatcher::MatchStatus::NOT_PROCESSED;
  size_t raw_match_count = 0;
  size_t verified_inlier_count = 0;
  TwoViewGeometry::ConfigurationType geometry_configuration =
      TwoViewGeometry::UNDEFINED;
  bool graph_insertion_attempted = false;
  CorrespondenceGraph::AddCorrespondencesResult graph_insertion;
  bool reconstruction_pair_sync_attempted = false;
  bool reconstruction_pair_sync_success = false;
  OnlineMapperPairRejectionReason rejection_reason =
      OnlineMapperPairRejectionReason::NONE;
};

struct OnlineMapperFrameResult {
  OnlineMapperFrameStatus status = OnlineMapperFrameStatus::REJECTED;
  OnlineMapperFrameReason reason = OnlineMapperFrameReason::NONE;
  std::string detail;
  size_t keypoint_count = 0;
  size_t descriptor_count = 0;
  std::vector<OnlineMapperPairAudit> pairs;
};

// Synchronous and intentionally single-threaded. All injected objects must
// outlive this frontend. Before construction, camera rows must be loaded into
// both caches and the reconstruction; the extractor and matcher must each be
// set up exactly once; and Reconstruction::Load plus SetUp must be complete. A
// FATAL_INCONSISTENCY result terminally latches the instance.
class OnlineMapperFrontend {
 public:
  OnlineMapperFrontend(const OnlineMapperFrontendOptions& options,
                       Database* database,
                       FeatureMatcherCache* matcher_cache,
                       DatabaseCache* database_cache,
                       Reconstruction* reconstruction,
                       SingleImageFeatureExtractor* extractor,
                       SiftFeatureMatcher* matcher);

  // Extracts, persists, and synchronizes one fresh image exactly once. A
  // successful call replaces the current image accepted by the matching stage.
  OnlineMapperFrameResult IngestFrame(Image image,
                                      Bitmap* bitmap,
                                      const Bitmap* frame_mask);

  // Matches one ordered batch against the most recently ingested current image.
  // Up to two default-sized batches may be submitted for the same current.
  // Repeated pairs return CACHED audits and never touch the matcher or graphs.
  OnlineMapperFrameResult MatchExplicitReferences(
      image_t current_image_id,
      const std::vector<image_t>& ordered_reference_image_ids);

  // Compatibility wrapper equivalent to IngestFrame followed by
  // MatchExplicitReferences for the same image identifier.
  OnlineMapperFrameResult ProcessFrame(
      Image image,
      Bitmap* bitmap,
      const Bitmap* frame_mask,
      const std::vector<image_t>& reference_image_ids);

 private:
  OnlineMapperFrameResult Reject(OnlineMapperFrameReason reason,
                                 const std::string& detail) const;
  OnlineMapperFrameResult RejectForCurrent(OnlineMapperFrameReason reason,
                                           const std::string& detail) const;
  OnlineMapperFrameResult Fatal(OnlineMapperFrameReason reason,
                                const std::string& detail,
                                size_t keypoint_count,
                                size_t descriptor_count,
                                std::vector<OnlineMapperPairAudit> pairs = {});
  OnlineMapperFrameResult LatchedFatal() const;

  const OnlineMapperFrontendOptions options_;
  Database* const database_;
  FeatureMatcherCache* const matcher_cache_;
  DatabaseCache* const database_cache_;
  Reconstruction* const reconstruction_;
  SingleImageFeatureExtractor* const extractor_;
  SiftFeatureMatcher* const matcher_;
  const std::thread::id control_thread_id_;
  bool fatal_inconsistency_ = false;
  OnlineMapperFrameReason fatal_reason_ = OnlineMapperFrameReason::NONE;
  std::string fatal_detail_;
  image_t current_image_id_ = kInvalidImageId;
  size_t current_keypoint_count_ = 0;
  size_t current_descriptor_count_ = 0;
  size_t current_submitted_reference_count_ = 0;
  std::unordered_map<image_pair_t, OnlineMapperPairAudit>
      processed_pair_audits_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_FRONTEND_H_
