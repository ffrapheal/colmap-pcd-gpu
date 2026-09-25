#include "controllers/online_mapper_state.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "base/database.h"

namespace colmap {
namespace {

bool IsValidImageId(const image_t image_id) {
  return image_id != kInvalidImageId && image_id < Database::kMaxNumImages;
}

bool NormalizePose(const KnownPoseSE3& pose, KnownPoseSE3* normalized_pose) {
  if (!pose.qvec.allFinite() || !pose.tvec.allFinite()) {
    return false;
  }

  const double qvec_norm = pose.qvec.norm();
  if (!std::isfinite(qvec_norm) ||
      !(qvec_norm > std::numeric_limits<double>::epsilon())) {
    return false;
  }

  normalized_pose->qvec = pose.qvec / qvec_norm;
  normalized_pose->tvec = pose.tvec;
  return normalized_pose->qvec.allFinite();
}

}  // namespace

KnownPoseSE3::KnownPoseSE3()
    : qvec(Eigen::Vector4d::Constant(
          std::numeric_limits<double>::quiet_NaN())),
      tvec(Eigen::Vector3d::Constant(
          std::numeric_limits<double>::quiet_NaN())) {}

KnownPoseSE3::KnownPoseSE3(const Eigen::Vector4d& qvec,
                           const Eigen::Vector3d& tvec)
    : qvec(qvec), tvec(tvec) {}

CatchupFailureEvidence::CatchupFailureEvidence(
    const size_t failed_active_edge_evidence_version,
    const size_t failed_trigger_actual_valid_lidar_residual_count,
    const size_t failed_lidar_map_version)
    : failed_active_edge_evidence_version(
          failed_active_edge_evidence_version),
      failed_trigger_actual_valid_lidar_residual_count(
          failed_trigger_actual_valid_lidar_residual_count),
      failed_lidar_map_version(failed_lidar_map_version) {}

KnownPosePromotion::KnownPosePromotion(const image_t image_id,
                                       const KnownPoseSE3& latest_T_cw)
    : image_id(image_id), latest_T_cw(latest_T_cw) {}

KnownPoseLatestPoseUpdate::KnownPoseLatestPoseUpdate(
    const image_t image_id, const KnownPoseSE3& latest_T_cw)
    : image_id(image_id), latest_T_cw(latest_T_cw) {}

KnownPoseCatchupFailureUpdate::KnownPoseCatchupFailureUpdate(
    const image_t image_id, const CatchupFailureEvidence& evidence)
    : image_id(image_id), evidence(evidence) {}

PreparedKnownPoseRegistryCommit::PreparedKnownPoseRegistryCommit(
    PreparedKnownPoseRegistryCommit&& other) {
  *this = std::move(other);
}

PreparedKnownPoseRegistryCommit&
PreparedKnownPoseRegistryCommit::operator=(
    PreparedKnownPoseRegistryCommit&& other) {
  if (this != &other) {
    Reset();
    registry_ = other.registry_;
    owner_thread_id_ = other.owner_thread_id_;
    expected_version_ = other.expected_version_;
    normalized_commit_ = std::move(other.normalized_commit_);
    prepared_ = other.prepared_;
    other.Reset();
  }
  return *this;
}

bool PreparedKnownPoseRegistryCommit::IsPrepared() const noexcept {
  return prepared_;
}

void PreparedKnownPoseRegistryCommit::Reset() noexcept {
  registry_ = nullptr;
  owner_thread_id_ = std::thread::id();
  expected_version_ = 0;
  normalized_commit_.expected_version = 0;
  normalized_commit_.promotions.clear();
  normalized_commit_.latest_pose_updates.clear();
  normalized_commit_.catchup_failures.clear();
  prepared_ = false;
}

KnownPoseRegistry::KnownPoseRegistry()
    : owner_thread_id_(std::this_thread::get_id()) {}

uint64_t KnownPoseRegistry::Version() const noexcept { return version_; }

KnownPoseRegistryResult KnownPoseRegistry::AddKnownPose(
    const image_t image_id,
    const size_t frame_index,
    const KnownPoseSE3& fastlio_T_cw) {
  if (!IsOwnerThread()) {
    return Reject(KnownPoseRegistryStatus::WRONG_THREAD,
                  "AddKnownPose must run on the registry owner thread");
  }
  if (!IsValidImageId(image_id)) {
    return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                  "image_id is outside the COLMAP database range");
  }
  if (frame_index == 0 ||
      frame_index == std::numeric_limits<size_t>::max()) {
    return Reject(KnownPoseRegistryStatus::INVALID_FRAME_INDEX,
                  "frame_index must be finite, non-zero, and one-based");
  }
  if (image_id_to_record_index_.count(image_id) != 0) {
    return Reject(KnownPoseRegistryStatus::DUPLICATE_IMAGE_ID,
                  "image_id is already registered");
  }
  if (frame_index < next_frame_index_) {
    return Reject(KnownPoseRegistryStatus::DUPLICATE_FRAME_INDEX,
                  "frame_index is already registered");
  }
  if (frame_index != next_frame_index_) {
    return Reject(KnownPoseRegistryStatus::NONCONTIGUOUS_FRAME_INDEX,
                  "frame_index must continue the one-based arrival sequence");
  }
  if (next_registration_sequence_ == std::numeric_limits<size_t>::max()) {
    return Reject(KnownPoseRegistryStatus::REGISTRATION_SEQUENCE_EXHAUSTED,
                  "registration_sequence cannot be incremented");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(KnownPoseRegistryStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }

  KnownPoseSE3 normalized_fastlio_T_cw;
  if (!NormalizePose(fastlio_T_cw, &normalized_fastlio_T_cw)) {
    return Reject(KnownPoseRegistryStatus::INVALID_POSE,
                  "fastlio_T_cw must be a finite valid SE3");
  }

  KnownPoseRecord record;
  record.image_id = image_id;
  record.frame_index = frame_index;
  record.registration_sequence = next_registration_sequence_;
  record.fastlio_T_cw = normalized_fastlio_T_cw;
  record.latest_T_cw = normalized_fastlio_T_cw;

  const size_t record_index = records_.size();
  records_.push_back(record);
  image_id_to_record_index_.emplace(image_id, record_index);
  ++next_frame_index_;
  ++next_registration_sequence_;
  ++version_;

  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.registration_sequence = record.registration_sequence;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::PromoteToVisualActive(
    const image_t image_id, const KnownPoseSE3& latest_T_cw) {
  if (!IsOwnerThread()) {
    return Reject(
        KnownPoseRegistryStatus::WRONG_THREAD,
        "PromoteToVisualActive must run on the registry owner thread");
  }
  if (!IsValidImageId(image_id)) {
    return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                  "image_id is outside the COLMAP database range");
  }

  size_t record_index = 0;
  if (!FindRecordIndex(image_id, &record_index)) {
    return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                  "image_id is not registered");
  }
  KnownPoseRecord& record = records_[record_index];
  if (record.visual_state == KnownPoseVisualState::VISUAL_ACTIVE) {
    return Reject(KnownPoseRegistryStatus::ALREADY_VISUAL_ACTIVE,
                  "visual state cannot be promoted more than once");
  }

  KnownPoseSE3 normalized_latest_T_cw;
  if (!NormalizePose(latest_T_cw, &normalized_latest_T_cw)) {
    return Reject(KnownPoseRegistryStatus::INVALID_POSE,
                  "latest_T_cw must be a finite valid SE3");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(KnownPoseRegistryStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }

  record.latest_T_cw = normalized_latest_T_cw;
  record.visual_state = KnownPoseVisualState::VISUAL_ACTIVE;
  ++version_;

  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.registration_sequence = record.registration_sequence;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::UpdateLatestPose(
    const image_t image_id, const KnownPoseSE3& latest_T_cw) {
  if (!IsOwnerThread()) {
    return Reject(KnownPoseRegistryStatus::WRONG_THREAD,
                  "UpdateLatestPose must run on the registry owner thread");
  }
  if (!IsValidImageId(image_id)) {
    return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                  "image_id is outside the COLMAP database range");
  }

  size_t record_index = 0;
  if (!FindRecordIndex(image_id, &record_index)) {
    return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                  "image_id is not registered");
  }
  KnownPoseRecord& record = records_[record_index];
  if (record.visual_state != KnownPoseVisualState::VISUAL_ACTIVE) {
    return Reject(KnownPoseRegistryStatus::REQUIRES_VISUAL_ACTIVE,
                  "latest pose updates require VISUAL_ACTIVE state");
  }

  KnownPoseSE3 normalized_latest_T_cw;
  if (!NormalizePose(latest_T_cw, &normalized_latest_T_cw)) {
    return Reject(KnownPoseRegistryStatus::INVALID_POSE,
                  "latest_T_cw must be a finite valid SE3");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(KnownPoseRegistryStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }

  record.latest_T_cw = normalized_latest_T_cw;
  ++version_;

  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.registration_sequence = record.registration_sequence;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::RecordCatchupFailure(
    const image_t image_id, const CatchupFailureEvidence& evidence) {
  if (!IsOwnerThread()) {
    return Reject(KnownPoseRegistryStatus::WRONG_THREAD,
                  "RecordCatchupFailure must run on the registry owner thread");
  }
  if (!IsValidImageId(image_id)) {
    return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                  "image_id is outside the COLMAP database range");
  }

  size_t record_index = 0;
  if (!FindRecordIndex(image_id, &record_index)) {
    return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                  "image_id is not registered");
  }
  KnownPoseRecord& record = records_[record_index];
  if (record.visual_state != KnownPoseVisualState::POSE_ONLY) {
    return Reject(KnownPoseRegistryStatus::REQUIRES_POSE_ONLY,
                  "CATCHUP failure evidence requires POSE_ONLY state");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(KnownPoseRegistryStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }

  record.last_catchup_failure = evidence;
  record.has_catchup_failure_evidence = true;
  ++version_;

  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.registration_sequence = record.registration_sequence;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::ValidateCommit(
    const KnownPoseRegistryCommit& commit) const {
  if (!IsOwnerThread()) {
    return Reject(KnownPoseRegistryStatus::WRONG_THREAD,
                  "commit validation must run on the registry owner thread");
  }
  if (commit.expected_version != version_) {
    return Reject(KnownPoseRegistryStatus::STALE_CANONICAL_VERSION,
                  "commit expected_version does not match canonical state");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(KnownPoseRegistryStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }
  if (commit.promotions.empty() && commit.latest_pose_updates.empty() &&
      commit.catchup_failures.empty()) {
    return Reject(KnownPoseRegistryStatus::EMPTY_COMMIT,
                  "registry commit has no writes");
  }

  std::set<image_t> written_image_ids;
  for (const KnownPosePromotion& promotion : commit.promotions) {
    if (!IsValidImageId(promotion.image_id)) {
      return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                    "promotion image_id is invalid");
    }
    if (!written_image_ids.insert(promotion.image_id).second) {
      return Reject(KnownPoseRegistryStatus::DUPLICATE_WRITE,
                    "registry commit writes one image more than once");
    }
    size_t record_index = 0;
    if (!FindRecordIndex(promotion.image_id, &record_index)) {
      return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                    "promotion image_id is not registered");
    }
    if (records_[record_index].visual_state ==
        KnownPoseVisualState::VISUAL_ACTIVE) {
      return Reject(KnownPoseRegistryStatus::ALREADY_VISUAL_ACTIVE,
                    "promotion requires a POSE_ONLY record");
    }
    KnownPoseSE3 normalized_pose;
    if (!NormalizePose(promotion.latest_T_cw, &normalized_pose)) {
      return Reject(KnownPoseRegistryStatus::INVALID_POSE,
                    "promotion latest_T_cw is invalid");
    }
  }

  for (const KnownPoseLatestPoseUpdate& update :
       commit.latest_pose_updates) {
    if (!IsValidImageId(update.image_id)) {
      return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                    "latest-pose image_id is invalid");
    }
    if (!written_image_ids.insert(update.image_id).second) {
      return Reject(KnownPoseRegistryStatus::DUPLICATE_WRITE,
                    "registry commit writes one image more than once");
    }
    size_t record_index = 0;
    if (!FindRecordIndex(update.image_id, &record_index)) {
      return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                    "latest-pose image_id is not registered");
    }
    if (records_[record_index].visual_state !=
        KnownPoseVisualState::VISUAL_ACTIVE) {
      return Reject(KnownPoseRegistryStatus::REQUIRES_VISUAL_ACTIVE,
                    "latest-pose update requires VISUAL_ACTIVE state");
    }
    KnownPoseSE3 normalized_pose;
    if (!NormalizePose(update.latest_T_cw, &normalized_pose)) {
      return Reject(KnownPoseRegistryStatus::INVALID_POSE,
                    "latest-pose update is invalid");
    }
  }

  for (const KnownPoseCatchupFailureUpdate& failure :
       commit.catchup_failures) {
    if (!IsValidImageId(failure.image_id)) {
      return Reject(KnownPoseRegistryStatus::INVALID_IMAGE_ID,
                    "catchup-failure image_id is invalid");
    }
    if (!written_image_ids.insert(failure.image_id).second) {
      return Reject(KnownPoseRegistryStatus::DUPLICATE_WRITE,
                    "registry commit writes one image more than once");
    }
    size_t record_index = 0;
    if (!FindRecordIndex(failure.image_id, &record_index)) {
      return Reject(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                    "catchup-failure image_id is not registered");
    }
    if (records_[record_index].visual_state !=
        KnownPoseVisualState::POSE_ONLY) {
      return Reject(KnownPoseRegistryStatus::REQUIRES_POSE_ONLY,
                    "catchup failure requires POSE_ONLY state");
    }
  }

  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::PrepareCommit(
    const KnownPoseRegistryCommit& commit,
    PreparedKnownPoseRegistryCommit* prepared) const {
  if (prepared == nullptr) {
    return Reject(KnownPoseRegistryStatus::INVALID_PREPARED_COMMIT,
                  "prepared commit output is null");
  }
  prepared->Reset();
  const KnownPoseRegistryResult validation = ValidateCommit(commit);
  if (!validation.IsSuccess()) {
    return validation;
  }

  PreparedKnownPoseRegistryCommit next;
  next.registry_ = this;
  next.owner_thread_id_ = owner_thread_id_;
  next.expected_version_ = commit.expected_version;
  next.normalized_commit_.expected_version = commit.expected_version;
  next.normalized_commit_.promotions.reserve(commit.promotions.size());
  for (const KnownPosePromotion& promotion : commit.promotions) {
    KnownPoseSE3 normalized_pose;
    NormalizePose(promotion.latest_T_cw, &normalized_pose);
    next.normalized_commit_.promotions.emplace_back(promotion.image_id,
                                                    normalized_pose);
  }
  next.normalized_commit_.latest_pose_updates.reserve(
      commit.latest_pose_updates.size());
  for (const KnownPoseLatestPoseUpdate& update :
       commit.latest_pose_updates) {
    KnownPoseSE3 normalized_pose;
    NormalizePose(update.latest_T_cw, &normalized_pose);
    next.normalized_commit_.latest_pose_updates.emplace_back(update.image_id,
                                                             normalized_pose);
  }
  next.normalized_commit_.catchup_failures = commit.catchup_failures;
  next.prepared_ = true;
  *prepared = std::move(next);
  return validation;
}

KnownPoseRegistryResult KnownPoseRegistry::ValidatePreparedCommit(
    const PreparedKnownPoseRegistryCommit& prepared) const {
  if (!IsOwnerThread() ||
      std::this_thread::get_id() != prepared.owner_thread_id_) {
    return Reject(KnownPoseRegistryStatus::WRONG_THREAD,
                  "prepared commit must remain on the registry owner thread");
  }
  if (!prepared.prepared_ || prepared.registry_ != this ||
      prepared.normalized_commit_.expected_version !=
          prepared.expected_version_) {
    return Reject(KnownPoseRegistryStatus::INVALID_PREPARED_COMMIT,
                  "prepared commit does not belong to this registry");
  }
  if (prepared.expected_version_ != version_) {
    return Reject(KnownPoseRegistryStatus::STALE_CANONICAL_VERSION,
                  "prepared commit expected_version is stale");
  }
  KnownPoseRegistryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.canonical_version = version_;
  return result;
}

KnownPoseRegistryResult KnownPoseRegistry::Commit(
    const KnownPoseRegistryCommit& commit) {
  PreparedKnownPoseRegistryCommit prepared;
  KnownPoseRegistryResult result = PrepareCommit(commit, &prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  result = ValidatePreparedCommit(prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  CommitPrepared(&prepared);
  result.canonical_version = version_;
  return result;
}

void KnownPoseRegistry::CommitPrepared(
    PreparedKnownPoseRegistryCommit* prepared) noexcept {
  for (const KnownPosePromotion& promotion :
       prepared->normalized_commit_.promotions) {
    KnownPoseRecord& record =
        records_[image_id_to_record_index_.find(promotion.image_id)->second];
    record.latest_T_cw = promotion.latest_T_cw;
    record.visual_state = KnownPoseVisualState::VISUAL_ACTIVE;
  }
  for (const KnownPoseLatestPoseUpdate& update :
       prepared->normalized_commit_.latest_pose_updates) {
    records_[image_id_to_record_index_.find(update.image_id)->second]
        .latest_T_cw = update.latest_T_cw;
  }
  for (const KnownPoseCatchupFailureUpdate& failure :
       prepared->normalized_commit_.catchup_failures) {
    KnownPoseRecord& record =
        records_[image_id_to_record_index_.find(failure.image_id)->second];
    record.last_catchup_failure = failure.evidence;
    record.has_catchup_failure_evidence = true;
  }
  version_ = prepared->expected_version_ + 1;
  prepared->registry_ = nullptr;
  prepared->owner_thread_id_ = std::thread::id();
  prepared->expected_version_ = 0;
  prepared->prepared_ = false;
}

KnownPoseRecordQueryResult KnownPoseRegistry::GetByImageId(
    const image_t image_id) const {
  if (!IsOwnerThread()) {
    return RejectRecordQuery(
        KnownPoseRegistryStatus::WRONG_THREAD,
        "GetByImageId must run on the registry owner thread");
  }
  if (!IsValidImageId(image_id)) {
    return RejectRecordQuery(
        KnownPoseRegistryStatus::INVALID_IMAGE_ID,
        "image_id is outside the COLMAP database range");
  }

  size_t record_index = 0;
  if (!FindRecordIndex(image_id, &record_index)) {
    return RejectRecordQuery(KnownPoseRegistryStatus::IMAGE_NOT_FOUND,
                             "image_id is not registered");
  }

  KnownPoseRecordQueryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.has_record = true;
  result.record = records_[record_index];
  return result;
}

KnownPoseRecordQueryResult KnownPoseRegistry::GetByRegistrationSequence(
    const size_t registration_sequence) const {
  if (!IsOwnerThread()) {
    return RejectRecordQuery(
        KnownPoseRegistryStatus::WRONG_THREAD,
        "GetByRegistrationSequence must run on the registry owner thread");
  }
  if (registration_sequence == 0 ||
      registration_sequence == std::numeric_limits<size_t>::max()) {
    return RejectRecordQuery(
        KnownPoseRegistryStatus::INVALID_REGISTRATION_SEQUENCE,
        "registration_sequence must be finite, non-zero, and one-based");
  }

  const size_t record_index = registration_sequence - 1;
  if (record_index >= records_.size() ||
      records_[record_index].registration_sequence != registration_sequence) {
    return RejectRecordQuery(
        KnownPoseRegistryStatus::REGISTRATION_SEQUENCE_NOT_FOUND,
        "registration_sequence is not registered");
  }

  KnownPoseRecordQueryResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.has_record = true;
  result.record = records_[record_index];
  return result;
}

KnownPoseImageIdsResult KnownPoseRegistry::GetRegisteredImageIds() const {
  return GetImageIds(KnownPoseVisualState::POSE_ONLY, false);
}

KnownPoseImageIdsResult KnownPoseRegistry::GetPoseOnlyImageIds() const {
  return GetImageIds(KnownPoseVisualState::POSE_ONLY, true);
}

KnownPoseImageIdsResult KnownPoseRegistry::GetVisualActiveImageIds() const {
  return GetImageIds(KnownPoseVisualState::VISUAL_ACTIVE, true);
}

bool KnownPoseRegistry::IsOwnerThread() const noexcept {
  return std::this_thread::get_id() == owner_thread_id_;
}

bool KnownPoseRegistry::FindRecordIndex(const image_t image_id,
                                        size_t* record_index) const {
  const auto it = image_id_to_record_index_.find(image_id);
  if (it == image_id_to_record_index_.end()) {
    return false;
  }
  *record_index = it->second;
  return true;
}

KnownPoseRegistryResult KnownPoseRegistry::Reject(
    const KnownPoseRegistryStatus status, const std::string& detail) const {
  KnownPoseRegistryResult result;
  result.status = status;
  result.detail = detail;
  result.canonical_version = IsOwnerThread() ? version_ : 0;
  return result;
}

KnownPoseRecordQueryResult KnownPoseRegistry::RejectRecordQuery(
    const KnownPoseRegistryStatus status, const std::string& detail) const {
  KnownPoseRecordQueryResult result;
  result.status = status;
  result.detail = detail;
  return result;
}

KnownPoseImageIdsResult KnownPoseRegistry::RejectImageIdsQuery(
    const KnownPoseRegistryStatus status, const std::string& detail) const {
  KnownPoseImageIdsResult result;
  result.status = status;
  result.detail = detail;
  return result;
}

KnownPoseImageIdsResult KnownPoseRegistry::GetImageIds(
    const KnownPoseVisualState state, const bool filter_by_state) const {
  if (!IsOwnerThread()) {
    return RejectImageIdsQuery(
        KnownPoseRegistryStatus::WRONG_THREAD,
        "image ID queries must run on the registry owner thread");
  }

  KnownPoseImageIdsResult result;
  result.status = KnownPoseRegistryStatus::SUCCESS;
  result.image_ids.reserve(records_.size());
  for (const KnownPoseRecord& record : records_) {
    if (!filter_by_state || record.visual_state == state) {
      result.image_ids.push_back(record.image_id);
    }
  }
  return result;
}

Point3DOwnerInput::Point3DOwnerInput(
    const point3D_t point3D_id,
    const image_t image_id,
    const point2D_t point2D_idx,
    const uint64_t association_identity)
    : point3D_id(point3D_id),
      image_id(image_id),
      point2D_idx(point2D_idx),
      association_identity(association_identity) {}

PreparedPoint3DOwnerTableCommit::PreparedPoint3DOwnerTableCommit(
    PreparedPoint3DOwnerTableCommit&& other) {
  *this = std::move(other);
}

PreparedPoint3DOwnerTableCommit&
PreparedPoint3DOwnerTableCommit::operator=(
    PreparedPoint3DOwnerTableCommit&& other) {
  if (this != &other) {
    Reset();
    table_ = other.table_;
    owner_thread_id_ = other.owner_thread_id_;
    expected_version_ = other.expected_version_;
    committed_version_ = other.committed_version_;
    next_creation_sequence_ = other.next_creation_sequence_;
    prepared_ = other.prepared_;
    owners_ = std::move(other.owners_);
    observation_to_point3D_ = std::move(other.observation_to_point3D_);
    association_to_point3D_ = std::move(other.association_to_point3D_);
    other.Reset();
  }
  return *this;
}

bool PreparedPoint3DOwnerTableCommit::IsPrepared() const noexcept {
  return prepared_;
}

void PreparedPoint3DOwnerTableCommit::Reset() noexcept {
  table_ = nullptr;
  owner_thread_id_ = std::thread::id();
  expected_version_ = 0;
  committed_version_ = 0;
  next_creation_sequence_ = 1;
  prepared_ = false;
  owners_.clear();
  observation_to_point3D_.clear();
  association_to_point3D_.clear();
}

Point3DOwnerTable::Point3DOwnerTable()
    : owner_thread_id_(std::this_thread::get_id()) {}

uint64_t Point3DOwnerTable::Version() const noexcept { return version_; }

bool Point3DOwnerTable::IsOwnerThread() const noexcept {
  return std::this_thread::get_id() == owner_thread_id_;
}

size_t Point3DOwnerTable::Size() const noexcept { return owners_.size(); }

Point3DOwnerQueryResult Point3DOwnerTable::Get(
    const point3D_t point3D_id) const {
  Point3DOwnerQueryResult result;
  if (!IsOwnerThread()) {
    result.status = Point3DOwnerTableStatus::WRONG_THREAD;
    result.detail = "owner query must run on the table owner thread";
    return result;
  }
  result.canonical_version = version_;
  const auto owner = owners_.find(point3D_id);
  if (owner == owners_.end()) {
    result.status = Point3DOwnerTableStatus::OWNER_NOT_FOUND;
    result.detail = "Point3D owner is not registered";
    return result;
  }
  result.status = Point3DOwnerTableStatus::SUCCESS;
  result.has_owner = true;
  result.owner = owner->second;
  return result;
}

Point3DOwnerTableSnapshot Point3DOwnerTable::Snapshot() const {
  Point3DOwnerTableSnapshot snapshot;
  if (!IsOwnerThread()) {
    snapshot.status = Point3DOwnerTableStatus::WRONG_THREAD;
    snapshot.detail = "owner snapshot must run on the table owner thread";
    return snapshot;
  }
  snapshot.canonical_version = version_;
  snapshot.next_creation_sequence = next_creation_sequence_;
  snapshot.status = Point3DOwnerTableStatus::SUCCESS;
  snapshot.owners.reserve(owners_.size());
  for (const auto& owner : owners_) {
    snapshot.owners.push_back(owner.second);
  }
  std::sort(snapshot.owners.begin(),
            snapshot.owners.end(),
            [](const Point3DOwner& lhs, const Point3DOwner& rhs) {
              return std::tie(lhs.creation_sequence, lhs.point3D_id) <
                     std::tie(rhs.creation_sequence, rhs.point3D_id);
            });
  return snapshot;
}

Point3DOwnerTableResult Point3DOwnerTable::Add(
    const Point3DOwnerInput& owner) {
  Point3DOwnerTableCommit commit;
  commit.expected_version = version_;
  commit.owners.push_back(owner);
  return Commit(commit);
}

Point3DOwnerTableResult Point3DOwnerTable::ValidateCommit(
    const Point3DOwnerTableCommit& commit) const {
  if (!IsOwnerThread()) {
    return Reject(Point3DOwnerTableStatus::WRONG_THREAD,
                  "commit validation must run on the table owner thread");
  }
  if (commit.expected_version != version_) {
    return Reject(Point3DOwnerTableStatus::STALE_CANONICAL_VERSION,
                  "commit expected_version does not match canonical state");
  }
  if (version_ == std::numeric_limits<uint64_t>::max()) {
    return Reject(Point3DOwnerTableStatus::CANONICAL_VERSION_EXHAUSTED,
                  "canonical version cannot be incremented");
  }
  if (commit.owners.empty()) {
    return Reject(Point3DOwnerTableStatus::EMPTY_COMMIT,
                  "owner-table commit has no writes");
  }
  if (next_creation_sequence_ == 0 ||
      commit.owners.size() >
          std::numeric_limits<uint64_t>::max() - next_creation_sequence_) {
    return Reject(Point3DOwnerTableStatus::CREATION_SEQUENCE_EXHAUSTED,
                  "creation sequence cannot cover the commit");
  }

  std::set<point3D_t> point3D_ids;
  std::set<ObservationKey> observations;
  std::set<uint64_t> association_identities;
  for (const Point3DOwnerInput& input : commit.owners) {
    if (input.point3D_id == kInvalidPoint3DId) {
      return Reject(Point3DOwnerTableStatus::INVALID_POINT3D_ID,
                    "owner point3D_id is invalid");
    }
    if (!IsValidImageId(input.image_id)) {
      return Reject(Point3DOwnerTableStatus::INVALID_IMAGE_ID,
                    "owner image_id is invalid");
    }
    if (input.point2D_idx == kInvalidPoint2DIdx) {
      return Reject(Point3DOwnerTableStatus::INVALID_POINT2D_INDEX,
                    "owner point2D_idx is invalid");
    }
    if (input.association_identity == 0) {
      return Reject(Point3DOwnerTableStatus::INVALID_ASSOCIATION_IDENTITY,
                    "association identity must be non-zero");
    }
    if (owners_.count(input.point3D_id) != 0 ||
        !point3D_ids.insert(input.point3D_id).second) {
      return Reject(Point3DOwnerTableStatus::DUPLICATE_POINT3D_ID,
                    "Point3D already has an owner");
    }
    const ObservationKey observation(input.image_id, input.point2D_idx);
    if (observation_to_point3D_.count(observation) != 0 ||
        !observations.insert(observation).second) {
      return Reject(Point3DOwnerTableStatus::DUPLICATE_OWNER_OBSERVATION,
                    "owner observation is already assigned");
    }
    if (association_to_point3D_.count(input.association_identity) != 0 ||
        !association_identities.insert(input.association_identity).second) {
      return Reject(
          Point3DOwnerTableStatus::DUPLICATE_ASSOCIATION_IDENTITY,
          "association identity is already assigned");
    }
  }
  Point3DOwnerTableResult result;
  result.status = Point3DOwnerTableStatus::SUCCESS;
  result.canonical_version = version_;
  return result;
}

Point3DOwnerTableResult Point3DOwnerTable::PrepareCommit(
    const Point3DOwnerTableCommit& commit,
    PreparedPoint3DOwnerTableCommit* prepared) const {
  if (prepared == nullptr) {
    return Reject(Point3DOwnerTableStatus::INVALID_PREPARED_COMMIT,
                  "prepared commit output is null");
  }
  prepared->Reset();
  const Point3DOwnerTableResult validation = ValidateCommit(commit);
  if (!validation.IsSuccess()) {
    return validation;
  }

  PreparedPoint3DOwnerTableCommit next;
  next.table_ = this;
  next.owner_thread_id_ = owner_thread_id_;
  next.expected_version_ = commit.expected_version;
  next.committed_version_ = commit.expected_version + 1;
  next.next_creation_sequence_ = next_creation_sequence_;
  next.owners_ = owners_;
  next.observation_to_point3D_ = observation_to_point3D_;
  next.association_to_point3D_ = association_to_point3D_;
  for (const Point3DOwnerInput& input : commit.owners) {
    Point3DOwner owner;
    owner.point3D_id = input.point3D_id;
    owner.image_id = input.image_id;
    owner.point2D_idx = input.point2D_idx;
    owner.association_identity = input.association_identity;
    owner.creation_sequence = next.next_creation_sequence_++;
    next.owners_.emplace(owner.point3D_id, owner);
    next.observation_to_point3D_.emplace(
        ObservationKey(owner.image_id, owner.point2D_idx), owner.point3D_id);
    next.association_to_point3D_.emplace(owner.association_identity,
                                         owner.point3D_id);
  }
  next.prepared_ = true;
  *prepared = std::move(next);
  return validation;
}

Point3DOwnerTableResult Point3DOwnerTable::ValidatePreparedCommit(
    const PreparedPoint3DOwnerTableCommit& prepared) const {
  if (!IsOwnerThread() ||
      std::this_thread::get_id() != prepared.owner_thread_id_) {
    return Reject(Point3DOwnerTableStatus::WRONG_THREAD,
                  "prepared commit must remain on the table owner thread");
  }
  if (!prepared.prepared_ || prepared.table_ != this ||
      prepared.committed_version_ != prepared.expected_version_ + 1) {
    return Reject(Point3DOwnerTableStatus::INVALID_PREPARED_COMMIT,
                  "prepared commit does not belong to this owner table");
  }
  if (prepared.expected_version_ != version_) {
    return Reject(Point3DOwnerTableStatus::STALE_CANONICAL_VERSION,
                  "prepared commit expected_version is stale");
  }
  Point3DOwnerTableResult result;
  result.status = Point3DOwnerTableStatus::SUCCESS;
  result.canonical_version = version_;
  return result;
}

Point3DOwnerTableResult Point3DOwnerTable::Commit(
    const Point3DOwnerTableCommit& commit) {
  PreparedPoint3DOwnerTableCommit prepared;
  Point3DOwnerTableResult result = PrepareCommit(commit, &prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  result = ValidatePreparedCommit(prepared);
  if (!result.IsSuccess()) {
    return result;
  }
  CommitPrepared(&prepared);
  result.canonical_version = version_;
  return result;
}

void Point3DOwnerTable::CommitPrepared(
    PreparedPoint3DOwnerTableCommit* prepared) noexcept {
  owners_.swap(prepared->owners_);
  observation_to_point3D_.swap(prepared->observation_to_point3D_);
  association_to_point3D_.swap(prepared->association_to_point3D_);
  next_creation_sequence_ = prepared->next_creation_sequence_;
  version_ = prepared->committed_version_;
  prepared->table_ = nullptr;
  prepared->owner_thread_id_ = std::thread::id();
  prepared->expected_version_ = 0;
  prepared->committed_version_ = 0;
  prepared->next_creation_sequence_ = 1;
  prepared->prepared_ = false;
}

Point3DOwnerTableResult Point3DOwnerTable::Reject(
    const Point3DOwnerTableStatus status, const std::string& detail) const {
  Point3DOwnerTableResult result;
  result.status = status;
  result.detail = detail;
  result.canonical_version = IsOwnerThread() ? version_ : 0;
  return result;
}

bool OnlineMapperCanonicalVersion::operator==(
    const OnlineMapperCanonicalVersion& other) const {
  return reconstruction == other.reconstruction &&
         known_pose_registry == other.known_pose_registry &&
         active_covisibility_graph == other.active_covisibility_graph &&
         point3D_owner_table == other.point3D_owner_table;
}

bool OnlineMapperCanonicalVersion::operator!=(
    const OnlineMapperCanonicalVersion& other) const {
  return !(*this == other);
}

namespace {

OnlineMapperTransactionResult MakeOnlineMapperTransactionResult(
    const OnlineMapperTransactionStatus status,
    const std::string& detail,
    const OnlineMapperCanonicalVersion& canonical_version) {
  OnlineMapperTransactionResult result;
  result.status = status;
  result.detail = detail;
  result.canonical_version = canonical_version;
  return result;
}

OnlineMapperCanonicalVersion CurrentOnlineMapperCanonicalVersion(
    const Reconstruction& reconstruction,
    const KnownPoseRegistry& known_pose_registry,
    const ActiveCovisibilityGraph& active_covisibility_graph,
    const Point3DOwnerTable& point3D_owner_table) {
  OnlineMapperCanonicalVersion version;
  version.reconstruction = reconstruction.CanonicalVersion();
  version.known_pose_registry = known_pose_registry.Version();
  version.active_covisibility_graph = active_covisibility_graph.Version();
  version.point3D_owner_table = point3D_owner_table.Version();
  return version;
}

bool HasKnownPoseWrites(const KnownPoseRegistryCommit& commit) {
  return !commit.promotions.empty() || !commit.latest_pose_updates.empty() ||
         !commit.catchup_failures.empty();
}

bool HasGraphWrites(const ActiveCovisibilityCommit& commit) {
  return !commit.nodes.empty() || !commit.ordinary_edges.empty() ||
         !commit.loop_edges.empty();
}

bool HasOwnerWrites(const Point3DOwnerTableCommit& commit) {
  return !commit.owners.empty();
}

}  // namespace

OnlineMapperPreparedTransaction::OnlineMapperPreparedTransaction(
    OnlineMapperPreparedTransaction&& other) {
  *this = std::move(other);
}

OnlineMapperPreparedTransaction& OnlineMapperPreparedTransaction::operator=(
    OnlineMapperPreparedTransaction&& other) {
  if (this != &other) {
    Reset();
    reconstruction_state_ = other.reconstruction_state_;
    known_pose_registry_state_ = other.known_pose_registry_state_;
    active_covisibility_graph_state_ =
        other.active_covisibility_graph_state_;
    point3D_owner_table_state_ = other.point3D_owner_table_state_;
    expected_version_ = other.expected_version_;
    reconstruction_ = std::move(other.reconstruction_);
    known_pose_registry_ = std::move(other.known_pose_registry_);
    active_covisibility_graph_ =
        std::move(other.active_covisibility_graph_);
    point3D_owner_table_ = std::move(other.point3D_owner_table_);
    has_known_pose_registry_commit_ =
        other.has_known_pose_registry_commit_;
    has_active_covisibility_graph_commit_ =
        other.has_active_covisibility_graph_commit_;
    has_point3D_owner_table_commit_ =
        other.has_point3D_owner_table_commit_;
    prepared_ = other.prepared_;
    other.Reset();
  }
  return *this;
}

bool OnlineMapperPreparedTransaction::IsPrepared() const noexcept {
  return prepared_;
}

void OnlineMapperPreparedTransaction::Reset() {
  reconstruction_state_ = nullptr;
  known_pose_registry_state_ = nullptr;
  active_covisibility_graph_state_ = nullptr;
  point3D_owner_table_state_ = nullptr;
  expected_version_ = OnlineMapperCanonicalVersion();
  reconstruction_ = ReconstructionTransaction();
  known_pose_registry_ = PreparedKnownPoseRegistryCommit();
  active_covisibility_graph_ = PreparedActiveCovisibilityCommit();
  point3D_owner_table_ = PreparedPoint3DOwnerTableCommit();
  has_known_pose_registry_commit_ = false;
  has_active_covisibility_graph_commit_ = false;
  has_point3D_owner_table_commit_ = false;
  prepared_ = false;
}

OnlineMapperTransactionResult CaptureOnlineMapperCanonicalVersion(
    const Reconstruction* reconstruction,
    const KnownPoseRegistry* known_pose_registry,
    const ActiveCovisibilityGraph* active_covisibility_graph,
    const Point3DOwnerTable* point3D_owner_table) {
  if (reconstruction == nullptr || known_pose_registry == nullptr ||
      active_covisibility_graph == nullptr || point3D_owner_table == nullptr) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::NULL_STATE,
        "online mapper canonical state pointer is null",
        OnlineMapperCanonicalVersion());
  }
  if (!reconstruction->IsTransactionOwnerThread() ||
      !known_pose_registry->IsOwnerThread() ||
      !active_covisibility_graph->IsOwnerThread() ||
      !point3D_owner_table->IsOwnerThread()) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::WRONG_THREAD,
        "all canonical state must be accessed by its owner thread",
        OnlineMapperCanonicalVersion());
  }
  const OnlineMapperCanonicalVersion version =
      CurrentOnlineMapperCanonicalVersion(*reconstruction,
                                          *known_pose_registry,
                                          *active_covisibility_graph,
                                          *point3D_owner_table);
  return MakeOnlineMapperTransactionResult(
      OnlineMapperTransactionStatus::SUCCESS, std::string(), version);
}

OnlineMapperTransactionResult ValidateOnlineMapperCanonicalVersion(
    const Reconstruction* reconstruction,
    const KnownPoseRegistry* known_pose_registry,
    const ActiveCovisibilityGraph* active_covisibility_graph,
    const Point3DOwnerTable* point3D_owner_table,
    const OnlineMapperCanonicalVersion& expected_version) {
  OnlineMapperTransactionResult result = CaptureOnlineMapperCanonicalVersion(
      reconstruction,
      known_pose_registry,
      active_covisibility_graph,
      point3D_owner_table);
  if (!result.IsSuccess()) {
    return result;
  }
  if (result.canonical_version != expected_version) {
    result.status = OnlineMapperTransactionStatus::STALE_CANONICAL_VERSION;
    result.detail = "online mapper canonical version drifted";
  }
  return result;
}

OnlineMapperTransactionResult BeginOnlineMapperTransaction(
    Reconstruction* reconstruction,
    KnownPoseRegistry* known_pose_registry,
    ActiveCovisibilityGraph* active_covisibility_graph,
    Point3DOwnerTable* point3D_owner_table,
    OnlineMapperTransactionPayload* payload) {
  if (payload == nullptr) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PAYLOAD,
        "online mapper transaction payload is null",
        OnlineMapperCanonicalVersion());
  }
  OnlineMapperTransactionResult result = CaptureOnlineMapperCanonicalVersion(
      reconstruction,
      known_pose_registry,
      active_covisibility_graph,
      point3D_owner_table);
  if (!result.IsSuccess()) {
    return result;
  }

  payload->reconstruction = ReconstructionTransaction();
  payload->known_pose_registry = KnownPoseRegistryCommit();
  payload->active_covisibility_graph = ActiveCovisibilityCommit();
  payload->point3D_owner_table = Point3DOwnerTableCommit();
  const ReconstructionTransactionResult reconstruction_result =
      reconstruction->CreateTransactionSnapshot(&payload->reconstruction);
  if (!reconstruction_result.IsSuccess()) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::COMPONENT_PREPARATION_FAILED,
        reconstruction_result.detail,
        result.canonical_version);
  }

  payload->expected_version = result.canonical_version;
  payload->known_pose_registry.expected_version =
      result.canonical_version.known_pose_registry;
  payload->active_covisibility_graph.expected_version =
      result.canonical_version.active_covisibility_graph;
  payload->point3D_owner_table.expected_version =
      result.canonical_version.point3D_owner_table;
  return result;
}

OnlineMapperTransactionResult PrepareOnlineMapperTransaction(
    Reconstruction* reconstruction,
    KnownPoseRegistry* known_pose_registry,
    ActiveCovisibilityGraph* active_covisibility_graph,
    Point3DOwnerTable* point3D_owner_table,
    OnlineMapperTransactionPayload* payload,
    OnlineMapperPreparedTransaction* prepared) {
  if (payload == nullptr || prepared == nullptr) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PAYLOAD,
        "online mapper payload or prepared output is null",
        OnlineMapperCanonicalVersion());
  }
  prepared->Reset();
  OnlineMapperTransactionResult result = ValidateOnlineMapperCanonicalVersion(
      reconstruction,
      known_pose_registry,
      active_covisibility_graph,
      point3D_owner_table,
      payload->expected_version);
  if (!result.IsSuccess()) {
    return result;
  }
  if (payload->reconstruction.ExpectedVersion() !=
          payload->expected_version.reconstruction ||
      payload->known_pose_registry.expected_version !=
          payload->expected_version.known_pose_registry ||
      payload->active_covisibility_graph.expected_version !=
          payload->expected_version.active_covisibility_graph ||
      payload->point3D_owner_table.expected_version !=
          payload->expected_version.point3D_owner_table) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PAYLOAD,
        "component expected versions differ from the unified version",
        result.canonical_version);
  }

  const Reconstruction* candidate = payload->reconstruction.Candidate();
  if (candidate == nullptr) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PAYLOAD,
        "Reconstruction candidate is missing or on the wrong thread",
        result.canonical_version);
  }
  for (const Point3DOwnerInput& owner : payload->point3D_owner_table.owners) {
    if (!candidate->ExistsPoint3D(owner.point3D_id) ||
        !candidate->ExistsImage(owner.image_id) ||
        owner.point2D_idx >= candidate->Image(owner.image_id).NumPoints2D()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::INVALID_PAYLOAD,
          "new Point3D owner does not exist in the candidate Reconstruction",
          result.canonical_version);
    }
    const Point2D& point2D =
        candidate->Image(owner.image_id).Point2D(owner.point2D_idx);
    if (!point2D.HasPoint3D() || point2D.Point3DId() != owner.point3D_id) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::INVALID_PAYLOAD,
          "new Point3D owner observation does not point back to its point",
          result.canonical_version);
    }
  }

  ReconstructionTransactionResult reconstruction_result =
      reconstruction->PrepareTransactionCommit(&payload->reconstruction);
  if (!reconstruction_result.IsSuccess()) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::COMPONENT_PREPARATION_FAILED,
        reconstruction_result.detail,
        result.canonical_version);
  }

  const bool has_registry = HasKnownPoseWrites(payload->known_pose_registry);
  PreparedKnownPoseRegistryCommit prepared_registry;
  if (has_registry) {
    const KnownPoseRegistryResult registry_result =
        known_pose_registry->PrepareCommit(payload->known_pose_registry,
                                           &prepared_registry);
    if (!registry_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::COMPONENT_PREPARATION_FAILED,
          registry_result.detail,
          result.canonical_version);
    }
  }

  const bool has_graph = HasGraphWrites(payload->active_covisibility_graph);
  PreparedActiveCovisibilityCommit prepared_graph;
  if (has_graph) {
    const ActiveCovisibilityResult graph_result =
        active_covisibility_graph->PrepareCommit(
            payload->active_covisibility_graph, &prepared_graph);
    if (!graph_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::COMPONENT_PREPARATION_FAILED,
          graph_result.detail,
          result.canonical_version);
    }
  }

  const bool has_owners = HasOwnerWrites(payload->point3D_owner_table);
  PreparedPoint3DOwnerTableCommit prepared_owners;
  if (has_owners) {
    const Point3DOwnerTableResult owner_result =
        point3D_owner_table->PrepareCommit(payload->point3D_owner_table,
                                           &prepared_owners);
    if (!owner_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::COMPONENT_PREPARATION_FAILED,
          owner_result.detail,
          result.canonical_version);
    }
  }

  result = ValidateOnlineMapperCanonicalVersion(
      reconstruction,
      known_pose_registry,
      active_covisibility_graph,
      point3D_owner_table,
      payload->expected_version);
  if (!result.IsSuccess()) {
    return result;
  }

  prepared->reconstruction_state_ = reconstruction;
  prepared->known_pose_registry_state_ = known_pose_registry;
  prepared->active_covisibility_graph_state_ = active_covisibility_graph;
  prepared->point3D_owner_table_state_ = point3D_owner_table;
  prepared->expected_version_ = payload->expected_version;
  prepared->reconstruction_ = std::move(payload->reconstruction);
  prepared->known_pose_registry_ = std::move(prepared_registry);
  prepared->active_covisibility_graph_ = std::move(prepared_graph);
  prepared->point3D_owner_table_ = std::move(prepared_owners);
  prepared->has_known_pose_registry_commit_ = has_registry;
  prepared->has_active_covisibility_graph_commit_ = has_graph;
  prepared->has_point3D_owner_table_commit_ = has_owners;
  prepared->prepared_ = true;
  return result;
}

OnlineMapperTransactionResult ValidatePreparedOnlineMapperTransaction(
    const OnlineMapperPreparedTransaction& prepared) {
  if (!prepared.prepared_ || prepared.reconstruction_state_ == nullptr ||
      prepared.known_pose_registry_state_ == nullptr ||
      prepared.active_covisibility_graph_state_ == nullptr ||
      prepared.point3D_owner_table_state_ == nullptr) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PREPARED_TRANSACTION,
        "online mapper transaction is not prepared",
        OnlineMapperCanonicalVersion());
  }
  OnlineMapperTransactionResult result = ValidateOnlineMapperCanonicalVersion(
      prepared.reconstruction_state_,
      prepared.known_pose_registry_state_,
      prepared.active_covisibility_graph_state_,
      prepared.point3D_owner_table_state_,
      prepared.expected_version_);
  if (!result.IsSuccess()) {
    return result;
  }
  const ReconstructionTransactionResult reconstruction_result =
      prepared.reconstruction_state_->ValidatePreparedTransaction(
          prepared.reconstruction_);
  if (!reconstruction_result.IsSuccess()) {
    return MakeOnlineMapperTransactionResult(
        OnlineMapperTransactionStatus::INVALID_PREPARED_TRANSACTION,
        reconstruction_result.detail,
        result.canonical_version);
  }
  if (prepared.has_known_pose_registry_commit_) {
    const KnownPoseRegistryResult registry_result =
        prepared.known_pose_registry_state_->ValidatePreparedCommit(
            prepared.known_pose_registry_);
    if (!registry_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::INVALID_PREPARED_TRANSACTION,
          registry_result.detail,
          result.canonical_version);
    }
  }
  if (prepared.has_active_covisibility_graph_commit_) {
    const ActiveCovisibilityResult graph_result =
        prepared.active_covisibility_graph_state_->ValidatePreparedCommit(
            prepared.active_covisibility_graph_);
    if (!graph_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::INVALID_PREPARED_TRANSACTION,
          graph_result.detail,
          result.canonical_version);
    }
  }
  if (prepared.has_point3D_owner_table_commit_) {
    const Point3DOwnerTableResult owner_result =
        prepared.point3D_owner_table_state_->ValidatePreparedCommit(
            prepared.point3D_owner_table_);
    if (!owner_result.IsSuccess()) {
      return MakeOnlineMapperTransactionResult(
          OnlineMapperTransactionStatus::INVALID_PREPARED_TRANSACTION,
          owner_result.detail,
          result.canonical_version);
    }
  }
  return result;
}

void CommitPreparedOnlineMapperTransaction(
    OnlineMapperPreparedTransaction* prepared) noexcept {
  prepared->reconstruction_state_->CommitPreparedTransaction(
      &prepared->reconstruction_);
  if (prepared->has_known_pose_registry_commit_) {
    prepared->known_pose_registry_state_->CommitPrepared(
        &prepared->known_pose_registry_);
  }
  if (prepared->has_active_covisibility_graph_commit_) {
    prepared->active_covisibility_graph_state_->CommitPrepared(
        &prepared->active_covisibility_graph_);
  }
  if (prepared->has_point3D_owner_table_commit_) {
    prepared->point3D_owner_table_state_->CommitPrepared(
        &prepared->point3D_owner_table_);
  }
  prepared->reconstruction_state_ = nullptr;
  prepared->known_pose_registry_state_ = nullptr;
  prepared->active_covisibility_graph_state_ = nullptr;
  prepared->point3D_owner_table_state_ = nullptr;
  prepared->has_known_pose_registry_commit_ = false;
  prepared->has_active_covisibility_graph_commit_ = false;
  prepared->has_point3D_owner_table_commit_ = false;
  prepared->prepared_ = false;
}

}  // namespace colmap
