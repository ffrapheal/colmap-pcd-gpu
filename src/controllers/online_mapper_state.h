#ifndef COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_STATE_H_
#define COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_STATE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "base/reconstruction.h"
#include "sfm/active_covisibility_graph.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

struct KnownPoseSE3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KnownPoseSE3();
  KnownPoseSE3(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec);

  // World-to-camera quaternion coefficients are ordered as (w, x, y, z).
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
};

enum class KnownPoseVisualState {
  POSE_ONLY,
  VISUAL_ACTIVE,
};

struct CatchupFailureEvidence {
  CatchupFailureEvidence() = default;
  CatchupFailureEvidence(
      size_t failed_active_edge_evidence_version,
      size_t failed_trigger_actual_valid_lidar_residual_count,
      size_t failed_lidar_map_version);

  size_t failed_active_edge_evidence_version = 0;
  size_t failed_trigger_actual_valid_lidar_residual_count = 0;
  size_t failed_lidar_map_version = 0;
};

struct KnownPoseRecord {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  image_t image_id = kInvalidImageId;
  size_t frame_index = std::numeric_limits<size_t>::max();
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  KnownPoseSE3 fastlio_T_cw;
  KnownPoseSE3 latest_T_cw;
  KnownPoseVisualState visual_state = KnownPoseVisualState::POSE_ONLY;
  bool has_catchup_failure_evidence = false;
  CatchupFailureEvidence last_catchup_failure;
};

enum class KnownPoseRegistryStatus {
  NOT_PROCESSED,
  SUCCESS,
  WRONG_THREAD,
  INVALID_IMAGE_ID,
  INVALID_FRAME_INDEX,
  NONCONTIGUOUS_FRAME_INDEX,
  DUPLICATE_IMAGE_ID,
  DUPLICATE_FRAME_INDEX,
  INVALID_REGISTRATION_SEQUENCE,
  REGISTRATION_SEQUENCE_NOT_FOUND,
  REGISTRATION_SEQUENCE_EXHAUSTED,
  INVALID_POSE,
  IMAGE_NOT_FOUND,
  REQUIRES_POSE_ONLY,
  REQUIRES_VISUAL_ACTIVE,
  ALREADY_VISUAL_ACTIVE,
  STALE_CANONICAL_VERSION,
  CANONICAL_VERSION_EXHAUSTED,
  EMPTY_COMMIT,
  DUPLICATE_WRITE,
  INVALID_PREPARED_COMMIT,
};

struct KnownPoseRegistryResult {
  KnownPoseRegistryStatus status = KnownPoseRegistryStatus::NOT_PROCESSED;
  std::string detail;
  size_t registration_sequence = std::numeric_limits<size_t>::max();
  uint64_t canonical_version = 0;

  bool IsSuccess() const { return status == KnownPoseRegistryStatus::SUCCESS; }
};

struct KnownPosePromotion {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KnownPosePromotion() = default;
  KnownPosePromotion(image_t image_id, const KnownPoseSE3& latest_T_cw);

  image_t image_id = kInvalidImageId;
  KnownPoseSE3 latest_T_cw;
};

struct KnownPoseLatestPoseUpdate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KnownPoseLatestPoseUpdate() = default;
  KnownPoseLatestPoseUpdate(image_t image_id,
                            const KnownPoseSE3& latest_T_cw);

  image_t image_id = kInvalidImageId;
  KnownPoseSE3 latest_T_cw;
};

struct KnownPoseCatchupFailureUpdate {
  KnownPoseCatchupFailureUpdate() = default;
  KnownPoseCatchupFailureUpdate(image_t image_id,
                                const CatchupFailureEvidence& evidence);

  image_t image_id = kInvalidImageId;
  CatchupFailureEvidence evidence;
};

struct KnownPoseRegistryCommit {
  uint64_t expected_version = 0;
  std::vector<KnownPosePromotion,
              Eigen::aligned_allocator<KnownPosePromotion>>
      promotions;
  std::vector<KnownPoseLatestPoseUpdate,
              Eigen::aligned_allocator<KnownPoseLatestPoseUpdate>>
      latest_pose_updates;
  std::vector<KnownPoseCatchupFailureUpdate> catchup_failures;
};

class KnownPoseRegistry;

class PreparedKnownPoseRegistryCommit {
 public:
  PreparedKnownPoseRegistryCommit() = default;
  PreparedKnownPoseRegistryCommit(
      PreparedKnownPoseRegistryCommit&& other);
  PreparedKnownPoseRegistryCommit& operator=(
      PreparedKnownPoseRegistryCommit&& other);

  PreparedKnownPoseRegistryCommit(
      const PreparedKnownPoseRegistryCommit&) = delete;
  PreparedKnownPoseRegistryCommit& operator=(
      const PreparedKnownPoseRegistryCommit&) = delete;

  bool IsPrepared() const noexcept;

 private:
  friend class KnownPoseRegistry;

  void Reset() noexcept;

  const KnownPoseRegistry* registry_ = nullptr;
  std::thread::id owner_thread_id_;
  uint64_t expected_version_ = 0;
  KnownPoseRegistryCommit normalized_commit_;
  bool prepared_ = false;
};

struct KnownPoseRecordQueryResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KnownPoseRegistryStatus status = KnownPoseRegistryStatus::NOT_PROCESSED;
  std::string detail;
  bool has_record = false;
  KnownPoseRecord record;

  bool IsSuccess() const {
    return status == KnownPoseRegistryStatus::SUCCESS && has_record;
  }
};

struct KnownPoseImageIdsResult {
  KnownPoseRegistryStatus status = KnownPoseRegistryStatus::NOT_PROCESSED;
  std::string detail;
  std::vector<image_t> image_ids;

  bool IsSuccess() const { return status == KnownPoseRegistryStatus::SUCCESS; }
};

// Synchronous state owned by the thread that constructs it. Every operation,
// including reads, rejects calls from another thread without touching records.
// Frame indices are unique, contiguous, and one-based. Registration sequences
// are allocated on successful insertion only and are also one-based.
class KnownPoseRegistry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KnownPoseRegistry();

  KnownPoseRegistry(const KnownPoseRegistry&) = delete;
  KnownPoseRegistry& operator=(const KnownPoseRegistry&) = delete;
  KnownPoseRegistry(KnownPoseRegistry&&) = delete;
  KnownPoseRegistry& operator=(KnownPoseRegistry&&) = delete;

  uint64_t Version() const noexcept;
  bool IsOwnerThread() const noexcept;

  // A finite qvec with Euclidean norm greater than double epsilon is divided
  // by that norm before storage. Non-finite translation values are rejected.
  KnownPoseRegistryResult AddKnownPose(image_t image_id,
                                       size_t frame_index,
                                       const KnownPoseSE3& fastlio_T_cw);

  // Promotion atomically stores the first visual pose and changes the state.
  // This preserves latest_T_cw == fastlio_T_cw for every POSE_ONLY record.
  KnownPoseRegistryResult PromoteToVisualActive(
      image_t image_id, const KnownPoseSE3& latest_T_cw);

  // Only an already active record may receive subsequent optimized poses.
  KnownPoseRegistryResult UpdateLatestPose(
      image_t image_id, const KnownPoseSE3& latest_T_cw);

  KnownPoseRegistryResult RecordCatchupFailure(
      image_t image_id, const CatchupFailureEvidence& evidence);

  KnownPoseRegistryResult ValidateCommit(
      const KnownPoseRegistryCommit& commit) const;
  KnownPoseRegistryResult PrepareCommit(
      const KnownPoseRegistryCommit& commit,
      PreparedKnownPoseRegistryCommit* prepared) const;
  KnownPoseRegistryResult ValidatePreparedCommit(
      const PreparedKnownPoseRegistryCommit& prepared) const;
  KnownPoseRegistryResult Commit(const KnownPoseRegistryCommit& commit);

  // Precondition: ValidatePreparedCommit succeeded and no owner-thread
  // operation ran afterwards. Applying normalized record fields cannot throw.
  void CommitPrepared(PreparedKnownPoseRegistryCommit* prepared) noexcept;

  // Record queries return copies so callers cannot mutate registry state.
  KnownPoseRecordQueryResult GetByImageId(image_t image_id) const;
  KnownPoseRecordQueryResult GetByRegistrationSequence(
      size_t registration_sequence) const;

  // Every ID list is ordered by registration_sequence, then image_id.
  KnownPoseImageIdsResult GetRegisteredImageIds() const;
  KnownPoseImageIdsResult GetPoseOnlyImageIds() const;
  KnownPoseImageIdsResult GetVisualActiveImageIds() const;

 private:
  using RecordVector =
      std::vector<KnownPoseRecord, Eigen::aligned_allocator<KnownPoseRecord>>;

  bool FindRecordIndex(image_t image_id, size_t* record_index) const;
  KnownPoseRegistryResult Reject(KnownPoseRegistryStatus status,
                                 const std::string& detail) const;
  KnownPoseRecordQueryResult RejectRecordQuery(
      KnownPoseRegistryStatus status, const std::string& detail) const;
  KnownPoseImageIdsResult RejectImageIdsQuery(
      KnownPoseRegistryStatus status, const std::string& detail) const;
  KnownPoseImageIdsResult GetImageIds(KnownPoseVisualState state,
                                      bool filter_by_state) const;

  const std::thread::id owner_thread_id_;
  size_t next_frame_index_ = 1;
  size_t next_registration_sequence_ = 1;
  uint64_t version_ = 0;
  RecordVector records_;
  std::unordered_map<image_t, size_t> image_id_to_record_index_;
};

struct Point3DOwnerInput {
  Point3DOwnerInput() = default;
  Point3DOwnerInput(point3D_t point3D_id,
                    image_t image_id,
                    point2D_t point2D_idx,
                    uint64_t association_identity);

  point3D_t point3D_id = kInvalidPoint3DId;
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;
  uint64_t association_identity = 0;
};

struct Point3DOwner {
  point3D_t point3D_id = kInvalidPoint3DId;
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;
  uint64_t association_identity = 0;
  uint64_t creation_sequence = 0;
};

enum class Point3DOwnerTableStatus {
  SUCCESS,
  WRONG_THREAD,
  INVALID_POINT3D_ID,
  INVALID_IMAGE_ID,
  INVALID_POINT2D_INDEX,
  INVALID_ASSOCIATION_IDENTITY,
  DUPLICATE_POINT3D_ID,
  DUPLICATE_OWNER_OBSERVATION,
  DUPLICATE_ASSOCIATION_IDENTITY,
  STALE_CANONICAL_VERSION,
  CANONICAL_VERSION_EXHAUSTED,
  CREATION_SEQUENCE_EXHAUSTED,
  EMPTY_COMMIT,
  OWNER_NOT_FOUND,
  INVALID_PREPARED_COMMIT,
};

struct Point3DOwnerTableResult {
  Point3DOwnerTableStatus status = Point3DOwnerTableStatus::SUCCESS;
  std::string detail;
  uint64_t canonical_version = 0;

  bool IsSuccess() const {
    return status == Point3DOwnerTableStatus::SUCCESS;
  }
};

struct Point3DOwnerQueryResult : Point3DOwnerTableResult {
  bool has_owner = false;
  Point3DOwner owner;

  bool IsSuccess() const {
    return Point3DOwnerTableResult::IsSuccess() && has_owner;
  }
};

struct Point3DOwnerTableSnapshot : Point3DOwnerTableResult {
  uint64_t next_creation_sequence = 1;
  std::vector<Point3DOwner> owners;
};

struct Point3DOwnerTableCommit {
  uint64_t expected_version = 0;
  std::vector<Point3DOwnerInput> owners;
};

class Point3DOwnerTable;

class PreparedPoint3DOwnerTableCommit {
 public:
  PreparedPoint3DOwnerTableCommit() = default;
  PreparedPoint3DOwnerTableCommit(
      PreparedPoint3DOwnerTableCommit&& other);
  PreparedPoint3DOwnerTableCommit& operator=(
      PreparedPoint3DOwnerTableCommit&& other);

  PreparedPoint3DOwnerTableCommit(
      const PreparedPoint3DOwnerTableCommit&) = delete;
  PreparedPoint3DOwnerTableCommit& operator=(
      const PreparedPoint3DOwnerTableCommit&) = delete;

  bool IsPrepared() const noexcept;

 private:
  friend class Point3DOwnerTable;

  using ObservationKey = std::pair<image_t, point2D_t>;

  void Reset() noexcept;

  const Point3DOwnerTable* table_ = nullptr;
  std::thread::id owner_thread_id_;
  uint64_t expected_version_ = 0;
  uint64_t committed_version_ = 0;
  uint64_t next_creation_sequence_ = 1;
  bool prepared_ = false;
  std::map<point3D_t, Point3DOwner> owners_;
  std::map<ObservationKey, point3D_t> observation_to_point3D_;
  std::map<uint64_t, point3D_t> association_to_point3D_;
};

// Stable lineage for online-created visual points. Owners are append-only:
// point IDs, owner observations, association identities, and creation
// sequences are each globally unique within one table lifetime.
class Point3DOwnerTable {
 public:
  Point3DOwnerTable();

  Point3DOwnerTable(const Point3DOwnerTable&) = delete;
  Point3DOwnerTable& operator=(const Point3DOwnerTable&) = delete;
  Point3DOwnerTable(Point3DOwnerTable&&) = delete;
  Point3DOwnerTable& operator=(Point3DOwnerTable&&) = delete;

  uint64_t Version() const noexcept;
  bool IsOwnerThread() const noexcept;
  size_t Size() const noexcept;

  Point3DOwnerQueryResult Get(point3D_t point3D_id) const;
  Point3DOwnerTableSnapshot Snapshot() const;
  Point3DOwnerTableResult Add(const Point3DOwnerInput& owner);
  Point3DOwnerTableResult ValidateCommit(
      const Point3DOwnerTableCommit& commit) const;
  Point3DOwnerTableResult PrepareCommit(
      const Point3DOwnerTableCommit& commit,
      PreparedPoint3DOwnerTableCommit* prepared) const;
  Point3DOwnerTableResult ValidatePreparedCommit(
      const PreparedPoint3DOwnerTableCommit& prepared) const;
  Point3DOwnerTableResult Commit(const Point3DOwnerTableCommit& commit);

  // Precondition: ValidatePreparedCommit succeeded and no owner-thread
  // operation ran afterwards. Publication only swaps prepared maps.
  void CommitPrepared(PreparedPoint3DOwnerTableCommit* prepared) noexcept;

 private:
  using ObservationKey = std::pair<image_t, point2D_t>;

  Point3DOwnerTableResult Reject(Point3DOwnerTableStatus status,
                                 const std::string& detail) const;

  const std::thread::id owner_thread_id_;
  uint64_t version_ = 0;
  uint64_t next_creation_sequence_ = 1;
  std::map<point3D_t, Point3DOwner> owners_;
  std::map<ObservationKey, point3D_t> observation_to_point3D_;
  std::map<uint64_t, point3D_t> association_to_point3D_;
};

struct OnlineMapperCanonicalVersion {
  ReconstructionCanonicalVersion reconstruction;
  uint64_t known_pose_registry = 0;
  uint64_t active_covisibility_graph = 0;
  uint64_t point3D_owner_table = 0;

  bool operator==(const OnlineMapperCanonicalVersion& other) const;
  bool operator!=(const OnlineMapperCanonicalVersion& other) const;
};

enum class OnlineMapperTransactionStatus {
  SUCCESS,
  WRONG_THREAD,
  NULL_STATE,
  INVALID_PAYLOAD,
  STALE_CANONICAL_VERSION,
  COMPONENT_PREPARATION_FAILED,
  INVALID_PREPARED_TRANSACTION,
};

struct OnlineMapperTransactionResult {
  OnlineMapperTransactionStatus status =
      OnlineMapperTransactionStatus::INVALID_PAYLOAD;
  std::string detail;
  OnlineMapperCanonicalVersion canonical_version;

  bool IsSuccess() const {
    return status == OnlineMapperTransactionStatus::SUCCESS;
  }
};

struct OnlineMapperTransactionPayload {
  OnlineMapperCanonicalVersion expected_version;
  ReconstructionTransaction reconstruction;
  KnownPoseRegistryCommit known_pose_registry;
  ActiveCovisibilityCommit active_covisibility_graph;
  Point3DOwnerTableCommit point3D_owner_table;
};

class OnlineMapperPreparedTransaction {
 public:
  OnlineMapperPreparedTransaction() = default;
  OnlineMapperPreparedTransaction(OnlineMapperPreparedTransaction&& other);
  OnlineMapperPreparedTransaction& operator=(
      OnlineMapperPreparedTransaction&& other);

  OnlineMapperPreparedTransaction(
      const OnlineMapperPreparedTransaction&) = delete;
  OnlineMapperPreparedTransaction& operator=(
      const OnlineMapperPreparedTransaction&) = delete;

  bool IsPrepared() const noexcept;

 private:
  friend OnlineMapperTransactionResult PrepareOnlineMapperTransaction(
      Reconstruction*,
      KnownPoseRegistry*,
      ActiveCovisibilityGraph*,
      Point3DOwnerTable*,
      OnlineMapperTransactionPayload*,
      OnlineMapperPreparedTransaction*);
  friend OnlineMapperTransactionResult ValidatePreparedOnlineMapperTransaction(
      const OnlineMapperPreparedTransaction&);
  friend void CommitPreparedOnlineMapperTransaction(
      OnlineMapperPreparedTransaction*) noexcept;

  void Reset();

  Reconstruction* reconstruction_state_ = nullptr;
  KnownPoseRegistry* known_pose_registry_state_ = nullptr;
  ActiveCovisibilityGraph* active_covisibility_graph_state_ = nullptr;
  Point3DOwnerTable* point3D_owner_table_state_ = nullptr;
  OnlineMapperCanonicalVersion expected_version_;
  ReconstructionTransaction reconstruction_;
  PreparedKnownPoseRegistryCommit known_pose_registry_;
  PreparedActiveCovisibilityCommit active_covisibility_graph_;
  PreparedPoint3DOwnerTableCommit point3D_owner_table_;
  bool has_known_pose_registry_commit_ = false;
  bool has_active_covisibility_graph_commit_ = false;
  bool has_point3D_owner_table_commit_ = false;
  bool prepared_ = false;
};

OnlineMapperTransactionResult CaptureOnlineMapperCanonicalVersion(
    const Reconstruction* reconstruction,
    const KnownPoseRegistry* known_pose_registry,
    const ActiveCovisibilityGraph* active_covisibility_graph,
    const Point3DOwnerTable* point3D_owner_table);

OnlineMapperTransactionResult ValidateOnlineMapperCanonicalVersion(
    const Reconstruction* reconstruction,
    const KnownPoseRegistry* known_pose_registry,
    const ActiveCovisibilityGraph* active_covisibility_graph,
    const Point3DOwnerTable* point3D_owner_table,
    const OnlineMapperCanonicalVersion& expected_version);

OnlineMapperTransactionResult BeginOnlineMapperTransaction(
    Reconstruction* reconstruction,
    KnownPoseRegistry* known_pose_registry,
    ActiveCovisibilityGraph* active_covisibility_graph,
    Point3DOwnerTable* point3D_owner_table,
    OnlineMapperTransactionPayload* payload);

OnlineMapperTransactionResult PrepareOnlineMapperTransaction(
    Reconstruction* reconstruction,
    KnownPoseRegistry* known_pose_registry,
    ActiveCovisibilityGraph* active_covisibility_graph,
    Point3DOwnerTable* point3D_owner_table,
    OnlineMapperTransactionPayload* payload,
    OnlineMapperPreparedTransaction* prepared);

OnlineMapperTransactionResult ValidatePreparedOnlineMapperTransaction(
    const OnlineMapperPreparedTransaction& prepared);

// After ValidatePreparedOnlineMapperTransaction succeeds on the owner thread,
// this publishes all four components without allocation or a fallible branch.
void CommitPreparedOnlineMapperTransaction(
    OnlineMapperPreparedTransaction* prepared) noexcept;

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_ONLINE_MAPPER_STATE_H_
