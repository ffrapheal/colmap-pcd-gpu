#define TEST_NAME "controllers/online_mapper_state_test"
#include "util/testing.h"

#include "controllers/online_mapper_state.h"

#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "base/correspondence_graph.h"
#include "base/database.h"

using namespace colmap;

namespace {

KnownPoseSE3 IdentityPose(const Eigen::Vector3d& tvec) {
  return KnownPoseSE3(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), tvec);
}

KnownPoseSE3 IdentityPose() { return IdentityPose(Eigen::Vector3d::Zero()); }

void CheckImageIds(const KnownPoseImageIdsResult& result,
                   const std::vector<image_t>& expected) {
  BOOST_REQUIRE(result.IsSuccess());
  BOOST_REQUIRE_EQUAL(result.image_ids.size(), expected.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(result.image_ids.begin(),
                                result.image_ids.end(), expected.begin(),
                                expected.end());
}

void CheckPose(const KnownPoseSE3& pose,
               const Eigen::Vector4d& expected_qvec,
               const Eigen::Vector3d& expected_tvec) {
  BOOST_CHECK(pose.qvec.isApprox(expected_qvec));
  BOOST_CHECK(pose.tvec.isApprox(expected_tvec));
}

void CheckPoseExactly(const KnownPoseSE3& pose,
                      const KnownPoseSE3& expected) {
  BOOST_CHECK_EQUAL_COLLECTIONS(
      pose.qvec.data(), pose.qvec.data() + pose.qvec.size(),
      expected.qvec.data(), expected.qvec.data() + expected.qvec.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      pose.tvec.data(), pose.tvec.data() + pose.tvec.size(),
      expected.tvec.data(), expected.tvec.data() + expected.tvec.size());
}

void CheckStateAndPosesUnchanged(const KnownPoseRecord& record,
                                 const KnownPoseRecord& expected) {
  BOOST_CHECK(record.visual_state == expected.visual_state);
  CheckPoseExactly(record.fastlio_T_cw, expected.fastlio_T_cw);
  CheckPoseExactly(record.latest_T_cw, expected.latest_T_cw);
}

void CheckRecordExactly(const KnownPoseRecord& record,
                        const KnownPoseRecord& expected) {
  BOOST_CHECK_EQUAL(record.image_id, expected.image_id);
  BOOST_CHECK_EQUAL(record.frame_index, expected.frame_index);
  BOOST_CHECK_EQUAL(record.registration_sequence,
                    expected.registration_sequence);
  CheckStateAndPosesUnchanged(record, expected);
  BOOST_CHECK_EQUAL(record.has_catchup_failure_evidence,
                    expected.has_catchup_failure_evidence);
  BOOST_CHECK_EQUAL(
      record.last_catchup_failure.failed_active_edge_evidence_version,
      expected.last_catchup_failure.failed_active_edge_evidence_version);
  BOOST_CHECK_EQUAL(
      record.last_catchup_failure
          .failed_trigger_actual_valid_lidar_residual_count,
      expected.last_catchup_failure
          .failed_trigger_actual_valid_lidar_residual_count);
  BOOST_CHECK_EQUAL(
      record.last_catchup_failure.failed_lidar_map_version,
      expected.last_catchup_failure.failed_lidar_map_version);
}

void SetUpTransactionalReconstruction(
    Reconstruction* reconstruction,
    CorrespondenceGraph* correspondence_graph,
    const uint64_t owner_epoch) {
  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("PINHOLE", 10, 10, 10);
  reconstruction->AddCamera(camera);

  Image image;
  image.SetImageId(1);
  image.SetCameraId(1);
  image.SetName("online_state_image");
  image.SetPoints2D(
      std::vector<Eigen::Vector2d>(2, Eigen::Vector2d::Zero()));
  reconstruction->AddImage(image);
  reconstruction->RegisterImage(1);
  correspondence_graph->AddImage(1, 2);
  reconstruction->SetUp(correspondence_graph);
  reconstruction->BeginStructureJournal(owner_epoch, 32);
}

}  // namespace

static_assert(
    noexcept(CommitPreparedOnlineMapperTransaction(
        std::declval<OnlineMapperPreparedTransaction*>())),
    "coordinated canonical publication must be noexcept");
static_assert(
    noexcept(std::declval<KnownPoseRegistry&>().CommitPrepared(
        std::declval<PreparedKnownPoseRegistryCommit*>())),
    "prepared registry publication must be noexcept");
static_assert(
    noexcept(std::declval<Point3DOwnerTable&>().CommitPrepared(
        std::declval<PreparedPoint3DOwnerTableCommit*>())),
    "prepared owner-table publication must be noexcept");

BOOST_AUTO_TEST_CASE(FirstFrameRegistersPoseOnlyAndNormalizesPose) {
  KnownPoseRegistry registry;
  const KnownPoseSE3 raw_pose(Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
                              Eigen::Vector3d(1.0, 2.0, 3.0));

  const KnownPoseRegistryResult add_result =
      registry.AddKnownPose(17, 1, raw_pose);
  BOOST_REQUIRE(add_result.IsSuccess());
  BOOST_CHECK_EQUAL(add_result.registration_sequence, 1);

  const KnownPoseRecordQueryResult by_image = registry.GetByImageId(17);
  BOOST_REQUIRE(by_image.IsSuccess());
  BOOST_CHECK_EQUAL(by_image.record.frame_index, 1);
  BOOST_CHECK_EQUAL(by_image.record.registration_sequence, 1);
  BOOST_CHECK(by_image.record.visual_state == KnownPoseVisualState::POSE_ONLY);
  BOOST_CHECK(!by_image.record.has_catchup_failure_evidence);
  CheckPose(by_image.record.fastlio_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 2.0, 3.0));
  CheckPose(by_image.record.latest_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 2.0, 3.0));

  const KnownPoseRecordQueryResult by_sequence =
      registry.GetByRegistrationSequence(1);
  BOOST_REQUIRE(by_sequence.IsSuccess());
  BOOST_CHECK_EQUAL(by_sequence.record.image_id, 17);
}

BOOST_AUTO_TEST_CASE(RegistrationEnforcesOrderDuplicatesAndUniqueSequences) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(30, 1, IdentityPose()).IsSuccess());

  BOOST_CHECK(registry.AddKnownPose(30, 2, IdentityPose()).status ==
              KnownPoseRegistryStatus::DUPLICATE_IMAGE_ID);
  BOOST_CHECK(registry.AddKnownPose(10, 1, IdentityPose()).status ==
              KnownPoseRegistryStatus::DUPLICATE_FRAME_INDEX);
  BOOST_CHECK(registry.AddKnownPose(10, 3, IdentityPose()).status ==
              KnownPoseRegistryStatus::NONCONTIGUOUS_FRAME_INDEX);

  const KnownPoseRegistryResult second =
      registry.AddKnownPose(10, 2, IdentityPose());
  BOOST_REQUIRE(second.IsSuccess());
  BOOST_CHECK_EQUAL(second.registration_sequence, 2);

  const KnownPoseRecordQueryResult second_by_sequence =
      registry.GetByRegistrationSequence(2);
  BOOST_REQUIRE(second_by_sequence.IsSuccess());
  BOOST_CHECK_EQUAL(second_by_sequence.record.image_id, 10);
  BOOST_CHECK(registry.GetByRegistrationSequence(0).status ==
              KnownPoseRegistryStatus::INVALID_REGISTRATION_SEQUENCE);
  BOOST_CHECK(registry.GetByRegistrationSequence(3).status ==
              KnownPoseRegistryStatus::REGISTRATION_SEQUENCE_NOT_FOUND);
}

BOOST_AUTO_TEST_CASE(InvalidIdentifiersAndTransformsDoNotConsumeSequence) {
  KnownPoseRegistry registry;
  const image_t database_limit =
      static_cast<image_t>(Database::kMaxNumImages);
  BOOST_CHECK(
      registry.AddKnownPose(kInvalidImageId, 1, IdentityPose()).status ==
      KnownPoseRegistryStatus::INVALID_IMAGE_ID);
  BOOST_CHECK(registry.AddKnownPose(database_limit, 1, IdentityPose()).status ==
              KnownPoseRegistryStatus::INVALID_IMAGE_ID);
  BOOST_CHECK(registry.AddKnownPose(1, 0, IdentityPose()).status ==
              KnownPoseRegistryStatus::INVALID_FRAME_INDEX);
  BOOST_CHECK(registry.AddKnownPose(1, 2, IdentityPose()).status ==
              KnownPoseRegistryStatus::NONCONTIGUOUS_FRAME_INDEX);

  const KnownPoseSE3 zero_quaternion(Eigen::Vector4d::Zero(),
                                     Eigen::Vector3d::Zero());
  BOOST_CHECK(registry.AddKnownPose(1, 1, zero_quaternion).status ==
              KnownPoseRegistryStatus::INVALID_POSE);

  const KnownPoseSE3 nonfinite_translation(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
      Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0, 0.0));
  BOOST_CHECK(registry.AddKnownPose(1, 1, nonfinite_translation).status ==
              KnownPoseRegistryStatus::INVALID_POSE);

  const KnownPoseSE3 nonfinite_quaternion(
      Eigen::Vector4d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0,
                      0.0),
      Eigen::Vector3d::Zero());
  BOOST_CHECK(registry.AddKnownPose(1, 1, nonfinite_quaternion).status ==
              KnownPoseRegistryStatus::INVALID_POSE);

  const KnownPoseRegistryResult valid =
      registry.AddKnownPose(1, 1, IdentityPose());
  BOOST_REQUIRE(valid.IsSuccess());
  BOOST_CHECK_EQUAL(valid.registration_sequence, 1);
  CheckImageIds(registry.GetRegisteredImageIds(), {1});
}

BOOST_AUTO_TEST_CASE(FastlioPoseIsImmutableAndLatestPoseRequiresActiveState) {
  KnownPoseRegistry registry;
  const KnownPoseSE3 fastlio_pose(
      Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
      Eigen::Vector3d(1.0, 2.0, 3.0));
  BOOST_REQUIRE(registry.AddKnownPose(8, 1, fastlio_pose).IsSuccess());

  const KnownPoseSE3 first_visual_pose(
      Eigen::Vector4d(0.0, 0.0, 0.0, 2.0),
      Eigen::Vector3d(4.0, 5.0, 6.0));
  const KnownPoseRecordQueryResult before_rejected_update =
      registry.GetByImageId(8);
  BOOST_REQUIRE(before_rejected_update.IsSuccess());
  BOOST_CHECK(registry.UpdateLatestPose(8, first_visual_pose).status ==
              KnownPoseRegistryStatus::REQUIRES_VISUAL_ACTIVE);
  const KnownPoseRecordQueryResult after_rejected_update =
      registry.GetByImageId(8);
  BOOST_REQUIRE(after_rejected_update.IsSuccess());
  BOOST_REQUIRE(after_rejected_update.record.visual_state ==
                KnownPoseVisualState::POSE_ONLY);
  CheckPoseExactly(after_rejected_update.record.fastlio_T_cw,
                   before_rejected_update.record.fastlio_T_cw);
  CheckPoseExactly(after_rejected_update.record.latest_T_cw,
                   before_rejected_update.record.latest_T_cw);
  BOOST_REQUIRE(
      registry.PromoteToVisualActive(8, first_visual_pose).IsSuccess());

  KnownPoseRecordQueryResult record = registry.GetByImageId(8);
  BOOST_REQUIRE(record.IsSuccess());
  BOOST_CHECK(record.record.visual_state ==
              KnownPoseVisualState::VISUAL_ACTIVE);
  CheckPose(record.record.fastlio_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 2.0, 3.0));
  CheckPose(record.record.latest_T_cw,
            Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
            Eigen::Vector3d(4.0, 5.0, 6.0));

  const KnownPoseSE3 second_visual_pose(
      Eigen::Vector4d(0.0, 3.0, 0.0, 0.0),
      Eigen::Vector3d(7.0, 8.0, 9.0));
  BOOST_REQUIRE(registry.UpdateLatestPose(8, second_visual_pose).IsSuccess());
  record = registry.GetByImageId(8);
  BOOST_REQUIRE(record.IsSuccess());
  CheckPose(record.record.fastlio_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 2.0, 3.0));
  CheckPose(record.record.latest_T_cw,
            Eigen::Vector4d(0.0, 1.0, 0.0, 0.0),
            Eigen::Vector3d(7.0, 8.0, 9.0));

  const KnownPoseRegistryResult invalid_update = registry.UpdateLatestPose(
      8, KnownPoseSE3(Eigen::Vector4d::Zero(), Eigen::Vector3d::Zero()));
  BOOST_CHECK(invalid_update.status == KnownPoseRegistryStatus::INVALID_POSE);
  const KnownPoseRecordQueryResult after_invalid = registry.GetByImageId(8);
  BOOST_REQUIRE(after_invalid.IsSuccess());
  CheckPose(after_invalid.record.latest_T_cw,
            Eigen::Vector4d(0.0, 1.0, 0.0, 0.0),
            Eigen::Vector3d(7.0, 8.0, 9.0));
}

BOOST_AUTO_TEST_CASE(VisualStatePromotionIsOneWay) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(4, 1, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.RecordCatchupFailure(
                            4, CatchupFailureEvidence(3, 49, 7))
                    .IsSuccess());

  const KnownPoseRecordQueryResult before_invalid_promotion =
      registry.GetByImageId(4);
  BOOST_REQUIRE(before_invalid_promotion.IsSuccess());
  const KnownPoseRecord before_invalid_promotion_record =
      before_invalid_promotion.record;

  const KnownPoseSE3 invalid_pose(Eigen::Vector4d::Zero(),
                                  Eigen::Vector3d::Zero());
  BOOST_CHECK(registry.PromoteToVisualActive(4, invalid_pose).status ==
              KnownPoseRegistryStatus::INVALID_POSE);
  const KnownPoseRecordQueryResult after_invalid = registry.GetByImageId(4);
  BOOST_REQUIRE(after_invalid.IsSuccess());
  CheckRecordExactly(after_invalid.record, before_invalid_promotion_record);

  const KnownPoseSE3 accepted_pose =
      IdentityPose(Eigen::Vector3d(1.0, 0.0, 0.0));
  BOOST_REQUIRE(registry.PromoteToVisualActive(4, accepted_pose).IsSuccess());
  const KnownPoseRecordQueryResult before_repeated_promotion =
      registry.GetByImageId(4);
  BOOST_REQUIRE(before_repeated_promotion.IsSuccess());
  const KnownPoseRegistryResult repeated_promotion =
      registry.PromoteToVisualActive(
          4, IdentityPose(Eigen::Vector3d(9.0, 0.0, 0.0)));
  BOOST_CHECK(repeated_promotion.status ==
              KnownPoseRegistryStatus::ALREADY_VISUAL_ACTIVE);

  const KnownPoseRecordQueryResult record = registry.GetByImageId(4);
  BOOST_REQUIRE(record.IsSuccess());
  CheckStateAndPosesUnchanged(record.record,
                              before_repeated_promotion.record);
  CheckPose(record.record.latest_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 0.0, 0.0));
}

BOOST_AUTO_TEST_CASE(RecordQueriesReturnCopies) {
  KnownPoseRegistry registry;
  const KnownPoseSE3 fastlio_pose(
      Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
      Eigen::Vector3d(1.0, 2.0, 3.0));
  BOOST_REQUIRE(registry.AddKnownPose(12, 1, fastlio_pose).IsSuccess());
  const KnownPoseSE3 visual_pose(
      Eigen::Vector4d(0.0, 0.0, 0.0, 3.0),
      Eigen::Vector3d(4.0, 5.0, 6.0));
  BOOST_REQUIRE(registry.PromoteToVisualActive(12, visual_pose).IsSuccess());

  KnownPoseRecordQueryResult external_copy = registry.GetByImageId(12);
  BOOST_REQUIRE(external_copy.IsSuccess());
  const KnownPoseRecord expected_internal_record = external_copy.record;
  external_copy.record.visual_state = KnownPoseVisualState::POSE_ONLY;
  external_copy.record.fastlio_T_cw =
      IdentityPose(Eigen::Vector3d(7.0, 8.0, 9.0));
  external_copy.record.latest_T_cw = external_copy.record.fastlio_T_cw;

  const KnownPoseRecordQueryResult internal_record =
      registry.GetByImageId(12);
  BOOST_REQUIRE(internal_record.IsSuccess());
  CheckRecordExactly(internal_record.record, expected_internal_record);
}

BOOST_AUTO_TEST_CASE(RegistrationSequenceQueriesReturnCopies) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(13, 1, IdentityPose()).IsSuccess());

  KnownPoseRecordQueryResult external_copy =
      registry.GetByRegistrationSequence(1);
  BOOST_REQUIRE(external_copy.IsSuccess());
  const KnownPoseRecord expected_internal_record = external_copy.record;
  external_copy.record.fastlio_T_cw =
      IdentityPose(Eigen::Vector3d(7.0, 8.0, 9.0));
  external_copy.record.latest_T_cw = external_copy.record.fastlio_T_cw;
  external_copy.record.has_catchup_failure_evidence = true;
  external_copy.record.last_catchup_failure =
      CatchupFailureEvidence(4, 101, 9);

  const KnownPoseRecordQueryResult internal_record =
      registry.GetByRegistrationSequence(1);
  BOOST_REQUIRE(internal_record.IsSuccess());
  CheckRecordExactly(internal_record.record, expected_internal_record);
}

BOOST_AUTO_TEST_CASE(ImageIdListsAreDeterministicByRegistrationSequence) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(30, 1, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(10, 2, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(20, 3, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(
      registry.PromoteToVisualActive(10, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(
      registry.PromoteToVisualActive(20, IdentityPose()).IsSuccess());

  CheckImageIds(registry.GetRegisteredImageIds(), {30, 10, 20});
  CheckImageIds(registry.GetPoseOnlyImageIds(), {30});
  CheckImageIds(registry.GetVisualActiveImageIds(), {10, 20});
}

BOOST_AUTO_TEST_CASE(WrongThreadCallsAreRejectedWithoutMutation) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());

  KnownPoseRegistryResult add_result;
  KnownPoseRegistryResult promote_result;
  KnownPoseRegistryResult update_result;
  KnownPoseRegistryResult evidence_result;
  KnownPoseImageIdsResult ids_result;
  std::thread other_thread([&]() {
    add_result = registry.AddKnownPose(2, 2, IdentityPose());
    promote_result = registry.PromoteToVisualActive(1, IdentityPose());
    update_result = registry.UpdateLatestPose(
        1, IdentityPose(Eigen::Vector3d(1.0, 0.0, 0.0)));
    evidence_result = registry.RecordCatchupFailure(
        1, CatchupFailureEvidence(3, 49, 7));
    ids_result = registry.GetRegisteredImageIds();
  });
  other_thread.join();

  BOOST_CHECK(add_result.status == KnownPoseRegistryStatus::WRONG_THREAD);
  BOOST_CHECK(promote_result.status == KnownPoseRegistryStatus::WRONG_THREAD);
  BOOST_CHECK(update_result.status == KnownPoseRegistryStatus::WRONG_THREAD);
  BOOST_CHECK(evidence_result.status == KnownPoseRegistryStatus::WRONG_THREAD);
  BOOST_CHECK(ids_result.status == KnownPoseRegistryStatus::WRONG_THREAD);
  BOOST_CHECK(ids_result.image_ids.empty());

  const KnownPoseRecordQueryResult first = registry.GetByImageId(1);
  BOOST_REQUIRE(first.IsSuccess());
  BOOST_CHECK(first.record.visual_state == KnownPoseVisualState::POSE_ONLY);
  BOOST_CHECK(!first.record.has_catchup_failure_evidence);
  CheckPose(first.record.latest_T_cw,
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
            Eigen::Vector3d::Zero());

  const KnownPoseRegistryResult second =
      registry.AddKnownPose(2, 2, IdentityPose());
  BOOST_REQUIRE(second.IsSuccess());
  BOOST_CHECK_EQUAL(second.registration_sequence, 2);
}

BOOST_AUTO_TEST_CASE(CatchupFailureEvidenceKeepsMostRecentAttempt) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(5, 1, IdentityPose()).IsSuccess());

  BOOST_REQUIRE(registry.RecordCatchupFailure(
                            5, CatchupFailureEvidence(3, 49, 7))
                    .IsSuccess());
  KnownPoseRecordQueryResult record = registry.GetByImageId(5);
  BOOST_REQUIRE(record.IsSuccess());
  BOOST_REQUIRE(record.record.has_catchup_failure_evidence);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_active_edge_evidence_version,
      3);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_trigger_actual_valid_lidar_residual_count,
      49);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure.failed_lidar_map_version, 7);

  BOOST_REQUIRE(registry.RecordCatchupFailure(
                            5, CatchupFailureEvidence(4, 101, 9))
                    .IsSuccess());
  record = registry.GetByImageId(5);
  BOOST_REQUIRE(record.IsSuccess());
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_active_edge_evidence_version,
      4);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_trigger_actual_valid_lidar_residual_count,
      101);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure.failed_lidar_map_version, 9);

  BOOST_REQUIRE(
      registry.PromoteToVisualActive(5, IdentityPose()).IsSuccess());
  const KnownPoseRegistryResult active_failure = registry.RecordCatchupFailure(
      5, CatchupFailureEvidence(5, 151, 10));
  BOOST_CHECK(active_failure.status ==
              KnownPoseRegistryStatus::REQUIRES_POSE_ONLY);
  record = registry.GetByImageId(5);
  BOOST_REQUIRE(record.IsSuccess());
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_active_edge_evidence_version,
      4);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure
          .failed_trigger_actual_valid_lidar_residual_count,
      101);
  BOOST_CHECK_EQUAL(
      record.record.last_catchup_failure.failed_lidar_map_version, 9);
}

BOOST_AUTO_TEST_CASE(RegistryPreparedCommitIsAtomicAndVersioned) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(2, 2, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.PromoteToVisualActive(2, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(3, 3, IdentityPose()).IsSuccess());
  const uint64_t before_version = registry.Version();

  KnownPoseRegistryCommit commit;
  commit.expected_version = before_version;
  commit.promotions.emplace_back(
      1, IdentityPose(Eigen::Vector3d(1.0, 0.0, 0.0)));
  commit.latest_pose_updates.emplace_back(
      2, IdentityPose(Eigen::Vector3d(2.0, 0.0, 0.0)));
  commit.catchup_failures.emplace_back(
      3, CatchupFailureEvidence(7, 41, 11));
  PreparedKnownPoseRegistryCommit prepared;
  BOOST_REQUIRE(registry.PrepareCommit(commit, &prepared).IsSuccess());
  BOOST_REQUIRE(registry.ValidatePreparedCommit(prepared).IsSuccess());
  registry.CommitPrepared(&prepared);

  BOOST_CHECK(!prepared.IsPrepared());
  BOOST_CHECK_EQUAL(registry.Version(), before_version + 1);
  const KnownPoseRecordQueryResult first = registry.GetByImageId(1);
  const KnownPoseRecordQueryResult second = registry.GetByImageId(2);
  const KnownPoseRecordQueryResult third = registry.GetByImageId(3);
  BOOST_REQUIRE(first.IsSuccess());
  BOOST_REQUIRE(second.IsSuccess());
  BOOST_REQUIRE(third.IsSuccess());
  BOOST_CHECK(first.record.visual_state ==
              KnownPoseVisualState::VISUAL_ACTIVE);
  BOOST_CHECK(first.record.latest_T_cw.tvec ==
              Eigen::Vector3d(1.0, 0.0, 0.0));
  BOOST_CHECK(second.record.latest_T_cw.tvec ==
              Eigen::Vector3d(2.0, 0.0, 0.0));
  BOOST_CHECK(third.record.has_catchup_failure_evidence);
  BOOST_CHECK_EQUAL(
      third.record.last_catchup_failure.failed_lidar_map_version, 11);
}

BOOST_AUTO_TEST_CASE(RegistryCommitFailureAndDriftDoNotPartiallyMutate) {
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());
  BOOST_REQUIRE(registry.AddKnownPose(2, 2, IdentityPose()).IsSuccess());

  KnownPoseRegistryCommit invalid;
  invalid.expected_version = registry.Version();
  invalid.promotions.emplace_back(1, IdentityPose());
  invalid.latest_pose_updates.emplace_back(2, IdentityPose());
  BOOST_CHECK(invalid.expected_version == registry.Version());
  BOOST_CHECK(registry.Commit(invalid).status ==
              KnownPoseRegistryStatus::REQUIRES_VISUAL_ACTIVE);
  BOOST_CHECK(registry.GetByImageId(1).record.visual_state ==
              KnownPoseVisualState::POSE_ONLY);

  KnownPoseRegistryCommit stale;
  stale.expected_version = registry.Version();
  stale.catchup_failures.emplace_back(
      1, CatchupFailureEvidence(3, 5, 7));
  PreparedKnownPoseRegistryCommit prepared;
  BOOST_REQUIRE(registry.PrepareCommit(stale, &prepared).IsSuccess());
  BOOST_REQUIRE(registry.RecordCatchupFailure(
                            2, CatchupFailureEvidence(4, 6, 8))
                    .IsSuccess());
  BOOST_CHECK(registry.ValidatePreparedCommit(prepared).status ==
              KnownPoseRegistryStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(registry.Commit(stale).status ==
              KnownPoseRegistryStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(!registry.GetByImageId(1).record.has_catchup_failure_evidence);
}

BOOST_AUTO_TEST_CASE(Point3DOwnerTableIsStableUniqueAndSnapshotable) {
  Point3DOwnerTable table;
  Point3DOwnerTableCommit commit;
  commit.expected_version = table.Version();
  commit.owners.emplace_back(9, 2, 4, 101);
  commit.owners.emplace_back(3, 1, 7, 102);
  PreparedPoint3DOwnerTableCommit prepared;
  BOOST_REQUIRE(table.PrepareCommit(commit, &prepared).IsSuccess());
  BOOST_REQUIRE(table.ValidatePreparedCommit(prepared).IsSuccess());
  table.CommitPrepared(&prepared);

  BOOST_CHECK_EQUAL(table.Version(), 1);
  BOOST_CHECK_EQUAL(table.Size(), 2);
  const Point3DOwnerQueryResult owner9 = table.Get(9);
  const Point3DOwnerQueryResult owner3 = table.Get(3);
  BOOST_REQUIRE(owner9.IsSuccess());
  BOOST_REQUIRE(owner3.IsSuccess());
  BOOST_CHECK_EQUAL(owner9.owner.creation_sequence, 1);
  BOOST_CHECK_EQUAL(owner3.owner.creation_sequence, 2);
  const Point3DOwnerTableSnapshot snapshot = table.Snapshot();
  BOOST_REQUIRE(snapshot.IsSuccess());
  BOOST_REQUIRE_EQUAL(snapshot.owners.size(), 2);
  BOOST_CHECK_EQUAL(snapshot.owners[0].point3D_id, 9);
  BOOST_CHECK_EQUAL(snapshot.owners[1].point3D_id, 3);

  Point3DOwnerQueryResult wrong_thread;
  std::thread worker([&]() { wrong_thread = table.Get(9); });
  worker.join();
  BOOST_CHECK(wrong_thread.status == Point3DOwnerTableStatus::WRONG_THREAD);

  BOOST_CHECK(table.Add(Point3DOwnerInput(9, 3, 1, 103)).status ==
              Point3DOwnerTableStatus::DUPLICATE_POINT3D_ID);
  BOOST_CHECK(table.Add(Point3DOwnerInput(10, 2, 4, 103)).status ==
              Point3DOwnerTableStatus::DUPLICATE_OWNER_OBSERVATION);
  BOOST_CHECK(table.Add(Point3DOwnerInput(10, 3, 1, 101)).status ==
              Point3DOwnerTableStatus::DUPLICATE_ASSOCIATION_IDENTITY);
  BOOST_CHECK_EQUAL(table.Size(), 2);

  Point3DOwnerTableCommit stale;
  stale.expected_version = table.Version();
  stale.owners.emplace_back(10, 3, 1, 103);
  PreparedPoint3DOwnerTableCommit stale_prepared;
  BOOST_REQUIRE(table.PrepareCommit(stale, &stale_prepared).IsSuccess());
  BOOST_REQUIRE(table.Add(Point3DOwnerInput(11, 4, 2, 104)).IsSuccess());
  BOOST_CHECK(table.ValidatePreparedCommit(stale_prepared).status ==
              Point3DOwnerTableStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(!table.Get(10).IsSuccess());
}

BOOST_AUTO_TEST_CASE(CoordinatedTransactionPublishesAllCanonicalState) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  SetUpTransactionalReconstruction(
      &reconstruction, &correspondence_graph, 801);
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());
  ActiveCovisibilityGraph graph;
  Point3DOwnerTable owner_table;

  OnlineMapperTransactionPayload payload;
  BOOST_REQUIRE(BeginOnlineMapperTransaction(&reconstruction,
                                              &registry,
                                              &graph,
                                              &owner_table,
                                              &payload)
                    .IsSuccess());
  Reconstruction* candidate = payload.reconstruction.MutableCandidate();
  BOOST_REQUIRE(candidate != nullptr);
  Track track;
  track.AddElement(1, 0);
  const point3D_t point3D_id =
      candidate->AddPoint3D(Eigen::Vector3d(1.0, 2.0, 3.0), track);
  payload.known_pose_registry.promotions.emplace_back(1, IdentityPose());
  payload.active_covisibility_graph.nodes.emplace_back(
      1, 1, ActiveCovisibilityNodeState::VISUAL_ACTIVE);
  payload.point3D_owner_table.owners.emplace_back(
      point3D_id, 1, 0, 501);

  OnlineMapperPreparedTransaction prepared;
  BOOST_REQUIRE(PrepareOnlineMapperTransaction(&reconstruction,
                                                &registry,
                                                &graph,
                                                &owner_table,
                                                &payload,
                                                &prepared)
                    .IsSuccess());
  BOOST_REQUIRE(ValidatePreparedOnlineMapperTransaction(prepared).IsSuccess());
  CommitPreparedOnlineMapperTransaction(&prepared);

  BOOST_CHECK(!prepared.IsPrepared());
  BOOST_CHECK(reconstruction.ExistsPoint3D(point3D_id));
  BOOST_CHECK(registry.GetByImageId(1).record.visual_state ==
              KnownPoseVisualState::VISUAL_ACTIVE);
  BOOST_CHECK(graph.HasNode(1));
  const Point3DOwnerQueryResult owner = owner_table.Get(point3D_id);
  BOOST_REQUIRE(owner.IsSuccess());
  BOOST_CHECK_EQUAL(owner.owner.image_id, 1);
  BOOST_CHECK_EQUAL(owner.owner.point2D_idx, 0);
  BOOST_CHECK_EQUAL(owner.owner.association_identity, 501);
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(CoordinatedPrepareFailureAvoidsHalfCommit) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  SetUpTransactionalReconstruction(
      &reconstruction, &correspondence_graph, 802);
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());
  ActiveCovisibilityGraph graph;
  Point3DOwnerTable owner_table;
  const OnlineMapperCanonicalVersion before =
      CaptureOnlineMapperCanonicalVersion(
          &reconstruction, &registry, &graph, &owner_table)
          .canonical_version;

  OnlineMapperTransactionPayload payload;
  BOOST_REQUIRE(BeginOnlineMapperTransaction(&reconstruction,
                                              &registry,
                                              &graph,
                                              &owner_table,
                                              &payload)
                    .IsSuccess());
  Reconstruction* candidate = payload.reconstruction.MutableCandidate();
  BOOST_REQUIRE(candidate != nullptr);
  candidate->Image(1).SetTvec(Eigen::Vector3d(9.0, 0.0, 0.0));
  payload.known_pose_registry.promotions.emplace_back(1, IdentityPose());
  payload.active_covisibility_graph.nodes.emplace_back(
      1, 1, ActiveCovisibilityNodeState::POSE_ONLY);

  OnlineMapperPreparedTransaction prepared;
  BOOST_CHECK(!PrepareOnlineMapperTransaction(&reconstruction,
                                               &registry,
                                               &graph,
                                               &owner_table,
                                               &payload,
                                               &prepared)
                   .IsSuccess());
  BOOST_CHECK(reconstruction.CanonicalVersion() == before.reconstruction);
  BOOST_CHECK(reconstruction.Image(1).Tvec() == Eigen::Vector3d::Zero());
  BOOST_CHECK(registry.GetByImageId(1).record.visual_state ==
              KnownPoseVisualState::POSE_ONLY);
  BOOST_CHECK(!graph.HasNode(1));
  BOOST_CHECK_EQUAL(owner_table.Size(), 0);
  reconstruction.EndStructureJournal();
}

BOOST_AUTO_TEST_CASE(CoordinatedTransactionDetectsAnyComponentDrift) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  SetUpTransactionalReconstruction(
      &reconstruction, &correspondence_graph, 803);
  KnownPoseRegistry registry;
  BOOST_REQUIRE(registry.AddKnownPose(1, 1, IdentityPose()).IsSuccess());
  ActiveCovisibilityGraph graph;
  Point3DOwnerTable owner_table;

  OnlineMapperTransactionPayload payload;
  BOOST_REQUIRE(BeginOnlineMapperTransaction(&reconstruction,
                                              &registry,
                                              &graph,
                                              &owner_table,
                                              &payload)
                    .IsSuccess());
  payload.known_pose_registry.promotions.emplace_back(1, IdentityPose());
  BOOST_REQUIRE(registry.RecordCatchupFailure(
                            1, CatchupFailureEvidence(1, 2, 3))
                    .IsSuccess());

  OnlineMapperPreparedTransaction prepared;
  const OnlineMapperTransactionResult result =
      PrepareOnlineMapperTransaction(&reconstruction,
                                     &registry,
                                     &graph,
                                     &owner_table,
                                     &payload,
                                     &prepared);
  BOOST_CHECK(result.status ==
              OnlineMapperTransactionStatus::STALE_CANONICAL_VERSION);
  BOOST_CHECK(!prepared.IsPrepared());
  BOOST_CHECK(registry.GetByImageId(1).record.visual_state ==
              KnownPoseVisualState::POSE_ONLY);
  reconstruction.EndStructureJournal();
}
