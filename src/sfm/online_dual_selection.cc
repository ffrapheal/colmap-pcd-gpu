#include "sfm/online_dual_selection.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include <Eigen/Geometry>
#include <boost/multiprecision/cpp_int.hpp>

namespace colmap {
namespace {

constexpr size_t kDualHopThreshold = 10;
constexpr size_t kRobustPairThreshold = 15;
constexpr size_t kMinimumRobustPairsPerSide = 3;
constexpr size_t kMinimumPnPInliers = 30;
constexpr size_t kSideWindowLimit = 10;
constexpr double kQuaternionNormTolerance = 1e-8;
constexpr double kRotationMatrixTolerance = 1e-8;
constexpr double kIdentityTolerance = 1e-12;

using boost::multiprecision::cpp_int;
using OnlineDualSE3Map =
    std::map<image_t,
             OnlineDualSE3,
             std::less<image_t>,
             Eigen::aligned_allocator<std::pair<const image_t, OnlineDualSE3>>>;
using OnlineDualSourceCorrectionMap =
    std::map<image_t,
             OnlineDualSourceCorrection,
             std::less<image_t>,
             Eigen::aligned_allocator<
                 std::pair<const image_t, OnlineDualSourceCorrection>>>;
using OnlineDualImageCorrectionMap =
    std::map<image_t,
             OnlineDualImageCorrection,
             std::less<image_t>,
             Eigen::aligned_allocator<
                 std::pair<const image_t, OnlineDualImageCorrection>>>;

template <typename Result>
void SetResult(Result* result,
               const OnlineDualResultClass result_class,
               const OnlineDualReason reason,
               const std::string& detail) {
  result->result_class = result_class;
  result->reason = reason;
  result->detail = detail;
}

bool IsValidImageId(const image_t image_id) {
  return image_id != kInvalidImageId;
}

bool IsValidRegistrationSequence(const size_t registration_sequence) {
  return registration_sequence > 0 &&
         registration_sequence != std::numeric_limits<size_t>::max();
}

Eigen::Quaterniond QuaternionFromVector(const Eigen::Vector4d& qvec) {
  return Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3));
}

Eigen::Vector4d VectorFromQuaternion(const Eigen::Quaterniond& quaternion) {
  return Eigen::Vector4d(quaternion.w(),
                         quaternion.x(),
                         quaternion.y(),
                         quaternion.z());
}

Eigen::Quaterniond CanonicalQuaternion(const Eigen::Vector4d& qvec) {
  Eigen::Quaterniond quaternion = QuaternionFromVector(qvec).normalized();
  const bool negate =
      quaternion.w() < 0.0 ||
      (quaternion.w() == 0.0 && quaternion.x() < 0.0) ||
      (quaternion.w() == 0.0 && quaternion.x() == 0.0 &&
       quaternion.y() < 0.0) ||
      (quaternion.w() == 0.0 && quaternion.x() == 0.0 &&
       quaternion.y() == 0.0 && quaternion.z() < 0.0);
  if (negate) {
    quaternion.coeffs() *= -1.0;
  }
  return quaternion;
}

double RotationAngle(const OnlineDualSE3& transform) {
  const Eigen::Quaterniond quaternion = CanonicalQuaternion(transform.qvec);
  return 2.0 * std::atan2(quaternion.vec().norm(), quaternion.w());
}

OnlineDualSE3 RotationFraction(const OnlineDualSE3& transform,
                               const double alpha) {
  const Eigen::Quaterniond quaternion = CanonicalQuaternion(transform.qvec);
  const double vector_norm = quaternion.vec().norm();
  Eigen::Quaterniond fraction = Eigen::Quaterniond::Identity();
  if (vector_norm > 0.0 && alpha > 0.0) {
    const double angle = 2.0 * std::atan2(vector_norm, quaternion.w());
    fraction = Eigen::Quaterniond(
        Eigen::AngleAxisd(alpha * angle, quaternion.vec() / vector_norm));
  }
  return OnlineDualSE3(VectorFromQuaternion(fraction),
                       alpha * transform.tvec);
}

const OnlineDualClusterSummary* FindCluster(
    const OnlineDualClassificationResult& classification,
    const OnlineDualClusterId cluster_id) {
  for (const OnlineDualClusterSummary& cluster : classification.clusters) {
    if (cluster.cluster_id == cluster_id) {
      return &cluster;
    }
  }
  return nullptr;
}

OnlineDualClusterSummary* FindCluster(
    OnlineDualClassificationResult* classification,
    const OnlineDualClusterId cluster_id) {
  for (OnlineDualClusterSummary& cluster : classification->clusters) {
    if (cluster.cluster_id == cluster_id) {
      return &cluster;
    }
  }
  return nullptr;
}

bool ReferenceLess(const OnlineDualReference& lhs,
                   const OnlineDualReference& rhs) {
  return std::tie(lhs.registration_sequence, lhs.image_id) <
         std::tie(rhs.registration_sequence, rhs.image_id);
}

OnlineDualReason ValidateReferences(
    const ActiveCovisibilityGraph& graph,
    const std::vector<OnlineDualReference>& references,
    std::string* detail) {
  const ActiveCovisibilitySnapshot snapshot = graph.Snapshot();
  std::map<image_t, size_t> canonical_sequences;
  for (const ActiveCovisibilityNode& node : snapshot.nodes) {
    canonical_sequences.emplace(node.image_id, node.registration_sequence);
  }
  std::set<image_t> image_ids;
  std::set<size_t> registration_sequences;
  for (const OnlineDualReference& reference : references) {
    if (!IsValidImageId(reference.image_id) ||
        !IsValidRegistrationSequence(reference.registration_sequence)) {
      *detail = "reference has an invalid image_id or registration_sequence";
      return OnlineDualReason::INVALID_REFERENCE;
    }
    if (!reference.matched_this_event) {
      *detail = "DUAL references must be fully matched in the current event";
      return OnlineDualReason::REFERENCE_NOT_MATCHED_THIS_EVENT;
    }
    if (!reference.positive_verified || reference.verified_inliers == 0) {
      *detail = "DUAL references must be positive verified pairs";
      return OnlineDualReason::REFERENCE_NOT_POSITIVE_VERIFIED;
    }
    if (reference.visual_state !=
        ActiveCovisibilityNodeState::VISUAL_ACTIVE) {
      *detail = "POSE_ONLY references are forbidden from DUAL classification";
      return OnlineDualReason::REFERENCE_NOT_VISUAL_ACTIVE;
    }
    if (!image_ids.insert(reference.image_id).second) {
      *detail = "DUAL reference image_id is duplicated";
      return OnlineDualReason::DUPLICATE_REFERENCE;
    }
    if (!registration_sequences.insert(reference.registration_sequence)
             .second) {
      *detail = "DUAL reference registration_sequence is duplicated";
      return OnlineDualReason::DUPLICATE_REGISTRATION_SEQUENCE;
    }
    const auto canonical = canonical_sequences.find(reference.image_id);
    if (canonical == canonical_sequences.end()) {
      *detail = "DUAL reference is not in the frozen canonical graph";
      return OnlineDualReason::REFERENCE_NOT_CANONICAL;
    }
    if (canonical->second != reference.registration_sequence) {
      *detail = "DUAL reference registration_sequence differs from canonical";
      return OnlineDualReason::REGISTRATION_SEQUENCE_MISMATCH;
    }
  }
  return OnlineDualReason::NONE;
}

size_t MatrixIndex(const HopDistanceMatrixResult& matrix,
                   const image_t image_id) {
  return static_cast<size_t>(
      std::lower_bound(matrix.image_ids.begin(), matrix.image_ids.end(), image_id) -
      matrix.image_ids.begin());
}

const HopDistance& MatrixDistance(const HopDistanceMatrixResult& matrix,
                                  const image_t image_id1,
                                  const image_t image_id2) {
  return matrix.distances.at(MatrixIndex(matrix, image_id1))
      .at(MatrixIndex(matrix, image_id2));
}

bool AccumulateSupport(const OnlineDualReference& reference,
                       OnlineDualClusterSummary* cluster) {
  if (reference.verified_inliers >= kRobustPairThreshold) {
    ++cluster->robust_pair_count;
  }
  cluster->best_verified_inliers =
      std::max(cluster->best_verified_inliers, reference.verified_inliers);
  if (std::numeric_limits<uint64_t>::max() -
          cluster->total_verified_inliers <
      reference.verified_inliers) {
    return false;
  }
  cluster->total_verified_inliers += reference.verified_inliers;
  cluster->reference_image_ids.push_back(reference.image_id);
  if (std::tie(reference.registration_sequence, reference.image_id) <
      std::tie(cluster->minimum_registration_sequence,
               cluster->minimum_image_id)) {
    cluster->minimum_registration_sequence = reference.registration_sequence;
    cluster->minimum_image_id = reference.image_id;
  }
  return true;
}

OnlineDualPnPProbeRequest MakeProbeRequest(
    const OnlineDualClusterSummary& cluster,
    const OnlineDualPnPCorrespondenceSet& correspondences) {
  OnlineDualPnPProbeRequest request;
  request.cluster_id = cluster.cluster_id;
  request.reference_image_ids = cluster.reference_image_ids;
  request.points2D = correspondences.points2D;
  request.points3D = correspondences.points3D;
  return request;
}

OnlineDualPnPProbeAudit EvaluateProbe(
    const OnlineDualPnPProbeRequest& request,
    const OnlineDualPnPProbeAdapter& adapter) {
  OnlineDualPnPProbeAudit audit;
  audit.cluster_id = request.cluster_id;
  audit.reference_image_ids = request.reference_image_ids;
  audit.correspondence_count = request.points2D.size();
  if (request.cluster_id == OnlineDualClusterId::NONE ||
      request.points2D.size() != request.points3D.size()) {
    audit.status = OnlineDualPnPProbeStatus::INVALID_INPUT;
    audit.detail = "PnP correspondence arrays must have equal size";
    return audit;
  }
  if (!adapter.IsConfigured()) {
    audit.status = OnlineDualPnPProbeStatus::ADAPTER_NOT_CONFIGURED;
    audit.detail = "PnP probe adapter is not configured";
    return audit;
  }

  const OnlineDualPnPSolverOutput output = adapter.Probe(request);
  audit.probe_T_cw = output.probe_T_cw;
  audit.detail = output.detail;
  if (!output.success) {
    audit.status = OnlineDualPnPProbeStatus::SOLVER_FAILED;
    return audit;
  }
  if (output.inlier_mask.size() != request.points2D.size() ||
      std::find_if(output.inlier_mask.begin(),
                   output.inlier_mask.end(),
                   [](const uint8_t value) { return value > 1; }) !=
          output.inlier_mask.end()) {
    audit.status = OnlineDualPnPProbeStatus::MALFORMED_INLIER_MASK;
    audit.detail = "PnP solver returned a malformed inlier mask";
    return audit;
  }
  audit.inlier_count = static_cast<size_t>(
      std::count(output.inlier_mask.begin(), output.inlier_mask.end(), 1));
  if (audit.correspondence_count > 0) {
    audit.inlier_ratio = static_cast<double>(audit.inlier_count) /
                         static_cast<double>(audit.correspondence_count);
  }
  if (audit.inlier_count < kMinimumPnPInliers) {
    audit.status = OnlineDualPnPProbeStatus::INLIERS_BELOW_MINIMUM;
    audit.detail = "PnP probe has fewer than 30 inliers";
    return audit;
  }
  const size_t minimum_ratio_inliers =
      audit.correspondence_count / 4 +
      (audit.correspondence_count % 4 == 0 ? 0 : 1);
  if (audit.inlier_count < minimum_ratio_inliers) {
    audit.status = OnlineDualPnPProbeStatus::INLIER_RATIO_BELOW_MINIMUM;
    audit.detail = "PnP probe inlier ratio is below 0.25";
    return audit;
  }
  audit.pose_is_valid_se3 = IsValidOnlineDualSE3(audit.probe_T_cw);
  if (!audit.pose_is_valid_se3) {
    audit.status = OnlineDualPnPProbeStatus::INVALID_SE3;
    audit.detail = "PnP probe pose is not a finite valid SE3";
    return audit;
  }
  audit.status = OnlineDualPnPProbeStatus::SUCCESS;
  return audit;
}

OnlineDualReason ProbeReason(const OnlineDualPnPProbeStatus status) {
  switch (status) {
    case OnlineDualPnPProbeStatus::SUCCESS:
      return OnlineDualReason::NONE;
    case OnlineDualPnPProbeStatus::INVALID_INPUT:
      return OnlineDualReason::INVALID_PNP_INPUT;
    case OnlineDualPnPProbeStatus::ADAPTER_NOT_CONFIGURED:
      return OnlineDualReason::PNP_ADAPTER_NOT_CONFIGURED;
    case OnlineDualPnPProbeStatus::SOLVER_FAILED:
      return OnlineDualReason::PNP_SOLVER_FAILED;
    case OnlineDualPnPProbeStatus::MALFORMED_INLIER_MASK:
      return OnlineDualReason::PNP_MALFORMED_INLIER_MASK;
    case OnlineDualPnPProbeStatus::INLIERS_BELOW_MINIMUM:
      return OnlineDualReason::PNP_INLIERS_BELOW_MINIMUM;
    case OnlineDualPnPProbeStatus::INLIER_RATIO_BELOW_MINIMUM:
      return OnlineDualReason::PNP_INLIER_RATIO_BELOW_MINIMUM;
    case OnlineDualPnPProbeStatus::INVALID_SE3:
      return OnlineDualReason::PNP_INVALID_SE3;
    case OnlineDualPnPProbeStatus::NOT_RUN:
      return OnlineDualReason::INVALID_PNP_INPUT;
  }
  return OnlineDualReason::INVALID_PNP_INPUT;
}

const OnlineDualWindowImage* FindWindowImage(
    const OnlineDualWindowResult& window, const image_t image_id) {
  for (const OnlineDualWindowImage& image : window.images) {
    if (image.image_id == image_id) {
      return &image;
    }
  }
  return nullptr;
}

struct ExactCost {
  cpp_int numerator = 0;
  cpp_int denominator = 1;

  void AddInverse(const size_t strength) {
    const cpp_int exact_strength = strength;
    numerator = numerator * exact_strength + denominator;
    denominator *= exact_strength;
    cpp_int lhs = numerator;
    cpp_int rhs = denominator;
    while (rhs != 0) {
      const cpp_int remainder = lhs % rhs;
      lhs = rhs;
      rhs = remainder;
    }
    numerator /= lhs;
    denominator /= lhs;
  }
};

int CompareExactCost(const ExactCost& lhs, const ExactCost& rhs) {
  const cpp_int lhs_scaled = lhs.numerator * rhs.denominator;
  const cpp_int rhs_scaled = rhs.numerator * lhs.denominator;
  if (lhs_scaled < rhs_scaled) {
    return -1;
  }
  if (lhs_scaled > rhs_scaled) {
    return 1;
  }
  return 0;
}

std::map<std::pair<image_t, image_t>, size_t> SnapshotStrengths(
    const ActiveCovisibilitySnapshot& snapshot) {
  std::map<std::pair<image_t, image_t>, size_t> strengths;
  for (const ActiveCovisibilityEdge& edge : snapshot.edges) {
    strengths.emplace(std::make_pair(edge.image_id1, edge.image_id2),
                      edge.strength);
  }
  return strengths;
}

std::pair<image_t, image_t> EdgeKey(const image_t image_id1,
                                   const image_t image_id2) {
  return image_id1 < image_id2 ? std::make_pair(image_id1, image_id2)
                               : std::make_pair(image_id2, image_id1);
}

bool ExactPathCost(
    const std::vector<image_t>& path,
    const std::map<std::pair<image_t, image_t>, size_t>& strengths,
    ExactCost* cost) {
  for (size_t index = 1; index < path.size(); ++index) {
    const auto edge = strengths.find(EdgeKey(path[index - 1], path[index]));
    if (edge == strengths.end() || edge->second == 0) {
      return false;
    }
    cost->AddInverse(edge->second);
  }
  return true;
}

double ComponentMedian(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return values[middle - 1] / 2.0 + values[middle] / 2.0;
}

}  // namespace

const char* ToString(const OnlineDualResultClass value) {
  switch (value) {
    case OnlineDualResultClass::SUCCESS:
      return "SUCCESS";
    case OnlineDualResultClass::SINGLE_FALLBACK:
      return "SINGLE_FALLBACK";
    case OnlineDualResultClass::INPUT_ERROR:
      return "INPUT_ERROR";
    case OnlineDualResultClass::DUAL_FATAL:
      return "DUAL_FATAL";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualDecision value) {
  switch (value) {
    case OnlineDualDecision::SINGLE:
      return "SINGLE";
    case OnlineDualDecision::DUAL_CANDIDATE:
      return "DUAL_CANDIDATE";
    case OnlineDualDecision::DUAL_READY:
      return "DUAL_READY";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualReason value) {
  switch (value) {
    case OnlineDualReason::NONE:
      return "NONE";
    case OnlineDualReason::FEWER_THAN_TWO_REFERENCES:
      return "FEWER_THAN_TWO_REFERENCES";
    case OnlineDualReason::MAX_HOP_NOT_GREATER_THAN_TEN:
      return "MAX_HOP_NOT_GREATER_THAN_TEN";
    case OnlineDualReason::INVALID_REFERENCE:
      return "INVALID_REFERENCE";
    case OnlineDualReason::REFERENCE_NOT_MATCHED_THIS_EVENT:
      return "REFERENCE_NOT_MATCHED_THIS_EVENT";
    case OnlineDualReason::REFERENCE_NOT_POSITIVE_VERIFIED:
      return "REFERENCE_NOT_POSITIVE_VERIFIED";
    case OnlineDualReason::REFERENCE_NOT_VISUAL_ACTIVE:
      return "REFERENCE_NOT_VISUAL_ACTIVE";
    case OnlineDualReason::DUPLICATE_REFERENCE:
      return "DUPLICATE_REFERENCE";
    case OnlineDualReason::DUPLICATE_REGISTRATION_SEQUENCE:
      return "DUPLICATE_REGISTRATION_SEQUENCE";
    case OnlineDualReason::REFERENCE_NOT_CANONICAL:
      return "REFERENCE_NOT_CANONICAL";
    case OnlineDualReason::REGISTRATION_SEQUENCE_MISMATCH:
      return "REGISTRATION_SEQUENCE_MISMATCH";
    case OnlineDualReason::GRAPH_QUERY_FAILED:
      return "GRAPH_QUERY_FAILED";
    case OnlineDualReason::DISCONNECTED_ACTIVE_GRAPH:
      return "DISCONNECTED_ACTIVE_GRAPH";
    case OnlineDualReason::SUPPORT_SUM_OVERFLOW:
      return "SUPPORT_SUM_OVERFLOW";
    case OnlineDualReason::INSUFFICIENT_ROBUST_PAIRS:
      return "INSUFFICIENT_ROBUST_PAIRS";
    case OnlineDualReason::WEAK_SUPPORT_NOT_STRICTLY_GREATER_THAN_HALF:
      return "WEAK_SUPPORT_NOT_STRICTLY_GREATER_THAN_HALF";
    case OnlineDualReason::CLASSIFICATION_NOT_DUAL_CANDIDATE:
      return "CLASSIFICATION_NOT_DUAL_CANDIDATE";
    case OnlineDualReason::INVALID_PNP_INPUT:
      return "INVALID_PNP_INPUT";
    case OnlineDualReason::PNP_ADAPTER_NOT_CONFIGURED:
      return "PNP_ADAPTER_NOT_CONFIGURED";
    case OnlineDualReason::PNP_SOLVER_FAILED:
      return "PNP_SOLVER_FAILED";
    case OnlineDualReason::PNP_MALFORMED_INLIER_MASK:
      return "PNP_MALFORMED_INLIER_MASK";
    case OnlineDualReason::PNP_INLIERS_BELOW_MINIMUM:
      return "PNP_INLIERS_BELOW_MINIMUM";
    case OnlineDualReason::PNP_INLIER_RATIO_BELOW_MINIMUM:
      return "PNP_INLIER_RATIO_BELOW_MINIMUM";
    case OnlineDualReason::PNP_INVALID_SE3:
      return "PNP_INVALID_SE3";
    case OnlineDualReason::PNP_DELTA_INVALID_SE3:
      return "PNP_DELTA_INVALID_SE3";
    case OnlineDualReason::INVALID_WINDOW_INPUT:
      return "INVALID_WINDOW_INPUT";
    case OnlineDualReason::STALE_FROZEN_GRAPH:
      return "STALE_FROZEN_GRAPH";
    case OnlineDualReason::WINDOW_EXPANSION_FAILED:
      return "WINDOW_EXPANSION_FAILED";
    case OnlineDualReason::BACKBONE_UNAVAILABLE:
      return "BACKBONE_UNAVAILABLE";
    case OnlineDualReason::BACKBONE_QUERY_FAILED:
      return "BACKBONE_QUERY_FAILED";
    case OnlineDualReason::EMPTY_NEWER_WINDOW:
      return "EMPTY_NEWER_WINDOW";
    case OnlineDualReason::MISSING_WINDOW_DELTA:
      return "MISSING_WINDOW_DELTA";
    case OnlineDualReason::DUPLICATE_WINDOW_DELTA:
      return "DUPLICATE_WINDOW_DELTA";
    case OnlineDualReason::EXTRA_WINDOW_DELTA:
      return "EXTRA_WINDOW_DELTA";
    case OnlineDualReason::INVALID_WINDOW_DELTA:
      return "INVALID_WINDOW_DELTA";
    case OnlineDualReason::INVALID_BACKBONE_COST:
      return "INVALID_BACKBONE_COST";
    case OnlineDualReason::OWNERSHIP_FAILED:
      return "OWNERSHIP_FAILED";
    case OnlineDualReason::OWNERSHIP_MISSING:
      return "OWNERSHIP_MISSING";
    case OnlineDualReason::DUPLICATE_POSE_ONLY_IMAGE:
      return "DUPLICATE_POSE_ONLY_IMAGE";
    case OnlineDualReason::POSE_ONLY_IMAGE_IN_ACTIVE_GRAPH:
      return "POSE_ONLY_IMAGE_IN_ACTIVE_GRAPH";
    case OnlineDualReason::INVALID_LINEAGE:
      return "INVALID_LINEAGE";
    case OnlineDualReason::DUPLICATE_POINT:
      return "DUPLICATE_POINT";
    case OnlineDualReason::INVALID_POINT_INPUT:
      return "INVALID_POINT_INPUT";
    case OnlineDualReason::MISSING_POINT_OWNER:
      return "MISSING_POINT_OWNER";
    case OnlineDualReason::MISSING_OWNER_CORRECTION:
      return "MISSING_OWNER_CORRECTION";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualClusterId value) {
  switch (value) {
    case OnlineDualClusterId::NONE:
      return "NONE";
    case OnlineDualClusterId::FIRST:
      return "FIRST";
    case OnlineDualClusterId::SECOND:
      return "SECOND";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualVisualSide value) {
  switch (value) {
    case OnlineDualVisualSide::NONE:
      return "NONE";
    case OnlineDualVisualSide::MAIN:
      return "MAIN";
    case OnlineDualVisualSide::LOOP:
      return "LOOP";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualTemporalSide value) {
  switch (value) {
    case OnlineDualTemporalSide::NONE:
      return "NONE";
    case OnlineDualTemporalSide::OLDER:
      return "OLDER";
    case OnlineDualTemporalSide::NEWER:
      return "NEWER";
  }
  return "UNKNOWN";
}

OnlineDualResolvedPolicy GetOnlineDualResolvedPolicy() {
  OnlineDualResolvedPolicy policy;
  policy.dual_hop_threshold = kDualHopThreshold;
  policy.robust_pair_minimum_inliers = kRobustPairThreshold;
  policy.minimum_robust_pairs_per_side = kMinimumRobustPairsPerSide;
  policy.weak_to_strong_strict_ratio = 0.5;
  policy.minimum_pnp_inliers_per_side = kMinimumPnPInliers;
  policy.minimum_pnp_inlier_ratio = 0.25;
  policy.maximum_window_size = ActiveCovisibilityGraph::kMaxWindowSize;
  policy.maximum_window_size_per_side = kSideWindowLimit;
  return policy;
}

OnlineDualSE3::OnlineDualSE3()
    : qvec(1.0, 0.0, 0.0, 0.0), tvec(Eigen::Vector3d::Zero()) {}

OnlineDualSE3::OnlineDualSE3(const Eigen::Vector4d& qvec_value,
                             const Eigen::Vector3d& tvec_value)
    : qvec(qvec_value), tvec(tvec_value) {}

bool IsValidOnlineDualSE3(const OnlineDualSE3& transform) {
  if (!transform.qvec.allFinite() || !transform.tvec.allFinite()) {
    return false;
  }
  const double norm = transform.qvec.norm();
  if (!std::isfinite(norm) ||
      std::abs(norm - 1.0) > kQuaternionNormTolerance) {
    return false;
  }
  const Eigen::Matrix3d rotation =
      QuaternionFromVector(transform.qvec).toRotationMatrix();
  return rotation.allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .cwiseAbs()
                 .maxCoeff() <= kRotationMatrixTolerance &&
         rotation.determinant() > 0.0 &&
         std::abs(rotation.determinant() - 1.0) <=
             kRotationMatrixTolerance;
}

bool IsIdentityOnlineDualSE3(const OnlineDualSE3& transform) {
  if (!IsValidOnlineDualSE3(transform)) {
    return false;
  }
  return transform.tvec.cwiseAbs().maxCoeff() <= kIdentityTolerance &&
         RotationAngle(transform) <= kIdentityTolerance;
}

OnlineDualSE3 InverseOnlineDualSE3(const OnlineDualSE3& transform) {
  const Eigen::Quaterniond quaternion = CanonicalQuaternion(transform.qvec);
  const Eigen::Quaterniond inverse = quaternion.conjugate();
  return OnlineDualSE3(VectorFromQuaternion(inverse),
                       inverse * -transform.tvec);
}

OnlineDualSE3 ComposeOnlineDualSE3(const OnlineDualSE3& lhs,
                                   const OnlineDualSE3& rhs) {
  const Eigen::Quaterniond lhs_quaternion = CanonicalQuaternion(lhs.qvec);
  const Eigen::Quaterniond rhs_quaternion = CanonicalQuaternion(rhs.qvec);
  const Eigen::Quaterniond composed = lhs_quaternion * rhs_quaternion;
  return OnlineDualSE3(VectorFromQuaternion(composed.normalized()),
                       lhs_quaternion * rhs.tvec + lhs.tvec);
}

OnlineDualSE3 ComputeOnlineDualDelta(const OnlineDualSE3& before_T_wc,
                                     const OnlineDualSE3& after_T_wc) {
  return ComposeOnlineDualSE3(after_T_wc,
                              InverseOnlineDualSE3(before_T_wc));
}

Eigen::Vector3d ApplyOnlineDualSE3(const OnlineDualSE3& transform,
                                  const Eigen::Vector3d& point) {
  return CanonicalQuaternion(transform.qvec) * point + transform.tvec;
}

OnlineDualReference::OnlineDualReference() = default;

OnlineDualReference::OnlineDualReference(
    const image_t image_id_value,
    const size_t registration_sequence_value,
    const size_t verified_inliers_value,
    const ActiveCovisibilityNodeState visual_state_value,
    const bool matched_this_event_value,
    const bool positive_verified_value)
    : image_id(image_id_value),
      registration_sequence(registration_sequence_value),
      verified_inliers(verified_inliers_value),
      visual_state(visual_state_value),
      matched_this_event(matched_this_event_value),
      positive_verified(positive_verified_value) {}

OnlineDualClassificationResult ClassifyOnlineDualReferences(
    const ActiveCovisibilityGraph& frozen_graph,
    const std::vector<OnlineDualReference>& references) {
  OnlineDualClassificationResult result;
  result.canonical_graph_version = frozen_graph.Version();
  std::string validation_detail;
  const OnlineDualReason validation =
      ValidateReferences(frozen_graph, references, &validation_detail);
  if (validation != OnlineDualReason::NONE) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              validation,
              validation_detail);
    return result;
  }

  result.ordered_references = references;
  std::sort(result.ordered_references.begin(),
            result.ordered_references.end(),
            ReferenceLess);
  if (result.ordered_references.size() < 2) {
    SetResult(&result,
              OnlineDualResultClass::SINGLE_FALLBACK,
              OnlineDualReason::FEWER_THAN_TWO_REFERENCES,
              "fewer than two effective VISUAL_ACTIVE references");
    return result;
  }

  std::vector<image_t> image_ids;
  for (const OnlineDualReference& reference : result.ordered_references) {
    image_ids.push_back(reference.image_id);
  }
  const HopDistanceMatrixResult matrix =
      frozen_graph.HopDistanceMatrix(image_ids);
  if (!matrix.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::GRAPH_QUERY_FAILED,
              matrix.detail);
    return result;
  }
  if (matrix.canonical_version != result.canonical_graph_version ||
      frozen_graph.Version() != result.canonical_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "canonical graph changed during DUAL classification");
    return result;
  }

  size_t maximum_hops = 0;
  size_t first_seed_index = 0;
  size_t second_seed_index = 1;
  for (size_t index1 = 0; index1 < result.ordered_references.size(); ++index1) {
    for (size_t index2 = index1 + 1;
         index2 < result.ordered_references.size();
         ++index2) {
      const HopDistance& distance = MatrixDistance(
          matrix,
          result.ordered_references[index1].image_id,
          result.ordered_references[index2].image_id);
      if (!distance.IsFinite()) {
        SetResult(&result,
                  OnlineDualResultClass::DUAL_FATAL,
                  OnlineDualReason::DISCONNECTED_ACTIVE_GRAPH,
                  "effective references are disconnected in canonical graph");
        return result;
      }
      if (distance.hops > maximum_hops) {
        maximum_hops = distance.hops;
        first_seed_index = index1;
        second_seed_index = index2;
        result.farthest_pair_tie_count = 1;
      } else if (distance.hops == maximum_hops) {
        ++result.farthest_pair_tie_count;
      }
    }
  }
  result.max_hop_defined = true;
  result.maximum_hops = maximum_hops;
  if (maximum_hops <= kDualHopThreshold) {
    SetResult(&result,
              OnlineDualResultClass::SINGLE_FALLBACK,
              OnlineDualReason::MAX_HOP_NOT_GREATER_THAN_TEN,
              "maximum canonical BFS hop is not greater than 10");
    return result;
  }

  const OnlineDualReference& first_seed =
      result.ordered_references[first_seed_index];
  const OnlineDualReference& second_seed =
      result.ordered_references[second_seed_index];
  result.first_seed_image_id = first_seed.image_id;
  result.second_seed_image_id = second_seed.image_id;
  result.clusters.resize(2);
  result.clusters[0].cluster_id = OnlineDualClusterId::FIRST;
  result.clusters[0].seed_image_id = first_seed.image_id;
  result.clusters[1].cluster_id = OnlineDualClusterId::SECOND;
  result.clusters[1].seed_image_id = second_seed.image_id;

  for (const OnlineDualReference& reference : result.ordered_references) {
    OnlineDualReferenceAssignment assignment;
    assignment.reference = reference;
    assignment.hops_to_first_seed =
        MatrixDistance(matrix, reference.image_id, first_seed.image_id).hops;
    assignment.hops_to_second_seed =
        MatrixDistance(matrix, reference.image_id, second_seed.image_id).hops;
    if (assignment.hops_to_first_seed < assignment.hops_to_second_seed) {
      assignment.cluster_id = OnlineDualClusterId::FIRST;
    } else if (assignment.hops_to_second_seed <
               assignment.hops_to_first_seed) {
      assignment.cluster_id = OnlineDualClusterId::SECOND;
    } else {
      assignment.cluster_id =
          ReferenceLess(first_seed, second_seed)
              ? OnlineDualClusterId::FIRST
              : OnlineDualClusterId::SECOND;
      assignment.assigned_by_seed_tie_break = true;
    }
    OnlineDualClusterSummary* cluster =
        FindCluster(&result, assignment.cluster_id);
    if (cluster == nullptr || !AccumulateSupport(reference, cluster)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::SUPPORT_SUM_OVERFLOW,
                "cluster verified support sum overflowed");
      return result;
    }
    result.assignments.push_back(assignment);
  }

  OnlineDualClusterSummary& first_cluster = result.clusters[0];
  OnlineDualClusterSummary& second_cluster = result.clusters[1];
  if (first_cluster.robust_pair_count < kMinimumRobustPairsPerSide ||
      second_cluster.robust_pair_count < kMinimumRobustPairsPerSide) {
    SetResult(&result,
              OnlineDualResultClass::SINGLE_FALLBACK,
              OnlineDualReason::INSUFFICIENT_ROBUST_PAIRS,
              "each DUAL cluster requires at least three robust pairs");
  } else {
    const size_t weak_best =
        std::min(first_cluster.best_verified_inliers,
                 second_cluster.best_verified_inliers);
    const size_t strong_best =
        std::max(first_cluster.best_verified_inliers,
                 second_cluster.best_verified_inliers);
    if (weak_best <= strong_best / 2) {
      SetResult(
          &result,
          OnlineDualResultClass::SINGLE_FALLBACK,
          OnlineDualReason::WEAK_SUPPORT_NOT_STRICTLY_GREATER_THAN_HALF,
          "weaker best support must be strictly greater than half the stronger");
    }
  }

  const bool support_tie = first_cluster.total_verified_inliers ==
                           second_cluster.total_verified_inliers;
  result.main_selected_by_tie_break = support_tie;
  const bool first_is_main =
      first_cluster.total_verified_inliers >
          second_cluster.total_verified_inliers ||
      (support_tie &&
       std::tie(first_cluster.minimum_registration_sequence,
                first_cluster.minimum_image_id) <
           std::tie(second_cluster.minimum_registration_sequence,
                    second_cluster.minimum_image_id));
  result.main_cluster_id = first_is_main ? OnlineDualClusterId::FIRST
                                         : OnlineDualClusterId::SECOND;
  result.loop_cluster_id = first_is_main ? OnlineDualClusterId::SECOND
                                         : OnlineDualClusterId::FIRST;

  const bool first_is_older =
      std::tie(first_cluster.minimum_registration_sequence,
               first_cluster.minimum_image_id) <
      std::tie(second_cluster.minimum_registration_sequence,
               second_cluster.minimum_image_id);
  result.older_cluster_id = first_is_older ? OnlineDualClusterId::FIRST
                                           : OnlineDualClusterId::SECOND;
  result.newer_cluster_id = first_is_older ? OnlineDualClusterId::SECOND
                                           : OnlineDualClusterId::FIRST;
  first_cluster.visual_side =
      result.main_cluster_id == OnlineDualClusterId::FIRST
          ? OnlineDualVisualSide::MAIN
          : OnlineDualVisualSide::LOOP;
  second_cluster.visual_side =
      result.main_cluster_id == OnlineDualClusterId::SECOND
          ? OnlineDualVisualSide::MAIN
          : OnlineDualVisualSide::LOOP;
  first_cluster.temporal_side =
      result.older_cluster_id == OnlineDualClusterId::FIRST
          ? OnlineDualTemporalSide::OLDER
          : OnlineDualTemporalSide::NEWER;
  second_cluster.temporal_side =
      result.older_cluster_id == OnlineDualClusterId::SECOND
          ? OnlineDualTemporalSide::OLDER
          : OnlineDualTemporalSide::NEWER;

  if (result.result_class == OnlineDualResultClass::INPUT_ERROR) {
    SetResult(&result,
              OnlineDualResultClass::SUCCESS,
              OnlineDualReason::NONE,
              std::string());
    result.decision = OnlineDualDecision::DUAL_CANDIDATE;
  }
  return result;
}

OnlineDualPnPProbeAdapter::OnlineDualPnPProbeAdapter() = default;

OnlineDualPnPProbeAdapter::OnlineDualPnPProbeAdapter(Solver solver)
    : solver_(std::move(solver)) {}

bool OnlineDualPnPProbeAdapter::IsConfigured() const {
  return static_cast<bool>(solver_);
}

OnlineDualPnPSolverOutput OnlineDualPnPProbeAdapter::Probe(
    const OnlineDualPnPProbeRequest& request) const {
  if (!solver_) {
    OnlineDualPnPSolverOutput output;
    output.detail = "PnP probe adapter is not configured";
    return output;
  }
  try {
    return solver_(request);
  } catch (const std::exception& exception) {
    OnlineDualPnPSolverOutput output;
    output.detail = std::string("PnP probe threw: ") + exception.what();
    return output;
  } catch (...) {
    OnlineDualPnPSolverOutput output;
    output.detail = "PnP probe threw a non-standard exception";
    return output;
  }
}

const char* ToString(const OnlineDualPnPProbeStatus value) {
  switch (value) {
    case OnlineDualPnPProbeStatus::NOT_RUN:
      return "NOT_RUN";
    case OnlineDualPnPProbeStatus::SUCCESS:
      return "SUCCESS";
    case OnlineDualPnPProbeStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case OnlineDualPnPProbeStatus::ADAPTER_NOT_CONFIGURED:
      return "ADAPTER_NOT_CONFIGURED";
    case OnlineDualPnPProbeStatus::SOLVER_FAILED:
      return "SOLVER_FAILED";
    case OnlineDualPnPProbeStatus::MALFORMED_INLIER_MASK:
      return "MALFORMED_INLIER_MASK";
    case OnlineDualPnPProbeStatus::INLIERS_BELOW_MINIMUM:
      return "INLIERS_BELOW_MINIMUM";
    case OnlineDualPnPProbeStatus::INLIER_RATIO_BELOW_MINIMUM:
      return "INLIER_RATIO_BELOW_MINIMUM";
    case OnlineDualPnPProbeStatus::INVALID_SE3:
      return "INVALID_SE3";
  }
  return "UNKNOWN";
}

OnlineDualPnPDecisionResult RunOnlineDualTwoSidePnPProbe(
    const OnlineDualClassificationResult& classification,
    const std::vector<OnlineDualPnPCorrespondenceSet>& correspondence_sets,
    const OnlineDualPnPProbeAdapter& adapter) {
  OnlineDualPnPDecisionResult result;
  if (!classification.IsDualCandidate()) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::CLASSIFICATION_NOT_DUAL_CANDIDATE,
              "two-side PnP may only run for a DUAL candidate");
    return result;
  }
  std::map<OnlineDualClusterId, const OnlineDualPnPCorrespondenceSet*> sets;
  for (const OnlineDualPnPCorrespondenceSet& set : correspondence_sets) {
    if (set.cluster_id == OnlineDualClusterId::NONE ||
        !sets.emplace(set.cluster_id, &set).second) {
      SetResult(&result,
                OnlineDualResultClass::INPUT_ERROR,
                OnlineDualReason::INVALID_PNP_INPUT,
                "PnP input requires one unique set per cluster");
      return result;
    }
  }
  if (sets.size() != 2 ||
      sets.count(OnlineDualClusterId::FIRST) == 0 ||
      sets.count(OnlineDualClusterId::SECOND) == 0) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_PNP_INPUT,
              "PnP input must contain FIRST and SECOND cluster sets");
    return result;
  }

  for (const OnlineDualClusterId cluster_id :
       {OnlineDualClusterId::FIRST, OnlineDualClusterId::SECOND}) {
    const OnlineDualClusterSummary* cluster =
        FindCluster(classification, cluster_id);
    if (cluster == nullptr) {
      SetResult(&result,
                OnlineDualResultClass::INPUT_ERROR,
                OnlineDualReason::INVALID_PNP_INPUT,
                "classification is missing a cluster summary");
      return result;
    }
    result.probes.push_back(EvaluateProbe(
        MakeProbeRequest(*cluster, *sets.at(cluster_id)), adapter));
  }

  for (const OnlineDualPnPProbeAudit& probe : result.probes) {
    if (probe.status != OnlineDualPnPProbeStatus::SUCCESS) {
      result.failed_cluster_id = probe.cluster_id;
      SetResult(&result,
                OnlineDualResultClass::SINGLE_FALLBACK,
                ProbeReason(probe.status),
                probe.detail);
      return result;
    }
  }

  const OnlineDualPnPProbeAudit* older_probe = nullptr;
  const OnlineDualPnPProbeAudit* newer_probe = nullptr;
  for (const OnlineDualPnPProbeAudit& probe : result.probes) {
    if (probe.cluster_id == classification.older_cluster_id) {
      older_probe = &probe;
    }
    if (probe.cluster_id == classification.newer_cluster_id) {
      newer_probe = &probe;
    }
  }
  if (older_probe == nullptr || newer_probe == nullptr) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_PNP_INPUT,
              "classification older/newer mapping is incomplete");
    return result;
  }
  const OnlineDualSE3 older_T_wc =
      InverseOnlineDualSE3(older_probe->probe_T_cw);
  const OnlineDualSE3 newer_T_wc =
      InverseOnlineDualSE3(newer_probe->probe_T_cw);
  result.delta_init = ComposeOnlineDualSE3(
      older_T_wc, InverseOnlineDualSE3(newer_T_wc));
  if (!IsValidOnlineDualSE3(result.delta_init)) {
    SetResult(&result,
              OnlineDualResultClass::SINGLE_FALLBACK,
              OnlineDualReason::PNP_DELTA_INVALID_SE3,
              "PnP-only Delta_init is not a finite valid SE3");
    return result;
  }
  result.delta_translation_norm = result.delta_init.tvec.norm();
  result.delta_rotation_radians = RotationAngle(result.delta_init);
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  result.decision = OnlineDualDecision::DUAL_CANDIDATE;
  return result;
}

OnlineDualWindowResult BuildOnlineDualFrozenWindow(
    const ActiveCovisibilityGraph& frozen_graph,
    const ActiveCovisibilityOverlay& current_overlay,
    const image_t current_image_id,
    const size_t current_registration_sequence,
    const OnlineDualClassificationResult& classification,
    const size_t max_window_size) {
  OnlineDualWindowResult result;
  result.frozen_graph_version = frozen_graph.Version();
  result.current_image_id = current_image_id;
  if (max_window_size < 2 ||
      max_window_size > ActiveCovisibilityGraph::kMaxWindowSize ||
      !classification.IsDualCandidate() || !IsValidImageId(current_image_id) ||
      !IsValidRegistrationSequence(current_registration_sequence) ||
      frozen_graph.HasNode(current_image_id)) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "DUAL window requires a non-canonical current and candidate classification");
    return result;
  }
  if (classification.canonical_graph_version != result.frozen_graph_version ||
      current_overlay.base_version != result.frozen_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "classification, overlay, and graph versions differ");
    return result;
  }
  const ActiveCovisibilityResult overlay_validation =
      frozen_graph.ValidateTentativeOverlay(current_overlay);
  if (!overlay_validation.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              overlay_validation.detail);
    return result;
  }
  size_t current_declarations = 0;
  for (const ActiveCovisibilityNode& node : current_overlay.temporary_nodes) {
    if (node.image_id == current_image_id &&
        node.registration_sequence == current_registration_sequence) {
      ++current_declarations;
    }
  }
  const auto all_edges_touch_current = [current_image_id](
                                           const std::vector<
                                               ActiveCovisibilityEdgeInput>&
                                               edges) {
    return std::all_of(
        edges.begin(),
        edges.end(),
        [current_image_id](const ActiveCovisibilityEdgeInput& edge) {
          return edge.image_id1 == current_image_id ||
                 edge.image_id2 == current_image_id;
        });
  };
  if (current_declarations != 1 ||
      current_overlay.temporary_nodes.size() != 1 ||
      !all_edges_touch_current(current_overlay.ordinary_edges) ||
      !all_edges_touch_current(current_overlay.loop_edges)) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "overlay may only declare current and edges incident to current");
    return result;
  }

  result.current_cluster_id = classification.main_cluster_id;
  const OnlineDualClusterSummary* older_cluster =
      FindCluster(classification, classification.older_cluster_id);
  const OnlineDualClusterSummary* newer_cluster =
      FindCluster(classification, classification.newer_cluster_id);
  if (older_cluster == nullptr || newer_cluster == nullptr) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "classification older/newer clusters are missing");
    return result;
  }
  result.older_side.cluster_id = older_cluster->cluster_id;
  result.older_side.temporal_side = OnlineDualTemporalSide::OLDER;
  result.newer_side.cluster_id = newer_cluster->cluster_id;
  result.newer_side.temporal_side = OnlineDualTemporalSide::NEWER;

  const auto expansion_for = [&](const OnlineDualClusterSummary& cluster) {
    std::vector<image_t> seeds = cluster.reference_image_ids;
    if (cluster.cluster_id == classification.main_cluster_id) {
      seeds.push_back(current_image_id);
    }
    return frozen_graph.ExpandWindow(
        seeds, ActiveCovisibilityGraph::kMaxWindowSize, &current_overlay);
  };
  const WindowExpansionResult older_expansion = expansion_for(*older_cluster);
  if (!older_expansion.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::WINDOW_EXPANSION_FAILED,
              older_expansion.detail);
    return result;
  }
  const WindowExpansionResult newer_expansion = expansion_for(*newer_cluster);
  if (!newer_expansion.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::WINDOW_EXPANSION_FAILED,
              newer_expansion.detail);
    return result;
  }

  const size_t side_window_limit = max_window_size / 2;
  std::set<image_t> selected;
  const auto append_side = [&](const WindowExpansionResult& expansion,
                               const OnlineDualClusterSummary& cluster,
                               const OnlineDualTemporalSide temporal_side,
                               const bool older_priority,
                               OnlineDualWindowSideAudit* audit,
                               std::vector<OnlineDualWindowImage>* images,
                               std::set<image_t>* all_selected) {
    std::set<image_t> cluster_seeds(cluster.reference_image_ids.begin(),
                                    cluster.reference_image_ids.end());
    for (const WindowSelection& selection : expansion.selection_order) {
      audit->expansion_order.push_back(selection.image_id);
      if (selection.image_id == current_image_id &&
          cluster.cluster_id != classification.main_cluster_id) {
        audit->skipped_current_wrong_side_image_ids.push_back(
            selection.image_id);
        continue;
      }
      if (!older_priority && all_selected->count(selection.image_id) != 0) {
        audit->skipped_overlap_image_ids.push_back(selection.image_id);
        continue;
      }
      if (audit->selected_image_ids.size() >= side_window_limit) {
        continue;
      }
      if (cluster.cluster_id == classification.main_cluster_id &&
          selection.image_id != current_image_id &&
          all_selected->count(current_image_id) == 0 &&
          audit->selected_image_ids.size() + 1 == side_window_limit) {
        continue;
      }
      if (!all_selected->insert(selection.image_id).second) {
        continue;
      }
      OnlineDualWindowImage image;
      image.image_id = selection.image_id;
      image.cluster_id = cluster.cluster_id;
      image.visual_side = cluster.visual_side;
      image.temporal_side = temporal_side;
      image.side_selection_rank = audit->selected_image_ids.size();
      image.expansion_support_strength = selection.support_strength;
      image.is_current = selection.image_id == current_image_id;
      image.is_cluster_seed = cluster_seeds.count(selection.image_id) != 0;
      audit->selected_image_ids.push_back(selection.image_id);
      images->push_back(image);
    }
  };
  append_side(older_expansion,
              *older_cluster,
              OnlineDualTemporalSide::OLDER,
              true,
              &result.older_side,
              &result.images,
              &selected);
  append_side(newer_expansion,
              *newer_cluster,
              OnlineDualTemporalSide::NEWER,
              false,
              &result.newer_side,
              &result.images,
              &selected);

  const size_t current_count = static_cast<size_t>(std::count_if(
      result.images.begin(), result.images.end(), [current_image_id](
                                                   const OnlineDualWindowImage& image) {
        return image.image_id == current_image_id;
      }));
  if (current_count != 1 || result.images.size() > max_window_size ||
      result.older_side.selected_image_ids.size() > side_window_limit ||
      result.newer_side.selected_image_ids.size() > side_window_limit) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "DUAL window violates current, total, or per-side limits");
    return result;
  }
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  return result;
}

OnlineDualBackboneResult SelectOnlineDualBackbone(
    const ActiveCovisibilityGraph& frozen_graph,
    const OnlineDualWindowResult& window) {
  OnlineDualBackboneResult result;
  result.frozen_graph_version = frozen_graph.Version();
  if (!window.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::INPUT_ERROR,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "backbone requires a valid frozen DUAL window");
    return result;
  }
  if (window.frozen_graph_version != result.frozen_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "canonical graph changed before backbone selection");
    return result;
  }

  std::vector<image_t> older_endpoints;
  std::vector<image_t> newer_endpoints;
  for (const OnlineDualWindowImage& image : window.images) {
    if (!frozen_graph.HasNode(image.image_id)) {
      continue;
    }
    if (image.temporal_side == OnlineDualTemporalSide::OLDER) {
      older_endpoints.push_back(image.image_id);
    } else if (image.temporal_side == OnlineDualTemporalSide::NEWER) {
      newer_endpoints.push_back(image.image_id);
    }
  }
  std::sort(older_endpoints.begin(), older_endpoints.end());
  std::sort(newer_endpoints.begin(), newer_endpoints.end());
  const ActiveCovisibilitySnapshot snapshot = frozen_graph.Snapshot();
  const auto strengths = SnapshotStrengths(snapshot);
  bool found = false;
  ExactCost best_cost;
  ShortestBackboneResult best_backbone;
  for (const image_t older_image_id : older_endpoints) {
    for (const image_t newer_image_id : newer_endpoints) {
      const ShortestBackboneResult backbone =
          frozen_graph.ShortestBackbone(older_image_id, newer_image_id);
      OnlineDualBackboneCandidateAudit candidate;
      candidate.older_image_id = older_image_id;
      candidate.newer_image_id = newer_image_id;
      candidate.reachable = backbone.IsSuccess() && backbone.reachable;
      candidate.total_cost = backbone.total_cost;
      candidate.path = backbone.path;
      result.candidates.push_back(candidate);
      if (backbone.status == ActiveCovisibilityStatus::UNREACHABLE) {
        continue;
      }
      if (!backbone.IsSuccess()) {
        SetResult(&result,
                  OnlineDualResultClass::DUAL_FATAL,
                  OnlineDualReason::BACKBONE_QUERY_FAILED,
                  backbone.detail);
        return result;
      }
      ExactCost candidate_cost;
      if (backbone.path.size() < 2 ||
          !ExactPathCost(backbone.path, strengths, &candidate_cost)) {
        SetResult(&result,
                  OnlineDualResultClass::DUAL_FATAL,
                  OnlineDualReason::BACKBONE_QUERY_FAILED,
                  "canonical backbone path does not match frozen edges");
        return result;
      }
      const int cost_order = found ? CompareExactCost(candidate_cost, best_cost)
                                   : -1;
      if (!found || cost_order < 0 ||
          (cost_order == 0 &&
           std::tie(older_image_id, newer_image_id) <
               std::tie(result.older_endpoint_image_id,
                        result.newer_endpoint_image_id))) {
        found = true;
        best_cost = candidate_cost;
        best_backbone = backbone;
        result.older_endpoint_image_id = older_image_id;
        result.newer_endpoint_image_id = newer_image_id;
        result.minimum_cost_endpoint_pair_count = 1;
      } else if (cost_order == 0) {
        ++result.minimum_cost_endpoint_pair_count;
      }
    }
  }
  if (frozen_graph.Version() != result.frozen_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "canonical graph changed during backbone selection");
    return result;
  }
  if (!found) {
    SetResult(&result,
              OnlineDualResultClass::SINGLE_FALLBACK,
              OnlineDualReason::BACKBONE_UNAVAILABLE,
              "no finite canonical cross-side backbone exists");
    return result;
  }
  result.total_cost = best_backbone.total_cost;
  result.path = best_backbone.path;
  result.edge_costs = best_backbone.edge_costs;
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  result.decision = OnlineDualDecision::DUAL_READY;
  return result;
}

const char* ToString(const OnlineDualImageCorrectionKind value) {
  switch (value) {
    case OnlineDualImageCorrectionKind::DIRECT_WINDOW_RESULT:
      return "DIRECT_WINDOW_RESULT";
    case OnlineDualImageCorrectionKind::PROPAGATED:
      return "PROPAGATED";
    case OnlineDualImageCorrectionKind::IDENTITY_OLDER_DOMAIN:
      return "IDENTITY_OLDER_DOMAIN";
    case OnlineDualImageCorrectionKind::IDENTITY_TRANSFORM:
      return "IDENTITY_TRANSFORM";
    case OnlineDualImageCorrectionKind::POSE_ONLY:
      return "POSE_ONLY";
  }
  return "UNKNOWN";
}

OnlineDualPropagationPlan BuildOnlineDualPropagationPlan(
    const ActiveCovisibilityGraph& frozen_graph,
    const OnlineDualWindowResult& window,
    const OnlineDualBackboneResult& backbone,
    const OnlineDualImageDeltaVector& window_deltas,
    const std::vector<image_t>& pose_only_image_ids) {
  OnlineDualPropagationPlan result;
  result.frozen_graph_version = frozen_graph.Version();
  if (!window.IsSuccess() || !backbone.IsDualReady()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::INVALID_WINDOW_INPUT,
              "propagation requires a valid window and DUAL-ready backbone");
    return result;
  }
  if (window.frozen_graph_version != result.frozen_graph_version ||
      backbone.frozen_graph_version != result.frozen_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "canonical graph changed before propagation planning");
    return result;
  }

  OnlineDualSE3Map deltas;
  for (const OnlineDualImageDelta& image_delta : window_deltas) {
    if (!IsValidImageId(image_delta.image_id)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_WINDOW_DELTA,
                "window Delta has an invalid image_id");
      return result;
    }
    if (!deltas.emplace(image_delta.image_id, image_delta.delta).second) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::DUPLICATE_WINDOW_DELTA,
                "window Delta image_id is duplicated");
      return result;
    }
    if (!IsValidOnlineDualSE3(image_delta.delta)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_WINDOW_DELTA,
                "window Delta is not a finite valid SE3");
      return result;
    }
  }
  std::set<image_t> window_image_ids;
  std::vector<image_t> newer_image_ids;
  std::set<image_t> older_canonical_sources;
  for (const OnlineDualWindowImage& image : window.images) {
    window_image_ids.insert(image.image_id);
    if (deltas.count(image.image_id) == 0) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::MISSING_WINDOW_DELTA,
                "every frozen window image requires an actual Delta_i");
      return result;
    }
    if (image.temporal_side == OnlineDualTemporalSide::NEWER) {
      newer_image_ids.push_back(image.image_id);
    } else if (image.temporal_side == OnlineDualTemporalSide::OLDER &&
               frozen_graph.HasNode(image.image_id)) {
      older_canonical_sources.insert(image.image_id);
    }
  }
  if (deltas.size() != window_image_ids.size()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::EXTRA_WINDOW_DELTA,
              "window Delta set contains a non-window image");
    return result;
  }
  if (newer_image_ids.empty()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::EMPTY_NEWER_WINDOW,
              "newer DUAL window has no actual Delta_i");
    return result;
  }

  std::vector<double> x_values;
  std::vector<double> y_values;
  std::vector<double> z_values;
  for (const image_t image_id : newer_image_ids) {
    x_values.push_back(deltas.at(image_id).tvec.x());
    y_values.push_back(deltas.at(image_id).tvec.y());
    z_values.push_back(deltas.at(image_id).tvec.z());
  }
  result.componentwise_translation_median =
      Eigen::Vector3d(ComponentMedian(x_values),
                      ComponentMedian(y_values),
                      ComponentMedian(z_values));
  if (!result.componentwise_translation_median.allFinite()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::INVALID_WINDOW_DELTA,
              "componentwise translation median is not finite");
    return result;
  }
  std::sort(newer_image_ids.begin(), newer_image_ids.end());
  double best_squared_distance = std::numeric_limits<double>::infinity();
  for (const image_t image_id : newer_image_ids) {
    const double squared_distance =
        (deltas.at(image_id).tvec -
         result.componentwise_translation_median)
            .squaredNorm();
    if (!std::isfinite(squared_distance)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_WINDOW_DELTA,
                "representative translation distance is not finite");
      return result;
    }
    OnlineDualRepresentativeCandidateAudit candidate;
    candidate.image_id = image_id;
    candidate.translation_distance_to_median = std::sqrt(squared_distance);
    result.representative_candidates.push_back(candidate);
    if (squared_distance < best_squared_distance ||
        (squared_distance == best_squared_distance &&
         image_id < result.representative_image_id)) {
      best_squared_distance = squared_distance;
      result.representative_image_id = image_id;
      result.minimum_distance_representative_count = 1;
    } else if (squared_distance == best_squared_distance) {
      ++result.minimum_distance_representative_count;
    }
  }
  result.representative_delta = deltas.at(result.representative_image_id);

  if (backbone.path.size() < 2 ||
      backbone.edge_costs.size() + 1 != backbone.path.size()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::INVALID_BACKBONE_COST,
              "backbone path and edge costs are inconsistent");
    return result;
  }
  double total_cost = 0.0;
  for (const double edge_cost : backbone.edge_costs) {
    if (!std::isfinite(edge_cost) || edge_cost <= 0.0) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_BACKBONE_COST,
                "backbone edge cost must be finite and positive");
      return result;
    }
    total_cost += edge_cost;
  }
  if (!std::isfinite(total_cost) || total_cost <= 0.0) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::INVALID_BACKBONE_COST,
              "backbone total cost must be finite and positive");
    return result;
  }

  OnlineDualSourceCorrectionMap source_corrections;
  double cumulative_cost = 0.0;
  for (size_t index = 0; index < backbone.path.size(); ++index) {
    if (index > 0) {
      cumulative_cost += backbone.edge_costs[index - 1];
    }
    OnlineDualSourceCorrection source;
    source.source_image_id = backbone.path[index];
    source.backbone_alpha = cumulative_cost / total_cost;
    source.forced_identity_by_older_window =
        older_canonical_sources.count(source.source_image_id) != 0;
    source.delta = source.forced_identity_by_older_window
                       ? OnlineDualSE3()
                       : RotationFraction(result.representative_delta,
                                          source.backbone_alpha);
    if (!std::isfinite(source.backbone_alpha) ||
        source.backbone_alpha < 0.0 || source.backbone_alpha > 1.0 ||
        !IsValidOnlineDualSE3(source.delta)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_BACKBONE_COST,
                "backbone alpha produced an invalid source correction");
      return result;
    }
    source_corrections[source.source_image_id] = source;
  }
  for (const image_t source_image_id : older_canonical_sources) {
    OnlineDualSourceCorrection source;
    source.source_image_id = source_image_id;
    source.forced_identity_by_older_window = true;
    source.delta = OnlineDualSE3();
    source_corrections[source_image_id] = source;
  }
  for (const auto& source : source_corrections) {
    result.source_corrections.push_back(source.second);
  }

  std::vector<image_t> source_ids;
  for (const auto& source : source_corrections) {
    source_ids.push_back(source.first);
  }
  const MultiSourceOwnershipResult ownership =
      frozen_graph.ComputeMultiSourceOwnership(source_ids);
  if (!ownership.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::OWNERSHIP_FAILED,
              ownership.detail);
    return result;
  }
  if (ownership.canonical_version != result.frozen_graph_version ||
      frozen_graph.Version() != result.frozen_graph_version) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::STALE_FROZEN_GRAPH,
              "canonical graph changed during ownership planning");
    return result;
  }

  OnlineDualImageCorrectionMap corrections;
  for (const OnlineDualWindowImage& image : window.images) {
    OnlineDualImageCorrection correction;
    correction.image_id = image.image_id;
    correction.kind = OnlineDualImageCorrectionKind::DIRECT_WINDOW_RESULT;
    correction.delta = OnlineDualSE3();
    corrections.emplace(correction.image_id, correction);
  }
  for (const MultiSourceOwnership& owner : ownership.ownership) {
    if (!owner.reachable || !std::isfinite(owner.distance)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::OWNERSHIP_MISSING,
                "canonical active image has no finite ownership source");
      return result;
    }
    if (corrections.count(owner.node_image_id) != 0) {
      continue;
    }
    const auto source = source_corrections.find(owner.source_image_id);
    if (source == source_corrections.end()) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::OWNERSHIP_MISSING,
                "ownership result references an unknown source");
      return result;
    }
    OnlineDualImageCorrection correction;
    correction.image_id = owner.node_image_id;
    correction.owner_source_image_id = owner.source_image_id;
    correction.ownership_distance = owner.distance;
    correction.backbone_alpha = source->second.backbone_alpha;
    correction.delta = source->second.delta;
    correction.apply_correction =
        !IsIdentityOnlineDualSE3(correction.delta);
    if (source->second.forced_identity_by_older_window) {
      correction.kind =
          OnlineDualImageCorrectionKind::IDENTITY_OLDER_DOMAIN;
    } else if (correction.apply_correction) {
      correction.kind = OnlineDualImageCorrectionKind::PROPAGATED;
    } else {
      correction.kind = OnlineDualImageCorrectionKind::IDENTITY_TRANSFORM;
    }
    corrections.emplace(correction.image_id, correction);
  }

  std::vector<image_t> sorted_pose_only = pose_only_image_ids;
  std::sort(sorted_pose_only.begin(), sorted_pose_only.end());
  if (std::adjacent_find(sorted_pose_only.begin(), sorted_pose_only.end()) !=
      sorted_pose_only.end()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::DUPLICATE_POSE_ONLY_IMAGE,
              "POSE_ONLY image list contains a duplicate");
    return result;
  }
  for (const image_t image_id : sorted_pose_only) {
    if (!IsValidImageId(image_id) || frozen_graph.HasNode(image_id) ||
        corrections.count(image_id) != 0) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::POSE_ONLY_IMAGE_IN_ACTIVE_GRAPH,
                "POSE_ONLY image overlaps canonical active or DUAL window");
      return result;
    }
    OnlineDualImageCorrection correction;
    correction.image_id = image_id;
    correction.kind = OnlineDualImageCorrectionKind::POSE_ONLY;
    corrections.emplace(image_id, correction);
  }
  for (const auto& correction : corrections) {
    result.image_corrections.push_back(correction.second);
  }
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  return result;
}

OnlineDualPointLineageResult ResolveOnlineDualPointLineage(
    const std::vector<OnlineDualPointLineage>& lineages) {
  OnlineDualPointLineageResult result;
  std::set<point3D_t> point_ids;
  for (const OnlineDualPointLineage& lineage : lineages) {
    if (lineage.final_point3D_id == kInvalidPoint3DId ||
        lineage.ancestors.empty() ||
        !point_ids.insert(lineage.final_point3D_id).second) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_LINEAGE,
                "point lineage requires a unique final id and ancestors");
      return result;
    }
    bool has_direct = false;
    for (const OnlineDualPointLineageAncestor& ancestor : lineage.ancestors) {
      if (!IsValidImageId(ancestor.owner_image_id) ||
          !IsValidRegistrationSequence(
              ancestor.owner_registration_sequence)) {
        SetResult(&result,
                  OnlineDualResultClass::DUAL_FATAL,
                  OnlineDualReason::INVALID_LINEAGE,
                  "point lineage ancestor has an invalid owner");
        return result;
      }
      has_direct = has_direct || ancestor.directly_optimized;
    }
    const OnlineDualPointLineageAncestor* best = nullptr;
    for (const OnlineDualPointLineageAncestor& ancestor : lineage.ancestors) {
      if (has_direct && !ancestor.directly_optimized) {
        continue;
      }
      if (best == nullptr ||
          std::tie(ancestor.owner_registration_sequence,
                   ancestor.owner_image_id) <
              std::tie(best->owner_registration_sequence,
                       best->owner_image_id)) {
        best = &ancestor;
      }
    }
    OnlineDualResolvedPointOwner owner;
    owner.point3D_id = lineage.final_point3D_id;
    owner.owner_image_id = best->owner_image_id;
    owner.owner_registration_sequence = best->owner_registration_sequence;
    owner.directly_optimized = has_direct;
    result.owners.push_back(owner);
  }
  std::sort(result.owners.begin(),
            result.owners.end(),
            [](const OnlineDualResolvedPointOwner& lhs,
               const OnlineDualResolvedPointOwner& rhs) {
              return lhs.point3D_id < rhs.point3D_id;
            });
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  return result;
}

const char* ToString(const OnlineDualPointKind value) {
  switch (value) {
    case OnlineDualPointKind::VISUAL:
      return "VISUAL";
    case OnlineDualPointKind::LIDAR:
      return "LIDAR";
  }
  return "UNKNOWN";
}

const char* ToString(const OnlineDualPointCorrectionKind value) {
  switch (value) {
    case OnlineDualPointCorrectionKind::DIRECT_OPTIMIZED:
      return "DIRECT_OPTIMIZED";
    case OnlineDualPointCorrectionKind::OWNER_NEWER_WINDOW_DELTA:
      return "OWNER_NEWER_WINDOW_DELTA";
    case OnlineDualPointCorrectionKind::OWNER_PROPAGATED_DELTA:
      return "OWNER_PROPAGATED_DELTA";
    case OnlineDualPointCorrectionKind::OWNER_OLDER_WINDOW:
      return "OWNER_OLDER_WINDOW";
    case OnlineDualPointCorrectionKind::OWNER_IDENTITY_DOMAIN:
      return "OWNER_IDENTITY_DOMAIN";
    case OnlineDualPointCorrectionKind::OWNER_POSE_ONLY:
      return "OWNER_POSE_ONLY";
    case OnlineDualPointCorrectionKind::LIDAR_UNCHANGED:
      return "LIDAR_UNCHANGED";
  }
  return "UNKNOWN";
}

OnlineDualPointCorrectionPlan BuildOnlineDualPointCorrectionPlan(
    const OnlineDualWindowResult& window,
    const OnlineDualImageDeltaVector& window_deltas,
    const OnlineDualPropagationPlan& propagation,
    const std::vector<OnlineDualPointState>& points) {
  OnlineDualPointCorrectionPlan result;
  if (!window.IsSuccess() || !propagation.IsSuccess()) {
    SetResult(&result,
              OnlineDualResultClass::DUAL_FATAL,
              OnlineDualReason::INVALID_POINT_INPUT,
              "point planning requires valid window and propagation plans");
    return result;
  }
  OnlineDualSE3Map deltas;
  for (const OnlineDualImageDelta& delta : window_deltas) {
    if (!deltas.emplace(delta.image_id, delta.delta).second ||
        !IsValidOnlineDualSE3(delta.delta)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::INVALID_WINDOW_DELTA,
                "point planning received invalid or duplicate window Delta");
      return result;
    }
  }
  std::map<image_t, const OnlineDualImageCorrection*> image_corrections;
  for (const OnlineDualImageCorrection& correction :
       propagation.image_corrections) {
    image_corrections.emplace(correction.image_id, &correction);
  }
  std::set<point3D_t> point_ids;
  std::vector<OnlineDualPointState> sorted_points = points;
  std::sort(sorted_points.begin(),
            sorted_points.end(),
            [](const OnlineDualPointState& lhs,
               const OnlineDualPointState& rhs) {
              return lhs.point3D_id < rhs.point3D_id;
            });
  for (const OnlineDualPointState& point : sorted_points) {
    if (point.point3D_id == kInvalidPoint3DId ||
        !point_ids.insert(point.point3D_id).second) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::DUPLICATE_POINT,
                "point correction input has invalid or duplicate point id");
      return result;
    }
    OnlineDualPointCorrection correction;
    correction.point3D_id = point.point3D_id;
    correction.owner_image_id = point.owner_image_id;
    if (point.point_kind == OnlineDualPointKind::LIDAR) {
      correction.kind = OnlineDualPointCorrectionKind::LIDAR_UNCHANGED;
      result.corrections.push_back(correction);
      continue;
    }
    if (!IsValidImageId(point.owner_image_id) ||
        !IsValidRegistrationSequence(point.owner_registration_sequence)) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::MISSING_POINT_OWNER,
                "visual point has no valid frozen owner");
      return result;
    }
    if (point.directly_optimized) {
      correction.kind = OnlineDualPointCorrectionKind::DIRECT_OPTIMIZED;
      result.corrections.push_back(correction);
      continue;
    }
    if (point.owner_visual_state == ActiveCovisibilityNodeState::POSE_ONLY) {
      correction.kind = OnlineDualPointCorrectionKind::OWNER_POSE_ONLY;
      result.corrections.push_back(correction);
      continue;
    }
    const OnlineDualWindowImage* owner_window =
        FindWindowImage(window, point.owner_image_id);
    if (owner_window != nullptr) {
      if (owner_window->temporal_side == OnlineDualTemporalSide::OLDER) {
        correction.kind = OnlineDualPointCorrectionKind::OWNER_OLDER_WINDOW;
      } else {
        const auto delta = deltas.find(point.owner_image_id);
        if (delta == deltas.end()) {
          SetResult(&result,
                    OnlineDualResultClass::DUAL_FATAL,
                    OnlineDualReason::MISSING_WINDOW_DELTA,
                    "newer-window point owner lacks its actual Delta_i");
          return result;
        }
        correction.kind =
            OnlineDualPointCorrectionKind::OWNER_NEWER_WINDOW_DELTA;
        correction.delta = delta->second;
        correction.apply_correction =
            !IsIdentityOnlineDualSE3(correction.delta);
      }
      result.corrections.push_back(correction);
      continue;
    }
    const auto owner_correction = image_corrections.find(point.owner_image_id);
    if (owner_correction == image_corrections.end()) {
      SetResult(&result,
                OnlineDualResultClass::DUAL_FATAL,
                OnlineDualReason::MISSING_OWNER_CORRECTION,
                "active point owner has no propagated image correction");
      return result;
    }
    correction.delta = owner_correction->second->delta;
    correction.apply_correction =
        owner_correction->second->apply_correction;
    correction.kind = correction.apply_correction
                          ? OnlineDualPointCorrectionKind::
                                OWNER_PROPAGATED_DELTA
                          : OnlineDualPointCorrectionKind::
                                OWNER_IDENTITY_DOMAIN;
    result.corrections.push_back(correction);
  }
  SetResult(&result,
            OnlineDualResultClass::SUCCESS,
            OnlineDualReason::NONE,
            std::string());
  return result;
}

}  // namespace colmap
