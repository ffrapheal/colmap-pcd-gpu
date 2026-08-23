#include "gpu_ba/ceres_envelope.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace gpu_ba {
namespace {

const char* const kTraceMetrics[] = {
    "cost",          "cost_change",       "gradient_max_norm",
    "gradient_norm", "step_norm",         "relative_decrease",
    "trust_region_radius", "eta",         "step_size"};

const char* const kStateMetrics[] = {
    "quaternion_ambient", "rotation_degrees", "translation_m",
    "point_m", "camera"};

std::string EscapeJson(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': stream << "\\\\"; break;
      case '"': stream << "\\\""; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (ch < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(ch) << std::dec;
        } else {
          stream << ch;
        }
    }
  }
  return stream.str();
}

bool ReadTextFile(const std::string& path,
                  std::string* contents,
                  std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    *error = "Cannot open Ceres envelope policy: " + path;
    return false;
  }
  contents->assign(std::istreambuf_iterator<char>(input),
                   std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    *error = "Cannot read complete Ceres envelope policy: " + path;
    return false;
  }
  return true;
}

bool IsSha256(const std::string& value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
  });
}

bool SameProvenance(const FidelityProvenanceSnapshot& lhs,
                    const FidelityProvenanceSnapshot& rhs) {
  return lhs.executable_sha256 == rhs.executable_sha256 &&
         lhs.libceres_sha256 == rhs.libceres_sha256 &&
         lhs.git_head == rhs.git_head &&
         lhs.dirty_diff_sha256 == rhs.dirty_diff_sha256;
}

bool ValidRecordedProvenance(const FidelityProvenanceSnapshot& value) {
  return !value.executable_path.empty() &&
         IsSha256(value.executable_sha256) && !value.libceres_path.empty() &&
         IsSha256(value.libceres_sha256) && !value.git_head.empty() &&
         !value.dirty_diff_sha256.empty();
}

template <typename SnapshotType, typename IdType>
bool UniqueIds(const std::vector<SnapshotType>& values,
               IdType SnapshotType::*member,
               const std::string& label,
               std::set<IdType>* ids,
               std::string* error) {
  ids->clear();
  for (const SnapshotType& value : values) {
    if (!ids->insert(value.*member).second) {
      *error = "duplicate_" + label + "_id=" +
               std::to_string(value.*member);
      return false;
    }
  }
  return true;
}

bool ValidateSnapshotIds(const Snapshot& snapshot, std::string* error) {
  std::set<uint32_t> camera_ids;
  std::set<uint32_t> image_ids;
  std::set<uint64_t> point_ids;
  return UniqueIds(snapshot.cameras, &CameraSnapshot::camera_id, "camera",
                   &camera_ids, error) &&
         UniqueIds(snapshot.images, &ImageSnapshot::image_id, "image",
                   &image_ids, error) &&
         UniqueIds(snapshot.points, &PointSnapshot::point3D_id, "point",
                   &point_ids, error);
}

bool SameIdSets(const Snapshot& reference,
                const Snapshot& candidate,
                std::string* reason) {
  std::set<uint32_t> ref_cameras;
  std::set<uint32_t> cand_cameras;
  std::set<uint32_t> ref_images;
  std::set<uint32_t> cand_images;
  std::set<uint64_t> ref_points;
  std::set<uint64_t> cand_points;
  std::string error;
  if (!UniqueIds(reference.cameras, &CameraSnapshot::camera_id, "camera",
                 &ref_cameras, &error) ||
      !UniqueIds(candidate.cameras, &CameraSnapshot::camera_id, "camera",
                 &cand_cameras, &error) ||
      !UniqueIds(reference.images, &ImageSnapshot::image_id, "image",
                 &ref_images, &error) ||
      !UniqueIds(candidate.images, &ImageSnapshot::image_id, "image",
                 &cand_images, &error) ||
      !UniqueIds(reference.points, &PointSnapshot::point3D_id, "point",
                 &ref_points, &error) ||
      !UniqueIds(candidate.points, &PointSnapshot::point3D_id, "point",
                 &cand_points, &error)) {
    *reason = error;
    return false;
  }
  if (ref_cameras != cand_cameras) {
    *reason = "camera_id_set_mismatch";
    return false;
  }
  if (ref_images != cand_images) {
    *reason = "image_id_set_mismatch";
    return false;
  }
  if (ref_points != cand_points) {
    *reason = "point_id_set_mismatch";
    return false;
  }
  return true;
}

bool SameOrderEntry(const OrderEntrySnapshot& lhs,
                    const OrderEntrySnapshot& rhs) {
  return lhs.source_index == rhs.source_index &&
         lhs.residual_kind == rhs.residual_kind &&
         lhs.image_id == rhs.image_id && lhs.point2D_idx == rhs.point2D_idx &&
         lhs.point3D_id == rhs.point3D_id;
}

bool SameResidualOrder(const Snapshot& lhs, const Snapshot& rhs) {
  if (lhs.source_insertion_order.size() != rhs.source_insertion_order.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.source_insertion_order.size(); ++i) {
    if (!SameOrderEntry(lhs.source_insertion_order[i],
                        rhs.source_insertion_order[i])) {
      return false;
    }
  }
  return true;
}

bool SameParameter(const ParameterBlockSnapshot& lhs,
                   const ParameterBlockSnapshot& rhs) {
  return lhs.source_index == rhs.source_index && lhs.kind == rhs.kind &&
         lhs.entity_id == rhs.entity_id &&
         lhs.ambient_size == rhs.ambient_size &&
         lhs.tangent_size == rhs.tangent_size && lhs.constant == rhs.constant;
}

bool SameParameterOrder(const Snapshot& lhs, const Snapshot& rhs) {
  if (lhs.parameter_blocks_source_order.size() !=
      rhs.parameter_blocks_source_order.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.parameter_blocks_source_order.size(); ++i) {
    if (!SameParameter(lhs.parameter_blocks_source_order[i],
                       rhs.parameter_blocks_source_order[i])) {
      return false;
    }
  }
  return true;
}

bool SameLosses(const std::vector<LossSpecificationSnapshot>& lhs,
                const std::vector<LossSpecificationSnapshot>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].residual_class != rhs[i].residual_class ||
        lhs[i].type != rhs[i].type || lhs[i].scale != rhs[i].scale ||
        lhs[i].residual_block_count != rhs[i].residual_block_count) {
      return false;
    }
  }
  return true;
}

bool SameConstraints(const std::vector<ParameterConstraintSnapshot>& lhs,
                     const std::vector<ParameterConstraintSnapshot>& rhs,
                     std::string* reason) {
  if (lhs.size() != rhs.size()) {
    *reason = "parameter_constraint_count_mismatch";
    return false;
  }
  std::set<std::pair<ParameterKind, uint64_t>> lhs_ids;
  std::set<std::pair<ParameterKind, uint64_t>> rhs_ids;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto lhs_id = std::make_pair(lhs[i].kind, lhs[i].entity_id);
    const auto rhs_id = std::make_pair(rhs[i].kind, rhs[i].entity_id);
    if (!lhs_ids.insert(lhs_id).second) {
      *reason = "duplicate_reference_parameter_id";
      return false;
    }
    if (!rhs_ids.insert(rhs_id).second) {
      *reason = "duplicate_candidate_parameter_id";
      return false;
    }
    if (lhs[i].kind != rhs[i].kind || lhs[i].entity_id != rhs[i].entity_id) {
      *reason = "parameter_id_or_first_insertion_order_mismatch";
      return false;
    }
    if (lhs[i].constant != rhs[i].constant ||
        lhs[i].local_parameterization != rhs[i].local_parameterization ||
        lhs[i].constant_indices != rhs[i].constant_indices) {
      *reason = "constant_or_subset_parameterization_mismatch";
      return false;
    }
  }
  if (lhs_ids != rhs_ids) {
    *reason = "parameter_id_set_mismatch";
    return false;
  }
  return true;
}

bool SameEffectiveOptions(const CeresFidelityRecord& lhs,
                          const CeresFidelityRecord& rhs) {
  return lhs.problem.effective_options_sha256 ==
             rhs.problem.effective_options_sha256 &&
         EffectiveCeresOptionsSha256(lhs.effective_options) ==
             EffectiveCeresOptionsSha256(rhs.effective_options);
}

bool ActualOptionsConsistent(const CeresFidelityRecord& value) {
  const CeresSummarySnapshot& summary = value.summary;
  const EffectiveCeresOptionsSnapshot& options = value.effective_options;
  return summary.minimizer_type == options.minimizer_type &&
         summary.trust_region_strategy_type ==
             options.trust_region_strategy_type &&
         summary.dogleg_type == options.dogleg_type &&
         summary.linear_solver_type_given == options.linear_solver_type &&
         summary.linear_solver_type_used == options.linear_solver_type &&
         summary.num_threads_given == options.num_threads &&
         summary.num_threads_used == options.num_threads &&
         summary.num_linear_solver_threads_given ==
             options.num_linear_solver_threads &&
         summary.num_linear_solver_threads_used ==
             options.num_linear_solver_threads &&
         summary.inner_iterations_given == options.use_inner_iterations &&
         summary.inner_iterations_used == options.use_inner_iterations;
}

std::string TerminationCategory(const CeresSummarySnapshot& summary) {
  std::string message = summary.termination_message;
  std::transform(message.begin(), message.end(), message.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (message.find("gradient") != std::string::npos) return "gradient";
  if (message.find("function") != std::string::npos) return "function";
  if (message.find("parameter") != std::string::npos) return "parameter";
  if (message.find("maximum") != std::string::npos) return "max_iterations";
  if (message.find("trust region radius") != std::string::npos) {
    return "min_trust_region_radius";
  }
  return std::to_string(summary.termination_type);
}

bool SameTraceStructure(const CeresSummarySnapshot& lhs,
                        const CeresSummarySnapshot& rhs,
                        std::string* reason) {
  if (lhs.iterations.size() != rhs.iterations.size()) {
    *reason = "trace_length_mismatch";
    return false;
  }
  for (size_t i = 0; i < lhs.iterations.size(); ++i) {
    const CeresIterationSnapshot& a = lhs.iterations[i];
    const CeresIterationSnapshot& b = rhs.iterations[i];
    if (a.iteration != b.iteration) {
      *reason = "trace_iteration_id_mismatch_at_index=" + std::to_string(i);
      return false;
    }
    if (a.step_is_valid != b.step_is_valid) {
      *reason = "trace_step_valid_mismatch_at_iteration=" +
                std::to_string(a.iteration);
      return false;
    }
    if (a.step_is_successful != b.step_is_successful) {
      *reason = "trace_step_success_mismatch_at_iteration=" +
                std::to_string(a.iteration);
      return false;
    }
    if (a.step_is_nonmonotonic != b.step_is_nonmonotonic) {
      *reason = "trace_nonmonotonic_mismatch_at_iteration=" +
                std::to_string(a.iteration);
      return false;
    }
    if (a.linear_solver_iterations != b.linear_solver_iterations) {
      *reason = "trace_linear_iterations_mismatch_at_iteration=" +
                std::to_string(a.iteration);
      return false;
    }
  }
  if (lhs.successful_steps != rhs.successful_steps) {
    *reason = "successful_step_count_mismatch";
    return false;
  }
  if (lhs.unsuccessful_steps != rhs.unsuccessful_steps) {
    *reason = "unsuccessful_step_count_mismatch";
    return false;
  }
  if (lhs.invalid_steps != rhs.invalid_steps) {
    *reason = "invalid_step_count_mismatch";
    return false;
  }
  if (lhs.num_linear_solves != rhs.num_linear_solves) {
    *reason = "linear_solve_count_mismatch";
    return false;
  }
  return true;
}

double IterationMetric(const CeresIterationSnapshot& value,
                       const std::string& name) {
  if (name == "cost") return value.cost;
  if (name == "cost_change") return value.cost_change;
  if (name == "gradient_max_norm") return value.gradient_max_norm;
  if (name == "gradient_norm") return value.gradient_norm;
  if (name == "step_norm") return value.step_norm;
  if (name == "relative_decrease") return value.relative_decrease;
  if (name == "trust_region_radius") return value.trust_region_radius;
  if (name == "eta") return value.eta;
  if (name == "step_size") return value.step_size;
  return std::numeric_limits<double>::quiet_NaN();
}

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = std::min(
      values.size() - 1,
      static_cast<size_t>(std::ceil(fraction * values.size()) - 1));
  return values[index];
}

struct PairMetric {
  double maximum = 0.0;
  double rms = 0.0;
  double p95 = 0.0;
  std::string worst_id;
};

using PairMetrics = std::map<std::string, PairMetric>;

bool ComputePairMetrics(const LoadedCeresFidelitySample& reference,
                        const LoadedCeresFidelitySample& candidate,
                        PairMetrics* metrics,
                        std::string* error) {
  metrics->clear();
  if (reference.record.summary.iterations.size() !=
      candidate.record.summary.iterations.size()) {
    *error = "trace_length_mismatch";
    return false;
  }
  for (const char* metric_name : kTraceMetrics) {
    std::vector<double> errors;
    errors.reserve(reference.record.summary.iterations.size());
    long double squared = 0.0;
    PairMetric metric;
    for (size_t i = 0; i < reference.record.summary.iterations.size(); ++i) {
      const double lhs = IterationMetric(
          reference.record.summary.iterations[i], metric_name);
      const double rhs = IterationMetric(
          candidate.record.summary.iterations[i], metric_name);
      if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        *error = "nonfinite_trace_metric=" + std::string(metric_name) +
                 " iteration=" + std::to_string(i);
        return false;
      }
      const double absolute = std::abs(lhs - rhs);
      errors.push_back(absolute);
      squared += static_cast<long double>(absolute) * absolute;
      if (absolute > metric.maximum || metric.worst_id.empty()) {
        metric.maximum = absolute;
        metric.worst_id = "iteration=" + std::to_string(i) + ":" +
                          metric_name;
      }
    }
    if (!errors.empty()) {
      metric.rms = std::sqrt(static_cast<double>(squared / errors.size()));
      metric.p95 = Percentile(errors, 0.95);
    }
    (*metrics)[metric_name] = metric;
  }

  CeresFidelityComparison comparison;
  std::string comparison_error;
  if (!CompareCeresFidelityRecords(
          reference.record, reference.post_state, candidate.record,
          candidate.post_state, &comparison, &comparison_error)) {
    *error = comparison_error;
    return false;
  }
  auto copy_state = [&](const std::string& name,
                        const FidelityErrorSummary& source) {
    PairMetric metric;
    metric.maximum = source.max_absolute_error;
    metric.rms = source.rms_absolute_error;
    metric.p95 = source.p95_absolute_error;
    metric.worst_id = source.worst_id;
    (*metrics)[name] = metric;
  };
  copy_state("quaternion_ambient", comparison.quaternion_ambient);
  copy_state("rotation_degrees", comparison.rotation_degrees);
  copy_state("translation_m", comparison.translation);
  copy_state("point_m", comparison.points);
  copy_state("camera", comparison.cameras);
  return true;
}

void AddFailure(const std::string& reason,
                CeresEnvelopeCandidateResult* result) {
  if (std::find(result->failure_reasons.begin(), result->failure_reasons.end(),
                reason) == result->failure_reasons.end()) {
    result->failure_reasons.push_back(reason);
  }
}

bool ValidateReferenceSet(
    const std::vector<LoadedCeresFidelitySample>& references,
    const CeresEnvelopePolicy& policy,
    bool leave_one_out_training,
    std::string* error) {
  const size_t expected = leave_one_out_training
                              ? policy.required_reference_samples - 1
                              : policy.required_reference_samples;
  if (references.size() != expected || references.empty()) {
    *error = "reference_sample_count_mismatch: expected=" +
             std::to_string(expected) + " actual=" +
             std::to_string(references.size());
    return false;
  }
  std::set<std::string> run_ids;
  const LoadedCeresFidelitySample& first = references.front();
  for (const LoadedCeresFidelitySample& sample : references) {
    if (!run_ids.insert(sample.record.run_id).second) {
      *error = "duplicate_reference_run_id=" + sample.record.run_id;
      return false;
    }
    if (!ValidRecordedProvenance(sample.record.provenance) ||
        !SameProvenance(sample.record.provenance,
                        policy.expected_reference_provenance)) {
      *error = "reference_provenance_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    if (sample.record.snapshot_id != first.record.snapshot_id ||
        sample.record.snapshot_payload_sha256 !=
            first.record.snapshot_payload_sha256 ||
        sample.record.problem.complete_fingerprint_sha256 !=
            first.record.problem.complete_fingerprint_sha256) {
      *error = "reference_problem_fingerprint_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    if (!SameResidualOrder(first.solve_pre_state, sample.solve_pre_state) ||
        sample.record.problem.source_residual_order_sha256 !=
            first.record.problem.source_residual_order_sha256) {
      *error = "reference_source_residual_order_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    if (!SameParameterOrder(first.solve_pre_state, sample.solve_pre_state) ||
        sample.record.problem.source_parameter_order_sha256 !=
            first.record.problem.source_parameter_order_sha256) {
      *error = "reference_parameter_first_insertion_order_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    std::string id_reason;
    if (!SameIdSets(first.solve_pre_state, sample.solve_pre_state, &id_reason) ||
        !SameIdSets(first.post_state, sample.post_state, &id_reason)) {
      *error = "reference_" + id_reason + " run_id=" + sample.record.run_id;
      return false;
    }
    if (!SameLosses(first.record.problem.loss_specifications,
                    sample.record.problem.loss_specifications)) {
      *error = "reference_loss_mismatch run_id=" + sample.record.run_id;
      return false;
    }
    std::string constraint_reason;
    if (!SameConstraints(first.record.problem.parameter_constraints,
                         sample.record.problem.parameter_constraints,
                         &constraint_reason)) {
      *error = "reference_" + constraint_reason + " run_id=" +
               sample.record.run_id;
      return false;
    }
    if (!SameEffectiveOptions(first.record, sample.record)) {
      *error = "reference_effective_options_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    if (!ActualOptionsConsistent(sample.record)) {
      *error = "reference_actual_options_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
    std::string trace_reason;
    if (!SameTraceStructure(first.record.summary, sample.record.summary,
                            &trace_reason)) {
      *error = "reference_" + trace_reason + " run_id=" +
               sample.record.run_id;
      return false;
    }
    if (first.record.summary.termination_type !=
            sample.record.summary.termination_type ||
        TerminationCategory(first.record.summary) !=
            TerminationCategory(sample.record.summary)) {
      *error = "reference_termination_mismatch run_id=" +
               sample.record.run_id;
      return false;
    }
  }
  return true;
}

bool Ratio(double error, double limit, double* ratio) {
  if (limit == 0.0) {
    *ratio = error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  } else {
    *ratio = error / limit;
  }
  return std::isfinite(*ratio) || std::isinf(*ratio);
}

std::string PolicyJson(const CeresEnvelopePolicy& policy) {
  std::ostringstream stream;
  stream << std::setprecision(17)
         << "{\"schema_version\":" << policy.schema_version
         << ",\"policy_id\":\"" << EscapeJson(policy.policy_id)
         << "\",\"file_sha256\":\"" << policy.file_sha256
         << "\",\"required_reference_samples\":"
         << policy.required_reference_samples
         << ",\"empirical_diameter_multiplier\":"
         << policy.empirical_diameter_multiplier
         << ",\"joint_distance\":\"" << EscapeJson(policy.joint_distance)
         << "\",\"zero_diameter_rule\":\""
         << EscapeJson(policy.zero_diameter_rule) << "\"}";
  return stream.str();
}

}  // namespace

bool ReadCeresEnvelopePolicy(const std::string& path,
                             CeresEnvelopePolicy* policy,
                             std::string* error) {
  if (policy == nullptr || error == nullptr) return false;
  *policy = CeresEnvelopePolicy();
  std::string contents;
  if (!ReadTextFile(path, &contents, error)) return false;
  boost::property_tree::ptree root;
  try {
    std::istringstream stream(contents);
    boost::property_tree::read_json(stream, root);
    policy->schema_version = root.get<uint32_t>("schema_version");
    policy->policy_id = root.get<std::string>("policy_id");
    policy->required_reference_samples =
        root.get<uint64_t>("required_reference_samples");
    policy->empirical_diameter_multiplier =
        root.get<double>("empirical_diameter_multiplier");
    policy->joint_distance = root.get<std::string>("joint_distance");
    policy->zero_diameter_rule =
        root.get<std::string>("zero_diameter_rule");
    const auto& provenance = root.get_child("expected_reference_provenance");
    policy->expected_reference_provenance.executable_path =
        provenance.get<std::string>("executable_path");
    policy->expected_reference_provenance.executable_sha256 =
        provenance.get<std::string>("executable_sha256");
    policy->expected_reference_provenance.libceres_path =
        provenance.get<std::string>("libceres_path");
    policy->expected_reference_provenance.libceres_sha256 =
        provenance.get<std::string>("libceres_sha256");
    policy->expected_reference_provenance.git_head =
        provenance.get<std::string>("git_head");
    policy->expected_reference_provenance.dirty_diff_sha256 =
        provenance.get<std::string>("dirty_diff_sha256");
    for (const auto& item : root.get_child("trace_absolute_caps")) {
      policy->trace_absolute_caps.emplace(item.first,
                                           item.second.get_value<double>());
    }
    for (const auto& item : root.get_child("state_absolute_caps")) {
      policy->state_absolute_caps.emplace(item.first,
                                           item.second.get_value<double>());
    }
  } catch (const std::exception& exception) {
    *error = "Invalid Ceres envelope policy JSON: " +
             std::string(exception.what());
    return false;
  }
  policy->file_sha256 = Sha256Hex(contents);
  if (policy->schema_version != 1 || policy->policy_id.empty() ||
      policy->required_reference_samples != 8 ||
      !std::isfinite(policy->empirical_diameter_multiplier) ||
      policy->empirical_diameter_multiplier < 1.0 ||
      policy->joint_distance != "single_reference_joint_linf" ||
      policy->zero_diameter_rule != "exact") {
    *error = "Unsupported or invalid Ceres envelope policy semantics";
    return false;
  }
  if (!ValidRecordedProvenance(policy->expected_reference_provenance)) {
    *error = "Ceres envelope policy has invalid reference provenance";
    return false;
  }
  for (const char* name : kTraceMetrics) {
    const auto it = policy->trace_absolute_caps.find(name);
    if (it == policy->trace_absolute_caps.end() || !std::isfinite(it->second) ||
        it->second < 0.0) {
      *error = "Missing or invalid trace absolute cap: " + std::string(name);
      return false;
    }
  }
  for (const char* name : kStateMetrics) {
    const auto it = policy->state_absolute_caps.find(name);
    if (it == policy->state_absolute_caps.end() || !std::isfinite(it->second) ||
        it->second < 0.0) {
      *error = "Missing or invalid state absolute cap: " + std::string(name);
      return false;
    }
  }
  if (policy->trace_absolute_caps.size() !=
          sizeof(kTraceMetrics) / sizeof(kTraceMetrics[0]) ||
      policy->state_absolute_caps.size() !=
          sizeof(kStateMetrics) / sizeof(kStateMetrics[0])) {
    *error = "Ceres envelope policy contains an unknown metric";
    return false;
  }
  return true;
}

bool LoadCeresFidelitySample(const std::string& record_path,
                             LoadedCeresFidelitySample* sample,
                             std::string* error) {
  if (sample == nullptr || error == nullptr) return false;
  *sample = LoadedCeresFidelitySample();
  sample->record_path = record_path;
  if (!ReadCeresFidelityRecord(record_path, &sample->record,
                               &sample->record_read, error) ||
      !ReadSnapshot(sample->record.snapshot_manifest_path,
                    &sample->solve_pre_state, &sample->solve_pre_read,
                    error) ||
      !ReadSnapshot(sample->record.post_state_manifest_path,
                    &sample->post_state, &sample->post_state_read, error)) {
    return false;
  }
  if (sample->solve_pre_state.metadata.snapshot_id !=
          sample->record.snapshot_id ||
      sample->solve_pre_read.integrity.payload_sha256 !=
          sample->record.snapshot_payload_sha256 ||
      sample->solve_pre_read.integrity.manifest_sha256 !=
          sample->record.snapshot_manifest_sha256 ||
      sample->post_state_read.integrity.payload_sha256 !=
          sample->record.post_state_payload_sha256 ||
      CanonicalStateSha256(sample->post_state) !=
          sample->record.final_state_sha256) {
    *error = "Ceres fidelity sample input/post-state integrity mismatch: " +
             record_path;
    return false;
  }
  if (!ValidateSnapshotIds(sample->solve_pre_state, error) ||
      !ValidateSnapshotIds(sample->post_state, error)) {
    return false;
  }
  return true;
}

bool ApplyCeresEnvelopePerturbation(
    const std::string& perturbation,
    LoadedCeresFidelitySample* candidate,
    std::string* error) {
  if (candidate == nullptr || error == nullptr) return false;
  if (perturbation.empty() || perturbation == "none") return true;
  if (perturbation == "option_max_linear_plus_one") {
    ++candidate->record.effective_options.max_linear_solver_iterations;
    return true;
  }
  if (perturbation == "loss_scale_plus_0_1") {
    if (candidate->record.problem.loss_specifications.empty()) {
      *error = "Cannot perturb an empty loss specification";
      return false;
    }
    candidate->record.problem.loss_specifications.front().scale += 0.1;
    return true;
  }
  if (perturbation == "swap_first_two_residuals") {
    if (candidate->solve_pre_state.source_insertion_order.size() < 2) {
      *error = "Cannot perturb fewer than two residuals";
      return false;
    }
    std::swap(candidate->solve_pre_state.source_insertion_order[0],
              candidate->solve_pre_state.source_insertion_order[1]);
    candidate->record.problem.source_residual_order_sha256[0] =
        candidate->record.problem.source_residual_order_sha256[0] == '0'
            ? '1'
            : '0';
    return true;
  }
  if (perturbation == "parameter_id_plus_one") {
    if (candidate->record.problem.parameter_constraints.empty() ||
        candidate->solve_pre_state.parameter_blocks_source_order.empty()) {
      *error = "Cannot perturb an empty parameter order";
      return false;
    }
    ++candidate->record.problem.parameter_constraints.front().entity_id;
    ++candidate->solve_pre_state.parameter_blocks_source_order.front()
          .entity_id;
    return true;
  }
  auto perturb_trace = [&](const std::string& field) {
    if (candidate->record.summary.iterations.size() < 2) return false;
    CeresIterationSnapshot& item = candidate->record.summary.iterations[1];
    if (field == "cost") item.cost += 1e-3;
    if (field == "gradient") item.gradient_max_norm += 1e-2;
    if (field == "radius") item.trust_region_radius += 1.0;
    return true;
  };
  if (perturbation == "trace_cost_plus_1e_3") {
    if (!perturb_trace("cost")) *error = "Cannot perturb trace cost";
    return error->empty();
  }
  if (perturbation == "trace_gradient_plus_1e_2") {
    if (!perturb_trace("gradient")) *error = "Cannot perturb trace gradient";
    return error->empty();
  }
  if (perturbation == "trace_radius_plus_one") {
    if (!perturb_trace("radius")) *error = "Cannot perturb trace radius";
    return error->empty();
  }
  if (perturbation == "final_pose_translation_plus_1e_3") {
    if (candidate->post_state.images.empty()) {
      *error = "Cannot perturb an empty final pose set";
      return false;
    }
    candidate->post_state.images.front().tvec[0] += 1e-3;
    return true;
  }
  *error = "Unknown Ceres envelope perturbation: " + perturbation;
  return false;
}

bool EvaluateCeresEnvelopeCandidate(
    const std::vector<LoadedCeresFidelitySample>& references,
    const LoadedCeresFidelitySample& candidate,
    const CeresEnvelopePolicy& policy,
    CeresEnvelopeCandidateResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = CeresEnvelopeCandidateResult();
  result->candidate_run_id = candidate.record.run_id;
  const bool loo_training =
      references.size() + 1 == policy.required_reference_samples;
  if (!ValidateReferenceSet(references, policy, loo_training, error)) {
    return false;
  }
  const LoadedCeresFidelitySample& base = references.front();
  result->integrity_pass = true;
  result->provenance_pass = ValidRecordedProvenance(candidate.record.provenance);
  if (!result->provenance_pass) AddFailure("candidate_provenance_invalid", result);

  result->problem_pass =
      candidate.record.snapshot_id == base.record.snapshot_id &&
      candidate.record.snapshot_payload_sha256 ==
          base.record.snapshot_payload_sha256 &&
      candidate.record.problem.complete_fingerprint_sha256 ==
          base.record.problem.complete_fingerprint_sha256;
  if (!result->problem_pass) AddFailure("problem_fingerprint_mismatch", result);

  result->source_order_pass =
      SameResidualOrder(base.solve_pre_state, candidate.solve_pre_state) &&
      candidate.record.problem.source_residual_order_sha256 ==
          base.record.problem.source_residual_order_sha256;
  if (!result->source_order_pass) {
    AddFailure("source_residual_order_mismatch", result);
  }
  result->parameter_order_pass =
      SameParameterOrder(base.solve_pre_state, candidate.solve_pre_state) &&
      candidate.record.problem.source_parameter_order_sha256 ==
          base.record.problem.source_parameter_order_sha256;
  if (!result->parameter_order_pass) {
    AddFailure("parameter_first_insertion_order_mismatch", result);
  }
  std::string id_reason;
  result->id_sets_pass =
      SameIdSets(base.solve_pre_state, candidate.solve_pre_state, &id_reason) &&
      SameIdSets(base.post_state, candidate.post_state, &id_reason);
  if (!result->id_sets_pass) AddFailure(id_reason, result);

  result->loss_pass = SameLosses(base.record.problem.loss_specifications,
                                 candidate.record.problem.loss_specifications);
  if (!result->loss_pass) AddFailure("loss_type_or_scale_mismatch", result);
  std::string constraint_reason;
  result->constraints_pass = SameConstraints(
      base.record.problem.parameter_constraints,
      candidate.record.problem.parameter_constraints, &constraint_reason);
  if (!result->constraints_pass) AddFailure(constraint_reason, result);
  result->effective_options_pass = SameEffectiveOptions(base.record, candidate.record);
  if (!result->effective_options_pass) {
    AddFailure("requested_or_effective_options_mismatch", result);
  }
  result->actual_options_pass = ActualOptionsConsistent(candidate.record);
  if (!result->actual_options_pass) AddFailure("actual_options_mismatch", result);
  std::string trace_reason;
  result->trace_structure_pass = SameTraceStructure(
      base.record.summary, candidate.record.summary, &trace_reason);
  if (!result->trace_structure_pass) AddFailure(trace_reason, result);
  result->termination_pass =
      base.record.summary.termination_type ==
          candidate.record.summary.termination_type &&
      TerminationCategory(base.record.summary) ==
          TerminationCategory(candidate.record.summary);
  if (!result->termination_pass) AddFailure("termination_mismatch", result);

  const bool exact_structure = result->problem_pass &&
      result->source_order_pass && result->parameter_order_pass &&
      result->id_sets_pass && result->loss_pass && result->constraints_pass &&
      result->effective_options_pass && result->actual_options_pass &&
      result->trace_structure_pass && result->termination_pass;
  if (!exact_structure) {
    result->pass = false;
    return true;
  }

  std::map<std::string, double> diameters;
  for (const char* name : kTraceMetrics) diameters[name] = 0.0;
  for (const char* name : kStateMetrics) diameters[name] = 0.0;
  for (size_t i = 0; i < references.size(); ++i) {
    for (size_t j = i + 1; j < references.size(); ++j) {
      PairMetrics pair;
      if (!ComputePairMetrics(references[i], references[j], &pair, error)) {
        return false;
      }
      for (const auto& item : pair) {
        diameters[item.first] = std::max(diameters[item.first],
                                         item.second.maximum);
      }
    }
  }

  double best_ratio = std::numeric_limits<double>::infinity();
  std::string best_reference;
  std::string best_worst_metric;
  std::vector<CeresEnvelopeMetricResult> best_metrics;
  for (const LoadedCeresFidelitySample& reference : references) {
    PairMetrics errors;
    if (!ComputePairMetrics(reference, candidate, &errors, error)) return false;
    double joint_ratio = 0.0;
    std::string worst_metric;
    std::vector<CeresEnvelopeMetricResult> metric_results;
    for (const auto& item : errors) {
      CeresEnvelopeMetricResult metric;
      metric.name = item.first;
      metric.candidate_error = item.second.maximum;
      metric.empirical_diameter = diameters.at(item.first);
      metric.empirical_limit = policy.empirical_diameter_multiplier *
                               metric.empirical_diameter;
      const auto trace_cap = policy.trace_absolute_caps.find(item.first);
      metric.absolute_cap = trace_cap != policy.trace_absolute_caps.end()
                                ? trace_cap->second
                                : policy.state_absolute_caps.at(item.first);
      Ratio(metric.candidate_error, metric.empirical_limit,
            &metric.empirical_ratio);
      Ratio(metric.candidate_error, metric.absolute_cap,
            &metric.absolute_cap_ratio);
      metric.joint_ratio =
          std::max(metric.empirical_ratio, metric.absolute_cap_ratio);
      metric.rms_absolute_error = item.second.rms;
      metric.p95_absolute_error = item.second.p95;
      metric.worst_id = item.second.worst_id;
      if (metric.joint_ratio > joint_ratio || worst_metric.empty()) {
        joint_ratio = metric.joint_ratio;
        worst_metric = metric.name;
      }
      metric_results.push_back(metric);
    }
    if (joint_ratio < best_ratio || best_reference.empty()) {
      best_ratio = joint_ratio;
      best_reference = reference.record.run_id;
      best_worst_metric = worst_metric;
      best_metrics = std::move(metric_results);
    }
  }
  result->best_reference_run_id = best_reference;
  result->best_joint_ratio = best_ratio;
  result->worst_joint_metric = best_worst_metric;
  result->metrics = std::move(best_metrics);
  result->numerical_envelope_pass = true;
  result->absolute_caps_pass = true;
  for (const CeresEnvelopeMetricResult& metric : result->metrics) {
    if (metric.empirical_ratio > 1.0) result->numerical_envelope_pass = false;
    if (metric.absolute_cap_ratio > 1.0) result->absolute_caps_pass = false;
  }
  if (!result->numerical_envelope_pass) {
    AddFailure("empirical_envelope_exceeded:" + result->worst_joint_metric,
               result);
  }
  if (!result->absolute_caps_pass) {
    AddFailure("fixed_absolute_cap_exceeded:" + result->worst_joint_metric,
               result);
  }
  result->pass = result->integrity_pass && result->provenance_pass &&
                 exact_structure && result->numerical_envelope_pass &&
                 result->absolute_caps_pass;
  return true;
}

bool EvaluateCeresEnvelopeLeaveOneOut(
    const std::vector<LoadedCeresFidelitySample>& samples,
    const CeresEnvelopePolicy& policy,
    CeresEnvelopeLeaveOneOutResult* result,
    std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = CeresEnvelopeLeaveOneOutResult();
  if (samples.size() != policy.required_reference_samples) {
    *error = "leave_one_out_requires_exact_reference_sample_count";
    return false;
  }
  for (size_t held_out = 0; held_out < samples.size(); ++held_out) {
    std::vector<LoadedCeresFidelitySample> training;
    training.reserve(samples.size() - 1);
    for (size_t i = 0; i < samples.size(); ++i) {
      if (i != held_out) training.push_back(samples[i]);
    }
    CeresEnvelopeCandidateResult fold;
    if (!EvaluateCeresEnvelopeCandidate(training, samples[held_out], policy,
                                        &fold, error)) {
      return false;
    }
    result->folds.push_back(std::move(fold));
  }
  result->pass = std::all_of(result->folds.begin(), result->folds.end(),
                             [](const CeresEnvelopeCandidateResult& fold) {
                               return fold.pass;
                             });
  return true;
}

std::string CeresEnvelopeCandidateJson(
    const CeresEnvelopeCandidateResult& result,
    const CeresEnvelopePolicy& policy,
    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string i2(indent_spaces + 2, ' ');
  const std::string i4(indent_spaces + 4, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17) << indent << "{\n"
         << i2 << "\"pass\":" << (result.pass ? "true" : "false")
         << ",\"candidate_run_id\":\""
         << EscapeJson(result.candidate_run_id)
         << "\",\"best_reference_run_id\":\""
         << EscapeJson(result.best_reference_run_id)
         << "\",\"best_joint_ratio\":" << result.best_joint_ratio
         << ",\"worst_joint_metric\":\""
         << EscapeJson(result.worst_joint_metric) << "\",\n"
         << i2 << "\"policy\":" << PolicyJson(policy) << ",\n"
         << i2 << "\"gates\":{\"integrity\":"
         << (result.integrity_pass ? "true" : "false")
         << ",\"provenance\":"
         << (result.provenance_pass ? "true" : "false")
         << ",\"problem\":" << (result.problem_pass ? "true" : "false")
         << ",\"source_order\":"
         << (result.source_order_pass ? "true" : "false")
         << ",\"parameter_order\":"
         << (result.parameter_order_pass ? "true" : "false")
         << ",\"id_sets\":" << (result.id_sets_pass ? "true" : "false")
         << ",\"loss\":" << (result.loss_pass ? "true" : "false")
         << ",\"constraints\":"
         << (result.constraints_pass ? "true" : "false")
         << ",\"effective_options\":"
         << (result.effective_options_pass ? "true" : "false")
         << ",\"actual_options\":"
         << (result.actual_options_pass ? "true" : "false")
         << ",\"trace_structure\":"
         << (result.trace_structure_pass ? "true" : "false")
         << ",\"termination\":"
         << (result.termination_pass ? "true" : "false")
         << ",\"numerical_envelope\":"
         << (result.numerical_envelope_pass ? "true" : "false")
         << ",\"absolute_caps\":"
         << (result.absolute_caps_pass ? "true" : "false") << "},\n"
         << i2 << "\"failure_reasons\":[";
  for (size_t i = 0; i < result.failure_reasons.size(); ++i) {
    if (i != 0) stream << ',';
    stream << '"' << EscapeJson(result.failure_reasons[i]) << '"';
  }
  stream << "],\n" << i2 << "\"metrics\":[\n";
  for (size_t i = 0; i < result.metrics.size(); ++i) {
    const CeresEnvelopeMetricResult& metric = result.metrics[i];
    stream << i4 << "{\"name\":\"" << metric.name
           << "\",\"candidate_error\":" << metric.candidate_error
           << ",\"empirical_diameter\":" << metric.empirical_diameter
           << ",\"empirical_limit\":" << metric.empirical_limit
           << ",\"absolute_cap\":" << metric.absolute_cap
           << ",\"empirical_ratio\":" << metric.empirical_ratio
           << ",\"absolute_cap_ratio\":" << metric.absolute_cap_ratio
           << ",\"joint_ratio\":" << metric.joint_ratio
           << ",\"rms_absolute_error\":" << metric.rms_absolute_error
           << ",\"p95_absolute_error\":" << metric.p95_absolute_error
           << ",\"worst_id\":\"" << EscapeJson(metric.worst_id)
           << "\"}";
    if (i + 1 != result.metrics.size()) stream << ',';
    stream << '\n';
  }
  stream << i2 << "]\n" << indent << '}';
  return stream.str();
}

std::string CeresEnvelopeLeaveOneOutJson(
    const CeresEnvelopeLeaveOneOutResult& result,
    const CeresEnvelopePolicy& policy,
    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string i2(indent_spaces + 2, ' ');
  std::ostringstream stream;
  stream << indent << "{\n" << i2 << "\"pass\":"
         << (result.pass ? "true" : "false")
         << ",\"mode\":\"leave_one_out\",\"policy\":"
         << PolicyJson(policy) << ",\"folds\":[\n";
  for (size_t i = 0; i < result.folds.size(); ++i) {
    stream << CeresEnvelopeCandidateJson(result.folds[i], policy,
                                         indent_spaces + 4);
    if (i + 1 != result.folds.size()) stream << ',';
    stream << '\n';
  }
  stream << i2 << "]\n" << indent << '}';
  return stream.str();
}

}  // namespace gpu_ba
}  // namespace colmap
