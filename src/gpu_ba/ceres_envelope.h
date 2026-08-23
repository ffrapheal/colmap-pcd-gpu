#ifndef COLMAP_SRC_GPU_BA_CERES_ENVELOPE_H_
#define COLMAP_SRC_GPU_BA_CERES_ENVELOPE_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gpu_ba/ceres_fidelity.h"

namespace colmap {
namespace gpu_ba {

// The production envelope is deliberately separate from the direct pairwise
// fidelity comparison.  Direct comparison remains the deterministic oracle;
// this policy describes the bounded run-to-run variation of an explicitly
// supplied set of original multi-threaded Ceres runs.
struct CeresEnvelopePolicy {
  uint32_t schema_version = 0;
  std::string policy_id;
  uint64_t required_reference_samples = 0;
  double empirical_diameter_multiplier = 0.0;
  std::string joint_distance;
  std::string zero_diameter_rule;
  FidelityProvenanceSnapshot expected_reference_provenance;
  std::map<std::string, double> trace_absolute_caps;
  std::map<std::string, double> state_absolute_caps;
  std::string file_sha256;
};

struct CeresEnvelopeMetricResult {
  std::string name;
  double candidate_error = 0.0;
  double empirical_diameter = 0.0;
  double empirical_limit = 0.0;
  double absolute_cap = 0.0;
  double empirical_ratio = 0.0;
  double absolute_cap_ratio = 0.0;
  double joint_ratio = 0.0;
  double rms_absolute_error = 0.0;
  double p95_absolute_error = 0.0;
  std::string worst_id;
};

struct CeresEnvelopeCandidateResult {
  bool pass = false;
  bool integrity_pass = false;
  bool provenance_pass = false;
  bool problem_pass = false;
  bool source_order_pass = false;
  bool parameter_order_pass = false;
  bool id_sets_pass = false;
  bool loss_pass = false;
  bool constraints_pass = false;
  bool effective_options_pass = false;
  bool actual_options_pass = false;
  bool trace_structure_pass = false;
  bool termination_pass = false;
  bool numerical_envelope_pass = false;
  bool absolute_caps_pass = false;
  std::string candidate_run_id;
  std::string best_reference_run_id;
  double best_joint_ratio = 0.0;
  std::string worst_joint_metric;
  std::vector<std::string> failure_reasons;
  std::vector<CeresEnvelopeMetricResult> metrics;
};

struct CeresEnvelopeLeaveOneOutResult {
  bool pass = false;
  std::vector<CeresEnvelopeCandidateResult> folds;
};

struct LoadedCeresFidelitySample {
  std::string record_path;
  CeresFidelityRecord record;
  CeresFidelityWriteResult record_read;
  Snapshot solve_pre_state;
  SnapshotReadResult solve_pre_read;
  Snapshot post_state;
  SnapshotReadResult post_state_read;
};

bool ReadCeresEnvelopePolicy(const std::string& path,
                             CeresEnvelopePolicy* policy,
                             std::string* error);

bool LoadCeresFidelitySample(const std::string& record_path,
                             LoadedCeresFidelitySample* sample,
                             std::string* error);

bool ApplyCeresEnvelopePerturbation(
    const std::string& perturbation,
    LoadedCeresFidelitySample* candidate,
    std::string* error);

bool EvaluateCeresEnvelopeCandidate(
    const std::vector<LoadedCeresFidelitySample>& references,
    const LoadedCeresFidelitySample& candidate,
    const CeresEnvelopePolicy& policy,
    CeresEnvelopeCandidateResult* result,
    std::string* error);

bool EvaluateCeresEnvelopeLeaveOneOut(
    const std::vector<LoadedCeresFidelitySample>& samples,
    const CeresEnvelopePolicy& policy,
    CeresEnvelopeLeaveOneOutResult* result,
    std::string* error);

std::string CeresEnvelopeCandidateJson(
    const CeresEnvelopeCandidateResult& result,
    const CeresEnvelopePolicy& policy,
    size_t indent_spaces = 0);

std::string CeresEnvelopeLeaveOneOutJson(
    const CeresEnvelopeLeaveOneOutResult& result,
    const CeresEnvelopePolicy& policy,
    size_t indent_spaces = 0);

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_CERES_ENVELOPE_H_
