#include "exe/gpu_ba.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "gpu_ba/ceres_envelope.h"
#include "gpu_ba/fixed_linearization.h"
#include "gpu_ba/ceres_fidelity.h"
#include "gpu_ba/snapshot.h"
#include "gpu_ba/validation.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {
namespace {

bool ValidateOrder(const gpu_ba::Snapshot& snapshot, std::string* error) {
  const size_t num_residuals = snapshot.source_insertion_order.size();
  if (snapshot.canonical_order.size() != num_residuals ||
      snapshot.observations.size() + snapshot.lidar.size() != num_residuals) {
    *error = "Residual order counts do not agree";
    return false;
  }
  std::set<uint64_t> source_indices;
  std::set<uint64_t> canonical_indices;
  for (size_t i = 0; i < num_residuals; ++i) {
    if (snapshot.source_insertion_order[i].source_index != i) {
      *error = "Source insertion indices are not contiguous";
      return false;
    }
    source_indices.insert(snapshot.source_insertion_order[i].source_index);
    canonical_indices.insert(snapshot.canonical_order[i].source_index);
  }
  if (source_indices != canonical_indices) {
    *error = "Canonical residual order is not a source-order permutation";
    return false;
  }

  const size_t num_parameters =
      snapshot.parameter_blocks_source_order.size();
  if (snapshot.parameter_blocks_canonical_order.size() != num_parameters) {
    *error = "Canonical parameter order count does not agree";
    return false;
  }
  std::set<uint64_t> parameter_indices;
  for (const uint64_t index : snapshot.parameter_blocks_canonical_order) {
    if (index >= num_parameters || !parameter_indices.insert(index).second) {
      *error = "Canonical parameter order is not a permutation";
      return false;
    }
  }
  return true;
}

std::string SolvedStateSha256(const gpu_ba::Snapshot& state) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "gpu-ba-solved-state-v1\n";
  for (const gpu_ba::ImageSnapshot& image : state.images) {
    stream << "image " << image.image_id;
    for (const double value : image.qvec) stream << ' ' << value;
    for (const double value : image.tvec) stream << ' ' << value;
    stream << '\n';
  }
  for (const gpu_ba::PointSnapshot& point : state.points) {
    stream << "point " << point.point3D_id;
    for (const double value : point.xyz) stream << ' ' << value;
    stream << '\n';
  }
  return gpu_ba::Sha256Hex(stream.str());
}

std::vector<std::string> SplitCommaSeparated(const std::string& value) {
  std::vector<std::string> output;
  std::istringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) output.push_back(item);
  }
  return output;
}

bool LoadCustomCpuOptionsFromOracle(
    const std::string& oracle_path,
    const gpu_ba::Snapshot& snapshot,
    const gpu_ba::SnapshotReadResult& snapshot_read,
    gpu_ba::CeresFidelityRecord* oracle,
    gpu_ba::CustomCpuSolverOptions* output,
    std::string* error) {
  gpu_ba::CeresFidelityWriteResult oracle_read;
  if (!gpu_ba::ReadCeresFidelityRecord(oracle_path, oracle, &oracle_read,
                                       error)) {
    return false;
  }
  if (oracle->snapshot_id != snapshot.metadata.snapshot_id ||
      oracle->snapshot_payload_sha256 != snapshot_read.integrity.payload_sha256 ||
      oracle->problem.initial_state_sha256 !=
          gpu_ba::CanonicalStateSha256(snapshot) ||
      oracle->problem.source_residual_order_sha256 !=
          snapshot_read.integrity.source_order_sha256 ||
      oracle->problem.lidar_correspondence_sha256 !=
          snapshot_read.integrity.lidar_correspondence_sha256) {
    *error = "Oracle does not identify the supplied solve-pre snapshot";
    return false;
  }
  if (oracle->problem.selected_image_count !=
          static_cast<uint64_t>(std::count_if(
              snapshot.images.begin(), snapshot.images.end(),
              [](const gpu_ba::ImageSnapshot& image) {
                return image.selected;
              })) ||
      oracle->problem.residual_block_count !=
          snapshot.source_insertion_order.size()) {
    *error = "Oracle problem counts differ from solve-pre snapshot";
    return false;
  }
  *output = gpu_ba::CustomCpuSolverOptions();
  const auto& source = oracle->effective_options;
  output->linear_solver_type = ceres::LinearSolverTypeToString(
      static_cast<ceres::LinearSolverType>(source.linear_solver_type));
  output->requested_num_threads = source.original_requested_num_threads;
  output->effective_num_threads = source.num_threads;
  output->max_num_iterations = source.max_num_iterations;
  output->min_linear_solver_iterations = source.min_linear_solver_iterations;
  output->max_linear_solver_iterations = source.max_linear_solver_iterations;
  output->max_num_consecutive_invalid_steps =
      source.max_num_consecutive_invalid_steps;
  output->max_solver_time_in_seconds = source.max_solver_time_in_seconds;
  output->function_tolerance = source.function_tolerance;
  output->gradient_tolerance = source.gradient_tolerance;
  output->parameter_tolerance = source.parameter_tolerance;
  output->initial_trust_region_radius = source.initial_trust_region_radius;
  output->min_trust_region_radius = source.min_trust_region_radius;
  output->max_trust_region_radius = source.max_trust_region_radius;
  output->min_lm_diagonal = source.min_lm_diagonal;
  output->max_lm_diagonal = source.max_lm_diagonal;
  output->min_relative_decrease = source.min_relative_decrease;
  output->eta = source.eta;
  output->jacobi_scaling = source.jacobi_scaling;
  output->use_nonmonotonic_steps = source.use_nonmonotonic_steps;
  output->use_inner_iterations = source.use_inner_iterations;

  bool found_loss = false;
  uint64_t loss_residual_count = 0;
  for (const gpu_ba::LossSpecificationSnapshot& specification :
       oracle->problem.loss_specifications) {
    if (!found_loss) {
      output->loss_type = specification.type;
      output->loss_scale = specification.scale;
      found_loss = true;
    } else if (output->loss_type != specification.type ||
               output->loss_scale != specification.scale) {
      *error = "custom_cpu requires one common captured loss for all residuals";
      return false;
    }
    loss_residual_count += specification.residual_block_count;
  }
  std::string snapshot_loss = snapshot.metadata.loss_function;
  std::transform(snapshot_loss.begin(), snapshot_loss.end(),
                 snapshot_loss.begin(),
                 [](unsigned char value) { return std::toupper(value); });
  if (!found_loss ||
      loss_residual_count != snapshot.source_insertion_order.size() ||
      output->loss_type != snapshot_loss) {
    *error = "Oracle loss specification differs from solve-pre snapshot";
    return false;
  }
  return true;
}

int32_t CustomTerminationAsCeres(
    gpu_ba::CpuTerminationType termination_type) {
  switch (termination_type) {
    case gpu_ba::CpuTerminationType::kConvergence:
      return static_cast<int32_t>(ceres::CONVERGENCE);
    case gpu_ba::CpuTerminationType::kNoConvergence:
      return static_cast<int32_t>(ceres::NO_CONVERGENCE);
    case gpu_ba::CpuTerminationType::kFailure:
      return static_cast<int32_t>(ceres::FAILURE);
  }
  return static_cast<int32_t>(ceres::FAILURE);
}

gpu_ba::CeresSummarySnapshot BuildCustomCpuCompatibilitySummary(
    const gpu_ba::CeresFidelityRecord& oracle,
    const gpu_ba::CustomCpuSolveResult& custom) {
  gpu_ba::CeresSummarySnapshot summary = oracle.summary;
  summary.termination_type = CustomTerminationAsCeres(custom.termination_type);
  summary.termination_message = custom.termination_reason;
  summary.initial_cost = custom.initial_cost;
  summary.final_cost = custom.final_cost;
  summary.successful_steps = custom.accepted_steps;
  summary.unsuccessful_steps = custom.rejected_steps;
  summary.invalid_steps = custom.invalid_steps;
  summary.num_linear_solves = custom.num_linear_solves;
  summary.preprocessor_time_in_seconds = 0.0;
  summary.minimizer_time_in_seconds = 0.0;
  summary.postprocessor_time_in_seconds = 0.0;
  summary.total_time_in_seconds = 0.0;
  summary.linear_solver_time_in_seconds = 0.0;
  summary.iterations.clear();
  summary.iterations.reserve(custom.trace.size());
  for (const gpu_ba::CustomCpuIteration& source : custom.trace) {
    gpu_ba::CeresIterationSnapshot item;
    item.iteration = source.iteration;
    item.step_is_valid = source.step_valid;
    item.step_is_nonmonotonic = false;
    item.step_is_successful = source.accepted;
    item.cost = source.iteration == 0 ? source.cost_before : source.trial_cost;
    item.cost_change = source.iteration == 0 ? 0.0 : source.actual_reduction;
    item.gradient_max_norm = source.projected_gradient_max_norm;
    item.gradient_norm = source.gradient_norm;
    item.step_norm = source.step_norm;
    item.relative_decrease = source.rho;
    item.trust_region_radius = source.radius_after;
    item.eta = source.eta;
    item.step_size = 0.0;
    item.linear_solver_iterations = source.linear_solver_iterations;
    summary.iterations.push_back(item);
  }
  return summary;
}

bool WriteCustomCpuFidelityCandidate(
    const gpu_ba::Snapshot& solve_pre,
    const gpu_ba::SnapshotReadResult& solve_pre_read,
    const gpu_ba::Snapshot& solved_state,
    const gpu_ba::CustomCpuSolveResult& custom_result,
    const gpu_ba::CeresFidelityRecord& oracle,
    const std::string& output_path,
    const std::string& run_id,
    gpu_ba::CeresFidelityWriteResult* write_result,
    std::string* error) {
  const std::string candidate_dir =
      JoinPaths(output_path, "custom-fidelity-" + run_id);
  CreateDirIfNotExists(candidate_dir, true);
  gpu_ba::Snapshot persisted_state = solved_state;
  persisted_state.metadata.backend = "custom_cpu";
  gpu_ba::SnapshotWriteResult state_write;
  if (!gpu_ba::WriteSnapshot(persisted_state, candidate_dir, &state_write,
                             error)) {
    return false;
  }

  gpu_ba::CeresFidelityRecord candidate;
  candidate.record_kind = "custom_cpu_independent_ceres14_compatible";
  candidate.run_id = run_id;
  if (!gpu_ba::CaptureFidelityProvenance(&candidate.provenance, error)) {
    return false;
  }
  candidate.snapshot_id = solve_pre.metadata.snapshot_id;
  candidate.snapshot_manifest_path = solve_pre_read.manifest_path;
  candidate.snapshot_payload_path = solve_pre_read.payload_path;
  candidate.snapshot_payload_sha256 = solve_pre_read.integrity.payload_sha256;
  candidate.snapshot_manifest_sha256 =
      solve_pre_read.integrity.manifest_sha256;
  candidate.post_state_manifest_path = state_write.manifest_path;
  candidate.post_state_payload_path = state_write.payload_path;
  candidate.post_state_payload_sha256 = state_write.integrity.payload_sha256;
  candidate.final_state_sha256 = gpu_ba::CanonicalStateSha256(solved_state);
  candidate.effective_options = oracle.effective_options;
  candidate.problem = oracle.problem;
  candidate.summary = BuildCustomCpuCompatibilitySummary(oracle, custom_result);
  return gpu_ba::WriteCeresFidelityRecord(candidate, candidate_dir,
                                          write_result, error);
}

}  // namespace

int RunGpuBaReplay(int argc, char** argv) {
  std::string snapshot_path;
  std::string backend = "ceres_cpu";
  std::string mode = "fixed_linearization";
  std::string output_path;
  std::string ceres_replay_semantics = "deterministic_diagnostic";
  std::string oracle_path;
  std::string oracle_paths;
  std::string candidate_oracle_path;
  std::string threshold_policy_path;
  std::string fidelity_envelope_mode = "candidate";
  std::string fidelity_run_id = "replay";
  std::string fidelity_perturbation = "none";
  double lambda = 1e-4;

  OptionManager options;
  options.AddDefaultOption("snapshot_path", &snapshot_path);
  options.AddDefaultOption("backend", &backend);
  options.AddDefaultOption("mode", &mode);
  options.AddDefaultOption("lambda", &lambda);
  options.AddDefaultOption("ceres_replay_semantics", &ceres_replay_semantics);
  options.AddDefaultOption("oracle_path", &oracle_path);
  options.AddDefaultOption("oracle_paths", &oracle_paths);
  options.AddDefaultOption("candidate_oracle_path", &candidate_oracle_path);
  options.AddDefaultOption("threshold_policy_path", &threshold_policy_path);
  options.AddDefaultOption("fidelity_envelope_mode", &fidelity_envelope_mode);
  options.AddDefaultOption("fidelity_run_id", &fidelity_run_id);
  options.AddDefaultOption("fidelity_perturbation", &fidelity_perturbation);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  if (ceres_replay_semantics != "deterministic_diagnostic" &&
      ceres_replay_semantics != "original_fidelity" &&
      ceres_replay_semantics != "fidelity_record_compare" &&
      ceres_replay_semantics != "fidelity_envelope_gate") {
    std::cerr << "ERROR: Invalid Ceres replay semantics: "
              << ceres_replay_semantics << std::endl;
    return EXIT_FAILURE;
  }
  if (ceres_replay_semantics == "fidelity_envelope_gate") {
    const std::vector<std::string> reference_paths =
        SplitCommaSeparated(oracle_paths);
    if (reference_paths.empty() || threshold_policy_path.empty() ||
        (fidelity_envelope_mode != "candidate" &&
         fidelity_envelope_mode != "leave_one_out") ||
        (fidelity_envelope_mode == "candidate" &&
         candidate_oracle_path.empty())) {
      std::cerr
          << "ERROR: fidelity_envelope_gate requires --oracle_paths, "
             "--threshold_policy_path, mode candidate|leave_one_out, and a "
             "candidate path in candidate mode"
          << std::endl;
      return EXIT_FAILURE;
    }
    CreateDirIfNotExists(output_path, true);
    gpu_ba::CeresEnvelopePolicy policy;
    std::string envelope_error;
    if (!gpu_ba::ReadCeresEnvelopePolicy(threshold_policy_path, &policy,
                                         &envelope_error)) {
      std::cerr << "ERROR: Envelope policy failed: " << envelope_error
                << std::endl;
      return EXIT_FAILURE;
    }
    std::vector<gpu_ba::LoadedCeresFidelitySample> references;
    references.reserve(reference_paths.size());
    for (const std::string& path : reference_paths) {
      gpu_ba::LoadedCeresFidelitySample sample;
      if (!gpu_ba::LoadCeresFidelitySample(path, &sample, &envelope_error)) {
        std::cerr << "ERROR: Envelope reference failed: " << envelope_error
                  << std::endl;
        return EXIT_FAILURE;
      }
      references.push_back(std::move(sample));
    }
    bool pass = false;
    std::string result_json;
    if (fidelity_envelope_mode == "leave_one_out") {
      gpu_ba::CeresEnvelopeLeaveOneOutResult leave_one_out;
      if (!gpu_ba::EvaluateCeresEnvelopeLeaveOneOut(
              references, policy, &leave_one_out, &envelope_error)) {
        std::cerr << "ERROR: Leave-one-out gate could not run: "
                  << envelope_error << std::endl;
        return EXIT_FAILURE;
      }
      pass = leave_one_out.pass;
      result_json =
          gpu_ba::CeresEnvelopeLeaveOneOutJson(leave_one_out, policy, 2);
    } else {
      gpu_ba::LoadedCeresFidelitySample candidate;
      if (!gpu_ba::LoadCeresFidelitySample(candidate_oracle_path, &candidate,
                                           &envelope_error) ||
          !gpu_ba::ApplyCeresEnvelopePerturbation(
              fidelity_perturbation, &candidate, &envelope_error)) {
        std::cerr << "ERROR: Envelope candidate failed: " << envelope_error
                  << std::endl;
        return EXIT_FAILURE;
      }
      gpu_ba::CeresEnvelopeCandidateResult candidate_result;
      if (!gpu_ba::EvaluateCeresEnvelopeCandidate(
              references, candidate, policy, &candidate_result,
              &envelope_error)) {
        std::cerr << "ERROR: Candidate envelope gate could not run: "
                  << envelope_error << std::endl;
        return EXIT_FAILURE;
      }
      pass = candidate_result.pass;
      result_json =
          gpu_ba::CeresEnvelopeCandidateJson(candidate_result, policy, 2);
    }
    const std::string report_path = JoinPaths(
        output_path, "fidelity-envelope-" + fidelity_run_id + ".json");
    std::ofstream report(report_path, std::ios::trunc);
    report << "{\"semantics\":\"fidelity_envelope_gate\",\n"
           << "\"mode\":\"" << fidelity_envelope_mode << "\",\n"
           << "\"perturbation\":\"" << fidelity_perturbation << "\",\n"
           << "\"threshold_policy_path\":\"" << threshold_policy_path
           << "\",\n\"result\":" << result_json << "\n}\n";
    report.close();
    if (!report) {
      std::cerr << "ERROR: Failed to write envelope report" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Ceres production envelope gate" << std::endl
              << "  mode: " << fidelity_envelope_mode << std::endl
              << "  policy_sha256: " << policy.file_sha256 << std::endl
              << "  pass: " << (pass ? "true" : "false") << std::endl
              << "  report: " << report_path << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (ceres_replay_semantics == "fidelity_record_compare") {
    if (oracle_path.empty() || candidate_oracle_path.empty()) {
      std::cerr << "ERROR: fidelity_record_compare requires --oracle_path "
                   "and --candidate_oracle_path"
                << std::endl;
      return EXIT_FAILURE;
    }
    CreateDirIfNotExists(output_path, true);
    gpu_ba::CeresFidelityRecord reference_record;
    gpu_ba::CeresFidelityRecord candidate_record;
    gpu_ba::CeresFidelityWriteResult reference_read;
    gpu_ba::CeresFidelityWriteResult candidate_read;
    gpu_ba::Snapshot reference_state;
    gpu_ba::Snapshot candidate_state;
    gpu_ba::SnapshotReadResult reference_state_read;
    gpu_ba::SnapshotReadResult candidate_state_read;
    gpu_ba::CeresFidelityComparison comparison;
    std::string comparison_error;
    if (!gpu_ba::ReadCeresFidelityRecord(
            oracle_path, &reference_record, &reference_read,
            &comparison_error) ||
        !gpu_ba::ReadCeresFidelityRecord(
            candidate_oracle_path, &candidate_record, &candidate_read,
            &comparison_error) ||
        !gpu_ba::ReadSnapshot(reference_record.post_state_manifest_path,
                              &reference_state, &reference_state_read,
                              &comparison_error) ||
        !gpu_ba::ReadSnapshot(candidate_record.post_state_manifest_path,
                              &candidate_state, &candidate_state_read,
                              &comparison_error) ||
        !gpu_ba::CompareCeresFidelityRecords(
            reference_record, reference_state, candidate_record,
            candidate_state, &comparison, &comparison_error)) {
      std::cerr << "ERROR: Fidelity record comparison failed: "
                << comparison_error << std::endl;
      return EXIT_FAILURE;
    }
    const std::string report_path = JoinPaths(
        output_path, "fidelity-record-compare-" + fidelity_run_id + ".json");
    std::ofstream report(report_path, std::ios::trunc);
    report << std::setprecision(17)
           << "{\"semantics\":\"fidelity_record_compare\",\n"
           << "\"reference_record_path\":\"" << oracle_path << "\",\n"
           << "\"candidate_record_path\":\"" << candidate_oracle_path
           << "\",\n\"reference_record\":"
           << gpu_ba::CeresFidelityRecordJson(reference_record, 2)
           << ",\n\"candidate_record\":"
           << gpu_ba::CeresFidelityRecordJson(candidate_record, 2)
           << ",\n\"comparison\":"
           << gpu_ba::CeresFidelityComparisonJson(comparison, 2)
           << "\n}\n";
    report.close();
    if (!report) {
      std::cerr << "ERROR: Failed to write fidelity record comparison"
                << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Ceres fidelity record comparison" << std::endl
              << "  pass: " << (comparison.pass ? "true" : "false")
              << std::endl
              << "  problem_fingerprint_pass: "
              << (comparison.problem_fingerprint_pass ? "true" : "false")
              << std::endl
              << "  trace_structure_pass: "
              << (comparison.trace_structure_pass ? "true" : "false")
              << std::endl
              << "  final_state_bitwise_identical: "
              << (comparison.final_state_bitwise_identical ? "true" :
                                                               "false")
              << std::endl
              << "  report: " << report_path << std::endl;
    return comparison.pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (ceres_replay_semantics == "original_fidelity") {
    if (backend != "ceres_cpu" || mode != "solve" || oracle_path.empty() ||
        snapshot_path.empty()) {
      std::cerr << "ERROR: original_fidelity requires --backend ceres_cpu, "
                   "--mode solve, and --oracle_path"
                << std::endl;
      return EXIT_FAILURE;
    }
    CreateDirIfNotExists(output_path, true);
    gpu_ba::CeresFidelityReplayResult fidelity_result;
    std::string fidelity_error;
    if (!gpu_ba::RunOriginalFidelityReplay(
            snapshot_path, oracle_path, output_path, fidelity_run_id,
            fidelity_perturbation, &fidelity_result, &fidelity_error)) {
      std::cerr << "ERROR: Original-fidelity replay failed to execute: "
                << fidelity_error << std::endl;
      return EXIT_FAILURE;
    }
    const std::string report_path = JoinPaths(
        output_path, fidelity_result.replay_record.snapshot_id +
                         "-ceres-original-fidelity-" + fidelity_run_id +
                         ".json");
    std::ofstream report(report_path, std::ios::trunc);
    report << std::setprecision(17)
           << gpu_ba::CeresFidelityReplayJson(fidelity_result) << '\n';
    report.close();
    if (!report) {
      std::cerr << "ERROR: Failed to write fidelity replay report"
                << std::endl;
      return EXIT_FAILURE;
    }
    const auto& recorded = fidelity_result.oracle_record.effective_options;
    const auto& replayed = fidelity_result.replay_record.effective_options;
    const auto& actual = fidelity_result.replay_record.summary;
    std::cout << "Ceres original-fidelity options" << std::endl
              << "  max_invalid recorded/replay/actual_input: "
              << recorded.max_num_consecutive_invalid_steps << "/"
              << replayed.max_num_consecutive_invalid_steps << "/"
              << replayed.max_num_consecutive_invalid_steps << std::endl
              << "  max_linear recorded/replay/actual_input: "
              << recorded.max_linear_solver_iterations << "/"
              << replayed.max_linear_solver_iterations << "/"
              << replayed.max_linear_solver_iterations << std::endl
              << "  threads recorded/replay/Ceres_actual: "
              << recorded.num_threads << "/" << replayed.num_threads << "/"
              << actual.num_threads_used << std::endl
              << "  linear solver recorded/replay/Ceres_actual: "
              << ceres::LinearSolverTypeToString(
                     static_cast<ceres::LinearSolverType>(
                         recorded.linear_solver_type))
              << "/"
              << ceres::LinearSolverTypeToString(
                     static_cast<ceres::LinearSolverType>(
                         replayed.linear_solver_type))
              << "/"
              << ceres::LinearSolverTypeToString(
                     static_cast<ceres::LinearSolverType>(
                         actual.linear_solver_type_used))
              << std::endl
              << "  report: " << report_path << std::endl;
    return fidelity_result.pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (snapshot_path.empty()) {
    std::cerr << "ERROR: --snapshot_path is required" << std::endl;
    return EXIT_FAILURE;
  }

  if (backend != "ceres_cpu" && backend != "custom_cpu" &&
      backend != "custom_cuda" && backend != "compare") {
    std::cerr << "ERROR: Invalid backend: " << backend << std::endl;
    return EXIT_FAILURE;
  }
  if (mode != "fixed_linearization" && mode != "single_step" &&
      mode != "ceres14_single_step" && mode != "solve") {
    std::cerr << "ERROR: Invalid replay mode: " << mode << std::endl;
    return EXIT_FAILURE;
  }

  gpu_ba::Snapshot snapshot;
  gpu_ba::SnapshotReadResult read_result;
  std::string error;
  if (!gpu_ba::ReadSnapshot(snapshot_path, &snapshot, &read_result, &error)) {
    std::cerr << "ERROR: Snapshot validation failed: " << error << std::endl;
    return EXIT_FAILURE;
  }
  if (!ValidateOrder(snapshot, &error)) {
    std::cerr << "ERROR: Snapshot order validation failed: " << error
              << std::endl;
    return EXIT_FAILURE;
  }

  // Every replay backend receives an independent state copy. Solver-specific
  // work is deliberately gated until the fixed-linearization implementation.
  const gpu_ba::Snapshot replay_state = snapshot;
  const gpu_ba::Snapshot comparison_state = snapshot;
  const bool deep_copy_valid =
      replay_state.points.size() == comparison_state.points.size() &&
      replay_state.images.size() == comparison_state.images.size();
  const size_t selected_images = std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const gpu_ba::ImageSnapshot& image) { return image.selected; });
  const size_t constant_poses = std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const gpu_ba::ImageSnapshot& image) { return image.pose_constant; });
  const size_t constant_points = std::count_if(
      snapshot.points.begin(), snapshot.points.end(),
      [](const gpu_ba::PointSnapshot& point) { return point.constant; });
  size_t camera_parameter_blocks = 0;
  size_t quaternion_parameter_blocks = 0;
  size_t translation_parameter_blocks = 0;
  size_t point_parameter_blocks = 0;
  for (const gpu_ba::ParameterBlockSnapshot& parameter :
       snapshot.parameter_blocks_source_order) {
    switch (parameter.kind) {
      case gpu_ba::ParameterKind::kCamera:
        ++camera_parameter_blocks;
        break;
      case gpu_ba::ParameterKind::kQuaternion:
        ++quaternion_parameter_blocks;
        break;
      case gpu_ba::ParameterKind::kTranslation:
        ++translation_parameter_blocks;
        break;
      case gpu_ba::ParameterKind::kPoint3D:
        ++point_parameter_blocks;
        break;
    }
  }

  CreateDirIfNotExists(output_path, true);

  gpu_ba::LinearizationValidationOptions validation_options;
  gpu_ba::LinearizationValidationResult validation_result;
  gpu_ba::FixedLinearizationOptions fixed_options;
  fixed_options.lambda = lambda;
  gpu_ba::FixedLinearizationResult fixed_result;
  const bool ceres14_single_step = mode == "ceres14_single_step";
  const bool run_linearization_validation =
      (mode == "fixed_linearization" || ceres14_single_step) &&
      backend == "compare";
  if (run_linearization_validation &&
      !gpu_ba::ValidateResidualsAndJacobians(
          snapshot, validation_options, &validation_result, &error)) {
    std::cerr << "ERROR: Residual/Jacobian validation could not run: "
              << error << std::endl;
    return EXIT_FAILURE;
  }
  const bool run_fixed_linearization = run_linearization_validation;
  fixed_options.step_semantics =
      ceres14_single_step ? gpu_ba::FixedStepSemantics::kCeres14
                          : gpu_ba::FixedStepSemantics::kLegacy;
  if (run_fixed_linearization &&
      !gpu_ba::RunFixedLinearizationComparison(
          snapshot, fixed_options, &fixed_result, &error)) {
    std::cerr << "ERROR: Fixed linearization could not run: " << error
              << std::endl;
    return EXIT_FAILURE;
  }

  gpu_ba::CustomCpuSolveResult custom_result;
  gpu_ba::CeresCpuSolveResult ceres_result;
  gpu_ba::CpuSolveComparisonResult solve_comparison;
  gpu_ba::Snapshot custom_state;
  gpu_ba::Snapshot ceres_state;
  const bool solver_mode = mode == "single_step" || ceres14_single_step ||
                           mode == "solve";
  const bool run_custom_cpu =
      solver_mode && (backend == "custom_cpu" || backend == "compare");
  const bool run_ceres_cpu =
      solver_mode && (backend == "ceres_cpu" || backend == "compare");
  const bool compare_cpu_solves = run_custom_cpu && run_ceres_cpu;
  if (solver_mode && backend == "custom_cuda") {
    std::cerr << "ERROR: custom_cuda replay is unavailable until phase 6"
              << std::endl;
    return EXIT_FAILURE;
  }
  const gpu_ba::CustomCpuSolveMode custom_mode =
      mode == "single_step"
          ? gpu_ba::CustomCpuSolveMode::kLegacySingleStep
          : (ceres14_single_step
                 ? gpu_ba::CustomCpuSolveMode::kCeres14SingleStep
                 : gpu_ba::CustomCpuSolveMode::kFull);
  gpu_ba::CustomCpuSolverOptions custom_options;
  gpu_ba::CeresFidelityRecord custom_reference_oracle;
  bool custom_options_from_oracle = false;
  if (run_custom_cpu && !oracle_path.empty()) {
    if (!LoadCustomCpuOptionsFromOracle(
            oracle_path, snapshot, read_result, &custom_reference_oracle,
            &custom_options, &error)) {
      std::cerr << "ERROR: custom_cpu oracle options could not be loaded: "
                << error << std::endl;
      return EXIT_FAILURE;
    }
    custom_options_from_oracle = true;
  }
  const bool custom_ran =
      !run_custom_cpu ||
      (custom_options_from_oracle
           ? gpu_ba::RunCustomCpuSolve(snapshot, custom_options, custom_mode,
                                      &custom_result, &custom_state, &error)
           : gpu_ba::RunCustomCpuSolve(snapshot, lambda, custom_mode,
                                      &custom_result, &custom_state, &error));
  if (!custom_ran) {
    std::cerr << "ERROR: custom_cpu solve could not run: " << error
              << std::endl;
    return EXIT_FAILURE;
  }
  gpu_ba::CeresFidelityWriteResult custom_fidelity_write;
  const bool write_custom_fidelity =
      run_custom_cpu && custom_options_from_oracle && mode == "solve";
  if (write_custom_fidelity &&
      !WriteCustomCpuFidelityCandidate(
          snapshot, read_result, custom_state, custom_result,
          custom_reference_oracle, output_path, fidelity_run_id,
          &custom_fidelity_write, &error)) {
    std::cerr << "ERROR: custom_cpu fidelity record could not be written: "
              << error << std::endl;
    return EXIT_FAILURE;
  }
  std::string damping_oracle_directory;
  if (run_ceres_cpu && ceres14_single_step) {
    damping_oracle_directory = JoinPaths(output_path, "ceres_damping_oracle");
    CreateDirIfNotExists(damping_oracle_directory, true);
  }
  if (run_ceres_cpu &&
      !gpu_ba::RunCeresCpuSolve(snapshot, mode != "solve",
                               damping_oracle_directory,
                               &ceres_result, &ceres_state, &error)) {
    std::cerr << "ERROR: ceres_cpu solve could not run: " << error
              << std::endl;
    return EXIT_FAILURE;
  }
  if (compare_cpu_solves &&
      !gpu_ba::CompareCpuSolveResults(
          ceres_state, custom_state, ceres_result, custom_result,
          &solve_comparison, &error)) {
    std::cerr << "ERROR: CPU solve comparison could not run: " << error
              << std::endl;
    return EXIT_FAILURE;
  }
  const bool solver_pass =
      (!run_custom_cpu || custom_result.success) &&
      (!run_ceres_cpu || ceres_result.success) &&
      (!compare_cpu_solves || solve_comparison.pass);
  const bool replay_pass =
      deep_copy_valid &&
      (!run_linearization_validation || validation_result.pass) &&
      (!run_fixed_linearization || fixed_result.pass) && solver_pass;

  std::string implementation_status = "schema_replay_only_phase_2";
  if (run_fixed_linearization) {
    implementation_status =
        ceres14_single_step
            ? "ceres14_single_step_with_fixed_layers_phase_5_v4"
            : "legacy_canonical_fixed_linearization_phase_4";
  } else if (solver_mode && backend == "compare") {
    implementation_status =
        mode == "solve"
            ? "custom_cpu_full_lm_comparison_phase_5"
            : (ceres14_single_step
                   ? "ceres14_single_step_comparison_phase_5_v4"
                   : "legacy_single_step_comparison_phase_5_v4");
  } else if (run_custom_cpu) {
    implementation_status = mode == "solve"
                                ? "custom_cpu_full_lm_phase_5"
                                : "custom_cpu_single_step_phase_5";
  } else if (run_ceres_cpu) {
    implementation_status = "ceres_cpu_replay_phase_5";
  }

  const std::string report_path = JoinPaths(
      output_path, snapshot.metadata.snapshot_id + "-" + backend + "-" + mode +
                       ".json");
  std::ofstream report(report_path, std::ios::trunc);
  if (!report.is_open()) {
    std::cerr << "ERROR: Cannot write replay report: " << report_path
              << std::endl;
    return EXIT_FAILURE;
  }
  report << std::setprecision(17)
         << "{\n"
         << "  \"snapshot_id\": \"" << snapshot.metadata.snapshot_id << "\",\n"
         << "  \"backend\": \"" << backend << "\",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"lambda\": " << lambda << ",\n"
         << "  \"schema_version\": " << gpu_ba::kSnapshotSchemaVersion << ",\n"
         << "  \"ba_kind\": \"" << gpu_ba::BaKindName(snapshot.metadata.ba_kind)
         << "\",\n"
         << "  \"registered_image_count\": "
         << snapshot.metadata.registered_image_count << ",\n"
         << "  \"ba_call_index\": " << snapshot.metadata.ba_call_index << ",\n"
         << "  \"refinement_index\": " << snapshot.metadata.refinement_index
         << ",\n"
         << "  \"trigger_image_id\": " << snapshot.metadata.trigger_image_id
         << ",\n"
         << "  \"implementation_status\": \""
         << implementation_status
         << "\",\n"
         << "  \"deep_copy_valid\": "
         << (deep_copy_valid ? "true" : "false") << ",\n"
         << "  \"counts\": {\"cameras\": " << snapshot.cameras.size()
         << ", \"images\": " << snapshot.images.size()
         << ", \"points\": " << snapshot.points.size()
         << ", \"observations\": " << snapshot.observations.size()
         << ", \"lidar\": " << snapshot.lidar.size()
         << ", \"parameter_blocks\": "
         << snapshot.parameter_blocks_source_order.size()
         << ", \"selected_images\": " << selected_images
         << ", \"variable_poses\": "
         << snapshot.images.size() - constant_poses
         << ", \"constant_poses\": " << constant_poses
         << ", \"variable_points\": "
         << snapshot.points.size() - constant_points
         << ", \"constant_points\": " << constant_points << "},\n"
         << "  \"problem_coverage\": {\"reconstruction_image_count\": "
         << snapshot.metadata.registered_image_count
         << ", \"snapshot_image_record_count\": " << snapshot.images.size()
         << ", \"actual_selected_ba_image_count\": " << selected_images
         << ", \"camera_parameter_block_count\": "
         << camera_parameter_blocks
         << ", \"quaternion_parameter_block_count\": "
         << quaternion_parameter_blocks
         << ", \"translation_parameter_block_count\": "
         << translation_parameter_blocks
         << ", \"pose_parameter_block_count\": "
         << quaternion_parameter_blocks + translation_parameter_blocks
         << ", \"point_parameter_block_count\": " << point_parameter_blocks
         << ", \"point_count\": " << snapshot.points.size()
         << ", \"visual_residual_block_count\": "
         << snapshot.observations.size()
         << ", \"lidar_residual_block_count\": " << snapshot.lidar.size()
         << ", \"total_residual_block_count\": "
         << snapshot.observations.size() + snapshot.lidar.size()
         << ", \"scalar_residual_count\": "
         << 2 * snapshot.observations.size() + snapshot.lidar.size()
         << ", \"loss_type\": \"" << snapshot.metadata.loss_function
         << "\", \"requested_solver\": \""
         << (run_ceres_cpu ? ceres_result.requested_linear_solver_type
                           : "not_run")
         << "\", \"actual_solver\": \""
         << (run_ceres_cpu ? ceres_result.actual_linear_solver_type
                           : "not_run")
         << "\"},\n"
         << "  \"manifest_sha256\": \""
         << read_result.integrity.manifest_sha256 << "\",\n"
         << "  \"payload_sha256\": \""
         << read_result.integrity.payload_sha256 << "\",\n"
         << "  \"source_order_sha256\": \""
         << read_result.integrity.source_order_sha256 << "\",\n"
         << "  \"canonical_order_sha256\": \""
         << read_result.integrity.canonical_order_sha256 << "\",\n"
         << "  \"lidar_correspondence_sha256\": \""
         << read_result.integrity.lidar_correspondence_sha256 << "\",\n";
  if (run_linearization_validation) {
    report << gpu_ba::LinearizationValidationJson(
                  validation_result, validation_options, 2)
           << ",\n";
  }
  if (run_fixed_linearization) {
    report << gpu_ba::FixedLinearizationJson(fixed_result, fixed_options, 2)
           << ",\n";
  }
  if (run_custom_cpu) {
    report << "  \"custom_state_sha256\": \""
           << SolvedStateSha256(custom_state) << "\",\n"
           << gpu_ba::CustomCpuSolveJson(custom_result, 2) << ",\n";
    if (write_custom_fidelity) {
      report << "  \"custom_fidelity_manifest_path\": \""
             << custom_fidelity_write.manifest_path << "\",\n"
             << "  \"custom_fidelity_binary_sha256\": \""
             << custom_fidelity_write.binary_sha256 << "\",\n";
    }
  }
  if (run_ceres_cpu) {
    report << "  \"ceres_state_sha256\": \""
           << SolvedStateSha256(ceres_state) << "\",\n"
           << gpu_ba::CeresCpuSolveJson(ceres_result, 2) << ",\n";
  }
  if (compare_cpu_solves) {
    report << gpu_ba::CpuSolveComparisonJson(solve_comparison, 2) << ",\n";
  }
  report << "  \"pass\": " << (replay_pass ? "true" : "false") << "\n"
         << "}\n";
  report.close();
  if (!report) {
    std::cerr << "ERROR: Failed while writing replay report" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Validated GPU BA snapshot " << snapshot.metadata.snapshot_id
            << std::endl
            << "  report: " << report_path << std::endl
            << "  payload_sha256: "
            << read_result.integrity.payload_sha256 << std::endl;
  return replay_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace colmap
