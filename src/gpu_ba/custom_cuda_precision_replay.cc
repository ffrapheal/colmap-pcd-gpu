#include "gpu_ba/custom_cuda.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "gpu_ba/snapshot.h"

namespace colmap {
namespace gpu_ba {
namespace {

std::string JsonString(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (const char character : value) {
    if (character == '"') stream << "\\\"";
    else if (character == '\\') stream << "\\\\";
    else if (character == '\n') stream << "\\n";
    else stream << character;
  }
  stream << '"';
  return stream.str();
}

std::string NumberOrNull(const double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

void WriteStatistics(std::ostream& stream,
                     const CudaPrecisionErrorStatistics& value) {
  stream << "{\"count\":" << value.count
         << ",\"finite_count\":" << value.finite_count
         << ",\"max_abs_error\":" << NumberOrNull(value.max_abs_error)
         << ",\"max_relative_error\":"
         << NumberOrNull(value.max_relative_error)
         << ",\"mean_abs_error\":" << NumberOrNull(value.mean_abs_error)
         << ",\"rms_abs_error\":" << NumberOrNull(value.rms_abs_error)
         << ",\"p50_abs_error\":" << NumberOrNull(value.p50_abs_error)
         << ",\"p95_abs_error\":" << NumberOrNull(value.p95_abs_error)
         << ",\"p99_abs_error\":" << NumberOrNull(value.p99_abs_error)
         << ",\"p50_relative_error\":"
         << NumberOrNull(value.p50_relative_error)
         << ",\"p95_relative_error\":"
         << NumberOrNull(value.p95_relative_error)
         << ",\"p99_relative_error\":"
         << NumberOrNull(value.p99_relative_error)
         << ",\"reference_max_magnitude\":"
         << NumberOrNull(value.reference_max_magnitude)
         << ",\"candidate_max_magnitude\":"
         << NumberOrNull(value.candidate_max_magnitude)
         << ",\"normalized_max_error\":"
         << NumberOrNull(value.normalized_max_error)
         << ",\"normalized_frobenius_error\":"
         << NumberOrNull(value.normalized_frobenius_error)
         << ",\"worst_index\":" << value.worst_index
         << ",\"worst_entity_kind\":" << JsonString(value.worst_entity_kind)
         << ",\"worst_entity_id\":" << value.worst_entity_id
         << ",\"worst_row\":" << value.worst_row
         << ",\"worst_column\":" << value.worst_column
         << ",\"worst_reference\":" << NumberOrNull(value.worst_reference)
         << ",\"worst_candidate\":" << NumberOrNull(value.worst_candidate)
         << '}';
}

void WriteMixedStableResult(std::ostream& output,
                            const char* snapshot,
                            const double lambda,
                            const size_t repeats,
                            const bool ran,
                            const CudaMixedStableComponentResult& result,
                            const std::string& error) {
  output << "{\n  \"schema\":\"phase10p1d_fp32_products_fp64_dense_component_v1\","
         << "\n  \"snapshot\":" << JsonString(snapshot)
         << ",\n  \"lambda\":" << NumberOrNull(lambda)
         << ",\n  \"repeats\":" << repeats
         << ",\n  \"ran\":" << (ran ? "true" : "false")
         << ",\n  \"success\":" << (result.success ? "true" : "false")
         << ",\n  \"error\":"
         << JsonString(error.empty() ? result.error : error)
         << ",\n  \"stages\":{";
  const std::pair<const char*, const CudaPrecisionErrorStatistics*> stages[] = {
      {"residual_jacobian", &result.residual_jacobian},
      {"robust_scale", &result.robust_scale},
      {"pose_hessian", &result.pose_hessian},
      {"pose_gradient", &result.pose_gradient},
      {"point_hessian", &result.point_hessian},
      {"point_gradient", &result.point_gradient},
      {"edge_blocks", &result.edge_blocks},
      {"jacobi_damping", &result.jacobi_damping},
      {"point_inverse", &result.point_inverse},
      {"inverse_gradient", &result.inverse_gradient},
      {"transformed_edge", &result.transformed_edge},
      {"schur", &result.schur},
      {"fp32_products_fp64_dense_accum", &result.f32_schur_then_cast},
      {"rhs", &result.rhs},
      {"camera_delta", &result.camera_delta},
      {"point_delta", &result.point_delta},
      {"trial_state", &result.trial_state},
  };
  bool first = true;
  for (const auto& stage : stages) {
    output << (first ? "\n    " : ",\n    ") << JsonString(stage.first)
           << ':';
    WriteStatistics(output, *stage.second);
    first = false;
  }
  output << "\n  },\n  \"scalars\":{\"fp64_predicted_reduction\":"
         << NumberOrNull(result.fp64_predicted_reduction)
         << ",\"mixed_predicted_reduction\":"
         << NumberOrNull(result.mixed_predicted_reduction)
         << ",\"fp64_backward_error\":"
         << NumberOrNull(result.fp64_backward_error)
         << ",\"mixed_backward_error\":"
         << NumberOrNull(result.mixed_backward_error)
         << ",\"fp64_schur_solve_backward_error\":"
         << NumberOrNull(result.fp64_schur_solve_backward_error)
         << ",\"mixed_schur_solve_backward_error\":"
         << NumberOrNull(result.mixed_schur_solve_backward_error)
         << ",\"fp64_trial_cost\":" << NumberOrNull(result.fp64_trial_cost)
         << ",\"mixed_trial_cost\":"
         << NumberOrNull(result.mixed_trial_cost) << "},"
         << "\n  \"solver\":{\"fp64_info\":" << result.fp64_solver_info
         << ",\"mixed_info\":" << result.mixed_solver_info
         << ",\"mixed_dpotrf_calls\":" << result.mixed_dpotrf_calls
         << ",\"mixed_dpotrs_calls\":" << result.mixed_dpotrs_calls
         << ",\"mixed_spotrf_calls\":" << result.mixed_spotrf_calls
         << ",\"mixed_spotrs_calls\":" << result.mixed_spotrs_calls
         << "},\n  \"resources\":{\"mixed_schur_math_effective\":"
         << JsonString(result.mixed_schur_math_effective)
         << ",\"float_buffer_bytes\":" << result.mixed_float_buffer_bytes
         << ",\"double_buffer_bytes\":" << result.mixed_double_buffer_bytes
         << ",\"full_array_d2h_bytes\":"
         << result.mixed_full_array_d2h_bytes
         << ",\"serial_full_scan_kernel_count\":"
         << result.mixed_serial_full_scan_kernel_count
         << ",\"post_initialize_allocation_calls\":"
         << result.mixed_post_initialize_allocation_calls
         << ",\"post_ready_arena_grow_calls\":"
         << result.mixed_post_ready_arena_grow_calls << "},"
         << "\n  \"timing_ms\":{\"fp64_complete\":"
         << NumberOrNull(result.fp64_complete_milliseconds)
         << ",\"mixed_complete_median\":"
         << NumberOrNull(result.mixed_complete_milliseconds)
         << ",\"residual_jacobian\":"
         << NumberOrNull(result.mixed_residual_jacobian_milliseconds)
         << ",\"hessian\":"
         << NumberOrNull(result.mixed_hessian_milliseconds)
         << ",\"gradient\":"
         << NumberOrNull(result.mixed_gradient_milliseconds)
         << ",\"point_inverse\":"
         << NumberOrNull(result.mixed_point_inverse_milliseconds)
         << ",\"schur\":" << NumberOrNull(result.mixed_schur_milliseconds)
         << ",\"rhs\":" << NumberOrNull(result.mixed_rhs_milliseconds)
         << ",\"factorization\":"
         << NumberOrNull(result.mixed_factorization_milliseconds)
         << ",\"back_substitution\":"
         << NumberOrNull(result.mixed_back_substitution_milliseconds)
         << ",\"diagnostics\":"
         << NumberOrNull(result.mixed_diagnostics_milliseconds)
         << "},\n  \"schur_candidate_pairs\":[";
  const size_t pair_count = std::min(
      result.promoted_double_schur_complete_milliseconds.size(),
      result.f32_schur_then_cast_complete_milliseconds.size());
  for (size_t index = 0; index < pair_count; ++index) {
    output << (index == 0 ? "" : ",") << "{\"pair\":" << index
           << ",\"promoted_double_ms\":"
           << NumberOrNull(
                  result.promoted_double_schur_complete_milliseconds[index])
           << ",\"fp32_products_fp64_dense_accum_ms\":"
           << NumberOrNull(
                  result.f32_schur_then_cast_complete_milliseconds[index])
           << '}';
  }
  output << "]\n}\n";
}

int Main(const int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " SNAPSHOT OUTPUT_JSON [LAMBDA=1] [REPEATS=3]"
                 " [legacy_fp32|mixed_stable]\n";
    return 2;
  }
  const double lambda = argc >= 4 ? std::stod(argv[3]) : 1.0;
  const size_t repeats = argc >= 5 ? std::stoull(argv[4]) : 3;
  const std::string mode = argc >= 6 ? argv[5] : "legacy_fp32";
  if (mode != "legacy_fp32" && mode != "mixed_stable") {
    std::cerr << "precision component mode must be legacy_fp32 or mixed_stable\n";
    return 2;
  }
  Snapshot snapshot;
  SnapshotReadResult read;
  std::string error;
  if (!ReadSnapshot(argv[1], &snapshot, &read, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::ofstream output(argv[2], std::ios::trunc);
  if (!output) {
    std::cerr << "cannot open output JSON\n";
    return 1;
  }
  if (mode == "mixed_stable") {
    CudaMixedStableComponentResult mixed_result;
    const bool ran = RunCudaMixedStableComponentForTesting(
        snapshot, lambda, repeats, &mixed_result, &error);
    WriteMixedStableResult(output, argv[1], lambda, repeats, ran,
                           mixed_result, error);
    output.close();
    if (!output) {
      std::cerr << "failed writing output JSON\n";
      return 1;
    }
    if (!ran) std::cerr << error << '\n';
    return ran && mixed_result.success ? 0 : 1;
  }
  CudaPrecisionComponentResult result;
  const bool ran = RunCudaPrecisionComponentForTesting(
      snapshot, lambda, repeats, &result, &error);
  output << "{\n  \"schema\":\"phase10p1a_precision_component_v1\","
         << "\n  \"snapshot\":" << JsonString(argv[1])
         << ",\n  \"lambda\":" << NumberOrNull(lambda)
         << ",\n  \"fp32_repeats\":" << repeats
         << ",\n  \"ran\":" << (ran ? "true" : "false")
         << ",\n  \"success\":" << (result.success ? "true" : "false")
         << ",\n  \"error\":"
         << JsonString(error.empty() ? result.error : error)
         << ",\n  \"factorization\":{\"fp64_routine\":"
         << JsonString(result.fp64_factorization_routine)
         << ",\"fp32_routine\":" << JsonString(result.fp32_factorization_routine)
         << ",\"fp64_solver_info\":" << result.fp64_solver_info
         << ",\"fp32_solver_info\":" << result.fp32_solver_info
         << ",\"dpotrf_calls\":" << result.dpotrf_calls
         << ",\"dpotrs_calls\":" << result.dpotrs_calls
         << ",\"spotrf_calls\":" << result.spotrf_calls
         << ",\"spotrs_calls\":" << result.spotrs_calls << "},"
         << "\n  \"stages\":{";
  const std::pair<const char*, const CudaPrecisionErrorStatistics*> stages[] = {
      {"input_cast", &result.input_cast},
      {"residual_jacobian", &result.residual_jacobian},
      {"robust_scale", &result.robust_scale},
      {"pose_hessian_gradient", &result.pose_hessian_gradient},
      {"point_hessian_gradient", &result.point_hessian_gradient},
      {"edge_blocks", &result.edge_blocks},
      {"jacobi_damping", &result.jacobi_damping},
      {"point_inverse", &result.point_inverse},
      {"transformed_edge", &result.transformed_edge},
      {"schur", &result.schur},
      {"rhs", &result.rhs},
      {"camera_delta", &result.camera_delta},
      {"point_delta", &result.point_delta},
      {"trial_state", &result.trial_state},
      {"fp64_trial_update_consistency",
       &result.fp64_trial_update_consistency},
      {"fp32_trial_update_consistency",
       &result.fp32_trial_update_consistency},
  };
  bool first = true;
  for (const auto& stage : stages) {
    output << (first ? "\n    " : ",\n    ") << JsonString(stage.first) << ':';
    WriteStatistics(output, *stage.second);
    first = false;
  }
  output << "\n  },\n  \"scalars\":{\"fp64_predicted_reduction\":"
         << NumberOrNull(result.fp64_predicted_reduction)
         << ",\"fp32_predicted_reduction\":"
         << NumberOrNull(result.fp32_predicted_reduction)
         << ",\"fp64_backward_error\":"
         << NumberOrNull(result.fp64_backward_error)
         << ",\"fp32_backward_error\":"
         << NumberOrNull(result.fp32_backward_error)
         << ",\"fp64_trial_cost\":" << NumberOrNull(result.fp64_trial_cost)
         << ",\"fp32_trial_cost\":" << NumberOrNull(result.fp32_trial_cost)
         << "},\n  \"condition_proxy\":{\"pose_diagonal_min\":"
         << NumberOrNull(result.pose_diagonal_min)
         << ",\"pose_diagonal_max\":" << NumberOrNull(result.pose_diagonal_max)
         << ",\"point_positive_diagonal_min\":"
         << NumberOrNull(result.point_positive_diagonal_min)
         << ",\"point_diagonal_max\":" << NumberOrNull(result.point_diagonal_max)
         << ",\"point_determinant_min_abs\":"
         << NumberOrNull(result.point_determinant_min_abs)
         << ",\"point_inverse_residual_max\":"
         << NumberOrNull(result.point_inverse_residual_max)
         << ",\"schur_diagonal_min\":" << NumberOrNull(result.schur_diagonal_min)
         << ",\"schur_diagonal_max\":" << NumberOrNull(result.schur_diagonal_max)
         << ",\"schur_spd_reference\":"
         << (result.schur_spd_reference ? "true" : "false")
         << ",\"metric\":" << JsonString(result.schur_condition_metric)
         << ",\"value\":" << NumberOrNull(result.schur_condition_number)
         << "},\n  \"diagnostic_pose\":{\"image_id\":"
         << result.diagnostic_pose_image_id << ",\"initial_qvec\":[";
  for (size_t i = 0; i < 4; ++i)
    output << (i == 0 ? "" : ",")
           << NumberOrNull(result.diagnostic_initial_qvec[i]);
  output << "],\"fp64_trial_qvec\":[";
  for (size_t i = 0; i < 4; ++i)
    output << (i == 0 ? "" : ",")
           << NumberOrNull(result.diagnostic_fp64_trial_qvec[i]);
  output << "],\"fp32_trial_qvec\":[";
  for (size_t i = 0; i < 4; ++i)
    output << (i == 0 ? "" : ",")
           << NumberOrNull(result.diagnostic_fp32_trial_qvec[i]);
  output << "],\"fp64_camera_delta\":[";
  for (size_t i = 0; i < 6; ++i)
    output << (i == 0 ? "" : ",")
           << NumberOrNull(result.diagnostic_fp64_camera_delta[i]);
  output << "],\"fp32_camera_delta\":[";
  for (size_t i = 0; i < 6; ++i)
    output << (i == 0 ? "" : ",")
           << NumberOrNull(result.diagnostic_fp32_camera_delta[i]);
  output << "]}\n}\n";
  output.close();
  if (!output) {
    std::cerr << "failed writing output JSON\n";
    return 1;
  }
  if (!ran) std::cerr << error << '\n';
  return ran && result.success ? 0 : 1;
}

}  // namespace
}  // namespace gpu_ba
}  // namespace colmap

int main(int argc, char** argv) {
  return colmap::gpu_ba::Main(argc, argv);
}
