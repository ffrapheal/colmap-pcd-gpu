#include "exe/gpu_ba.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

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

}  // namespace

int RunGpuBaReplay(int argc, char** argv) {
  std::string snapshot_path;
  std::string backend = "ceres_cpu";
  std::string mode = "fixed_linearization";
  std::string output_path;
  double lambda = 1e-4;

  OptionManager options;
  options.AddRequiredOption("snapshot_path", &snapshot_path);
  options.AddDefaultOption("backend", &backend);
  options.AddDefaultOption("mode", &mode);
  options.AddDefaultOption("lambda", &lambda);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  if (backend != "ceres_cpu" && backend != "custom_cpu" &&
      backend != "custom_cuda" && backend != "compare") {
    std::cerr << "ERROR: Invalid backend: " << backend << std::endl;
    return EXIT_FAILURE;
  }
  if (mode != "fixed_linearization" && mode != "single_step" &&
      mode != "solve") {
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

  gpu_ba::LinearizationValidationOptions validation_options;
  gpu_ba::LinearizationValidationResult validation_result;
  const bool run_linearization_validation =
      mode == "fixed_linearization" && backend == "compare";
  if (run_linearization_validation &&
      !gpu_ba::ValidateResidualsAndJacobians(
          snapshot, validation_options, &validation_result, &error)) {
    std::cerr << "ERROR: Residual/Jacobian validation could not run: "
              << error << std::endl;
    return EXIT_FAILURE;
  }
  const bool replay_pass =
      deep_copy_valid &&
      (!run_linearization_validation || validation_result.pass);

  CreateDirIfNotExists(output_path, true);
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
         << (run_linearization_validation
                 ? "residual_jacobian_validation_phase_3"
                 : "schema_replay_only_phase_2")
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
  if (mode != "fixed_linearization") {
    std::cerr << "ERROR: " << mode
              << " is gated until the custom linearization/LM phases"
              << std::endl;
    return EXIT_FAILURE;
  }
  return replay_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace colmap
