#include "gpu_ba/ceres_fidelity.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <openssl/sha.h>
#include <unistd.h>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "util/misc.h"

// A transitive platform/PCL header defines U64(x) as an integer-literal
// helper. Prevent that unrelated macro from rewriting the codec method name.
#ifdef U64
#undef U64
#endif

namespace colmap {
namespace gpu_ba {
namespace {

constexpr char kFidelityMagic[8] = {'C', 'E', 'R', 'E', 'S', 'F', 'I', 'D'};

class Writer {
 public:
  void U8(uint8_t value) { data_.push_back(value); }
  void U32(uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
      data_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
    }
  }
  void I32(int32_t value) { U32(static_cast<uint32_t>(value)); }
  void U64(uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
      data_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
    }
  }
  void Double(double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "binary64 required");
    std::memcpy(&bits, &value, sizeof(bits));
    U64(bits);
  }
  void String(const std::string& value) {
    U64(value.size());
    data_.insert(data_.end(), value.begin(), value.end());
  }
  void Bytes(const uint8_t* data, size_t size) {
    data_.insert(data_.end(), data, data + size);
  }
  const std::vector<uint8_t>& Data() const { return data_; }
  std::vector<uint8_t> MoveData() { return std::move(data_); }

 private:
  std::vector<uint8_t> data_;
};

class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& data) : data_(data) {}
  bool U8(uint8_t* value) {
    if (!Require(1)) return false;
    *value = data_[offset_++];
    return true;
  }
  bool U32(uint32_t* value) {
    if (!Require(4)) return false;
    *value = 0;
    for (size_t i = 0; i < 4; ++i) {
      *value |= static_cast<uint32_t>(data_[offset_++]) << (8 * i);
    }
    return true;
  }
  bool I32(int32_t* value) {
    uint32_t raw = 0;
    if (!U32(&raw)) return false;
    *value = static_cast<int32_t>(raw);
    return true;
  }
  bool U64(uint64_t* value) {
    if (!Require(8)) return false;
    *value = 0;
    for (size_t i = 0; i < 8; ++i) {
      *value |= static_cast<uint64_t>(data_[offset_++]) << (8 * i);
    }
    return true;
  }
  bool Double(double* value) {
    uint64_t bits = 0;
    if (!U64(&bits)) return false;
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }
  bool String(std::string* value) {
    uint64_t size = 0;
    if (!U64(&size) || size > Remaining()) return false;
    value->assign(reinterpret_cast<const char*>(data_.data() + offset_),
                  static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return true;
  }
  bool Bytes(size_t size, std::vector<uint8_t>* value) {
    if (!Require(size)) return false;
    value->assign(data_.begin() + offset_, data_.begin() + offset_ + size);
    offset_ += size;
    return true;
  }
  bool AtEnd() const { return offset_ == data_.size(); }
  size_t Remaining() const { return data_.size() - offset_; }

 private:
  bool Require(size_t size) const { return size <= Remaining(); }
  const std::vector<uint8_t>& data_;
  size_t offset_ = 0;
};

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

std::string SanitizeToken(const std::string& value) {
  std::string result;
  for (const unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '-' || ch == '_') result.push_back(ch);
  }
  return result.empty() ? "run" : result;
}

std::string RemoveSuffix(const std::string& value, const std::string& suffix) {
  if (value.size() >= suffix.size() &&
      value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return value.substr(0, value.size() - suffix.size());
  }
  return value;
}

bool ReadFile(const std::string& path,
              std::vector<uint8_t>* data,
              std::string* error) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    *error = "Cannot open fidelity file: " + path;
    return false;
  }
  const std::streamoff size = input.tellg();
  if (size < 0) {
    *error = "Cannot determine fidelity file size: " + path;
    return false;
  }
  input.seekg(0, std::ios::beg);
  data->resize(static_cast<size_t>(size));
  if (size > 0 &&
      !input.read(reinterpret_cast<char*>(data->data()), size)) {
    *error = "Cannot read complete fidelity file: " + path;
    return false;
  }
  return true;
}

bool WriteFile(const std::string& path,
               const std::vector<uint8_t>& data,
               std::string* error) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    *error = "Cannot open fidelity file for writing: " + path;
    return false;
  }
  if (!data.empty()) {
    output.write(reinterpret_cast<const char*>(data.data()), data.size());
  }
  output.close();
  if (!output) {
    *error = "Cannot write complete fidelity file: " + path;
    return false;
  }
  return true;
}

bool ExtractJsonString(const std::string& json,
                       const std::string& key,
                       std::string* value) {
  const std::string token = "\"" + key + "\":\"";
  const size_t begin = json.find(token);
  if (begin == std::string::npos) return false;
  const size_t value_begin = begin + token.size();
  const size_t end = json.find('"', value_begin);
  if (end == std::string::npos) return false;
  *value = json.substr(value_begin, end - value_begin);
  return true;
}

std::string Sha256File(const std::string& path, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    *error = "Cannot open file for SHA256: " + path;
    return "";
  }
  SHA256_CTX context;
  SHA256_Init(&context);
  std::array<char, 1 << 16> buffer;
  while (input) {
    input.read(buffer.data(), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) {
      SHA256_Update(&context, buffer.data(), static_cast<size_t>(count));
    }
  }
  if (!input.eof()) {
    *error = "Failed while hashing file: " + path;
    return "";
  }
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
  SHA256_Final(digest.data(), &context);
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const uint8_t value : digest) {
    stream << std::setw(2) << static_cast<int>(value);
  }
  return stream.str();
}

std::string RequireEnvironment(const char* name, std::string* error) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    *error = std::string("Fidelity capture requires environment variable ") +
             name;
    return "";
  }
  return value;
}

void WriteIntVector(const std::vector<int32_t>& values, Writer* writer) {
  writer->U64(values.size());
  for (const int32_t value : values) writer->I32(value);
}

bool ReadIntVector(Reader* reader, std::vector<int32_t>* values) {
  uint64_t size = 0;
  if (!reader->U64(&size) || size > reader->Remaining() / 4) return false;
  values->resize(size);
  for (int32_t& value : *values) {
    if (!reader->I32(&value)) return false;
  }
  return true;
}

void WriteProvenance(const FidelityProvenanceSnapshot& value,
                     Writer* writer) {
  writer->String(value.executable_path);
  writer->String(value.executable_sha256);
  writer->String(value.libceres_path);
  writer->String(value.libceres_sha256);
  writer->String(value.git_head);
  writer->String(value.dirty_diff_sha256);
}

bool ReadProvenance(Reader* reader, FidelityProvenanceSnapshot* value) {
  return reader->String(&value->executable_path) &&
         reader->String(&value->executable_sha256) &&
         reader->String(&value->libceres_path) &&
         reader->String(&value->libceres_sha256) &&
         reader->String(&value->git_head) &&
         reader->String(&value->dirty_diff_sha256);
}

void WriteEffectiveOptions(const EffectiveCeresOptionsSnapshot& value,
                           Writer* writer) {
  writer->I32(value.minimizer_type);
  writer->I32(value.trust_region_strategy_type);
  writer->I32(value.dogleg_type);
  writer->I32(value.linear_solver_type);
  writer->I32(value.preconditioner_type);
  writer->I32(value.visibility_clustering_type);
  writer->I32(value.dense_linear_algebra_library_type);
  writer->I32(value.sparse_linear_algebra_library_type);
  writer->I32(value.logging_type);
  writer->I32(value.original_requested_num_threads);
  writer->I32(value.original_requested_num_linear_solver_threads);
  writer->I32(value.num_threads);
  writer->I32(value.num_linear_solver_threads);
  writer->I32(value.min_num_residuals_for_multi_threading);
  writer->I32(value.max_num_iterations);
  writer->I32(value.min_linear_solver_iterations);
  writer->I32(value.max_linear_solver_iterations);
  writer->I32(value.max_num_consecutive_invalid_steps);
  writer->I32(value.max_consecutive_nonmonotonic_steps);
  writer->Double(value.max_solver_time_in_seconds);
  writer->Double(value.function_tolerance);
  writer->Double(value.gradient_tolerance);
  writer->Double(value.parameter_tolerance);
  writer->Double(value.initial_trust_region_radius);
  writer->Double(value.min_trust_region_radius);
  writer->Double(value.max_trust_region_radius);
  writer->Double(value.min_lm_diagonal);
  writer->Double(value.max_lm_diagonal);
  writer->Double(value.min_relative_decrease);
  writer->Double(value.eta);
  writer->Double(value.inner_iteration_tolerance);
  writer->Double(value.gradient_check_relative_precision);
  writer->Double(value.gradient_check_numeric_derivative_relative_step_size);
  writer->U8(value.jacobi_scaling);
  writer->U8(value.use_nonmonotonic_steps);
  writer->U8(value.use_inner_iterations);
  writer->U8(value.use_explicit_schur_complement);
  writer->U8(value.use_postordering);
  writer->U8(value.dynamic_sparsity);
  writer->U8(value.minimizer_progress_to_stdout);
  writer->U8(value.check_gradients);
  writer->U8(value.update_state_every_iteration);
  writer->U8(value.linear_solver_ordering_present);
  writer->U8(value.inner_iteration_ordering_present);
  writer->U8(value.evaluation_callback_present);
  writer->U64(value.callback_count);
  writer->U64(value.trust_region_dump_iteration_count);
  writer->String(value.trust_region_problem_dump_directory);
  writer->I32(value.trust_region_problem_dump_format_type);
}

bool ReadEffectiveOptions(Reader* reader,
                          EffectiveCeresOptionsSnapshot* value) {
  uint8_t flags[12] = {};
  if (!reader->I32(&value->minimizer_type) ||
      !reader->I32(&value->trust_region_strategy_type) ||
      !reader->I32(&value->dogleg_type) ||
      !reader->I32(&value->linear_solver_type) ||
      !reader->I32(&value->preconditioner_type) ||
      !reader->I32(&value->visibility_clustering_type) ||
      !reader->I32(&value->dense_linear_algebra_library_type) ||
      !reader->I32(&value->sparse_linear_algebra_library_type) ||
      !reader->I32(&value->logging_type) ||
      !reader->I32(&value->original_requested_num_threads) ||
      !reader->I32(&value->original_requested_num_linear_solver_threads) ||
      !reader->I32(&value->num_threads) ||
      !reader->I32(&value->num_linear_solver_threads) ||
      !reader->I32(&value->min_num_residuals_for_multi_threading) ||
      !reader->I32(&value->max_num_iterations) ||
      !reader->I32(&value->min_linear_solver_iterations) ||
      !reader->I32(&value->max_linear_solver_iterations) ||
      !reader->I32(&value->max_num_consecutive_invalid_steps) ||
      !reader->I32(&value->max_consecutive_nonmonotonic_steps) ||
      !reader->Double(&value->max_solver_time_in_seconds) ||
      !reader->Double(&value->function_tolerance) ||
      !reader->Double(&value->gradient_tolerance) ||
      !reader->Double(&value->parameter_tolerance) ||
      !reader->Double(&value->initial_trust_region_radius) ||
      !reader->Double(&value->min_trust_region_radius) ||
      !reader->Double(&value->max_trust_region_radius) ||
      !reader->Double(&value->min_lm_diagonal) ||
      !reader->Double(&value->max_lm_diagonal) ||
      !reader->Double(&value->min_relative_decrease) ||
      !reader->Double(&value->eta) ||
      !reader->Double(&value->inner_iteration_tolerance) ||
      !reader->Double(&value->gradient_check_relative_precision) ||
      !reader->Double(
          &value->gradient_check_numeric_derivative_relative_step_size)) {
    return false;
  }
  for (uint8_t& flag : flags) {
    if (!reader->U8(&flag) || flag > 1) return false;
  }
  value->jacobi_scaling = flags[0];
  value->use_nonmonotonic_steps = flags[1];
  value->use_inner_iterations = flags[2];
  value->use_explicit_schur_complement = flags[3];
  value->use_postordering = flags[4];
  value->dynamic_sparsity = flags[5];
  value->minimizer_progress_to_stdout = flags[6];
  value->check_gradients = flags[7];
  value->update_state_every_iteration = flags[8];
  value->linear_solver_ordering_present = flags[9];
  value->inner_iteration_ordering_present = flags[10];
  value->evaluation_callback_present = flags[11];
  return reader->U64(&value->callback_count) &&
         reader->U64(&value->trust_region_dump_iteration_count) &&
         reader->String(&value->trust_region_problem_dump_directory) &&
         reader->I32(&value->trust_region_problem_dump_format_type);
}

void WriteConstraint(const ParameterConstraintSnapshot& value,
                     Writer* writer) {
  writer->U8(static_cast<uint8_t>(value.kind));
  writer->U64(value.entity_id);
  writer->U8(value.constant);
  writer->String(value.local_parameterization);
  WriteIntVector(value.constant_indices, writer);
}

bool ReadConstraint(Reader* reader, ParameterConstraintSnapshot* value) {
  uint8_t kind = 0;
  uint8_t constant = 0;
  if (!reader->U8(&kind) || kind > static_cast<uint8_t>(ParameterKind::kCamera) ||
      !reader->U64(&value->entity_id) || !reader->U8(&constant) ||
      constant > 1 || !reader->String(&value->local_parameterization) ||
      !ReadIntVector(reader, &value->constant_indices)) {
    return false;
  }
  value->kind = static_cast<ParameterKind>(kind);
  value->constant = constant;
  return true;
}

void WriteLoss(const LossSpecificationSnapshot& value, Writer* writer) {
  writer->String(value.residual_class);
  writer->String(value.type);
  writer->Double(value.scale);
  writer->U64(value.residual_block_count);
}

bool ReadLoss(Reader* reader, LossSpecificationSnapshot* value) {
  return reader->String(&value->residual_class) &&
         reader->String(&value->type) && reader->Double(&value->scale) &&
         reader->U64(&value->residual_block_count);
}

void WriteProblem(const FidelityProblemSnapshot& value, Writer* writer) {
  writer->U64(value.config_image_count);
  writer->U64(value.selected_image_count);
  writer->U64(value.reconstruction_image_count);
  writer->U64(value.snapshot_image_record_count);
  writer->U64(value.visual_residual_block_count);
  writer->U64(value.lidar_residual_block_count);
  writer->U64(value.residual_block_count);
  writer->U64(value.scalar_residual_count);
  writer->U64(value.parameter_block_count);
  writer->U64(value.ambient_parameter_dimension);
  writer->U64(value.tangent_parameter_dimension);
  writer->U64(value.constant_pose_count);
  writer->U64(value.constant_point_count);
  writer->U64(value.constant_camera_count);
  writer->U64(value.quaternion_manifold_count);
  writer->U64(value.subset_translation_count);
  writer->U8(value.has_bounds_or_constraints);
  writer->U64(value.camera_model_ids.size());
  for (const int32_t model : value.camera_model_ids) writer->I32(model);
  writer->U64(value.parameter_constraints.size());
  for (const auto& constraint : value.parameter_constraints) {
    WriteConstraint(constraint, writer);
  }
  writer->U64(value.loss_specifications.size());
  for (const auto& loss : value.loss_specifications) WriteLoss(loss, writer);
  writer->String(value.source_residual_order_sha256);
  writer->String(value.canonical_residual_order_sha256);
  writer->String(value.source_parameter_order_sha256);
  writer->String(value.canonical_parameter_order_sha256);
  writer->String(value.parameter_constraints_sha256);
  writer->String(value.loss_specification_sha256);
  writer->String(value.initial_state_sha256);
  writer->String(value.lidar_correspondence_sha256);
  writer->String(value.effective_options_sha256);
  writer->String(value.complete_fingerprint_sha256);
}

bool ReadProblem(Reader* reader, FidelityProblemSnapshot* value) {
  uint8_t constrained = 0;
  uint64_t size = 0;
  if (!reader->U64(&value->config_image_count) ||
      !reader->U64(&value->selected_image_count) ||
      !reader->U64(&value->reconstruction_image_count) ||
      !reader->U64(&value->snapshot_image_record_count) ||
      !reader->U64(&value->visual_residual_block_count) ||
      !reader->U64(&value->lidar_residual_block_count) ||
      !reader->U64(&value->residual_block_count) ||
      !reader->U64(&value->scalar_residual_count) ||
      !reader->U64(&value->parameter_block_count) ||
      !reader->U64(&value->ambient_parameter_dimension) ||
      !reader->U64(&value->tangent_parameter_dimension) ||
      !reader->U64(&value->constant_pose_count) ||
      !reader->U64(&value->constant_point_count) ||
      !reader->U64(&value->constant_camera_count) ||
      !reader->U64(&value->quaternion_manifold_count) ||
      !reader->U64(&value->subset_translation_count) ||
      !reader->U8(&constrained) || constrained > 1 || !reader->U64(&size) ||
      size > reader->Remaining() / 4) {
    return false;
  }
  value->has_bounds_or_constraints = constrained;
  value->camera_model_ids.resize(size);
  for (int32_t& model : value->camera_model_ids) {
    if (!reader->I32(&model)) return false;
  }
  if (!reader->U64(&size) || size > reader->Remaining() / 16) return false;
  value->parameter_constraints.resize(size);
  for (auto& constraint : value->parameter_constraints) {
    if (!ReadConstraint(reader, &constraint)) return false;
  }
  if (!reader->U64(&size) || size > reader->Remaining() / 16) return false;
  value->loss_specifications.resize(size);
  for (auto& loss : value->loss_specifications) {
    if (!ReadLoss(reader, &loss)) return false;
  }
  return reader->String(&value->source_residual_order_sha256) &&
         reader->String(&value->canonical_residual_order_sha256) &&
         reader->String(&value->source_parameter_order_sha256) &&
         reader->String(&value->canonical_parameter_order_sha256) &&
         reader->String(&value->parameter_constraints_sha256) &&
         reader->String(&value->loss_specification_sha256) &&
         reader->String(&value->initial_state_sha256) &&
         reader->String(&value->lidar_correspondence_sha256) &&
         reader->String(&value->effective_options_sha256) &&
         reader->String(&value->complete_fingerprint_sha256);
}

void WriteIteration(const CeresIterationSnapshot& value, Writer* writer) {
  writer->I32(value.iteration);
  writer->U8(value.step_is_valid);
  writer->U8(value.step_is_nonmonotonic);
  writer->U8(value.step_is_successful);
  writer->Double(value.cost);
  writer->Double(value.cost_change);
  writer->Double(value.gradient_max_norm);
  writer->Double(value.gradient_norm);
  writer->Double(value.step_norm);
  writer->Double(value.relative_decrease);
  writer->Double(value.trust_region_radius);
  writer->Double(value.eta);
  writer->Double(value.step_size);
  writer->I32(value.line_search_function_evaluations);
  writer->I32(value.line_search_gradient_evaluations);
  writer->I32(value.line_search_iterations);
  writer->I32(value.linear_solver_iterations);
  writer->Double(value.iteration_time_in_seconds);
  writer->Double(value.step_solver_time_in_seconds);
  writer->Double(value.cumulative_time_in_seconds);
}

bool ReadIteration(Reader* reader, CeresIterationSnapshot* value) {
  uint8_t valid = 0;
  uint8_t nonmonotonic = 0;
  uint8_t successful = 0;
  if (!reader->I32(&value->iteration) || !reader->U8(&valid) ||
      !reader->U8(&nonmonotonic) || !reader->U8(&successful) || valid > 1 ||
      nonmonotonic > 1 || successful > 1 || !reader->Double(&value->cost) ||
      !reader->Double(&value->cost_change) ||
      !reader->Double(&value->gradient_max_norm) ||
      !reader->Double(&value->gradient_norm) ||
      !reader->Double(&value->step_norm) ||
      !reader->Double(&value->relative_decrease) ||
      !reader->Double(&value->trust_region_radius) ||
      !reader->Double(&value->eta) || !reader->Double(&value->step_size) ||
      !reader->I32(&value->line_search_function_evaluations) ||
      !reader->I32(&value->line_search_gradient_evaluations) ||
      !reader->I32(&value->line_search_iterations) ||
      !reader->I32(&value->linear_solver_iterations) ||
      !reader->Double(&value->iteration_time_in_seconds) ||
      !reader->Double(&value->step_solver_time_in_seconds) ||
      !reader->Double(&value->cumulative_time_in_seconds)) {
    return false;
  }
  value->step_is_valid = valid;
  value->step_is_nonmonotonic = nonmonotonic;
  value->step_is_successful = successful;
  return true;
}

void WriteSummary(const CeresSummarySnapshot& value, Writer* writer) {
  writer->I32(value.minimizer_type);
  writer->I32(value.termination_type);
  writer->String(value.termination_message);
  writer->Double(value.initial_cost);
  writer->Double(value.final_cost);
  writer->Double(value.fixed_cost);
  writer->I32(value.successful_steps);
  writer->I32(value.unsuccessful_steps);
  writer->I32(value.invalid_steps);
  writer->I32(value.num_inner_iteration_steps);
  writer->I32(value.num_line_search_steps);
  writer->I32(value.num_linear_solves);
  writer->Double(value.preprocessor_time_in_seconds);
  writer->Double(value.minimizer_time_in_seconds);
  writer->Double(value.postprocessor_time_in_seconds);
  writer->Double(value.total_time_in_seconds);
  writer->Double(value.linear_solver_time_in_seconds);
  writer->I32(value.num_parameter_blocks);
  writer->I32(value.num_parameters);
  writer->I32(value.num_effective_parameters);
  writer->I32(value.num_residual_blocks);
  writer->I32(value.num_residuals);
  writer->I32(value.num_parameter_blocks_reduced);
  writer->I32(value.num_parameters_reduced);
  writer->I32(value.num_effective_parameters_reduced);
  writer->I32(value.num_residual_blocks_reduced);
  writer->I32(value.num_residuals_reduced);
  writer->U8(value.is_constrained);
  writer->I32(value.num_threads_given);
  writer->I32(value.num_threads_used);
  writer->I32(value.num_linear_solver_threads_given);
  writer->I32(value.num_linear_solver_threads_used);
  writer->I32(value.linear_solver_type_given);
  writer->I32(value.linear_solver_type_used);
  writer->I32(value.preconditioner_type_given);
  writer->I32(value.preconditioner_type_used);
  writer->I32(value.trust_region_strategy_type);
  writer->I32(value.dogleg_type);
  writer->I32(value.dense_linear_algebra_library_type);
  writer->I32(value.sparse_linear_algebra_library_type);
  writer->U8(value.inner_iterations_given);
  writer->U8(value.inner_iterations_used);
  WriteIntVector(value.linear_solver_ordering_given, writer);
  WriteIntVector(value.linear_solver_ordering_used, writer);
  WriteIntVector(value.inner_iteration_ordering_given, writer);
  WriteIntVector(value.inner_iteration_ordering_used, writer);
  writer->String(value.schur_structure_given);
  writer->String(value.schur_structure_used);
  writer->U64(value.iterations.size());
  for (const auto& iteration : value.iterations) {
    WriteIteration(iteration, writer);
  }
}

bool ReadSummary(Reader* reader, CeresSummarySnapshot* value) {
  uint8_t constrained = 0;
  uint8_t inner_given = 0;
  uint8_t inner_used = 0;
  uint64_t size = 0;
  if (!reader->I32(&value->minimizer_type) ||
      !reader->I32(&value->termination_type) ||
      !reader->String(&value->termination_message) ||
      !reader->Double(&value->initial_cost) ||
      !reader->Double(&value->final_cost) ||
      !reader->Double(&value->fixed_cost) ||
      !reader->I32(&value->successful_steps) ||
      !reader->I32(&value->unsuccessful_steps) ||
      !reader->I32(&value->invalid_steps) ||
      !reader->I32(&value->num_inner_iteration_steps) ||
      !reader->I32(&value->num_line_search_steps) ||
      !reader->I32(&value->num_linear_solves) ||
      !reader->Double(&value->preprocessor_time_in_seconds) ||
      !reader->Double(&value->minimizer_time_in_seconds) ||
      !reader->Double(&value->postprocessor_time_in_seconds) ||
      !reader->Double(&value->total_time_in_seconds) ||
      !reader->Double(&value->linear_solver_time_in_seconds) ||
      !reader->I32(&value->num_parameter_blocks) ||
      !reader->I32(&value->num_parameters) ||
      !reader->I32(&value->num_effective_parameters) ||
      !reader->I32(&value->num_residual_blocks) ||
      !reader->I32(&value->num_residuals) ||
      !reader->I32(&value->num_parameter_blocks_reduced) ||
      !reader->I32(&value->num_parameters_reduced) ||
      !reader->I32(&value->num_effective_parameters_reduced) ||
      !reader->I32(&value->num_residual_blocks_reduced) ||
      !reader->I32(&value->num_residuals_reduced) ||
      !reader->U8(&constrained) || constrained > 1 ||
      !reader->I32(&value->num_threads_given) ||
      !reader->I32(&value->num_threads_used) ||
      !reader->I32(&value->num_linear_solver_threads_given) ||
      !reader->I32(&value->num_linear_solver_threads_used) ||
      !reader->I32(&value->linear_solver_type_given) ||
      !reader->I32(&value->linear_solver_type_used) ||
      !reader->I32(&value->preconditioner_type_given) ||
      !reader->I32(&value->preconditioner_type_used) ||
      !reader->I32(&value->trust_region_strategy_type) ||
      !reader->I32(&value->dogleg_type) ||
      !reader->I32(&value->dense_linear_algebra_library_type) ||
      !reader->I32(&value->sparse_linear_algebra_library_type) ||
      !reader->U8(&inner_given) || !reader->U8(&inner_used) ||
      constrained > 1 || inner_given > 1 || inner_used > 1 ||
      !ReadIntVector(reader, &value->linear_solver_ordering_given) ||
      !ReadIntVector(reader, &value->linear_solver_ordering_used) ||
      !ReadIntVector(reader, &value->inner_iteration_ordering_given) ||
      !ReadIntVector(reader, &value->inner_iteration_ordering_used) ||
      !reader->String(&value->schur_structure_given) ||
      !reader->String(&value->schur_structure_used) || !reader->U64(&size) ||
      size > reader->Remaining() / 32) {
    return false;
  }
  value->is_constrained = constrained;
  value->inner_iterations_given = inner_given;
  value->inner_iterations_used = inner_used;
  value->iterations.resize(size);
  for (auto& iteration : value->iterations) {
    if (!ReadIteration(reader, &iteration)) return false;
  }
  return true;
}

void WriteRecord(const CeresFidelityRecord& value, Writer* writer) {
  writer->String(value.record_kind);
  writer->String(value.run_id);
  WriteProvenance(value.provenance, writer);
  writer->String(value.snapshot_id);
  writer->String(value.snapshot_manifest_path);
  writer->String(value.snapshot_payload_path);
  writer->String(value.snapshot_payload_sha256);
  writer->String(value.snapshot_manifest_sha256);
  writer->String(value.post_state_manifest_path);
  writer->String(value.post_state_payload_path);
  writer->String(value.post_state_payload_sha256);
  writer->String(value.final_state_sha256);
  WriteEffectiveOptions(value.effective_options, writer);
  WriteProblem(value.problem, writer);
  WriteSummary(value.summary, writer);
}

bool ReadRecord(Reader* reader, CeresFidelityRecord* value) {
  return reader->String(&value->record_kind) &&
         reader->String(&value->run_id) &&
         ReadProvenance(reader, &value->provenance) &&
         reader->String(&value->snapshot_id) &&
         reader->String(&value->snapshot_manifest_path) &&
         reader->String(&value->snapshot_payload_path) &&
         reader->String(&value->snapshot_payload_sha256) &&
         reader->String(&value->snapshot_manifest_sha256) &&
         reader->String(&value->post_state_manifest_path) &&
         reader->String(&value->post_state_payload_path) &&
         reader->String(&value->post_state_payload_sha256) &&
         reader->String(&value->final_state_sha256) &&
         ReadEffectiveOptions(reader, &value->effective_options) &&
         ReadProblem(reader, &value->problem) &&
         ReadSummary(reader, &value->summary);
}

template <typename T>
std::vector<int32_t> CopyIntVector(const std::vector<T>& input) {
  return std::vector<int32_t>(input.begin(), input.end());
}

std::string HashWriter(const Writer& writer) {
  return Sha256Hex(writer.Data());
}

void WriteOrder(const std::vector<OrderEntrySnapshot>& order,
                Writer* writer) {
  writer->U64(order.size());
  for (const auto& entry : order) {
    writer->U64(entry.source_index);
    writer->U8(static_cast<uint8_t>(entry.residual_kind));
    writer->U32(entry.image_id);
    writer->U32(entry.point2D_idx);
    writer->U64(entry.point3D_id);
  }
}

void WriteParameterOrder(const Snapshot& snapshot,
                         bool canonical,
                         Writer* writer) {
  writer->U64(snapshot.parameter_blocks_source_order.size());
  auto write_parameter = [writer](const ParameterBlockSnapshot& parameter) {
    writer->U64(parameter.source_index);
    writer->U8(static_cast<uint8_t>(parameter.kind));
    writer->U64(parameter.entity_id);
    writer->U32(parameter.ambient_size);
    writer->U32(parameter.tangent_size);
    writer->U8(parameter.constant);
  };
  if (!canonical) {
    for (const auto& parameter : snapshot.parameter_blocks_source_order) {
      write_parameter(parameter);
    }
  } else {
    for (const uint64_t index : snapshot.parameter_blocks_canonical_order) {
      writer->U64(index);
      if (index < snapshot.parameter_blocks_source_order.size()) {
        write_parameter(snapshot.parameter_blocks_source_order[index]);
      }
    }
  }
}

void WriteConstraints(
    const std::vector<ParameterConstraintSnapshot>& constraints,
    Writer* writer) {
  writer->U64(constraints.size());
  for (const auto& constraint : constraints) WriteConstraint(constraint, writer);
}

void WriteLossHashDomain(
    const Snapshot& snapshot,
    const std::vector<LossSpecificationSnapshot>& specifications,
    Writer* writer) {
  std::map<std::string, const LossSpecificationSnapshot*> lookup;
  for (const auto& specification : specifications) {
    lookup.emplace(specification.residual_class, &specification);
  }
  writer->U64(snapshot.source_insertion_order.size());
  for (const auto& entry : snapshot.source_insertion_order) {
    const std::string residual_class =
        entry.residual_kind == ResidualKind::kVisual ? "visual" : "lidar";
    writer->U64(entry.source_index);
    writer->String(residual_class);
    const auto it = lookup.find(residual_class);
    if (it == lookup.end()) {
      writer->String("missing");
      writer->Double(std::numeric_limits<double>::quiet_NaN());
    } else {
      writer->String(it->second->type);
      writer->Double(it->second->scale);
    }
  }
}

void WriteCanonicalState(const Snapshot& snapshot, Writer* writer) {
  std::vector<const CameraSnapshot*> cameras;
  std::vector<const ImageSnapshot*> images;
  std::vector<const PointSnapshot*> points;
  for (const auto& camera : snapshot.cameras) cameras.push_back(&camera);
  for (const auto& image : snapshot.images) images.push_back(&image);
  for (const auto& point : snapshot.points) points.push_back(&point);
  std::sort(cameras.begin(), cameras.end(),
            [](const CameraSnapshot* a, const CameraSnapshot* b) {
              return a->camera_id < b->camera_id;
            });
  std::sort(images.begin(), images.end(),
            [](const ImageSnapshot* a, const ImageSnapshot* b) {
              return a->image_id < b->image_id;
            });
  std::sort(points.begin(), points.end(),
            [](const PointSnapshot* a, const PointSnapshot* b) {
              return a->point3D_id < b->point3D_id;
            });
  writer->String("ceres-fidelity-canonical-state-v1");
  writer->U64(cameras.size());
  for (const CameraSnapshot* camera : cameras) {
    writer->U32(camera->camera_id);
    writer->I32(camera->model_id);
    writer->U64(camera->params.size());
    for (const double parameter : camera->params) writer->Double(parameter);
  }
  writer->U64(images.size());
  for (const ImageSnapshot* image : images) {
    writer->U32(image->image_id);
    for (const double value : image->qvec) writer->Double(value);
    for (const double value : image->tvec) writer->Double(value);
  }
  writer->U64(points.size());
  for (const PointSnapshot* point : points) {
    writer->U64(point->point3D_id);
    for (const double value : point->xyz) writer->Double(value);
  }
}

std::string ProblemCompleteHash(const FidelityProblemSnapshot& problem) {
  Writer writer;
  writer.String("ceres-fidelity-problem-v1");
  writer.U64(problem.config_image_count);
  writer.U64(problem.selected_image_count);
  writer.U64(problem.reconstruction_image_count);
  writer.U64(problem.visual_residual_block_count);
  writer.U64(problem.lidar_residual_block_count);
  writer.U64(problem.scalar_residual_count);
  writer.U64(problem.parameter_block_count);
  writer.U64(problem.ambient_parameter_dimension);
  writer.U64(problem.tangent_parameter_dimension);
  writer.U8(problem.has_bounds_or_constraints);
  writer.String(problem.source_residual_order_sha256);
  writer.String(problem.canonical_residual_order_sha256);
  writer.String(problem.source_parameter_order_sha256);
  writer.String(problem.canonical_parameter_order_sha256);
  writer.String(problem.parameter_constraints_sha256);
  writer.String(problem.loss_specification_sha256);
  writer.String(problem.initial_state_sha256);
  writer.String(problem.lidar_correspondence_sha256);
  writer.String(problem.effective_options_sha256);
  return HashWriter(writer);
}

}  // namespace

EffectiveCeresOptionsSnapshot CaptureEffectiveCeresOptions(
    const ceres::Solver::Options& original_requested,
    const ceres::Solver::Options& effective,
    int min_num_residuals_for_multi_threading) {
  EffectiveCeresOptionsSnapshot output;
  output.minimizer_type = effective.minimizer_type;
  output.trust_region_strategy_type = effective.trust_region_strategy_type;
  output.dogleg_type = effective.dogleg_type;
  output.linear_solver_type = effective.linear_solver_type;
  output.preconditioner_type = effective.preconditioner_type;
  output.visibility_clustering_type = effective.visibility_clustering_type;
  output.dense_linear_algebra_library_type =
      effective.dense_linear_algebra_library_type;
  output.sparse_linear_algebra_library_type =
      effective.sparse_linear_algebra_library_type;
  output.logging_type = effective.logging_type;
  output.original_requested_num_threads = original_requested.num_threads;
#if CERES_VERSION_MAJOR < 2
  output.original_requested_num_linear_solver_threads =
      original_requested.num_linear_solver_threads;
  output.num_linear_solver_threads = effective.num_linear_solver_threads;
#else
  output.original_requested_num_linear_solver_threads =
      original_requested.num_threads;
  output.num_linear_solver_threads = effective.num_threads;
#endif
  output.num_threads = effective.num_threads;
  output.min_num_residuals_for_multi_threading =
      min_num_residuals_for_multi_threading;
  output.max_num_iterations = effective.max_num_iterations;
  output.min_linear_solver_iterations = effective.min_linear_solver_iterations;
  output.max_linear_solver_iterations = effective.max_linear_solver_iterations;
  output.max_num_consecutive_invalid_steps =
      effective.max_num_consecutive_invalid_steps;
  output.max_consecutive_nonmonotonic_steps =
      effective.max_consecutive_nonmonotonic_steps;
  output.max_solver_time_in_seconds = effective.max_solver_time_in_seconds;
  output.function_tolerance = effective.function_tolerance;
  output.gradient_tolerance = effective.gradient_tolerance;
  output.parameter_tolerance = effective.parameter_tolerance;
  output.initial_trust_region_radius = effective.initial_trust_region_radius;
  output.min_trust_region_radius = effective.min_trust_region_radius;
  output.max_trust_region_radius = effective.max_trust_region_radius;
  output.min_lm_diagonal = effective.min_lm_diagonal;
  output.max_lm_diagonal = effective.max_lm_diagonal;
  output.min_relative_decrease = effective.min_relative_decrease;
  output.eta = effective.eta;
  output.inner_iteration_tolerance = effective.inner_iteration_tolerance;
  output.gradient_check_relative_precision =
      effective.gradient_check_relative_precision;
  output.gradient_check_numeric_derivative_relative_step_size =
      effective.gradient_check_numeric_derivative_relative_step_size;
  output.jacobi_scaling = effective.jacobi_scaling;
  output.use_nonmonotonic_steps = effective.use_nonmonotonic_steps;
  output.use_inner_iterations = effective.use_inner_iterations;
  output.use_explicit_schur_complement =
      effective.use_explicit_schur_complement;
  output.use_postordering = effective.use_postordering;
  output.dynamic_sparsity = effective.dynamic_sparsity;
  output.minimizer_progress_to_stdout =
      effective.minimizer_progress_to_stdout;
  output.check_gradients = effective.check_gradients;
  output.update_state_every_iteration = effective.update_state_every_iteration;
  output.linear_solver_ordering_present =
      effective.linear_solver_ordering != nullptr;
  output.inner_iteration_ordering_present =
      effective.inner_iteration_ordering != nullptr;
  output.evaluation_callback_present = effective.evaluation_callback != nullptr;
  output.callback_count = effective.callbacks.size();
  output.trust_region_dump_iteration_count =
      effective.trust_region_minimizer_iterations_to_dump.size();
  output.trust_region_problem_dump_directory =
      effective.trust_region_problem_dump_directory;
  output.trust_region_problem_dump_format_type =
      effective.trust_region_problem_dump_format_type;
  return output;
}

bool ApplyEffectiveCeresOptions(const EffectiveCeresOptionsSnapshot& input,
                                ceres::Solver::Options* options,
                                std::string* error) {
  if (options == nullptr || error == nullptr) return false;
  if (input.linear_solver_ordering_present ||
      input.inner_iteration_ordering_present || input.callback_count != 0 ||
      input.evaluation_callback_present ||
      input.trust_region_dump_iteration_count != 0) {
    *error = "Fidelity replay does not support recorded callbacks, custom "
             "orderings, evaluation callbacks, or diagnostic dumps";
    return false;
  }
  *options = ceres::Solver::Options();
  options->minimizer_type =
      static_cast<ceres::MinimizerType>(input.minimizer_type);
  options->trust_region_strategy_type =
      static_cast<ceres::TrustRegionStrategyType>(
          input.trust_region_strategy_type);
  options->dogleg_type = static_cast<ceres::DoglegType>(input.dogleg_type);
  options->linear_solver_type =
      static_cast<ceres::LinearSolverType>(input.linear_solver_type);
  options->preconditioner_type =
      static_cast<ceres::PreconditionerType>(input.preconditioner_type);
  options->visibility_clustering_type =
      static_cast<ceres::VisibilityClusteringType>(
          input.visibility_clustering_type);
  options->dense_linear_algebra_library_type =
      static_cast<ceres::DenseLinearAlgebraLibraryType>(
          input.dense_linear_algebra_library_type);
  options->sparse_linear_algebra_library_type =
      static_cast<ceres::SparseLinearAlgebraLibraryType>(
          input.sparse_linear_algebra_library_type);
  options->logging_type = static_cast<ceres::LoggingType>(input.logging_type);
  options->num_threads = input.num_threads;
#if CERES_VERSION_MAJOR < 2
  options->num_linear_solver_threads = input.num_linear_solver_threads;
#endif
  options->max_num_iterations = input.max_num_iterations;
  options->min_linear_solver_iterations = input.min_linear_solver_iterations;
  options->max_linear_solver_iterations = input.max_linear_solver_iterations;
  options->max_num_consecutive_invalid_steps =
      input.max_num_consecutive_invalid_steps;
  options->max_consecutive_nonmonotonic_steps =
      input.max_consecutive_nonmonotonic_steps;
  options->max_solver_time_in_seconds = input.max_solver_time_in_seconds;
  options->function_tolerance = input.function_tolerance;
  options->gradient_tolerance = input.gradient_tolerance;
  options->parameter_tolerance = input.parameter_tolerance;
  options->initial_trust_region_radius = input.initial_trust_region_radius;
  options->min_trust_region_radius = input.min_trust_region_radius;
  options->max_trust_region_radius = input.max_trust_region_radius;
  options->min_lm_diagonal = input.min_lm_diagonal;
  options->max_lm_diagonal = input.max_lm_diagonal;
  options->min_relative_decrease = input.min_relative_decrease;
  options->eta = input.eta;
  options->inner_iteration_tolerance = input.inner_iteration_tolerance;
  options->gradient_check_relative_precision =
      input.gradient_check_relative_precision;
  options->gradient_check_numeric_derivative_relative_step_size =
      input.gradient_check_numeric_derivative_relative_step_size;
  options->jacobi_scaling = input.jacobi_scaling;
  options->use_nonmonotonic_steps = input.use_nonmonotonic_steps;
  options->use_inner_iterations = input.use_inner_iterations;
  options->use_explicit_schur_complement =
      input.use_explicit_schur_complement;
  options->use_postordering = input.use_postordering;
  options->dynamic_sparsity = input.dynamic_sparsity;
  options->minimizer_progress_to_stdout = input.minimizer_progress_to_stdout;
  options->check_gradients = input.check_gradients;
  options->update_state_every_iteration = input.update_state_every_iteration;
  options->trust_region_problem_dump_directory =
      input.trust_region_problem_dump_directory;
  options->trust_region_problem_dump_format_type =
      static_cast<ceres::DumpFormatType>(
          input.trust_region_problem_dump_format_type);
  std::string validation_error;
  if (!options->IsValid(&validation_error)) {
    *error = "Recorded effective Ceres options are invalid: " +
             validation_error;
    return false;
  }
  return true;
}

std::string EffectiveCeresOptionsSha256(
    const EffectiveCeresOptionsSnapshot& options) {
  Writer writer;
  writer.String("effective-ceres-options-v1");
  WriteEffectiveOptions(options, &writer);
  return HashWriter(writer);
}

CeresSummarySnapshot CaptureCeresSummary(
    const ceres::Solver::Summary& summary) {
  CeresSummarySnapshot output;
  output.minimizer_type = summary.minimizer_type;
  output.termination_type = summary.termination_type;
  output.termination_message = summary.message;
  output.initial_cost = summary.initial_cost;
  output.final_cost = summary.final_cost;
  output.fixed_cost = summary.fixed_cost;
  output.successful_steps = summary.num_successful_steps;
  output.unsuccessful_steps = summary.num_unsuccessful_steps;
  output.invalid_steps = static_cast<int32_t>(std::count_if(
      summary.iterations.begin(), summary.iterations.end(),
      [](const ceres::IterationSummary& value) {
        return value.iteration > 0 && !value.step_is_valid;
      }));
  output.num_inner_iteration_steps = summary.num_inner_iteration_steps;
  output.num_line_search_steps = summary.num_line_search_steps;
  output.num_linear_solves = summary.num_linear_solves;
  output.preprocessor_time_in_seconds = summary.preprocessor_time_in_seconds;
  output.minimizer_time_in_seconds = summary.minimizer_time_in_seconds;
  output.postprocessor_time_in_seconds = summary.postprocessor_time_in_seconds;
  output.total_time_in_seconds = summary.total_time_in_seconds;
  output.linear_solver_time_in_seconds = summary.linear_solver_time_in_seconds;
  output.num_parameter_blocks = summary.num_parameter_blocks;
  output.num_parameters = summary.num_parameters;
  output.num_effective_parameters = summary.num_effective_parameters;
  output.num_residual_blocks = summary.num_residual_blocks;
  output.num_residuals = summary.num_residuals;
  output.num_parameter_blocks_reduced = summary.num_parameter_blocks_reduced;
  output.num_parameters_reduced = summary.num_parameters_reduced;
  output.num_effective_parameters_reduced =
      summary.num_effective_parameters_reduced;
  output.num_residual_blocks_reduced = summary.num_residual_blocks_reduced;
  output.num_residuals_reduced = summary.num_residuals_reduced;
  output.is_constrained = summary.is_constrained;
  output.num_threads_given = summary.num_threads_given;
  output.num_threads_used = summary.num_threads_used;
  output.num_linear_solver_threads_given =
      summary.num_linear_solver_threads_given;
  output.num_linear_solver_threads_used =
      summary.num_linear_solver_threads_used;
  output.linear_solver_type_given = summary.linear_solver_type_given;
  output.linear_solver_type_used = summary.linear_solver_type_used;
  output.preconditioner_type_given = summary.preconditioner_type_given;
  output.preconditioner_type_used = summary.preconditioner_type_used;
  output.trust_region_strategy_type = summary.trust_region_strategy_type;
  output.dogleg_type = summary.dogleg_type;
  output.dense_linear_algebra_library_type =
      summary.dense_linear_algebra_library_type;
  output.sparse_linear_algebra_library_type =
      summary.sparse_linear_algebra_library_type;
  output.inner_iterations_given = summary.inner_iterations_given;
  output.inner_iterations_used = summary.inner_iterations_used;
  output.linear_solver_ordering_given =
      CopyIntVector(summary.linear_solver_ordering_given);
  output.linear_solver_ordering_used =
      CopyIntVector(summary.linear_solver_ordering_used);
  output.inner_iteration_ordering_given =
      CopyIntVector(summary.inner_iteration_ordering_given);
  output.inner_iteration_ordering_used =
      CopyIntVector(summary.inner_iteration_ordering_used);
  output.schur_structure_given = summary.schur_structure_given;
  output.schur_structure_used = summary.schur_structure_used;
  output.iterations.reserve(summary.iterations.size());
  for (const ceres::IterationSummary& source : summary.iterations) {
    CeresIterationSnapshot item;
    item.iteration = source.iteration;
    item.step_is_valid = source.step_is_valid;
    item.step_is_nonmonotonic = source.step_is_nonmonotonic;
    item.step_is_successful = source.step_is_successful;
    item.cost = source.cost;
    item.cost_change = source.cost_change;
    item.gradient_max_norm = source.gradient_max_norm;
    item.gradient_norm = source.gradient_norm;
    item.step_norm = source.step_norm;
    item.relative_decrease = source.relative_decrease;
    item.trust_region_radius = source.trust_region_radius;
    item.eta = source.eta;
    item.step_size = source.step_size;
    item.line_search_function_evaluations =
        source.line_search_function_evaluations;
    item.line_search_gradient_evaluations =
        source.line_search_gradient_evaluations;
    item.line_search_iterations = source.line_search_iterations;
    item.linear_solver_iterations = source.linear_solver_iterations;
    item.iteration_time_in_seconds = source.iteration_time_in_seconds;
    item.step_solver_time_in_seconds = source.step_solver_time_in_seconds;
    item.cumulative_time_in_seconds = source.cumulative_time_in_seconds;
    output.iterations.push_back(item);
  }
  return output;
}

std::string CanonicalStateSha256(const Snapshot& snapshot) {
  Writer writer;
  WriteCanonicalState(snapshot, &writer);
  return HashWriter(writer);
}

bool UpdateSnapshotStateFromReconstruction(const Reconstruction& reconstruction,
                                           Snapshot* snapshot,
                                           std::string* error) {
  if (snapshot == nullptr || error == nullptr) return false;
  for (CameraSnapshot& camera : snapshot->cameras) {
    if (!reconstruction.ExistsCamera(camera.camera_id)) {
      *error = "Post-solve reconstruction is missing camera=" +
               std::to_string(camera.camera_id);
      return false;
    }
    const Camera& source = reconstruction.Camera(camera.camera_id);
    camera.params.assign(source.Params().begin(), source.Params().end());
  }
  for (ImageSnapshot& image : snapshot->images) {
    if (!reconstruction.ExistsImage(image.image_id)) {
      *error = "Post-solve reconstruction is missing image=" +
               std::to_string(image.image_id);
      return false;
    }
    const Image& source = reconstruction.Image(image.image_id);
    for (size_t i = 0; i < 4; ++i) image.qvec[i] = source.Qvec()[i];
    for (size_t i = 0; i < 3; ++i) image.tvec[i] = source.Tvec()[i];
  }
  for (PointSnapshot& point : snapshot->points) {
    if (!reconstruction.ExistsPoint3D(point.point3D_id)) {
      *error = "Post-solve reconstruction is missing point3D=" +
               std::to_string(point.point3D_id);
      return false;
    }
    const Point3D& source = reconstruction.Point3D(point.point3D_id);
    for (size_t i = 0; i < 3; ++i) point.xyz[i] = source.XYZ()[i];
  }
  return true;
}

bool RestoreReconstructionStateFromSnapshot(const Snapshot& snapshot,
                                            Reconstruction* reconstruction,
                                            std::string* error) {
  if (reconstruction == nullptr || error == nullptr) return false;
  for (const CameraSnapshot& camera : snapshot.cameras) {
    if (!reconstruction->ExistsCamera(camera.camera_id)) {
      *error = "Reconstruction is missing camera=" +
               std::to_string(camera.camera_id);
      return false;
    }
    Camera& destination = reconstruction->Camera(camera.camera_id);
    if (destination.Params().size() != camera.params.size()) {
      *error = "Camera parameter size mismatch while restoring camera=" +
               std::to_string(camera.camera_id);
      return false;
    }
    std::copy(camera.params.begin(), camera.params.end(),
              destination.Params().begin());
  }
  for (const ImageSnapshot& image : snapshot.images) {
    if (!reconstruction->ExistsImage(image.image_id)) {
      *error = "Reconstruction is missing image=" +
               std::to_string(image.image_id);
      return false;
    }
    Image& destination = reconstruction->Image(image.image_id);
    for (size_t i = 0; i < 4; ++i) destination.Qvec()[i] = image.qvec[i];
    for (size_t i = 0; i < 3; ++i) destination.Tvec()[i] = image.tvec[i];
  }
  for (const PointSnapshot& point : snapshot.points) {
    if (!reconstruction->ExistsPoint3D(point.point3D_id)) {
      *error = "Reconstruction is missing point3D=" +
               std::to_string(point.point3D_id);
      return false;
    }
    Point3D& destination = reconstruction->Point3D(point.point3D_id);
    for (size_t i = 0; i < 3; ++i) destination.XYZ()[i] = point.xyz[i];
  }
  return true;
}

bool CaptureFidelityProvenance(FidelityProvenanceSnapshot* provenance,
                               std::string* error) {
  if (provenance == nullptr || error == nullptr) return false;
  std::array<char, 4096> executable_path;
  const ssize_t size =
      readlink("/proc/self/exe", executable_path.data(), executable_path.size());
  if (size <= 0 || static_cast<size_t>(size) >= executable_path.size()) {
    *error = "Cannot resolve /proc/self/exe for fidelity provenance";
    return false;
  }
  provenance->executable_path.assign(executable_path.data(), size);
  provenance->executable_sha256 =
      Sha256File(provenance->executable_path, error);
  if (provenance->executable_sha256.empty()) return false;
  provenance->libceres_path =
      RequireEnvironment("COLMAP_FIDELITY_LIBCERES_PATH", error);
  if (provenance->libceres_path.empty()) return false;
  provenance->libceres_sha256 =
      Sha256File(provenance->libceres_path, error);
  if (provenance->libceres_sha256.empty()) return false;
  provenance->git_head =
      RequireEnvironment("COLMAP_FIDELITY_GIT_HEAD", error);
  if (provenance->git_head.empty()) return false;
  provenance->dirty_diff_sha256 =
      RequireEnvironment("COLMAP_FIDELITY_DIRTY_DIFF_SHA256", error);
  return !provenance->dirty_diff_sha256.empty();
}

FidelityProblemSnapshot BuildFidelityProblemSnapshot(
    const Snapshot& snapshot,
    const SnapshotIntegrity& integrity,
    const EffectiveCeresOptionsSnapshot& effective_options,
    uint64_t config_image_count,
    uint64_t scalar_residual_count,
    uint64_t parameter_block_count,
    uint64_t ambient_parameter_dimension,
    const std::vector<ParameterConstraintSnapshot>& parameter_constraints,
    const std::vector<LossSpecificationSnapshot>& loss_specifications,
    bool has_bounds_or_constraints) {
  FidelityProblemSnapshot output;
  output.config_image_count = config_image_count;
  output.selected_image_count = static_cast<uint64_t>(std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const ImageSnapshot& image) { return image.selected; }));
  output.reconstruction_image_count = snapshot.metadata.registered_image_count;
  output.snapshot_image_record_count = snapshot.images.size();
  output.visual_residual_block_count = snapshot.observations.size();
  output.lidar_residual_block_count = snapshot.lidar.size();
  output.residual_block_count = snapshot.source_insertion_order.size();
  output.scalar_residual_count = scalar_residual_count;
  output.parameter_block_count = parameter_block_count;
  output.ambient_parameter_dimension = ambient_parameter_dimension;
  output.tangent_parameter_dimension = 0;
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    if (!parameter.constant) {
      output.tangent_parameter_dimension += parameter.tangent_size;
    }
  }
  output.constant_pose_count = static_cast<uint64_t>(std::count_if(
      snapshot.images.begin(), snapshot.images.end(),
      [](const ImageSnapshot& image) { return image.pose_constant; }));
  output.constant_point_count = static_cast<uint64_t>(std::count_if(
      snapshot.points.begin(), snapshot.points.end(),
      [](const PointSnapshot& point) { return point.constant; }));
  output.constant_camera_count = static_cast<uint64_t>(std::count_if(
      snapshot.cameras.begin(), snapshot.cameras.end(),
      [](const CameraSnapshot& camera) { return camera.constant; }));
  output.quaternion_manifold_count = static_cast<uint64_t>(std::count_if(
      parameter_constraints.begin(), parameter_constraints.end(),
      [](const ParameterConstraintSnapshot& value) {
        return value.local_parameterization == "quaternion";
      }));
  output.subset_translation_count = static_cast<uint64_t>(std::count_if(
      parameter_constraints.begin(), parameter_constraints.end(),
      [](const ParameterConstraintSnapshot& value) {
        return value.kind == ParameterKind::kTranslation &&
               !value.constant_indices.empty();
      }));
  output.has_bounds_or_constraints = has_bounds_or_constraints;
  for (const auto& camera : snapshot.cameras) {
    output.camera_model_ids.push_back(camera.model_id);
  }
  std::sort(output.camera_model_ids.begin(), output.camera_model_ids.end());
  output.camera_model_ids.erase(
      std::unique(output.camera_model_ids.begin(), output.camera_model_ids.end()),
      output.camera_model_ids.end());
  output.parameter_constraints = parameter_constraints;
  output.loss_specifications = loss_specifications;
  Writer writer;
  WriteOrder(snapshot.source_insertion_order, &writer);
  output.source_residual_order_sha256 = HashWriter(writer);
  writer = Writer();
  WriteOrder(snapshot.canonical_order, &writer);
  output.canonical_residual_order_sha256 = HashWriter(writer);
  writer = Writer();
  WriteParameterOrder(snapshot, false, &writer);
  output.source_parameter_order_sha256 = HashWriter(writer);
  writer = Writer();
  WriteParameterOrder(snapshot, true, &writer);
  output.canonical_parameter_order_sha256 = HashWriter(writer);
  writer = Writer();
  WriteConstraints(parameter_constraints, &writer);
  output.parameter_constraints_sha256 = HashWriter(writer);
  writer = Writer();
  WriteLossHashDomain(snapshot, loss_specifications, &writer);
  output.loss_specification_sha256 = HashWriter(writer);
  output.initial_state_sha256 = CanonicalStateSha256(snapshot);
  output.lidar_correspondence_sha256 =
      integrity.lidar_correspondence_sha256;
  output.effective_options_sha256 =
      EffectiveCeresOptionsSha256(effective_options);
  output.complete_fingerprint_sha256 = ProblemCompleteHash(output);
  return output;
}

bool WriteCeresFidelityRecord(const CeresFidelityRecord& record,
                              const std::string& output_dir,
                              CeresFidelityWriteResult* result,
                              std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  if (record.snapshot_id.empty() || record.run_id.empty() ||
      output_dir.empty()) {
    *error = "Fidelity record requires snapshot ID, run ID, and output dir";
    return false;
  }
  CreateDirIfNotExists(output_dir, true);
  const std::string prefix = JoinPaths(
      output_dir, record.snapshot_id + "-" + SanitizeToken(record.run_id) +
                      ".oracle");
  const std::string binary_path = prefix + ".bin";
  const std::string manifest_path = prefix + ".manifest.json";
  if (ExistsFile(binary_path) || ExistsFile(manifest_path)) {
    *error = "Refusing to reuse stale fidelity oracle: " + prefix;
    return false;
  }
  Writer writer;
  writer.Bytes(reinterpret_cast<const uint8_t*>(kFidelityMagic),
               sizeof(kFidelityMagic));
  writer.U32(kCeresFidelitySchemaVersion);
  writer.U32(kSnapshotLittleEndianMarker);
  WriteRecord(record, &writer);
  const std::vector<uint8_t> binary = writer.Data();
  const std::string binary_sha256 = Sha256Hex(binary);
  const std::string record_json = CeresFidelityRecordJson(record, 2);
  std::ostringstream manifest;
  manifest << std::setprecision(17)
           << "{\"schema_version\":" << kCeresFidelitySchemaVersion
           << ",\"binary_sha256\":\"" << binary_sha256
           << "\",\"record\":\n" << record_json << "\n}\n";
  const std::string manifest_text = manifest.str();
  const std::vector<uint8_t> manifest_bytes(manifest_text.begin(),
                                            manifest_text.end());
  const std::string binary_tmp = binary_path + ".tmp";
  const std::string manifest_tmp = manifest_path + ".tmp";
  std::remove(binary_tmp.c_str());
  std::remove(manifest_tmp.c_str());
  if (!WriteFile(binary_tmp, binary, error) ||
      !WriteFile(manifest_tmp, manifest_bytes, error)) {
    std::remove(binary_tmp.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }
  if (std::rename(binary_tmp.c_str(), binary_path.c_str()) != 0) {
    *error = "Cannot finalize fidelity binary: " +
             std::string(std::strerror(errno));
    std::remove(binary_tmp.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }
  if (std::rename(manifest_tmp.c_str(), manifest_path.c_str()) != 0) {
    *error = "Cannot finalize fidelity manifest: " +
             std::string(std::strerror(errno));
    std::remove(binary_path.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }
  result->prefix_path = prefix;
  result->binary_path = binary_path;
  result->manifest_path = manifest_path;
  result->binary_sha256 = binary_sha256;
  result->manifest_sha256 = Sha256Hex(manifest_text);
  return true;
}

bool ReadCeresFidelityRecord(const std::string& path,
                             CeresFidelityRecord* record,
                             CeresFidelityWriteResult* result,
                             std::string* error) {
  if (record == nullptr || result == nullptr || error == nullptr) return false;
  std::string prefix = RemoveSuffix(path, ".manifest.json");
  prefix = RemoveSuffix(prefix, ".bin");
  const std::string binary_path = prefix + ".bin";
  const std::string manifest_path = prefix + ".manifest.json";
  std::vector<uint8_t> binary;
  std::vector<uint8_t> manifest_bytes;
  if (!ReadFile(binary_path, &binary, error) ||
      !ReadFile(manifest_path, &manifest_bytes, error)) {
    return false;
  }
  const std::string manifest(manifest_bytes.begin(), manifest_bytes.end());
  std::string expected_sha256;
  if (!ExtractJsonString(manifest, "binary_sha256", &expected_sha256)) {
    *error = "Fidelity manifest is missing binary_sha256";
    return false;
  }
  const std::string actual_sha256 = Sha256Hex(binary);
  if (expected_sha256 != actual_sha256) {
    *error = "Fidelity binary SHA256 mismatch";
    return false;
  }
  Reader reader(binary);
  std::vector<uint8_t> magic;
  uint32_t version = 0;
  uint32_t endian = 0;
  if (!reader.Bytes(sizeof(kFidelityMagic), &magic) ||
      std::memcmp(magic.data(), kFidelityMagic, sizeof(kFidelityMagic)) != 0 ||
      !reader.U32(&version) || version != kCeresFidelitySchemaVersion ||
      !reader.U32(&endian) || endian != kSnapshotLittleEndianMarker ||
      !ReadRecord(&reader, record) || !reader.AtEnd()) {
    *error = "Invalid or unsupported fidelity oracle binary";
    return false;
  }
  result->prefix_path = prefix;
  result->binary_path = binary_path;
  result->manifest_path = manifest_path;
  result->binary_sha256 = actual_sha256;
  result->manifest_sha256 = Sha256Hex(manifest);
  return true;
}

namespace {

const char* BoolText(bool value) { return value ? "true" : "false"; }

std::string LossJson(const LossSpecificationSnapshot& value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "{\"residual_class\":\""
         << EscapeJson(value.residual_class) << "\",\"type\":\""
         << EscapeJson(value.type) << "\",\"scale\":" << value.scale
         << ",\"residual_block_count\":" << value.residual_block_count
         << '}';
  return stream.str();
}

std::string EnumNameMinimizer(int32_t value) {
  return ceres::MinimizerTypeToString(static_cast<ceres::MinimizerType>(value));
}
std::string EnumNameTrustRegion(int32_t value) {
  return ceres::TrustRegionStrategyTypeToString(
      static_cast<ceres::TrustRegionStrategyType>(value));
}
std::string EnumNameLinearSolver(int32_t value) {
  return ceres::LinearSolverTypeToString(
      static_cast<ceres::LinearSolverType>(value));
}
std::string EnumNameSparse(int32_t value) {
  return ceres::SparseLinearAlgebraLibraryTypeToString(
      static_cast<ceres::SparseLinearAlgebraLibraryType>(value));
}
std::string EnumNamePreconditioner(int32_t value) {
  return ceres::PreconditionerTypeToString(
      static_cast<ceres::PreconditionerType>(value));
}
std::string EnumNameTermination(int32_t value) {
  return ceres::TerminationTypeToString(
      static_cast<ceres::TerminationType>(value));
}

}  // namespace

std::string CeresFidelityRecordJson(const CeresFidelityRecord& record,
                                    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string i2(indent_spaces + 2, ' ');
  const std::string i4(indent_spaces + 4, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << indent << "{\n";
  stream << i2 << "\"record_kind\":\"" << EscapeJson(record.record_kind)
         << "\",\"run_id\":\"" << EscapeJson(record.run_id) << "\",\n";
  stream << i2 << "\"provenance\":{\"executable_path\":\""
         << EscapeJson(record.provenance.executable_path)
         << "\",\"executable_sha256\":\""
         << record.provenance.executable_sha256
         << "\",\"libceres_path\":\""
         << EscapeJson(record.provenance.libceres_path)
         << "\",\"libceres_sha256\":\""
         << record.provenance.libceres_sha256 << "\",\"git_head\":\""
         << record.provenance.git_head << "\",\"dirty_diff_sha256\":\""
         << record.provenance.dirty_diff_sha256 << "\"},\n";
  stream << i2 << "\"input\":{\"snapshot_id\":\""
         << EscapeJson(record.snapshot_id)
         << "\",\"snapshot_manifest_path\":\""
         << EscapeJson(record.snapshot_manifest_path)
         << "\",\"snapshot_payload_path\":\""
         << EscapeJson(record.snapshot_payload_path)
         << "\",\"snapshot_payload_sha256\":\""
         << record.snapshot_payload_sha256
         << "\",\"snapshot_manifest_sha256\":\""
         << record.snapshot_manifest_sha256 << "\"},\n";
  stream << i2 << "\"post_state\":{\"manifest_path\":\""
         << EscapeJson(record.post_state_manifest_path)
         << "\",\"payload_path\":\""
         << EscapeJson(record.post_state_payload_path)
         << "\",\"payload_sha256\":\""
         << record.post_state_payload_sha256
         << "\",\"canonical_state_sha256\":\""
         << record.final_state_sha256 << "\"},\n";
  const auto& option = record.effective_options;
  stream << i2 << "\"effective_options\":{\n"
         << i4 << "\"minimizer_type\":\""
         << EnumNameMinimizer(option.minimizer_type)
         << "\",\"trust_region_strategy\":\""
         << EnumNameTrustRegion(option.trust_region_strategy_type)
         << "\",\"dogleg_type_enum\":" << option.dogleg_type
         << ",\"requested_linear_solver\":\""
         << EnumNameLinearSolver(option.linear_solver_type)
         << "\",\"sparse_library\":\""
         << EnumNameSparse(option.sparse_linear_algebra_library_type)
         << "\",\"preconditioner\":\""
         << EnumNamePreconditioner(option.preconditioner_type) << "\",\n"
         << i4 << "\"visibility_clustering_type_enum\":"
         << option.visibility_clustering_type
         << ",\"dense_linear_algebra_library_type_enum\":"
         << option.dense_linear_algebra_library_type
         << ",\"sparse_linear_algebra_library_type_enum\":"
         << option.sparse_linear_algebra_library_type
         << ",\"logging_type_enum\":" << option.logging_type << ",\n"
         << i4 << "\"original_requested_num_threads\":"
         << option.original_requested_num_threads
         << ",\"num_threads\":" << option.num_threads
         << ",\"original_requested_num_linear_solver_threads\":"
         << option.original_requested_num_linear_solver_threads
         << ",\"num_linear_solver_threads\":"
         << option.num_linear_solver_threads
         << ",\"min_residuals_for_multithreading\":"
         << option.min_num_residuals_for_multi_threading << ",\n"
         << i4 << "\"max_num_iterations\":" << option.max_num_iterations
         << ",\"min_linear_solver_iterations\":"
         << option.min_linear_solver_iterations
         << ",\"max_linear_solver_iterations\":"
         << option.max_linear_solver_iterations
         << ",\"max_num_consecutive_invalid_steps\":"
         << option.max_num_consecutive_invalid_steps
         << ",\"function_tolerance\":" << option.function_tolerance
         << ",\"gradient_tolerance\":" << option.gradient_tolerance
         << ",\"parameter_tolerance\":" << option.parameter_tolerance
         << ",\n"
         << i4 << "\"initial_trust_region_radius\":"
         << option.initial_trust_region_radius
         << ",\"min_trust_region_radius\":"
         << option.min_trust_region_radius
         << ",\"max_trust_region_radius\":"
         << option.max_trust_region_radius
         << ",\"min_lm_diagonal\":" << option.min_lm_diagonal
         << ",\"max_lm_diagonal\":" << option.max_lm_diagonal
         << ",\"min_relative_decrease\":" << option.min_relative_decrease
         << ",\"eta\":" << option.eta << ",\"max_solver_time_seconds\":"
         << option.max_solver_time_in_seconds
         << ",\"inner_iteration_tolerance\":"
         << option.inner_iteration_tolerance
         << ",\"gradient_check_relative_precision\":"
         << option.gradient_check_relative_precision
         << ",\"gradient_check_numeric_derivative_relative_step_size\":"
         << option.gradient_check_numeric_derivative_relative_step_size
         << ",\n"
         << i4 << "\"jacobi_scaling\":" << BoolText(option.jacobi_scaling)
         << ",\"use_nonmonotonic_steps\":"
         << BoolText(option.use_nonmonotonic_steps)
         << ",\"max_consecutive_nonmonotonic_steps\":"
         << option.max_consecutive_nonmonotonic_steps
         << ",\"use_inner_iterations\":"
         << BoolText(option.use_inner_iterations)
         << ",\"use_explicit_schur_complement\":"
         << BoolText(option.use_explicit_schur_complement)
         << ",\"use_postordering\":" << BoolText(option.use_postordering)
         << ",\"dynamic_sparsity\":" << BoolText(option.dynamic_sparsity)
         << ",\"minimizer_progress_to_stdout\":"
         << BoolText(option.minimizer_progress_to_stdout)
         << ",\"check_gradients\":" << BoolText(option.check_gradients)
         << ",\"update_state_every_iteration\":"
         << BoolText(option.update_state_every_iteration)
         << ",\"linear_solver_ordering_present\":"
         << BoolText(option.linear_solver_ordering_present)
         << ",\"inner_iteration_ordering_present\":"
         << BoolText(option.inner_iteration_ordering_present)
         << ",\"evaluation_callback_present\":"
         << BoolText(option.evaluation_callback_present)
         << ",\"callback_count\":" << option.callback_count
         << ",\"dump_iteration_count\":"
         << option.trust_region_dump_iteration_count
         << ",\"trust_region_problem_dump_directory\":\""
         << EscapeJson(option.trust_region_problem_dump_directory)
         << "\",\"trust_region_problem_dump_format_type_enum\":"
         << option.trust_region_problem_dump_format_type << "},\n";
  const auto& problem = record.problem;
  stream << i2 << "\"problem\":{\"config_image_count\":"
         << problem.config_image_count << ",\"selected_image_count\":"
         << problem.selected_image_count
         << ",\"reconstruction_image_count\":"
         << problem.reconstruction_image_count
         << ",\"snapshot_image_record_count\":"
         << problem.snapshot_image_record_count
         << ",\"visual_residual_blocks\":"
         << problem.visual_residual_block_count
         << ",\"lidar_residual_blocks\":"
         << problem.lidar_residual_block_count
         << ",\"residual_blocks\":" << problem.residual_block_count
         << ",\"scalar_residuals\":" << problem.scalar_residual_count
         << ",\"parameter_blocks\":" << problem.parameter_block_count
         << ",\"ambient_dimension\":"
         << problem.ambient_parameter_dimension
         << ",\"tangent_dimension\":"
         << problem.tangent_parameter_dimension
         << ",\"constant_poses\":" << problem.constant_pose_count
         << ",\"constant_points\":" << problem.constant_point_count
         << ",\"constant_cameras\":" << problem.constant_camera_count
         << ",\"quaternion_manifolds\":"
         << problem.quaternion_manifold_count
         << ",\"subset_translations\":"
         << problem.subset_translation_count
         << ",\"has_bounds_or_constraints\":"
         << BoolText(problem.has_bounds_or_constraints) << ",\n"
         << i4 << "\"source_residual_order_sha256\":\""
         << problem.source_residual_order_sha256
         << "\",\"canonical_residual_order_sha256\":\""
         << problem.canonical_residual_order_sha256
         << "\",\"source_parameter_order_sha256\":\""
         << problem.source_parameter_order_sha256
         << "\",\"canonical_parameter_order_sha256\":\""
         << problem.canonical_parameter_order_sha256
         << "\",\"parameter_constraints_sha256\":\""
         << problem.parameter_constraints_sha256
         << "\",\"loss_specification_sha256\":\""
         << problem.loss_specification_sha256
         << "\",\"initial_state_sha256\":\""
         << problem.initial_state_sha256
         << "\",\"lidar_correspondence_sha256\":\""
         << problem.lidar_correspondence_sha256
         << "\",\"effective_options_sha256\":\""
         << problem.effective_options_sha256
         << "\",\"complete_fingerprint_sha256\":\""
         << problem.complete_fingerprint_sha256 << "\",\n"
         << i4 << "\"loss_specifications\":[";
  for (size_t i = 0; i < problem.loss_specifications.size(); ++i) {
    if (i != 0) stream << ',';
    stream << LossJson(problem.loss_specifications[i]);
  }
  stream << "],\"camera_model_ids\":[";
  for (size_t i = 0; i < problem.camera_model_ids.size(); ++i) {
    if (i != 0) stream << ',';
    stream << problem.camera_model_ids[i];
  }
  stream << "],\"parameter_constraints\":[";
  for (size_t i = 0; i < problem.parameter_constraints.size(); ++i) {
    if (i != 0) stream << ',';
    const auto& constraint = problem.parameter_constraints[i];
    stream << "{\"kind\":" << static_cast<int>(constraint.kind)
           << ",\"entity_id\":" << constraint.entity_id
           << ",\"constant\":" << BoolText(constraint.constant)
           << ",\"local_parameterization\":\""
           << EscapeJson(constraint.local_parameterization)
           << "\",\"constant_indices\":[";
    for (size_t j = 0; j < constraint.constant_indices.size(); ++j) {
      if (j != 0) stream << ',';
      stream << constraint.constant_indices[j];
    }
    stream << "]}";
  }
  stream << "]},\n";
  const auto& summary = record.summary;
  stream << i2 << "\"summary\":{\"initial_cost\":"
         << summary.initial_cost << ",\"final_cost\":" << summary.final_cost
         << ",\"fixed_cost\":" << summary.fixed_cost
         << ",\"successful_steps\":" << summary.successful_steps
         << ",\"unsuccessful_steps\":" << summary.unsuccessful_steps
         << ",\"invalid_steps\":" << summary.invalid_steps
         << ",\"termination_type\":\""
         << EnumNameTermination(summary.termination_type)
         << "\",\"termination_message\":\""
         << EscapeJson(summary.termination_message)
         << "\",\"requested_solver\":\""
         << EnumNameLinearSolver(summary.linear_solver_type_given)
         << "\",\"actual_solver\":\""
         << EnumNameLinearSolver(summary.linear_solver_type_used)
         << "\",\"requested_threads\":" << summary.num_threads_given
         << ",\"actual_threads\":" << summary.num_threads_used
         << ",\"requested_linear_solver_threads\":"
         << summary.num_linear_solver_threads_given
         << ",\"actual_linear_solver_threads\":"
         << summary.num_linear_solver_threads_used
         << ",\"preconditioner_given_enum\":"
         << summary.preconditioner_type_given
         << ",\"preconditioner_used_enum\":"
         << summary.preconditioner_type_used
         << ",\"trust_region_strategy_type_enum\":"
         << summary.trust_region_strategy_type
         << ",\"dogleg_type_enum\":" << summary.dogleg_type
         << ",\"dense_linear_algebra_library_type_enum\":"
         << summary.dense_linear_algebra_library_type
         << ",\"sparse_linear_algebra_library_type_enum\":"
         << summary.sparse_linear_algebra_library_type
         << ",\"inner_iterations_given\":"
         << BoolText(summary.inner_iterations_given)
         << ",\"inner_iterations_used\":"
         << BoolText(summary.inner_iterations_used)
         << ",\"is_constrained\":" << BoolText(summary.is_constrained)
         << ",\"num_parameter_blocks\":" << summary.num_parameter_blocks
         << ",\"num_parameters\":" << summary.num_parameters
         << ",\"num_effective_parameters\":"
         << summary.num_effective_parameters
         << ",\"num_residual_blocks\":" << summary.num_residual_blocks
         << ",\"num_residuals\":" << summary.num_residuals
         << ",\"num_parameter_blocks_reduced\":"
         << summary.num_parameter_blocks_reduced
         << ",\"num_parameters_reduced\":"
         << summary.num_parameters_reduced
         << ",\"num_effective_parameters_reduced\":"
         << summary.num_effective_parameters_reduced
         << ",\"num_residual_blocks_reduced\":"
         << summary.num_residual_blocks_reduced
         << ",\"num_residuals_reduced\":"
         << summary.num_residuals_reduced
         << ",\"num_inner_iteration_steps\":"
         << summary.num_inner_iteration_steps
         << ",\"num_line_search_steps\":"
         << summary.num_line_search_steps
         << ",\"num_linear_solves\":" << summary.num_linear_solves
         << ",\"preprocessor_time_seconds\":"
         << summary.preprocessor_time_in_seconds
         << ",\"minimizer_time_seconds\":"
         << summary.minimizer_time_in_seconds
         << ",\"postprocessor_time_seconds\":"
         << summary.postprocessor_time_in_seconds
         << ",\"total_time_seconds\":"
         << summary.total_time_in_seconds
         << ",\"linear_solver_time_seconds\":"
         << summary.linear_solver_time_in_seconds
         << ",\"iterations\":[\n";
  for (size_t i = 0; i < summary.iterations.size(); ++i) {
    const auto& item = summary.iterations[i];
    stream << i4 << "{\"iteration\":" << item.iteration
           << ",\"cost\":" << item.cost
           << ",\"cost_change\":" << item.cost_change
           << ",\"gradient_max_norm\":" << item.gradient_max_norm
           << ",\"gradient_norm\":" << item.gradient_norm
           << ",\"step_norm\":" << item.step_norm
           << ",\"relative_decrease\":" << item.relative_decrease
           << ",\"trust_region_radius\":" << item.trust_region_radius
           << ",\"eta\":" << item.eta
           << ",\"step_size\":" << item.step_size
           << ",\"linear_solver_iterations\":"
           << item.linear_solver_iterations
           << ",\"line_search_function_evaluations\":"
           << item.line_search_function_evaluations
           << ",\"line_search_gradient_evaluations\":"
           << item.line_search_gradient_evaluations
           << ",\"line_search_iterations\":"
           << item.line_search_iterations
           << ",\"iteration_time_seconds\":"
           << item.iteration_time_in_seconds
           << ",\"step_solver_time_seconds\":"
           << item.step_solver_time_in_seconds
           << ",\"cumulative_time_seconds\":"
           << item.cumulative_time_in_seconds << ",\"step_is_valid\":"
           << BoolText(item.step_is_valid) << ",\"step_is_successful\":"
           << BoolText(item.step_is_successful)
           << ",\"step_is_nonmonotonic\":"
           << BoolText(item.step_is_nonmonotonic) << '}';
    if (i + 1 != summary.iterations.size()) stream << ',';
    stream << '\n';
  }
  stream << i2 << "]}\n" << indent << '}';
  return stream.str();
}

namespace {

std::string TerminationCategory(const CeresSummarySnapshot& summary) {
  const auto type = static_cast<ceres::TerminationType>(summary.termination_type);
  if (type == ceres::FAILURE || type == ceres::USER_FAILURE) return "failure";
  std::string message = summary.termination_message;
  std::transform(message.begin(), message.end(), message.begin(),
                 [](unsigned char value) { return std::tolower(value); });
  if (message.find("gradient") != std::string::npos) return "gradient";
  if (message.find("function") != std::string::npos) return "function";
  if (message.find("parameter") != std::string::npos) return "parameter";
  if (message.find("maximum") != std::string::npos) return "max_iterations";
  if (message.find("trust region radius") != std::string::npos) {
    return "min_trust_region_radius";
  }
  return type == ceres::CONVERGENCE ? "other_convergence"
                                    : "other_no_convergence";
}

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = std::min(
      values.size() - 1,
      static_cast<size_t>(std::ceil(fraction * values.size()) - 1));
  return values[index];
}

class ErrorAccumulator {
 public:
  ErrorAccumulator(double atol, double rtol) : atol_(atol), rtol_(rtol) {}

  void Add(const std::string& id,
           double reference,
           double candidate,
           const std::vector<double>& reference_values = {},
           const std::vector<double>& candidate_values = {}) {
    const double absolute = std::abs(reference - candidate);
    const double scale = std::max(std::abs(reference), std::abs(candidate));
    const double relative = scale == 0.0 ? 0.0 : absolute / scale;
    ++summary_.count;
    uint64_t ref_bits = 0;
    uint64_t cand_bits = 0;
    std::memcpy(&ref_bits, &reference, sizeof(ref_bits));
    std::memcpy(&cand_bits, &candidate, sizeof(cand_bits));
    if (ref_bits != cand_bits) ++summary_.bitwise_differences;
    if (!std::isfinite(reference) || !std::isfinite(candidate) ||
        absolute > atol_ + rtol_ * scale) {
      ++summary_.tolerance_failures;
    }
    squared_sum_ += static_cast<long double>(absolute) * absolute;
    absolute_errors_.push_back(absolute);
    if (absolute > summary_.max_absolute_error || summary_.worst_id.empty()) {
      summary_.max_absolute_error = absolute;
      summary_.worst_id = id;
      summary_.worst_reference = reference;
      summary_.worst_candidate = candidate;
      summary_.worst_reference_values = reference_values.empty()
                                            ? std::vector<double>{reference}
                                            : reference_values;
      summary_.worst_candidate_values = candidate_values.empty()
                                            ? std::vector<double>{candidate}
                                            : candidate_values;
    }
    summary_.max_relative_error =
        std::max(summary_.max_relative_error, relative);
  }

  FidelityErrorSummary Finish() {
    if (summary_.count != 0) {
      summary_.rms_absolute_error = std::sqrt(
          static_cast<double>(squared_sum_ / summary_.count));
      summary_.p95_absolute_error = Percentile(absolute_errors_, 0.95);
    }
    return summary_;
  }

 private:
  double atol_;
  double rtol_;
  FidelityErrorSummary summary_;
  long double squared_sum_ = 0.0;
  std::vector<double> absolute_errors_;
};

bool SameProvenance(const FidelityProvenanceSnapshot& reference,
                    const FidelityProvenanceSnapshot& candidate) {
  return reference.executable_sha256 == candidate.executable_sha256 &&
         reference.libceres_sha256 == candidate.libceres_sha256 &&
         reference.git_head == candidate.git_head &&
         reference.dirty_diff_sha256 == candidate.dirty_diff_sha256;
}

bool SameDoubleBits(double lhs, double rhs) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits == rhs_bits;
}

std::string FirstEffectiveOptionMismatch(
    const EffectiveCeresOptionsSnapshot& reference,
    const EffectiveCeresOptionsSnapshot& candidate) {
#define CHECK_OPTION_FIELD(field)       \
  if (reference.field != candidate.field) return #field
#define CHECK_OPTION_DOUBLE(field)             \
  if (!SameDoubleBits(reference.field, candidate.field)) return #field
  CHECK_OPTION_FIELD(minimizer_type);
  CHECK_OPTION_FIELD(trust_region_strategy_type);
  CHECK_OPTION_FIELD(dogleg_type);
  CHECK_OPTION_FIELD(linear_solver_type);
  CHECK_OPTION_FIELD(preconditioner_type);
  CHECK_OPTION_FIELD(visibility_clustering_type);
  CHECK_OPTION_FIELD(dense_linear_algebra_library_type);
  CHECK_OPTION_FIELD(sparse_linear_algebra_library_type);
  CHECK_OPTION_FIELD(logging_type);
  CHECK_OPTION_FIELD(original_requested_num_threads);
  CHECK_OPTION_FIELD(original_requested_num_linear_solver_threads);
  CHECK_OPTION_FIELD(num_threads);
  CHECK_OPTION_FIELD(num_linear_solver_threads);
  CHECK_OPTION_FIELD(min_num_residuals_for_multi_threading);
  CHECK_OPTION_FIELD(max_num_iterations);
  CHECK_OPTION_FIELD(min_linear_solver_iterations);
  CHECK_OPTION_FIELD(max_linear_solver_iterations);
  CHECK_OPTION_FIELD(max_num_consecutive_invalid_steps);
  CHECK_OPTION_FIELD(max_consecutive_nonmonotonic_steps);
  CHECK_OPTION_DOUBLE(max_solver_time_in_seconds);
  CHECK_OPTION_DOUBLE(function_tolerance);
  CHECK_OPTION_DOUBLE(gradient_tolerance);
  CHECK_OPTION_DOUBLE(parameter_tolerance);
  CHECK_OPTION_DOUBLE(initial_trust_region_radius);
  CHECK_OPTION_DOUBLE(min_trust_region_radius);
  CHECK_OPTION_DOUBLE(max_trust_region_radius);
  CHECK_OPTION_DOUBLE(min_lm_diagonal);
  CHECK_OPTION_DOUBLE(max_lm_diagonal);
  CHECK_OPTION_DOUBLE(min_relative_decrease);
  CHECK_OPTION_DOUBLE(eta);
  CHECK_OPTION_DOUBLE(inner_iteration_tolerance);
  CHECK_OPTION_DOUBLE(gradient_check_relative_precision);
  CHECK_OPTION_DOUBLE(gradient_check_numeric_derivative_relative_step_size);
  CHECK_OPTION_FIELD(jacobi_scaling);
  CHECK_OPTION_FIELD(use_nonmonotonic_steps);
  CHECK_OPTION_FIELD(use_inner_iterations);
  CHECK_OPTION_FIELD(use_explicit_schur_complement);
  CHECK_OPTION_FIELD(use_postordering);
  CHECK_OPTION_FIELD(dynamic_sparsity);
  CHECK_OPTION_FIELD(minimizer_progress_to_stdout);
  CHECK_OPTION_FIELD(check_gradients);
  CHECK_OPTION_FIELD(update_state_every_iteration);
  CHECK_OPTION_FIELD(linear_solver_ordering_present);
  CHECK_OPTION_FIELD(inner_iteration_ordering_present);
  CHECK_OPTION_FIELD(evaluation_callback_present);
  CHECK_OPTION_FIELD(callback_count);
  CHECK_OPTION_FIELD(trust_region_dump_iteration_count);
  CHECK_OPTION_FIELD(trust_region_problem_dump_directory);
  CHECK_OPTION_FIELD(trust_region_problem_dump_format_type);
#undef CHECK_OPTION_DOUBLE
#undef CHECK_OPTION_FIELD
  return "";
}

void RecordProblemMismatch(const std::string& field,
                           CeresFidelityComparison* comparison) {
  if (comparison->first_problem_mismatch.empty()) {
    comparison->first_problem_mismatch = field;
  }
}

void CompareProblem(const FidelityProblemSnapshot& reference,
                    const FidelityProblemSnapshot& candidate,
                    CeresFidelityComparison* comparison) {
  if (reference.complete_fingerprint_sha256 !=
      candidate.complete_fingerprint_sha256) {
    if (reference.source_residual_order_sha256 !=
        candidate.source_residual_order_sha256) {
      RecordProblemMismatch("source_residual_order_sha256", comparison);
    } else if (reference.source_parameter_order_sha256 !=
               candidate.source_parameter_order_sha256) {
      RecordProblemMismatch("source_parameter_order_sha256", comparison);
    } else if (reference.loss_specification_sha256 !=
               candidate.loss_specification_sha256) {
      RecordProblemMismatch("loss_specification_sha256", comparison);
    } else if (reference.initial_state_sha256 !=
               candidate.initial_state_sha256) {
      RecordProblemMismatch("initial_state_sha256", comparison);
    } else if (reference.effective_options_sha256 !=
               candidate.effective_options_sha256) {
      RecordProblemMismatch("effective_options_sha256", comparison);
    } else if (reference.parameter_constraints_sha256 !=
               candidate.parameter_constraints_sha256) {
      RecordProblemMismatch("parameter_constraints_sha256", comparison);
    } else if (reference.lidar_correspondence_sha256 !=
               candidate.lidar_correspondence_sha256) {
      RecordProblemMismatch("lidar_correspondence_sha256", comparison);
    } else {
      RecordProblemMismatch("problem_count_or_identity", comparison);
    }
  }
  comparison->problem_fingerprint_pass =
      reference.complete_fingerprint_sha256 ==
      candidate.complete_fingerprint_sha256;
}

void RecordTraceStructureMismatch(int32_t iteration,
                                  const std::string& field,
                                  CeresFidelityComparison* comparison) {
  if (comparison->trace_structure_mismatch_count == 0) {
    comparison->first_trace_structure_mismatch_iteration = iteration;
    comparison->first_trace_structure_mismatch_field = field;
  }
  ++comparison->trace_structure_mismatch_count;
}

void CompareTrace(const CeresSummarySnapshot& reference,
                  const CeresSummarySnapshot& candidate,
                  CeresFidelityComparison* comparison) {
  if (reference.iterations.size() != candidate.iterations.size()) {
    RecordTraceStructureMismatch(-1, "trace_length", comparison);
  }
  const size_t common =
      std::min(reference.iterations.size(), candidate.iterations.size());
  for (size_t i = 0; i < common; ++i) {
    const auto& ref = reference.iterations[i];
    const auto& cand = candidate.iterations[i];
    if (ref.iteration != cand.iteration) {
      RecordTraceStructureMismatch(ref.iteration, "iteration", comparison);
    }
    if (ref.step_is_valid != cand.step_is_valid) {
      RecordTraceStructureMismatch(ref.iteration, "step_is_valid", comparison);
    }
    if (ref.step_is_successful != cand.step_is_successful) {
      RecordTraceStructureMismatch(ref.iteration, "step_is_successful",
                                   comparison);
    }
    if (ref.step_is_nonmonotonic != cand.step_is_nonmonotonic) {
      RecordTraceStructureMismatch(ref.iteration, "step_is_nonmonotonic",
                                   comparison);
    }
    if (ref.linear_solver_iterations != cand.linear_solver_iterations) {
      RecordTraceStructureMismatch(ref.iteration,
                                   "linear_solver_iterations", comparison);
    }
    if (ref.line_search_function_evaluations !=
        cand.line_search_function_evaluations) {
      RecordTraceStructureMismatch(
          ref.iteration, "line_search_function_evaluations", comparison);
    }
    if (ref.line_search_gradient_evaluations !=
        cand.line_search_gradient_evaluations) {
      RecordTraceStructureMismatch(
          ref.iteration, "line_search_gradient_evaluations", comparison);
    }
    if (ref.line_search_iterations != cand.line_search_iterations) {
      RecordTraceStructureMismatch(ref.iteration, "line_search_iterations",
                                   comparison);
    }
  }
  if (reference.successful_steps != candidate.successful_steps) {
    RecordTraceStructureMismatch(-1, "successful_steps", comparison);
  }
  if (reference.unsuccessful_steps != candidate.unsuccessful_steps) {
    RecordTraceStructureMismatch(-1, "unsuccessful_steps", comparison);
  }
  if (reference.invalid_steps != candidate.invalid_steps) {
    RecordTraceStructureMismatch(-1, "invalid_steps", comparison);
  }
  if (reference.termination_type != candidate.termination_type) {
    RecordTraceStructureMismatch(-1, "termination_type", comparison);
  }
  if (TerminationCategory(reference) != TerminationCategory(candidate)) {
    RecordTraceStructureMismatch(-1, "termination_category", comparison);
  }
  if (reference.num_linear_solves != candidate.num_linear_solves) {
    RecordTraceStructureMismatch(-1, "num_linear_solves", comparison);
  }
  comparison->trace_structure_pass =
      comparison->trace_structure_mismatch_count == 0;

  auto record_numeric = [&](int32_t iteration, const std::string& field,
                            double ref, double cand) {
    if (comparison->first_trace_numeric_divergence_iteration >= 0) return;
    comparison->first_trace_numeric_divergence_iteration = iteration;
    comparison->first_trace_numeric_divergence_field = field;
    comparison->first_trace_reference = ref;
    comparison->first_trace_candidate = cand;
    comparison->first_trace_absolute_error = std::abs(ref - cand);
    const double scale = std::max(std::abs(ref), std::abs(cand));
    comparison->first_trace_relative_error =
        scale == 0.0 ? 0.0 : comparison->first_trace_absolute_error / scale;
  };
  auto compare_numeric = [&](int32_t iteration, const std::string& field,
                             double ref, double cand) {
    const double scale = std::max(std::abs(ref), std::abs(cand));
    if (!std::isfinite(ref) || !std::isfinite(cand) ||
        std::abs(ref - cand) > 1e-12 + 1e-12 * scale) {
      record_numeric(iteration, field, ref, cand);
    }
  };
  for (size_t i = 0; i < common; ++i) {
    const auto& ref = reference.iterations[i];
    const auto& cand = candidate.iterations[i];
    compare_numeric(ref.iteration, "cost", ref.cost, cand.cost);
    compare_numeric(ref.iteration, "cost_change", ref.cost_change,
                    cand.cost_change);
    compare_numeric(ref.iteration, "gradient_max_norm", ref.gradient_max_norm,
                    cand.gradient_max_norm);
    compare_numeric(ref.iteration, "gradient_norm", ref.gradient_norm,
                    cand.gradient_norm);
    compare_numeric(ref.iteration, "step_norm", ref.step_norm, cand.step_norm);
    compare_numeric(ref.iteration, "relative_decrease", ref.relative_decrease,
                    cand.relative_decrease);
    compare_numeric(ref.iteration, "trust_region_radius",
                    ref.trust_region_radius, cand.trust_region_radius);
    compare_numeric(ref.iteration, "eta", ref.eta, cand.eta);
    compare_numeric(ref.iteration, "step_size", ref.step_size,
                    cand.step_size);
  }
  comparison->trace_numeric_pass =
      comparison->first_trace_numeric_divergence_iteration < 0;
}

template <typename SnapshotType, typename IdType>
bool IndexUniqueById(const std::vector<SnapshotType>& values,
                     IdType SnapshotType::*member,
                     const std::string& side,
                     const std::string& kind,
                     std::unordered_map<IdType, const SnapshotType*>* output,
                     std::string* error) {
  output->clear();
  for (const auto& value : values) {
    if (!output->emplace(value.*member, &value).second) {
      *error = "Fidelity state duplicate " + side + " " + kind + " ID=" +
               std::to_string(value.*member);
      return false;
    }
  }
  return true;
}

template <typename IdType, typename SnapshotType>
bool SameIndexedIds(
    const std::unordered_map<IdType, const SnapshotType*>& reference,
    const std::unordered_map<IdType, const SnapshotType*>& candidate,
    const std::string& kind,
    std::string* error) {
  for (const auto& item : reference) {
    if (candidate.count(item.first) == 0) {
      *error = "Fidelity state candidate is missing " + kind + " ID=" +
               std::to_string(item.first);
      return false;
    }
  }
  for (const auto& item : candidate) {
    if (reference.count(item.first) == 0) {
      *error = "Fidelity state candidate has extra " + kind + " ID=" +
               std::to_string(item.first);
      return false;
    }
  }
  if (reference.size() != candidate.size()) {
    *error = "Fidelity state " + kind + " ID count mismatch: reference=" +
             std::to_string(reference.size()) + " candidate=" +
             std::to_string(candidate.size());
    return false;
  }
  return true;
}

void CompareStates(const Snapshot& reference,
                   const Snapshot& candidate,
                   CeresFidelityComparison* comparison,
                   std::string* error) {
  std::unordered_map<uint32_t, const CameraSnapshot*> reference_cameras;
  std::unordered_map<uint32_t, const CameraSnapshot*> candidate_cameras;
  std::unordered_map<uint32_t, const ImageSnapshot*> reference_images;
  std::unordered_map<uint32_t, const ImageSnapshot*> candidate_images;
  std::unordered_map<uint64_t, const PointSnapshot*> reference_points;
  std::unordered_map<uint64_t, const PointSnapshot*> candidate_points;
  if (!IndexUniqueById(reference.cameras, &CameraSnapshot::camera_id,
                       "reference", "camera", &reference_cameras, error) ||
      !IndexUniqueById(candidate.cameras, &CameraSnapshot::camera_id,
                       "candidate", "camera", &candidate_cameras, error) ||
      !IndexUniqueById(reference.images, &ImageSnapshot::image_id,
                       "reference", "image", &reference_images, error) ||
      !IndexUniqueById(candidate.images, &ImageSnapshot::image_id,
                       "candidate", "image", &candidate_images, error) ||
      !IndexUniqueById(reference.points, &PointSnapshot::point3D_id,
                       "reference", "point", &reference_points, error) ||
      !IndexUniqueById(candidate.points, &PointSnapshot::point3D_id,
                       "candidate", "point", &candidate_points, error) ||
      !SameIndexedIds(reference_cameras, candidate_cameras, "camera", error) ||
      !SameIndexedIds(reference_images, candidate_images, "image", error) ||
      !SameIndexedIds(reference_points, candidate_points, "point", error)) {
    return;
  }
  ErrorAccumulator quaternion(1e-13, 1e-13);
  ErrorAccumulator rotation(1e-10, 0.0);
  ErrorAccumulator translation(1e-10, 1e-12);
  ErrorAccumulator points(1e-9, 1e-12);
  ErrorAccumulator cameras(1e-12, 1e-12);
  for (const auto& ref : reference.cameras) {
    const auto it = candidate_cameras.find(ref.camera_id);
    if (it == candidate_cameras.end() ||
        it->second->params.size() != ref.params.size()) {
      *error = "Fidelity state camera identity mismatch";
      return;
    }
    for (size_t i = 0; i < ref.params.size(); ++i) {
      cameras.Add("camera=" + std::to_string(ref.camera_id) + ":param=" +
                      std::to_string(i),
                  ref.params[i], it->second->params[i]);
    }
  }
  for (const auto& ref : reference.images) {
    const auto it = candidate_images.find(ref.image_id);
    if (it == candidate_images.end()) {
      *error = "Fidelity state image identity mismatch";
      return;
    }
    const ImageSnapshot& cand = *it->second;
    for (size_t i = 0; i < 4; ++i) {
      quaternion.Add("image=" + std::to_string(ref.image_id) + ":q=" +
                         std::to_string(i),
                     ref.qvec[i], cand.qvec[i]);
    }
    double ref_norm = 0.0;
    double cand_norm = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < 4; ++i) {
      ref_norm += ref.qvec[i] * ref.qvec[i];
      cand_norm += cand.qvec[i] * cand.qvec[i];
      dot += ref.qvec[i] * cand.qvec[i];
    }
    dot = std::abs(dot) / std::sqrt(ref_norm * cand_norm);
    dot = std::max(-1.0, std::min(1.0, dot));
    const double degrees = 2.0 * std::acos(dot) * 180.0 / std::acos(-1.0);
    rotation.Add("image=" + std::to_string(ref.image_id), 0.0, degrees,
                 std::vector<double>(ref.qvec.begin(), ref.qvec.end()),
                 std::vector<double>(cand.qvec.begin(), cand.qvec.end()));
    double translation_distance = 0.0;
    double translation_scale = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const double difference = ref.tvec[i] - cand.tvec[i];
      translation_distance += difference * difference;
      translation_scale += ref.tvec[i] * ref.tvec[i];
    }
    translation.Add(
        "image=" + std::to_string(ref.image_id),
        std::sqrt(translation_scale),
        std::sqrt(translation_scale) + std::sqrt(translation_distance),
        std::vector<double>(ref.tvec.begin(), ref.tvec.end()),
        std::vector<double>(cand.tvec.begin(), cand.tvec.end()));
  }
  for (const auto& ref : reference.points) {
    const auto it = candidate_points.find(ref.point3D_id);
    if (it == candidate_points.end()) {
      *error = "Fidelity state point identity mismatch";
      return;
    }
    double distance = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const double difference = ref.xyz[i] - it->second->xyz[i];
      distance += difference * difference;
      norm += ref.xyz[i] * ref.xyz[i];
    }
    points.Add(
        "point3D=" + std::to_string(ref.point3D_id), std::sqrt(norm),
        std::sqrt(norm) + std::sqrt(distance),
        std::vector<double>(ref.xyz.begin(), ref.xyz.end()),
        std::vector<double>(it->second->xyz.begin(), it->second->xyz.end()));
  }
  comparison->quaternion_ambient = quaternion.Finish();
  comparison->rotation_degrees = rotation.Finish();
  comparison->translation = translation.Finish();
  comparison->points = points.Finish();
  comparison->cameras = cameras.Finish();
  comparison->final_state_bitwise_identical =
      CanonicalStateSha256(reference) == CanonicalStateSha256(candidate);
  comparison->final_state_strict_pass =
      comparison->quaternion_ambient.tolerance_failures == 0 &&
      comparison->rotation_degrees.tolerance_failures == 0 &&
      comparison->translation.tolerance_failures == 0 &&
      comparison->points.tolerance_failures == 0 &&
      comparison->cameras.tolerance_failures == 0;
}

}  // namespace

bool CompareCeresFidelityRecords(const CeresFidelityRecord& reference,
                                 const Snapshot& reference_state,
                                 const CeresFidelityRecord& candidate,
                                 const Snapshot& candidate_state,
                                 CeresFidelityComparison* comparison,
                                 std::string* error) {
  if (comparison == nullptr || error == nullptr) return false;
  *comparison = CeresFidelityComparison();
  comparison->provenance_pass =
      SameProvenance(reference.provenance, candidate.provenance);
  CompareProblem(reference.problem, candidate.problem, comparison);
  comparison->first_option_mismatch = FirstEffectiveOptionMismatch(
      reference.effective_options, candidate.effective_options);
  comparison->effective_options_pass =
      comparison->first_option_mismatch.empty() &&
      reference.problem.effective_options_sha256 ==
          candidate.problem.effective_options_sha256;
  if (!comparison->effective_options_pass) {
    if (comparison->first_option_mismatch.empty()) {
      comparison->first_option_mismatch = "effective_options_sha256";
    }
  }
  comparison->actual_options_pass =
      candidate.summary.minimizer_type ==
          candidate.effective_options.minimizer_type &&
      candidate.summary.trust_region_strategy_type ==
          candidate.effective_options.trust_region_strategy_type &&
      candidate.summary.dogleg_type ==
          candidate.effective_options.dogleg_type &&
      candidate.summary.linear_solver_type_given ==
          candidate.effective_options.linear_solver_type &&
      candidate.summary.preconditioner_type_given ==
          candidate.effective_options.preconditioner_type &&
      candidate.summary.dense_linear_algebra_library_type ==
          candidate.effective_options.dense_linear_algebra_library_type &&
      candidate.summary.sparse_linear_algebra_library_type ==
          candidate.effective_options.sparse_linear_algebra_library_type &&
      candidate.summary.num_threads_given ==
          candidate.effective_options.num_threads &&
      candidate.summary.num_linear_solver_threads_given ==
          candidate.effective_options.num_linear_solver_threads &&
      candidate.summary.inner_iterations_given ==
          candidate.effective_options.use_inner_iterations &&
      candidate.summary.linear_solver_type_used ==
          reference.summary.linear_solver_type_used &&
      candidate.summary.preconditioner_type_used ==
          reference.summary.preconditioner_type_used &&
      candidate.summary.num_threads_used == reference.summary.num_threads_used &&
      candidate.summary.num_linear_solver_threads_used ==
          reference.summary.num_linear_solver_threads_used &&
      candidate.summary.is_constrained == reference.summary.is_constrained;
  CompareTrace(reference.summary, candidate.summary, comparison);
  CompareStates(reference_state, candidate_state, comparison, error);
  if (!error->empty()) return false;
  const bool single_thread = reference.summary.num_threads_used == 1 &&
                             candidate.summary.num_threads_used == 1;
  comparison->pass =
      comparison->provenance_pass && comparison->problem_fingerprint_pass &&
      comparison->effective_options_pass && comparison->actual_options_pass &&
      comparison->trace_structure_pass && comparison->trace_numeric_pass &&
      comparison->final_state_strict_pass &&
      (!single_thread || comparison->final_state_bitwise_identical);
  return true;
}

namespace {

void ErrorSummaryJson(std::ostringstream* stream,
                      const FidelityErrorSummary& value) {
  *stream << std::setprecision(17) << "{\"count\":" << value.count
          << ",\"bitwise_differences\":" << value.bitwise_differences
          << ",\"tolerance_failures\":" << value.tolerance_failures
          << ",\"max_absolute_error\":" << value.max_absolute_error
          << ",\"max_relative_error\":" << value.max_relative_error
          << ",\"rms_absolute_error\":" << value.rms_absolute_error
          << ",\"p95_absolute_error\":" << value.p95_absolute_error
          << ",\"worst_id\":\"" << EscapeJson(value.worst_id)
          << "\",\"worst_reference\":" << value.worst_reference
          << ",\"worst_candidate\":" << value.worst_candidate
          << ",\"worst_reference_values\":[";
  for (size_t i = 0; i < value.worst_reference_values.size(); ++i) {
    if (i != 0) *stream << ',';
    *stream << value.worst_reference_values[i];
  }
  *stream << "],\"worst_candidate_values\":[";
  for (size_t i = 0; i < value.worst_candidate_values.size(); ++i) {
    if (i != 0) *stream << ',';
    *stream << value.worst_candidate_values[i];
  }
  *stream << "]}";
}

}  // namespace

std::string CeresFidelityComparisonJson(
    const CeresFidelityComparison& comparison,
    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string i2(indent_spaces + 2, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17) << indent << "{\n";
  stream << i2 << "\"pass\":" << BoolText(comparison.pass)
         << ",\"provenance_pass\":"
         << BoolText(comparison.provenance_pass)
         << ",\"problem_fingerprint_pass\":"
         << BoolText(comparison.problem_fingerprint_pass)
         << ",\"effective_options_pass\":"
         << BoolText(comparison.effective_options_pass)
         << ",\"actual_options_pass\":"
         << BoolText(comparison.actual_options_pass)
         << ",\"first_option_mismatch\":\""
         << EscapeJson(comparison.first_option_mismatch)
         << "\",\"first_problem_mismatch\":\""
         << EscapeJson(comparison.first_problem_mismatch) << "\",\n";
  stream << i2 << "\"trace_structure\":{\"pass\":"
         << BoolText(comparison.trace_structure_pass)
         << ",\"mismatch_count\":"
         << comparison.trace_structure_mismatch_count
         << ",\"first_mismatch_iteration\":"
         << comparison.first_trace_structure_mismatch_iteration
         << ",\"first_mismatch_field\":\""
         << EscapeJson(comparison.first_trace_structure_mismatch_field)
         << "\"},\n";
  stream << i2 << "\"trace_numeric\":{\"pass\":"
         << BoolText(comparison.trace_numeric_pass)
         << ",\"first_divergence_iteration\":"
         << comparison.first_trace_numeric_divergence_iteration
         << ",\"first_divergence_field\":\""
         << EscapeJson(comparison.first_trace_numeric_divergence_field)
         << "\",\"reference\":" << comparison.first_trace_reference
         << ",\"candidate\":" << comparison.first_trace_candidate
         << ",\"absolute_error\":"
         << comparison.first_trace_absolute_error
         << ",\"relative_error\":"
         << comparison.first_trace_relative_error << "},\n";
  stream << i2 << "\"final_state\":{\"bitwise_identical\":"
         << BoolText(comparison.final_state_bitwise_identical)
         << ",\"strict_pass\":"
         << BoolText(comparison.final_state_strict_pass)
         << ",\"quaternion_ambient\":";
  ErrorSummaryJson(&stream, comparison.quaternion_ambient);
  stream << ",\"rotation_degrees\":";
  ErrorSummaryJson(&stream, comparison.rotation_degrees);
  stream << ",\"translation\":";
  ErrorSummaryJson(&stream, comparison.translation);
  stream << ",\"points\":";
  ErrorSummaryJson(&stream, comparison.points);
  stream << ",\"cameras\":";
  ErrorSummaryJson(&stream, comparison.cameras);
  stream << "}\n" << indent << '}';
  return stream.str();
}

namespace {

struct MutableSnapshotLookup {
  std::unordered_map<uint32_t, CameraSnapshot*> cameras;
  std::unordered_map<uint32_t, ImageSnapshot*> images;
  std::unordered_map<uint64_t, PointSnapshot*> points;
  std::vector<const ObservationSnapshot*> observations_by_source;
  std::vector<const LidarSnapshot*> lidar_by_source;
};

bool BuildMutableLookup(Snapshot* state,
                        MutableSnapshotLookup* lookup,
                        std::string* error) {
  const size_t residual_count = state->source_insertion_order.size();
  lookup->observations_by_source.assign(residual_count, nullptr);
  lookup->lidar_by_source.assign(residual_count, nullptr);
  for (auto& camera : state->cameras) {
    if (!lookup->cameras.emplace(camera.camera_id, &camera).second) {
      *error = "Duplicate replay camera";
      return false;
    }
  }
  for (auto& image : state->images) {
    if (!lookup->images.emplace(image.image_id, &image).second) {
      *error = "Duplicate replay image";
      return false;
    }
  }
  for (auto& point : state->points) {
    if (!lookup->points.emplace(point.point3D_id, &point).second) {
      *error = "Duplicate replay point";
      return false;
    }
  }
  for (const auto& observation : state->observations) {
    if (observation.source_index >= residual_count ||
        lookup->observations_by_source[observation.source_index] != nullptr) {
      *error = "Invalid replay visual source index";
      return false;
    }
    lookup->observations_by_source[observation.source_index] = &observation;
  }
  for (const auto& lidar : state->lidar) {
    if (lidar.source_index >= residual_count ||
        lookup->lidar_by_source[lidar.source_index] != nullptr) {
      *error = "Invalid replay LiDAR source index";
      return false;
    }
    lookup->lidar_by_source[lidar.source_index] = &lidar;
  }
  return true;
}

ceres::CostFunction* CreateVariableVisualCost(int32_t model_id,
                                               const std::array<double, 2>& xy,
                                               std::string* error) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::kModelId:                                               \
    return BundleAdjustmentCostFunction<CameraModel>::Create(               \
        Eigen::Vector2d(xy[0], xy[1]));
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  *error = "Unsupported camera model in fidelity replay: " +
           std::to_string(model_id);
  return nullptr;
}

ceres::CostFunction* CreateConstantVisualCost(
    int32_t model_id,
    const ImageSnapshot& image,
    const std::array<double, 2>& xy,
    std::string* error) {
  const Eigen::Vector4d quaternion(image.qvec[0], image.qvec[1], image.qvec[2],
                                   image.qvec[3]);
  const Eigen::Vector3d translation(image.tvec[0], image.tvec[1],
                                    image.tvec[2]);
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::kModelId:                                               \
    return BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(   \
        quaternion, translation, Eigen::Vector2d(xy[0], xy[1]));
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  *error = "Unsupported camera model in fidelity replay: " +
           std::to_string(model_id);
  return nullptr;
}

ceres::LossFunction* CreateLoss(const LossSpecificationSnapshot& loss,
                                std::string* error) {
  if (!std::isfinite(loss.scale)) {
    *error = "Non-finite fidelity loss scale";
    return nullptr;
  }
  if (loss.type == "TRIVIAL" || loss.type == "trivial") {
    return new ceres::TrivialLoss();
  }
  if (loss.type == "SOFT_L1" || loss.type == "soft_l1") {
    if (loss.scale <= 0.0) {
      *error = "SOFT_L1 fidelity loss scale must be positive";
      return nullptr;
    }
    return new ceres::SoftLOneLoss(loss.scale);
  }
  if (loss.type == "CAUCHY" || loss.type == "cauchy") {
    if (loss.scale <= 0.0) {
      *error = "CAUCHY fidelity loss scale must be positive";
      return nullptr;
    }
    return new ceres::CauchyLoss(loss.scale);
  }
  *error = "Unsupported fidelity loss type: " + loss.type;
  return nullptr;
}

const LossSpecificationSnapshot* FindLoss(
    const std::vector<LossSpecificationSnapshot>& losses,
    const std::string& residual_class) {
  for (const auto& loss : losses) {
    if (loss.residual_class == residual_class) return &loss;
  }
  return nullptr;
}

bool SameLoss(const LossSpecificationSnapshot& lhs,
              const LossSpecificationSnapshot& rhs) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs.scale, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs.scale, sizeof(rhs_bits));
  return lhs.type == rhs.type && lhs_bits == rhs_bits;
}

using ParameterKey = std::pair<ParameterKind, uint64_t>;

struct ReplayProblemBuild {
  std::unique_ptr<ceres::Problem> problem;
  Snapshot fingerprint_snapshot;
  std::vector<ParameterConstraintSnapshot> constraints;
};

bool BuildReplayProblem(
    Snapshot* state,
    const std::vector<OrderEntrySnapshot>& source_order,
    const std::vector<ParameterConstraintSnapshot>& recorded_constraints,
    const std::vector<LossSpecificationSnapshot>& losses,
    ReplayProblemBuild* output,
    std::string* error) {
  MutableSnapshotLookup lookup;
  if (!BuildMutableLookup(state, &lookup, error)) return false;
  output->problem.reset(new ceres::Problem());
  const LossSpecificationSnapshot* visual_loss_spec =
      FindLoss(losses, "visual");
  const LossSpecificationSnapshot* lidar_loss_spec = FindLoss(losses, "lidar");
  if ((state->observations.size() != 0 && visual_loss_spec == nullptr) ||
      (state->lidar.size() != 0 && lidar_loss_spec == nullptr)) {
    *error = "Fidelity replay is missing a per-class loss specification";
    return false;
  }
  ceres::LossFunction* visual_loss = nullptr;
  ceres::LossFunction* lidar_loss = nullptr;
  if (visual_loss_spec != nullptr) {
    visual_loss = CreateLoss(*visual_loss_spec, error);
    if (visual_loss == nullptr) return false;
  }
  if (lidar_loss_spec != nullptr) {
    if (visual_loss_spec != nullptr && SameLoss(*visual_loss_spec, *lidar_loss_spec)) {
      lidar_loss = visual_loss;
    } else {
      lidar_loss = CreateLoss(*lidar_loss_spec, error);
      if (lidar_loss == nullptr) return false;
    }
  }

  std::map<ParameterKey, double*> parameter_pointers;
  std::set<double*> seen_pointers;
  std::vector<ParameterBlockSnapshot> parameter_order;
  auto track_parameter = [&](ParameterKind kind, uint64_t entity_id,
                             double* pointer) {
    parameter_pointers.emplace(std::make_pair(kind, entity_id), pointer);
    if (!seen_pointers.insert(pointer).second) return;
    ParameterBlockSnapshot parameter;
    parameter.source_index = parameter_order.size();
    parameter.kind = kind;
    parameter.entity_id = entity_id;
    parameter_order.push_back(parameter);
  };

  for (const OrderEntrySnapshot& entry : source_order) {
    if (entry.source_index >= state->source_insertion_order.size()) {
      *error = "Fidelity source order contains an invalid source index";
      return false;
    }
    if (entry.residual_kind == ResidualKind::kVisual) {
      const ObservationSnapshot* observation =
          lookup.observations_by_source[entry.source_index];
      if (observation == nullptr) {
        *error = "Fidelity source order is missing a visual residual";
        return false;
      }
      ImageSnapshot* image = lookup.images.at(observation->image_id);
      PointSnapshot* point = lookup.points.at(observation->point3D_id);
      CameraSnapshot* camera = lookup.cameras.at(image->camera_id);
      ceres::CostFunction* cost = nullptr;
      if (observation->pose_constant) {
        cost = CreateConstantVisualCost(camera->model_id, *image,
                                        observation->xy, error);
        if (cost == nullptr) return false;
        track_parameter(ParameterKind::kPoint3D, point->point3D_id,
                        point->xyz.data());
        track_parameter(ParameterKind::kCamera, camera->camera_id,
                        camera->params.data());
        output->problem->AddResidualBlock(cost, visual_loss, point->xyz.data(),
                                          camera->params.data());
      } else {
        cost = CreateVariableVisualCost(camera->model_id, observation->xy,
                                        error);
        if (cost == nullptr) return false;
        track_parameter(ParameterKind::kQuaternion, image->image_id,
                        image->qvec.data());
        track_parameter(ParameterKind::kTranslation, image->image_id,
                        image->tvec.data());
        track_parameter(ParameterKind::kPoint3D, point->point3D_id,
                        point->xyz.data());
        track_parameter(ParameterKind::kCamera, camera->camera_id,
                        camera->params.data());
        output->problem->AddResidualBlock(
            cost, visual_loss, image->qvec.data(), image->tvec.data(),
            point->xyz.data(), camera->params.data());
      }
    } else {
      const LidarSnapshot* lidar = lookup.lidar_by_source[entry.source_index];
      if (lidar == nullptr) {
        *error = "Fidelity source order is missing a LiDAR residual";
        return false;
      }
      PointSnapshot* point = lookup.points.at(lidar->point3D_id);
      Eigen::Matrix<double, 4, 1> plane;
      for (size_t i = 0; i < 4; ++i) plane[i] = lidar->plane[i];
      ceres::CostFunction* cost =
          BundleAdjustmentLidarCostFunction::Create(plane, lidar->weight);
      track_parameter(ParameterKind::kPoint3D, point->point3D_id,
                      point->xyz.data());
      output->problem->AddResidualBlock(cost, lidar_loss, point->xyz.data());
    }
  }

  std::map<ParameterKey, const ParameterConstraintSnapshot*> constraint_lookup;
  for (const auto& constraint : recorded_constraints) {
    if (!constraint_lookup
             .emplace(std::make_pair(constraint.kind, constraint.entity_id),
                      &constraint)
             .second) {
      *error = "Duplicate recorded fidelity parameter constraint";
      return false;
    }
  }
  output->constraints.clear();
  output->constraints.reserve(parameter_order.size());
  for (ParameterBlockSnapshot& parameter : parameter_order) {
    const ParameterKey key(parameter.kind, parameter.entity_id);
    const auto constraint_it = constraint_lookup.find(key);
    const auto pointer_it = parameter_pointers.find(key);
    if (constraint_it == constraint_lookup.end() ||
        pointer_it == parameter_pointers.end()) {
      *error = "Fidelity parameter order lacks a recorded constraint";
      return false;
    }
    const ParameterConstraintSnapshot& constraint = *constraint_it->second;
    double* pointer = pointer_it->second;
    if (constraint.constant) {
      output->problem->SetParameterBlockConstant(pointer);
    } else if (constraint.local_parameterization == "quaternion") {
      SetQuaternionManifold(output->problem.get(), pointer);
    } else if (constraint.local_parameterization == "subset") {
      std::vector<int> constant_indices(constraint.constant_indices.begin(),
                                        constraint.constant_indices.end());
      SetSubsetManifold(output->problem->ParameterBlockSize(pointer),
                        constant_indices, output->problem.get(), pointer);
    } else if (constraint.local_parameterization != "euclidean") {
      *error = "Unsupported recorded local parameterization: " +
               constraint.local_parameterization;
      return false;
    }
    parameter.ambient_size = output->problem->ParameterBlockSize(pointer);
    parameter.tangent_size = output->problem->ParameterBlockLocalSize(pointer);
    parameter.constant = output->problem->IsParameterBlockConstant(pointer);
    if (parameter.constant != constraint.constant) {
      *error = "Fidelity parameter constant state was not restored";
      return false;
    }
    output->constraints.push_back(constraint);
  }

  output->fingerprint_snapshot = *state;
  output->fingerprint_snapshot.source_insertion_order = source_order;
  output->fingerprint_snapshot.canonical_order = source_order;
  std::sort(output->fingerprint_snapshot.canonical_order.begin(),
            output->fingerprint_snapshot.canonical_order.end(),
            [](const OrderEntrySnapshot& lhs,
               const OrderEntrySnapshot& rhs) {
              return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                              lhs.point3D_id, lhs.source_index) <
                     std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                              rhs.point3D_id, rhs.source_index);
            });
  output->fingerprint_snapshot.parameter_blocks_source_order = parameter_order;
  output->fingerprint_snapshot.parameter_blocks_canonical_order.resize(
      parameter_order.size());
  for (size_t i = 0; i < parameter_order.size(); ++i) {
    output->fingerprint_snapshot.parameter_blocks_canonical_order[i] = i;
  }
  std::sort(
      output->fingerprint_snapshot.parameter_blocks_canonical_order.begin(),
      output->fingerprint_snapshot.parameter_blocks_canonical_order.end(),
      [&parameter_order](uint64_t lhs, uint64_t rhs) {
        const auto& a = parameter_order[lhs];
        const auto& b = parameter_order[rhs];
        return std::tie(a.kind, a.entity_id, a.source_index) <
               std::tie(b.kind, b.entity_id, b.source_index);
      });
  return true;
}

bool ApplyPerturbation(const std::string& perturbation,
                       std::vector<OrderEntrySnapshot>* source_order,
                       std::vector<LossSpecificationSnapshot>* losses,
                       EffectiveCeresOptionsSnapshot* options,
                       std::string* error) {
  if (perturbation.empty() || perturbation == "none") return true;
  if (perturbation == "option_invalid_plus_one") {
    ++options->max_num_consecutive_invalid_steps;
    return true;
  }
  if (perturbation == "option_max_linear_plus_one") {
    ++options->max_linear_solver_iterations;
    return true;
  }
  if (perturbation == "loss_scale_plus_0_1") {
    if (losses->empty()) {
      *error = "Cannot perturb missing loss specifications";
      return false;
    }
    losses->front().scale += 0.1;
    return true;
  }
  if (perturbation == "swap_first_two_residuals") {
    if (source_order->size() < 2) {
      *error = "Cannot perturb a source order with fewer than two residuals";
      return false;
    }
    std::swap((*source_order)[0], (*source_order)[1]);
    return true;
  }
  *error = "Unknown fidelity perturbation: " + perturbation;
  return false;
}

}  // namespace

bool RunOriginalFidelityReplay(const std::string& snapshot_path,
                               const std::string& oracle_path,
                               const std::string& output_dir,
                               const std::string& run_id,
                               const std::string& perturbation,
                               CeresFidelityReplayResult* result,
                               std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  *result = CeresFidelityReplayResult();
  result->perturbation = perturbation.empty() ? "none" : perturbation;
  CeresFidelityRecord oracle;
  CeresFidelityWriteResult oracle_read;
  if (!ReadCeresFidelityRecord(oracle_path, &oracle, &oracle_read, error)) {
    return false;
  }
  result->oracle_record = oracle;
  Snapshot initial_state;
  SnapshotReadResult snapshot_read;
  if (!ReadSnapshot(snapshot_path, &initial_state, &snapshot_read, error)) {
    return false;
  }
  if (initial_state.metadata.snapshot_id != oracle.snapshot_id ||
      snapshot_read.integrity.payload_sha256 !=
          oracle.snapshot_payload_sha256) {
    *error = "Fidelity oracle does not belong to the supplied snapshot";
    return false;
  }
  Snapshot original_final_state;
  SnapshotReadResult original_final_read;
  if (!ReadSnapshot(oracle.post_state_manifest_path, &original_final_state,
                    &original_final_read, error)) {
    return false;
  }
  if (original_final_read.integrity.payload_sha256 !=
          oracle.post_state_payload_sha256 ||
      CanonicalStateSha256(original_final_state) != oracle.final_state_sha256) {
    *error = "Original fidelity post-state integrity mismatch";
    return false;
  }

  Snapshot replay_state = initial_state;
  std::vector<OrderEntrySnapshot> source_order =
      initial_state.source_insertion_order;
  std::vector<LossSpecificationSnapshot> losses =
      oracle.problem.loss_specifications;
  EffectiveCeresOptionsSnapshot effective = oracle.effective_options;
  if (!ApplyPerturbation(result->perturbation, &source_order, &losses,
                         &effective, error)) {
    return false;
  }
  ReplayProblemBuild build;
  if (!BuildReplayProblem(&replay_state, source_order,
                          oracle.problem.parameter_constraints, losses,
                          &build, error)) {
    return false;
  }
  CeresFidelityRecord replay;
  replay.record_kind = "original_fidelity_replay";
  replay.run_id = run_id.empty() ? "replay" : run_id;
  if (!CaptureFidelityProvenance(&replay.provenance, error)) return false;
  replay.snapshot_id = initial_state.metadata.snapshot_id;
  replay.snapshot_manifest_path = snapshot_read.manifest_path;
  replay.snapshot_payload_path = snapshot_read.payload_path;
  replay.snapshot_payload_sha256 = snapshot_read.integrity.payload_sha256;
  replay.snapshot_manifest_sha256 = snapshot_read.integrity.manifest_sha256;
  replay.effective_options = effective;
  replay.problem = BuildFidelityProblemSnapshot(
      build.fingerprint_snapshot, snapshot_read.integrity, effective,
      oracle.problem.config_image_count, build.problem->NumResiduals(),
      build.problem->NumParameterBlocks(), build.problem->NumParameters(),
      build.constraints, losses, false);
  result->replay_record = replay;

  CompareProblem(oracle.problem, replay.problem, &result->comparison);
  result->comparison.provenance_pass =
      SameProvenance(oracle.provenance, replay.provenance);
  result->comparison.first_option_mismatch = FirstEffectiveOptionMismatch(
      oracle.effective_options, replay.effective_options);
  result->comparison.effective_options_pass =
      result->comparison.first_option_mismatch.empty() &&
      oracle.problem.effective_options_sha256 ==
      replay.problem.effective_options_sha256;
  if (!result->comparison.effective_options_pass) {
    if (result->comparison.first_option_mismatch.empty()) {
      result->comparison.first_option_mismatch =
          "effective_options_sha256";
    }
  }
  if (!result->comparison.provenance_pass ||
      !result->comparison.problem_fingerprint_pass ||
      !result->comparison.effective_options_pass) {
    result->pass = false;
    return true;
  }

  ceres::Solver::Options solver_options;
  if (!ApplyEffectiveCeresOptions(effective, &solver_options, error)) {
    return false;
  }
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, build.problem.get(), &summary);
  replay.summary = CaptureCeresSummary(summary);
  Snapshot post_state = replay_state;
  post_state.metadata.snapshot_id =
      initial_state.metadata.snapshot_id + "-replay-" +
      SanitizeToken(replay.run_id) + "-post";
  const std::string post_dir = JoinPaths(output_dir, "post_states");
  if (!WriteSnapshot(post_state, post_dir, &result->post_state_write, error)) {
    return false;
  }
  replay.post_state_manifest_path = result->post_state_write.manifest_path;
  replay.post_state_payload_path = result->post_state_write.payload_path;
  replay.post_state_payload_sha256 =
      result->post_state_write.integrity.payload_sha256;
  replay.final_state_sha256 = CanonicalStateSha256(post_state);
  if (!WriteCeresFidelityRecord(replay, output_dir,
                                &result->replay_record_write, error)) {
    return false;
  }
  result->replay_record = replay;
  if (!CompareCeresFidelityRecords(oracle, original_final_state, replay,
                                   post_state, &result->comparison, error)) {
    return false;
  }
  result->pass = result->comparison.pass;
  return true;
}

std::string CeresFidelityReplayJson(const CeresFidelityReplayResult& result,
                                    size_t indent_spaces) {
  const std::string indent(indent_spaces, ' ');
  const std::string i2(indent_spaces + 2, ' ');
  std::ostringstream stream;
  stream << std::setprecision(17) << indent << "{\n";
  stream << i2 << "\"pass\":" << BoolText(result.pass)
         << ",\"semantics\":\"" << result.semantics
         << "\",\"perturbation\":\"" << EscapeJson(result.perturbation)
         << "\",\"used_residual_order\":\""
         << result.used_residual_order << "\",\"used_parameter_order\":\""
         << result.used_parameter_order << "\",\n";
  stream << i2 << "\"replay_record_binary_path\":\""
         << EscapeJson(result.replay_record_write.binary_path)
         << "\",\"replay_record_binary_sha256\":\""
         << result.replay_record_write.binary_sha256 << "\",\n";
  stream << i2 << "\"oracle_record\":"
         << CeresFidelityRecordJson(result.oracle_record, indent_spaces + 2)
         << ",\n";
  stream << i2 << "\"replay_record\":"
         << CeresFidelityRecordJson(result.replay_record, indent_spaces + 2)
         << ",\n";
  stream << i2 << "\"comparison\":"
         << CeresFidelityComparisonJson(result.comparison, indent_spaces + 2)
         << '\n' << indent << '}';
  return stream.str();
}

}  // namespace gpu_ba
}  // namespace colmap
