#include "exe/online_i3dgs_replay.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
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
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <jsoncpp/json/json.h>
#ifdef __linux__
#include <linux/fs.h>
#endif
#include <openssl/sha.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/database.h"
#include "base/database_cache.h"
#include "base/image.h"
#include "base/pose.h"
#include "controllers/incremental_mapper.h"
#include "controllers/online_i3dgs_mapper.h"
#include "estimators/pose.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "lidar/incremental_causal_lidar_map.h"
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_triangulator.h"
#include "sfm/online_local_ba_executor.h"
#include "sfm/online_local_ba_postprocess.h"
#include "util/bitmap.h"

namespace colmap {
namespace {

constexpr char kManifestSchema[] =
    "online_i3dgs_phase1_input_manifest_v1";
constexpr char kConfigSchema[] =
    "online_i3dgs_phase1_resolved_config_v1";
constexpr char kEventSchema[] = "online_i3dgs_phase1_frame_event_v1";
constexpr char kPhase1SummarySchema[] =
    "online_i3dgs_phase1_run_summary_v1";
constexpr char kExpectedI3dgsCommit[] =
    "cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42";
constexpr size_t kMaximumJsonBytes = 256 * 1024 * 1024;

volatile sig_atomic_t g_stop_requested = 0;

class ReplayError : public std::runtime_error {
 public:
  explicit ReplayError(const std::string& message) : std::runtime_error(message) {}
};

struct FileSeal {
  std::string path;
  std::string real_path;
  std::string sha256;
  uint64_t size_bytes = 0;
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t mode = 0;
  int64_t mtime_ns = 0;
  int64_t ctime_ns = 0;
  bool path_is_symlink = false;
};

struct ReplayFrame {
  OnlineI3dgsReplayFrameInput input;
  std::string event_sha256;
};

struct ReplayPlan {
  std::string sealed_artifact;
  std::string session_real_path;
  std::string i3dgs_path;
  std::string mapper_binary_path;
  std::string manifest_text;
  std::string config_text;
  std::string phase1_frames_text;
  std::string manifest_sha256;
  std::string config_sha256;
  std::string phase1_frames_sha256;
  std::string event_chain_tail_sha256;
  OnlineI3dgsReplayCamera camera;
  size_t expected_frame_count = 0;
  std::vector<ReplayFrame> frames;
  std::map<std::string, FileSeal> sealed_files;
  std::set<std::string> session_relative_paths;
};

std::string ErrnoMessage(const std::string& operation,
                         const std::string& path) {
  return operation + " failed for " + path + ": " + std::strerror(errno);
}

bool IsLowerHex(const std::string& value, size_t size) {
  if (value.size() != size) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

std::string HexDigest(const unsigned char* digest, size_t size) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (size_t index = 0; index < size; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return stream.str();
}

std::string Sha256(const std::string& value) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  if (::SHA256(reinterpret_cast<const unsigned char*>(value.data()),
               value.size(), digest.data()) == nullptr) {
    throw ReplayError("OpenSSL SHA-256 failed");
  }
  return HexDigest(digest.data(), digest.size());
}

void WriteAll(int descriptor, const char* data, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t written = ::write(descriptor, data + offset, size - offset);
    if (written < 0) {
      if (errno == EINTR) continue;
      throw ReplayError("write failed: " + std::string(std::strerror(errno)));
    }
    offset += static_cast<size_t>(written);
  }
}

std::string ReadFile(const std::string& path) {
  struct stat file_stat;
  if (::lstat(path.c_str(), &file_stat) != 0) {
    throw ReplayError(ErrnoMessage("lstat", path));
  }
  if (S_ISLNK(file_stat.st_mode) || !S_ISREG(file_stat.st_mode)) {
    throw ReplayError("JSON input must be a regular non-symlink file: " + path);
  }
  if (file_stat.st_size < 0 ||
      static_cast<uint64_t>(file_stat.st_size) > kMaximumJsonBytes) {
    throw ReplayError("JSON input is too large: " + path);
  }
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) throw ReplayError(ErrnoMessage("open", path));
  std::string contents(static_cast<size_t>(file_stat.st_size), '\0');
  size_t offset = 0;
  try {
    while (offset < contents.size()) {
      const ssize_t count =
          ::read(descriptor, &contents[offset], contents.size() - offset);
      if (count < 0) {
        if (errno == EINTR) continue;
        throw ReplayError(ErrnoMessage("read", path));
      }
      if (count == 0) break;
      offset += static_cast<size_t>(count);
    }
    struct stat after;
    if (::fstat(descriptor, &after) != 0) {
      throw ReplayError(ErrnoMessage("fstat", path));
    }
    if (after.st_dev != file_stat.st_dev || after.st_ino != file_stat.st_ino ||
        after.st_size != file_stat.st_size ||
        after.st_mtim.tv_sec != file_stat.st_mtim.tv_sec ||
        after.st_mtim.tv_nsec != file_stat.st_mtim.tv_nsec ||
        offset != contents.size()) {
      throw ReplayError("File changed while reading: " + path);
    }
  } catch (...) {
    ::close(descriptor);
    throw;
  }
  ::close(descriptor);
  return contents;
}

Json::Value ParseJson(const std::string& text, const std::string& context) {
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  builder["allowComments"] = false;
  builder["strictRoot"] = true;
  builder["allowDroppedNullPlaceholders"] = false;
  builder["allowNumericKeys"] = false;
  builder["allowSingleQuotes"] = false;
  builder["failIfExtra"] = true;
  builder["rejectDupKeys"] = true;
  builder["allowSpecialFloats"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value root;
  std::string errors;
  if (!reader->parse(text.data(), text.data() + text.size(), &root, &errors)) {
    throw ReplayError("Invalid JSON in " + context + ": " + errors);
  }
  return root;
}

std::string CompactJson(const Json::Value& value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  builder["commentStyle"] = "None";
  builder["enableYAMLCompatibility"] = false;
  builder["dropNullPlaceholders"] = false;
  builder["useSpecialFloats"] = false;
  std::string output = Json::writeString(builder, value);
  if (!output.empty() && output.back() == '\n') output.pop_back();
  return output;
}

const Json::Value& Member(const Json::Value& object,
                          const char* name,
                          const std::string& context) {
  if (!object.isObject() || !object.isMember(name)) {
    throw ReplayError(context + " is missing member " + name);
  }
  return object[name];
}

void RequireObject(const Json::Value& value, const std::string& context) {
  if (!value.isObject()) throw ReplayError(context + " must be an object");
}

void RequireArray(const Json::Value& value, const std::string& context) {
  if (!value.isArray()) throw ReplayError(context + " must be an array");
}

void RequireOnlyMembers(const Json::Value& value,
                        std::initializer_list<const char*> expected,
                        const std::string& context) {
  RequireObject(value, context);
  std::set<std::string> expected_names;
  for (const char* name : expected) expected_names.insert(name);
  const Json::Value::Members actual = value.getMemberNames();
  const std::set<std::string> actual_names(actual.begin(), actual.end());
  if (actual_names != expected_names) {
    throw ReplayError(context + " has an unexpected member set");
  }
}

std::string RequireString(const Json::Value& value,
                          const std::string& context) {
  if (!value.isString()) throw ReplayError(context + " must be a string");
  return value.asString();
}

bool RequireBool(const Json::Value& value, const std::string& context) {
  if (!value.isBool()) throw ReplayError(context + " must be a boolean");
  return value.asBool();
}

uint64_t RequireUInt(const Json::Value& value, const std::string& context) {
  if (!value.isIntegral() || value.asLargestInt() < 0) {
    throw ReplayError(context + " must be a non-negative integer");
  }
  return value.asLargestUInt();
}

double RequireFiniteNumber(const Json::Value& value,
                           const std::string& context) {
  if (!value.isNumeric()) throw ReplayError(context + " must be numeric");
  const double number = value.asDouble();
  if (!std::isfinite(number)) throw ReplayError(context + " must be finite");
  return number;
}

void RequireStringValue(const Json::Value& value,
                        const std::string& expected,
                        const std::string& context) {
  if (RequireString(value, context) != expected) {
    throw ReplayError(context + " has an unexpected value");
  }
}

void RequireBoolValue(const Json::Value& value,
                      bool expected,
                      const std::string& context) {
  if (RequireBool(value, context) != expected) {
    throw ReplayError(context + " has an unexpected value");
  }
}

void RequireUIntValue(const Json::Value& value,
                      uint64_t expected,
                      const std::string& context) {
  if (RequireUInt(value, context) != expected) {
    throw ReplayError(context + " has an unexpected value");
  }
}

std::string RealPath(const std::string& path) {
  std::unique_ptr<char, decltype(&std::free)> resolved(
      ::realpath(path.c_str(), nullptr), &std::free);
  if (!resolved) throw ReplayError(ErrnoMessage("realpath", path));
  return resolved.get();
}

int64_t StatNanoseconds(const timespec& value) {
  if (value.tv_sec > std::numeric_limits<int64_t>::max() / 1000000000LL) {
    throw ReplayError("File timestamp is out of range");
  }
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

FileSeal ParseFileSeal(const Json::Value& value,
                       const std::string& context) {
  RequireOnlyMembers(value,
                     {"path", "real_path", "size_bytes", "mtime_ns",
                      "mtime_utc", "ctime_ns", "ctime_utc", "sha256",
                      "device", "inode", "mode", "mode_octal",
                      "path_is_symlink"},
                     context);
  FileSeal seal;
  seal.path = RequireString(Member(value, "path", context), context + ".path");
  seal.real_path = RequireString(Member(value, "real_path", context),
                                 context + ".real_path");
  seal.size_bytes =
      RequireUInt(Member(value, "size_bytes", context), context + ".size_bytes");
  seal.mtime_ns = static_cast<int64_t>(RequireUInt(
      Member(value, "mtime_ns", context), context + ".mtime_ns"));
  seal.ctime_ns = static_cast<int64_t>(RequireUInt(
      Member(value, "ctime_ns", context), context + ".ctime_ns"));
  RequireString(Member(value, "mtime_utc", context), context + ".mtime_utc");
  RequireString(Member(value, "ctime_utc", context), context + ".ctime_utc");
  seal.sha256 =
      RequireString(Member(value, "sha256", context), context + ".sha256");
  if (!IsLowerHex(seal.sha256, 64)) {
    throw ReplayError(context + ".sha256 must be lowercase SHA-256");
  }
  seal.device =
      RequireUInt(Member(value, "device", context), context + ".device");
  seal.inode = RequireUInt(Member(value, "inode", context), context + ".inode");
  RequireString(Member(value, "mode", context), context + ".mode");
  const std::string mode_octal =
      RequireString(Member(value, "mode_octal", context),
                    context + ".mode_octal");
  if (mode_octal.size() < 3 || mode_octal.compare(0, 2, "0o") != 0) {
    throw ReplayError(context + ".mode_octal is invalid");
  }
  try {
    seal.mode = std::stoull(mode_octal.substr(2), nullptr, 8);
  } catch (const std::exception&) {
    throw ReplayError(context + ".mode_octal is invalid");
  }
  seal.path_is_symlink = RequireBool(
      Member(value, "path_is_symlink", context), context + ".path_is_symlink");
  if (seal.path.empty() || seal.path.front() != '/' || seal.real_path.empty() ||
      seal.real_path.front() != '/') {
    throw ReplayError(context + " paths must be absolute");
  }
  return seal;
}

void RegisterSeal(const FileSeal& seal,
                  const std::string& context,
                  ReplayPlan* plan) {
  const auto inserted = plan->sealed_files.emplace(seal.path, seal);
  if (!inserted.second) {
    const FileSeal& previous = inserted.first->second;
    if (previous.real_path != seal.real_path ||
        previous.sha256 != seal.sha256 ||
        previous.size_bytes != seal.size_bytes ||
        previous.device != seal.device || previous.inode != seal.inode ||
        previous.mode != seal.mode || previous.mtime_ns != seal.mtime_ns ||
        previous.ctime_ns != seal.ctime_ns ||
        previous.path_is_symlink != seal.path_is_symlink) {
      throw ReplayError("Conflicting sealed file records for " + seal.path +
                        " at " + context);
    }
  }
}

std::string HashFileAndVerifyMetadata(const FileSeal& seal) {
  struct stat entry;
  if (::lstat(seal.path.c_str(), &entry) != 0) {
    throw ReplayError(ErrnoMessage("lstat", seal.path));
  }
  const bool is_symlink = S_ISLNK(entry.st_mode);
  if (is_symlink != seal.path_is_symlink) {
    throw ReplayError("Sealed path symlink state changed: " + seal.path);
  }
  const std::string real_path = RealPath(seal.path);
  if (real_path != seal.real_path) {
    throw ReplayError("Sealed path target changed: " + seal.path);
  }
  int flags = O_RDONLY | O_CLOEXEC;
  if (!seal.path_is_symlink) flags |= O_NOFOLLOW;
  const int descriptor = ::open(seal.path.c_str(), flags);
  if (descriptor < 0) throw ReplayError(ErrnoMessage("open", seal.path));

  SHA256_CTX context;
  if (SHA256_Init(&context) != 1) {
    ::close(descriptor);
    throw ReplayError("SHA256_Init failed for " + seal.path);
  }
  std::array<unsigned char, 1024 * 1024> buffer;
  struct stat before;
  if (::fstat(descriptor, &before) != 0) {
    ::close(descriptor);
    throw ReplayError(ErrnoMessage("fstat", seal.path));
  }
  try {
    while (true) {
      const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
      if (count < 0) {
        if (errno == EINTR) continue;
        throw ReplayError(ErrnoMessage("read", seal.path));
      }
      if (count == 0) break;
      if (SHA256_Update(&context, buffer.data(), static_cast<size_t>(count)) !=
          1) {
        throw ReplayError("SHA256_Update failed for " + seal.path);
      }
    }
    struct stat after;
    if (::fstat(descriptor, &after) != 0) {
      throw ReplayError(ErrnoMessage("fstat", seal.path));
    }
    if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_mode != after.st_mode || before.st_size != after.st_size ||
        StatNanoseconds(before.st_mtim) != StatNanoseconds(after.st_mtim) ||
        StatNanoseconds(before.st_ctim) != StatNanoseconds(after.st_ctim)) {
      throw ReplayError("Sealed file changed while hashing: " + seal.path);
    }
  } catch (...) {
    ::close(descriptor);
    throw;
  }
  ::close(descriptor);

  struct stat current_entry;
  if (::lstat(seal.path.c_str(), &current_entry) != 0 ||
      RealPath(seal.path) != seal.real_path ||
      (S_ISLNK(current_entry.st_mode) != seal.path_is_symlink)) {
    throw ReplayError("Sealed path changed while hashing: " + seal.path);
  }
  if (!seal.path_is_symlink &&
      (current_entry.st_dev != before.st_dev ||
       current_entry.st_ino != before.st_ino ||
       current_entry.st_mode != before.st_mode ||
       current_entry.st_size != before.st_size ||
       StatNanoseconds(current_entry.st_mtim) !=
           StatNanoseconds(before.st_mtim) ||
       StatNanoseconds(current_entry.st_ctim) !=
           StatNanoseconds(before.st_ctim))) {
    throw ReplayError("Sealed path entry changed while hashing: " + seal.path);
  }

  if (!S_ISREG(before.st_mode) || static_cast<uint64_t>(before.st_dev) != seal.device ||
      static_cast<uint64_t>(before.st_ino) != seal.inode ||
      static_cast<uint64_t>(before.st_size) != seal.size_bytes ||
      static_cast<uint64_t>(before.st_mode & 07777) != seal.mode ||
      StatNanoseconds(before.st_mtim) != seal.mtime_ns ||
      StatNanoseconds(before.st_ctim) != seal.ctime_ns) {
    throw ReplayError("Sealed file metadata changed: " + seal.path);
  }

  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  if (SHA256_Final(digest.data(), &context) != 1) {
    throw ReplayError("SHA256_Final failed for " + seal.path);
  }
  const std::string actual = HexDigest(digest.data(), digest.size());
  if (actual != seal.sha256) {
    throw ReplayError("Sealed file SHA-256 changed: " + seal.path);
  }
  return actual;
}

std::string RelativeTo(const std::string& path, const std::string& root) {
  if (path.size() <= root.size() || path.compare(0, root.size(), root) != 0 ||
      path[root.size()] != '/') {
    throw ReplayError("Session file escapes sealed session: " + path);
  }
  return path.substr(root.size() + 1);
}

bool PathIsWithin(const std::string& path, const std::string& root) {
  return path == root ||
         (path.size() > root.size() && path.compare(0, root.size(), root) == 0 &&
          path[root.size()] == '/');
}

std::string ProspectiveOutputPath(const std::string& requested) {
  const boost::filesystem::path absolute =
      boost::filesystem::absolute(requested);
  boost::system::error_code error;
  const boost::filesystem::path parent =
      boost::filesystem::canonical(absolute.parent_path(), error);
  if (error || !boost::filesystem::is_directory(parent)) {
    throw ReplayError("Output parent must be an existing directory");
  }
  return (parent / absolute.filename()).string();
}

void CollectSessionInventory(const std::string& root,
                             const boost::filesystem::path& directory,
                             std::set<std::string>* output) {
  boost::system::error_code error;
  boost::filesystem::directory_iterator iterator(directory, error);
  if (error) throw ReplayError("Cannot enumerate sealed session: " + error.message());
  const boost::filesystem::directory_iterator end;
  std::vector<boost::filesystem::path> entries;
  for (; iterator != end; iterator.increment(error)) {
    if (error) throw ReplayError("Cannot enumerate sealed session: " + error.message());
    entries.push_back(iterator->path());
  }
  std::sort(entries.begin(), entries.end());
  for (const auto& entry : entries) {
    const boost::filesystem::file_status status =
        boost::filesystem::symlink_status(entry, error);
    if (error) throw ReplayError("Cannot inspect sealed session entry: " + entry.string());
    if (boost::filesystem::is_symlink(status)) {
      throw ReplayError("Sealed session contains a symlink: " + entry.string());
    }
    if (boost::filesystem::is_directory(status)) {
      CollectSessionInventory(root, entry, output);
    } else if (boost::filesystem::is_regular_file(status)) {
      output->insert(RelativeTo(entry.string(), root));
    } else {
      throw ReplayError("Sealed session entry is not a regular file: " +
                        entry.string());
    }
  }
}

std::string VerifyAllSealedInputs(const ReplayPlan& plan) {
  SHA256_CTX aggregate;
  if (SHA256_Init(&aggregate) != 1) {
    throw ReplayError("SHA256_Init failed for sealed input aggregate");
  }
  for (const auto& item : plan.sealed_files) {
    const std::string digest = HashFileAndVerifyMetadata(item.second);
    std::string row = item.first;
    row.push_back('\0');
    row += digest;
    row.push_back('\n');
    if (SHA256_Update(&aggregate, row.data(), row.size()) != 1) {
      throw ReplayError("SHA256_Update failed for sealed input aggregate");
    }
  }
  std::set<std::string> actual_inventory;
  CollectSessionInventory(plan.session_real_path, plan.session_real_path,
                          &actual_inventory);
  if (actual_inventory != plan.session_relative_paths) {
    throw ReplayError("Sealed session inventory changed");
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  if (SHA256_Final(digest.data(), &aggregate) != 1) {
    throw ReplayError("SHA256_Final failed for sealed input aggregate");
  }
  return HexDigest(digest.data(), digest.size());
}

void ValidateNumericArray(const Json::Value& value,
                          size_t expected_size,
                          const std::string& context) {
  RequireArray(value, context);
  if (value.size() != expected_size) {
    throw ReplayError(context + " has an unexpected length");
  }
  for (Json::ArrayIndex index = 0; index < value.size(); ++index) {
    RequireFiniteNumber(value[index], context + "[" + std::to_string(index) + "]");
  }
}

Eigen::Matrix3d ValidatePose(const Json::Value& pose,
                             const std::string& context,
                             Eigen::Vector3d* translation) {
  RequireOnlyMembers(pose, {"timestamp", "source_values", "fastlio", "colmap_prior"},
                     context);
  RequireString(Member(pose, "timestamp", context), context + ".timestamp");
  ValidateNumericArray(Member(pose, "source_values", context), 8,
                       context + ".source_values");

  const Json::Value& fastlio = Member(pose, "fastlio", context);
  RequireOnlyMembers(fastlio,
                     {"convention", "translation_m", "source_quaternion_wxyz",
                      "normalized_quaternion_wxyz", "source_quaternion_norm",
                      "T_wc"},
                     context + ".fastlio");
  RequireStringValue(Member(fastlio, "convention", context), "T_wc",
                     context + ".fastlio.convention");
  ValidateNumericArray(Member(fastlio, "translation_m", context), 3,
                       context + ".fastlio.translation_m");
  ValidateNumericArray(Member(fastlio, "source_quaternion_wxyz", context), 4,
                       context + ".fastlio.source_quaternion_wxyz");
  ValidateNumericArray(Member(fastlio, "normalized_quaternion_wxyz", context), 4,
                       context + ".fastlio.normalized_quaternion_wxyz");
  RequireFiniteNumber(Member(fastlio, "source_quaternion_norm", context),
                      context + ".fastlio.source_quaternion_norm");
  const Json::Value& fastlio_matrix = Member(fastlio, "T_wc", context);
  RequireArray(fastlio_matrix, context + ".fastlio.T_wc");
  if (fastlio_matrix.size() != 4) throw ReplayError(context + ".fastlio.T_wc must be 4x4");
  for (Json::ArrayIndex row = 0; row < 4; ++row) {
    ValidateNumericArray(fastlio_matrix[row], 4,
                         context + ".fastlio.T_wc[" + std::to_string(row) + "]");
  }

  const Json::Value& prior = Member(pose, "colmap_prior", context);
  RequireOnlyMembers(prior,
                     {"convention", "camera_center_world_m", "rotation_cw",
                      "translation_cw_m", "T_cw", "pose_prior_ply_values"},
                     context + ".colmap_prior");
  RequireStringValue(Member(prior, "convention", context), "T_cw",
                     context + ".colmap_prior.convention");
  ValidateNumericArray(Member(prior, "camera_center_world_m", context), 3,
                       context + ".colmap_prior.camera_center_world_m");
  ValidateNumericArray(Member(prior, "translation_cw_m", context), 3,
                       context + ".colmap_prior.translation_cw_m");
  ValidateNumericArray(Member(prior, "pose_prior_ply_values", context), 6,
                       context + ".colmap_prior.pose_prior_ply_values");

  const Json::Value& rotation = Member(prior, "rotation_cw", context);
  RequireArray(rotation, context + ".colmap_prior.rotation_cw");
  if (rotation.size() != 3) {
    throw ReplayError(context + ".colmap_prior.rotation_cw must be 3x3");
  }
  Eigen::Matrix3d matrix;
  for (Json::ArrayIndex row = 0; row < 3; ++row) {
    ValidateNumericArray(rotation[row], 3,
                         context + ".colmap_prior.rotation_cw[" +
                             std::to_string(row) + "]");
    for (Json::ArrayIndex column = 0; column < 3; ++column) {
      matrix(row, column) = rotation[row][column].asDouble();
    }
  }
  const Json::Value& translation_json =
      Member(prior, "translation_cw_m", context);
  for (Json::ArrayIndex index = 0; index < 3; ++index) {
    (*translation)(index) = translation_json[index].asDouble();
  }

  const Json::Value& transform = Member(prior, "T_cw", context);
  RequireArray(transform, context + ".colmap_prior.T_cw");
  if (transform.size() != 4) {
    throw ReplayError(context + ".colmap_prior.T_cw must be 4x4");
  }
  for (Json::ArrayIndex row = 0; row < 4; ++row) {
    ValidateNumericArray(transform[row], 4,
                         context + ".colmap_prior.T_cw[" +
                             std::to_string(row) + "]");
  }
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      if (std::abs(transform[row][column].asDouble() - matrix(row, column)) >
          1e-12) {
        throw ReplayError(context + ".colmap_prior matrices disagree");
      }
    }
    if (std::abs(transform[row][3].asDouble() - (*translation)(row)) > 1e-12) {
      throw ReplayError(context + ".colmap_prior translations disagree");
    }
  }
  if ((matrix.transpose() * matrix - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
          1e-8 ||
      std::abs(matrix.determinant() - 1.0) > 1e-8 ||
      std::abs(transform[3][0].asDouble()) > 1e-12 ||
      std::abs(transform[3][1].asDouble()) > 1e-12 ||
      std::abs(transform[3][2].asDouble()) > 1e-12 ||
      std::abs(transform[3][3].asDouble() - 1.0) > 1e-12) {
    throw ReplayError(context + ".colmap_prior.T_cw is not a valid SE(3)");
  }
  return matrix;
}

OnlineI3dgsReplayCamera ParseCamera(const Json::Value& value,
                                    bool source_file_required,
                                    ReplayPlan* plan,
                                    const std::string& context) {
  if (source_file_required) {
    RequireOnlyMembers(value,
                       {"model", "width", "height", "params", "param_order",
                        "source_distortion_model", "source_distortion",
                        "source_file"},
                       context);
  } else {
    RequireOnlyMembers(value,
                       {"model", "width", "height", "params", "param_order",
                        "source_distortion_model", "source_distortion"},
                       context);
  }
  OnlineI3dgsReplayCamera camera;
  camera.model = RequireString(Member(value, "model", context), context + ".model");
  if (camera.model != "OPENCV") throw ReplayError(context + ".model must be OPENCV");
  camera.width = RequireUInt(Member(value, "width", context), context + ".width");
  camera.height = RequireUInt(Member(value, "height", context), context + ".height");
  if (camera.width == 0 || camera.height == 0) {
    throw ReplayError(context + " dimensions must be positive");
  }
  const Json::Value& params = Member(value, "params", context);
  ValidateNumericArray(params, 8, context + ".params");
  for (const auto& item : params) camera.params.push_back(item.asDouble());
  const std::array<std::string, 8> expected_order =
      {{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"}};
  const Json::Value& order = Member(value, "param_order", context);
  RequireArray(order, context + ".param_order");
  if (order.size() != expected_order.size()) {
    throw ReplayError(context + ".param_order has an unexpected length");
  }
  for (Json::ArrayIndex index = 0; index < order.size(); ++index) {
    RequireStringValue(order[index], expected_order[index],
                       context + ".param_order");
  }
  const Json::Value& distortion_model =
      Member(value, "source_distortion_model", context);
  if (!distortion_model.isNull() && !distortion_model.isString()) {
    throw ReplayError(context + ".source_distortion_model has an invalid type");
  }
  const Json::Value& distortion = Member(value, "source_distortion", context);
  RequireArray(distortion, context + ".source_distortion");
  if (distortion.size() < 4) {
    throw ReplayError(context + ".source_distortion is incomplete");
  }
  for (Json::ArrayIndex index = 0; index < distortion.size(); ++index) {
    RequireFiniteNumber(distortion[index], context + ".source_distortion");
  }
  if (source_file_required) {
    const FileSeal seal = ParseFileSeal(Member(value, "source_file", context),
                                        context + ".source_file");
    RegisterSeal(seal, context + ".source_file", plan);
  }
  return camera;
}

void ValidateCamRecord(const Json::Value& value, const std::string& context) {
  RequireOnlyMembers(value, {"format", "world_to_camera_3x4", "calibration_values"},
                     context);
  RequireStringValue(Member(value, "format", context), "MVE_CAM",
                     context + ".format");
  const Json::Value& transform = Member(value, "world_to_camera_3x4", context);
  RequireArray(transform, context + ".world_to_camera_3x4");
  if (transform.size() != 3) throw ReplayError(context + " transform must be 3x4");
  for (Json::ArrayIndex row = 0; row < transform.size(); ++row) {
    ValidateNumericArray(transform[row], 4, context + ".world_to_camera_3x4");
  }
  ValidateNumericArray(Member(value, "calibration_values", context), 6,
                       context + ".calibration_values");
}

std::string BaseName(const std::string& path) {
  return boost::filesystem::path(path).filename().string();
}

void ParseManifestFrames(const Json::Value& frames,
                         ReplayPlan* plan,
                         std::vector<Json::Value>* frame_poses) {
  RequireArray(frames, "input-manifest.frames");
  if (frames.empty()) throw ReplayError("input-manifest.frames must not be empty");
  for (Json::ArrayIndex offset = 0; offset < frames.size(); ++offset) {
    const Json::Value& frame = frames[offset];
    const std::string context =
        "input-manifest.frames[" + std::to_string(offset) + "]";
    RequireOnlyMembers(frame, {"frame_index", "image_name", "image", "files", "cam", "pose"},
                       context);
    const size_t frame_index =
        RequireUInt(Member(frame, "frame_index", context), context + ".frame_index");
    if (frame_index != static_cast<size_t>(offset) + 1) {
      throw ReplayError("Manifest frame indices must be contiguous from 1");
    }
    const std::string image_name =
        RequireString(Member(frame, "image_name", context), context + ".image_name");
    if (image_name != "imgs_" + std::to_string(frame_index) + ".jpg") {
      throw ReplayError(context + ".image_name does not match frame index");
    }
    const Json::Value& image = Member(frame, "image", context);
    RequireOnlyMembers(image, {"width", "height"}, context + ".image");
    RequireUIntValue(Member(image, "width", context), plan->camera.width,
                     context + ".image.width");
    RequireUIntValue(Member(image, "height", context), plan->camera.height,
                     context + ".image.height");
    ValidateCamRecord(Member(frame, "cam", context), context + ".cam");

    Eigen::Vector3d translation;
    const Json::Value& pose = Member(frame, "pose", context);
    const Eigen::Matrix3d rotation = ValidatePose(pose, context + ".pose", &translation);
    frame_poses->push_back(pose);

    ReplayFrame replay_frame;
    replay_frame.input.frame_index = frame_index;
    replay_frame.input.image_id = static_cast<uint32_t>(frame_index);
    replay_frame.input.image_name = image_name;
    const Eigen::Vector4d quaternion = RotationMatrixToQuaternion(rotation);
    for (size_t index = 0; index < 4; ++index) replay_frame.input.qvec[index] = quaternion(index);
    for (size_t index = 0; index < 3; ++index) replay_frame.input.tvec[index] = translation(index);

    const Json::Value& files = Member(frame, "files", context);
    RequireOnlyMembers(files, {"jpg", "cam", "odom", "scan"}, context + ".files");
    const std::array<std::string, 4> kinds = {{"jpg", "cam", "odom", "scan"}};
    std::map<std::string, FileSeal> parsed;
    for (const std::string& kind : kinds) {
      const FileSeal seal = ParseFileSeal(files[kind], context + ".files." + kind);
      if (seal.path_is_symlink) {
        throw ReplayError("Frame inputs must not be symlinks: " + seal.path);
      }
      RegisterSeal(seal, context + ".files." + kind, plan);
      plan->session_relative_paths.insert(RelativeTo(seal.real_path,
                                                     plan->session_real_path));
      parsed.emplace(kind, seal);
    }
    const std::map<std::string, std::string> expected_names = {
        {"jpg", "imgs_" + std::to_string(frame_index) + ".jpg"},
        {"cam", "imgs_" + std::to_string(frame_index) + ".CAM"},
        {"odom", "odoms_" + std::to_string(frame_index) + ".txt"},
        {"scan", "scans_" + std::to_string(frame_index) + ".pcd"}};
    for (const auto& expected : expected_names) {
      if (BaseName(parsed.at(expected.first).path) != expected.second) {
        throw ReplayError(context + ".files." + expected.first +
                          " basename does not match frame index");
      }
    }
    replay_frame.input.image_path = parsed["jpg"].path;
    replay_frame.input.camera_path = parsed["cam"].path;
    replay_frame.input.odometry_path = parsed["odom"].path;
    replay_frame.input.scan_path = parsed["scan"].path;
    replay_frame.input.image_sha256 = parsed["jpg"].sha256;
    replay_frame.input.camera_sha256 = parsed["cam"].sha256;
    replay_frame.input.odometry_sha256 = parsed["odom"].sha256;
    replay_frame.input.scan_sha256 = parsed["scan"].sha256;
    replay_frame.input.scan_size_bytes = parsed["scan"].size_bytes;
    plan->frames.push_back(std::move(replay_frame));
  }
}

void ValidateRepository(const Json::Value& value,
                        ReplayPlan* plan,
                        const std::string& context) {
  RequireObject(value, context);
  const std::string commit =
      RequireString(Member(value, "commit", context), context + ".commit");
  const std::string tree =
      RequireString(Member(value, "tree", context), context + ".tree");
  if (!IsLowerHex(commit, 40) || !IsLowerHex(tree, 40)) {
    throw ReplayError(context + " commit/tree is invalid");
  }
  RequireBool(Member(value, "dirty", context), context + ".dirty");
  RequireArray(Member(value, "status", context), context + ".status");
  const std::string status_hash =
      RequireString(Member(value, "status_sha256", context),
                    context + ".status_sha256");
  if (!IsLowerHex(status_hash, 64)) throw ReplayError(context + " status hash is invalid");
  const Json::Value& files = Member(value, "key_source_files", context);
  RequireArray(files, context + ".key_source_files");
  for (Json::ArrayIndex index = 0; index < files.size(); ++index) {
    const FileSeal seal = ParseFileSeal(
        files[index], context + ".key_source_files[" + std::to_string(index) + "]");
    RegisterSeal(seal, context + ".key_source_files", plan);
  }
  RequireString(Member(value, "key_source_snapshot_set_sha256", context),
                context + ".key_source_snapshot_set_sha256");
  RequireArray(Member(value, "untracked_key_source_files", context),
               context + ".untracked_key_source_files");
}

std::vector<std::string> RunProcess(const std::vector<std::string>& arguments) {
  if (arguments.empty()) throw ReplayError("Cannot run an empty command");
  int pipe_descriptors[2];
  if (::pipe2(pipe_descriptors, O_CLOEXEC) != 0) {
    throw ReplayError("pipe2 failed: " + std::string(std::strerror(errno)));
  }
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(pipe_descriptors[0]);
    ::close(pipe_descriptors[1]);
    throw ReplayError("fork failed: " + std::string(std::strerror(errno)));
  }
  if (child == 0) {
    ::dup2(pipe_descriptors[1], STDOUT_FILENO);
    ::dup2(pipe_descriptors[1], STDERR_FILENO);
    ::close(pipe_descriptors[0]);
    ::close(pipe_descriptors[1]);
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    ::execvp(argv[0], argv.data());
    _exit(127);
  }
  ::close(pipe_descriptors[1]);
  std::string output;
  std::array<char, 4096> buffer;
  while (true) {
    const ssize_t count = ::read(pipe_descriptors[0], buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR) continue;
      ::close(pipe_descriptors[0]);
      throw ReplayError("read from child failed: " +
                        std::string(std::strerror(errno)));
    }
    if (count == 0) break;
    output.append(buffer.data(), static_cast<size_t>(count));
  }
  ::close(pipe_descriptors[0]);
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) throw ReplayError("waitpid failed");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw ReplayError("Command failed: " + arguments.front() + ": " + output);
  }
  std::vector<std::string> lines;
  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) lines.push_back(line);
  return lines;
}

void ValidateRuntimeEnvironment(const ReplayPlan& plan,
                                const OnlineI3dgsReplayOptions& options) {
#ifndef GPU_BA_CUDA_ENABLED
  throw ReplayError("online_i3dgs_mapper requires GPU_BA_CUDA_ENABLED");
#endif
  std::string executable = options.executable_path;
  if (executable.empty()) {
    std::array<char, 4096> buffer;
    const ssize_t size = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (size < 0) throw ReplayError("Cannot resolve /proc/self/exe");
    buffer[static_cast<size_t>(size)] = '\0';
    executable.assign(buffer.data());
  }
  const std::string executable_real_path = RealPath(executable);
  const FileSeal& mapper = plan.sealed_files.at(plan.mapper_binary_path);
  if (executable_real_path != mapper.real_path) {
    throw ReplayError("Executing mapper binary differs from sealed mapper binary");
  }
  HashFileAndVerifyMetadata(mapper);
  const std::vector<std::string> commit =
      RunProcess({"git", "-C", plan.i3dgs_path, "rev-parse", "HEAD"});
  if (commit.size() != 1 || commit.front() != kExpectedI3dgsCommit) {
    throw ReplayError("i3dgs checkout commit changed after Phase 1 sealing");
  }
  const std::vector<std::string> status = RunProcess(
      {"git", "-C", plan.i3dgs_path, "status", "--porcelain=v1",
       "--untracked-files=normal"});
  if (!status.empty()) throw ReplayError("i3dgs checkout is dirty");
}

void ValidateI3dgs(const Json::Value& value,
                    ReplayPlan* plan,
                    const std::string& context) {
  RequireObject(value, context);
  plan->i3dgs_path =
      RequireString(Member(value, "path", context), context + ".path");
  const std::string commit =
      RequireString(Member(value, "commit", context), context + ".commit");
  const std::string expected = RequireString(
      Member(value, "expected_commit", context), context + ".expected_commit");
  if (commit != kExpectedI3dgsCommit || expected != kExpectedI3dgsCommit) {
    throw ReplayError("Unexpected sealed i3dgs commit");
  }
  RequireBoolValue(Member(value, "dirty", context), false, context + ".dirty");
  RequireArray(Member(value, "status", context), context + ".status");
  if (!Member(value, "status", context).empty()) {
    throw ReplayError(context + ".status must be empty");
  }
  const std::string status_sha = RequireString(
      Member(value, "status_sha256", context), context + ".status_sha256");
  if (status_sha != Sha256("")) {
    throw ReplayError(context + ".status_sha256 does not represent a clean checkout");
  }
}

void ParseManifest(const Json::Value& manifest,
                   ReplayPlan* plan,
                   std::vector<Json::Value>* frame_poses) {
  RequireOnlyMembers(
      manifest,
      {"schema", "phase", "run_id", "generated_at", "artifact_dir",
       "hash_algorithm", "session", "frame_count", "frame_range",
       "frame_order", "frames", "camera", "session_assets",
       "required_session_assets", "binaries", "repository", "i3dgs",
       "pose_conversion", "stability_verification"},
      "input-manifest");
  RequireStringValue(Member(manifest, "schema", "input-manifest"),
                     kManifestSchema, "input-manifest.schema");
  RequireUIntValue(Member(manifest, "phase", "input-manifest"), 1,
                   "input-manifest.phase");
  RequireString(Member(manifest, "run_id", "input-manifest"),
                "input-manifest.run_id");
  RequireString(Member(manifest, "generated_at", "input-manifest"),
                "input-manifest.generated_at");
  RequireStringValue(Member(manifest, "hash_algorithm", "input-manifest"),
                     "SHA-256", "input-manifest.hash_algorithm");
  const std::string artifact_dir = RequireString(
      Member(manifest, "artifact_dir", "input-manifest"),
      "input-manifest.artifact_dir");
  if (RealPath(artifact_dir) != plan->sealed_artifact) {
    throw ReplayError("input-manifest.artifact_dir differs from --sealed_artifact");
  }

  const Json::Value& session = Member(manifest, "session", "input-manifest");
  RequireOnlyMembers(session,
                     {"path", "real_path", "read_only_policy",
                      "inventory_file_count", "inventory_relative_paths_sha256"},
                     "input-manifest.session");
  RequireString(Member(session, "path", "input-manifest.session"),
                "input-manifest.session.path");
  plan->session_real_path = RequireString(
      Member(session, "real_path", "input-manifest.session"),
      "input-manifest.session.real_path");
  RequireBoolValue(Member(session, "read_only_policy", "input-manifest.session"),
                   true, "input-manifest.session.read_only_policy");
  RequireUInt(Member(session, "inventory_file_count", "input-manifest.session"),
              "input-manifest.session.inventory_file_count");
  const std::string inventory_sha = RequireString(
      Member(session, "inventory_relative_paths_sha256", "input-manifest.session"),
      "input-manifest.session.inventory_relative_paths_sha256");
  if (!IsLowerHex(inventory_sha, 64)) {
    throw ReplayError("input-manifest session inventory hash is invalid");
  }

  plan->expected_frame_count = RequireUInt(
      Member(manifest, "frame_count", "input-manifest"),
      "input-manifest.frame_count");
  RequireStringValue(Member(manifest, "frame_order", "input-manifest"),
                     "NUMERIC_SUFFIX_ASCENDING", "input-manifest.frame_order");
  const Json::Value& range = Member(manifest, "frame_range", "input-manifest");
  RequireArray(range, "input-manifest.frame_range");
  if (range.size() != 2 || RequireUInt(range[0], "input-manifest.frame_range[0]") != 1 ||
      RequireUInt(range[1], "input-manifest.frame_range[1]") !=
          plan->expected_frame_count) {
    throw ReplayError("input-manifest.frame_range is invalid");
  }

  plan->camera = ParseCamera(Member(manifest, "camera", "input-manifest"), true,
                             plan, "input-manifest.camera");
  ParseManifestFrames(Member(manifest, "frames", "input-manifest"), plan,
                      frame_poses);
  if (plan->frames.size() != plan->expected_frame_count) {
    throw ReplayError("input-manifest frame_count does not match frames array");
  }

  const Json::Value& assets = Member(manifest, "session_assets", "input-manifest");
  RequireArray(assets, "input-manifest.session_assets");
  std::set<std::string> asset_paths;
  for (Json::ArrayIndex index = 0; index < assets.size(); ++index) {
    const Json::Value& asset = assets[index];
    const std::string context =
        "input-manifest.session_assets[" + std::to_string(index) + "]";
    RequireOnlyMembers(asset, {"relative_path", "required_for_phase1", "file"},
                       context);
    const std::string relative = RequireString(
        Member(asset, "relative_path", context), context + ".relative_path");
    if (relative.empty() || relative.front() == '/' || relative.find("..") != std::string::npos ||
        !asset_paths.insert(relative).second) {
      throw ReplayError(context + ".relative_path is invalid or duplicated");
    }
    RequireBool(Member(asset, "required_for_phase1", context),
                context + ".required_for_phase1");
    const FileSeal seal = ParseFileSeal(Member(asset, "file", context),
                                        context + ".file");
    if (seal.path_is_symlink || RelativeTo(seal.real_path, plan->session_real_path) != relative) {
      throw ReplayError(context + " does not identify its session relative path");
    }
    RegisterSeal(seal, context + ".file", plan);
    plan->session_relative_paths.insert(relative);
  }
  const uint64_t inventory_count = RequireUInt(
      Member(session, "inventory_file_count", "input-manifest.session"),
      "input-manifest.session.inventory_file_count");
  if (inventory_count != plan->session_relative_paths.size()) {
    throw ReplayError("input-manifest session inventory count is inconsistent");
  }

  const Json::Value& required_assets =
      Member(manifest, "required_session_assets", "input-manifest");
  RequireArray(required_assets, "input-manifest.required_session_assets");
  for (Json::ArrayIndex index = 0; index < required_assets.size(); ++index) {
    const std::string relative = RequireString(
        required_assets[index], "input-manifest.required_session_assets");
    if (asset_paths.count(relative) == 0) {
      throw ReplayError("Required session asset is absent from session_assets");
    }
  }

  const Json::Value& binaries = Member(manifest, "binaries", "input-manifest");
  RequireOnlyMembers(binaries, {"frontend", "mapper", "texrecon"},
                     "input-manifest.binaries");
  for (const std::string& name : {"frontend", "mapper", "texrecon"}) {
    const FileSeal seal = ParseFileSeal(binaries[name],
                                        "input-manifest.binaries." + name);
    RegisterSeal(seal, "input-manifest.binaries." + name, plan);
    if (::access(seal.path.c_str(), X_OK) != 0) {
      throw ReplayError("Sealed binary is not executable: " + seal.path);
    }
    if (name == "mapper") plan->mapper_binary_path = seal.path;
  }
  ValidateRepository(Member(manifest, "repository", "input-manifest"), plan,
                     "input-manifest.repository");
  ValidateI3dgs(Member(manifest, "i3dgs", "input-manifest"), plan,
                "input-manifest.i3dgs");
  RequireObject(Member(manifest, "pose_conversion", "input-manifest"),
                "input-manifest.pose_conversion");
  const Json::Value& conversion =
      Member(manifest, "pose_conversion", "input-manifest");
  RequireStringValue(Member(conversion, "fastlio_pose_convention", "pose_conversion"),
                     "T_wc", "input-manifest.pose_conversion.fastlio_pose_convention");
  RequireStringValue(Member(conversion, "colmap_prior_convention", "pose_conversion"),
                     "T_cw", "input-manifest.pose_conversion.colmap_prior_convention");
  const FileSeal implementation = ParseFileSeal(
      Member(conversion, "implementation_file", "pose_conversion"),
      "input-manifest.pose_conversion.implementation_file");
  RegisterSeal(implementation, "input-manifest.pose_conversion.implementation_file",
               plan);
  RequireObject(Member(manifest, "stability_verification", "input-manifest"),
                "input-manifest.stability_verification");
}

void ValidateResolvedConfig(const Json::Value& config,
                            const Json::Value& manifest,
                            const ReplayPlan& plan) {
  RequireOnlyMembers(
      config,
      {"schema", "phase", "run_mode", "session_dir", "artifact_dir",
       "expected_frame_count", "resolved_frame_count", "input_manifest_sha256",
       "input_policy", "replay", "camera", "frontend_baseline_not_executed",
       "mapper_baseline_not_executed", "texrecon_not_executed",
       "pose_conversion", "i3dgs_reference"},
      "resolved-config");
  RequireStringValue(Member(config, "schema", "resolved-config"), kConfigSchema,
                     "resolved-config.schema");
  RequireUIntValue(Member(config, "phase", "resolved-config"), 1,
                   "resolved-config.phase");
  RequireStringValue(Member(config, "run_mode", "resolved-config"),
                     "PREPARE_ONLY_DRY_RUN", "resolved-config.run_mode");
  RequireStringValue(Member(config, "session_dir", "resolved-config"),
                     plan.session_real_path, "resolved-config.session_dir");
  if (RealPath(RequireString(Member(config, "artifact_dir", "resolved-config"),
                             "resolved-config.artifact_dir")) !=
      plan.sealed_artifact) {
    throw ReplayError("resolved-config.artifact_dir differs from sealed artifact");
  }
  const Json::Value& expected = Member(config, "expected_frame_count", "resolved-config");
  if (!expected.isNull() && RequireUInt(expected, "resolved-config.expected_frame_count") !=
                                plan.expected_frame_count) {
    throw ReplayError("resolved-config expected frame count is inconsistent");
  }
  RequireUIntValue(Member(config, "resolved_frame_count", "resolved-config"),
                   plan.expected_frame_count, "resolved-config.resolved_frame_count");
  RequireStringValue(Member(config, "input_manifest_sha256", "resolved-config"),
                     plan.manifest_sha256, "resolved-config.input_manifest_sha256");
  const Json::Value& policy = Member(config, "input_policy", "resolved-config");
  RequireObject(policy, "resolved-config.input_policy");
  RequireBoolValue(Member(policy, "artifact_must_not_exist", "input_policy"), true,
                   "resolved-config.input_policy.artifact_must_not_exist");
  RequireBoolValue(Member(policy, "overwrite_allowed", "input_policy"), false,
                   "resolved-config.input_policy.overwrite_allowed");
  RequireStringValue(Member(policy, "input_mutation_detection", "input_policy"),
                     "FULL_REHASH_BEFORE_AND_AFTER_REPLAY",
                     "resolved-config.input_policy.input_mutation_detection");
  const Json::Value& replay = Member(config, "replay", "resolved-config");
  RequireObject(replay, "resolved-config.replay");
  RequireStringValue(Member(replay, "event_order", "replay"),
                     "ARRIVED_BY_NUMERIC_FRAME_INDEX",
                     "resolved-config.replay.event_order");
  RequireStringValue(Member(replay, "visibility", "replay"),
                     "CURRENT_AND_HISTORY_ONLY", "resolved-config.replay.visibility");
  RequireBoolValue(Member(replay, "release_future_paths", "replay"), false,
                   "resolved-config.replay.release_future_paths");
  RequireBoolValue(Member(replay, "release_future_features", "replay"), false,
                   "resolved-config.replay.release_future_features");

  const OnlineI3dgsReplayCamera config_camera = ParseCamera(
      Member(config, "camera", "resolved-config"), false, nullptr,
      "resolved-config.camera");
  if (config_camera.model != plan.camera.model ||
      config_camera.width != plan.camera.width ||
      config_camera.height != plan.camera.height ||
      config_camera.params != plan.camera.params) {
    throw ReplayError("resolved-config camera differs from input-manifest");
  }
  const Json::Value& mapper =
      Member(config, "mapper_baseline_not_executed", "resolved-config");
  RequireObject(mapper, "resolved-config.mapper_baseline_not_executed");
  RequireUIntValue(Member(mapper, "Mapper.online_mode", "mapper"), 1,
                   "resolved-config.Mapper.online_mode");
  RequireUIntValue(Member(mapper, "Mapper.ba_global_enabled", "mapper"), 0,
                   "resolved-config.Mapper.ba_global_enabled");
  RequireUIntValue(Member(mapper, "Mapper.ba_refine_focal_length", "mapper"), 0,
                   "resolved-config.Mapper.ba_refine_focal_length");
  RequireUIntValue(Member(mapper, "Mapper.ba_refine_principal_point", "mapper"), 0,
                   "resolved-config.Mapper.ba_refine_principal_point");
  RequireUIntValue(Member(mapper, "Mapper.ba_refine_extra_params", "mapper"), 0,
                   "resolved-config.Mapper.ba_refine_extra_params");
  if (Member(mapper, "binary", "mapper") !=
      Member(Member(manifest, "binaries", "input-manifest"), "mapper", "binaries")) {
    throw ReplayError("resolved-config mapper binary differs from manifest");
  }
  const Json::Value& frontend =
      Member(config, "frontend_baseline_not_executed", "resolved-config");
  RequireObject(frontend, "resolved-config.frontend_baseline_not_executed");
  if (Member(frontend, "binary", "frontend") !=
      Member(Member(manifest, "binaries", "input-manifest"), "frontend", "binaries")) {
    throw ReplayError("resolved-config frontend binary differs from manifest");
  }
  const Json::Value& texrecon =
      Member(config, "texrecon_not_executed", "resolved-config");
  RequireObject(texrecon, "resolved-config.texrecon_not_executed");
  if (Member(texrecon, "binary", "texrecon") !=
      Member(Member(manifest, "binaries", "input-manifest"), "texrecon", "binaries")) {
    throw ReplayError("resolved-config texrecon binary differs from manifest");
  }
  if (Member(config, "i3dgs_reference", "resolved-config") !=
      Member(manifest, "i3dgs", "input-manifest")) {
    throw ReplayError("resolved-config i3dgs reference differs from manifest");
  }
}

std::string RemoveEventHashField(const std::string& line,
                                 const std::string& event_hash) {
  const std::string needle = "\"event_sha256\":\"" + event_hash + "\",";
  const size_t position = line.find(needle);
  if (position == std::string::npos || line.find(needle, position + 1) != std::string::npos) {
    throw ReplayError("Frame event is not in the sealed canonical JSONL form");
  }
  std::string canonical = line;
  canonical.erase(position, needle.size());
  return canonical;
}

void ValidateIndexArray(const Json::Value& value,
                        size_t last,
                        const std::string& context) {
  RequireArray(value, context);
  if (value.size() != last) throw ReplayError(context + " has an invalid length");
  for (Json::ArrayIndex index = 0; index < value.size(); ++index) {
    RequireUIntValue(value[index], static_cast<uint64_t>(index) + 1, context);
  }
}

void ParseEvents(const std::string& text,
                 const std::vector<Json::Value>& frame_poses,
                 ReplayPlan* plan) {
  if (text.empty() || text.back() != '\n') {
    throw ReplayError("logs/frames.jsonl must end with a newline");
  }
  std::istringstream stream(text);
  std::string line;
  std::string previous_hash;
  size_t offset = 0;
  while (std::getline(stream, line)) {
    if (line.empty()) throw ReplayError("logs/frames.jsonl contains an empty line");
    if (offset >= plan->frames.size()) {
      throw ReplayError("logs/frames.jsonl has more events than the manifest");
    }
    const std::string context = "logs/frames.jsonl line " + std::to_string(offset + 1);
    const Json::Value event = ParseJson(line, context);
    RequireOnlyMembers(
        event,
        {"schema", "event_sequence", "event_type", "frame_index", "image_name",
         "state_before", "state_after", "phase1_disposition", "attempt_no",
         "registration_attempted", "released_inputs", "released_pose", "causality",
         "previous_event_sha256", "event_sha256"},
        context);
    RequireStringValue(Member(event, "schema", context), kEventSchema,
                       context + ".schema");
    RequireUIntValue(Member(event, "event_sequence", context), offset + 1,
                     context + ".event_sequence");
    RequireUIntValue(Member(event, "frame_index", context), offset + 1,
                     context + ".frame_index");
    RequireStringValue(Member(event, "event_type", context), "ARRIVED",
                       context + ".event_type");
    RequireStringValue(Member(event, "state_before", context), "SEALED",
                       context + ".state_before");
    RequireStringValue(Member(event, "state_after", context), "ARRIVED",
                       context + ".state_after");
    RequireStringValue(Member(event, "phase1_disposition", context),
                       "PREPARED_ONLY_NO_MAPPER",
                       context + ".phase1_disposition");
    RequireUIntValue(Member(event, "attempt_no", context), 0,
                     context + ".attempt_no");
    RequireBoolValue(Member(event, "registration_attempted", context), false,
                     context + ".registration_attempted");
    RequireStringValue(Member(event, "image_name", context),
                       plan->frames[offset].input.image_name,
                       context + ".image_name");

    const Json::Value& previous = Member(event, "previous_event_sha256", context);
    if (offset == 0) {
      if (!previous.isNull()) throw ReplayError("First event previous hash must be null");
    } else {
      RequireStringValue(previous, previous_hash, context + ".previous_event_sha256");
    }
    const std::string event_hash = RequireString(
        Member(event, "event_sha256", context), context + ".event_sha256");
    if (!IsLowerHex(event_hash, 64) ||
        Sha256(RemoveEventHashField(line, event_hash)) != event_hash) {
      throw ReplayError("Frame event SHA-256 chain is invalid at frame " +
                        std::to_string(offset + 1));
    }

    const Json::Value& released = Member(event, "released_inputs", context);
    RequireOnlyMembers(released, {"jpg", "cam", "odom", "scan"},
                       context + ".released_inputs");
    struct ExpectedReleased {
      const char* kind;
      const std::string* path;
      const std::string* sha256;
    };
    const OnlineI3dgsReplayFrameInput& input = plan->frames[offset].input;
    const std::array<ExpectedReleased, 4> expected = {{
        {"jpg", &input.image_path, &input.image_sha256},
        {"cam", &input.camera_path, &input.camera_sha256},
        {"odom", &input.odometry_path, &input.odometry_sha256},
        {"scan", &input.scan_path, &input.scan_sha256},
    }};
    for (const auto& item : expected) {
      const Json::Value& record = Member(released, item.kind, context);
      RequireOnlyMembers(record, {"path", "sha256"},
                         context + ".released_inputs." + item.kind);
      RequireStringValue(Member(record, "path", context), *item.path,
                         context + ".released_inputs.path");
      RequireStringValue(Member(record, "sha256", context), *item.sha256,
                         context + ".released_inputs.sha256");
    }
    if (Member(event, "released_pose", context) != frame_poses[offset]) {
      throw ReplayError(context + ".released_pose differs from input-manifest");
    }

    const Json::Value& causality = Member(event, "causality", context);
    RequireOnlyMembers(
        causality,
        {"visibility_rule", "visible_frame_indices", "released_pose_frame_indices",
         "max_visible_frame_index", "visible_feature_frame_indices",
         "visible_match_pairs", "future_paths_exposed", "future_features_exposed"},
        context + ".causality");
    RequireStringValue(Member(causality, "visibility_rule", context),
                       "CURRENT_AND_HISTORY_ONLY", context + ".causality.visibility_rule");
    RequireUIntValue(Member(causality, "max_visible_frame_index", context), offset + 1,
                     context + ".causality.max_visible_frame_index");
    ValidateIndexArray(Member(causality, "visible_frame_indices", context), offset + 1,
                       context + ".causality.visible_frame_indices");
    ValidateIndexArray(Member(causality, "released_pose_frame_indices", context),
                       offset + 1, context + ".causality.released_pose_frame_indices");
    RequireArray(Member(causality, "visible_feature_frame_indices", context),
                 context + ".causality.visible_feature_frame_indices");
    RequireArray(Member(causality, "visible_match_pairs", context),
                 context + ".causality.visible_match_pairs");
    if (!Member(causality, "visible_feature_frame_indices", context).empty() ||
        !Member(causality, "visible_match_pairs", context).empty()) {
      throw ReplayError(context + " exposes Phase 1 feature state");
    }
    RequireBoolValue(Member(causality, "future_paths_exposed", context), false,
                     context + ".causality.future_paths_exposed");
    RequireBoolValue(Member(causality, "future_features_exposed", context), false,
                     context + ".causality.future_features_exposed");

    plan->frames[offset].event_sha256 = event_hash;
    previous_hash = event_hash;
    ++offset;
  }
  if (offset != plan->frames.size()) {
    throw ReplayError("logs/frames.jsonl event count differs from manifest");
  }
  plan->event_chain_tail_sha256 = previous_hash;
}

FileSeal SnapshotUnrecordedFile(const std::string& path) {
  struct stat entry;
  if (::lstat(path.c_str(), &entry) != 0 || !S_ISREG(entry.st_mode) ||
      S_ISLNK(entry.st_mode)) {
    throw ReplayError("Phase 1 contract file is not a regular file: " + path);
  }
  FileSeal seal;
  seal.path = path;
  seal.real_path = RealPath(path);
  seal.size_bytes = static_cast<uint64_t>(entry.st_size);
  seal.device = static_cast<uint64_t>(entry.st_dev);
  seal.inode = static_cast<uint64_t>(entry.st_ino);
  seal.mode = static_cast<uint64_t>(entry.st_mode & 07777);
  seal.mtime_ns = StatNanoseconds(entry.st_mtim);
  seal.ctime_ns = StatNanoseconds(entry.st_ctim);
  seal.path_is_symlink = false;

  SHA256_CTX context;
  if (SHA256_Init(&context) != 1) throw ReplayError("SHA256_Init failed");
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) throw ReplayError(ErrnoMessage("open", path));
  std::array<unsigned char, 1024 * 1024> buffer;
  while (true) {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR) continue;
      ::close(descriptor);
      throw ReplayError(ErrnoMessage("read", path));
    }
    if (count == 0) break;
    SHA256_Update(&context, buffer.data(), static_cast<size_t>(count));
  }
  ::close(descriptor);
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  SHA256_Final(digest.data(), &context);
  seal.sha256 = HexDigest(digest.data(), digest.size());
  return seal;
}

void ValidatePhase1Summary(const Json::Value& summary,
                           const ReplayPlan& plan) {
  RequireObject(summary, "Phase 1 run-summary");
  RequireStringValue(Member(summary, "schema", "Phase 1 run-summary"),
                     kPhase1SummarySchema, "Phase 1 run-summary.schema");
  RequireStringValue(Member(summary, "status", "Phase 1 run-summary"),
                     "COMPLETED", "Phase 1 run-summary.status");
  RequireUIntValue(Member(summary, "frame_count", "Phase 1 run-summary"),
                   plan.expected_frame_count, "Phase 1 run-summary.frame_count");
  RequireUIntValue(Member(summary, "causal_event_count", "Phase 1 run-summary"),
                   plan.expected_frame_count,
                   "Phase 1 run-summary.causal_event_count");
  RequireUIntValue(Member(summary, "future_path_exposure_count", "Phase 1 run-summary"),
                   0, "Phase 1 run-summary.future_path_exposure_count");
  RequireStringValue(Member(summary, "event_chain_tail_sha256", "Phase 1 run-summary"),
                     plan.event_chain_tail_sha256,
                     "Phase 1 run-summary.event_chain_tail_sha256");
  const Json::Value& outputs = Member(summary, "outputs", "Phase 1 run-summary");
  RequireObject(outputs, "Phase 1 run-summary.outputs");
  RequireStringValue(Member(Member(outputs, "input_manifest", "outputs"), "sha256", "input_manifest"),
                     plan.manifest_sha256, "Phase 1 input manifest hash");
  RequireStringValue(Member(Member(outputs, "resolved_config", "outputs"), "sha256", "resolved_config"),
                     plan.config_sha256, "Phase 1 resolved config hash");
  RequireStringValue(Member(Member(outputs, "frames", "outputs"), "sha256", "frames"),
                     plan.phase1_frames_sha256, "Phase 1 frames hash");
}

ReplayPlan LoadReplayPlan(const OnlineI3dgsReplayOptions& options) {
  if (options.sealed_artifact.empty() || options.output_path.empty()) {
    throw ReplayError("--sealed_artifact and --output_path are required");
  }
  ReplayPlan plan;
  plan.sealed_artifact = RealPath(options.sealed_artifact);
  struct stat artifact_stat;
  if (::stat(plan.sealed_artifact.c_str(), &artifact_stat) != 0 ||
      !S_ISDIR(artifact_stat.st_mode)) {
    throw ReplayError("--sealed_artifact must be a directory");
  }
  const std::string manifest_path = plan.sealed_artifact + "/input-manifest.json";
  const std::string config_path = plan.sealed_artifact + "/resolved-config.json";
  const std::string frames_path = plan.sealed_artifact + "/logs/frames.jsonl";
  const std::string summary_path = plan.sealed_artifact + "/run-summary.json";
  plan.manifest_text = ReadFile(manifest_path);
  plan.config_text = ReadFile(config_path);
  plan.phase1_frames_text = ReadFile(frames_path);
  const std::string summary_text = ReadFile(summary_path);
  plan.manifest_sha256 = Sha256(plan.manifest_text);
  plan.config_sha256 = Sha256(plan.config_text);
  plan.phase1_frames_sha256 = Sha256(plan.phase1_frames_text);

  const Json::Value manifest = ParseJson(plan.manifest_text, manifest_path);
  std::vector<Json::Value> frame_poses;
  ParseManifest(manifest, &plan, &frame_poses);
  if (plan.expected_frame_count != options.required_frame_count) {
    throw ReplayError("Sealed expected frame count is not " +
                      std::to_string(options.required_frame_count));
  }
  const Json::Value config = ParseJson(plan.config_text, config_path);
  ValidateResolvedConfig(config, manifest, plan);
  ParseEvents(plan.phase1_frames_text, frame_poses, &plan);
  ValidatePhase1Summary(ParseJson(summary_text, summary_path), plan);

  RegisterSeal(SnapshotUnrecordedFile(manifest_path), manifest_path, &plan);
  RegisterSeal(SnapshotUnrecordedFile(config_path), config_path, &plan);
  RegisterSeal(SnapshotUnrecordedFile(frames_path), frames_path, &plan);
  RegisterSeal(SnapshotUnrecordedFile(summary_path), summary_path, &plan);
  const Json::Value summary = ParseJson(summary_text, summary_path);
  const Json::Value& pose_output =
      Member(Member(summary, "outputs", "Phase 1 run-summary"), "pose_prior", "outputs");
  const std::string pose_path =
      RequireString(Member(pose_output, "path", "pose_prior"), "pose_prior.path");
  FileSeal pose_seal = SnapshotUnrecordedFile(pose_path);
  const std::string expected_pose_hash =
      RequireString(Member(pose_output, "sha256", "pose_prior"), "pose_prior.sha256");
  if (pose_seal.sha256 != expected_pose_hash) {
    throw ReplayError("Phase 1 pose-prior SHA-256 mismatch");
  }
  RegisterSeal(pose_seal, "Phase 1 pose-prior", &plan);

  if (options.max_frames > plan.frames.size()) {
    throw ReplayError("--max_frames exceeds the sealed frame count");
  }
  return plan;
}

void FsyncDirectory(const boost::filesystem::path& path) {
  const int descriptor = ::open(path.string().c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (descriptor < 0) throw ReplayError(ErrnoMessage("open directory", path.string()));
  if (::fsync(descriptor) != 0) {
    ::close(descriptor);
    throw ReplayError(ErrnoMessage("fsync directory", path.string()));
  }
  ::close(descriptor);
}

void RenameNoReplace(const std::string& source, const std::string& destination) {
#if defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
  if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                destination.c_str(), RENAME_NOREPLACE) == 0) {
    return;
  }
  if (errno != ENOSYS && errno != EINVAL) {
    throw ReplayError(ErrnoMessage("renameat2", destination));
  }
#endif
  if (::link(source.c_str(), destination.c_str()) != 0) {
    throw ReplayError(ErrnoMessage("link", destination));
  }
  if (::unlink(source.c_str()) != 0) {
    throw ReplayError(ErrnoMessage("unlink", source));
  }
}

class ArtifactWriter {
 public:
  explicit ArtifactWriter(const std::string& requested_path) {
    root_ = ProspectiveOutputPath(requested_path);
    struct stat existing;
    if (::lstat(root_.c_str(), &existing) == 0 || errno != ENOENT) {
      throw ReplayError("Output path already exists; refusing to overwrite: " + root_);
    }
    if (::mkdir(root_.c_str(), 0700) != 0) {
      throw ReplayError(ErrnoMessage("mkdir", root_));
    }
    CreateDirectory("logs");
    CreateDirectory("graph");
    CreateDirectory("models");
    WriteJson("INCOMPLETE", InitialIncomplete(), false);
  }

  const std::string& Root() const { return root_; }

  void WriteText(const std::string& relative,
                 const std::string& contents,
                 bool replace) {
    const std::string destination = Path(relative);
    const std::string temporary =
        destination + ".tmp-" + std::to_string(::getpid()) + "-" +
        std::to_string(++temporary_sequence_);
    const int descriptor = ::open(temporary.c_str(),
                                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                  0600);
    if (descriptor < 0) throw ReplayError(ErrnoMessage("open", temporary));
    try {
      WriteAll(descriptor, contents.data(), contents.size());
      if (::fsync(descriptor) != 0) {
        throw ReplayError(ErrnoMessage("fsync", temporary));
      }
    } catch (...) {
      ::close(descriptor);
      ::unlink(temporary.c_str());
      throw;
    }
    ::close(descriptor);
    try {
      if (replace) {
        if (::rename(temporary.c_str(), destination.c_str()) != 0) {
          throw ReplayError(ErrnoMessage("rename", destination));
        }
      } else {
        RenameNoReplace(temporary, destination);
      }
      FsyncDirectory(boost::filesystem::path(destination).parent_path());
    } catch (...) {
      ::unlink(temporary.c_str());
      throw;
    }
  }

  void WriteJson(const std::string& relative,
                 const Json::Value& value,
                 bool replace) {
    WriteText(relative, CompactJson(value) + "\n", replace);
  }

  void AppendText(const std::string& relative, const std::string& contents) {
    if (contents.empty()) return;
    const std::string destination = Path(relative);
    const int descriptor = ::open(destination.c_str(),
                                  O_WRONLY | O_APPEND | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) throw ReplayError(ErrnoMessage("open", destination));
    struct stat previous;
    if (::fstat(descriptor, &previous) != 0 || !S_ISREG(previous.st_mode)) {
      ::close(descriptor);
      throw ReplayError("Append destination is not a regular file: " + destination);
    }
    try {
      WriteAll(descriptor, contents.data(), contents.size());
      if (::fsync(descriptor) != 0) {
        throw ReplayError(ErrnoMessage("fsync", destination));
      }
    } catch (...) {
      const bool restored = ::ftruncate(descriptor, previous.st_size) == 0 &&
                            ::fsync(descriptor) == 0;
      ::close(descriptor);
      if (!restored) {
        throw ReplayError("Failed to restore incomplete log append: " + destination);
      }
      throw;
    }
    ::close(descriptor);
  }

  void PublishModel(const OnlineI3dgsReplayController& controller) {
    const std::string staging =
        Path("models/.final.tmp-" + std::to_string(::getpid()));
    if (::mkdir(staging.c_str(), 0700) != 0) {
      throw ReplayError(ErrnoMessage("mkdir", staging));
    }
    std::string error;
    if (!controller.WriteModel(staging, &error)) {
      throw ReplayError("Failed to write final model: " + error);
    }
    FsyncTree(staging);
    RenameNoReplace(staging, Path("models/final"));
    FsyncDirectory(boost::filesystem::path(Path("models")));
  }

  void MarkCompleted(const Json::Value& summary, bool smoke) {
    WriteJson("run-summary.json", summary, true);
    Json::Value marker(Json::objectValue);
    marker["schema"] = "online_i3dgs_replay_completion_v1";
    marker["status"] = smoke ? "SMOKE_COMPLETED" : "COMPLETED";
    marker["run_summary_sha256"] = Sha256(CompactJson(summary) + "\n");
    WriteJson(smoke ? "SMOKE_COMPLETED" : "COMPLETED", marker, false);
    if (::unlink(Path("INCOMPLETE").c_str()) != 0 && errno != ENOENT) {
      throw ReplayError(ErrnoMessage("unlink", Path("INCOMPLETE")));
    }
    FsyncDirectory(root_);
  }

  void MarkIncomplete(const Json::Value& summary) noexcept {
    try {
      WriteJson("run-summary.json", summary, true);
      Json::Value marker(Json::objectValue);
      marker["schema"] = "online_i3dgs_replay_incomplete_v1";
      marker["status"] = "INCOMPLETE";
      marker["last_consistent_frame_index"] =
          summary.get("last_consistent_frame_index", Json::UInt64(0));
      marker["error"] = summary.get("error", "unknown failure");
      WriteJson("INCOMPLETE", marker, true);
    } catch (...) {
    }
  }

 private:
  Json::Value InitialIncomplete() const {
    Json::Value marker(Json::objectValue);
    marker["schema"] = "online_i3dgs_replay_incomplete_v1";
    marker["status"] = "INCOMPLETE";
    marker["last_consistent_frame_index"] = Json::UInt64(0);
    marker["error"] = "replay started but did not publish completion";
    return marker;
  }

  std::string Path(const std::string& relative) const {
    if (relative.empty() || relative.front() == '/' ||
        relative.find("..") != std::string::npos) {
      throw ReplayError("Invalid artifact relative path: " + relative);
    }
    return root_ + "/" + relative;
  }

  void CreateDirectory(const std::string& relative) {
    const std::string path = Path(relative);
    if (::mkdir(path.c_str(), 0700) != 0) {
      throw ReplayError(ErrnoMessage("mkdir", path));
    }
    FsyncDirectory(boost::filesystem::path(path).parent_path());
  }

  void FsyncTree(const std::string& root) {
    boost::system::error_code error;
    std::vector<boost::filesystem::path> directories;
    directories.push_back(root);
    boost::filesystem::recursive_directory_iterator iterator(root, error);
    const boost::filesystem::recursive_directory_iterator end;
    if (error) throw ReplayError("Cannot inspect staged model: " + error.message());
    for (; iterator != end; iterator.increment(error)) {
      if (error) throw ReplayError("Cannot inspect staged model: " + error.message());
      const auto status = boost::filesystem::symlink_status(iterator->path(), error);
      if (error || boost::filesystem::is_symlink(status)) {
        throw ReplayError("Staged model contains an invalid entry");
      }
      if (boost::filesystem::is_directory(status)) {
        directories.push_back(iterator->path());
      } else if (boost::filesystem::is_regular_file(status)) {
        const int descriptor =
            ::open(iterator->path().string().c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (descriptor < 0 || ::fsync(descriptor) != 0) {
          if (descriptor >= 0) ::close(descriptor);
          throw ReplayError("Cannot fsync staged model file");
        }
        ::close(descriptor);
      } else {
        throw ReplayError("Staged model contains a non-regular entry");
      }
    }
    for (auto iterator = directories.rbegin(); iterator != directories.rend(); ++iterator) {
      FsyncDirectory(*iterator);
    }
  }

  std::string root_;
  uint64_t temporary_sequence_ = 0;
};

Json::Value ParseControllerJson(const std::string& text,
                                const std::string& context,
                                bool allow_empty) {
  if (text.empty()) {
    if (allow_empty) return Json::Value(Json::nullValue);
    throw ReplayError(context + " is missing structured JSON");
  }
  Json::Value value = ParseJson(text, context);
  RequireObject(value, context);
  return value;
}

Json::Value BaseSummary(const ReplayPlan& plan,
                        size_t selected_frame_count,
                        const OnlineI3dgsReplayResult& result) {
  Json::Value summary(Json::objectValue);
  summary["schema"] = "online_i3dgs_replay_run_summary_v1";
  summary["status"] = "INCOMPLETE";
  summary["complete"] = false;
  summary["smoke"] = selected_frame_count < plan.frames.size();
  summary["sealed_artifact"] = plan.sealed_artifact;
  summary["input_manifest_sha256"] = plan.manifest_sha256;
  summary["phase1_frames_sha256"] = plan.phase1_frames_sha256;
  summary["event_chain_tail_sha256"] = plan.event_chain_tail_sha256;
  summary["expected_frame_count"] = Json::UInt64(plan.frames.size());
  summary["selected_frame_count"] = Json::UInt64(selected_frame_count);
  summary["processed_frame_count"] = Json::UInt64(result.processed_frame_count);
  summary["last_consistent_frame_index"] =
      Json::UInt64(result.last_consistent_frame_index);
  summary["global_ba_call_count"] = Json::UInt64(result.global_ba_call_count);
  summary["error"] = result.error;
  return summary;
}

BundleAdjustmentOptions ProductionBundleAdjustmentOptions();
OnlineLidarAssociationOptions ProductionAssociationOptions();

Json::Value ResolvedReplayConfig(const ReplayPlan& plan,
                                 const OnlineI3dgsReplayOptions& options,
                                 const std::string& output_path,
                                 size_t selected_frame_count) {
  Json::Value config(Json::objectValue);
  config["schema"] = "online_i3dgs_replay_resolved_config_v1";
  config["sealed_artifact"] = plan.sealed_artifact;
  config["output_path"] = output_path;
  config["input_manifest_sha256"] = plan.manifest_sha256;
  config["phase1_resolved_config_sha256"] = plan.config_sha256;
  config["phase1_frames_sha256"] = plan.phase1_frames_sha256;
  config["expected_frame_count"] = Json::UInt64(plan.frames.size());
  config["selected_frame_count"] = Json::UInt64(selected_frame_count);
  config["max_frames"] = Json::UInt64(options.max_frames);
  config["ba_window_size"] = Json::UInt64(options.ba_window_size);
  config["run_mode"] = selected_frame_count < plan.frames.size()
                           ? "SMOKE_REPLAY"
                           : "FULL_REPLAY";
  config["input_mutation_detection"] = "FULL_REHASH_BEFORE_AND_AFTER_REPLAY";
  config["future_input_release"] = "CURRENT_FRAME_ONLY";
  Json::Value backend(Json::objectValue);
  backend["compiled_gpu_ba_cuda"] = true;
  backend["requested"] = "custom_cuda";
  backend["fallback_allowed"] = false;
  const BundleAdjustmentOptions ba_options = ProductionBundleAdjustmentOptions();
  backend["audit_profile"] = ba_options.ba_cuda_audit_profile;
  backend["arithmetic_precision"] = ba_options.ba_cuda_arithmetic_precision;
  backend["hessian_assembly_backend"] =
      ba_options.ba_cuda_hessian_assembly_backend;
  backend["hot_kernel_mode"] = ba_options.ba_cuda_hot_kernel_mode;
  backend["schur_contribution_backend"] =
      ba_options.ba_cuda_schur_contribution_backend;
  backend["host_problem_store"] = ba_options.ba_cuda_host_problem_store;
  backend["lidar_residual"] = ba_options.ba_lidar_residual;
  backend["proj_lidar_constraint_weight"] = ba_options.proj_lidar_constraint_weight;
  backend["icp_lidar_constraint_weight"] = ba_options.icp_lidar_constraint_weight;
  backend["icp_ground_lidar_constraint_weight"] =
      ba_options.icp_ground_lidar_constraint_weight;
  backend["max_num_iterations"] = ba_options.solver_options.max_num_iterations;
  backend["function_tolerance"] = ba_options.solver_options.function_tolerance;
  backend["gradient_tolerance"] = ba_options.solver_options.gradient_tolerance;
  backend["parameter_tolerance"] = ba_options.solver_options.parameter_tolerance;
  backend["loss_function_type"] =
      ba_options.loss_function_type ==
              BundleAdjustmentOptions::LossFunctionType::SOFT_L1
          ? "SOFT_L1"
          : "TRIVIAL";
  backend["loss_function_scale"] = ba_options.loss_function_scale;
  config["backend"] = backend;
  const OnlineLidarAssociationOptions association_options =
      ProductionAssociationOptions();
  Json::Value association(Json::objectValue);
  association["local_lidar_kdtree_only"] =
      association_options.local_lidar_kdtree_only;
  association["min_proj_num"] = association_options.min_proj_num;
  association["kdtree_max_search_range"] =
      association_options.kdtree_max_search_range;
  association["kdtree_min_search_range"] =
      association_options.kdtree_min_search_range;
  association["search_range_drop_speed"] =
      association_options.search_range_drop_speed;
  association["ba_match_features_threshold"] =
      association_options.ba_match_features_threshold;
  config["association"] = association;
  Json::Value camera(Json::objectValue);
  camera["model"] = plan.camera.model;
  camera["width"] = Json::UInt64(plan.camera.width);
  camera["height"] = Json::UInt64(plan.camera.height);
  for (double parameter : plan.camera.params) camera["params"].append(parameter);
  config["camera"] = camera;
  return config;
}

std::string SerializeBaRecords(const std::vector<std::string>& source,
                               size_t frame_index,
                               size_t previous_record_count) {
  std::string output;
  for (size_t index = 0; index < source.size(); ++index) {
    Json::Value record = ParseControllerJson(
        source[index], "controller BA pass record", false);
    record["replay_frame_index"] = Json::UInt64(frame_index);
    record["replay_pass_sequence"] =
        Json::UInt64(previous_record_count + index + 1);
    output += CompactJson(record) + "\n";
  }
  return output;
}

#ifdef GPU_BA_CUDA_ENABLED

Json::Value ImageIdsJson(const std::vector<image_t>& image_ids) {
  Json::Value output(Json::arrayValue);
  for (const image_t image_id : image_ids) output.append(Json::UInt64(image_id));
  return output;
}

Json::Value BaPassJson(const OnlineI3dgsBaPassAudit& pass) {
  Json::Value output(Json::objectValue);
  output["pass_index"] = pass.pass_index;
  output["solve_native_invoked"] = pass.solve_native_invoked;
  output["solve_native_success"] = pass.solve_native_success;
  output["finite_costs"] = pass.finite_costs;
  output["finite_output"] = pass.finite_output;
  output["submitted_visual_residual_count"] =
      Json::UInt64(pass.submitted_visual_residual_count);
  output["submitted_lidar_constraint_count"] =
      Json::UInt64(pass.submitted_lidar_constraint_count);
  output["trigger_submitted_lidar_constraint_count"] =
      Json::UInt64(pass.trigger_submitted_lidar_constraint_count);
  output["trigger_preflight_checked"] = pass.trigger_preflight_checked;
  output["trigger_preflight_lidar_constraint_count"] =
      Json::UInt64(pass.trigger_preflight_lidar_constraint_count);
  output["constant_camera_pose_count"] =
      Json::UInt64(pass.constant_camera_pose_count);
  output["postprocess_count"] = Json::UInt64(pass.postprocess_count);
  output["requested_backend"] = pass.requested_backend;
  output["executed_backend"] = pass.executed_backend;
  output["requested_problem_source"] = pass.requested_problem_source;
  output["executed_problem_source"] = pass.executed_problem_source;
  output["fallback_used"] = pass.fallback_used;
  output["termination"] = pass.termination;
  output["detail"] = pass.detail;
  return output;
}

Json::Value ExecutionResultJson(const OnlineLocalBaPassResult& pass) {
  const BundleAdjustmentExecutionResult& execution = pass.execution;
  Json::Value output(Json::objectValue);
  output["schema"] = "online_i3dgs_existing_execution_result_v1";
  output["mode"] = ToString(pass.mode);
  output["attempt_id"] = Json::UInt64(pass.attempt_id);
  output["pass_index"] = pass.pass_index;
  output["trigger_image_id"] = Json::UInt64(pass.trigger_image_id);
  output["success"] = pass.success;
  output["failure_reason"] = ToString(pass.failure_reason);
  output["failure_detail"] = pass.failure_detail;
  output["call_index"] = Json::UInt64(execution.call_index);
  output["ba_kind"] = execution.ba_kind;
  output["requested_backend"] = execution.requested_backend;
  output["executed_backend"] = execution.executed_backend;
  output["success_from_execution_result"] = execution.success;
  output["termination"] = execution.termination;
  output["stable_error"] = execution.stable_error;
  output["diagnostic_message"] = execution.diagnostic_message;
  output["fallback_used"] = execution.fallback_used;
  output["fallback_reason"] = execution.fallback_reason;
  output["wall_seconds"] = execution.wall_seconds;
  output["initial_cost"] = execution.initial_cost;
  output["final_cost"] = execution.final_cost;
  output["residuals"] = Json::UInt64(execution.residuals);
  output["residual_blocks"] = Json::UInt64(execution.residual_blocks);
  output["ceres_solve_calls"] = Json::UInt64(execution.ceres_solve_calls);
  output["solver_invoked"] = pass.solver_called;
  output["converged"] = pass.termination_converged;
  output["execution_profile"] = execution.execution_profile;
  output["audit_profile"] = execution.audit_profile_effective;
  output["arithmetic_precision"] = execution.arithmetic_precision_effective;
  output["trial_steps"] = execution.trial_steps;
  output["accepted_steps"] = execution.accepted_steps;
  output["rejected_steps"] = execution.rejected_steps;
  output["submitted_lidar_constraints"] =
      Json::UInt64(pass.submitted_lidar_constraint_count);
  output["trigger_submitted_lidar_constraints"] =
      Json::UInt64(pass.trigger_submitted_lidar_constraint_count);
  output["association_input_points"] =
      Json::UInt64(pass.association_audit.input_point_count);
  output["association_projection_route_points"] =
      Json::UInt64(pass.association_audit.projection_route_point_count);
  output["association_kdtree_route_points"] =
      Json::UInt64(pass.association_audit.kdtree_route_point_count);
  output["association_projection_calls"] =
      Json::UInt64(pass.association_audit.projection_call_count);
  output["intent_build_milliseconds"] = pass.timing.intent_build_milliseconds;
  output["request_validation_milliseconds"] =
      pass.timing.request_validation_milliseconds;
  output["transaction_discard_milliseconds"] =
      pass.timing.transaction_discard_milliseconds;
  output["trigger_preflight_checked"] = pass.trigger_gate.checked;
  output["trigger_preflight_passed"] = pass.trigger_gate.passed;
  output["trigger_preflight_lidar_constraints"] =
      Json::UInt64(pass.trigger_gate.actual_count);
  output["trigger_preflight_nearest_queries"] =
      Json::UInt64(pass.trigger_gate.nearest_query_count);
  output["native_run_milliseconds"] = pass.timing.native_run_milliseconds;
  output["merge_milliseconds"] = pass.timing.merge_milliseconds;
  output["complete_milliseconds"] = pass.timing.complete_milliseconds;
  output["filter_milliseconds"] = pass.timing.filter_milliseconds;
  output["visual_state_hash_milliseconds"] =
      pass.timing.visual_state_hash_milliseconds;
  output["total_milliseconds"] = pass.timing.total_milliseconds;
  return output;
}

Json::Value GraphSnapshotJson(const ActiveCovisibilityGraph& graph) {
  const ActiveCovisibilitySnapshot snapshot = graph.Snapshot();
  Json::Value output(Json::objectValue);
  output["schema"] = "online_i3dgs_active_graph_snapshot_v1";
  output["canonical_version"] = Json::UInt64(snapshot.version);
  output["node_count"] = Json::UInt64(snapshot.nodes.size());
  output["edge_count"] = Json::UInt64(snapshot.edges.size());
  output["nodes"] = Json::Value(Json::arrayValue);
  for (const ActiveCovisibilityNode& node : snapshot.nodes) {
    Json::Value item(Json::objectValue);
    item["image_id"] = Json::UInt64(node.image_id);
    item["registration_sequence"] = Json::UInt64(node.registration_sequence);
    output["nodes"].append(std::move(item));
  }
  output["edges"] = Json::Value(Json::arrayValue);
  for (const ActiveCovisibilityEdge& edge : snapshot.edges) {
    Json::Value item(Json::objectValue);
    item["image_id1"] = Json::UInt64(edge.image_id1);
    item["image_id2"] = Json::UInt64(edge.image_id2);
    item["strength"] = Json::UInt64(edge.strength);
    item["birth_frame"] = Json::UInt64(edge.birth_frame);
    item["kind"] = edge.kind == ActiveCovisibilityEdgeKind::ORDINARY
                       ? "ORDINARY"
                       : "LOOP";
    output["edges"].append(std::move(item));
  }
  return output;
}

Json::Value CanonicalVersionJson(const OnlineMapperCanonicalVersion& version) {
  Json::Value output(Json::objectValue);
  output["reconstruction"]["structure_owner_epoch"] =
      Json::UInt64(version.reconstruction.structure_owner_epoch);
  output["reconstruction"]["structure_revision"] =
      Json::UInt64(version.reconstruction.structure_revision);
  output["reconstruction"]["publish_version"] =
      Json::UInt64(version.reconstruction.publish_version);
  output["reconstruction"]["correspondence_graph_identity"] = Json::UInt64(
      static_cast<uint64_t>(version.reconstruction.correspondence_graph_identity));
  output["known_pose_registry"] = Json::UInt64(version.known_pose_registry);
  output["active_covisibility_graph"] =
      Json::UInt64(version.active_covisibility_graph);
  output["point3D_owner_table"] = Json::UInt64(version.point3D_owner_table);
  return output;
}

Json::Value FrameAuditJson(const OnlineI3dgsFrameResult& result) {
  const OnlineI3dgsFrameAudit& audit = result.audit;
  Json::Value output(Json::objectValue);
  output["schema"] = "online_i3dgs_controller_frame_audit_v1";
  output["status"] = ToString(result.status);
  output["frame_index"] = Json::UInt64(audit.frame_index);
  output["image_id"] = Json::UInt64(audit.image_id);
  output["max_visible_frame_index"] =
      Json::UInt64(audit.max_visible_frame_index);
  output["lidar_map_version"] = Json::UInt64(audit.lidar_map_version);
  output["max_scan_index"] = Json::UInt64(audit.max_scan_index);
  output["registration_sequence"] = Json::UInt64(audit.registration_sequence);
  output["event_start_canonical_version"] =
      CanonicalVersionJson(audit.event_start_canonical_version);
  output["current_recomputed_canonical_version"] =
      CanonicalVersionJson(audit.current_recomputed_canonical_version);
  output["canonical_version_after"] =
      CanonicalVersionJson(audit.canonical_version_after);
  output["event_start_registry_version"] =
      Json::UInt64(audit.event_start_registry_version);
  output["event_start_graph_version"] =
      Json::UInt64(audit.event_start_graph_version);
  output["current_recomputed_registry_version"] =
      Json::UInt64(audit.current_recomputed_registry_version);
  output["current_recomputed_graph_version"] =
      Json::UInt64(audit.current_recomputed_graph_version);
  output["first_batch_reference_image_ids"] =
      ImageIdsJson(audit.first_batch_reference_image_ids);
  output["second_batch_reference_image_ids"] =
      ImageIdsJson(audit.second_batch_reference_image_ids);
  output["matching_extended_to_ten"] = audit.matching_extended_to_ten;
  output["maximum_finite_hop_defined"] = audit.maximum_finite_hop_defined;
  output["maximum_finite_hop"] = Json::UInt64(audit.maximum_finite_hop);
  output["current_recomputed_after_catchup"] =
      audit.current_recomputed_after_catchup;
  output["mode"] = ToString(audit.mode);
  output["dual_probe_attempted"] = audit.dual_probe_attempted;
  output["dual_fell_back_to_single"] = audit.dual_fell_back_to_single;
  output["frozen_image_ids"] = ImageIdsJson(audit.frozen_image_ids);
  output["ba_succeeded"] = audit.ba_succeeded;
  output["commit_attempted"] = audit.commit_attempted;
  output["commit_succeeded"] = audit.commit_succeeded;
  output["current_visual_active"] = audit.current_visual_active;
  output["registry_version_after"] = Json::UInt64(audit.registry_version_after);
  output["graph_version_after"] = Json::UInt64(audit.graph_version_after);
  output["global_ba_call_count"] = Json::UInt64(audit.global_ba_call_count);
  output["termination"] = audit.termination;
  output["detail"] = audit.detail;
  output["candidates"] = Json::Value(Json::arrayValue);
  for (const ConfirmedNormalizedSE3Reference& candidate : audit.candidates) {
    Json::Value item(Json::objectValue);
    item["image_id"] = Json::UInt64(candidate.image_id);
    item["score"] = candidate.score;
    item["translation_distance_meters"] =
        candidate.translation_distance_meters;
    item["rotation_distance_degrees"] = candidate.rotation_distance_degrees;
    item["registration_sequence"] =
        Json::UInt64(candidate.registration_sequence);
    item["visual_state"] =
        candidate.visual_state == KnownPoseVisualState::VISUAL_ACTIVE
            ? "VISUAL_ACTIVE"
            : "POSE_ONLY";
    output["candidates"].append(std::move(item));
  }
  output["matched_pairs"] = Json::Value(Json::arrayValue);
  for (const OnlineI3dgsPairMatch& pair : audit.matched_pairs) {
    Json::Value item(Json::objectValue);
    item["reference_image_id"] = Json::UInt64(pair.reference_image_id);
    item["raw_match_count"] = Json::UInt64(pair.raw_match_count);
    item["verified_inlier_count"] = Json::UInt64(pair.verified_inlier_count);
    output["matched_pairs"].append(std::move(item));
  }
  output["ba_passes"] = Json::Value(Json::arrayValue);
  for (const OnlineI3dgsBaPassAudit& pass : audit.ba_passes) {
    output["ba_passes"].append(BaPassJson(pass));
  }
  output["stage_milliseconds"] = Json::Value(Json::objectValue);
  for (const auto& stage : audit.stage_milliseconds) {
    output["stage_milliseconds"][stage.first] = stage.second;
  }
  return output;
}

Json::Value ControllerSummaryJson(const OnlineI3dgsRunSummary& summary) {
  Json::Value output(Json::objectValue);
  output["schema"] = "online_i3dgs_controller_run_summary_v1";
  output["status"] = ToString(summary.status);
  output["arrived_event_count"] = Json::UInt64(summary.arrived_event_count);
  output["known_pose_registered_count"] =
      Json::UInt64(summary.known_pose_registered_count);
  output["visual_active_online_count"] =
      Json::UInt64(summary.visual_active_online_count);
  output["visual_active_after_flush_count"] =
      Json::UInt64(summary.visual_active_after_flush_count);
  output["pose_only_final_count"] = Json::UInt64(summary.pose_only_final_count);
  output["bootstrap_call_count"] = Json::UInt64(summary.bootstrap_call_count);
  output["single_call_count"] = Json::UInt64(summary.single_call_count);
  output["catchup_call_count"] = Json::UInt64(summary.catchup_call_count);
  output["dual_call_count"] = Json::UInt64(summary.dual_call_count);
  output["ba_pass_count"] = Json::UInt64(summary.ba_pass_count);
  output["catchup_commit_count"] = Json::UInt64(summary.catchup_commit_count);
  output["dual_atomic_commit_count"] =
      Json::UInt64(summary.dual_atomic_commit_count);
  output["partial_commit_detection_count"] =
      Json::UInt64(summary.partial_commit_detection_count);
  output["pnp_probe_call_count"] = Json::UInt64(summary.pnp_probe_call_count);
  output["pnp_registration_call_count"] =
      Json::UInt64(summary.pnp_registration_call_count);
  output["global_ba_call_count"] = Json::UInt64(summary.global_ba_call_count);
  output["last_consistent_stage"] = summary.last_consistent_stage;
  output["incomplete_reason"] = summary.incomplete_reason;
  return output;
}

SiftExtractionOptions ProductionExtractionOptions() {
  SiftExtractionOptions options;
  options.num_threads = 8;
  options.use_gpu = true;
  options.gpu_index = "0";
  options.max_image_size = 612;
  return options;
}

SiftMatchingOptions ProductionMatchingOptions() {
  SiftMatchingOptions options;
  options.num_threads = 8;
  options.use_gpu = true;
  options.gpu_index = "0";
  options.min_num_inliers = 1;
  return options;
}

BundleAdjustmentOptions ProductionBundleAdjustmentOptions() {
  IncrementalMapperOptions mapper_options;
  mapper_options.icp_lidar_constraint_weight = 10.0;
  mapper_options.icp_ground_lidar_constraint_weight = 10.0;
  BundleAdjustmentOptions options = mapper_options.LocalBundleAdjustment();
  options.ba_backend = "custom_cuda";
  options.ba_fallback_to_ceres = false;
  options.ba_cuda_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
  options.ba_cuda_host_problem_store = "host_prepared_store";
  options.ba_cuda_execution_profile =
      BundleAdjustmentOptions::CudaExecutionProfile::COMPACT_CONTROL;
  options.ba_cuda_audit_profile = "production";
  options.ba_cuda_arithmetic_precision = "fp32_mixed";
  options.ba_cuda_hessian_assembly_backend = "observation_segmented";
  options.ba_cuda_hot_kernel_mode = "transformed";
  options.ba_cuda_schur_contribution_backend = "segmented";
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.refine_extrinsics = true;
  options.if_add_lidar_constraint = true;
  options.ba_match_features_threshold = 200;
  options.print_summary = false;
  options.solver_options.num_threads = 8;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = 8;
#endif
  return options;
}

OnlineLidarAssociationOptions ProductionAssociationOptions() {
  OnlineLidarAssociationOptions options;
  options.local_lidar_kdtree_only = true;
  options.min_proj_num = 1;
  options.kdtree_max_search_range = 0.15;
  options.kdtree_min_search_range = 0.05;
  options.search_range_drop_speed = 0.01;
  options.ba_match_features_threshold = 200;
  return options;
}

OnlineDualSE3 ImageTcw(const Image& image) {
  return OnlineDualSE3(image.Qvec(), image.Tvec());
}

Json::Value OnlineDualSe3Json(const OnlineDualSE3& pose) {
  Json::Value output(Json::objectValue);
  output["qvec_wxyz"] = Json::Value(Json::arrayValue);
  output["tvec"] = Json::Value(Json::arrayValue);
  for (int index = 0; index < 4; ++index) {
    output["qvec_wxyz"].append(pose.qvec(index));
  }
  for (int index = 0; index < 3; ++index) {
    output["tvec"].append(pose.tvec(index));
  }
  return output;
}

class ProductionBaCandidate final : public OnlineI3dgsBaCandidate {
 public:
  OnlineMapperTransactionPayload payload;
  std::set<point3D_t> directly_optimized_points;
  std::map<image_t, OnlineDualSE3> dual_before_T_wc;
  std::unordered_set<image_t> post_commit_dirty_image_ids;
  std::unordered_set<point3D_t> post_commit_dirty_point3D_ids;
};

Json::Value NeighborhoodSearchAuditJson(
    const lidar::NeighborhoodSearchAudit& audit) {
  Json::Value output(Json::objectValue);
  output["query_count"] = Json::UInt64(audit.query_count);
  output["block_probe_count"] = Json::UInt64(audit.block_probe_count);
  output["visited_block_count"] = Json::UInt64(audit.visited_block_count);
  output["visited_voxel_count"] = Json::UInt64(audit.visited_voxel_count);
  output["hit_count"] = Json::UInt64(audit.hit_count);
  output["inserted_key_count"] = Json::UInt64(audit.inserted_key_count);
  return output;
}

Json::Value LidarAppendAuditJson(const lidar::AppendScanAudit& audit) {
  Json::Value output(Json::objectValue);
  output["schema"] = "online_lidar_append_profile_v1";
  output["scan_index"] = Json::UInt64(audit.scan_index);
  output["file_size_bytes"] = Json::UInt64(audit.file_size_bytes);
  output["input_record_count"] = Json::UInt64(audit.input_record_count);
  output["accepted_record_count"] = Json::UInt64(audit.accepted_record_count);
  output["touched_voxel_count"] = Json::UInt64(audit.touched_voxel_count);
  output["new_voxel_count"] = Json::UInt64(audit.new_voxel_count);
  output["count_only_voxel_count"] = Json::UInt64(audit.count_only_voxel_count);
  output["geometry_changed_voxel_count"] =
      Json::UInt64(audit.geometry_changed_voxel_count);
  output["affected_query_count"] = Json::UInt64(audit.affected_query_count);
  output["support_voxel_count"] = Json::UInt64(audit.support_voxel_count);
  output["map_voxel_count"] = Json::UInt64(audit.map_voxel_count);
  output["map_block_count"] = Json::UInt64(audit.map_block_count);
  output["mutable_block_count"] = Json::UInt64(audit.mutable_block_count);
  output["all_voxels_queried"] = audit.all_voxels_queried;
  output["outer_valid_count"] = Json::UInt64(audit.outer_valid_count);
  output["inner_valid_count"] = Json::UInt64(audit.inner_valid_count);
  output["affected_search"] = NeighborhoodSearchAuditJson(audit.affected_search);
  output["support_search"] = NeighborhoodSearchAuditJson(audit.support_search);
  output["stage_milliseconds"] = Json::Value(Json::objectValue);
  for (const auto& stage : audit.stage_milliseconds) {
    output["stage_milliseconds"][stage.first] = stage.second;
  }
  output["cuda_milliseconds"]["upload"] = audit.cuda_timing.upload_ms;
  output["cuda_milliseconds"]["index"] = audit.cuda_timing.index_ms;
  output["cuda_milliseconds"]["normals"] = audit.cuda_timing.normals_ms;
  output["cuda_milliseconds"]["download"] = audit.cuda_timing.download_ms;
  output["total_milliseconds"] = audit.total_milliseconds;
  output["snapshot_sha256"] = audit.snapshot_sha256_after;
  output["geometry_sha256"] = audit.geometry_sha256_after;
  return output;
}

class ProductionOnlineReplay final : public OnlineI3dgsReplayController,
                                     public OnlineI3dgsFrontendDependency,
                                     public OnlineI3dgsLidarDependency,
                                     public OnlineI3dgsPnPDependency,
                                     public OnlineI3dgsLocalBaDependency,
                                     public OnlineI3dgsStateDependency,
                                     public OnlineLocalBaPostprocessAdapter {
 public:
  explicit ProductionOnlineReplay(
      const OnlineI3dgsReplayControllerConfig& config)
      : config_(config),
        database_path_(config.output_path + "/database.db.staging"),
        database_(database_path_),
        matcher_cache_(std::max<size_t>(64, config.expected_frame_count * 2),
                       &database_),
        extraction_options_(ProductionExtractionOptions()),
        matching_options_(ProductionMatchingOptions()),
        ba_options_(ProductionBundleAdjustmentOptions()),
        extractor_(extraction_options_),
        matcher_(matching_options_, &database_, &matcher_cache_),
        frontend_(OnlineMapperFrontendOptions(),
                  &database_,
                  &matcher_cache_,
                  &database_cache_,
                  &reconstruction_,
                  &extractor_,
                  &matcher_) {
    Camera camera;
    camera.SetCameraId(1);
    camera.SetModelIdFromName(config.camera.model);
    camera.SetWidth(config.camera.width);
    camera.SetHeight(config.camera.height);
    camera.SetParams(config.camera.params);
    camera.SetPriorFocalLength(true);
    if (!camera.VerifyParams() || database_.WriteCamera(camera, true) != 1) {
      throw ReplayError("Failed to initialize the fixed online camera");
    }
    matcher_cache_.Setup();
    database_cache_.Load(database_, 1, false, {});
    reconstruction_.Load(database_cache_);
    reconstruction_.SetUp(&database_cache_.CorrespondenceGraph());
    reconstruction_.BeginStructureJournal(1);
    const SingleImageFeatureExtractionResult extractor_setup = extractor_.Setup();
    if (!extractor_setup.success) {
      throw ReplayError("CUDA SIFT setup failed: " +
                        extractor_setup.rejection_reason);
    }
    if (!matcher_.SetupForMaxNumFeatures(extraction_options_.max_num_features)) {
      throw ReplayError("CUDA SIFT matcher setup failed");
    }
    OnlineI3dgsMapperOptions options;
    options.expected_frame_count = config.expected_frame_count;
    options.ba_window_size = config.ba_window_size;
    OnlineI3dgsMapperDependencies dependencies;
    dependencies.frontend = this;
    dependencies.lidar = this;
    dependencies.pnp = this;
    dependencies.local_ba = this;
    dependencies.state = this;
    mapper_.reset(new OnlineI3dgsMapper(options, dependencies));
    if (mapper_->Status() == OnlineI3dgsMapperStatus::INCOMPLETE) {
      throw ReplayError("OnlineI3dgsMapper rejected production dependencies");
    }
  }

  ~ProductionOnlineReplay() override {
    if (reconstruction_.StructureJournalEnabled()) {
      reconstruction_.EndStructureJournal();
    }
  }

  OnlineI3dgsReplayControllerFrameResult ProcessFrame(
      const OnlineI3dgsReplayFrameInput& frame) override {
    ba_execution_json_.clear();
    lidar_ingest_profile_ = Json::Value();
    OnlineI3dgsFrameInput input;
    input.frame_index = frame.frame_index;
    input.image_id = static_cast<image_t>(frame.image_id);
    input.image_path = frame.image_path;
    input.camera_path = frame.camera_path;
    input.odometry_path = frame.odometry_path;
    input.scan_path = frame.scan_path;
    input.image_sha256 = frame.image_sha256;
    input.camera_sha256 = frame.camera_sha256;
    input.odometry_sha256 = frame.odometry_sha256;
    input.scan_sha256 = frame.scan_sha256;
    input.scan_size_bytes = frame.scan_size_bytes;
    input.fastlio_T_cw = KnownPoseSE3(
        Eigen::Vector4d(frame.qvec[0], frame.qvec[1], frame.qvec[2],
                        frame.qvec[3]),
        Eigen::Vector3d(frame.tvec[0], frame.tvec[1], frame.tvec[2]));
    const OnlineI3dgsFrameResult native = mapper_->ProcessArrivedFrame(input);
    OnlineI3dgsReplayControllerFrameResult result;
    result.success = native.IsSuccess();
    result.ready_to_flush =
        native.status == OnlineI3dgsMapperStatus::READY_TO_FLUSH;
    result.global_ba_call_count = native.audit.global_ba_call_count;
    result.termination = native.audit.termination.empty()
                             ? ToString(native.status)
                             : native.audit.termination;
    Json::Value frame_audit = FrameAuditJson(native);
    if (!lidar_ingest_profile_.isNull()) {
      frame_audit["lidar_ingest_profile"] = lidar_ingest_profile_;
    }
    result.audit_json = CompactJson(frame_audit);
    result.ba_pass_json = ba_execution_json_;
    result.graph_snapshot_json = CompactJson(GraphSnapshotJson(active_graph_));
    return result;
  }

  OnlineI3dgsReplayControllerFinishResult Finish(
      const bool full_sequence) override {
    ba_execution_json_.clear();
    OnlineI3dgsReplayControllerFinishResult result;
    if (full_sequence) {
      const OnlineI3dgsFinishResult finish = mapper_->Finish();
      result.success = finish.IsComplete();
      result.global_ba_call_count = finish.summary.global_ba_call_count;
      result.termination = ToString(finish.status);
      result.summary_json = CompactJson(ControllerSummaryJson(finish.summary));
    } else {
      const OnlineI3dgsRunSummary& summary = mapper_->Summary();
      result.success = mapper_->Status() == OnlineI3dgsMapperStatus::RUNNING;
      result.global_ba_call_count = summary.global_ba_call_count;
      result.termination = "SMOKE_STOPPED_BEFORE_FLUSH";
      result.summary_json = CompactJson(ControllerSummaryJson(summary));
    }
    result.ba_pass_json = ba_execution_json_;
    result.graph_snapshot_json = CompactJson(GraphSnapshotJson(active_graph_));
    return result;
  }

  bool WriteModel(const std::string& staging_path,
                  std::string* error) const override {
    try {
      reconstruction_.Write(staging_path);
      if (!database_published_) {
        database_.Close();
        RenameNoReplace(database_path_, config_.output_path + "/database.db");
        FsyncDirectory(config_.output_path);
        database_published_ = true;
      }
      return true;
    } catch (const std::exception& exception) {
      if (error != nullptr) *error = exception.what();
      return false;
    } catch (...) {
      if (error != nullptr) *error = "unknown model write failure";
      return false;
    }
  }

  OnlineI3dgsDependencyResult IngestFrame(
      const OnlineI3dgsFrameInput& frame,
      const size_t max_visible_frame_index) override {
    if (max_visible_frame_index != frame.frame_index) {
      return {false, "frontend visibility limit differs from current frame"};
    }
    Bitmap bitmap;
    if (!bitmap.Read(frame.image_path, false)) {
      return {false, "failed to decode current frame image"};
    }
    Image image;
    image.SetImageId(frame.image_id);
    image.SetName(BaseName(frame.image_path));
    image.SetCameraId(1);
    const OnlineMapperFrameResult result =
        frontend_.IngestFrame(std::move(image), &bitmap, nullptr);
    return {result.status == OnlineMapperFrameStatus::SUCCESS, result.detail};
  }

  OnlineI3dgsMatchBatchResult MatchExplicitReferences(
      const image_t current_image_id,
      const std::vector<image_t>& ordered_reference_image_ids,
      const size_t max_visible_frame_index) override {
    OnlineI3dgsMatchBatchResult output;
    if (current_image_id != max_visible_frame_index) {
      output.detail = "matcher visibility limit differs from current image";
      return output;
    }
    const OnlineMapperFrameResult result = frontend_.MatchExplicitReferences(
        current_image_id, ordered_reference_image_ids);
    output.success = result.status == OnlineMapperFrameStatus::SUCCESS;
    output.detail = result.detail;
    for (const OnlineMapperPairAudit& pair : result.pairs) {
      OnlineI3dgsPairMatch item;
      item.reference_image_id = pair.reference_image_id;
      item.raw_match_count = pair.raw_match_count;
      item.verified_inlier_count = pair.verified_inlier_count;
      output.pairs.push_back(item);
    }
    return output;
  }

  OnlineI3dgsLidarIngestResult IngestScan(
      const OnlineI3dgsFrameInput& frame,
      const size_t max_visible_frame_index) override {
    OnlineI3dgsLidarIngestResult result;
    if (max_visible_frame_index != frame.frame_index) {
      result.detail = "LiDAR visibility limit differs from current frame";
      return result;
    }
    lidar::ScanSource source;
    source.scan_index = frame.frame_index;
    source.pcd_path = frame.scan_path;
    source.input_frame = lidar::LidarCoordinateFrame::FASTLIO_WORLD;
    source.expected_size_bytes = frame.scan_size_bytes;
    source.expected_sha256 = frame.scan_sha256;
    lidar::AppendScanAudit audit;
    std::string error;
    result.success = lidar_map_.AppendScan(source, &audit, &error);
    lidar_ingest_profile_ = LidarAppendAuditJson(audit);
    result.detail = error;
    result.opened_scan_index = audit.opened_scan_index;
    result.map_version_before = audit.version_before;
    result.map_version_after = audit.version_after;
    result.max_scan_index = audit.max_scan_index_after;
    result.point_transform_count_min = audit.point_transform_count_min;
    result.point_transform_count_max = audit.point_transform_count_max;
    result.normal_transform_count_min = audit.source_normal_transform_count_min;
    result.normal_transform_count_max = audit.source_normal_transform_count_max;
    result.snapshot_sha256 = audit.snapshot_sha256_after;
    result.geometry_sha256 = audit.geometry_sha256_after;
    return result;
  }

  OnlineDualPnPDecisionResult ProbeTwoSides(
      const OnlineI3dgsPnPRequest& request) override {
    std::map<OnlineDualClusterId, std::vector<image_t>> cluster_images;
    for (const OnlineDualReferenceAssignment& assignment :
         request.classification.assignments) {
      cluster_images[assignment.cluster_id].push_back(
          assignment.reference.image_id);
    }
    std::vector<OnlineDualPnPCorrespondenceSet> sets;
    for (const auto& cluster : cluster_images) {
      if (cluster.first == OnlineDualClusterId::NONE) continue;
      OnlineDualPnPCorrespondenceSet set;
      set.cluster_id = cluster.first;
      std::set<std::pair<point2D_t, point3D_t>> seen;
      for (const image_t reference_image_id : cluster.second) {
        const FeatureMatches matches = database_cache_.CorrespondenceGraph()
                                           .FindCorrespondencesBetweenImages(
                                               request.current_image_id,
                                               reference_image_id);
        const Image& current = reconstruction_.Image(request.current_image_id);
        const Image& reference = reconstruction_.Image(reference_image_id);
        for (const FeatureMatch& match : matches) {
          if (match.point2D_idx1 >= current.NumPoints2D() ||
              match.point2D_idx2 >= reference.NumPoints2D()) {
            continue;
          }
          const Point2D& reference_point = reference.Point2D(match.point2D_idx2);
          if (!reference_point.HasPoint3D()) continue;
          const point3D_t point3D_id = reference_point.Point3DId();
          if (!reconstruction_.ExistsPoint3D(point3D_id) ||
              !seen.emplace(match.point2D_idx1, point3D_id).second) {
            continue;
          }
          set.points2D.push_back(current.Point2D(match.point2D_idx1).XY());
          set.points3D.push_back(reconstruction_.Point3D(point3D_id).XYZ());
        }
      }
      sets.push_back(std::move(set));
    }
    OnlineDualPnPProbeAdapter adapter(
        [this](const OnlineDualPnPProbeRequest& probe) {
          OnlineDualPnPSolverOutput output;
          Camera camera = reconstruction_.Camera(1);
          AbsolutePoseEstimationOptions options;
          options.estimate_focal_length = false;
          options.num_threads = 8;
          options.ransac_options.max_error = 12.0;
          options.ransac_options.min_inlier_ratio = 0.25;
          Eigen::Vector4d qvec(1.0, 0.0, 0.0, 0.0);
          Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
          size_t num_inliers = 0;
          std::vector<char> inlier_mask;
          output.success = EstimateAbsolutePose(options,
                                                probe.points2D,
                                                probe.points3D,
                                                &qvec,
                                                &tvec,
                                                &camera,
                                                &num_inliers,
                                                &inlier_mask);
          output.probe_T_cw = OnlineDualSE3(qvec, tvec);
          output.inlier_mask.reserve(inlier_mask.size());
          for (const char inlier : inlier_mask) {
            output.inlier_mask.push_back(inlier != 0 ? 1 : 0);
          }
          if (!output.success) output.detail = "EstimateAbsolutePose failed";
          return output;
        });
    return RunOnlineDualTwoSidePnPProbe(request.classification, sets, adapter);
  }

  const KnownPoseRegistry& KnownPoses() const override { return registry_; }
  const ActiveCovisibilityGraph& ActiveGraph() const override {
    return active_graph_;
  }

  OnlineMapperCanonicalVersion CanonicalVersion() const override {
    const OnlineMapperTransactionResult result =
        CaptureOnlineMapperCanonicalVersion(&reconstruction_, &registry_,
                                            &active_graph_, &owners_);
    return result.IsSuccess() ? result.canonical_version
                              : OnlineMapperCanonicalVersion();
  }

  OnlineI3dgsKnownPoseRegistrationResult RegisterKnownPose(
      const OnlineI3dgsFrameInput& frame) override {
    OnlineI3dgsKnownPoseRegistrationResult output;
    if (!reconstruction_.ExistsImage(frame.image_id) ||
        reconstruction_.Image(frame.image_id).IsRegistered()) {
      output.detail = "frontend image is missing or already registered";
      return output;
    }
    Image& image = reconstruction_.Image(frame.image_id);
    image.SetQvec(frame.fastlio_T_cw.qvec);
    image.SetTvec(frame.fastlio_T_cw.tvec);
    reconstruction_.RegisterImage(frame.image_id);
    const KnownPoseRegistryResult result = registry_.AddKnownPose(
        frame.image_id, frame.frame_index, frame.fastlio_T_cw);
    output.success = result.IsSuccess();
    output.detail = result.detail;
    output.registration_sequence = result.registration_sequence;
    return output;
  }

  OnlineI3dgsCatchupFailureQueryResult GetCatchupFailure(
      const image_t image_id) const override {
    OnlineI3dgsCatchupFailureQueryResult result;
    const KnownPoseRecordQueryResult record = registry_.GetByImageId(image_id);
    if (!record.IsSuccess()) {
      result.detail = record.detail;
      return result;
    }
    result.success = true;
    const auto evidence = catchup_failures_.find(image_id);
    if (evidence != catchup_failures_.end()) {
      result.has_evidence = true;
      result.evidence = evidence->second;
    }
    return result;
  }

  OnlineI3dgsDependencyResult RecordCatchupFailure(
      const image_t image_id,
      const OnlineI3dgsCatchupFailureEvidence& evidence) override {
    const KnownPoseRecordQueryResult record = registry_.GetByImageId(image_id);
    if (!record.IsSuccess() ||
        record.record.visual_state != KnownPoseVisualState::POSE_ONLY) {
      return {false, "CATCHUP evidence requires a POSE_ONLY registry image"};
    }
    catchup_failures_[image_id] = evidence;
    return {true, ""};
  }

  uint64_t GlobalBaCallCount() const override { return 0; }

  bool Merge(Reconstruction* candidate,
             const OnlineLocalBaRequest&,
             uint64_t* merged_observation_count,
             std::string* error) override {
    return RunTriangulatorPostprocess(candidate, "merge", merged_observation_count,
                                      error);
  }

  bool Complete(Reconstruction* candidate,
                const OnlineLocalBaRequest&,
                uint64_t* completed_observation_count,
                std::string* error) override {
    return RunTriangulatorPostprocess(candidate, "complete",
                                      completed_observation_count, error);
  }

  bool Filter(Reconstruction* candidate,
              const OnlineLocalBaRequest&,
              uint64_t* filtered_observation_count,
              std::string* error) override {
    if (candidate == nullptr || filtered_observation_count == nullptr ||
        error == nullptr) {
      return false;
    }
    try {
      postprocess_full_point_count_ = candidate->NumPoints3D();
      const OnlineLocalBaTouchedFilterResult result =
          FilterOnlineLocalBaTouchedPoints(candidate, postprocess_point3D_ids_,
                                           4.0, 1.5);
      postprocess_filter_point_count_ = result.existing_input_point_count;
      *filtered_observation_count = result.filtered_observation_count;
      *error = result.detail;
      return result.success;
    } catch (const std::exception& exception) {
      *error = exception.what();
      return false;
    }
  }

  OnlineI3dgsBaExecution Execute(
      const OnlineI3dgsBaRequest& request) override {
    const auto preparation_start = std::chrono::steady_clock::now();
    postprocess_point3D_ids_ = pending_filter_point3D_ids_;
    postprocess_filter_point_count_ = 0;
    postprocess_full_point_count_ = 0;
    if (CanonicalVersion() != request.expected_canonical_version ||
        registry_.Version() != request.expected_registry_version ||
        active_graph_.Version() != request.expected_graph_version) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             "BA preparation canonical version changed");
    }
    uint64_t trigger_observation_upper_bound = 0;
    std::string bound_error;
    const bool allow_preparation_shortcut =
        request.mode != OnlineI3dgsMapperMode::DUAL &&
        request.mode != OnlineI3dgsMapperMode::CATCHUP;
    if (allow_preparation_shortcut &&
        !CountOnlineLocalBaPotentialTriggerObservations(
            reconstruction_, database_cache_.CorrespondenceGraph(),
            request.trigger_image_id, &trigger_observation_upper_bound,
            &bound_error)) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             bound_error);
    }
    if (allow_preparation_shortcut &&
        trigger_observation_upper_bound <
            kOnlineLocalBaMinimumTriggerLidarResidualCount) {
      OnlineLocalBaPassResult pass;
      pass.mode = LocalMode(request.mode);
      pass.attempt_id = request.attempt_id;
      pass.pass_index = 1;
      pass.trigger_image_id = request.trigger_image_id;
      pass.failure_reason = OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM;
      pass.failure_detail =
          "trigger has fewer than 50 potentially triangulatable observations";
      pass.timing.total_milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - preparation_start).count();
      Json::Value output = ExecutionResultJson(pass);
      output["transaction_clone_milliseconds"] = 0.0;
      output["triangulation_milliseconds"] = 0.0;
      output["trigger_observation_upper_bound"] =
          Json::UInt64(trigger_observation_upper_bound);
      output["preparation_skipped"] = true;
      ba_execution_json_.push_back(CompactJson(output));
      return ConvertSingleExecution(request, pass, nullptr);
    }
    std::unique_ptr<ProductionBaCandidate> candidate(
        new ProductionBaCandidate());
    const OnlineMapperTransactionResult begin = BeginOnlineMapperTransaction(
        &reconstruction_, &registry_, &active_graph_, &owners_,
        &candidate->payload);
    if (!begin.IsSuccess() ||
        begin.canonical_version != request.expected_canonical_version ||
        begin.canonical_version.known_pose_registry !=
            request.expected_registry_version ||
        begin.canonical_version.active_covisibility_graph !=
            request.expected_graph_version) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             begin.detail.empty() ? "cannot begin BA transaction"
                                                  : begin.detail);
    }
    Reconstruction* reconstruction =
        candidate->payload.reconstruction.MutableCandidate();
    const auto clone_end = std::chrono::steady_clock::now();
    if (reconstruction == nullptr) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             "BA transaction has no candidate");
    }
    std::string error;
    if (!TriangulateFrozenWindow(request.frozen_image_ids, reconstruction,
                                 &candidate->directly_optimized_points,
                                 &postprocess_point3D_ids_,
                                 &error)) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             error);
    }
    if (candidate->directly_optimized_points.empty()) {
      return FailedExecution(
          request,
          request.mode == OnlineI3dgsMapperMode::DUAL
              ? OnlineI3dgsBaDisposition::FATAL_FAILURE
              : OnlineI3dgsBaDisposition::RECOVERABLE_FAILURE,
          OnlineI3dgsBaFailureReason::PREPARE_FAILED,
          "frozen window produced no triangulated visual points");
    }
    postprocess_point3D_ids_.insert(candidate->directly_optimized_points.begin(),
                                    candidate->directly_optimized_points.end());
    for (const image_t image_id : pending_filter_image_ids_) {
      if (!reconstruction->ExistsImage(image_id)) continue;
      for (const Point2D& point2D : reconstruction->Image(image_id).Points2D()) {
        if (point2D.HasPoint3D()) {
          postprocess_point3D_ids_.insert(point2D.Point3DId());
        }
      }
    }
    const auto triangulation_end = std::chrono::steady_clock::now();
    const auto record_pass = [&](const OnlineLocalBaPassResult& pass) {
      Json::Value output = ExecutionResultJson(pass);
      output["transaction_clone_milliseconds"] = pass.pass_index == 1
          ? std::chrono::duration<double, std::milli>(
                clone_end - preparation_start).count()
          : 0.0;
      output["triangulation_milliseconds"] = pass.pass_index == 1
          ? std::chrono::duration<double, std::milli>(
                triangulation_end - clone_end).count()
          : 0.0;
      output["postprocess_filter_point_count"] =
          Json::UInt64(pass.postprocess.filter_call_count != 0
                           ? postprocess_filter_point_count_ : 0);
      output["postprocess_full_point_count"] =
          Json::UInt64(pass.postprocess.filter_call_count != 0
                           ? postprocess_full_point_count_ : 0);
      ba_execution_json_.push_back(CompactJson(output));
    };

    if (request.mode == OnlineI3dgsMapperMode::DUAL &&
        !ApplyDualPrealignment(request, candidate.get(), &error)) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             error);
    }

    OnlineLocalBaRequest pass1;
    if (!BuildLocalBaRequest(request, 1, reconstruction,
                             candidate->directly_optimized_points, &pass1,
                             &error)) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             error);
    }
    OnlineLocalBaExecutorDependencies dependencies;
    dependencies.postprocess = this;
    if (request.required_pass_count == 1) {
      const OnlineLocalBaPassResult pass = ExecuteOnlineLocalBa(
          ba_options_, pass1, &candidate->payload.reconstruction, dependencies);
      record_pass(pass);
      return ConvertSingleExecution(request, pass, std::move(candidate));
    }
    if (request.required_pass_count != 2 ||
        request.mode != OnlineI3dgsMapperMode::DUAL) {
      return FailedExecution(request,
                             OnlineI3dgsBaDisposition::FATAL_FAILURE,
                             OnlineI3dgsBaFailureReason::PREPARE_FAILED,
                             "unsupported online BA pass count");
    }
    OnlineLocalBaRequest pass2 = pass1;
    pass1.mode = OnlineLocalBaMode::DUAL_PASS1;
    pass1.allow_postprocess = false;
    pass2.mode = OnlineLocalBaMode::DUAL_PASS2;
    pass2.pass_index = 2;
    pass2.selection_revision = pass1.selection_revision + 1;
    pass2.allow_postprocess = true;
    const OnlineDualLocalBaResult dual = ExecuteOnlineDualLocalBa(
        ba_options_, pass1, pass2, &candidate->payload.reconstruction,
        dependencies);
    record_pass(dual.pass1);
    record_pass(dual.pass2);
    return ConvertDualExecution(request, dual, std::move(candidate));
  }

  OnlineI3dgsCommitResult Commit(
      const OnlineI3dgsCommitRequest& request,
      OnlineI3dgsBaExecution* execution) override {
    OnlineI3dgsCommitResult output;
    const OnlineMapperTransactionResult captured_before =
        CaptureOnlineMapperCanonicalVersion(&reconstruction_, &registry_,
                                            &active_graph_, &owners_);
    output.canonical_version_before = captured_before.canonical_version;
    output.canonical_version_after = output.canonical_version_before;
    output.registry_version_before = registry_.Version();
    output.graph_version_before = active_graph_.Version();
    output.registry_version_after = output.registry_version_before;
    output.graph_version_after = output.graph_version_before;
    if (execution == nullptr || execution->candidate == nullptr) {
      output.detail = "BA commit is missing its candidate";
      return output;
    }
    ProductionBaCandidate* candidate =
        dynamic_cast<ProductionBaCandidate*>(execution->candidate.get());
    if (!captured_before.IsSuccess() || candidate == nullptr ||
        request.expected_canonical_version !=
            output.canonical_version_before ||
        request.expected_registry_version != registry_.Version() ||
        request.expected_graph_version != active_graph_.Version()) {
      output.detail = "BA commit candidate or canonical version is invalid";
      return output;
    }
    Reconstruction* candidate_reconstruction =
        candidate->payload.reconstruction.MutableCandidate();
    if (candidate_reconstruction == nullptr) {
      output.detail = "BA commit candidate reconstruction disappeared";
      return output;
    }

    const std::set<image_t> promoted(request.promoted_image_ids.begin(),
                                     request.promoted_image_ids.end());
    for (const image_t image_id : registry_.GetRegisteredImageIds().image_ids) {
      const KnownPoseRecordQueryResult record = registry_.GetByImageId(image_id);
      if (!record.IsSuccess() || !candidate_reconstruction->ExistsImage(image_id)) {
        output.detail = "BA commit cannot reconcile registered images";
        return output;
      }
      if (record.record.visual_state == KnownPoseVisualState::POSE_ONLY &&
          promoted.count(image_id) == 0) {
        const Image& image = candidate_reconstruction->Image(image_id);
        if (image.Qvec() != record.record.fastlio_T_cw.qvec ||
            image.Tvec() != record.record.fastlio_T_cw.tvec) {
          candidate->post_commit_dirty_image_ids.insert(image_id);
        }
        candidate_reconstruction->Image(image_id).SetQvec(
            record.record.fastlio_T_cw.qvec);
        candidate_reconstruction->Image(image_id).SetTvec(
            record.record.fastlio_T_cw.tvec);
      }
    }

    candidate->payload.known_pose_registry.promotions.clear();
    candidate->payload.known_pose_registry.latest_pose_updates.clear();
    for (const OnlineI3dgsImagePoseUpdate& update : request.pose_updates) {
      if (promoted.count(update.image_id) != 0) {
        candidate->payload.known_pose_registry.promotions.emplace_back(
            update.image_id, update.latest_T_cw);
      } else {
        const KnownPoseRecordQueryResult record =
            registry_.GetByImageId(update.image_id);
        if (!record.IsSuccess()) {
          output.detail = "BA commit pose update is absent from registry";
          return output;
        }
        if (record.record.visual_state == KnownPoseVisualState::VISUAL_ACTIVE) {
          candidate->payload.known_pose_registry.latest_pose_updates.emplace_back(
              update.image_id, update.latest_T_cw);
        }
      }
    }
    candidate->payload.active_covisibility_graph.nodes.clear();
    for (const image_t image_id : request.promoted_image_ids) {
      const KnownPoseRecordQueryResult record = registry_.GetByImageId(image_id);
      if (!record.IsSuccess()) {
        output.detail = "BA commit promoted image is absent from registry";
        return output;
      }
      candidate->payload.active_covisibility_graph.nodes.emplace_back(
          image_id, record.record.registration_sequence,
          ActiveCovisibilityNodeState::VISUAL_ACTIVE);
    }
    candidate->payload.active_covisibility_graph.ordinary_edges =
        request.ordinary_edges;
    candidate->payload.active_covisibility_graph.loop_edges = request.loop_edges;

    OnlineMapperPreparedTransaction prepared;
    const OnlineMapperTransactionResult preparation =
        PrepareOnlineMapperTransaction(&reconstruction_, &registry_,
                                       &active_graph_, &owners_,
                                       &candidate->payload, &prepared);
    if (!preparation.IsSuccess()) {
      output.detail = preparation.detail;
      return output;
    }
    const OnlineMapperTransactionResult validation =
        ValidatePreparedOnlineMapperTransaction(prepared);
    if (!validation.IsSuccess()) {
      output.detail = validation.detail;
      return output;
    }
    CommitPreparedOnlineMapperTransaction(&prepared);
    pending_filter_image_ids_ =
        std::move(candidate->post_commit_dirty_image_ids);
    pending_filter_point3D_ids_ =
        std::move(candidate->post_commit_dirty_point3D_ids);
    execution->candidate.reset();
    const OnlineMapperTransactionResult captured_after =
        CaptureOnlineMapperCanonicalVersion(&reconstruction_, &registry_,
                                            &active_graph_, &owners_);
    if (!captured_after.IsSuccess()) {
      output.detail = captured_after.detail;
      return output;
    }
    output.success = true;
    output.atomic_publication = true;
    output.canonical_version_after = captured_after.canonical_version;
    output.registry_version_after = registry_.Version();
    output.graph_version_after = active_graph_.Version();
    return output;
  }

 private:
  static OnlineLocalBaMode LocalMode(const OnlineI3dgsMapperMode mode) {
    switch (mode) {
      case OnlineI3dgsMapperMode::BOOTSTRAP:
        return OnlineLocalBaMode::BOOTSTRAP;
      case OnlineI3dgsMapperMode::CATCHUP:
        return OnlineLocalBaMode::CATCHUP;
      case OnlineI3dgsMapperMode::SINGLE:
      case OnlineI3dgsMapperMode::DUAL:
      case OnlineI3dgsMapperMode::END_OF_SEQUENCE_FLUSH:
      case OnlineI3dgsMapperMode::NONE:
        return OnlineLocalBaMode::SINGLE;
    }
    return OnlineLocalBaMode::SINGLE;
  }

  bool RunTriangulatorPostprocess(Reconstruction* candidate,
                                  const std::string& operation,
                                  uint64_t* observation_count,
                                  std::string* error) {
    if (candidate == nullptr || observation_count == nullptr || error == nullptr) {
      return false;
    }
    try {
      IncrementalTriangulator triangulator(
          &database_cache_.CorrespondenceGraph(), candidate);
      IncrementalTriangulator::Options options;
      options.ignore_two_view_tracks = false;
      if (operation == "merge") {
        *observation_count = triangulator.MergeAllTracks(options);
      } else {
        *observation_count = triangulator.CompleteAllTracks(options);
      }
      const auto& modified = triangulator.GetModifiedPoints3D();
      postprocess_point3D_ids_.insert(modified.begin(), modified.end());
      return true;
    } catch (const std::exception& exception) {
      *error = exception.what();
      return false;
    }
  }

  bool TriangulateFrozenWindow(const std::vector<image_t>& image_ids,
                               Reconstruction* candidate,
                               std::set<point3D_t>* selected_points,
                               std::unordered_set<point3D_t>* modified_points,
                               std::string* error) {
    try {
      IncrementalTriangulator triangulator(
          &database_cache_.CorrespondenceGraph(), candidate);
      IncrementalTriangulator::Options options;
      options.ignore_two_view_tracks = false;
      for (const image_t image_id : image_ids) {
        if (!candidate->ExistsImage(image_id) ||
            !candidate->Image(image_id).IsRegistered()) {
          *error = "frozen BA image is missing or unregistered";
          return false;
        }
        triangulator.TriangulateImage(options, image_id);
      }
      const auto& modified = triangulator.GetModifiedPoints3D();
      modified_points->insert(modified.begin(), modified.end());
      selected_points->clear();
      for (const image_t image_id : image_ids) {
        const Image& image = candidate->Image(image_id);
        for (const Point2D& point : image.Points2D()) {
          if (point.HasPoint3D()) selected_points->insert(point.Point3DId());
        }
      }
      return true;
    } catch (const std::exception& exception) {
      *error = std::string("frozen-window triangulation failed: ") +
               exception.what();
      return false;
    }
  }

  bool BuildLocalBaRequest(const OnlineI3dgsBaRequest& source,
                           const uint32_t pass_index,
                           Reconstruction* candidate,
                           const std::set<point3D_t>& selected_points,
                           OnlineLocalBaRequest* output,
                           std::string* error) {
    output->mode = LocalMode(source.mode);
    output->attempt_id = source.attempt_id;
    output->pass_index = pass_index;
    output->trigger_image_id = source.trigger_image_id;
    output->ordered_frozen_image_ids = source.frozen_image_ids;
    output->point3D_ids.assign(selected_points.begin(), selected_points.end());
    output->expected_candidate_version = candidate->CanonicalVersion();
    output->owner_epoch = candidate->StructureOwnerEpoch();
    output->topology_revision = candidate->StructureRevision();
    output->selection_revision = source.attempt_id * 2 + pass_index;
    output->expected_map_version = source.lidar_map_version;
    output->expected_max_scan_index = source.max_scan_index;
    output->expected_snapshot_sha256 = source.lidar_snapshot_sha256;
    output->expected_geometry_sha256 = source.lidar_geometry_sha256;
    output->map_snapshot = lidar_map_.GetSnapshot();
    output->association_options = ProductionAssociationOptions();
    output->requested_backend = source.requested_backend;
    output->requested_problem_source = gpu_ba::CudaProblemSource::kNativeGraph;
    output->fallback_allowed = source.fallback_allowed;
    output->all_camera_poses_variable = source.all_camera_poses_variable;
    output->constant_camera_pose_ids.clear();
    output->allow_postprocess = true;
    output->require_trigger_residual_increase =
        source.mode == OnlineI3dgsMapperMode::CATCHUP &&
        source.has_previous_catchup_failure &&
        source.active_edge_evidence_version <=
            source.previous_catchup_failure.active_edge_evidence_version;
    output->previous_trigger_lidar_constraint_count =
        source.previous_catchup_failure.trigger_submitted_lidar_constraint_count;
    return CaptureOnlineLocalBaFixedIntrinsics(
        *candidate, source.frozen_image_ids, &output->fixed_intrinsics, error);
  }

  OnlineI3dgsBaExecution FailedExecution(
      const OnlineI3dgsBaRequest& request,
      const OnlineI3dgsBaDisposition disposition,
      const OnlineI3dgsBaFailureReason reason,
      const std::string& detail) const {
    OnlineI3dgsBaExecution output;
    output.disposition = disposition;
    output.failure_reason = reason;
    output.detail = detail;
    OnlineI3dgsBaPassAudit pass;
    pass.pass_index = 1;
    pass.requested_backend = request.requested_backend;
    pass.termination = "NOT_RUN";
    pass.detail = detail;
    output.passes.push_back(pass);
    return output;
  }

  OnlineI3dgsBaPassAudit ConvertPass(
      const OnlineLocalBaPassResult& source) const {
    OnlineI3dgsBaPassAudit output;
    output.pass_index = source.pass_index;
    output.solve_native_invoked = source.solver_called;
    output.solve_native_success = source.execution.success;
    output.finite_costs = source.finite_costs;
    output.finite_output = source.success;
    output.submitted_lidar_constraint_count =
        source.submitted_lidar_constraint_count;
    if (source.execution.residuals >= source.execution.residual_blocks) {
      output.submitted_visual_residual_count =
          source.execution.residuals - source.execution.residual_blocks;
    }
    output.trigger_submitted_lidar_constraint_count =
        source.trigger_submitted_lidar_constraint_count;
    output.trigger_preflight_checked = source.trigger_gate.checked;
    output.trigger_preflight_lidar_constraint_count =
        source.trigger_gate.actual_count;
    output.constant_camera_pose_count = 0;
    output.postprocess_count =
        source.postprocess.merge_call_count == 1 &&
                source.postprocess.complete_call_count == 1 &&
                source.postprocess.filter_call_count == 1
            ? 1
            : 0;
    output.requested_backend = source.execution.requested_backend.empty()
                                   ? "custom_cuda"
                                   : source.execution.requested_backend;
    output.executed_backend = source.execution.executed_backend;
    output.requested_problem_source =
        source.execution.problem_source_requested.empty()
            ? "native_graph"
            : source.execution.problem_source_requested;
    output.executed_problem_source =
        source.execution.problem_source_effective;
    output.fallback_used = source.execution.fallback_used;
    output.termination = source.termination_converged
                             ? "CONVERGENCE"
                             : (source.solver_called &&
                                        source.solver_summary.termination_type ==
                                            ceres::NO_CONVERGENCE
                                    ? "NO_CONVERGENCE"
                                    : source.execution.termination);
    output.detail = source.failure_detail;
    return output;
  }

  OnlineI3dgsBaFailureReason ConvertFailureReason(
      const OnlineLocalBaFailureReason reason) const {
    if (reason == OnlineLocalBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED) {
      return OnlineI3dgsBaFailureReason::RETRY_EVIDENCE_NOT_INCREASED;
    }
    if (reason == OnlineLocalBaFailureReason::TRIGGER_BELOW_MINIMUM) {
      return OnlineI3dgsBaFailureReason::
          TRIGGER_SUBMITTED_LIDAR_CONSTRAINTS_BELOW_MINIMUM;
    }
    if (reason == OnlineLocalBaFailureReason::POSTPROCESS_MERGE_FAILED ||
        reason == OnlineLocalBaFailureReason::POSTPROCESS_COMPLETE_FAILED ||
        reason == OnlineLocalBaFailureReason::POSTPROCESS_FILTER_FAILED) {
      return OnlineI3dgsBaFailureReason::POSTPROCESS_FAILED;
    }
    if (reason == OnlineLocalBaFailureReason::INVALID_REQUEST ||
        reason == OnlineLocalBaFailureReason::CANDIDATE_UNAVAILABLE ||
        reason == OnlineLocalBaFailureReason::FIXED_INTRINSICS_MISMATCH ||
        reason == OnlineLocalBaFailureReason::INTENT_BUILD_FAILED ||
        reason == OnlineLocalBaFailureReason::INTENT_AUDIT_FAILED) {
      return OnlineI3dgsBaFailureReason::PREPARE_FAILED;
    }
    return OnlineI3dgsBaFailureReason::SOLVE_NATIVE_FAILED;
  }

  bool PopulateSuccessfulExecution(const OnlineI3dgsBaRequest& request,
                                   const OnlineLocalBaPassResult& last_pass,
                                   ProductionBaCandidate* candidate,
                                   OnlineI3dgsBaExecution* output,
                                   std::string* error) {
    Reconstruction* reconstruction =
        candidate->payload.reconstruction.MutableCandidate();
    if (reconstruction == nullptr) {
      *error = "successful BA result lost its transaction candidate";
      return false;
    }
    for (const image_t image_id : registry_.GetRegisteredImageIds().image_ids) {
      if (!reconstruction->ExistsImage(image_id)) continue;
      const Image& image = reconstruction->Image(image_id);
      output->pose_updates.push_back(
          {image_id, KnownPoseSE3(image.Qvec(), image.Tvec())});
    }
    for (const image_t image_id : request.frozen_image_ids) {
      const auto count =
          last_pass.per_image_submitted_lidar_constraint_count.find(
              image_id);
      if (count ==
          last_pass.per_image_submitted_lidar_constraint_count.end()) {
        *error = "successful BA result omitted a submitted LiDAR count";
        return false;
      }
      output->per_image_submitted_lidar_constraint_counts.push_back(
          {image_id, count->second});
    }
    output->total_postprocess_count =
        last_pass.postprocess.merge_call_count == 1 &&
                last_pass.postprocess.complete_call_count == 1 &&
                last_pass.postprocess.filter_call_count == 1
            ? 1
            : 0;
    return true;
  }

  OnlineI3dgsBaExecution ConvertSingleExecution(
      const OnlineI3dgsBaRequest& request,
      const OnlineLocalBaPassResult& pass,
      std::unique_ptr<ProductionBaCandidate> candidate) {
    OnlineI3dgsBaExecution output;
    output.passes.push_back(ConvertPass(pass));
    if (!pass.success) {
      output.disposition = pass.fatal
                               ? OnlineI3dgsBaDisposition::FATAL_FAILURE
                               : OnlineI3dgsBaDisposition::RECOVERABLE_FAILURE;
      output.failure_reason = ConvertFailureReason(pass.failure_reason);
      output.detail = pass.failure_detail;
      return output;
    }
    output.disposition = OnlineI3dgsBaDisposition::SUCCESS;
    output.failure_reason = OnlineI3dgsBaFailureReason::NONE;
    std::string error;
    if (!PopulateSuccessfulExecution(
            request, pass, candidate.get(), &output, &error)) {
      output.disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      output.failure_reason = OnlineI3dgsBaFailureReason::INVALID_RESULT;
      output.detail = error;
      return output;
    }
    output.candidate = std::move(candidate);
    return output;
  }

  OnlineI3dgsBaExecution ConvertDualExecution(
      const OnlineI3dgsBaRequest& request,
      const OnlineDualLocalBaResult& dual,
      std::unique_ptr<ProductionBaCandidate> candidate) {
    OnlineI3dgsBaExecution output;
    output.passes.push_back(ConvertPass(dual.pass1));
    output.passes.push_back(ConvertPass(dual.pass2));
    if (!dual.success) {
      output.disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      output.failure_reason = ConvertFailureReason(
          dual.pass2.failure_reason == OnlineLocalBaFailureReason::NONE
              ? dual.pass1.failure_reason
              : dual.pass2.failure_reason);
      output.detail = dual.failure_detail;
      return output;
    }
    std::string error;
    if (!ApplyDualPropagation(request, candidate.get(), &error)) {
      output.disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      output.failure_reason = OnlineI3dgsBaFailureReason::PROPAGATION_FAILED;
      output.detail = error;
      return output;
    }
    output.disposition = OnlineI3dgsBaDisposition::SUCCESS;
    output.failure_reason = OnlineI3dgsBaFailureReason::NONE;
    output.propagation_applied = true;
    if (!PopulateSuccessfulExecution(
            request, dual.pass2, candidate.get(), &output, &error)) {
      output.disposition = OnlineI3dgsBaDisposition::FATAL_FAILURE;
      output.failure_reason = OnlineI3dgsBaFailureReason::INVALID_RESULT;
      output.detail = error;
      return output;
    }
    output.candidate = std::move(candidate);
    return output;
  }

  bool ApplyDualPrealignment(const OnlineI3dgsBaRequest& request,
                             ProductionBaCandidate* candidate,
                             std::string* error) {
    Json::Value audit(Json::objectValue);
    audit["schema"] = "online_i3dgs_dual_prealignment_v1";
    audit["record_type"] = "DUAL_PREALIGNMENT";
    audit["attempt_id"] = Json::UInt64(request.attempt_id);
    audit["frame_index"] = Json::UInt64(request.frame_index);
    audit["delta_init"] = OnlineDualSe3Json(request.dual_pnp.delta_init);
    audit["images"] = Json::Value(Json::arrayValue);
    const auto fail = [&](const std::string& detail) {
      audit["success"] = false;
      audit["detail"] = detail;
      ba_execution_json_.push_back(CompactJson(audit));
      *error = detail;
      return false;
    };
    if (!request.has_dual_context ||
        !IsValidOnlineDualSE3(request.dual_pnp.delta_init) ||
        request.dual_window.frozen_graph_version != active_graph_.Version()) {
      return fail("DUAL prealignment has invalid frozen context or delta_init");
    }
    Reconstruction* reconstruction =
        candidate->payload.reconstruction.MutableCandidate();
    if (reconstruction == nullptr) {
      return fail("DUAL prealignment candidate is unavailable");
    }
    const std::shared_ptr<const lidar::LidarMapSnapshot> lidar_before =
        lidar_map_.GetSnapshot();
    if (lidar_before == nullptr ||
        lidar_before->Version() != request.lidar_map_version ||
        lidar_before->MaxScanIndex() != request.max_scan_index ||
        lidar_before->SnapshotSha256() != request.lidar_snapshot_sha256 ||
        lidar_before->GeometrySha256() != request.lidar_geometry_sha256) {
      return fail("DUAL prealignment LiDAR snapshot identity is invalid");
    }

    const std::set<image_t> frozen(request.frozen_image_ids.begin(),
                                   request.frozen_image_ids.end());
    if (frozen.size() != request.frozen_image_ids.size() ||
        request.dual_window.current_image_id != request.trigger_image_id) {
      return fail("DUAL prealignment frozen window identity is invalid");
    }
    std::set<image_t> observed;
    size_t current_count = 0;
    candidate->dual_before_T_wc.clear();
    for (const OnlineDualWindowImage& window_image :
         request.dual_window.images) {
      if (frozen.count(window_image.image_id) == 0 ||
          !observed.insert(window_image.image_id).second ||
          !reconstruction->ExistsImage(window_image.image_id) ||
          window_image.temporal_side == OnlineDualTemporalSide::NONE) {
        return fail("DUAL prealignment window is inconsistent with frozen images");
      }
      Image& image = reconstruction->Image(window_image.image_id);
      const OnlineDualSE3 before_T_wc =
          InverseOnlineDualSE3(ImageTcw(image));
      if (!IsValidOnlineDualSE3(before_T_wc)) {
        return fail("DUAL prealignment encountered an invalid before pose");
      }
      candidate->dual_before_T_wc.emplace(window_image.image_id, before_T_wc);
      if (window_image.is_current) {
        ++current_count;
        if (window_image.image_id != request.trigger_image_id) {
          return fail("DUAL prealignment current image identity is invalid");
        }
      }
    }
    if (observed != frozen || observed.count(request.trigger_image_id) == 0 ||
        current_count != 1) {
      return fail("DUAL prealignment did not cover the exact frozen window");
    }

    for (const OnlineDualWindowImage& window_image :
         request.dual_window.images) {
      Image& image = reconstruction->Image(window_image.image_id);
      const OnlineDualSE3& before_T_wc =
          candidate->dual_before_T_wc.at(window_image.image_id);
      OnlineDualSE3 after_T_wc = before_T_wc;
      const bool apply =
          window_image.temporal_side == OnlineDualTemporalSide::NEWER;
      if (apply) {
        after_T_wc = ComposeOnlineDualSE3(request.dual_pnp.delta_init,
                                          before_T_wc);
        if (!IsValidOnlineDualSE3(after_T_wc)) {
          return fail("DUAL delta_init produced an invalid prealigned pose");
        }
        const OnlineDualSE3 after_T_cw = InverseOnlineDualSE3(after_T_wc);
        image.SetQvec(after_T_cw.qvec);
        image.SetTvec(after_T_cw.tvec);
      }
      const OnlineDualSE3 candidate_T_wc =
          InverseOnlineDualSE3(ImageTcw(image));
      if (!IsValidOnlineDualSE3(candidate_T_wc)) {
        return fail("DUAL prealignment stored an invalid candidate pose");
      }
      Json::Value item(Json::objectValue);
      item["image_id"] = Json::UInt64(window_image.image_id);
      item["is_current"] = window_image.is_current;
      item["temporal_side"] = ToString(window_image.temporal_side);
      item["delta_init_applied"] = apply;
      item["before_T_wc"] = OnlineDualSe3Json(before_T_wc);
      item["after_T_wc"] = OnlineDualSe3Json(after_T_wc);
      audit["images"].append(std::move(item));
    }
    const std::shared_ptr<const lidar::LidarMapSnapshot> lidar_after =
        lidar_map_.GetSnapshot();
    if (lidar_after.get() != lidar_before.get() ||
        lidar_after->Version() != lidar_before->Version() ||
        lidar_after->SnapshotSha256() != lidar_before->SnapshotSha256() ||
        lidar_after->GeometrySha256() != lidar_before->GeometrySha256()) {
      return fail("DUAL prealignment modified the frozen LiDAR map");
    }
    audit["success"] = true;
    audit["detail"] = "NEWER T_wc poses left-multiplied by delta_init";
    audit["lidar_snapshot_sha256_before"] = lidar_before->SnapshotSha256();
    audit["lidar_snapshot_sha256_after"] = lidar_after->SnapshotSha256();
    audit["lidar_geometry_sha256_before"] = lidar_before->GeometrySha256();
    audit["lidar_geometry_sha256_after"] = lidar_after->GeometrySha256();
    ba_execution_json_.push_back(CompactJson(audit));
    return true;
  }

  bool ApplyDualPropagation(const OnlineI3dgsBaRequest& request,
                            ProductionBaCandidate* candidate,
                            std::string* error) {
    Reconstruction* reconstruction =
        candidate->payload.reconstruction.MutableCandidate();
    const OnlineDualBackboneResult backbone =
        SelectOnlineDualBackbone(active_graph_, request.dual_window);
    if (!backbone.IsDualReady()) {
      *error = backbone.detail;
      return false;
    }
    OnlineDualImageDeltaVector deltas;
    for (const OnlineDualWindowImage& window_image : request.dual_window.images) {
      const auto before = candidate->dual_before_T_wc.find(window_image.image_id);
      if (before == candidate->dual_before_T_wc.end() ||
          !reconstruction->ExistsImage(window_image.image_id)) {
        *error = "DUAL propagation window image is missing";
        return false;
      }
      const OnlineDualSE3 after_twc =
          InverseOnlineDualSE3(ImageTcw(reconstruction->Image(window_image.image_id)));
      OnlineDualImageDelta delta;
      delta.image_id = window_image.image_id;
      delta.delta = ComputeOnlineDualDelta(before->second, after_twc);
      deltas.push_back(delta);
    }
    const KnownPoseImageIdsResult pose_only = registry_.GetPoseOnlyImageIds();
    if (!pose_only.IsSuccess()) {
      *error = pose_only.detail;
      return false;
    }
    const OnlineDualPropagationPlan propagation = BuildOnlineDualPropagationPlan(
        active_graph_, request.dual_window, backbone, deltas,
        pose_only.image_ids);
    if (!propagation.IsSuccess()) {
      *error = propagation.detail;
      return false;
    }
    for (const OnlineDualImageCorrection& correction :
         propagation.image_corrections) {
      if (!correction.apply_correction ||
          !reconstruction->ExistsImage(correction.image_id)) {
        continue;
      }
      Image& image = reconstruction->Image(correction.image_id);
      const OnlineDualSE3 corrected_twc = ComposeOnlineDualSE3(
          correction.delta, InverseOnlineDualSE3(ImageTcw(image)));
      const OnlineDualSE3 corrected_tcw = InverseOnlineDualSE3(corrected_twc);
      image.SetQvec(corrected_tcw.qvec);
      image.SetTvec(corrected_tcw.tvec);
      candidate->post_commit_dirty_image_ids.insert(correction.image_id);
    }

    std::vector<OnlineDualPointState> point_states;
    for (const point3D_t point3D_id : reconstruction->Point3DIds()) {
      const Point3D& point = reconstruction->Point3D(point3D_id);
      OnlineDualPointState state;
      state.point3D_id = point3D_id;
      state.directly_optimized =
          candidate->directly_optimized_points.count(point3D_id) != 0;
      size_t best_sequence = std::numeric_limits<size_t>::max();
      for (const TrackElement& element : point.Track().Elements()) {
        const KnownPoseRecordQueryResult owner =
            registry_.GetByImageId(element.image_id);
        if (owner.IsSuccess() &&
            std::make_pair(owner.record.registration_sequence, element.image_id) <
                std::make_pair(best_sequence, state.owner_image_id)) {
          best_sequence = owner.record.registration_sequence;
          state.owner_image_id = element.image_id;
          state.owner_registration_sequence = best_sequence;
          state.owner_visual_state =
              owner.record.visual_state == KnownPoseVisualState::VISUAL_ACTIVE ||
                      element.image_id == request.trigger_image_id
                  ? ActiveCovisibilityNodeState::VISUAL_ACTIVE
                  : ActiveCovisibilityNodeState::POSE_ONLY;
        }
      }
      if (state.owner_image_id == kInvalidImageId) {
        *error = "DUAL point has no registered owner";
        return false;
      }
      point_states.push_back(state);
    }
    const OnlineDualPointCorrectionPlan point_plan =
        BuildOnlineDualPointCorrectionPlan(request.dual_window, deltas,
                                           propagation, point_states);
    if (!point_plan.IsSuccess()) {
      *error = point_plan.detail;
      return false;
    }
    for (const OnlineDualPointCorrection& correction : point_plan.corrections) {
      if (correction.apply_correction &&
          reconstruction->ExistsPoint3D(correction.point3D_id)) {
        Point3D& point = reconstruction->Point3D(correction.point3D_id);
        point.SetXYZ(ApplyOnlineDualSE3(correction.delta, point.XYZ()));
        candidate->post_commit_dirty_point3D_ids.insert(correction.point3D_id);
      }
    }
    return true;
  }

  const OnlineI3dgsReplayControllerConfig config_;
  const std::string database_path_;
  mutable Database database_;
  FeatureMatcherCache matcher_cache_;
  DatabaseCache database_cache_;
  Reconstruction reconstruction_;
  const SiftExtractionOptions extraction_options_;
  const SiftMatchingOptions matching_options_;
  const BundleAdjustmentOptions ba_options_;
  PersistentCudaSiftFeatureExtractor extractor_;
  SiftFeatureMatcher matcher_;
  OnlineMapperFrontend frontend_;
  lidar::IncrementalCausalLidarMap lidar_map_;
  KnownPoseRegistry registry_;
  ActiveCovisibilityGraph active_graph_;
  Point3DOwnerTable owners_;
  std::map<image_t, OnlineI3dgsCatchupFailureEvidence> catchup_failures_;
  std::unordered_set<image_t> pending_filter_image_ids_;
  std::unordered_set<point3D_t> pending_filter_point3D_ids_;
  std::unordered_set<point3D_t> postprocess_point3D_ids_;
  size_t postprocess_filter_point_count_ = 0;
  size_t postprocess_full_point_count_ = 0;
  std::unique_ptr<OnlineI3dgsMapper> mapper_;
  std::vector<std::string> ba_execution_json_;
  Json::Value lidar_ingest_profile_;
  mutable bool database_published_ = false;
};

#endif  // GPU_BA_CUDA_ENABLED

class SignalScope {
 public:
  SignalScope() {
    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = &SignalScope::Handle;
    sigemptyset(&action.sa_mask);
    if (::sigaction(SIGINT, &action, &old_int_) != 0 ||
        ::sigaction(SIGTERM, &action, &old_term_) != 0) {
      throw ReplayError("Failed to install signal handlers");
    }
    installed_ = true;
  }

  ~SignalScope() {
    if (installed_) {
      ::sigaction(SIGINT, &old_int_, nullptr);
      ::sigaction(SIGTERM, &old_term_, nullptr);
    }
  }

 private:
  static void Handle(int) { g_stop_requested = 1; }

  struct sigaction old_int_;
  struct sigaction old_term_;
  bool installed_ = false;
};

}  // namespace

OnlineI3dgsReplayResult RunOnlineI3dgsReplay(
    const OnlineI3dgsReplayOptions& options,
    const OnlineI3dgsReplayControllerFactory& controller_factory) {
  OnlineI3dgsReplayResult result;
  std::unique_ptr<ArtifactWriter> artifact;
  Json::Value summary;
  Json::Value timing(Json::objectValue);
  auto stage_start = std::chrono::steady_clock::now();
  const auto record_stage = [&](const char* name) {
    const auto now = std::chrono::steady_clock::now();
    timing[name] = timing.get(name, 0.0).asDouble() +
        std::chrono::duration<double, std::milli>(now - stage_start).count();
    stage_start = now;
  };
  size_t ba_record_count = 0;
  try {
    if (!controller_factory) throw ReplayError("Controller factory is not configured");
    if (options.ba_window_size < 2 ||
        options.ba_window_size > ActiveCovisibilityGraph::kMaxWindowSize) {
      throw ReplayError("--ba_window_size must be in [2, 20]");
    }
    struct stat output_stat;
    if (::lstat(options.output_path.c_str(), &output_stat) == 0 || errno != ENOENT) {
      throw ReplayError("Output path already exists; refusing to overwrite: " +
                        options.output_path);
    }
    const ReplayPlan plan = LoadReplayPlan(options);
    const std::string resolved_output = ProspectiveOutputPath(options.output_path);
    if (PathIsWithin(resolved_output, plan.sealed_artifact) ||
        PathIsWithin(resolved_output, plan.session_real_path) ||
        PathIsWithin(resolved_output, plan.i3dgs_path)) {
      throw ReplayError("Output artifact must be isolated from all sealed inputs");
    }
    record_stage("load_plan_milliseconds");
    const std::string pre_replay_input_sha256 = VerifyAllSealedInputs(plan);
    record_stage("pre_replay_input_verification_milliseconds");
    if (options.verify_runtime_environment) ValidateRuntimeEnvironment(plan, options);
    record_stage("pre_replay_environment_milliseconds");
    const size_t selected_frame_count =
        options.max_frames == 0 ? plan.frames.size() : options.max_frames;

    artifact.reset(new ArtifactWriter(options.output_path));
    artifact->WriteText("input-manifest.json", plan.manifest_text, false);
    artifact->WriteText("phase1-resolved-config.json", plan.config_text, false);
    artifact->WriteText("phase1-frames.jsonl", plan.phase1_frames_text, false);
    Json::Value reference(Json::objectValue);
    reference["schema"] = "online_i3dgs_replay_input_reference_v1";
    reference["sealed_artifact"] = plan.sealed_artifact;
    reference["input_manifest_sha256"] = plan.manifest_sha256;
    reference["phase1_resolved_config_sha256"] = plan.config_sha256;
    reference["phase1_frames_sha256"] = plan.phase1_frames_sha256;
    reference["pre_replay_input_set_sha256"] = pre_replay_input_sha256;
    artifact->WriteJson("input-manifest-reference.json", reference, false);
    artifact->WriteJson(
        "resolved-config.json",
        ResolvedReplayConfig(plan, options, artifact->Root(), selected_frame_count),
        false);
    artifact->WriteText("logs/frames.jsonl", "", false);
    artifact->WriteText("logs/ba-passes.jsonl", "", false);
    record_stage("artifact_initialization_milliseconds");

    OnlineI3dgsReplayControllerConfig controller_config;
    controller_config.camera = plan.camera;
    controller_config.expected_frame_count = plan.frames.size();
    controller_config.output_path = artifact->Root();
    controller_config.ba_window_size = options.ba_window_size;
    std::string factory_error;
    std::unique_ptr<OnlineI3dgsReplayController> controller =
        controller_factory(controller_config, &factory_error);
    if (!controller) {
      throw ReplayError("Controller initialization failed: " + factory_error);
    }
    record_stage("controller_initialization_milliseconds");

    ResetOnlineI3dgsReplayStopForTesting();
    SignalScope signals;
    for (size_t offset = 0; offset < selected_frame_count; ++offset) {
      if (g_stop_requested != 0) throw ReplayError("Replay interrupted by signal");
      const ReplayFrame& frame = plan.frames[offset];
      const OnlineI3dgsReplayControllerFrameResult frame_result =
          controller->ProcessFrame(frame.input);
      record_stage("controller_frames_milliseconds");
      result.global_ba_call_count = frame_result.global_ba_call_count;

      Json::Value record(Json::objectValue);
      record["schema"] = "online_i3dgs_replay_frame_v1";
      record["event_sequence"] = Json::UInt64(offset + 1);
      record["frame_index"] = Json::UInt64(frame.input.frame_index);
      record["image_id"] = Json::UInt64(frame.input.image_id);
      record["phase1_event_sha256"] = frame.event_sha256;
      record["max_visible_frame_index"] = Json::UInt64(frame.input.frame_index);
      record["released_inputs"]["jpg"]["path"] = frame.input.image_path;
      record["released_inputs"]["jpg"]["sha256"] = frame.input.image_sha256;
      record["released_inputs"]["cam"]["path"] = frame.input.camera_path;
      record["released_inputs"]["cam"]["sha256"] = frame.input.camera_sha256;
      record["released_inputs"]["odom"]["path"] = frame.input.odometry_path;
      record["released_inputs"]["odom"]["sha256"] = frame.input.odometry_sha256;
      record["released_inputs"]["scan"]["path"] = frame.input.scan_path;
      record["released_inputs"]["scan"]["sha256"] = frame.input.scan_sha256;
      record["controller_success"] = frame_result.success;
      record["controller_ready_to_flush"] = frame_result.ready_to_flush;
      record["global_ba_call_count"] =
          Json::UInt64(frame_result.global_ba_call_count);
      record["termination"] = frame_result.termination;
      record["controller_audit"] = ParseControllerJson(
          frame_result.audit_json, "controller frame audit", false);
      const std::string ba_lines = SerializeBaRecords(
          frame_result.ba_pass_json, frame.input.frame_index, ba_record_count);
      artifact->AppendText("logs/frames.jsonl", CompactJson(record) + "\n");
      artifact->AppendText("logs/ba-passes.jsonl", ba_lines);
      ba_record_count += frame_result.ba_pass_json.size();

      Json::Value graph = ParseControllerJson(
          frame_result.graph_snapshot_json, "controller graph snapshot", false);
      graph["replay_frame_index"] = Json::UInt64(frame.input.frame_index);
      std::ostringstream graph_name;
      graph_name << "graph/frame-" << std::setw(6) << std::setfill('0')
                 << frame.input.frame_index << ".json";
      artifact->WriteJson(graph_name.str(), graph, false);

      if (frame_result.global_ba_call_count != 0) {
        throw ReplayError("global_ba_call_count became non-zero");
      }
      if (!frame_result.success) {
        throw ReplayError("Controller rejected frame " +
                          std::to_string(frame.input.frame_index) + ": " +
                          frame_result.termination);
      }
      ++result.processed_frame_count;
      result.last_consistent_frame_index = frame.input.frame_index;
      summary = BaseSummary(plan, selected_frame_count, result);
      summary["status"] = "RUNNING";
      summary["pre_replay_input_set_sha256"] = pre_replay_input_sha256;
      artifact->WriteJson("run-summary.json", summary, true);
      record_stage("frame_artifacts_milliseconds");
    }
    if (g_stop_requested != 0) throw ReplayError("Replay interrupted by signal");

    const bool full_sequence = selected_frame_count == plan.frames.size();
    const OnlineI3dgsReplayControllerFinishResult finish =
        controller->Finish(full_sequence);
    record_stage("controller_finish_milliseconds");
    result.global_ba_call_count = finish.global_ba_call_count;
    artifact->AppendText(
        "logs/ba-passes.jsonl",
        SerializeBaRecords(finish.ba_pass_json,
                            full_sequence ? plan.frames.size() : 0,
                            ba_record_count));
    Json::Value final_graph = ParseControllerJson(
        finish.graph_snapshot_json, "controller final graph snapshot", false);
    artifact->WriteJson("graph/final-active-graph.json", final_graph, false);
    if (finish.global_ba_call_count != 0) {
      throw ReplayError("global_ba_call_count is non-zero at replay completion");
    }
    if (!finish.success) {
      throw ReplayError("Controller finish failed: " + finish.termination);
    }
    record_stage("finish_artifacts_milliseconds");

    const std::string post_replay_input_sha256 = VerifyAllSealedInputs(plan);
    if (post_replay_input_sha256 != pre_replay_input_sha256) {
      throw ReplayError("Sealed input aggregate changed after replay");
    }
    record_stage("post_replay_input_verification_milliseconds");
    if (options.verify_runtime_environment) {
      ValidateRuntimeEnvironment(plan, options);
    }
    record_stage("post_replay_environment_milliseconds");
    artifact->PublishModel(*controller);
    record_stage("model_publication_milliseconds");

    result.status = full_sequence ? OnlineI3dgsReplayStatus::COMPLETED
                                  : OnlineI3dgsReplayStatus::SMOKE_COMPLETED;
    result.error.clear();
    summary = BaseSummary(plan, selected_frame_count, result);
    summary["status"] = full_sequence ? "COMPLETED" : "SMOKE_COMPLETED";
    summary["complete"] = full_sequence;
    summary["pre_replay_input_set_sha256"] = pre_replay_input_sha256;
    summary["post_replay_input_set_sha256"] = post_replay_input_sha256;
    summary["input_stability"] = "VERIFIED";
    summary["controller_termination"] = finish.termination;
    summary["controller_summary"] = ParseControllerJson(
        finish.summary_json, "controller run summary", false);
    summary["replay_timing"] = timing;
    artifact->MarkCompleted(summary, !full_sequence);
    return result;
  } catch (const std::exception& error) {
    result.error = error.what();
    if (artifact) {
      result.status = OnlineI3dgsReplayStatus::INCOMPLETE;
      if (summary.isNull()) {
        summary = Json::Value(Json::objectValue);
        summary["schema"] = "online_i3dgs_replay_run_summary_v1";
      }
      summary["status"] = "INCOMPLETE";
      summary["complete"] = false;
      summary["processed_frame_count"] = Json::UInt64(result.processed_frame_count);
      summary["last_consistent_frame_index"] =
          Json::UInt64(result.last_consistent_frame_index);
      summary["global_ba_call_count"] = Json::UInt64(result.global_ba_call_count);
      summary["error"] = result.error;
      artifact->MarkIncomplete(summary);
    } else {
      result.status = OnlineI3dgsReplayStatus::REJECTED;
    }
    return result;
  } catch (...) {
    result.error = "Unknown replay failure";
    if (artifact) {
      result.status = OnlineI3dgsReplayStatus::INCOMPLETE;
      summary["schema"] = "online_i3dgs_replay_run_summary_v1";
      summary["status"] = "INCOMPLETE";
      summary["complete"] = false;
      summary["processed_frame_count"] = Json::UInt64(result.processed_frame_count);
      summary["last_consistent_frame_index"] =
          Json::UInt64(result.last_consistent_frame_index);
      summary["global_ba_call_count"] = Json::UInt64(result.global_ba_call_count);
      summary["error"] = result.error;
      artifact->MarkIncomplete(summary);
    }
    return result;
  }
}

OnlineI3dgsReplayControllerFactory CreateOnlineI3dgsControllerFactory() {
  return [](const OnlineI3dgsReplayControllerConfig& config,
            std::string* error) -> std::unique_ptr<OnlineI3dgsReplayController> {
#ifdef GPU_BA_CUDA_ENABLED
    try {
      return std::unique_ptr<OnlineI3dgsReplayController>(
          new ProductionOnlineReplay(config));
    } catch (const std::exception& exception) {
      if (error != nullptr) *error = exception.what();
      return nullptr;
    } catch (...) {
      if (error != nullptr) *error = "unknown controller construction failure";
      return nullptr;
    }
#else
    static_cast<void>(config);
    if (error != nullptr) {
      *error = "online_i3dgs_mapper requires GPU_BA_CUDA_ENABLED";
    }
    return nullptr;
#endif
  };
}

void RequestOnlineI3dgsReplayStopForTesting() { g_stop_requested = 1; }

void ResetOnlineI3dgsReplayStopForTesting() { g_stop_requested = 0; }

}  // namespace colmap
