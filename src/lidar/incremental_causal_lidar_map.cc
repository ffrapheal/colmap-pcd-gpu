#include "lidar/incremental_causal_lidar_map.h"

#include <openssl/sha.h>

#include <flann/algorithms/dist.h>
#include <flann/algorithms/kdtree_single_index.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>

namespace colmap {
namespace lidar {

constexpr float IncrementalCausalLidarMap::kVoxelLeafMeters;
constexpr int64_t IncrementalCausalLidarMap::kVoxelsPerBlockAxis;
constexpr float IncrementalCausalLidarMap::kOuterNormalRadiusMeters;
constexpr float IncrementalCausalLidarMap::kInnerNormalRadiusMeters;

bool VoxelKey::operator==(const VoxelKey& other) const noexcept {
  return x == other.x && y == other.y && z == other.z;
}

bool VoxelKey::operator!=(const VoxelKey& other) const noexcept {
  return !(*this == other);
}

bool VoxelKey::operator<(const VoxelKey& other) const noexcept {
  if (x != other.x) return x < other.x;
  if (y != other.y) return y < other.y;
  return z < other.z;
}

bool VoxelBlockKey::operator==(const VoxelBlockKey& other) const noexcept {
  return x == other.x && y == other.y && z == other.z;
}

bool VoxelBlockKey::operator!=(const VoxelBlockKey& other) const noexcept {
  return !(*this == other);
}

bool VoxelBlockKey::operator<(const VoxelBlockKey& other) const noexcept {
  if (x != other.x) return x < other.x;
  if (y != other.y) return y < other.y;
  return z < other.z;
}

namespace {

using Sha256Digest = std::array<uint8_t, SHA256_DIGEST_LENGTH>;

int64_t FloorDivide(const int64_t value, const int64_t divisor) noexcept {
  int64_t quotient = value / divisor;
  const int64_t remainder = value % divisor;
  if (remainder < 0) --quotient;
  return quotient;
}

float CanonicalZero(const float value) noexcept {
  return value == 0.0f ? 0.0f : value;
}

double CanonicalZero(const double value) noexcept {
  return value == 0.0 ? 0.0 : value;
}

uint32_t FloatBits(const float value) noexcept {
  uint32_t bits = 0;
  const float canonical = CanonicalZero(value);
  std::memcpy(&bits, &canonical, sizeof(bits));
  return bits;
}

uint64_t DoubleBits(const double value) noexcept {
  uint64_t bits = 0;
  const double canonical = CanonicalZero(value);
  std::memcpy(&bits, &canonical, sizeof(bits));
  return bits;
}

constexpr double kNormalUpdateCellMeters = 0.16;
static_assert(kNormalUpdateCellMeters >
                  IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
              "normal update cells must exceed the largest normal radius");

struct NormalUpdateCellHash {
  size_t operator()(const VoxelKey& cell) const noexcept {
    size_t result = std::hash<int64_t>()(cell.x);
    result ^= std::hash<int64_t>()(cell.y) + 0x9e3779b9u +
              (result << 6) + (result >> 2);
    result ^= std::hash<int64_t>()(cell.z) + 0x9e3779b9u +
              (result << 6) + (result >> 2);
    return result;
  }
};

bool PointToNormalUpdateCell(const std::array<float, 3>& point,
                             VoxelKey* cell) {
  std::array<int64_t, 3> coordinates;
  constexpr double kCoordinateLimit = static_cast<double>(int64_t{1} << 60);
  for (size_t axis = 0; axis < 3; ++axis) {
    const double coordinate =
        std::floor(static_cast<double>(point[axis]) / kNormalUpdateCellMeters);
    if (!std::isfinite(coordinate) || coordinate < -kCoordinateLimit ||
        coordinate > kCoordinateLimit) {
      return false;
    }
    coordinates[axis] = static_cast<int64_t>(coordinate);
  }
  *cell = {coordinates[0], coordinates[1], coordinates[2]};
  return true;
}

class CanonicalWriter {
 public:
  explicit CanonicalWriter(const size_t reserve_bytes = 0) {
    bytes_.reserve(reserve_bytes);
  }

  void AppendUint8(const uint8_t value) { Allocate(1)[0] = value; }

  void AppendUint32(const uint32_t value) {
    uint8_t* output = Allocate(4);
    output[0] = static_cast<uint8_t>(value & 0xffu);
    output[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
    output[2] = static_cast<uint8_t>((value >> 16) & 0xffu);
    output[3] = static_cast<uint8_t>((value >> 24) & 0xffu);
  }

  void AppendUint64(const uint64_t value) {
    uint8_t* output = Allocate(8);
    output[0] = static_cast<uint8_t>(value & 0xffu);
    output[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
    output[2] = static_cast<uint8_t>((value >> 16) & 0xffu);
    output[3] = static_cast<uint8_t>((value >> 24) & 0xffu);
    output[4] = static_cast<uint8_t>((value >> 32) & 0xffu);
    output[5] = static_cast<uint8_t>((value >> 40) & 0xffu);
    output[6] = static_cast<uint8_t>((value >> 48) & 0xffu);
    output[7] = static_cast<uint8_t>((value >> 56) & 0xffu);
  }

  void AppendInt64(const int64_t value) {
    AppendUint64(static_cast<uint64_t>(value));
  }
  void Float(const float value) { AppendUint32(FloatBits(value)); }
  void Double(const double value) { AppendUint64(DoubleBits(value)); }

  void String(const std::string& value) {
    AppendUint64(static_cast<uint64_t>(value.size()));
    AppendBytes(value.data(), value.size());
  }

  template <size_t Size>
  void String(const char (&value)[Size]) {
    AppendUint64(static_cast<uint64_t>(Size - 1));
    AppendBytes(value, Size - 1);
  }

  void Digest(const Sha256Digest& digest) {
    AppendBytes(digest.data(), digest.size());
  }

  const std::vector<uint8_t>& bytes() {
    Flush();
    return bytes_;
  }

 private:
  uint8_t* Allocate(const size_t size) {
    if (buffer_.size() - buffered_bytes_ < size) Flush();
    uint8_t* output = buffer_.data() + buffered_bytes_;
    buffered_bytes_ += size;
    return output;
  }

  void AppendBytes(const void* data, size_t size) {
    const uint8_t* input = static_cast<const uint8_t*>(data);
    while (size > 0) {
      if (buffered_bytes_ == buffer_.size()) Flush();
      const size_t copy_size =
          std::min(size, buffer_.size() - buffered_bytes_);
      std::memcpy(buffer_.data() + buffered_bytes_, input, copy_size);
      buffered_bytes_ += copy_size;
      input += copy_size;
      size -= copy_size;
    }
  }

  void Flush() {
    if (buffered_bytes_ == 0) return;
    bytes_.insert(bytes_.end(), buffer_.data(),
                  buffer_.data() + buffered_bytes_);
    buffered_bytes_ = 0;
  }

  std::array<uint8_t, 4096> buffer_;
  size_t buffered_bytes_ = 0;
  std::vector<uint8_t> bytes_;
};

constexpr char kGeometryBlockHashDomain[] =
    "INCREMENTAL_CAUSAL_LIDAR_GEOMETRY_BLOCK_V1";
constexpr char kSnapshotBlockHashDomain[] =
    "INCREMENTAL_CAUSAL_LIDAR_SNAPSHOT_BLOCK_V1";
constexpr char kGeometryHashDomain[] = "INCREMENTAL_CAUSAL_LIDAR_GEOMETRY_V1";
constexpr char kSnapshotHashDomain[] = "INCREMENTAL_CAUSAL_LIDAR_SNAPSHOT_V1";
constexpr size_t kCanonicalUint8Bytes = 1;
constexpr size_t kCanonicalUint32Bytes = 4;
constexpr size_t kCanonicalUint64Bytes = 8;
constexpr size_t kCanonicalKeyBytes = 3 * kCanonicalUint64Bytes;
constexpr size_t kCanonicalNormalGeometryBytes =
    kCanonicalUint8Bytes + 4 * kCanonicalUint32Bytes;
constexpr size_t kCanonicalNormalSnapshotBytes =
    kCanonicalNormalGeometryBytes + kCanonicalUint64Bytes;
constexpr size_t kCanonicalGeometryVoxelBytes =
    kCanonicalKeyBytes + kCanonicalUint8Bytes +
    3 * kCanonicalUint32Bytes + 2 * kCanonicalNormalGeometryBytes;
constexpr size_t kCanonicalSnapshotVoxelBytes =
    kCanonicalKeyBytes + kCanonicalUint8Bytes + kCanonicalUint64Bytes +
    3 * kCanonicalUint64Bytes + 3 * kCanonicalUint32Bytes +
    2 * kCanonicalUint64Bytes + 2 * kCanonicalNormalSnapshotBytes;
constexpr size_t kCanonicalBlockDigestBytes =
    kCanonicalKeyBytes + SHA256_DIGEST_LENGTH;

template <size_t Size>
constexpr size_t CanonicalStringBytes(const char (&)[Size]) {
  return kCanonicalUint64Bytes + Size - 1;
}

constexpr size_t kCanonicalGeometryBlockHeaderBytes =
    CanonicalStringBytes(kGeometryBlockHashDomain) + kCanonicalKeyBytes +
    kCanonicalUint64Bytes;
constexpr size_t kCanonicalSnapshotBlockHeaderBytes =
    CanonicalStringBytes(kSnapshotBlockHashDomain) + kCanonicalKeyBytes +
    kCanonicalUint64Bytes;
constexpr size_t kCanonicalGeometryHeaderBytes =
    CanonicalStringBytes(kGeometryHashDomain) + kCanonicalUint8Bytes +
    kCanonicalUint32Bytes + kCanonicalUint64Bytes +
    2 * kCanonicalUint32Bytes + 2 * kCanonicalUint64Bytes;
constexpr size_t kCanonicalSnapshotHeaderBytes =
    CanonicalStringBytes(kSnapshotHashDomain) + 2 * kCanonicalUint64Bytes +
    kCanonicalUint8Bytes + kCanonicalUint32Bytes + kCanonicalUint64Bytes +
    2 * kCanonicalUint32Bytes + 2 * kCanonicalUint64Bytes;

Sha256Digest Sha256(const std::vector<uint8_t>& bytes) {
  Sha256Digest digest;
  SHA256(bytes.empty() ? nullptr : bytes.data(), bytes.size(), digest.data());
  return digest;
}

std::string HexDigest(const Sha256Digest& digest) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const uint8_t byte : digest) {
    stream << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return stream.str();
}

bool NormalizeSha256(const std::string& input, std::string* output) {
  if (input.size() != SHA256_DIGEST_LENGTH * 2) return false;
  output->resize(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const char value = input[i];
    if (value >= '0' && value <= '9') {
      (*output)[i] = value;
    } else if (value >= 'a' && value <= 'f') {
      (*output)[i] = value;
    } else if (value >= 'A' && value <= 'F') {
      (*output)[i] = static_cast<char>(value - 'A' + 'a');
    } else {
      output->clear();
      return false;
    }
  }
  return true;
}

bool DefaultStatFile(const std::string& path,
                     uint64_t* size,
                     std::string* error) {
  struct stat metadata;
  if (stat(path.c_str(), &metadata) != 0) {
    *error = "stat failed for " + path + ": " + std::strerror(errno);
    return false;
  }
  if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
    *error = "PCD path is not a regular file: " + path;
    return false;
  }
  *size = static_cast<uint64_t>(metadata.st_size);
  return true;
}

bool DefaultReadFile(const std::string& path,
                     std::vector<uint8_t>* bytes,
                     std::string* error) {
  bytes->clear();
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    *error = "cannot open PCD file: " + path;
    return false;
  }
  const std::streamoff end = file.tellg();
  if (end < 0 ||
      static_cast<uint64_t>(end) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      static_cast<uint64_t>(end) > static_cast<uint64_t>(
                                       std::numeric_limits<std::streamsize>::max())) {
    *error = "PCD file size is not representable: " + path;
    return false;
  }
  file.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(end));
  if (end > 0 &&
      !file.read(reinterpret_cast<char*>(bytes->data()),
                 static_cast<std::streamsize>(end))) {
    bytes->clear();
    *error = "cannot read complete PCD file: " + path;
    return false;
  }
  return true;
}

bool DefaultSha256(const std::vector<uint8_t>& bytes,
                   std::string* digest,
                   std::string* error) {
  (void)error;
  *digest = HexDigest(Sha256(bytes));
  return true;
}

std::string Trim(const std::string& input) {
  size_t begin = 0;
  while (begin < input.size() &&
         std::isspace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }
  size_t end = input.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(begin, end - begin);
}

std::vector<std::string> SplitWhitespace(const std::string& input) {
  std::istringstream stream(input);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) tokens.push_back(token);
  return tokens;
}

std::string UpperAscii(std::string value) {
  for (char& character : value) {
    if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  return value;
}

std::string LowerAscii(std::string value) {
  for (char& character : value) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  return value;
}

bool ParseUint64(const std::string& token,
                 uint64_t* value,
                 std::string* error) {
  if (token.empty() || token[0] == '-') {
    *error = "invalid unsigned PCD header value: " + token;
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(token.c_str(), &end, 10);
  if (errno == ERANGE || end == token.c_str() || *end != '\0') {
    *error = "invalid unsigned PCD header value: " + token;
    return false;
  }
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool CheckedMultiply(const uint64_t lhs,
                     const uint64_t rhs,
                     uint64_t* product) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *product = lhs * rhs;
  return true;
}

struct PcdField {
  std::string name;
  uint64_t size = 0;
  char type = '\0';
  uint64_t count = 0;
  uint64_t byte_offset = 0;
  uint64_t scalar_offset = 0;
};

struct PcdLayout {
  std::vector<PcdField> fields;
  std::array<size_t, 8> required_field_indices{{0, 0, 0, 0, 0, 0, 0, 0}};
  uint64_t point_count = 0;
  uint64_t point_step = 0;
  uint64_t scalar_count = 0;
  size_t data_offset = 0;
  bool binary = false;
};

const std::array<const char*, 8> kRequiredPcdFields = {{
    "x", "y", "z", "intensity", "normal_x", "normal_y", "normal_z",
    "curvature",
}};

bool ParsePcdHeader(const std::vector<uint8_t>& bytes,
                    PcdLayout* layout,
                    std::string* error) {
  std::vector<std::string> names;
  std::vector<uint64_t> sizes;
  std::vector<char> types;
  std::vector<uint64_t> counts;
  uint64_t width = 0;
  uint64_t height = 0;
  uint64_t points = 0;
  bool has_names = false;
  bool has_sizes = false;
  bool has_types = false;
  bool has_counts = false;
  bool has_width = false;
  bool has_height = false;
  bool has_points = false;
  bool has_data = false;

  size_t position = 0;
  while (position < bytes.size()) {
    size_t line_end = position;
    while (line_end < bytes.size() && bytes[line_end] != '\n') ++line_end;
    std::string line(reinterpret_cast<const char*>(bytes.data() + position),
                     line_end - position);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t next = line_end < bytes.size() ? line_end + 1 : line_end;
    position = next;
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> tokens = SplitWhitespace(line);
    if (tokens.empty()) continue;
    const std::string keyword = UpperAscii(tokens[0]);

    if (keyword == "FIELDS" || keyword == "COLUMNS") {
      if (has_names || tokens.size() < 2) {
        *error = "invalid or duplicate PCD FIELDS header";
        return false;
      }
      names.assign(tokens.begin() + 1, tokens.end());
      has_names = true;
    } else if (keyword == "SIZE") {
      if (has_sizes || tokens.size() < 2) {
        *error = "invalid or duplicate PCD SIZE header";
        return false;
      }
      for (size_t i = 1; i < tokens.size(); ++i) {
        uint64_t value = 0;
        if (!ParseUint64(tokens[i], &value, error)) return false;
        sizes.push_back(value);
      }
      has_sizes = true;
    } else if (keyword == "TYPE") {
      if (has_types || tokens.size() < 2) {
        *error = "invalid or duplicate PCD TYPE header";
        return false;
      }
      for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i].size() != 1) {
          *error = "invalid PCD TYPE entry: " + tokens[i];
          return false;
        }
        types.push_back(UpperAscii(tokens[i])[0]);
      }
      has_types = true;
    } else if (keyword == "COUNT") {
      if (has_counts || tokens.size() < 2) {
        *error = "invalid or duplicate PCD COUNT header";
        return false;
      }
      for (size_t i = 1; i < tokens.size(); ++i) {
        uint64_t value = 0;
        if (!ParseUint64(tokens[i], &value, error)) return false;
        counts.push_back(value);
      }
      has_counts = true;
    } else if (keyword == "WIDTH" || keyword == "HEIGHT" ||
               keyword == "POINTS") {
      if (tokens.size() != 2) {
        *error = "invalid PCD " + keyword + " header";
        return false;
      }
      uint64_t value = 0;
      if (!ParseUint64(tokens[1], &value, error)) return false;
      bool* flag = keyword == "WIDTH" ? &has_width
                   : keyword == "HEIGHT" ? &has_height
                                         : &has_points;
      uint64_t* destination = keyword == "WIDTH" ? &width
                              : keyword == "HEIGHT" ? &height
                                                    : &points;
      if (*flag) {
        *error = "duplicate PCD " + keyword + " header";
        return false;
      }
      *flag = true;
      *destination = value;
    } else if (keyword == "DATA") {
      if (has_data || tokens.size() != 2) {
        *error = "invalid or duplicate PCD DATA header";
        return false;
      }
      const std::string mode = LowerAscii(tokens[1]);
      if (mode != "ascii" && mode != "binary") {
        *error = "unsupported PCD DATA mode: " + tokens[1];
        return false;
      }
      layout->binary = mode == "binary";
      layout->data_offset = position;
      has_data = true;
      break;
    }
  }

  if (!has_data || !has_names || !has_sizes || !has_types) {
    *error = "PCD header is missing FIELDS, SIZE, TYPE, or DATA";
    return false;
  }
  if (names.size() != sizes.size() || names.size() != types.size()) {
    *error = "PCD FIELDS, SIZE, and TYPE lengths differ";
    return false;
  }
  if (!has_counts) counts.assign(names.size(), uint64_t{1});
  if (counts.size() != names.size()) {
    *error = "PCD COUNT length differs from FIELDS";
    return false;
  }

  if (has_width != has_height) {
    *error = "PCD WIDTH and HEIGHT must be supplied together";
    return false;
  }
  uint64_t dimensions_count = 0;
  if (has_width && !CheckedMultiply(width, height, &dimensions_count)) {
    *error = "PCD WIDTH times HEIGHT overflows";
    return false;
  }
  if (!has_points && !has_width) {
    *error = "PCD header is missing POINTS or WIDTH/HEIGHT";
    return false;
  }
  if (has_points && has_width && points != dimensions_count) {
    *error = "PCD POINTS differs from WIDTH times HEIGHT";
    return false;
  }
  layout->point_count = has_points ? points : dimensions_count;

  std::map<std::string, size_t> field_indices;
  uint64_t byte_offset = 0;
  uint64_t scalar_offset = 0;
  for (size_t i = 0; i < names.size(); ++i) {
    const std::string name = LowerAscii(names[i]);
    if (!field_indices.emplace(name, i).second) {
      *error = "duplicate PCD field: " + name;
      return false;
    }
    if (counts[i] == 0 ||
        !(types[i] == 'F' || types[i] == 'I' || types[i] == 'U') ||
        !(sizes[i] == 1 || sizes[i] == 2 || sizes[i] == 4 ||
          sizes[i] == 8)) {
      *error = "unsupported PCD field schema for: " + name;
      return false;
    }
    uint64_t field_bytes = 0;
    if (!CheckedMultiply(sizes[i], counts[i], &field_bytes) ||
        byte_offset > std::numeric_limits<uint64_t>::max() - field_bytes ||
        scalar_offset > std::numeric_limits<uint64_t>::max() - counts[i]) {
      *error = "PCD point layout overflows";
      return false;
    }
    PcdField field;
    field.name = name;
    field.size = sizes[i];
    field.type = types[i];
    field.count = counts[i];
    field.byte_offset = byte_offset;
    field.scalar_offset = scalar_offset;
    layout->fields.push_back(field);
    byte_offset += field_bytes;
    scalar_offset += counts[i];
  }
  layout->point_step = byte_offset;
  layout->scalar_count = scalar_offset;
  if (layout->point_step == 0 || layout->scalar_count == 0) {
    *error = "PCD point layout is empty";
    return false;
  }

  for (size_t i = 0; i < kRequiredPcdFields.size(); ++i) {
    const auto field_it = field_indices.find(kRequiredPcdFields[i]);
    if (field_it == field_indices.end()) {
      *error = "PCD is missing required field: " +
               std::string(kRequiredPcdFields[i]);
      return false;
    }
    const PcdField& field = layout->fields[field_it->second];
    if (field.type != 'F' || field.size != 4 || field.count != 1) {
      *error = "PCD field is not PointXYZINormal float32 scalar: " +
               field.name;
      return false;
    }
    layout->required_field_indices[i] = field_it->second;
  }
  return true;
}

float ReadLittleEndianFloat(const uint8_t* bytes) {
  const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
                        (static_cast<uint32_t>(bytes[1]) << 8) |
                        (static_cast<uint32_t>(bytes[2]) << 16) |
                        (static_cast<uint32_t>(bytes[3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename Callback>
bool ParsePcd(const std::vector<uint8_t>& bytes,
              Callback callback,
              uint64_t* point_count,
              std::string* error) {
  PcdLayout layout;
  if (!ParsePcdHeader(bytes, &layout, error)) return false;
  *point_count = layout.point_count;

  if (layout.binary) {
    uint64_t payload_size = 0;
    if (!CheckedMultiply(layout.point_count, layout.point_step, &payload_size) ||
        payload_size > std::numeric_limits<size_t>::max() ||
        layout.data_offset > bytes.size() ||
        bytes.size() - layout.data_offset != static_cast<size_t>(payload_size)) {
      *error = "binary PCD payload size does not match POINTS and schema";
      return false;
    }
    for (uint64_t ordinal = 0; ordinal < layout.point_count; ++ordinal) {
      const uint8_t* record =
          bytes.data() + layout.data_offset +
          static_cast<size_t>(ordinal * layout.point_step);
      std::array<float, 8> values;
      for (size_t field_index = 0; field_index < values.size(); ++field_index) {
        const PcdField& field =
            layout.fields[layout.required_field_indices[field_index]];
        values[field_index] =
            ReadLittleEndianFloat(record + field.byte_offset);
      }
      if (!callback(ordinal, values, error)) return false;
    }
    return true;
  }

  if (layout.data_offset > bytes.size()) {
    *error = "ASCII PCD data offset exceeds file size";
    return false;
  }
  const char* cursor = reinterpret_cast<const char*>(bytes.data()) +
                       layout.data_offset;
  const char* const end = reinterpret_cast<const char*>(bytes.data()) +
                          bytes.size();
  std::vector<int> required_by_scalar;
  if (layout.scalar_count > std::numeric_limits<size_t>::max()) {
    *error = "ASCII PCD scalar count is not representable";
    return false;
  }
  required_by_scalar.assign(static_cast<size_t>(layout.scalar_count), -1);
  for (size_t i = 0; i < layout.required_field_indices.size(); ++i) {
    const PcdField& field = layout.fields[layout.required_field_indices[i]];
    required_by_scalar[static_cast<size_t>(field.scalar_offset)] =
        static_cast<int>(i);
  }

  for (uint64_t ordinal = 0; ordinal < layout.point_count; ++ordinal) {
    std::array<float, 8> values;
    for (uint64_t scalar = 0; scalar < layout.scalar_count; ++scalar) {
      while (cursor < end &&
             std::isspace(static_cast<unsigned char>(*cursor))) {
        ++cursor;
      }
      if (cursor == end) {
        *error = "ASCII PCD payload ended before POINTS records";
        return false;
      }
      const char* token_limit = cursor;
      while (token_limit < end &&
             !std::isspace(static_cast<unsigned char>(*token_limit))) {
        ++token_limit;
      }
      const std::string token(cursor, token_limit - cursor);
      char* token_end = nullptr;
      errno = 0;
      const float parsed = std::strtof(token.c_str(), &token_end);
      if (token_end == token.c_str() || *token_end != '\0') {
        *error = "invalid numeric token in ASCII PCD record " +
                 std::to_string(ordinal);
        return false;
      }
      const int required_index =
          required_by_scalar[static_cast<size_t>(scalar)];
      if (required_index >= 0) {
        values[static_cast<size_t>(required_index)] = parsed;
      }
      cursor = token_limit;
    }
    if (!callback(ordinal, values, error)) return false;
  }
  while (cursor < end && std::isspace(static_cast<unsigned char>(*cursor))) {
    ++cursor;
  }
  if (cursor != end) {
    *error = "ASCII PCD payload has records beyond POINTS";
    return false;
  }
  return true;
}

bool PointValuesAreFinite(const std::array<float, 8>& values) {
  for (const float value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool CoordinateToVoxelIndex(const float coordinate,
                            int64_t* index,
                            std::string* error) {
  const double scaled = static_cast<double>(coordinate) * 100.0;
  const double floored = std::floor(scaled);
  if (!std::isfinite(floored) ||
      floored < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      floored >= -static_cast<double>(std::numeric_limits<int64_t>::min())) {
    *error = "transformed point exceeds int64 1 cm voxel-key range";
    return false;
  }
  *index = static_cast<int64_t>(floored);
  return true;
}

bool PointToVoxelKey(const std::array<float, 3>& point,
                     VoxelKey* key,
                     std::string* error) {
  return CoordinateToVoxelIndex(point[0], &key->x, error) &&
         CoordinateToVoxelIndex(point[1], &key->y, error) &&
         CoordinateToVoxelIndex(point[2], &key->z, error);
}

void WriteKey(const VoxelKey& key, CanonicalWriter* writer) {
  writer->AppendInt64(key.x);
  writer->AppendInt64(key.y);
  writer->AppendInt64(key.z);
}

void WriteBlockKey(const VoxelBlockKey& key, CanonicalWriter* writer) {
  writer->AppendInt64(key.x);
  writer->AppendInt64(key.y);
  writer->AppendInt64(key.z);
}

void WriteNormalGeometry(const LidarNormalValue& normal,
                         CanonicalWriter* writer) {
  writer->AppendUint8(normal.valid ? uint8_t{1} : uint8_t{0});
  if (normal.valid) {
    writer->Float(normal.normal[0]);
    writer->Float(normal.normal[1]);
    writer->Float(normal.normal[2]);
    writer->Float(normal.curvature);
  } else {
    writer->Float(0.0f);
    writer->Float(0.0f);
    writer->Float(0.0f);
    writer->Float(0.0f);
  }
}

void WriteNormalSnapshot(const LidarNormalValue& normal,
                         CanonicalWriter* writer) {
  WriteNormalGeometry(normal, writer);
  writer->AppendUint64(normal.revision);
}

bool IsValidScale(const LidarNormalScale scale) {
  return scale == LidarNormalScale::OUTER_0_15_M ||
         scale == LidarNormalScale::INNER_0_05_M;
}

const LidarNormalValue* SelectNormal(const LidarVoxelRecord& record,
                                    const LidarNormalScale scale) {
  if (scale == LidarNormalScale::OUTER_0_15_M) {
    return &record.outer_normal;
  }
  if (scale == LidarNormalScale::INNER_0_05_M) {
    return &record.inner_normal;
  }
  return nullptr;
}

PlaneSample MakePlaneSample(const LidarVoxelRecord& record,
                            const LidarNormalScale scale) {
  const LidarNormalValue* normal = SelectNormal(record, scale);
  PlaneSample sample;
  sample.key = record.key;
  sample.frame = record.frame;
  sample.scale = scale;
  sample.point = record.centroid;
  sample.normal = normal->normal;
  sample.curvature = normal->curvature;
  sample.normal_revision = normal->revision;
  sample.voxel_count = record.count;
  return sample;
}

bool TimingIsValid(const CudaNormalEstimationTiming& timing) {
  return std::isfinite(timing.upload_ms) && timing.upload_ms >= 0.0 &&
         std::isfinite(timing.index_ms) && timing.index_ms >= 0.0 &&
         std::isfinite(timing.normals_ms) && timing.normals_ms >= 0.0 &&
         std::isfinite(timing.download_ms) && timing.download_ms >= 0.0;
}

bool DecodeNormalOutput(const std::vector<float>& values,
                        const size_t query_count,
                        const uint64_t revision,
                        const char* label,
                        std::vector<LidarNormalValue>* normals,
                        uint64_t* valid_count,
                        uint64_t* invalid_count,
                        std::string* error) {
  if (query_count > std::numeric_limits<size_t>::max() / 4 ||
      values.size() != query_count * 4) {
    *error = std::string(label) +
             " normal output length is not exactly query_count * 4";
    return false;
  }
  normals->clear();
  normals->reserve(query_count);
  *valid_count = 0;
  *invalid_count = 0;
  for (size_t i = 0; i < query_count; ++i) {
    const float nx = values[4 * i];
    const float ny = values[4 * i + 1];
    const float nz = values[4 * i + 2];
    const float curvature = values[4 * i + 3];
    const bool all_nan = std::isnan(nx) && std::isnan(ny) &&
                         std::isnan(nz) && std::isnan(curvature);
    if (all_nan) {
      LidarNormalValue invalid;
      invalid.revision = revision;
      normals->push_back(invalid);
      ++*invalid_count;
      continue;
    }
    if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz) ||
        !std::isfinite(curvature)) {
      *error = std::string(label) +
               " normal output contains partial NaN or infinity at query " +
               std::to_string(i);
      return false;
    }
    const double squared_norm = static_cast<double>(nx) * nx +
                                static_cast<double>(ny) * ny +
                                static_cast<double>(nz) * nz;
    if (!std::isfinite(squared_norm) || !(squared_norm > 0.0)) {
      *error = std::string(label) +
               " finite normal cannot be normalized at query " +
               std::to_string(i);
      return false;
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    LidarNormalValue normal;
    normal.normal = {{CanonicalZero(static_cast<float>(nx * inverse_norm)),
                      CanonicalZero(static_cast<float>(ny * inverse_norm)),
                      CanonicalZero(static_cast<float>(nz * inverse_norm))}};
    normal.curvature = CanonicalZero(curvature);
    normal.valid = true;
    normal.revision = revision;
    normals->push_back(normal);
    ++*valid_count;
  }
  return true;
}

int64_t SaturatingFloorCentimeters(const double coordinate) {
  const long double scaled = static_cast<long double>(coordinate) * 100.0L;
  if (scaled <= static_cast<long double>(
                    std::numeric_limits<int64_t>::min())) {
    return std::numeric_limits<int64_t>::min();
  }
  if (scaled >= static_cast<long double>(
                    std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::max();
  }
  return static_cast<int64_t>(std::floor(scaled));
}

int64_t SaturatingAdd(const int64_t value, const int64_t delta) {
  if (delta > 0 && value > std::numeric_limits<int64_t>::max() - delta) {
    return std::numeric_limits<int64_t>::max();
  }
  if (delta < 0 && value < std::numeric_limits<int64_t>::min() - delta) {
    return std::numeric_limits<int64_t>::min();
  }
  return value + delta;
}

}  // namespace

VoxelBlockKey VoxelKeyToBlockKey(const VoxelKey& key) noexcept {
  VoxelBlockKey block;
  block.x = FloorDivide(key.x,
                        IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  block.y = FloorDivide(key.y,
                        IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  block.z = FloorDivide(key.z,
                        IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  return block;
}

namespace internal {

struct IncrementalCausalLidarMapBlock {
  std::map<VoxelKey, LidarVoxelRecord> voxels;
  std::array<float, 3> centroid_min{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> centroid_max{{0.0f, 0.0f, 0.0f}};
  Sha256Digest geometry_digest{{0}};
  Sha256Digest snapshot_digest{{0}};
};

struct IncrementalCausalLidarMapSpatialYLine {
  std::map<int64_t,
           std::shared_ptr<const IncrementalCausalLidarMapBlock>>
      blocks_by_z;
};

struct IncrementalCausalLidarMapSpatialXPlane {
  std::map<int64_t,
           std::shared_ptr<const IncrementalCausalLidarMapSpatialYLine>>
      lines_by_y;
};

struct IncrementalCausalLidarMapNearestIndex {
  using Tree = flann::KDTreeSingleIndex<flann::L2_Simple<double>>;

  struct Entry {
    VoxelKey key;
    bool outer_valid = false;
    bool inner_valid = false;
  };

  class Result : public flann::ResultSet<double> {
   public:
    Result(const std::vector<Entry>& entries,
           const LidarNormalScale scale,
           const double max_squared_distance)
        : squared_distance(max_squared_distance),
          entries_(entries),
          scale_(scale) {}

    bool full() const override { return found; }

    double worstDist() const override {
      return std::nextafter(squared_distance,
                            std::numeric_limits<double>::infinity());
    }

    void addPoint(const double distance, const size_t index) override {
      const Entry& entry = entries_[index];
      const bool valid = scale_ == LidarNormalScale::OUTER_0_15_M
                             ? entry.outer_valid
                             : entry.inner_valid;
      if (!valid || distance > squared_distance) return;
      if (!found || distance < squared_distance ||
          entry.key < entries_[best_index].key) {
        found = true;
        squared_distance = distance;
        best_index = index;
      }
    }

    bool found = false;
    double squared_distance;
    size_t best_index = 0;

   private:
    const std::vector<Entry>& entries_;
    const LidarNormalScale scale_;
  };

  std::vector<Entry> entries;
  std::vector<double> coordinates;
  std::unique_ptr<const Tree> tree;
};

struct IncrementalCausalLidarMapSupportPoint {
  VoxelKey key;
  std::array<float, 3> centroid;
};

struct IncrementalCausalLidarMapState {
  uint64_t version = 0;
  uint64_t max_scan_index = 0;
  LidarCoordinateFrame frame = LidarCoordinateFrame::COLMAP_WORLD;
  size_t voxel_count = 0;
  std::map<VoxelBlockKey,
           std::shared_ptr<const IncrementalCausalLidarMapBlock>>
      blocks;
  std::map<int64_t,
           std::shared_ptr<const IncrementalCausalLidarMapSpatialXPlane>>
      spatial_planes_by_x;
  std::shared_ptr<const IncrementalCausalLidarMapNearestIndex> nearest_index;
  std::shared_ptr<const std::vector<IncrementalCausalLidarMapSupportPoint>>
      normal_support =
          std::make_shared<std::vector<IncrementalCausalLidarMapSupportPoint>>();
  std::string geometry_sha256;
  std::string snapshot_sha256;
};

}  // namespace internal

namespace {

using MapBlock = internal::IncrementalCausalLidarMapBlock;
using MapState = internal::IncrementalCausalLidarMapState;
using SpatialXPlane = internal::IncrementalCausalLidarMapSpatialXPlane;
using SpatialYLine = internal::IncrementalCausalLidarMapSpatialYLine;
using NearestIndex = internal::IncrementalCausalLidarMapNearestIndex;
using SupportPoint = internal::IncrementalCausalLidarMapSupportPoint;

void RebuildNearestIndex(MapState* state) {
  auto index = std::make_shared<NearestIndex>();
  index->entries.reserve(state->voxel_count);
  index->coordinates.reserve(3 * state->voxel_count);
  for (const auto& block_entry : state->blocks) {
    for (const auto& voxel_entry : block_entry.second->voxels) {
      const LidarVoxelRecord& record = voxel_entry.second;
      if (!record.outer_normal.valid && !record.inner_normal.valid) continue;
      NearestIndex::Entry entry;
      entry.key = record.key;
      entry.outer_valid = record.outer_normal.valid;
      entry.inner_valid = record.inner_normal.valid;
      index->entries.push_back(entry);
      for (const float coordinate : record.centroid) {
        index->coordinates.push_back(static_cast<double>(coordinate));
      }
    }
  }
  if (!index->entries.empty()) {
    const flann::Matrix<double> dataset(
        index->coordinates.data(), index->entries.size(), 3);
    std::unique_ptr<NearestIndex::Tree> tree(new NearestIndex::Tree(
        dataset, flann::KDTreeSingleIndexParams(10, false)));
    tree->buildIndex();
    index->tree = std::move(tree);
  }
  state->nearest_index = std::move(index);
}

struct MutableSpatialIndex {
  std::map<int64_t, std::shared_ptr<SpatialXPlane>> planes_by_x;
  std::map<std::pair<int64_t, int64_t>, std::shared_ptr<SpatialYLine>>
      lines_by_xy;
};

const LidarVoxelRecord* FindRecord(const MapState& state,
                                   const VoxelKey& key) {
  const auto block_it = state.blocks.find(VoxelKeyToBlockKey(key));
  if (block_it == state.blocks.end()) return nullptr;
  const auto voxel_it = block_it->second->voxels.find(key);
  return voxel_it == block_it->second->voxels.end() ? nullptr
                                                     : &voxel_it->second;
}

std::shared_ptr<MapBlock> EnsureMutableBlock(
    MapState* state,
    std::map<VoxelBlockKey, std::shared_ptr<MapBlock>>* mutable_blocks,
    const VoxelBlockKey& key) {
  const auto mutable_it = mutable_blocks->find(key);
  if (mutable_it != mutable_blocks->end()) return mutable_it->second;

  std::shared_ptr<MapBlock> block;
  const auto state_it = state->blocks.find(key);
  if (state_it == state->blocks.end()) {
    block = std::make_shared<MapBlock>();
  } else {
    block = std::make_shared<MapBlock>(*state_it->second);
  }
  mutable_blocks->emplace(key, block);
  state->blocks[key] = block;
  return block;
}

void UpdateSpatialBlockIndex(MapState* state,
                             MutableSpatialIndex* mutable_spatial_index,
                             const VoxelBlockKey& key,
                             const std::shared_ptr<MapBlock>& block) {
  std::shared_ptr<SpatialXPlane> plane;
  const auto mutable_plane_it =
      mutable_spatial_index->planes_by_x.find(key.x);
  if (mutable_plane_it != mutable_spatial_index->planes_by_x.end()) {
    plane = mutable_plane_it->second;
  } else {
    const auto state_plane_it = state->spatial_planes_by_x.find(key.x);
    plane = state_plane_it == state->spatial_planes_by_x.end()
                ? std::make_shared<SpatialXPlane>()
                : std::make_shared<SpatialXPlane>(*state_plane_it->second);
    mutable_spatial_index->planes_by_x.emplace(key.x, plane);
    state->spatial_planes_by_x[key.x] = plane;
  }

  const std::pair<int64_t, int64_t> xy(key.x, key.y);
  std::shared_ptr<SpatialYLine> line;
  const auto mutable_line_it =
      mutable_spatial_index->lines_by_xy.find(xy);
  if (mutable_line_it != mutable_spatial_index->lines_by_xy.end()) {
    line = mutable_line_it->second;
  } else {
    const auto plane_line_it = plane->lines_by_y.find(key.y);
    line = plane_line_it == plane->lines_by_y.end()
               ? std::make_shared<SpatialYLine>()
               : std::make_shared<SpatialYLine>(*plane_line_it->second);
    mutable_spatial_index->lines_by_xy.emplace(xy, line);
    plane->lines_by_y[key.y] = line;
  }
  line->blocks_by_z[key.z] = block;
}

LidarVoxelRecord* MutableRecord(
    std::map<VoxelBlockKey, std::shared_ptr<MapBlock>>* mutable_blocks,
    const VoxelKey& key) {
  const VoxelBlockKey block_key = VoxelKeyToBlockKey(key);
  const auto mutable_it = mutable_blocks->find(block_key);
  if (mutable_it == mutable_blocks->end()) return nullptr;
  const auto voxel_it = mutable_it->second->voxels.find(key);
  return voxel_it == mutable_it->second->voxels.end() ? nullptr
                                                       : &voxel_it->second;
}

void RecomputeBlockDerivedData(const VoxelBlockKey& block_key,
                               MapBlock* block) {
  if (block->voxels.empty()) {
    throw std::logic_error("incremental LiDAR map contains an empty block");
  }
  block->centroid_min = block->voxels.begin()->second.centroid;
  block->centroid_max = block->centroid_min;

  const size_t voxel_count = block->voxels.size();
  CanonicalWriter geometry_writer(kCanonicalGeometryBlockHeaderBytes +
                                  voxel_count *
                                      kCanonicalGeometryVoxelBytes);
  geometry_writer.String(kGeometryBlockHashDomain);
  WriteBlockKey(block_key, &geometry_writer);
  geometry_writer.AppendUint64(static_cast<uint64_t>(voxel_count));

  CanonicalWriter snapshot_writer(kCanonicalSnapshotBlockHeaderBytes +
                                  voxel_count *
                                      kCanonicalSnapshotVoxelBytes);
  snapshot_writer.String(kSnapshotBlockHashDomain);
  WriteBlockKey(block_key, &snapshot_writer);
  snapshot_writer.AppendUint64(static_cast<uint64_t>(voxel_count));

  for (const auto& voxel_entry : block->voxels) {
    const LidarVoxelRecord& record = voxel_entry.second;
    for (size_t axis = 0; axis < 3; ++axis) {
      block->centroid_min[axis] =
          std::min(block->centroid_min[axis], record.centroid[axis]);
      block->centroid_max[axis] =
          std::max(block->centroid_max[axis], record.centroid[axis]);
    }

    WriteKey(record.key, &geometry_writer);
    geometry_writer.AppendUint8(static_cast<uint8_t>(record.frame));
    for (const float coordinate : record.centroid) {
      geometry_writer.Float(coordinate);
    }
    WriteNormalGeometry(record.outer_normal, &geometry_writer);
    WriteNormalGeometry(record.inner_normal, &geometry_writer);

    WriteKey(record.key, &snapshot_writer);
    snapshot_writer.AppendUint8(static_cast<uint8_t>(record.frame));
    snapshot_writer.AppendUint64(record.count);
    for (const double sum : record.xyz_sum) snapshot_writer.Double(sum);
    for (const float coordinate : record.centroid) {
      snapshot_writer.Float(coordinate);
    }
    snapshot_writer.AppendUint64(record.first_scan_index);
    snapshot_writer.AppendUint64(record.last_scan_index);
    WriteNormalSnapshot(record.outer_normal, &snapshot_writer);
    WriteNormalSnapshot(record.inner_normal, &snapshot_writer);
  }
  block->geometry_digest = Sha256(geometry_writer.bytes());
  block->snapshot_digest = Sha256(snapshot_writer.bytes());
}

void RecomputeTopLevelHashes(MapState* state) {
  const size_t block_count = state->blocks.size();
  CanonicalWriter geometry_writer(kCanonicalGeometryHeaderBytes +
                                  block_count * kCanonicalBlockDigestBytes);
  geometry_writer.String(kGeometryHashDomain);
  geometry_writer.AppendUint8(static_cast<uint8_t>(state->frame));
  geometry_writer.Float(IncrementalCausalLidarMap::kVoxelLeafMeters);
  geometry_writer.AppendInt64(
      IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  geometry_writer.Float(
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters);
  geometry_writer.Float(
      IncrementalCausalLidarMap::kInnerNormalRadiusMeters);
  geometry_writer.AppendUint64(
      static_cast<uint64_t>(state->voxel_count));
  geometry_writer.AppendUint64(
      static_cast<uint64_t>(state->blocks.size()));
  for (const auto& block_entry : state->blocks) {
    WriteBlockKey(block_entry.first, &geometry_writer);
    geometry_writer.Digest(block_entry.second->geometry_digest);
  }
  state->geometry_sha256 = HexDigest(Sha256(geometry_writer.bytes()));

  CanonicalWriter snapshot_writer(kCanonicalSnapshotHeaderBytes +
                                  block_count * kCanonicalBlockDigestBytes);
  snapshot_writer.String(kSnapshotHashDomain);
  snapshot_writer.AppendUint64(state->version);
  snapshot_writer.AppendUint64(state->max_scan_index);
  snapshot_writer.AppendUint8(static_cast<uint8_t>(state->frame));
  snapshot_writer.Float(IncrementalCausalLidarMap::kVoxelLeafMeters);
  snapshot_writer.AppendInt64(
      IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  snapshot_writer.Float(
      IncrementalCausalLidarMap::kOuterNormalRadiusMeters);
  snapshot_writer.Float(
      IncrementalCausalLidarMap::kInnerNormalRadiusMeters);
  snapshot_writer.AppendUint64(
      static_cast<uint64_t>(state->voxel_count));
  snapshot_writer.AppendUint64(
      static_cast<uint64_t>(state->blocks.size()));
  for (const auto& block_entry : state->blocks) {
    WriteBlockKey(block_entry.first, &snapshot_writer);
    snapshot_writer.Digest(block_entry.second->snapshot_digest);
  }
  state->snapshot_sha256 = HexDigest(Sha256(snapshot_writer.bytes()));
}

struct BlockKeyRange {
  VoxelBlockKey min;
  VoxelBlockKey max;
  bool empty = false;
};

BlockKeyRange ComputeAabbBlockKeyRange(const LidarAabb& bounds) {
  BlockKeyRange range;
  int64_t* minimum_axes[3] = {&range.min.x, &range.min.y, &range.min.z};
  int64_t* maximum_axes[3] = {&range.max.x, &range.max.y, &range.max.z};
  for (size_t axis = 0; axis < 3; ++axis) {
    *minimum_axes[axis] = SaturatingAdd(
        FloorDivide(SaturatingFloorCentimeters(bounds.min[axis]),
                    IncrementalCausalLidarMap::kVoxelsPerBlockAxis),
        -1);
    *maximum_axes[axis] = SaturatingAdd(
        FloorDivide(SaturatingFloorCentimeters(bounds.max[axis]),
                    IncrementalCausalLidarMap::kVoxelsPerBlockAxis),
        1);
  }
  return range;
}

bool ComputeNearestBlockKeyRange(const std::array<double, 3>& query,
                                 const double max_distance,
                                 BlockKeyRange* range,
                                 std::string* error) {
  constexpr long double kMaxBlockCombinations = 1000000.0L;
  const int64_t minimum_block = FloorDivide(
      std::numeric_limits<int64_t>::min(),
      IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  const int64_t maximum_block = FloorDivide(
      std::numeric_limits<int64_t>::max(),
      IncrementalCausalLidarMap::kVoxelsPerBlockAxis);
  int64_t* minimum_axes[3] = {&range->min.x, &range->min.y, &range->min.z};
  int64_t* maximum_axes[3] = {&range->max.x, &range->max.y, &range->max.z};

  for (size_t axis = 0; axis < 3; ++axis) {
    const long double lower_centimeters =
        (static_cast<long double>(query[axis]) -
         static_cast<long double>(max_distance)) *
        100.0L;
    const long double upper_centimeters =
        (static_cast<long double>(query[axis]) +
         static_cast<long double>(max_distance)) *
        100.0L;
    if (lower_centimeters >
            static_cast<long double>(std::numeric_limits<int64_t>::max()) ||
        upper_centimeters <
            static_cast<long double>(std::numeric_limits<int64_t>::min())) {
      range->empty = true;
      return true;
    }
    const int64_t minimum_voxel =
        lower_centimeters <=
                static_cast<long double>(std::numeric_limits<int64_t>::min())
            ? std::numeric_limits<int64_t>::min()
            : static_cast<int64_t>(std::floor(lower_centimeters));
    const int64_t maximum_voxel =
        upper_centimeters >=
                static_cast<long double>(std::numeric_limits<int64_t>::max())
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(std::floor(upper_centimeters));
    *minimum_axes[axis] = std::max(
        minimum_block,
        SaturatingAdd(FloorDivide(
                          minimum_voxel,
                          IncrementalCausalLidarMap::kVoxelsPerBlockAxis),
                      -1));
    *maximum_axes[axis] = std::min(
        maximum_block,
        SaturatingAdd(FloorDivide(
                          maximum_voxel,
                          IncrementalCausalLidarMap::kVoxelsPerBlockAxis),
                      1));
  }

  long double combination_count = 1.0L;
  const int64_t minimum_axes_values[3] = {
      range->min.x, range->min.y, range->min.z};
  const int64_t maximum_axes_values[3] = {
      range->max.x, range->max.y, range->max.z};
  for (size_t axis = 0; axis < 3; ++axis) {
    const long double axis_count =
        static_cast<long double>(maximum_axes_values[axis]) -
        static_cast<long double>(minimum_axes_values[axis]) + 1.0L;
    if (!(axis_count > 0.0L) ||
        combination_count > kMaxBlockCombinations / axis_count) {
      *error =
          "nearest-plane block-key interval exceeds 1000000 combinations";
      return false;
    }
    combination_count *= axis_count;
  }
  return true;
}

bool BlockIntersectsAabb(const MapBlock& block, const LidarAabb& bounds) {
  for (size_t axis = 0; axis < 3; ++axis) {
    if (static_cast<double>(block.centroid_max[axis]) < bounds.min[axis] ||
        static_cast<double>(block.centroid_min[axis]) > bounds.max[axis]) {
      return false;
    }
  }
  return true;
}

bool PointInsideAabb(const std::array<float, 3>& point,
                     const LidarAabb& bounds) {
  for (size_t axis = 0; axis < 3; ++axis) {
    if (static_cast<double>(point[axis]) < bounds.min[axis] ||
        static_cast<double>(point[axis]) > bounds.max[axis]) {
      return false;
    }
  }
  return true;
}

std::string ExceptionMessage(const char* stage,
                             const std::exception& exception) {
  return std::string(stage) + " threw exception: " + exception.what();
}

}  // namespace

LidarMapSnapshot::LidarMapSnapshot(
    std::shared_ptr<const internal::IncrementalCausalLidarMapState> state)
    : state_(std::move(state)) {}

uint64_t LidarMapSnapshot::Version() const noexcept { return state_->version; }

uint64_t LidarMapSnapshot::MaxScanIndex() const noexcept {
  return state_->max_scan_index;
}

LidarCoordinateFrame LidarMapSnapshot::Frame() const noexcept {
  return state_->frame;
}

size_t LidarMapSnapshot::VoxelCount() const noexcept {
  return state_->voxel_count;
}

size_t LidarMapSnapshot::BlockCount() const noexcept {
  return state_->blocks.size();
}

const std::string& LidarMapSnapshot::GeometrySha256() const noexcept {
  return state_->geometry_sha256;
}

const std::string& LidarMapSnapshot::SnapshotSha256() const noexcept {
  return state_->snapshot_sha256;
}

bool LidarMapSnapshot::FindVoxel(const VoxelKey& key,
                                 LidarVoxelRecord* record) const noexcept {
  if (record == nullptr) return false;
  const LidarVoxelRecord* found = FindRecord(*state_, key);
  if (found == nullptr) return false;
  *record = *found;
  return true;
}

std::vector<VoxelKey> LidarMapSnapshot::VoxelKeys() const {
  std::vector<VoxelKey> keys;
  keys.reserve(state_->voxel_count);
  for (const auto& block_entry : state_->blocks) {
    for (const auto& voxel_entry : block_entry.second->voxels) {
      keys.push_back(voxel_entry.first);
    }
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

NearestPlaneResult LidarMapSnapshot::FindNearestPlane(
    const std::array<double, 3>& query_world,
    const LidarCoordinateFrame query_frame,
    const LidarNormalScale scale,
    const double max_distance) const noexcept {
  NearestPlaneResult result;
  try {
    if (query_frame != LidarCoordinateFrame::COLMAP_WORLD) {
      result.error = "nearest-plane query frame must be COLMAP_WORLD";
      return result;
    }
    if (!IsValidScale(scale)) {
      result.error = "nearest-plane normal scale is invalid";
      return result;
    }
    for (const double coordinate : query_world) {
      if (!std::isfinite(coordinate)) {
        result.error = "nearest-plane query contains non-finite coordinate";
        return result;
      }
    }
    if (!std::isfinite(max_distance) || max_distance < 0.0 ||
        max_distance > std::sqrt(std::numeric_limits<double>::max())) {
      result.error = "nearest-plane max distance is invalid";
      return result;
    }

    const double max_squared_distance = max_distance * max_distance;
    BlockKeyRange block_range;
    if (!ComputeNearestBlockKeyRange(query_world, max_distance, &block_range,
                                     &result.error)) {
      return result;
    }
    if (block_range.empty) {
      result.ok = true;
      return result;
    }
    if (!state_->nearest_index || !state_->nearest_index->tree) {
      result.ok = true;
      return result;
    }
    const NearestIndex& index = *state_->nearest_index;
    NearestIndex::Result nearest(index.entries, scale, max_squared_distance);
    index.tree->findNeighbors(nearest, query_world.data(),
                               flann::SearchParams(-1, 0.0f));
    if (nearest.found) {
      const LidarVoxelRecord* record =
          FindRecord(*state_, index.entries[nearest.best_index].key);
      if (record == nullptr) {
        result.ok = false;
        result.error = "nearest-plane index references a missing voxel";
        return result;
      }
      result.found = true;
      result.visited_block_count = 1;
      result.plane = MakePlaneSample(*record, scale);
      result.squared_distance = nearest.squared_distance;
    }
    result.ok = true;
    return result;
  } catch (const std::exception& exception) {
    result.error = ExceptionMessage("nearest-plane query", exception);
    return result;
  } catch (...) {
    result.error = "nearest-plane query threw non-standard exception";
    return result;
  }
}

NearestPlaneResult LidarMapSnapshot::FindNearestPlane(
    const std::array<double, 3>& query_world,
    const LidarNormalScale scale,
    const double max_distance) const noexcept {
  return FindNearestPlane(query_world, LidarCoordinateFrame::COLMAP_WORLD,
                          scale, max_distance);
}

PlaneCollectionResult LidarMapSnapshot::CollectPlanesInAabb(
    const LidarAabb& bounds, const LidarNormalScale scale) const noexcept {
  PlaneCollectionResult result;
  try {
    if (bounds.frame != LidarCoordinateFrame::COLMAP_WORLD) {
      result.error = "plane AABB frame must be COLMAP_WORLD";
      return result;
    }
    if (!IsValidScale(scale)) {
      result.error = "plane AABB normal scale is invalid";
      return result;
    }
    for (size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(bounds.min[axis]) ||
          !std::isfinite(bounds.max[axis]) ||
          bounds.min[axis] > bounds.max[axis]) {
        result.error = "plane AABB bounds are invalid";
        return result;
      }
    }
    const BlockKeyRange block_range = ComputeAabbBlockKeyRange(bounds);
    auto plane_it =
        state_->spatial_planes_by_x.lower_bound(block_range.min.x);
    while (plane_it != state_->spatial_planes_by_x.end() &&
           plane_it->first <= block_range.max.x) {
      auto line_it =
          plane_it->second->lines_by_y.lower_bound(block_range.min.y);
      while (line_it != plane_it->second->lines_by_y.end() &&
             line_it->first <= block_range.max.y) {
        auto block_it =
            line_it->second->blocks_by_z.lower_bound(block_range.min.z);
        while (block_it != line_it->second->blocks_by_z.end() &&
               block_it->first <= block_range.max.z) {
          ++result.visited_block_count;
          const MapBlock& block = *block_it->second;
          if (BlockIntersectsAabb(block, bounds)) {
            for (const auto& voxel_entry : block.voxels) {
              const LidarVoxelRecord& record = voxel_entry.second;
              const LidarNormalValue* normal = SelectNormal(record, scale);
              if (normal != nullptr && normal->valid &&
                  PointInsideAabb(record.centroid, bounds)) {
                result.planes.push_back(MakePlaneSample(record, scale));
              }
            }
          }
          ++block_it;
        }
        ++line_it;
      }
      ++plane_it;
    }
    std::sort(result.planes.begin(), result.planes.end(),
              [](const PlaneSample& lhs, const PlaneSample& rhs) {
                return lhs.key < rhs.key;
              });
    result.ok = true;
    return result;
  } catch (const std::exception& exception) {
    result.planes.clear();
    result.error = ExceptionMessage("plane AABB query", exception);
    return result;
  } catch (...) {
    result.planes.clear();
    result.error = "plane AABB query threw non-standard exception";
    return result;
  }
}

namespace internal {

struct IncrementalCausalLidarMapImpl {
  explicit IncrementalCausalLidarMapImpl(
      IncrementalCausalLidarMapDependencies requested_dependencies)
      : dependencies(std::move(requested_dependencies)) {
    if (!dependencies.stat_file) dependencies.stat_file = DefaultStatFile;
    if (!dependencies.read_file) dependencies.read_file = DefaultReadFile;
    if (!dependencies.sha256) dependencies.sha256 = DefaultSha256;
    if (!dependencies.estimate_normals) {
      dependencies.estimate_normals =
          EstimateDualRadiusNormalsForQueriesCuda;
    }

    std::shared_ptr<MapState> state = std::make_shared<MapState>();
    RecomputeTopLevelHashes(state.get());
    std::shared_ptr<const LidarMapSnapshot> initial(
        new LidarMapSnapshot(state));
    std::atomic_store_explicit(&current_snapshot, initial,
                               std::memory_order_release);
  }

  bool AppendLocked(const ScanSource& source,
                    AppendScanAudit* output_audit,
                    std::string* output_error) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point profile_start = Clock::now();
    Clock::time_point stage_start = profile_start;
    AppendScanAudit audit;
    const auto mark_stage = [&](const char* name) {
      const Clock::time_point now = Clock::now();
      audit.stage_milliseconds[name] =
          std::chrono::duration<double, std::milli>(now - stage_start).count();
      stage_start = now;
    };
    const std::shared_ptr<const LidarMapSnapshot> parent_snapshot =
        std::atomic_load_explicit(&current_snapshot,
                                  std::memory_order_acquire);
    const MapState& parent = *parent_snapshot->state_;

    if (parent.max_scan_index == std::numeric_limits<uint64_t>::max() ||
        source.scan_index != parent.max_scan_index + 1) {
      *output_error = "scan_index must equal max_scan_index + 1";
      return false;
    }
    if (source.input_frame != LidarCoordinateFrame::FASTLIO_WORLD) {
      *output_error = "scan input frame must be FASTLIO_WORLD";
      return false;
    }
    if (source.point_transform_count != 0 ||
        source.normal_transform_count != 0) {
      *output_error =
          "scan point and normal transform counts must both be zero";
      return false;
    }
    if (source.pcd_path.empty()) {
      *output_error = "scan PCD path is empty";
      return false;
    }

    uint64_t stat_size = 0;
    std::string stage_error;
    try {
      if (!dependencies.stat_file(source.pcd_path, &stat_size, &stage_error)) {
        *output_error = "PCD stat failed: " +
                        (stage_error.empty() ? std::string("unspecified error")
                                             : stage_error);
        return false;
      }
    } catch (const std::exception& exception) {
      *output_error = ExceptionMessage("PCD stat", exception);
      return false;
    } catch (...) {
      *output_error = "PCD stat threw non-standard exception";
      return false;
    }
    if (source.expected_size_bytes != 0 &&
        source.expected_size_bytes != stat_size) {
      *output_error = "PCD file size does not match expected_size_bytes";
      return false;
    }

    std::vector<uint8_t> bytes;
    stage_error.clear();
    try {
      if (!dependencies.read_file(source.pcd_path, &bytes, &stage_error)) {
        *output_error = "PCD read failed: " +
                        (stage_error.empty() ? std::string("unspecified error")
                                             : stage_error);
        return false;
      }
    } catch (const std::exception& exception) {
      *output_error = ExceptionMessage("PCD read", exception);
      return false;
    } catch (...) {
      *output_error = "PCD read threw non-standard exception";
      return false;
    }
    if (bytes.size() != stat_size) {
      *output_error = "PCD file size changed between stat and read";
      return false;
    }
    mark_stage("input_read");

    std::string file_sha256;
    stage_error.clear();
    try {
      if (!dependencies.sha256(bytes, &file_sha256, &stage_error)) {
        *output_error = "PCD SHA-256 failed: " +
                        (stage_error.empty() ? std::string("unspecified error")
                                             : stage_error);
        return false;
      }
    } catch (const std::exception& exception) {
      *output_error = ExceptionMessage("PCD SHA-256", exception);
      return false;
    } catch (...) {
      *output_error = "PCD SHA-256 threw non-standard exception";
      return false;
    }
    std::string normalized_file_sha256;
    if (!NormalizeSha256(file_sha256, &normalized_file_sha256)) {
      *output_error = "PCD SHA-256 hook returned a non-SHA256 digest";
      return false;
    }
    if (!source.expected_sha256.empty()) {
      std::string normalized_expected_sha256;
      if (!NormalizeSha256(source.expected_sha256,
                           &normalized_expected_sha256)) {
        *output_error = "expected_sha256 is not a 64-digit hexadecimal digest";
        return false;
      }
      if (normalized_expected_sha256 != normalized_file_sha256) {
        *output_error = "PCD SHA-256 does not match expected_sha256";
        return false;
      }
    }

    mark_stage("input_hash");
    std::shared_ptr<MapState> candidate = std::make_shared<MapState>(parent);
    std::map<VoxelBlockKey, std::shared_ptr<MapBlock>> mutable_blocks;
    MutableSpatialIndex mutable_spatial_index;
    std::set<VoxelKey> touched_keys;
    audit.scan_index = source.scan_index;
    audit.opened_scan_index = source.scan_index;
    audit.version_before = parent.version;
    audit.max_scan_index_before = parent.max_scan_index;
    audit.source_frame = source.input_frame;
    audit.output_frame = LidarCoordinateFrame::COLMAP_WORLD;
    audit.derived_normal_frame = LidarCoordinateFrame::COLMAP_WORLD;
    audit.derived_normal_transform_count = 0;
    audit.file_size_bytes = stat_size;
    audit.file_sha256 = normalized_file_sha256;
    audit.geometry_sha256_before = parent.geometry_sha256;
    audit.snapshot_sha256_before = parent.snapshot_sha256;
    mark_stage("snapshot_copy");

    uint64_t parsed_point_count = 0;
    stage_error.clear();
    const auto ingest = [&](const uint64_t ordinal,
                            const std::array<float, 8>& values,
                            std::string* ingest_error) {
      (void)ordinal;
      if (!PointValuesAreFinite(values)) {
        ++audit.dropped_non_finite_record_count;
        return true;
      }

      std::array<float, 3> point{{CanonicalZero(-values[1]),
                                  CanonicalZero(-values[2]),
                                  CanonicalZero(values[0])}};
      std::array<float, 3> source_normal{{CanonicalZero(-values[5]),
                                          CanonicalZero(-values[6]),
                                          CanonicalZero(values[4])}};
      const double normal_squared_norm =
          static_cast<double>(source_normal[0]) * source_normal[0] +
          static_cast<double>(source_normal[1]) * source_normal[1] +
          static_cast<double>(source_normal[2]) * source_normal[2];
      if (normal_squared_norm == 0.0) {
        source_normal = {{0.0f, 0.0f, 0.0f}};
        ++audit.zero_source_normal_count;
      } else {
        const double inverse_norm = 1.0 / std::sqrt(normal_squared_norm);
        for (size_t axis = 0; axis < 3; ++axis) {
          source_normal[axis] = CanonicalZero(
              static_cast<float>(source_normal[axis] * inverse_norm));
        }
        ++audit.nonzero_source_normal_count;
      }

      VoxelKey key;
      if (!PointToVoxelKey(point, &key, ingest_error)) return false;
      const VoxelBlockKey block_key = VoxelKeyToBlockKey(key);
      const std::shared_ptr<MapBlock> block = EnsureMutableBlock(
          candidate.get(), &mutable_blocks, block_key);
      auto insertion = block->voxels.emplace(key, LidarVoxelRecord());
      LidarVoxelRecord& record = insertion.first->second;
      if (insertion.second) {
        record.key = key;
        record.frame = LidarCoordinateFrame::COLMAP_WORLD;
        record.first_scan_index = source.scan_index;
      }
      if (record.count == std::numeric_limits<uint64_t>::max()) {
        *ingest_error = "voxel accumulator count overflows uint64";
        return false;
      }
      for (size_t axis = 0; axis < 3; ++axis) {
        const double updated_sum =
            record.xyz_sum[axis] + static_cast<double>(point[axis]);
        if (!std::isfinite(updated_sum)) {
          *ingest_error = "voxel accumulator sum became non-finite";
          return false;
        }
        record.xyz_sum[axis] = updated_sum;
      }
      ++record.count;
      record.last_scan_index = source.scan_index;
      touched_keys.insert(key);

      if (!audit.has_first_accepted_record) {
        audit.has_first_accepted_record = true;
        audit.first_transformed_point = point;
        audit.first_transformed_source_normal = source_normal;
      }
      ++audit.accepted_record_count;
      return true;
    };

    try {
      if (!ParsePcd(bytes, ingest, &parsed_point_count, &stage_error)) {
        *output_error = "PCD parse/ingest failed: " +
                        (stage_error.empty() ? std::string("unspecified error")
                                             : stage_error);
        return false;
      }
    } catch (const std::exception& exception) {
      *output_error = ExceptionMessage("PCD parse/ingest", exception);
      return false;
    } catch (...) {
      *output_error = "PCD parse/ingest threw non-standard exception";
      return false;
    }
    audit.input_record_count = parsed_point_count;
    if (audit.accepted_record_count == 0) {
      *output_error = "PCD contains no finite PointXYZINormal records";
      return false;
    }
    audit.point_transform_count_min = 1;
    audit.point_transform_count_max = 1;
    audit.source_normal_transform_count_min = 1;
    audit.source_normal_transform_count_max = 1;
    mark_stage("parse_and_voxelize");

    std::set<VoxelKey> geometry_changed_keys;
    for (const VoxelKey& key : touched_keys) {
      LidarVoxelRecord* record =
          MutableRecord(&mutable_blocks, key);
      if (record == nullptr || record->count == 0) {
        *output_error = "internal touched voxel is missing";
        return false;
      }
      std::array<float, 3> centroid;
      for (size_t axis = 0; axis < 3; ++axis) {
        centroid[axis] = CanonicalZero(static_cast<float>(
            record->xyz_sum[axis] / static_cast<double>(record->count)));
        if (!std::isfinite(centroid[axis])) {
          *output_error = "voxel centroid became non-finite";
          return false;
        }
      }
      const LidarVoxelRecord* old_record = FindRecord(parent, key);
      record->centroid = centroid;
      if (old_record == nullptr) {
        ++audit.new_voxel_count;
        geometry_changed_keys.insert(key);
      } else {
        bool changed = false;
        for (size_t axis = 0; axis < 3; ++axis) {
          changed = changed ||
                    FloatBits(old_record->centroid[axis]) !=
                        FloatBits(record->centroid[axis]);
        }
        if (changed) {
          ++audit.geometry_changed_voxel_count;
          geometry_changed_keys.insert(key);
        } else {
          ++audit.count_only_voxel_count;
        }
      }
    }
    audit.touched_voxel_count = static_cast<uint64_t>(touched_keys.size());
    if (audit.new_voxel_count >
        std::numeric_limits<size_t>::max() - parent.voxel_count) {
      *output_error = "voxel count exceeds size_t capacity";
      return false;
    }
    candidate->voxel_count =
        parent.voxel_count + static_cast<size_t>(audit.new_voxel_count);
    mark_stage("update_centroids");

    std::unordered_set<VoxelKey, NormalUpdateCellHash> changed_cells;
    bool update_all_normals = parent.voxel_count == 0;
    if (!update_all_normals) {
      for (const VoxelKey& key : geometry_changed_keys) {
        const LidarVoxelRecord* records[] = {FindRecord(parent, key),
                                             FindRecord(*candidate, key)};
        for (const LidarVoxelRecord* record : records) {
          if (record == nullptr) continue;
          VoxelKey cell;
          if (!PointToNormalUpdateCell(record->centroid, &cell)) {
            update_all_normals = true;
            break;
          }
          changed_cells.insert(cell);
        }
        if (update_all_normals) break;
      }
    }
    std::unordered_set<VoxelKey, NormalUpdateCellHash> dirty_cells;
    if (!update_all_normals) {
      for (const VoxelKey& cell : changed_cells) {
        for (int64_t offset_z = -1; offset_z <= 1; ++offset_z) {
          for (int64_t offset_y = -1; offset_y <= 1; ++offset_y) {
            for (int64_t offset_x = -1; offset_x <= 1; ++offset_x) {
              dirty_cells.insert({cell.x + offset_x, cell.y + offset_y,
                                  cell.z + offset_z});
            }
          }
        }
      }
    }
    mark_stage("affected_search");
    audit.stage_milliseconds["support_search"] = 0.0;
    std::vector<VoxelKey> affected_keys;
    if (!geometry_changed_keys.empty()) {
      auto support = std::make_shared<std::vector<SupportPoint>>();
      support->reserve(candidate->voxel_count);
      auto changed = geometry_changed_keys.begin();
      const auto append_changed = [&]() {
        const LidarVoxelRecord* record = FindRecord(*candidate, *changed);
        if (record == nullptr) {
          *output_error = "internal changed support voxel is missing";
          return false;
        }
        support->push_back({record->key, record->centroid});
        ++changed;
        return true;
      };
      for (const SupportPoint& previous : *parent.normal_support) {
        while (changed != geometry_changed_keys.end() &&
               *changed < previous.key) {
          if (!append_changed()) return false;
        }
        if (changed != geometry_changed_keys.end() &&
            *changed == previous.key) {
          if (!append_changed()) return false;
        } else {
          support->push_back(previous);
        }
      }
      while (changed != geometry_changed_keys.end()) {
        if (!append_changed()) return false;
      }
      if (support->size() != candidate->voxel_count) {
        *output_error = "normal support cache does not cover the complete map";
        return false;
      }
      candidate->normal_support = std::move(support);
    }
    const std::vector<SupportPoint> empty_support;
    const auto& support_records = geometry_changed_keys.empty()
                                      ? empty_support
                                      : *candidate->normal_support;
    std::vector<float> support_xyz;
    std::vector<float> query_xyz;
    support_xyz.reserve(support_records.size() * 3);
    for (const SupportPoint& record : support_records) {
      support_xyz.insert(support_xyz.end(), record.centroid.begin(),
                         record.centroid.end());
      VoxelKey cell;
      if (update_all_normals ||
          !PointToNormalUpdateCell(record.centroid, &cell) ||
          dirty_cells.count(cell) != 0) {
        affected_keys.push_back(record.key);
        query_xyz.insert(query_xyz.end(), record.centroid.begin(),
                         record.centroid.end());
      }
    }
    audit.affected_query_count =
        static_cast<uint64_t>(affected_keys.size());
    audit.support_voxel_count =
        static_cast<uint64_t>(support_records.size());
    audit.all_voxels_queried = affected_keys.size() == candidate->voxel_count;
    mark_stage("pack_normal_inputs");

    std::vector<float> outer_output;
    std::vector<float> inner_output;
    std::vector<LidarNormalValue> outer_normals;
    std::vector<LidarNormalValue> inner_normals;
    audit.stage_milliseconds["normal_estimation"] = 0.0;
    audit.stage_milliseconds["decode_normals"] = 0.0;
    if (!affected_keys.empty()) {
      stage_error.clear();
      try {
        if (!dependencies.estimate_normals(
                support_xyz,
                audit.all_voxels_queried ? support_xyz : query_xyz,
                IncrementalCausalLidarMap::kOuterNormalRadiusMeters,
                IncrementalCausalLidarMap::kInnerNormalRadiusMeters,
                &outer_output, &inner_output, &audit.cuda_timing,
                &stage_error)) {
          *output_error =
              "CUDA dual-normal estimation failed: " +
              (stage_error.empty() ? std::string("unspecified error")
                                   : stage_error);
          return false;
        }
      } catch (const std::exception& exception) {
        *output_error = ExceptionMessage("CUDA dual-normal estimator",
                                         exception);
        return false;
      } catch (...) {
        *output_error =
            "CUDA dual-normal estimator threw non-standard exception";
        return false;
      }
      mark_stage("normal_estimation");
      if (!TimingIsValid(audit.cuda_timing)) {
        *output_error = "CUDA dual-normal estimator returned invalid timing";
        return false;
      }
      if (!DecodeNormalOutput(
              outer_output, affected_keys.size(), source.scan_index, "outer",
              &outer_normals, &audit.outer_valid_count,
              &audit.outer_invalid_count, output_error) ||
          !DecodeNormalOutput(
              inner_output, affected_keys.size(), source.scan_index, "inner",
              &inner_normals, &audit.inner_valid_count,
              &audit.inner_invalid_count, output_error)) {
        return false;
      }
      mark_stage("decode_normals");

      size_t query_index = 0;
      for (const VoxelKey& key : affected_keys) {
        const VoxelBlockKey block_key = VoxelKeyToBlockKey(key);
        const std::shared_ptr<MapBlock> block = EnsureMutableBlock(
            candidate.get(), &mutable_blocks, block_key);
        const auto record_it = block->voxels.find(key);
        if (record_it == block->voxels.end()) {
          *output_error = "internal normal-update voxel is missing";
          return false;
        }
        record_it->second.outer_normal = outer_normals[query_index];
        record_it->second.inner_normal = inner_normals[query_index];
        ++query_index;
      }
    }
    mark_stage("writeback_normals");

    double block_derived_ms = 0.0;
    double spatial_index_ms = 0.0;
    for (const auto& mutable_entry : mutable_blocks) {
      const Clock::time_point derived_start = Clock::now();
      RecomputeBlockDerivedData(mutable_entry.first,
                                mutable_entry.second.get());
      const Clock::time_point index_start = Clock::now();
      block_derived_ms += std::chrono::duration<double, std::milli>(
                              index_start - derived_start).count();
      UpdateSpatialBlockIndex(candidate.get(), &mutable_spatial_index,
                              mutable_entry.first, mutable_entry.second);
      spatial_index_ms += std::chrono::duration<double, std::milli>(
                              Clock::now() - index_start).count();
    }
    audit.stage_milliseconds["block_derived"] = block_derived_ms;
    audit.stage_milliseconds["spatial_index"] = spatial_index_ms;
    stage_start = Clock::now();
    if (!geometry_changed_keys.empty()) RebuildNearestIndex(candidate.get());
    mark_stage("nearest_index");
    candidate->version = source.scan_index;
    candidate->max_scan_index = source.scan_index;
    RecomputeTopLevelHashes(candidate.get());

    audit.version_after = candidate->version;
    audit.max_scan_index_after = candidate->max_scan_index;
    audit.geometry_sha256_after = candidate->geometry_sha256;
    audit.snapshot_sha256_after = candidate->snapshot_sha256;
    audit.map_voxel_count = candidate->voxel_count;
    audit.map_block_count = candidate->blocks.size();
    audit.mutable_block_count = mutable_blocks.size();
    std::shared_ptr<const LidarMapSnapshot> published(
        new LidarMapSnapshot(candidate));
    mark_stage("snapshot_finalize");
    audit.total_milliseconds = std::chrono::duration<double, std::milli>(
                                  Clock::now() - profile_start).count();
    if (output_audit != nullptr) *output_audit = audit;
    std::atomic_store_explicit(&current_snapshot, published,
                               std::memory_order_release);
    return true;
  }

  IncrementalCausalLidarMapDependencies dependencies;
  mutable std::shared_ptr<const LidarMapSnapshot> current_snapshot;
  std::mutex writer_mutex;
};

}  // namespace internal

IncrementalCausalLidarMap::IncrementalCausalLidarMap()
    : IncrementalCausalLidarMap(IncrementalCausalLidarMapDependencies()) {}

IncrementalCausalLidarMap::IncrementalCausalLidarMap(
    IncrementalCausalLidarMapDependencies dependencies)
    : impl_(new internal::IncrementalCausalLidarMapImpl(
          std::move(dependencies))) {}

IncrementalCausalLidarMap::~IncrementalCausalLidarMap() = default;

std::shared_ptr<const LidarMapSnapshot>
IncrementalCausalLidarMap::GetSnapshot() const noexcept {
  return std::atomic_load_explicit(&impl_->current_snapshot,
                                   std::memory_order_acquire);
}

bool IncrementalCausalLidarMap::AppendScan(const ScanSource& source,
                                           AppendScanAudit* audit,
                                           std::string* error) noexcept {
  if (audit != nullptr) *audit = AppendScanAudit();
  if (error != nullptr) error->clear();
  std::string local_error;
  try {
    std::lock_guard<std::mutex> lock(impl_->writer_mutex);
    const bool success = impl_->AppendLocked(source, audit, &local_error);
    if (!success && error != nullptr) *error = local_error;
    return success;
  } catch (const std::exception& exception) {
    local_error = ExceptionMessage("incremental LiDAR append", exception);
  } catch (...) {
    local_error = "incremental LiDAR append threw non-standard exception";
  }
  if (audit != nullptr) *audit = AppendScanAudit();
  if (error != nullptr) *error = local_error;
  return false;
}

}  // namespace lidar
}  // namespace colmap
