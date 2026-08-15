#include "gpu_ba/snapshot.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <openssl/sha.h>

#include "util/misc.h"

namespace colmap {
namespace gpu_ba {
namespace {

constexpr char kPayloadMagic[8] = {'G', 'P', 'U', 'B', 'A', 'S', 'N', 'P'};

enum class SectionType : uint32_t {
  kMetadata = 1,
  kCameras = 2,
  kImages = 3,
  kPoints = 4,
  kObservations = 5,
  kTracks = 6,
  kLidar = 7,
  kParameterBlocks = 8,
  kParameterCanonicalOrder = 9,
  kSourceOrder = 10,
  kCanonicalOrder = 11,
};

struct Section {
  SectionType type;
  std::string name;
  std::vector<uint8_t> data;
  uint32_t crc32 = 0;
  std::string sha256;
};

class BinaryWriter {
 public:
  void U8(uint8_t value) { data_.push_back(value); }

  void U32(uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
      data_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
    }
  }

  void I32(int32_t value) { U32(static_cast<uint32_t>(value)); }

  void UInt64(uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
      data_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
    }
  }

  void Double(double value) {
    static_assert(sizeof(double) == sizeof(uint64_t),
                  "Snapshot schema requires IEEE-754 binary64");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    UInt64(bits);
  }

  void String(const std::string& value) {
    UInt64(value.size());
    Raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
  }

  void Raw(const uint8_t* bytes, size_t size) {
    if (size == 0) return;
    data_.insert(data_.end(), bytes, bytes + size);
  }

  void Raw(const std::vector<uint8_t>& bytes) {
    data_.insert(data_.end(), bytes.begin(), bytes.end());
  }

  const std::vector<uint8_t>& Data() const { return data_; }
  std::vector<uint8_t> MoveData() { return std::move(data_); }

 private:
  std::vector<uint8_t> data_;
};

class BinaryReader {
 public:
  explicit BinaryReader(const std::vector<uint8_t>& data) : data_(&data) {}

  bool U8(uint8_t* value) {
    if (!Require(1)) return false;
    *value = (*data_)[offset_++];
    return true;
  }

  bool U32(uint32_t* value) {
    if (!Require(4)) return false;
    *value = 0;
    for (size_t i = 0; i < 4; ++i) {
      *value |= static_cast<uint32_t>((*data_)[offset_++]) << (8 * i);
    }
    return true;
  }

  bool I32(int32_t* value) {
    uint32_t bits = 0;
    if (!U32(&bits)) return false;
    *value = static_cast<int32_t>(bits);
    return true;
  }

  bool UInt64(uint64_t* value) {
    if (!Require(8)) return false;
    *value = 0;
    for (size_t i = 0; i < 8; ++i) {
      *value |= static_cast<uint64_t>((*data_)[offset_++]) << (8 * i);
    }
    return true;
  }

  bool Double(double* value) {
    uint64_t bits = 0;
    if (!UInt64(&bits)) return false;
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool String(std::string* value) {
    uint64_t size = 0;
    if (!UInt64(&size) || size > Remaining()) return false;
    value->assign(reinterpret_cast<const char*>(data_->data() + offset_),
                  static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return true;
  }

  bool Raw(size_t size, std::vector<uint8_t>* value) {
    if (!Require(size)) return false;
    value->assign(data_->begin() + offset_, data_->begin() + offset_ + size);
    offset_ += size;
    return true;
  }

  bool Skip(size_t size) {
    if (!Require(size)) return false;
    offset_ += size;
    return true;
  }

  size_t Remaining() const { return data_->size() - offset_; }
  size_t Offset() const { return offset_; }
  bool AtEnd() const { return offset_ == data_->size(); }

 private:
  bool Require(size_t size) const { return size <= data_->size() - offset_; }

  const std::vector<uint8_t>* data_;
  size_t offset_ = 0;
};

std::string Hex(const uint8_t* data, size_t size) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (size_t i = 0; i < size; ++i) {
    stream << std::setw(2) << static_cast<int>(data[i]);
  }
  return stream.str();
}

bool HexToBytes(const std::string& hex, std::vector<uint8_t>* bytes) {
  if (hex.size() % 2 != 0) return false;
  bytes->clear();
  bytes->reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    unsigned int value = 0;
    std::istringstream stream(hex.substr(i, 2));
    stream >> std::hex >> value;
    if (!stream || value > 255) return false;
    bytes->push_back(static_cast<uint8_t>(value));
  }
  return true;
}

std::string Sha256Bytes(const uint8_t* data, size_t size) {
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
  SHA256(size == 0 ? nullptr : data, size, digest.data());
  return Hex(digest.data(), digest.size());
}

bool ReadFile(const std::string& path,
              std::vector<uint8_t>* data,
              std::string* error) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    *error = "Cannot open file: " + path;
    return false;
  }
  const std::streamoff size = file.tellg();
  if (size < 0) {
    *error = "Cannot determine file size: " + path;
    return false;
  }
  file.seekg(0, std::ios::beg);
  data->resize(static_cast<size_t>(size));
  if (size > 0 && !file.read(reinterpret_cast<char*>(data->data()), size)) {
    *error = "Cannot read complete file: " + path;
    return false;
  }
  return true;
}

bool WriteFile(const std::string& path,
               const std::vector<uint8_t>& data,
               std::string* error) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    *error = "Cannot open file for writing: " + path;
    return false;
  }
  if (!data.empty()) {
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
  }
  file.close();
  if (!file) {
    *error = "Cannot write complete file: " + path;
    return false;
  }
  return true;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': stream << "\\\\"; break;
      case '"': stream << "\\\""; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
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

template <typename T>
void JsonIntegerArray(std::ostringstream* stream, const std::vector<T>& values) {
  *stream << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) *stream << ',';
    *stream << static_cast<uint64_t>(values[i]);
  }
  *stream << ']';
}

std::string SectionManifest(const std::vector<Section>& sections) {
  std::ostringstream stream;
  stream << '[';
  for (size_t i = 0; i < sections.size(); ++i) {
    if (i > 0) stream << ',';
    stream << "{\"name\":\"" << JsonEscape(sections[i].name)
           << "\",\"type\":" << static_cast<uint32_t>(sections[i].type)
           << ",\"bytes\":" << sections[i].data.size()
           << ",\"crc32\":" << sections[i].crc32
           << ",\"sha256\":\"" << sections[i].sha256 << "\"}";
  }
  stream << ']';
  return stream.str();
}

std::vector<uint32_t> SelectedImageIds(const Snapshot& snapshot) {
  std::vector<uint32_t> ids;
  for (const auto& image : snapshot.images) {
    if (image.selected) ids.push_back(image.image_id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<uint32_t> PoseIds(const Snapshot& snapshot, bool constant) {
  std::vector<uint32_t> ids;
  for (const auto& image : snapshot.images) {
    if (image.pose_constant == constant) ids.push_back(image.image_id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<uint64_t> PointIds(const Snapshot& snapshot, bool constant) {
  std::vector<uint64_t> ids;
  for (const auto& point : snapshot.points) {
    if (point.constant == constant) ids.push_back(point.point3D_id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::string BuildManifestCore(const Snapshot& snapshot,
                              const std::vector<Section>& sections) {
  const auto selected_images = SelectedImageIds(snapshot);
  const auto variable_poses = PoseIds(snapshot, false);
  const auto constant_poses = PoseIds(snapshot, true);
  const auto variable_points = PointIds(snapshot, false);
  const auto constant_points = PointIds(snapshot, true);

  std::ostringstream stream;
  stream << "{\"schema_version\":" << kSnapshotSchemaVersion
         << ",\"snapshot_id\":\"" << JsonEscape(snapshot.metadata.snapshot_id)
         << "\",\"ba_kind\":\"" << BaKindName(snapshot.metadata.ba_kind)
         << "\",\"registered_image_count\":"
         << snapshot.metadata.registered_image_count
         << ",\"ba_call_index\":" << snapshot.metadata.ba_call_index
         << ",\"refinement_index\":" << snapshot.metadata.refinement_index
         << ",\"trigger_image_id\":" << snapshot.metadata.trigger_image_id
         << ",\"optimize_phrase\":\""
         << JsonEscape(snapshot.metadata.optimize_phrase)
         << "\",\"backend\":\"" << JsonEscape(snapshot.metadata.backend)
         << "\",\"loss\":\"" << JsonEscape(snapshot.metadata.loss_function)
         << "\",\"lidar_residual_mode\":\""
         << JsonEscape(snapshot.metadata.lidar_residual_mode)
         << "\",\"lidar_correspondence_version\":\""
         << JsonEscape(snapshot.metadata.lidar_correspondence_version)
         << "\",\"data_types\":{\"endianness\":\"little\","
            "\"floating_point\":\"IEEE-754 binary64\","
            "\"signed_integer\":\"two-complement int32\","
            "\"identifier\":\"uint32/uint64\"},\"counts\":{"
         << "\"cameras\":" << snapshot.cameras.size()
         << ",\"images\":" << snapshot.images.size()
         << ",\"points\":" << snapshot.points.size()
         << ",\"observations\":" << snapshot.observations.size()
         << ",\"track_elements\":" << snapshot.tracks.size()
         << ",\"lidar\":" << snapshot.lidar.size()
         << ",\"parameter_blocks\":"
         << snapshot.parameter_blocks_source_order.size()
         << ",\"residuals\":" << snapshot.source_insertion_order.size()
         << "},\"selected_image_ids\":";
  JsonIntegerArray(&stream, selected_images);
  stream << ",\"variable_pose_ids\":";
  JsonIntegerArray(&stream, variable_poses);
  stream << ",\"constant_pose_ids\":";
  JsonIntegerArray(&stream, constant_poses);
  stream << ",\"variable_point_ids\":";
  JsonIntegerArray(&stream, variable_points);
  stream << ",\"constant_point_ids\":";
  JsonIntegerArray(&stream, constant_points);
  stream << ",\"search_ranges\":[";
  bool first_range = true;
  for (const auto& point : snapshot.points) {
    if (!point.has_search_range) continue;
    if (!first_range) stream << ',';
    first_range = false;
    stream << "{\"point3D_id\":" << point.point3D_id
           << ",\"meters\":" << std::setprecision(17) << point.search_range
           << '}';
  }
  stream << "],\"source_insertion_order\":{\"section\":\"source_order\","
            "\"count\":" << snapshot.source_insertion_order.size()
         << "},\"canonical_comparison_order\":{\"section\":"
            "\"canonical_order\",\"count\":" << snapshot.canonical_order.size()
         << "},\"sections\":" << SectionManifest(sections) << '}';
  return stream.str();
}

void SerializeMetadata(const SnapshotMetadata& value, BinaryWriter* writer) {
  writer->String(value.snapshot_id);
  writer->U8(static_cast<uint8_t>(value.ba_kind));
  writer->UInt64(value.registered_image_count);
  writer->UInt64(value.ba_call_index);
  writer->UInt64(value.refinement_index);
  writer->U32(value.trigger_image_id);
  writer->String(value.optimize_phrase);
  writer->String(value.backend);
  writer->String(value.loss_function);
  writer->String(value.lidar_residual_mode);
  writer->String(value.lidar_correspondence_version);
  writer->String(value.schur_mode);
  writer->U8(value.refine_focal_length);
  writer->U8(value.refine_principal_point);
  writer->U8(value.refine_extra_params);
  writer->U8(value.refine_extrinsics);
  writer->Double(value.proj_lidar_weight);
  writer->Double(value.icp_lidar_weight);
  writer->Double(value.icp_ground_lidar_weight);
  writer->Double(value.function_tolerance);
  writer->Double(value.gradient_tolerance);
  writer->Double(value.parameter_tolerance);
  writer->I32(value.max_num_iterations);
  writer->I32(value.max_linear_solver_iterations);
  writer->I32(value.max_consecutive_invalid_steps);
}

bool DeserializeMetadata(BinaryReader* reader, SnapshotMetadata* value) {
  uint8_t kind = 0;
  uint8_t refine_focal = 0;
  uint8_t refine_principal = 0;
  uint8_t refine_extra = 0;
  uint8_t refine_extrinsics = 0;
  if (!reader->String(&value->snapshot_id) || !reader->U8(&kind) ||
      kind > static_cast<uint8_t>(BaKind::kWhole) ||
      !reader->UInt64(&value->registered_image_count) ||
      !reader->UInt64(&value->ba_call_index) ||
      !reader->UInt64(&value->refinement_index) ||
      !reader->U32(&value->trigger_image_id) ||
      !reader->String(&value->optimize_phrase) ||
      !reader->String(&value->backend) ||
      !reader->String(&value->loss_function) ||
      !reader->String(&value->lidar_residual_mode) ||
      !reader->String(&value->lidar_correspondence_version) ||
      !reader->String(&value->schur_mode) || !reader->U8(&refine_focal) ||
      !reader->U8(&refine_principal) || !reader->U8(&refine_extra) ||
      !reader->U8(&refine_extrinsics) ||
      !reader->Double(&value->proj_lidar_weight) ||
      !reader->Double(&value->icp_lidar_weight) ||
      !reader->Double(&value->icp_ground_lidar_weight) ||
      !reader->Double(&value->function_tolerance) ||
      !reader->Double(&value->gradient_tolerance) ||
      !reader->Double(&value->parameter_tolerance) ||
      !reader->I32(&value->max_num_iterations) ||
      !reader->I32(&value->max_linear_solver_iterations) ||
      !reader->I32(&value->max_consecutive_invalid_steps)) {
    return false;
  }
  if (refine_focal > 1 || refine_principal > 1 || refine_extra > 1 ||
      refine_extrinsics > 1) {
    return false;
  }
  value->ba_kind = static_cast<BaKind>(kind);
  value->refine_focal_length = refine_focal;
  value->refine_principal_point = refine_principal;
  value->refine_extra_params = refine_extra;
  value->refine_extrinsics = refine_extrinsics;
  return reader->AtEnd();
}

template <size_t N>
void SerializeArray(const std::array<double, N>& values, BinaryWriter* writer) {
  for (const double value : values) writer->Double(value);
}

template <size_t N>
bool DeserializeArray(BinaryReader* reader, std::array<double, N>* values) {
  for (double& value : *values) {
    if (!reader->Double(&value)) return false;
  }
  return true;
}

Section MakeSection(SectionType type,
                    const std::string& name,
                    std::vector<uint8_t> data) {
  Section section;
  section.type = type;
  section.name = name;
  section.data = std::move(data);
  section.crc32 = Crc32(section.data);
  section.sha256 = Sha256Hex(section.data);
  return section;
}

std::vector<Section> BuildSections(const Snapshot& snapshot) {
  std::vector<Section> sections;
  BinaryWriter writer;
  SerializeMetadata(snapshot.metadata, &writer);
  sections.push_back(MakeSection(SectionType::kMetadata, "metadata",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.cameras.size());
  for (const auto& camera : snapshot.cameras) {
    writer.U32(camera.camera_id);
    writer.I32(camera.model_id);
    writer.UInt64(camera.width);
    writer.UInt64(camera.height);
    writer.U8(camera.constant);
    writer.UInt64(camera.params.size());
    for (const double param : camera.params) writer.Double(param);
  }
  sections.push_back(MakeSection(SectionType::kCameras, "cameras",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.images.size());
  for (const auto& image : snapshot.images) {
    writer.U32(image.image_id);
    writer.U32(image.camera_id);
    writer.U8(image.selected);
    writer.U8(image.pose_constant);
    writer.U8(image.has_pose_parameter_blocks);
    writer.U8(image.constant_tvec_mask);
    SerializeArray(image.qvec, &writer);
    SerializeArray(image.tvec, &writer);
  }
  sections.push_back(MakeSection(SectionType::kImages, "images",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.points.size());
  for (const auto& point : snapshot.points) {
    writer.UInt64(point.point3D_id);
    writer.U8(point.constant);
    writer.U8(point.config_role);
    writer.U8(point.has_search_range);
    writer.Double(point.search_range);
    SerializeArray(point.xyz, &writer);
  }
  sections.push_back(MakeSection(SectionType::kPoints, "points",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.observations.size());
  for (const auto& observation : snapshot.observations) {
    writer.UInt64(observation.source_index);
    writer.U32(observation.image_id);
    writer.U32(observation.point2D_idx);
    writer.UInt64(observation.point3D_id);
    writer.U8(observation.pose_constant);
    SerializeArray(observation.xy, &writer);
  }
  sections.push_back(MakeSection(SectionType::kObservations, "observations",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.tracks.size());
  for (const auto& track : snapshot.tracks) {
    writer.UInt64(track.point3D_id);
    writer.U32(track.image_id);
    writer.U32(track.point2D_idx);
  }
  sections.push_back(MakeSection(SectionType::kTracks, "tracks",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.lidar.size());
  for (const auto& lidar : snapshot.lidar) {
    writer.UInt64(lidar.source_index);
    writer.UInt64(lidar.point3D_id);
    writer.U8(lidar.lidar_type);
    writer.U8(lidar.has_search_range);
    writer.Double(lidar.search_range);
    writer.Double(lidar.weight);
    SerializeArray(lidar.lidar_xyz, &writer);
    SerializeArray(lidar.plane, &writer);
  }
  sections.push_back(MakeSection(SectionType::kLidar, "lidar",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.parameter_blocks_source_order.size());
  for (const auto& parameter : snapshot.parameter_blocks_source_order) {
    writer.UInt64(parameter.source_index);
    writer.U8(static_cast<uint8_t>(parameter.kind));
    writer.UInt64(parameter.entity_id);
    writer.U32(parameter.ambient_size);
    writer.U32(parameter.tangent_size);
    writer.U8(parameter.constant);
  }
  sections.push_back(MakeSection(SectionType::kParameterBlocks,
                                 "parameter_blocks_source_order",
                                 writer.MoveData()));

  writer = BinaryWriter();
  writer.UInt64(snapshot.parameter_blocks_canonical_order.size());
  for (const uint64_t index : snapshot.parameter_blocks_canonical_order) {
    writer.UInt64(index);
  }
  sections.push_back(MakeSection(SectionType::kParameterCanonicalOrder,
                                 "parameter_blocks_canonical_order",
                                 writer.MoveData()));

  auto SerializeOrder = [](const std::vector<OrderEntrySnapshot>& order) {
    BinaryWriter order_writer;
    order_writer.UInt64(order.size());
    for (const auto& entry : order) {
      order_writer.UInt64(entry.source_index);
      order_writer.U8(static_cast<uint8_t>(entry.residual_kind));
      order_writer.U32(entry.image_id);
      order_writer.U32(entry.point2D_idx);
      order_writer.UInt64(entry.point3D_id);
    }
    return order_writer.MoveData();
  };
  sections.push_back(MakeSection(SectionType::kSourceOrder, "source_order",
                                 SerializeOrder(snapshot.source_insertion_order)));
  sections.push_back(MakeSection(SectionType::kCanonicalOrder, "canonical_order",
                                 SerializeOrder(snapshot.canonical_order)));
  return sections;
}

const Section* FindSection(const std::vector<Section>& sections,
                           SectionType type) {
  for (const auto& section : sections) {
    if (section.type == type) return &section;
  }
  return nullptr;
}

bool ReadCount(BinaryReader* reader, uint64_t* count) {
  constexpr uint64_t kMaximumRecordCount = 1000000000ull;
  return reader->UInt64(count) && *count <= kMaximumRecordCount;
}

bool ParseSections(const std::vector<Section>& sections,
                   Snapshot* snapshot,
                   std::string* error) {
  const std::array<SectionType, 11> required{{
      SectionType::kMetadata, SectionType::kCameras, SectionType::kImages,
      SectionType::kPoints, SectionType::kObservations, SectionType::kTracks,
      SectionType::kLidar, SectionType::kParameterBlocks,
      SectionType::kParameterCanonicalOrder, SectionType::kSourceOrder,
      SectionType::kCanonicalOrder}};
  for (const auto type : required) {
    if (FindSection(sections, type) == nullptr) {
      *error = "Missing required payload section type " +
               std::to_string(static_cast<uint32_t>(type));
      return false;
    }
  }

  BinaryReader reader(FindSection(sections, SectionType::kMetadata)->data);
  if (!DeserializeMetadata(&reader, &snapshot->metadata)) {
    *error = "Invalid metadata section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kCameras)->data);
  uint64_t count = 0;
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid camera count";
    return false;
  }
  snapshot->cameras.resize(count);
  for (auto& camera : snapshot->cameras) {
    uint8_t constant = 0;
    uint64_t num_params = 0;
    if (!reader.U32(&camera.camera_id) || !reader.I32(&camera.model_id) ||
        !reader.UInt64(&camera.width) || !reader.UInt64(&camera.height) ||
        !reader.U8(&constant) || constant > 1 ||
        !ReadCount(&reader, &num_params)) {
      *error = "Invalid camera record";
      return false;
    }
    camera.constant = constant;
    camera.params.resize(num_params);
    for (double& param : camera.params) {
      if (!reader.Double(&param)) {
        *error = "Truncated camera parameters";
        return false;
      }
    }
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in camera section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kImages)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid image count";
    return false;
  }
  snapshot->images.resize(count);
  for (auto& image : snapshot->images) {
    uint8_t selected = 0;
    uint8_t constant = 0;
    uint8_t has_blocks = 0;
    if (!reader.U32(&image.image_id) || !reader.U32(&image.camera_id) ||
        !reader.U8(&selected) || !reader.U8(&constant) ||
        !reader.U8(&has_blocks) || !reader.U8(&image.constant_tvec_mask) ||
        selected > 1 || constant > 1 || has_blocks > 1 ||
        !DeserializeArray(&reader, &image.qvec) ||
        !DeserializeArray(&reader, &image.tvec)) {
      *error = "Invalid image record";
      return false;
    }
    image.selected = selected;
    image.pose_constant = constant;
    image.has_pose_parameter_blocks = has_blocks;
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in image section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kPoints)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid point count";
    return false;
  }
  snapshot->points.resize(count);
  for (auto& point : snapshot->points) {
    uint8_t constant = 0;
    uint8_t has_range = 0;
    if (!reader.UInt64(&point.point3D_id) || !reader.U8(&constant) ||
        !reader.U8(&point.config_role) || !reader.U8(&has_range) ||
        constant > 1 || has_range > 1 ||
        !reader.Double(&point.search_range) ||
        !DeserializeArray(&reader, &point.xyz)) {
      *error = "Invalid point record";
      return false;
    }
    point.constant = constant;
    point.has_search_range = has_range;
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in point section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kObservations)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid observation count";
    return false;
  }
  snapshot->observations.resize(count);
  for (auto& observation : snapshot->observations) {
    uint8_t constant = 0;
    if (!reader.UInt64(&observation.source_index) ||
        !reader.U32(&observation.image_id) ||
        !reader.U32(&observation.point2D_idx) ||
        !reader.UInt64(&observation.point3D_id) || !reader.U8(&constant) ||
        constant > 1 || !DeserializeArray(&reader, &observation.xy)) {
      *error = "Invalid observation record";
      return false;
    }
    observation.pose_constant = constant;
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in observation section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kTracks)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid track count";
    return false;
  }
  snapshot->tracks.resize(count);
  for (auto& track : snapshot->tracks) {
    if (!reader.UInt64(&track.point3D_id) || !reader.U32(&track.image_id) ||
        !reader.U32(&track.point2D_idx)) {
      *error = "Invalid track record";
      return false;
    }
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in track section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kLidar)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid LiDAR count";
    return false;
  }
  snapshot->lidar.resize(count);
  for (auto& lidar : snapshot->lidar) {
    uint8_t has_range = 0;
    if (!reader.UInt64(&lidar.source_index) || !reader.UInt64(&lidar.point3D_id) ||
        !reader.U8(&lidar.lidar_type) || !reader.U8(&has_range) ||
        has_range > 1 || !reader.Double(&lidar.search_range) ||
        !reader.Double(&lidar.weight) ||
        !DeserializeArray(&reader, &lidar.lidar_xyz) ||
        !DeserializeArray(&reader, &lidar.plane)) {
      *error = "Invalid LiDAR record";
      return false;
    }
    lidar.has_search_range = has_range;
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in LiDAR section";
    return false;
  }

  reader = BinaryReader(FindSection(sections, SectionType::kParameterBlocks)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid parameter block count";
    return false;
  }
  snapshot->parameter_blocks_source_order.resize(count);
  for (auto& parameter : snapshot->parameter_blocks_source_order) {
    uint8_t kind = 0;
    uint8_t constant = 0;
    if (!reader.UInt64(&parameter.source_index) || !reader.U8(&kind) ||
        kind > static_cast<uint8_t>(ParameterKind::kCamera) ||
        !reader.UInt64(&parameter.entity_id) ||
        !reader.U32(&parameter.ambient_size) ||
        !reader.U32(&parameter.tangent_size) || !reader.U8(&constant) ||
        constant > 1) {
      *error = "Invalid parameter block record";
      return false;
    }
    parameter.kind = static_cast<ParameterKind>(kind);
    parameter.constant = constant;
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in parameter block section";
    return false;
  }

  reader = BinaryReader(
      FindSection(sections, SectionType::kParameterCanonicalOrder)->data);
  if (!ReadCount(&reader, &count)) {
    *error = "Invalid canonical parameter count";
    return false;
  }
  snapshot->parameter_blocks_canonical_order.resize(count);
  for (uint64_t& index : snapshot->parameter_blocks_canonical_order) {
    if (!reader.UInt64(&index)) {
      *error = "Invalid canonical parameter order";
      return false;
    }
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes in canonical parameter section";
    return false;
  }

  auto ParseOrder = [error](const Section& section,
                            std::vector<OrderEntrySnapshot>* order) {
    BinaryReader order_reader(section.data);
    uint64_t order_count = 0;
    if (!ReadCount(&order_reader, &order_count)) {
      *error = "Invalid residual order count";
      return false;
    }
    order->resize(order_count);
    for (auto& entry : *order) {
      uint8_t kind = 0;
      if (!order_reader.UInt64(&entry.source_index) || !order_reader.U8(&kind) ||
          kind > static_cast<uint8_t>(ResidualKind::kLidar) ||
          !order_reader.U32(&entry.image_id) ||
          !order_reader.U32(&entry.point2D_idx) ||
          !order_reader.UInt64(&entry.point3D_id)) {
        *error = "Invalid residual order record";
        return false;
      }
      entry.residual_kind = static_cast<ResidualKind>(kind);
    }
    if (!order_reader.AtEnd()) {
      *error = "Trailing bytes in residual order section";
      return false;
    }
    return true;
  };
  if (!ParseOrder(*FindSection(sections, SectionType::kSourceOrder),
                  &snapshot->source_insertion_order) ||
      !ParseOrder(*FindSection(sections, SectionType::kCanonicalOrder),
                  &snapshot->canonical_order)) {
    return false;
  }
  return true;
}

std::string RemoveSuffix(const std::string& value, const std::string& suffix) {
  if (value.size() >= suffix.size() &&
      value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return value.substr(0, value.size() - suffix.size());
  }
  return value;
}

bool ExtractJsonString(const std::string& json,
                       const std::string& key,
                       std::string* value) {
  const std::string token = "\"" + key + "\":\"";
  const size_t start = json.find(token);
  if (start == std::string::npos) return false;
  size_t pos = start + token.size();
  std::ostringstream decoded;
  bool escaped = false;
  for (; pos < json.size(); ++pos) {
    const char ch = json[pos];
    if (escaped) {
      switch (ch) {
        case '\\': decoded << '\\'; break;
        case '"': decoded << '"'; break;
        case 'n': decoded << '\n'; break;
        case 'r': decoded << '\r'; break;
        case 't': decoded << '\t'; break;
        default: decoded << ch; break;
      }
      escaped = false;
    } else if (ch == '\\') {
      escaped = true;
    } else if (ch == '"') {
      *value = decoded.str();
      return true;
    } else {
      decoded << ch;
    }
  }
  return false;
}

bool ExtractManifestCore(const std::string& manifest, std::string* core) {
  const std::string token = "\"manifest_core\":";
  const size_t key_pos = manifest.find(token);
  if (key_pos == std::string::npos) return false;
  const size_t start = manifest.find('{', key_pos + token.size());
  if (start == std::string::npos) return false;
  size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t pos = start; pos < manifest.size(); ++pos) {
    const char ch = manifest[pos];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      if (--depth == 0) {
        *core = manifest.substr(start, pos - start + 1);
        return true;
      }
    }
  }
  return false;
}

}  // namespace

std::string BaKindName(BaKind kind) {
  switch (kind) {
    case BaKind::kLocal: return "local";
    case BaKind::kGlobal: return "global";
    case BaKind::kWhole: return "whole";
  }
  return "unknown";
}

bool ParseBaKind(const std::string& value, BaKind* kind) {
  if (value == "local") {
    *kind = BaKind::kLocal;
  } else if (value == "global") {
    *kind = BaKind::kGlobal;
  } else if (value == "whole") {
    *kind = BaKind::kWhole;
  } else {
    return false;
  }
  return true;
}

uint32_t Crc32(const std::vector<uint8_t>& data) {
  uint32_t crc = 0xffffffffu;
  for (const uint8_t byte : data) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = 0u - (crc & 1u);
      crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
  }
  return ~crc;
}

std::string Sha256Hex(const std::vector<uint8_t>& data) {
  return Sha256Bytes(data.empty() ? nullptr : data.data(), data.size());
}

std::string Sha256Hex(const std::string& data) {
  return Sha256Bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

bool ValidateSnapshot(const Snapshot& snapshot, std::string* error) {
  if (error == nullptr) return false;
  error->clear();
  if (snapshot.metadata.snapshot_id.empty()) {
    *error = "Snapshot ID is empty";
    return false;
  }

  std::set<uint32_t> camera_ids;
  for (const auto& camera : snapshot.cameras) {
    if (!camera_ids.insert(camera.camera_id).second || camera.params.empty()) {
      *error = "Duplicate camera ID or empty camera parameter block";
      return false;
    }
  }
  std::set<uint32_t> image_ids;
  for (const auto& image : snapshot.images) {
    if (!image_ids.insert(image.image_id).second ||
        camera_ids.count(image.camera_id) == 0) {
      *error = "Duplicate image ID or missing image camera";
      return false;
    }
  }
  std::set<uint64_t> point_ids;
  for (const auto& point : snapshot.points) {
    if (!point_ids.insert(point.point3D_id).second ||
        (point.has_search_range &&
         (!std::isfinite(point.search_range) || point.search_range < 0.0))) {
      *error = "Duplicate point ID or invalid search range";
      return false;
    }
  }

  std::set<std::tuple<uint64_t, uint32_t, uint32_t>> tracks;
  for (const auto& track : snapshot.tracks) {
    const auto key =
        std::make_tuple(track.point3D_id, track.image_id, track.point2D_idx);
    if (point_ids.count(track.point3D_id) == 0 ||
        image_ids.count(track.image_id) == 0 || !tracks.insert(key).second) {
      *error = "Invalid or duplicate track element";
      return false;
    }
  }

  const size_t num_residuals = snapshot.source_insertion_order.size();
  if (snapshot.canonical_order.size() != num_residuals ||
      snapshot.observations.size() + snapshot.lidar.size() != num_residuals) {
    *error = "Residual and order counts differ";
    return false;
  }
  std::vector<uint8_t> residual_records(num_residuals, 0);
  for (size_t i = 0; i < num_residuals; ++i) {
    if (snapshot.source_insertion_order[i].source_index != i) {
      *error = "Source residual indices are not contiguous";
      return false;
    }
  }
  for (const auto& observation : snapshot.observations) {
    if (observation.source_index >= num_residuals ||
        residual_records[observation.source_index]++ != 0 ||
        image_ids.count(observation.image_id) == 0 ||
        point_ids.count(observation.point3D_id) == 0 ||
        tracks.count(std::make_tuple(observation.point3D_id,
                                     observation.image_id,
                                     observation.point2D_idx)) == 0) {
      *error = "Invalid visual residual reference";
      return false;
    }
    const auto& order = snapshot.source_insertion_order[observation.source_index];
    if (order.residual_kind != ResidualKind::kVisual ||
        order.image_id != observation.image_id ||
        order.point2D_idx != observation.point2D_idx ||
        order.point3D_id != observation.point3D_id) {
      *error = "Visual residual disagrees with source order";
      return false;
    }
  }
  for (const auto& lidar : snapshot.lidar) {
    if (lidar.source_index >= num_residuals ||
        residual_records[lidar.source_index]++ != 0 ||
        point_ids.count(lidar.point3D_id) == 0 ||
        (lidar.has_search_range &&
         (!std::isfinite(lidar.search_range) || lidar.search_range < 0.0))) {
      *error = "Invalid LiDAR residual reference";
      return false;
    }
    const auto& order = snapshot.source_insertion_order[lidar.source_index];
    if (order.residual_kind != ResidualKind::kLidar ||
        order.point3D_id != lidar.point3D_id) {
      *error = "LiDAR residual disagrees with source order";
      return false;
    }
  }
  if (std::find(residual_records.begin(), residual_records.end(), 0) !=
      residual_records.end()) {
    *error = "Source order contains an unreferenced residual";
    return false;
  }

  std::vector<uint8_t> canonical_residuals(num_residuals, 0);
  auto ResidualLess = [](const OrderEntrySnapshot& lhs,
                         const OrderEntrySnapshot& rhs) {
    return std::tie(lhs.residual_kind, lhs.image_id, lhs.point2D_idx,
                    lhs.point3D_id, lhs.source_index) <
           std::tie(rhs.residual_kind, rhs.image_id, rhs.point2D_idx,
                    rhs.point3D_id, rhs.source_index);
  };
  for (size_t i = 0; i < snapshot.canonical_order.size(); ++i) {
    const auto& entry = snapshot.canonical_order[i];
    if (entry.source_index >= num_residuals ||
        canonical_residuals[entry.source_index]++ != 0 ||
        (i > 0 && ResidualLess(entry, snapshot.canonical_order[i - 1]))) {
      *error = "Canonical residual order is invalid";
      return false;
    }
    const auto& source = snapshot.source_insertion_order[entry.source_index];
    if (std::tie(entry.residual_kind, entry.image_id, entry.point2D_idx,
                 entry.point3D_id) !=
        std::tie(source.residual_kind, source.image_id, source.point2D_idx,
                 source.point3D_id)) {
      *error = "Canonical residual entry differs from source entry";
      return false;
    }
  }

  const size_t num_parameters = snapshot.parameter_blocks_source_order.size();
  if (snapshot.parameter_blocks_canonical_order.size() != num_parameters) {
    *error = "Parameter order counts differ";
    return false;
  }
  std::vector<uint8_t> canonical_parameters(num_parameters, 0);
  for (size_t i = 0; i < num_parameters; ++i) {
    const auto& parameter = snapshot.parameter_blocks_source_order[i];
    if (parameter.source_index != i || parameter.ambient_size == 0) {
      *error = "Invalid source parameter block";
      return false;
    }
    const bool entity_exists =
        (parameter.kind == ParameterKind::kCamera &&
         camera_ids.count(static_cast<uint32_t>(parameter.entity_id)) != 0) ||
        ((parameter.kind == ParameterKind::kQuaternion ||
          parameter.kind == ParameterKind::kTranslation) &&
         image_ids.count(static_cast<uint32_t>(parameter.entity_id)) != 0) ||
        (parameter.kind == ParameterKind::kPoint3D &&
         point_ids.count(parameter.entity_id) != 0);
    if (!entity_exists) {
      *error = "Parameter block references a missing entity";
      return false;
    }
  }
  for (size_t i = 0; i < num_parameters; ++i) {
    const uint64_t index = snapshot.parameter_blocks_canonical_order[i];
    if (index >= num_parameters || canonical_parameters[index]++ != 0) {
      *error = "Canonical parameter order is not a permutation";
      return false;
    }
    if (i > 0) {
      const auto& previous = snapshot.parameter_blocks_source_order[
          snapshot.parameter_blocks_canonical_order[i - 1]];
      const auto& current = snapshot.parameter_blocks_source_order[index];
      if (std::tie(current.kind, current.entity_id, current.source_index) <
          std::tie(previous.kind, previous.entity_id, previous.source_index)) {
        *error = "Canonical parameter order is not sorted";
        return false;
      }
    }
  }
  return true;
}

bool WriteSnapshot(const Snapshot& snapshot,
                   const std::string& output_dir,
                   SnapshotWriteResult* result,
                   std::string* error) {
  if (result == nullptr || error == nullptr) return false;
  error->clear();
  if (snapshot.metadata.snapshot_id.empty()) {
    *error = "Snapshot ID must not be empty";
    return false;
  }
  if (!ValidateSnapshot(snapshot, error)) return false;
  if (output_dir.empty()) {
    *error = "Snapshot output directory must not be empty";
    return false;
  }
  CreateDirIfNotExists(output_dir, true);
  const std::string prefix = JoinPaths(output_dir, snapshot.metadata.snapshot_id);
  const std::string payload_path = prefix + ".payload.bin";
  const std::string manifest_path = prefix + ".manifest.json";
  if (ExistsFile(payload_path) || ExistsFile(manifest_path)) {
    *error = "Refusing to overwrite existing snapshot: " + prefix;
    return false;
  }

  const std::vector<Section> sections = BuildSections(snapshot);
  const std::string manifest_core = BuildManifestCore(snapshot, sections);
  const std::string manifest_core_sha256 = Sha256Hex(manifest_core);
  std::vector<uint8_t> manifest_core_digest;
  if (!HexToBytes(manifest_core_sha256, &manifest_core_digest) ||
      manifest_core_digest.size() != SHA256_DIGEST_LENGTH) {
    *error = "Internal SHA256 encoding failure";
    return false;
  }

  BinaryWriter payload;
  payload.Raw(reinterpret_cast<const uint8_t*>(kPayloadMagic),
              sizeof(kPayloadMagic));
  payload.U32(kSnapshotSchemaVersion);
  payload.U32(kSnapshotLittleEndianMarker);
  payload.U32(64);
  payload.U32(32);
  payload.U32(64);
  payload.U32(sections.size());
  payload.Raw(manifest_core_digest);
  for (const auto& section : sections) {
    payload.U32(static_cast<uint32_t>(section.type));
    payload.U32(1);
    payload.UInt64(section.data.size());
    payload.U32(section.crc32);
    payload.U32(section.name.size());
    payload.Raw(reinterpret_cast<const uint8_t*>(section.name.data()),
                section.name.size());
    payload.Raw(section.data);
  }
  const std::vector<uint8_t> payload_data = payload.Data();
  const std::string payload_sha256 = Sha256Hex(payload_data);

  std::ostringstream manifest;
  manifest << "{\"manifest_core_sha256\":\"" << manifest_core_sha256
           << "\",\"payload_sha256\":\"" << payload_sha256
           << "\",\"hash_domain\":\"manifest_core is hashed exactly as the "
              "embedded JSON object; payload_sha256 covers the complete binary "
              "file\",\"manifest_core\":" << manifest_core << "}\n";
  const std::string manifest_data = manifest.str();
  const std::string manifest_sha256 = Sha256Hex(manifest_data);
  const std::vector<uint8_t> manifest_bytes(manifest_data.begin(),
                                            manifest_data.end());

  const std::string payload_tmp = payload_path + ".tmp";
  const std::string manifest_tmp = manifest_path + ".tmp";
  std::remove(payload_tmp.c_str());
  std::remove(manifest_tmp.c_str());
  if (!WriteFile(payload_tmp, payload_data, error) ||
      !WriteFile(manifest_tmp, manifest_bytes, error)) {
    std::remove(payload_tmp.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }
  if (std::rename(payload_tmp.c_str(), payload_path.c_str()) != 0) {
    *error = "Cannot finalize payload: " + std::string(std::strerror(errno));
    std::remove(payload_tmp.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }
  if (std::rename(manifest_tmp.c_str(), manifest_path.c_str()) != 0) {
    *error = "Cannot finalize manifest: " + std::string(std::strerror(errno));
    std::remove(payload_path.c_str());
    std::remove(manifest_tmp.c_str());
    return false;
  }

  const Section* lidar = FindSection(sections, SectionType::kLidar);
  const Section* source = FindSection(sections, SectionType::kSourceOrder);
  const Section* canonical = FindSection(sections, SectionType::kCanonicalOrder);
  result->prefix_path = prefix;
  result->manifest_path = manifest_path;
  result->payload_path = payload_path;
  result->integrity.manifest_core_sha256 = manifest_core_sha256;
  result->integrity.manifest_sha256 = manifest_sha256;
  result->integrity.payload_sha256 = payload_sha256;
  result->integrity.lidar_correspondence_sha256 = lidar->sha256;
  result->integrity.source_order_sha256 = source->sha256;
  result->integrity.canonical_order_sha256 = canonical->sha256;
  result->integrity.payload_bytes = payload_data.size();
  return true;
}

bool ReadSnapshot(const std::string& snapshot_path,
                  Snapshot* snapshot,
                  SnapshotReadResult* result,
                  std::string* error) {
  if (snapshot == nullptr || result == nullptr || error == nullptr) return false;
  error->clear();
  std::string prefix = RemoveSuffix(snapshot_path, ".manifest.json");
  prefix = RemoveSuffix(prefix, ".payload.bin");
  const std::string payload_path = prefix + ".payload.bin";
  const std::string manifest_path = prefix + ".manifest.json";

  std::vector<uint8_t> payload_data;
  std::vector<uint8_t> manifest_bytes;
  if (!ReadFile(payload_path, &payload_data, error) ||
      !ReadFile(manifest_path, &manifest_bytes, error)) {
    return false;
  }
  if (payload_data.size() < 8 + 6 * 4 + SHA256_DIGEST_LENGTH) {
    *error = "Payload header is truncated";
    return false;
  }

  BinaryReader reader(payload_data);
  std::vector<uint8_t> magic;
  uint32_t schema_version = 0;
  uint32_t endian_marker = 0;
  uint32_t float_bits = 0;
  uint32_t int_bits = 0;
  uint32_t id_bits = 0;
  uint32_t section_count = 0;
  std::vector<uint8_t> manifest_core_digest;
  if (!reader.Raw(sizeof(kPayloadMagic), &magic) ||
      std::memcmp(magic.data(), kPayloadMagic, sizeof(kPayloadMagic)) != 0) {
    *error = "Invalid payload magic";
    return false;
  }
  if (!reader.U32(&schema_version) || schema_version != kSnapshotSchemaVersion) {
    *error = "Unsupported snapshot schema version";
    return false;
  }
  if (!reader.U32(&endian_marker) ||
      endian_marker != kSnapshotLittleEndianMarker) {
    *error = "Snapshot is not encoded as little-endian";
    return false;
  }
  if (!reader.U32(&float_bits) || !reader.U32(&int_bits) ||
      !reader.U32(&id_bits) || float_bits != 64 || int_bits != 32 ||
      id_bits != 64) {
    *error = "Unsupported snapshot data type description";
    return false;
  }
  if (!reader.U32(&section_count) || section_count != 11 ||
      !reader.Raw(SHA256_DIGEST_LENGTH, &manifest_core_digest)) {
    *error = "Invalid snapshot section table";
    return false;
  }

  std::vector<Section> sections;
  std::set<uint32_t> seen_types;
  for (uint32_t i = 0; i < section_count; ++i) {
    uint32_t type = 0;
    uint32_t version = 0;
    uint64_t length = 0;
    uint32_t expected_crc = 0;
    uint32_t name_length = 0;
    if (!reader.U32(&type) || !reader.U32(&version) ||
        !reader.UInt64(&length) || !reader.U32(&expected_crc) ||
        !reader.U32(&name_length) || version != 1 ||
        name_length > reader.Remaining() || length > reader.Remaining() ||
        !seen_types.insert(type).second) {
      *error = "Invalid or duplicate payload section header";
      return false;
    }
    std::vector<uint8_t> name_bytes;
    std::vector<uint8_t> data;
    if (!reader.Raw(name_length, &name_bytes) || length > reader.Remaining() ||
        !reader.Raw(static_cast<size_t>(length), &data)) {
      *error = "Truncated payload section";
      return false;
    }
    if (Crc32(data) != expected_crc) {
      *error = "CRC32 mismatch in payload section";
      return false;
    }
    Section section;
    section.type = static_cast<SectionType>(type);
    section.name.assign(name_bytes.begin(), name_bytes.end());
    section.data = std::move(data);
    section.crc32 = expected_crc;
    section.sha256 = Sha256Hex(section.data);
    sections.push_back(std::move(section));
  }
  if (!reader.AtEnd()) {
    *error = "Trailing bytes after payload section table";
    return false;
  }

  const std::string manifest(manifest_bytes.begin(), manifest_bytes.end());
  std::string expected_core_sha;
  std::string expected_payload_sha;
  std::string manifest_core;
  if (!ExtractJsonString(manifest, "manifest_core_sha256", &expected_core_sha) ||
      !ExtractJsonString(manifest, "payload_sha256", &expected_payload_sha) ||
      !ExtractManifestCore(manifest, &manifest_core)) {
    *error = "Invalid snapshot manifest";
    return false;
  }
  const std::string actual_core_sha = Sha256Hex(manifest_core);
  const std::string actual_payload_sha = Sha256Hex(payload_data);
  const std::string header_core_sha =
      Hex(manifest_core_digest.data(), manifest_core_digest.size());
  if (expected_core_sha != actual_core_sha || header_core_sha != actual_core_sha) {
    *error = "Manifest core SHA256 cross-reference mismatch";
    return false;
  }
  if (expected_payload_sha != actual_payload_sha) {
    *error = "Payload SHA256 mismatch";
    return false;
  }

  Snapshot parsed;
  if (!ParseSections(sections, &parsed, error)) return false;
  if (!ValidateSnapshot(parsed, error)) return false;
  if (manifest_core.find("\"snapshot_id\":\"" +
                         JsonEscape(parsed.metadata.snapshot_id) + "\"") ==
      std::string::npos) {
    *error = "Manifest and payload snapshot IDs differ";
    return false;
  }

  const Section* lidar = FindSection(sections, SectionType::kLidar);
  const Section* source = FindSection(sections, SectionType::kSourceOrder);
  const Section* canonical = FindSection(sections, SectionType::kCanonicalOrder);
  *snapshot = std::move(parsed);
  result->prefix_path = prefix;
  result->manifest_path = manifest_path;
  result->payload_path = payload_path;
  result->integrity.manifest_core_sha256 = actual_core_sha;
  result->integrity.manifest_sha256 = Sha256Hex(manifest);
  result->integrity.payload_sha256 = actual_payload_sha;
  result->integrity.lidar_correspondence_sha256 = lidar->sha256;
  result->integrity.source_order_sha256 = source->sha256;
  result->integrity.canonical_order_sha256 = canonical->sha256;
  result->integrity.payload_bytes = payload_data.size();
  return true;
}

}  // namespace gpu_ba
}  // namespace colmap
