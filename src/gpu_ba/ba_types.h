#ifndef COLMAP_SRC_GPU_BA_BA_TYPES_H_
#define COLMAP_SRC_GPU_BA_BA_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

namespace colmap {
namespace gpu_ba {

enum class BaKind : uint8_t { kLocal = 0, kGlobal = 1, kWhole = 2 };
enum class ResidualKind : uint8_t { kVisual = 0, kLidar = 1 };
enum class ParameterKind : uint8_t {
  kQuaternion = 0,
  kTranslation = 1,
  kPoint3D = 2,
  kCamera = 3,
};

struct ParameterIdentityKey {
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;

  bool operator==(const ParameterIdentityKey& other) const noexcept {
    return kind == other.kind && entity_id == other.entity_id;
  }
};

struct ParameterIdentityKeyHash {
  size_t operator()(const ParameterIdentityKey& value) const noexcept {
    const size_t entity_hash = std::hash<uint64_t>()(value.entity_id);
    const size_t kind_hash =
        std::hash<uint8_t>()(static_cast<uint8_t>(value.kind));
    return entity_hash ^ (kind_hash + 0x9e3779b9u + (entity_hash << 6) +
                          (entity_hash >> 2));
  }
};

}  // namespace gpu_ba
}  // namespace colmap

#endif  // COLMAP_SRC_GPU_BA_BA_TYPES_H_
