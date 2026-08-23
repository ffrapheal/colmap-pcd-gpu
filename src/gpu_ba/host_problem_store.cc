#include "gpu_ba/host_problem_store_internal.h"

#include <algorithm>
#include <new>

#include "base/reconstruction.h"

namespace colmap {
namespace gpu_ba {
namespace {

uint64_t SaturatingAddBytes(const uint64_t lhs, const uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs
      ? std::numeric_limits<uint64_t>::max()
      : lhs + rhs;
}

uint64_t SaturatingMulBytes(const uint64_t lhs, const uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs
      ? std::numeric_limits<uint64_t>::max()
      : lhs * rhs;
}

uint64_t EstimateCatalogResidentBytes(const StaticProblemDataCatalog& value) {
  uint64_t bytes = sizeof(value);
  bytes = SaturatingAddBytes(
      bytes, SaturatingMulBytes(value.cameras.size(),
                                sizeof(HostCatalogCamera)));
  bytes = SaturatingAddBytes(
      bytes, SaturatingMulBytes(value.images.size(),
                                sizeof(HostCatalogImage)));
  bytes = SaturatingAddBytes(
      bytes, SaturatingMulBytes(value.points.size(),
                                sizeof(HostCatalogPoint)));
  bytes = SaturatingAddBytes(
      bytes, SaturatingMulBytes(value.observations.size(),
                                sizeof(HostCatalogObservation)));
  for (const auto& item : value.points) {
    bytes = SaturatingAddBytes(
        bytes, SaturatingMulBytes(item.second.track.size(),
                                  sizeof(HostCatalogTrackElement)));
  }
  bytes = SaturatingAddBytes(
      bytes,
      SaturatingMulBytes(
          SaturatingAddBytes(
              SaturatingAddBytes(value.deleted_cameras.size(),
                                 value.deleted_images.size()),
              SaturatingAddBytes(value.deleted_points.size(),
                                 value.deleted_observations.size())),
          sizeof(uint64_t)));
  return bytes;
}

HostCatalogTrackElement ConvertTrackElement(const TrackElement& input) {
  HostCatalogTrackElement output;
  output.image_id = input.image_id;
  output.point2D_idx = input.point2D_idx;
  return output;
}

bool AddCatalogPointFromReconstruction(
    const Reconstruction& reconstruction,
    const uint64_t point3D_id,
    StaticProblemDataCatalog* catalog,
    std::string* error) {
  if (!reconstruction.ExistsPoint3D(point3D_id)) {
    *error = "catalog delta references a missing Point3D: point=" +
             std::to_string(point3D_id);
    return false;
  }
  const Point3D& point = reconstruction.Point3D(point3D_id);
  HostCatalogPoint output;
  output.point3D_id = point3D_id;
  output.track.reserve(point.Track().Length());
  for (const TrackElement& element : point.Track().Elements()) {
    output.track.push_back(ConvertTrackElement(element));
  }
  catalog->deleted_points.erase(point3D_id);
  catalog->points[point3D_id] = output;
  for (const HostCatalogTrackElement& element : output.track) {
    const bool image_exists = reconstruction.ExistsImage(element.image_id);
    const size_t point2D_count = image_exists
        ? reconstruction.Image(element.image_id).NumPoints2D() : 0;
    if (!image_exists || element.point2D_idx >= point2D_count) {
      *error = "catalog Point3D track references a missing observation: point=" +
          std::to_string(point3D_id) + " image=" +
          std::to_string(element.image_id) + " point2D=" +
          std::to_string(element.point2D_idx) + " exists=" +
          std::to_string(image_exists) + " num_points2D=" +
          std::to_string(point2D_count);
      return false;
    }
    const Point2D& point2D =
        reconstruction.Image(element.image_id).Point2D(element.point2D_idx);
    if (!point2D.HasPoint3D() || point2D.Point3DId() != point3D_id) {
      *error = "catalog Point3D track is not reciprocal: point=" +
          std::to_string(point3D_id) + " image=" +
          std::to_string(element.image_id) + " point2D=" +
          std::to_string(element.point2D_idx);
      return false;
    }
    HostCatalogObservation observation;
    observation.point3D_id = point3D_id;
    observation.image_id = element.image_id;
    observation.point2D_idx = element.point2D_idx;
    observation.xy = {{point2D.X(), point2D.Y()}};
    const uint64_t key = HostCatalogObservationKey(
        observation.image_id, observation.point2D_idx);
    catalog->deleted_observations.erase(key);
    catalog->observations[key] = observation;
  }
  return true;
}

const HostCatalogPoint* FindCatalogPointIncludingDelta(
    const StaticProblemDataCatalog& delta,
    const std::shared_ptr<const StaticProblemDataCatalog>& base,
    const uint64_t id) {
  if (delta.deleted_points.count(id) != 0) return nullptr;
  const auto local = delta.points.find(id);
  if (local != delta.points.end()) return &local->second;
  return FindCatalogPoint(base, id);
}

HostStructureEvent ConvertStructureEvent(
    const ReconstructionStructureEvent& input) {
  HostStructureEvent output;
  output.kind = static_cast<HostStructureEventKind>(input.kind);
  output.image_id = input.image_id;
  output.old_camera_id = input.old_camera_id;
  output.new_camera_id = input.new_camera_id;
  output.point3D_id = input.point3D_id;
  output.old_point3D_id1 = input.old_point3D_id1;
  output.old_point3D_id2 = input.old_point3D_id2;
  output.point2D_idx = input.point2D_idx;
  output.old_registration = input.old_registration;
  output.new_registration = input.new_registration;
  output.old_track.reserve(input.old_track.size());
  for (const TrackElement& element : input.old_track) {
    output.old_track.push_back(ConvertTrackElement(element));
  }
  output.reason = input.reason;
  return output;
}

}  // namespace

uint64_t HostReconstructionOwnerEpoch(
    const Reconstruction* reconstruction) noexcept {
  return reconstruction == nullptr ? 0 : reconstruction->StructureOwnerEpoch();
}

uint64_t HostReconstructionStructureRevision(
    const Reconstruction* reconstruction) noexcept {
  return reconstruction == nullptr ? 0 : reconstruction->StructureRevision();
}

bool ReadHostStructureEventsSince(
    const Reconstruction* reconstruction,
    const uint64_t owner_epoch,
    const uint64_t cursor,
    HostStructureReadResult* result,
    std::string* error) {
  if (reconstruction == nullptr || result == nullptr || error == nullptr) {
    if (error != nullptr) *error = "invalid host structure journal arguments";
    return false;
  }
  try {
    const ReconstructionStructureReadResult input =
        reconstruction->ReadStructureEventsSince(owner_epoch, cursor);
    HostStructureReadResult output;
    output.complete = input.complete;
    output.gap = input.gap;
    output.owner_epoch = input.owner_epoch;
    output.cursor = input.cursor;
    output.current_revision = input.current_revision;
    output.oldest_retained_revision = input.oldest_retained_revision;
    output.batches.reserve(input.batches.size());
    for (const ReconstructionStructureBatch& input_batch : input.batches) {
      HostStructureBatch output_batch;
      output_batch.revision = input_batch.revision;
      output_batch.events.reserve(input_batch.events.size());
      for (const ReconstructionStructureEvent& input_event :
           input_batch.events) {
        output_batch.events.push_back(ConvertStructureEvent(input_event));
      }
      output.batches.push_back(std::move(output_batch));
    }
    *result = std::move(output);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "host structure journal copy allocation failed";
    return false;
  }
}

bool BuildStaticProblemDataCatalog(
    const Reconstruction* reconstruction,
    const uint64_t owner_epoch,
    const uint64_t revision,
    const uint64_t generation,
    std::shared_ptr<const StaticProblemDataCatalog>* output,
    std::string* error) {
  if (reconstruction == nullptr || output == nullptr || error == nullptr ||
      owner_epoch == 0 || revision == 0 || generation == 0 ||
      reconstruction->StructureOwnerEpoch() != owner_epoch ||
      reconstruction->StructureRevision() != revision) {
    if (error != nullptr) *error = "invalid static catalog build arguments";
    return false;
  }
  try {
    std::shared_ptr<StaticProblemDataCatalog> catalog(
        new StaticProblemDataCatalog());
    catalog->owner_epoch = owner_epoch;
    catalog->revision = revision;
    catalog->generation = generation;
    catalog->cameras.reserve(reconstruction->NumCameras());
    catalog->images.reserve(reconstruction->NumImages());
    catalog->points.reserve(reconstruction->NumPoints3D());
    for (const auto& item : reconstruction->Cameras()) {
      const Camera& camera = item.second;
      HostCatalogCamera value;
      value.camera_id = camera.CameraId();
      value.model_id = camera.ModelId();
      value.width = camera.Width();
      value.height = camera.Height();
      value.parameter_count = camera.Params().size();
      if (!catalog->cameras.emplace(value.camera_id, value).second) {
        *error = "duplicate camera in static catalog";
        return false;
      }
    }
    for (const auto& item : reconstruction->Images()) {
      const Image& image = item.second;
      HostCatalogImage value;
      value.image_id = image.ImageId();
      value.camera_id = image.CameraId();
      value.registered = image.IsRegistered();
      if (!catalog->images.emplace(value.image_id, value).second) {
        *error = "duplicate image in static catalog";
        return false;
      }
    }
    for (const auto& item : reconstruction->Points3D()) {
      if (!AddCatalogPointFromReconstruction(
              *reconstruction, item.first, catalog.get(), error)) {
        return false;
      }
    }
    catalog->resident_bytes = EstimateCatalogResidentBytes(*catalog);
    *output = std::move(catalog);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "static catalog build allocation failed";
    return false;
  }
}

bool ApplyStructureJournalToCatalog(
    const Reconstruction* reconstruction,
    const HostStructureReadResult& journal,
    const uint64_t generation,
    const std::shared_ptr<const StaticProblemDataCatalog>& base,
    std::shared_ptr<const StaticProblemDataCatalog>* output,
    std::string* error) {
  if (reconstruction == nullptr || base == nullptr || output == nullptr ||
      error == nullptr || generation == 0 || !journal.complete || journal.gap ||
      journal.owner_epoch != base->owner_epoch ||
      reconstruction->StructureOwnerEpoch() != base->owner_epoch ||
      reconstruction->StructureRevision() != journal.current_revision ||
      journal.cursor != base->revision ||
      journal.current_revision < journal.cursor) {
    if (error != nullptr) *error = "invalid static catalog delta arguments";
    return false;
  }
  try {
    uint64_t previous_revision = journal.cursor;
    for (const HostStructureBatch& batch : journal.batches) {
      if (batch.revision <= previous_revision ||
          batch.revision > journal.current_revision) {
        *error = "static catalog journal batch order is invalid";
        return false;
      }
      previous_revision = batch.revision;
    }
    if (previous_revision != journal.current_revision) {
      *error = "static catalog journal does not cover the current revision";
      return false;
    }
    std::shared_ptr<StaticProblemDataCatalog> delta(
        new StaticProblemDataCatalog());
    delta->owner_epoch = base->owner_epoch;
    delta->revision = journal.current_revision;
    delta->generation = generation;
    delta->overlay_depth = base->overlay_depth + 1;
    delta->parent = base;
    for (const HostStructureBatch& batch : journal.batches) {
      for (const HostStructureEvent& event : batch.events) {
        switch (event.kind) {
          case HostStructureEventKind::kCameraAdded: {
            if (!reconstruction->ExistsCamera(event.new_camera_id)) {
              *error = "camera-add delta references a missing camera";
              return false;
            }
            const Camera& camera = reconstruction->Camera(event.new_camera_id);
            HostCatalogCamera value;
            value.camera_id = camera.CameraId();
            value.model_id = camera.ModelId();
            value.width = camera.Width();
            value.height = camera.Height();
            value.parameter_count = camera.Params().size();
            delta->deleted_cameras.erase(value.camera_id);
            delta->cameras[value.camera_id] = value;
            break;
          }
          case HostStructureEventKind::kImageAdded:
          case HostStructureEventKind::kImageRegistrationChanged:
          case HostStructureEventKind::kCameraAssociationChanged: {
            if (!reconstruction->ExistsImage(event.image_id)) {
              *error = "image delta references a missing image";
              return false;
            }
            const Image& image = reconstruction->Image(event.image_id);
            HostCatalogImage value;
            value.image_id = image.ImageId();
            value.camera_id = image.CameraId();
            value.registered = image.IsRegistered();
            delta->deleted_images.erase(value.image_id);
            delta->images[value.image_id] = value;
            break;
          }
          case HostStructureEventKind::kPointAdded:
            if (!AddCatalogPointFromReconstruction(
                    *reconstruction, event.point3D_id, delta.get(), error)) {
              return false;
            }
            break;
          case HostStructureEventKind::kPointDeleted:
            delta->points.erase(event.point3D_id);
            delta->deleted_points.insert(event.point3D_id);
            for (const HostCatalogTrackElement& element : event.old_track) {
              const uint64_t key = HostCatalogObservationKey(
                  element.image_id, element.point2D_idx);
              delta->observations.erase(key);
              delta->deleted_observations.insert(key);
            }
            break;
          case HostStructureEventKind::kPointMerged:
            for (const uint64_t old_id :
                 {event.old_point3D_id1, event.old_point3D_id2}) {
              const HostCatalogPoint* old_point =
                  FindCatalogPointIncludingDelta(*delta, base, old_id);
              if (old_point == nullptr) {
                *error = "point-merge delta cannot recover an old track";
                return false;
              }
              delta->deleted_points.insert(old_id);
              for (const HostCatalogTrackElement& element : old_point->track) {
                delta->deleted_observations.insert(HostCatalogObservationKey(
                    element.image_id, element.point2D_idx));
              }
            }
            if (!AddCatalogPointFromReconstruction(
                    *reconstruction, event.point3D_id, delta.get(), error)) {
              return false;
            }
            break;
          case HostStructureEventKind::kObservationAdded:
          case HostStructureEventKind::kObservationDeleted:
            if (event.kind == HostStructureEventKind::kObservationDeleted) {
              const uint64_t key = HostCatalogObservationKey(
                  event.image_id, event.point2D_idx);
              delta->observations.erase(key);
              delta->deleted_observations.insert(key);
            }
            if (reconstruction->ExistsPoint3D(event.point3D_id) &&
                !AddCatalogPointFromReconstruction(
                    *reconstruction, event.point3D_id, delta.get(), error)) {
              return false;
            }
            break;
          case HostStructureEventKind::kBulkUnknown:
            *error = "BulkUnknown cannot be applied as a catalog delta";
            return false;
        }
      }
    }
    delta->resident_bytes = EstimateCatalogResidentBytes(*delta);
    *output = std::move(delta);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "static catalog delta allocation failed";
    return false;
  }
}

bool FlattenStaticProblemDataCatalog(
    const std::shared_ptr<const StaticProblemDataCatalog>& root,
    HostIndexedCatalogData* output,
    std::string* error) {
  if (root == nullptr || output == nullptr || error == nullptr ||
      root->owner_epoch == 0 || root->generation == 0) {
    if (error != nullptr) *error = "invalid indexed catalog flatten arguments";
    return false;
  }
  error->clear();
  try {
    HostIndexedCatalogData flat;
    flat.owner_epoch = root->owner_epoch;
    flat.revision = root->revision;
    flat.generation = root->generation;
    std::unordered_set<uint32_t> seen_cameras;
    std::unordered_set<uint32_t> seen_images;
    std::unordered_set<uint64_t> seen_points;
    std::unordered_set<uint64_t> seen_observations;
    for (auto node = root; node != nullptr; node = node->parent) {
      seen_cameras.insert(node->deleted_cameras.begin(),
                          node->deleted_cameras.end());
      seen_images.insert(node->deleted_images.begin(),
                         node->deleted_images.end());
      seen_points.insert(node->deleted_points.begin(),
                         node->deleted_points.end());
      seen_observations.insert(node->deleted_observations.begin(),
                               node->deleted_observations.end());
      for (const auto& item : node->cameras) {
        if (seen_cameras.insert(item.first).second)
          flat.cameras.push_back(item.second);
      }
      for (const auto& item : node->images) {
        if (seen_images.insert(item.first).second)
          flat.images.push_back(item.second);
      }
      for (const auto& item : node->points) {
        if (seen_points.insert(item.first).second)
          flat.points.push_back(item.second);
      }
      for (const auto& item : node->observations) {
        if (seen_observations.insert(item.first).second)
          flat.observations.push_back(item.second);
      }
    }
    std::sort(flat.cameras.begin(), flat.cameras.end(),
              [](const HostCatalogCamera& lhs, const HostCatalogCamera& rhs) {
                return lhs.camera_id < rhs.camera_id;
              });
    std::sort(flat.images.begin(), flat.images.end(),
              [](const HostCatalogImage& lhs, const HostCatalogImage& rhs) {
                return lhs.image_id < rhs.image_id;
              });
    std::sort(flat.points.begin(), flat.points.end(),
              [](const HostCatalogPoint& lhs, const HostCatalogPoint& rhs) {
                return lhs.point3D_id < rhs.point3D_id;
              });
    std::sort(flat.observations.begin(), flat.observations.end(),
              [](const HostCatalogObservation& lhs,
                 const HostCatalogObservation& rhs) {
                return HostCatalogObservationKey(lhs.image_id,
                                                 lhs.point2D_idx) <
                       HostCatalogObservationKey(rhs.image_id,
                                                 rhs.point2D_idx);
              });

    std::unordered_set<uint32_t> camera_ids;
    std::unordered_set<uint32_t> image_ids;
    std::unordered_set<uint64_t> point_ids;
    camera_ids.reserve(flat.cameras.size());
    image_ids.reserve(flat.images.size());
    point_ids.reserve(flat.points.size());
    for (const HostCatalogCamera& camera : flat.cameras) {
      if (!camera_ids.insert(camera.camera_id).second) {
        *error = "flattened indexed catalog contains duplicate camera";
        return false;
      }
    }
    for (const HostCatalogImage& image : flat.images) {
      if (!image_ids.insert(image.image_id).second ||
          camera_ids.count(image.camera_id) == 0) {
        *error = "flattened indexed catalog image identity is invalid";
        return false;
      }
    }
    for (const HostCatalogPoint& point : flat.points) {
      if (!point_ids.insert(point.point3D_id).second) {
        *error = "flattened indexed catalog contains duplicate point";
        return false;
      }
    }
    std::unordered_map<uint64_t, uint64_t> observation_points;
    observation_points.reserve(flat.observations.size());
    for (const HostCatalogObservation& observation : flat.observations) {
      const uint64_t key = HostCatalogObservationKey(
          observation.image_id, observation.point2D_idx);
      if (image_ids.count(observation.image_id) == 0 ||
          point_ids.count(observation.point3D_id) == 0 ||
          !observation_points.emplace(key, observation.point3D_id).second) {
        *error = "flattened indexed catalog observation identity is invalid";
        return false;
      }
    }
    for (const HostCatalogPoint& point : flat.points) {
      for (const HostCatalogTrackElement& element : point.track) {
        const auto observation = observation_points.find(
            HostCatalogObservationKey(element.image_id, element.point2D_idx));
        if (observation == observation_points.end() ||
            observation->second != point.point3D_id) {
          *error = "flattened indexed catalog track is not reciprocal";
          return false;
        }
      }
    }
    flat.resident_bytes = sizeof(flat);
    flat.resident_bytes = SaturatingAddBytes(
        flat.resident_bytes,
        SaturatingMulBytes(flat.cameras.size(), sizeof(HostCatalogCamera)));
    flat.resident_bytes = SaturatingAddBytes(
        flat.resident_bytes,
        SaturatingMulBytes(flat.images.size(), sizeof(HostCatalogImage)));
    flat.resident_bytes = SaturatingAddBytes(
        flat.resident_bytes,
        SaturatingMulBytes(flat.points.size(), sizeof(HostCatalogPoint)));
    flat.resident_bytes = SaturatingAddBytes(
        flat.resident_bytes,
        SaturatingMulBytes(flat.observations.size(),
                           sizeof(HostCatalogObservation)));
    for (const HostCatalogPoint& point : flat.points) {
      flat.resident_bytes = SaturatingAddBytes(
          flat.resident_bytes,
          SaturatingMulBytes(point.track.size(),
                             sizeof(HostCatalogTrackElement)));
    }
    *output = std::move(flat);
    return true;
  } catch (const std::bad_alloc&) {
    *error = "indexed catalog flatten allocation failed";
    return false;
  }
}

}  // namespace gpu_ba
}  // namespace colmap
