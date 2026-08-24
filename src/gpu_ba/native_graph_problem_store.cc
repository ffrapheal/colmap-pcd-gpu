#include "gpu_ba/native_graph_problem_store.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Core>

#include "base/reconstruction.h"
#include "gpu_ba/host_problem_store_internal.h"

namespace colmap {
namespace gpu_ba {
namespace {

struct ParameterKey {
  ParameterKind kind = ParameterKind::kPoint3D;
  uint64_t entity_id = 0;
  bool operator==(const ParameterKey& other) const noexcept {
    return kind == other.kind && entity_id == other.entity_id;
  }
};

struct ParameterKeyHash {
  size_t operator()(const ParameterKey& value) const noexcept {
    return std::hash<uint64_t>()(
        (value.entity_id << 3) ^ static_cast<uint64_t>(value.kind));
  }
};

bool Finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

template <size_t N>
bool Finite(const std::array<double, N>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

uint64_t ObservationKey(const uint32_t image_id,
                        const uint32_t point2D_idx) noexcept {
  return (static_cast<uint64_t>(image_id) << 32) | point2D_idx;
}

bool BuildColdInput(const Reconstruction& reconstruction,
                    const uint64_t owner_epoch,
                    HostBaGraphColdInput* output,
                    std::string* error) {
  if (output == nullptr || error == nullptr || owner_epoch == 0 ||
      reconstruction.StructureOwnerEpoch() != owner_epoch) {
    if (error != nullptr) *error = "invalid native graph cold-build identity";
    return false;
  }
  const uint64_t revision = reconstruction.StructureRevision();
  if (revision == 0) {
    *error = "native graph requires an active structure journal";
    return false;
  }
  HostBaGraphColdInput result;
  result.owner_epoch = owner_epoch;
  result.topology_revision = revision;
  std::vector<uint32_t> camera_ids;
  std::vector<uint32_t> image_ids;
  std::vector<uint64_t> point_ids;
  camera_ids.reserve(reconstruction.NumCameras());
  image_ids.reserve(reconstruction.NumImages());
  point_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& item : reconstruction.Cameras()) camera_ids.push_back(item.first);
  for (const auto& item : reconstruction.Images()) image_ids.push_back(item.first);
  for (const auto& item : reconstruction.Points3D()) point_ids.push_back(item.first);
  std::sort(camera_ids.begin(), camera_ids.end());
  std::sort(image_ids.begin(), image_ids.end());
  std::sort(point_ids.begin(), point_ids.end());
  result.cameras.reserve(camera_ids.size());
  result.images.reserve(image_ids.size());
  result.points.reserve(point_ids.size());
  for (const uint32_t id : camera_ids) {
    const Camera& camera = reconstruction.Camera(id);
    result.cameras.push_back({id, camera.ModelId(), camera.Width(),
                              camera.Height(),
                              static_cast<uint32_t>(camera.Params().size())});
  }
  for (const uint32_t id : image_ids) {
    const Image& image = reconstruction.Image(id);
    if (!reconstruction.ExistsCamera(image.CameraId())) {
      *error = "native graph image references a missing camera";
      return false;
    }
    result.images.push_back({id, image.CameraId(), image.IsRegistered()});
  }
  std::unordered_set<uint64_t> observation_keys;
  for (const uint64_t id : point_ids) {
    const Point3D& point = reconstruction.Point3D(id);
    result.points.push_back({id});
    for (const TrackElement& element : point.Track().Elements()) {
      if (!reconstruction.ExistsImage(element.image_id) ||
          element.point2D_idx >=
              reconstruction.Image(element.image_id).NumPoints2D()) {
        *error = "native graph point track references a missing observation";
        return false;
      }
      const Point2D& point2D =
          reconstruction.Image(element.image_id).Point2D(element.point2D_idx);
      if (!point2D.HasPoint3D() || point2D.Point3DId() != id ||
          !observation_keys.insert(
              ObservationKey(element.image_id, element.point2D_idx)).second) {
        *error = "native graph point track is not reciprocal or is duplicated";
        return false;
      }
      result.observations.push_back(
          {element.image_id, element.point2D_idx, id,
           {{point2D.X(), point2D.Y()}}});
    }
  }
  std::sort(result.observations.begin(), result.observations.end(),
            [](const HostBaObservationRecord& lhs,
               const HostBaObservationRecord& rhs) {
              return std::tie(lhs.image_id, lhs.point2D_idx) <
                     std::tie(rhs.image_id, rhs.point2D_idx);
            });
  if (reconstruction.StructureOwnerEpoch() != owner_epoch ||
      reconstruction.StructureRevision() != revision) {
    *error = "Reconstruction changed during native graph cold build";
    return false;
  }
  *output = std::move(result);
  return true;
}

bool BuildCoalescedMutation(const Reconstruction& reconstruction,
                            const HostStructureReadResult& journal,
                            CoalescedBaGraphMutation* output,
                            bool* full_rebuild,
                            NativeGraphPrepareRuntime* runtime,
                            std::string* error) {
  if (output == nullptr || full_rebuild == nullptr || runtime == nullptr ||
      error == nullptr || !journal.complete || journal.gap) {
    return false;
  }
  *full_rebuild = false;
  CoalescedBaGraphMutation mutation;
  mutation.owner_epoch = journal.owner_epoch;
  mutation.revision_before = journal.cursor;
  mutation.revision_after = journal.current_revision;
  std::unordered_set<uint32_t> cameras;
  std::unordered_set<uint32_t> images;
  std::unordered_set<uint64_t> points;
  std::unordered_set<uint64_t> observations;
  for (const HostStructureBatch& batch : journal.batches) {
    ++runtime->journal_batches;
    for (const HostStructureEvent& event : batch.events) {
      ++runtime->journal_events;
      switch (event.kind) {
        case HostStructureEventKind::kCameraAdded:
          cameras.insert(event.new_camera_id);
          break;
        case HostStructureEventKind::kImageAdded:
        case HostStructureEventKind::kImageRegistrationChanged:
        case HostStructureEventKind::kCameraAssociationChanged:
          images.insert(event.image_id);
          break;
        case HostStructureEventKind::kPointAdded:
          points.insert(event.point3D_id);
          break;
        case HostStructureEventKind::kPointDeleted:
          points.insert(event.point3D_id);
          for (const HostCatalogTrackElement& element : event.old_track) {
            observations.insert(ObservationKey(element.image_id,
                                               element.point2D_idx));
          }
          break;
        case HostStructureEventKind::kPointMerged:
          points.insert(event.old_point3D_id1);
          points.insert(event.old_point3D_id2);
          points.insert(event.point3D_id);
          break;
        case HostStructureEventKind::kObservationAdded:
        case HostStructureEventKind::kObservationDeleted:
          observations.insert(ObservationKey(event.image_id,
                                             event.point2D_idx));
          break;
        case HostStructureEventKind::kBulkUnknown:
          *full_rebuild = true;
          return true;
      }
    }
  }
  std::vector<uint32_t> sorted_cameras(cameras.begin(), cameras.end());
  std::vector<uint32_t> sorted_images(images.begin(), images.end());
  std::vector<uint64_t> sorted_points(points.begin(), points.end());
  std::sort(sorted_cameras.begin(), sorted_cameras.end());
  std::sort(sorted_images.begin(), sorted_images.end());
  std::sort(sorted_points.begin(), sorted_points.end());
  for (const uint32_t id : sorted_cameras) {
    if (!reconstruction.ExistsCamera(id)) {
      *full_rebuild = true;
      return true;
    }
    const Camera& camera = reconstruction.Camera(id);
    mutation.camera_upserts.push_back(
        {id, camera.ModelId(), camera.Width(), camera.Height(),
         static_cast<uint32_t>(camera.Params().size())});
  }
  for (const uint32_t id : sorted_images) {
    if (!reconstruction.ExistsImage(id)) {
      *full_rebuild = true;
      return true;
    }
    const Image& image = reconstruction.Image(id);
    mutation.image_upserts.push_back(
        {id, image.CameraId(), image.IsRegistered()});
  }
  for (const uint64_t id : sorted_points) {
    if (!reconstruction.ExistsPoint3D(id)) {
      mutation.tombstone_point_ids.push_back(id);
      continue;
    }
    mutation.point_upserts.push_back({id});
    const Point3D& point = reconstruction.Point3D(id);
    for (const TrackElement& element : point.Track().Elements()) {
      observations.insert(ObservationKey(element.image_id,
                                         element.point2D_idx));
    }
  }
  std::vector<uint64_t> sorted_observations(observations.begin(),
                                            observations.end());
  std::sort(sorted_observations.begin(), sorted_observations.end());
  for (const uint64_t key : sorted_observations) {
    const uint32_t image_id = static_cast<uint32_t>(key >> 32);
    const uint32_t point2D_idx = static_cast<uint32_t>(key);
    if (!reconstruction.ExistsImage(image_id) ||
        point2D_idx >= reconstruction.Image(image_id).NumPoints2D()) {
      mutation.tombstone_observations.push_back({image_id, point2D_idx});
      continue;
    }
    const Point2D& point2D = reconstruction.Image(image_id).Point2D(point2D_idx);
    if (!point2D.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2D.Point3DId())) {
      mutation.tombstone_observations.push_back({image_id, point2D_idx});
      continue;
    }
    mutation.observation_upserts.push_back(
        {image_id, point2D_idx, point2D.Point3DId(),
         {{point2D.X(), point2D.Y()}}});
  }
  if (reconstruction.StructureOwnerEpoch() != journal.owner_epoch ||
      reconstruction.StructureRevision() != journal.current_revision) {
    *error = "Reconstruction changed during native graph delta conversion";
    return false;
  }
  *output = std::move(mutation);
  return true;
}

const ActiveBaParameterBlockSpec* FindParameter(
    const NativeActiveSolveInputs& inputs,
    const ParameterKind kind,
    const uint64_t entity_id) {
  for (size_t i = 0; i < inputs.parameter_blocks.size; ++i) {
    const ActiveBaParameterBlockSpec& parameter = inputs.parameter_blocks[i];
    if (parameter.kind == kind && parameter.entity_id == entity_id)
      return &parameter;
  }
  return nullptr;
}

template <typename SnapshotType, typename IdType, typename GetId>
const SnapshotType* FindById(const std::vector<SnapshotType>& values,
                             const IdType id,
                             const GetId& get_id) {
  const auto found = std::find_if(values.begin(), values.end(),
                                  [&](const SnapshotType& value) {
                                    return get_id(value) == id;
                                  });
  return found == values.end() ? nullptr : &*found;
}

}  // namespace

struct NativeGraphStoreState {
  NativeGraphStoreState(const Reconstruction* reconstruction_in,
                        const uint64_t owner_epoch_in)
      : reconstruction(reconstruction_in),
        owner_epoch(owner_epoch_in),
        graph(owner_epoch_in) {}

  const Reconstruction* reconstruction = nullptr;
  uint64_t owner_epoch = 0;
  uint64_t journal_cursor = 0;
  uint64_t next_config_generation = 0;
  uint64_t next_selection_generation = 0;
  uint64_t next_state_generation = 0;
  uint64_t active_prepares = 0;
  bool closing = false;
  std::mutex mutex;
  HostBaGraphStore graph;
  NativeHostSolveMaterializer materializer;
};

struct PreparedNativeActiveSolve::Data {
  std::shared_ptr<NativeGraphStoreState> state;
  NativeHostSolveView view;
  ActiveStateBuffer initial_state;
  CudaFullLmOptions options;
  NativeGraphPrepareRuntime runtime;
  bool active = false;
};

PreparedNativeActiveSolve::PreparedNativeActiveSolve() = default;
PreparedNativeActiveSolve::~PreparedNativeActiveSolve() {
  std::string ignored;
  Complete(&ignored);
}
PreparedNativeActiveSolve::PreparedNativeActiveSolve(
    PreparedNativeActiveSolve&&) noexcept = default;
PreparedNativeActiveSolve& PreparedNativeActiveSolve::operator=(
    PreparedNativeActiveSolve&&) noexcept = default;

bool PreparedNativeActiveSolve::valid() const noexcept {
  return data_ != nullptr && data_->active && data_->state != nullptr &&
         data_->view.catalog.valid();
}

NativeCudaSolveRequest PreparedNativeActiveSolve::request() const noexcept {
  NativeCudaSolveRequest request;
  if (valid()) {
    request.view = &data_->view;
    request.initial_state = &data_->initial_state;
    request.options = &data_->options;
  }
  return request;
}

const NativeHostSolveView* PreparedNativeActiveSolve::view() const noexcept {
  return valid() ? &data_->view : nullptr;
}

const ActiveStateBuffer* PreparedNativeActiveSolve::initial_state()
    const noexcept {
  return valid() ? &data_->initial_state : nullptr;
}

const NativeGraphPrepareRuntime& PreparedNativeActiveSolve::runtime()
    const noexcept {
  static const NativeGraphPrepareRuntime empty;
  return data_ == nullptr ? empty : data_->runtime;
}

bool PreparedNativeActiveSolve::Complete(std::string* error) noexcept {
  if (error == nullptr) return false;
  error->clear();
  std::shared_ptr<Data> data = std::move(data_);
  if (data == nullptr || !data->active || data->state == nullptr) return true;
  {
    std::lock_guard<std::mutex> lock(data->state->mutex);
    if (data->state->active_prepares == 0) {
      *error = "native graph prepared lease accounting underflow";
      data->active = false;
      return false;
    }
    --data->state->active_prepares;
    data->active = false;
  }
  data->view = NativeHostSolveView();
  return true;
}

bool MakeNativeActiveSolveInputs(const ActiveBaSolveSpec& spec,
                                 NativeActiveSolveInputs* inputs,
                                 std::string* error) {
  if (inputs == nullptr || error == nullptr || spec.owner_epoch == 0 ||
      spec.catalog_revision == 0) {
    if (error != nullptr) *error = "active spec lacks native graph identity";
    return false;
  }
  NativeActiveSolveInputs result;
  result.problem = &spec.problem;
  result.parameter_blocks = {spec.parameter_blocks.data(),
                             spec.parameter_blocks.size()};
  result.owner_epoch = spec.owner_epoch;
  result.catalog_revision = spec.catalog_revision;
  result.residual_block_count = spec.residual_block_count;
  result.scalar_residual_count = spec.scalar_residual_count;
  result.ambient_parameter_count = spec.ambient_parameter_count;
  result.effective_parameter_count = spec.effective_parameter_count;
  *inputs = result;
  error->clear();
  return true;
}

std::shared_ptr<NativeGraphStoreState> CreateNativeGraphStoreState(
    const Reconstruction* reconstruction, const uint64_t owner_epoch) {
  if (reconstruction == nullptr || owner_epoch == 0) return nullptr;
  try {
    return std::make_shared<NativeGraphStoreState>(reconstruction,
                                                   owner_epoch);
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

bool ShutdownNativeGraphStoreState(
    const std::shared_ptr<NativeGraphStoreState>& state,
    std::string* error) noexcept {
  if (error == nullptr) return false;
  error->clear();
  if (state == nullptr) return true;
  std::lock_guard<std::mutex> lock(state->mutex);
  state->closing = true;
  if (state->active_prepares != 0) {
    *error = "StoreBusy: native graph has an active prepared solve";
    return false;
  }
  return true;
}

bool PrepareCudaNativeActiveSolve(
    const NativeActiveSolveInputs& inputs,
    const CudaFullLmOptions& options,
    const CudaHostStoreBinding& binding,
    PreparedNativeActiveSolve* prepared,
    std::string* error) {
  if (prepared == nullptr || error == nullptr || inputs.problem == nullptr) {
    if (error != nullptr) *error = "invalid native graph prepare arguments";
    return false;
  }
  std::string ignored;
  prepared->Complete(&ignored);
  error->clear();
  if (binding.mode != CudaHostProblemStoreMode::kHostPreparedStore ||
      binding.store == nullptr || binding.owner_epoch == 0 ||
      binding.owner_epoch != inputs.owner_epoch ||
      binding.store->native_graph_state_ == nullptr) {
    *error = "native_graph requires a matching enabled host store";
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  std::shared_ptr<NativeGraphStoreState> state =
      binding.store->native_graph_state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  if (state->closing || state->active_prepares != 0 ||
      state->owner_epoch != binding.owner_epoch ||
      state->reconstruction == nullptr) {
    *error = state->active_prepares != 0
        ? "StoreBusy: native graph already has an active solve"
        : "native graph store is closed or has mismatched identity";
    return false;
  }
  std::shared_ptr<PreparedNativeActiveSolve::Data> data;
  try {
    data = std::make_shared<PreparedNativeActiveSolve::Data>();
  } catch (const std::bad_alloc&) {
    *error = "native graph prepared solve allocation failed";
    return false;
  }
  data->state = state;
  data->runtime.journal_cursor_before = state->journal_cursor;
  const auto sync_start = std::chrono::steady_clock::now();
  const auto cold_build = [&](const bool rebuild) -> bool {
    HostBaGraphColdInput cold;
    if (!BuildColdInput(*state->reconstruction, state->owner_epoch, &cold,
                        error)) {
      return false;
    }
    HostBaGraphUpdateResult update;
    if (!state->graph.ColdBuild(cold, &update, error)) {
      if (update.reader_busy) ++data->runtime.reader_busy;
      return false;
    }
    state->journal_cursor = cold.topology_revision;
    ++data->runtime.catalog_cold_builds;
    if (rebuild) ++data->runtime.catalog_full_rebuilds;
    return true;
  };
  if (state->graph.empty()) {
    if (!cold_build(false)) return false;
  } else {
    HostStructureReadResult journal;
    if (!ReadHostStructureEventsSince(state->reconstruction,
                                      state->owner_epoch,
                                      state->journal_cursor,
                                      &journal, error)) {
      return false;
    }
    bool rebuild = journal.gap || !journal.complete;
    CoalescedBaGraphMutation mutation;
    if (!rebuild && journal.current_revision != state->journal_cursor) {
      if (!BuildCoalescedMutation(*state->reconstruction, journal, &mutation,
                                  &rebuild, &data->runtime, error)) {
        return false;
      }
    }
    if (rebuild) {
      if (!cold_build(true)) return false;
    } else if (journal.current_revision != state->journal_cursor) {
      HostBaGraphUpdateResult update;
      if (!state->graph.ApplyCoalescedMutation(mutation, &update, error)) {
        if (update.reader_busy) {
          ++data->runtime.reader_busy;
          return false;
        }
        if (!cold_build(true)) return false;
      } else {
        state->journal_cursor = journal.current_revision;
        ++data->runtime.catalog_delta_updates;
      }
    }
  }
  data->runtime.journal_cursor_after = state->journal_cursor;
  data->runtime.catalog_sync_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - sync_start).count();
  if (inputs.catalog_revision != state->journal_cursor) {
    *error = "native active selection and catalog revisions differ";
    return false;
  }
  CatalogReadLease lease = state->graph.AcquireReadLease();
  if (!lease.valid()) {
    *error = "native graph catalog lease is unavailable";
    return false;
  }
  if (++state->next_config_generation == 0)
    ++state->next_config_generation;
  if (++state->next_selection_generation == 0)
    ++state->next_selection_generation;
  if (++state->next_state_generation == 0)
    ++state->next_state_generation;
  NativeCudaResolvedConfig resolved_config;
  if (!ResolveNativeCudaConfiguration(
          *inputs.problem, options, state->next_config_generation,
          &data->options, &resolved_config, error)) {
    return false;
  }

  const CudaSolveProblem& problem = *inputs.problem;
  BaSolveIntent intent;
  intent.owner_epoch = inputs.owner_epoch;
  intent.catalog_revision = lease.topology_revision();
  intent.catalog_generation = lease.generation();
  intent.selection_revision = state->next_selection_generation;
  intent.kind = problem.metadata.ba_kind;
  intent.config = resolved_config;
  DenseActiveState dense;
  dense.owner_epoch = inputs.owner_epoch;
  dense.state_generation = state->next_state_generation;
  std::unordered_set<uint32_t> selected_images;
  std::unordered_set<uint32_t> relevant_cameras;
  std::unordered_set<uint32_t> relevant_points;
  for (const ImageSnapshot& image : problem.images) {
    const HostBaImageSlot* slot = lease.FindImageById(image.image_id);
    if (slot == nullptr || !slot->header.alive ||
        slot->camera_slot >= lease.cameras().size ||
        lease.cameras()[slot->camera_slot].camera_id != image.camera_id) {
      *error = "native active image is inconsistent with the graph catalog";
      return false;
    }
    DenseImageState state_value;
    state_value.image_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.quaternion = image.qvec;
    state_value.translation = image.tvec;
    dense.images.push_back(state_value);
    if (image.selected) {
      if (!selected_images.insert(image.image_id).second) {
        *error = "native active image selection is duplicated";
        return false;
      }
      intent.active_image_slots.push_back(slot->header.slot);
      if (image.pose_constant) intent.fixed_pose_slots.push_back(slot->header.slot);
      const ActiveBaParameterBlockSpec* translation = FindParameter(
          inputs, ParameterKind::kTranslation, image.image_id);
      if (translation != nullptr) {
        intent.translation_subsets.push_back(
            {slot->header.slot, translation->translation_subset_mask,
             {0, 0, 0}});
      }
    }
    relevant_cameras.insert(image.camera_id);
  }
  for (const CameraSnapshot& camera : problem.cameras) {
    if (relevant_cameras.count(camera.camera_id) == 0) continue;
    const HostBaCameraSlot* slot = lease.FindCameraById(camera.camera_id);
    if (slot == nullptr || !slot->header.alive ||
        slot->model_id != camera.model_id || slot->width != camera.width ||
        slot->height != camera.height ||
        slot->parameter_count != camera.params.size()) {
      *error = "native active camera is inconsistent with the graph catalog";
      return false;
    }
    DenseCameraState state_value;
    state_value.camera_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.parameters = camera.params;
    dense.cameras.push_back(std::move(state_value));
    CameraParameterPolicy policy;
    policy.camera_slot = slot->header.slot;
    const ActiveBaParameterBlockSpec* parameter =
        FindParameter(inputs, ParameterKind::kCamera, camera.camera_id);
    policy.constant = parameter == nullptr || parameter->constant;
    if (parameter != nullptr) {
      policy.fixed_parameter_indices =
          parameter->fixed_camera_parameter_indices;
    }
    intent.camera_policies.push_back(std::move(policy));
  }
  for (const PointSnapshot& point : problem.points) {
    const HostBaPointSlot* slot = lease.FindPointById(point.point3D_id);
    if (slot == nullptr || !slot->header.alive) {
      *error = "native active point is missing from the graph catalog";
      return false;
    }
    DensePointState state_value;
    state_value.point_slot = slot->header.slot;
    state_value.state_generation = dense.state_generation;
    state_value.xyz = point.xyz;
    dense.points.push_back(state_value);
    relevant_points.insert(slot->header.slot);
    PointFixedPolicy policy;
    policy.point_slot = slot->header.slot;
    policy.constant = point.constant;
    policy.config_role = point.config_role;
    policy.has_search_range = point.has_search_range;
    policy.search_range = point.search_range;
    intent.point_policies.push_back(policy);
    if (point.config_role == 1) {
      intent.explicit_variable_point_slots.push_back(slot->header.slot);
    }
  }
  std::unordered_map<uint64_t, uint32_t> visual_source_slots;
  std::unordered_map<uint64_t, uint32_t> lidar_source_slots;
  visual_source_slots.reserve(problem.observations.size());
  lidar_source_slots.reserve(problem.lidar.size());
  for (const ObservationSnapshot& observation : problem.observations) {
    const HostBaObservationSlot* slot =
        lease.FindObservation(observation.image_id, observation.point2D_idx);
    if (slot == nullptr || !slot->header.alive ||
        slot->point_slot >= lease.points().size ||
        lease.points()[slot->point_slot].point3D_id != observation.point3D_id ||
        slot->xy != observation.xy ||
        !visual_source_slots.emplace(observation.source_index,
                                     slot->header.slot).second) {
      *error = "native visual residual is inconsistent with the graph catalog";
      return false;
    }
  }
  intent.lidar.lidar_map_generation = state->journal_cursor;
  intent.lidar.match_config_generation = resolved_config.config_generation;
  intent.lidar.constraints.reserve(problem.lidar.size());
  for (size_t index = 0; index < problem.lidar.size(); ++index) {
    const LidarSnapshot& lidar = problem.lidar[index];
    const HostBaPointSlot* point = lease.FindPointById(lidar.point3D_id);
    if (point == nullptr || !point->header.alive ||
        relevant_points.count(point->header.slot) == 0 ||
        !lidar_source_slots.emplace(lidar.source_index,
                                    static_cast<uint32_t>(index)).second) {
      *error = "native LiDAR residual is inconsistent with the active points";
      return false;
    }
    LidarConstraintRecord constraint;
    constraint.point_slot = point->header.slot;
    constraint.constraint_slot = static_cast<uint32_t>(index);
    constraint.physical_identity = lidar.source_index;
    constraint.lidar_type = lidar.lidar_type;
    constraint.plane = lidar.plane;
    constraint.lidar_xyz = lidar.lidar_xyz;
    constraint.weight = lidar.weight;
    constraint.search_range = lidar.search_range;
    constraint.point_state_generation = dense.state_generation;
    intent.lidar.constraints.push_back(constraint);
  }
  const size_t residual_count =
      problem.observations.size() + problem.lidar.size();
  if (problem.source_insertion_order.size() != residual_count) {
    *error = "native source order does not cover all residuals";
    return false;
  }
  intent.source_insertion_order.reserve(residual_count);
  for (size_t index = 0; index < residual_count; ++index) {
    const OrderEntrySnapshot& entry = problem.source_insertion_order[index];
    if (entry.source_index != index) {
      *error = "native source order is not a dense solve-local sequence";
      return false;
    }
    if (entry.residual_kind == ResidualKind::kVisual) {
      const auto found = visual_source_slots.find(index);
      if (found == visual_source_slots.end()) {
        *error = "native source order references a missing visual residual";
        return false;
      }
      intent.source_insertion_order.push_back(
          {ResidualKind::kVisual, found->second});
    } else {
      const auto found = lidar_source_slots.find(index);
      if (found == lidar_source_slots.end()) {
        *error = "native source order references a missing LiDAR residual";
        return false;
      }
      intent.source_insertion_order.push_back(
          {ResidualKind::kLidar, found->second});
    }
  }
  const auto materialize_start = std::chrono::steady_clock::now();
  NativeHostSolvePreparationRuntime materialize_runtime;
  if (!state->materializer.Materialize(
          lease, intent, dense, &data->view, &data->initial_state,
          &materialize_runtime, error)) {
    return false;
  }
  data->runtime.materialize_wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - materialize_start).count();
  if (data->view.residual_block_count != inputs.residual_block_count ||
      data->view.scalar_residual_count != inputs.scalar_residual_count ||
      data->view.ambient_parameter_count != inputs.ambient_parameter_count ||
      data->view.effective_parameter_count !=
          inputs.effective_parameter_count ||
      data->view.visual_observation_slots.size() !=
          problem.observations.size() ||
      data->view.lidar.constraints.size() != problem.lidar.size() ||
      data->view.parameter_ordinals.size() != inputs.parameter_blocks.size) {
    *error = "native materialized view count contract mismatch";
    return false;
  }
  const auto graph_cameras = data->view.catalog.cameras();
  const auto graph_images = data->view.catalog.images();
  const auto graph_points = data->view.catalog.points();
  for (size_t i = 0; i < inputs.parameter_blocks.size; ++i) {
    const ActiveBaParameterBlockSpec& expected = inputs.parameter_blocks[i];
    const ParameterOrdinal& actual = data->view.parameter_ordinals[i];
    uint64_t entity_id = 0;
    if (actual.kind == ParameterKind::kQuaternion ||
        actual.kind == ParameterKind::kTranslation) {
      if (actual.entity_slot >= graph_images.size) return false;
      entity_id = graph_images[actual.entity_slot].image_id;
    } else if (actual.kind == ParameterKind::kPoint3D) {
      if (actual.entity_slot >= graph_points.size) return false;
      entity_id = graph_points[actual.entity_slot].point3D_id;
    } else {
      if (actual.entity_slot >= graph_cameras.size) return false;
      entity_id = graph_cameras[actual.entity_slot].camera_id;
    }
    if (actual.kind != expected.kind || entity_id != expected.entity_id ||
        actual.ambient_size != expected.ambient_size ||
        actual.tangent_size != expected.tangent_size ||
        static_cast<bool>(actual.constant) != expected.constant ||
        actual.translation_subset_mask !=
            expected.translation_subset_mask) {
      *error = "native parameter ordinal contract mismatch";
      return false;
    }
  }
  ++state->active_prepares;
  ++data->runtime.prepared_views;
  data->active = true;
  prepared->data_ = std::move(data);
  (void)start;
  return true;
}

bool ValidateAndCommitNativeBaState(
    const NativeActiveSolveInputs& inputs,
    const PreparedNativeActiveSolve& prepared,
    const DenseActiveState& candidate,
    Reconstruction* reconstruction,
    std::string* error) {
  if (!prepared.valid() || inputs.problem == nullptr ||
      reconstruction == nullptr || error == nullptr) {
    if (error != nullptr) *error = "invalid native graph commit arguments";
    return false;
  }
  error->clear();
  const auto& data = *prepared.data_;
  const NativeHostSolveView& view = data.view;
  if (candidate.owner_epoch != view.identity.owner_epoch ||
      candidate.state_generation < data.initial_state.state_generation ||
      candidate.cameras.size() != data.initial_state.cameras.size() ||
      candidate.images.size() != data.initial_state.images.size() ||
      candidate.points.size() != data.initial_state.points.size()) {
    *error = "native final state identity or count mismatch";
    return false;
  }
  struct ImageUpdate {
    uint32_t id = 0;
    std::array<double, 4> q;
    std::array<double, 3> t;
    uint8_t mask = 0;
  };
  struct PointUpdate {
    uint64_t id = 0;
    std::array<double, 3> xyz;
  };
  std::vector<ImageUpdate> image_updates;
  std::vector<PointUpdate> point_updates;
  image_updates.reserve(candidate.images.size());
  point_updates.reserve(candidate.points.size());
  const auto cameras = view.catalog.cameras();
  const auto images = view.catalog.images();
  const auto points = view.catalog.points();
  std::unordered_set<uint32_t> camera_slots;
  std::unordered_set<uint32_t> image_slots;
  std::unordered_set<uint32_t> point_slots;
  for (const DenseCameraState& value : candidate.cameras) {
    if (value.camera_slot >= cameras.size ||
        !camera_slots.insert(value.camera_slot).second ||
        !Finite(value.parameters)) {
      *error = "native final camera state is invalid";
      return false;
    }
    const CameraSnapshot* before = FindById(
        inputs.problem->cameras, cameras[value.camera_slot].camera_id,
        [](const CameraSnapshot& camera) { return camera.camera_id; });
    if (before == nullptr || before->params != value.parameters ||
        !reconstruction->ExistsCamera(before->camera_id)) {
      *error = "native fixed camera changed or disappeared";
      return false;
    }
  }
  for (const DenseImageState& value : candidate.images) {
    if (value.image_slot >= images.size ||
        !image_slots.insert(value.image_slot).second ||
        !Finite(value.quaternion) || !Finite(value.translation)) {
      *error = "native final image state is invalid";
      return false;
    }
    const ImageSnapshot* before = FindById(
        inputs.problem->images, images[value.image_slot].image_id,
        [](const ImageSnapshot& image) { return image.image_id; });
    const auto policy = std::find_if(
        view.fixed.images.begin(), view.fixed.images.end(),
        [&](const ImageFixedPolicyResult& item) {
          return item.image_slot == value.image_slot;
        });
    if (before == nullptr || policy == view.fixed.images.end() ||
        !reconstruction->ExistsImage(before->image_id)) {
      *error = "native final image identity is missing";
      return false;
    }
    if (policy->pose_constant &&
        (before->qvec != value.quaternion ||
         before->tvec != value.translation)) {
      *error = "native fixed pose changed";
      return false;
    }
    for (size_t component = 0; component < 3; ++component) {
      if ((policy->translation_subset_mask & (1u << component)) != 0 &&
          before->tvec[component] != value.translation[component]) {
        *error = "native fixed translation component changed";
        return false;
      }
    }
    const double norm = std::sqrt(
        value.quaternion[0] * value.quaternion[0] +
        value.quaternion[1] * value.quaternion[1] +
        value.quaternion[2] * value.quaternion[2] +
        value.quaternion[3] * value.quaternion[3]);
    if (!policy->pose_constant &&
        (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-10)) {
      *error = "native final quaternion norm is invalid";
      return false;
    }
    if (!policy->pose_constant) {
      image_updates.push_back({before->image_id, value.quaternion,
                               value.translation,
                               policy->translation_subset_mask});
    }
  }
  for (const DensePointState& value : candidate.points) {
    if (value.point_slot >= points.size ||
        !point_slots.insert(value.point_slot).second || !Finite(value.xyz)) {
      *error = "native final point state is invalid";
      return false;
    }
    const PointSnapshot* before = FindById(
        inputs.problem->points, points[value.point_slot].point3D_id,
        [](const PointSnapshot& point) { return point.point3D_id; });
    const auto policy = std::find_if(
        view.fixed.points.begin(), view.fixed.points.end(),
        [&](const PointFixedPolicyResult& item) {
          return item.point_slot == value.point_slot;
        });
    if (before == nullptr || policy == view.fixed.points.end() ||
        !reconstruction->ExistsPoint3D(before->point3D_id)) {
      *error = "native final point identity is missing";
      return false;
    }
    if (policy->constant && before->xyz != value.xyz) {
      *error = "native fixed point changed";
      return false;
    }
    if (!policy->constant)
      point_updates.push_back({before->point3D_id, value.xyz});
  }
  if (camera_slots.size() != data.initial_state.cameras.size() ||
      image_slots.size() != data.initial_state.images.size() ||
      point_slots.size() != data.initial_state.points.size()) {
    *error = "native final state slot coverage mismatch";
    return false;
  }
  for (const ImageUpdate& update : image_updates) {
    Image& image = reconstruction->Image(update.id);
    image.SetQvec(Eigen::Map<const Eigen::Vector4d>(update.q.data()));
    Eigen::Vector3d translation = image.Tvec();
    for (size_t component = 0; component < 3; ++component) {
      if ((update.mask & (1u << component)) == 0)
        translation[component] = update.t[component];
    }
    image.SetTvec(translation);
  }
  for (const PointUpdate& update : point_updates) {
    reconstruction->Point3D(update.id).XYZ() =
        Eigen::Map<const Eigen::Vector3d>(update.xyz.data());
  }
  return true;
}

}  // namespace gpu_ba
}  // namespace colmap
