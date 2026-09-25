#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"
#include "pcd_projection.h"
#include "sfm/nonba_profiler.h"

#ifdef GPU_BA_CUDA_ENABLED
#include "base/camera_models.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace colmap{
namespace lidar{

using namespace Eigen;

namespace {

bool IsRegularFile(const std::string& path) {
    struct stat status;
    return stat(path.c_str(), &status) == 0 && S_ISREG(status.st_mode);
}

std::string JoinPath(const std::string& directory,
                     const std::string& filename) {
    if (directory.empty() || directory.back() == '/') {
        return directory + filename;
    }
    return directory + "/" + filename;
}

bool ParseFrameId(const std::string& image_name, std::string* frame_id) {
    const size_t slash = image_name.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot = image_name.find_last_of('.');
    size_t end = dot == std::string::npos || dot < begin
                     ? image_name.size()
                     : dot;
    size_t digits_begin = end;
    while (digits_begin > begin &&
           std::isdigit(static_cast<unsigned char>(image_name[digits_begin - 1]))) {
        --digits_begin;
    }
    if (digits_begin == end) {
        return false;
    }
    try {
        *frame_id = std::to_string(
            std::stoull(image_name.substr(digits_begin, end - digits_begin)));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool RunProcess(const std::vector<std::string>& arguments) {
    if (arguments.empty()) {
        return false;
    }
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t child = fork();
    if (child < 0) {
        std::cerr << "Failed to fork mesh-depth renderer: "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    if (child == 0) {
        execv(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace

namespace internal {

bool IntersectCameraRayWithWorldPlane(
        const Camera& camera,
        const Eigen::Matrix3d& rotation_cw,
        const Eigen::Vector3d& translation_cw,
        const Eigen::Vector2d& image_point,
        const Eigen::Vector3d& plane_point_world,
        const Eigen::Vector3d& plane_normal_world,
        Eigen::Vector3d* intersection_world) {
    if (intersection_world == nullptr) {
        return false;
    }
    intersection_world->setZero();
    if (!rotation_cw.allFinite() || !translation_cw.allFinite() ||
        !image_point.allFinite() || !plane_point_world.allFinite() ||
        !plane_normal_world.allFinite()) {
        return false;
    }

    const Eigen::Vector2d normalized_point = camera.ImageToWorld(image_point);
    const Eigen::Vector3d ray_camera(
        normalized_point.x(), normalized_point.y(), 1.0);
    if (!ray_camera.allFinite()) {
        return false;
    }

    const Eigen::Matrix3d rotation_wc = rotation_cw.transpose();
    const Eigen::Vector3d ray_world = rotation_wc * ray_camera;
    const Eigen::Vector3d camera_center_world =
        -rotation_wc * translation_cw;
    const double ray_norm = ray_world.norm();
    const double normal_norm = plane_normal_world.norm();
    if (!ray_world.allFinite() || !camera_center_world.allFinite() ||
        !std::isfinite(ray_norm) || !std::isfinite(normal_norm) ||
        ray_norm == 0.0 || normal_norm == 0.0) {
        return false;
    }

    const double denominator = plane_normal_world.dot(ray_world);
    constexpr double kMinAbsRayPlaneCosine = 1e-10;
    if (!std::isfinite(denominator) ||
        std::abs(denominator) <=
            kMinAbsRayPlaneCosine * normal_norm * ray_norm) {
        return false;
    }

    const double lambda = plane_normal_world.dot(
                              plane_point_world - camera_center_world) /
                          denominator;
    if (!std::isfinite(lambda) || lambda <= 0.0) {
        return false;
    }

    const Eigen::Vector3d candidate_world =
        camera_center_world + lambda * ray_world;
    if (!candidate_world.allFinite()) {
        return false;
    }
    *intersection_world = candidate_world;
    return true;
}

}  // namespace internal

bool PcdProj::HasInitialMeshDepth() const {
    return !options_.initial_mesh_depth_path.empty() &&
           (IsRegularFile(options_.initial_mesh_depth_path) ||
            (!options_.initial_mesh_depth_generator_path.empty() &&
             !options_.initial_mesh_path.empty() &&
             !options_.initial_mesh_depth_dataset_path.empty() &&
             !options_.initial_mesh_depth_intrinsics_path.empty()));
}

double PcdProj::InitialMeshDepthPnpMaxError() const {
    return options_.initial_mesh_depth_pnp_max_error;
}

std::string PcdProj::ResolveInitialMeshDepthPath(const Image& image) const {
    if (IsRegularFile(options_.initial_mesh_depth_path)) {
        return options_.initial_mesh_depth_path;
    }

    std::string frame_id;
    if (!ParseFrameId(image.Name(), &frame_id)) {
        std::cerr << "Cannot extract frame id for mesh depth from image name: "
                  << image.Name() << std::endl;
        return std::string();
    }

    const std::string depth_path = JoinPath(
        options_.initial_mesh_depth_path,
        "imgs_" + frame_id + "_mesh_depth_m.tiff");
    std::lock_guard<std::mutex> lock(initial_mesh_depth_mutex_);
    if (IsRegularFile(depth_path)) {
        std::cout << "Initial mesh depth cache hit: " << depth_path
                  << std::endl;
        return depth_path;
    }

    const std::string odometry_path = JoinPath(
        options_.initial_mesh_depth_dataset_path,
        "odoms_" + frame_id + ".txt");
    const std::vector<std::string> arguments = {
        options_.initial_mesh_depth_generator_path,
        options_.initial_mesh_depth_dataset_path,
        frame_id,
        options_.initial_mesh_path,
        "--output-directory",
        options_.initial_mesh_depth_path,
        "--intrinsics",
        options_.initial_mesh_depth_intrinsics_path,
        "--odometry",
        odometry_path};
    std::cout << "Rendering initial mesh depth for " << image.Name()
              << " from " << options_.initial_mesh_path << std::endl;
    if (!RunProcess(arguments) || !IsRegularFile(depth_path)) {
        std::cerr << "Failed to generate initial mesh depth: " << depth_path
                  << std::endl;
        return std::string();
    }
    return depth_path;
}

bool PcdProj::SetInitialImageFromMeshDepth(
        const Image& image,
        const Camera& camera,
        std::vector<std::pair<Eigen::Vector2d, bool>, Eigen::aligned_allocator<std::pair<Eigen::Vector2d, bool>>>& pt_xys,
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pt_xyzs) const {
    pt_xyzs.clear();
    pt_xyzs.reserve(pt_xys.size());

    const std::string depth_path = ResolveInitialMeshDepthPath(image);
    if (depth_path.empty()) {
        return false;
    }
    const cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth.empty() || depth.type() != CV_32FC1) {
        std::cerr << "Failed to load float32 initial mesh depth: "
                  << depth_path << std::endl;
        return false;
    }
    if (!(options_.initial_mesh_depth_fx > 0.0) ||
        !(options_.initial_mesh_depth_fy > 0.0)) {
        std::cerr << "Initial mesh depth intrinsics must have positive fx/fy."
                  << std::endl;
        return false;
    }

    const Eigen::Matrix3d rotation_wc = image.RotationMatrix().transpose();
    const Eigen::Vector3d center_world = image.ProjectionCenter();
    size_t valid_depths = 0;
    for (auto& pt_xy : pt_xys) {
        const Eigen::Vector2d normalized = camera.ImageToWorld(pt_xy.first);
        const double depth_u =
            options_.initial_mesh_depth_fx * normalized.x() +
            options_.initial_mesh_depth_cx;
        const double depth_v =
            options_.initial_mesh_depth_fy * normalized.y() +
            options_.initial_mesh_depth_cy;
        const int u = static_cast<int>(std::lround(depth_u));
        const int v = static_cast<int>(std::lround(depth_v));
        if (u < 0 || u >= depth.cols || v < 0 || v >= depth.rows) {
            pt_xy.second = false;
            pt_xyzs.push_back(Eigen::Vector3d::Zero());
            continue;
        }

        const float z = depth.at<float>(v, u);
        if (!std::isfinite(z) || z <= 0.0f) {
            pt_xy.second = false;
            pt_xyzs.push_back(Eigen::Vector3d::Zero());
            continue;
        }

        const Eigen::Vector3d point_camera(
            static_cast<double>(z) * normalized.x(),
            static_cast<double>(z) * normalized.y(),
            static_cast<double>(z));
        pt_xy.second = true;
        pt_xyzs.push_back(center_world + rotation_wc * point_camera);
        ++valid_depths;
    }

    std::cout << "Initial mesh depth: " << valid_depths << "/"
              << pt_xys.size() << " matched features have valid depth from "
              << depth_path << std::endl;
    return true;
}

void PcdProj::SetNewImage(const Image& image, const Camera& camera, std::map<point3D_t,Eigen::Matrix<double,6,1>>& map){
    NonBaStageSink* const non_ba_profiler = non_ba_profiler_;
    NonBaStageScope set_new_image(
        non_ba_profiler, NonBaStageId::kLocalPcdSetNewImage, 1);
    const size_t output_map_size_before =
        non_ba_profiler == nullptr ? 0 : map.size();

    // Create a new image struct
    Eigen::Quaterniond q_cw(image.Qvec()[0],image.Qvec()[1],image.Qvec()[2],image.Qvec()[3]);
    Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();
    Eigen::Vector3d t_cw = image.Tvec();
    
    // Only for opencv camera model
    std::vector<double> params = camera.Params();
    double scale = options_.depth_image_scale;
    int img_h = static_cast<int>(camera.Height() * scale);
    int img_w = static_cast<int>(camera.Width() * scale);

    // Save the pixel position and 3d_id
    std::set<Eigen::Matrix<int,2,1>,fea_compare> features;
    {
        NonBaStageScope feature_collection(
            non_ba_profiler,
            NonBaStageId::kLocalPcdFeatureCollectionIndex,
            [&image]() { return image.Points2D().size(); });
        for (const Point2D& point2D : image.Points2D()){
            if (!point2D.HasPoint3D()) {
                continue;
            }
            Eigen::Matrix<int,2,1> uv = (point2D.XY() * scale).cast<int>();
            if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;
            features.insert(uv);
        }
        if (non_ba_profiler != nullptr) {
            feature_collection.SetOutputItems(features.size());
        }
    }

    LImage img(features,rot_cw,t_cw);

    img.img_height = img_h;
    img.img_width = img_w;
    img.img_name = image.Name();
    img.fx = params[0] * scale;
    img.fy = params[1] * scale;
    img.cx = params[2] * scale;
    img.cy = params[3] * scale;

    // Search which nodes in the map correspond to the current image
    ImageMapType img_nodes;

    {
        NonBaStageScope search_submap(
            non_ba_profiler, NonBaStageId::kLocalPcdSearchSubmap, 1);
        SearchSubMap(img, img_nodes);
        if (non_ba_profiler != nullptr) {
            search_submap.SetOutputItems(img_nodes.size());
        }
    }

    // Project lidar points to image
    {
        NonBaStageScope image_map_projection(
            non_ba_profiler, NonBaStageId::kLocalPcdImageMapProj,
            [&img_nodes]() {
              uint64_t selected_lidar_points = 0;
              for (const NodeType* node : img_nodes) {
                selected_lidar_points += node->size();
              }
              return selected_lidar_points;
            });
        ImageMapProj(img, img_nodes, camera);
        if (non_ba_profiler != nullptr) {
            image_map_projection.SetOutputItems(img.feature_pts_map.size());
        }
    }

    const size_t association_map_size_before =
        non_ba_profiler == nullptr ? 0 : map.size();
    {
        NonBaStageScope association_extraction(
            non_ba_profiler,
            NonBaStageId::kLocalPcdAssociationExtraction,
            [&image]() { return image.Points2D().size(); });
        for (const Point2D& point2D : image.Points2D()){
            if (!point2D.HasPoint3D()) {
                continue;
            }

            Eigen::Matrix<int,2,1> uv = (point2D.XY() * scale).cast<int>();
            if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;
            auto iter = img.feature_pts_map.find(uv);
            if (iter != img.feature_pts_map.end()){
                point3D_t id = point2D.GetPoint3DId();
                Eigen::Matrix<double,6,1> pt_lidar;
                pt_lidar <<static_cast<double>(iter->second.first.x),
                            static_cast<double>(iter->second.first.y),
                            static_cast<double>(iter->second.first.z),
                            static_cast<double>(iter->second.first.normal_x),
                            static_cast<double>(iter->second.first.normal_y),
                            static_cast<double>(iter->second.first.normal_z);
                map.insert({id,pt_lidar});
                img.succeed_match +=1;
            }
        }
        if (non_ba_profiler != nullptr) {
            association_extraction.SetOutputItems(
                map.size() - association_map_size_before);
        }
    }

    if (options_.if_save_depth_image){
        SaveDepthImage(img);
        std::cout<<"Saved depth image "<<img.img_name<<std::endl;
    }
    if (non_ba_profiler != nullptr) {
        set_new_image.SetOutputItems(map.size() - output_map_size_before);
    }
}

void PcdProj::SetNewImage(const Image& image, 
            const Camera& camera, 
            std::vector<std::pair<Eigen::Vector2d, bool>,Eigen::aligned_allocator<std::pair<Eigen::Vector2d, bool>>>& pt_xys, 
            std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>>& pt_xyzs){
    Eigen::Quaterniond q_cw(image.Qvec()[0],image.Qvec()[1],image.Qvec()[2],image.Qvec()[3]);
    Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();
    Eigen::Vector3d t_cw = image.Tvec();
    
    // Only for OpenCV camera model
    std::vector<double> params = camera.Params();
    double scale = options_.depth_image_scale;
    int img_h = static_cast<int>(camera.Height() * scale);
    int img_w = static_cast<int>(camera.Width() * scale);

    std::set<Eigen::Matrix<int,2,1>,fea_compare> features;
    for (const auto& pt_xy : pt_xys){

        Eigen::Matrix<int,2,1> uv = (pt_xy.first * scale).cast<int>();
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;
        features.insert(uv);
    }

    LImage img(features,rot_cw,t_cw);
    img.img_height = img_h;
    img.img_width = img_w;
    img.img_name = image.Name();
    img.fx = params[0] * scale;
    img.fy = params[1] * scale;
    img.cx = params[2] * scale;
    img.cy = params[3] * scale;

    ImageMapType img_nodes;

    SearchSubMap(img, img_nodes);

    ImageMapProj(img, img_nodes, camera);

    for (auto& pt_xy : pt_xys){

        Eigen::Matrix<int,2,1> uv = (pt_xy.first * scale).cast<int>();
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ){
            pt_xy.second = false;
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xyzs.push_back(pt_xyz);
            continue;
        } 

        auto iter = img.feature_pts_map.find(uv);
        if (iter != img.feature_pts_map.end()){
            const Eigen::Vector3d plane_point_world(
                static_cast<double>(iter->second.first.x),
                static_cast<double>(iter->second.first.y),
                static_cast<double>(iter->second.first.z));
            const Eigen::Vector3d plane_normal_world(
                static_cast<double>(iter->second.first.normal_x),
                static_cast<double>(iter->second.first.normal_y),
                static_cast<double>(iter->second.first.normal_z));
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xy.second = internal::IntersectCameraRayWithWorldPlane(
                camera,
                rot_cw,
                t_cw,
                pt_xy.first,
                plane_point_world,
                plane_normal_world,
                &pt_xyz);
            pt_xyzs.push_back(pt_xyz);
        } else {
            pt_xy.second = false;
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xyzs.push_back(pt_xyz);
        }
    }    
}

#ifdef GPU_BA_CUDA_ENABLED
bool PcdProj::ProjectImageToSnapshot(
        const Image& image,
        const Camera& camera,
        const LidarMapSnapshot& snapshot,
        std::map<point3D_t, SnapshotProjectionMatch>* matches,
        SnapshotProjectionAudit* audit,
        std::string* error) {
    if (matches != nullptr) {
        matches->clear();
    }
    if (audit != nullptr) {
        *audit = SnapshotProjectionAudit();
    }
    if (error != nullptr) {
        error->clear();
    }

    const auto fail = [audit, error](const std::string& message) {
        if (audit != nullptr) {
            *audit = SnapshotProjectionAudit();
        }
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (matches == nullptr) {
        return fail("snapshot projection output map is null");
    }

    try {
        if (snapshot.Version() == 0) {
            return fail("snapshot projection requires map version greater than zero");
        }
        if (snapshot.Frame() != LidarCoordinateFrame::COLMAP_WORLD) {
            return fail("snapshot projection requires a COLMAP_WORLD snapshot");
        }
        if (snapshot.SnapshotSha256().empty()) {
            return fail("snapshot projection requires a non-empty snapshot hash");
        }
        if (!ExistsCameraModelWithId(camera.ModelId()) ||
            !camera.VerifyParams()) {
            return fail("snapshot projection camera model or parameters are invalid");
        }
        if (camera.ModelId() != OpenCVCameraModel::model_id) {
            return fail("snapshot projection requires the OPENCV camera model");
        }
        if (camera.Width() == 0 || camera.Height() == 0) {
            return fail("snapshot projection camera dimensions must be positive");
        }

        const std::vector<double>& params = camera.Params();
        for (const double parameter : params) {
            if (!std::isfinite(parameter)) {
                return fail("snapshot projection camera parameters must be finite");
            }
        }
        const double fx = params[0];
        const double fy = params[1];
        const double cx = params[2];
        const double cy = params[3];
        if (!(fx > 0.0) || !(fy > 0.0)) {
            return fail("snapshot projection focal lengths must be positive");
        }

        if (!std::isfinite(options_.depth_image_scale) ||
            !(options_.depth_image_scale > 0.0) ||
            !std::isfinite(static_cast<double>(options_.choose_meter)) ||
            !(options_.choose_meter > 0.0f) ||
            !std::isfinite(options_.min_proj_dist) ||
            !(options_.min_proj_dist >= 0.0) ||
            !(options_.min_proj_dist <
              static_cast<double>(options_.choose_meter)) ||
            !std::isfinite(options_.min_lidar_proj_dist) ||
            !(options_.min_lidar_proj_dist >= 0.0) ||
            !(options_.min_lidar_proj_dist <= options_.min_proj_dist) ||
            options_.min_proj_scale < 0 || options_.max_proj_scale < 0 ||
            options_.min_proj_scale > options_.max_proj_scale) {
            return fail("snapshot projection options are invalid");
        }

        const double scaled_height =
            static_cast<double>(camera.Height()) * options_.depth_image_scale;
        const double scaled_width =
            static_cast<double>(camera.Width()) * options_.depth_image_scale;
        if (!std::isfinite(scaled_height) || !std::isfinite(scaled_width) ||
            scaled_height < 1.0 || scaled_width < 1.0 ||
            scaled_height > static_cast<double>(std::numeric_limits<int>::max()) ||
            scaled_width > static_cast<double>(std::numeric_limits<int>::max())) {
            return fail("snapshot projection scaled image dimensions are invalid");
        }
        const int image_height = static_cast<int>(scaled_height);
        const int image_width = static_cast<int>(scaled_width);

        const Eigen::Vector4d& qvec = image.Qvec();
        const Eigen::Vector3d& tvec = image.Tvec();
        const double quaternion_norm = qvec.norm();
        if (!qvec.allFinite() || !tvec.allFinite() ||
            !std::isfinite(quaternion_norm) ||
            std::abs(quaternion_norm - 1.0) > 1e-6) {
            return fail("snapshot projection image pose is invalid");
        }

        Eigen::Quaterniond q_cw(qvec[0], qvec[1], qvec[2], qvec[3]);
        Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();
        Eigen::Vector3d t_cw = tvec;
        struct FeatureObservation {
            point3D_t point3D_id = kInvalidPoint3DId;
            point2D_t point2D_idx = kInvalidPoint2DIdx;
            Eigen::Matrix<int, 2, 1> scaled_pixel =
                Eigen::Matrix<int, 2, 1>::Zero();
        };

        if (image.Points2D().size() >
            static_cast<size_t>(kInvalidPoint2DIdx)) {
            return fail("snapshot projection image has too many features");
        }
        std::set<point3D_t> point3D_ids;
        std::set<Eigen::Matrix<int,2,1>,fea_compare> feature_pixels;
        std::vector<FeatureObservation> observations;
        observations.reserve(image.Points2D().size());

        SnapshotProjectionAudit local_audit;
        local_audit.map_version = snapshot.Version();
        local_audit.snapshot_sha256 = snapshot.SnapshotSha256();
        local_audit.geometry_sha256 = snapshot.GeometrySha256();
        local_audit.snapshot_voxel_count = snapshot.VoxelCount();
        local_audit.snapshot_block_count = snapshot.BlockCount();
        bool has_nonzero_opencv_distortion = false;
        for (size_t parameter_index = 4; parameter_index < params.size();
             ++parameter_index) {
            has_nonzero_opencv_distortion =
                has_nonzero_opencv_distortion ||
                params[parameter_index] != 0.0;
        }
        const bool use_local_undistorted_cone =
            camera.IsUndistorted() && !has_nonzero_opencv_distortion;
        local_audit.full_snapshot_due_to_distortion =
            !use_local_undistorted_cone;

        for (size_t point2D_idx = 0; point2D_idx < image.Points2D().size();
             ++point2D_idx) {
            const Point2D& point2D = image.Points2D()[point2D_idx];
            if (!point2D.HasPoint3D()) {
                continue;
            }
            ++local_audit.input_feature_count;
            if (!point2D.XY().allFinite()) {
                return fail("snapshot projection image contains a non-finite feature");
            }
            const point3D_t point3D_id = point2D.GetPoint3DId();
            if (!point3D_ids.insert(point3D_id).second) {
                return fail(
                    "snapshot projection image contains a duplicate point3D_id");
            }

            const Eigen::Vector2d scaled =
                point2D.XY() * options_.depth_image_scale;
            if (!scaled.allFinite() || scaled.x() < 0.0 ||
                scaled.x() >= static_cast<double>(image_width) ||
                scaled.y() < 0.0 ||
                scaled.y() >= static_cast<double>(image_height)) {
                continue;
            }
            const double truncated_u = std::trunc(scaled.x());
            const double truncated_v = std::trunc(scaled.y());

            FeatureObservation observation;
            observation.point3D_id = point3D_id;
            observation.point2D_idx = static_cast<point2D_t>(point2D_idx);
            observation.scaled_pixel << static_cast<int>(truncated_u),
                                         static_cast<int>(truncated_v);
            observations.push_back(observation);
            feature_pixels.insert(observation.scaled_pixel);
            ++local_audit.in_image_feature_count;
        }

        LImage projection_image(feature_pixels, rot_cw, t_cw);
        projection_image.img_height = image_height;
        projection_image.img_width = image_width;
        projection_image.img_name = image.Name();
        projection_image.fx = fx * options_.depth_image_scale;
        projection_image.fy = fy * options_.depth_image_scale;
        projection_image.cx = cx * options_.depth_image_scale;
        projection_image.cy = cy * options_.depth_image_scale;
        if (!std::isfinite(projection_image.fx) ||
            !std::isfinite(projection_image.fy) ||
            !std::isfinite(projection_image.cx) ||
            !std::isfinite(projection_image.cy) ||
            !(projection_image.fx > 0.0) || !(projection_image.fy > 0.0)) {
            return fail("snapshot projection scaled intrinsics are invalid");
        }

        const double max_proj_scale_x =
            static_cast<double>(options_.max_proj_scale) * (fx / 3039.0) *
            (options_.depth_image_scale / 0.2);
        const double max_proj_scale_y =
            static_cast<double>(options_.max_proj_scale) * (fy / 3039.0) *
            (options_.depth_image_scale / 0.2);
        const double min_proj_scale_x =
            static_cast<double>(options_.min_proj_scale) * (fx / 3039.0) *
            (options_.depth_image_scale / 0.2);
        const double min_proj_scale_y =
            static_cast<double>(options_.min_proj_scale) * (fy / 3039.0) *
            (options_.depth_image_scale / 0.2);
        const double a_x = (max_proj_scale_x - min_proj_scale_x) /
                           (options_.min_proj_dist -
                            static_cast<double>(options_.choose_meter));
        const double b_x = min_proj_scale_x -
                           a_x * static_cast<double>(options_.choose_meter);
        const double a_y = (max_proj_scale_y - min_proj_scale_y) /
                           (options_.min_proj_dist -
                            static_cast<double>(options_.choose_meter));
        const double b_y = min_proj_scale_y -
                           a_y * static_cast<double>(options_.choose_meter);
        const std::array<double, 8> coverage_coefficients = {{
            max_proj_scale_x, max_proj_scale_y, min_proj_scale_x,
            min_proj_scale_y, a_x, b_x, a_y, b_y}};
        for (const double coefficient : coverage_coefficients) {
            if (!std::isfinite(coefficient)) {
                return fail(
                    "snapshot projection coverage coefficients are invalid");
            }
        }
        if (max_proj_scale_x < 0.0 || max_proj_scale_y < 0.0 ||
            static_cast<long double>(max_proj_scale_x) >
                static_cast<long double>(
                    std::numeric_limits<int64_t>::max()) ||
            static_cast<long double>(max_proj_scale_y) >
                static_cast<long double>(
                    std::numeric_limits<int64_t>::max())) {
            return fail(
                "snapshot projection coverage radius exceeds int64 range");
        }

        const Eigen::Matrix3d rot_wc = rot_cw.transpose();
        const Eigen::Vector3d t_wc = -rot_wc * t_cw;
        const double far_distance =
            static_cast<double>(options_.choose_meter);
        if (use_local_undistorted_cone) {
            const double expanded_minimum_u = -max_proj_scale_x - 0.5;
            const double expanded_maximum_u =
                static_cast<double>(image_width) - 0.5 + max_proj_scale_x;
            const double expanded_minimum_v = -max_proj_scale_y - 0.5;
            const double expanded_maximum_v =
                static_cast<double>(image_height) - 0.5 + max_proj_scale_y;
            const std::array<Eigen::Vector2d, 4> expanded_pixel_corners = {{
                Eigen::Vector2d(expanded_maximum_u,
                                expanded_maximum_v) /
                    options_.depth_image_scale,
                Eigen::Vector2d(expanded_maximum_u,
                                expanded_minimum_v) /
                    options_.depth_image_scale,
                Eigen::Vector2d(expanded_minimum_u,
                                expanded_minimum_v) /
                    options_.depth_image_scale,
                Eigen::Vector2d(expanded_minimum_u,
                                expanded_maximum_v) /
                    options_.depth_image_scale}};
            std::array<Eigen::Vector3d, 4> far_corners;
            for (size_t corner_index = 0;
                 corner_index < expanded_pixel_corners.size();
                 ++corner_index) {
                const Eigen::Vector2d normalized =
                    camera.ImageToWorld(expanded_pixel_corners[corner_index]);
                if (!expanded_pixel_corners[corner_index].allFinite() ||
                    !normalized.allFinite()) {
                    return fail(
                        "snapshot projection camera boundary is non-finite");
                }
                far_corners[corner_index] =
                    t_wc +
                    rot_wc * Eigen::Vector3d(normalized.x() * far_distance,
                                             normalized.y() * far_distance,
                                             far_distance);
                if (!far_corners[corner_index].allFinite()) {
                    return fail("snapshot projection frustum is non-finite");
                }
            }
            for (size_t axis = 0; axis < 3; ++axis) {
                double minimum = t_wc(axis);
                double maximum = t_wc(axis);
                for (const Eigen::Vector3d& corner : far_corners) {
                    minimum = std::min(minimum, corner(axis));
                    maximum = std::max(maximum, corner(axis));
                }
                local_audit.projection_aabb.min[axis] = std::nextafter(
                    minimum, std::numeric_limits<double>::lowest());
                local_audit.projection_aabb.max[axis] = std::nextafter(
                    maximum, std::numeric_limits<double>::max());
            }
        } else {
            local_audit.projection_aabb.min = {{
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest()}};
            local_audit.projection_aabb.max = {{
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max()}};
        }
        local_audit.projection_aabb.frame =
            LidarCoordinateFrame::COLMAP_WORLD;

        const PlaneCollectionResult collection = snapshot.CollectPlanesInAabb(
            local_audit.projection_aabb, LidarNormalScale::PROJECTION);
        if (!collection.ok) {
            return fail("snapshot projection AABB query failed: " +
                        collection.error);
        }
        local_audit.aabb_visited_block_count =
            collection.visited_block_count;
        local_audit.candidate_plane_count =
            static_cast<uint64_t>(collection.planes.size());

        struct PixelWinner {
            PlaneSample plane;
            float camera_distance = 0.0f;
        };
        std::map<Eigen::Matrix<int,2,1>, PixelWinner, fea_compare>
            pixel_winners;
        bool has_previous_key = false;
        VoxelKey previous_key;
        for (const PlaneSample& plane : collection.planes) {
            if (plane.frame != LidarCoordinateFrame::COLMAP_WORLD ||
                plane.scale != LidarNormalScale::PROJECTION ||
                plane.voxel_count == 0 ||
                (has_previous_key && !(previous_key < plane.key))) {
                return fail("snapshot projection received invalid plane metadata");
            }
            double normal_squared_norm = 0.0;
            for (size_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(plane.point[axis]) ||
                    !std::isfinite(plane.normal[axis])) {
                    return fail(
                        "snapshot projection received a non-finite plane");
                }
                normal_squared_norm +=
                    static_cast<double>(plane.normal[axis]) *
                    plane.normal[axis];
            }
            if (!std::isfinite(normal_squared_norm) ||
                !(normal_squared_norm > 0.0)) {
                return fail("snapshot projection received an invalid normal");
            }
            previous_key = plane.key;
            has_previous_key = true;

            const Eigen::Vector3d point_world(
                static_cast<double>(plane.point[0]),
                static_cast<double>(plane.point[1]),
                static_cast<double>(plane.point[2]));
            const Eigen::Vector3d point_camera =
                rot_cw * point_world + t_cw;
            if (!point_camera.allFinite()) {
                return fail(
                    "snapshot projection produced a non-finite camera point");
            }
            const double axial_distance = point_camera(2);
            if (!(axial_distance > 0.0)) {
                continue;
            }
            ++local_audit.positive_depth_hit_count;
            if (!(axial_distance <= far_distance)) {
                continue;
            }
            ++local_audit.axial_depth_hit_count;

            int64_t scale_x = 0;
            int64_t scale_y = 0;
            if (axial_distance < options_.min_lidar_proj_dist) {
                continue;
            } else if (options_.min_lidar_proj_dist <= axial_distance &&
                       axial_distance <= options_.min_proj_dist) {
                scale_x = static_cast<int64_t>(max_proj_scale_x);
                scale_y = static_cast<int64_t>(max_proj_scale_y);
            } else {
                const double radius_x = a_x * axial_distance + b_x;
                const double radius_y = a_y * axial_distance + b_y;
                if (!std::isfinite(radius_x) || !std::isfinite(radius_y) ||
                    radius_x < 0.0 || radius_y < 0.0 ||
                    static_cast<long double>(radius_x) >
                        static_cast<long double>(
                            std::numeric_limits<int64_t>::max()) ||
                    static_cast<long double>(radius_y) >
                        static_cast<long double>(
                            std::numeric_limits<int64_t>::max())) {
                    return fail(
                        "snapshot projection coverage radius is invalid");
                }
                scale_x = static_cast<int64_t>(radius_x);
                scale_y = static_cast<int64_t>(radius_y);
            }

            const Eigen::Vector2d distorted_pixel = camera.WorldToImage(
                Eigen::Vector2d(point_camera(0) / axial_distance,
                                point_camera(1) / axial_distance));
            const double rounded_u = std::round(
                distorted_pixel(0) * options_.depth_image_scale);
            const double rounded_v = std::round(
                distorted_pixel(1) * options_.depth_image_scale);
            if (!distorted_pixel.allFinite() || !std::isfinite(rounded_u) ||
                !std::isfinite(rounded_v)) {
                return fail(
                    "snapshot projection produced a non-finite image point");
            }
            if (static_cast<long double>(rounded_u) + scale_x < 0.0L ||
                static_cast<long double>(rounded_u) - scale_x >
                    static_cast<long double>(image_width) - 1.0L ||
                static_cast<long double>(rounded_v) + scale_y < 0.0L ||
                static_cast<long double>(rounded_v) - scale_y >
                    static_cast<long double>(image_height) - 1.0L) {
                continue;
            }
            const long double minimum_int64 = static_cast<long double>(
                std::numeric_limits<int64_t>::min());
            const long double maximum_int64 = static_cast<long double>(
                std::numeric_limits<int64_t>::max());
            if (static_cast<long double>(rounded_u) <
                    minimum_int64 + scale_x ||
                static_cast<long double>(rounded_u) >
                    maximum_int64 - scale_x ||
                static_cast<long double>(rounded_v) <
                    minimum_int64 + scale_y ||
                static_cast<long double>(rounded_v) >
                    maximum_int64 - scale_y) {
                return fail(
                    "snapshot projection rounded center exceeds int64 range");
            }
            const int64_t u0 = static_cast<int64_t>(rounded_u);
            const int64_t v0 = static_cast<int64_t>(rounded_v);
            const int64_t minimum_u = u0 - scale_x;
            const int64_t maximum_u = u0 + scale_x;
            const int64_t minimum_v = v0 - scale_y;
            const int64_t maximum_v = v0 + scale_y;
            const int64_t clipped_minimum_u = std::max<int64_t>(0, minimum_u);
            const int64_t clipped_maximum_u = std::min<int64_t>(
                static_cast<int64_t>(image_width) - 1, maximum_u);
            const int64_t clipped_minimum_v = std::max<int64_t>(0, minimum_v);
            const int64_t clipped_maximum_v = std::min<int64_t>(
                static_cast<int64_t>(image_height) - 1, maximum_v);
            if (clipped_minimum_u > clipped_maximum_u ||
                clipped_minimum_v > clipped_maximum_v) {
                continue;
            }
            ++local_audit.pixel_hit_count;
            const uint64_t visited_width = static_cast<uint64_t>(
                clipped_maximum_u - clipped_minimum_u + 1);
            const uint64_t visited_height = static_cast<uint64_t>(
                clipped_maximum_v - clipped_minimum_v + 1);
            if (visited_width >
                    std::numeric_limits<uint64_t>::max() / visited_height ||
                local_audit.coverage_pixel_visit_count >
                    std::numeric_limits<uint64_t>::max() -
                        visited_width * visited_height) {
                return fail(
                    "snapshot projection pixel visit count overflowed");
            }
            const float camera_distance =
                static_cast<float>(point_camera.norm());
            if (!std::isfinite(camera_distance)) {
                return fail(
                    "snapshot projection camera distance is non-finite");
            }
            for (int64_t u = clipped_minimum_u; u <= clipped_maximum_u; ++u) {
                for (int64_t v = clipped_minimum_v; v <= clipped_maximum_v;
                     ++v) {
                    ++local_audit.coverage_pixel_visit_count;
                    Eigen::Matrix<int,2,1> feature_pixel;
                    feature_pixel << static_cast<int>(u), static_cast<int>(v);
                    if (projection_image.feature_points.find(feature_pixel) ==
                        projection_image.feature_points.end()) {
                        continue;
                    }
                    ++local_audit.feature_coverage_hit_count;
                    auto winner = pixel_winners.find(feature_pixel);
                    if (winner == pixel_winners.end()) {
                        PixelWinner new_winner;
                        new_winner.plane = plane;
                        new_winner.camera_distance = camera_distance;
                        pixel_winners.emplace(feature_pixel, new_winner);
                    } else if (
                        camera_distance < winner->second.camera_distance ||
                        (camera_distance == winner->second.camera_distance &&
                         plane.key < winner->second.plane.key)) {
                        winner->second.plane = plane;
                        winner->second.camera_distance = camera_distance;
                    }
                }
            }
        }
        local_audit.feature_pixel_hit_count =
            static_cast<uint64_t>(pixel_winners.size());

        std::map<point3D_t, SnapshotProjectionMatch> local_matches;
        for (const FeatureObservation& observation : observations) {
            const auto winner = pixel_winners.find(observation.scaled_pixel);
            if (winner == pixel_winners.end()) {
                continue;
            }
            SnapshotProjectionMatch match;
            match.point3D_id = observation.point3D_id;
            match.point2D_idx = observation.point2D_idx;
            match.scaled_pixel = {{observation.scaled_pixel(0),
                                   observation.scaled_pixel(1)}};
            match.plane = winner->second.plane;
            match.plane_key = winner->second.plane.key;
            match.map_version = snapshot.Version();
            match.camera_distance = winner->second.camera_distance;
            if (!local_matches.emplace(match.point3D_id, match).second) {
                return fail(
                    "snapshot projection produced a duplicate point3D match");
            }
            ++local_audit.feature_hit_count;
        }

        matches->swap(local_matches);
        if (audit != nullptr) {
            *audit = std::move(local_audit);
        }
        return true;
    } catch (const std::exception& exception) {
        matches->clear();
        return fail(std::string("snapshot projection threw exception: ") +
                    exception.what());
    } catch (...) {
        matches->clear();
        return fail("snapshot projection threw non-standard exception");
    }
}
#endif

void PcdProj::BuildSubMap(const MapType& ptr){
    global_map_ptr_ = ptr;
    for (PointType& pt : ptr->points){
        global_map_min_x_ = std::min(global_map_min_x_, pt.x);
        global_map_max_x_ = std::max(global_map_max_x_, pt.x);
        global_map_min_y_ = std::min(global_map_min_y_, pt.y);
        global_map_max_y_ = std::max(global_map_max_y_, pt.y);
        global_map_min_z_ = std::min(global_map_min_z_, pt.z);
        global_map_max_z_ = std::max(global_map_max_z_, pt.z);

        auto key = GetKeyType(pt);
        auto iter = submap_.find(key);
        if (iter == submap_.end()){
            NodeType node;
            node.push_back(pt);
            submap_.insert({key,node});
            submap_num_ +=1;
        } else {
            iter->second.push_back(pt);
        }
    }
}

void PcdProj::SearchSubMap(const LImage& img, ImageMapType& image_map){

    Eigen::Matrix3f rot_wc = img.rot_cw.cast<float>().transpose();
    Eigen::Vector3f t_wc = - rot_wc * img.t_cw.cast<float>();

    Eigen::Vector3f center_v(0.0,0.0,1.0);
    Eigen::Vector3f x_bar_v(1.0,0.0,0.0);
    Eigen::Vector3f y_bar_v(0.0,1.0,0.0);
    float x_bar_min = -img.cx / img.fx;
    float x_bar_max = (img.img_width-img.cx) / img.fx;
    float y_bar_min = -img.cy / img.fy;
    float y_bar_max = (img.img_height - img.cy) / img.fy;

    Eigen::Vector3f corner_1 = x_bar_v * x_bar_max  + y_bar_v * y_bar_max;
    Eigen::Vector3f corner_2 = x_bar_v * x_bar_max  + y_bar_v * y_bar_min;
    Eigen::Vector3f corner_3 = x_bar_v * x_bar_min  + y_bar_v * y_bar_min;
    Eigen::Vector3f corner_4 = x_bar_v * x_bar_min  + y_bar_v * y_bar_max;
    // Get the four corners of the pyramid
    corner_1 = (t_wc + rot_wc * (center_v + corner_1) * options_.choose_meter).eval();
    corner_2 = (t_wc + rot_wc * (center_v + corner_2) * options_.choose_meter).eval();
    corner_3 = (t_wc + rot_wc * (center_v + corner_3) * options_.choose_meter).eval();
    corner_4 = (t_wc + rot_wc * (center_v + corner_4) * options_.choose_meter).eval();

    // Initialize the pyramid
    QuadPyramid quad_pyramid(t_wc,corner_1,corner_2,corner_3,corner_4);
  
    SearchImageMap(quad_pyramid,image_map);
}

void PcdProj::ImageMapProj(LImage& img, ImageMapType& image_map, const Camera& camera){
    Eigen::Matrix3f rot_cw = img.rot_cw.cast<float>();
    Eigen::Vector3f t_cw = img.t_cw.cast<float>();

    std::ofstream ofs;
    if (options_.if_save_lidar_frame) {
        std::string substr;
        std::stringstream s_stream(img.img_name);
        std::getline(s_stream, substr, '.');
        std::string point_cloud_write_path = options_.lidar_frame_folder + "/" 
                                            + substr + ".txt";
        ofs.open(point_cloud_write_path, std::ios::out | std::ios::trunc);
    }
    int num = image_map.size();
    # pragma omp parallel for
    for (int i = 0; i < num; i++){
        NodeType* node_ptr = image_map[i];
        // for (NodeType** iter = image_map.begin(); iter != image_map.end(); iter++){
        // NodeType* node_ptr = *iter;
        for (PointType& pt : *node_ptr){

            Eigen::Vector3f pt_w = pt.getVector3fMap();
            // Write out point cloud file
            if (options_.if_save_lidar_frame) {
                std::lock_guard<std::mutex> lock(proj_mutex_);
                ofs << pt_w(2)<<" "<<-pt_w(0)<<" "<<-pt_w(1)<<std::endl;
            }

            Eigen::Vector3f pt_c = rot_cw * pt_w + t_cw;
            if (pt_c(2) < 0) continue;
            // Point's coordinate of the original image
            std::vector<double> params = camera.Params();
            double fx = params[0]; 
            double fy = params[1];
            double cx = params[2];
            double cy = params[3];
            double depth_image_scale = options_.depth_image_scale;
            double u_ori = fx * static_cast<double>(pt_c(0) / pt_c(2)) + cx;
            double v_ori = fy * static_cast<double>(pt_c(1) / pt_c(2)) + cy;

            // Distort pixel
            Vector2d uv_ori;
            uv_ori << u_ori, v_ori;
            Vector2d uv_dis;
            uv_dis = DistortOpenCV(uv_ori, camera);
            int u0 = int(round(uv_dis(0) * depth_image_scale));
            int v0 = int(round(uv_dis(1) * depth_image_scale));
            // lidar point near the image should have large scale
            // lidar point far from the image should have small scale
            // The most appropriate scale can make the lidar projection cover the image
            // Adjust scale by focal length
            float dist = pt_c(2);
            // scale = ax + b;
            int scale_x;
            int scale_y;

            double max_proj_scale_x = static_cast<double>(options_.max_proj_scale) * (fx/3039.0) * (depth_image_scale/0.2);
            double max_proj_scale_y = static_cast<double>(options_.max_proj_scale) * (fy/3039.0) * (depth_image_scale/0.2);

            double min_proj_scale_x = static_cast<double>(options_.min_proj_scale) * (fx/3039.0) * (depth_image_scale/0.2);
            double min_proj_scale_y = static_cast<double>(options_.min_proj_scale) * (fy/3039.0) * (depth_image_scale/0.2);

            static double a_x = (max_proj_scale_x - min_proj_scale_x)/
                        (options_.min_proj_dist - static_cast<double>(options_.choose_meter));
            static double b_x = min_proj_scale_x - a_x * static_cast<double>(options_.choose_meter);

            static double a_y = (max_proj_scale_y - min_proj_scale_y)/
                        (options_.min_proj_dist - static_cast<double>(options_.choose_meter));
            static double b_y = static_cast<double>(options_.min_proj_scale) - a_y * static_cast<double>(options_.choose_meter);

            if (dist < options_.min_lidar_proj_dist) {
                continue;
            } else if (options_.min_lidar_proj_dist <= dist && dist<= options_.min_proj_dist) {
                scale_x = static_cast<int>(max_proj_scale_x); 
                scale_y = static_cast<int>(max_proj_scale_y);
            } else if (dist > options_.min_proj_dist){
                scale_x = static_cast<int>(a_x * dist + b_x);
                scale_y = static_cast<int>(a_y * dist + b_y);
            } else {
                std::cout<<"Please resolve the parameter conflict"<<std::endl;
                continue;
            }
            for (int u = u0 - scale_x; u <= u0 + scale_x; u++){
            for (int v = v0 - scale_y; v <= v0 + scale_y; v++){
                if(u < 0 || u >= img.img_width || v < 0 || v >= img.img_height){
                    continue;
                }
                Eigen::Matrix<int,2,1> uv;
                uv << u, v;
                float dist = pt_c.norm();

                if(options_.if_save_depth_image){
                    // The distance of the lidar point from the center of the camera
                    std::lock_guard<std::mutex> lock(proj_mutex_);
                    auto iter = img.dist_map.find(uv);
                    if (iter != img.dist_map.end()){
                        iter->second = std::min(iter->second,dist);
                    } else {
                        img.dist_map.insert({uv,dist});
                    }

                } 
                
                if (img.feature_points.find(uv)==img.feature_points.end()) continue;
                // The distance of the lidar point from the center of the camera
                std::lock_guard<std::mutex> lock(proj_mutex_);
                auto iter = img.feature_pts_map.find(uv);
                if (iter!= img.feature_pts_map.end()){
                    if (iter->second.second > dist){
                        iter->second = std::make_pair(pt,dist);
                    } else {continue;}
                } else {
                    img.feature_pts_map.insert({uv,std::make_pair(pt,dist)});
                }
                
            }
            }
        }
    }
}

void PcdProj::SaveDepthImage(const LImage& img){
    std::string folder_path = options_.original_image_folder + "/";
    std::string image_path = folder_path + img.img_name;
    cv::Mat original_image = cv::imread(image_path);

    cv::resize(original_image, original_image, cv::Size(img.img_width,img.img_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat depth_image(img.img_height,img.img_width,CV_8UC3,cv::Scalar(255,255,255));
    float color_scale = 255/options_.choose_meter;
    for (auto iter : img.dist_map){
        
        int u = iter.first(0);
        int v = iter.first(1);
        depth_image.at<cv::Vec3b>(v,u)[0] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
        depth_image.at<cv::Vec3b>(v,u)[1] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
        depth_image.at<cv::Vec3b>(v,u)[2] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
    }

    cv::Mat result_image(img.img_height,img.img_width,CV_8UC3);
    cv::addWeighted(depth_image,0.8,original_image,0.2,0,result_image);

    for(auto& point : img.feature_points){
		cv::circle(result_image, cv::Point(point(0),point(1)), 4, cv::Scalar(0, 255, 120), -1);
	}

    std::string result_image_path = options_.depth_image_folder + "/";
    std::string result_image_name = result_image_path + img.img_name;
    cv::imwrite(result_image_name,result_image);
}

void PcdProj::SearchImageMap(QuadPyramid& quad, ImageMapType& image_map){

    std::vector<float> x{quad.vertex(0),quad.corner_1(0),quad.corner_2(0),quad.corner_3(0),
            quad.corner_4(0), global_map_min_x_,global_map_max_x_};
    std::vector<float> y{quad.vertex(1),quad.corner_1(1),quad.corner_2(1),quad.corner_3(1),
            quad.corner_4(1), global_map_min_y_,global_map_max_y_};
    std::vector<float> z{quad.vertex(2),quad.corner_1(2),quad.corner_2(2),quad.corner_3(2),
            quad.corner_4(2), global_map_min_z_,global_map_max_z_};

    std::sort(x.begin(),x.end());
    std::sort(y.begin(),y.end());
    std::sort(z.begin(),z.end());

    int x_min = round(x.front()/options_.submap_length);
    int x_max = round(x.back()/options_.submap_length);
    int y_min = round(y.front()/options_.submap_height);
    int y_max = round(y.back()/options_.submap_height);
    int z_min = round(z.front()/options_.submap_width);
    int z_max = round(z.back()/options_.submap_width);
    
    for(int idx = x_min-1; idx <= x_max+1; idx++){
        float x = idx * options_.submap_length;
        for(int idy = y_min-1; idy <= y_max+1; idy++){
            float y = idy * options_.submap_height;
            for(int idz = z_min-1; idz <= z_max+1; idz++){
                float z = idz * options_.submap_width;
                bool condition_0 = (quad.plane_0(0)*x + quad.plane_0(1)*y + quad.plane_0(2)*z + quad.plane_0(3) <= 0.0);
                bool condition_1 = (quad.plane_1(0)*x + quad.plane_1(1)*y + quad.plane_1(2)*z + quad.plane_1(3) <= 0.0);
                bool condition_2 = (quad.plane_2(0)*x + quad.plane_2(1)*y + quad.plane_2(2)*z + quad.plane_2(3) <= 0.0);
                bool condition_3 = (quad.plane_3(0)*x + quad.plane_3(1)*y + quad.plane_3(2)*z + quad.plane_3(3) <= 0.0);
                bool condition_4 = (quad.plane_4(0)*x + quad.plane_4(1)*y + quad.plane_4(2)*z + quad.plane_4(3) <= 0.0);
                if(condition_1 && condition_2 && condition_3 && condition_4 && condition_0){
                    KeyType key;
                    key << idx, idy, idz;
                    auto iter = submap_.find(key);
                    if (iter != submap_.end()){image_map.push_back(&(iter->second));}
                }
            }
        } 
    }
    return ;
}

Eigen::Vector2d PcdProj::DistortOpenCV(Eigen::Vector2d& ori_uv, const Camera& camera){
    
    std::vector<double> params = camera.Params();
    double fx = params[0]; 
    double fy = params[1];
    double cx = params[2];
    double cy = params[3];
    double k1 = params[4];
    double k2 = params[5];
    double p1 = params[6];
    double p2 = params[7];

    // Normalization
    double x_corrected = (ori_uv(0) - cx) / fx;
    double y_corrected = (ori_uv(1) - cy) / fy;

    double r2 = x_corrected * x_corrected + y_corrected * y_corrected;
    double deltaRa = 1. + k1 * r2 + k2 * r2 * r2;
    double deltaRb = 1;
    double deltaTx = 2. * p1 * x_corrected * y_corrected + p2 * (r2 + 2. * x_corrected * x_corrected);
    double deltaTy = p1 * (r2 + 2. * y_corrected * y_corrected) + 2. * p2 * x_corrected * y_corrected;

    double distort_u0 = x_corrected * deltaRa * deltaRb + deltaTx;
    double distort_v0 = y_corrected * deltaRa * deltaRb + deltaTy;

    distort_u0 = distort_u0 * fx + cx;
    distort_v0 = distort_v0 * fy + cy;

    Eigen::Vector2d dis_uv;
    dis_uv << distort_u0, distort_v0;

    return dis_uv;

}

} //namespace lidar
} //namespace colmap
