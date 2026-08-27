#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"
#include "pcd_projection.h"
#include "sfm/nonba_profiler.h"

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

    // Get the equation of the plane
    double fx = params[0];
    double fy = params[1];
    double cx = params[2];
    double cy = params[3];

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
            pt_xy.second = true;
            // u = fx * x / z + cx
            // v = fy * y / z + cy
            // z * (u - cx)/fx = x
            // z * (v - cy)/fy = y
            // ax + by + cz + d = 0
            // (a * (u - cx)/fx + b * (v - cy)/fy + c ) * z + d = 0
            double a = static_cast<double>(iter->second.first.normal_x);
            double b = static_cast<double>(iter->second.first.normal_y);
            double c = static_cast<double>(iter->second.first.normal_z);
            double d = 0 - a * static_cast<double>(iter->second.first.x)
                         - b * static_cast<double>(iter->second.first.y)
                         - c * static_cast<double>(iter->second.first.z);
            double u = pt_xy.first(0);
            double v = pt_xy.first(1);
            double z = - d / (a * (u - cx)/fx + b * (v - cy)/fy + c );
            double x = z * (u - cx)/fx;
            double y = z * (v - cy)/fy;
            Eigen::Vector3d pt_xyz;
            pt_xyz << x,y,z;
            pt_xyzs.push_back(pt_xyz);
        } else {
            pt_xy.second = false;
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xyzs.push_back(pt_xyz);
        }
    }    
}
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
