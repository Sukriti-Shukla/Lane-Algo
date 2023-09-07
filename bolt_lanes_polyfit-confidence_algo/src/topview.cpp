#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_ros/transform_listener.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv4/opencv2/imgproc.hpp>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;
using namespace cv_bridge;
using namespace sensor_msgs;

image_transport::Publisher image_pub;
tf2_ros::Buffer tf_buffer;
image_geometry::PinholeCameraModel camera_model;

std::string ground_frame = "map";

double scale = 15;
double x_offset = -2.0;
double y_offset = 10.0;
double height = 10.0;
double width = 20.0;

std::string frame_id_depth = "zed2i_left_camera_optical_frame";

[[nodiscard]] std::array<int, 2> worldToImgNoBounds(double x, double y)
{
    return {
        static_cast<int>((x + x_offset) * scale),
        static_cast<int>((y + y_offset) * scale)};
}

[[nodiscard]] bool imgCheckBounds(Mat &img, int i, int j)
{
    return ((i >= 0) && (i < img.rows) && (j >= 0) && (j < img.cols));
}

[[nodiscard]] std::array<double, 2> imgToWorld(int i, int j)
{
    return {
        static_cast<double>(i) / scale - x_offset,
        static_cast<double>(j) / scale - y_offset,
    };
}

[[nodiscard]] std::array<double, 2> imgDirToWorld(int i, int j)
{
    // Use when we don't need the offset,only the relative direction
    return {
        static_cast<double>(i) / scale,
        static_cast<double>(j) / scale,
    };
}

void imagecallback(const sensor_msgs::ImageConstPtr &msg)
{
    auto time = ros::Time::now();
    // ROS_INFO("Message is received");
    const auto &camera_info_ptr = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
        "/zed2i/zed_node/rgb/camera_info");

    if (!camera_info_ptr)
    {
        ROS_FATAL("Could not get camera_info from the topic /zed2i/zed_node/rgb/camera_info");
        // ros::shutdown();
    }

    CvImagePtr cv_ptr;
    Mat birdseye = Mat::zeros(static_cast<int>(height * scale), static_cast<int>(width * scale), CV_8UC1);
    try
    {
        cv_ptr = toCvCopy(msg, image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    Mat img_rec = cv_ptr->image;
    const int rows = img_rec.rows, cols = img_rec.cols;
    const auto rows_to_keep = static_cast<int>(0.4 * rows);
    // ROS_WARN("Is this even executed2?");
    //  auto rows_to_keep = rows;
    cv::Mat img{img_rec, cv::Rect{0, rows - rows_to_keep, cols, rows_to_keep}};
    // cv::imshow("Cropped image", img);
    // cv::waitKey(0);
    geometry_msgs::TransformStamped transform;
    try
    {
        transform = tf_buffer.lookupTransform(ground_frame, frame_id_depth, ros::Time(0));
    }
    catch (const tf2::TransformException &e)
    {
        // Catches tf2::LookupException and tf2::ConnectivityException
        // ROS_ERROR_THROTTLE(1.0, "Couldn't get transform from %s to %s", frame_id_depth, ground_frame.c_str());
    }
    // ROS_ERROR("%s", frame_id_depth.c_str());
    Eigen::Transform<double, 3, Eigen::Isometry> transformer = Eigen::Translation3d(transform.transform.translation.x, transform.transform.translation.y,
                                                                                    transform.transform.translation.z) *
                                                               Eigen::Quaternion<double>(
                                                                   transform.transform.rotation.w, transform.transform.rotation.x,
                                                                   transform.transform.rotation.y, transform.transform.rotation.z);

    auto base_footprint_trans = transformer * Eigen::Vector3d(0, 0, 0);
    base_footprint_trans.z() -= 1.0; // 1.0 is the fake camera height

    Eigen::Vector3d forward_dir = transformer * Eigen::Vector3d(0, 0, 1) - base_footprint_trans;
    forward_dir.z() = 0.0;
    forward_dir.normalize();
    auto theta = std::atan2(forward_dir.y(), forward_dir.x());

    // Transform point from zed_optical_frame to 'base_footprint'
    auto transform_point = [&transformer, &base_footprint_trans,
                            ctheta = std::cos(theta), stheta = std::sin(theta)](double x, double y, double z) -> Eigen::Vector3d
    {
        const auto ray_e = transformer * Eigen::Vector3d(x, y, z) - base_footprint_trans;

        // Rotate to zed yaw frame
        return {ctheta * ray_e.x() + stheta * ray_e.y(),
                -stheta * ray_e.x() + ctheta * ray_e.y(),
                ray_e.z()};
    };

    camera_model.fromCameraInfo(*camera_info_ptr);
    auto rows_start = rows - rows_to_keep;
    auto project_subpixel_to_ground = [rows_start,
                                       &cm = camera_model,
                                       camera_origin = transform_point(0, 0, 0),
                                       &transform_point](double x_cv, double y_cv) -> std::array<double, 3>
    {
        Eigen::Vector3d camera_ray{
            (x_cv - cm.cx() - cm.Tx()) / cm.fx(),
            ((rows_start + y_cv) - cm.cy() - cm.Ty()) / cm.fy(),
            1.0};
        // auto camera_ray = camera_model.projectPixelTo3dRay(cv::Point{x_cv, rows_start + y_cv});

        auto pt = transform_point(camera_ray[0], camera_ray[1], camera_ray[2]);

        // Project to ground
        const auto lambda = camera_origin[2] / (camera_origin[2] - pt[2]);
        auto p_g = (1 - lambda) * camera_origin + lambda * pt;

        // ROS_INFO("%f\t%f\t%f\t", p_g.x(), p_g.y(), p_g.z());
        return {p_g.x(), p_g.y(), p_g.z()};
    };

    for (auto x = 0; x < img.cols - 1; ++x)
    {
        for (auto y = 0; y < img.rows - 1; ++y)
        {

            if (img.at<uint8_t>(cv::Point{x, y}) != 255)
                continue;

            const auto [x_g, y_g, z_g] = project_subpixel_to_ground(x + 0.5, y + 0.5);
            // this is the world coordinates ROS_WARN("%f %f %f", x_g, y_g, z_g);
            //  Convert from base_footprint to image frame
            const auto [bi, bj] = worldToImgNoBounds(x_g, y_g);
            if (imgCheckBounds(birdseye, bi, bj))
                birdseye.at<uint8_t>(bi, bj) = 255;
        }
    }
    /*cv::rotate(birdseye, birdseye, ROTATE_180);
    const int rows_birdseye = birdseye.rows;
    const int cols_birdseye = birdseye.cols;
    const int rowstokeep = static_cast<int>(0.6 * rows_birdseye);
    cv::Mat top{birdseye, cv::Rect{0, rows_birdseye - rowstokeep, cols_birdseye, rowstokeep}};*/

    std_msgs::Header header;
    header.frame_id = "Top_view";
    header.stamp = ros::Time::now();
    sensor_msgs::ImagePtr msg_pub = cv_bridge::CvImage(header, "mono8", birdseye).toImageMsg();
    image_pub.publish(msg_pub);
    // ROS_ERROR("%lf", double((header.stamp - time).toSec()));
}
/**
void depth_callback(const sensor_msgs::ImageConstPtr &msg)
{
    frame_id_depth = msg->header.frame_id.c_str();
}*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "topview");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("lanes", 1, imagecallback);
    // image_transport::Subscriber depth_sub = it.subscribe("/zed2i/zed_node/depth/depth_registered", 1, depth_callback);
    image_pub = it.advertise("top_view", 1);
    tf2_ros::TransformListener tf_listener(tf_buffer);
    if (!tf_buffer.canTransform(ground_frame, frame_id_depth.c_str(), ros::Time(0), ros::Duration(10)))
    {
        ROS_FATAL("Could not get transform from %s to %s.", frame_id_depth.c_str(), ground_frame.c_str());
        // ros::shutdown();
    }
    ros::spin();
    return 0;
}
