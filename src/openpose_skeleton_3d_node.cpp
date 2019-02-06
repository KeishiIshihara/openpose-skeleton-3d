//Auther: Keishi Ishihara
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/MarkerArray.h>
//#include <aisl_openpose_ros/COCO_ARR.h>
#include <coco_keypoints/COCO_ARR.h>
#include <openpose_skeleton_3d/COCO3d_ARR.h>

template<typename COCO, bool Const>
struct COCOKeypointType {
  using keypoint = const decltype(COCO::Neck);
};

template<typename COCO>
struct COCOKeypointType<COCO, false> {
  using keypoint = decltype(COCO::Neck);
};

template<typename COCO>
struct COCOProxy {
private:
  using keypoint = typename COCOKeypointType<COCO, std::is_const<COCO>::value>::keypoint;

public:
  COCOProxy(COCO& coco)
    : coco(coco),
      keypoints({
                &coco.Nose,
                &coco.Neck,
                &coco.RShoulder,
                &coco.RElbow,
                &coco.RWrist,
                &coco.LShoulder,
                &coco.LElbow,
                &coco.LWrist,
                &coco.RHip,
                &coco.RKnee,
                &coco.RAnkle,
                &coco.LHip,
                &coco.LKnee,
                &coco.LAnkle,
                &coco.REye,
                &coco.LEye,
                &coco.REar,
                &coco.LEar,
                &coco.Waist})
  {}

  size_t size() const {
    return keypoints.size();
  }

  keypoint& operator [] (int n) {
    return *keypoints[n];
  }

  const keypoint& operator [] (int n) const {
    return *keypoints[n];
  }

  typename std::array<keypoint*, 19>::iterator begin() {
    return keypoints.begin();
  }

  typename std::array<keypoint*, 19>::iterator end() {
    return keypoints.end();
  }

  typename std::array<keypoint*, 19>::const_iterator begin() const {
    return keypoints.begin();
  }

  typename std::array<keypoint*, 19>::const_iterator end() const {
    return keypoints.end();
  }


private:
  COCO& coco;
  std::array<keypoint*, 19> keypoints;
};

class OpenPoseSkeleton3dNode {
public:
  OpenPoseSkeleton3dNode()
    : nh(),
      private_nh("~"),
      //cocos_sub(nh, "/openpose/detections/pose", 10),
      cocos_sub(nh, "/coco_keypoints/pose", 10),
      image_sub(nh, "/kinect2/hd/image_depth_rect", 45),
      camera_info_sub(nh, "/kinect2/hd/camera_info", 45),
      sync(SyncPolicy(45), cocos_sub, image_sub, camera_info_sub),
      cocos_3d_pub(private_nh.advertise<openpose_skeleton_3d::COCO3d_ARR>("pose3d", 30)),
      markers_pub(private_nh.advertise<visualization_msgs::MarkerArray>("markers", 30))
  {
    sync.registerCallback(boost::bind(&OpenPoseSkeleton3dNode::callback, this, _1, _2, _3));
  }

private:
  void callback(const coco_keypoints::COCO_ARRConstPtr& cocos_msg, const sensor_msgs::ImageConstPtr& depth_image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    const Eigen::Matrix3d camera_matrix = Eigen::Map<const Eigen::Matrix3d>(camera_info_msg->K.data()).transpose();
    const Eigen::Matrix3d inv_camera_matrix = camera_matrix.inverse();

    auto depth_image = cv_bridge::toCvCopy(depth_image_msg);
    //cv::imshow("depth",depth_image->image);
    //cv::waitKey(50);

    openpose_skeleton_3d::COCO3d_ARRPtr cocos_3d_msg(new openpose_skeleton_3d::COCO3d_ARR());
    cocos_3d_msg->header = cocos_msg->header;
    cocos_3d_msg->num_person = cocos_msg->num_person;
    cocos_3d_msg->data.resize(cocos_msg->data.size());

    for(int i=0; i<cocos_msg->data.size(); i++) {
      COCOProxy<const coco_keypoints::COCO> coco_proxy(cocos_msg->data[i]);
      COCOProxy<openpose_skeleton_3d::COCO3d> coco3d_proxy(cocos_3d_msg->data[i]);
      std::cout << "------------------------" << std::endl;
      std::cout << cocos_msg->data[i].Neck.x << std::endl;

      for(int j=0; j<coco_proxy.size(); j++) {
        double depth = median(depth_image->image, cv::Point(coco_proxy[j].x, coco_proxy[j].y), 5) / 1000.0;
	std::cout << depth << std::endl;

        Eigen::Vector3d uv1(coco_proxy[j].x, coco_proxy[j].y, 1.0);
        Eigen::Vector3d xyz = depth * inv_camera_matrix * uv1;

        coco3d_proxy[j].x = xyz.x();
	std::cout << xyz.x() << std::endl;
        coco3d_proxy[j].y = xyz.y();
        coco3d_proxy[j].z = depth;
        coco3d_proxy[j].confidence = coco_proxy[j].confidence;
	// std::cout << "------------------------" << std::endl;
	std::cout << coco3d_proxy[j].confidence << std::endl;
	std::cout << coco3d_proxy[j].x << std::endl;
	std::cout << coco3d_proxy[j].y << std::endl;
	std::cout << coco3d_proxy[j].z << std::endl;
      }
    }

    cocos_3d_pub.publish(cocos_3d_msg);

    if(markers_pub.getNumSubscribers()) {
      markers_pub.publish(create_markers(cocos_3d_msg));
    }
  }

  int median(const cv::Mat& depth, const cv::Point& pt, int extent) const {
    cv::Point extents(extent, extent);
    cv::Rect image_region(cv::Point(0, 0), depth.size());
    cv::Rect region = cv::Rect(pt - extents, pt + extents) & image_region;
    std::cout << region << std::endl;

    if(region.width <= 0 || region.height <= 0) {
      return 0;
    }

    cv::Mat roi(depth, region);

    std::vector<int> values;
    values.reserve(extent * extent);
    for(auto itr = roi.begin<uint16_t>(); itr != roi.end<uint16_t>(); itr++) {
      if(*itr > 50) {
        values.push_back(*itr);
      }
    }

    if(values.empty()) {
      return 0;
    }

    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
  }

  visualization_msgs::MarkerArrayPtr create_markers(const openpose_skeleton_3d::COCO3d_ARRConstPtr& cocos_3d_msg) const {
    visualization_msgs::MarkerArrayPtr markers_msg(new visualization_msgs::MarkerArray());
    markers_msg->markers.resize(cocos_3d_msg->data.size());

    for(int i=0; i<cocos_3d_msg->data.size(); i++) {
      visualization_msgs::Marker& marker = markers_msg->markers[i];
      marker.header = cocos_3d_msg->header;
      marker.ns = (boost::format("person%d") % i).str();
      marker.id = i;

      marker.type = visualization_msgs::Marker::POINTS;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;

      marker.lifetime = ros::Duration();

      COCOProxy<const openpose_skeleton_3d::COCO3d> coco_3d(cocos_3d_msg->data[i]);

      marker.points.resize(coco_3d.size());
      marker.colors.resize(coco_3d.size());
      for(int j=0; j<coco_3d.size(); j++) {
        marker.points[j].x = coco_3d[j].x;
        marker.points[j].y = coco_3d[j].y;
        marker.points[j].z = coco_3d[j].z;

        marker.colors[j].r = 1.0;
        marker.colors[j].g = 0.0;
        marker.colors[j].b = 0.0;
        marker.colors[j].a = 1.0;
      }
    }

    return markers_msg;
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<coco_keypoints::COCO_ARR, sensor_msgs::Image, sensor_msgs::CameraInfo>;

  message_filters::Subscriber<coco_keypoints::COCO_ARR> cocos_sub;
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub;
  message_filters::Synchronizer<SyncPolicy> sync;

  ros::Publisher cocos_3d_pub;
  ros::Publisher markers_pub;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "openpose_skeleton_3d_node");
  std::unique_ptr<OpenPoseSkeleton3dNode> node(new OpenPoseSkeleton3dNode());
  ros::spin();
  return 0;
}
