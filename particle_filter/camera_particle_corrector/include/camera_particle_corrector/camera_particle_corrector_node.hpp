// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ll2_cost_map/hierarchical_cost_map.hpp>
#include <modularized_particle_filter/correction/abst_corrector.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv4/opencv2/core.hpp>

namespace yabloc::modularized_particle_filter {

cv::Point2f cv2pt(const Eigen::Vector3f v);

class CameraParticleCorrector
  : public modularized_particle_filter::AbstCorrector {
public:
  using LineSegment  = pcl::PointXYZLNormal;
  using LineSegments = pcl::PointCloud<LineSegment>;
  using PointCloud2  = sensor_msgs::msg::PointCloud2;
  using PoseStamped  = geometry_msgs::msg::PoseStamped;
  using Image        = sensor_msgs::msg::Image;
  using MarkerArray  = visualization_msgs::msg::MarkerArray;
  using Pose         = geometry_msgs::msg::Pose;
  using Bool         = std_msgs::msg::Bool;
  using String       = std_msgs::msg::String;
  using SetBool      = std_srvs::srv::SetBool;
  CameraParticleCorrector();

private:
  const float min_prob_;
  const float far_weight_gain_;
  HierarchicalCostMap cost_map_;

  rclcpp::Subscription<PointCloud2>::SharedPtr bounding_box_sub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr line_segments_cloud_sub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr ll2_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Service<SetBool>::SharedPtr switch_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Publisher<Image>::SharedPtr image_pub_;
  rclcpp::Publisher<Image>::SharedPtr map_image_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr scored_cloud_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr scored_posteriori_cloud_pub_;
  rclcpp::Publisher<String>::SharedPtr string_pub_;

  Eigen::Vector3f last_mean_position_;
  std::optional<PoseStamped> latest_pose_stamped_{std::nullopt};
  std::function<float(float)> score_converter_;

  bool enable_switch_{true};

  void on_line_segments   (const PointCloud2& msg);
  void on_ll2_road_marking(const PointCloud2& msg);
  void on_ll2_bounding_box(const PointCloud2& msg);
  void on_pose            (const PoseStamped& msg);
  void on_switch(
    SetBool::Request::ConstSharedPtr request,
    SetBool::Response::SharedPtr     response
  );
  void on_timer();

  std::pair<LineSegments, LineSegments> split_line_segments(
    const PointCloud2& msg
  );

  float compute_logit(
    const LineSegments& line_segments_cloud,
    const Eigen::Vector3f& self_position
  );

  pcl::PointCloud<pcl::PointXYZI> evaluate_cloud(
    const LineSegments& line_segments_cloud,
    const Eigen::Vector3f& self_position
  );

  std::pair<LineSegments, LineSegments> filt(const LineSegments& lines);
};
} // namespace yabloc::modularized_particle_filter
