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

#include "camera_particle_corrector/camera_particle_corrector_node.hpp"
#include "camera_particle_corrector/logit_probit.hpp"
#include "camera_particle_corrector/score.hpp"

#include <yabloc_common/color.hpp>
#include <yabloc_common/pose_conversions.hpp>
#include <yabloc_common/pub_sub.hpp>
#include <yabloc_common/timer.hpp>
#include <yabloc_common/transform_line_segments.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <opencv4/opencv2/imgproc.hpp>

namespace yabloc::modularized_particle_filter {

cv::Point2f cv2pt(const Eigen::Vector3f v) {
  const float kMetersPerPixel = 0.05;
  const float kImageRadius    = 400;
  return {
    -v.y() / kMetersPerPixel + kImageRadius,
    -v.x() / kMetersPerPixel + 2 * kImageRadius
  };
}

CameraParticleCorrector::CameraParticleCorrector()
  : AbstCorrector("camera_particle_corrector")
  , min_prob_(declare_parameter<float>("min_prob", 0.01))
  , far_weight_gain_(declare_parameter<float>("far_weight_gain", 0.001))
  , cost_map_(this)
{
  using std::placeholders::_1;
  using std::placeholders::_2;

  // Declare parameters.
  enable_switch_ = declare_parameter<bool>("enabled_at_first", true);

  // Declare publishers.
  image_pub_        = create_publisher<Image>      ("match_image"   , 10);
  map_image_pub_    = create_publisher<Image>      ("cost_map_image", 10);
  marker_pub_       = create_publisher<MarkerArray>("cost_map_range", 10);
  string_pub_       = create_publisher<String>     ("state_string"  , 10);
  scored_cloud_pub_ = create_publisher<PointCloud2>("scored_cloud"  , 10);
  scored_posteriori_cloud_pub_
    = create_publisher<PointCloud2>("scored_post_cloud", 10);

  // Declare subscribers.
  line_segments_cloud_sub_ = create_subscription<PointCloud2>(
    "line_segments_cloud",                                          // Topic
    10,                                                             // QoS
    std::bind(&CameraParticleCorrector::on_line_segments, this, _1) // Callback
  );
  ll2_sub_ = create_subscription<PointCloud2>(
    "ll2_road_marking",
    10,
    std::bind(&CameraParticleCorrector::on_ll2_road_marking, this, _1)
  );
  bounding_box_sub_ = create_subscription<PointCloud2>(
    "ll2_bounding_box",
    10,
    std::bind(&CameraParticleCorrector::on_ll2_bounding_box, this, _1)
  );
  pose_sub_ = create_subscription<PoseStamped>(
    "pose",
    10,
    std::bind(&CameraParticleCorrector::on_pose, this, _1)
  );

  // Declare services.
  switch_srv_ = create_service<SetBool>(
    "switch_srv",                                                // Service
    std::bind(&CameraParticleCorrector::on_switch, this, _1, _2) // Callback
  );

  // Declare timers.
  timer_ = rclcpp::create_timer(
    this,                                               // Node
    get_clock(),                                        // Clock
    rclcpp::Rate(1).period(),                           // Period
    std::bind(&CameraParticleCorrector::on_timer, this) // Callback
  );
}

void CameraParticleCorrector::on_line_segments(const PointCloud2& msg) {
  common::Timer timer;

  // Get synchronized particles. If not available, return.
  const rclcpp::Time stamp = msg.header.stamp;
  auto sync_particles      = get_synchronized_particle_array(stamp);
  if (!sync_particles.has_value())  {
    return;
  }

  // Check timestamp gap between image and particles. If it is too large, warn.
  const rclcpp::Duration dt = (stamp - sync_particles->header.stamp);
  if (std::abs(dt.seconds()) > 0.1) {
    RCLCPP_WARN_STREAM(
      get_logger(),
      "Timestamp (s) gap between image and particles is LARGE: " << dt.seconds()
    );
  }

  // Check if the travel distance is enough long to publish weighted particles.
  bool publish_weighted_particles = true;
  const Pose mean_pose            = compute_mean_pose(*sync_particles);
  {
    // Compute travel distance.
    Eigen::Vector3f mean_position
      = common::pose_to_affine(mean_pose).translation();

    // If the travel distance is too short, warn and skip particle weighting.
    constexpr float kMinTravelDistance = 1.0f;
    if ((mean_position - last_mean_position_).squaredNorm() > kMinTravelDistance) {
      last_mean_position_ = mean_position; // Update last mean position.
    } else {
      using namespace std::literals::chrono_literals;
      publish_weighted_particles = false;
      RCLCPP_WARN_STREAM_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Skip particle weighting due to almost the same position"
      );
    }
  }

  auto [reliable_line_segments, iffy_line_segments] = split_line_segments(msg);
  cost_map_.set_height(mean_pose.position.z);

  // Publish weighted particles if the travel distance is enough long.
  auto weighted_particles = sync_particles.value();
  if (publish_weighted_particles) {
    for (auto& particle : weighted_particles.particles) {
      // Convert particle pose to SE3.
      Sophus::SE3f transform = common::pose_to_se3(particle.pose);
      // Transform line segments back to the world frame.
      LineSegments transformed_ones;
      transformed_ones += common::transform_line_segments(
        reliable_line_segments,
        transform
      );
      transformed_ones += common::transform_line_segments(
        iffy_line_segments,
        transform
      );
      // Compute logit and update particle weight.
      const float logit = compute_logit(transformed_ones, transform.translation());
      particle.weight   = logit_to_probit(logit, 0.01f);
    }

    if (enable_switch_) {
      set_weighted_particle_array(weighted_particles);
    }
  }

  cost_map_.erase_obsolete(); // NOTE:
  marker_pub_->publish(cost_map_.show_map_range());

  // DEBUG: just visualization
  {
    Pose meaned_pose       = compute_mean_pose(weighted_particles);
    Sophus::SE3f transform = common::pose_to_se3(meaned_pose);

    pcl::PointCloud<pcl::PointXYZI> cloud = evaluate_cloud(
      common::transform_line_segments(reliable_line_segments, transform),
      transform.translation()
    );
    pcl::PointCloud<pcl::PointXYZI> iffy_cloud = evaluate_cloud(
      common::transform_line_segments(iffy_line_segments, transform),
      transform.translation()
    );

    pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud2;

    float max_score = 0;
    for (const auto p : cloud) {
      max_score = std::max(max_score, std::abs(p.intensity));
    }
    for (const auto p : cloud) {
      pcl::PointXYZRGB rgb;
      rgb.getVector3fMap() = p.getVector3fMap();
      rgb.rgba = common::color_scale::blue_red(p.intensity / max_score);
      rgb_cloud.push_back(rgb);
    }
    for (const auto p : iffy_cloud) {
      pcl::PointXYZRGB rgb;
      rgb.getVector3fMap() = p.getVector3fMap();
      rgb.rgba = common::color_scale::blue_red(p.intensity / max_score);
      rgb_cloud2.push_back(rgb);
    }

    common::publish_cloud(
      *scored_cloud_pub_, rgb_cloud, msg.header.stamp
    );
    common::publish_cloud(
      *scored_posteriori_cloud_pub_, rgb_cloud2, msg.header.stamp
    );
  }

  if (timer.milli_seconds() > 80) {
    RCLCPP_WARN_STREAM(get_logger(), "on_line_segments: " << timer);
  } else {
    RCLCPP_INFO_STREAM(get_logger(), "on_line_segments: " << timer);
  }

  // Publish status as string.
  {
    String status;
    std::stringstream ss;
    ss << "-- Camera particle corrector --" << std::endl;
    ss << (enable_switch_ ? "ENABLED" : "disabled") << std::endl;
    ss << "time: " << timer << std::endl;
    status.data = ss.str();
    string_pub_->publish(status);
  }
}

void CameraParticleCorrector::on_ll2_road_marking(const PointCloud2& ll2_msg) {
  pcl::PointCloud<pcl::PointNormal> ll2_cloud;
  pcl::fromROSMsg(ll2_msg, ll2_cloud);
  cost_map_.set_cloud(ll2_cloud);
  RCLCPP_INFO_STREAM(get_logger(), "Set LL2 cloud into hierarchical cost map");
}

void CameraParticleCorrector::on_ll2_bounding_box(const PointCloud2& msg) {
  // NOTE: Under construction
  pcl::PointCloud<pcl::PointXYZL> ll2_bounding_box;
  pcl::fromROSMsg(msg, ll2_bounding_box);
  cost_map_.set_bounding_box(ll2_bounding_box);
  RCLCPP_INFO_STREAM(get_logger(), "Set bounding box into hierarchical cost map");
}

void CameraParticleCorrector::on_pose(const PoseStamped& msg) {
  latest_pose_stamped_ = msg;
}

void CameraParticleCorrector::on_switch(
  SetBool::Request::ConstSharedPtr request,
  SetBool::Response::SharedPtr response
) {
  enable_switch_    = request->data;
  response->success = true;
  if (enable_switch_) {
    RCLCPP_INFO_STREAM(get_logger(), "camera_corrector is enabled");
  } else {
    RCLCPP_INFO_STREAM(get_logger(), "camera_corrector is disabled");
  }
}

void CameraParticleCorrector::on_timer() {
  if (latest_pose_stamped_.has_value()) {
    // Publish cost map image at the latest pose for debugging.
    common::publish_image(
      *map_image_pub_,
      cost_map_.get_map_image(latest_pose_stamped_->pose),
      latest_pose_stamped_->header.stamp
    );
  }
}

std::pair<
  CameraParticleCorrector::LineSegments,
  CameraParticleCorrector::LineSegments
> CameraParticleCorrector::split_line_segments(const PointCloud2& msg) {
  LineSegments all_line_segments;
  pcl::fromROSMsg(msg, all_line_segments);
  LineSegments reliable_ones, iffy_ones;
  {
    for (const auto& ls : all_line_segments) {
      if (ls.label == 0) {
        iffy_ones.push_back(ls);
      } else {
        reliable_ones.push_back(ls);
      }
    }
  }

  auto [good_ones, bad_ones] = filt(iffy_ones);
  {
    cv::Mat debug_image = cv::Mat::zeros(800, 800, CV_8UC3);
    auto draw = [&debug_image](LineSegments& line_segments, cv::Scalar color) -> void {
      for (const auto& ls : line_segments) {
        const Eigen::Vector3f start = ls.getVector3fMap();
        const Eigen::Vector3f end   = ls.getNormalVector3fMap();
        cv::line(debug_image, cv2pt(start), cv2pt(end), color, 2);
      }
    };

    draw(reliable_ones, cv::Scalar(  0,   0, 255));
    draw(good_ones    , cv::Scalar(  0, 255,   0));
    draw(bad_ones     , cv::Scalar(100, 100, 100));
    common::publish_image(*image_pub_, debug_image, msg.header.stamp);
  }

  return {reliable_ones, good_ones};
}

float CameraParticleCorrector::compute_logit(
  const LineSegments& line_segments,
  const Eigen::Vector3f& position
) {
  float logit = 0;
  for (const LineSegment& ls : line_segments) {
    const auto& start = ls.getVector3fMap();
    const auto& end   = ls.getNormalVector3fMap();

    const Eigen::Vector3f direction = (end - start).normalized();
    const float           length    = (end - start).norm();

    // Loop through the line segment and sum up the logit.
    constexpr float step = 0.1f;
    for (float distance = 0; distance < length; distance += step) {
      Eigen::Vector3f curr = start + direction * distance;

      // NOTE: Close points are prioritized
      const float squared_norm = (curr - position).topRows(2).squaredNorm();
      const float gain = std::exp(-far_weight_gain_ * squared_norm); // 0 < gain < 1

      // Get cost map pixel value at the current point.
      const CostMapValue pixel = cost_map_.at(curr.topRows(2));

      // logit does not change if target pixel is unmapped.
      if (pixel.unmapped) {
        continue;
      }

      // Compute the closeness between the direction of the line segment and the
      // pixel direction.
      const float closeness = score_closeness_by_dot_product(
        direction.topRows(2).normalized(),
        pixel.angle
      );

      // Update logit based on the label.
      const float increment = gain * (closeness * pixel.intensity - 0.5f);
      if (ls.label == 0) {
        // posteriori.
        logit += 0.2f * increment;
      } else {
        // a priori.
        logit += increment;
      }
    }
  }
  return logit;
}

pcl::PointCloud<pcl::PointXYZI> CameraParticleCorrector::evaluate_cloud(
  const LineSegments& line_segments,
  const Eigen::Vector3f& position
) {
  pcl::PointCloud<pcl::PointXYZI> cloud;
  for (const LineSegment& ls : line_segments) {
    const auto start = ls.getVector3fMap();
    const auto end   = ls.getNormalVector3fMap();

    const Eigen::Vector3f direction = (end - start).normalized();
    const float           length    = (end - start).norm();

    constexpr float step = 0.1f;
    for (float distance = 0; distance < length; distance += step) {
      Eigen::Vector3f curr = start + direction * distance;

      // NOTE: Close points are prioritized
      float squared_norm = (curr - position).topRows(2).squaredNorm();
      float gain         = std::exp(-far_weight_gain_ * squared_norm);

      // Get cost map pixel value at the current point.
      CostMapValue pixel = cost_map_.at(curr.topRows(2));

      // Compute logit based on the label.
      float logit = 0;
      if (!pixel.unmapped) {
        // Compute the closeness between the direction of the line segment and
        // the pixel direction.
        const float closeness = score_closeness_by_dot_product(
          direction.topRows(2).normalized(),
          pixel.angle
        );
        // Compute logit.
        logit = gain * (closeness * pixel.intensity - 0.5f);
      }

      // Update cloud.
      pcl::PointXYZI xyzi(logit_to_probit(logit, 10.f));
      xyzi.getVector3fMap() = curr;
      cloud.push_back(xyzi);
    }
  }
  return cloud;
}

std::pair<
  CameraParticleCorrector::LineSegments,
  CameraParticleCorrector::LineSegments
> CameraParticleCorrector::filt(const LineSegments& iffy_line_segments) {
  LineSegments good_ones, bad_ones;
  if (!latest_pose_stamped_.has_value()) {
    throw std::runtime_error("latest_pose_ is nullopt");
  }

  const auto transform = common::pose_to_se3(latest_pose_stamped_.value().pose);

  for (const auto& ls : iffy_line_segments) {
    const Eigen::Vector3f start     = ls.getVector3fMap();
    const Eigen::Vector3f end       = ls.getNormalVector3fMap();
    const float           length    = (start - end).norm();
    const Eigen::Vector3f direction = (start - end).normalized();

    float score          = 0;
    std::size_t count    = 0;
    constexpr float step = 0.1f;
    for (float distance = 0; distance < length; distance += step) {
      const Eigen::Vector3f curr = transform * (end + direction * distance);
      const Eigen::Vector3f transformed_direction = transform.so3() * direction;

      const CostMapValue pixel = cost_map_.at(curr.topRows(2));
      const float closeness = score_closeness_by_angle_diff(
        transformed_direction.topRows(2),
        pixel.angle
      );
      score += (closeness * pixel.intensity);

      count++;
    }

    score /= static_cast<float>(count);
    if (score > 0.5f) {
      good_ones.push_back(ls);
    } else {
      bad_ones.push_back(ls);
    }
  }

  return {good_ones, bad_ones};
}

} // namespace yabloc::modularized_particle_filter
