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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/StdVector>
#include <boost/functional/hash.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <opencv4/opencv2/core.hpp>

#include <rclcpp/node.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <yabloc_common/gamma_converter.hpp>

namespace yabloc::lanelet2_cost_map {

using EigenPosition = Eigen::Vector2f; // position in world frame
using CvPixel       = cv::Point2i;     // pixel in image frame

using BoostPoint   = boost::geometry::model::d2::point_xy<double>;
using BoostBox     = boost::geometry::model::box<BoostPoint>;
using BoostPolygon = boost::geometry::model::polygon<BoostPoint>;

using Marker      = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using Pose        = geometry_msgs::msg::Pose;

// TODO: define following structures/aliases:
// - Pose (maybe from Eigen)
// - BoostLineSegment and BoostLineSegments
// - LabelPoint, FrozenPolygon, and FrozenPolygons


inline geometry_msgs::msg::Point to_ros(const EigenPosition& position) {
  geometry_msgs::msg::Point point;
  point.x = position.x();
  point.y = position.y();
  return point;
}

inline BoostPoint to_boost(const EigenPosition& position) {
  return {position.x(), position.y()};
}

inline EigenPosition to_eigen(const BoostPoint& point) {
  return {point.x(), point.y()};
}

// TODO: use center as origin instead of bottom-left to have cost map of 360 deg.
// Area class represents a unit discretized area of the cost map.
struct CostArea {
  CostArea() {}

  // Constructor with real scale bottom-left position in world frame.
  CostArea(const EigenPosition& position) {
    if (kMetricSize < 0) {
      std::cerr << "CostArea::kMetricSize is not initialized.\n";
      throw std::runtime_error("invalid CostArea::kMetricSize");
    }

    x = static_cast<int>(std::floor(position.x() / kMetricSize));
    y = static_cast<int>(std::floor(position.y() / kMetricSize));
  }

  friend bool operator==(const CostArea& one, const CostArea& other) {
    return one.x == other.x && one.y == other.y;
  }

  friend bool operator!=(const CostArea& one, const CostArea& other) {
    return !(one == other);
  }

  // Hash function for CostArea.
  std::size_t operator()(const CostArea& index) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, index.x);
    boost::hash_combine(seed, index.y);
    return seed;
  }

  CvPixel to_pixel(const EigenPosition& position, const int image_size) const {
    const EigenPosition relative = position - bottom_left();
    return {
      static_cast<int>(relative.x() / kMetricSize * image_size),
      static_cast<int>(relative.y() / kMetricSize * image_size)
    };
  }

  // Return the real scale of the corner positions of the area in world frame.
  EigenPosition bottom_left() const {
    return {x * kMetricSize, y * kMetricSize};
  };

  EigenPosition top_right() const {
    return bottom_left() + EigenPosition(kMetricSize, kMetricSize);
  };

  EigenPosition top_left() const {
    return bottom_left() + EigenPosition(0, kMetricSize);
  };

  EigenPosition bottom_right() const {
    return bottom_left() + EigenPosition(kMetricSize, 0);
  };

  int x, y;                 // bottom-left pixel
  static float kMetricSize; // unit size of the area in meters
};

// CostMapValue represents a pixel value of the cost map.
struct CostMapValue {
  CostMapValue(const float occupancy, const int direction, const bool frozen)
    : occupancy(occupancy), direction(direction), frozen(frozen)
  {}

  float occupancy;   // [0, 1] Cost value representing current pixel's occupancy
                     // from cloud, where 0 is free and 1 is occupied.
  int   direction;   // [0, 180] Cost value representing current pixel's
                     // angle of direction from cloud.
  bool  frozen   ;   // Boolean value representing whether the pixel is frozen
                     // or not. It lies in polygons whose label is bounding box.
};

// TODO: use meters_per_pixel instead of image_size.
struct Parameters {
  float                max_range = 40.0f; // Maximum range of the cost map [m].
  int                 image_size = 800  ; // Size of the cost map image [pixel].
  std::size_t        max_nb_maps = 10   ; // Maximum number of maps to be stored.
  float                    gamma = 4.0f ; // Gamma value for gamma correction.
  float min_height_diff_to_reset = 2.0f ; // Minimum height difference to reset maps [m].
  float max_height_diff_to_build = 4.0f ; // Maximum height difference to build maps [m].
};

class HierarchicalCostMap {
public:
  HierarchicalCostMap(const Parameters& parameters = Parameters());

  // Set height of the cost map and reset all maps.
  void set_height(const float height);

  // Set the frozen polygons, who are sent as a point cloud from lanelet2 map,
  // where the points of a polygon have the same label.
  // TODO: refactor data structure. The current one is supposed to a list of
  // label points, where label is used for same-polygon points.
  void set_frozen_polygons(const pcl::PointCloud<pcl::PointXYZL>& cloud);

  // Set the cloud of line segments to be used for the cost map.
  // TODO: refactor data structure. The current one is supposed to a list of
  // normal points, where point is start and normal is end.
  void set_road_markings(const pcl::PointCloud<pcl::PointNormal>& road_markings);

  // Get pixel value at specified real scale position in world frame.
  CostMapValue at(const EigenPosition& position);

  MarkerArray show_map_range() const;

  // TODO: refactor this function.
  cv::Mat get_map_image(const Pose& pose);

  void erase_obsolete();

private:
  // Build the cost map for the given area.
  void build_map(const CostArea& area);

  // Compute the cost occupancy map from the given occupancy map, where:
  // - If occupied, then the occupancy is 0.
  // - If free, then the occupancy is 100.
  cv::Mat compute_occupancy_cost(const cv::Mat& occupancy) const;

  // Compute the cost direction map from the given direction map and occupancy
  // map, by propagating the occupancy and direction information from the
  // top-left and bottom-right corners.
  cv::Mat compute_direction_cost(
    const cv::Mat& direction,
    const cv::Mat& occupancy
  );

  // Compute the frozen map from the given area.
  cv::Mat compute_frozen(const CostArea& area) const;

private:
  Parameters params_;

  // A list of all generated areas.
  std::list<CostArea> generated_areas_;
  // A list of all accessed areas.
  std::unordered_map<CostArea, bool, CostArea> accessed_areas_;
  // A list of all current maps.
  std::unordered_map<CostArea, cv::Mat, CostArea> maps_;

  std::optional<float> height_ = std::nullopt;
  std::vector<BoostPolygon> frozen_polygons_;
  std::optional<pcl::PointCloud<pcl::PointNormal>> cloud_ = std::nullopt;

  common::GammaConverter gamma_converter_;
};

} // namespace yabloc::lanelet2_cost_map
