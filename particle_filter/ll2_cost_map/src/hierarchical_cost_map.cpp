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

#include <boost/geometry/algorithms/disjoint.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <yabloc_common/color.hpp>

#include "ll2_cost_map/hierarchical_cost_map.hpp"

namespace yabloc::lanelet2_cost_map {

float CostArea::kMetricSize = -1.f;

HierarchicalCostMap::HierarchicalCostMap(const Parameters& parameters)
  : params_(parameters) {
  CostArea::kMetricSize = params_.max_range;
  gamma_converter_.reset(params_.gamma);
}

void HierarchicalCostMap::set_height(const float height) {
  // Only reset the map when the height changes significantly.
  if (height_) {
    if (std::abs(*height_ - height) > params_.min_height_diff_to_reset) {
      generated_areas_.clear();
      maps_.clear();
      accessed_areas_.clear();
    }
  }

  height_ = height;
}

void HierarchicalCostMap::set_frozen_polygons(
  const pcl::PointCloud<pcl::PointXYZL>& cloud
) {
  // Skip if the cloud is empty.
  if (cloud.empty()) {
    return;
  }

  // TODO: Whether to clear the existing bounding boxes?
  // TODO: find the max label from the cloud to resize the polygons_.

  // Convert a cloud of vertices to a list of polygons, where each polygon has
  // vertices with the same label.
  BoostPolygon polygon;
  std::optional<uint32_t> last_label = std::nullopt;
  for (const auto& point : cloud) {
    if (last_label) {
      // If the label changes, then the current polygon is finished.
      if ((*last_label) != point.label) {
        frozen_polygons_.push_back(polygon);
        polygon.outer().clear();
      }
    }
    // Add the point to the current polygon and update the last label.
    polygon.outer().push_back(BoostPoint(point.x, point.y));
    last_label = point.label;
  }

  frozen_polygons_.push_back(polygon);
}

void HierarchicalCostMap::set_road_markings(
  const pcl::PointCloud<pcl::PointNormal>& cloud
) {
  cloud_ = cloud;
}

CostMapValue HierarchicalCostMap::at(const EigenPosition& position) {
  // Return a default value if the cloud is not set.
  if (!cloud_) {
    return CostMapValue{0.5f, 0, true};
  }

  // Build the map if the area is not generated yet and mark it as accessed.
  CostArea area(position);
  if (maps_.count(area) == 0) {
    build_map(area);
  }
  accessed_areas_[area] = true;

  // Get the pixel value from the generated map.
  const CvPixel     pixel = area.to_pixel(position, params_.image_size);
  const cv::Mat&    map   = maps_.at(area)                             ;
  const cv::Vec3b   b3    = map.at<cv::Vec3b>(pixel)                   ;

  return {b3[0] / 255.f, b3[1], b3[2] == 1};
}

MarkerArray HierarchicalCostMap::show_map_range() const {
  MarkerArray markers;

  int id = 0;
  for (const CostArea& area : generated_areas_) {
    Marker marker;
    marker.header.frame_id = "map";
    marker.id              = id++;
    marker.type            = Marker::LINE_STRIP;
    marker.color           = common::Color(0.0f, 0.0f, 1.0f, 1.0f);
    marker.scale.x         = 0.1;

    const EigenPosition& bottom_left  = area.bottom_left ();
    const EigenPosition& bottom_right = area.bottom_right();
    const EigenPosition& top_right    = area.top_right   ();
    const EigenPosition& top_left     = area.top_left    ();
    marker.points.push_back(to_ros(bottom_left ));
    marker.points.push_back(to_ros(bottom_right));
    marker.points.push_back(to_ros(top_right   ));
    marker.points.push_back(to_ros(top_left    ));
    marker.points.push_back(to_ros(bottom_left ));

    markers.markers.push_back(marker);
  }

  return markers;
}

cv::Mat HierarchicalCostMap::get_map_image(const Pose& pose) {
  EigenPosition center;
  center << pose.position.x, pose.position.y;

  float w = pose.orientation.w;
  float z = pose.orientation.z;
  Eigen::Matrix2f R
    = Eigen::Rotation2Df(2.f * std::atan2(z, w) - M_PI_2).toRotationMatrix();

  auto toVector2f = [this, center, R](float h, float w) -> EigenPosition {
    EigenPosition offset;
    offset.x() =  (w / params_.image_size - 0.5f) * params_.max_range * 1.5f;
    offset.y() = -(h / params_.image_size - 0.5f) * params_.max_range * 1.5f;
    return center + R * offset;
  };

  cv::Mat image = cv::Mat::zeros(cv::Size(params_.image_size, params_.image_size), CV_8UC3);
  for (int w = 0; w < params_.image_size; w++) {
    for (int h = 0; h < params_.image_size; h++) {
      CostMapValue v3 = at(toVector2f(h, w));
      // TODO: swap direction and occupancy.
      if (v3.frozen) {
        image.at<cv::Vec3b>(h, w)
          = cv::Vec3b(v3.direction, 255 * v3.occupancy, 50);
      } else {
        image.at<cv::Vec3b>(h, w)
          = cv::Vec3b(v3.direction, 255 * v3.occupancy, 255 * v3.occupancy);
      }
    }
  }

  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_HSV2BGR);
  return rgb_image;
}

void HierarchicalCostMap::erase_obsolete() {
  if (maps_.size() < params_.max_nb_maps) {
    return;
  }

  for (auto it = generated_areas_.begin(); it != generated_areas_.end();) {
    // Skip the map if it is accessed.
    if (accessed_areas_[*it]) {
      ++it;
      continue;
    }
    // Otherwise, erase the map from the list of generated ones and the
    // currently used ones.
    maps_.erase(*it);
    it = generated_areas_.erase(it);
  }

  // Clear the list of accessed maps.
  accessed_areas_.clear();
}

void HierarchicalCostMap::build_map(const CostArea& area) {
  // Skip building the map if the cloud is not set.
  if (!cloud_) {
    return;
  }

  const cv::Size kSize(params_.image_size, params_.image_size);

  // Initialize the occupancy and direction maps, where:
  // - the occupancy map is a binary map where 0 = free, 255 = occupied,
  // - the direction map is a grayscale map where pixel value is in [0, 180].
  cv::Mat occupancy = cv::Mat(kSize, CV_8UC1, cv::Scalar::all(255));
  cv::Mat direction = cv::Mat(kSize, CV_8UC1, cv::Scalar::all(  0));
  for (const auto pn : cloud_.value()) {
    // TODO: We can speed up by skipping too far line_segments.

    // Skip the point if it is outside the area.
    if (height_) {
      const bool is_outside
        = std::abs(pn.z        - *height_) > params_.max_height_diff_to_build
       || std::abs(pn.normal_z - *height_) > params_.max_height_diff_to_build;
      if (is_outside) {
        continue;
      }
    }

    // Convert the points from the world frame to the pixel frame.
    const auto from_pos = pn.getVector3fMap()      .topRows(2);
    const auto to_pos   = pn.getNormalVector3fMap().topRows(2);
    const auto from_px  = area.to_pixel(from_pos, params_.image_size);
    const auto to_px    = area.to_pixel(to_pos  , params_.image_size);

    // Calculate the angle of the line segment, where angle lies in [0, 180].
    float radian = std::atan2(from_px.y - to_px.y, from_px.x - to_px.x);
    if (radian < 0) {
      radian += M_PI;
    }
    const float degree = radian * 180.f / M_PI;

    // Draw the line segment on the occupancy and direction maps.
    cv::line(occupancy, from_px, to_px, cv::Scalar::all(     0), 1);
    cv::line(direction, from_px, to_px, cv::Scalar::all(degree), 1);
  }

  // Compute the cost map from different sources.
  const cv::Mat occupancy_cost = compute_occupancy_cost(occupancy);
  const cv::Mat direction_cost = compute_direction_cost(direction, occupancy);
  const cv::Mat frozen         = compute_frozen(area);

  cv::Mat cost_map;
  cv::merge(
    std::vector<cv::Mat>{occupancy_cost, direction_cost, frozen},
    cost_map
  );

  // Store the generated map.
  maps_[area] = cost_map;
  generated_areas_.push_back(area);

  // TODO: log here as successful to build map at the area of bottom_left...
}

// Compute the distance transform of the occupancy map, then threshold and
// invert it so that the pixel value is in [0, 100]. Finally, propagate it
// through a gamma correction to highlight the occupancy.
cv::Mat HierarchicalCostMap::compute_occupancy_cost(
  const cv::Mat& occupancy
) const {
  cv::Mat cost;

  cv::distanceTransform(occupancy, cost, cv::DIST_L2, 3);
  // TODO: add a parameter to control the distance thresholding (be careful that
  // it's in pixel).
  // TODO: scale the distance transform to [0, 100].
  cv::threshold(cost, cost, 100, 100, cv::THRESH_TRUNC);
  // Invert the cost map so that: [0, 100] -> [255, 0].
  cost.convertTo(cost, CV_8UC1, -2.55, 255);
  cost = gamma_converter_(cost);

  return cost;
}

cv::Mat HierarchicalCostMap::compute_direction_cost(
  const cv::Mat& direction,
  const cv::Mat& occupancy
) {
  // Error if the size of the cost map and the intensity map are different.
  if (direction.size() != occupancy.size()) {
    throw std::runtime_error(
      "The size of the cost map and the intensity map must be the same."
    );
  }

  constexpr int kInfinity = std::numeric_limits<int>::max();
  const     int rows      = direction.rows;
  const     int cols      = direction.cols;

  // Rescale the occupancy map from [0, 255] to [0, inf] to avoid overflow.
  // - If occupied, then the occupancy is 0.
  // - If free, then the occupancy is inf.
  std::vector<std::vector<int>> scaled_occupancy;
  scaled_occupancy.resize(rows);
  for (int r = 0; r < rows; r++) {
    // Initialize all occupancies to infinity.
    scaled_occupancy.at(r).resize(cols);
    std::fill(
      scaled_occupancy.at(r).begin(),
      scaled_occupancy.at(r).end(),
      kInfinity
    );

    // Set the occupancy to 0 if the cell is occupied.
    const uchar* ptr = occupancy.ptr<uchar>(r);
    for (int c = 0; c < cols; c++) {
      if (ptr[c] == 0) {
        scaled_occupancy.at(r).at(c) = 0;
      }
    }
  }

  // Initialize the cost map with the direction map.
  cv::Mat cost = direction.clone();

  // Forward pass.
  for (int r = 1; r < rows; r++) {
    for (int c = 1; c < cols; c++) {
      const uchar  top_dir  = cost.at<uchar>(r - 1, c    );
      const uchar  left_dir = cost.at<uchar>(r    , c - 1);
            uchar& curr_dir = cost.at<uchar>(r    , c    );

      const int    top_occ  = scaled_occupancy.at(r - 1).at(c    );
      const int    left_occ = scaled_occupancy.at(r    ).at(c - 1);
            int&   curr_occ = scaled_occupancy.at(r    ).at(c    );

      // Propagate the occupancy and the direction from the top or the left.
      if (top_occ < left_occ) {
        if (curr_occ < top_occ + 1) {
          continue;
        }
        curr_occ = top_occ + 1;
        curr_dir = top_dir;
      } else {
        if (curr_occ < left_occ + 1) {
          continue;
        }
        curr_occ = left_occ + 1;
        curr_dir = left_dir;
      }
    }
  }

  // Backward pass.
  for (int r = rows - 2; r >= 0; r--) {
    for (int c = cols - 2; c >= 0; c--) {
      const uchar  bottom_dir = cost.at<uchar>(r + 1, c    );
      const uchar  right_dir  = cost.at<uchar>(r    , c + 1);
            uchar& curr_dir   = cost.at<uchar>(r    , c    );

      const int    bottom_occ = scaled_occupancy.at(r + 1).at(c    );
      const int    right_occ  = scaled_occupancy.at(r    ).at(c + 1);
            int&   curr_occ   = scaled_occupancy.at(r    ).at(c    );

      // Propagate the occupancy and the direction from the bottom or the right.
      if (bottom_occ < right_occ) {
        if (curr_occ < bottom_occ + 1) {
          continue;
        }
        curr_occ = bottom_occ + 1;
        curr_dir = bottom_dir;
      } else {
        if (curr_occ < right_occ + 1) {
          continue;
        }
        curr_occ = right_occ + 1;
        curr_dir = right_dir;
      }
    }
  }

  return cost;
}

cv::Mat HierarchicalCostMap::compute_frozen(const CostArea& area) const {
  const cv::Size kSize(params_.image_size, params_.image_size);
  cv::Mat frozen = cv::Mat::zeros(kSize, CV_8UC1);

  // Skip if the bounding boxes are not set.
  if (frozen_polygons_.empty()) {
    return frozen;
  }

  // Define current area.
  BoostBox area_box;
  area_box.min_corner() = to_boost(area.bottom_left());
  area_box.max_corner() = to_boost(area.top_right  ());

  // Draw the contours of the bounding boxes on the frozen map, where they must
  // be joined with the area.
  std::vector<std::vector<CvPixel>> contours;
  for (const auto& polygon : frozen_polygons_) {
    // Skip the polygon if it is disjoint with the area.
    if (boost::geometry::disjoint(area_box, polygon)) {
      continue;
    }

    // Convert the polygon from the world frame to the pixel frame.
    std::vector<CvPixel> contour;
    contour.reserve(polygon.outer().size());
    for (const auto& position : polygon.outer()) {
      const auto pixel = area.to_pixel(to_eigen(position), params_.image_size);
      contour.push_back(pixel);
    }

    contours.push_back(contour);
  }

  // Draw the contours on the frozen map.
  cv::drawContours(frozen, contours, -1, cv::Scalar::all(1), -1);

  return frozen;
}

} // namespace yabloc::lanelet2_cost_map
