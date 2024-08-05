#pragma once

#include <Eigen/Core>

namespace yabloc::modularized_particle_filter {

// These 2 functions are used to evaluate how close a direction vector is to a
// given angle in 2D space.

float score_closeness_by_dot_product(
  const Eigen::Vector2f& direction,
  const float angle_deg
);

float score_closeness_by_angle_diff(
  const Eigen::Vector2f& direction,
  const float angle_deg
);

} // namespace yabloc::modularized_particle_filter
