#include "camera_particle_corrector/fast_trigonometry.hpp"
#include "camera_particle_corrector/score.hpp"

#include <cmath>

namespace yabloc::modularized_particle_filter {

FastTrigonometry fast_trigonometry;

float score_closeness_by_dot_product(
  const Eigen::Vector2f& direction,
  const float angle_deg
) {
  const Eigen::Vector2f angle_direction = {
    fast_trigonometry.cos(angle_deg),
    fast_trigonometry.sin(angle_deg)
  };
  return std::abs(direction.normalized().dot(angle_direction));
}

float score_closeness_by_angle_diff(
  const Eigen::Vector2f& direction,
  const float angle_deg
) {
  float diff = std::atan2(direction.y(), direction.x()) - angle_deg * M_PI / 180.f;

  // normalize to [0, pi]
  diff = std::fmod(diff, M_PI);
  if (diff < 0) {
    diff = -diff;
  }

  if (diff < M_PI_2) {
    // Decrease linearly 1 to 0.
    return 1 - diff / M_PI_2;
  } else if (diff < M_PI) {
    // Increase linearly 0 to 1.
    return diff / M_PI_2 - 1;
  } else {
    throw std::runtime_error("0?" + std::to_string(diff));
  }
}

} // namespace yabloc::modularized_particle_filter
