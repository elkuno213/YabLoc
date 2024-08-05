#include <cmath>

#include "camera_particle_corrector/fast_trigonometry.hpp"

namespace yabloc::modularized_particle_filter {

FastTrigonometry::FastTrigonometry(const int bin) {
  for (int i = 0; i <= bin; ++i) {
    cos_.push_back(std::cos(i * M_PI / 180.f));
  }
}
float FastTrigonometry::cos(float deg) const {
  // Normalize deg to [0, 360).
  while (deg < 0) {
    deg += 360;
  }
  while (deg > 360) {
    deg -= 360;
  }

  // Use the lookup table.
  if (deg < 90) {
    return cos_.at(int(deg));
  } else if (deg < 180) {
    return -cos_.at(int(180 - deg));
  } else if (deg < 270) {
    return -cos_.at(int(deg - 180));
  } else {
    return cos_.at(int(360 - deg));
  }
}

float FastTrigonometry::sin(float deg) const {
  // sin(x) = cos(x - 90).
  return cos(deg - 90.f);
}

} // namespace yabloc::modularized_particle_filter
