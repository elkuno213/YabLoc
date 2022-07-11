#include "modularized_particle_filter/correction/abst_corrector.hpp"

#include "modularized_particle_filter/correction/correction_util.hpp"

AbstCorrector::AbstCorrector(const std::string & node_name)
: Node(node_name), visualize_(declare_parameter<bool>("visualize", false))
{
  using std::placeholders::_1;
  particle_pub_ = create_publisher<ParticleArray>("/weighted_particles", 10);
  particle_sub_ = create_subscription<ParticleArray>(
    "/predicted_particles", 10, std::bind(&AbstCorrector::particleArrayCallback, this, _1));

  if (visualize_) pub_marker_array_ = create_publisher<MarkerArray>("/marker_array_particles", 10);
}

void AbstCorrector::particleArrayCallback(const ParticleArray & particle_array)
{
  particle_array_buffer_.push_back(particle_array);
}

std::optional<AbstCorrector::ParticleArray> AbstCorrector::getSyncronizedParticleArray(
  const rclcpp::Time & stamp)
{
  auto itr = particle_array_buffer_.begin();
  while (itr != particle_array_buffer_.end()) {
    rclcpp::Duration dt = rclcpp::Time(itr->header.stamp) - stamp;
    if (dt.seconds() < -1.0)
      particle_array_buffer_.erase(itr++);
    else
      break;
  }

  if (particle_array_buffer_.empty()) return std::nullopt;

  auto comp = [stamp](ParticleArray & x1, ParticleArray & x2) -> bool {
    auto dt1 = rclcpp::Time(x1.header.stamp) - stamp;
    auto dt2 = rclcpp::Time(x2.header.stamp) - stamp;
    return std::abs(dt1.seconds()) < std::abs(dt2.seconds());
  };
  return *std::min_element(particle_array_buffer_.begin(), particle_array_buffer_.end(), comp);
}

void AbstCorrector::setWeightedParticleArray(const ParticleArray & particle_array)
{
  particle_pub_->publish(particle_array);
  if (visualize_) visualize(particle_array);
}

void AbstCorrector::visualize(const ParticleArray & msg)
{
  visualization_msgs::msg::MarkerArray marker_array;
  auto minmax_weight = std::minmax_element(
    msg.particles.begin(), msg.particles.end(),
    [](const Particle & a, const Particle & b) -> bool { return a.weight < b.weight; });

  float min = minmax_weight.first->weight;
  float max = minmax_weight.second->weight;
  max = std::max(max, min + 1e-7f);
  auto boundWeight = [min, max](float raw) -> float { return (raw - min) / (max - min); };

  RCLCPP_INFO_STREAM(get_logger(), "min: " << min << " max: " << max);
  int id = 0;
  for (const Particle & p : msg.particles) {
    visualization_msgs::msg::Marker marker;
    marker.frame_locked = true;
    marker.header.frame_id = "map";
    marker.id = id++;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.scale.x = 0.3;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color = computeColor(boundWeight(p.weight));
    marker.pose.orientation = p.pose.orientation;
    marker.pose.position.x = p.pose.position.x;
    marker.pose.position.y = p.pose.position.y;
    marker.pose.position.z = p.pose.position.z;
    marker_array.markers.push_back(marker);
  }

  pub_marker_array_->publish(marker_array);
}