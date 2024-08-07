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

#ifndef MODULARIZED_PARTICLE_FILTER__COMMON__MEAN_HPP_
#define MODULARIZED_PARTICLE_FILTER__COMMON__MEAN_HPP_

#include <eigen3/Eigen/StdVector>

#include <geometry_msgs/msg/pose.hpp>
#include <modularized_particle_filter_msgs/msg/particle_array.hpp>

namespace yabloc
{
namespace modularized_particle_filter
{
geometry_msgs::msg::Pose compute_mean_pose(
  const modularized_particle_filter_msgs::msg::ParticleArray & particle_array);

Eigen::Matrix3f std_of_distribution(
  const modularized_particle_filter_msgs::msg::ParticleArray & particle_array);

float std_of_weight(const modularized_particle_filter_msgs::msg::ParticleArray & particle_array);
}  // namespace modularized_particle_filter
}  // namespace yabloc

#endif  // MODULARIZED_PARTICLE_FILTER__COMMON__MEAN_HPP_
