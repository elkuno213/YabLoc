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

#include "camera_particle_corrector/logit_probit.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace yabloc {

namespace {

struct ProbitToLogitLUT {
  ProbitToLogitLUT() {
    for (int i = 0; i < 100; ++i) {
      float p      = i / 100.0f;
      table_.at(i) = std::log(p / std::max(1 - p, 1e-6f));
    }
  }
  float operator()(float prob) const {
    int index = std::clamp(int(prob * 100), 0, 99);
    return table_.at(index);
  }

  std::array<float, 100> table_;
} probit_to_logit_lut;

} // namespace

float logit_to_probit(float logit, float gain) {
  return 1.f / (1 + std::exp(-gain * logit));
}

float probit_to_logit(float prob) {
  return probit_to_logit_lut(prob);
}

} // namespace yabloc
