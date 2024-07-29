#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include "../verlet/point.hpp"
#include "utils/GPUVector.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Connection>& connections,
                  const sf::FloatRect& bounds,
                  float dt = 1.0f);

#endif