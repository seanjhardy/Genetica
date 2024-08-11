#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include <modules/verlet/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUVector.hpp"

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Connection>& connections,
                  GPUVector<AngleConstraint>& angles,
                  const sf::FloatRect& bounds,
                  float dt = 1.0f);

#endif