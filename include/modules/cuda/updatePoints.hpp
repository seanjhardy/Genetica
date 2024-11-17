#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUVector.hpp"
#include "CGPUValue.hpp"

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Connection>& connections,
                  CGPUValue<sf::FloatRect>& bounds,
                  float dt = 1.0f);

#endif