#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include "../verlet/point.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void updatePointsOnGPU(std::vector<Point>& points, const sf::FloatRect& bounds, float dt = 1.0f);

#endif