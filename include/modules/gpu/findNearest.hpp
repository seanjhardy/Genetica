#ifndef FIND_NEAREST
#define FIND_NEAREST

#include <modules/physics/point.hpp>
// COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "modules/gpu/structures/GPUVector.hpp"
#include "modules/gpu/structures/CGPUValue.hpp"

std::pair<int, float> findNearest(const GPUVector<Point>& points,
  float x, float y, float minDistance);

int movePoint(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos);

#endif