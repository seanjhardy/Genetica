#ifndef FIND_NEAREST
#define FIND_NEAREST

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPUVector.hpp"
#include "GPUValue.hpp"

std::pair<int, float> findNearest(GPUVector<Point>& points,
                  float x, float y, float minDistance);

#endif