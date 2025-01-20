#ifndef FIND_NEAREST
#define FIND_NEAREST

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "modules/cuda/structures/GPUVector.hpp"
#include "modules/cuda/structures/CGPUValue.hpp"

std::pair<int, float> findNearest(const GPUVector<Point>& points,
                                  float x, float y, float minDistance);

#endif