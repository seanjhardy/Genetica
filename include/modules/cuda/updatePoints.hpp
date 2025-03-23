#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "modules/cuda/structures/GPUVector.hpp"
#include "modules/cuda/structures/CGPUValue.hpp"
#include <geneticAlgorithm/cellParts/cellLink.hpp>

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Cell>& cells,
                  GPUVector<CellLink>& cellLinks,
                  CGPUValue<sf::FloatRect>& bounds);

int movePoint(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos);

#endif