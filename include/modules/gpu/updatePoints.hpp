#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include <modules/physics/point.hpp>
#include "modules/gpu/structures/GPUVector.hpp"
#include "modules/gpu/structures/CGPUValue.hpp"
#include <geneticAlgorithm/cellParts/cellLink.hpp>

void updatePoints(GPUVector<Point>& points,
  CGPUValue<sf::FloatRect>& bounds);

#endif