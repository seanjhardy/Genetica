#ifndef UPDATE_CELL_LINKS
#define UPDATE_CELL_LINKS

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "modules/gpu/structures/GPUVector.hpp"
#include "modules/gpu/structures/CGPUValue.hpp"
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include "modules/utils/gpu/mathUtils.hpp"

void updateCellLinks(GPUVector<Point>& points,
  GPUVector<Cell>& cells,
  GPUVector<CellLink>& cellLinks);

#endif