#ifndef UPDATE_BLUEPRINT
#define UPDATE_BLUEPRINT

#include <geneticAlgorithm/lifeform.hpp>
#include <modules/physics/point.hpp>
#include "modules/cuda/structures/GPUVector.hpp"
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

void updateBlueprint(
                LifeForm& lifeform,
                GPUVector<Point>& points,
                  GPUVector<Cell>& cells,
                  GPUVector<CellLink>& cellLinks);

#endif