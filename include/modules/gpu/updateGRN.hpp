#ifndef UPDATE_GRN
#define UPDATE_GRN

#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"

void updateGRN(LifeForm& lifeForm, GPUVector<Point>& points, GPUVector<Cell>& cells,
    GPUVector<CellLink>& cellLinks);

#endif