#ifndef UPDATE_GRN
#define UPDATE_GRN

#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"

__device__ void updateGRN(LifeForm& lifeForm,
               GPUVector<Cell>& cells,
               GPUVector<CellLink>& cellLinks,
               GPUVector<Point>& points);

#endif