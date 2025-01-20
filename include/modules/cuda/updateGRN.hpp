#ifndef UPDATE_GRN
#define UPDATE_GRN

#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"

__device__ void updateGRN(GeneRegulatoryNetwork& grn, GPUVector<int>& cellIdxs, GPUVector<Cell>& cells, GPUVector<Point>& points);

#endif