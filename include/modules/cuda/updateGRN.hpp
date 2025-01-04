#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"

void computeAffinities(GeneRegulatoryNetwork &grn);

void updateGRN(GeneRegulatoryNetwork& grn, const GPUVector<int>& cellIdxs, const GPUVector<Cell>& cells, const GPUVector<Point>& points);

#endif