#ifndef UPDATE_POINTS
#define UPDATE_POINTS

#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>

void computeAffinities(GeneRegulatoryNetwork &grn);

void updateGRN(GeneRegulatoryNetwork& grn, Cell* cells, Point* points, int numCells);

#endif