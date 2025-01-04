#ifndef GENE_REGULATORY_NETWORK
#define GENE_REGULATORY_NETWORK

#include <vector>
#include <unordered_map>
#include <map>
#include "modules/cuda/structures/CGPUValue.hpp"

#include "./gene.hpp"
#include "./promoter.hpp"
#include "./effector.hpp"
#include "./regulatoryUnit.hpp"
#include <geneticAlgorithm/cellParts/cell.hpp>
#include "modules/cuda/structures/GPUVector.hpp"

class LifeForm;

class GeneRegulatoryNetwork {
public:
    StaticGPUVector<Gene> factors;
    StaticGPUVector<Promoter> promoters;
    StaticGPUVector<Effector> effectors;
    StaticGPUVector<RegulatoryUnit> regulatoryUnits;

    StaticGPUVector<float> promoterFactorAffinities;
    StaticGPUVector<float> factorEffectorAffinities;
    StaticGPUVector<float> factorReceptorAffinities;

    StaticGPUVector<float> cellDistances;
};

#endif