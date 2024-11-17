#ifndef GENE_REGULATORY_NETWORK
#define GENE_REGULATORY_NETWORK

#include <vector>
#include <unordered_map>
#include <map>
#include <modules/cuda/CGPUValue.hpp>

#include "./gene.hpp"
#include "./promoter.hpp"
#include "./effector.hpp"
#include "./regulatoryUnit.hpp"
#include "geneticAlgorithm/cellParts/cell.hpp"

class LifeForm;

class GeneRegulatoryNetwork {
public:
    Gene* factors;
    int numFactors;
    Promoter* promoters;
    int numPromoters;
    Effector* effectors;
    int numEffectors;

    RegulatoryUnit* regulatoryUnits;
    int numRegulatoryUnits;

    float* promoterFactorAffinities;
    float* factorEffectorAffinities;
    float* factorReceptorAffinities;

    std::map<std::pair<const Cell*, const Cell*>, float> cellDistances;

    void render(VertexManager& vertexManager);

    void precomputeAffinities();
    void precomputeDistances();
    void updateProductConcentrations(float deltaTime);
    void updateGeneExpression(float deltaTime);
};

#endif