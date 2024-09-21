#ifndef GENE_REGULATORY_NETWORK
#define GENE_REGULATORY_NETWORK

#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include "./gene.hpp"
#include "./promoter.hpp"
#include "./effector.hpp"
#include "./regulatoryUnit.hpp"
#include "geneticAlgorithm/cellParts/cell.hpp"
#include <modules/utils/floatOps.hpp>

class LifeForm;

class GeneRegulatoryNetwork {
public:
    LifeForm* lifeForm;
    std::vector<Gene> factors;
    std::vector<Promoter> promoters;
    std::vector<Effector> effectors;

    std::vector<RegulatoryUnit> regulatoryUnits;

    std::map<std::pair<Promoter*, Gene*>, float> promoterFactorAffinities;
    std::map<std::pair<Gene*, Effector*>, float> factorEffectorAffinities;
    std::map<std::pair<Gene*, Gene*>, float> factorReceptorAffinities;

    std::map<std::pair<const Cell*, const Cell*>, float> cellDistances;

    void update(float deltaTime);
    void render(VertexManager& vertexManager);

    void precomputeAffinities();
    void precomputeDistances();
    void updateProductConcentrations(float deltaTime);
    void updateGeneExpression(float deltaTime);
};

#endif