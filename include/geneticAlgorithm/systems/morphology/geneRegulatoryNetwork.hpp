#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include "./geneticUnit.hpp"
#include "./regulatoryUnit.hpp"
#include "geneticAlgorithm/cellParts/cell.hpp"
#include <modules/utils/floatOps.hpp>

class LifeForm;

class GeneRegulatoryNetwork {
public:
    LifeForm* lifeForm;
    std::vector<GeneticUnit> elements;
    std::vector<Gene> factors;
    std::vector<RegulatoryUnit> regulatoryUnits;

    std::map<std::pair<Promoter*, Gene*>, float> promoterFactorAffinities;
    std::map<Promoter*, float> promoterActivities;

    void precomputeAffinities();
    std::unordered_map<Gene*, float> calculateMorphogenPerception(const Cell& cell);
    void updateProductConcentrations(float deltaTime);
    std::vector<double> calculateExternalDivisionVector(const Cell& cell);
};