#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include "./geneticUnit.hpp"
#include "./regulatoryUnit.hpp"
#include "geneticAlgorithm/cellParts/cell.hpp"
#include <modules/utils/floatOps.hpp>

class GeneRegulatoryNetwork {
public:
    std::vector<Cell*> cells;
    std::vector<GeneticUnit> elements;
    std::vector<Factor> factors;
    std::vector<RegulatoryUnit> regulatoryUnits;

    std::map<std::pair<Promoter*, Factor*>, float> promoterFactorAffinities;
    std::map<Promoter*, float> promoterActivities;

    void precomputeAffinities() {
        for (auto factor : factors) {
            if (factor.factorType != Factor::FactorType::ExternalProduct) {
                continue;
            }
            for (auto promoter : elements) {
                if (promoter.type != GeneticUnit::Type::Promoter) {
                    continue;
                }
                Promoter promoterGene = static_cast<Promoter&>(promoter);
                Factor factorGene = static_cast<Factor&>(factor);
                auto pair = std::make_pair(&promoterGene,&factorGene);
                const float affinity = promoterGene.calculateAffinity(factorGene);
                promoterFactorAffinities.insert({pair, affinity});
            }
        }
    }

    std::unordered_map<Factor*, float> calculateMorphogenPerception(const Cell& cell) {
        std::unordered_map<Factor*, float> perceivedLevels;
        // For every other cell (that can be producing morphogens)
        for (const auto& otherCell : cells) {
            if (otherCell != &cell) continue;

            float distance = distanceBetween(cell.pos, otherCell->pos);

            float distanceScale = 1.0f / (1.0f + distance);

            // Get the distance from our cell to the source cell
            for (const auto& product : cell.products) {
                if (product.first->factorType == Factor::FactorType::ExternalProduct) {
                    float perceivedLevel = distanceScale * product.second;
                    perceivedLevels[product.first] += perceivedLevel;
                }
            }
        }
        return perceivedLevels;
    }

    void updateProductConcentrations(float deltaTime) {
        for (auto& cell : cells) {
            std::unordered_map<Factor*, float> deltaLevels
            = calculateMorphogenPerception(*cell);
            for (auto& product : cell->products) {
                product.second += perce
            }
        }
    }

    std::vector<double> calculateExternalDivisionVector(const Cell& cell) {
        float externalDivisionRotation = 0;
        for (auto& other : cells) {
            if (&other == &cell) continue;
            for (auto& receptor : factors) {
                if (receptor.factorType != Factor::FactorType::Receptor) continue;
                for (auto& morphogen : factors) {
                    if (morphogen.factorType != Factor::FactorType::ExternalProduct) continue;
                    float affinity = 0.0;
                    // Calculate affinity between receptor and morphogen
                    // (This would require access to the genetic elements, which is not provided in this simplified version)

                    float distance = distanceBetween(cell.pos, other.pos);

                    float morphogenLevel = morphogen.level / (1.0 + distance);

                    float direction = other.pos - cell.pos;
                    float distance = direction.length();
                    direction.normalize();

                    externalDivisionRotation += receptor.level * affinity * morphogenLevel * direction;
                }
            }
        }
        for (const auto& receptor : cell.products) {
            if (receptor.type == ProductType::Receptor) {
                for (const auto& morphogen : cell.products) {
                    if (morphogen.type == ProductType::External) {
                        for (const auto& sourceCell : allCells) {
                            if (&sourceCell != &cell) {
                                double affinity = 0.0;
                                // Calculate affinity between receptor and morphogen
                                // (This would require access to the genetic elements, which is not provided in this simplified version)

                                std::vector<double> direction(cell.position.size());
                                double distance = 0.0;
                                for (size_t i = 0; i < cell.position.size(); ++i) {
                                    direction[i] = sourceCell.position[i] - cell.position[i];
                                    distance += std::pow(direction[i], 2);
                                }
                                distance = std::sqrt(distance);

                                double morphogenLevel = morphogen.level / (1.0 + distance);

                                for (size_t i = 0; i < externalVector.size(); ++i) {
                                    externalVector[i] += receptor.level * affinity * morphogenLevel * (direction[i] / distance);
                                }
                            }
                        }
                    }
                }
            }
        }
        return externalVector;
    }

    void divide(Cell& mother) {
        Cell daughter = mother;
        daughter.products = mother.products;
        // TODO: reset value of division products (?)
        daughter.internalDivisionRotation = mother.internalDivisionRotation;
        daughter.pos = mother.pos + vec(mother.externalDivisionRotation) * mother.childDistance;
        cells.push_back(daughter);
    }
};