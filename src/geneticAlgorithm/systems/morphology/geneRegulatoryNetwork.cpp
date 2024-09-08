#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <simulator/entities/lifeform.hpp>

void GeneRegulatoryNetwork::precomputeAffinities() {
    for (auto factor : factors) {
        if (factor.factorType != Gene::FactorType::ExternalProduct) {
            continue;
        }
        for (auto promoter : elements) {
            if (promoter.type != GeneticUnit::Type::Promoter) {
                continue;
            }
            Promoter promoterGene = static_cast<Promoter&>(promoter);
            Gene factorGene = static_cast<Gene&>(factor);
            auto pair = std::make_pair(&promoterGene,&factorGene);
            const float affinity = promoterGene.calculateAffinity(factorGene);
            promoterFactorAffinities.insert({pair, affinity});
        }
    }
}

std::unordered_map<Gene*, float> GeneRegulatoryNetwork::calculateMorphogenPerception(const Cell& cell) {
    std::unordered_map<Gene*, float> perceivedLevels;
    Point* cellPoint = lifeForm->getEnv()->getPoint(cell.pointIdx);

    // For every other cell (that can be producing morphogens)
    for (const auto& otherCell : lifeForm->cells) {
        if (otherCell != &cell) continue;

        Point* otherCellPoint = lifeForm->getEnv()->getPoint(otherCell->pointIdx);

        float distance = distanceBetween(cellPoint->pos, otherCellPoint->pos);

        float distanceScale = 1.0f / (1.0f + distance);

        // Get the distance from our cell to the source cell
        for (const auto& product : cell.products) {
            if (product.first->factorType == Gene::FactorType::ExternalProduct) {
                float perceivedLevel = distanceScale * product.second;
                perceivedLevels[product.first] += perceivedLevel;
            }
        }
    }
    return perceivedLevels;
}

void GeneRegulatoryNetwork::updateProductConcentrations(float deltaTime) {
    for (auto& cell : lifeForm->cells) {
        std::unordered_map<Gene*, float> deltaLevels
                                    = calculateMorphogenPerception(*cell);
        for (auto& product : cell->products) {
            product.second += perce
        }
    }
}

std::vector<double> GeneRegulatoryNetwork::calculateExternalDivisionVector(const Cell& cell) {
    float externalDivisionRotation = 0;
    for (auto& other : lifeForm->cells) {
        if (&other == &cell) continue;

        for (auto& receptor : factors) {
            if (receptor.factorType != Gene::FactorType::Receptor) continue;
            for (auto& morphogen : factors) {
                if (morphogen.factorType != Gene::FactorType::ExternalProduct) continue;
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
        if (receptor.type != Gene::FactorType::Receptor) continue;
        for (const auto& morphogen : cell.products) {
            if (morphogen.type != Gene::FactorType::External) continue;
            for (const auto& sourceCell : lifeForm->cells) {
                if (&sourceCell == &cell) continue;
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
    return externalVector;
}