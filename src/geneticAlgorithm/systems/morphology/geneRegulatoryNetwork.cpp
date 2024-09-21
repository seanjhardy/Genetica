#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <simulator/entities/lifeform.hpp>
#include <simulator/simulator.hpp>

void GeneRegulatoryNetwork::update(float dt) {
    // Compute distances between cells
    precomputeDistances();

    // Set the external factors (like lifeform energy) for each cell and decay internal/external products
    updateProductConcentrations(dt);

    // For each cell, update it's products based on the regulatory units
    for (auto& cell : lifeForm->cells) {
        print("cell:", cell->products.size());
        std::unordered_map<Gene*, float> newFactorLevels;
        for (auto& unit : regulatoryUnits) {
            std::unordered_map<Gene*, float> factorLevels = unit.calculateActivation(promoters,
                                                                                     factors,
                                                                                     cell->products,
                                                                                     promoterFactorAffinities);
            for (auto& [factor, level] : factorLevels) {
                newFactorLevels[factor] += level;
            }
        }
        // Add factor levels back to cell's products
        print("newFactorLevels:", newFactorLevels.size());
        for (auto& [factor, level] : newFactorLevels) {
            cell->products[factor] += level;
            print("factor:", cell->products[factor]);
        }
    }

    // Update the gene expression of each cell
    updateGeneExpression(dt);
}

void GeneRegulatoryNetwork::render(VertexManager& vertexManager) {
    float size = 0.05f;
    // Loop over affinities
    for (auto& [pair, affinity] : promoterFactorAffinities) {
        auto& [first, second] = pair;
        float2 pos1 = {first->embedding[0], first->embedding[1]};
        float2 pos2 = {second->embedding[0], second->embedding[1]};
        sf::Color color;
        if (affinity == 0) continue;
        if (affinity < 0) {
            color = interpolate(sf::Color::Yellow, sf::Color::Red, -affinity/10);
        } else {
            color = interpolate(sf::Color::Yellow, sf::Color::Green, affinity/10);
        }
        vertexManager.addLine(pos1, pos2, color, size/6);
    }
    for (auto& [pair, affinity] : factorEffectorAffinities) {
        auto& [first, second] = pair;
        float2 pos1 = {first->embedding[0], first->embedding[1]};
        float2 pos2 = {second->embedding[0], second->embedding[1]};
        sf::Color color;
        if (affinity == 0) continue;
        if (affinity < 0) {
            color = interpolate(sf::Color::Yellow, sf::Color::Red, -affinity);
        } else {
            color = interpolate(sf::Color::Yellow, sf::Color::Green, affinity);
        }
        vertexManager.addLine(pos1, pos2, color, size/6);
    }
    for (auto& [pair, affinity] : factorReceptorAffinities) {
        auto& [first, second] = pair;
        float2 pos1 = {first->embedding[0], first->embedding[1]};
        float2 pos2 = {second->embedding[0], second->embedding[1]};
        sf::Color color;
        if (affinity == 0) continue;
        if (affinity < 0) {
            color = interpolate(sf::Color(255, 0, 0, 0), sf::Color::Red, -affinity * 1000);
        } else {
            color = interpolate(sf::Color(0, 255, 0, 0), sf::Color::Green, affinity * 1000);
        }
        vertexManager.addLine(pos1, pos2, color, size/6);
    }
    for (auto& factor: factors) {
        float2 pos = {factor.embedding[0], factor.embedding[1]};
        vertexManager.addCircle(pos,size, sf::Color::Blue);
    }
    for (auto& gene: promoters) {
        float2 pos = {gene.embedding[0], gene.embedding[1]};
        vertexManager.addCircle(pos, size, sf::Color::Red);
    }
    for (auto& effector: effectors) {
        float2 pos = {effector.embedding[0], effector.embedding[1]};
        vertexManager.addCircle(pos, size, sf::Color::Green);
    }
}

void GeneRegulatoryNetwork::precomputeAffinities() {
    for (auto& factor : factors) {
        for (auto& promoter : promoters) {
            auto pair = std::make_pair(&promoter,&factor);
            const float affinity = promoter.calculateAffinity(factor);
            promoterFactorAffinities.insert({pair, affinity});
        }
    }
    for (auto& factor : factors) {
        for (auto& effector : effectors) {
            auto pair = std::make_pair(&factor, &effector);
            const float affinity = effector.calculateAffinity(factor);
            factorEffectorAffinities.insert({pair, affinity});
        }
    }
    for (auto& factor : factors) {
        for (auto& receptor : factors) {
            if (&factor == &receptor) continue;
            auto pair = std::make_pair(&factor,&receptor);
            const float affinity = receptor.calculateAffinity(factor);
            factorReceptorAffinities.insert({pair, affinity});
        }
    }
}

void GeneRegulatoryNetwork::precomputeDistances(){
    cellDistances.clear();
    for (auto& cell : lifeForm->cells) {
        if (cell->frozen) continue;
        for (auto& other : lifeForm->cells) {
            if (other->frozen) continue;
            if (&cell == &other) continue;
            if (cellDistances.find({cell.get(), other.get()}) != cellDistances.end()) continue;
            Point* p1 = lifeForm->getEnv()->getPoint(cell->pointIdx);
            Point* p2 = lifeForm->getEnv()->getPoint(other->pointIdx);

            float d = p1->distanceTo(*p2);
            cellDistances.insert({{cell.get(), other.get()}, d});
            cellDistances.insert({{other.get(), cell.get()}, d});
        }
    }
}

// GRN Inputs
void GeneRegulatoryNetwork::updateProductConcentrations(float deltaTime) {
    float2 headPos = lifeForm->getEnv()->getPoint(lifeForm->head->pointIdx)->pos;
    float decayRate = pow(0.99, deltaTime);

    // Update the concentration of products in every cell
    for (const auto& cell : lifeForm->cells) {
        if (cell->frozen) continue;
        Point* p1 = lifeForm->getEnv()->getPoint(cell->pointIdx);
        float2 divisionVector = {0, 0};

        for (auto& [product, amount] : cell->products) {
            // Update product quantities in cell
            if (product->factorType == Gene::FactorType::MaternalFactor) {
                float2 factorPos = rotate(headPos + product->extra, lifeForm->head->rotation);
                amount = distanceBetween(factorPos, lifeForm->getEnv()->getPoint(cell->pointIdx)->pos);
            }
            if (product->factorType == Gene::FactorType::Time) {
                amount = product->extra.y * (product->sign ? 1 : -1)
                  + (lifeForm->birthdate - Simulator::get().getStep()) / std::abs(10000 * product->extra.x);
            }
            if (product->factorType == Gene::FactorType::Constant) {
                amount = product->extra.x;
            }
            if (product->factorType == Gene::FactorType::Generation) {
                amount = (float)cell->generation;
            }
            if (product->factorType == Gene::FactorType::Energy) {
                amount = lifeForm->energy / product->extra.x;
            }
            if (product->factorType == Gene::FactorType::Crowding) {
                amount = 0;
                for (const auto& otherCell: lifeForm->cells) {
                    if (otherCell->frozen) continue;
                    if (&otherCell == &cell) continue;
                    float distance = cellDistances.at({cell.get(), otherCell.get()});
                    amount += 1.0f / (1.0f + distance);
                }
            }
            //Decay products
            if (product->factorType == Gene::FactorType::InternalProduct) {
                amount *= decayRate;
            }
            if (product->factorType == Gene::FactorType::ExternalProduct) {
                amount *= decayRate;
                for (const auto& otherCell: lifeForm->cells) {
                    if (otherCell->frozen) continue;
                    if (&otherCell == &cell) continue;
                    float distance = cellDistances.at({cell.get(), otherCell.get()});
                    float distanceScale = 1.0f / (1.0f + distance);

                    cell->products[product] += distanceScale * otherCell->products[product];
                }
            }
            if (product->factorType == Gene::FactorType::Receptor) {
                for (const auto& otherCell: lifeForm->cells) {
                    if (otherCell->frozen) continue;
                    if (&otherCell == &cell) continue;
                    float distance = cellDistances.at({cell.get(), otherCell.get()});
                    Point* p2 = lifeForm->getEnv()->getPoint(otherCell->pointIdx);
                    float2 normalisedVectorToCell = p1->pos - p2->pos / distance;

                    for(auto& [otherProduct, otherAmount] : otherCell->products) {
                        if (otherProduct->factorType != Gene::FactorType::ExternalProduct) continue;
                        float affinity = factorReceptorAffinities.at({otherProduct, product});
                        divisionVector += amount * affinity * otherAmount * normalisedVectorToCell;
                    }
                }
            }
            cell->divisionRotation = std::atan2(divisionVector.y, divisionVector.x);
        }
    }
}

void GeneRegulatoryNetwork::updateGeneExpression(float deltaTime) {
    for (auto& cell : lifeForm->cells) {
        if (cell->frozen) continue;
        for (auto& effector : effectors) {
            float expression = 0.0f;
            for (auto& [gene, level] : cell->products) {
                if (gene->factorType != Gene::FactorType::InternalProduct) continue;
                expression += level * factorEffectorAffinities.at({gene, &effector});
            }
            if (effector.effectorType == Effector::EffectorType::Die) {
                if (expression > 0.9) {
                    cell->die();
                }
            }
            if (effector.effectorType == Effector::EffectorType::Divide) {
                if (expression > 0.9) {
                    cell->divide();
                }
            }
            if (effector.effectorType == Effector::EffectorType::Freeze) {
                if (expression > 0.9) {
                    cell->frozen = true;
                }
            }
            if (effector.effectorType == Effector::EffectorType::DaughterDistance) {
                for (auto& cellLink : lifeForm->links) {
                    cellLink->adjustSize(0.1f * expression);
                }
            }
            if (effector.effectorType == Effector::EffectorType::Radius) {
                cell->adjustSize(0.1f * expression);
            }
            //if (effector.effectorType == Effector::EffectorType::Rotate) {
            //    cell->rotation += 0.1f * expression;
            //}
        }
    }
}