#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include "geneticAlgorithm/lifeform.hpp"
#include <simulator/simulator.hpp>
#include <modules/utils/GUIUtils.hpp>
/*
void GeneRegulatoryNetwork::update(float dt) {
    // Compute distances between cells
    precomputeDistances();

    // Set the external factors (like lifeform energy) for each cell and decay internal/external products
    updateProductConcentrations(dt);

    // For each cell, update it's products based on the regulatory units
    for (auto& cell : lifeForm->cells) {
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
        for (auto& [factor, level] : newFactorLevels) {
            cell->products[factor] += level;
        }
    }

    // Update the gene expression of each cell
    updateGeneExpression(dt);
}

void GeneRegulatoryNetwork::precomputeDistances(){
    cellDistances.clear();
    for (auto& cell : lifeForm->cells) {
        if (cell->frozen) continue;
        for (auto& other : lifeForm->cells) {
            if (other->frozen) continue;
            if (&cell == &other) continue;
            if (cellDistances.find({cell, other}) != cellDistances.end()) continue;
            Point* p1 = lifeForm->getEnv()->getPoint(cell->pointIdx);
            Point* p2 = lifeForm->getEnv()->getPoint(other->pointIdx);

            float d = p1->distanceTo(*p2);
            cellDistances.insert({{cell, other}, d});
            cellDistances.insert({{other, cell}, d});
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
                float2 factorPos = rotate(product->extra * 10.0f, lifeForm->head->rotation) + headPos;
                amount = distanceBetween(factorPos, p1->pos);
            }
            if (product->factorType == Gene::FactorType::Time) {
                amount = product->extra.y * (product->sign ? 1.0f : -1.0f)
                  + product->extra.x * (float)(Simulator::get().getStep() - lifeForm->birthdate)/100000.0;
            }
            if (product->factorType == Gene::FactorType::Constant) {
                amount = (float)product->extra.x;
            }
            if (product->factorType == Gene::FactorType::Generation) {
                amount = (float)cell->generation;
            }
            if (product->factorType == Gene::FactorType::Energy) {
                amount = lifeForm->energy * max(product->extra.x, 0.1f);
            }
            if (product->factorType == Gene::FactorType::Crowding) {
                amount = 0;
                for (const auto& otherCell: lifeForm->cells) {
                    if (otherCell->frozen) continue;
                    if (&otherCell == &cell) continue;
                    float distance = cellDistances.at({cell, otherCell});
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
                    float distance = cellDistances.at({cell, otherCell});
                    float distanceScale = 1.0f / (1.0f + distance);

                    cell->products[product] += distanceScale * otherCell->products[product];
                }
            }
            if (product->factorType == Gene::FactorType::Receptor && cell != lifeForm->head) {
                for (const auto& otherCell: lifeForm->cells) {
                    if (otherCell->frozen) continue;
                    if (&otherCell == &cell) continue;
                    float distance = cellDistances.at({cell, otherCell});
                    Point* p2 = lifeForm->getEnv()->getPoint(otherCell->pointIdx);
                    float2 normalisedVectorToCell = p1->pos - p2->pos / distance;

                    for(auto& [otherProduct, otherAmount] : otherCell->products) {
                        if (otherProduct->factorType != Gene::FactorType::ExternalProduct) continue;
                        float affinity = factorReceptorAffinities.at({otherProduct, product});
                        divisionVector += amount * affinity * otherAmount * normalisedVectorToCell;
                    }
                }
                consoleLog("Cell division rotation: ", cell->divisionRotation);
                cell->divisionRotation = std::atan2(divisionVector.y, divisionVector.x);
            }
        }
    }
}

void GeneRegulatoryNetwork::updateGeneExpression(float deltaTime) {
    auto cells = lifeForm->cells;
    for (auto cell : cells) {
        if (cell->frozen) continue;
        for (auto& effector : effectors) {
            float expression = 0.0f;
            for (auto& [gene, level] : cell->products) {
                if (gene->factorType != Gene::FactorType::InternalProduct) continue;
                expression += level * factorEffectorAffinities.at({gene, &effector});
            }
            if (expression == 0) continue;
            if (effector.effectorType == Effector::EffectorType::Die) {
                if (expression > 0.5 && cell != lifeForm->head) {
                    cell->die();
                }
            }
            if (effector.effectorType == Effector::EffectorType::Divide) {
                if (expression > 0.0001) {
                    cell->dividing = true;
                }
            }
            if (effector.effectorType == Effector::EffectorType::Freeze) {
                if (expression > 0.5) {
                    cell->frozen = true;
                }
            }
            if (effector.effectorType == Effector::EffectorType::Distance) {
                for (auto& cellLink : lifeForm->links) {
                    cellLink->adjustSize(max(expression, 0.0f));
                }
            }
            if (effector.effectorType == Effector::EffectorType::Radius) {
                cell->adjustSize(max(expression, 0.0f));
            }
            if (effector.effectorType == Effector::EffectorType::Red) {
                cell->updateHue(PIGMENT::Red, expression * 0.1f);
            }
            if (effector.effectorType == Effector::EffectorType::Green) {
                cell->updateHue(PIGMENT::Green, expression * 0.1f);
            }
            if (effector.effectorType == Effector::EffectorType::Blue) {
                cell->updateHue(PIGMENT::Blue, expression * 0.1f);
            }
        }
    }
}

void GeneRegulatoryNetwork::render(VertexManager& vertexManager) {
    int materialFactorCount = 0;
    for (auto& factor: factors) {
        if (factor.factorType != Gene::FactorType::ExternalProduct &&
            factor.factorType != Gene::FactorType::InternalProduct &&
            factor.factorType != Gene::FactorType::Receptor) {
            materialFactorCount++;
        }
    }

    int nodesPerRow = 7;
    int maternalFactorRows = ceil((float)materialFactorCount / nodesPerRow);
    int regulatoryUnitRows = ceil((float)regulatoryUnits.size() / nodesPerRow);
    int effectorRows = ceil((float)effectors.size() / nodesPerRow);

    float rowHeight = 0.9f / max(float(maternalFactorRows + regulatoryUnitRows + effectorRows - 1), 1.0f);
    float colWidth = 1.2f / float(nodesPerRow - 1);

    auto getPos = [nodesPerRow, rowHeight, colWidth,
                   maternalFactorRows, regulatoryUnitRows](int type, int index) {
        int col = 0;
        int row = 0;
        if (type == 0) {
            col = index % nodesPerRow;
            row = floor(index / nodesPerRow);
        } else if (type == 1) {
            col = index %  nodesPerRow;
            row = floor((index + (maternalFactorRows) * nodesPerRow) / nodesPerRow);
        } else if (type == 2) {
            col = index % nodesPerRow;
            row = floor((index + (maternalFactorRows + regulatoryUnitRows) * nodesPerRow) / nodesPerRow);
        }
        return make_float2(-0.1f + col * colWidth, 0.05f + row * rowHeight);
    };

    auto getLineColour = [](float affinity) {
        if (affinity < 0) {
            return sf::Color(255, 0, 0, 200.0f * min(-affinity * 1000.0f, 1.0f));
        } else {
            return sf::Color(0, 255, 0, 200.0f * min(affinity * 1000.0f, 1.0f));
        }
    };

    float size = 0.05f;
    float affinityScale = 1000.0f;

    // Maternal Factor to regulatory unit lines
    for (int regulatoryUnitIndex = 0; regulatoryUnitIndex < regulatoryUnits.size(); regulatoryUnitIndex++) {
        float2 pos = getPos(1, regulatoryUnitIndex)  + make_float2(0, -size/2);
        for (auto &promoterIndex: regulatoryUnits[regulatoryUnitIndex].promoters) {
            int factorIndex = 0;
            for (auto &factor: factors) {
                if (factor.factorType == Gene::FactorType::ExternalProduct ||
                    factor.factorType == Gene::FactorType::InternalProduct ||
                    factor.factorType == Gene::FactorType::Receptor) {
                    continue;
                }
                float2 pos2 = getPos(0, factorIndex);
                factorIndex += 1;
                float affinity = promoterFactorAffinities.at({&promoters[promoterIndex], &factor});
                if (affinity == 0) continue;
                vertexManager.addLine(pos, pos2, getLineColour(affinity), size / 6);
            }
        }
    }
    // Maternal factor -> effector lines
    for (int effectorIndex = 0; effectorIndex < effectors.size(); effectorIndex++) {
        float2 pos = getPos(2, effectorIndex);
        int factorIndex = 0;
        for (auto& factor: factors) {
            if (factor.factorType == Gene::FactorType::ExternalProduct ||
                factor.factorType == Gene::FactorType::InternalProduct ||
                factor.factorType == Gene::FactorType::Receptor) {
                continue;
            }
            float2 pos2 = getPos(0, factorIndex);
            factorIndex += 1;
            float affinity = factorEffectorAffinities.at({&factors[factorIndex], &effectors[effectorIndex]});
            if (affinity == 0) continue;
            vertexManager.addLine(pos, pos2, getLineColour(affinity), size / 6);
        }
    }


    for (int regulatoryUnitIndex = 0; regulatoryUnitIndex < regulatoryUnits.size(); regulatoryUnitIndex++) {
        float2 pos = getPos(1, regulatoryUnitIndex) + make_float2(0, size/2);
        for (auto& factorIndex : regulatoryUnits[regulatoryUnitIndex].factors) {
            // Regulatory unit to effector lines
            for (int effectorIndex = 0; effectorIndex < effectors.size(); effectorIndex++) {
                float2 pos2 = getPos(2, effectorIndex);
                float affinity = factorEffectorAffinities.at({&factors[factorIndex], &effectors[effectorIndex]});
                if (affinity == 0) continue;
                vertexManager.addLine(pos, pos2, getLineColour(affinity), size / 6);
            }
            // To other regulatory units
            for (int regulatoryUnitIndex2 = 0; regulatoryUnitIndex2 < regulatoryUnits.size(); regulatoryUnitIndex2++) {
                for (int promoter: regulatoryUnits[regulatoryUnitIndex2].promoters) {
                    float2 pos2 = getPos(1, regulatoryUnitIndex2) + make_float2(0, -size/2);
                    float affinity = promoterFactorAffinities.at({&promoters[promoter], &factors[factorIndex]});
                    vertexManager.addLine(pos, pos2, getLineColour(affinity), size / 6);
                }
            }
        }
    }

    Cell* head = lifeForm->head;
    int maternalFactorIndex = 0;
    for (auto& factor: factors) {
        if (factor.factorType == Gene::FactorType::ExternalProduct ||
            factor.factorType == Gene::FactorType::InternalProduct ||
            factor.factorType == Gene::FactorType::Receptor) {
            continue;
        }
        float2 pos = getPos(0, maternalFactorIndex);
        vertexManager.addCircle(pos,size, sf::Color::Blue);
        std::string name;
        switch (factor.factorType) {
            case Gene::FactorType::MaternalFactor:
                name = "Fctr";
                break;
            case Gene::FactorType::Crowding:
                name = "Crwd";
                break;
            case Gene::FactorType::Constant:
                name = "Cnst";
                break;
            case Gene::FactorType::Generation:
                name = "Gen";
                break;
            case Gene::FactorType::Energy:
                name = "Enrgy";
                break;
            case Gene::FactorType::Time:
                name = "Time";
                break;
        }
        float amount = head->products[&factor];
        vertexManager.addText(name + ":\n" + roundToDecimalPlaces(amount, 3),
                              pos, 0.0015f, sf::Color::White, TextAlignment::Center);
        maternalFactorIndex += 1;
    }
    for (int regulatoryUnitIndex = 0; regulatoryUnitIndex < regulatoryUnits.size(); regulatoryUnitIndex++) {
        float2 pos = getPos(1, regulatoryUnitIndex);
        regulatoryUnits[regulatoryUnitIndex].render(vertexManager, pos, size, promoters, factors, lifeForm->head->products);
    }
    int effectorIndex = 0;
    for (auto& effector: effectors) {
        float2 pos = getPos(2, effectorIndex);
        vertexManager.addCircle(pos, size, sf::Color::Green);
        std::string name;
        switch (effector.effectorType) {
            case Effector::EffectorType::Die:
                name = "Die";
                break;
            case Effector::EffectorType::Divide:
                name = "Divd";
                break;
            case Effector::EffectorType::Freeze:
                name = "Frz";
                break;
            case Effector::EffectorType::Distance:
                name = "Dst";
                break;
            case Effector::EffectorType::Radius:
                name = "Rad";
                break;
            case Effector::EffectorType::Red:
                name = "Red";
                break;
            case Effector::EffectorType::Green:
                name = "Green";
                break;
            case Effector::EffectorType::Blue:
                name = "Blue";
                break;
            case Effector::EffectorType::Chloroplast:
                name = "Chlr";
                break;
            case Effector::EffectorType::TouchSensor:
                name = "Tch";
                break;
        }
        vertexManager.addText(name, pos, 0.0015f, sf::Color::White, TextAlignment::Center, 2.0f);
        effectorIndex += 1;
    }
}

*/
