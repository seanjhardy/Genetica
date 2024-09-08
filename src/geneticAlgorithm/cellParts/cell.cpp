#include <simulator/entities/lifeform.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

Cell::Cell(LifeForm* lifeForm, int point) : lifeForm(lifeForm), pointIdx(point) {
    color = sf::Color(150,100,100);
    childDistance = 0;
    growthProgress = 0;
    internalDivisionRotation = 0;
    externalDivisionRotation = 0;
}

void Cell::simulate(float dt) {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);

    lifeForm->energy -= 0.0001f * dt; // Add a small constant to prevent part spam (penalises lots of points)
    lifeForm->energy -= LifeForm::ENERGY_DECREASE_RATE * pointObj->mass * dt;

    // Grow cell
    if (growthProgress == 1) return;

    growthProgress += LifeForm::BUILD_RATE;
}

void Cell::updateGeneExpression() {
    for (auto& unit : lifeForm->grn.regulatoryUnits) {
        for (auto& promoter : unit.promoters) {
            float promoterActivity = promoter.calculateActivation(unit.genes,
                                                                  lifeForm->grn.promoterFactorAffinities,
                                                                  products);
        }
        float activity = unit.calculateActivation();
    }
};