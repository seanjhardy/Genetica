#include <simulator/entities/lifeform.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

Cell::Cell(LifeForm* lifeForm, Cell* mother, float2 pos, float radius)
: lifeForm(lifeForm), mother(mother) {
    pointIdx = lifeForm->getEnv()->addPoint(lifeForm->entityID,
                                            pos.x, pos.y, radius);
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
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);

    // Define external factors for this cell
    float externalFactors[10];
    externalFactors[0] = distanceBetween(pointObj->pos, {lifeForm->pos.x, lifeForm->pos.y});

    for (auto& unit : lifeForm->grn.regulatoryUnits) {
        std::map<Promoter*, float> promoterActivities;
        for (auto& promoter : unit.promoters) {
            float promoterActivity = promoter.calculateActivity(unit.genes,
                                                                  lifeForm->grn.promoterFactorAffinities,
                                                                  products);
            promoterActivities.insert({&promoter, promoterActivity});
        }
        float activity = unit.calculateActivation(promoterActivities);
    }
};

void Cell::divide() {
    Cell daughter = *this;
    daughter.products = products;
    // TODO: reset value of division products (?)
    daughter.internalDivisionRotation = internalDivisionRotation;
    Point* motherPoint = lifeForm->getEnv()->getPoint(pointIdx);
    Point* daughterPoint = lifeForm->getEnv()->getPoint(daughter.pointIdx);
    daughterPoint->pos = motherPoint->pos
      + vec(externalDivisionRotation) * childDistance;
    lifeForm->cells.push_back(&daughter);
}