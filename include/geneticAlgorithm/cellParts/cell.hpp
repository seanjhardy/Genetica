#include "simulator/entities/entity.hpp"
#include "geneticAlgorithm/systems/morphology/geneticUnit.hpp"

class Cell {
public:
    LifeForm* lifeForm;
    int pointIdx;

    std::unordered_map<Gene*, float> products;
    float internalDivisionRotation;
    float externalDivisionRotation;
    sf::Color color;
    float childDistance;
    float growthProgress;

    Cell(LifeForm* lifeForm, int point);

    void simulate(float dt);

    void freeze();
    void die();
    void divide();

    void updateGeneExpression();
};