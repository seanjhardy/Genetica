#include "simulator/entities/entity.hpp"
#include "geneticAlgorithm/systems/morphology/geneticUnit.hpp"

class Cell {
public:
    LifeForm* lifeForm;
    int pointIdx;

    Cell* mother;
    std::unordered_map<Gene*, float> products;

    float rotation;
    float internalDivisionRotation;
    float externalDivisionRotation;

    sf::Color color;
    float childDistance;
    float growthProgress;

    Cell(LifeForm* lifeForm, Cell* mother, float2 pos, float radius = 0.1);

    void simulate(float dt);

    void freeze();
    void die();
    void divide();

    void updateGeneExpression();
};