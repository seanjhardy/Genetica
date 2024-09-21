#ifndef CELL
#define CELL

#include <simulator/entities/entity.hpp>
#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>
#include <geneticAlgorithm/systems/morphology/gene.hpp>
#include <geneticAlgorithm/systems/morphology/promoter.hpp>

class LifeForm;

class Cell {
public:
    LifeForm* lifeForm;
    Cell* mother;
    int pointIdx;
    int generation = 0.0f;

    std::unordered_map<Gene*, float> products;

    float rotation = 0.0f;
    float divisionRotation = 0.0f;
    bool frozen = false;

    sf::Color color = sf::Color(200, 150, 150);

    Cell(LifeForm* lifeForm, Cell* mother, float2 pos, float radius = 0.1);

    void simulate(float dt);
    void render(VertexManager& vertexManager);
    void adjustSize(float sizeChange);

    void die();
    void divide();
    void fuse(Cell* other);
};

#endif