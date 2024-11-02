#ifndef CELL
#define CELL

#include <simulator/entities/entity.hpp>
#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>
#include <geneticAlgorithm/systems/morphology/gene.hpp>
#include <geneticAlgorithm/systems/morphology/promoter.hpp>
#include <modules/utils/GUIUtils.hpp>
#include <modules/utils/random.hpp>

class LifeForm;

class Cell {
public:
    LifeForm* lifeForm;
    Cell* mother;
    int pointIdx;
    int generation = 0.0f;
    int lastDivideTime = 0;
    int divisionFrequency = 10 + Random::random(20);

    std::unordered_map<Gene*, float> products;

    float rotation = 0.0f;
    float divisionRotation = 0.0f;
    bool frozen = false;
    bool dividing = false;

    float hue = 200.0f, saturation = 0.0f, luminosity = 0.0f;
    sf::Color color = sf::Color(HSVtoRGB(hue, saturation, 127 + luminosity));

    Cell(LifeForm* lifeForm, Cell* mother, float2 pos, float radius = 0.1);

    void simulate(float dt);
    void render(VertexManager& vertexManager) const;
    void adjustSize(float sizeChange) const;

    void die();
    void divide();
    void fuse(Cell* other);
    void updateHue(PIGMENT pigment, float amount);
};

#endif