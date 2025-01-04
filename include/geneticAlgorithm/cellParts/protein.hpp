#include <vector>
#include "geneticAlgorithm/lifeform.hpp"

class Protein {
public:
    LifeForm *lifeForm;
    Cell* parent;
    std::vector<float> parameters;
    float size;

    Protein(LifeForm *lifeform, Cell* parent) : lifeForm(lifeform), parent(parent) {}

    virtual void simulate(float dt);
    virtual void render(VertexManager& vertexManager);

    float getParameter(int index) {
        return parameters[index];
    }
};