#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

// Fish.hpp
#include <vector>
#include <cmath>
#include <modules/physics/point.hpp>
#include "simulator/environment.hpp"
#include "modules/utils/random.hpp"
#include <simulator/entities/entity.hpp>
#include "entity.hpp"
#include "geneticAlgorithm/species.hpp"
#include <unordered_map>
#include <map>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <geneticAlgorithm/genome.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include <geneticAlgorithm/cellParts/protein.hpp>

using namespace std;

class LifeForm : public Entity {
public:
    static constexpr int GROWTH_INTERVAL = 50;
    static constexpr float BUILD_COST_SCALE = 0.00001f;
    static constexpr float BUILD_RATE = 50.0f;
    static constexpr float ENERGY_DECREASE_RATE = 0.0000001;

    Species* species{};
    Environment* env;
    Genome genome;

    GeneRegulatoryNetwork grn;
    Cell* head{};
    std::vector<std::unique_ptr<Cell>> cells;
    std::vector<std::unique_ptr<CellLink>> links;

    vector<Protein*> inputs;
    vector<Protein*> outputs;

    float energy = 0;
    int numChildren = 0;
    int birthdate = 0;

    LifeForm(Environment* env, float2 pos, Genome& genome);
    void init();

    void simulate(float dt) override;
    void render(VertexManager& viewer) override;

    LifeForm& combine(LifeForm *partner);
    LifeForm& clone(bool mutate);
    void kill();

    void grow(float dt);
    void addCell(Cell* cell);
    void addCellLink(CellLink* cellLink);
    void addInput(Protein* protein);
    void addOutput(Protein* protein);

    [[nodiscard]] Environment* getEnv() const;
    [[nodiscard]] Species* getSpecies() const;

};

#endif