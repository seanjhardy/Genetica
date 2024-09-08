#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

// Fish.hpp
#include <vector>
#include <cmath>
#include <modules/physics/point.hpp>
#include "simulator/environment.hpp"
#include "modules/utils/random.hpp"
#include "geneticAlgorithm/cellParts/cellPartInstance.hpp"
#include "geneticAlgorithm/cellParts/segmentInstance.hpp"
#include "geneticAlgorithm/cellParts/cellPartType.hpp"
#include "geneticAlgorithm/cellParts/cellPartSchematic.hpp"
#include <simulator/entities/entity.hpp>
#include "entity.hpp"
#include "geneticAlgorithm/species.hpp"
#include <unordered_map>
#include <map>
#include <geneticAlgorithm/cellParts/cell.hpp>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <geneticAlgorithm/genome.hpp>

using namespace std;

class LifeForm : public Entity {
public:
    Species* species{};
    Environment* env;
    Genome genome;

    GeneRegulatoryNetwork grn;
    Cell* head{};
    std::vector<Cell*> cells;

    vector<CellPartInstance*> inputs;
    vector<CellPartInstance*> outputs;

    static int GROWTH_INTERVAL;
    static float BUILD_COST_SCALE, BUILD_RATE, ENERGY_DECREASE_RATE;

    float energy = 0;
    int numChildren = 0;
    int birthdate = 0;

    LifeForm(Environment* env, float2 pos, Genome& genome);

    void simulate(float dt) override;
    void render(VertexManager& viewer) override;

    LifeForm& combine(LifeForm *partner);
    LifeForm& clone(bool mutate);
    void kill();
    void init();

    void grow(float dt);
    void addInput(CellPartInstance* cellPartInstance);
    void addOutput(CellPartInstance* cellPartInstance);

    [[nodiscard]] Environment* getEnv() const;

    [[nodiscard]] Genome getGenome() const;
    void setGenome(const Genome& genomeArr);

    [[nodiscard]] Species* getSpecies() const;

};

#endif