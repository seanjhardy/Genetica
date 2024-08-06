#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

// Fish.hpp
#include "vector"
#include "cmath"
#include <modules/verlet/point.hpp>
#include <geneticAlgorithm/environment.hpp>
#include <geneticAlgorithm/individual.hpp>
#include "hyperLife.hpp"
#include "unordered_map"
#include "map"

class CellPartInstance;
class CellPartType;
class CellPartSchematic;

class LifeForm : public Individual {
public:
    enum class ReproductionType {
        SEXUAL, ASEXUAL
    };
    enum class SymmetryType {
        NONE, LOCAL, GLOBAL, RADIAL
    };

    std::unordered_map<int, CellPartType&> cellParts{};
    vector<CellPartInstance*> cellPartInstances;
    vector<CellPartInstance*> inputs;
    vector<CellPartInstance*> outputs;
    std::multimap<int, std::pair<SegmentInstance*, CellPartSchematic*>> buildQueue;
    SegmentInstance* head{};

    static int HEADER_SIZE, CELL_DATA_SIZE;
    static float BUILD_COST_SCALE;
    float2 pos{};

    //Meta variables
    ReproductionType reproductionType{};
    SymmetryType symmetryType{};
    float energy = 0, growthEnergy = 0,
    currentGrowthEnergy = 0, growthPriority = 0,
    regenerationFraction = 0, childEnergy = 0;
    int numChildren = 0;
    float size = 0;

    int randomTime = int(Random::random(0, 50));


    LifeForm(HyperLife* env, float2 pos, std::unordered_map<int, string> genome);

    void simulate(float dt) override;
    void render(VertexManager& viewer) override;
    void mutate() override;
    Individual& combine(Individual *partner) override;
    Individual& clone(bool mutate) override;
    void init() override;

    void grow(float dt);

    HyperLife* getEnv() override {
        return dynamic_cast<HyperLife *>(Individual::getEnv());
    }

    void addCellPartInstance(CellPartInstance* cellPartInstance);

    std::vector<CellPartInstance> body{};
};

#endif