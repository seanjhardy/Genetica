#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

// Fish.hpp
#include "vector"
#include "cmath"
#include "modules/verlet/point.hpp"
#include "geneticAlgorithm/environment.hpp"
#include "modules/noise/random.hpp"
#include "geneticAlgorithm/cellParts/cellPartInstance.hpp"
#include "geneticAlgorithm/cellParts/segmentInstance.hpp"
#include "geneticAlgorithm/cellParts/cellPartType.hpp"
#include "geneticAlgorithm/cellParts/cellPartSchematic.hpp"
#include "geneticAlgorithm/species.hpp"
#include "unordered_map"
#include "map"

using namespace std;

class LifeForm {
public:
    enum class ReproductionType {
        SEXUAL, ASEXUAL
    };
    enum class SymmetryType {
        NONE, LOCAL, GLOBAL, RADIAL
    };

    std::unordered_map<int, std::shared_ptr<CellPartType>> cellParts{};
    vector<CellPartInstance*> cellPartInstances;
    vector<CellPartInstance*> inputs;
    vector<CellPartInstance*> outputs;
    std::multimap<int, std::pair<SegmentInstance*, CellPartSchematic*>> buildQueue;
    std::multimap<int, CellPartInstance*> buildsInProgress;
    SegmentInstance* head{};

    static int HEADER_SIZE, CELL_DATA_SIZE, GROWTH_INTERVAL;
    static float BUILD_COST_SCALE, BUILD_RATE, ENERGY_DECREASE_RATE;
    float2 pos{};

    int id;
    unordered_map<int, string> genome;
    Species* species{};
    Environment* env;

    //Meta variables
    ReproductionType reproductionType{};
    SymmetryType symmetryType{};
    float energy = 0, growthEnergy = 0,
    currentGrowthEnergy = 0, growthRate = 0,
    regenerationFraction = 0, childEnergy = 0;
    int numChildren = 0;
    int lastGrow = 0;
    float size = 0;

    std::vector<CellPartInstance> body{};


    LifeForm(Environment* env, float2 pos, const std::unordered_map<int, string>& genome);

    void simulate(float dt);
    void render(VertexManager& viewer);
    void mutate();
    LifeForm& combine(LifeForm *partner);
    LifeForm& clone(bool mutate);
    void init();

    void grow(float dt);
    void addCellPartInstance(CellPartInstance* cellPartInstance);

    Environment* getEnv();

    unordered_map<int, string> getGenome();
    void setGenome(const unordered_map<int, string>& genomeArr);

    int getID();
    void setID(int ID);

    Species* getSpecies();

};

#endif