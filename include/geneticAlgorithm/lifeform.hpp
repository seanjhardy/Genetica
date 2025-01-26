#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

#include <cmath>
#include <geneticAlgorithm/species.hpp>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include "genome.hpp"
#include <geneticAlgorithm/cellParts/protein.hpp>
#include <modules/utils/structures/DynamicStableVector.hpp>

using namespace std;

class Environment;

class LifeForm {
public:
    static constexpr int GROWTH_INTERVAL = 200;
    static constexpr float BUILD_COST_SCALE = 0.00001f;
    static constexpr float BUILD_RATE = 50.0f;
    static constexpr float ENERGY_DECREASE_RATE = 0.0000001f;

    struct LfUpdateData {
        struct NEW_CELL {
            int motherIdx{};
            int motherPointIdx{};
            float2 pos{};
            float radius{};
            float rotation{};
            float divisionRotation{};

            int generation{};
            float hue{};
            float saturation{};
            float luminosity{};

            //StaticGPUVector<float> products{};
        } newCell;
        bool cellAdded;
        float energyChange;
    };

    size_t idx;

    Genome genome;

    GeneRegulatoryNetwork grn;
    int lastGrnUpdate = 0;
    DynamicStableVector<int> cells;
    DynamicStableVector<int> links;

    //DynamicStableVector<Protein> inputs;
    //DynamicStableVector<Protein> outputs;

    double energy = 0;
    int numChildren = 0;
    int birthdate = 0;

    LifeForm() = default;
    LifeForm(Genome& genome);
    void init();

    void update();

    void combine(LifeForm *partner);
    void clone(bool mutate);
    void kill();

    void addCell(const LfUpdateData::NEW_CELL& newCell);

    [[nodiscard]] Species* getSpecies() const;
};

#endif