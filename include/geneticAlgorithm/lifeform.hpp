#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

#include <cmath>
#include <geneticAlgorithm/species.hpp>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include "genome.hpp"
#include <geneticAlgorithm/cellParts/protein.hpp>

using namespace std;

class Environment;

class LifeForm {
public:
    static constexpr int GROWTH_INTERVAL = 200;
    static constexpr float BUILD_COST_SCALE = 0.00001f;
    static constexpr float BUILD_RATE = 50.0f;
    static constexpr float ENERGY_DECREASE_RATE = 0.0000001f;

    size_t idx;

    Genome genome;

    GeneRegulatoryNetwork grn;
    int lastGrnUpdate = 0;
    GPUVector<int> cells;
    GPUVector<int> links;

    GPUVector<Protein> inputs;
    GPUVector<Protein> outputs;

    double energy = 0;
    int numChildren = 0;
    int birthdate = 0;

    LifeForm(Genome& genome);
    void init();

    void update();

    void combine(LifeForm *partner);
    void clone(bool mutate);
    void kill();

    void grow(float dt);
    void addInput(const Protein& protein);
    void addOutput(const Protein& protein);

    [[nodiscard]] Species* getSpecies() const;

};

#endif