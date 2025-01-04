#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

// Fish.hpp
#include <vector>
#include <cmath>
#include "modules/physics/point.hpp"
#include "simulator/environment.hpp"
#include "modules/utils/random.hpp"
#include "geneticAlgorithm/species.hpp"
#include <unordered_map>
#include <map>
#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"
#include "genome.hpp"
#include "geneticAlgorithm/cellParts/cell.hpp"
#include "geneticAlgorithm/cellParts/cellLink.hpp"
#include "geneticAlgorithm/cellParts/protein.hpp"

using namespace std;

class LifeForm {
public:
    static constexpr int GROWTH_INTERVAL = 200;
    static constexpr float BUILD_COST_SCALE = 0.00001f;
    static constexpr float BUILD_RATE = 50.0f;
    static constexpr float ENERGY_DECREASE_RATE = 0.0000001f;

    size_t idx;

    Species* species{};
    Environment* env;
    size_t genomeIdx;

    GeneRegulatoryNetwork grn;
    GPUVector<int> cells;
    GPUVector<int> links;

    GPUVector<Protein> inputs;
    GPUVector<Protein> outputs;

    double energy = 0;
    int numChildren = 0;
    int birthdate = 0;

    __host__ __device__ LifeForm(size_t id, Environment* env, Genome& genome, size_t genomeIdx, float2 pos);
    __host__ __device__ void init();

    //void simulate(float dt);
    //void render(VertexManager& viewer);

    int combine(LifeForm *partner);
    int clone(bool mutate);
    void kill();

    void grow(float dt);
    void addCell(const Cell& cell);
    void addCellLink(const CellLink& cellLink);
    void addInput(const Protein& protein);
    void addOutput(const Protein& protein);

    [[nodiscard]] Environment* getEnv() const;
    [[nodiscard]] Species* getSpecies() const;

};

#endif