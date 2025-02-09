#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

#include <geneticAlgorithm/species.hpp>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include "genome.hpp"
#include <modules/utils/structures/DynamicStableVector.hpp>

#include "cellParts/cellLink.hpp"

using namespace std;

class Environment;

class LifeForm {
public:
    static constexpr int GROWTH_INTERVAL = 200;
    static constexpr float BUILD_COST_SCALE = 0.00001f;
    static constexpr float BUILD_RATE = 50.0f;

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
    void render(VertexManager& vertexManager, vector<Cell>& cells, vector<CellLink>& cellLinks, vector<Point>& points);

    void combine(LifeForm *partner);
    void clone(bool mutate);
    void kill();

    void addCell(size_t motherIdx, const Cell& mother, const Point& point);

    [[nodiscard]] Species* getSpecies() const;
};

#endif