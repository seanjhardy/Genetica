#ifndef LIFEFORM_HPP
#define LIFEFORM_HPP

#include <geneticAlgorithm/species.hpp>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include "genome.hpp"
#include "modules/cuda/structures/CGPUVector.hpp"
#include "cellParts/cellLink.hpp"
#include <modules/utils/structures/AdjacencyMatrix.hpp>

#define MAX_CELLS 32
#define GRN_INTERVAL 5000

using namespace std;

class Environment;

class LifeForm {
public:
    size_t idx;

    Genome genome;

    GeneRegulatoryNetwork grn;
    int lastGrnUpdate = 0;
    vector<size_t> cellIdxs = vector<size_t>(0);
    vector<size_t> linkIdxs = vector<size_t>(0);
    AdjacencyMatrix<size_t> cellLinksMatrix = AdjacencyMatrix<size_t>();

    int numChildren = 0;
    int birthdate = 0;

    LifeForm() = default;
    LifeForm(size_t idx, Genome& genome, float2 pos);

    void update();
    void render(VertexManager& vertexManager, vector<Cell>& cells, vector<CellLink>& cellLinks, vector<Point>& points);
    void renderBlueprint(VertexManager& vertexManager, vector<Cell>& cells, vector<CellLink>& cellLinks);

    void combine(LifeForm *partner);
    void clone(bool mutate);
    void kill();

    void addCell(size_t motherIdx, const Cell& mother, const Point& point);

    [[nodiscard]] Species* getSpecies() const;
};

#endif