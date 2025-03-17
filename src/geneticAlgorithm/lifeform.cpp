#include "geneticAlgorithm/lifeform.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"

using namespace std;

LifeForm::LifeForm(Genome& genome)
    : genome(genome) {
    birthdate = Simulator::get().getStep();
}

void LifeForm::update() {
    // Update the GRN every GROWTH_INTERVAL steps
    if (Simulator::get().getStep() - lastGrnUpdate > GROWTH_INTERVAL) {
        lastGrnUpdate = Simulator::get().getStep();
        updateGRN(*this,
            Simulator::get().getEnv().getPoints(),
            Simulator::get().getEnv().getCells(),
            Simulator::get().getEnv().getCellLinks());
    }
}

void LifeForm::render(VertexManager& vertexManager, vector<Cell>& hostCells, vector<CellLink>& cellLinks, vector<Point>& points) {
    // Render outline first
    for (int cell : cells.hostData()) {
        hostCells[cell].renderCellWalls(vertexManager, points );
    }
    for (int link : links.hostData()) {
        cellLinks[link].renderCellWalls(vertexManager, hostCells, points);
    }

    // Render body next
    for (int cell : cells.hostData()) {
        hostCells[cell].renderBody(vertexManager, points);
    }
    for (int link : links.hostData()) {
        cellLinks[link].renderBody(vertexManager, hostCells, points);
    }
}



void LifeForm::clone(bool mutate){
    // TODO: Implement this
}

void LifeForm::kill() {
    //TODO: Implement this
}

void LifeForm::addCell(size_t motherIdx, const Cell& mother, const Point& point) {
    float2 pos = point.getPos() + vec(mother.rotation + mother.divisionRotation) * point.radius;
    auto cell = Cell(idx, pos, point.radius);
    cell.generation = mother.generation;
    cell.hue = mother.hue;
    cell.saturation = mother.saturation;
    cell.luminosity = mother.luminosity;
    cell.products = mother.products.copy();
    cell.rotation = mother.rotation + mother.divisionRotation;

    cell.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(cell);
    cells.push(cell.idx);

    auto cellLink = CellLink(idx,
        cell.idx,
        motherIdx,
        cell.pointIdx,
        mother.pointIdx,
        point.radius*2,
        Random::random(0, 2 * M_PI), 0.01);

    const size_t linkIdx = Simulator::get().getEnv().nextCellLinkIdx();
    Simulator::get().getEnv().addCellLink(cellLink);
    links.push(linkIdx);
    grn.cellDistances.destroy();
    grn.cellDistances = StaticGPUVector<float>((cells.size() * (cells.size() - 1)) / 2);
}