#include "geneticAlgorithm/lifeform.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"
#include <geneticAlgorithm/sequencer.hpp>

using namespace std;

LifeForm::LifeForm(size_t idx, Genome& genome, float2 pos)
    : idx(idx), genome(genome) {
    birthdate = Simulator::get().getStep();
    sequence(*this, pos);
}

void LifeForm::update() {
    // Update the GRN every GROWTH_INTERVAL steps
    if (Simulator::get().getStep() - lastGrnUpdate > GRN_INTERVAL) {
        lastGrnUpdate = Simulator::get().getStep();
        updateGRN(*this,
            Simulator::get().getEnv().getPoints(),
            Simulator::get().getEnv().getCells(),
            Simulator::get().getEnv().getCellLinks());
    }
}

void LifeForm::render(VertexManager& vertexManager, vector<Cell>& hostCells, vector<CellLink>& cellLinks, vector<Point>& points) {
    // Render outline first
    for (int cell : cellIdxs) {
        hostCells[cell].renderCellWalls(vertexManager, points );
    }
    for (int link : linkIdxs) {
        cellLinks[link].renderCellWalls(vertexManager, hostCells, points);
    }

    // Render body next
    for (int cell : cellIdxs) {
        hostCells[cell].renderBody(vertexManager, points);
    }
    for (int link : linkIdxs) {
        cellLinks[link].renderBody(vertexManager, hostCells, points);
    }

    // Render details
    for (int link : linkIdxs) {
        cellLinks[link].renderDetails(vertexManager, hostCells, points);
    }
    for (int cell : cellIdxs) {
        hostCells[cell].renderDetails(vertexManager, points);
    }
}

void LifeForm::renderBlueprint(VertexManager& vertexManager, vector<Cell>& hostCells, vector<CellLink>& hostCellLinks) {
    float width = 300.0f;
    float height = 400.0f;
    float minX, minY, maxX, maxY = 0.0f;
    for (int cell : cellIdxs) {
        const Cell& hostCell = hostCells[cell];
        minX = min(minX, hostCell.bluePrintPosition.x);
        minY = min(minY, hostCell.bluePrintPosition.y);
        maxX = max(maxX, hostCell.bluePrintPosition.x);
        maxY = max(maxY, hostCell.bluePrintPosition.y);
    }
    float scaleX = width / (maxX - minX);
    float scaleY = height / (maxY - minY);
    float scale = min(scaleX, scaleY);
    float offsetX = (width - (maxX - minX) * scale) / 2;
    float offsetY = (height - (maxY - minY) * scale) / 2;
    for (int cell : cellIdxs) {
        const Cell& hostCell = hostCells[cell];
        const Point& point = Simulator::get().getEnv().getPoints().itemToHost(hostCell.pointIdx);
        float2 pos = point.getPos() * scale + make_float2(offsetX, offsetY);
        vertexManager.addCircle(pos, point.radius * scale, hostCell.getColor());
    }
    for (int link : linkIdxs) {
        const CellLink& hostCellLink = hostCellLinks[link];
        const Cell& cell1 = hostCells[hostCellLink.cellAIdx];
        const Cell& cell2 = hostCells[hostCellLink.cellBIdx];
        const Point& point1 = Simulator::get().getEnv().getPoints().itemToHost(cell1.pointIdx);
        const Point& point2 = Simulator::get().getEnv().getPoints().itemToHost(cell2.pointIdx);
        float2 pos1 = point1.getPos() * scale + make_float2(offsetX, offsetY);
        float2 pos2 = point2.getPos() * scale + make_float2(offsetX, offsetY);
        vertexManager.addLine(pos1, pos2, sf::Color::White, 0.5f);
    }
}




void LifeForm::clone(bool mutate){
    // TODO: Implement this
}

void LifeForm::kill() {
    //TODO: Implement this
}

void LifeForm::addCell(size_t motherIdx, const Cell& mother, const Point& motherPoint) {
    if (cellIdxs.size() >= MAX_CELLS) return;

    // Set position and velocity relative to mother
    float2 pos = motherPoint.getPos() + vec(motherPoint.angle + mother.divisionRotation) * motherPoint.radius * 2;
    double2 prevPos = motherPoint.prevPos + vec(motherPoint.angle + mother.divisionRotation) * motherPoint.radius * 2;
    Point daughterPoint = Point(idx, pos.x, pos.y, motherPoint.radius);
    daughterPoint.angle = motherPoint.angle + mother.divisionRotation + M_PI;
    daughterPoint.prevPos = prevPos;

    // Create the daughterCell
    auto daughter = Cell(idx, daughterPoint);
    daughter.generation = mother.generation + 1;
    daughter.hue = mother.hue;
    daughter.saturation = mother.saturation;
    daughter.luminosity = mother.luminosity;
    daughter.products = mother.products.copy();
    daughter.divisionRotation = mother.divisionRotation;
    daughter.lastDivideTime = mother.lastDivideTime;
    daughter.targetRadius = mother.targetRadius;
    daughter.energy = mother.energy;
    daughter.bluePrintPosition = mother.bluePrintPosition + vec(mother.blueprintAngle + mother.divisionRotation) * motherPoint.radius * 2;
    daughter.blueprintAngle = mother.blueprintAngle + mother.divisionRotation + M_PI;

    daughter.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(daughter);
    cellIdxs.push_back(daughter.idx);

    grn.cellDistances.destroy();
    grn.cellDistances = staticGPUVector<float>((cellIdxs.size() * (cellIdxs.size() - 1)) / 2);

    // Create a link to the mother cell
    auto cellLink = CellLink(idx,
        daughter.idx,
        motherIdx,
        mother.pointIdx,
        daughter.pointIdx,
        motherPoint.radius * 2,
        mother.targetRadius * 2,
        mother.divisionRotation,
        M_PI - mother.divisionRotation,
        0.01);

    const size_t linkIdx = Simulator::get().getEnv().nextCellLinkIdx();
    Simulator::get().getEnv().addCellLink(cellLink);
    linkIdxs.push_back(linkIdx);
    cellLinksMatrix.addEdge(daughter.idx, motherIdx);

    // Add links to all cells within the binding distance
    for (size_t otherCellIdx : cellIdxs) {
        if (otherCellIdx == daughter.idx || otherCellIdx == mother.idx) continue;
        if (cellLinksMatrix.isConnected(daughter.idx, otherCellIdx)) continue;

        const Cell other = Simulator::get().getEnv().getCells().itemToHost(otherCellIdx);
        const Point otherPoint = Simulator::get().getEnv().getPoints().itemToHost(other.pointIdx);
        float distance = daughterPoint.distanceTo(otherPoint);
        if (distance >= (otherPoint.radius + daughterPoint.radius) * 0.8) continue;
        float angleFromDaughter = daughterPoint.angleTo(otherPoint) - daughterPoint.angle;
        float angleFromOther = otherPoint.angleTo(daughterPoint) - otherPoint.angle;

        auto link = CellLink(idx,
            daughter.idx,
            otherCellIdx,
            daughter.pointIdx,
            other.pointIdx,
            distance,
            distance * 2,
            angleFromDaughter,
            angleFromOther, 0.01);
        const size_t newLinkIdx = Simulator::get().getEnv().nextCellLinkIdx();
        Simulator::get().getEnv().addCellLink(link);
        linkIdxs.push_back(newLinkIdx);
        cellLinksMatrix.addEdge(daughter.idx, otherCellIdx);
    }
}