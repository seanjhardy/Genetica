#include "geneticAlgorithm/lifeform.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"
#include <geneticAlgorithm/sequencer.hpp>
#include <modules/utils/GPU/mathUtils.hpp>
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

void LifeForm::render(VertexManager& vertexManager, vector<Cell>& hostCells, vector<CellLink>& cellLinks,
                      vector<Point>& points) {
    // Render outline first
    for (int cell : cellIdxs) {
        hostCells[cell].renderCellWalls(vertexManager, points);
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
    float width = 400.0f;
    float height = 400.0f;
    float minX = INFINITY, minY = INFINITY, maxX = -INFINITY, maxY = -INFINITY;
    for (int cell : cellIdxs) {
        const Cell& hostCell = hostCells[cell];
        const Point& blueprintPos = blueprintPoints.itemToHost(hostCell.blueprintPointIdx);
        minX = min(minX, blueprintPos.getPos().x);
        minY = min(minY, blueprintPos.getPos().y);
        maxX = max(maxX, blueprintPos.getPos().x);
        maxY = max(maxY, blueprintPos.getPos().y);
    }
    float scaleX = width / (maxX - minX);
    float scaleY = height / (maxY - minY);
    float scale = min(scaleX, scaleY);
    float offsetX = (width - (maxX - minX) * scale) / 2;
    float offsetY = (height - (maxY - minY) * scale) / 2;
    for (int cell : cellIdxs) {
        const Cell& hostCell = hostCells[cell];
        const Point& blueprintPoint = blueprintPoints.itemToHost(hostCell.blueprintPointIdx);
        float2 pos = blueprintPoint.getPos() * scale + make_float2(offsetX, offsetY);
        print(pos, scale);
        vertexManager.addCircle(pos, max(blueprintPoint.radius * scale, 1.0), hostCell.getColor());
    }
    for (int link : linkIdxs) {
        const CellLink& hostCellLink = hostCellLinks[link];
        const Cell& cell1 = hostCells[hostCellLink.cellAIdx];
        const Cell& cell2 = hostCells[hostCellLink.cellBIdx];
        const Point& blueprintPoint1 = blueprintPoints.itemToHost(cell1.blueprintPointIdx);
        const Point& blueprintPoint2 = blueprintPoints.itemToHost(cell2.blueprintPointIdx);
        float2 pos1 = blueprintPoint1.getPos() * scale + make_float2(offsetX, offsetY);
        float2 pos2 = blueprintPoint2.getPos() * scale + make_float2(offsetX, offsetY);
        const sf::Color cell1Color = brightness(cell1.getColor(), 0.6);
        const sf::Color cell2Color = brightness(cell2.getColor(), 0.6);
        const float angle = atan2(pos2.y - pos1.y, pos2.x - pos1.x);
        const float angle1 = angle + M_PI_HALF;
        const float angle2 = angle - M_PI_HALF;
        float cell1Thickness = max(cell1.membraneThickness * scale, 1.0f);
        float cell2Thickness = max(cell2.membraneThickness * scale, 1.0f);
        float2 v1 = pos1 + make_float2(cos(angle1), sin(angle1)) * (blueprintPoint1.radius * scale + cell1Thickness);
        float2 v2 = pos2 + make_float2(cos(angle1), sin(angle1)) * (blueprintPoint2.radius * scale + cell2Thickness);
        float2 v3 = pos2 + make_float2(cos(angle2), sin(angle2)) * (blueprintPoint2.radius * scale + cell2Thickness);
        float2 v4 = pos1 + make_float2(cos(angle2), sin(angle2)) * (blueprintPoint1.radius * scale + cell1Thickness);
        vertexManager.addPolygon(std::vector<VertexManager::Vertex>({
            {v1, cell1Color},
            {v2, cell2Color},
            {v3, cell2Color},

            {v3, cell2Color},
            {v4, cell1Color},
            {v1, cell1Color}
        }));
    }
}


void LifeForm::clone(bool mutate) {
    // TODO: Implement this
}

void LifeForm::kill() {
    //TODO: Implement this
}

void LifeForm::addCell(size_t motherIdx, const Cell& mother) {
    if (cellIdxs.size() >= MAX_CELLS) return;

    // Set position and velocity relative to mother
    Point motherPoint = Simulator::get().getEnv().getPoints().itemToHost(mother.pointIdx);
    float2 pos = motherPoint.getPos() + vec(motherPoint.angle + mother.divisionRotation) * motherPoint.radius * 2;
    double2 prevPos = motherPoint.prevPos + vec(motherPoint.angle + mother.divisionRotation) * motherPoint.radius * 2;
    Point daughterPoint = Point(idx, pos.x, pos.y, motherPoint.radius);
    daughterPoint.angle = motherPoint.angle + mother.divisionRotation + M_PI;
    daughterPoint.prevPos = prevPos;

    Point motherBlueprintPoint = blueprintPoints.itemToHost(mother.blueprintPointIdx);
    float2 blueprintPos = motherPoint.getPos() + vec(motherPoint.angle + mother.divisionRotation) * motherBlueprintPoint
        .radius * 2;
    Point daughterBlueprintPoint = Point(idx, blueprintPos.x, blueprintPos.y, motherBlueprintPoint.radius);
    daughterBlueprintPoint.angle = motherBlueprintPoint.angle + mother.divisionRotation + M_PI;
    daughterBlueprintPoint.prevPos = prevPos;

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
    daughter.blueprintPointIdx = blueprintPoints.getNextIndex();

    daughter.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(daughter);
    blueprintPoints.push(daughterBlueprintPoint);
    cellIdxs.push_back(daughter.idx);

    grn.cellDistances.destroy();
    grn.cellDistances = StaticGPUVector<float>((cellIdxs.size() * (cellIdxs.size() - 1)) / 2);

    // Create a link to the mother cell
    auto cellLink = CellLink(daughter.idx,
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
        const Point otherBlueprintPoint = blueprintPoints.itemToHost(other.blueprintPointIdx);

        float distance = distanceBetween(otherBlueprintPoint.getPos(), blueprintPos);
        if (distance >= (otherBlueprintPoint.radius + daughterPoint.radius) * 0.8) continue;

        float angleFromDaughter = angleBetween(blueprintPos, otherBlueprintPoint.getPos()) - daughterPoint.angle;
        float angleFromOther = angleBetween(otherBlueprintPoint.getPos(), otherBlueprintPoint.getPos()) -
            otherBlueprintPoint.angle;

        auto link = CellLink(daughter.idx,
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
