#include "geneticAlgorithm/lifeform.hpp"
#include <simulator/simulator.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

Cell::Cell(int lifeFormIdx, Point& point) : lifeFormIdx(lifeFormIdx), products(0) {
    pointIdx = Simulator::get().getEnv().addPoint(point);

    hue = 100.0f; //Random::random(255.0f);
    saturation = 0.2f; //Random::random(0.0f, 0.4f);
}

void Cell::renderBody(VertexManager& vertexManager, vector<Point>& points) const {
    const Point point = points[pointIdx];
    const float2 pos = point.getPos();
    const float radius = point.radius;
    const auto color = getColor();
    vertexManager.addCircle(pos, radius, color);
}

void Cell::renderDetails(VertexManager& vertexManager, vector<Point>& points) const {
    const Point point = points[pointIdx];
    const float2 pos = point.getPos();
    const float radius = point.radius;
    const auto color = getColor();
    float nucleusSize = point.radius / 3;
    vertexManager.addCircle(point.getPos(), nucleusSize, brightness(color, 1.2), 10);
    vertexManager.addLine(pos, pos + vec(point.angle + divisionRotation) * radius, sf::Color::Red, 0.2f);
}

void Cell::renderCellWalls(VertexManager& vertexManager, vector<Point>& points) const {
    const Point pointObj = points[pointIdx];
    const float2 pos = pointObj.getPos();
    const float radius = pointObj.radius;
    if (vertexManager.getSizeInView(radius) < 5) return;
    const auto color = brightness(getColor(), 0.6);
    vertexManager.addCircle(pos, radius + membraneThickness, color);
}

/*
void Cell::die() {
    // Remove this point from environment
    lifeForm->getEnv()->removePoint(pointIdx);
    // Remove this cell from lifeform
    lifeForm->cells.erase(std::remove_if(lifeForm->cells.begin(), lifeForm->cells.end(),
                                         [this](const Cell* cell) {
                                             return cell == this;
                                         }), lifeForm->cells.end());
    // Destroy all links
    lifeForm->links.erase(std::remove_if(lifeForm->links.begin(), lifeForm->links.end(),
                                         [this](const std::unique_ptr<CellLink>& cellLink) {
                                             return cellLink->cell1 == this || cellLink->cell2 == this;
                                         }), lifeForm->links.end());
    // Finally delete the memory itself
    delete this;
}*/
