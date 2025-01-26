#include "geneticAlgorithm/lifeform.hpp"
#include <simulator/simulator.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

Cell::Cell(int lifeFormIdx, const float2& pos, float radius) {
    pointIdx = Simulator::get().getEnv().addPoint(Point(lifeFormIdx, pos.x, pos.y, radius));
    this->lifeFormIdx = lifeFormIdx;

    hue = Random::random(255.0f);
    saturation = Random::random(0.0f, 0.4f);
}

void Cell::render(VertexManager& vertexManager, vector<Point>& points) const {
    const Point pointObj = points[pointIdx];
    const float2 pos = pointObj.getPos();
    const float radius = pointObj.radius;
    const auto color = getColor();
    vertexManager.addCircle(pos, radius, color);
    vertexManager.addLine(pos, pos + vec(rotation) * radius, sf::Color::Blue, 0.4f);
    vertexManager.addLine(pos, pos + vec(rotation + divisionRotation) * radius, sf::Color::Red, 0.2f);
}


/*
void Cell::divide() {
    if (Simulator::get().getStep() - lastDivideTime < divisionFrequency) return;
    Point* motherPoint = lifeForm->getEnv()->getPoint(pointIdx);
    if (motherPoint->radius < 1.0f) return;
    dividing = false;

    lastDivideTime = Simulator::get().getStep();
    motherPoint->radius /= SQRT_2;
    float2 pos = motherPoint->pos + vec(rotation + divisionRotation) * CellLink::INITIAL_DISTANCE;
    auto* daughter = new Cell(lifeForm, this, pos, motherPoint->radius);
    Point* daughterPoint = lifeForm->getEnv()->getPoint(daughter->pointIdx);

    lifeForm->getEnv()->updatePoint(pointIdx, *motherPoint);
    lifeForm->getEnv()->updatePoint(daughter->pointIdx, *daughterPoint);
    lifeForm->addCell(daughter);

    // Create cell links
    auto* cellLink = new CellLink(lifeForm, this, daughter, CellLink::INITIAL_DISTANCE);
    lifeForm->addCellLink(cellLink);
    /*for(auto cell : lifeForm->cells) {
        if (cell == this || &cell == &daughter) continue;
        Point* otherPoint = lifeForm->getEnv()->getPoint(cell->pointIdx);
        float distance = daughterPoint->distanceTo(*otherPoint);
        // If overlapping another cell, create a link
        if (distance < daughterPoint->radius + otherPoint->radius) {
            auto* newCellLink = new CellLink(lifeForm, daughter, cell, CellLink::INITIAL_DISTANCE);
            lifeForm->addCellLink(newCellLink);
        }
    }
}

void Cell::fuse(Cell* other) {
    lifeForm->getEnv()->removePoint(other->pointIdx);
    // Move all cell links on other to this
    for(auto& cellLink : lifeForm->links) {
        if (cellLink->cell1 == other) {
            cellLink->moveCell1(this);
        }
        if (cellLink->cell2 == other) {
            cellLink->moveCell2(this);
        }
    }
    other->die();
}

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