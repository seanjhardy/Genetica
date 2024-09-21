#include <simulator/entities/lifeform.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/utils/print.hpp>

Cell::Cell(LifeForm* lifeForm, Cell* mother, float2 pos, float radius)
: lifeForm(lifeForm), mother(mother) {
    pointIdx = lifeForm->getEnv()->addPoint(lifeForm->entityID,
                                            pos.x, pos.y, radius);
    if (mother == nullptr) return;
    generation = mother->generation + 1;
    color = mother->color;
    products = mother->products;
    rotation = mother->rotation + mother->divisionRotation;
}


void Cell::simulate(float dt) {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);

    lifeForm->energy -= 0.0001f * dt; // Add a small constant to prevent part spam (penalises lots of points)
    lifeForm->energy -= LifeForm::ENERGY_DECREASE_RATE * pointObj->mass * dt;
}

void Cell::render(VertexManager& vertexManager) {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);
    float2 pos = pointObj->pos;
    float radius = pointObj->mass;

    vertexManager.addCircle(pos, radius, color);
    vertexManager.addLine(pos, pos + vec(rotation) * radius, sf::Color::White);
    vertexManager.addLine(pos, pos + vec(rotation + divisionRotation) * radius, sf::Color::Red);
}

void Cell::adjustSize(float sizeChange) {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);
    pointObj->mass += sizeChange;
    if (pointObj->mass < 0) {
        die();
    } else {
        lifeForm->getEnv()->updatePoint(pointIdx, *pointObj);
    }
}

void Cell::divide() {
    auto* daughter = new Cell(*this);
    Point* motherPoint = lifeForm->getEnv()->getPoint(pointIdx);
    Point* daughterPoint = lifeForm->getEnv()->getPoint(daughter->pointIdx);

    daughterPoint->setPos(motherPoint->pos + vec(rotation + divisionRotation) * CellLink::INITIAL_DISTANCE);

    lifeForm->addCell(daughter);

    // Create cell links
    auto* cellLink = new CellLink(lifeForm, this, daughter, CellLink::INITIAL_DISTANCE);
    lifeForm->addCellLink(cellLink);
    for(auto& cell : lifeForm->cells) {
        if (cell.get() == this || cell.get() == daughter) continue;
        Point* otherPoint = lifeForm->getEnv()->getPoint(cell->pointIdx);
        float distance = daughterPoint->distanceTo(*otherPoint);
        // If overlapping another cell, create a link
        if (distance < daughterPoint->mass + otherPoint->mass) {
            auto* newCellLink = new CellLink(lifeForm, daughter, cell.get(), CellLink::INITIAL_DISTANCE);
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
                                         [this](const std::unique_ptr<Cell>& cell) {
                                             return cell.get() == this;
                                         }), lifeForm->cells.end());
    // Destroy all links
    lifeForm->links.erase(std::remove_if(lifeForm->links.begin(), lifeForm->links.end(),
                                         [this](const std::unique_ptr<CellLink>& cellLink) {
                                             return cellLink->cell1 == this || cellLink->cell2 == this;
                                         }), lifeForm->links.end());
    // Finally delete the memory itself
    delete this;
}