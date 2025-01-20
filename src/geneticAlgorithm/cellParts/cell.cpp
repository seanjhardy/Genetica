#include "geneticAlgorithm/lifeform.hpp"
#include <simulator/simulator.hpp>

Cell::Cell(LifeForm& lifeForm, Cell* mother, const float2& pos, float radius) {
    pointIdx = Simulator::get().getEnv().addPoint(Point(lifeForm.idx, pos.x, pos.y, radius));
    lifeFormIdx = lifeForm.idx;

    if (mother == nullptr) return;

    generation = mother->generation + 1;
    //color = mother->color;
    products = mother->products;
    rotation = mother->rotation + mother->divisionRotation;
}

void Cell::render(VertexManager& vertexManager, vector<Point>& points) const {
    const Point pointObj = points[pointIdx];
    const float2 pos = pointObj.pos;
    const float radius = pointObj.radius;
    const auto color = sf::Color(HSVtoRGB(hue, saturation, 127 + luminosity));
    vertexManager.addCircle(pos, radius, color);
    vertexManager.addLine(pos, pos + vec(rotation) * radius, sf::Color::Blue, 0.4f);
    vertexManager.addLine(pos, pos + vec(rotation + divisionRotation) * radius, sf::Color::Red, 0.2f);
}


/*
void Cell::simulate(float dt) {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);
    lifeForm->energy -= LifeForm::ENERGY_DECREASE_RATE * (1.0f + pointObj->radius) * dt;
}*/

/*
void Cell::render(VertexManager& vertexManager) const {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);
    float2 pos = pointObj->pos;
    float radius = pointObj->radius;
    vertexManager.addCircle(pos, radius, color);
    vertexManager.addLine(pos, pos + vec(rotation) * radius, sf::Color::White, 0.2f);
    vertexManager.addLine(pos, pos + vec(rotation + divisionRotation) * radius, sf::Color::Red, 0.1f);
}*/
/*
void Cell::adjustSize(float sizeChange) const {
    Point* pointObj = lifeForm->getEnv()->getPoint(pointIdx);
    lifeForm->energy -= (2 * M_PI * pointObj->radius * sizeChange + M_PI * pow(sizeChange, 2));
    pointObj->radius = max(pointObj->radius + sizeChange, 0.1f);
    lifeForm->getEnv()->updatePoint(pointIdx, *pointObj);
}

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

void Cell::updateHue(PIGMENT pigment, float amount) {
    float target_hue = pigment == PIGMENT::Red ? 0.0f :
                       pigment == PIGMENT::Green ? 120.0f :
                       pigment == PIGMENT::Blue ? 240.0f : 360.0f;
    float delta_hue = int(target_hue - hue) % 360;

    if (delta_hue > 180) {
        delta_hue -= 360;
    }

    saturation = clamp(0.0, saturation + amount * 0.1f, 1.0f);
    float new_hue = hue + amount * (delta_hue != 0 ? delta_hue / abs(delta_hue) : 0);
    color = sf::Color(HSVtoRGB(new_hue, saturation, 127 + luminosity));
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