#include <simulator/environment.hpp>
#include "geneticAlgorithm/lifeform.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/cuda/updatePoints.hpp>
#include <simulator/simulator.hpp>
#include <modules/cuda/findNearest.hpp>
#include <modules/utils/print.hpp>
#include <modules/cuda/updateCells.hpp>

Environment::Environment(sf::FloatRect bounds) :
    initialBounds(bounds),
    bounds(bounds), fluidSimulator(
      0.1, (size_t)bounds.width, (size_t)bounds.height,
      {}) {
    planet = &Planet::planets["Delune"];
}

void Environment::simulate(float deltaTime) {
    if (fluidEnabled && Simulator::get().getStep() % 500 == 0) {
        fluidSimulator.update(0.02);
    }
    planet->update();
    updatePoints(points, cellLinks, bounds, deltaTime);
    updateCells(getGA().getPopulation(), cells, points);
};

void Environment::render(VertexManager& vertexManager) {
    planet->render(vertexManager);

    if (fluidEnabled) {
        fluidSimulator.render(vertexManager, bounds.hostData());
    }

    // Draw a grid of columns and rows inside of bounds:
    if (vertexManager.getSizeInView(1) > 0.2 && gridLinesVisible) {
        int opacity = (int)clamp(10.0f, vertexManager.camera->getZoom() * 10.0f, 30.0f);
        float thickness = clamp(1.0f, 1.0f / vertexManager.camera->getZoom(), 5.0f);
        sf::Color gridColor = sf::Color(255, 255, 255, opacity);
        for (int i = 0; i < bounds.hostData().width + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left + i, bounds.hostData().top},
                                  {bounds.hostData().left + i, bounds.hostData().top + bounds.hostData().height}, gridColor,
                                  thickness);
        }
        for (int i = 0; i < bounds.hostData().height + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left, bounds.hostData().top + i},
                                  {bounds.hostData().left + bounds.hostData().width, bounds.hostData().top + i}, gridColor,
                                  thickness);
        }
    }

    dragHandler.render(vertexManager, bounds.hostData());
};

std::pair<bool, int> Environment::handleEvent(const sf::Event& event, const sf::Vector2f mousePos) {
    dragHandler.handleEvent(event);

    /*if (!dragHandler.isDragging() && planet->getBounds() != bounds.hostData()) {
        planet->setBounds(bounds.hostData());
    }*/

    if (event.type == sf::Event::MouseButtonReleased &&
        event.mouseButton.button == sf::Mouse::Left) {
        heldPoint = -1;
        return {false, -1};
    }

    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            // Find the nearest point in the quadtree (within a radius of 20)
            std::pair<int, float> nearestPoint = findNearest(points, mousePos.x, mousePos.y, 20);

            if (nearestPoint.first != -1) {
                if (points.size() > nearestPoint.first) {
                    heldPoint = nearestPoint.first;
                    int newSelectedEntityId = movePoint(points, heldPoint, mousePos);
                    return {true, newSelectedEntityId};
                }
            } else {
                heldPoint = -1;
                return {true, -1};
            }
        }
    }

    return {false, -1};
};

void Environment::update(const sf::Vector2f& worldCoords, float zoom, bool UIHovered) {
    if (heldPoint != -1) {
        movePoint(points, heldPoint, worldCoords);
    }

    if (!UIHovered) {
        sf::FloatRect deltaBounds = dragHandler.update(worldCoords, bounds.hostData(), 15.0f / zoom);

        if (deltaBounds.left != 0 || deltaBounds.top != 0 || deltaBounds.width != 0 || deltaBounds.height != 0) {
            bounds = {bounds.hostData().left + deltaBounds.left,
                      bounds.hostData().top + deltaBounds.top,
                    bounds.hostData().width + deltaBounds.width,
                  bounds.hostData().height + deltaBounds.height};

            planet->setBounds(bounds.hostData());

            if (fluidEnabled) {
                fluidSimulator = FluidSimulator(0.05, (size_t)bounds.hostData().width, (size_t)bounds.hostData().height, {});
            }
        }
    } else {
        dragHandler.reset();
    }

    if (!fluidEnabled) return;

    std::swap(mousePos1, mousePos2);
    mousePos2 = {(worldCoords.x - bounds.hostData().left) * fluidSimulator.scale,
                 (worldCoords.y - bounds.hostData().top) * fluidSimulator.scale};
    if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
        fluidSimulator.addForce(mousePos2, mousePos2 - mousePos1);
    }
}

void Environment::reset() {
    points.clear();
    cellLinks.clear();
    geneticAlgorithm.reset();
    fluidSimulator.init();
    bounds = initialBounds;
    planet->reset();
    planet->setBounds(bounds.hostData());
};

void Environment::cleanup() {
    points.clear();
    cellLinks.clear();
    geneticAlgorithm.reset();
    fluidSimulator.reset();
}

size_t Environment::addPoint(Point p) {
    return points.push(p);
}

void Environment::removePoint(int index) {
    points.remove(index);
}

size_t Environment::addCellLink(const CellLink &cellLink) {
    return cellLinks.push(cellLink);
}

size_t Environment::addCell(const Cell& cell) {
    return cells.push(cell);
}

void Environment::removeCell(int index) {
    cells.remove(index);
}

sf::FloatRect* Environment::getBounds() {
    return bounds.hostDataPtr();
}

float2 Environment::randomPos() {
    return {Random::random(bounds.hostData().left, bounds.hostData().left + bounds.hostData().width),
            Random::random(bounds.hostData().top, bounds.hostData().top + bounds.hostData().height)};
}

void Environment::toggleGridLinesVisible() {
    gridLinesVisible = !gridLinesVisible;
}

bool Environment::getGridLineVisibility() const {
    return gridLinesVisible;
}

bool Environment::getFluidEnabled() const {
    return fluidEnabled;
}
void Environment::toggleFluidEnabled() {
    fluidEnabled = !fluidEnabled;
}

Planet& Environment::getPlanet() {
    return *planet;
}
void Environment::setPlanet(Planet* newPlanet) {
    planet = newPlanet;
    planet->setBounds(bounds.hostData());
}