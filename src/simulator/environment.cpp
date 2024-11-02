#include <simulator/environment.hpp>
#include <simulator/entities/lifeform.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/cuda/updatePoints.hpp>
#include <simulator/simulator.hpp>
#include <modules/cuda/findNearest.hpp>

Environment::Environment(sf::FloatRect bounds) :
    initialBounds(bounds),
    bounds(bounds), fluidSimulator(
      0.1, bounds.width, bounds.height,
      {}) {
    planet = &Planet::planets["Delune"];
}

void Environment::simulate(float deltaTime) {
    if (fluidEnabled && Simulator::get().getStep() % 500 == 0) {
        fluidSimulator.update(0.02);
    }
    planet->update();

    points.syncToDevice();
    updatePoints(points, connections, bounds, deltaTime);
    points.syncToHost();
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
        for (float i = 0; i < bounds.hostData().width + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left + i, bounds.hostData().top},
                                  {bounds.hostData().left + i, bounds.hostData().top + bounds.hostData().height}, gridColor,
                                  thickness);
        }
        for (float i = 0; i < bounds.hostData().height + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left, bounds.hostData().top + i},
                                  {bounds.hostData().left + bounds.hostData().width, bounds.hostData().top + i}, gridColor,
                                  thickness);
        }
    }

    dragHandler.render(vertexManager, bounds.hostData());
};

bool Environment::handleEvent(const sf::Event& event, const sf::Vector2f mousePos, Entity** selectedEntity) {
    dragHandler.handleEvent(event);

    /*if (!dragHandler.isDragging() && planet->getBounds() != bounds.hostData()) {
        planet->setBounds(bounds.hostData());
    }*/

    if (event.type == sf::Event::MouseButtonReleased) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            heldPoint = nullptr;
        }
    }

    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            // Find the nearest point in the quadtree (within a radius of 10)
            std::pair<int, float> nearestPoint = findNearest(points, mousePos.x, mousePos.y, 20);
            if (nearestPoint.first != -1) {
                if (points.size() > nearestPoint.first) {
                    heldPoint = &points[nearestPoint.first];
                    *selectedEntity = entities[heldPoint->entityID];
                }
            } else {
                heldPoint = nullptr;
                *selectedEntity = nullptr;
            }
            return true;
        }
    }
    return false;
}

void Environment::update(const sf::Vector2f& worldCoords, float zoom, bool UIHovered) {
    if (heldPoint != nullptr) {
        heldPoint->setPos({worldCoords.x, worldCoords.y});
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
                fluidSimulator = FluidSimulator(0.05, bounds.hostData().width, bounds.hostData().height, {});
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
    connections.clear();
    fluidSimulator.init();
    bounds = initialBounds;
    planet->reset();
    planet->setBounds(bounds.hostData());
};

void Environment::cleanup() {
    points.clear();
    connections.clear();
    fluidSimulator.reset();
}

int Environment::addPoint(int id, float x, float y, float mass) {
    //Add a point to the "points" vector and return a pointer to it
    points.push_back(Point(id, x, y, mass));
    return points.size() - 1;
}

void Environment::removePoint(int index) {
    points.remove(index);
}

void Environment::updatePoint(int index, Point updatedPoint) {
    points.update(index, updatedPoint);

}

void Environment::addEntity(int id, Entity* entity) {
    entities[id] = entity;
}

Point* Environment::getPoint(int index) {
    return &points[index];
}

int Environment::addConnection(int a, int b, float distance){
    connections.push_back(Connection(a, b, distance));
    return connections.size() - 1;
}

Connection* Environment::getConnection(int index) {
    return &connections[index];
}

void Environment::updateConnection(int index, Connection updatedConnection) {
    connections.update(index, updatedConnection);
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

int Environment::nextEntityID() {
    return entityID++;
}