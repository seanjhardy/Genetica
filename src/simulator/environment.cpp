#include <simulator/environment.hpp>
#include <simulator/entities/lifeform.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/cuda/updatePoints.hpp>
#include <modules/utils/print.hpp>
#include <simulator/simulator.hpp>
#include <modules/cuda/findNearest.hpp>
#include <modules/graphics/cursorManager.hpp>

Environment::Environment(sf::FloatRect bounds) :
    bounds(bounds){
}

void Environment::simulate(float deltaTime) {
    points.syncToDevice();
    updatePoints(points, connections, parentChildLinks, bounds, deltaTime);
    points.syncToHost();
};

void Environment::render(VertexManager& vertexManager) {
    vertexManager.addFloatRect(bounds.hostData(), sf::Color(10, 10, 20));

    // Draw a grid of columns and rows inside of bounds:
    if (vertexManager.getSizeInView(1) > 0.5 && gridLinesVisible) {
        sf::Color gridColor = sf::Color(30, 30, 40);
        for (float i = 0; i < bounds.hostData().width + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left + i, bounds.hostData().top},
                                  {bounds.hostData().left + i, bounds.hostData().top + bounds.hostData().height}, gridColor, 1);
        }
        for (float i = 0; i < bounds.hostData().height + 1; i += 20) {
            vertexManager.addLine({bounds.hostData().left, bounds.hostData().top + i},
                                  {bounds.hostData().left + bounds.hostData().width, bounds.hostData().top + i}, gridColor, 1);
        }
    }
    dragHandler.render(vertexManager, bounds.hostData());
};

bool Environment::handleEvent(const sf::Event& event, const sf::Vector2f mousePos, Entity** selectedEntity) {
    dragHandler.handleEvent(mousePos, event);

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

void Environment::update(const sf::Vector2f& mousePos) {
    sf::Vector2f worldCoords = Simulator::get().getCamera().getCoords(mousePos);

    if (heldPoint != nullptr) {
        heldPoint->setPos({worldCoords.x, worldCoords.y});
    }

    sf::FloatRect deltaBounds = dragHandler.update(worldCoords, bounds.hostData());

    if (deltaBounds.left != 0 || deltaBounds.top != 0 || deltaBounds.width != 0 || deltaBounds.height != 0) {
        bounds = {bounds.hostData().left + deltaBounds.left,
                  bounds.hostData().top + deltaBounds.top,
                  bounds.hostData().width + deltaBounds.width,
                  bounds.hostData().height + deltaBounds.height};
        Simulator::get().getCamera().setBounds(bounds.hostData());
    }
}

void Environment::reset() {
    points.clear();
    connections.clear();
    parentChildLinks.clear();
};

int Environment::addPoint(int id, float x, float y, float mass) {
    //Add a point to the "points" vector and return a pointer to it
    points.push_back(Point(id, x, y, mass));
    return points.size() - 1;
}

void Environment::addEntity(int id, Entity* entity) {
    entities[id] = entity;
}

Point* Environment::getPoint(int index) {
    return &points[index];
}

ParentChildLink* Environment::getParentChildLink(int index) {
    return &parentChildLinks[index];
}

void Environment::addConnection(int a, int b, float distance){
    connections.push_back(Connection(a, b, distance));
}

int Environment::addParentChildLink(int a, int b, int parentStart, int parentEnd,
                                    float2 pointOnParent, float targetAngle, float stiffness) {
    parentChildLinks.push_back(ParentChildLink(a, b,
                                               parentStart, parentEnd,
                                               pointOnParent, targetAngle, stiffness));
    return parentChildLinks.size() - 1;
}

sf::FloatRect Environment::getBounds() const {
    return bounds.hostData();
}

void Environment::toggleGridLinesVisible() {
    gridLinesVisible = !gridLinesVisible;
}

bool Environment::getGridLineVisibility() const {
    return gridLinesVisible;
}

int Environment::nextEntityID() {
    return entityID++;
}