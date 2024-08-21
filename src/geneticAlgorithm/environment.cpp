#include "simulator/environment.hpp"
#include "simulator/entities/lifeform.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/cuda/updatePoints.hpp>
#include <modules/utils/print.hpp>
#include <simulator/simulator.hpp>

Environment::Environment(sf::FloatRect bounds) :
    bounds(bounds),
    quadtree({bounds.left + bounds.width / 2, bounds.top + bounds.height / 2},
                                                 {bounds.width, bounds.height}) {

}

void Environment::simulate(float deltaTime) {
    points.syncToDevice();
    updatePoints(points, connections, parentChildLinks, bounds, deltaTime);
    points.syncToHost();
    quadtree.update();
};

void Environment::render(VertexManager& vertexManager) {
    vertexManager.addFloatRect(bounds.hostData(), sf::Color(10, 10, 20));

    // Draw a grid of columns and rows inside of bounds:
    if (vertexManager.getSizeInView(1) > 0.5) {
        sf::Color gridColor = sf::Color(30, 30, 40);
        for (float i = 0; i < bounds.hostData().width + 1; i += 20) {
            vertexManager.addLine({i, 0}, {i, bounds.hostData().height}, gridColor, 1);
        }
        for (float i = 0; i < bounds.hostData().height + 1; i += 20) {
            vertexManager.addLine({0, i}, {bounds.hostData().width, i}, gridColor, 1);
        }
    }
    if (quadTreeVisible) {
        quadtree.render(vertexManager);
    }
};

LifeForm& Environment::createRandomLifeForm() {
    map<int, string> genome = Simulator::get().getGA().createRandomGenome();
    auto* lifeForm = new LifeForm(this, {Random::random(bounds.hostData().width),
                                          Random::random(bounds.hostData().height)},
                                   genome);
    lifeForm->energy = 100;
    Simulator::get().getGA().addLifeForm(lifeForm);
    return *lifeForm;
}

int Environment::handleEvent(const sf::Event& event, const sf::Vector2f mousePos) {
    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            // Find the nearest point in the quadtree (within a radius of 10)
            heldPoint = quadtree.findNearestPoint({mousePos.x, mousePos.y}, 20);

        }
    }
    if (event.type == sf::Event::MouseButtonReleased) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            heldPoint = nullptr;
        }
    }
    if (heldPoint) {
        return heldPoint->entityID;
    } else {
        return -1;
    }
}

void Environment::update(const sf::Vector2f& mousePos) {
    if (heldPoint != nullptr) {
        heldPoint->setPos({mousePos.x, mousePos.y});
    }
}

void Environment::reset() {
    points.clear();
    connections.clear();
    parentChildLinks.clear();
    quadtree.reset();
};

int Environment::addPoint(int id, float x, float y, float mass) {
    //Add a point to the "points" vector and return a pointer to it
    points.push_back(Point(id, x, y, mass));
    quadtree.insert(points.back());
    return points.size() - 1;
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

void Environment::toggleQuadTreeVisible() {
    quadTreeVisible = !quadTreeVisible;
}

bool Environment::isQuadTreeVisible() const {
    return quadTreeVisible;
}

Quadtree* Environment::getQuadtree() {
    return &quadtree;
}
