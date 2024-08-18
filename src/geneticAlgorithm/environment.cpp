#include <geneticAlgorithm/environment.hpp>
#include "geneticAlgorithm/entities/lifeform.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/cuda/updatePoints.hpp>

Environment::Environment(sf::FloatRect bounds) :
    title("HyperLife"),
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
    unordered_map<int, string> genome = GeneticAlgorithm::get().createRandomGenome();
    auto* lifeForm = new LifeForm(this, {Random::random(bounds.hostData().width),
                                          Random::random(bounds.hostData().height)},
                                   genome);
    lifeForm->energy = 100;
    GeneticAlgorithm::get().addLifeForm(lifeForm);
    return *lifeForm;
}

void Environment::reset() {
    points.clear();
    connections.clear();
    parentChildLinks.clear();
    quadtree.reset();
};

int Environment::addPoint(float x, float y, float mass) {
    //Add a point to the "points" vector and return a pointer to it
    points.push_back(Point(x, y, mass));
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

char* Environment::getTitle() const {
    char* result = new char[title.size() + 1];
    // Copy the input string into the myString array
    strcpy_s(result, title.size() + 1, title.c_str());
    // Ensure null termination
    result[title.size()] = '\0';
    return result;
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
