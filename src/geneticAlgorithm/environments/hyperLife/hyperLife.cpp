#include <geneticAlgorithm/environment.hpp>
#include <modules/verlet/point.hpp>
#include <geneticAlgorithm/environments/hyperLife/hyperLife.hpp>
#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include <geneticAlgorithm/individual.hpp>
#include <modules/cuda/updatePoints.hpp>

HyperLife::HyperLife(const sf::FloatRect& bounds): Environment("HyperLife", bounds) {

}

void HyperLife::simulate(float deltaTime) {
    updatePoints(points, connections, angleConstraints, bounds, deltaTime);
};

void HyperLife::render(VertexManager& window) {
    window.addFloatRect(bounds, sf::Color(10, 10, 20));

    // Draw a grid of columns and rows inside of bounds:
    sf::Color gridColor = sf::Color(30, 30, 40);
    for (float i = 0; i < bounds.width + 1; i += 20) {
        window.addLine({i, 0}, {i, bounds.height}, gridColor, 1);
    }
    for (float i = 0; i < bounds.height + 1; i += 20) {
        window.addLine({0, i}, {bounds.width, i}, gridColor, 1);
    }
};

Individual& HyperLife::createRandomIndividual() {
    unordered_map<int, string> genome = GeneticAlgorithm::get().createRandomGenome();
    auto* lifeForm = new LifeForm(this, {Random::random(bounds.width),
                                          Random::random(bounds.height)},
                                   genome);
    lifeForm->energy = 100;
    GeneticAlgorithm::get().addIndividual(dynamic_cast<Individual *>(lifeForm));
    return dynamic_cast<Individual &>(*lifeForm);
}

void HyperLife::reset() {
    points.clear();
    connections.clear();
    angleConstraints.clear();
};

size_t HyperLife::addPoint(float x, float y, float mass) {
    //Add a point to the "points" vector and return a pointer to it
    points.push_back(Point(x, y, mass));
    return points.size() - 1;
}

Point* HyperLife::getPoint(size_t index) {
    return &points[index];
}

void HyperLife::addConnection(size_t a, size_t b, float distance){
    connections.push_back(Connection(a, b, distance));
}

void HyperLife::addAngleConstraint(size_t a, size_t b, size_t parentStart, size_t parentEnd, float targetAngle, float stiffness) {
    angleConstraints.push_back(AngleConstraint(a, b, parentStart, parentEnd, targetAngle, stiffness));
}
