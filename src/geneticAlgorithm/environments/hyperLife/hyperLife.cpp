#include "geneticAlgorithm/environment.hpp"
#include "modules/verlet/point.hpp"
#include "geneticAlgorithm/environments/hyperLife/hyperLife.hpp"
#include "modules/cuda/updatePoints.hpp"

HyperLife::HyperLife(const sf::FloatRect& bounds): Environment("HyperLife", bounds) {

}

void HyperLife::simulate(float deltaTime) {
    updatePoints(points, connections, angleConstraints, bounds, deltaTime);
};

void HyperLife::render(VertexManager& window) {
    window.addFloatRect(bounds, sf::Color(100, 100, 100));
};

void HyperLife::reset() {
    points.clear();
    connections.clear();
    angleConstraints.clear();
};

Point* HyperLife::addPoint(float x, float y, float mass) {
    points.push_back(Point(x, y, mass));
    return points.back();
}

void HyperLife::addConnection(Point* a, Point* b, float distance){
    connections.push_back(Connection(a, b, distance));
}

void HyperLife::addAngleConstraint(Point* a, Point* b, Point* parentStart, Point* parentEnd, float targetAngle, float stiffness) {
    angleConstraints.push_back(AngleConstraint(a, b, parentStart, parentEnd, targetAngle, stiffness));
}
