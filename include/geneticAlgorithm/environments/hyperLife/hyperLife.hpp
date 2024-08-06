// dot_environment.hpp
#ifndef HYPERLIFE_HPP
#define HYPERLIFE_HPP

#include "vector"
#include <geneticAlgorithm/environment.hpp>
#include "random"
#include <modules/verlet/point.hpp>
#include <modules/cuda/GPUVector.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentInstance.hpp>

/**
 * A 2D top-down simulation of multi-cellular organisms with modular neural networks.
 * The simulation is based on the Verlet integration method.
 */
class HyperLife : public Environment {
public:
    explicit HyperLife(const sf::FloatRect& bounds);
    void simulate(float deltaTime) override;
    void render(VertexManager& window) override;
    void reset() override;

    Point* addPoint(float x, float y, float mass);
    void addConnection(Point* a, Point* b, float distance);
    void addAngleConstraint(Point* a, Point* b,
                            Point* parentStart, Point* parentEbd,
                            float targetAngle, float stiffness);

private:
    //TODO: Fix random counts and avoid reallocating pointers
    GPUVector<Point> points = GPUVector<Point>(2000);
    GPUVector<Connection> connections = GPUVector<Connection>(2000);
    GPUVector<AngleConstraint> angleConstraints = GPUVector<AngleConstraint>(2000);
};

#endif