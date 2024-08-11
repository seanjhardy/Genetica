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
private:
    //TODO: Fix random counts and avoid reallocating pointers
    GPUVector<Point> points = GPUVector<Point>(1000);
    GPUVector<Connection> connections = GPUVector<Connection>(100);
    GPUVector<AngleConstraint> angleConstraints = GPUVector<AngleConstraint>(100);
public:
    explicit HyperLife(const sf::FloatRect& bounds);
    void simulate(float deltaTime) override;
    void render(VertexManager& window) override;
    void reset() override;
    Individual& createRandomIndividual() override;

    size_t addPoint(float x, float y, float mass);
    Point* getPoint(size_t index);
    void addConnection(size_t a, size_t b, float distance);
    void addAngleConstraint(size_t a, size_t b,
                            size_t parentStart, size_t parentEbd,
                            float targetAngle, float stiffness);

};

#endif