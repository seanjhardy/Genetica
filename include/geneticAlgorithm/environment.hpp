// dot_environment.hpp
#ifndef HYPERLIFE_HPP
#define HYPERLIFE_HPP

#include "vector"
#include "random"
#include "modules/verlet/point.hpp"
#include "modules/cuda/GPUVector.hpp"
#include "modules/cuda/GPUValue.hpp"
#include "modules/graphics/vertexManager.hpp"
#include "modules/quadtree/quadtree.hpp"

class LifeForm;

/**
 * A 2D top-down simulation of multi-cellular organisms with modular neural networks.
 * The simulation is based on the Verlet integration method.
 */
class Environment {
private:
    std::string title;
    GPUValue<sf::FloatRect> bounds;

    GPUVector<Point> points = GPUVector<Point>();
    GPUVector<Connection> connections = GPUVector<Connection>();
    GPUVector<ParentChildLink> parentChildLinks = GPUVector<ParentChildLink>();
    Quadtree quadtree;
    bool quadTreeVisible = false;

public:
    explicit Environment(sf::FloatRect bounds);
    void simulate(float deltaTime);
    void render(VertexManager& window);
    void reset();
    LifeForm& createRandomLifeForm();

    int addPoint(float x, float y, float mass);
    Point* getPoint(int index);
    ParentChildLink* getParentChildLink(int index);
    void addConnection(int a, int b, float distance);
    int addParentChildLink(int a, int b,
                         int parentStart, int parentEnd,
                         float2 pointOnParent,
                         float targetAngle, float stiffness);
    GPUVector<ParentChildLink>& getParentChildLinks() { return parentChildLinks; }

    [[nodiscard]] char* getTitle() const;
    [[nodiscard]] sf::FloatRect getBounds() const;
    void toggleQuadTreeVisible();
    bool isQuadTreeVisible() const;

};

#endif