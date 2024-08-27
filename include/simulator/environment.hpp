// dot_environment.hpp
#ifndef ENVIRONMENT
#define ENVIRONMENT

#include <vector>
#include <random>
#include <modules/physics/point.hpp>
#include <modules/cuda/GPUVector.hpp>
#include <modules/cuda/GPUValue.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <simulator/entities/entity.hpp>
#include <modules/utils/dragHandler.hpp>

class LifeForm;

/**
 * A 2D top-down simulation of multi-cellular organisms with modular neural networks.
 * The simulation is based on the Verlet integration method.
 */
class Environment {
private:
    GPUValue<sf::FloatRect> bounds;

    GPUVector<Point> points = GPUVector<Point>();
    GPUVector<Connection> connections = GPUVector<Connection>();
    GPUVector<ParentChildLink> parentChildLinks = GPUVector<ParentChildLink>();
    bool gridLinesVisible = true;
    Point* heldPoint = nullptr;
    std::unordered_map<int, Entity*> entities;
    int entityID;
    DragHandler dragHandler;

public:
    explicit Environment(sf::FloatRect bounds);
    void simulate(float deltaTime);
    void render(VertexManager& window);
    void reset();
    bool handleEvent(const sf::Event& event, sf::Vector2f mousePos, Entity** selectedEntity);
    void update(const sf::Vector2f& mousePos);

    int addPoint(int id, float x, float y, float mass);
    void addEntity(int id, Entity* entity);
    Point* getPoint(int index);
    ParentChildLink* getParentChildLink(int index);
    void addConnection(int a, int b, float distance);
    int addParentChildLink(int a, int b,
                         int parentStart, int parentEnd,
                         float2 pointOnParent,
                         float targetAngle, float stiffness);
    GPUVector<ParentChildLink>& getParentChildLinks() { return parentChildLinks; }

    [[nodiscard]] sf::FloatRect getBounds() const;
    void toggleGridLinesVisible();
    [[nodiscard]] bool getGridLineVisibility() const;

    int nextEntityID();

};

#endif