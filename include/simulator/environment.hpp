// dot_environment.hpp
#ifndef ENVIRONMENT
#define ENVIRONMENT

#include <modules/physics/point.hpp>
#include <modules/cuda/GPUVector.hpp>
#include <modules/cuda/CGPUValue.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <simulator/entities/entity.hpp>
#include "modules/graphics/dragHandler.hpp"
#include "modules/physics/fluid.hpp"
#include <simulator/planet.hpp>

class LifeForm;

/**
 * The environment contains information about all physics objects in the scene, as well as various
 * environmental variables such as the background map, an optional fluid simulator, grid lines,
 * the bounds of the environment and more
 */
class Environment {
    Planet* planet = nullptr;

    sf::FloatRect initialBounds;
    CGPUValue<sf::FloatRect> bounds;

    GPUVector<Point> points = GPUVector<Point>();
    GPUVector<Connection> connections = GPUVector<Connection>();

    bool gridLinesVisible = true;
    bool fluidEnabled = false;
    int entityID = 0;
    Point* heldPoint = nullptr;
    std::unordered_map<int, Entity*> entities;
    DragHandler dragHandler;

    FluidSimulator fluidSimulator;
    float2 mousePos1{}, mousePos2{};

public:
    explicit Environment(sf::FloatRect bounds);

    void simulate(float deltaTime);
    void render(VertexManager& window);
    void reset();
    bool handleEvent(const sf::Event& event, sf::Vector2f mousePos, Entity** selectedEntity);
    void update(const sf::Vector2f& worldCoords, float zoom, bool UIHovered);
    void cleanup();

    int addPoint(int id, float x, float y, float mass);
    void removePoint(int index);
    void addEntity(int id, Entity* entity);
    Point* getPoint(int index);
    void updatePoint(int index, Point updatedPoint);
    GPUVector<Point>& getPoints() { return points; }

    int addConnection(int a, int b, float distance);
    Connection* getConnection(int index);
    GPUVector<Connection>& getConnections() { return connections; }
    void updateConnection(int index, Connection updatedConnection);

    [[nodiscard]] sf::FloatRect* getBounds();
    [[nodiscard]] float2 randomPos();
    void toggleGridLinesVisible();
    [[nodiscard]] bool getGridLineVisibility() const;
    bool getFluidEnabled() const;
    void toggleFluidEnabled();
    Planet& getPlanet();
    void setPlanet(Planet* newPlanet);

    int nextEntityID();

};

#endif