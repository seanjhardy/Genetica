// dot_environment.hpp
#ifndef ENVIRONMENT
#define ENVIRONMENT

#include <modules/physics/point.hpp>
#include "modules/cuda/structures/GPUVector.hpp"
#include "modules/cuda/structures/CGPUValue.hpp"
#include <modules/graphics/vertexManager.hpp>
#include <modules/graphics/dragHandler.hpp>
#include <simulator/planet.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>

/**
 * The environment contains information about all physics objects in the scene, as well as various
 * environmental variables such as the background map, an optional fluid simulator, grid lines,
 * the bounds of the environment and more
 */
class Environment {
    // Core environment data
    Planet* planet = nullptr;
    sf::FloatRect initialBounds;
    CGPUValue<sf::FloatRect> bounds;
    sf::FloatRect tempBounds;
    GeneticAlgorithm geneticAlgorithm;
    int entityID = 0;

    // UI
    bool gridLinesVisible = true;

    // GPU data storage
    GPUVector<Point> points = GPUVector<Point>();
    GPUVector<CellLink> cellLinks = GPUVector<CellLink>();
    GPUVector<Cell> cells = GPUVector<Cell>();

    // Drag handler for holding points
    int heldPoint = -1;
    DragHandler dragHandler;

public:
    explicit Environment(sf::FloatRect bounds);

    void simulate();
    void render(VertexManager& window);
    void reset();
    std::pair<bool, int> handleEvent(const sf::Event& event, sf::Vector2f mousePos);
    void update(const sf::Vector2f& worldCoords, float zoom, bool UIHovered);
    void cleanup();
    void drawGrid(VertexManager& vertexManager);
    GeneticAlgorithm& getGA() { return geneticAlgorithm; }

    // Point management
    size_t addPoint(const Point& p);
    void removePoint(int index);
    GPUVector<Point>& getPoints() { return points; }

    // Cell management
    size_t nextCellIdx();
    size_t nextCellLinkIdx();
    size_t addCell(const Cell& cell);
    void removeCell(int index);
    GPUVector<Cell>& getCells() { return cells; }
    size_t addCellLink(const CellLink& cellLink);
    GPUVector<CellLink>& getCellLinks() { return cellLinks; }

    [[nodiscard]] sf::FloatRect* getBounds();
    [[nodiscard]] float2 randomPos();
    void toggleGridLinesVisible();
    [[nodiscard]] bool getGridLineVisibility() const;
    Planet& getPlanet();
    void setPlanet(Planet* newPlanet);

    int nextEntityID();
};

#endif