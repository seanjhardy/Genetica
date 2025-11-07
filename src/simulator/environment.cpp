#include <simulator/environment.hpp>
#include "geneticAlgorithm/lifeform.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/gpu/updatePoints.hpp>
#include <simulator/simulator.hpp>
#include <modules/gpu/findNearest.hpp>
#include <modules/gpu/OpenCLManager.hpp>
#include <modules/utils/random.hpp>
#include <chrono>
// COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
// #include <modules/gpu/updateCells.hpp>
// #include <modules/gpu/updateCellLinks.hpp>

Environment::Environment(sf::FloatRect bounds) :
    initialBounds(bounds),
    bounds(bounds),
    lastBoundsUpdate(std::chrono::steady_clock::now()) {
    planet = &Planet::planets["Delune"];
}

void Environment::simulate() {
    planet->update();
    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    // geneticAlgorithm.simulate();

    // Physics simulation of life forms
    updatePoints(points, bounds);
    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    // updateCells(getGA().getPopulation(), cells, points, cellDivisionData);
    // updateCellLinks(points, cells, cellLinks);
};

void Environment::render(VertexManager& vertexManager) {
    planet->render(vertexManager);

    // Draw a grid of columns and rows inside of bounds
    drawGrid(vertexManager);

    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    //draw life forms
    // geneticAlgorithm.render(vertexManager, cells, cellLinks, points);

    // Render points
    renderPoints(vertexManager);

    // Render draggable environment bounds
    dragHandler.render(vertexManager, bounds.hostData());
};

std::pair<bool, int> Environment::handleEvent(const sf::Event& event, const sf::Vector2f mousePos) {
    dragHandler.handleEvent(event);

    // Update planet bounds when dragging stops or if there's a pending update
    if ((!dragHandler.isDragging() && (planet->getBounds() != bounds.hostData() || hasPendingBoundsUpdate))) {
        consoleLog("Dragging stopped, applying pending bounds update if needed. Current planet bounds: ", planet->getBounds(), " Environment bounds: ", bounds.hostData());
        planet->setBounds(bounds.hostData());
        hasPendingBoundsUpdate = false;
        lastBoundsUpdate = std::chrono::steady_clock::now();
        consoleLog("Applied bounds update after dragging stopped");
    }

    if (event.type == sf::Event::MouseButtonReleased &&
        event.mouseButton.button == sf::Mouse::Left) {
        heldPoint = -1;
        return { false, -1 };
    }

    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
            // Find the nearest point in the quadtree (within a radius of 20)
            // std::pair<int, float> nearestPoint = findNearest(points, mousePos.x, mousePos.y, 20);

            // if (nearestPoint.first != -1) {
            //     if (points.size() > nearestPoint.first) {
            //         heldPoint = nearestPoint.first;
            //         int newSelectedEntityId = movePoint(points, heldPoint, mousePos);
            //         return { true, newSelectedEntityId };
            //     }
            // }
            // else {
            heldPoint = -1;
            return { true, -1 };
            // }
        }
    }

    return { false, -1 };
};

void Environment::update(const sf::Vector2f& worldCoords, float zoom, bool UIHovered) {
    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    // if (heldPoint != -1) {
    //     movePoint(points, heldPoint, worldCoords);
    // }

    if (!UIHovered) {
        sf::FloatRect deltaBounds = dragHandler.update(worldCoords, tempBounds, 15.0f / zoom);
        if (deltaBounds.left != 0 || deltaBounds.top != 0 || deltaBounds.width != 0 || deltaBounds.height != 0) {
            tempBounds += deltaBounds;
            sf::FloatRect newBounds = {
                round(tempBounds.left / 20) * 20, round(tempBounds.top / 20) * 20,
                round(tempBounds.width / 20) * 20, round(tempBounds.height / 20) * 20
            };

            // Only update planet bounds if they actually changed to avoid unnecessary noise map recalculations
            sf::FloatRect currentBounds = bounds.hostData();
            if (newBounds.left != currentBounds.left ||
                newBounds.top != currentBounds.top ||
                newBounds.width != currentBounds.width ||
                newBounds.height != currentBounds.height ||
                hasPendingBoundsUpdate) {

                bounds = newBounds;

                // Throttle expensive noise map updates to prevent lag during camera movement + resizing
                auto now = std::chrono::steady_clock::now();
                auto timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastBoundsUpdate);

                if (timeSinceLastUpdate >= BOUNDS_UPDATE_THROTTLE_MS || !dragHandler.isDragging()) {
                    planet->setBounds(bounds.hostData());
                    lastBoundsUpdate = now;
                    hasPendingBoundsUpdate = false;
                }
                else {
                    hasPendingBoundsUpdate = true;
                }
            }
        }
    }
    else {
        dragHandler.reset();
    }
}

void Environment::drawGrid(VertexManager& vertexManager) {
    // Draw bounding box
    const auto bbox = bounds.hostData();
    const auto borderColor = sf::Color(255, 255, 255, 255);
    const auto lineSize = 2.0f / vertexManager.camera->getZoom();
    vertexManager.addLine(
        { bbox.left, bbox.top },
        { bbox.left, bbox.top + bbox.height }, borderColor,
        lineSize);
    vertexManager.addLine(
        { bbox.left + bbox.width, bbox.top },
        { bbox.left + bbox.width, bbox.top + bbox.height }, borderColor,
        lineSize);
    vertexManager.addLine(
        { bbox.left, bbox.top },
        { bbox.left + bbox.width, bbox.top }, borderColor,
        lineSize);
    vertexManager.addLine(
        { bbox.left, bbox.top + bbox.height },
        { bbox.left + bbox.width, bbox.top + bbox.height }, borderColor,
        lineSize);

    if (vertexManager.getSizeInView(1) < 0.2 || !gridLinesVisible) return;

    // Draw gridlines
    const int opacity = (int)clamp(10.0f, vertexManager.camera->getZoom() * 10.0f, 60.0f);
    const float thickness = clamp(1.0f, 1.0f / vertexManager.camera->getZoom(), 5.0f);
    const auto gridColor = sf::Color(0, 0, 0, opacity);
    const auto numWidthLines = floor(bbox.width / 20);
    const auto numHeightLines = floor(bbox.height / 20);
    for (int i = 1; i < numWidthLines - 2; i++) {
        vertexManager.addLine({ bbox.left + i * 20, bbox.top },
            { bbox.left + i * 20, bbox.top + bbox.height }, gridColor,
            thickness);
    }
    for (int i = 1; i < numHeightLines - 2; i++) {
        vertexManager.addLine({ bbox.left, bbox.top + i * 20 },
            { bbox.left + bbox.width, bbox.top + i * 20 }, gridColor,
            thickness);
    }
}

void Environment::reset() {
    points.destroy();
    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    // cells.destroy();
    // cellLinks.destroy();
    // geneticAlgorithm.reset();
    bounds = initialBounds;
    tempBounds = initialBounds;
    planet->reset();
    planet->setBounds(bounds.hostData());

    // Spawn test points after reset
    spawnTestPoints(100);
};

void Environment::cleanup() {
    points.destroy();
    // COMMENTED OUT FOR BAREBONES VERSION - TODO: Refactor for OpenCL
    // cells.destroy();
    // cellLinks.destroy();
    // geneticAlgorithm.reset();
}

size_t Environment::addPoint(const Point& p) {
    return points.push(p);
}

void Environment::removePoint(int index) {
    points.remove(index);
}

size_t Environment::addCellLink(const CellLink& cellLink) {
    return cellLinks.push(cellLink);
}


size_t Environment::nextCellIdx() {
    return cells.getNextIndex();
}

size_t Environment::nextCellLinkIdx() {
    return cellLinks.getNextIndex();
}

size_t Environment::addCell(const Cell& cell) {
    return cells.push(cell);
}

void Environment::removeCell(int index) {
    cells.remove(index);
}

sf::FloatRect* Environment::getBounds() {
    return bounds.hostDataPtr();
}

float2 Environment::randomPos() {
    return {
        Random::random(bounds.hostData().left, bounds.hostData().left + bounds.hostData().width),
        Random::random(bounds.hostData().top, bounds.hostData().top + bounds.hostData().height)
    };
}

void Environment::toggleGridLinesVisible() {
    gridLinesVisible = !gridLinesVisible;
}

bool Environment::getGridLineVisibility() const {
    return gridLinesVisible;
}

Planet& Environment::getPlanet() {
    return *planet;
}

void Environment::setPlanet(Planet* newPlanet) {
    planet = newPlanet;
    planet->setBounds(bounds.hostData());
}

int Environment::nextEntityID() {
    return entityID++;
}

// Spawn test points
void Environment::spawnTestPoints(int count) {
    sf::FloatRect b = bounds.hostData();
    std::cout << "Bounds: left=" << b.left << " top=" << b.top
        << " width=" << b.width << " height=" << b.height << std::endl;

    for (int i = 0; i < count; i++) {
        // Random position within bounds
        float x = Random::random(b.left + 20, b.left + b.width - 20);
        float y = Random::random(b.top + 20, b.top + b.height - 20);

        // Create point with random initial velocity
        Point p(nextEntityID(), x, y, 5.0f);

        // Give it a random initial velocity
        float vx = Random::random(-5.0f, 5.0f);
        float vy = Random::random(-5.0f, 5.0f);
        p.prevPos = make_float2(x - vx, y - vy);

        size_t index = addPoint(p);

        if (i == 0) {
            std::cout << "First point: entityID=" << p.entityID
                << " pos=(" << p.pos.s[0] << "," << p.pos.s[1] << ")"
                << " radius=" << p.radius << " index=" << index << std::endl;
        }
    }

    std::cout << "Spawned " << count << " test points, total points.size()=" << points.size() << std::endl;
}

// Render points
void Environment::renderPoints(VertexManager& vertexManager) {
    static bool firstRender = true;

    if (points.size() == 0) {
        if (firstRender) {
            std::cout << "renderPoints: points.size() is 0!" << std::endl;
            firstRender = false;
        }
        return;
    }

    // Read points from GPU
    auto hostPoints = points.toHost();

    if (firstRender) {
        std::cout << "renderPoints: points.size()=" << points.size()
            << ", hostPoints.size()=" << hostPoints.size() << std::endl;
        if (hostPoints.size() > 0) {
            std::cout << "First point position: (" << hostPoints[0].pos.s[0]
                << "," << hostPoints[0].pos.s[1] << "), radius=" << hostPoints[0].radius << std::endl;
        }
        firstRender = false;
    }

    // Render each point as a circle
    for (const auto& point : hostPoints) {
        sf::Color color = sf::Color(100, 150, 255, 255); // Blue color
        vertexManager.addCircle(point.getPos(), point.radius, color);
    }
}
