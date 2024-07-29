#include "fishTank.hpp"
#include "fish.hpp"
#include "mapGenerator.cpp"
#include "../../modules/cuda/updatePoints.hpp"
#include <cuda_runtime.h>
#include "rock.hpp"

FishTank::FishTank(const sf::FloatRect& bounds)
        : Environment("FishTank", bounds),
          rng(std::random_device{}()),
          xDist(bounds.left, bounds.left + bounds.width),
          yDist(bounds.top, bounds.top + bounds.height),
          sizeDist(1.0f, 5.0f) {
    time = 0;
    rocks = generate_map({bounds.width, bounds.height}, 20,0.6, 0.1);
}


void FishTank::simulate(float deltaTime) {
    dt = 1;//deltaTime;

    for (Fish &fish : fishArray) {
        fish.step(*this, fish.random_policy(dt));
    }
    points.syncToDevice();
    updatePoints(points, connections, bounds, dt);
}

void FishTank::render(VertexManager& viewer) {
    for (Rock rock : rocks) {
        rock.render(viewer);
    }

    for (Fish fish : fishArray) {
        fish.render(viewer, false, true);
    }

}

void FishTank::reset() {
    fishArray.clear();
    for (int i = 0; i < 100; ++i) {
        auto [x, y] = get_random_pos();
        fishArray.emplace_back(*this, x, y, 0);
    }
}

std::pair<int, int> FishTank::get_random_pos() {
    bool valid_pos = false;
    int x = 0, y = 0;

    while (!valid_pos) {
        x = rand() % static_cast<int>(bounds.width);
        y = rand() % static_cast<int>(bounds.height);
        valid_pos = true;
    }

    return std::make_pair(x, y);
}

Point* FishTank::addPoint(float x, float y, float mass = 1.0f) {
    points.push_back(Point(x, y, mass));
    points.syncToDevice();
    numPoints += 1;
    return points.back();
}

void FishTank::addConnection(Point *a, Point *b, float distance) {
    // Convert CPU pointers to GPU pointers
    a = &points.deviceData()[a - &points.hostData()[0]];
    b = &points.deviceData()[b - &points.hostData()[0]];

    Connection conn(a, b, distance);
    connections.push_back(conn);
}

