#include <geneticAlgorithm/environments/fishTank/fish.hpp>
#include <geneticAlgorithm/environments/fishTank/fishTank.hpp>
#include "mapGenerator.cpp"
#include "cuda_runtime.h"
#include <modules/noise/random.hpp>

FishTank::FishTank(const sf::FloatRect& bounds)
        : Environment("FishTank", bounds) {
    time = 0;
    rocks = GPUVector<Rock>(generate_map({bounds.width, bounds.height}, 20,0.6, 0.1));
    rocks.syncToDevice();
}


void FishTank::simulate(float deltaTime) {
    dt = 1;//deltaTime;

    for (Fish &fish : fishArray) {
        fish.simulate(deltaTime);
    }
    points.syncToDevice();
    //updatePoints(points, connections, bounds, dt);
}

void FishTank::render(VertexManager& viewer) {
    for (Rock rock : rocks.hostData()) {
        rock.render(viewer);
    }

    for (Fish fish : fishArray) {
        fish.render(viewer);
    }
}

void FishTank::reset() {
    fishArray.clear();
    for (int i = 0; i < 100; ++i) {
        auto [x, y] = get_random_pos();
        fishArray.emplace_back(this, x, y);
    }
}

std::pair<int, int> FishTank::get_random_pos() {
    bool valid_pos = false;
    int x = 0, y = 0;

    while (!valid_pos) {
        x = Random::random(bounds.width);
        y = Random::random(bounds.height);
        valid_pos = true;
    }

    return std::make_pair(x, y);
}

Point* FishTank::addPoint(float x, float y, float mass = 1.0f) {
    points.push_back(Point(x, y, mass));
    points.syncToDevice();
    return points.back();
}

void FishTank::addConnection(Point *a, Point *b, float distance) {
    // Convert CPU pointers to GPU pointers
    a = &points.deviceData()[a - &points.hostData()[0]];
    b = &points.deviceData()[b - &points.hostData()[0]];

    Connection conn(a, b, distance);
    connections.push_back(conn);
}

