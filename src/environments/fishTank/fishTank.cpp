// dot_environment.cpp
#include "fishTank.hpp"
#include "fish.hpp"
#include <iostream>
#include "mapGenerator.cpp"
#include "../../modules/cuda/updatePoints.hpp"
#include <cuda_runtime.h>

FishTank::FishTank(const sf::FloatRect& bounds)
        : Environment("FishTank", bounds),
          rng(std::random_device{}()),
          xDist(bounds.left, bounds.left + bounds.width),
          yDist(bounds.top, bounds.top + bounds.height),
          sizeDist(1.0f, 5.0f) {
    time = 0;
    map = generate_map({bounds.width, bounds.height}, 20,0.6, 0.1);
}


void FishTank::simulate(float deltaTime) {
    if (time == 0) {
        this->reset();
    }
    dt = 1;//deltaTime;

    for (Fish &fish : fishArray) {
        fish.step(*this, fish.random_policy(dt));
    }

    //updatePointsOnGPU(points, bounds, dt);
    time++;
}

void FishTank::render(sf::RenderWindow& window) {
    for (int x = 0; x < map.size(); ++x) {
        for (int y = 0; y < map[x].size(); ++y) {
            if (!map[x][y]) continue;
            sf::RectangleShape rect(sf::Vector2f(20, 20));
            rect.setPosition(x * 20, y * 20);
            rect.setFillColor(sf::Color(100, 100, 100));window.draw(rect);
            window.draw(rect);
        }
    }
    for (Fish fish : fishArray) {
        fish.render(window, false, true);
    }
}

void FishTank::reset() {
    fishArray.clear();
    for (int i = 0; i < 50; ++i) {
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
        valid_pos = !map[std::floor(x / 20.0)][std::floor(y / 20.0)];
    }

    return std::make_pair(x, y);
}

Point* FishTank::addPoint(float x, float y, float mass = 1.0f) {
    points.emplace_back(x, y, mass);
    return &points.back();
}
