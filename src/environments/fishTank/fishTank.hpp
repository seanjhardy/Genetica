// dot_environment.hpp
#ifndef FISH_TANK_HPP
#define FISH_TANK_HPP

#include "../environment.hpp"
#include <random>
#include "../../modules/verlet/point.hpp"
#include "fish.hpp"

class FishTank : public Environment {
public:
    explicit FishTank(const sf::FloatRect& bounds);
    void simulate(float deltaTime) override;
    void render(sf::RenderWindow& window) override;
    void reset() override;

    Point* addPoint(float x, float y, float mass);
    std::pair<int, int> get_random_pos();

private:
    std::vector<Point> points;
    std::vector<Fish> fishArray;
    std::mt19937 rng;
    std::uniform_real_distribution<float> xDist, yDist, sizeDist;
    std::vector<std::vector<bool>> map;
};

#endif