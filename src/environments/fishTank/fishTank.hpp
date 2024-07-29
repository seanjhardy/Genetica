// dot_environment.hpp
#ifndef FISH_TANK_HPP
#define FISH_TANK_HPP

#include "../environment.hpp"
#include <random>
#include "../../modules/verlet/point.hpp"
#include "fish.hpp"
#include "../../modules/cuda/utils/GPUVector.hpp"
#include "../../modules/graphics/vertexManager.hpp"
#include "rock.hpp"

class FishTank : public Environment {
public:
    explicit FishTank(const sf::FloatRect& bounds);
    void simulate(float deltaTime) override;
    void render(VertexManager& window) override;
    void reset() override;

    Point* addPoint(float x, float y, float mass);
    void addConnection(Point* a, Point* b, float distance);
    std::pair<int, int> get_random_pos();

private:
    int numPoints = 0;
    GPUVector<Point> points = GPUVector<Point>(2000);
    GPUVector<Connection> connections = GPUVector<Connection>(2000);

    std::vector<Fish> fishArray;
    std::mt19937 rng;
    std::uniform_real_distribution<float> xDist, yDist, sizeDist;
    std::vector<Rock> rocks;
};

#endif