// dot_environment.hpp
#ifndef FISH_TANK_HPP
#define FISH_TANK_HPP

#include <geneticAlgorithm/environment.hpp>
#include "random"
#include <modules/verlet/point.hpp>
#include <modules/cuda/GPUVector.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <geneticAlgorithm/environments/fishTank/rock.hpp>

class Fish;

class FishTank : public Environment {
public:
    explicit FishTank(const sf::FloatRect& bounds);
    void simulate(float deltaTime) override;
    void render(VertexManager& window) override;
    void reset() override;

    size_t addPoint(float x, float y, float mass);
    Point* getPoint(size_t index);
    void addConnection(size_t a, size_t b, float distance);
    std::pair<int, int> get_random_pos();

private:
    GPUVector<Point> points = GPUVector<Point>(2000);
    GPUVector<Connection> connections = GPUVector<Connection>(2000);
    GPUVector<Rock> rocks;

    std::vector<Fish> fishArray;
};

#endif