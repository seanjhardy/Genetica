#ifndef FISH_HPP
#define FISH_HPP

// Fish.hpp
#include "vector"
#include "cmath"
#include <modules/verlet/point.hpp>
#include <geneticAlgorithm/environment.hpp>
#include <geneticAlgorithm/individual.hpp>
#include <geneticAlgorithm/environments/fishTank/fishTank.hpp>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

class Fish : public Individual {
public:
    static const float maxAng; // Convert degrees to radians
    static const float maxAccel;
    static const sf::Color colour;

    Fish(FishTank* fishTank, float x, float y);

    void set_position(float x, float y);
    std::tuple<float, float> random_policy(float deltaTime);
    void simulate(float dt) override;
    void render(VertexManager& viewer) override;
    void drawVentralFins(VertexManager& viewer, float2 pos,
                         float angle, float finSize, const sf::Color& finColour, float skew);
    void mutate() override {};
    void init() override;
    Individual& combine(Individual *partner) override { return (Individual &) *this; };
    Individual& clone(bool mutate) override {return (Individual &) *this; };

    FishTank* getEnv() override;
    std::vector<size_t> body{};

    float dir_change;
    float dir_change_avg;
    float dir;
    float target_dir;
    float target_speed;
    int size;

    std::vector<float> rayDist;
    std::vector<float> collision_force;
};

#endif