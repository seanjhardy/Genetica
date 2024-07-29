#ifndef FISH_HPP
#define FISH_HPP

// Fish.hpp
#include <vector>
#include <cmath>
#include "../../modules/verlet/point.hpp"
#include "../environment.hpp"
#include "../environment.hpp"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

class FishTank;

class Fish {
public:
    static constexpr float maxAng = 10 * (M_PI / 180); // Convert degrees to radians
    static constexpr float maxAccel = 10.0f;
    static const sf::Color colour;

    Fish(FishTank& fishTank, float x, float y, int obs_pixels);

    void set_position(float x, float y);
    void reset(const Environment& env);
    std::vector<float> random_policy(float deltaTime);
    void step(Environment& env, const std::vector<float>& action);
    void render(VertexManager& viewer, bool stereoscopic, bool NPC);
    void drawVentralFins(VertexManager& viewer, float2 pos,
                         float angle, float finSize, const sf::Color& finColour, float skew);

    std::vector<Point*> body{};

    float dir_change;
    float dir_change_avg;
    float dir;
    float target_dir;
    float target_speed;
    float accel;
    int size;
    int fov;
    int fidelity;
    int view_dist;

    std::vector<float> rayDist;
    std::vector<float> collision_force;
};

#endif