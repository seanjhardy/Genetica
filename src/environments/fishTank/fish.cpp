#ifndef FISH_CPP
#define FISH_CPP

// Fish.cpp
#include "fish.hpp"
#include "fishTank.hpp"
#include <cmath>
#include <algorithm>
#include "../../modules/utils/mathUtils.cpp"
#include "../../modules/utils/GUIUtils.cpp"
#include "../../modules/verlet/constraints.cu"

const sf::Color Fish::colour = sf::Color(255, 0, 0);

Fish::Fish(FishTank& fishTank, float x, float y, int obs_pixels)
        : dir_change(0), dir_change_avg(0), dir(0), target_dir(0), target_speed(0),
          accel(0), size(5), fov(120), fidelity(obs_pixels),
          view_dist(200), rayDist(obs_pixels, 200),
          collision_force(4, 0) {

    Point* p = fishTank.addPoint(x + 20, y, 4);
    body.push_back(p);

    Point* p2 = fishTank.addPoint(x + 5, y, 5);
    body.push_back(p2);

    Point* p3 = fishTank.addPoint(x - 10, y, 2);
    body.push_back(p3);

    Point* p4 = fishTank.addPoint(x - 25, y, 0.6);
    body.push_back(p4);
}

void Fish::set_position(float x, float y) {
    for (int i = 0; i < body.size(); ++i) {
        body[i]->pos = make_float2(x - 15.0f * i, y);
        body[i]->prevPos = body[i]->pos;
    }
}

void Fish::reset(const Environment& env) {
    float x = env.getBounds().width / 2, y = env.getBounds().height / 2;
    set_position(x, y);
    accel = 0;
}

// Random policy method implementation
std::vector<float> Fish::random_policy(float deltaTime) {
    bool reset = target_speed == 0 && target_dir == 0 && dir_change_avg == 0;
    float dir_change = angleDiff(dir, target_dir);
    dir_change = std::clamp(dir_change * 0.05f, -Fish::maxAng, Fish::maxAng);
    float dir_change_input = dir_change / Fish::maxAng;

    //float2 vel = body[0]->getVelocity();
    if ((static_cast<float>(rand()) / RAND_MAX) < deltaTime || reset) {
        target_dir = static_cast<float>(rand()) / RAND_MAX * M_PI * 2;
    }

    if (static_cast<float>(rand()) / RAND_MAX < (deltaTime) || reset) {
        target_speed = static_cast<float>(rand()) / RAND_MAX * (static_cast<float>(rand()) / RAND_MAX < 0.2f ? -1 : 1);
    }

    if (std::abs(collision_force[0]) > 0.1f) {
        target_dir = dir + M_PI;
    }
    std::vector<float> actions = {target_speed, dir_change_input};

    return actions;
}

// Step method implementation
void Fish::step(Environment& env, const std::vector<float>& action) {
    if (!action.empty()) {
        accel = action[0] * (action[0] >= 0 ? 1 : -0.2) * Fish::maxAccel;

        dir_change = action[1] * Fish::maxAng;
        dir_change_avg = dir_change_avg * 0.9 + dir_change * 0.1;
        dir += dir_change * env.dt;
        
        float2 force = make_float2(accel * cos(dir), accel * sin(dir));
        body[0]->applyForce(force);
    }

    // Apply physics constraints to body
    float delta = angleDiff(std::atan2(body[1]->pos.y - body[0]->pos.y,
                                       body[1]->pos.x - body[0]->pos.x), dir + M_PI);

    if (delta <= -M_PI) {
        delta += 2 * M_PI;
    } else if (delta >= M_PI) {
        delta -= 2 * M_PI;
    }

    if (std::abs(delta) > 0.01) {
        body[0]->rotate(body[0]->pos * 2 - body[1]->pos, -delta * 0.01 * env.dt);
        body[1]->rotate(body[0]->pos, delta * 0.01 * env.dt);
    }

    for (int i = 0; i < body.size(); i++) {
        if (i + 1 < body.size()) {
            constrainDistance(*body[i], *body[i + 1], 15, 1.0f);
        }
        if (i + 2 < body.size()) {
            constrainDistance(*body[i], *body[i + 2], 30, 0.05f);
        }
    }
}

void Fish::render(sf::RenderWindow& viewer, bool stereoscopic, bool NPC) {
    if (!NPC) {
        float halfFidelity = static_cast<float>(fidelity) * 0.5f;
        for (int i = 0; i < fidelity; ++i) {
            float start_x = body[0]->pos.x;
            float start_y = body[0]->pos.y;
            int index = i;
            if (stereoscopic) {
                if (i % 2 == 0) { // Odd index
                    index = (fidelity + 1) / 2 + (i - 1) / 2;
                } else { // Even index
                    index = i / 2;
                }
                float sign = i % 2 == 0 ? 1.0f : -1.0f;
                start_x += cos(dir + sign * (M_PI / 2)) * 5;
                start_y += sin(dir + sign * (M_PI / 2)) * 5;
            }
            float ang = (i - halfFidelity) / fidelity * fov;
            float dist = rayDist[index];
            float end_x = start_x + cos(dir + ang) * dist;
            float end_y = start_y + sin(dir + ang) * dist;

            sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(start_x, start_y), sf::Color(255, 0, 0, 51)),
                    sf::Vertex(sf::Vector2f(end_x, end_y), sf::Color(255, 0, 0, 51))
            };
            viewer.draw(line, 2, sf::Lines);
        }
    }
    auto finColour = interpolate(colour, sf::Color(255, 255, 255), 0.3f);
    auto bodyFinColour = interpolate(colour, sf::Color(0, 0, 0), 0.4f);

    float a2 = body[1]->angleTo(*body[0]);
    float a3 = body[2]->angleTo(*body[1]);
    float a4 = body[3]->angleTo(*body[2]);

    float2 midPos = body[0]->pos + make_float2(cos(dir), sin(dir));
    float2 p1 = midPos - body[0]->pos;
    float2 p2 = body[2]->pos - body[3]->pos;
    float skew = clockwiseAngleDiff(p1, p2);

    float headToMid = angleDiff(dir, a3);
    float totalCurvature = angleDiff(dir, a4);

    drawVentralFins(viewer, body[0]->pos.x, body[0]->pos.y, a2, body[0]->mass, finColour, skew);

    float angle = std::atan2(body[1]->pos.y - body[2]->pos.y, body[1]->pos.x - body[2]->pos.x);
    drawVentralFins(viewer, (body[2]->pos.x + body[1]->pos.x) * 0.5f, (body[2]->pos.y + body[1]->pos.y) * 0.5f, angle, body[2]->mass * 1.2f, finColour, skew);

    // Construct body polygon from body points
    for (size_t i = 0; i < body.size(); ++i) {
        body[i]->render(viewer, sf::Color(colour));

        if (i + 1 >= body.size()) continue;
        auto poly1 = findPerpendicularPoints(*body[i], *body[i + 1], body[i]->mass, body[i + 1]->mass);
        drawPolygon(viewer, poly1, sf::Color(colour));
    }

    std::vector<float2> dorsalFinPoints = {
        make_float2((body[0]->pos.x + body[1]->pos.x) * 0.5f,
        (body[0]->pos.y + body[1]->pos.y) * 0.5f),
        body[1]->pos, body[2]->pos
    };

    /*auto bezierPoints = bezier(body[2]->pos.x, body[2]->pos.y,
                               body[1]->pos.x + cos(a3 + M_PI / 2) * headToMid * 3,
                               body[1]->pos.y + sin(a3 + M_PI / 2) * headToMid * 3,
                               (body[0]->pos.x + body[1]->pos.x) * 0.5f,
                               (body[0]->pos.y + body[1]->pos.y) * 0.5f, 10);

    for (auto& p : bezierPoints) {
        dorsalFinPoints.emplace_back(sf::Vector2f(p.first, p.second));
    }

    drawPolygon(viewer, dorsalFinPoints, bodyFinColour);*/

    angle = body[3]->angleTo(*body[2]) + M_PI;
    float tailWidth = 0.5f - std::clamp(std::abs(totalCurvature * 0.3f), 0.0f, 0.3f);

    std::vector<float2> tailFinPoints = {body[2]->pos * 0.6f + body[3]->pos * 0.4f};

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle + tailWidth) * 10,
                         tailFinPoints[0].y + sin(angle + tailWidth) * 10);

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle) * 20,
                         tailFinPoints[0].y + sin(angle) * 20);

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle - tailWidth) * 10,
                         tailFinPoints[0].y + sin(angle - tailWidth) * 10);

    drawPolygon(viewer, tailFinPoints, finColour);

    // Eyes
    float eye1x = body[0]->pos.x + cos(dir + M_PI * 0.5f + 0.3f) * size * 0.7f;
    float eye1y = body[0]->pos.y + sin(dir + M_PI * 0.5f + 0.3f) * size * 0.7f;
    sf::Vector2f eye1 = sf::Vector2f(eye1x, eye1y);
    drawCircle(viewer, eye1, size * 0.5f, sf::Color(255, 255, 255));

    float eye2x = body[0]->pos.x + cos(dir - M_PI * 0.5f - 0.3f) * size * 0.7f;
    float eye2y = body[0]->pos.y + sin(dir - M_PI * 0.5f - 0.3f) * size * 0.7f;
    sf::Vector2f eye2 = sf::Vector2f(eye2x, eye2y);
    drawCircle(viewer, eye2, size * 0.5f, sf::Color(255, 255, 255));

    float iris_x = cos(dir + dir_change_avg * 10) * size * 0.25;
    float iris_y = sin(dir + dir_change_avg * 10) * size * 0.25;
    sf::Vector2f iris1 = sf::Vector2f(eye1x + iris_x, eye1y + iris_y);
    drawCircle(viewer, iris1, size * 0.3f, sf::Color(0, 0, 0));

    sf::Vector2f iris2 = sf::Vector2f(eye2x + iris_x, eye2y + iris_y);
    drawCircle(viewer, iris2, size * 0.3f, sf::Color(0, 0, 0));
}

void Fish::drawVentralFins(sf::RenderWindow& viewer, float x, float y, float angle, float size, const sf::Color& colour, float skew) {
    const float deg90 = M_PI / 2;
    skew = sin(skew);
    float skewLeft = skew > 0 ? 0 : std::abs(skew + 0.2f);
    float skewRight = skew < 0 ? 0 : std::abs(skew - 0.2f);

    sf::ConvexShape finLeft;
    finLeft.setPointCount(3);
    finLeft.setPoint(0, sf::Vector2f(cos(angle + deg90) * size, sin(angle + deg90) * size));
    finLeft.setPoint(1, sf::Vector2f(cos(angle + deg90 * (2 + 0.5f * skewLeft)) * size * 3, sin(angle + deg90 * (2 + 0.5f * skewLeft)) * size * 3));
    finLeft.setPoint(2, sf::Vector2f(cos(angle + deg90 * (1.5f + 0.3f * skewLeft)) * size * 5, sin(angle + deg90 * (1.5f + 0.3f * skewLeft)) * size * 5));
    finLeft.setFillColor(colour);
    finLeft.setPosition(x, y);
    viewer.draw(finLeft);

    sf::ConvexShape finRight;
    finRight.setPointCount(3);
    finRight.setPoint(0, sf::Vector2f(cos(angle - deg90) * size, sin(angle - deg90) * size));
    finRight.setPoint(1, sf::Vector2f(cos(angle - deg90 * (2 + 0.5f * skewRight)) * size * 3, sin(angle - deg90 * (2 + 0.5f * skewRight)) * size * 3));
    finRight.setPoint(2, sf::Vector2f(cos(angle - deg90 * (1.5f + 0.3f * skewRight)) * size * 5, sin(angle - deg90 * (1.5f + 0.3f * skewRight)) * size * 5));
    finRight.setFillColor(colour);
    finRight.setPosition(x, y);
    viewer.draw(finRight);
}

#endif // FISH_HPP