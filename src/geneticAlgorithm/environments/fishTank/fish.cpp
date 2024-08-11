#ifndef FISH_CPP
#define FISH_CPP

// Fish.cpp
#include <geneticAlgorithm/environments/fishTank/fish.hpp>
#include <geneticAlgorithm/environments/fishTank/fishTank.hpp>
#include "cmath"
#include "algorithm"
#include <modules/utils/mathUtils.hpp>
#include "../../../modules/verlet/constraints.cu"
#include <modules/noise/random.hpp>

const sf::Color Fish::colour = sf::Color(255, 0, 0);
const float Fish::maxAng = 10 * (M_PI / 180);
const float Fish::maxAccel = 10.0f;

Fish::Fish(FishTank* fishTank, float x, float y)
        : Individual(fishTank, unordered_map<int, string>()),
          dir_change(0), dir_change_avg(0), dir(0), target_dir(0),
          target_speed(0), size(5),
          collision_force(4, 0) {

    std::vector<float3> points = {
        make_float3(x + 20, y, 4.0f),
        make_float3(x + 5, y, 5.0f),
        make_float3(x - 10, y, 2.0f),
        make_float3(x - 25, y, 0.6f)
    };

    for (int i = 0; i < points.size(); ++i) {
        size_t p = fishTank->addPoint(points[i].x, points[i].y, points[i].z);
        body.push_back(p);
        if (i > 0) {
            fishTank->addConnection(p, body[i - 1], 15);
        }
    }
}

void Fish::set_position(float x, float y) {
    for (int i = 0; i < body.size(); ++i) {
        getEnv()->getPoint(body[i])->setPos(make_float2(x - 15.0f * i, y));
    }
}

void Fish::init() {
    float x = Individual::getEnv()->getBounds().width / 2, y = Individual::getEnv()->getBounds().height / 2;
    set_position(x, y);
}

// Random policy method implementation
std::tuple<float, float> Fish::random_policy(float deltaTime) {
    bool reset = target_speed == 0 && target_dir == 0 && dir_change_avg == 0;

    float dir_change = angleDiff(dir, target_dir);
    dir_change = std::clamp(dir_change * 0.05f, -Fish::maxAng, Fish::maxAng);
    dir_change /= Fish::maxAng;

    //float2 vel = body[0]->getVelocity();
    if (Random::random() < 0.2 * deltaTime || reset) {
        target_dir = Random::random() * M_PI * 2;
    }

    if (Random::random() < 0.2 * deltaTime || reset) {
        target_speed = Random::random() * (Random::random() < 0.2f ? -1.0f : 1.0f);
    }

    if (std::abs(collision_force[0]) > 0.1f) {
        target_dir = dir + M_PI;
    }
    return {target_speed, dir_change};
}

// Step method implementation
void Fish::simulate(float dt) {
    std::tuple<float, float> action = random_policy(dt);
    float accel = std::get<0>(action) * (std::get<0>(action) >= 0 ? 1 : -0.2) * Fish::maxAccel;

    dir_change = std::get<1>(action) * Fish::maxAng;
    dir_change_avg = dir_change_avg * 0.9 + dir_change * 0.1;
    dir += dir_change * dt;

    getEnv()->getPoint(body[0])->force += accel * vec(dir);
}

void Fish::render(VertexManager &viewer) {
    auto finColour = interpolate(colour, sf::Color(255, 255, 255), 0.3f);
    auto bodyFinColour = interpolate(colour, sf::Color(0, 0, 0), 0.4f);

    Point* body0 = getEnv()->getPoint(body[0]);
    Point* body1 = getEnv()->getPoint(body[0]);
    Point* body2 = getEnv()->getPoint(body[0]);
    Point* body3 = getEnv()->getPoint(body[0]);

    float a2 = body1->angleTo(*body0);
    float a3 = body2->angleTo(*body1);
    float a4 = body3->angleTo(*body2);

    float2 midPos = body0->pos + vec(dir);

    float2 p1 = midPos - body0->pos;
    float2 p2 = body2->pos - body3->pos;
    float skew = clockwiseAngleDiff(p1, p2);

    float headToMid = angleDiff(dir, a3);
    float totalCurvature = angleDiff(dir, a4);

    drawVentralFins(viewer, body0->pos, a2, body0->mass, finColour, skew);

    float angle = std::atan2(body1->pos.y - body2->pos.y, body1->pos.x - body2->pos.x);
    drawVentralFins(viewer, (body2->pos + body1->pos) * 0.5f, angle, body2->mass * 1.2f, finColour, skew);

    // Construct body polygon from body points
    for (size_t i = 0; i < body.size(); ++i) {
        Point* bodyi = getEnv()->getPoint(body[i]);
        Point* bodyi1 = getEnv()->getPoint(body[i + 1]);
        bodyi->render(viewer, colour);

        if (i + 1 >= body.size()) continue;
        auto poly1 = findPerpendicularPoints(*bodyi, *bodyi1, bodyi->mass, bodyi1->mass);
        viewer.addPolygon(poly1, sf::Color(colour));
    }

    std::vector<float2> dorsalFinPoints = {
            make_float2((body0->pos.x + body1->pos.x) * 0.5f,
                        (body0->pos.y + body1->pos.y) * 0.5f),
            body1->pos, body2->pos
    };


    angle = body3->angleTo(*body2) + M_PI;
    float tailWidth = 0.5f - std::clamp(std::abs(totalCurvature * 0.3f), 0.0f, 0.3f);

    std::vector<float2> tailFinPoints = {body2->pos * 0.6f + body3->pos * 0.4f};

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle + tailWidth) * 10,
                               tailFinPoints[0].y + sin(angle + tailWidth) * 10);

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle) * 20,
                               tailFinPoints[0].y + sin(angle) * 20);

    tailFinPoints.emplace_back(tailFinPoints[0].x + cos(angle - tailWidth) * 10,
                               tailFinPoints[0].y + sin(angle - tailWidth) * 10);

    viewer.addPolygon(tailFinPoints, finColour);

    // Eyes
    float2 eye1 = body0->pos + vec(a2 + M_PI * 0.5f + 0.3f) * size * 0.7f;
    viewer.addCircle(eye1, size * 0.5f, sf::Color(255, 255, 255));
    float2 eye2 = body0->pos + vec(a2 - M_PI * 0.5f - 0.3f) * size * 0.7f;
    viewer.addCircle(eye2, size * 0.5f, sf::Color(255, 255, 255));

    float2 iris = vec(a2 + dir_change_avg * 10) * size * 0.25;
    viewer.addCircle(eye1 + iris, size * 0.3f, sf::Color(0, 0, 0));
    viewer.addCircle(eye2 + iris, size * 0.3f, sf::Color(0, 0, 0));
}

void Fish::drawVentralFins(VertexManager &viewer, float2 pos, float angle, float finSize, const sf::Color &finColour,
                           float skew) {
    const float deg90 = M_PI / 2;
    skew = sin(skew);
    float skewLeft = skew > 0 ? 0 : std::abs(skew + 0.2f);
    float skewRight = skew < 0 ? 0 : std::abs(skew - 0.2f);

    viewer.addPolygon({
                          pos + vec(angle + deg90) * finSize,
                          pos + vec(angle + deg90 * (2 + 0.5f * skewLeft)) * finSize * 3,
                          pos + vec(angle + deg90 * (1.5f + 0.3f * skewLeft)) * finSize * 5
                      }, finColour);

    viewer.addPolygon({
                          pos + vec(angle - deg90) * finSize,
                          pos + vec(angle - deg90 * (2 + 0.5f * skewRight)) * finSize * 3,
                          pos + vec(angle - deg90 * (1.5f + 0.3f * skewRight)) * finSize * 5
                      }, finColour);
}

FishTank *Fish::getEnv() {
    return dynamic_cast<FishTank *>(Individual::getEnv());
}

#endif