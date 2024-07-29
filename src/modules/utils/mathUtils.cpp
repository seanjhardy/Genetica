#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <cmath>
#include <utility> // For std::pair
#include "../verlet/point.hpp"
#include "floatOps.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline std::vector<float2> findPerpendicularPoints(const Point& point1, const Point& point2, float r1, float r2) {
    float x1 = point1.pos.x, y1 = point1.pos.y;
    float x2 = point2.pos.x, y2 = point2.pos.y;

    // Ensure the points are not identical to avoid division by zero in atan2
    if (point1.pos == point2.pos) {
        x1 += 0.0001;
    }

    double angle = std::atan2(y2 - y1, x2 - x1);
    float anglePlus90 = angle + M_PI / 2;
    float angleMinus90 = angle - M_PI / 2;

    // Calculate the coordinates of the points on each circle's circumference
    float2 point1Circle1 = point1.pos + vec(anglePlus90) * r1;
    float2 point2Circle1 = point1.pos + vec(angleMinus90) * r1;
    float2 point1Circle2 = point2.pos + vec(angleMinus90) * r2;
    float2 point2Circle2 = point2.pos + vec(anglePlus90) * r2;

    return {point1Circle1, point2Circle1, point1Circle2, point2Circle2};
}

inline float getVelocity(const Point& point) {
    return std::sqrt(std::pow(point.pos.x - point.prevPos.x, 2) + std::pow(point.pos.y - point.prevPos.y, 2));
}

inline sf::Color interpolate(const sf::Color c1, const sf::Color c2, float x) {
    int r = static_cast<int>(c1.r + (c2.r - c1.r) * x);
    int g = static_cast<int>(c1.g + (c2.g - c1.g) * x);
    int b = static_cast<int>(c1.b + (c2.b - c1.b) * x);
    return sf::Color(r, g, b);
}

inline float clamp(float min_val, float x, float max_val) {
    return std::max(min_val, std::min(x, max_val));
}

inline float normAngle(float angle) {
    return std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
}

inline float angleDiff(float angle1, float angle2, bool norm = true) {
    float diff = angle2 - angle1;
    if (norm) {
        diff = normAngle(diff);
    }
    return diff;
}

inline float clockwiseAngleDiff(const float2& p1, const float2& p2) {
    return std::atan2(p1.x * p2.y - p1.y * p2.x,
                      p1.x * p2.x + p1.y * p2.y);
}

inline std::vector<std::pair<int, int>> bezier(int x0, int y0, int x1, int y1, int x2, int y2, int num_points = 10) {
    auto bezierInterpolation = [](float t, int p0, int p1, int p2) {
        float x = std::pow(1 - t, 2) * p0 + 2 * (1 - t) * t * p1 + std::pow(t, 2) * p2;
        return static_cast<int>(x);
    };

    std::vector<std::pair<int, int>> points;
    for (int i = 0; i <= num_points; ++i) {
        float t = static_cast<float>(i) / num_points;
        int x = bezierInterpolation(t, x0, x1, x2);
        int y = bezierInterpolation(t, y0, y1, y2);
        points.emplace_back(x, y);
    }
    return points;
}

inline float smoothAngle(float angle1, float angle2, float tolerance = 90) {
    float diff = angleDiff(angle1, angle2);
    tolerance = tolerance * M_PI / 180;

    if (std::abs(diff) < tolerance) {
        if (diff > 0) {
            return normAngle(angle2 - tolerance);
        } else {
            return normAngle(angle2 + tolerance);
        }
    }
    return angle1;
}


#endif // MATHUTILS_HPP