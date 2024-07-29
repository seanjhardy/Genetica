#include "point.hpp"
#include <cmath>
#include "../utils/floatOps.hpp"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

__host__ __device__ inline void constrainDistance(Point& point1, Point& point2, float distance, float factor = 1.0f) {
    float currentDistance = point1.distanceTo(point2);
    float deltaDistance = factor * (distance - currentDistance);

    if (currentDistance == 0) {
        currentDistance += 1e-8f; // Avoid division by zero
    }

    if (std::abs(deltaDistance) < 0.001) {
        return; // No significant change needed
    }

    float2 delta = (point2.pos - point1.pos) * deltaDistance / currentDistance;

    float massRatio = point1.mass / (point1.mass + point2.mass);

    point1.pos -= delta * (1 - massRatio);
    point2.pos += delta * massRatio;
}

__host__ __device__ inline void constrainAngle(Point& point1, Point& point2, Point& point3, float desiredAngle, float factor = 0.001f) {
    float angle1 = point2.angleTo(point1);
    float angle2 = point2.angleTo(point3);
    float currentAngle = angle2 - angle1;

    // Normalize the angle difference
    float deltaAngle = desiredAngle - currentAngle;
    if (deltaAngle <= -M_PI) {
        deltaAngle += 2 * M_PI;
    } else if (deltaAngle >= M_PI) {
        deltaAngle -= 2 * M_PI;
    }

    if (std::abs(deltaAngle) * factor < 1e-3) {
        return; // No significant change needed
    }

    point1.rotate(point2.pos, factor * deltaAngle);
    point3.rotate(point2.pos, factor * deltaAngle);
}

__host__ __device__ inline float constrainPosition(Point& point, sf::FloatRect bounds) {
    float updateDist = 0.0f;
    float minMax[4] = {bounds.left + point.mass,
                       bounds.width - point.mass,
                       bounds.top + point.mass,
                       bounds.height - point.mass};

    if (point.pos.x < minMax[0]) {
        updateDist += std::abs(point.pos.x - minMax[0]);
        point.pos.x = minMax[0];
    }
    if (point.pos.x > minMax[1]) {
        updateDist += std::abs(point.pos.x - minMax[1]);
        point.pos.x = minMax[1];
    }

    if (point.pos.y < minMax[2]) {
        updateDist += std::abs(point.pos.y - minMax[2]);
        point.pos.y = minMax[2];
    }
    if (point.pos.y > minMax[3]) {
        updateDist += std::abs(point.pos.y - minMax[3]);
        point.pos.y = minMax[3];
    }

    return updateDist;
}
