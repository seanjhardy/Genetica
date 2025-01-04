#include <modules/physics/point.hpp>
#include <modules/utils/floatOps.hpp>
#include <modules/utils/print.hpp>
#include "cmath"
#include <modules/utils/mathUtils.hpp>

__host__ __device__ inline void constrainDistance(Point& point1, Point& point2, float distance, float factor = 1.0f) {
    float currentDistance = point1.distanceTo(point2);
    float deltaDistance = factor * (distance - currentDistance);

    if (currentDistance == 0) {
        currentDistance += 1e-5f; // Avoid division by zero
    }

    if (std::abs(deltaDistance) < 0.01) {
        return; // No significant change needed
    }

    float2 delta = (point2.pos - point1.pos) * deltaDistance / currentDistance;

    float massRatio = point1.radius / (point1.radius + point2.radius);
    point1.pos -= delta * (1 - massRatio) * 0.01;
    point2.pos += delta * massRatio * 0.01;
}

__host__ __device__ inline void constrainAngle(Point& point1, Point& point2, float targetAngle, float stiffness, float dt) {
    float length = point1.distanceTo(point2);
    float2 newPos = point1.pos + vec(targetAngle) * length;
    point2.prevPos += (newPos - point2.pos) * stiffness * dt * 0.99;
    point2.pos += (newPos - point2.pos) * stiffness * dt;
}

__host__ __device__ inline float constrainPosition(Point& point, sf::FloatRect bounds) {
    float updateDist = 0.0f;
    float minMax[4] = {bounds.left + point.radius,
                       bounds.width - point.radius,
                       bounds.top + point.radius,
                       bounds.height - point.radius};

    if (point.pos.x < minMax[0]) {
        point.prevPos.x = point.pos.x;
        point.pos.x = minMax[0];
    }
    if (point.pos.x > minMax[1]) {
        point.prevPos.x = point.pos.x;
        point.pos.x = minMax[1];
    }

    if (point.pos.y < minMax[2]) {
        point.prevPos.y = point.pos.y;
        point.pos.y = minMax[2];
    }
    if (point.pos.y > minMax[3]) {
        point.prevPos.y = point.pos.y;
        point.pos.y = minMax[3];
    }

    return updateDist;
}

__host__ __device__ inline void checkCollisionCircleRec(Point& circle, Point& rect) {
    float xOverlap = 0.0f;
    float yOverlap = 0.0f;
    float x = circle.pos.x;
    float y = circle.pos.y;
    float pX = circle.pos.x;
    float pY = circle.pos.y;
    float size = circle.radius;
    float minX = rect.pos.x - rect.radius / 2;
    float maxX = rect.pos.x + rect.radius / 2;
    float minY = rect.pos.y - rect.radius / 2;
    float maxY = rect.pos.y + rect.radius / 2;

    // Check if the circle is completely outside the rectangle
    if (pX < minX - size || pX > maxX + size || pY < minY - size || pY > maxY + size) {
        return;
    }

    // Compute xOverlap
    if (pX < minX) {
        pX = minX;
    } else if (pX >= maxX) {
        pX = maxX;
    } else {
        float recCenterX = (minX + maxX) / 2.0f;
        xOverlap = (pX < recCenterX) ? (minX - pX) : (maxX - pX);
    }

    // Compute yOverlap
    if (pY < minY) {
        pY = minY;
    } else if (pY >= maxY) {
        pY = maxY;
    } else {
        float recCenterY = (minY + maxY) / 2.0f;
        yOverlap = (pY < recCenterY) ? (minY - pY) : (maxY - pY);
    }

    // Check collision
    if ((x - pX) * (x - pX) + (y - pY) * (y - pY) < size * size) {
        float contactX = pX;
        float contactY = pY;

        if (fabsf(xOverlap) < fabsf(yOverlap)) {
            contactX += xOverlap;
        } else if (fabsf(yOverlap) < fabsf(xOverlap)) {
            contactY += yOverlap;
        }

        float penX = x - contactX;
        float penY = y - contactY;
        float pLength = sqrtf(penX * penX + penY * penY);
        float depth = size - pLength;
        float normalX = penX / pLength;
        float normalY = penY / pLength;

        if (xOverlap and yOverlap) {
            depth -= size * 2.0f;
        }

        circle.pos.x += normalX * depth * 1.1;
        circle.pos.y += normalY * depth * 1.1;
    }
}