#include <modules/physics/point.hpp>
#include <modules/utils/floatOps.hpp>
#include <modules/utils/print.hpp>
#include "cmath"
#include <modules/utils/GPU/atomicOps.hpp>

 __device__ inline void constrainDistance(Point& pointA, Point& pointB, double distance) {
    double currentDistance = pointA.distanceTo(pointB);
    double deltaDistance = 0.5 * (distance - currentDistance);

    if (currentDistance == 0) {
        currentDistance += 1e-6f; // Avoid division by zero
    }

    if (std::abs(deltaDistance) < 1e-7) {
        return; // No significant change needed
    }

    double2 delta = (pointB.pos - pointA.pos) * deltaDistance / currentDistance;

    double pointAMass = pointA.radius * pointA.radius;
    double pointBMass = pointB.radius * pointB.radius;
    double massRatio = pointAMass / (pointAMass + pointBMass);
    atomicAddDouble(&pointA.deltaPos.x, - delta.x * (1 - massRatio));
    atomicAddDouble(&pointA.deltaPos.y, - delta.y * (1 - massRatio));
    atomicAddDouble(&pointB.deltaPos.x, delta.x * massRatio);
    atomicAddDouble(&pointB.deltaPos.y, delta.y * massRatio);
    atomicAdd(&pointA.connections, 1);
    atomicAdd(&pointB.connections, 1);
}

__device__ inline void constrainAngle(Point& pointA, Point& pointB, float prevAngle, float targetAngle, float stiffness) {
    float currentAngle = dir(pointA.getPos(), pointB.getPos());
    float angleChange = currentAngle - prevAngle;
    pointA.angle += angleChange;
    pointB.angle += angleChange;
    return;
    
    float angleDiff = (pointA.angle + targetAngle) - currentAngle;

    // Normalize angleDiff to the interval [-pi, pi]
    while(angleDiff > M_PI)  angleDiff -= 2 * M_PI;
    while(angleDiff < -M_PI) angleDiff += 2 * M_PI;

    // Calculate the correction based on stiffness.
    float correction = stiffness * angleDiff;
    float halfCorrection = correction / 2.0f;

    // Find the midpoint between A and B.
    float pointAMass = pointA.radius * pointA.radius;
    float pointBMass = pointB.radius * pointB.radius;
    double massRatio = pointBMass / (pointAMass + pointBMass);
    double2 centerOfMass = pointA.pos + (pointB.pos - pointA.pos) * massRatio;

    // Rotate A around center by -halfCorrection.
    //pointA.pos = centerOfMass + rotate(pointA.pos - centerOfMass, -halfCorrection);
    // Rotate B around center by +halfCorrection.
    //pointB.pos = centerOfMass + rotate(pointB.pos - centerOfMass, halfCorrection);

    // Update the points' angles.
    pointA.angle -= halfCorrection;
    pointB.angle += halfCorrection;
}

__host__ __device__ inline float constrainPosition(Point& point, sf::FloatRect bounds) {
    float updateDist = 0.0f;
    float minMax[4] = {bounds.left + (float)point.radius,
                       bounds.width - (float)point.radius,
                       bounds.top + (float)point.radius,
                       bounds.height - (float)point.radius};

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