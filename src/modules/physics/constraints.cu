#include <modules/physics/point.hpp>
#include <modules/utils/operations.hpp>
#include <modules/utils/print.hpp>
#include "modules/utils/GPU/mathUtils.hpp"
#include "cmath"
#include <modules/utils/GPU/atomicOps.hpp>
#include "cuda_runtime.h"

__device__ inline void constrainDistance(Point& pointA, Point& pointB, double distance) {
    double currentDistance = pointA.distanceTo(pointB);
    double deltaDistance = distance - currentDistance;

    if (currentDistance == 0) {
        currentDistance += 1e-6f; // Avoid division by zero
    }

    if (std::abs(deltaDistance) < 1e-7) {
        return; // No significant change needed
    }

    double2 delta = 0.5 * deltaDistance * (pointB.pos - pointA.pos) / currentDistance;

    double pointAMass = pointA.radius * pointA.radius;
    double pointBMass = pointB.radius * pointB.radius;
    double massRatio = pointAMass / (pointAMass + pointBMass);
    double2 forceA = delta * (massRatio - 1);
    double2 forceB = delta * massRatio;

    atomicAddDouble(&pointA.force.x, forceA.x);
    atomicAddDouble(&pointA.force.y, forceA.y);
    atomicAddDouble(&pointB.force.x, forceB.x);
    atomicAddDouble(&pointB.force.y, forceB.y);
}

__device__ inline void constrainMinDistance(Point& pointA, Point& pointB, float minDistance) {
    double2 posA = pointA.pos;
    double2 posB = pointB.pos;
    float distance = distanceBetween(posA, posB);

    if (distance >= minDistance) return;

    // Calculate overlap and resistive force
    float overlap = minDistance - distance;
    float resistiveForceMagnitude = overlap * overlap * 0.01;

    double2 direction = posA - posB;
    float length = magnitude(direction);

    if (length < 1e-6f) return;

    // Normalize direction
    direction.x /= length;
    direction.y /= length;

    // Apply resistive force proportionally
    double2 forceA = direction * resistiveForceMagnitude;
    double2 forceB = direction * -resistiveForceMagnitude;

    atomicAddDouble(&pointA.force.x, forceA.x);
    atomicAddDouble(&pointA.force.y, forceA.y);
    atomicAddDouble(&pointB.force.x, forceB.x);
    atomicAddDouble(&pointB.force.y, forceB.y);
}

__device__ inline void constrainAngle(Point& pointA, Point& pointB, float angleFromA, float angleFromB,
                                      float stiffness) {
    // 1. Compute the current angle of the AB link.
    float theta_AB = dir(pointA.getPos(), pointB.getPos());

    // 2. Compute the two target directions for the link:
    //    For A, the desired outgoing direction is its own angle plus the offset.
    float target_A = pointA.angle + angleFromA;
    //    For B, the desired incoming direction (for the link, which is opposite to its outgoing side)
    //    is the opposite of (pointB.angle + angleFromB). That is, if you add π to B's desired direction,
    //    you get the target for the AB link.
    float target_B = pointB.angle + angleFromB;

    // 3. Determine how far the current link is from each target.
    //    errorA is how much the link’s angle must change to match A’s target.
    float errorA = normAngle(target_A - theta_AB);
    //    errorB is how much it must change to match B’s target.
    float errorB = normAngle(target_B - theta_AB);

    // 4. Ideally both errors would be the same. If not, we let the link itself rotate by the average.
    float errorMean = (errorA + errorB) * 0.5f;
    // 5. The remaining error for each point is the difference between its own error and the mean.
    float deltaA_angle = errorA;
    float deltaB_angle = errorB;

    // 6. Scale corrections by stiffness.
    deltaA_angle *= stiffness;
    deltaB_angle *= stiffness;

    // 7. Rotate both points around the weighted center-of-mass by the average error.
    float massA = pointA.radius * pointA.radius;
    float massB = pointB.radius * pointB.radius;
    float totalMass = massA + massB;
    double2 centerOfMass = (pointA.pos * massA + pointB.pos * massB) / totalMass;

    //pointA.rotate(centerOfMass, errorMean * 0.0001f);
    //pointB.rotate(centerOfMass, errorMean * 0.0001f);

    // 8. Finally, adjust the stored angles so that A and B “pay” for the remainder of the correction.
    atomicAdd(&pointA.angle, deltaA_angle);
    atomicAdd(&pointB.angle, deltaB_angle);
}


__host__ __device__ inline float constrainPosition(Point& point, sf::FloatRect bounds) {
    float updateDist = 0.0f;
    float minMax[4] = {
        bounds.left + (float)point.radius,
        bounds.width - (float)point.radius,
        bounds.top + (float)point.radius,
        bounds.height - (float)point.radius
    };

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
    }
    else if (pX >= maxX) {
        pX = maxX;
    }
    else {
        float recCenterX = (minX + maxX) / 2.0f;
        xOverlap = (pX < recCenterX) ? (minX - pX) : (maxX - pX);
    }

    // Compute yOverlap
    if (pY < minY) {
        pY = minY;
    }
    else if (pY >= maxY) {
        pY = maxY;
    }
    else {
        float recCenterY = (minY + maxY) / 2.0f;
        yOverlap = (pY < recCenterY) ? (minY - pY) : (maxY - pY);
    }

    // Check collision
    if ((x - pX) * (x - pX) + (y - pY) * (y - pY) < size * size) {
        float contactX = pX;
        float contactY = pY;

        if (fabsf(xOverlap) < fabsf(yOverlap)) {
            contactX += xOverlap;
        }
        else if (fabsf(yOverlap) < fabsf(xOverlap)) {
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
