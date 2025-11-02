#include <modules/physics/point.hpp>
#include <modules/utils/vector_functions.hpp>
#include <modules/utils/operations.hpp>
#include <modules/utils/print.hpp>
#include <modules/utils/gpu/mathUtils.hpp>
#include "cmath"

inline void constrainDistance(Point& pointA, Point& pointB, double distance) {
    double currentDistance = pointA.distanceTo(pointB);
    double deltaDistance = distance - currentDistance;

    if (currentDistance == 0) {
        currentDistance += 1e-6f; // Avoid division by zero
    }

    if (std::abs(deltaDistance) < 1e-7) {
        return; // No significant change needed
    }

    float2 delta = 0.5 * deltaDistance * (pointB.pos - pointA.pos) / currentDistance;

    float pointAMass = pointA.radius * pointA.radius;
    float pointBMass = pointB.radius * pointB.radius;
    float massRatio = pointAMass / (pointAMass + pointBMass);
    float2 forceA = delta * (massRatio - 1);
    float2 forceB = delta * massRatio;

    pointA.force.s[0] += forceA.s[0];
    pointA.force.s[1] += forceA.s[1];
    pointB.force.s[0] += forceB.s[0];
    pointB.force.s[1] += forceB.s[1];
}

inline void constrainMinDistance(Point& pointA, Point& pointB, float minDistance) {
    float2 posA = pointA.pos;
    float2 posB = pointB.pos;
    float distance = distanceBetween(posA, posB);

    if (distance >= minDistance) return;

    // Calculate overlap and resistive force
    float overlap = minDistance - distance;
    float resistiveForceMagnitude = overlap * overlap * 0.01;

    float2 direction = posA - posB;
    float length = magnitude(direction);

    if (length < 1e-6f) return;

    // Normalize direction
    direction.s[0] /= length;
    direction.s[1] /= length;

    // Apply resistive force proportionally
    float2 forceA = direction * resistiveForceMagnitude;
    float2 forceB = direction * -resistiveForceMagnitude;

    pointA.force.s[0] += forceA.s[0];
    pointA.force.s[1] += forceA.s[1];
    pointB.force.s[0] += forceB.s[0];
    pointB.force.s[1] += forceB.s[1];
}

inline void constrainAngle(Point& pointA, Point& pointB, float angleFromA, float angleFromB,
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
    float2 centerOfMass = (pointA.pos * massA + pointB.pos * massB) / totalMass;

    //pointA.rotate(centerOfMass, errorMean * 0.0001f);
    //pointB.rotate(centerOfMass, errorMean * 0.0001f);

    // 8. Finally, adjust the stored angles so that A and B “pay” for the remainder of the correction.
    pointA.angle += deltaA_angle;
    pointB.angle += deltaB_angle;
}


inline float constrainPosition(Point& point, sf::FloatRect bounds) {
    float updateDist = 0.0f;
    float minMax[4] = {
        bounds.left + (float)point.radius,
        bounds.width - (float)point.radius,
        bounds.top + (float)point.radius,
        bounds.height - (float)point.radius
    };

    if (point.pos.s[0] < minMax[0]) {
        point.prevPos.s[0] = point.pos.s[0];
        point.pos.s[0] = minMax[0];
    }
    if (point.pos.s[0] > minMax[1]) {
        point.prevPos.s[0] = point.pos.s[0];
        point.pos.s[0] = minMax[1];
    }

    if (point.pos.s[1] < minMax[2]) {
        point.prevPos.s[1] = point.pos.s[1];
        point.pos.s[1] = minMax[2];
    }
    if (point.pos.s[1] > minMax[3]) {
        point.prevPos.s[1] = point.pos.s[1];
        point.pos.s[1] = minMax[3];
    }

    return updateDist;
}

inline void checkCollisionCircleRec(Point& circle, Point& rect) {
    float xOverlap = 0.0f;
    float yOverlap = 0.0f;
    float x = circle.pos.s[0];
    float y = circle.pos.s[1];
    float pX = circle.pos.s[0];
    float pY = circle.pos.s[1];
    float size = circle.radius;
    float minX = rect.pos.s[0] - rect.radius / 2;
    float maxX = rect.pos.s[0] + rect.radius / 2;
    float minY = rect.pos.s[1] - rect.radius / 2;
    float maxY = rect.pos.s[1] + rect.radius / 2;

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

        circle.pos.s[0] += normalX * depth * 1.1;
        circle.pos.s[1] += normalY * depth * 1.1;
    }
}
