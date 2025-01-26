#include <modules/physics/point.hpp>
#include "../physics/constraints.cu"
#include "cuda_runtime.h"
#include <modules/cuda/updatePoints.hpp>
#include <SFML/Graphics.hpp>
#include <modules/cuda/logging.hpp>
#include <modules/utils/floatOps.hpp>

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void updatePointsKernel(GPUVector<Point> points, float dt, sf::FloatRect *bounds) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= points.size()) return;

    Point &point = points[index];
    point.update(dt);
    constrainPosition(point, *bounds);
}


__global__ void constrainDistancesKernel(GPUVector<Point> points, GPUVector<CellLink> cellLinks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cellLinks.size()) return;

    const CellLink& cellLink = cellLinks[index];
    constrainDistance(points[cellLink.p1], points[cellLink.p2], cellLink.length, 0.2f);
}

/*
__global__ void updateParentChildLinkKernel(Point *points, ParentChildLink *angles, int numAngles, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numAngles) {
        ParentChildLink &pcl = angles[index];
        float parentAngle = FastMath::atan2f(points[pcl.parentEndPoint].pos.y -
                                                    points[pcl.parentStartPoint].pos.y,
                                                    points[pcl.parentEndPoint].pos.x -
                                                    points[pcl.parentStartPoint].pos.x);
        constrainAngle(points[pcl.startPoint], points[pcl.endPoint], parentAngle + pcl.targetAngle, pcl.stiffness, dt);
        points[pcl.startPoint].pos = points[pcl.parentStartPoint].pos + rotate(pcl.pointOnParent, parentAngle);
    }
}
*/

/*
__global__ void computeCollisions(GPUVector<Point> points) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (a >= points.size() || b >= points.size() || a >= b) return;

    Point& pointA = points[a];
    Point& pointB = points[b];

    if (pointA.entityID == pointB.entityID) return;

    double2 posA = pointA.pos;
    double2 posB = pointB.pos;
    float distance = distanceBetween(posA, posB);
    float minDistance = pointA.radius + pointB.radius;

    if (distance > minDistance || fabs(distance - minDistance) < 1e-6f) return;

    // Calculate the vector to separate the points
    float overlap = minDistance - distance;
    double2 direction = posA - posB;
    float length = sqrt(sum(direction * direction));

    // Adjust positions to resolve overlap (optional for immediate collision resolution)
    double2 adjustment = direction * overlap * 0.5f;
    atomicAddDouble(&pointA.pos.x, adjustment.x);
    atomicAddDouble(&pointA.pos.y, adjustment.y);
    atomicAddDouble(&pointB.pos.x, -adjustment.x);
    atomicAddDouble(&pointB.pos.y, -adjustment.y);

    if (length < 1e-6f) return;

    // Normalize the direction
    direction.x /= length;
    direction.y /= length;

    // Retrieve velocities and masses
    double massA = pointA.radius * pointA.radius;
    double massB = pointB.radius * pointB.radius;

    if (massA + massB < 1e-6f) return;

    // Compute the force (based on overlap) and split proportionally by mass
    double forceMagnitude = overlap / (massA + massB);

    // Apply forces in the opposite directions
    double2 forceA = make_double2(-direction.x * forceMagnitude * massB, -direction.y * forceMagnitude * massB);
    double2 forceB = make_double2(direction.x * forceMagnitude * massA, direction.y * forceMagnitude * massA);

    // Add forces to points
    atomicAddDouble(&pointA.force.x, forceA.x);
    atomicAddDouble(&pointA.force.y, forceA.y);
    atomicAddDouble(&pointB.force.x, forceB.x);
    atomicAddDouble(&pointB.force.y, forceB.y);
}*/

// Soft collisions (resistive forces)
__global__ void computeCollisions(GPUVector<Point> points) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (a >= points.size() || b >= points.size() || a >= b) return;

    Point& pointA = points[a];
    Point& pointB = points[b];

    if (pointA.entityID == pointB.entityID) return;

    double2 posA = pointA.pos;
    double2 posB = pointB.pos;
    float distance = distanceBetween(posA, posB);
    float minDistance = pointA.radius + pointB.radius;

    if (distance >= minDistance) return;

    // Calculate overlap and resistive force
    float overlap = minDistance - distance;
    float resistiveForceMagnitude = overlap * overlap * 0.01;

    double2 direction = posA - posB;
    float length = sqrt(sum(direction * direction));

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

void updatePoints(GPUVector<Point>& points,
                  GPUVector<CellLink>& cellLinks,
                  CGPUValue<sf::FloatRect> &bounds,
                  float dt) {

    int blockSize = 256; // Number of threads per block
    int numBlocks = (points.size() + blockSize - 1) / blockSize;

    if (points.size() == 0) return;

    // Update the points
    updatePointsKernel<<<numBlocks, blockSize>>>(points, dt, bounds.deviceData());

    // Update connections
    numBlocks = (cellLinks.size() + blockSize - 1) / blockSize;
    constrainDistancesKernel<<<numBlocks, blockSize>>>(points, cellLinks);

    dim3 threadsPerBlock(32, 32);
    dim3 numCollisionBlocks((points.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (points.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    computeCollisions<<<numCollisionBlocks, threadsPerBlock>>>(points);
}

__global__ void movePointKernel(GPUVector<Point> points, int pointIndex, const float2 newPos, int* entityID) {
    Point &point = points[pointIndex];
    point.pos.x = newPos.x;
    point.pos.y = newPos.y;
    *entityID = point.entityID;
}

int movePoint(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos) {
    int* entityID = nullptr;
    cudaLog(cudaMalloc(&entityID, sizeof(int)));
    movePointKernel<<<1, 1>>>(points, pointIndex, {newPos.x, newPos.y}, entityID);
    int entityIDHost;
    cudaMemcpy(&entityIDHost, entityID, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(entityID);
    return entityIDHost;
}
