#include <modules/physics/point.hpp>
#include "../physics/constraints.cu"
#include "cuda_runtime.h"
#include <modules/cuda/updatePoints.hpp>
#include <SFML/Graphics.hpp>

__global__ void updatePointsKernel(GPUVector<Point> points, sf::FloatRect *bounds) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= points.size()) return;

    Point &point = points[index];
    point.update();
    constrainPosition(point, *bounds);
}

// Soft collisions (resistive forces)
__global__ void computeCollisions(GPUVector<Point> points) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (a >= points.size() || b >= points.size() || a >= b) return;

    Point& pointA = points[a];
    Point& pointB = points[b];

    //if (pointA.entityID == pointB.entityID) return;

    double2 posA = pointA.pos;
    double2 posB = pointB.pos;
    float distance = distanceBetween(posA, posB);
    float minDistance = pointA.radius + pointB.radius;

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

void updatePoints(GPUVector<Point>& points,
                  CGPUValue<sf::FloatRect> &bounds) {

    int blockSize = 256; // Number of threads per block
    int numBlocks = 0;

    if (points.size() == 0) return;

    // Update the points
    numBlocks = (points.size() + blockSize - 1) / blockSize;
    updatePointsKernel<<<numBlocks, blockSize>>>(points, bounds.deviceData());

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
