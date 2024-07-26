#include "../verlet/point.hpp"
#include "../verlet/constraints.cu"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "updatePoints.hpp"

__global__ void updatePointsKernel(Point* points, int numParticles, const sf::FloatRect& bounds, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numParticles) {
        Point& point = points[index];
        constrainPosition(point, bounds);
        point.update(dt);
    }
}

void updatePointsOnGPU(std::vector<Point>& points, const sf::FloatRect& bounds, float dt) {
    Point* d_points;
    sf::FloatRect* d_bounds;

    size_t pointsSize = points.size() * sizeof(Point);
    size_t boundsSize = sizeof(float2);

    cudaMalloc(&d_points, pointsSize);
    cudaMalloc(&d_bounds, boundsSize);

    cudaMemcpy(d_points, points.data(), pointsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bounds, &bounds, boundsSize, cudaMemcpyHostToDevice);

    int numPoints = points.size();
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    updatePointsKernel<<<numBlocks, blockSize>>>(d_points, numPoints, *d_bounds, dt);

    cudaMemcpy(points.data(), d_points, pointsSize, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_bounds);
}