#include "../verlet/point.hpp"
#include "../verlet/constraints.cu"
#include <cuda_runtime.h>
#include "updatePoints.hpp"
#include "../utils/floatOps.hpp"
#include "error_check.cu"
#include "utils/GPUVector.hpp"

__global__ void updatePointsKernel(Point* points, int numParticles, const sf::FloatRect& bounds, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numParticles) {
        Point& point = points[index];
        constrainPosition(point, bounds);
        point.update(dt);
    }
}


__global__ void constrainDistancesKernel(Connection* connections, int numConnections, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numConnections) {
        Connection& connection = connections[index];
        constrainDistance(*connection.a, *connection.b, connection.distance, 0.9f);
    }
}

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Connection>& connections,
                  const sf::FloatRect& bounds,
                  float dt) {
    int numPoints = points.size();
    int numConnections = connections.size();

    sf::FloatRect* d_bounds;
    size_t boundsSize = sizeof(sf::FloatRect);
    cudaMalloc(&d_bounds, boundsSize);
    cudaMemcpy(d_bounds, &bounds, boundsSize, cudaMemcpyHostToDevice);

    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    updatePointsKernel<<<numBlocks, blockSize>>>(points.deviceData(),
                                                 numPoints, *d_bounds, dt);

    numBlocks = (numConnections + blockSize - 1) / blockSize;
    constrainDistancesKernel<<<numBlocks, blockSize>>>(connections.deviceData(),
                                                       numConnections, dt);

    points.syncToHost();
    cudaFree(d_bounds);
    //cudaCheckError();
}