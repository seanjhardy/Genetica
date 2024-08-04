#include "modules/verlet/point.hpp"
#include "../verlet/constraints.cu"
#include "cuda_runtime.h"
#include "modules/cuda/updatePoints.hpp"
#include "SFML/Graphics.hpp"
#include "geneticAlgorithm/environments/fishTank/rock.hpp"

__global__ void updatePointsKernel(Point* points, int numParticles, float dt, sf::FloatRect* bounds) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numParticles) {
        Point& point = points[index];
        point.update(dt);
        constrainPosition(point, *bounds);
    }
}


__global__ void constrainDistancesKernel(Connection* connections, int numConnections) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numConnections) {
        Connection& connection = connections[index];
        constrainDistance(*connection.a, *connection.b, connection.distance, 0.9f);
    }
}

__global__ void constrainAnglesKernel(AngleConstraint* angles, int numAngles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numAngles) {
        AngleConstraint& angle = angles[index];
        float targetAngle = angle.targetAngle + atan2(angle.b->pos.y - angle.a->pos.y, angle.b->pos.x - angle.a->pos.x);
        constrainAngle(*angle.a, *angle.b, targetAngle, angle.stiffness);
    }
}

__global__ void computeCollisions(Point* points, int numPoints, Rock* rocks, int numRocks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < numPoints && y < numRocks) {
        Point& point = points[x];
        auto rock = static_cast<Point>(rocks[y]);
        checkCollisionCircleRec(point, rock);
    }
}

void updatePoints(GPUVector<Point>& points,
                  GPUVector<Connection>& connections,
                  GPUVector<AngleConstraint>& angles,
                  const sf::FloatRect& bounds,
                  float dt) {
    int numPoints = points.size();
    int numConnections = connections.size();
    int numAngles = angles.size();

    sf::FloatRect* d_bounds;
    size_t boundsSize = sizeof(sf::FloatRect);
    cudaMalloc(&d_bounds, boundsSize);
    cudaMemcpy(d_bounds, &bounds, boundsSize, cudaMemcpyHostToDevice);

    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    updatePointsKernel<<<numBlocks, blockSize>>>(points.deviceData(),
                                                 numPoints, dt, d_bounds);

    numBlocks = (numConnections + blockSize - 1) / blockSize;
    constrainAnglesKernel<<<numBlocks, blockSize>>>(angles.deviceData(),  numAngles);

    numBlocks = (numConnections + blockSize - 1) / blockSize;
    constrainDistancesKernel<<<numBlocks, blockSize>>>(connections.deviceData(), numConnections);

    /*int threadsPerBlock = 16;
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((numPoints + threadsPerBlock - 1) / threadsPerBlock,
                       (numRocks + threadsPerBlock - 1) / threadsPerBlock, 1);
    computeCollisions<<<blocksPerGrid, threadsPerBlockDim>>>(points.deviceData(),
                                                numPoints, rocks.deviceData(), numRocks);*/

    points.syncToHost();
    cudaFree(d_bounds);
}