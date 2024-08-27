#include <modules/physics/point.hpp>
#include "../physics/constraints.cu"
#include "cuda_runtime.h"
#include <modules/cuda/updatePoints.hpp>
#include <SFML/Graphics.hpp>
#include "simulator/entities/rock.hpp"

__global__ void updatePointsKernel(Point *points, int numParticles, float dt, sf::FloatRect *bounds) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numParticles) {
        Point &point = points[index];
        point.update(dt);
        constrainPosition(point, *bounds);
    }
}


__global__ void constrainDistancesKernel(Point *points, Connection *connections, int numConnections, int numPoints) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numConnections) {
        Connection connection = connections[index];
        constrainDistance(points[connection.a], points[connection.b], connection.distance, 0.2f);
    }
}

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

__global__ void computeCollisions(Point *points, int numPoints, Rock *rocks, int numRocks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < numPoints && y < numRocks) {
        Point &point = points[x];
        auto rock = static_cast<Point>(rocks[y]);
        checkCollisionCircleRec(point, rock);
    }
}

void updatePoints(GPUVector<Point> &points,
                  GPUVector<Connection> &connections,
                  GPUVector<ParentChildLink> &parentChildLinks,
                  GPUValue<sf::FloatRect> &bounds,
                  float dt) {
    int numPoints = points.size();
    int numConnections = connections.size();
    int numParentChildLinks = parentChildLinks.size();

    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    updatePointsKernel<<<numBlocks, blockSize>>>(points.deviceData(),
                                                 numPoints, dt, bounds.deviceData());

    numBlocks = (numParentChildLinks + blockSize - 1) / blockSize;
    updateParentChildLinkKernel<<<numBlocks, blockSize>>>(points.deviceData(), parentChildLinks.deviceData(),
                                                          numParentChildLinks, dt);

    numBlocks = (numConnections + blockSize - 1) / blockSize;
    constrainDistancesKernel<<<numBlocks, blockSize>>>(points.deviceData(), connections.deviceData(), numConnections,
                                                       numPoints);

    /*int threadsPerBlock = 16;
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((numPoints + threadsPerBlock - 1) / threadsPerBlock,
                       (numRocks + threadsPerBlock - 1) / threadsPerBlock, 1);
    computeCollisions<<<blocksPerGrid, threadsPerBlockDim>>>(points.deviceData(),
                                                numPoints, rocks.deviceData(), numRocks);*/
    //cudaDeviceSynchronize();
}