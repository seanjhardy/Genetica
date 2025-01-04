#include <modules/physics/point.hpp>
#include "../physics/constraints.cu"
#include "cuda_runtime.h"
#include <modules/cuda/updatePoints.hpp>
#include <SFML/Graphics.hpp>

__global__ void updatePointsKernel(GPUVector<Point>& points, float dt, sf::FloatRect *bounds) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points.size()) return;

    Point &point = points[index];
    point.update(dt);
    constrainPosition(point, *bounds);
}


__global__ void constrainDistancesKernel(GPUVector<Point>& points, GPUVector<CellLink>& cellLinks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cellLinks.size()) return;

    CellLink cellLink = cellLinks[index];
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
__global__ void computeCollisions(Point *points, int numPoints, Rock *rocks, int numRocks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < numPoints && y < numRocks) {
        Point &point = points[x];
        auto rock = static_cast<Point>(rocks[y]);
        checkCollisionCircleRec(point, rock);
    }
}*/

void updatePoints(GPUVector<Point>& points,
                  GPUVector<CellLink>& cellLinks,
                  CGPUValue<sf::FloatRect> &bounds,
                  float dt) {
    int numPoints = points.size();
    int numConnections = cellLinks.size();

    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    updatePointsKernel<<<numBlocks, blockSize>>>(points, dt, bounds.deviceData());

    /*numBlocks = (numParentChildLinks + blockSize - 1) / blockSize;
    updateParentChildLinkKernel<<<numBlocks, blockSize>>>(points.deviceData(), parentChildLinks.deviceData(),
                                                          numParentChildLinks, dt);*/

    numBlocks = (numConnections + blockSize - 1) / blockSize;
    constrainDistancesKernel<<<numBlocks, blockSize>>>(points, cellLinks);

    /*int threadsPerBlock = 16;
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((numPoints + threadsPerBlock - 1) / threadsPerBlock,
                       (numRocks + threadsPerBlock - 1) / threadsPerBlock, 1);
    computeCollisions<<<blocksPerGrid, threadsPerBlockDim>>>(points.deviceData(),
                                                numPoints, rocks.deviceData(), numRocks);*/
    //cudaDeviceSynchronize();
}

__global__ void movePointKernel(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos, int* entityID) {
    Point &point = points[pointIndex];
    point.pos.x = newPos.x;
    point.pos.y = newPos.y;
    *entityID = point.entityID;
}

int movePoint(GPUVector<Point>& points, int pointIndex, const sf::Vector2f& newPos) {
    int* entityID;
    cudaMalloc(&entityID, sizeof(int));
    movePointKernel<<<1, 1>>>(points, pointIndex, newPos, entityID);
    int entityIDHost;
    cudaMemcpy(&entityIDHost, entityID, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(entityID);
    return entityIDHost;
}
