#include <modules/physics/point.hpp>
#include "../physics/constraints.cu"
#include "cuda_runtime.h"
#include <modules/cuda/updateCellLinks.hpp>
#include "modules/utils/GPU/mathUtils.hpp"

__global__ void updateCellLink(GPUVector<Point> points, GPUVector<Cell> cells, GPUVector<CellLink> cellLinks) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cellLinks.size()) return;

    CellLink& cellLink = cellLinks[index];
    Cell& cellA = cells[cellLink.cellAIdx];
    Cell& cellB = cells[cellLink.cellBIdx];
    Point& pointA = points[cellLink.pointAIdx];
    Point& pointB = points[cellLink.pointBIdx];

    // Constrain distance between points
    constrainDistance(pointA, pointB, cellLink.length + pointA.radius + pointB.radius);

    // Update the link to maintain its angle relative to the cell
    constrainAngle(pointA, pointB, cellLink.angleFromA, cellLink.angleFromB, 0.1); //cellLink.stiffness

    if (cellLink.length < cellLink.targetLength && cellA.energy > 0 && cellB.energy > 0) {
        float delta = 0.1f * 2 / (pointA.radius + pointB.radius);
        //cellLink.length += delta;
    }

    // Share energy between cells
    // TODO: this should be based on distance, genes for energy transfer speed, and coefficient of sharing
    float maxTransferSpeed = (pointA.radius + pointB.radius) / 2; // Proportional to average cross-section of the link
    float totalEnergy = cellA.energy + cellB.energy;
    float equilibriumA = totalEnergy * 0.5f; // Assume equal weighting for now
    float transferAtoB = equilibriumA - cellA.energy;
    float actualTransfer = clamp(-maxTransferSpeed, transferAtoB, maxTransferSpeed);

    if (actualTransfer > 0) {
        cellA.energy += actualTransfer;
        cellB.energy -= actualTransfer;
    }
}


void updateCellLinks(GPUVector<Point>& points,
                     GPUVector<Cell>& cells,
                     GPUVector<CellLink>& cellLinks) {
    int blockSize = 256; // Number of threads per block

    if (points.size() == 0) return;

    // Update connections
    int numBlocks = (cellLinks.size() + blockSize - 1) / blockSize;
    updateCellLink<<<numBlocks, blockSize>>>(points, cells, cellLinks);
}
