/*
#include <modules/gpu/updateCells.hpp>
#include <simulator/simulator.hpp>

__global__ void updateCell(GPUVector<Cell> cells, GPUVector<Point> points, size_t step, cellGrowthData output) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;

    Cell& cell = cells[index];
    Point& point = points[cell.pointIdx];

    // - Base metabolic rate (proportional to area)
    cell.energy -= 0.00000001f * M_PI * point.radius * point.radius;

    // TODO: Implement these
    // - Energy spent when moving
    // - Photosynthesis, eating, thermal induction
    // - Reduce energy loss due to insulation based on cell wall properties
    // - Energy used by organelles

    // - Growth - if target size not reached, gain energy according to area gained
    if (abs(point.radius - cell.targetRadius) > 0.0f) {
        // TODO: Dynamic growth rate (1.0f) from GRN
        float newArea = sqrt((M_PI * point.radius * point.radius + 0.1f) / M_PI);
        float areaChange = newArea - point.radius;
        float growthEnergy = 0.0001f * areaChange;
        if (cell.energy > growthEnergy) {
            cell.energy -= growthEnergy;
            point.radius = newArea;
        }
    }

    // Check if the cell should divide
    if ((step - cell.lastDivideTime) > 200 &&
        point.radius >= cell.targetRadius &&
        //cell.numDivisions == 0 && cell.generation == 0 &&
        cell.energy > 0) {
        auto hasDivided = output.push(cell.idx);
        if (hasDivided) {
            // Update cell properties
            cell.numDivisions += 1;
            cell.lastDivideTime = step;
            point.radius *= 0.7f; // 30% reduction in size
            cell.energy *= 0.5f; // half energy for this cell and daughter cell
        }
    }
}

__global__ void updateCellCellInteractions(GPUVector<Cell> cells, GPUVector<CellLink> cellLinks,
    GPUVector<Point> points, size_t step, cellGrowthData output) {
    size_t indexA = blockIdx.x * blockDim.x + threadIdx.x;
    size_t indexB = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexA >= cells.size() || indexB >= cells.size() || indexA >= indexB) return;

    Cell& cellA = cells[indexA];
    Cell& cellB = cells[indexB];
    Point& pointA = points[cellA.pointIdx];
    Point& pointB = points[cellB.pointIdx];

    if (cellA.lifeFormIdx == cellB.lifeFormIdx) {
        if (cellA.lastDivideTime <= 100 || cellB.lastDivideTime <= 100) {
            float distance = distanceBetween(pointA.pos, pointB.pos);
            float linkDistance = pointA.radius + pointB.radius;
            //if (distance < linkDistance);
        }
    }
}

void updateCells(dynamicStableVector<LifeForm>& lifeForms,
    const GPUVector<Cell>& cells,
    const GPUVector<Point>& points,
    cellGrowthData& cellDivisionData) {
    // Only run every 10 simulation steps
    if (Simulator::get().getStep() % 10 != 0) {
        return;
    }

    if (cells.size() == 0) {
        return;
    }

    // Only process division events every 100 steps to prevent slow gpu calls
    cellDivisionData.setEnabled(Simulator::get().getStep() % 100 == 0);
    cellDivisionData.reset();

    // Launch kernel
    int threadsPerBlock = 512;
    int numCellBlocks = (cells.size() + threadsPerBlock - 1) / threadsPerBlock;
    updateCell << <numCellBlocks, threadsPerBlock >> > (cells, points, Simulator::get().getStep(), cellDivisionData);

    dim3 cellThreadsPerBlock(32, 32);
    dim3 numCellCellBlocks((cells.size() + cellThreadsPerBlock.x - 1) / cellThreadsPerBlock.x,
        (cells.size() + cellThreadsPerBlock.y - 1) / cellThreadsPerBlock.y);
    //updateCellCellInteractions<<<numCellBlocks, threadsPerBlock>>>(cells, points);

    int numDivisions = cellDivisionData.getNumDivisions();
    if (numDivisions > 0) {
        int eventsToProcess = min(numDivisions, MAX_DIVISION_EVENTS);
        auto* hostDividingIndices = new size_t[eventsToProcess];

        cellDivisionData.getDividingCellIndices(hostDividingIndices, eventsToProcess);

        for (int i = 0; i < eventsToProcess; i++) {
            size_t motherIdx = hostDividingIndices[i];
            Cell mother = cells.itemToHost(motherIdx);
            lifeForms[mother.lifeFormIdx].addCell(motherIdx, mother);
        }

        delete[] hostDividingIndices;
    }
}
*/
