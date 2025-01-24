#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"


// This uses a triangular matrix to store distances between each pair of cells
// This is because the distance between cell i and cell j is the same as the distance between cell j and cell i
// This reduces the amount of memory needed to store the distances
// The formula for the linear index of the distance between cell i and cell j is:
// i * numCells - (i * (i + 1)) / 2 + (j - i - 1)
__global__ void calculateDistances(const GPUVector<Cell> cells, const GPUVector<Point> points, StaticGPUVector<float> output) {
    size_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 >= cells.size() || idx2 >= cells.size() || idx1 >= idx2) return;

    size_t linearIdx = idx1 * cells.size() - (idx1 * (idx1 + 1)) / 2 + (idx2 - idx1 - 1);
    auto* cell1 = cells + idx1;
    auto* cell2 = cells + idx2;
    const Point* a = points + cell1->pointIdx;
    const Point* b = points + cell2->pointIdx;
    output[linearIdx] = a->distanceTo(*b);
}



__global__ void updateProductConcentration(GeneRegulatoryNetwork& grn,
                                           const GPUVector<Cell>& cells, const GPUVector<Point>& points,
                                           const StaticGPUVector<float>& cellDistances) {
    size_t productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cellIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (productIdx >= grn.factors.size() || cellIdx >= cells.size()) return;

    float decayRate = 0.99;
    auto cell = cells + cellIdx;
    float* amount = cell->products + productIdx;

    if (cell->frozen) return;

    Gene* factor = grn.factors + productIdx;

    // Update product quantities in cell
    /*if (factor->factorType == Gene::FactorType::MaternalFactor) {
        float2 factorPos = rotate(factor->extra * 10.0f, lifeForm->head->rotation) + headPos;
        *amount = distanceBetween(factorPos, p1->pos);
    }*/
    /*if (factor->factorType == Gene::FactorType::Time) {
        *amount = product->extra.y * (factor->sign ? 1.0f : -1.0f)
          + product->extra.x * (float)(simulationStep - lifeForm->birthdate)/100000.0;
    }*/
    /*if (factor->factorType == Gene::FactorType::Constant) {
        *amount = (float)factor->extra.x;
    }
    if (factor->factorType == Gene::FactorType::Generation) {
        *amount = (float)cell->generation;
    }*/
    /*if (factor->factorType == Gene::FactorType::Energy) {
        *amount = lifeForm->energy * max(factor->extra.x, 0.1f);
    }*/
    /*if (factor->factorType == Gene::FactorType::Crowding) {
        *amount = 0;
        for (const auto& otherCell: lifeForm->cells) {
            if (otherCell->frozen) continue;
            if (&otherCell == &cell) continue;
            float distance = cellDistances.at({cell, otherCell});
            *amount += 1.0f / (1.0f + distance);
        }
    }*/
    //Decay products
    if (factor->factorType == Gene::FactorType::InternalProduct) {
        *amount *= decayRate;
    }
}

__device__ float getCellDistance(const int cellIdx, const int otherCellIdx,
                                 const size_t numCells, const StaticGPUVector<float>& cellDistances) {
    size_t linearIdx = cellIdx * numCells - (cellIdx * (cellIdx + 1)) / 2 + (otherCellIdx - cellIdx - 1);
    return cellDistances[linearIdx];
};

__global__ void updateNSquaredProductConcentration(
  GeneRegulatoryNetwork& grn, const GPUVector<Cell>& cells, const GPUVector<Point>& points,
  const StaticGPUVector<float>& cellDistances) {
    size_t productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cellIdx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t cellIdx2 = blockIdx.z * blockDim.z + threadIdx.z;

    if (productIdx >= grn.factors.size() || cellIdx >= cells.size() || cellIdx2 >= cells.size()) return;

    float decayRate = 0.99;
    auto cell = cells + cellIdx;
    auto otherCell = cells + cellIdx2;

    float* amount = cell->products + productIdx;
    float otherAmount = otherCell->products[productIdx];

    if (cell->frozen) return;
    if (otherCell->frozen) return;
    if (cellIdx == cellIdx2) return;

    Point* p1 = points + cell->pointIdx;
    Point* p2 = points + otherCell->pointIdx;
    float2 divisionVector = {0, 0};
    Gene* factor = grn.factors + productIdx;

    float cellDistance = getCellDistance(cellIdx, cellIdx2, cells.size(), cellDistances);

    if (factor->factorType == Gene::FactorType::ExternalProduct) {
        float distanceScale = 1.0f / (1.0f + cellDistance);
        *amount += distanceScale * otherAmount;
    }

    if (factor->factorType == Gene::FactorType::Receptor && cellIdx != 0) {
        float2 normalisedVectorToCell = p1->pos - p2->pos / cellDistance;

        for(int i = 0; i < grn.factors.size(); i++) {
            if (grn.factors[i].factorType != Gene::FactorType::ExternalProduct) continue;
            float receptorAmount = otherCell->products[i];
            int affinityIndex = (int)(productIdx * grn.factors.size() + i);
            float affinity = grn.factorReceptorAffinities[affinityIndex];
            divisionVector += (*amount) * affinity * receptorAmount * normalisedVectorToCell;
        }
        cell->divisionRotation = std::atan2(divisionVector.y, divisionVector.x);
    }

    *amount *= decayRate;
}

__global__ void updateRegulatoryUnitExpression(GeneRegulatoryNetwork& grn,
                                               const GPUVector<Cell>& cells, const GPUVector<Point>& points) {
    size_t cellIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cellIdx >= cells.size()) return;

    auto cell = cells + cellIdx;
    if (cell->frozen) return;

    // For each cell, update it's products based on the regulatory units
    for (int i = 0; i < grn.regulatoryUnits.size(); i++) {
        float* factorLevels = grn.regulatoryUnits[i].calculateActivation(grn.promoters,
                                                         grn.factors,
                                                         cell->products,
                                                         grn.promoterFactorAffinities);
        // Add factor levels back to cell's products
        for (int j = 0; j < grn.factors.size(); j++) {
            cell->products[j] += factorLevels[j];
        }
    }
}

void updateGRN(GeneRegulatoryNetwork& grn,
                GPUVector<int>& cellIdxs,
               GPUVector<Cell>& cells,
               GPUVector<Point>& points) {
    // Calculate distances between each pair of cells
    grn.cellDistances = StaticGPUVector<float>((cells.size() * (cells.size() - 1)) / 2);
    dim3 threadsPerBlock(16, 16);
    dim3 numDistanceBlocks((cellIdxs.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (cellIdxs.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    calculateDistances<<<numDistanceBlocks, threadsPerBlock>>>(cells, points, grn.cellDistances);
    /*
    // Update product concentration
    dim3 numProductBlocks((grn.factors.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (cellIdxs.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateProductConcentration<<<numProductBlocks, threadsPerBlock>>>(grn, cells, points, grn.cellDistances);

    dim3 threadsPerCellProductBlock(16, 16, 16);
    dim3 numNSquaredProductBlocks((grn.factors.size() + threadsPerCellProductBlock.x - 1) / threadsPerCellProductBlock.x,
                                  (cellIdxs.size() + threadsPerCellProductBlock.y - 1) / threadsPerCellProductBlock.y,
                                  (cellIdxs.size() + threadsPerCellProductBlock.z - 1) / threadsPerCellProductBlock.z);
    updateNSquaredProductConcentration<<<numNSquaredProductBlocks, threadsPerCellProductBlock>>>(grn, cells, points, grn.cellDistances);


    // Update each cell's products based on regulatory expression
    dim3 numCellBlocks((cellIdxs.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (cellIdxs.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateRegulatoryUnitExpression<<<numCellBlocks, threadsPerBlock>>>(grn, cells, points);*/
}