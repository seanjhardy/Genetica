#include <geneticAlgorithm/lifeform.hpp>
#include <simulator/simulator.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"

// This uses a triangular matrix to store distances between each pair of cells
// This is because the distance between cell i and cell j is the same as the distance between cell j and cell i
// This reduces the amount of memory needed to store the distances
// The formula for the linear index of the distance between cell i and cell j is:
// i * numCells - (i * (i + 1)) / 2 + (j - i - 1)
__global__ void calculateDistances(const StaticGPUVector<Cell*> cells, const GPUVector<Point> points, StaticGPUVector<float> output) {
    size_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 >= cells.size() || idx2 >= cells.size() || idx1 >= idx2) return;

    size_t linearIdx = idx1 * cells.size() - (idx1 * (idx1 + 1)) / 2 + (idx2 - idx1 - 1);
    auto cell1 = *(cells + idx1);
    auto cell2 = *(cells + idx2);
    const Point* a = points + cell1->pointIdx;
    const Point* b = points + cell2->pointIdx;
    output[linearIdx] = a->distanceTo(*b);
}

__device__ float getCellDistance(const int cellIdx, const int otherCellIdx,
                                 const size_t numCells, const StaticGPUVector<float> cellDistances) {
    size_t linearIdx = cellIdx * numCells - (cellIdx * (cellIdx + 1)) / 2 + (otherCellIdx - cellIdx - 1);
    return cellDistances[linearIdx];
};


__global__ void updateProductConcentration(GeneRegulatoryNetwork grn,
                                            StaticGPUVector<Cell*> cells,
                                           const GPUVector<Point> points,
                                           const int simulationStep,
                                           const int birthdate,
                                           const float energy) {
    int productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cellIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (productIdx >= grn.factors.size() || cellIdx >= cells.size()) return;

    //Head
    auto head = *cells[0];
    if (head.pointIdx >= points.size()) return;
    auto headPos = points[head.pointIdx].getPos();

    auto cell = cells[cellIdx];
    // Print before accessing products
    /*if (productIdx == 0) {
        // Print a range of memory values around the products vector
        StaticGPUVector<float>* products_vec = &(cell->products);
        void** memory = (void**)products_vec;
        printf("Memory at %p:\n", memory);
        for (int i = -2; i <= 2; i++) {
            printf("  Offset %d: %p\n", i, memory[i]);
        }
    }*/
    if (cell == nullptr) {
        printf("Error: null cell pointer at idx %llu, %llu\n", cell->idx, cell->lifeFormIdx);
        return;
    }

    float* products_data = cell->products.data();
    if (products_data == nullptr) {
        printf("Error: null products pointer for cell %llu %llu\n", cell->idx, cell->lifeFormIdx);
        return;
    }

    // Now try to access
    float* amount = products_data + productIdx;
    if (cell->products.data() == nullptr) return;

    if (cell->pointIdx >= points.size()) return;
    Point* p1 = points + cell->pointIdx;
    float decayRate = 0.99;
    if (cell->frozen) return;

    if (productIdx >= cell->products.size()) return;
    Gene* factor = grn.factors + productIdx;

    // Update product quantities in cell
    if (factor->factorType == Gene::FactorType::MaternalFactor) {
        float2 factorPos = rotate(factor->extra * 10.0f, head.rotation) + headPos;
        *amount = distanceBetween(factorPos, p1->getPos());
    }
    /*if (factor->factorType == Gene::FactorType::Time) {
        *amount = factor->extra.y * (factor->sign ? 1.0f : -1.0f)
          + factor->extra.x * (simulationStep - birthdate)/100000.0;
    }
    if (factor->factorType == Gene::FactorType::Constant) {
        *amount = factor->extra.x;
    }
    if (factor->factorType == Gene::FactorType::Generation) {
        *amount = cell->generation;
    }
    if (factor->factorType == Gene::FactorType::Energy) {
        *amount = energy * max(factor->extra.x, 0.1f);
    }
    //Decay products
    if (factor->factorType == Gene::FactorType::InternalProduct) {
        *amount *= decayRate;
    }*/
}

__global__ void updateNSquaredProductConcentration(
  GeneRegulatoryNetwork grn, const StaticGPUVector<Cell*> cells, const GPUVector<Point> points) {
    size_t productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cellIdx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t cellIdx2 = blockIdx.z * blockDim.z + threadIdx.z;

    if (productIdx >= grn.factors.size() || cellIdx >= cells.size() || cellIdx2 >= cells.size()) return;
    float decayRate = 0.99;
    auto cell = *(cells + cellIdx);
    auto otherCell = *(cells + cellIdx2);

    float* amount = cell->products + productIdx;
    float otherAmount = otherCell->products[productIdx];

    if (cell->frozen) return;
    if (otherCell->frozen) return;
    if (cellIdx == cellIdx2) return;

    Point* p1 = points + cell->pointIdx;
    Point* p2 = points + otherCell->pointIdx;
    float2 divisionVector = {0, 0};
    Gene* factor = grn.factors + productIdx;

    float cellDistance = getCellDistance(cellIdx, cellIdx2, cells.size(), grn.cellDistances);

    if (factor->factorType == Gene::FactorType::ExternalProduct) {
        float distanceScale = 1.0f / (1.0f + cellDistance);
        *amount += distanceScale * otherAmount;
    }

    if (factor->factorType == Gene::FactorType::Receptor && cellIdx != 0) {
        float2 normalisedVectorToCell = p1->getPos() - p2->getPos() / cellDistance;

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

__global__ void updateRegulatoryUnitExpression(GeneRegulatoryNetwork grn,
                                               const StaticGPUVector<Cell*> cells,
                                               const GPUVector<Point> points) {
    size_t cellIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cellIdx >= cells.size()) return;

    auto cell = *(cells + cellIdx);
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

__global__ void updateGeneExpression(GeneRegulatoryNetwork grn,
    StaticGPUVector<Cell*> cellPtrs,
    StaticGPUVector<CellLink*> cellLinkPtrs,
    const GPUVector<Point> points) {
    size_t cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t effectorIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (cellIdx >= cellPtrs.size() || effectorIdx >= grn.effectors.size()) return;

    auto cell = *(cellPtrs + cellIdx);
    auto effector = grn.effectors + effectorIdx;

    if (cell->frozen) return;

    float expression = 0.0f;
    for (int productIdx = 0; productIdx < cell->products.size(); productIdx++) {
        auto gene = grn.factors + productIdx;
        auto level = cell->products[productIdx];
        if (gene->factorType != Gene::FactorType::InternalProduct) continue;
        int affinityIndex = productIdx * grn.factors.size() + effectorIdx;
        expression += level * grn.factorEffectorAffinities[affinityIndex];
    }

    if (expression == 0) return;

    if (effector->effectorType == Effector::EffectorType::Die) {
        if (expression > 0.5 && cellIdx != 0) {
            cell->dead = true;
        }
    }
    if (effector->effectorType == Effector::EffectorType::Divide) {
        if (expression > 0.0001 && !cell->dividing) {
            cell->dividing = true;
        }
    }
    if (effector->effectorType == Effector::EffectorType::Freeze) {
        if (expression > 0.5) {
            cell->frozen = true;
        }
    }
    if (effector->effectorType == Effector::EffectorType::Distance) {
        /*for (auto cellLink : cellLinkPtrs) {
            if (cellLink->cellAId == cellIdx || cellLink->cellBId == cellIdx) {
                cellLink->adjustSize(min(expression, 0.0f));
            }
        }*/
    }
    if (effector->effectorType == Effector::EffectorType::Radius) {
        float sizeChange = expression;
        int pointIdx = cell->pointIdx;
        Point* pointObj = points + pointIdx;
        if (cell->targetRadius + sizeChange < 20) {
            cell->targetRadius = max(pointObj->radius + sizeChange, 0.5f);
        }
    }
    if (effector->effectorType == Effector::EffectorType::Red) {
        cell->updateHue(Red, expression * 0.1f);
    }
    if (effector->effectorType == Effector::EffectorType::Green) {
        cell->updateHue(Green, expression * 0.1f);
    }
    if (effector->effectorType == Effector::EffectorType::Blue) {
        cell->updateHue(Blue, expression * 0.1f);
    }
}

void updateGRN(LifeForm& lifeForm,
               GPUVector<Point>& points) {

    auto cellsPtrs = static_cast<StaticGPUVector<Cell*>>(lifeForm.cellPointers);
    auto cellLinksPtrs =  static_cast<StaticGPUVector<CellLink*>>(lifeForm.cellLinkPointers);

    if (cellsPtrs.size() > Simulator::get().getEnv().getCells().size() || cellsPtrs.size() == 0) return;

    /*for (int i = 0 ; i < cellsPtrs.size(); i++) {
        Cell* cell;
        // copy cell to cpu
        cudaLog(cudaMemcpy(&cell, &cellsPtrs[i], sizeof(Cell*), cudaMemcpyDeviceToHost));
        if (((size_t)cell->products.data() - 30103272960) > 10000000000 || ((size_t)cell->products.data() - 30103272960) < -10000000000) {
            printf("something weird1: %p, %llu\n", cellsPtrs[i]->products.data(), cellsPtrs[i]->products.size());
        }
    }*/
    // Calculate distances between each pair of cells
    lifeForm.grn.cellDistances.destroy();
    lifeForm.grn.cellDistances = StaticGPUVector<float>((cellsPtrs.size() * (cellsPtrs.size() - 1)) / 2);
    dim3 threadsPerBlock(32, 32);
    dim3 numDistanceBlocks((cellsPtrs.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (cellsPtrs.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    calculateDistances<<<numDistanceBlocks, threadsPerBlock>>>(cellsPtrs, points, lifeForm.grn.cellDistances);

    // Update product concentration
    dim3 numProductBlocks((lifeForm.grn.factors.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (cellsPtrs.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateProductConcentration<<<numProductBlocks, threadsPerBlock>>>(
        lifeForm.grn,
        cellsPtrs,
        points,
        Simulator::get().getStep(),
        lifeForm.birthdate,
        lifeForm.energy);

    // Update product concentration based on n squared interactions
    /*dim3 threadsPerCellProductBlock(32, 32, 32);
    dim3 numNSquaredProductBlocks((lifeForm.grn.factors.size() + threadsPerCellProductBlock.x - 1) / threadsPerCellProductBlock.x,
                                  (cellsPtrs.size() + threadsPerCellProductBlock.y - 1) / threadsPerCellProductBlock.y,
                                  (cellsPtrs.size() + threadsPerCellProductBlock.z - 1) / threadsPerCellProductBlock.z);
    updateNSquaredProductConcentration<<<numNSquaredProductBlocks, threadsPerCellProductBlock>>>(lifeForm.grn, cellsPtrs, points);

    // Update each cell's products based on regulatory expression
    size_t numCellBlocks((cellsPtrs.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
    updateRegulatoryUnitExpression<<<numCellBlocks, threadsPerBlock>>>(lifeForm.grn, cellsPtrs, points);

    // Update gene expression based on the products
    dim3 numEffectorBlocks((cellsPtrs.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (lifeForm.grn.effectors.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateGeneExpression<<<numEffectorBlocks, threadsPerBlock>>>(lifeForm.grn, cellsPtrs, cellLinksPtrs, points);*/
}