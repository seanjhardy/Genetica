#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>

// Compute affinities for all promoters, factors, and effectors
__global__ void calculateAffinity(const GeneticUnit* a_elements, const GeneticUnit* b_elements,
                                    const int num_a, const int num_b,
                                    float* output) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 >= num_a || idx2 >= num_b) return;

    int resultIdx = idx1 * num_b + idx2; // Flatten 2D indices into a 1D result index
    GeneticUnit a = a_elements[idx1];
    GeneticUnit b = b_elements[idx2];
    float distance = distanceBetween(a.embedding, b.embedding);

    if (distance > GeneticUnit::DISTANCE_THRESHOLD) {
        output[resultIdx] = 0.0f;
    } else {
        float affinitySign = (a.sign == b.sign) ? 1.0f : -1.0f;

        output[resultIdx] = affinitySign *
             (2.0f * std::abs(a.modifier * b.modifier)
                * (GeneticUnit::DISTANCE_THRESHOLD - distance)) /
             (10.0f * distance + std::abs(a.modifier * b.modifier));
    }
}

// This uses a triangular matrix to store distances between each pair of cells
// This is because the distance between cell i and cell j is the same as the distance between cell j and cell i
// This reduces the amount of memory needed to store the distances
// The formula for the linear index of the distance between cell i and cell j is:
// i * numCells - (i * (i + 1)) / 2 + (j - i - 1)
__global__ void calculateDistances(const Cell* cells, const Point points*, const int numCells, float* output) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 >= numCells || idx2 >= numCells || idx1 >= idx2) return;

    int linearIdx = idx1 * numCells - (idx1 * (idx1 + 1)) / 2 + (idx2 - idx1 - 1);
    auto* cell1 = cells + idx1;
    auto* cell2 = cells + idx2;
    const Point* a = points + cell1->pointIdx;
    const Point* b = points + cell2->pointIdx;
    output[linearIdx] = a->distanceTo(*b);
}

void computeAffinities(GeneRegulatoryNetwork &grn) {
    dim3 threadsPerBlock(16, 16);

    dim3 numPromoterFactorBlocks((grn.numPromoters + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (grn.numFactors + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaMalloc(&grn.promoterFactorAffinities, grn.numPromoters * grn.numFactors * sizeof(float));
    calculateAffinity<<<numPromoterFactorBlocks, threadsPerBlock>>>(grn.promoters, grn.factors,
                                                      grn.numPromoters, grn.numFactors,
                                                      grn.promoterFactorAffinities);

    dim3 numFactorEffectorBlocks((grn.numFactors + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                    (grn.numEffectors + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaMalloc(&grn.factorEffectorAffinities, grn.numFactors * grn.numEffectors * sizeof(float));
    calculateAffinity<<<numFactorEffectorBlocks, threadsPerBlock>>>(grn.factors, grn.effectors,
                                                      grn.numFactors, grn.numEffectors,
                                                      grn.factorEffectorAffinities);

    dim3 numFactorReceptorBlocks((grn.numFactors + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                    (grn.numFactors + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaMalloc(&grn.factorReceptorAffinities, grn.numFactors * grn.numFactors * sizeof(float));
    calculateAffinity<<<numFactorReceptorBlocks, threadsPerBlock>>>(grn.factors, grn.factors,
                                                      grn.numFactors, grn.numFactors,
                                                      grn.factorReceptorAffinities);
}

__global__ void updateProductConcentration(GeneRegulatoryNetwork& grn, Cell* cells, Point points*,
    const float* cellDistances, const int numProducts, const int numCells) {
    int productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int cellIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (productIdx >= numProducts || cellIdx >= numCells) return;

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
    if (factor->factorType == Gene::FactorType::Constant) {
        *amount = (float)factor->extra.x;
    }
    if (factor->factorType == Gene::FactorType::Generation) {
        *amount = (float)cell->generation;
    }
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

__global__ float getCellDistance(const int cellIdx, const int otherCellIdx, const float* cellDistances, const int numCells) {
    int linearIdx = cellIdx * numCells - (cellIdx * (cellIdx + 1)) / 2 + (otherCellIdx - cellIdx - 1);
    return cellDistances[linearIdx];
};

__global__ void updateNSquaredProductConcentration(GeneRegulatoryNetwork& grn, Cell* cells, Point points*,
    const float* cellDistances, const int numCells) {
    int productIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int cellIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int cellIdx2 = blockIdx.z * blockDim.z + threadIdx.z;

    if (productIdx >= grn.numFactors || cellIdx >= numCells || cellIdx2 >= numCells) return;

    float decayRate = 0.99;
    auto cell = cells + cellIdx;
    auto otherCell = cells + cellIdx2;

    float* amount = cell->products + productIdx;
    float otherAmount = *(otherCell->products + productIdx);

    if (cell->frozen) return;
    if (otherCell->frozen) return;
    if (cellIdx == cellIdx2) return;

    Point* p1 = points + cell->pointIdx;
    Point* p2 = points + otherCell->pointIdx;
    float2 divisionVector = {0, 0};
    Gene* factor = grn.factors + productIdx;

    float cellDistance = getCellDistance(cellIdx, cellIdx2, cellDistances, numCells);

    if (factor->factorType == Gene::FactorType::ExternalProduct) {
        float distanceScale = 1.0f / (1.0f + cellDistance);
        *amount += distanceScale * otherAmount;
    }

    if (factor->factorType == Gene::FactorType::Receptor && cellIdx != 0) {
        float2 normalisedVectorToCell = p1->pos - p2->pos / cellDistance;

        for(auto& [otherProduct, otherAmount] : otherCell->products) {
            if (otherProduct->factorType != Gene::FactorType::ExternalProduct) continue;
            int affinityIndex = productIdx * grn.numFactors + otherProduct->index;
            float affinity = grn.factorReceptorAffinities[affinityIndex];
            divisionVector += amount * affinity * otherAmount * normalisedVectorToCell;
        }
        cell->divisionRotation = std::atan2(divisionVector.y, divisionVector.x);
    }

    *amount *= decayRate;
}

__global__ void updateRegulatoryUnitExpression(GeneRegulatoryNetwork& grn, Cell* cells, Point points*, const int numCells) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;

    auto cell = cells + cellIdx;
    if (cell->frozen) return;

    // For each cell, update it's products based on the regulatory units
    float* newFactorLevels;
    for (auto& unit : grn.regulatoryUnits) {
        float* factorLevels = unit.calculateActivation(grn.promoters,
                                                         grn.factors,
                                                         cell->products,
                                                         grn.promoterFactorAffinities);
        for (auto& [factor, level] : factorLevels) {
            newFactorLevels[factor] += level;
        }
    }
    // Add factor levels back to cell's products
    for (auto& [factor, level] : newFactorLevels) {
        cell->products[factor] += level;
    }
}

float updateGRN(GeneRegulatoryNetwork& grn, Cell* cells, Point* points, const int numCells) {
    // Calculate distances between each pair of cells
    int numDistances = (numCells * (numCells - 1)) / 2;
    float* cellDistances;
    cudaMalloc(&cellDistances, numDistances * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numDistanceBlocks((numCells + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numCells + threadsPerBlock.y - 1) / threadsPerBlock.y);
    calculateDistances<<<numDistanceBlocks, threadsPerBlock>>>(cells, points, numCells, cellDistances);
    return cellDistances[0];

    // Update product concentration
    dim3 numProductBlocks((grn.numFactors + threadsPerBlock.x - 1) / threadsPerBlock.x,
                          (numCells + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateProductConcentration<<<numProductBlocks, threadsPerBlock>>>(grn, cells, points, cellDistances,
        grn.numFactors, numCells);

    dim3 threadsPerCellProductBlock(16, 16, 16);
    dim3 numNSquaredProductBlocks((grn.numFactors + threadsPerCellProductBlock.x - 1) / threadsPerCellProductBlock.x,
                                  (numCells + threadsPerCellProductBlock.y - 1) / threadsPerCellProductBlock.y,
                                  (numCells + threadsPerCellProductBlock.z - 1) / threadsPerCellProductBlock.z);
    updateNSquaredProductConcentration<<<numNSquaredProductBlocks, threadsPerCellProductBlock>>>(grn, cells, points, cellDistances, numCells);


    // Update each cell's products based on regulatory expression
    dim3 numCellBlocks((numCells + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numCells + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateRegulatoryUnitExpression<<<numCellBlocks, threadsPerBlock>>>(grn, cells, points, numCells);
}