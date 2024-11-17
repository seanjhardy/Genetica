#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>

// Compute affinities for all promoters, factors, and effectors
__global__ void calculateAffinity(const GeneticUnit* a_elements, const GeneticUnit* b_elements,
                                    const int num_a, const int num_b,
                                    float* output) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 < num_a && idx2 < num_b) {
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
}

// This uses a triangular matrix to store distances between each pair of cells
// This is because the distance between cell i and cell j is the same as the distance between cell j and cell i
// This reduces the amount of memory needed to store the distances
// The formula for the linear index of the distance between cell i and cell j is:
// i * numCells - (i * (i + 1)) / 2 + (j - i - 1)
__global__ void calculateDistances(const Cell* cells, const Point points*, const int numCells, float* output) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 < numCells && idx2 < numCells && idx1 < idx2) {
        int linearIdx = idx1 * numCells - (idx1 * (idx1 + 1)) / 2 + (idx2 - idx1 - 1);
        Point a = points[cells[idx1].pointIdx];
        Point b = points[cells[idx2].pointIdx];
        output[linearIdx] = a.distanceTo(b);
    }
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

void updateGRN(GeneRegulatoryNetwork& grn, const Cell* cells, const Point* points, const int numCells) {
    // Calculate distances between each pair of cells
    int numDistances = (numCells * (numCells - 1)) / 2;
    float* cellDistances;
    cudaMalloc(&cellDistances, numDistances * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numCellBlocks((numCells + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numCells + threadsPerBlock.y - 1) / threadsPerBlock.y);
    calculateDistances<<<numCellBlocks, threadsPerBlock>>>(cells, points, numCells, cellDistances);

    // Update product concentrations
}