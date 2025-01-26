#include "geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp"
#include <modules/cuda/calculateAffinities.hpp>

template <typename T, typename U>
__global__ void calculateAffinity(const StaticGPUVector<T> a_elements, const StaticGPUVector<U> b_elements, StaticGPUVector<float> output) {
    size_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx1 >= a_elements.size() || idx2 >= b_elements.size()) return;

    size_t resultIdx = idx1 * b_elements.size() + idx2; // Flatten 2D indices into a 1D result index
    const GeneticUnit a = a_elements[idx1];
    const GeneticUnit b = b_elements[idx2];

    const float distance = distanceBetween(a.embedding, b.embedding);

    if (distance > GeneticUnit::DISTANCE_THRESHOLD) {
        output[resultIdx] = 0.0f;
    } else {
        const float affinitySign = (a.sign == b.sign) ? 1.0f : -1.0f;

        output[resultIdx] = affinitySign *
             (2.0f * std::abs(a.modifier * b.modifier)
                * (GeneticUnit::DISTANCE_THRESHOLD - distance)) /
             (10.0f * distance + std::abs(a.modifier * b.modifier));
    }
}

void calculateAffinities(GeneRegulatoryNetwork &grn) {
    dim3 threadsPerBlock(32, 32);

    dim3 numPromoterFactorBlocks((grn.promoters.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (grn.factors.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grn.promoterFactorAffinities = StaticGPUVector<float>(grn.promoters.size() * grn.factors.size());
    calculateAffinity<<<numPromoterFactorBlocks, threadsPerBlock>>>(
      grn.promoters,
      grn.factors,
      grn.promoterFactorAffinities);

    dim3 numFactorEffectorBlocks((grn.factors.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                    (grn.effectors.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    grn.factorEffectorAffinities = StaticGPUVector<float>(grn.factors.size() * grn.effectors.size());
    calculateAffinity<<<numFactorEffectorBlocks, threadsPerBlock>>>(
      grn.factors,
    grn.effectors,
    grn.factorEffectorAffinities);

    dim3 numFactorReceptorBlocks((grn.factors.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                    (grn.factors.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    grn.factorReceptorAffinities = StaticGPUVector<float>(grn.factors.size() * grn.factors.size());
    calculateAffinity<<<numFactorReceptorBlocks, threadsPerBlock>>>(
      grn.factors,
      grn.factors,
      grn.factorReceptorAffinities);
}