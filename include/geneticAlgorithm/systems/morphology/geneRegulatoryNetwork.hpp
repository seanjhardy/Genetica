#ifndef GENE_REGULATORY_NETWORK
#define GENE_REGULATORY_NETWORK

#include "./gene.hpp"
#include "./promoter.hpp"
#include "./effector.hpp"
#include "./regulatoryUnit.hpp"
#include <geneticAlgorithm/cellParts/cell.hpp>

class LifeForm;

/** The GRN is a network of genes which can activate and deactivate one another.
 *
 * Factors - Genes which represent levels of a particular protein in the cell, they can
 * diffuse to other cells externally, diffuse internally, be a changing quantity, or an explicit value like the
 * age of the cell, the number of neighbours nearby, the energy of the cell etc, these are basically inputs
 *
 * Effectors - Output genes which produce some effect, e.g. cause the cell to divide, grow, change colour, grow new organelles etc
 *
 * Promoters - when specific factors bind to promoters, these promote the production factors.
 *
 * Regulatory units - A regualatory unit is composed of a set of Promoters, and a set of factors.
 * The promoters get activated based on their binding distance to other factors in the cell, and this in turn causes
 * the factors in the regulatory unit to get produced.
 *
 * Imagine a network of nodes where there are some inputs (e.g. chemicals in the environment, the age of the cell, its energy,
 * factors produced by nearby cells etc) which then feed into regulatory units. One regulatory unit is like one node
 * where a bunch of factors can promote it, which then in turn increases the production of other factors, which feed into other
 * regulatory units (intermediate nodes) and effectors (output nodes). The network dynamically adapts over time, producing
 * specific cell growth at specific points in the growth of the organism
*/
class GeneRegulatoryNetwork {
public:
    staticGPUVector<Gene> factors;
    staticGPUVector<Promoter> promoters;
    staticGPUVector<Effector> effectors;
    staticGPUVector<RegulatoryUnit> regulatoryUnits;

    staticGPUVector<float> promoterFactorAffinities;
    staticGPUVector<float> factorEffectorAffinities;
    staticGPUVector<float> factorReceptorAffinities;

    // We need to keep track of the distances between each pair of cells (here stored as a flat triangular matrix)
    // To calculate how factors diffuse as they get further away from where they're produced
    staticGPUVector<float> cellDistances;

    void destroy() {
        factors.destroy();
        promoters.destroy();
        effectors.destroy();
        regulatoryUnits.destroy();
        promoterFactorAffinities.destroy();
        factorEffectorAffinities.destroy();
        factorReceptorAffinities.destroy();
        cellDistances.destroy();
    }
};

#endif