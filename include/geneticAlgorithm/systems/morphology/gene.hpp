#ifndef GENE
#define GENE

#include "./geneticUnit.hpp"
#include <modules/utils/operations.hpp>

/**
 * A gene is a genetic unit that can be expressed in a cell
 */
struct Gene : GeneticUnit {
    enum class FactorType {
        // External Factors
        Constant, // A value of 1
        Time, // A value that increases over time
        Generation, // The generation of a particular
        Energy, // The energy of the life form (maybe add energy per cell?)
        MaternalFactor, // A factor from the genome
        Crowding, // The number of nearby cells

        // Genetic factors
        InternalProduct, // Morphogen that acts inside the cell
        ExternalProduct, // Morphogen that can interact with other cells
        Receptor, // Binds to morphogens to control divide direction
    } factorType{};

    float2 extra;

    Gene() : factorType(FactorType::Constant) {}
    Gene(FactorType factorType, bool sign, float modifier, float3 embedding)
        : GeneticUnit(sign, modifier, embedding), factorType(factorType) {
    }
    Gene(FactorType factorType, bool sign, float modifier, float3 embedding, float2 extra)
        : GeneticUnit(sign, modifier, embedding), factorType(factorType), extra(extra) {
    }
};

#endif