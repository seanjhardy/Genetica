#ifndef GENE
#define GENE

#include "./geneticUnit.hpp"
#include <vector_types.h>
#include <modules/utils/floatOps.hpp>

/**
 * A gene is a genetic unit that can be expressed in a cell
 */
class Gene : public GeneticUnit {
public:
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

    Gene(FactorType factorType,
         bool sign,
         float modifier,
         const float* embedding,
         float2 extra={0,0})
      : GeneticUnit(sign, modifier, embedding),
        factorType(factorType), extra(extra){}

};

#endif