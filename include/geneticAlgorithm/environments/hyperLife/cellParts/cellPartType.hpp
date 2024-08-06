#ifndef CELL_PART
#define CELL_PART

#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include "SFML/Graphics.hpp"

/**
 * An abstract class representing the blueprint for building a cell part.
 * One cell part type can exist in multiple places on one lifeform, and can be represented
 * as a segment, or a protein.
 */
class CellPartType {
public:
    enum class Type {
        SEGMENT,
        PROTEIN,
    };

    LifeForm* lifeForm;
    int id;
    Type type;

    CellPartType(LifeForm* lifeForm, int id, Type type);

    virtual float getBuildCost() const = 0;

    friend class SegmentInstance;
};

#endif