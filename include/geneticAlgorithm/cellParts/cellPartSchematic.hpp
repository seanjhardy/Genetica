#ifndef CELL_PART_DATA
#define CELL_PART_DATA

class CellPartType;

/**
 * A cellPartSchematic represents the concept of a particular instance of a given cell part,
 * and how it relates to a parent.
 *
 * For instance, imagine a module defining a single hair, its structure is the same across the body,
 * but on one part of the body its angled one way, and grows at a certain rate,
 * and on another part of the body its angled another way, and grows a different amount.
 */
class CellPartSchematic {
public:
    CellPartType* type;
    int buildPriority;
    float angleOnBody, angleFromBody;
    bool flipped;

    CellPartSchematic(CellPartType* partType, bool isFlipped, int buildPriority,
                      float angleOnBody, float angleFromBody);
};

#endif