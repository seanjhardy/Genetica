#include <geneticAlgorithm/cellParts/cellPartSchematic.hpp>

CellPartSchematic::CellPartSchematic(CellPartType* type, bool isFlipped, int buildPriority,
                                     float angleOnBody, float angleFromBody):
  type(type), buildPriority(buildPriority),
  angleOnBody(angleOnBody), angleFromBody(angleFromBody),
  flipped(isFlipped) {
}