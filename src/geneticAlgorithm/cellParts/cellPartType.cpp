#include <geneticAlgorithm/cellParts/cellPartType.hpp>

CellPartType::CellPartType(float partCode, Type type)
: partCode(partCode), type(type) {

};

float CellPartType::getBuildCost() const {
    return 0;
}