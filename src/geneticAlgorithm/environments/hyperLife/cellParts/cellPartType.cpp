#include <geneticAlgorithm/environments/hyperLife/cellParts/cellPartType.hpp>

CellPartType::CellPartType(int id, Type type)
: id(id), type(type) {

};

float CellPartType::getBuildCost() const {
    return 0;
}