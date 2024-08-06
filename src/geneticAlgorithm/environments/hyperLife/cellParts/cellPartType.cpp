#include <geneticAlgorithm/environments/hyperLife/cellParts/cellPartType.hpp>

CellPartType::CellPartType(LifeForm* lifeForm, int id, Type type)
: lifeForm(lifeForm), id(id), type(type) {

};

float CellPartType::getBuildCost() const {
    return 0;
}