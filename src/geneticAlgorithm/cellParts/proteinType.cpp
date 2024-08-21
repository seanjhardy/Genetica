#include <geneticAlgorithm/cellParts/proteinType.hpp>
#include "simulator/entities/lifeform.hpp"

ProteinType::ProteinType(LifeForm* lifeform, int type, float partCode) : CellPartType(partCode, Type::PROTEIN) {
    this->type = type;
    this->size = 1.0f;
    this->parameters = std::vector<float>(4);
    for (int i = 0; i < 4; i++) {
        this->parameters[i] = Random::random();
    }
}