#include "unordered_map"
#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/cellPartType.hpp>

std::unordered_map<int, CellPartType&> sequence(LifeForm* lifeForm, const std::unordered_map<int, string>& genome);
std::pair<int, CellPartType&> sequenceChromosome(LifeForm* lifeForm, int key, string chromosome);
void construct(LifeForm* lifeForm, int key, string chromosome);