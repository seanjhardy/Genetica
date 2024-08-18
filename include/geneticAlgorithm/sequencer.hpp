#include "unordered_map"
#include "geneticAlgorithm/entities/lifeform.hpp"
#include "geneticAlgorithm/cellParts/cellPartType.hpp"

void sequence(LifeForm* lifeForm, const std::unordered_map<int, string>& genome);
std::shared_ptr<CellPartType> sequenceChromosome(int key, string chromosome);
void construct(LifeForm* lifeForm, int key, string chromosome);

std::unordered_map<int, string> plantGenome();