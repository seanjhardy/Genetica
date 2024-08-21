#include "unordered_map"
#include "simulator/entities/lifeform.hpp"
#include "geneticAlgorithm/cellParts/cellPartType.hpp"

void sequence(LifeForm* lifeForm, const std::map<int, string>& genome);
std::shared_ptr<CellPartType> sequenceChromosome(int key, string chromosome);
void construct(LifeForm* lifeForm, int key, string chromosome);

std::map<int, string> plantGenome();