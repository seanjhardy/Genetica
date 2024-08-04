#include "unordered_map"
#include "geneticAlgorithm/environments/hyperLife/lifeform.hpp"
#include "geneticAlgorithm/environments/hyperLife/cellParts/cellPartType.hpp"

inline unordered_map<int, CellPartType&> sequence(LifeForm* lifeForm, const unordered_map<int, string>& genome);
inline std::pair<int, CellPartType&> sequenceChromosome(LifeForm* lifeForm, int key, string chromosome);
inline void construct(LifeForm* lifeForm, int key, string chromosome);