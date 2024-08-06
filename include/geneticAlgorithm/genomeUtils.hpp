#include "iostream"
#include "vector"
#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include "unordered_map"

using namespace std;

/**
 * Consumes the first base in the RNA string and returns it as an integer.
 * @param rna
 * @return
 */
int readBase(string& rna);

// Consumes a range of bases from the RNA string;
float readBaseRange(string& rna, int length);
float readUniqueBaseRange(string& rna, int length);
float readExpBaseRange(string& rna, int length);

// Compare genetic similarity of two genes
float compareGeneBases(string gene1, string gene2);
float getCompatibility(LifeForm* a, LifeForm* b,
                              float geneDifferenceScalar, float baseDifferenceScalar);

unordered_map<int, string> crossover(const unordered_map<int, string>& genome1,
                                                const unordered_map<int, string>& genome2,
                                                int header = 0, int cellDataSize = 1,
                                                float crossoverChance = 0.2f);
string crossoverGene(string gene1, string gene2,
                            int header = 0, int cellDataSize = 1,
                            float crossoverChance = 0.2f);
