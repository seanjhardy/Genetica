#ifndef GENOME_UTILS
#define GENOME_UTILS

#include "iostream"
#include "vector"
#include "geneticAlgorithm/lifeform.hpp"
#include "unordered_map"

using namespace std;

/**
 * Consumes the first base in the RNA string and returns it as an integer.
 * @param rna
 * @return
 */
class RNAExhaustedException : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override;
};

int readBase(string& rna);

// Consumes a range of bases from the RNA string;
float readBaseRange(string& rna, int length);
float readUniqueBaseRange(string& rna, int length);
float readExpBaseRange(string& rna, int length);

// Compare genetic similarity of two genes
float compareGeneBases(string gene1, string gene2);
float getCompatibility(const Genome& a, const Genome& b,
                      float geneDifferenceScalar, float baseDifferenceScalar);

Genome& crossover(const Genome& genome1,
                    const Genome& genome2,
                    float crossoverChance = 0.2f);
string crossoverGene(string gene1, string gene2,
                            float crossoverChance = 0.2f);


#endif