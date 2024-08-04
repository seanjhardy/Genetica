#include <iostream>
#include <utility>
#include "geneticAlgorithm/genomeUtils.hpp"
#include "modules/utils/mathUtils.hpp"

using namespace std;

// =====================================
// READ BASES
// =====================================

int readBase(string& rna) {
    if (rna.empty()) {
        return -1;
    }
    int base = rna[0] - '0';
    rna.erase(0, 1);
    return base;
}

float readBaseRange(string& rna, int length) {
    float result = 0;
    for (int i = 0; i < length; i++) {
        result += (float) readBase(rna);
    }
    return result / (4.0f * (float) length);
}

float readUniqueBaseRange(string& rna, int length) {
    float result = 0;

    //inclusive
    for(int i = 0; i < length; i++){
        result += float(readBase(rna) * pow(0.25f, i + 1));
    }
    return result;
}

float readExpBaseRange(string& gene, int length) {
    float result = readBaseRange(gene, length);
    result = pow(1.45f * result - 0.6f, 3.0f) + result / 5.0f + 0.2f;
    return result;
}

float compareGeneBases(string gene1, string gene2) {
    int baseDiff = 0;
    int maxLength = max(gene1.length(), gene2.length());
    for (int i = 0; i < maxLength; i++) {
        int base1 = readBase(gene1);
        int base2 = readBase(gene2);
        baseDiff += (base1 != base2);
    }
    return baseDiff;
}


// =====================================
// COMPATIBILITY
// =====================================

float getCompatibility(LifeForm* a, LifeForm* b, float geneDifferenceScalar, float baseDifferenceScalar) {
    float geneDifference = 0;
    float baseDifference = 0;

    for (auto& [key, value]: a->getGenome()) {
        if (!b->cellParts.contains(key)) {
            geneDifference += 1;
        } else {
            baseDifference += compareGeneBases(value, b->getGenome().at(key));
        }
    }

    for (auto& [key, value]: b->cellParts) {
        if (!a->cellParts.contains(key)) {
            geneDifference += 1;
        }
    }

    return geneDifference * geneDifferenceScalar + baseDifference * baseDifferenceScalar;
}

// =====================================
// CROSSOVER
// =====================================

unordered_map<int, string> crossover(const unordered_map<int, string>& parent1,
                                     const unordered_map<int, string>& parent2,
                                     int header, int cellDataSize, float crossoverChance) {
    std::unordered_map<int, string> childGenome;
    for (auto& [key, value]: parent1) {
        if (parent2.contains(key)) {
            if (key == 0) {
                // Treat the header gene as a special case where the "header" is the full length of the gene
                // So it either crosses over entirely, or takes either parent's gene
                childGenome.insert({key,
                                    crossoverGene(value, parent2.at(key),
                                                  max(parent1.size(), parent2.size()))});
            } else {
                childGenome.insert({key,
                                    crossoverGene(value, parent2.at(key),
                                                  header, cellDataSize, crossoverChance)});
            }
        } else {
            childGenome.insert({key, value});
        }
    }

    for (auto& [key, value]: parent2) {
        if (!parent1.contains(key)) {
            childGenome.insert({key, value});
        }
    }

    return childGenome;
}

string crossoverGene(string gene1, string gene2, int header, int cellDataSize, float crossoverChance) {
    int geneLength = max(gene1.size(), gene2.size());
    int headerSize = min(geneLength, header);
    bool crossover, parent;
    string childGene;
    //Crossover headers
    for (int i = 0; i < geneLength; i++) {
        // Randomly switch mode after the header, at intervals of "cellDataSize"
        // (when header is 0 and cellDataSize is 1, this is effectively the same as random switching)
        if (i >= headerSize and ((i - headerSize) % cellDataSize == 0)) {
            crossover = getRandom() < crossoverChance;
            parent = randomBool();
        }

        // Randomly combine both parents
        if (crossover) {
            if (i < gene1.size() && i < gene2.size()) {
                childGene += randomBool() ? gene1[i] : gene2[i];
            } else if (i < gene1.size()) {
                childGene += gene1[i];
            } else {
                childGene += gene2[i];
            }
        } else {
            // Take base pairs from one parent
            if (parent) {
                childGene += i < gene1.size() ? gene1[i] : gene2[i];
            } else {
                childGene += i < gene2.size() ? gene2[i] : gene1[i];
            }
        }
    }

    return childGene;
}