#include <iostream>
#include <utility>
#include "modules/utils/genomeUtils.hpp"

using namespace std;

// =====================================
// READ BASES
// =====================================

/**
 * Reads the first base from the gene and removes it from the gene
 */
int readBase(string& rna) {
    if (rna.empty()) {
        return -1;
    }
    int base = rna[0] - '0';
    rna.erase(0, 1);
    return base;
}

/**
 * Reads the base linearly, adding up each bases contribution to the total
 * Used for most functions as it is differentiable and linear
 */
float readBaseRange(string& rna, int length) {
    float result = 0;
    for (int i = 0; i < length; i++) {

        result += (float) readBase(rna);
    }
    return result / (3.0f * (float) length);
}

/**
 * Turns the base range into a unique number by taking
 * 0.25 * first base + 0.25^2 * second base + 0.25^3 * third base + ...base
 * Used for things like part IDs since they're non-differentiable
 */
float readUniqueBaseRange(string& rna, int length) {
    float result = 0;
    for(int i = 0; i < length; i++){
        result += float(readBase(rna) * pow(0.25f, i + 1));
    }
    return result;
}

/**
 * Essentially extreme base ranges (either all 3s or all 0s) take extreme values whereas most are average
 * Exponential tail ends of the distribution, slighly weighted to smaller values ~0.35 on average
 * Used for things like size where we want nonlinear effects at the tail ends so it gets harder
 * to make progress to extreme values
 */
float readExpBaseRange(string& gene, int length) {
    float result = readBaseRange(gene, length);
    result = pow(1.45f * result - 0.6f, 3.0f) + result / 5.0f + 0.25f;
    return result;
}

float compareGeneBases(string gene1, string gene2) {
    string& gene1Cpy = gene1;
    string& gene2Cpy = gene2;
    int baseDiff = 0;
    int maxLength = max(gene1.length(), gene2.length());
    for (int i = 0; i < maxLength; i++) {
        int base1 = readBase(gene1Cpy);
        int base2 = readBase(gene2Cpy);
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

map<int, string> crossover(const map<int, string>& parent1,
                             const map<int, string>& parent2,
                             int header, int cellDataSize, float crossoverChance) {
    std::map<int, string> childGenome;
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
            crossover = Random::random() < crossoverChance;
            parent = Random::randomBool();
        }

        // Randomly combine both parents
        if (crossover) {
            if (i < gene1.size() && i < gene2.size()) {
                childGene += Random::randomBool() ? gene1[i] : gene2[i];
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