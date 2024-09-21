#include <iostream>
#include <utility>
#include "modules/utils/genomeUtils.hpp"

using namespace std;

// =====================================
// READ BASES
// =====================================

/**
 * An exception representing if the rna string is empty
 */
const char* RNAExhaustedException::what() const noexcept {
    return "RNA sequence exhausted";
}


/**
 * Reads the first base from the gene and removes it from the gene
 */
int readBase(string& rna) {
    if (rna.empty()) {
        throw RNAExhaustedException();
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
    float result = 0.0f;
    for (int i = 0; i < length; i++) {
        result += (float) readBase(rna);
    }
    return result / (3.0f * (float) length);
}

/**
 * Turns the base range into a unique number by taking
 * 0.25 * first base + 0.25^2 * second base + 0.25^3 * third base + ...
 * Used for things like random floats since they're uniformly distributed (but non-continuous)
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

float getCompatibility(const Genome& a, const Genome& b,
                       float geneDifferenceScalar, float baseDifferenceScalar) {
    float geneDifference = 0;
    float baseDifference = 0;

    for (auto& [key, value] : a.getGenes()) {
        if (!b.getGenes().contains(key)) {
            geneDifference += 1;
        } else {
            baseDifference += compareGeneBases(value, b.at(key));
        }
    }

    for (auto& [key, value]: b.getGenes()) {
        if (!a.getGenes().contains(key)) {
            geneDifference += 1;
        }
    }

    return geneDifference * geneDifferenceScalar + baseDifference * baseDifferenceScalar;
}

// =====================================
// CROSSOVER
// =====================================

Genome& crossover(const Genome& parent1,
                 const Genome& parent2, float crossoverChance) {
    Genome* childGenome = new Genome();

    // Add each gene in order of the earliest index in the parents
    // (to prevent duplicate genes being added)
    map<int, bool> hoxGenesAdded;
    vector<int> hoxOrder;
    int size = max(parent1.hoxGeneOrder.size(), parent2.hoxGeneOrder.size());
    for (int i = 0; i < size; i++) {
        if (i < parent1.hoxGeneOrder.size() && !hoxGenesAdded.contains(parent1.hoxGeneOrder[i])) {
            hoxOrder.push_back(parent1.hoxGeneOrder[i]);
            hoxGenesAdded.insert({parent1.hoxGeneOrder[i], true});
        }
        if (i < parent2.hoxGeneOrder.size() && !hoxGenesAdded.contains(parent2.hoxGeneOrder[i])) {
            hoxOrder.push_back(parent2.hoxGeneOrder[i]);
            hoxGenesAdded.insert({parent2.hoxGeneOrder[i], true});
        }
    }

    // Crossover genes from both parents
    for (auto& [key, value]: parent1.getGenes()) {
        if (parent2.contains(key)) {
            childGenome->addHoxGene(key, crossoverGene(value,
                                          parent2.at(key),
                                          crossoverChance));
        } else {
            // Add all genes from parent 1 not in parent 2
            childGenome->addHoxGene(key, value);
        }
    }

    for (auto& [key, value]: parent2.getGenes()) {
        if (!parent1.contains(key)) {
            // Add all genes from parent 2 not in parent 1
            childGenome->addHoxGene(key, value);
        }
    }

    return *childGenome;
}

string crossoverGene(string gene1, string gene2, float crossoverChance) {
    int geneLength = max(gene1.size(), gene2.size());

    bool parent = Random::randomBool();
    string childGene;

    //Crossover headers
    for (int i = 0; i < geneLength; i++) {
        // Randomly switch mode after the header, at intervals of "cellDataSize"
        // (when header is 0 and cellDataSize is 1, this is effectively the same as random switching)
        if (Random::random() < crossoverChance) {
            parent = !parent;
        }

        if (i < gene1.size() && i < gene2.size()) {
            childGene += parent ? gene1[i] : gene2[i];
        } else if (i < gene1.size()) {
            childGene += gene1[i];
        } else {
            childGene += gene2[i];
        }
    }

    return childGene;
}
