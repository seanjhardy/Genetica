#include <geneticAlgorithm/genome.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/utils/random.hpp>
#include <simulator/simulator.hpp>

Genome::Genome() {}

bool Genome::contains(int key) const {
    return hoxGenes.contains(key);
}

string Genome::at(int key) const {
    return hoxGenes.at(key);
}

map<int, string> Genome::getGenes() const {
    return hoxGenes;
}

void Genome::addHoxGene(int key, const string& value, int position) {
    hoxGenes.insert({key, value});
    if (position == -1) {
        position = hoxGeneOrder.size();
    }
    hoxGeneOrder.insert(hoxGeneOrder.begin() + position, key);
}

void Genome::removeGene(int key) {
    hoxGenes.erase(key);
    hoxGeneOrder.erase(std::remove(hoxGeneOrder.begin(), hoxGeneOrder.end(), key), hoxGeneOrder.end());
}

string Genome::toString() const {
    string genomeString;
    for (auto& [key, value]: hoxGenes) {
        genomeString += value;
    }
    return genomeString;
}

/**
 * DEFINE TEMPLATE GENERATORS
 */

void Genome::init(Template templateType) {
    if (templateType == Template::RANDOM) {
        int numHoxGenes = (int) Random::random(1, 50);
        int geneLength = 85;
        for (int i = 0; i < numHoxGenes; i++) {
            string randomGene;
            for (int j = 0; j < geneLength; j++) {
                randomGene += Random::randomBase();
            }
            int index = Simulator::get().getGA().nextGeneID();
            hoxGenes.insert({index, randomGene});
            hoxGeneOrder.push_back(index);
        }
    }

    if (templateType == Template::PROKARYOTE) {

    }
}


