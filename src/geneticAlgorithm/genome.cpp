#include <geneticAlgorithm/genome.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/utils/random.hpp>
#include <simulator/simulator.hpp>

Genome::Genome() {
    int numHoxGenes = Random::random(1.0f, 10.0f);
    int geneLength = 85;
    for (int i = 0; i < numHoxGenes; i++) {
        string randomGene;
        for (int j = 0; j < geneLength; j++) {
            randomGene += Random::randomBase();
        }
        int index = Simulator::get().getGA().nextGeneID();
        hoxGenes.insert({index, randomGene});
        hoxOrder.push_back(index);
    }
}

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
        position = hoxOrder.size();
    }
    hoxOrder.insert(hoxOrder.begin() + position, key);
}

void Genome::removeGene(int key) {
    hoxGenes.erase(key);
    hoxOrder.erase(std::remove(hoxOrder.begin(), hoxOrder.end(), key), hoxOrder.end());
}

string Genome::toString() const {
    string genomeString;
    for (auto& [key, value]: hoxGenes) {
        genomeString += value;
    }
    return genomeString;
}

