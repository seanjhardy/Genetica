#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/noise/random.hpp>
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include <simulator/simulator.hpp>

void GeneticAlgorithm::simulate(float dt) {
    for (LifeForm* lifeform : population) {
        lifeform->simulate(dt);
    }
};

void GeneticAlgorithm::render(VertexManager& vertexManager) {
    for (LifeForm* lifeform : population) {
        lifeform->render(vertexManager);
    }
};

map<int, string> GeneticAlgorithm::mutate(const map<int, string>& genome,
                                            int headerSize, int cellDataSize) {
    map<int, string> mutatedGenome;
    // Clone genes
    for (auto& [key, value]: genome) {
        mutatedGenome.insert({key, value});
        if (Random::random() < cloneChromosomeChance) {
            mutatedGenome.insert({nextGeneID(), value});
        }
    }
    // Insert new genes
    float insertRandom = Random::random();
    if (insertRandom < insertChromosomeChance) {
        string newChromosome;
        int size = Random::random(headerSize, MAX_CHROMOSOME_SIZE);
        for(int i = 0; i < size; i++) {
            //TODO: Fix random character generation
            newChromosome += std::to_string(rand() % 4);
        }
        mutatedGenome.insert({nextGeneID(), newChromosome});
    }

    // Mutate genes
    for (auto& [key, value]: mutatedGenome) {
        string mutatedChromosome = mutateGene(mutatedGenome, key, value, headerSize, cellDataSize);
    }

    // Add cloned genes to the end of the chromosomes
    for (auto& [key, value]: mutatedGenome) {
        for(int i = 0; i <value.size(); i++) {
            int adjustedIndex = i - headerSize - (cellDataSize - 1);
            if(Random::random() < cloneBaseChance &&
            i >= headerSize && adjustedIndex % cellDataSize == 0){
                string lastItems = value.substr(i - (cellDataSize - 1),i + 1);
                value += lastItems;
            }
        }
        mutatedGenome.insert({key, value});
    }
    return mutatedGenome;
}

string GeneticAlgorithm::mutateGene(map<int, string> genome,
                                    const int key, string gene,
                                    int headerSize, int cellDataSize) const {
    string mutatedGene = gene;
    for (int i = 0; i < gene.size(); i++) {
        int base = readBase(gene);
        float mutateParentGene = Random::random();
        int adjustedIndex = i - headerSize - (cellDataSize - 1);
        // Mutate base
        if (Random::random() < mutateBaseChance) {
            base = rand() % 4;
        }
        // Delete base
        if (Random::random() > deleteBaseChance &&
            !(mutateParentGene < mutateSectionLocationChance &&
              i >= headerSize &&
              (adjustedIndex % cellDataSize == 0))) {
            mutatedGene += base;
        }

        if (i >= headerSize && adjustedIndex % cellDataSize == 0) {
            // Insert new section
            if (Random::random() < insertChromosomeChance) {
                for (int j = 0; i < cellDataSize; j++) {
                    mutatedGene += rand() % 4;
                }
            }
            //Change location of a section
            if (mutateParentGene < mutateSectionLocationChance) {
                string section = gene.substr(i - (cellDataSize - 1), i + 1);
                int randomIndex = rand() % genome.size();
                auto it = genome.begin();
                std::advance(it, randomIndex);
                int randomKey = it->first;
                if (randomKey == key) {
                    mutatedGene += section;
                } else {
                    genome.insert({randomKey,
                                   genome.at(randomKey) + section});
                }
            }
        }
    }
    return mutatedGene;
}

map<int, string> GeneticAlgorithm::createRandomGenome() {
    if (true) {
        return plantGenome();
    }
    // Create random genome
    map<int, string> genome;
    int num_chromosomes = (int)Random::random(2, 5);
    for (int i = 0; i < num_chromosomes; i++) {
        string chromosome;
        int size = LifeForm::HEADER_SIZE + LifeForm::CELL_DATA_SIZE * Random::random(0, 3);
        for (int j = 0; j < size; j++) {
            chromosome += Random::randomBase();
        }
        genome.insert({nextGeneID(), chromosome});
    }
    return genome;
}

void GeneticAlgorithm::addLifeForm(LifeForm* lifeform) {
    population.push_back(lifeform);
}

void GeneticAlgorithm::reset() {
    population.clear();
    species.clear();
    ancestors.clear();
    speciesID = 0;
    geneID = 0;
}

vector<LifeForm*> GeneticAlgorithm::getPopulation() {
    return population;
}

vector<Species*> GeneticAlgorithm::getSpecies() {
    return species;
}

int GeneticAlgorithm::nextGeneID() {
    return lifeFormID++;
}

int GeneticAlgorithm::nextSpeciesID() {
    return speciesID++;
}