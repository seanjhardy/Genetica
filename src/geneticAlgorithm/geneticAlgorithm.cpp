#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include "modules/utils/random.hpp"
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include <simulator/simulator.hpp>
/*
void GeneticAlgorithm::render(VertexManager& vertexManager) {
    for (LifeForm* lifeform : population) {
        lifeform->render(vertexManager);
    }
};*/

void GeneticAlgorithm::mutate(Genome& genome) {
    // Clone genes
    for (auto& [key, value]: genome.getGenes()) {
        if (Random::random() < cloneGeneChance) {
            genome.addHoxGene(nextGeneID(),
                              value,
                              Random::random(genome.hoxGeneOrder.size()));
        }
    }

    // Insert new genes
    float insertRandom = Random::random();
    if (insertRandom < insertGeneChance) {
        string newGene;
        for(int i = 0; i < Genome::HOX_SIZE; i++) {
            newGene += Random::randomBase();
        }
        genome.addHoxGene(nextGeneID(),
                          newGene,
                          Random::random(genome.hoxGeneOrder.size()));
    }

    // Delete genes
    for (auto& [key, value]: genome.getGenes()) {
        if (Random::random() < deleteGeneChance) {
            genome.removeGene(key);
        }
    }

    // Mutate genes
    for (auto& [key, gene] : genome.hoxGenes) {
        mutateGene(gene);
    }
}

void GeneticAlgorithm::mutateGene(string& gene) const {
    string mutatedGene;

    for (int i = 0; i < gene.size(); i++) {
        int base = readBase(gene);

        // Mutate base
        if (Random::random() < mutateBaseChance) {
            base = rand() % 4;
        }

        // Delete base
        if (Random::random() > deleteBaseChance) {
            mutatedGene += base;
        }

        // Insert base
        if (Random::random() < insertBaseChance) {
            mutatedGene += Random::randomBase();
        }
    }
    gene = mutatedGene;
}

void GeneticAlgorithm::createRandomLifeForm() {
    Genome genome = Genome();
    genome.init();
    size_t genomeID = nextGenomeID();
    genomes.insert({genomeID, genome});

    float2 pos = Simulator::get().getEnv().randomPos();
    size_t id = Simulator::get().getEnv().getGA().population.getNextIndex();
    LifeForm lifeForm = LifeForm(id, &Simulator::get().getEnv(), genome, genomeID, pos);
    lifeForm.energy = 100;
    Simulator::get().getEnv().getGA().addLifeForm(lifeForm);
}

size_t GeneticAlgorithm::addLifeForm(const LifeForm& lifeform) {
    return population.push(lifeform);
}

void GeneticAlgorithm::reset() {
    population.clear();
    species.clear();
    ancestors.clear();
    speciesID = 0;
    geneID = 0;
}

GPUVector<LifeForm>& GeneticAlgorithm::getPopulation() {
    return population;
}

GPUVector<Species>& GeneticAlgorithm::getSpecies() {
    return species;
}

size_t GeneticAlgorithm::nextGeneID() {
    return geneID++;
}

size_t GeneticAlgorithm::nextSpeciesID() {
    return speciesID++;
}

size_t GeneticAlgorithm::nextGenomeID() {
    return genomeID++;
}