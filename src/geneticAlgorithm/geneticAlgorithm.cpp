#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include "modules/utils/random.hpp"
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

LifeForm& GeneticAlgorithm::createRandomLifeForm() {
    Genome genome = Genome();
    genome.init();

    float2 pos = Simulator::get().getEnv().randomPos();
    auto* lifeForm = new LifeForm(&Simulator::get().getEnv(), pos, genome);
    lifeForm->energy = 100;
    Simulator::get().getGA().addLifeForm(lifeForm);
    return *lifeForm;
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
    return geneID++;
}

int GeneticAlgorithm::nextSpeciesID() {
    return speciesID++;
}