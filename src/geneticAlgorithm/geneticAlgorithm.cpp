#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include "modules/utils/random.hpp"
#include "modules/utils/genomeUtils.hpp"
#include <simulator/simulator.hpp>

void GeneticAlgorithm::simulate() {
    for (auto& lifeForm : population) {
        lifeForm.update();
    }
}

void GeneticAlgorithm::render(VertexManager& vertexManager, GPUVector<Segment>& segments, GPUVector<Point>& points) {
    auto hostSegments = vector<Cell>(segments.size());
    auto hostPoints = vector<Point>(points.size());

    cudaMemcpy(hostSegments.data(), segments.data(), segments.size() * sizeof(Segment), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPoints.data(), points.data(), points.size() * sizeof(Point), cudaMemcpyDeviceToHost);

    for (LifeForm& lifeForm : population) {
        lifeForm.render(vertexManager, hostSegments, hostPoints);
    }
}

void GeneticAlgorithm::mutate(Genome& genome) {
    // Clone genes
    for (auto& [key, value] : genome.getGenes()) {
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
        for (int i = 0; i < Genome::HOX_SIZE; i++) {
            newGene += Random::randomBase();
        }
        genome.addHoxGene(nextGeneID(),
                          newGene,
                          Random::random(genome.hoxGeneOrder.size()));
    }

    // Delete genes
    for (auto& [key, value] : genome.getGenes()) {
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
    for (int i = 0; i < 1; i++) {
        auto genome = Genome();
        genome.init();

        float2 pos = Simulator::get().getEnv().randomPos();
        auto lifeFormIdx = population.getNextIndex();
        auto lifeForm = LifeForm(lifeFormIdx, genome, pos);
        population.push(lifeForm);
    }
}


void GeneticAlgorithm::reset() {
    population.clear();
    species.clear();
    ancestors.clear();
    speciesID = 0;
    geneID = 0;
}

dynamicStableVector<LifeForm>& GeneticAlgorithm::getPopulation() {
    return population;
}

vector<Species>& GeneticAlgorithm::getSpecies() {
    return species;
}

size_t GeneticAlgorithm::nextGeneID() {
    return geneID++;
}

size_t GeneticAlgorithm::nextSpeciesID() {
    return speciesID++;
}
