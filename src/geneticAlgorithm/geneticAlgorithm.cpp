#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include "modules/utils/random.hpp"
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include <simulator/simulator.hpp>

void GeneticAlgorithm::simulate() {
    for (auto& lifeForm : population) {
        lifeForm.update();
    }
};

void GeneticAlgorithm::render(VertexManager& vertexManager, GPUVector<Cell>& cells, GPUVector<CellLink>& cellLinks, GPUVector<Point>& points) {
    auto hostCells = vector<Cell>(cells.size());
    auto hostCellLinks = vector<CellLink>(cellLinks.size());
    auto hostPoints = vector<Point>(points.size());

    cudaMemcpy(hostCells.data(), cells.data(), cells.size() * sizeof(Cell), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCellLinks.data(), cellLinks.data(), cellLinks.size() * sizeof(CellLink), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPoints.data(), points.data(), points.size() * sizeof(Point), cudaMemcpyDeviceToHost);
    printf("cells size: %d, cellLinks size: %d, points size: %d\n", cells.size() * sizeof(Cell), cellLinks.size() * sizeof(CellLink), points.size() * sizeof(Point));

    for (LifeForm& lifeForm : population) {
        lifeForm.render(vertexManager, hostCells, hostCellLinks, hostPoints);
    }
}

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
    for (int i = 0; i < 1; i++) {
        auto genome = Genome();
        genome.init();

        float2 pos = Simulator::get().getEnv().randomPos();
        auto lifeForm = LifeForm(genome);
        lifeForm.energy = 100;

        addLifeForm(lifeForm);

        sequence(population[lifeForm.idx], genome, pos);
    }
}

size_t GeneticAlgorithm::addLifeForm(LifeForm& lifeForm) {
    lifeForm.idx = population.getNextIndex();
    population.push(lifeForm);
    return lifeForm.idx;
}


void GeneticAlgorithm::reset() {
    population.clear();
    species.clear();
    ancestors.clear();
    speciesID = 0;
    geneID = 0;
}

DynamicStableVector<LifeForm>& GeneticAlgorithm::getPopulation() {
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