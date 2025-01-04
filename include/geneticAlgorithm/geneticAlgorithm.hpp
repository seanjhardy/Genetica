#ifndef GENETIC_ALGORITHM
#define GENETIC_ALGORITHM

#include <vector>
#include <geneticAlgorithm/species.hpp>
#include <string>
#include <map>
#include <modules/graphics/vertexManager.hpp>
#include <geneticAlgorithm/genome.hpp>
#include <modules/cuda/structures/GPUVector.hpp>

using namespace std;

/**
 * Genetic Algorithm class
 * Maintains population of lifeforms and species
 */
class GeneticAlgorithm {
private:
    int MAX_POPULATION = -1;
    int MAX_GENES = 500;

    // Species parameters
    float geneDifferenceScalar = 0.5f;
    float baseDifferenceScalar = 0.1f;
    float compatabilityDistanceThreshold = 5.0f;

    // Chromosome mutations
    float insertGeneChance = 0.00005f;
    float cloneGeneChance = 0.0001f;
    float deleteGeneChance = 0.0005f;

    // Base mutations
    float mutateBaseChance = 0.0005f;
    float insertBaseChance = 0.00003f;
    float deleteBaseChance = 0.00005f;
    float crossoverCellDataChance = 0.2f;// probability of switching from one parent to another

    std::map<size_t, Genome> genomes;

    GPUVector<LifeForm> population{};
    GPUVector<Species> species{};
    GPUVector<Species> ancestors{};

    size_t genomeID = 0;
    size_t geneID = 0;
    size_t speciesID = 0;

public:
    //void render(VertexManager& vertexManager);
    void reset();

    size_t addLifeForm(const LifeForm& lifeform);
    void mutate(Genome& genome);
    void mutateGene(string& gene) const;

    void createRandomLifeForm();

    GPUVector<LifeForm>& getPopulation();
    GPUVector<Species>& getSpecies();

    size_t nextGeneID();
    size_t nextSpeciesID();
    size_t nextGenomeID();
};

#endif