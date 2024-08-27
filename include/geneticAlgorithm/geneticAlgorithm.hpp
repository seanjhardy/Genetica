#ifndef GENETIC_ALGORITHM
#define GENETIC_ALGORITHM

#include <vector>
#include <geneticAlgorithm/species.hpp>
#include <string>
#include <map>
#include <modules/graphics/vertexManager.hpp>

using namespace std;

/**
 * Genetic Algorithm class
 * Maintains population of lifeforms and species
 */
class GeneticAlgorithm {
private:
    int MAX_POPULATION = -1;
    int MAX_CHROMOSOMES = 100;
    int MAX_CHROMOSOME_SIZE = 200;

    // Species parameters
    float geneDifferenceScalar = 0.5f;
    float baseDifferenceScalar = 0.1f;
    float compatabilityDistanceThreshold = 5.0f;

    // Chromosome mutations
    float insertChromosomeChance = 0.00004f;
    float cloneChromosomeChance = 0.0001f;
    float deleteChromosomeChance = 0.000005f;

    // Base mutations
    float cloneBaseChance = 0.0000000001f;
    float mutateSectionLocationChance = 0.005f;
    float mutateBaseChance = 0.0005f;
    float insertBaseChance = 0.00005f;
    float deleteBaseChance = 0.00005f;
    float crossoverCellDataChance = 0.2f;// 0 = mix bases, 1 = use one parent

    vector<LifeForm*> population{};
    vector<Species*> species{};
    vector<Species*> ancestors{};
    int speciesID = 0;
    int geneID = 0;
    int lifeFormID = 0;

public:
    void simulate(float dt);
    void render(VertexManager& vertexManager);
    void reset();

    void addLifeForm(LifeForm* lifeform);
    map<int, string> mutate(const map<int, string>& genome,
                                      int headerSize, int cellDataSize);
    [[nodiscard]] string mutateGene(map<int, string> genome, int key, string gene,
                      int headerSize, int cellDataSize) const;
    map<int, string> createRandomGenome();
    LifeForm& createRandomLifeForm();

    vector<LifeForm*> getPopulation();
    vector<Species*> getSpecies();

    int nextGeneID();
    int nextSpeciesID();
};

#endif