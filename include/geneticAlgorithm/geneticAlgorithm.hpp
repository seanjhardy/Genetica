#ifndef GENETIC_ALGORITHM
#define GENETIC_ALGORITHM

#include <vector>
#include <geneticAlgorithm/environment.hpp>
#include <geneticAlgorithm/species.hpp>

using namespace std;

/**
 * Genetic Algorithm class
 * Maintains population of lifeforms and species
 */
class GeneticAlgorithm {
private:

    GeneticAlgorithm() = default;
    GeneticAlgorithm(GeneticAlgorithm const&);              // Don't Implement
    void operator=(GeneticAlgorithm const&); // Don't implement

    int MAX_POPULATION = -1;
    int MAX_CHROMOSOMES = 100;
    int MAX_CHROMOSOME_SIZE = 200;

    // Species parameters
    float geneDifferenceScalar = 0.5f;
    float baseDifferenceScalar = 0.1f;
    float compatabilityDistanceThreshold = 5.0f;

    // Gene mutations
    float insertGeneChance = 0.0004f;
    float cloneGeneChance = 0.001f;
    float deleteGeneChance = 0.00005f;

    // Base mutations
    float cloneBaseChance = 0.0f;
    float mutateSectionLocationChance = 0.005f;
    float mutateBaseChance = 0.0005f;
    float insertBaseChance = 0.00005f;
    float deleteBaseChance = 0.00005f;
    float crossoverCellDataChance = 0.2f;// 0 = mix bases, 1 = use one parent

    Environment* env{};
    vector<LifeForm*> population{};
    vector<Species*> species{};
    vector<Species*> ancestors{};
    int speciesID = 0;
    int geneID = 0;
    int lifeFormID = 0;

public:
    static GeneticAlgorithm& get();

    int step = 0;

    void simulate(float dt);
    void render(VertexManager& vertexManager);
    void reset();

    void addLifeForm(LifeForm* lifeform);
    unordered_map<int, string> mutate(const unordered_map<int, string>& genome,
                                      int headerSize, int cellDataSize);
    [[nodiscard]] string mutateGene(unordered_map<int, string> genome, int key, string gene,
                      int headerSize, int cellDataSize) const;
    unordered_map<int, string> createRandomGenome();

    void setEnvironment(Environment& env);
    [[nodiscard]] Environment* getEnv() const;

    vector<LifeForm*> getPopulation();
    vector<Species*> getSpecies();

    int nextLifeFormID();
    int nextGeneID();
    int nextSpeciesID();
};

#endif