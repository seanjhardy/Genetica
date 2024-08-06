#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <geneticAlgorithm/individual.hpp>

Individual::Individual(Environment* env, unordered_map<int, string> genome)
    : env(env), genome(std::move(genome)){
    id = GeneticAlgorithm::get().nextIndividualID();
};