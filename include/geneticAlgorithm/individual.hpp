#ifndef INDIVIDUAL
#define INDIVIDUAL

#include <utility>

#include "vector"
#include <modules/graphics/vertexManager.hpp>
#include "environment.hpp"
#include "species.hpp"
#include "unordered_map"

using namespace std;

/**
 * An individual is a single entity in the genetic algorithm.
 * Individuals store their genome and are mutated, combined, and evaluated.
 */
class Species;

class Individual {
private:
    int id;
    unordered_map<int, string> genome;
    Species* species{};
    Environment* env;

public:

    Individual(Environment* env, unordered_map<int, string> genome);

    virtual void mutate() = 0;
    virtual Individual& combine(Individual *partner) = 0;
    virtual Individual& clone(bool mutate) = 0;
    virtual void init() {};
    virtual void simulate(float dt) = 0;
    virtual void render(VertexManager& viewer) = 0;

    unordered_map<int, string> getGenome() { return genome; };
    int getID() { return id; };
    Species* getSpecies() { return species; };

    void setGenome(const unordered_map<int, string>& genomeArr) { this->genome = genomeArr; };
    void setID(int ID) { this->id = ID; };
    virtual Environment* getEnv() { return env; };
};

#endif