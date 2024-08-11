#ifndef SPECIES
#define SPECIES

#include <utility>
#include "vector"

class Individual;

/**
 * Species are collections of individuals with similar genetic makeup that can interbreed.
 */
class Species {
private:
    Species* parent{};
    std::vector<Individual*> members{};
    Individual* mascot;
    int originTime = 0;
    int deathTime = -1;

public:
    Species(Species* parent, Individual* mascot, int originTime) :
    parent(parent), mascot(mascot), originTime(originTime) {};

    void addCreature(Individual* creature);
    void removeCreature(Individual* creature);
};
#endif