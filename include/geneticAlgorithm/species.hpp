#ifndef SPECIES
#define SPECIES

#include <utility>
#include <vector>

class LifeForm;

/**
 * Species are collections of individuals with similar genetic makeup that can interbreed.
 */
class Species {
private:
    Species* parent{};
    std::vector<LifeForm*> members{};
    LifeForm* mascot;
    int originTime = 0;
    int deathTime = -1;

public:
    Species(Species* parent, LifeForm* mascot, int originTime) :
    parent(parent), mascot(mascot), originTime(originTime) {};

    void addCreature(LifeForm* creature);
    void removeCreature(LifeForm* creature);
};
#endif