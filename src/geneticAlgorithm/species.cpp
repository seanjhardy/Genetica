#include <geneticAlgorithm/species.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>

void Species::addCreature(Individual* creature) {
    members.push_back(creature);
    creature->getSpecies()->removeCreature(creature);
};


void Species::removeCreature(Individual* creature) {
    members.erase(std::remove(members.begin(), members.end(), creature), members.end());
    if (members.empty()) {
        deathTime = GeneticAlgorithm::get().time;
    }
};