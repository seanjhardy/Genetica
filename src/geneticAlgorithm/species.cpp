#include <geneticAlgorithm/species.hpp>
#include "geneticAlgorithm/lifeform.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <simulator/simulator.hpp>

void Species::addCreature(LifeForm* creature) {
    members.push_back(creature);
};


void Species::removeCreature(LifeForm* creature) {
    members.erase(std::remove(members.begin(), members.end(), creature), members.end());
    if (members.empty()) {
        deathTime = Simulator::get().getStep();
    }
};
