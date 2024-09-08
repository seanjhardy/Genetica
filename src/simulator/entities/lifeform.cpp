#include <utility>
#include "vector_types.h"
#include "simulator/entities/lifeform.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/geneticAlgorithm.hpp"
#include "modules/utils/print.hpp"
#include "simulator/simulator.hpp"

using namespace std;

int LifeForm::GROWTH_INTERVAL = 5;
float LifeForm::BUILD_COST_SCALE = 0.00001f;
float LifeForm::BUILD_RATE = 50.0f;
float LifeForm::ENERGY_DECREASE_RATE = 0.0000001;

LifeForm::LifeForm(Environment* environment, float2 pos,
                   Genome& genome)
    : Entity(pos), env(environment), genome(genome){
    sequence(this, genome);
    birthdate = Simulator::get().getStep();
}

void LifeForm::simulate(float dt) {
    pos = env->getPoint(head->pointIdx)->pos;
    grow(dt);

}

void LifeForm::render(VertexManager& vertexManager) {
    // Create mesh from cells;
}

void LifeForm::grow(float dt) {
    if (head == nullptr) return;

    if (Simulator::get().getStep() % GROWTH_INTERVAL != 0) return;

    grn.updateMorphogenLevels();
    for (auto& cell : cells) {
        cell->updateGeneExpression();
    }
}

LifeForm& LifeForm::clone(bool mutate){
    // Copy genome
    Genome copiedGenome = genome;
    // Create new LifeForm
    auto* clone = new LifeForm(getEnv(), pos, copiedGenome);
    // Mutate lifeForm genome
    if (mutate) {
        Simulator::get().getGA().mutate(clone->genome);
    }

    // Setup lifeForm parameters
    // TODO: Fix energy splitting
    float energyChange = energy * 0.5;
    energy -= energyChange;
    clone->energy += energyChange;
    numChildren++;

    // Assign to species and add to population
    //TODO: Simulator::get().getGA()..assignSpecies(clone);
    Simulator::get().getGA().addLifeForm(clone);
    return *clone;
}

LifeForm& LifeForm::combine(LifeForm* partner) {
    // Combine genomes
    Genome combinedGenome = crossover(genome,partner->genome);
    // Create new LifeForm
    auto *child = new LifeForm(getEnv(), pos, combinedGenome);

    // Mutate lifeForm genome
    Simulator::get().getGA().mutate(child->genome);

    // Setup child lifeForm parameters - share energy from parents
    auto* partnerLF = dynamic_cast<LifeForm*>(partner);
    float energyChange = energy * 0.25;
    float partnerEnergyChange = partnerLF->energy * 0.25;
    energy -= energyChange;
    partnerLF->energy -= partnerEnergyChange;
    child->energy += energyChange + partnerEnergyChange;
    numChildren++; partner->numChildren++;

    // Assign to species and add to population
    //TODO: Simulator::get().getGA()..assignSpecies(clone);
    Simulator::get().getGA().addLifeForm(child);
    return *child;
}

void LifeForm::kill() {
    //TODO: Implement this
    //env->removePoint(this->entityID);
}

void LifeForm::addInput(CellPartInstance* cellPartInstance) {
    inputs.push_back(cellPartInstance);
}

void LifeForm::addOutput(CellPartInstance* cellPartInstance) {
    outputs.push_back(cellPartInstance);
}

void LifeForm::init(){
    head = nullptr;
    cells.clear();
    inputs.clear();
    outputs.clear();
    energy = 0;
    numChildren = 0;
}

Environment* LifeForm::getEnv() const {
    return env;
}

Species* LifeForm::getSpecies() const {
    return species;
};