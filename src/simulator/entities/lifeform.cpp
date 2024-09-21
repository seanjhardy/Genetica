#include <utility>
#include "vector_types.h"
#include "simulator/entities/lifeform.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/geneticAlgorithm.hpp"
#include "modules/utils/print.hpp"
#include "simulator/simulator.hpp"

using namespace std;

LifeForm::LifeForm(Environment* environment, float2 pos,
                   Genome& genome)
    : Entity(pos), env(environment), genome(genome){
    init();
    sequence(this, genome);
}

void LifeForm::simulate(float dt) {
    if (head == nullptr) return;
    pos = env->getPoint(head->pointIdx)->pos;
    grow(dt);

}

void LifeForm::render(VertexManager& vertexManager) {
    // Create mesh from cells;
    for (auto& cell : cells) {
        cell->render(vertexManager);
    }
}

void LifeForm::grow(float dt) {
    if (head == nullptr) return;
    if (Simulator::get().getStep() % GROWTH_INTERVAL != 0) return;
    grn.update(dt);
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
    float energyChange = energy * 0.5f;
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

void LifeForm::addCell(Cell* cell) {
    cells.push_back(unique_ptr<Cell>(cell));
}

void LifeForm::addCellLink(CellLink* cellLink) {
    links.push_back(unique_ptr<CellLink>(cellLink));
}

void LifeForm::addInput(Protein* protein) {
    inputs.push_back(protein);
}

void LifeForm::addOutput(Protein* protein) {
    outputs.push_back(protein);
}

void LifeForm::init(){
    head = nullptr;
    cells.clear();
    inputs.clear();
    outputs.clear();
    energy = 0;
    numChildren = 0;
    birthdate = Simulator::get().getStep();

}

Environment* LifeForm::getEnv() const {
    return env;
}

Species* LifeForm::getSpecies() const {
    return species;
};