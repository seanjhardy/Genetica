#include <utility>
#include "vector_types.h"
#include "geneticAlgorithm/lifeform.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"

using namespace std;

LifeForm::LifeForm(size_t idx,
                   Environment* environment,
                   Genome& genome,
                   size_t genomeIdx,
                   float2 pos)
    : idx(idx), env(environment), genomeIdx(genomeIdx) {
    init();
    sequence(*this, genome, pos);
}

//void LifeForm::simulate(float dt) {
    //pos = env->getPoint(cells[0].pointIdx)->pos;
    /*for (auto& cell : cells) {
        cell->simulate(dt);
    }*/
    //grow(dt);
//}

/*void LifeForm::render(VertexManager& vertexManager) {
    // Create mesh from cells;
    /*for (auto& link : links.data()) {
        auto& cell1 = link->cell1;
        auto& cell2 = link->cell2;
        Point* point1 = env->getPoint(cell1->pointIdx);
        Point* point2 = env->getPoint(cell2->pointIdx);
        vertexManager.addLine(point1->pos, point2->pos, sf::Color::Blue, 0.2f);
    }
    for (auto& cell : cells) {
        cell->render(vertexManager);
    }
}*/

void LifeForm::grow(float dt) {
    if (Simulator::get().getStep() % GROWTH_INTERVAL == 0) {
        updateGRN(grn, cells, env->getCells(), env->getPoints());
    };
}
/*
int LifeForm::clone(bool mutate){
    // Copy genome
    Genome copiedGenome = genome;
    // Create new LifeForm

    // TODO: Fix this mess
    auto clone = LifeForm(getEnv(), copiedGenome, {0, 0});
    // Mutate lifeForm genome
    if (mutate) {
        Simulator::get().getEnv().getGA().mutate(clone.genome);
    }

    // Setup lifeForm parameters
    // TODO: Fix energy splitting
    float energyChange = energy * 0.5f;
    energy -= energyChange;
    clone.energy += energyChange;
    numChildren++;

    // Assign to species and add to population
    //TODO: Simulator::get().getEnv().getGA().assignSpecies(clone);
    return Simulator::get().getEnv().getGA().addLifeForm(clone);
}*/

/*
int LifeForm::combine(LifeForm* partner) {
    // Combine genomes
    Genome combinedGenome = crossover(genome,partner->genome);
    // Create new LifeForm
    auto child = LifeForm(getEnv(), combinedGenome);

    // Mutate lifeForm genome
    Simulator::get().getEnv().getGA().mutate(child.genome);

    // Setup child lifeForm parameters - share energy from parents
    auto* partnerLF = dynamic_cast<LifeForm*>(partner);
    float energyChange = energy * 0.25;
    float partnerEnergyChange = partnerLF->energy * 0.25;
    energy -= energyChange;
    partnerLF->energy -= partnerEnergyChange;
    child.energy += energyChange + partnerEnergyChange;
    numChildren++; partner->numChildren++;

    // Assign to species and add to population
    //TODO: Simulator::get().getEnv().getGA()..assignSpecies(clone);
    return Simulator::get().getEnv().getGA().addLifeForm(child);
}
*/
void LifeForm::kill() {
    //TODO: Implement this
    //env->removePoint(this->entityID);
}

void LifeForm::addCell(const Cell& cell) {
    size_t cellIdx = env->addCell(cell);
    cells.push(cellIdx);
}

void LifeForm::addCellLink(const CellLink& cellLink) {
    size_t cellLinkIdx = env->addCellLink(cellLink);
    links.push(cellLinkIdx);
}

void LifeForm::addInput(const Protein& protein) {
    inputs.push(protein);
}

void LifeForm::addOutput(const Protein& protein) {
    outputs.push(protein);
}

void LifeForm::init(){
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