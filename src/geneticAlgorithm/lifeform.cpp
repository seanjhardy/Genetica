#include "geneticAlgorithm/lifeform.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"

using namespace std;

LifeForm::LifeForm(Genome& genome)
    : genome(genome) {
    init();
}

void LifeForm::update() {
    // Update the GRN
    if (Simulator::get().getStep() - lastGrnUpdate > GROWTH_INTERVAL) {
        lastGrnUpdate = Simulator::get().getStep();
        updateGRN(*this,
            Simulator::get().getEnv().getCells(),
            Simulator::get().getEnv().getCellLinks(),
            Simulator::get().getEnv().getPoints());
    }

    // Update NN
}


void LifeForm::clone(bool mutate){
    // Copy genome

    Genome copiedGenome = genome;
    // Create new LifeForm
    auto clone = LifeForm(copiedGenome);
    Simulator::get().getEnv().getGA().addLifeForm(clone);

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
}

/*
void LifeForm::combine(LifeForm* partner) {
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
}*/

void LifeForm::kill() {
    //TODO: Implement this
    //env->removePoint(this->entityID);
}


void LifeForm::addCell(const LfUpdateData::NEW_CELL& newCell) {
    auto cell = Cell(idx, newCell.pos, newCell.radius);
    cell.generation = newCell.generation;
    cell.hue = newCell.hue;
    cell.saturation = newCell.saturation;
    cell.luminosity = newCell.luminosity;
    //cell.products = newCell.products.copy();
    cell.rotation = newCell.rotation + newCell.divisionRotation;

    cell.idx = Simulator::get().getEnv().addCell(cell);
    cells.push(cell.idx);

    auto cellLink = CellLink(idx,
        cell.idx,
        newCell.motherIdx,
        cell.pointIdx,
        newCell.motherPointIdx,
        newCell.radius * 3);
    Simulator::get().getEnv().addCellLink(cellLink);
}

void LifeForm::init(){
    energy = 0;
    numChildren = 0;
    birthdate = Simulator::get().getStep();
}