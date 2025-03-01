#include "geneticAlgorithm/lifeform.hpp"
#include "simulator/simulator.hpp"
#include "modules/cuda/updateGRN.hpp"

using namespace std;

LifeForm::LifeForm(Genome& genome)
    : genome(genome) {
    init();
    birthdate = Simulator::get().getStep();
}

void LifeForm::update() {
    // Update the GRN
    if (Simulator::get().getStep() - lastGrnUpdate > GROWTH_INTERVAL) {
        lastGrnUpdate = Simulator::get().getStep();
        updateGRN(*this, Simulator::get().getEnv().getPoints());
    }

    // Update NN
}

void LifeForm::render(VertexManager& vertexManager, vector<Cell>& hostCells, vector<CellLink>& cellLinks, vector<Point>& points) {
    // Render outline first
    for (int cell : cells) {
        hostCells[cell].renderCellWalls(vertexManager, points );
    }
    for (int link : links) {
        cellLinks[link].renderCellWalls(vertexManager, hostCells, points);
    }

    // Render body next
    for (int cell : cells) {
        hostCells[cell].renderBody(vertexManager, points);
    }
    for (int link : links) {
        cellLinks[link].renderBody(vertexManager, hostCells, points);
    }
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

void LifeForm::addCell(size_t motherIdx, const Cell& mother, const Point& point) {
    print("Add cell (?)", motherIdx);
    auto cell = Cell(idx, point.getPos(), point.radius);
    cell.generation = mother.generation;
    cell.hue = mother.hue;
    cell.saturation = mother.saturation;
    cell.luminosity = mother.luminosity;
    cell.products = mother.products.copy();
    cell.rotation = mother.rotation + mother.divisionRotation;

    cell.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(cell);
    cells.push(cell.idx);
    cellPointers.push(Simulator::get().getEnv().getCells().data() + cell.idx);

    auto cellLink = CellLink(idx,
        cell.idx,
        motherIdx,
        cell.pointIdx,
        mother.pointIdx,
        point.radius*2);
    const size_t linkIdx = Simulator::get().getEnv().nextCellLinkIdx();
    Simulator::get().getEnv().addCellLink(cellLink);
    links.push(linkIdx);
    cellLinkPointers.push(Simulator::get().getEnv().getCellLinks().data() + linkIdx);
}

void LifeForm::init(){
    energy = 0;
    numChildren = 0;
    birthdate = Simulator::get().getStep();
}