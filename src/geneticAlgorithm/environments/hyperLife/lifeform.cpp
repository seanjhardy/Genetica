#include <utility>

#include "vector_types.h"
#include "geneticAlgorithm/environments/hyperLife/lifeform.hpp"
#include "geneticAlgorithm/environments/hyperLife/sequencer.hpp"
#include "geneticAlgorithm/genomeUtils.hpp"

int LifeForm::HEADER_SIZE = 35;
int LifeForm::CELL_DATA_SIZE = 10;
float LifeForm::BUILD_COST_SCALE = 5.0f;

LifeForm::LifeForm(HyperLife* hyperLife, float2 pos, unordered_map<int, string> genome)
    : Individual(hyperLife, std::move(genome)), pos(pos) {
    sequence(this, genome);
}

void LifeForm::simulate(float dt) {
    pos = head->startPoint->pos;
    if (GeneticAlgorithm::get().time % randomTime) {
        grow(dt);
        //TODO: Implement neural network
    }
}

void LifeForm::render(VertexManager& vertexManager) {
    for (auto& cellPartInstance : cellPartInstances) {
        cellPartInstance->render(vertexManager);
    }
}

void LifeForm::grow(float dt) {
    if (currentGrowthEnergy >= growthEnergy) return;
    if (head == nullptr) return;
    if (buildQueue.empty()) return;

    auto toBuild = buildQueue.rbegin();
    SegmentInstance* buildFrom = toBuild->second.first;
    CellPartSchematic* buildSchematic = toBuild->second.second;
    CellPartType* buildType = buildSchematic->type;

    if (buildType->type == CellPartType::Type::SEGMENT) {
        float energyCost = dynamic_cast<SegmentType*>(buildType)->getBuildCost() * CellPartInstance::initialSize;
        if (energy < energyCost * (1 - growthPriority)) return;

        energy -= energyCost;
        currentGrowthEnergy += energyCost;

        auto* segment = new SegmentInstance(this, buildSchematic, buildFrom);
        addCellPartInstance(segment);
        bool centerAligned = (segment->getAdjustedAngleOnBody() == (float)M_PI ||
                              segment->getAdjustedAngleOnBody() == 0) && segment->getAdjustedAngleFromBody() == 0;
        if (buildFrom->centered && centerAligned) {
            segment->centered = true;
        }
        buildFrom->children.push_back(segment);
    }
}

Individual& LifeForm::clone(bool mutate){
    // Copy genome
    unordered_map<int, string> copiedGenome = Individual::getGenome();
    // Create new LifeForm
    auto* clone = new LifeForm(dynamic_cast<HyperLife*>(Individual::getEnv()), pos, copiedGenome);
    // Mutate lifeForm genome
    if (mutate) clone->mutate();

    // Setup lifeForm parameters
    float energyChange = energy * childEnergy;
    energy -= energyChange;
    clone->energy += energyChange;
    clone->currentGrowthEnergy = 0;

    // Assign to species and add to population
    //TODO: GeneticAlgorithm::get().assignSpecies(clone);
    GeneticAlgorithm::get().addIndividual(clone);
    return *clone;
}

Individual& LifeForm::combine(Individual* partner) {
    // Combine genomes
    unordered_map<int, string> combinedGenome = crossover(Individual::getGenome(),
                                                          partner->getGenome(),
                                                          HEADER_SIZE, CELL_DATA_SIZE);
    // Create new LifeForm
    auto *child = new LifeForm(dynamic_cast<HyperLife *>(Individual::getEnv()), pos, combinedGenome);
    // Mutate lifeForm genome
    child->mutate();

    // Setup child lifeForm parameters - share energy from parents
    auto* partnerLF = dynamic_cast<LifeForm*>(partner);
    float energyChange = energy * childEnergy + partnerLF->energy * partnerLF->childEnergy;
    energy -= energyChange;
    child->energy += energyChange;
    child->currentGrowthEnergy = 0;

    // Assign to species and add to population
    //TODO: GeneticAlgorithm::get().assignSpecies(clone);
    GeneticAlgorithm::get().addIndividual(child);
    return *child;
}

void LifeForm::mutate() {
    unordered_map<int, string> mutatedGenome = GeneticAlgorithm::get().mutate(Individual::getGenome(),
                                                            HEADER_SIZE, CELL_DATA_SIZE);
    setGenome(mutatedGenome);
    sequence(this, mutatedGenome);
}

void LifeForm::addCellPartInstance(CellPartInstance* cellPartInstance){
    cellPartInstances.push_back(cellPartInstance);

    if (cellPartInstance->cellData->type->type != CellPartType::Type::SEGMENT)  return;

    // Automatically insert all child build orders by priority
    for (auto& child : dynamic_cast<SegmentType*>(cellPartInstance->cellData->type)->children) {
        // Calculate build priority (shallow nodes win in tiebreakers)
        int buildPriority = child.buildPriority - cellPartInstance->depth;
        auto buildData = std::make_pair(dynamic_cast<SegmentInstance*>(cellPartInstance), &child);
        buildQueue.insert({buildPriority, buildData});
    }
}

void LifeForm::init(){
    Individual::init();
    head = nullptr;
    cellParts.clear();
    cellPartInstances.clear();
    inputs.clear();
    outputs.clear();
    buildQueue.clear();
    energy = 0;
    growthEnergy = 0;
    currentGrowthEnergy = 0;
    growthPriority = 0;
    regenerationFraction = 0;
    childEnergy = 0;
    numChildren = 0;
}