#include <utility>
#include "vector_types.h"
#include "simulator/entities/lifeform.hpp"
#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include "geneticAlgorithm/geneticAlgorithm.hpp"
#include "geneticAlgorithm/cellParts/segmentType.hpp"
#include "geneticAlgorithm/cellParts/segmentInstance.hpp"
#include "unordered_map"
#include "modules/utils/print.hpp"
#include "simulator/simulator.hpp"

using namespace std;

int LifeForm::HEADER_SIZE = 50;
int LifeForm::GROWTH_INTERVAL = 5;
int LifeForm::CELL_DATA_SIZE = 28;
float LifeForm::BUILD_COST_SCALE = 0.0005f;
float LifeForm::BUILD_RATE = 50.0f;
float LifeForm::ENERGY_DECREASE_RATE = 0.0001;

LifeForm::LifeForm(Environment* environment, float2 pos, const map<int, string>& genome)
    : Entity(pos), env(environment), genome(genome){
    sequence(this, genome);
}

void LifeForm::simulate(float dt) {
    setPos(env->getPoint(head->startPoint)->pos);
    grow(dt);
    if (head != nullptr) {
        head->simulate(dt);
    }
}

void LifeForm::render(VertexManager& vertexManager) {
    pos = getEnv()->getPoint(head->startPoint)->pos;
    if (head != nullptr) {
        head->render(vertexManager);
    }
    vertexManager.addText(std::to_string(energy), pos, 24);
}

void LifeForm::grow(float dt) {
    if (currentGrowthEnergy >= growthEnergy) return;
    if (head == nullptr) return;
    // Exit if there's nothing to build
    if (buildQueue.empty() && buildsInProgress.empty()) return;

    //Calculate the sum of the build budget over builds in progress
    int currentBuildTotal = 0;
    for (auto& build : buildsInProgress) {
        currentBuildTotal += build.first;
    }

    // Only build on random intervals to stagger builds
    if ((Simulator::get().getStep() - lastGrow) < LifeForm::GROWTH_INTERVAL/dt) return;
    lastGrow = Simulator::get().getStep();

    // Grow the builds in progress by their percentage of the total growth rate
    for (auto it = buildsInProgress.begin(); it != buildsInProgress.end(); ) {
        auto [priority, segment] = *it;
        float massChange = LifeForm::BUILD_RATE * growthRate * (float)priority / 100.0f;
        bool built = segment->grow(dt * LifeForm::GROWTH_INTERVAL, massChange);
        // Remove the build from the buildsInProgress if it's done
        if (built) {
            it = buildsInProgress.erase(it);
        } else {
            ++it;
        }
    }

    // Get the next item to build
    if (buildQueue.empty()) return;
    auto nextToBuild = *buildQueue.rbegin();
    // Ensure the growth budget never goes above 100
    if (currentBuildTotal + nextToBuild.first >= 100) return;

    // If it's below the growth budget, create the segment and add it to the buildsInProgress
    SegmentInstance* buildFrom = nextToBuild.second.first;
    CellPartSchematic* buildSchematic = nextToBuild.second.second;
    CellPartType* buildType = buildSchematic->type;
    // Remove this element from the buildQueue (it's added to buildsInProgress below this function)
    buildQueue.erase(--buildQueue.end());

    if (buildType->type == CellPartType::Type::SEGMENT) {
        float energyCost = dynamic_cast<SegmentType*>(buildType)->getBuildCost()
          * LifeForm::BUILD_COST_SCALE * CellPartInstance::INITIAL_GROWTH_FRACTION * size;
        if (energy < energyCost) return;

        // Not multiplied by dt since this is a one-time instantaneous cost
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

LifeForm& LifeForm::clone(bool mutate){
    // Copy genome
    map<int, string> copiedGenome = getGenome();
    // Create new LifeForm
    auto* clone = new LifeForm(getEnv(), pos, copiedGenome);
    // Mutate lifeForm genome
    if (mutate) clone->mutate();

    // Setup lifeForm parameters
    float energyChange = energy * childEnergy;
    energy -= energyChange;
    clone->energy += energyChange;
    clone->currentGrowthEnergy = 0;

    // Assign to species and add to population
    //TODO: Simulator::get().getGA()..assignSpecies(clone);
    Simulator::get().getGA().addLifeForm(clone);
    return *clone;
}

LifeForm& LifeForm::combine(LifeForm* partner) {
    // Combine genomes
    map<int, string> combinedGenome = crossover(genome,
                                              partner->getGenome(),
                                              HEADER_SIZE, CELL_DATA_SIZE);
    // Create new LifeForm
    auto *child = new LifeForm(getEnv(), pos, combinedGenome);
    // Mutate lifeForm genome
    child->mutate();

    // Setup child lifeForm parameters - share energy from parents
    auto* partnerLF = dynamic_cast<LifeForm*>(partner);
    float energyChange = energy * childEnergy + partnerLF->energy * partnerLF->childEnergy;
    energy -= energyChange;
    child->energy += energyChange;
    child->currentGrowthEnergy = 0;

    // Assign to species and add to population
    //TODO: Simulator::get().getGA()..assignSpecies(clone);
    Simulator::get().getGA().addLifeForm(child);
    return *child;
}

void LifeForm::mutate() {
    map<int, string> mutatedGenome = Simulator::get().getGA().mutate(genome,
                                                            HEADER_SIZE, CELL_DATA_SIZE);
    setGenome(mutatedGenome);
    sequence(this, mutatedGenome);
}

void LifeForm::addCellPartInstance(CellPartInstance* cellPartInstance){
    cellPartInstances.push_back(cellPartInstance);

    //Add this item to buildsInProgress
    buildsInProgress.insert({cellPartInstance->schematic->buildPriority, cellPartInstance});

    if (cellPartInstance->schematic->type->type != CellPartType::Type::SEGMENT)  return;

    // Automatically insert all child build orders by priority
    for (auto& child : dynamic_cast<SegmentType*>(cellPartInstance->schematic->type)->children) {
        // Calculate build priority (shallower nodes win in tiebreakers)
        int buildPriority = child.buildPriority - cellPartInstance->depth;
        auto buildData = std::make_pair(dynamic_cast<SegmentInstance*>(cellPartInstance), &child);
        buildQueue.insert({buildPriority, buildData});
    }
}

void LifeForm::addInput(CellPartInstance* cellPartInstance) {
    inputs.push_back(cellPartInstance);
}

void LifeForm::addOutput(CellPartInstance* cellPartInstance) {
    outputs.push_back(cellPartInstance);
}

void LifeForm::init(){
    head = nullptr;
    cellParts.clear();
    cellPartInstances.clear();
    inputs.clear();
    outputs.clear();
    buildQueue.clear();
    energy = 0;
    growthEnergy = 0;
    currentGrowthEnergy = 0;
    growthRate = 0;
    regenerationFraction = 0;
    childEnergy = 0;
    numChildren = 0;
}

Environment* LifeForm::getEnv() {
    return env;
}

map<int, string> LifeForm::getGenome() {
    return genome;
};
void LifeForm::setGenome(const map<int, string>& genomeArr) {
    this->genome = genomeArr;
};

Species* LifeForm::getSpecies() {
    return species;
};