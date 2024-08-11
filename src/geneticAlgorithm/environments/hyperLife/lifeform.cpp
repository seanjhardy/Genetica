#include <utility>
#include "vector_types.h"
#include <geneticAlgorithm/individual.hpp>
#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include <geneticAlgorithm/environments/hyperLife/sequencer.hpp>
#include <geneticAlgorithm/genomeUtils.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp>
#include "unordered_map"

using namespace std;

int LifeForm::HEADER_SIZE = 35;
int LifeForm::CELL_DATA_SIZE = 12;
float LifeForm::BUILD_COST_SCALE = 0.0001f;
float LifeForm::BUILD_RATE = 50.0f;
float LifeForm::ENERGY_DECREASE_RATE = 0.0001;

LifeForm::LifeForm(HyperLife* hyperLife, float2 pos, const unordered_map<int, string>& genome)
    : Individual(hyperLife, genome), pos(pos) {
    sequence(this, genome);
}

void LifeForm::simulate(float dt) {
    //pos = head->startPoint.pos;
    grow(dt);
    //TODO: Implement neural network
}

void LifeForm::render(VertexManager& vertexManager) {
    getEnv()->getPoint(head->startPoint)->setPos(pos);
    for (auto& cellPartInstance : cellPartInstances) {
        cellPartInstance->render(vertexManager);
    }
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
    if ((GeneticAlgorithm::get().step + randomTime) % 5000 != 0) return;

    // Grow the builds in progress by their percentage of the total growth rate
    for (auto it = buildsInProgress.begin(); it != buildsInProgress.end(); ) {
        auto [priority, segment] = *it;
        float massChange = LifeForm::BUILD_RATE * growthRate * (float)priority / 100.0f;
        bool built = segment->grow(dt * 5000, massChange);
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
    CellPartType* buildType = buildSchematic->partType;
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

Individual& LifeForm::clone(bool mutate){
    // Copy genome
    unordered_map<int, string> copiedGenome = Individual::getGenome();
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

    //Add this item to buildsInProgress
    buildsInProgress.insert({cellPartInstance->cellData->buildPriority, cellPartInstance});

    if (cellPartInstance->cellData->partType->type != CellPartType::Type::SEGMENT)  return;

    // Automatically insert all child build orders by priority
    for (auto& child : dynamic_cast<SegmentType*>(cellPartInstance->cellData->partType)->children) {
        // Calculate build priority (shallower nodes win in tiebreakers)
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
    growthRate = 0;
    regenerationFraction = 0;
    childEnergy = 0;
    numChildren = 0;
}