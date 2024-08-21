#include <geneticAlgorithm/cellParts/segmentInstance.hpp>
#include <geneticAlgorithm/cellParts/segmentType.hpp>
#include "modules/utils/mathUtils.hpp"
#include "modules/utils/GUIUtils.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include "../../modules/physics/constraints.cu"
#include <simulator/simulator.hpp>
#include "cmath"

SegmentInstance::SegmentInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent)
    : CellPartInstance(lifeForm, type, parent) {

    float2 endPos;
    if (parent != nullptr) {
        realAngle = parent->realAngle + getAdjustedAngleFromBody() + getAdjustedAngleOnBody();
    } else {
        realAngle = Random::random() * M_PI * 2;
    }

    float length = dynamic_cast<SegmentType*>(schematic->type)->length
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;
    float startWidth = dynamic_cast<SegmentType*>(schematic->type)->startWidth
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;
    float endWidth = dynamic_cast<SegmentType*>(schematic->type)->startWidth
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;

    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    start->mass = startWidth;

    endPos = start->pos + vec(realAngle) * length;
    endPoint = lifeForm->getEnv()->addPoint(lifeForm->entityID, endPos.x, endPos.y, endWidth);
    constrainPosition(*lifeForm->getEnv()->getPoint(startPoint), lifeForm->getEnv()->getBounds());
    constrainPosition(*lifeForm->getEnv()->getPoint(endPoint), lifeForm->getEnv()->getBounds());

    if (parent != nullptr) {
        ParentChildLink* pcl = lifeForm->getEnv()->getParentChildLink(parentChildLink);
        pcl->endPoint = endPoint;
        pcl->targetAngle = getAdjustedAngleFromBody() + getAdjustedAngleOnBody();
        float stiffness = 0.1f + dynamic_cast<SegmentType*>(schematic->type)->boneDensity * 0.5f;
        stiffness = min(stiffness, 0.99f);
        pcl->stiffness = stiffness;
        lifeForm->getEnv()->getParentChildLinks().update(parentChildLink, *pcl);
    }
}

void SegmentInstance::simulate(float dt) {
    CellPartInstance::simulate(dt);

    for (auto& child : children ) {
        child->simulate(dt);
    }
    float cellLength = dynamic_cast<SegmentType*>(schematic->type)->length * growthFraction * lifeForm->size;
    float cellWidth = (dynamic_cast<SegmentType*>(schematic->type)->startWidth +
                       dynamic_cast<SegmentType*>(schematic->type)->endWidth) * 0.5f * growthFraction * lifeForm->size;
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);
    realAngle = start->angleTo(*end);

    if (growthFraction != 1 && Simulator::get().getStep() % 5 == 0) {
        float length = dynamic_cast<SegmentType*>(schematic->type)->length;
        constrainDistance(*start, *end, length * growthFraction * lifeForm->size);
    }

    if (lifeForm == nullptr) return;

    // TODO: Adjust this value
    lifeForm->energy -= 0.0001f; // Add a small constant to prevent part spam (penalises lots of points)
    lifeForm->energy -= LifeForm::ENERGY_DECREASE_RATE * cellLength * cellWidth * dt;
}

bool SegmentInstance::grow(float dt, float massChange) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);

    float startWidth = (dynamic_cast<SegmentType*>(schematic->type)->startWidth);
    float endWidth = (dynamic_cast<SegmentType*>(schematic->type)->endWidth);
    float length = dynamic_cast<SegmentType*>(schematic->type)->length;
    float avgWidth = (startWidth + endWidth) * 0.5f;

    // Don't ask me how I got this equation :skull:
    float deltaGrowth = -growthFraction + sqrt(growthFraction*growthFraction
                                               + (massChange * dt) / (avgWidth * length * lifeForm->size * lifeForm->size));

    // Ensure growthFraction + deltaGrowth does not go above 1
    double newGrowthFraction = min(growthFraction + deltaGrowth, 1.0f);
    deltaGrowth = newGrowthFraction - growthFraction;
    // Multiplied out all growthFraction terms
    // Using a minimum cost so lifeforms can't spam tiny parts without incurring a small activation cost
    float growthEnergyCost = schematic->type->getBuildCost() * LifeForm::BUILD_COST_SCALE * avgWidth * length * deltaGrowth;

    if (lifeForm->energy < growthEnergyCost) {
        return false;
    }

    lastGrowthFraction = growthFraction;
    growthFraction = newGrowthFraction;
    lifeForm->energy -= growthEnergyCost;
    // Calculate the growthFraction of each point based on gene, segment growthFraction %, and lifeform growthFraction
    start->mass = startWidth * growthFraction * lifeForm->size;
    end->mass = endWidth * growthFraction * lifeForm->size;

    // Continue building if not done
    if (growthFraction != 1) return false;

    float cellLength = dynamic_cast<SegmentType*>(schematic->type)->length * 1.0f * lifeForm->size;
    lifeForm->getEnv()->addConnection(startPoint, endPoint, cellLength);
    // Return true if fully built
    return true;
}

float2 SegmentInstance::getPointAtAngle(float angle) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);
    float length = (dynamic_cast<SegmentType*>(schematic->type)->length) * growthFraction * lifeForm->size;

    float2 point = getPointOnSegment(length, start->mass/2, end->mass/2, angle);

    return start->pos + rotate(point, realAngle);
}

/**
 * Activates the muscle output of the segment
 * @param dt - simulation deltaTime
 * @param signal - signal strength
 */
void SegmentInstance::activateOutput(float dt, float signal) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);

    float length = hypot(end->pos.y - start->pos.y,
                         end->pos.x - start->pos.x);

    float width = (start->mass + end->mass) * 0.5f;
    float forceWidthRatio = length / width - std::abs(end->mass - start->mass) * 0.2f;

    float muscleStrength = 0.02f
            * dynamic_cast<SegmentType*>(schematic->type)->muscleStrength
            * signal * (flipped ? -1.0f : 1.0f) * dt;

    float energyCost = 0.05f * abs(muscleStrength) * growthFraction;
    float oldAngle = realAngle + M_PI;
    float newAngle = oldAngle + muscleStrength;

    if(energyCost < lifeForm->energy && muscleStrength != lastMuscle){
        lifeForm->energy -= energyCost;

        end->setPos(start->pos + vec(newAngle)*length);

        //add force
        float magnitude = -10 * forceWidthRatio * abs(muscleStrength - lastMuscle);
        start->force += vec(newAngle) * magnitude;
    }

    lastMuscle = muscleStrength;
}

void SegmentInstance::render(VertexManager& vertexManager) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);

    for (auto& child : children) {
        child->render(vertexManager);
    }

    //Don't render this segment if it's out of bounds of the camera
    //if (!vertexManager.camera->isCircleVisible(start->pos, start->mass) &&
    //    !vertexManager.camera->isCircleVisible(end->pos, end->mass)) {
    //    return;
    //}

    float startRadius = start->mass / 2;
    float endRadius = end->mass / 2;
    float length = dynamic_cast<SegmentType*>(schematic->type)->length;

    // Automatically cull small segments
    if ((startRadius < 0.5f && endRadius < 0.5f) || length < 0.5f) return;

    sf::Color color = dynamic_cast<SegmentType*>(schematic->type)->color;
    float lineWidth = min((startRadius + endRadius) * 0.5f, length) * growthFraction * lifeForm->size * 0.5f;

    float boneDensity = dynamic_cast<SegmentType*>(schematic->type)->boneDensity
            * dynamic_cast<SegmentType*>(schematic->type)->bone;

    if (boneDensity > 0 and parent != nullptr) {
        float boneAngle = 40 * M_PI / 180;
        float2 line1Start = getPointAtAngle(-boneAngle);
        float2 line1End = parent->getPointAtAngle(getAdjustedAngleOnBody()  + boneAngle);

        float2 line2Start = getPointAtAngle(boneAngle);
        float2 line2End = parent->getPointAtAngle(getAdjustedAngleOnBody() - boneAngle);

        float lineThickness = 0.4f * lineWidth * boneDensity * growthFraction;
        vertexManager.addLine(line1Start, line1End, sf::Color::White, lineThickness);
        vertexManager.addLine(line2Start, line2End, sf::Color::White, lineThickness);
    }
    if (vertexManager.getSizeInView((startRadius + endRadius) / 2) > 2) {
        sf::Color darker = brightness(color, 0.5f);
        vertexManager.addSegment(start->pos, end->pos, start->mass + 2.0f, end->mass + 2.0f, realAngle, darker);
    }
    vertexManager.addSegment(start->pos, end->pos, start->mass, end->mass, realAngle, color);

    float muscleStrength = dynamic_cast<SegmentType*>(schematic->type)->muscleStrength
                        * dynamic_cast<SegmentType*>(schematic->type)->muscle;

    if (muscleStrength > 0 && false) {
        sf::Color muscleColour = sf::Color(255, 125, 125);
        float muscleWidth = lineWidth * muscleStrength;
        float muscleAngle = realAngle + M_PI/2;
        float2 musclePerp = vec(muscleAngle);
        float percent = startRadius / (length * 2);
        float2 diff = end->pos - start->pos;

        float2 pointA = start->pos + diff * percent;
        float2 pointB = start->pos + diff * (1 - percent);

        float2 line1Start = pointA + musclePerp * startRadius * 0.5f;
        float2 line1End = pointB + musclePerp * startRadius * 0.5f;

        float2 line2Start = pointA - musclePerp * endRadius * 0.5f;
        float2 line2End = pointB - musclePerp * endRadius * 0.5f;

        vertexManager.addLine(line1Start, line1End, muscleColour, muscleWidth);
        vertexManager.addLine(line2Start, line2End, muscleColour, muscleWidth);
    }
    vertexManager.addCircle(start->prevPos, 0.5, sf::Color::Blue);
    vertexManager.addCircle(end->prevPos, 0.5, sf::Color::Blue);
    vertexManager.addCircle(start->pos, 0.5, sf::Color::Red);
    vertexManager.addCircle(end->pos, 0.5, sf::Color::Red);
}

float SegmentInstance::getEnergyContent() {
    auto* type = dynamic_cast<SegmentType*>(schematic->type);
    float energyContent = type->getBuildCost()
      * LifeForm::BUILD_COST_SCALE
      * growthFraction * lifeForm->size
      * growthFraction * lifeForm->size;
    return energyContent;
}

void SegmentInstance::detach() {
    if (parent != nullptr) {
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), this),
                               parent->children.end());
    }
    parent = nullptr;
    detached = true;
}