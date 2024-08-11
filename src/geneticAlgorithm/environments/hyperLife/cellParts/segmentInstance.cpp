#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentInstance.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/cellPartSchematic.hpp>
#include <geneticAlgorithm/environments/hyperLife/lifeform.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp>
#include <modules/utils/mathUtils.hpp>
#include "../../../../modules/verlet/constraints.cu"
#include "cmath"

SegmentInstance::SegmentInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent)
    : CellPartInstance(lifeForm, type, parent) {

    float2 startPos;
    float2 endPos;
    if (parent != nullptr) {
        float parentAngle = parent->angle;
        realAngle = parentAngle + angle;
        startPos = getRotatedPoint();
    } else {
        realAngle = Random::random() * M_PI * 2;
        startPos = lifeForm->pos;
    }

    float length = dynamic_cast<SegmentType*>(cellData->partType)->length
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;
    float startWidth = dynamic_cast<SegmentType*>(cellData->partType)->startWidth
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;
    float endWidth = dynamic_cast<SegmentType*>(cellData->partType)->startWidth
      * CellPartInstance::INITIAL_GROWTH_FRACTION * lifeForm->size;

    endPos = startPos + vec(realAngle) * length;
    startPoint = lifeForm->getEnv()->addPoint(startPos.x, startPos.y, startWidth);
    endPoint = lifeForm->getEnv()->addPoint(endPos.x, endPos.y, endWidth);
}

void SegmentInstance::simulate(float dt) {
    CellPartInstance::simulate(dt);

    for (auto& child : children ) {
        child->simulate(dt);
    }
    float cellLength = dynamic_cast<SegmentType*>(cellData->partType)->length * growthFraction * lifeForm->size;
    float cellWidth = (dynamic_cast<SegmentType*>(cellData->partType)->startWidth +
                       dynamic_cast<SegmentType*>(cellData->partType)->endWidth) * 0.5f * growthFraction * lifeForm->size;

    if (lifeForm == nullptr) return;

    // TODO: Adjust this value
    lifeForm->energy -= 0.0001f; // Add a small constant to prevent part spam (penalises lots of points)
    lifeForm->energy -= LifeForm::ENERGY_DECREASE_RATE * cellLength * cellWidth * dt;
}

bool SegmentInstance::grow(float dt, float massChange) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    Point* end = lifeForm->getEnv()->getPoint(endPoint);

    float startWidth = (dynamic_cast<SegmentType*>(cellData->partType)->startWidth);
    float endWidth = (dynamic_cast<SegmentType*>(cellData->partType)->endWidth);
    float length = dynamic_cast<SegmentType*>(cellData->partType)->length;
    float avgWidth = (startWidth + endWidth) * 0.5f;

    // Don't ask me how I got this equation :skull:
    float deltaGrowth = -growthFraction + sqrt(pow(growthFraction, 2)
                                               + (massChange) / (avgWidth * length * pow(lifeForm->size, 2)));

    // Ensure growthFraction + deltaGrowth does not go above 1
    double newGrowthFraction = min(growthFraction + deltaGrowth * dt, 1.0f);
    deltaGrowth = newGrowthFraction - growthFraction;
    // Multiplied out all growthFraction terms
    // Using a minimum cost so lifeforms can't spam tiny parts without incurring a small activation cost
    float growthEnergyCost = cellData->partType->getBuildCost() * LifeForm::BUILD_COST_SCALE * avgWidth * length * deltaGrowth;

    if (lifeForm->energy > growthEnergyCost) {
        lastGrowthFraction = growthFraction;
        growthFraction = newGrowthFraction;
        lifeForm->energy -= growthEnergyCost;
        // Calculate the growthFraction of each point based on gene, segment growthFraction %, and lifeform growthFraction
        start->mass = startWidth * growthFraction * lifeForm->size;
        end->mass = endWidth * growthFraction * lifeForm->size;
    }

    constrainDistance(*start, *end, length * growthFraction * lifeForm->size);

    // Continue building if not done
    if (growthFraction != 1) return false;

    float cellLength = dynamic_cast<SegmentType*>(cellData->partType)->length * growthFraction;
    float stiffness = 0.1f + dynamic_cast<SegmentType*>(cellData->partType)->boneDensity * 0.5f;
    start->mass = startWidth * growthFraction * lifeForm->size;
    end->mass = endWidth * growthFraction * lifeForm->size;
    stiffness = min(stiffness, 0.99f);
    lifeForm->getEnv()->addConnection(startPoint, endPoint, cellLength);
    if (parent != nullptr) {
        lifeForm->getEnv()->addAngleConstraint(startPoint, endPoint,
                                               parent->startPoint, parent->endPoint,
                                               cellData->angleFromBody, stiffness);
    }
    // Return true if fully built
    return true;
}

float2 SegmentInstance::getPointAtAngle(float angle) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);
    float length = (dynamic_cast<SegmentType*>(cellData->partType)->length) * growthFraction * lifeForm->size;

    float2 point = getPointOnSegment(length, start->mass, start->mass, angle);

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
            * dynamic_cast<SegmentType*>(cellData->partType)->muscleStrength
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

    float startRadius = start->mass / 2;
    float endRadius = end->mass / 2;
    float length = dynamic_cast<SegmentType*>(cellData->partType)->length;

    // Automatically cull small segments
    //if ((startRadius < 0.5f && endRadius < 0.5f) || length < 0.5f) return;

    sf::Color color = dynamic_cast<SegmentType*>(cellData->partType)->color;
    float lineWidth = min((startRadius + endRadius) * 0.5f, length) * growthFraction * lifeForm->size * 0.5f;

    float boneDensity = dynamic_cast<SegmentType*>(cellData->partType)->boneDensity
            * dynamic_cast<SegmentType*>(cellData->partType)->bone;

    if (boneDensity > 0 and parent != nullptr) {
        float2 line1Start = getPointAtAngle(-40);
        float2 line1End = parent->getPointAtAngle(getAdjustedAngleOnBody() + 40);

        float2 line2Start = getPointAtAngle(40);
        float2 line2End = parent->getPointAtAngle(getAdjustedAngleOnBody() - 40);

        float lineThickness = 0.4f * lineWidth * boneDensity * growthFraction;
        vertexManager.addLine(line1Start, line1End, sf::Color::White, lineThickness);
        vertexManager.addLine(line2Start, line2End, sf::Color::White, lineThickness);
    }

    // TODO: Change number of points based on LOD
    vertexManager.addCircle(start->pos, startRadius, color, 15);
    vertexManager.addCircle(end->pos, endRadius, color, 15);
    float2 d = vec(realAngle + M_PI/2);
    vertexManager.addRectangle(start->pos - d * startRadius,
                               start->pos + d * startRadius,
                               end->pos + d * endRadius,
                               end->pos - d * endRadius, color);

    float muscleStrength = dynamic_cast<SegmentType*>(cellData->partType)->muscleStrength
                        * dynamic_cast<SegmentType*>(cellData->partType)->muscle;

    if (muscleStrength > 0) {
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
}

float SegmentInstance::getEnergyContent() {
    auto* type = dynamic_cast<SegmentType*>(cellData->partType);
    float energyContent = type->getBuildCost()
      * LifeForm::BUILD_COST_SCALE
      * pow(growthFraction * lifeForm->size, 2);
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