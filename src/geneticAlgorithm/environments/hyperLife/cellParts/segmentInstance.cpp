#include "geneticAlgorithm/environments/hyperLife/cellParts/segmentInstance.hpp"
#include "geneticAlgorithm/environments/hyperLife/cellParts/cellPartSchematic.hpp"
#include "geneticAlgorithm/environments/hyperLife/lifeform.hpp"
#include "geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp"
#include "modules/utils/mathUtils.hpp"
#include "../../../../modules/verlet/constraints.cu"
#include "cmath"

SegmentInstance::SegmentInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent)
    : CellPartInstance(lifeForm, type, parent) {
    if (parent != nullptr) {
        float2 point = getRotatedPoint();
        startPoint->pos = point;
        float length = dynamic_cast<SegmentType*>(cellData->type)->length * size;
        float parentAngle = parent->angle;
        realAngle = parentAngle + angle;
        endPoint->setPos(startPoint->pos + vec(realAngle) * length);
    } else {
        float length = dynamic_cast<SegmentType*>(cellData->type)->length * size;
        startPoint->setPos(lifeForm->pos);
        realAngle = getRandom() * M_PI * 2;
        endPoint->setPos(startPoint->pos + vec(realAngle) * length);
    }

    //TODO: Add segment to GA
    startPoint = lifeForm->getEnv()->addPoint(lifeForm->pos.x, lifeForm->pos.y, size);
    float2 endPos = startPoint->pos + vec(realAngle);
    endPoint = lifeForm->getEnv()->addPoint(endPos.x, endPos.y, size);
}

void SegmentInstance::simulate(float dt) {
    CellPartInstance::simulate(dt);

    calculatePosition(dt);

    for (auto& child : children ) {
        child.simulate(dt);
    }
    float cellLength = dynamic_cast<SegmentType*>(cellData->type)->length * size;
    float cellWidth = (dynamic_cast<SegmentType*>(cellData->type)->startWidth +
            dynamic_cast<SegmentType*>(cellData->type)->endWidth) * 0.5f * size;
    if (lifeForm != nullptr) {
        lifeForm->energy -= 0.001f * cellLength * cellWidth * dt;
    }
}

void SegmentInstance::calculatePosition(float dt) {
    //TODO: Add dynamic growth speed
    if (size < 1) {
        float newSize = 0.01f * dt;
        float growthEnergyCost = min(cellData->type->getBuildCost() * (newSize - size), 1.0f);
        if (lifeForm->energy > growthEnergyCost) {
            size = newSize;
            lifeForm->energy -= growthEnergyCost;
        }

        float length = dynamic_cast<SegmentType*>(cellData->type)->length * size;
        constrainDistance(*startPoint, *endPoint, length);
    } else if (!fullyGrown){
        fullyGrown = true;
        float cellLength = dynamic_cast<SegmentType*>(cellData->type)->length * size;
        float angle = cellData->angleFromBody;
        float stiffness = 0.1f + dynamic_cast<SegmentType*>(cellData->type)->boneDensity * 0.5f;
        stiffness = min(stiffness, 0.99f);
        lifeForm->getEnv()->addConnection(startPoint, endPoint, cellLength);
        lifeForm->getEnv()->addAngleConstraint(startPoint, endPoint,
                                               parent->startPoint, parent->endPoint,
                                               cellData->angleFromBody, stiffness);
    }

    //Simulate points
    //Add constraints
    //update real angle
}

float2 SegmentInstance::getPointAtAngle(float angle) {
    float r1 = (dynamic_cast<SegmentType*>(cellData->type)->startWidth) * size;
    float r2 = (dynamic_cast<SegmentType*>(cellData->type)->endWidth) * size;
    float length = (dynamic_cast<SegmentType*>(cellData->type)->length) * size;

    float2 point = getPointOnSegment(length, r1, r2, angle);

    return startPoint->pos + rotate(point, realAngle);
}

/**
 * Activates the muscle output of the segment
 * @param dt - simulation deltaTime
 * @param signal - signal strength
 */
void SegmentInstance::activateOutput(float dt, float signal) {
    float length = hypot(endPoint->pos.y - startPoint->pos.y,
                                   endPoint->pos.x - startPoint->pos.x);

    float startWidth = dynamic_cast<SegmentType*>(cellData->type)->startWidth;

    float muscleStrength = 0.02f
            * dynamic_cast<SegmentType*>(cellData->type)->muscleStrength
            * signal * (flipped ? -1.0f : 1.0f) * dt;

    float energyCost = 0.05f * abs(muscleStrength) * size;
    float oldAngle = realAngle + M_PI;
    float newAngle = oldAngle + muscleStrength;

    if(energyCost < lifeForm->energy && muscleStrength != lastMuscle){
        lifeForm->energy -= energyCost;

        endPoint->setPos(startPoint->pos + vec(newAngle)*length);

        //add force
        float magnitude = -10 * size * (length / startWidth) * abs(muscleStrength - lastMuscle);
        startPoint->force += vec(newAngle) * magnitude;
    }

    lastMuscle = muscleStrength;
}

void SegmentInstance::render(VertexManager& vertexManager) {
    for (auto& child : children) {
        child.render(vertexManager);
    }

    float startWidth = dynamic_cast<SegmentType*>(cellData->type)->startWidth;
    float endWidth = dynamic_cast<SegmentType*>(cellData->type)->endWidth;
    float length = dynamic_cast<SegmentType*>(cellData->type)->length;

    // Automatically cull small segments
    if ((startWidth < 1.0f && endWidth < 1.0f) || length < 1.0f) return;

    sf::Color color = dynamic_cast<SegmentType*>(cellData->type)->color;
    float lineWidth = min((startWidth + endWidth) * 0.5f, length) * 0.2f;

    float boneDensity = dynamic_cast<SegmentType*>(cellData->type)->boneDensity
            * dynamic_cast<SegmentType*>(cellData->type)->bone;

    if (boneDensity > 0 and parent != nullptr) {
        float2 line1Start = getPointAtAngle(-40);
        float2 line1End = parent->getPointAtAngle(getAdjustedAngleOnBody() + 40);

        float2 line2Start = getPointAtAngle(40);
        float2 line2End = parent->getPointAtAngle(getAdjustedAngleOnBody() - 40);

        float lineThickness = 0.4f * lineWidth *  boneDensity * size;
        vertexManager.addLine(line1Start, line1End, sf::Color::White, lineThickness);
        vertexManager.addLine(line2Start, line2End, sf::Color::White, lineThickness);
    }

    vertexManager.addCircle(startPoint->pos, startPoint->mass, color, 5);
    vertexManager.addCircle(endPoint->pos, endPoint->mass, color, 5);
    float2 d = vec(realAngle);
    vertexManager.addRectangle(startPoint->pos - d * startWidth * 0.5f,
                               startPoint->pos + d * startWidth * 0.5f,
                               endPoint->pos + d * endWidth * 0.5f,
                               endPoint->pos - d * endWidth * 0.5f, color);
    float muscleStrength = dynamic_cast<SegmentType*>(cellData->type)->muscleStrength
                        * dynamic_cast<SegmentType*>(cellData->type)->muscle;

    if (muscleStrength > 0) {
        sf::Color muscleColour = sf::Color(255, 125, 125);
        float muscleWidth = lineWidth * muscleStrength;
        float muscleAngle = realAngle + M_PI/2;
        float2 musclePerp = vec(muscleAngle);
        float percent = startWidth / (length * 2);
        float2 diff = endPoint->pos - startPoint->pos;

        float2 pointA = startPoint->pos + diff * percent;
        float2 pointB = startPoint->pos + diff * (1 - percent);

        float2 line1Start = pointA + musclePerp;
        float2 line1End = pointB + musclePerp;

        float2 line2Start = pointA - musclePerp;
        float2 line2End = pointB - musclePerp;

        vertexManager.addLine(line1Start, line1End, muscleColour, muscleWidth);
        vertexManager.addLine(line2Start, line2End, muscleColour, muscleWidth);

    }
}

float SegmentInstance::getEnergyContent() {
    auto* type = dynamic_cast<SegmentType*>(cellData->type);
    float length = type->length;
    float width = (type->startWidth + type->endWidth) * 0.5f;
    float muscle = type->muscle;

    float energyContent = width * length * (1 +
            0.1f * type->boneDensity * (float)type->bone +
            0.3f * type->muscleStrength * (float)type->muscle +
            2.0f * type->fatSize * (float)type->fat);

    /*for (auto& child : children) {
        if (dynamic_cast<ProteinInstance*>(child) != nullptr) {
            energyContent += dynamic_cast<ProteinInstance*>(child)->getEnergyContent();
        }
    }*/

    energyContent *= LifeForm::BUILD_COST_SCALE;
    return energyContent;
}

void SegmentInstance::detach() {
    if (parent != nullptr) {
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), this), parent->children.end());
    }
    parent = nullptr;
    detached = true;
}