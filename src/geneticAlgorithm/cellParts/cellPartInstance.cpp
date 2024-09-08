#include "modules/utils/mathUtils.hpp"
#include <geneticAlgorithm/cellParts/segmentType.hpp>

float CellPartInstance::INITIAL_GROWTH_FRACTION = 0.1;

CellPartInstance::CellPartInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent) :
  lifeForm(lifeForm), parent(parent), schematic(type), flipped(type->flipped){
    float2 startPos;

    if (parent == nullptr) {
        depth = 0;
        startPos = lifeForm->pos;
    } else {
        depth = parent->depth + 1;
        flipped = parent->flipped;
        CellPartType* type = parent->upcast()->schematic->type;

        float startWidth = dynamic_cast<SegmentType*>(type)->startWidth * lifeForm->size;
        float endWidth = dynamic_cast<SegmentType*>(type)->endWidth * lifeForm->size;
        float length = dynamic_cast<SegmentType*>(type)->length * lifeForm->size;

        pointOnParent = getPointOnSegment(length, startWidth, endWidth, getAdjustedAngleOnBody());
        scaledPointOnParent = pointOnParent * parent->growthFraction;
        startPos = rotate(pointOnParent, parent->realAngle) + lifeForm->getEnv()->getPoint(parent->startPoint)->pos;
    }

    angle = getAdjustedAngleFromBody() + getAdjustedAngleOnBody();
    startPoint = lifeForm->getEnv()->addPoint(lifeForm->entityID, startPos.x, startPos.y, 1);

    if (parent != nullptr) {
        parentChildLink = lifeForm->getEnv()->addParentChildLink(startPoint, -1,
                                                                 parent->startPoint, parent->endPoint,
                                                                 scaledPointOnParent, angle, -1);
    }
}

void CellPartInstance::simulate(float dt) {
    //Point* start = lifeForm->getEnv()->getPoint(startPoint);

    if (parent == nullptr) return;

    //Only update after a significant change in size
    if (parent->growthFraction - parent->lastGrowthFraction > 0.05) {
        // Update point on parent data if the parent is still growing
        scaledPointOnParent = pointOnParent * parent->growthFraction;
        ParentChildLink* pcl = lifeForm->getEnv()->getParentChildLink(parentChildLink);
        pcl->pointOnParent = rotate(pointOnParent, parent->realAngle);
        lifeForm->getEnv()->getParentChildLinks().update(parentChildLink, *pcl);
    }
}

float CellPartInstance::getAdjustedAngleFromBody() const {
    return schematic->angleFromBody * ((flipped && !schematic->flipped) ? -1.0 : 1.0);
}

float CellPartInstance::getAdjustedAngleOnBody() const {
    return schematic->angleOnBody * ((flipped && !schematic->flipped) ? -1.0 : 1.0)
           + ((flipped && !schematic->flipped) ? M_PI * 2 : 0);
}