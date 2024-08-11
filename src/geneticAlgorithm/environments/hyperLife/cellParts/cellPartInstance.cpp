#include <modules/utils/mathUtils.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/cellPartInstance.hpp>

float CellPartInstance::INITIAL_GROWTH_FRACTION = 0.1;

CellPartInstance::CellPartInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent) :
lifeForm(lifeForm), parent(parent), cellData(type), flipped(type->flipped){
    if (parent == nullptr) {
        depth = 0;
    } else {
        depth = parent->depth + 1;
        flipped = parent->flipped;
    }
    growthFraction = INITIAL_GROWTH_FRACTION;
    updatePointOnParent(getAdjustedAngleOnBody());
}

void CellPartInstance::simulate(float dt) {
    Point* start = lifeForm->getEnv()->getPoint(startPoint);

    if(abs(realAngle - lastAngle) > 0.005 ||
       abs(start->pos.x - start->prevPos.x) > 0.005 ||
       abs(start->pos.y - start->prevPos.y) > 0.005){
        lastAngle = realAngle;
    }

    if (parent == nullptr) return;

    if(parent->growthFraction != parent->lastGrowthFraction) {
        updatePointOnParent(getAdjustedAngleOnBody());
    }
}


void CellPartInstance::updatePointOnParent(float adjustedAngle) {
    if (parent == nullptr) return;

    CellPartType* type = parent->upcast()->cellData->partType;

    float startWidth = dynamic_cast<SegmentType*>(type)->startWidth * parent->growthFraction * lifeForm->size;
    float endWidth = dynamic_cast<SegmentType*>(type)->endWidth * parent->growthFraction * lifeForm->size;
    float length = dynamic_cast<SegmentType*>(type)->length * parent->growthFraction * lifeForm->size;

    pointOnParent = getPointOnSegment(length, startWidth, endWidth, adjustedAngle);
}

float2 CellPartInstance::getRotatedPoint() {
    Point* parentStart = lifeForm->getEnv()->getPoint(parent->startPoint);

    //Update point position if parent changes
    if(abs(parent->angle - parent->lastAngle) > 0.005 ||
       diff(parentStart->pos, parentStart->prevPos) > 0.005 ||
       (rotatedPoint.x == 0 && rotatedPoint.y == 0) ||
       parent->growthFraction != parent->lastGrowthFraction){

        rotatedPoint = rotate(pointOnParent, parent->angle);
        rotatedPoint += parentStart->pos;
    }
    return rotatedPoint;
}

float CellPartInstance::getAdjustedAngleFromBody() const {
    return cellData->angleFromBody * ((flipped && !cellData->flipped) ? -1.0 : 1.0);
}

float CellPartInstance::getAdjustedAngleOnBody() const {
    return cellData->angleOnBody * ((flipped && !cellData->flipped) ? -1.0 : 1.0)
           + ((flipped && !cellData->flipped)? M_PI * 2 : 0);
}