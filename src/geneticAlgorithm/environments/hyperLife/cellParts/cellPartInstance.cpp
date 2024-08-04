#include "geneticAlgorithm/environments/hyperLife/cellParts/cellPartInstance.hpp"
#include "modules/utils/mathUtils.hpp"
#include "geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp"
#include "geneticAlgorithm/environments/hyperLife/cellParts/segmentInstance.hpp"
#include "modules/utils/floatOps.hpp"

float CellPartInstance::initialSize = 0.2;

CellPartInstance::CellPartInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent) :
lifeForm(lifeForm), parent(parent), cellData(type), flipped(type->flipped){
    if (parent == nullptr) {
        depth = 0;
    } else {
        depth = parent->depth + 1;
        flipped = parent->flipped;
    }
    updatePointOnParent(getAdjustedAngleOnBody());
    //lifeForm->addCellPartInstance(*this);
}

void CellPartInstance::simulate(float dt) {
    if(abs(realAngle - lastAngle) > 0.005 ||
       abs(startPoint->pos.x - startPoint->prevPos.x) > 0.005 ||
       abs(startPoint->pos.y - startPoint->prevPos.y) > 0.005){
        lastAngle = realAngle;
    }

    if (parent == nullptr) return;

    if(parent->upcast()->size != parent->upcast()->lastSize) {
        updatePointOnParent(getAdjustedAngleOnBody());
    }
}


void CellPartInstance::updatePointOnParent(float adjustedAngle) {
    if (parent == nullptr) return;

    CellPartType* type = parent->upcast()->cellData->type;

    float startWidth = dynamic_cast<SegmentType*>(type)->startWidth * parent->upcast()->size;
    float endWidth = dynamic_cast<SegmentType*>(type)->endWidth * parent->upcast()->size;
    float length = dynamic_cast<SegmentType*>(type)->length * parent->upcast()->size;

    pointOnParent = getPointOnSegment(length, startWidth, endWidth, adjustedAngle);
}

float2 CellPartInstance::getRotatedPoint() {
    //Update point position if parent changes
    if(abs(parent->upcast()->angle - parent->upcast()->lastAngle) > 0.005 ||
       diff(parent->upcast()->startPoint->pos, parent->upcast()->startPoint->prevPos) > 0.005 ||
       (rotatedPoint.x == 0 && rotatedPoint.y == 0) ||
       parent->upcast()->size != parent->upcast()->lastSize){

        rotatedPoint = rotate(pointOnParent, parent->upcast()->angle);
        rotatedPoint += parent->upcast()->startPoint->pos;
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