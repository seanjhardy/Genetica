#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include <simulator/entities/lifeform.hpp>

CellLink::CellLink(LifeForm* lifeForm, Cell* a, Cell* b, float startLength)
: lifeForm(lifeForm), cell1(a), cell2(b) {
    connectionIdx = lifeForm->getEnv()->addConnection(a->pointIdx, b->pointIdx, startLength);
}

void CellLink::adjustSize(float distance) {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    connection->length += distance;
    if (connection->length <= 0) {
        cell1->fuse(cell1);
    } else {
        lifeForm->getEnv()->updateConnection(connectionIdx, *connection);
    }
}

void CellLink::moveCell1(Cell* newCell) {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    connection->p1 = newCell->pointIdx;
    lifeForm->getEnv()->updateConnection(connectionIdx, *connection);
    cell1 = newCell;
}

void CellLink::moveCell2(Cell* newCell) {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    connection->p2 = newCell->pointIdx;
    lifeForm->getEnv()->updateConnection(connectionIdx, *connection);
    cell2 = newCell;
}

[[nodiscard]]

float CellLink::getBuildCost() const {
    Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
    Point* p1 = lifeForm->getEnv()->getPoint(cell1->pointIdx);
    Point* p2 = lifeForm->getEnv()->getPoint(cell2->pointIdx);
    float buildCost = 1.0f;
    //+ 0.2f * (bone ? 1.0f : 0.0f) * boneDensity
    //+ 0.5f * (muscle ? 1.0f : 0.0f) * muscleStrength
    //+ 0.01f * (nerve ? 1.0f : 0.0f)
    //+ 0.01f * (fat ? 1.0f : 0.0f) * fatSize;
    buildCost *= connection->length * (p1->mass + p2->mass) / 2;
    return buildCost;
};

/*void simulate(float dt, float massChange) {
        Connection* connection = lifeForm->getEnv()->getConnection(connectionIdx);
        Point* p1 = lifeForm->getEnv()->getPoint(connection->p1);
        Point* p2 = lifeForm->getEnv()->getPoint(connection->p2);

        float startWidth = p1->mass;
        float endWidth = p2->mass;
        float avgWidth = (startWidth + endWidth) * 0.5f;

        // Don't ask me how I got this equation :skull:
        float deltaGrowth = -growthProgress + sqrt(growthProgress*growthProgress
                                                   + (massChange * dt) / (avgWidth * finalLength));

        // Ensure growthFraction + deltaGrowth does not go above 1
        double newGrowthFraction = min(growthProgress + deltaGrowth, 1.0f);
        deltaGrowth = newGrowthFraction - growthProgress;

        float growthEnergyCost = getBuildCost() * LifeForm::BUILD_COST_SCALE * deltaGrowth;

        if (lifeForm->energy < growthEnergyCost) return;

        lastGrowthProgress = growthProgress;
        growthProgress = newGrowthFraction;
        lifeForm->energy -= growthEnergyCost;

        // Calculate the growthFraction of each point based on gene, segment growthFraction %, and lifeform growthFraction
        p1->mass = startWidth * growthProgress;
        p2->mass = endWidth * growthProgress;


        if (lifeForm != nullptr || false) {
            float distToParentStart = sqrt(sum(pointOnParent * pointOnParent));
            lifeForm->getEnv()->addConnection(parent->startPoint, startPoint, distToParentStart);

            float parentLength = dynamic_cast<SegmentType *>(parent->schematic->type)->length * lifeForm->size;
            float2 d = pointOnParent - make_float2(parentLength, 0);
            float distToParentEnd = sqrt(sum(d * d));
            lifeForm->getEnv()->addConnection(parent->endPoint, startPoint, distToParentEnd);
        }

        // Return true if fully built
        return true;
    }*/
