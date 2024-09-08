#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>
#include <simulator/entities/lifeform.hpp>

class CellLink {
    int connectionIdx;

    LifeForm* lifeForm;
    float growthProgress = 0.1;
    float lastGrowthProgress = 0.1;
    float finalLength;

    Cell* cell1;
    Cell* cell2;

    CellLink(LifeForm* lifeForm, int a, int b, float length) {
        connectionIdx = lifeForm->getEnv()->addConnection(a, b, length*growthProgress);
        this->lifeForm = lifeForm;
        finalLength = length;
    }

    void simulate(float dt, float massChange) {
        if (growthProgress == 1) return;

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
    }

    float getBuildCost() const {
        Point* p1 = lifeForm->getEnv()->getPoint(connection.a);
        Point* p2 = lifeForm->getEnv()->getPoint(connection.b);
        float buildCost = 1.0f;
                          //+ 0.2f * (bone ? 1.0f : 0.0f) * boneDensity
                          //+ 0.5f * (muscle ? 1.0f : 0.0f) * muscleStrength
                          //+ 0.01f * (nerve ? 1.0f : 0.0f)
                          //+ 0.01f * (fat ? 1.0f : 0.0f) * fatSize;
        buildCost *= finalLength * (p1->mass + p2->mass) / 2;
        return buildCost;
    };
};