#ifndef SEGMENT_INSTANCE
#define SEGMENT_INSTANCE

#include "vector"
#include "cellPartInstance.hpp"
#include <modules/noise/random.hpp>
#include <modules/utils/floatOps.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/CellPartSchematic.hpp>

/**
 * A segment instance is a specific instantiation of a cell part, adding an endpoint to the base class,
 * as well as variables for its muscle position, and whether it is centered or detached.
 */
class SegmentInstance : public CellPartInstance {
public:
    std::vector<CellPartInstance*> children;
    size_t endPoint;
    bool centered = false;
    bool detached = false;
    bool dead = false;
    float lastMuscle = 0.0;

    SegmentInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent);

    void simulate(float dt) override;
    void render(VertexManager& window) override;
    bool grow(float dt, float massChange) override;

    float2 getPointAtAngle(float angle);
    void activateOutput(float dt, float signal);
    float getEnergyContent() override;
    void detach();

    CellPartInstance* upcast(){
        return static_cast<CellPartInstance*>(this);
    }
};
#endif