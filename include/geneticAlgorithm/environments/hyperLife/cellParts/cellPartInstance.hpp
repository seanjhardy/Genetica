#ifndef CELL_PART_INSTANCE
#define CELL_PART_INSTANCE

#include <modules/verlet/point.hpp>

class LifeForm;
class SegmentInstance;
class CellPartSchematic;

/**
 * An abstract cell part instance, defining its growthFraction, angle, point and parent
 */
class CellPartInstance {
public:
    LifeForm* lifeForm;
    CellPartSchematic* cellData;
    SegmentInstance* parent;
    size_t startPoint{};
    static float INITIAL_GROWTH_FRACTION;
    float lastGrowthFraction = INITIAL_GROWTH_FRACTION;
    float growthFraction = INITIAL_GROWTH_FRACTION;
    float angle = 0, realAngle{};
    float lastAngle = 0;
    float2 pointOnParent{};
    float2 rotatedPoint{};
    float blueprintAngle = 0;
    float2 blueprintPos{};
    int depth = 0;
    bool flipped = false;

    CellPartInstance(LifeForm* lifeForm, CellPartSchematic* type, SegmentInstance* parent);

    virtual void simulate(float dt);
    virtual void render(VertexManager& vertexManager) = 0;

    virtual bool grow(float dt, float massChange) = 0;
    virtual float getEnergyContent() = 0;
    void updatePointOnParent(float adjustedAngle);

    float2 getRotatedPoint();
    [[nodiscard]] float getAdjustedAngleFromBody() const;
    [[nodiscard]] float getAdjustedAngleOnBody() const;

    friend class SegmentInstance;
};

#endif