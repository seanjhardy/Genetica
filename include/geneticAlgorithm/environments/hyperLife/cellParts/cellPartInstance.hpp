#ifndef CELL_PART_INSTANCE
#define CELL_PART_INSTANCE

#include <modules/verlet/point.hpp>

class LifeForm;
class SegmentInstance;
class CellPartSchematic;

/**
 * An abstract cell part instance, defining its size, angle, point and parent
 */
class CellPartInstance {
public:
    LifeForm* lifeForm;
    CellPartSchematic* cellData;
    SegmentInstance* parent;
    Point* startPoint{};
    static float initialSize;
    float lastSize = initialSize;
    float size = initialSize;
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
    virtual float getEnergyContent() = 0;
    void updatePointOnParent(float adjustedAngle);

    float2 getRotatedPoint();
    [[nodiscard]] float getAdjustedAngleFromBody() const;
    [[nodiscard]] float getAdjustedAngleOnBody() const;

    friend class SegmentInstance;
};

#endif