#ifndef SEGMENT_TYPE
#define SEGMENT_TYPE

#include "geneticAlgorithm/entities/lifeform.hpp"
#include "cellPartType.hpp"

class LifeForm;

/**
 *  A specific instance of a cellPart Type defining a segment with a given
 *  width, length, and properties such as the presence of muscles, bones, nerves, and fat, and their strength.
 */
class SegmentType : public CellPartType {
public:
    std::vector<CellPartSchematic> children;
    float startWidth{}, endWidth{}, length{};
    bool bone{}, muscle{}, nerve{}, fat{};
    float boneDensity{}, muscleStrength{}, fatSize{};
    sf::Color color;

    explicit SegmentType(float partCode);

    void addChild(CellPartSchematic child);

    [[nodiscard]] float getBuildCost() const override;

    CellPartType* upcast() {
        return static_cast<CellPartType*>(this);
    }
};

#endif