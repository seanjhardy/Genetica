#ifndef SEGMENT_TYPE
#define SEGMENT_TYPE

#include "../lifeform.hpp"
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

    SegmentType(LifeForm* lifeForm, int id)
    : CellPartType(lifeForm, id, Type::SEGMENT) {};

    void addChild(CellPartSchematic child) {
        children.push_back(child);
    }

    [[nodiscard]] float getBuildCost() const override {
        float buildCost = 1.0f
                + 0.2f * (bone ? 1.0f : 0.0f) * boneDensity
                + 0.5f * (muscle ? 1.0f : 0.0f) * muscleStrength
                + 0.01f * (nerve ? 1.0f : 0.0f)
                + 0.01f * (fat ? 1.0f : 0.0f) * fatSize;
        buildCost *= length * (startWidth + endWidth) / 2;
        return buildCost;
    };

    CellPartType* upcast() {
        return static_cast<CellPartType*>(this);
    }
};

#endif