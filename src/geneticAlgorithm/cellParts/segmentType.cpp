#include <geneticAlgorithm/cellParts/segmentType.hpp>
#include <geneticAlgorithm/cellParts/cellPartSchematic.hpp>

SegmentType::SegmentType(float partCode)
: CellPartType(partCode, Type::SEGMENT) {};

void SegmentType::addChild(CellPartSchematic child) {
    children.push_back(child);
}

float SegmentType::getBuildCost() const {
    float buildCost = 1.0f
                      + 0.2f * (bone ? 1.0f : 0.0f) * boneDensity
                      + 0.5f * (muscle ? 1.0f : 0.0f) * muscleStrength
                      + 0.01f * (nerve ? 1.0f : 0.0f)
                      + 0.01f * (fat ? 1.0f : 0.0f) * fatSize;
    buildCost *= length * (startWidth + endWidth) / 2;
    return buildCost;
};