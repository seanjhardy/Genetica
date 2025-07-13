#ifndef SEGMENT
#define SEGMENT

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class Segment {
public:
    size_t lifeFormIdx;

    size_t startPointIdx;
    size_t startPointAttachedIdx;
    size_t endPointIdx;
    size_t endPointAttachedIdx;

    float stiffness = 1.0f;
    double startPointSize;
    double endPointSize;
    double length;
    double targetLength;
    float startPointTargetRadius = 0.0f;
    float endPointTargetRadius = 0.0f;

    // Blueprint pos
    size_t blueprintStartPointIdx;
    size_t blueprintStartPointAttachedIdx;
    size_t blueprintEndPointIdx;
    size_t blueprintEndPointAttachedIdx;

    // Cell properties
    StaticGPUVector<float> products;
    int generation = 0;
    size_t lastDivideTime = 0;
    float energy = 0.0f;
    float startDivisionRotation = 0.0f;
    float endDivisionRotation = 0.0f;
    int numDivisions = 0;
    bool frozen = false;
    bool dividing = false;
    bool dead = false;
    float hue = 200.0f, saturation = 0.0f, luminosity = 0.0f;
    float membraneThickness = 1.0f;

    Segment() = default;
    Segment(size_t startPointIdx, size_t endPointIdx, size_t startPointAttachedIdx, size_t endPointAttachedIdx,
             float startLength, float targetLength);

    __host__ void renderCellWall(VertexManager& vertexManager, vector<Point>& points);
    __host__ void renderBody(VertexManager& vertexManager, vector<Point>& points);
    __host__ void renderDetails(VertexManager& vertexManager, vector<Point>& points);

    __host__ __device__ void adjustSize(float lengthChange) {
        length += lengthChange;
    }


    __host__ __device__ void updateHue(PIGMENT pigment, float amount) {
        float target_hue = pigment == Red ? 0.0f :
                           pigment == Green ? 120.0f :
                           pigment == Blue ? 240.0f : 360.0f;
        float delta_hue = int(target_hue - hue) % 360;

        if (delta_hue > 180) {
            delta_hue -= 360;
        }

        saturation = clamp(0.0, saturation + amount * 0.01f, 1.0f);
        hue += amount * (delta_hue != 0 ? delta_hue / abs(delta_hue) : 0);
    }
    [[nodiscard]] sf::Color getColor() const {
        return HSVtoRGB(hue, saturation, 0.3 + luminosity);
    }
};

#endif