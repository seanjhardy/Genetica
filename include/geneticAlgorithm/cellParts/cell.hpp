#ifndef CELL
#define CELL

#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>
#include <modules/utils/GUIUtils.hpp>
#include <modules/utils/random.hpp>
#include <modules/cuda/structures/staticGPUVector.hpp>

class LifeForm;

class Cell {
public:
    size_t idx;
    size_t lifeFormIdx;
    size_t pointIdx;

    int generation = 0;
    size_t lastDivideTime = 0;

    staticGPUVector<float> products;

    float energy = 0.0f;
    float divisionRotation = 0.0f;
    float targetRadius = 0.0f;
    int numDivisions = 0;
    bool frozen = false;
    bool dividing = false;
    bool dead = false;

    float hue = 200.0f, saturation = 0.0f, luminosity = 0.0f;
    float thickness = 1.0f;

    Cell() = default;
    Cell(int lifeFormIdx, Point& point);

    __host__ __device__ void fuse(Cell* other);

    void renderBody(VertexManager& vertexManager, vector<Point>& points) const;
    void renderCellWalls(VertexManager& vertexManager, vector<Point>& points) const;
    void renderDetails(VertexManager& vertexManager, vector<Point>& points) const;

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