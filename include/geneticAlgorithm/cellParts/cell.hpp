#ifndef CELL
#define CELL

#include <geneticAlgorithm/systems/morphology/geneticUnit.hpp>
#include <geneticAlgorithm/systems/morphology/promoter.hpp>
#include <modules/utils/GUIUtils.hpp>
#include <modules/utils/random.hpp>

class LifeForm;

class Cell {
public:
    size_t idx;
    size_t lifeFormIdx;
    size_t pointIdx;

    int generation = 0;
    int lastDivideTime = 0;
    int divisionFrequency = 10 + Random::random(20);

    StaticGPUVector<float> products;

    float energyUse = 0.0f;
    float rotation = 0.0f;
    float divisionRotation = 0.0f;
    bool frozen = false;
    bool dividing = false;
    bool dead = false;

    float hue = 200.0f, saturation = 0.0f, luminosity = 0.0f;

    Cell() = default;
    Cell(int lifeFormIdx, const float2& pos, float radius = 0.1);

    __host__ __device__ void fuse(Cell* other);
    void render(VertexManager& vertexManager, vector<Point>& points) const;

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