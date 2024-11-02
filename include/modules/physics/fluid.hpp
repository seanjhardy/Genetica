#ifndef FLUID_SIMULATOR
#define FLUID_SIMULATOR

#include <SFML/Graphics.hpp>
#include "modules/graphics/vertexManager.hpp"

struct Color3f
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    __host__ __device__ Color3f operator+ (Color3f other)
    {
        Color3f res;
        res.x= this->x + other.x;
        res.y = this->y + other.y;
        res.z = this->z + other.z;
        return res;
    }

    __host__ __device__ Color3f operator* (float d)
    {
        Color3f res;
        res.x = this->x * d;
        res.y = this->y * d;
        res.z = this->z * d;
        return res;
    }
};

struct Particle
{
    float2 u; // velocity
    Color3f color;
};

class FluidSimulator {
public:
    struct Config
    {
        float velocityDiffusion = 0.8f;
        float pressure = 1.5f;
        float vorticity = 50.0f;
        float colorDiffusion = 0.8f;
        float densityDiffusion = 0.001f;
        float forceScale = 5000.0f;
        float bloomIntense = 0.1f;
        int radius = 5;
        bool bloomEnabled = false;
    } config;

    struct SystemConfig
    {
        int velocityIterations = 20;
        int pressureIterations = 40;
        int xThreads = 32;
        int yThreads = 32;
    } sConfig;

    Particle* newField;
    Particle* oldField;
    uint8_t* colorField;
    float* pressureNew;
    float* pressureOld;
    float* vorticityField;
    float deltaTime = 0.0f;

    float scale = 40;
    float width, height;
    sf::Texture texture;
    sf::Sprite sprite;
    std::vector<uint8_t> pixelBuffer;

    FluidSimulator(float scale, size_t width, size_t height, Config config);

    void init();
    void reset();
    void update(float dt);

    void addForce(float2 position, float2 vector);

    void render(VertexManager &vertexManager, sf::FloatRect bounds){
        texture.update(pixelBuffer.data());
        sprite.setTexture(texture);
        sprite.setPosition(bounds.left, bounds.top);
        sprite.setScale({ 1.0f/scale, 1.0f/scale });
        vertexManager.addSprite(sprite);
    }
private:
    void computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt);
    void computeDiffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt);

};

#endif