#ifndef NOISE
#define NOISE

#include <SFML/Graphics.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <utility>

class Noise {
public:
    std::vector<sf::Color> colours;
    int noiseOctaves = 6;
    float noiseFrequency = 1.0f;
    float noiseWarp = 0.0f;
    bool smoothNoise = false;
    bool animated = false;

    void applyToTexture(sf::RenderTexture* texture, sf::RenderStates states, float2 size, float seed);
};

class NoiseLayer {
public:
    bool update = true;
    Noise base;

    explicit NoiseLayer(Noise base) : base(std::move(base)) {}
    virtual void applyToTexture(sf::RenderTexture* texture, float2 size, float seed) = 0;
};

class Add : public NoiseLayer {
public:
    sf::RenderTexture cachedTexture;

    explicit Add(Noise base) : NoiseLayer(std::move(base)) {}

    void applyToTexture(sf::RenderTexture* texture, float2 size, float seed);
};

class Mask : public NoiseLayer {
public:
    Noise mask;
    sf::RenderTexture cachedTexture;

    Mask(Noise base, Noise mask) : NoiseLayer(std::move(base)), mask(std::move(mask)) {}

    void applyToTexture(sf::RenderTexture* texture, float2 size, float seed);
};


#endif