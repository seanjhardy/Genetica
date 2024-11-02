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

    void applyToTexture(sf::RenderTexture* texture, sf::RenderStates states, sf::FloatRect bounds, float seed);
};

class NoiseLayer {
public:
    bool update = true;
    Noise base;

    explicit NoiseLayer(Noise base) : base(std::move(base)) {}
    virtual void applyToTexture(sf::RenderTexture* texture, sf::FloatRect bounds, float seed) = 0;
};

class Add : public NoiseLayer {
public:
    sf::RenderTexture cachedTexture;

    explicit Add(Noise base) : NoiseLayer(std::move(base)) {}

    void applyToTexture(sf::RenderTexture* texture, sf::FloatRect bounds, float seed) override;
};

class Mask : public NoiseLayer {
public:
    Noise mask;
    sf::RenderTexture cachedTexture;

    Mask(Noise base, Noise mask) : NoiseLayer(std::move(base)), mask(std::move(mask)) {}

    void applyToTexture(sf::RenderTexture* texture, sf::FloatRect bounds, float seed) override;
};


#endif