#ifndef PLANET
#define PLANET

#include <utility>
#include <vector>
#include <modules/utils/random.hpp>
#include <SFML/Graphics.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <sstream>
#include <string>
#include <modules/noise/noise.hpp>

class Planet {
public:
    static std::map<std::string, Planet> planets;
    static std::vector<std::string> planetNames;
    static void init();
    static Planet* getRandom();
    static int current;

    // Info related to planet
    std::string name;
    std::string thumbnail;
    float temperature = 10.0f;

    // Information related to background map
    static constexpr float MAP_SCALE = 20.0f;
    sf::FloatRect mapBounds;
    sf::RenderTexture texture{};
    sf::Sprite mapSprite{};
    std::vector<NoiseLayer*> noise;
    double lastUpdate = 0;
    int mapSeed = Random::random(10000);

    Planet() = default;
    explicit Planet(std::string name);

    Planet(const Planet& other){
        name = other.name;
        thumbnail = other.thumbnail;
        temperature = other.temperature;
        mapSprite = other.mapSprite;
        mapSeed = other.mapSeed;
        noise = other.noise;
    };

    void update();
    void render (VertexManager& vertexManager);

    void updateMap();
    void setBounds(sf::FloatRect bounds);
    sf::FloatRect getBounds();
    void reset();
};


#endif