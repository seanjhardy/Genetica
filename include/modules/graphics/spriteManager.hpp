#ifndef SPRITE_MANAGER
#define SPRITE_MANAGER

#include <unordered_map>
#include <string>
#include "SFML/Graphics.hpp"

class SpriteManager {
public:
    // Loads sprites from a configuration file
    static void init();

    // Gets a sprite by its key
    static sf::Sprite* get(const std::string& key);

private:
    static std::unordered_map<std::string, sf::Texture> textures;
    static std::unordered_map<std::string, sf::Sprite> sprites;
};

#endif