#ifndef FONT_MANAGER
#define FONT_MANAGER

#include <unordered_map>
#include <string>
#include "SFML/Graphics.hpp"

class FontManager {
public:
    // Loads font from a configuration file
    static void init();

    // Gets a font by its key
    static sf::Font* getFont(const std::string& key);

private:
    static std::unordered_map<std::string, sf::Font> fonts;
};

#endif