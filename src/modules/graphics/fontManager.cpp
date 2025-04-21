#include "modules/graphics/fontManager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

std::unordered_map<std::string, sf::Font> FontManager::fonts;


void FontManager::init() {
    // Hardcoded map of key to file path
    std::unordered_map<std::string, std::string> fontMappings = {
        // Icons
        {"russo", "./assets/fonts/russoone-regular.ttf"}
    };

    for (const auto& pair : fontMappings) {
        const std::string& key = pair.first;
        const std::string& filePath = pair.second;

        sf::Font font;
        if (!font.loadFromFile(filePath)) {
            std::cerr << "Failed to load texture from file: " << filePath << std::endl;
            continue; // Skip to next texture
        }
        fonts[key] = font;
    }
}

sf::Font* FontManager::get(const std::string& key) {
    auto it = fonts.find(key);
    if (it != fonts.end()) {
        return &it->second;
    }
    return nullptr;
}
