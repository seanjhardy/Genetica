#include "modules/graphics/spriteManager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

std::unordered_map<std::string, sf::Texture> SpriteManager::textures;
std::unordered_map<std::string, sf::Sprite> SpriteManager::sprites;


void SpriteManager::init() {
    // Hardcoded map of key to file path
    std::unordered_map<std::string, std::string> spriteMappings = {
            // Icons
            {"creature", "./assets/icons/creature.png"},
            {"creatureHightlighted", "./assets/icons/creature_highlighted.png"},
            {"dna", "./assets/icons/dna.png"},
            {"dnaHighlighted", "./assets/icons/dna_highlighted.png"},
            {"play", "./assets/icons/play.png"},
            {"playHighlighted", "./assets/icons/play_highlighted.png"},
            {"pause", "./assets/icons/pause.png"},
            {"pauseHighlighted", "./assets/icons/pause_highlighted.png"},
            {"energy", "./assets/icons/energy.png"},
            {"energyHighlighted", "./assets/icons/energy_highlighted.png"},
            {"eye", "./assets/icons/eye.png"},
            {"eyeHighlighted", "./assets/icons/eye_highlighted.png"},
            {"folder", "./assets/icons/folder.png"},
            {"folderHighlighted", "./assets/icons/folder_highlighted.png"},
            {"mutation", "./assets/icons/mutation.png"},
            {"mutationHighlighted", "./assets/icons/mutation_highlighted.png"},
            {"noEye", "./assets/icons/no_eye.png"},
            {"noEyeHighlighted", "./assets/icons/no_eye_highlighted.png"},
            {"save", "./assets/icons/save.png"},
            {"saveHighlighted", "./assets/icons/save_highlighted.png"},
            {"simulation", "./assets/icons/simulation.png"},
            {"simulationHighlighted", "./assets/icons/simulation_highlighted.png"},
            {"tools", "./assets/icons/tools.png"},
            {"settings", "./assets/icons/settings.png"},
            {"settingsHighlighted", "./assets/icons/settings_highlighted.png"},
            {"fastForward", "./assets/icons/fast_forward.png"},
            {"fastForwardHighlighted", "./assets/icons/fast_forward_highlighted.png"},
            {"slowDown", "./assets/icons/slow_down.png"},
            {"slowDownHighlighted", "./assets/icons/slow_down_highlighted.png"},
            // Textures
            {"default", "./assets/textures/default.png"},
            {"dnaBanner", "./assets/textures/dna_banner.png"},
            {"dnaBannerMedium", "./assets/textures/dna_banner_medium.png"},
            {"dnaBannerSmall", "./assets/textures/dna_banner_small.png"},
            {"jawsClosed", "./assets/textures/jaws_closed.png"},
            {"jawsOpen", "./assets/textures/jaws_open.png"},
    };

    for (const auto& pair : spriteMappings) {
        const std::string& key = pair.first;
        const std::string& filePath = pair.second;

        sf::Texture texture;
        if (!texture.loadFromFile(filePath)) {
            std::cerr << "Failed to load texture from file: " << filePath << std::endl;
            continue; // Skip to next texture
        }

        texture.setSmooth(true);
        textures[key] = texture;
        sprites[key] = sf::Sprite(textures[key]);
    }
}

sf::Sprite* SpriteManager::getSprite(const std::string& key) {
    auto it = sprites.find(key);
    if (it != sprites.end()) {
        return &it->second;
    }
    return nullptr;
}