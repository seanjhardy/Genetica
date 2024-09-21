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
            {"clone", "./assets/icons/creature.png"},
            {"cloneHighlighted", "./assets/icons/creature_highlighted.png"},
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
            {"mutate", "./assets/icons/mutation.png"},
            {"mutateHighlighted", "./assets/icons/mutation_highlighted.png"},
            {"delete", "./assets/icons/delete.png"},
            {"deleteHighlighted", "./assets/icons/delete_highlighted.png"},
            {"noEye", "./assets/icons/no_eye.png"},
            {"noEyeHighlighted", "./assets/icons/no_eye_highlighted.png"},
            {"save", "./assets/icons/save.png"},
            {"saveHighlighted", "./assets/icons/save_highlighted.png"},
            {"simulation", "./assets/icons/simulation.png"},
            {"simulationHighlighted", "./assets/icons/simulation_highlighted.png"},
            {"tools", "./assets/icons/tools.png"},
            {"settings", "./assets/icons/settings.png"},
            {"settingsHighlighted", "./assets/icons/settings_highlighted.png"},
            {"speedUp", "./assets/icons/speed_up.png"},
            {"speedUpHighlighted", "./assets/icons/speed_up_highlighted.png"},
            {"slowDown", "./assets/icons/slow_down.png"},
            {"slowDownHighlighted", "./assets/icons/slow_down_highlighted.png"},
            {"map", "./assets/icons/map.png"},
            {"mapHighlighted", "./assets/icons/map_highlighted.png"},
            {"quadtree", "./assets/icons/quadtree.png"},
            {"quadtreeHighlighted", "./assets/icons/quadtree_highlighted.png"},
            {"fluidEnabled", "./assets/icons/fluid.png"},
            {"fluidEnabledHighlighted", "./assets/icons/fluid_highlighted.png"},
            {"fluidDisabled", "./assets/icons/no_fluid.png"},
            {"fluidDisabledHighlighted", "./assets/icons/no_fluid_highlighted.png"},
            // Textures
            {"default", "./assets/textures/default.png"},
            {"dnaBanner", "./assets/textures/dna_banner.png"},
            {"dnaBannerMedium", "./assets/textures/dna_banner_medium.png"},
            {"dnaBannerSmall", "./assets/textures/dna_banner_small.png"},
            {"jawsClosed", "./assets/textures/jaws_closed.png"},
            {"jawsOpen", "./assets/textures/jaws_open.png"},
            {"cellTexture", "./assets/textures/cellTexture.png"},
            {"mars", "./assets/textures/mars.png"},
            {"cyrus", "./assets/textures/cyrus.png"},
            {"syth", "./assets/textures/syth.png"},
            {"xeria", "./assets/textures/xeria.png"},
            {"xeriab", "./assets/textures/xeriab.png"},
            {"ichigo", "./assets/textures/ichigo.png"},
            {"b12axo", "./assets/textures/b12axo.png"},
            {"aridium", "./assets/textures/aridium.png"},
            {"ocea", "./assets/textures/ocea.png"},
            {"gundam", "./assets/textures/gundam.png"},
            {"roche", "./assets/textures/roche.png"},
            {"xia", "./assets/textures/xia.png"},
            {"glau", "./assets/textures/glau.png"},
            {"eden", "./assets/textures/eden.png"},
            {"zolo", "./assets/textures/zolo.png"},
            {"riula", "./assets/textures/riula.png"},
            {"flax", "./assets/textures/flax.png"},
            {"romea", "./assets/textures/romea.png"},
            {"emulo", "./assets/textures/emulo.png"},
            {"azeuros", "./assets/textures/azeuros.png"},
            {"iridium", "./assets/textures/iridium.png"},
            {"nauca", "./assets/textures/nauca.png"},
            {"abogee", "./assets/textures/abogee.png"},
            {"delune", "./assets/textures/delune.png"},
            {"cerebrus", "./assets/textures/cerebrus.png"}
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
        texture.setRepeated(true);
        textures[key] = texture;
        sprites[key] = sf::Sprite(textures[key]);
    }
}

sf::Sprite* SpriteManager::get(const std::string& key) {
    if (sprites.contains(key)) {
        return &sprites[key];
    }
    return &sprites["default"];
}

sf::Texture* SpriteManager::getTexture(const std::string& key) {
    if (textures.contains(key)) {
        return &textures[key];
    }
    return &textures["default"];
}