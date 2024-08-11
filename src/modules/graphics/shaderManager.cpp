#include "modules/graphics/shaderManager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

std::unordered_map<std::string, sf::Texture> ShaderManager::textures;
std::unordered_map<std::string, sf::Shader> ShaderManager::shaders;


void ShaderManager::init() {
    // Hardcoded map of key to file path
    std::unordered_map<std::string, std::string> shaderMappings = {
            // Icons
            {"blur", "./assets/shaders/blur.frag"},
    };

    for (const auto& pair : shaderMappings) {
        const std::string& key = pair.first;
        const std::string& filePath = pair.second;
        if (!shaders[key].loadFromFile(filePath, sf::Shader::Fragment)) {
            std::cerr << "Failed to load texture from file: " << filePath << std::endl;
            continue; // Skip to next texture
        }
    }
}

sf::Shader* ShaderManager::getShader(const std::string &key) {
    auto it = shaders.find(key);
    if (it != shaders.end()) {
        return &it->second;
    }
    return nullptr;
}