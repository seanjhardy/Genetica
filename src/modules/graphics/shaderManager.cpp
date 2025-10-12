#include "modules/graphics/shaderManager.hpp"
#include <fstream>
#include <modules/utils/print.hpp>
#include <modules/graphics/spriteManager.hpp>

std::unordered_map<std::string, sf::Texture> ShaderManager::textures;
std::unordered_map<std::string, sf::Shader> ShaderManager::shaders;


void ShaderManager::init() {
    // Hardcoded map of key to file path
    loadShader("texture", "./assets/shaders/texture.vert", "./assets/shaders/texture.frag");
    loadShader("perlin", "", "./assets/shaders/perlin.frag");
    loadShader("genome", "", "./assets/shaders/genome.frag");
}

void ShaderManager::loadShader(const std::string& key, const std::string& vertexPath, const std::string& fragmentPath) {
    if (fragmentPath.empty()) {
        if (!shaders[key].loadFromFile(vertexPath, sf::Shader::Vertex)) {
            consoleLog("Failed to load vertex shader ", key, " from ", vertexPath);
            return;
        }
        shaders[key].setUniform("texture", SpriteManager::get("default"));
        consoleLog("Loaded vertex shader: ", key);
        return;
    }
    if (vertexPath.empty()) {
        if (!shaders[key].loadFromFile(fragmentPath, sf::Shader::Fragment)) {
            consoleLog("Failed to load fragment shader ", key, " from ", fragmentPath);
            return;
        }
        shaders[key].setUniform("texture", SpriteManager::get("default"));
        consoleLog("Loaded fragment shader: ", key);
        return;
    }
    if (!shaders[key].loadFromFile(vertexPath, fragmentPath)) {
        consoleLog("Failed to load shader ", key, " from ", vertexPath, " and ", fragmentPath);
        return;
    }
    consoleLog("Loaded shader: ", key);
}

sf::Shader* ShaderManager::get(const std::string& key) {
    if (shaders.contains(key)) {
        return &shaders[key];
    }
    return nullptr;
}
