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
        shaders[key].loadFromFile(vertexPath, sf::Shader::Vertex);
        shaders[key].setUniform("texture", SpriteManager::get("default"));
        return;
    }
    if (vertexPath.empty()) {
        shaders[key].loadFromFile(fragmentPath, sf::Shader::Fragment);
        shaders[key].setUniform("texture", SpriteManager::get("default"));
        return;
    }
    if (!shaders[key].loadFromFile(vertexPath, fragmentPath)) {
        print("Failed to load shader ", key, " from ", vertexPath, " and ", fragmentPath);
        return;
    }
}

sf::Shader* ShaderManager::get(const std::string& key) {
    if (shaders.contains(key)) {
        return &shaders[key];
    }
    return nullptr;
}
