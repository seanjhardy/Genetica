#ifndef SHADER_MANAGER
#define SHADER_MANAGER

#include <unordered_map>
#include <string>
#include <SFML/Graphics.hpp>

class ShaderManager {
public:
    // Loads shaders from a configuration file
    static void init();

    // Gets a shader by its key
    static sf::Shader* get(const std::string& key);

private:
    static std::unordered_map<std::string, sf::Texture> textures;
    static std::unordered_map<std::string, sf::Shader> shaders;
};

#endif