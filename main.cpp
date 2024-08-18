#undef main

// main.cpp
#include "simulator/simulator.hpp"
#include "geneticAlgorithm/environment.hpp"
#include "SFML/Graphics.hpp"
#include "stdexcept"
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/graphics/styleManager.hpp>
#include <modules/utils/fastMath.hpp>

int main() {
    try {
        print("Genetica v0.1");

        // Initialising systems
        Styles::init();
        FastMath::init();
        SpriteManager::init();
        ShaderManager::init();
        FontManager::init();

        // Set up genetic algorithm with a given environment
        Environment env(sf::FloatRect(0, 0, 1000, 1000));
        GeneticAlgorithm::get().setEnvironment(env);

        // Set up simulation
        Simulator simulator(env, 800, 600);
        simulator.setup();
        simulator.run();

    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    return 0;
}