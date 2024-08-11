#undef main

// main.cpp
#include "simulator/simulator.hpp"
#include <geneticAlgorithm/environments/fishTank/fishTank.hpp>
#include <geneticAlgorithm/environments/hyperLife/hyperLife.hpp>
#include "SFML/Graphics.hpp"
#include "stdexcept"
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/shaderManager.hpp>

int main() {
    try {
        print("Genetica v0.1");

        // Initialising systems
        SpriteManager::init();
        ShaderManager::init();
        FontManager::init();

        // Set up genetic algorithm with a given environment
        HyperLife env({0, 0, 1000, 1000});
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