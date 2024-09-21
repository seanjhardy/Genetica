#undef main

// main.cpp
#include <modules/physics/fluid.hpp>
#include "simulator/simulator.hpp"
#include "stdexcept"
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/graphics/componentManager.hpp>
#include <modules/utils/fastMath.hpp>
#include <simulator/planet.hpp>

int main() {
    try {
        print("Starting Genetica v0.1");

        // Initialising systems
        Styles::init();
        FastMath::init();
        SpriteManager::init();
        ShaderManager::init();
        FontManager::init();
        ComponentManager::init();
        CursorManager::init();
        Planet::init();

        // Set up simulation
        Simulator::get().setup();
        Simulator::get().run();

    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }

    Simulator::get().cleanup();
    return 0;
}