#undef main

// main.cpp
#include "simulator/simulator.hpp"
#include "stdexcept"
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/graphics/componentManager.hpp>
#include <modules/graphics/cursorManager.hpp>
#include <modules/utils/fastMath.hpp>

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

        // Set up simulation
        Simulator::get().setup();
        Simulator::get().run();

    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    return 0;
}