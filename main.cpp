#undef main

// main.cpp
#include "simulator/simulator.hpp"
#include "stdexcept"
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/componentManager.hpp>
#include <modules/gpu/OpenCLManager.hpp>

int main() {
    try {
        consoleLog("Starting Genetica");
        // Initialise and load resources
        Styles::init();
        SpriteManager::init();
        ShaderManager::init();
        FontManager::init();
        ComponentManager::init();
        CursorManager::init();
        Planet::init();
        Simulator::get().init();
        OpenCLManager::init();

        // Set up and run simulation
        Simulator::get().run();

    }
    catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }

    Simulator::get().cleanup();
    OpenCLManager::cleanup();
    return 0;
}