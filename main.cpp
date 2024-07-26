#undef main

// main.cpp
#include "src/simulator/simulator.hpp"
#include "src/environments/fishTank/fishTank.hpp"
#include <SFML/Graphics.hpp>
#include <stdexcept>

int main() {
    try {
        std::cout << "Genetica v0.1" << std::endl;
        sf::FloatRect bounds(0, 0, 800, 600);
        FishTank env(bounds);
        Simulator simulator(env);
        simulator.run();
    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    return 0;
}