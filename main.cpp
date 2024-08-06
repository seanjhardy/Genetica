#undef main

// main.cpp
#include "simulator/simulator.hpp"
#include <geneticAlgorithm/environments/fishTank/fishTank.hpp>
#include <geneticAlgorithm/environments/hyperLife/hyperLife.hpp>
#include "SFML/Graphics.hpp"
#include "stdexcept"

int main() {
    try {
        print("Genetica v0.1");

        // Set up genetic algorithm with a given environment
        HyperLife env({0, 0, 2000, 2000});
        GeneticAlgorithm::get().setEnvironment(env);

        // Set up simulation
        Simulator simulator(env, 800, 600);
        simulator.run();

    } catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    return 0;
}