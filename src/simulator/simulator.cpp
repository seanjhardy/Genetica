// simulator.cpp
#include "simulator/simulator.hpp"
#include "modules/utils/print.hpp"
#include "screens/simulationScreen.cpp"

// Instantiate simulator
Simulator::Simulator(Environment& env, int width, int height)
        : window(sf::VideoMode(width, height), env.getTitle()),
        state(State::Playing),
        camera(CameraController(env.getBounds(), window)){

    print("Loading Environment: ", env.getTitle());

    uiManager.addScreen("simulation", getSimulationScreen(this));
    uiManager.setCurrentScreen("simulation");
}

// Run simulation step
void Simulator::run() {
    sf::Clock clock;
    while (window.isOpen()) {
        sf::Time elapsed = clock.restart(); // Restart the clock and get elapsed time
        float deltaTime = elapsed.asSeconds(); // Convert elapsed time to seconds

        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            camera.updateEvent(event);
        }
        camera.update(deltaTime);

        if (state == State::Playing) {
            GeneticAlgorithm::get().simulate(deltaTime);
        }

        std::clock_t now = std::clock();
        auto renderDelta = static_cast<double>(now - lastRenderTime);
        if (renderDelta >= FRAME_INTERVAL) {
            lastRenderTime = now;

            if (state != State::Fast) {
                window.clear();
                window.setView(camera.getView());
                GeneticAlgorithm::get().render(vertexManager);
                vertexManager.draw(window);

                window.setView(window.getDefaultView());
                uiManager.draw(window);
            }
            window.display();
        }
    }
}

void Simulator::setState(State newState) {
    state = newState;
}

void Simulator::reset() {
    GeneticAlgorithm::get().reset();
}