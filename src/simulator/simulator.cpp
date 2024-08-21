// simulator.cpp
#include "simulator/simulator.hpp"
#include "simulator/screens/simulationScreen.hpp"
#include <modules/utils/print.hpp>
#include <sstream>
#include <iomanip>

Simulator& Simulator::get(){
    static Simulator simulator;
    return simulator;
};

// Instantiate simulator
Simulator::Simulator()
      : env(sf::FloatRect(0, 0, 1000, 1000)),
        window(sf::VideoMode(600, 800), "Genetica"),
        state(State::Playing),
        uiManager(&window),
        camera(CameraController(env.getBounds(), &window)){

    vertexManager.setCamera(&camera);
}

void Simulator::setup() {
    uiManager.addScreen("simulation", getSimulationScreen(this));
    uiManager.setCurrentScreen("simulation");
    Simulator::reset();
}

// Run simulation step
void Simulator::run() {
    sf::Clock clock;
    while (window.isOpen()) {
        sf::Time elapsed = clock.restart(); // Restart the clock and get elapsed time
        float deltaTime = elapsed.asSeconds(); // Convert elapsed time to seconds

        sf::Event event{};
        sf::Vector2f mousePos = static_cast<sf::Vector2f>(sf::Mouse::getPosition(window));
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            // Handle discrete events (e.g. key press, mouse click)
            camera.updateEvent(event);
            uiManager.handleEvent(event);
            env.handleEvent(event, camera.getCoords(mousePos));
        }
        // Handle continuous events (e.g. holding down a key/hovering over a UI element)
        uiManager.update(deltaTime, mousePos);
        env.update(camera.getCoords(mousePos));
        camera.update(deltaTime);

        // Update simulation state
        if (state == State::Playing) {
            realTime += deltaTime * speed;
            env.simulate(speed);
            geneticAlgorithm.simulate(speed);
            step += 1;
        }

        std::clock_t now = std::clock();
        auto renderDelta = static_cast<double>(now - lastRenderTime);
        // Avoid re-rendering over 60fps
        if (renderDelta >= FRAME_INTERVAL) {
            lastRenderTime = now;

            if (state != State::Fast) {
                window.clear();

                // Render simulation
                window.setView(camera.getView());
                env.render(vertexManager);
                geneticAlgorithm.render(vertexManager);
                vertexManager.draw(window);

                // Render UI
                window.setView(camera.getWindowView());
                uiManager.draw(window);
            }
            window.display();
        }
    }
}

void Simulator::setState(State newState) {
    state = newState;
}

Simulator::State Simulator::getState() {
    return state;
}

sf::RenderWindow& Simulator::getWindow() {
    return window;
}

Environment& Simulator::getEnv() {
    return env;
}

GeneticAlgorithm& Simulator::getGA() {
    return geneticAlgorithm;
}

void Simulator::speedUp() {
    speed *= 1.5;
}

void Simulator::slowDown() {
    speed /= 1.5;
}

void Simulator::reset() {
    env.reset();
    geneticAlgorithm.reset();
}

std::string Simulator::getTimeString() const {
    float time = realTime;
    int hours = time / 3600;
    time -= hours * 3600;
    int minutes = time / 60;
    time -= minutes * 60;
    int seconds = time;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << seconds;

    return oss.str();
}

float Simulator::getSpeed() const {
    return speed;
}

int Simulator::getStep() const {
    return step;
}