// simulator.cpp
#include "simulator/simulator.hpp"
#include "simulator/screens/simulationScreen.hpp"
#include <modules/utils/print.hpp>
#include <sstream>
#include <iomanip>

// Instantiate simulator
Simulator::Simulator(Environment& env, int width, int height)
        : window(sf::VideoMode(width, height), env.getTitle()),
        state(State::Playing),
        uiManager(&window),
        camera(CameraController(env.getBounds(), &window)){

    print("Loading Environment: ", env.getTitle());

    vertexManager.setCamera(&camera);
}

void Simulator::setup() {
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
            uiManager.handleEvent(event);
        }
        uiManager.update(deltaTime, static_cast<sf::Vector2f>(sf::Mouse::getPosition(window)));

        camera.update(deltaTime);

        if (state == State::Playing) {
            realTime += deltaTime * speed;
            GeneticAlgorithm::get().simulate(deltaTime * speed);
        }

        std::clock_t now = std::clock();
        auto renderDelta = static_cast<double>(now - lastRenderTime);
        // Avoid re-rendering over 60fps
        if (renderDelta >= FRAME_INTERVAL) {
            lastRenderTime = now;

            if (state != State::Fast) {
                window.clear();

                window.setView(camera.getView());
                GeneticAlgorithm::get().render(vertexManager);
                vertexManager.draw(window);

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

void Simulator::speedUp() {
    speed *= 1.5;
}

void Simulator::slowDown() {
    speed /= 1.5;
}

void Simulator::reset() {
    GeneticAlgorithm::get().reset();
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