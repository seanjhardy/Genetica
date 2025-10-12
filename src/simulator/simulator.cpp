// simulator.cpp
#include "simulator/simulator.hpp"
#include "./screen/simulationScreen.hpp"
#include <sstream>
#include <iomanip>

Simulator& Simulator::get() {
    static Simulator simulator;
    return simulator;
};

// Instantiate simulator
Simulator::Simulator()
    : env(sf::FloatRect(0, 0, 500, 500)),
    window(sf::VideoMode(800, 600), "Genetica", sf::Style::Default, sf::ContextSettings(0, 0, 8)),
    state(State::Playing),
    uiManager(&window) {
    window.setMouseCursor(CursorManager::getDefault());
}

void Simulator::init() {
    // TOOO: add main menu in future
    uiManager.addScreen("simulation", getSimulationScreen(this));
    uiManager.setCurrentScreen("simulation");
    reset();
}

// Run simulation step
void Simulator::run() {
    sf::Clock clock;
    Viewport* simulation = dynamic_cast<Viewport*>(uiManager.getScreen("simulation")->getElement("simulation"));

    while (window.isOpen()) {
        sf::Time elapsed = clock.restart();
        float deltaTime = elapsed.asSeconds();

        // Manage all UI and simulation events
        handleEvents(simulation);

        // Handle continuous events (e.g. holding down a key/hovering over a UI element)
        auto mousePos = static_cast<sf::Vector2f>(sf::Mouse::getPosition(window));
        sf::Vector2f worldCoords = simulation->mapPixelToCoords(mousePos);
        bool UIHovered = uiManager.update(deltaTime, mousePos);
        env.update(worldCoords, simulation->getCamera()->getZoom(), UIHovered);

        // Update simulation state if playing
        if (state == State::Playing) {
            realTime += deltaTime * speed;
            env.simulate();
            step += 1;
        }

        std::clock_t now = std::clock();
        auto renderDelta = static_cast<double>(now - lastRenderTime);

        // Avoid re-rendering over 60fps
        if (renderDelta >= FRAME_INTERVAL) {
            lastRenderTime = now;
            window.clear();

            // Render simulation
            env.render(simulation->getVertexManager());

            // Render UI
            uiManager.draw(window);

            window.display();
        }
    }
}

// Handle all events in the simulation
void Simulator::handleEvents(Viewport* simulation) {
    sf::Event event{};
    auto mousePos = static_cast<sf::Vector2f>(sf::Mouse::getPosition(window));
    sf::Vector2f worldCoords = simulation->mapPixelToCoords(mousePos);

    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }
        if (event.type == sf::Event::Resized) {
            updateWindowView();
        }
        bool consumed = uiManager.handleEvent(event);
        if (consumed) continue;

        // If the UI hasn't consumed the event, handle it in the environment and
        // check if an entity was selected
        std::pair<bool, int> selectedEntity = env.handleEvent(event, worldCoords);
        if (selectedEntity.first && selectedEntity.second != selectedEntityId) {
            selectedEntityId = selectedEntity.second;
            setTab(selectedEntityId == -1 ? Tab::Simulation : Tab::LifeForm);
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

void Simulator::updateWindowView() {
    sf::Vector2f viewSize(window.getSize().x, window.getSize().y);

    // Set the view size and center
    windowView.setSize(viewSize);
    windowView.setCenter({ viewSize.x / 2, viewSize.y / 2 });

    window.setView(windowView);
}

void Simulator::setTab(Tab tab) {
    string lifeFormTab = std::string(tab == Tab::LifeForm ? "visible: true" : "visible: false");
    uiManager.getScreen("simulation")->getElement("genomeTab")->overrideProperty("style", lifeFormTab);
    uiManager.getScreen("simulation")->getElement("grnTab")->overrideProperty("style", lifeFormTab);
    uiManager.getScreen("simulation")->resize(window.getSize());
}

Environment& Simulator::getEnv() {
    return env;
}

void Simulator::speedUp() {
    speed *= 1.5;
}

void Simulator::slowDown() {
    speed /= 1.5;
}

void Simulator::reset() {
    selectedEntityId = -1;
    setTab(Tab::Simulation);
    env.reset();
}

std::string Simulator::getTimeString() const {
    int time = (int)realTime;
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

size_t Simulator::getStep() const {
    return step;
}

float Simulator::getRealTime() const {
    return (float)realTime;
}

size_t Simulator::getSelectedEntityId() const {
    return selectedEntityId;
}

void Simulator::cleanup() {
    env.cleanup();
}
