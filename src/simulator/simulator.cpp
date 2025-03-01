// simulator.cpp
#include "simulator/simulator.hpp"
#include "./screen/simulationScreen.hpp"
#include <sstream>
#include <iomanip>
#include <modules/cuda/GPUDirectRenderer.hpp>

Simulator& Simulator::get(){
    static Simulator simulator;
    return simulator;
};

// Instantiate simulator
Simulator::Simulator()
      : env(sf::FloatRect(0, 0, 500, 500)),
        window(sf::VideoMode(800, 600), "Genetica"),
        state(State::Playing),
        uiManager(&window){
    window.setMouseCursor(CursorManager::getDefault());
}

void Simulator::setup() {
    uiManager.addScreen("simulation", getSimulationScreen(this));
    uiManager.setCurrentScreen("simulation");
    reset();
}

// Run simulation step
void Simulator::run() {
    sf::Clock clock;

    Viewport* simulation = dynamic_cast<Viewport*>(uiManager.getScreen("simulation")->getElement("simulation"));

    while (window.isOpen()) {
        /*
        *Point p(123, 88.8115f, 53.1058f, 5.0f); // Initialize the host Point object

    // Allocate memory on the device
    Point* d_data_test = nullptr;
    cudaMalloc(&d_data_test, sizeof(Point));

    // Copy the host object to the device
    cudaMemcpy(d_data_test, &p, sizeof(Point), cudaMemcpyHostToDevice);

    // Allocate memory on the host for receiving data from the device
    Point* h_data_Test = new Point; // Dynamically allocate memory for the host Point

    // Copy the data back from the device to the host
    cudaMemcpy(h_data_Test, d_data_test, sizeof(Point), cudaMemcpyDeviceToHost);

    // Print the copied values to verify
    std::cout << "Point1(" << p.pos.x << ", " << p.pos.y << ", " << p.radius << ")"  << std::endl;
    std::cout << "Point2(" << h_data_Test->pos.x << ", " << h_data_Test->pos.y << ", " << h_data_Test->radius << ")" << std::endl;

    // Free device memory
    cudaFree(d_data_test);

    // Free host memory
    delete h_data_Test;
         */
        sf::Time elapsed = clock.restart(); // Restart the clock and get elapsed time
        float deltaTime = elapsed.asSeconds(); // Convert elapsed time to seconds

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
            // Handle discrete events (e.g. key press, mouse click)
            bool consumed = uiManager.handleEvent(event);
            if (consumed) continue;

            std::pair<bool, int> selectedEntity = env.handleEvent(event, worldCoords);

            if (selectedEntity.first && selectedEntity.second != selectedEntityId) {
                selectedEntityId = selectedEntity.second;
                setTab(selectedEntityId == -1 ? Tab::Simulation : Tab::LifeForm);
            }
        }
        // Handle continuous events (e.g. holding down a key/hovering over a UI element)
        bool UIHovered = uiManager.update(deltaTime, mousePos);
        env.update(worldCoords, simulation->getCamera()->getZoom(), UIHovered);

        // Update simulation state if playing
        if (state == State::Playing) {
            realTime += deltaTime * speed;
            env.simulate(1);
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
    windowView.setCenter({viewSize.x / 2, viewSize.y / 2});

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

int Simulator::getStep() const {
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