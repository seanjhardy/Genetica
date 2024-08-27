// simulator.hpp
#ifndef SIMULATION
#define SIMULATION

#include "SFML/Graphics.hpp"
#include "vector"
#include "Camera.hpp"
#include <modules/graphics/vertexManager.hpp>
#include "environment.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/graphics/UIManager.hpp>
#include "simulator/entities/entity.hpp"

class Simulator {
public:
    enum class State {
        Paused,
        Playing,
        Fast,
    };
    enum class Tab {
        Simulation,
        LifeForm,
    };
private:
    // Time and framerate
    double realTime = 0;
    float speed = 1.0;
    int step = 0;
    int MAX_FRAMERATE = 60;
    double FRAME_INTERVAL = CLOCKS_PER_SEC / MAX_FRAMERATE;
    std::clock_t lastRenderTime = std::clock();

    // Rendering
    sf::RenderWindow window{};
    VertexManager vertexManager{};
    UIManager uiManager;

    // Simulation state
    Camera camera{};
    State state;
    Environment env;
    GeneticAlgorithm geneticAlgorithm;
    Entity* selectedEntity = nullptr;

    // Singleton class functions
    Simulator();
    Simulator(Simulator const&);              // Don't Implement
    void operator=(Simulator const&); // Don't implement

public:
    void run();
    void reset();
    void setup();
    void setState(State newState);
    void speedUp();
    void slowDown();

    std::string getTimeString() const;
    float getSpeed() const;
    int getStep() const;
    State getState();
    sf::RenderWindow& getWindow();
    void setTab(Tab tab);

    Environment& getEnv();
    GeneticAlgorithm& getGA();
    Entity* getSelectedEntity();
    Camera& getCamera();

    static Simulator& get();
};

#endif