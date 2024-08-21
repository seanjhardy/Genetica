// simulator.hpp
#pragma once

#include "SFML/Graphics.hpp"
#include "vector"
#include "CameraController.hpp"
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
    CameraController camera;
    State state;
    Environment env;
    GeneticAlgorithm geneticAlgorithm;

    int entityID = 0;

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

    int nextEntityID();

    std::string getTimeString() const;
    float getSpeed() const;
    int getStep() const;
    State getState();
    sf::RenderWindow& getWindow();

    Environment& getEnv();
    GeneticAlgorithm& getGA();
    Entity* selectedEntity = nullptr;

    static Simulator& get();
};