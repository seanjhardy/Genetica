// simulator.hpp
#pragma once

#include "SFML/Graphics.hpp"
#include "vector"
#include "CameraController.hpp"
#include "modules/graphics/vertexManager.hpp"
#include "geneticAlgorithm/environment.hpp"
#include "geneticAlgorithm/geneticAlgorithm.hpp"
#include "modules/graphics/UIManager.hpp"

class Simulator {
public:
    enum class State {
        Paused,
        Playing,
        Fast,
    };

    explicit Simulator(Environment& environment, int width, int height);
    void run();
    void reset();
    void setState(State newState);

private:
    int time = 0;
    int MAX_FRAMERATE = 60;
    double FRAME_INTERVAL = CLOCKS_PER_SEC / MAX_FRAMERATE;
    std::clock_t lastRenderTime = std::clock();

    sf::RenderWindow window{};
    VertexManager vertexManager{};
    UIManager uiManager{};

    CameraController camera;
    State state;
};