// simulator.hpp
#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include "../environments/environment.hpp"
#include "camera/CameraController.hpp"
#include "../modules/graphics/VertexManager.hpp"

class Simulator {
public:
    enum class State {
        Paused,
        Playing
    };

    explicit Simulator(Environment& environment);
    void run();
    void setState(State newState);
    void reset();
    void setRendering(bool render);

private:
    int time = 0;
    int MAX_FRAMERATE = 60;
    double FRAME_INTERVAL = CLOCKS_PER_SEC / MAX_FRAMERATE;
    std::clock_t lastRenderTime = std::clock();

    sf::RenderWindow window{};
    VertexManager vertexManager{};
    CameraController camera;
    bool rendering;
    State state;

    Environment& environment;
};