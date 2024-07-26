// simulator.hpp
#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include "../environments/environment.hpp"
#include "camera/CameraController.hpp"

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

    sf::RenderWindow window{};
    Environment& environment;
    State state;
    bool rendering;
    CameraController camera;
};