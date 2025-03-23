// simulator.hpp
#ifndef SIMULATION
#define SIMULATION

#include "SFML/Graphics.hpp"
#include "environment.hpp"
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/graphics/UIManager.hpp>
#include <modules/graphics/components/viewport.hpp>

/** Singleton class that manages the simulation
 *
 * Fetched throughout the program using Simulator::get()
 **/
class Simulator {
public:
    enum class State {
        Paused,
        Playing,
    };
    enum class Tab {
        Simulation,
        LifeForm,
    };
private:
    // Time and framerate
    double realTime = 0;
    float speed = 1.0;
    size_t step = 0;
    int MAX_FRAMERATE = 60;
    double FRAME_INTERVAL = CLOCKS_PER_SEC / MAX_FRAMERATE;
    std::clock_t lastRenderTime = std::clock();

    // Rendering
    sf::RenderWindow window{};
    UIManager uiManager;
    sf::View windowView;

    // Simulation state
    State state;
    Environment env;
    size_t selectedEntityId = -1;

    // Singleton class functions
    Simulator();
    Simulator(Simulator const&);              // Don't Implement
    void operator=(Simulator const&); // Don't implement

public:
    void run();
    void handleEvents(Viewport* simulation);
    void reset();
    void init();
    void setState(State newState);
    void speedUp();
    void slowDown();
    void cleanup();

    std::string getTimeString() const;
    float getSpeed() const;
    size_t getStep() const;
    float getRealTime() const;
    State getState();
    sf::RenderWindow& getWindow();
    void updateWindowView();
    void setTab(Tab tab);

    Environment& getEnv();
    size_t getSelectedEntityId() const;

    static Simulator& get();
};

#endif