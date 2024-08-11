#include <modules/graphics/UI/screen.hpp>
#include <modules/graphics/UI/button.hpp>
#include <modules/graphics/UI/text.hpp>
#include "simulator/simulator.hpp"
#include "simulator/screens/simulationScreen.hpp"

Screen *getSimulationScreen(Simulator *simulator) {
    auto screen = new Screen();

    auto* infoLabel = new Label("", "font-size: 15px;");

    // Define play button
    auto* playButton = new Button("", nullptr,
               "flex: 1; height: 40px; border: 2px rgba(0,0,0,100) 5px; "
               "font-size: 15px; padding: 5px; icon: pause; background: #0e3c42;",
               "icon: pauseHighlighted; background: #20868a; ");
    auto togglePaused = [simulator, playButton]() {
        if (simulator->getState() == Simulator::State::Paused) {
            simulator->setState(Simulator::State::Playing);
            playButton->overrideStyle("icon: pause;");
            playButton->overrideStyleOnHover("icon: pauseHighlighted;");
        } else {
            simulator->setState(Simulator::State::Paused);
            playButton->overrideStyle("icon: play;");
            playButton->overrideStyleOnHover("icon: playHighlighted;");
        }
    };
    playButton->setOnClick(togglePaused);

    auto* slowDownButton = new Button("", [simulator]() {simulator->setState(Simulator::State::Paused);},
                                         "flex: 1; height: 40px; border: 2px rgba(0,0,0,100) 5px; "
                                         "font-size: 15px; padding: 5px; icon: slowDown; background: #0e3c42;",
                                         "icon: slowDownHighlighted; background: #20868a;");
    auto* fastForwardButton = new Button("", [simulator]() {simulator->setState(Simulator::State::Paused);},
                                   "flex: 1; height: 40px; border: 2px rgba(0,0,0,100) 5px; "
                                   "font-size: 15px; padding: 5px; icon: fastForward; background: #0e3c42;",
                                   "icon: fastForwardHighlighted; background: #20868a;");

    auto createRandom = []() {
        GeneticAlgorithm::get().getEnvironment()->createRandomIndividual();
    };

    auto* infoBox = new Container("flex-direction: column; width: 200px; height: 100%;"
                                    "align-row: center; align-col: start; background: #215057; "
                                    "border: 2px #215057 2px; padding: 5px; gap: 10px;", {
        infoLabel,
    });


    auto *root = new Container("width: 100%; height: 100%;"
                               "flex-direction: column; background: transparent;"
                               "align-row: center; align-col: end;");

    auto *bottomBar =
      new Container("flex-direction: row; width: 100%; height: 150px;"
                    "align-row: center; align-col: center; background: #2f5b61; "
                    "border: 2px #438891 0px; padding: 5px; gap: 5px;");

    auto* controlPanel =
    new Container("flex-direction: column; width: 200px; height: 100%;"
                  "align-row: center; align-col: start; background: #215057; "
                  "border: 2px #215057 2px; padding: 5px; gap: 10px;", {
                    new Container("flex-direction: row; width: 200px; flex: 1;"
                                  "align-row: center; align-col: start; background: #215057; "
                                  "border: 2px #215057 2px; padding: 5px; gap: 10px;", {
                                    slowDownButton, playButton, fastForwardButton
                                  }
                    ),
                    new Container("flex-direction: row; width: 200px; flex: 1;"
                                  "align-row: center; align-col: start; background: #215057; "
                                  "border: 2px #215057 2px; padding: 5px; gap: 10px;", {
                      new Button("Create Random", []() {GeneticAlgorithm::get().getEnvironment()->createRandomIndividual();},
                                 "flex: 1; height: 40px; border: 2px rgba(0,0,0,100) 5px; "
                                 "font-size: 15px; padding: 5px; background: #0e3c42;",
                                 "background: #20868a;")
                    }),
                  });

    bottomBar->addChild(controlPanel);
    bottomBar->addChild(infoBox);

    root->addChild(bottomBar);
    screen->addElement(root);
    return screen;
}