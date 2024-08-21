#include <modules/graphics/UI/screen.hpp>
#include <modules/graphics/UI/button.hpp>
#include <modules/graphics/UI/text.hpp>
#include "simulator/simulator.hpp"
#include "simulator/screens/simulationScreen.hpp"
#include <modules/graphics/styleManager.hpp>
#include <format>

Container *getSettings(Simulator *simulator, Screen *screen, Container *root) {
    auto *settings = new Container(Styles::get("container") + "flex-direction: column; width: 50px; height: 100%;");

    auto *quadTreeButton = new Button(Styles::get("miniBtn") + "icon: quadtree;",
                                      Styles::get("btnHover") + "icon: quadtreeHighlighted;");

    auto toggleQuadTree = [simulator, quadTreeButton]() {
        if (simulator->getEnv().isQuadTreeVisible()) {
            simulator->getEnv().toggleQuadTreeVisible();
            quadTreeButton->overrideStyle("icon: quadtree;");
            quadTreeButton->overrideStyleOnHover("icon: quadtreeHighlighted;");
        } else {
            simulator->getEnv().toggleQuadTreeVisible();
            quadTreeButton->overrideStyle("icon: map;");
            quadTreeButton->overrideStyleOnHover("icon: mapHighlighted;");
        }
    };
    quadTreeButton->setOnClick(toggleQuadTree);


    auto showUI = [simulator, screen, root]() {
        screen->reset();
        screen->addElement(root);
        screen->resize(simulator->getWindow().getSize());
    };
    auto *showUIBar = new Container("width: 100%; height: 100%;"
                                    "flex-direction: column;"
                                    "align-row: start; align-col: end; margin: 20px;", {
                                      new Button(showUI,
                                                 Styles::get("miniBtn") + "icon: eye;",
                                                 Styles::get("btnHover") + "icon: eyeHighlighted;"),
                                    });

    auto hideUI = [simulator, screen, showUIBar]() {
        screen->reset();
        screen->addElement(showUIBar);
        screen->resize(simulator->getWindow().getSize());
    };
    auto *UIButton = new Button(hideUI,
                                Styles::get("miniBtn") + "icon: noEye;",
                                Styles::get("btnHover") + "icon: noEyeHighlighted;");

    settings->addChild(quadTreeButton);
    settings->addChild(UIButton);
    return settings;
}


Screen *getSimulationScreen(Simulator *simulator) {
    auto screen = new Screen();

    // Define play button
    auto *playButton = new Button(Styles::get("mainBtn") + "height: 100%; icon: pause;",
                                  Styles::get("mainBtnHover") + "icon: pauseHighlighted");
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

    auto *slowDownButton = new Button([simulator]() { simulator->slowDown(); },
                                      Styles::get("mainBtn") + "height: 100%; icon: slowDown;",
                                      Styles::get("mainBtnHover") + "icon: slowDownHighlighted;");
    auto *fastForwardButton = new Button([simulator]() { simulator->speedUp(); },
                                         Styles::get("mainBtn") + "height: 100%; icon: fastForward;",
                                         Styles::get("mainBtnHover") + "icon: fastForwardHighlighted;");

    auto *infoLabel = new Text("", "font-size: 18px;");
    auto *infoBox = new Container(Styles::get("container") + "flex-direction: column; width: 200px; height: 100%;",
                                  {infoLabel});


    auto *root = new Container(Styles::get("root") + "align-col: end;");

    auto *bottomBar =
      new Container(Styles::get("container") + "flex-direction: row; width: 100%; height: 150px;"
                    "background: #2f5b61; border: 2px #438891 0px;");

    auto *controlPanel =
      new Container(Styles::get("container") + "width: 200px; height: 100%;", {
                      new Container("flex-direction: row; width: 100%; flex: 1;"
                                    "align-row: center; align-col: start; "
                                    "border: 2px #215057 2px; gap: 10px;", {
                                      slowDownButton, playButton, fastForwardButton
                                    }
                      ),
                      new Button("Create Random",
                                 [simulator]() {simulator->getEnv().createRandomLifeForm(); },
                                 Styles::get("mainBtn") + "width: 100%;", Styles::get("mainBtnHover")),
                      new Button("Reset",
                                 [simulator]() {simulator->reset(); },
                                 Styles::get("mainBtn") + "width: 100%;", Styles::get("mainBtnHover")),
                    });

    auto *settings = getSettings(simulator, screen, root);

    bottomBar->addChild(settings);
    bottomBar->addChild(controlPanel);
    bottomBar->addChild(infoBox);

    root->addChild(bottomBar);
    screen->addFunction([simulator, infoLabel]() {
        infoLabel->setText("\nTime: " + simulator->getTimeString() +
                           "\nStep: " + std::to_string(simulator->getStep()) +
                           "\nSpeed: " + std::format("{:.2f}", simulator->getSpeed()) +
                           "\nLifeForms: " + std::to_string(simulator->getGA().getPopulation().size()) +
                           "\nSpecies: " + std::to_string(simulator->getGA().getSpecies().size())
        );
    });
    screen->addElement(root);
    return screen;
}