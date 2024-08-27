#include "modules/graphics/components/screen.hpp"
#include "modules/graphics/components/text.hpp"
#include "simulator/simulator.hpp"
#include "modules/graphics/componentManager.hpp"
#include "modules/graphics/functionManager.hpp"
#include "modules/graphics/utils/HTMLParser.hpp"
#include "simulator/entities/lifeform.hpp"
#include <format>

inline Screen *getSimulationScreen(Simulator *simulator) {
    auto screen = new Screen();

    FunctionManager::add("slowDown", [simulator]() { simulator->slowDown(); });
    FunctionManager::add("speedUp", [simulator]() { simulator->speedUp(); });
    FunctionManager::add("createRandom", [simulator]() { simulator->getGA().createRandomLifeForm(); });
    FunctionManager::add("reset", [simulator]() { simulator->reset(); });

    // Define play button
    FunctionManager::add("togglePaused", [simulator, screen]() {
        if (simulator->getState() == Simulator::State::Paused) {
            simulator->setState(Simulator::State::Playing);
            screen->getElement("playBtnIcon")->overrideProperty("style", "image: pause");
            screen->getElement("playBtnIcon")->overrideProperty("styleOnHover", "image: pause");
        } else {
            simulator->setState(Simulator::State::Paused);
            screen->getElement("playBtnIcon")->overrideProperty("style", "image: play");
            screen->getElement("playBtnIcon")->overrideProperty("styleOnHover", "image: playHighlighted");
        }
    });

    //Settings buttons
    FunctionManager::add("toggleQuadTree", [simulator, screen]() {
        if (simulator->getEnv().getGridLineVisibility()) {
            simulator->getEnv().toggleGridLinesVisible();
            screen->getElement("quadtreeBtnIcon")->overrideProperty("style", "image: quadtree");
            screen->getElement("quadtreeBtnIcon")->overrideProperty("styleOnHover", "image: quadtreeHighlighted");
        } else {
            simulator->getEnv().toggleGridLinesVisible();
            screen->getElement("quadtreeBtnIcon")->overrideProperty("style", "image: map");
            screen->getElement("quadtreeBtnIcon")->overrideProperty("styleOnHover", "image: mapHighlighted");
        }
    });

    FunctionManager::add("showUI", [simulator, screen]() {
        screen->getElement("root")->overrideProperty("style", "visible: true");
        screen->getElement("showUIView")->overrideProperty("style", "visible: false");
        screen->resize(simulator->getWindow().getSize());
    });

    FunctionManager::add("hideUI", [simulator, screen]() {
        screen->getElement("root")->overrideProperty("style", "visible: false");
        screen->getElement("showUIView")->overrideProperty("style", "visible: true");
        screen->resize(simulator->getWindow().getSize());
    });

    FunctionManager::add("clone", [simulator]() {
        //
    });
    FunctionManager::add("mutate", [simulator]() {
        //
    });
    FunctionManager::add("energy", [simulator]() {
        //
    });
    FunctionManager::add("delete", [simulator]() {
        //
    });

    screen->addFunction([simulator, screen]() {
        if (simulator->getSelectedEntity() == nullptr) {
            string text = "Time: " + simulator->getTimeString() +
                   "\nStep: " + std::to_string(simulator->getStep()) +
                   "\nSpeed: " + std::format("{:.2f}", simulator->getSpeed()) +
                   "\nLifeForms: " + std::to_string(simulator->getGA().getPopulation().size()) +
                   "\nSpecies: " + std::to_string(simulator->getGA().getSpecies().size());
            dynamic_cast<Text*>(screen->getElement("simulationInfoLabel"))->setText(text);
        } else if (dynamic_cast<LifeForm*>(simulator->getSelectedEntity())) {
            auto selectedLifeform = dynamic_cast<LifeForm*>(simulator->getSelectedEntity());
            string text = "Energy: " + std::to_string(selectedLifeform->energy);
            dynamic_cast<Text*>(screen->getElement("lifeformInfoLabel"))->setText(text);
        }
    });

    vector<UIElement*> elements = ComponentManager::get("simulationScreen");
    for (const auto& child : elements) {
        screen->addElement(child);
    }
    return screen;
}