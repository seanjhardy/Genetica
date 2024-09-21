#include "modules/graphics/components/screen.hpp"
#include "modules/graphics/components/text.hpp"
#include "simulator/simulator.hpp"
#include "modules/graphics/componentManager.hpp"
#include "modules/graphics/functionManager.hpp"
#include "modules/graphics/utils/HTMLParser.hpp"
#include "simulator/entities/lifeform.hpp"
#include <modules/graphics/components/image.hpp>
#include <modules/graphics/components/viewport.hpp>
#include <simulator/planet.hpp>
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
        screen->getElement("bottomBar")->overrideProperty("style", "visible: true");
        screen->getElement("showUIView")->overrideProperty("style", "visible: false");
        screen->getElement("root")->onLayout();
    });

    FunctionManager::add("hideUI", [simulator, screen]() {
        screen->getElement("bottomBar")->overrideProperty("style", "visible: false");
        screen->getElement("showUIView")->overrideProperty("style", "visible: true");
        screen->getElement("root")->onLayout();
    });

    FunctionManager::add("toggleFluid", [simulator, screen]() {
        if (simulator->getEnv().getFluidEnabled()) {
            simulator->getEnv().toggleFluidEnabled();
            screen->getElement("fluidBtnIcon")->overrideProperty("style", "image: fluidEnabled");
            screen->getElement("fluidBtnIcon")->overrideProperty("styleOnHover", "image: fluidEnabledHighlighted");
        } else {
            simulator->getEnv().toggleFluidEnabled();
            screen->getElement("fluidBtnIcon")->overrideProperty("style", "image: fluidDisabled");
            screen->getElement("fluidBtnIcon")->overrideProperty("styleOnHover", "image: fluidDisabledHighlighted");
        }
    });

    FunctionManager::add("clone", [simulator]() {
        dynamic_cast<LifeForm*>(simulator->getSelectedEntity())->clone(false);
    });
    FunctionManager::add("mutate", [simulator]() {
        simulator->getGA().mutate(dynamic_cast<LifeForm*>(simulator->getSelectedEntity())->genome);
    });
    FunctionManager::add("energy", [simulator]() {
        dynamic_cast<LifeForm*>(simulator->getSelectedEntity())->energy += 100;
    });
    FunctionManager::add("delete", [simulator]() {
        dynamic_cast<LifeForm*>(simulator->getSelectedEntity())->kill();
    });

    FunctionManager::add("randomPlanet", [screen, simulator]() {
        Planet* planet;
        do {
            planet = Planet::getRandom();
        } while (simulator->getEnv().getPlanet().name == planet->thumbnail);

        simulator->getEnv().setPlanet(planet);

        ((ImageElement*)screen->getElement("planetBtnIcon"))
        ->overrideProperty("style", "image: " + planet->thumbnail);
        ((Text*)screen->getElement("planetBtnName"))->setText(planet->name);
    });

    screen->addFunction([simulator, screen]() {
        if (simulator->getSelectedEntity() == nullptr) {
            string text = "Time: " + simulator->getTimeString() +
                   "\nStep: " + std::to_string(simulator->getStep()) +
                   "\nSpeed: " + std::format("{:.2f}", simulator->getSpeed()) +
                   "\nLifeForms: " + std::to_string(simulator->getGA().getPopulation().size()) +
                   "\nCells: " + std::to_string(simulator->getEnv().getPoints().size()) +
                   "\nSpecies: " + std::to_string(simulator->getGA().getSpecies().size());
            dynamic_cast<Text*>(screen->getElement("simulationInfoLabel"))->setText(text);
        } else if (dynamic_cast<LifeForm*>(simulator->getSelectedEntity())) {
            auto selectedLifeform = dynamic_cast<LifeForm*>(simulator->getSelectedEntity());
            string text = "Energy: " + std::to_string(selectedLifeform->energy);
            dynamic_cast<Text*>(screen->getElement("lifeformInfoLabel"))->setText(text);

            selectedLifeform->grn.render(((Viewport*)screen->getElement("geneRegulatoryNetwork"))->getVertexManager());
            sf::FloatRect layout = ((Viewport*)screen->getElement("geneRegulatoryNetwork"))->layout;
            float zoom = std::min(layout.width, layout.height);
            ((Viewport*)screen->getElement("geneRegulatoryNetwork"))->getCamera()->setZoom(zoom);
            ((Viewport*)screen->getElement("geneRegulatoryNetwork"))->getCamera()->setPosition({0.5, 0.5});
       }
    });

    vector<UIElement*> elements = ComponentManager::get("simulationScreen");
    for (const auto& child : elements) {
        screen->addElement(child);
    }

    // Post setup functions
    ((Viewport*)screen->getElement("simulation"))->setCameraBounds(simulator->getEnv().getBounds());

    return screen;
}