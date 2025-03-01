#include "modules/graphics/components/screen.hpp"
#include "modules/graphics/components/text.hpp"
#include "simulator/simulator.hpp"
#include "modules/graphics/componentManager.hpp"
#include "modules/graphics/functionManager.hpp"
#include "modules/graphics/utils/HTMLParser.hpp"
#include "geneticAlgorithm/lifeform.hpp"
#include <modules/graphics/components/image.hpp>
#include <modules/graphics/components/viewport.hpp>
#include <simulator/planet.hpp>

inline Screen *getSimulationScreen(Simulator *simulator) {
    auto screen = new Screen();

    FunctionManager::add("slowDown", [simulator]() { simulator->slowDown(); });
    FunctionManager::add("speedUp", [simulator]() { simulator->speedUp(); });
    FunctionManager::add("createRandom", [simulator]() { simulator->getEnv().getGA().createRandomLifeForm(); });
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

    FunctionManager::add("showUI", [screen]() {
        screen->getElement("UI")->overrideProperty("style", "visible: true");
        screen->getElement("showUIView")->overrideProperty("style", "visible: false");
        screen->getElement("root")->onLayout();
    });

    FunctionManager::add("hideUI", [screen]() {
        screen->getElement("UI")->overrideProperty("style", "visible: false");
        screen->getElement("showUIView")->overrideProperty("style", "visible: true");
        screen->getElement("root")->onLayout();
    });

    FunctionManager::add("toggleGenomeTab", [screen]() {
        if (screen->getElement("genomePanel")->visible) {
            screen->getElement("genomePanel")->overrideProperty("style", "visible: false; ");
        } else {
            screen->getElement("genomePanel")->overrideProperty("style", "visible: true; ");
        }
        screen->getElement("root")->onLayout();
    });

    FunctionManager::add("copyGenome", [simulator]() {
        //dynamic_cast<LifeForm*>(simulator->getSelectedEntity())->genome;
    });

    FunctionManager::add("toggleGRNTab", [screen]() {
        if (screen->getElement("grnPanel")->visible) {
            screen->getElement("grnPanel")->overrideProperty("style", "visible: false; ");
        } else {
            screen->getElement("grnPanel")->overrideProperty("style", "visible: true; ");
        }
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
        simulator->getEnv().getGA().getPopulation()[simulator->getSelectedEntityId()].clone(false);
    });
    FunctionManager::add("mutate", [simulator]() {
        //simulator->getEnv().getGA().getPopulation()[simulator->getSelectedEntityId()].mutate();
    });
    FunctionManager::add("energy", [simulator]() {
        simulator->getEnv().getGA().getPopulation()[simulator->getSelectedEntityId()].energy = 100;
    });
    FunctionManager::add("delete", [simulator]() {
        simulator->getEnv().getGA().getPopulation()[simulator->getSelectedEntityId()].kill();
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
        dynamic_cast<Text*>(screen->getElement("time"))->setText(simulator->getTimeString());
        dynamic_cast<Text*>(screen->getElement("step"))->setText(std::to_string(simulator->getStep()));
        dynamic_cast<Text*>(screen->getElement("speed"))->setText("x" + roundToDecimalPlaces(simulator->getSpeed(), 2));

        dynamic_cast<Text*>(screen->getElement("species"))->setText(std::to_string(simulator->getEnv().getGA().getSpecies().size()));
        dynamic_cast<Text*>(screen->getElement("lifeforms"))->setText(std::to_string(simulator->getEnv().getGA().getPopulation().size()));
        dynamic_cast<Text*>(screen->getElement("cells"))->setText(std::to_string(simulator->getEnv().getPoints().size()));

        float temperature = simulator->getEnv().getPlanet().temperature;
        float octave1 = sin(0.1f * simulator->getRealTime());
        float octave2 = sin(0.05f * simulator->getRealTime());
        float octave3 = sin(2.0f * simulator->getRealTime());
        temperature += octave1 * 5.0f + octave2 * 1.0f + octave3 * 0.2f;
        float thermometerTemperature = clamp(-10, temperature, 50);
        sf::Color thermometerColor;
        if (temperature <= 10) {
            thermometerColor = interpolate(sf::Color(50, 50, 255), sf::Color(200, 200, 255),
                                                     (thermometerTemperature + 20)/30);
        } else {
            thermometerColor = interpolate(sf::Color(255, 200, 100), sf::Color(255, 100, 0),
                                                     (thermometerTemperature)/50);
        }
        std::string thermometerColorString = "rgb("
          + std::to_string(thermometerColor.r) + ","
          + std::to_string(thermometerColor.g) + ","
          + std::to_string(thermometerColor.b) + ")";
        dynamic_cast<Text*>(screen->getElement("temperature"))
        ->setText(roundToDecimalPlaces(temperature, 2) + "C");
        dynamic_cast<ImageElement*>(screen->getElement("thermometer"))
        ->overrideProperty("style", "tint: " + thermometerColorString);

        if (simulator->getSelectedEntityId() != -1) {
            auto selectedLifeform = Simulator::get().getEnv().getGA().getPopulation()[simulator->getSelectedEntityId()];
            string text = "Energy: " + std::to_string(selectedLifeform.energy);

            if (screen->getElement("genomePanel")->visible) {
                selectedLifeform.genome.render(
                  ((Viewport *) screen->getElement("genome"))->getVertexManager());
            }

            if (screen->getElement("grnPanel")->visible) {
                auto* grn = (Viewport *) screen->getElement("geneRegulatoryNetwork");
                /*if (grn != nullptr) {
                    selectedLifeform->grn.render(
                      (grn)->getVertexManager());
                    sf::FloatRect layout = ((Viewport *) screen->getElement("geneRegulatoryNetwork"))->layout;
                    float zoom = std::min(layout.width, layout.height);
                    ((Viewport *) screen->getElement("geneRegulatoryNetwork"))->getCamera()->setZoom(zoom);
                    ((Viewport *) screen->getElement("geneRegulatoryNetwork"))->getCamera()->setPosition({0.5, 0.5});
                }*/
            }
       }
    });

    vector<UIElement*> elements = ComponentManager::get("SimulationScreen");
    for (const auto& child : elements) {
        screen->addElement(child);
    }

    // Post setup functions
    ((Viewport*)screen->getElement("simulation"))->setCameraBounds(simulator->getEnv().getBounds());

    return screen;
}