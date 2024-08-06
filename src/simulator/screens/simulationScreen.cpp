#include <modules/graphics/UI/screen.hpp>
#include <modules/graphics/UI/button.hpp>
#include <modules/graphics/UI/text.hpp>
#include <modules/graphics/UI/image.hpp>
#include <modules/graphics/UI/container.hpp>
#include "simulator/simulator.hpp"
#include "simulator/screens/simulationScreen.hpp"

Screen* getSimulationScreen(Simulator* simulator) {
    auto screen = new Screen();
    auto* root(new Container(Container::Direction::Column, Container::Alignment::Start,
                             Container::Alignment::Center));

    auto* container(new Container(Container::Direction::Row, Container::Alignment::Center,
                                Container::Alignment::Center));

    // Add a child with fixed pixel size
    auto pause = [&simulator]() {
        simulator->setState(Simulator::State::Paused);
    };
    auto play = [&simulator]() {
        simulator->setState(Simulator::State::Playing);
    };

    container->addChild(new Button(sf::FloatRect(0,0,50,50),
                                  "Play", play),
                       Size::Pixel(100), Size::Pixel(50));

    // Add a child that flexes to fill available space
    container->addChild(new Button(sf::FloatRect(0,0,50,50),
                                  "Pause", pause), Size::Flex(1), Size::Flex(1));


    root->addChild(container, Size::Percent(100), Size::Pixel(100));
    screen->addElement(root);

    return screen;
}