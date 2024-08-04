#include "modules/graphics/UI/screen.hpp"
#include "modules/graphics/UI/button.hpp"
#include "modules/graphics/UI/text.hpp"
#include "modules/graphics/UI/image.hpp"
#include "modules/graphics/UI/container.hpp"
#include "simulator/simulator.hpp"

inline std::unique_ptr<Screen> getSimulationScreen(Simulator* simulator) {
    auto *screen = new Screen();

    unique_ptr<Container> container;

    // Add a child with fixed pixel size
    auto pause = [&simulator]() {
        simulator->setState(Simulator::State::Paused);
    };
    auto play = [&simulator]() {
        simulator->setState(Simulator::State::Playing);
    };

    container->addChild(new Button(sf::FloatRect(0,0,0,0),
                                  "Play", &play),
                       Size::Pixel(100), Size::Pixel(50));

    // Add a child that flexes to fill available space
    container->addChild(new Button(sf::FloatRect(0,0,0,0),
                                  "Pause", &pause), Size::Flex(1), Size::Flex(1));

    screen->addElement(reinterpret_cast<unique_ptr<UIElement> &&>(container));

    return std::unique_ptr<Screen>(screen);
}