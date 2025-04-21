#include <modules/graphics/components/viewport.hpp>
#include <simulator/simulator.hpp>

Viewport::Viewport(const unordered_map<string, string>& properties) : UIElement(properties) {
    styleSetters["background"] = [this](const string& value) {
        backgroundColor = parseColor(value);
    };
    propertySetters["camera"] = [this](const string& value) {
        camera.setLocked(value == "locked");
    };
    propertySetters["bounds"] = [this](const string& value) {
        Size boundArray[4] = {Size::Pixel(0), Size::Pixel(0), Size::Pixel(0), Size::Pixel(0)};
        parseMultiValue(value, boundArray);
        bounds = {
            boundArray[0].getValue(), boundArray[1].getValue(),
            boundArray[2].getValue(), boundArray[3].getValue()
        };
        camera.setBounds(&bounds);
    };

    camera = Camera(&viewport, &layout);
    vertexManager.setCamera(&camera);

    setProperties(properties);
    restyle();
}

void Viewport::setCameraBounds(sf::FloatRect* b) {
    camera.setBounds(b);
    camera.updateView();
}

void Viewport::draw(sf::RenderTarget& target) {
    viewport.clear(backgroundColor);
    vertexManager.draw(viewport);
    viewport.display();
    target.draw(viewportSprite);
}

bool Viewport::handleEvent(const sf::Event& event) {
    camera.handleEvent(event);
    return false;
}

bool Viewport::update(float dt, const sf::Vector2f& position) {
    UIElement::update(dt, position);
    camera.update(dt);
    return false;
}

sf::Vector2f Viewport::mapPixelToCoords(sf::Vector2f mousePos) {
    return camera.mapPixelToCoords(mousePos);
}

void Viewport::onLayout() {
    sf::FloatRect initialLayout = layout;
    UIElement::onLayout();

    if (layout.width <= 0 || layout.height <= 0) return;

    if (layout.width != initialLayout.width || layout.height != initialLayout.height) {
        viewport.create((int)layout.width, (int)layout.height);
    }
    camera.setTargetLayout(&layout);
    if (bounds.width != 0 && bounds.height != 0) {
        camera.setBounds(&bounds);
    }
    viewportSprite = sf::Sprite(viewport.getTexture());
    viewportSprite.setPosition(layout.left, layout.top);

    if (camera.getView().getSize().x == 0 || camera.getView().getSize().y == 0) {
        camera.setView(sf::View(sf::FloatRect(0, 0, layout.width, layout.height)));
    }
    camera.updateView();
}

Size Viewport::calculateWidth() {
    return width;
}

Size Viewport::calculateHeight() {
    return height;
}
