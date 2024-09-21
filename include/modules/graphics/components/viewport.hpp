#include <modules/graphics/utils/UIElement.hpp>

// For high performance rendering with its own VertexManager
class Viewport : public UIElement {
private:
    VertexManager vertexManager;
    sf::RenderTexture viewport;
    sf::Sprite viewportSprite;
    Camera camera;
    sf::FloatRect bounds;
    sf::Color backgroundColor = sf::Color::Black;
public:
    explicit Viewport(const unordered_map<string, string>& properties);
    VertexManager &getVertexManager() { return vertexManager; }

    void draw(sf::RenderTarget& target) override;
    bool handleEvent(const sf::Event& event) override;
    void update(float dt, const sf::Vector2f& position) override;
    void onLayout() override;

    void setCameraBounds(sf::FloatRect* bounds);
    sf::Vector2f mapPixelToCoords(sf::Vector2f mousePos);
    Camera* getCamera() { return &camera; }
};