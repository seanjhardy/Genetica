#include <modules/noise/noise.hpp>
#include <simulator/simulator.hpp>

void Noise::applyToTexture(sf::RenderTexture* texture, sf::RenderStates states, sf::FloatRect bounds, float seed) {
    ShaderManager::get("perlin")->setUniform("texture", sf::Shader::CurrentTexture);
    ShaderManager::get("perlin")->setUniform("seed", (float)(seed) / 10000.0f);
    auto* shaderMainColours = new sf::Glsl::Vec4[colours.size()];
    for (int i = 0; i < colours.size(); i++) {
        shaderMainColours[i] = sf::Glsl::Vec4((float)colours[i].r / 255.0f,
                                              (float)colours[i].g / 255.0f,
                                              (float)colours[i].b / 255.0f,
                                              (float)colours[i].a / 255.0f);
    }
    ShaderManager::get("perlin")->setUniformArray("colours", shaderMainColours, colours.size());
    ShaderManager::get("perlin")->setUniform("numColours", (int)colours.size());
    ShaderManager::get("perlin")->setUniform("offset", sf::Glsl::Vec2(bounds.left, bounds.top));
    ShaderManager::get("perlin")->setUniform("resolution",
                                             sf::Glsl::Vec2((float)bounds.width, (float)bounds.height));
    if (animated) {
        ShaderManager::get("perlin")->setUniform("time", Simulator::get().getRealTime() / 30.0f);
    }
    ShaderManager::get("perlin")->setUniform("noiseFrequency", noiseFrequency);
    ShaderManager::get("perlin")->setUniform("noiseOctaves", noiseOctaves);
    ShaderManager::get("perlin")->setUniform("smoothNoise", smoothNoise);
    ShaderManager::get("perlin")->setUniform("noiseWarp", noiseWarp);
    texture->draw(sf::Sprite(texture->getTexture()), states);
}

void Add::applyToTexture(sf::RenderTexture* texture, sf::FloatRect bounds, float seed) {
    if (!update && !base.animated) {
        texture->draw(sf::Sprite(cachedTexture.getTexture()));
        return;
    }
    sf::RenderStates states;
    cachedTexture.create(texture->getSize().x, texture->getSize().y);
    cachedTexture.clear(sf::Color::Transparent);

    states.texture = &cachedTexture.getTexture(); //&texture->getTexture();
    states.blendMode = sf::BlendMode(
        sf::BlendMode::One, // Source factor (for color)
        sf::BlendMode::OneMinusSrcColor, // Destination factor (for color)
        sf::BlendMode::Add, // Subtraction equation (for color)
        sf::BlendMode::One, // Source factor (for alpha)
        sf::BlendMode::One, // Destination factor (for alpha)
        sf::BlendMode::Add // Subtraction equation (for alpha)
    );
    states.shader = ShaderManager::get("perlin");

    base.applyToTexture(&cachedTexture, states, bounds, seed);
    cachedTexture.display();
    texture->draw(sf::Sprite(cachedTexture.getTexture()));
    update = false;
}

void Mask::applyToTexture(sf::RenderTexture* texture, sf::FloatRect bounds, float seed) {
    if (!update) {
        texture->draw(sf::Sprite(cachedTexture.getTexture()));
        return;
    }
    sf::RenderStates baseStates;
    baseStates.texture = &texture->getTexture();
    baseStates.blendMode = sf::BlendAlpha;
    baseStates.shader = ShaderManager::get("perlin");

    // Create a new texture, write the base texture, then cut out the mask
    if (!cachedTexture.create(texture->getSize().x, texture->getSize().y)) {
        return;
    }
    cachedTexture.clear(sf::Color::Transparent);


    base.applyToTexture(&cachedTexture, baseStates, bounds, seed);

    sf::RenderStates maskStates;
    maskStates.texture = &cachedTexture.getTexture();
    maskStates.blendMode = sf::BlendMode(
        sf::BlendMode::One, // Source factor (for color)
        sf::BlendMode::One, // Destination factor (for color)
        sf::BlendMode::Add, // Subtraction equation (for color)
        sf::BlendMode::One, // Source factor (for alpha)
        sf::BlendMode::One, // Destination factor (for alpha)
        sf::BlendMode::ReverseSubtract // Subtraction equation (for alpha)
    );
    maskStates.shader = ShaderManager::get("perlin");
    mask.applyToTexture(&cachedTexture, maskStates, bounds, seed + 10);
    cachedTexture.display();

    texture->draw(sf::Sprite(cachedTexture.getTexture()));
    update = false;
}
