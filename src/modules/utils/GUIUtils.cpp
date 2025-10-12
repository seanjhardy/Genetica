#include <SFML/Graphics.hpp>
#include <modules/utils/GUIUtils.hpp>
#include <modules/utils/vector_types.hpp>

sf::Color brightness(sf::Color color, float brightness) {
    return {
        static_cast<sf::Uint8>(color.r * brightness),
        static_cast<sf::Uint8>(color.g * brightness),
        static_cast<sf::Uint8>(color.b * brightness),
        color.a
    };
}

sf::FloatRect computeBoundingBox(std::vector<float2> points) {
    if (points.empty()) return {};

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (const auto& p : points) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    return { minX, minY, maxX - minX, maxY - minY };
}

sf::Color HSVtoRGB(float H, float S, float V) {
    float C = S * V; // Chroma
    float HPrime = std::fmod(H / 60, 6.f); // H'
    float X = C * (1 - std::fabs(std::fmod(HPrime, 2.f) - 1));
    float M = V - C;

    float R = 0.f;
    float G = 0.f;
    float B = 0.f;

    switch (static_cast<int>(HPrime)) {
    case 0: R = C;
        G = X;
        break; // [0, 1)
    case 1: R = X;
        G = C;
        break; // [1, 2)
    case 2: G = C;
        B = X;
        break; // [2, 3)
    case 3: G = X;
        B = C;
        break; // [3, 4)
    case 4: R = X;
        B = C;
        break; // [4, 5)
    case 5: R = C;
        B = X;
        break; // [5, 6)
    }

    R += M;
    G += M;
    B += M;

    sf::Color color;
    color.r = static_cast<sf::Uint8>(std::round(R * 255));
    color.g = static_cast<sf::Uint8>(std::round(G * 255));
    color.b = static_cast<sf::Uint8>(std::round(B * 255));

    return color;
}
