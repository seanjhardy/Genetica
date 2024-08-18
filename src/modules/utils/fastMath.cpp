#include <modules/utils/fastMath.hpp>
#include <cmath>

float FastMath::cosTable[FastMath::TABLE_SIZE];
float FastMath::sinTable[FastMath::TABLE_SIZE];

float FastMath::cos(float angle) {
    return getValue(angle, FastMath::cosTable);
}

float FastMath::sin(float angle) {
    return getValue(angle, FastMath::sinTable);
}

void FastMath::init() {
    float step = M_PI_2 / TABLE_SIZE;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        float angle = i * step;
        FastMath::cosTable[i] = std::cos(angle);
        FastMath::sinTable[i] = std::sin(angle);
    }
}

float FastMath::getValue(float angle, const float* table) {
    if (angle < 0) {
        angle = -angle;
    }
    // Normalize angle to range [0, 2π) and find the index
    int index = static_cast<int>((angle / M_PI_2 * TABLE_SIZE)) % TABLE_SIZE;
    return table[index];
}