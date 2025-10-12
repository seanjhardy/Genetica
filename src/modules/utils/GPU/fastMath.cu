#include <modules/utils/gpu/fastMath.hpp>
#include <cmath>
#include "modules/utils/gpu/mathUtils.hpp"

float FastMath::cosTable[FastMath::TABLE_SIZE];
float FastMath::sinTable[FastMath::TABLE_SIZE];

float __host__ FastMath::cos(float angle) {
    return getValue(angle, FastMath::cosTable);
}

float __host__ FastMath::sin(float angle) {
    return getValue(angle, FastMath::sinTable);
}

void __host__ FastMath::init() {
    float step = M_PI_2 / TABLE_SIZE;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        float angle = i * step;
        FastMath::cosTable[i] = std::cos(angle);
        FastMath::sinTable[i] = std::sin(angle);
    }
}

float __host__ FastMath::getValue(float angle, const float* table) {
    if (angle < 0) {
        angle = -angle;
    }
    // Normalize angle to range [0, 2π) and find the index
    int index = static_cast<int>((angle / M_PI_2 * TABLE_SIZE)) % TABLE_SIZE;
    return table[index];
}

float __host__ __device__ FastMath::atan2f(float y, float x) {
    float abs_y = fabs(y) + 1e-10f; // Add small number to prevent division by zero
    float r, angle;
    if (x >= 0) {
        r = (x - abs_y) / (x + abs_y);
        angle = 0.1963f * r * r * r - 0.9817f * r + M_PI_4;
    }
    else {
        r = (x + abs_y) / (abs_y - x);
        angle = 0.1963f * r * r * r - 0.9817f * r + 3.0f * M_PI_4;
    }
    return (y < 0) ? -angle : angle;
}
