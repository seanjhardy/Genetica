#ifndef RANDOM
#define RANDOM
#include <random>

inline float getRandom(float min = 0.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

#endif