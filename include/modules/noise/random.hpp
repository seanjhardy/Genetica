#ifndef RANDOM
#define RANDOM

#include "random"
#include "map"

class Random {
public:
    static bool randomBool() {
        if (1 == m_rand) {
            m_rand = std::uniform_int_distribution<uint64_t>{}(rng) | Random::s_mask_left1;
        }
        bool const ret = m_rand & 1;
        m_rand >>= 1;
        return ret;
    }

    static float random(float min = 0.0, float max = 1.0) {
        return (max - min) * static_cast <float> (rand()) /(static_cast <float> (RAND_MAX)) + min;
    }

    static constexpr const uint64_t s_mask_left1 = uint64_t(1) << (sizeof(uint64_t) * 8 - 1);
    static std::random_device rd;
    static std::mt19937 rng;
    static uint64_t m_rand;
};

#endif