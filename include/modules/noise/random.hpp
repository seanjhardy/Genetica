#ifndef RANDOM
#define RANDOM

#include "random"
#include "map"
#define UNLIKELY(x) __builtin_expect((x), 0)

class Rand {
public:
    bool operator()() {
        if (UNLIKELY(1 == m_rand)) {
            m_rand = std::uniform_int_distribution<uint64_t>{}(rng) | s_mask_left1;
        }
        bool const ret = m_rand & 1;
        m_rand >>= 1;
        return ret;
    }

private:
    static constexpr const uint64_t s_mask_left1 = uint64_t(1) << (sizeof(uint64_t) * 8 - 1);
    std::random_device rd;
    std::mt19937 rng = std::mt19937(rd());
    uint64_t m_rand = 1;
};

Rand randomBool;

inline float getRandom(float min = 0.0, float max = 1.0) {
    return (max - min) * static_cast <float> (rand()) /(static_cast <float> (RAND_MAX)) + min;
}

#endif