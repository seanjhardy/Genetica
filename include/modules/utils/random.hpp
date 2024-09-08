#ifndef RANDOM
#define RANDOM

#include <random>
#include <map>
#include <string>

class Random {
public:
    static bool randomBool() {
        if (m_bits_left == 0) {
            refreshRandom();
        }
        bool const ret = m_rand & 1;
        --m_bits_left;
        m_rand >>= 1;
        return ret;
    }

    static float random() {
        return (float)dist(rng);
    }

    static int random(int max) {
        return (int)(max * dist(rng));
    }

    static float random(float min, float max) {
        return (max - min) * dist(rng) + min;
    }

    static std::string randomBase() {
        if (m_bits_left < 2) {
            refreshRandom();
        }
        uint8_t const result = (m_rand & 3);  // Extract the lowest 2 bits
        m_rand >>= 2;
        m_bits_left -= 2;
        return std::to_string(result);  // Convert to string
    }

    static constexpr const uint64_t s_mask_left1 = uint64_t(1) << (sizeof(uint64_t) * 8 - 1);
    static std::random_device rd;
    static std::mt19937 rng;
    static uint64_t m_rand;
    static uint8_t m_bits_left;
    static std::uniform_real_distribution<double> dist;
private:
    static void refreshRandom() {
        m_rand = std::uniform_int_distribution<uint64_t>{}(rng);
        m_bits_left = sizeof(uint64_t) * 8;
    }
};

#endif