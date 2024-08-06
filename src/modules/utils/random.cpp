#include <modules/noise/random.hpp>

std::random_device Random::rd;
std::mt19937 Random::rng = std::mt19937(Random::rd());
uint64_t Random::m_rand = 1;