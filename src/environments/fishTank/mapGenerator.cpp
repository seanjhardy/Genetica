#include <vector>
#include <cmath>
#include <random>
#include "../../modules/noise/perlin.hpp"
#include "rock.hpp"

using namespace std;

inline vector<Rock> generate_map(pair<int, int> map_size, int square_size, double edge_probability, double center_probability) {
    int x_size = round(map_size.first / square_size);
    int y_size = round(map_size.second / square_size);
    vector<Rock> rocks;

    // Calculate the center of the map
    double center_x = x_size / 2.0;
    double center_y = y_size / 2.0;

    // Calculate the maximum distance from the center of the map
    double max_distance = sqrt(center_x * center_x + center_y * center_y);

    // Seed for randomness
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1000);
    int seed = dis(gen);

    // Initialize Perlin noise generator
    siv::PerlinNoise noiseMap{ std::random_device{} };

    noiseMap.reseed(time(0));

    // Iterate over each grid cell
    for (int x = 0; x < x_size; ++x) {
        for (int y = 0; y < y_size; ++y) {
            // Calculate the distance from the center of the map to the current grid cell
            double distance = sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));

            // Calculate the probability of the grid cell being filled based on the exponential function
            double edge_dropoff = (edge_probability - center_probability) * (distance / max_distance) + center_probability;

            // Calculate Perlin noise value at current position
            double noise_value = noiseMap.octave2D((x / static_cast<double>(x_size) + seed) * 3,
                                                      (y / static_cast<double>(y_size)) * 3, 4);

            // Map Perlin noise value to probability between edge_probability and center_probability
            double probability = (noise_value + 1) / 2.0;

            // Add the grid cell to the map
            if (probability < edge_dropoff) {
                rocks.emplace_back(x * square_size + square_size /2,
                                   y * square_size + square_size /2, square_size);
            }
        }
    }

    return rocks;
}
