#include <vector>
#include <geneticAlgorithm/systems/morphogen/morphogen.hpp>
#include <vector_types.h>
#include <unordered_map>

class MorphogenSystem {
public:
    void addMorphogen(int id, Morphogen morphogen, const std::unordered_map<int, float>& morphogenInteractions) {
        morphogens.insert({id, morphogen});
        concentrations.insert({id, 0.0});
        for (const auto& interaction : morphogenInteractions) {
            interactions.insert({{id, interaction.first}, interaction.second});
        }
    }

    [[nodiscard]] std::unordered_map<int, float> update(const float2& point) {
        for (const auto& morphogen : morphogens) {
            concentrations[morphogen.first] = morphogen.second.sample(point);
        }
        // Compute morphogen interactions
        for (const auto& interaction : interactions) {
            concentrations[interaction.first.second] += concentrations[interaction.first.first] * interaction.second;
            concentrations[interaction.first.first] *= (1.0 - abs(interaction.second));
        }
    }

private:
    std::unordered_map<int, Morphogen> morphogens;
    std::unordered_map<int, float> concentrations;
    std::map<std::pair<int, int>, float> interactions;
};