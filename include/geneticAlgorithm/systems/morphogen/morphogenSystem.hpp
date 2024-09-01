#include <vector>
#include <geneticAlgorithm/systems/morphogen/morphogen.hpp>
#include <vector_types.h>
#include <unordered_map>

class MorphogenSystem {
public:
    void addMorphogen(int id, Morphogen morphogen) {
        morphogens.insert({id, morphogen});
    }

    [[nodiscard]] std::unordered_map<int, double> sampleAllMorphogens(const float2& point) const {
        std::unordered_map<int, double> concentrations;
        for (const auto& morphogen : morphogens) {
            concentrations.insert({morphogen.first, morphogen.second.sample(point)});
        }
        // Compute morphogen interactions
        for (const auto& interaction : interactions) {
            concentrations[interaction.first.first] += concentrations[interaction.first.second] * interaction.second;
        }
        return concentrations;
    }

private:
    std::unordered_map<int, Morphogen> morphogens;
    std::map<std::pair<int, int>, float> interactions;
};