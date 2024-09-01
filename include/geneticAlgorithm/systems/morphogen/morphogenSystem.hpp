#include <vector>
#include <geneticAlgorithm/systems/morphogen/morphogen.hpp>
#include <vector_types.h>

class MorphogenSystem {
public:
    /*void addMorphogen(Morphogen morphogen) {
        morphogens.push_back(std::move(morphogen));
    }

    std::vector<double> sampleAllMorphogens(const float2& point) const {
        std::vector<double> concentrations;
        for (const auto& morphogen : morphogens) {
            concentrations.push_back(morphogen.sample(point));
        }
        return concentrations;
    }*/

private:
    std::vector<Morphogen> morphogens;
};