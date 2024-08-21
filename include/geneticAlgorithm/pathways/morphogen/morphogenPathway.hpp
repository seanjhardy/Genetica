#include <vector>
#include "./morphogen.hpp"

class MorphogenPathway {
private:
    std::vector<Morphogen> morphogens;
public:
    MorphogenPathway();

    void addMorphogen(Morphogen* morphogen);
    void removeMorphogen(Morphogen* morphogen);
    float morphogenConcentrationAtLocation(float x, float y);
};