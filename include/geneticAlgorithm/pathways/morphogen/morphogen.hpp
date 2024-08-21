#include <map>
#include <vector>

/**
 * A morphogen is a signalling chemical used in a morphogen cascade to describe the morphological development
 * of a creature. THese are determined by hox genes.
 */
class Morphogen {
private:
    std::map<Morphogen*, float> bindingAgents;

public:
    //Morphogen(std::map<Morphogen*, float co);
    float generator(std::vector<Morphogen> morphogens, float x, float y);
};