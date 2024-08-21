#include <vector>
#include <geneticAlgorithm/cellParts/cellPartType.hpp>

class ProteinType : CellPartType {
private:
    std::vector<float> parameters;
    int type;
    float size;
public:
    ProteinType(LifeForm* lifeform, int type, float partCode);

    float getParameter(int index) {
        return parameters[index];
    }
};