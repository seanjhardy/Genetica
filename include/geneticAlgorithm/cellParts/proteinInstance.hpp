#include <vector>
#include <geneticAlgorithm/cellParts/cellPartType.hpp>
#include <geneticAlgorithm/cellParts/proteins/touchSensor.hpp>

class ProteinInstance : public CellPartInstance {
protected:
    std::vector<float> parameters;
    int type;
    float size;
public:
    ProteinInstance(LifeForm* lifeform, CellPartSchematic* schematic, SegmentInstance* parent);

    float getParameter(int index) {
        return parameters[index];
    }

    static ProteinInstance* createProteinInstance(LifeForm* lifeform, CellPartSchematic* schematic, SegmentInstance* parent, int type) {
        switch (type) {
            case 0:
                return dynamic_cast<ProteinInstance*>(new TouchSensor(lifeform, schematic, parent));
        }
    }
};