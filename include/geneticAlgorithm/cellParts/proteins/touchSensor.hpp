#ifndef TOUCH
#define TOUCH

#include <geneticAlgorithm/cellParts/protein.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <simulator/simulator.hpp>

class TouchSensor : public Protein {
private:
    int endPoint;
    float overlapping;
public:
    TouchSensor(LifeForm* lifeform, Cell* parent)
        : Protein(lifeform, parent) {
        lifeform->addInput(this);
    }

    void simulate(float dt) override {
        Protein::simulate(dt);
        if (Simulator::get().getStep() % 10 == 0) {
            checkForOverlap();
        }
    }

    void checkForOverlap() {
        Point* end = lifeForm->getEnv()->getPoint(endPoint);
        /*std::vector<Point*> points = lifeForm->getEnv()->getQuadtree()->queryRange(end->pos - 1.25, end->pos + 1.25);
        for (Point* point : points) {
            if (point->entityID != lifeForm->entityID) {
                overlapping = true;
            }
        }*/
    }
};

#endif