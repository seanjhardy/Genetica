#ifndef EFFECTOR
#define EFFECTOR

#include "./geneticUnit.hpp"

class Effector : public GeneticUnit {
public:
    enum class EffectorType {
        //Actions
        Divide,
        Die,
        Freeze,

        //Parameters
        DaughterDistance,
        Radius,
        Red,
        Green,
        Blue,

        //Proteins
        Chloroplast,
        TouchSensor,
    } effectorType;

    Effector(EffectorType effectorType,
             bool sign,
             float modifier,
             const float* embedding)
      : GeneticUnit(sign, modifier, embedding),
        effectorType(effectorType) {}
};

#endif