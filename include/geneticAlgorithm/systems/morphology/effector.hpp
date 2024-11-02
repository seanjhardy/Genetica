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
        Distance,
        Radius,
        Red,
        Green,
        Blue,

        //Proteins
        Chloroplast,
        TouchSensor,
        EFFECTOR_LENGTH
    } effectorType;

    Effector(EffectorType effectorType,
             bool sign,
             float modifier,
             const float* embedding)
      : GeneticUnit(sign, modifier, embedding),
        effectorType(effectorType) {}
};

#endif