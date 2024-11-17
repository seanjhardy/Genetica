#ifndef EFFECTOR
#define EFFECTOR

#include "./geneticUnit.hpp"

struct Effector : GeneticUnit {
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

    Effector(EffectorType efectorType, bool sign, float modifier, float3 embedding)
        : GeneticUnit(sign, modifier, embedding), effectorType(efectorType) {}
};

#endif